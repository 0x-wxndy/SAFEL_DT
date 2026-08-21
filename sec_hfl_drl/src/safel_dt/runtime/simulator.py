"""Top-level simulator entry point.

Per-round flow (PR-6a onwards):

1. Each fog policy ``select(round_idx)`` -> local indices to include.
2. Cloud orchestrates the FL round on the (possibly reduced) cohort.
3. Cost accounting: per-fog ``CostBreakdown`` (paper eqs. 3-9).
4. Multiplier update (single global dual-ascent step over all fogs).
5. Per-fog Lagrangian reward (paper eq. 5) -- or the simple Δacc-minus-
   penalty reward if no ``LagrangianConfig`` was supplied.
6. Each fog policy ``observe_feedback(...)`` -> updates traces + (for
   SAC) pushes a transition into the replay buffer / takes a gradient
   step.

When ``fog_policies`` is ``None`` every client participates and the
per-round policy hooks become no-ops (PR-3/4 legacy behaviour).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from torch import nn
from torch.utils.data import Dataset

from safel_dt.attacks.schedule import MaliciousSchedule, NoMalice
from safel_dt.costs.reward import Multipliers
from safel_dt.crypto.calibration import CalibrationReport, run_calibration
from safel_dt.crypto.channel import PlaintextChannel, SecureChannel
from safel_dt.crypto.paillier import PaillierContext
from safel_dt.crypto.signing import Signer
from safel_dt.dt.profiles import sample_profile_params
from safel_dt.fl.adversary_features import ClientAdversaryFeatures
from safel_dt.fl.client import FederatedClient, LocalTrainConfig
from safel_dt.fl.cloud_server import CloudRoundOutcome, CloudServer
from safel_dt.fl.fog_server import FogServer
from safel_dt.rl.cloud_policy import CloudFeedback, CloudPolicy
from safel_dt.rl.policy import FogPolicy, RoundFeedback
from safel_dt.runtime.cost_accounting import (
    CostBreakdown,
    PrivacyConfig,
    TimingCoefficients,
    compute_fog_cost_breakdown,
    per_client_time,
)
from safel_dt.runtime.lagrangian import (
    LagrangianConfig,
    LagrangianState,
    compute_fog_reward,
)
from safel_dt.runtime.round_context import RoundContext
from safel_dt.runtime.tracing import JsonlWriter
from safel_dt.seeds import set_global_seed
from safel_dt.transport.timing import SimClock, measure
from safel_dt.types import DeviceDTState, FogDTState, QualityProfile

_ZERO_MULTIPLIERS = Multipliers()


@dataclass(frozen=True)
class RewardConfig:
    """Simple fallback reward: Δaccuracy minus a small selection-fraction penalty.

    Only used when :class:`SimulatorConfig.lagrangian` is ``None``.
    """

    selection_penalty: float = 0.05


@dataclass(frozen=True)
class EncryptionConfig:
    """Drive the channel choice + crypto cost calibration (PR-11 / PR-12).

    ``mode="plain"`` is the legacy zero-overhead path (a no-op
    :class:`safel_dt.crypto.channel.PlaintextChannel`). ``mode="paillier"``
    builds a single :class:`safel_dt.crypto.paillier.PaillierContext`
    with the requested key length, shared across every fog and the
    cloud.

    When ``calibrate=True`` the simulator runs a one-shot timing pass on
    the live channel + a sample signer at boot. The measured per-payload
    wall times replace every device's synthetic ``c_enc / c_auth /
    c_verify`` profile draw (scaled by per-device ``record_size_kb`` so
    heterogeneity is preserved). When ``False`` we keep the synthetic
    profile draws verbatim -- useful for reproducing pre-PR-11 numbers.

    ``paillier_keybits`` defaults to 1024 (Paillier 1024 takes ~few
    hundred ms per encrypt for a small MLP, which is tolerable in a
    smoke test). The paper canonical value is 2048; pass it explicitly
    when generating paper-quality results.
    """

    mode: str = "plain"
    paillier_keybits: int = 1024
    calibrate: bool = True
    n_warmup: int = 1
    n_trials: int = 3

    def __post_init__(self) -> None:
        if self.mode not in ("plain", "paillier"):
            raise ValueError(
                f"encryption mode must be 'plain' or 'paillier', got {self.mode!r}"
            )
        if self.paillier_keybits < 256:
            raise ValueError(
                f"paillier_keybits must be >= 256, got {self.paillier_keybits}"
            )
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {self.n_trials}")


def _build_channel(cfg: EncryptionConfig) -> SecureChannel:
    """Realise the :class:`SecureChannel` requested by ``cfg``.

    Returned channel is shared across every fog + the cloud so the
    Paillier private key only needs to be generated once and the cloud
    can decrypt the fog-level aggregate.
    """
    if cfg.mode == "plain":
        return PlaintextChannel()
    return PaillierContext.generate(n_length=cfg.paillier_keybits)


@dataclass
class SimulatorConfig:
    """Inputs the simulator needs to run a single seed."""

    seed: int
    rounds: int
    model_factory: Callable[[], nn.Module]
    client_train_sets: list[Dataset]
    client_to_fog: dict[int, list[int]]
    test_set: Dataset
    train_cfg: LocalTrainConfig = field(default_factory=LocalTrainConfig)
    # PR-11: Encryption choice + calibration. When ``channel_factory`` is
    # explicitly set it overrides ``encryption`` (the pre-PR-11 escape
    # hatch used by a handful of unit tests). New code should set
    # ``encryption`` and leave ``channel_factory`` at its default.
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    # Phase 1 PQ auth: hmac | ecdsa | mldsa (see safel_dt.crypto.signing).
    sig_alg: str = "hmac"
    channel_factory: Callable[[], SecureChannel] | None = None
    aggregator: str = "fedavg"
    aggregator_options: dict[str, object] = field(default_factory=dict)
    malicious_schedule: MaliciousSchedule = field(default_factory=NoMalice)
    trace_path: Path | None = None
    fog_policies: dict[int, FogPolicy] | None = None
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)

    # PR-6a: Lagrangian reward + dual-ascent (replaces ``reward_cfg`` when set).
    lagrangian: LagrangianConfig | None = None
    client_profiles: dict[int, QualityProfile] | None = None
    fog_states: dict[int, FogDTState] | None = None
    timing: TimingCoefficients = field(default_factory=TimingCoefficients)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    # PR-6b: cloud-level D3QN policy. When provided, it picks the
    # aggregator per round (overriding ``aggregator`` above).
    cloud_policy: CloudPolicy | None = None

    # PR-9b: stragglers / dropouts (off by default; explicit opt-in).
    # ``drop_prob`` already lives on each ``DeviceDTState`` (set by the
    # quality profile). When ``enable_random_drops`` is True we *consume*
    # it: each selected client is dropped from the round with probability
    # ``device.drop_prob`` (Bernoulli, independent across clients/rounds).
    # When ``drop_late`` is True the simulator also drops any client whose
    # per-client estimated round time exceeds its fog's deadline ``delta``.
    # Both lists are recorded in the per-round JSONL trace.
    enable_random_drops: bool = False
    drop_late: bool = False

    # PR-9a: per-round multiplicative lognormal jitter on ``lambda_i`` and
    # ``record_size_kb``. ``sigma=0`` (default) reproduces the deterministic
    # profile values; ``sigma=0.1`` gives ~+/-10% per-round variation around
    # the nominal profile draw. Profile means stay constant across the run.
    device_noise_sigma: float = 0.0

    # PR-14: per-client adversary-detection features. When True the
    # FogServer decrypts each client's delta and computes (delta_norm_ratio,
    # cos_dist_to_mean, loss_zscore); these are propagated through
    # CloudRoundOutcome and into the next round's RoundFeedback so SAC /
    # heuristic policies can learn to avoid adversaries. Off by default
    # to keep legacy traces reproducible.
    adversary_features: bool = False


# --- device / fog state bootstrap ----------------------------------------


def _build_device_states(
    *,
    client_train_sets: list[Dataset],
    client_to_fog: dict[int, list[int]],
    profiles: dict[int, QualityProfile] | None,
    rng: np.random.Generator,
) -> dict[int, DeviceDTState]:
    fog_of: dict[int, int] = {
        cid: fid for fid, cids in client_to_fog.items() for cid in cids
    }
    out: dict[int, DeviceDTState] = {}
    for client_id, train_set in enumerate(client_train_sets):
        profile: QualityProfile = (
            profiles.get(client_id, "medium") if profiles is not None else "medium"
        )
        params = sample_profile_params(profile, rng)
        out[client_id] = DeviceDTState(
            client_id=client_id,
            fog_id=fog_of.get(client_id, -1),
            n_samples=len(train_set),  # type: ignore[arg-type]
            profile=profile,
            record_size_kb=params["record_size_kb"],
            c_enc=params["c_enc"],
            c_auth=params["c_auth"],
            c_verify=params["c_verify"],
            lambda_i=params["lambda"],
            battery=params["battery"],
            cpu=params["cpu"],
            mem=params["mem"],
            link_quality=params["link_quality"],
            packet_loss=params["packet_loss"],
            data_fraction=params.get("data_fraction", 1.0),
            label_noise=params.get("label_noise", 0.0),
            drop_prob=params.get("drop_prob", 0.0),
        )
    return out


def _build_fog_states(
    *,
    client_to_fog: dict[int, list[int]],
    explicit: dict[int, FogDTState] | None,
) -> dict[int, FogDTState]:
    if explicit is not None:
        return dict(explicit)
    return {
        fog_id: FogDTState(fog_id=fog_id, device_ids=list(cids))
        for fog_id, cids in client_to_fog.items()
    }


# --- reward helpers ------------------------------------------------------


def _simple_fog_reward(
    *,
    outcome: CloudRoundOutcome,
    prev_accuracy: float | None,
    selected_count: int,
    cohort_size: int,
    reward_cfg: RewardConfig,
) -> float:
    util = outcome.accuracy - (prev_accuracy if prev_accuracy is not None else outcome.accuracy)
    penalty = reward_cfg.selection_penalty * (selected_count / max(cohort_size, 1))
    return float(util - penalty)


def _per_client_attack_family(schedule: object) -> dict[str, str] | None:
    """Per-cohort {client_id_str: attack_family_name} for mixed schedules.

    Returns ``None`` for non-mixed schedules so the JSONL field stays
    compact (one ``null`` per round vs duplicating the same dict).
    Duck-typed on :class:`MixedAdversary` -- any schedule exposing
    ``per_client_attacks() -> dict[int, Attack]`` qualifies.
    """
    fn = getattr(schedule, "per_client_attacks", None)
    if not callable(fn):
        return None
    try:
        per = fn()
    except Exception:
        return None
    return {
        str(int(cid)): getattr(att, "name", "unknown")
        for cid, att in per.items()
    }


def _active_malicious_at(
    schedule: object, round_idx: int
) -> set[int]:
    """Return the set of clients actively attacking at ``round_idx``.

    Prefers a fast ``active_malicious_ids(round_idx)`` method if the
    schedule provides one (PR-16's :class:`RampedAdversary`). Otherwise
    falls back to iterating ``attack_for`` over the static cohort, which
    works for any :class:`MaliciousSchedule` implementation including
    :class:`WindowedAdversary` / :class:`PeriodicAdversary`.
    """
    fast = getattr(schedule, "active_malicious_ids", None)
    if callable(fast):
        return {int(c) for c in fast(round_idx)}
    cohort = schedule.malicious_ids()  # type: ignore[attr-defined]
    if not cohort:
        return set()
    active: set[int] = set()
    for cid in cohort:
        att = schedule.attack_for(int(cid), int(round_idx))  # type: ignore[attr-defined]
        # ``NoAttack.name == "none"`` is the only safe duck-type check
        # without importing NoAttack here (would create a dependency
        # cycle via attacks -> simulator). We accept the small string
        # comparison cost; this runs once per round per ever-attacker.
        if getattr(att, "name", "none") != "none":
            active.add(int(cid))
    return active


def _per_fog_selection(
    fog_policies: dict[int, FogPolicy] | None,
    round_idx: int,
    fog_select_s: dict[int, float] | None = None,
) -> dict[int, list[int]] | None:
    """Run each fog policy's ``select`` once. When ``fog_select_s`` is
    provided, record per-fog wall-time so the simulator can publish RL
    inference overhead in the trace (PR-15)."""
    if fog_policies is None:
        return None
    out: dict[int, list[int]] = {}
    for fog_id, policy in fog_policies.items():
        with measure() as elapsed:
            chosen = policy.select(round_idx=round_idx)
        if fog_select_s is not None:
            fog_select_s[fog_id] = float(elapsed())
        out[fog_id] = chosen
    return out


def _local_indices_for_fog(
    fog_id: int,
    client_to_fog: dict[int, list[int]],
    selected_local: dict[int, list[int]] | None,
) -> list[int]:
    if selected_local is None:
        return list(range(len(client_to_fog.get(fog_id, []))))
    return list(selected_local.get(fog_id, []))


def _jittered_states(
    *,
    base: dict[int, DeviceDTState],
    sigma: float,
    rng: np.random.Generator,
) -> dict[int, DeviceDTState]:
    """Return per-round copies of ``base`` with ``lambda_i`` / ``record_size_kb``
    multiplied by independent lognormal(0, sigma) noise.

    ``sigma <= 0`` returns ``base`` unchanged (no copy). Other device fields
    (``n_samples``, security costs, etc.) are not jittered: ``lambda_i`` and
    ``record_size_kb`` are the two cost-driving fields that vary round-to-round
    in real deployments (CPU contention; per-round payload jitter), while the
    others are determined by the algorithm / hardware.
    """
    if sigma <= 0.0:
        return base
    from dataclasses import replace

    out: dict[int, DeviceDTState] = {}
    for cid, dev in base.items():
        noise_lambda = float(np.exp(rng.normal(0.0, sigma)))
        noise_size = float(np.exp(rng.normal(0.0, sigma)))
        out[cid] = replace(
            dev,
            lambda_i=dev.lambda_i * noise_lambda,
            record_size_kb=dev.record_size_kb * noise_size,
        )
    return out


def _apply_drops(
    *,
    per_fog_selected: dict[int, list[int]] | None,
    client_to_fog: dict[int, list[int]],
    device_states: dict[int, DeviceDTState],
    fog_states: dict[int, FogDTState],
    timing: TimingCoefficients,
    enable_random: bool,
    drop_late: bool,
    rng: np.random.Generator,
) -> tuple[dict[int, list[int]] | None, dict[int, list[int]], dict[int, list[int]]]:
    """Apply random + late dropouts to ``per_fog_selected``.

    Returns ``(survivors, dropped_random, dropped_late)``. ``survivors``
    maps fog_id -> local-index list after drops. ``dropped_*`` map
    fog_id -> *global* client ids that were excluded from this round's
    aggregation.

    Random drop: per selected client, Bernoulli(``device.drop_prob``).
    Late drop: when ``drop_late`` is set, drop any client whose
    :func:`per_client_time` exceeds its fog's ``delta``.

    When ``per_fog_selected`` is ``None`` (no fog policy in use), drops
    are still applied against the implicit "select all clients in the
    fog" cohort so traces remain consistent.
    """
    if not enable_random and not drop_late:
        return per_fog_selected, {}, {}
    if per_fog_selected is None:
        per_fog_selected = {
            fog_id: list(range(len(cids))) for fog_id, cids in client_to_fog.items()
        }
    survivors: dict[int, list[int]] = {}
    dropped_random: dict[int, list[int]] = {}
    dropped_late: dict[int, list[int]] = {}
    for fog_id, local_indices in per_fog_selected.items():
        clients = client_to_fog.get(fog_id, [])
        fog = fog_states.get(fog_id)
        kept: list[int] = []
        drop_r: list[int] = []
        drop_l: list[int] = []
        for loc in local_indices:
            if not (0 <= loc < len(clients)):
                continue
            cid = clients[loc]
            dev = device_states[cid]
            if enable_random and dev.drop_prob > 0.0 and rng.random() < dev.drop_prob:
                drop_r.append(cid)
                continue
            if (
                drop_late
                and fog is not None
                and per_client_time(dev, timing=timing) > fog.delta
            ):
                drop_l.append(cid)
                continue
            kept.append(loc)
        survivors[fog_id] = kept
        dropped_random[fog_id] = drop_r
        dropped_late[fog_id] = drop_l
    return survivors, dropped_random, dropped_late


def _apply_calibrated_costs(
    devices: dict[int, DeviceDTState],
    report: CalibrationReport,
) -> None:
    """Overwrite ``c_enc / c_auth / c_verify`` in-place with measured values.

    Heterogeneity preservation: the encryption cost scales linearly in
    the payload size, so we keep the per-device variation captured in
    ``record_size_kb`` and apply the calibrated *per-KB* time to it.
    The signing payload is the fixed-size digest of the canonical body
    -- it does not scale with the model, so every device gets the same
    ``c_auth / c_verify`` time.
    """
    if not devices:
        return
    sizes = np.array([d.record_size_kb for d in devices.values()], dtype=np.float64)
    mean_size = float(np.mean(sizes)) if sizes.size else 1.0
    if mean_size <= 0.0:
        mean_size = 1.0
    enc_per_kb = report.costs.c_enc / mean_size
    for dev in devices.values():
        dev.c_enc = float(enc_per_kb * dev.record_size_kb)
        dev.c_auth = float(report.costs.c_auth)
        dev.c_verify = float(report.costs.c_verify)


def run_simulation(cfg: SimulatorConfig) -> list[CloudRoundOutcome]:
    """Run ``cfg.rounds`` global rounds and return the per-round outcomes."""
    set_global_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    clock = SimClock()

    # PR-11: explicit ``channel_factory`` wins (legacy path); otherwise
    # build the channel from ``encryption`` and share the *instance*
    # across every FL participant. Paillier in particular only works
    # when fogs + cloud share the same key pair.
    if cfg.channel_factory is not None:
        channel = cfg.channel_factory()
    else:
        channel = _build_channel(cfg.encryption)

    clients: list[FederatedClient] = []
    for client_id, train_set in enumerate(cfg.client_train_sets):
        signer = Signer.generate(cfg.sig_alg)
        clients.append(
            FederatedClient(
                client_id=client_id,
                model_factory=cfg.model_factory,
                train_set=train_set,
                channel=channel,
                signer=signer,
                config=cfg.train_cfg,
            )
        )

    compute_homomorphic_agg = cfg.aggregator == "fedavg" or cfg.cloud_policy is not None
    fogs: list[FogServer] = []
    for fog_id, client_ids in cfg.client_to_fog.items():
        fog_clients = [clients[cid] for cid in client_ids]
        fogs.append(
            FogServer(
                fog_id=fog_id,
                clients=fog_clients,
                channel=channel,
                compute_homomorphic_aggregate=compute_homomorphic_agg,
                compute_adversary_features=cfg.adversary_features,
            )
        )

    cloud = CloudServer(
        model_factory=cfg.model_factory,
        fogs=fogs,
        channel=channel,
        test_set=cfg.test_set,
        aggregator=cfg.aggregator,
        aggregator_options=dict(cfg.aggregator_options),
    )

    device_states = _build_device_states(
        client_train_sets=cfg.client_train_sets,
        client_to_fog=cfg.client_to_fog,
        profiles=cfg.client_profiles,
        rng=np.random.default_rng(cfg.seed + 13),
    )
    fog_states = _build_fog_states(
        client_to_fog=cfg.client_to_fog,
        explicit=cfg.fog_states,
    )

    # PR-11 / PR-12: one-shot crypto calibration. We use the *live*
    # channel + a throw-away signer so the measured numbers reflect the
    # actual algorithms in use. Synthetic profile draws are discarded
    # whenever ``encryption.calibrate`` is set.
    #
    # The vector size is taken from ``cloud.global_param_count`` (already
    # computed during CloudServer construction) rather than from a fresh
    # ``cfg.model_factory()`` call: building an extra model would
    # advance the global ``torch`` RNG and silently desync every
    # subsequent model initialisation, breaking determinism-sensitive
    # downstream tests.
    calibration_report: CalibrationReport | None = None
    if cfg.channel_factory is None and cfg.encryption.calibrate:
        cal_signer = Signer.generate(cfg.sig_alg)
        calibration_report = run_calibration(
            channel=channel,
            signer=cal_signer,
            vec_size=cloud.global_param_count,
            encryption_mode=cfg.encryption.mode,
            paillier_keybits=(
                cfg.encryption.paillier_keybits
                if cfg.encryption.mode == "paillier"
                else None
            ),
            n_warmup=cfg.encryption.n_warmup,
            n_trials=cfg.encryption.n_trials,
        )
        _apply_calibrated_costs(device_states, calibration_report)

    lagrangian_state: LagrangianState | None = (
        LagrangianState(cfg=cfg.lagrangian) if cfg.lagrangian is not None else None
    )

    writer = JsonlWriter(cfg.trace_path) if cfg.trace_path is not None else None
    if cfg.trace_path is not None and calibration_report is not None:
        # PR-11: calibration is per-run, not per-round -- write it to a
        # sidecar so the JSONL trace stays homogeneous (every line is
        # exactly one round). Path: ``<trace>.calibration.json``.
        import json as _json

        sidecar = cfg.trace_path.with_suffix(cfg.trace_path.suffix + ".calibration.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("w", encoding="utf-8") as fh:
            _json.dump(calibration_report.as_dict(), fh, indent=2)
    outcomes: list[CloudRoundOutcome] = []
    prev_acc: float | None = None

    for r in range(cfg.rounds):
        ctx = RoundContext(round_idx=r, rng=rng, clock=clock)
        del ctx

        round_devices = _jittered_states(
            base=device_states, sigma=cfg.device_noise_sigma, rng=rng,
        )

        # PR-15: wall-clock instrumentation for the RL agents themselves.
        # Reviewer-2 #4 asked for explicit RL inference + online-training
        # overhead so it can be compared against ``cost_train/sec/comm``.
        # All times here are seconds measured on the host that ran the sim.
        rl_fog_select_s: dict[int, float] = {}
        rl_fog_learn_s: dict[int, float] = {}
        rl_cloud_select_s: float = 0.0
        rl_cloud_learn_s: float = 0.0

        per_fog_selected = _per_fog_selection(
            cfg.fog_policies, r, fog_select_s=rl_fog_select_s
        )
        per_fog_selected, dropped_random, dropped_late = _apply_drops(
            per_fog_selected=per_fog_selected,
            client_to_fog=cfg.client_to_fog,
            device_states=round_devices,
            fog_states=fog_states,
            timing=cfg.timing,
            enable_random=cfg.enable_random_drops,
            drop_late=cfg.drop_late,
            rng=rng,
        )
        cloud_aggregator: str | None = None
        cloud_debug: dict[str, object] = {}
        if cfg.cloud_policy is not None:
            with measure() as elapsed_cloud_sel:
                cloud_aggregator = cfg.cloud_policy.select(round_idx=r)
            rl_cloud_select_s = float(elapsed_cloud_sel())
            cloud_debug = cfg.cloud_policy.debug_info()
        outcome = cloud.run_round(
            round_idx=r,
            schedule=cfg.malicious_schedule,
            rng=rng,
            per_fog_selected_local=per_fog_selected,
            aggregator_override=cloud_aggregator,
        )
        outcomes.append(outcome)

        # --- per-fog cost breakdowns (always computed; cheap) ---
        breakdowns: dict[int, CostBreakdown] = {}
        for fog_id, client_ids in cfg.client_to_fog.items():
            local_sel = _local_indices_for_fog(fog_id, cfg.client_to_fog, per_fog_selected)
            devs = [round_devices[cid] for cid in client_ids]
            breakdowns[fog_id] = compute_fog_cost_breakdown(
                fog_state=fog_states[fog_id],
                devices=devs,
                selected_local_indices=local_sel,
                timing=cfg.timing,
                privacy=cfg.privacy,
            )

        # --- multiplier dual ascent (only when Lagrangian is active) ---
        if lagrangian_state is not None:
            lagrangian_state.step(list(breakdowns.values()))

        # --- per-fog rewards (always computed; fed to fog policies + cloud policy) ---
        per_fog_reward: dict[int, float] = {}
        prev = prev_acc if prev_acc is not None else outcome.accuracy
        utility = outcome.accuracy - prev
        for fog_id in cfg.client_to_fog:
            if lagrangian_state is not None:
                per_fog_reward[fog_id] = compute_fog_reward(
                    breakdown=breakdowns[fog_id],
                    utility=utility,
                    state=lagrangian_state,
                )
            else:
                selected_local = (
                    per_fog_selected.get(fog_id, []) if per_fog_selected is not None else []
                )
                per_fog_reward[fog_id] = _simple_fog_reward(
                    outcome=outcome,
                    prev_accuracy=prev_acc,
                    selected_count=len(selected_local),
                    cohort_size=len(cfg.client_to_fog.get(fog_id, [])),
                    reward_cfg=cfg.reward_cfg,
                )

        # --- fog-policy feedback ---
        if cfg.fog_policies is not None and per_fog_selected is not None:
            for fog_id, policy in cfg.fog_policies.items():
                selected_local = per_fog_selected.get(fog_id, [])
                fog_features = (
                    outcome.per_fog_adversary_features.get(fog_id)
                    if outcome.per_fog_adversary_features
                    else None
                )
                # ``per_fog_adversary_features`` is typed as
                # ``dict[int, object]`` at the cloud boundary to avoid
                # circular imports; cast back to the concrete type for
                # the policy layer.
                typed_features: dict[int, ClientAdversaryFeatures] | None = (
                    {
                        int(k): v
                        for k, v in fog_features.items()
                        if isinstance(v, ClientAdversaryFeatures)
                    }
                    if fog_features
                    else None
                )
                with measure() as elapsed_fog_learn:
                    policy.observe_feedback(
                        RoundFeedback(
                            round_idx=r,
                            selected_local_indices=selected_local,
                            client_losses=dict(outcome.per_client_losses),
                            reward=per_fog_reward[fog_id],
                            done=(r == cfg.rounds - 1),
                            client_features=typed_features,
                        )
                    )
                rl_fog_learn_s[fog_id] = float(elapsed_fog_learn())

        # --- cloud reward (always computed; used by trace + cloud policy) ---
        cloud_reward = (
            sum(per_fog_reward.values()) / max(len(per_fog_reward), 1)
            if per_fog_reward
            else utility
        )

        # --- cloud-policy feedback ---
        if cfg.cloud_policy is not None and cloud_aggregator is not None:
            multipliers_now = (
                lagrangian_state.multipliers
                if lagrangian_state is not None
                else _ZERO_MULTIPLIERS
            )
            with measure() as elapsed_cloud_learn:
                cfg.cloud_policy.observe_feedback(
                    CloudFeedback(
                        round_idx=r,
                        chosen_aggregator=cloud_aggregator,
                        outcome=outcome,
                        multipliers=multipliers_now,
                        per_fog_reward=per_fog_reward,
                        reward=float(cloud_reward),
                        done=(r == cfg.rounds - 1),
                    )
                )
            rl_cloud_learn_s = float(elapsed_cloud_learn())

        prev_acc = outcome.accuracy

        if writer is not None:
            writer.append(
                {
                    "round_idx": outcome.round_idx,
                    "accuracy": outcome.accuracy,
                    "loss": outcome.loss,
                    "n_clients_accepted": outcome.n_clients_accepted,
                    "n_clients_rejected": outcome.n_clients_rejected,
                    "n_fogs_accepted": outcome.n_fogs_accepted,
                    "aggregator": outcome.aggregator,
                    "selected_per_fog": (
                        {str(k): v for k, v in per_fog_selected.items()}
                        if per_fog_selected is not None
                        else None
                    ),
                    "costs": {
                        str(fid): {
                            "comm": b.cost_comm,
                            "train": b.cost_train,
                            "sec": b.cost_sec,
                            "t_round": b.t_round,
                            "g_lat": b.g_lat,
                            "g_cap": b.g_cap,
                            "g_priv": b.g_priv,
                        }
                        for fid, b in breakdowns.items()
                    },
                    "multipliers": (
                        {
                            "lat": lagrangian_state.multipliers.lat,
                            "cap": lagrangian_state.multipliers.cap,
                            "priv": lagrangian_state.multipliers.priv,
                        }
                        if lagrangian_state is not None
                        else None
                    ),
                    "cloud_aggregator": cloud_aggregator,
                    "cloud_debug": cloud_debug if cloud_debug else None,
                    "rl_overhead_s": {
                        "fog_select": {
                            str(k): float(v) for k, v in rl_fog_select_s.items()
                        },
                        "fog_learn": {
                            str(k): float(v) for k, v in rl_fog_learn_s.items()
                        },
                        "cloud_select": rl_cloud_select_s,
                        "cloud_learn": rl_cloud_learn_s,
                        "total": (
                            sum(rl_fog_select_s.values())
                            + sum(rl_fog_learn_s.values())
                            + rl_cloud_select_s
                            + rl_cloud_learn_s
                        ),
                    },
                    "adversary_features": (
                        {
                            str(fid): {
                                str(cid): f.as_dict()
                                for cid, f in feats.items()
                                if isinstance(f, ClientAdversaryFeatures)
                            }
                            for fid, feats in outcome.per_fog_adversary_features.items()
                        }
                        if outcome.per_fog_adversary_features
                        else None
                    ),
                    "utility": float(utility),
                    "per_fog_reward": {str(k): float(v) for k, v in per_fog_reward.items()},
                    "cloud_reward": float(cloud_reward),
                    "dropped_random": {str(k): v for k, v in dropped_random.items()},
                    "dropped_late": {str(k): v for k, v in dropped_late.items()},
                    "attack_name": getattr(cfg.malicious_schedule, "name", "none"),
                    "malicious_ids": sorted(cfg.malicious_schedule.malicious_ids()),
                    # PR-16: round-resolved active attackers. For
                    # non-ramped schedules this is either the full
                    # ``malicious_ids`` (when within the window) or
                    # empty (out of window). Analysis scripts should
                    # prefer this field when computing per-round
                    # attack-rate plots.
                    "active_malicious_ids": sorted(_active_malicious_at(
                        cfg.malicious_schedule, r
                    )),
                    # Mixed-attack scenario only: per-cohort attack
                    # family assignment (stable across rounds). None
                    # for single-family schedules.
                    "per_client_attack_family": _per_client_attack_family(
                        cfg.malicious_schedule
                    ),
                }
            )
    return outcomes
