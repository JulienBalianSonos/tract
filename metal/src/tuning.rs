//! Per-process scheduling defaults and overrides for the Metal runtime.
//!
//! Environment overrides take precedence over application overrides and the
//! baseline profile. The optional, budget-capped synthetic workload probe is
//! disabled by default; enable it with `TRACT_METAL_AUTOTUNE=1` or
//! [`set_autotune`]. Explicit overrides are excluded from its sweep.
//!
//! The first dispatch resolves the profile. Without a probe it freezes at
//! resolution; with a probe it freezes after the sweep. Late overrides fail.
//! Tuning state is process-local and is never persisted.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use tract_core::internal::*;

/// The resolved per-process Metal performance-tuning profile.
///
/// Read it through [`tuning`]; construct it directly only in tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalTuning {
    /// Context length (in cached tokens) where decode-shaped SDPA steps
    /// switch from the batched-gemm pipeline to the fused flash-attention
    /// kernel. The default `usize::MAX` keeps the batched-gemm path active.
    /// Env: `TRACT_METAL_FLASH_SDPA_MIN_T` (legacy
    /// `TRACT_METAL_GPT_OSS_FLASH_MIN_T`).
    pub flash_sdpa_min_t: usize,
    /// Key-block size of the block-wise prefill attention: prefill steps
    /// whose effective key length exceeds this loop key-blocks through a
    /// fixed scores buffer with a running-max/denominator f32 state instead
    /// of materializing the full [Hkv, group*S, T] scores/probs (which near
    /// 50k context reaches GBs per layer, touched 3-4 times). 0 disables.
    /// Env: `TRACT_METAL_SDPA_PREFILL_BLOCK`.
    pub sdpa_prefill_block: usize,
    /// Context length where the decode AV gemv switches to split-k (partial
    /// sums over key chunks + a small reduce).
    /// No env override (gate: `TRACT_METAL_DISABLE_SDPA_SPLIT_K`).
    pub sdpa_split_k_min_t: usize,
    /// Target keys per split-k chunk (chunk count is clamped to [2, 16]).
    /// No env override.
    pub sdpa_split_k_chunk: usize,
    /// Commit cadence: split each forward into command buffers every N
    /// `command_buffer()` acquisitions (roughly every N kernel dispatches),
    /// so the GPU starts executing early layers while the CPU still encodes
    /// late ones. 0 disables the cadence: a forward uses one buffer plus
    /// boundaries requested by operations. Applications can set a model-
    /// specific value with [`set_tuning_overrides`]. Env:
    /// `TRACT_METAL_COMMIT_EVERY_N_DISPATCHES`.
    pub commit_every_n_dispatches: usize,
    /// How many committed-but-unawaited command buffers `commit_current`
    /// keeps in flight (>= 1). The wait on the oldest buffer is the
    /// backpressure that bounds transient memory. Env:
    /// `TRACT_METAL_MAX_IN_FLIGHT`.
    pub max_command_buffers_in_flight: usize,
    /// Routed-MoE ops end their dispatch with a non-blocking command-buffer
    /// boundary only when the step routes more than this many (token,
    /// expert) pairs. The boundary is a correctness measure for prefill-sized
    /// batches (see `MetalRoutedQ40MatMul`); decode steps (route_count =
    /// top_k) end with the runtime's own blocking logits sync anyway, and
    /// Env:
    /// `TRACT_METAL_MOE_COMMIT_MIN_ROUTES`.
    pub moe_commit_min_routes: usize,
    /// Route count at which routed-q40 MoE matmuls take the expert-grouped
    /// path: bin the routes by expert (single-threadgroup counting sort),
    /// then let each threadgroup amortize weight reads across routes
    /// of one expert through the simdgroup-matrix pipeline. The per-route kernel re-reads an expert's weights once
    /// per route, so below this threshold (decode-sized route lists) it is
    /// the cheaper one. No env override (gate:
    /// `TRACT_METAL_DISABLE_GROUPED_MOE`).
    pub moe_grouped_min_routes: usize,
}

/// Partial profile an embedding application may register through
/// [`set_tuning_overrides`] before the first Metal dispatch. `None` fields
/// keep the baseline; env vars still win over these hints, and the load-time
/// probe does not sweep knobs these pin.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetalTuningOverrides {
    pub flash_sdpa_min_t: Option<usize>,
    pub sdpa_prefill_block: Option<usize>,
    pub sdpa_split_k_min_t: Option<usize>,
    pub sdpa_split_k_chunk: Option<usize>,
    pub commit_every_n_dispatches: Option<usize>,
    pub max_command_buffers_in_flight: Option<usize>,
    pub moe_commit_min_routes: Option<usize>,
    pub moe_grouped_min_routes: Option<usize>,
}

impl MetalTuningOverrides {
    /// Merge `other` over `self`, field by field: `Some` fields of `other`
    /// win, `None` fields keep the current value.
    fn merge_from(&mut self, other: &MetalTuningOverrides) {
        macro_rules! merge {
            ($($field:ident),*) => {
                $(if let Some(v) = other.$field { self.$field = Some(v); })*
            };
        }
        merge!(
            flash_sdpa_min_t,
            sdpa_prefill_block,
            sdpa_split_k_min_t,
            sdpa_split_k_chunk,
            commit_every_n_dispatches,
            max_command_buffers_in_flight,
            moe_commit_min_routes,
            moe_grouped_min_routes
        );
    }
}

/// Which probeable scheduling knobs already have an externally supplied
/// value (env var or app override): the load-time probe skips those, better
/// information already exists, and those layers outrank probe results
/// anyway (probing what is pinned is wasted load time).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PinnedKnobs {
    pub max_command_buffers_in_flight: bool,
    pub commit_every_n_dispatches: bool,
    pub moe_commit_min_routes: bool,
}

impl PinnedKnobs {
    /// Recompute from the same sources as [`MetalTuning::resolve`], injected
    /// for hermetic tests.
    fn compute(
        overrides: &MetalTuningOverrides,
        env: impl Fn(&str) -> Option<String>,
    ) -> PinnedKnobs {
        let parsed = |name: &str| env(name).and_then(|v| v.parse::<usize>().ok());
        PinnedKnobs {
            max_command_buffers_in_flight: parsed("TRACT_METAL_MAX_IN_FLIGHT")
                .filter(|&n| n >= 1)
                .is_some()
                || overrides.max_command_buffers_in_flight.is_some(),
            commit_every_n_dispatches: parsed("TRACT_METAL_COMMIT_EVERY_N_DISPATCHES").is_some()
                || overrides.commit_every_n_dispatches.is_some(),
            moe_commit_min_routes: parsed("TRACT_METAL_MOE_COMMIT_MIN_ROUTES").is_some()
                || overrides.moe_commit_min_routes.is_some(),
        }
    }
}

/// The pinned-knob mask for this process, from the live env/app-hint state.
pub(crate) fn pinned_knobs() -> PinnedKnobs {
    let overrides = app_overrides().lock().map(|o| o.clone()).unwrap_or_default();
    PinnedKnobs::compute(&overrides, |name| std::env::var(name).ok())
}

impl MetalTuning {
    /// The hardcoded defaults for this runtime.
    pub const BASELINE: MetalTuning = MetalTuning {
        flash_sdpa_min_t: usize::MAX,
        sdpa_prefill_block: 4096,
        sdpa_split_k_min_t: 8192,
        sdpa_split_k_chunk: 2048,
        commit_every_n_dispatches: 0,
        max_command_buffers_in_flight: 8,
        moe_commit_min_routes: 64,
        moe_grouped_min_routes: 64,
    };

    /// Resolve the profile from the baseline, the registered application
    /// overrides and the process environment. Called once per process by
    /// [`tuning`]; use that accessor instead.
    pub fn from_env_and_device() -> MetalTuning {
        let overrides = app_overrides().lock().map(|o| o.clone()).unwrap_or_default();
        Self::resolve(&overrides, |name| std::env::var(name).ok())
    }

    /// Pure resolution, env injected for hermetic tests.
    /// Hard ceiling for `moe_commit_min_routes`: 512 routes = 64 prefilled
    /// tokens at top_k 8, far below any corruption-prone long prefill, and
    /// far above any decode step (route count = top_k).
    pub const MOE_COMMIT_MIN_ROUTES_CEILING: usize = 512;

    fn resolve(
        overrides: &MetalTuningOverrides,
        env: impl Fn(&str) -> Option<String>,
    ) -> MetalTuning {
        let base = MetalTuning::BASELINE;
        // Seam for device-informed derivations: adjust `base` here, between
        // the baseline and the app overrides. Nothing today has a principled
        // cross-device scaling rule, so `base` is used as-is.
        let parsed = |name: &str| env(name).and_then(|v| v.parse::<usize>().ok());
        MetalTuning {
            flash_sdpa_min_t: parsed("TRACT_METAL_FLASH_SDPA_MIN_T")
                .or_else(|| parsed("TRACT_METAL_GPT_OSS_FLASH_MIN_T"))
                .or(overrides.flash_sdpa_min_t)
                .unwrap_or(base.flash_sdpa_min_t),
            sdpa_prefill_block: parsed("TRACT_METAL_SDPA_PREFILL_BLOCK")
                .or(overrides.sdpa_prefill_block)
                .unwrap_or(base.sdpa_prefill_block),
            sdpa_split_k_min_t: overrides.sdpa_split_k_min_t.unwrap_or(base.sdpa_split_k_min_t),
            sdpa_split_k_chunk: overrides.sdpa_split_k_chunk.unwrap_or(base.sdpa_split_k_chunk),
            commit_every_n_dispatches: parsed("TRACT_METAL_COMMIT_EVERY_N_DISPATCHES")
                .or(overrides.commit_every_n_dispatches)
                .unwrap_or(base.commit_every_n_dispatches),
            max_command_buffers_in_flight: parsed("TRACT_METAL_MAX_IN_FLIGHT")
                .filter(|&n| n >= 1)
                .or(overrides.max_command_buffers_in_flight.filter(|&n| n >= 1))
                .unwrap_or(base.max_command_buffers_in_flight),
            // The commit boundary this knob gates is CORRECTNESS-required on
            // prefill-sized route batches (see MetalRoutedQ40MatMul::eval):
            // clamp to a ceiling that keeps decode-sized batches uncommitted
            // while guaranteeing the boundary fires for real prefills, so no
            // override (env, hint or autotune) can disable it.
            moe_commit_min_routes: parsed("TRACT_METAL_MOE_COMMIT_MIN_ROUTES")
                .or(overrides.moe_commit_min_routes)
                .unwrap_or(base.moe_commit_min_routes)
                .min(Self::MOE_COMMIT_MIN_ROUTES_CEILING),
            moe_grouped_min_routes: overrides
                .moe_grouped_min_routes
                .unwrap_or(base.moe_grouped_min_routes),
        }
    }
}

fn app_overrides() -> &'static Mutex<MetalTuningOverrides> {
    static O: OnceLock<Mutex<MetalTuningOverrides>> = OnceLock::new();
    O.get_or_init(|| Mutex::new(MetalTuningOverrides::default()))
}

static RESOLVED: AtomicBool = AtomicBool::new(false);

/// The immutable process profile, set at freeze time. Before it is set,
/// reads go through [`probe_window`] (only the default-on load-time
/// autotune probe keeps that window open past the first read).
static FROZEN: OnceLock<MetalTuning> = OnceLock::new();

/// Profile storage between first resolution and freeze: the load-time
/// autotune probe swaps candidate values in here.
fn probe_window() -> &'static Mutex<Option<MetalTuning>> {
    static W: OnceLock<Mutex<Option<MetalTuning>>> = OnceLock::new();
    W.get_or_init(|| Mutex::new(None))
}

/// The autotune probe opt-in hint ([`set_autotune`]`(true)`); the probe is
/// disabled by default. `TRACT_METAL_AUTOTUNE` (0/1), when set, wins over the hint.
static AUTOTUNE_ENABLED_HINT: AtomicBool = AtomicBool::new(false);

pub(crate) fn autotune_enabled() -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let env = *ENV.get_or_init(|| std::env::var("TRACT_METAL_AUTOTUNE").ok().map(|v| v != "0"));
    autotune_enabled_from(env, AUTOTUNE_ENABLED_HINT.load(Ordering::Acquire))
}

/// Pure polarity rule, env pre-parsed for hermetic tests: default disabled, the
/// hint opts in, an explicit env value overrides the hint in both
/// directions.
fn autotune_enabled_from(env: Option<bool>, hint_enabled: bool) -> bool {
    env.unwrap_or(hint_enabled)
}

/// The process-wide tuning profile. Lifecycle: the first call resolves (and
/// logs) it; with the probe opted out it freezes right there and every
/// later call is a static read (env lookups on per-dispatch hot paths
/// measurably show up in decode CPU profiles, hence resolve-once). With the
/// default-on probe, reads go through a mutex until the load-time probe
/// freezes the profile (see `crate::autotune`); the values are still
/// coherent at every read, only the fast path is deferred.
pub fn tuning() -> MetalTuning {
    if let Some(t) = FROZEN.get() {
        return *t;
    }
    tuning_unfrozen()
}

#[cold]
fn tuning_unfrozen() -> MetalTuning {
    let mut window = probe_window().lock().unwrap_or_else(|e| e.into_inner());
    // Another thread may have frozen while this one waited on the lock.
    if let Some(t) = FROZEN.get() {
        return *t;
    }
    let t = *window.get_or_insert_with(|| {
        RESOLVED.store(true, Ordering::Release);
        let tuning = MetalTuning::from_env_and_device();
        log::debug!("resolved Metal tuning profile: {tuning:?}");
        tuning
    });
    if !autotune_enabled() {
        // Freeze the resolved profile when probing is disabled.
        let _ = FROZEN.set(t);
    }
    t
}

/// True once the process profile is immutable.
pub(crate) fn is_frozen() -> bool {
    FROZEN.get().is_some()
}

/// Swap the process profile during the load-time probe window. Only the
/// probe calls this, between fully quiesced runs; fails once frozen.
pub(crate) fn probe_set(t: MetalTuning) -> TractResult<()> {
    let mut window = probe_window().lock().unwrap_or_else(|e| e.into_inner());
    if FROZEN.get().is_some() {
        bail!("Metal tuning profile already frozen; probe window is closed");
    }
    RESOLVED.store(true, Ordering::Release);
    *window = Some(t);
    Ok(())
}

/// Close the probe window: `t` becomes the immutable process profile.
/// Idempotent; a first freeze from a racing plain read cannot happen while
/// the probe holds the window open (probes run single-threaded at load).
pub(crate) fn probe_freeze(t: MetalTuning) {
    let mut window = probe_window().lock().unwrap_or_else(|e| e.into_inner());
    RESOLVED.store(true, Ordering::Release);
    *window = Some(t);
    let _ = FROZEN.set(t);
}

/// Register application tuning hints. Must be called before the first Metal
/// dispatch of the process (i.e. before runtime prepare); fails once the
/// profile has been resolved. Later calls merge over earlier ones field by
/// field. Env vars still take precedence over these hints, and the
/// load-time probe does not sweep knobs these pin.
pub fn set_tuning_overrides(overrides: MetalTuningOverrides) -> TractResult<()> {
    let mut current = app_overrides().lock().map_err(|e| anyhow!("{e}"))?;
    if RESOLVED.load(Ordering::Acquire) {
        bail!(
            "Metal tuning profile already resolved; \
             set_tuning_overrides must run before the first Metal dispatch"
        );
    }
    current.merge_from(&overrides);
    Ok(())
}

/// Enable or disable the load-time autotune probe for this process (see
/// `crate::autotune`). The probe is disabled by default: at the end of the Metal
/// runtime's `prepare`, a short budget-capped synthetic-workload probe
/// sweeps the output-invariant scheduling knobs not already pinned by an
/// env var or an app override, and adopts winners in-memory for the
/// process. `set_autotune(true)` opts in; the `TRACT_METAL_AUTOTUNE` env
/// var (0/1), when set, wins over this hint. Same contract as
/// [`set_tuning_overrides`]: must run before the first Metal dispatch of
/// the process, fails once the profile has been resolved.
pub fn set_autotune(enable: bool) -> TractResult<()> {
    if RESOLVED.load(Ordering::Acquire) {
        bail!(
            "Metal tuning profile already resolved; \
             set_autotune must run before the first Metal dispatch"
        );
    }
    AUTOTUNE_ENABLED_HINT.store(enable, Ordering::Release);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    /// The default resolution uses the baseline when no overrides are set.
    #[test]
    fn default_profile_is_baseline() {
        let t = MetalTuning::resolve(&MetalTuningOverrides::default(), no_env);
        assert_eq!(t, MetalTuning::BASELINE);
        assert_eq!(t.flash_sdpa_min_t, usize::MAX);
        assert_eq!(t.sdpa_prefill_block, 4096);
        assert_eq!(t.sdpa_split_k_min_t, 8192);
        assert_eq!(t.sdpa_split_k_chunk, 2048);
        assert_eq!(t.commit_every_n_dispatches, 0);
        assert_eq!(t.max_command_buffers_in_flight, 8);
        assert_eq!(t.moe_commit_min_routes, 64);
        assert_eq!(t.moe_grouped_min_routes, 64);
    }

    #[test]
    fn env_override_wins_over_baseline_and_hints() {
        let env = |name: &str| match name {
            "TRACT_METAL_COMMIT_EVERY_N_DISPATCHES" => Some("20".to_string()),
            "TRACT_METAL_MAX_IN_FLIGHT" => Some("2".to_string()),
            "TRACT_METAL_SDPA_PREFILL_BLOCK" => Some("1024".to_string()),
            "TRACT_METAL_MOE_COMMIT_MIN_ROUTES" => Some("32".to_string()),
            _ => None,
        };
        let hints =
            MetalTuningOverrides { commit_every_n_dispatches: Some(10), ..Default::default() };
        let t = MetalTuning::resolve(&hints, env);
        // Env beats the application hint.
        assert_eq!(t.commit_every_n_dispatches, 20);
        assert_eq!(t.max_command_buffers_in_flight, 2);
        assert_eq!(t.sdpa_prefill_block, 1024);
        assert_eq!(t.moe_commit_min_routes, 32);
        // Untouched fields keep the baseline.
        assert_eq!(t.flash_sdpa_min_t, usize::MAX);
        assert_eq!(t.moe_grouped_min_routes, 64);
    }

    #[test]
    fn app_hint_wins_over_baseline() {
        let hints =
            MetalTuningOverrides { commit_every_n_dispatches: Some(10), ..Default::default() };
        let t = MetalTuning::resolve(&hints, no_env);
        assert_eq!(t.commit_every_n_dispatches, 10);
        assert_eq!(t.max_command_buffers_in_flight, 8);
    }

    /// Legacy GPT-OSS-era env name still works, new name wins over it.
    #[test]
    fn flash_min_t_legacy_env_fallback() {
        let legacy_only =
            |name: &str| (name == "TRACT_METAL_GPT_OSS_FLASH_MIN_T").then(|| "4096".to_string());
        let t = MetalTuning::resolve(&MetalTuningOverrides::default(), legacy_only);
        assert_eq!(t.flash_sdpa_min_t, 4096);
        let both = |name: &str| match name {
            "TRACT_METAL_FLASH_SDPA_MIN_T" => Some("0".to_string()),
            "TRACT_METAL_GPT_OSS_FLASH_MIN_T" => Some("4096".to_string()),
            _ => None,
        };
        let t = MetalTuning::resolve(&MetalTuningOverrides::default(), both);
        assert_eq!(t.flash_sdpa_min_t, 0);
    }

    /// Unparseable or out-of-domain env values fall through, matching the
    /// parsing behavior used at each call site (`parse().ok()`, in-flight `>= 1`).
    #[test]
    fn invalid_env_values_fall_through() {
        let env = |name: &str| match name {
            "TRACT_METAL_MAX_IN_FLIGHT" => Some("0".to_string()),
            "TRACT_METAL_COMMIT_EVERY_N_DISPATCHES" => Some("not-a-number".to_string()),
            _ => None,
        };
        let t = MetalTuning::resolve(&MetalTuningOverrides::default(), env);
        assert_eq!(t.max_command_buffers_in_flight, 8);
        assert_eq!(t.commit_every_n_dispatches, 0);
    }

    /// Probe polarity: disabled by default, hint opts in, explicit env wins over
    /// the hint in both directions.
    #[test]
    fn autotune_polarity() {
        assert!(!autotune_enabled_from(None, false), "default must be disabled");
        assert!(autotune_enabled_from(None, true), "hint must opt in");
        assert!(!autotune_enabled_from(Some(false), true), "env 0 must win over the hint");
        assert!(autotune_enabled_from(Some(true), false), "env 1 must opt in");
    }

    /// Knobs supplied by env or app override are pinned: the probe must not
    /// spend budget on them.
    #[test]
    fn pinned_knobs_from_env_and_hints() {
        // Nothing set: nothing pinned.
        let none = PinnedKnobs::compute(&MetalTuningOverrides::default(), no_env);
        assert_eq!(none, PinnedKnobs::default());
        // App override pins its knob.
        let hints =
            MetalTuningOverrides { commit_every_n_dispatches: Some(10), ..Default::default() };
        let p = PinnedKnobs::compute(&hints, no_env);
        assert!(p.commit_every_n_dispatches);
        assert!(!p.max_command_buffers_in_flight);
        assert!(!p.moe_commit_min_routes);
        // Env pins its knob; an out-of-domain value does not.
        let env = |name: &str| match name {
            "TRACT_METAL_MOE_COMMIT_MIN_ROUTES" => Some("32".to_string()),
            "TRACT_METAL_MAX_IN_FLIGHT" => Some("0".to_string()),
            _ => None,
        };
        let p = PinnedKnobs::compute(&MetalTuningOverrides::default(), env);
        assert!(p.moe_commit_min_routes);
        assert!(!p.max_command_buffers_in_flight);
        assert!(!p.commit_every_n_dispatches);
    }

    /// Profile lifecycle on the real process globals: with the default-on
    /// probe the first read leaves the window open, the probe can swap
    /// values, and the freeze makes every mutation path fail; with the
    /// opt-out (env) the first read froze already and only the frozen half
    /// applies.
    #[test]
    fn probe_window_and_freeze_lifecycle() {
        let first = tuning();
        if autotune_enabled() && !is_frozen() {
            // Probe window open: candidates can be swapped in and read back.
            let mut probed = first;
            probed.moe_grouped_min_routes = first.moe_grouped_min_routes + 1;
            probe_set(probed).unwrap();
            assert_eq!(tuning(), probed);
            probe_freeze(first);
        }
        assert!(is_frozen());
        assert_eq!(tuning(), first, "frozen profile must be stable");
        assert!(probe_set(MetalTuning::BASELINE).is_err(), "probe window must be closed");
        assert!(set_tuning_overrides(MetalTuningOverrides::default()).is_err());
        assert!(set_autotune(false).is_err());
        // Freezing again with the same value is an idempotent no-op.
        probe_freeze(first);
        assert_eq!(tuning(), first);
    }
}
