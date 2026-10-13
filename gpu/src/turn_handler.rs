use crate::memory::DeviceMemSchema;
use crate::memory::DeviceMemoryPool;
use crate::tensor::DeviceTensor;
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct DeviceTurnHandler {
    pub mem_schema: DeviceMemSchema,
    /// Set once the arena-resolution failure has been reported, so a decode
    /// loop does not log the same error on every step.
    arena_failure_logged: Arc<std::sync::atomic::AtomicBool>,
}

impl DeviceTurnHandler {
    pub fn from_plan(plan: &TypedSimplePlan, memory_hint: &SymbolValues) -> TractResult<Self> {
        let mem_schema =
            DeviceMemSchema::build(plan.model(), plan.order_without_consts(), memory_hint)?;
        Ok(Self { mem_schema, arena_failure_logged: Arc::new(false.into()) })
    }

    /// The arena a runtime installs on `plan`: sized by `hints` when given, and
    /// without any when every shape in the model is concrete. A symbolic model
    /// with no hints gets none, and its device tensors are allocated one by one.
    /// wgpu does not call it: its dispatches cannot bind two regions of one
    /// arena buffer as a read and a write.
    pub fn for_plan(
        plan: &TypedSimplePlan,
        hints: Option<&SymbolValues>,
    ) -> TractResult<Option<Self>> {
        let no_hints = SymbolValues::default();
        let hints = match hints {
            Some(hints) => hints,
            None if plan
                .model()
                .nodes
                .iter()
                .all(|n| n.outputs.iter().all(|o| o.fact.shape.is_concrete())) =>
            {
                &no_hints
            }
            None => return Ok(None),
        };
        Self::from_plan(plan, hints).context("While sizing memory arena. Missing hint ?").map(Some)
    }
}

impl TurnStateHandler for DeviceTurnHandler {
    fn before_plan_eval(&self, turn: &mut TurnState) -> TractResult<()> {
        // A schema that cannot be resolved yet (e.g. a symbol only known
        // mid-eval) is not a correctness error: ops fall back to per-node
        // allocation when no memory pool is installed. It is a heavy
        // performance loss though, so report it loudly (once per session),
        // and hard-error under TRACT_GPU_STRICT_ARENA=1.
        let resolved_mem_schema = match self.mem_schema.resolve(&turn.resolved_symbols) {
            Ok(schema) => schema,
            Err(e) => {
                if std::env::var("TRACT_GPU_STRICT_ARENA").is_ok_and(|v| v == "1") {
                    return Err(e.context(
                        "TRACT_GPU_STRICT_ARENA=1: device memory arena resolution failed",
                    ));
                }
                if !self.arena_failure_logged.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::error!(
                        "Device memory arena disabled for this session, ops fall back to \
                         per-node allocations (expect a performance loss): {e:?}"
                    );
                }
                return Ok(());
            }
        };
        if std::env::var("TRACT_GPU_LOG_ARENA").is_ok_and(|v| v == "1") {
            log::info!("device memory arena size: {} bytes", resolved_mem_schema.memory_size);
        }
        if turn.shared.get::<crate::memory::ArenaStorageCache>().is_none() {
            turn.shared.insert(crate::memory::ArenaStorageCache::default());
        }
        let cache = turn.shared.get::<crate::memory::ArenaStorageCache>().unwrap();
        let memory_pool = DeviceMemoryPool::from_schema_with_cache(resolved_mem_schema, cache)?;

        turn.shared.insert(memory_pool);
        ensure!(turn.shared.get::<DeviceMemoryPool>().is_some());
        Ok(())
    }

    fn after_plan_eval(&self, turn: &mut TurnState) -> TractResult<()> {
        turn.shared.remove::<DeviceMemoryPool>();
        Ok(())
    }
}

pub fn make_tensor_for_node(
    ctx: &EvalContext,
    dt: DatumType,
    shape: &[usize],
) -> TractResult<DeviceTensor> {
    ctx.shared
        .and_then(|s| s.get::<DeviceMemoryPool>())
        .map(|mem| mem.tensor_for_node(ctx.node_id, dt, shape))
        .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
}

/// Like [`make_tensor_for_node`] but for one output slot of a multi-output
/// node: the memory schema reserves one arena region per device output.
pub fn make_tensor_for_node_output(
    ctx: &EvalContext,
    slot: usize,
    dt: DatumType,
    shape: &[usize],
) -> TractResult<DeviceTensor> {
    ctx.shared
        .and_then(|s| s.get::<DeviceMemoryPool>())
        .map(|mem| mem.tensor_for_node_output(ctx.node_id, slot, dt, shape))
        .unwrap_or_else(|| DeviceTensor::uninitialized_dt(dt, shape))
}

pub fn make_scalar_exotic_tensor_for_node(
    ctx: &EvalContext,
    dt: DatumType,
    exotic_fact: Box<dyn ExoticFact>,
) -> TractResult<DeviceTensor> {
    match ctx.shared.and_then(|s| s.get::<DeviceMemoryPool>()) {
        Some(mem) => mem.scalar_exotic_tensor_for_node(ctx.node_id, dt, exotic_fact),
        None => DeviceTensor::uninitialized_exotic(exotic_fact),
    }
}
