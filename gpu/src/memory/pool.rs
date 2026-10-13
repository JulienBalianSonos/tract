use crate::device::get_context;
use crate::memory::DeviceResolvedMemSchema;
use crate::tensor::DeviceTensor;
use crate::tensor::DeviceView;
use crate::tensor::OwnedDeviceTensor;

use tract_core::internal::*;

#[derive(Debug)]
pub struct DeviceMemoryPool {
    storage: Arc<Box<dyn OwnedDeviceTensor>>,
    resolved_schema: DeviceResolvedMemSchema,
}

/// Arena storage retained across evaluations in `TurnState.shared`.
///
/// Storage is reused only while the cache holds its sole reference. Escaped
/// views and in-flight command buffers retain their own references and require
/// a fresh allocation. Growth reserves 25% headroom; a demand below half the
/// cached capacity releases that headroom.
#[derive(Debug, Default)]
pub struct ArenaStorageCache {
    #[allow(clippy::type_complexity)]
    storage: std::sync::Mutex<Option<(Arc<Box<dyn OwnedDeviceTensor>>, usize)>>,
}

impl DeviceMemoryPool {
    /// Arena allocations are rounded up to this granularity so consecutive
    /// steps of a growing-context decode request the SAME buffer size: one
    /// storage allocation then serves many steps instead of reallocating and
    /// wiring a fresh multi-MB device buffer per token (an IOGPU kernel trap
    /// on alloc and free, per step, growing with context).
    const ARENA_SIZE_BUCKET: usize = 16 * 1024 * 1024;

    fn bucketed(size: usize) -> usize {
        if size > Self::ARENA_SIZE_BUCKET {
            size.next_multiple_of(Self::ARENA_SIZE_BUCKET)
        } else {
            size
        }
    }

    pub fn from_schema_with_cache(
        resolved_schema: DeviceResolvedMemSchema,
        cache: &ArenaStorageCache,
    ) -> TractResult<Self> {
        let needed = Self::bucketed(resolved_schema.memory_size);
        let mut cached = cache.storage.lock().map_err(|e| anyhow!("{e:?}"))?;
        let storage = match &*cached {
            Some((storage, size))
                if *size >= needed
                    && *size <= needed.saturating_mul(2)
                    && Arc::strong_count(storage) == 1 =>
            {
                Arc::clone(storage)
            }
            _ => {
                // Headroom only helps the growth path; a shrink goes straight
                // to the demanded size.
                let size = if cached.as_ref().is_some_and(|(_, size)| *size < needed) {
                    Self::bucketed(needed.saturating_mul(5) / 4)
                } else {
                    needed
                };
                if std::env::var_os("TRACT_GPU_LOG_ARENA").is_some() {
                    eprintln!(
                        "arena alloc: {:.1} MB (demand {:.1} MB, cached {:.1} MB{})",
                        size as f64 / (1024.0 * 1024.0),
                        needed as f64 / (1024.0 * 1024.0),
                        cached.as_ref().map(|(_, s)| *s).unwrap_or(0) as f64 / (1024.0 * 1024.0),
                        if cached.as_ref().is_some_and(|(s, _)| Arc::strong_count(s) > 1) {
                            ", busy"
                        } else {
                            ""
                        },
                    );
                }
                let storage =
                    Arc::new(get_context()?.uninitialized_device_tensor(&[size], DatumType::U8)?);
                *cached = Some((Arc::clone(&storage), size));
                storage
            }
        };
        Ok(Self { storage, resolved_schema })
    }

    pub fn from_schema(resolved_schema: DeviceResolvedMemSchema) -> TractResult<Self> {
        Self::from_schema_with_cache(resolved_schema, &ArenaStorageCache::default())
    }

    pub fn tensor_for_node(
        &self,
        node_id: usize,
        dt: DatumType,
        shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        if let Some(offsets) = self.resolved_schema.offsets_by_node[node_id].as_ref() {
            ensure!(offsets.len() == 1, "'tensor_for_node' is for mono-output nodes only");
        }
        self.tensor_for_node_output(node_id, 0, dt, shape)
    }

    /// Allocate one output slot of a multi-output node from its arena region.
    pub fn tensor_for_node_output(
        &self,
        node_id: usize,
        slot: usize,
        dt: DatumType,
        shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        match self.resolved_schema.offsets_by_node[node_id].as_ref() {
            Some(offsets) if slot < offsets.len() && offsets[slot].len() == 1 => Ok(DeviceView {
                buffer: Arc::clone(&self.storage),
                dt,
                len: shape.iter().product(),
                shape: shape.into(),
                strides: Tensor::natural_strides(shape),
                offset_bytes: offsets[slot][0],
                exotic_fact: None,
            }
            .into()),
            _ => DeviceTensor::uninitialized_dt(dt, shape),
        }
    }

    pub fn scalar_exotic_tensor_for_node(
        &self,
        node_id: usize,
        dt: DatumType,
        exotic_fact: Box<dyn ExoticFact>,
    ) -> TractResult<DeviceTensor> {
        match self.resolved_schema.offsets_by_node[node_id].as_ref() {
            Some(offsets) => {
                ensure!(
                    offsets.len() == 1 && offsets[0].len() == 2,
                    "'scalar_exotic_tensor_for_node' is for mono-output nodes only"
                );
                Ok(DeviceView {
                    buffer: Arc::clone(&self.storage),
                    dt,
                    len: 1,
                    shape: tvec!(),
                    strides: tvec!(),
                    offset_bytes: offsets[0][1],
                    exotic_fact: Some(exotic_fact.clone()),
                }
                .into())
            }
            None => DeviceTensor::uninitialized_exotic(exotic_fact),
        }
    }
}
