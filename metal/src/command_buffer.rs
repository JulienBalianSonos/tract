use metal::{CommandBuffer, ComputeCommandEncoder, ComputeCommandEncoderRef};
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone)]
pub struct TCommandBuffer {
    inner: CommandBuffer,
    encoder: ComputeCommandEncoder,
}

impl TCommandBuffer {
    pub fn new(command_buffer: CommandBuffer) -> Self {
        let encoder =
            objc::rc::autoreleasepool(|| command_buffer.new_compute_command_encoder().to_owned());

        TCommandBuffer { inner: command_buffer, encoder }
    }

    pub fn encoder(&self) -> &ComputeCommandEncoder {
        &self.encoder
    }

    pub fn encode<EncodeCallback>(&self, encode_cb: EncodeCallback)
    where
        EncodeCallback: Fn(&ComputeCommandEncoderRef),
    {
        encode_cb(&self.encoder);
    }
}

impl Deref for TCommandBuffer {
    type Target = CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TCommandBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use crate::context::MetalStream;
    use metal::foreign_types::ForeignTypeRef;
    use objc::rc::{WeakPtr, autoreleasepool};
    use tract_core::internal::*;

    #[test]
    fn command_objects_released_without_caller_pool_drain() -> TractResult<()> {
        autoreleasepool(|| {
            let stream = MetalStream::new();
            let device = metal::Device::system_default().expect("Metal device");
            let library = device
                .new_library_with_source(
                    r#"
                #include <metal_stdlib>
                using namespace metal;
                kernel void lifetime_test(device const uint *input [[buffer(0)]],
                                          device uint *output [[buffer(1)]],
                                          uint i [[thread_position_in_grid]]) {
                    output[i] = 2 * input[i] + 1;
                }
                "#,
                    &metal::CompileOptions::new(),
                )
                .map_err(|e| anyhow!("{e}"))?;
            let function =
                library.get_function("lifetime_test", None).map_err(|e| anyhow!("{e}"))?;
            for _ in 0..4 {
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| anyhow!("{e}"))?;
                let values = [1u32, 2, 3, 4];
                let zeros = [0u32; 4];
                let input = device.new_buffer_with_data(
                    values.as_ptr().cast(),
                    16,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                let output = device.new_buffer_with_data(
                    zeros.as_ptr().cast(),
                    16,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                let command = stream.command_buffer();
                let weak_command = unsafe { WeakPtr::new(command.as_ptr().cast()) };
                let weak_encoder = unsafe { WeakPtr::new(command.encoder().as_ptr().cast()) };
                command.encode(|encoder| {
                    encoder.set_compute_pipeline_state(&pipeline);
                    encoder.set_buffer(0, Some(&input), 0);
                    encoder.set_buffer(1, Some(&output), 0);
                    encoder.dispatch_thread_groups(
                        metal::MTLSize::new(4, 1, 1),
                        metal::MTLSize::new(1, 1, 1),
                    );
                });
                drop(input);
                drop(pipeline);
                stream.wait_until_completed()?;
                assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
                let result =
                    unsafe { std::slice::from_raw_parts(output.contents().cast::<u32>(), 4) };
                assert_eq!(result, &[3, 5, 7, 9]);
                drop(command);
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                while (!weak_command.load().is_null() || !weak_encoder.load().is_null())
                    && std::time::Instant::now() < deadline
                {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert!(weak_command.load().is_null(), "command retained by outer pool");
                assert!(weak_encoder.load().is_null(), "encoder retained by outer pool");
            }
            Ok(())
        })
    }
}
