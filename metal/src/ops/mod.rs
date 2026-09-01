pub mod conv;
pub mod fused_axis_op;
pub mod fused_elementwise;
pub mod gemm;
pub mod multi_gemm;
pub mod pool;

pub use fused_axis_op::MetalFusedAxisOp;
pub use fused_elementwise::MetalFusedElementwise;
pub use gemm::MetalGemm;
pub use multi_gemm::MetalMultiGemm;
