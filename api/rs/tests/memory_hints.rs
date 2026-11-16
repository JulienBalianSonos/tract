use tract::prelude::*;

#[test]
fn memory_hints_validate_symbols_without_fixing_runtime_shapes() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    std::fs::write(
        dir.path().join("graph.nnef"),
        "version 1.0;\nextension tract_symbol T;\ngraph identity(x) -> (y) {\n x = external<scalar>(shape = [T]);\n y = copy(x);\n}\n",
    )?;
    let model = tract::nnef()?.load(dir.path())?;
    let runtime = tract::runtime_for_name("default")?;
    assert!(runtime.prepare_with_memory_sizing_hints(model.clone(), &[("missing", 4)]).is_err());
    assert!(runtime.prepare_with_memory_sizing_hints(model.clone(), &[("T", -1)]).is_err());
    let runnable = runtime.prepare_with_memory_sizing_hints(model, &[("T", 4)])?;
    for len in [2, 7] {
        let input = Tensor::from_slice(&[len], &vec![0.5f32; len])?;
        let output = runnable.run([input])?;
        assert_eq!(output[0].shape()?, vec![len]);
    }
    Ok(())
}
