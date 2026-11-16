import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import zipfile


def write_inputs(path, arrays):
    with zipfile.ZipFile(path, "w") as archive:
        for name, shape, values in arrays:
            header = repr({"descr": "<f4", "fortran_order": False, "shape": shape}).encode()
            header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
            data = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
            archive.writestr(name + ".npy", data + struct.pack("<" + "f" * len(values), *values))


cli = shlex.split(os.environ["TRACT_RUN"])
with_sinks = os.environ.get("TEST_SINKS") == "1"
metal = os.environ.get("TEST_METAL") == "1"
symbolic = os.environ.get("TEST_SYMBOLIC") == "1"
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    source = Path("graph.nnef").read_text()
    for dtype in ["f16", "f32"]:
        graph = source.replace("'f16'", repr(dtype))
        if with_sinks:
            graph = graph.replace("mask) ->", "mask, sinks) ->")
            graph = graph.replace("    q16 =", "    sinks = external<scalar>(shape = [4]);\n    q16 =")
            graph = graph.replace("mask16, scale", "mask16, sinks = sinks, scale")
        (root / "graph.nnef").write_text(graph)
        for query_len, key_len in [(1, 13), (1, 4096), (9, 17), (9, 64), (9, 65)]:
            cases = ["nan_mask", "one_nan", "all_masked", "masked_prefix", "finite"]
            cases += [f"masked_{field}_{value}" for field in ["k", "v"] for value in ["nan", "posinf", "neginf"]]
            if with_sinks:
                cases += ["finite_sink_masked", "nan_sink", "positive_inf_sink"]
            for case in cases:
                mask = [0.0] * (4 * query_len * key_len)
                if case in ["nan_mask", "all_masked", "finite_sink_masked"]:
                    mask = [float("nan") if case == "nan_mask" else -float("inf")] * len(mask)
                elif case == "one_nan":
                    mask[key_len - 1] = float("nan")
                elif case == "masked_prefix" or case.startswith("masked_k_") or case.startswith("masked_v_"):
                    mask = [-float("inf") if i % key_len < key_len // 2 else 0.0 for i in range(len(mask))]
                arrays = [
                    ("q", (2, 4, query_len, 64), [0.25] * (2 * 4 * query_len * 64)),
                    ("k", (2, 2, key_len, 64), [0.25] * (2 * 2 * key_len * 64)),
                    ("v", (2, 2, key_len, 64), [0.5] * (2 * 2 * key_len * 64)),
                    ("mask", (1, 4, query_len, key_len), mask),
                ]
                if case.startswith("masked_k_") or case.startswith("masked_v_"):
                    _, field, kind = case.split("_")
                    value = {"nan": float("nan"), "posinf": float("inf"), "neginf": -float("inf")}[kind]
                    array = arrays[1 if field == "k" else 2][2]
                    array[(key_len // 2 - 1) * 64] = value
                if with_sinks:
                    value = {"finite_sink_masked": 0.0, "nan_sink": float("nan"), "positive_inf_sink": float("inf")}.get(case, -float("inf"))
                    sink_values = [value, -float("inf"), 0.0, -2.0] if case in ["nan_sink", "positive_inf_sink"] else [value] * 4
                    arrays.append(("sinks", (4,), sink_values))
                inputs = root / "inputs.npz"
                outputs = root / "outputs.npz"
                write_inputs(inputs, arrays)
                symbols = ["--set", f"S={query_len}", "--set", f"T={key_len}"]
                common = cli + [str(root), "--opl"] + ([] if symbolic else symbols)
                run = ["run"] + (symbols if symbolic else []) + ["--input-from-npz", str(inputs)]
                print(f"{dtype} S={query_len} T={key_len} {case} sinks={with_sinks}", flush=True)
                subprocess.run(common + run + ["--save-outputs-npz", str(outputs)], check=True)
                if metal:
                    subprocess.run(common + ["--metal"] + run + ["--assert-output-bundle", str(outputs), "--approx", "approximate", "--assert-op-count", "MetalMlxSdpa", "1"], check=True)
