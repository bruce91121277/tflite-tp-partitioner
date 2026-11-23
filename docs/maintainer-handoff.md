# Engineering handoff — TFLite Tensor Parallel Partitioner

**Prepared:** 2026-11-19  
**Public baseline:** main at 99500df87ba12fb3b3427a23bd10751364acefe7 (2026-11-18; PR #6 was documentation-only).  
**Project status:** Experimental / model release BLOCKED.  
**Hardware:** MediaTek Genio 720, MDLA 5.3; NCC 8.2.22 and MTK Converter 8.16.0 were used in the cited private historical runs. Those are observations, not a guarantee that a new environment is configured.

## Read this first: separate code, evidence, and reports

1. **Public source and CI:** this repository provides a narrow Llama 3.1 8B W4A16 per-layer TFLite TP2/TP4 structural partitioner, a verifier backend, synthetic tests, and documentation. GitHub Actions on main succeeded on 2026-11-18. This only confirms package/CLI/synthetic tests.
2. **M9 NPU checkpoint:** [M9 evidence](m9-tp4-layer0-npu-evidence-20261027.md) is a **sanitized maintainer report** of 112 fresh invokes and fixture-scoped Layer0 output exactness on the tested P128/D1 candidates. It is **not** full-model or parallel four-EVK proof. The proprietary artifacts and NPU captures are not public.
3. **Later private M10–M18 checkpoints:** the latest source available for those measurements is [Issue #1 discussion](https://github.com/bruce91121277/tflite-tp-partitioner/issues/1), last updated 2026-11-02. These are maintainer-supplied private-run summaries, **not** implemented or independently reproduced by public GitHub CI. Some new failures supersede the narrower M9 single-fixture conclusions.
4. **Newest public change (PR #6):** explains checkpoint download and the external export/PTQ prerequisites; it did not add new numerical repairs or generalize parameter sizes.

See [numerical blockers and first-failure investigation](numerical-blockers.md). If private evidence is not accessible to the successor, report it as **maintainer-reported, awaiting reproduction**, never as independently certified.

## Repository map

| Area | Main files | Responsibility |
| --- | --- | --- |
| CLI | src/tflite_tp/cli.py, pyproject.toml | discover, build-tp2, build-tp4, top-level pipeline, canary commands |
| Input graphs | tflite_reader.py, flatbuffer_adapter.py, model.py | TFLite parse, tensor/quantization contracts |
| Discovery | discovery.py, analysis.py | identify projections, graph features, layers and profiles |
| Planner | planning.py, partition.py | TP2 row-reduce and TP4 four-all-gather policies |
| Rewriter | tflite_rewriter.py, packed_int4.py | rank models, INT4 slices, quant metadata and experimental padding |
| Golden / release | golden.py, diagnostics.py, pipeline.py, release.py | Host Golden, bounded repair, SHA manifests, fail-closed publication |
| MediaTek backend | backends/mediatek/tflite_tp_mediatek/npu_verify/ | compile, idle-gated invoke, capture, compare, stitch and reports |
| Verification | tests/, .github/workflows/tests.yml | synthetic software/unit checks only |
| History | docs/, Issue #1 | M1–M9 documentary evidence and private M10–M18 updates |

No model checkpoint, canonical/rank TFLite, DLA, NVIDIA/MediaTek SDK binary, private fixture, raw capture or remote access credential is included in this public repository.

## Implemented and not implemented

| Capability | Current public-code assessment |
| --- | --- |
| Graph discovery and seven-projection recognition | Implemented for recognized graph patterns |
| W4A16 packed INT4 slicing and quantization metadata rewrite | Implemented for supported encodings |
| TP2 and TP4 candidate generation, layer-directory batches | Implemented, **structural only** |
| P128 and D1 profile support | Implemented for supported Llama 3.1 8B shapes |
| MLP Gate width-padding (TP4 P128) | Experimental; every candidate SHA needs fresh verification |
| Attention context head-padding | Experimental, restricted to TP4 P128 Layer0 |
| Host Golden / bounded repair / SHA gate | Infrastructure exists; requires compatible external executor and complete references |
| NCC/Neuron verifier backend | Code exists; external compiler, device and actual inputs required |
| TP2 row-reduce arithmetic exactness | **UNPROVEN** |
| P128/D1 Full32 NPU exactness | **NOT_VERIFIED** |
| Concurrent four-EVK collectives and full E2E token generation | **NOT_VERIFIED in this repository** |
| Arbitrary Llama size, TP8, W8A16, arbitrary prefill shapes | **NOT_IMPLEMENTED** |
| Direct HF checkpoint -> PTQ -> TFLite -> NPU | **NOT_IMPLEMENTED**; export/PTQ are external |

The TP4 plan has four all-gather boundaries (AG0–AG3) per layer; TP2 is a separate two-row-reduce policy, not an interchangeable proof. The generated per-layer TP4 layout uses four ranks x six component models per layer: **768 per 32-layer profile**; 1,536 for both P128 and D1. This is an expected structural count, not 1,536 validated NPU models.

## Reproducible public software smoke

Requires authorized canonical per-layer W4A16 TFLite files, Python >=3.10 and a fresh output path.

~~~bash
python -m pip install -e .
python -m unittest discover -s tests -v
tflite-tp --help
tflite-tp-verify --help
tflite-tp-verify probe

# Minimal graph smoke:
tflite-tp discover --source ./canonical/D1/layer_00.tflite --expected-layers 1 --report ./audit/d1-l0.json
tflite-tp build-tp4 --source ./canonical/D1/layer_00.tflite --output ./out/tp4-d1-l0 --profile D1 --expected-layers 1

# Only after checking all 32 source files and their manifests:
tflite-tp build-tp4 --source ./canonical/P128 --output ./out/tp4-p128 --profile P128 --expected-layers 32
tflite-tp build-tp4 --source ./canonical/D1 --output ./out/tp4-d1 --profile D1 --expected-layers 32
~~~

These commands do **not** download HF weights, convert/quantize checkpoints, run NCC or certify NPU exactness. The verifier's hardware commands require an explicitly configured authorized backend and fresh read-only device/space/idle gates. Use its help and [backend README](../backends/mediatek/README.md); never run hardware tasks on a busy device.

## Major blockers and immediate order

**P0 — latest D1 Position 32/64 failure (M18):** Scheme B public Hidden mismatch **4,075/4,096** at valid KV=64 (max |diff|=839), **4,090/4,096** at valid KV=32 (max=979); K/V compared exact. The original T0 on the **same new positions** was not run. A later read-only input ABI/SHA audit passed for recorded manifests, but this does **not** prove internal production semantics. First do a controlled **Canonical C0 vs original T0 vs Scheme B** comparison on the same NPU, exact input bytes and coherent per-layer KV state. Confirm repeats/reset/position/mask/valid length and first divergent boundary before another graph rewrite.

**P0 — D1 original production L0:** independent Prompt A/B Hidden **282/4,096** and **266/4,096** (max 1 INT16 LSB); Gate/Up quantized INT16 outputs matched, but isolated FLOAT32 Gate Dequant reported 66/14,336 and 153/14,336 differences. Post-SwiGLU diagnostic substitution is not production repair. Experimental full-head Gate Dequant / Scheme B had three passing position-128 fixtures but failed at 32/64. Root cause and generalization are **UNPROVEN**.

**P1 — P128 L2 rank-local:** original L2 P128 NPU **2,914/786,432** mismatch (max 2 LSB); an instrumented Rank3 q_mul1 showed 11/131,072 one-LSB differences but a diagnostic tap does not prove the first production divergence. Centralized Canonical full-head Q/K/V with Host scatter achieved three-fixture exactness, but is a different experimental **sequential** architecture and not a scalable rank-local or four-EVK fix.

**P1 — D1 L2:** 3/4,096 and 5/4,096 Hidden mismatches on Prompt A/B, K/V exact. Revisit after L0 and L1 chained, multiple-position correctness.

**P1/P2 — full profile:** TP2 row-reduce signed accumulator, bias, rounding, saturation and requant semantics are unresolved; TP4 layers 3–31 and Full32, actual parallel transport, real prompt-to-output and TPS remain unverified. Separate structural, Host, compiler, NPU, and distributed gates.

## Private handoff package checklist (do NOT upload to GitHub)

- Exact approved HF checkpoint revision, model config, PTQ/calibration/quantization and shape-fix/export provenance; verify lineage instead of assuming matching file names.
- SHA256 manifest of **64 canonical P128/D1 TFLites** and complete source path/volume identities; verify readable current files. M12 reported 64/64 restored; M13 reported an additional independent-device copy; M11 had an unresolved file-visibility incident. Do not delete unique artifacts.
- M9 source/candidate/DLA SHA, compiler/runtime version and output contract; M12–M18 experimental candidate scripts, original **C0/T0/Scheme B** binaries and SHA, P128 L2 rank-local/centralized variants, and actual fixture bytes.
- Per-run input, position, valid KV, mask, RoPE, **layer-local** past-K/V, output QDomain/scales, reset and hardware execution manifests; preserve raw FLOAT32 bits when comparing dequant.
- Logs and reports from actual NPU runs, including APUSys failures separately from valid numeric comparisons; SSH device aliases/permissions supplied privately by infrastructure owner, **never** copied into the public repo.
- Restore procedures and storage capacity plan. Verify an independently housed off-host copy before high-volume builds or cleanup.

## Handoff priorities for the next owner

1. **Re-establish evidence and access:** public source SHA, canonical files and candidate hashes; locate the private M18 bundle and verify current authorized idle NPU access. Stop on missing evidence or resource gates.
2. **Reproduce P0 across valid positions:** same-device SHA-matched C0/T0/B for 32/64/128 with coherent caches; record first failure and a causal A/B. Make no undocumented threshold/tolerance workaround.
3. **Close D1 production L0 and L2:** require one untapped integrated candidate to match public Hidden, current K and current V on all recorded and new independent fixtures; then chained L0–L2.
4. **Repair P128 L2 rank-local production** rather than inheriting centralized Q/K/V experimental correctness as rank-local proof.
5. **Extend Full32** with verified layers and real P128->D1 layer-local KV handoff, then test actual concurrent four-EVK collectives and E2E TPS.
6. **Generalize Llama sizes as a separate design track** using a model-parameter IR and graph-derived dimensions, without changing the proven 8B numerical baseline. Source blockers: model.py EXPECTED_WEIGHT_SHAPES (4096/1024/14336), discovery.py classify_profile (4096, 128/1), analysis.py residual 4096, planning.py TP2/TP4-only and hardcoded collective shapes. Add head/GQA mapping, uneven sharding/padding constraints, new profiles and W8A16 only after spec/fixtures exist.

## Release and evidence contract

Never equate **STRUCTURAL_PASS**, GitHub CI PASS, NCC acceptance, Host exactness, an invoke manifest, or zero mismatch on one fixture with model release.

For each numerical claim record: source/canonical/candidate/DLA SHA256; toolchain and backend; input and per-layer KV SHA; exact profile, position, valid length and tensor shape/QDomain; public-output map; full compared element count, mismatch count, max diff and first mismatch; untapped-vs-tapped equivalence status; rank/chained/continuous/distributed scope. **NOT_RUN means mismatch_count=null, never zero.** Label historical evidence as historical and non-public evidence as maintainer-reported. Maintain **Experimental / BLOCKED** until required exact production and distributed gates pass.

Related documents: [numerical verification](numerical-verification.md), [known limitations](known-limitations.md), [roadmap](roadmap.md), [M9 checkpoint](m9-tp4-layer0-npu-evidence-20261027.md), [Issue #1](https://github.com/bruce91121277/tflite-tp-partitioner/issues/1).
