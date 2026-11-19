# Numerical blockers and evidence matrix — 2026-11-19

**Project:** tflite-tp-partitioner, Genio 720, Llama 3.1 8B W4A16.  
**Current public main baseline when drafted:** 99500df87ba12fb3b3427a23bd10751364acefe7.  
**Release state:** Experimental / BLOCKED. No general Full32 or distributed NPU exactness claim.

## Evidence provenance

- **Public code** provides TP2/TP4 partitioning for recognized graphs, tests, and an NPU verifier backend. Main GitHub Actions was green 2026-11-18; it tests software, not actual NPU.
- **Public M9 document** ([source](m9-tp4-layer0-npu-evidence-20261027.md)): sanitized account of private, SHA-bound **112 fresh Genio 720 NPU invokes**, fixture-scoped TP4 P128/D1 Layer0 outputs. Public repo cannot replay the binary captures.
- **M10–M18**: [Issue #1](https://github.com/bruce91121277/tflite-tp-partitioner/issues/1) has **maintainer-supplied private evidence summaries through 2026-11-02**. There were no public M18 source changes or CI runs. All claims below from these checkpoints must be **reproduced from retained private artifacts** before promotion to independently validated status.
- **Latest public PR #6 (merged 2026-11-18)** is documentation-only, not a numerical fix.

## Milestone matrix

| Milestone | Scope | Recorded result | What it does not prove |
| --- | --- | --- | --- |
| Phase 3C | Historical TP4 P128 L0 MLP unpadded | **117/458,752** INT16 mismatch, max 1; first [0,4,752] 16 vs 17 | Does not describe M9 candidate SHA |
| Phase 3C padded candidate | Saved four-rank MLP component comparisons | **0/5,505,024**, 12 comparisons, three fixtures; no new invoke during archive audit | Neither full L0 nor later production hashes |
| M9 (Oct 27) | TP4 P128 L0, rank/stitch/tap rechecks | **0/14,155,776** recorded comparisons, repeated elements | Full32, distinct-output count, distributed TP4 |
| M9 (Oct 27) | TP4 D1 L0, rank/stitch/tap rechecks | **0/116,736**, repeated elements | Position generalization, full decode |
| M12 (Oct 28) | P128 L1 original unpadded vs private headpad x4 | Original 53 public differences (first Rank3 attention context [0,44,3327], max 1); padded 0/786,432 on one fixture | The Layer1 padding code is not in public main |
| M13/14 | P128 L2 rank-local NPU vs Canonical | **2,914/786,432**, max 2; public Hidden first [0,4,12] 0 vs 1; K/V exact | A trusted first production operator |
| M15 | P128 L2 centralized full-head Q/K/V + Host scatter | **0/786,432** over three fixtures (maintainer-reported) | Rank-local fix or concurrent 4-EVK |
| M15–M18 | Original T0 D1 L0 Prompt A/B | **282/4,096** and **266/4,096**, max 1; current K/V exact | Exact production L0 for new prompts |
| M18 | Isolated D1 Gate Dequant FLOAT32 on A/B | **66/14,336** and **153/14,336** differences; Gate/Up INT16 outputs exact | A proven NCC/kernel root cause |
| M18 | Experimental D1 L0 Scheme B at position 128 | **0 Hidden/K/V** mismatch on three recorded fixtures | Integrated rank-local production or arbitrary positions |
| M18 | Scheme B new Science64/Garden32 | **4,075/4,096** (max 839) and **4,090/4,096** (max 979) Hidden mismatch; current K/V exact | Root cause: original T0 was NOT_RUN at these exact positions |
| M18 | D1 L2 on Prompt A/B | **3/4,096** and **5/4,096** Hidden mismatch; K/V exact | An exact Full32 decode |
| M18 (Nov 2) | Read-only ABI/source/fixture audit | Recorded INPUT_ABI_EXACT=PASS and GOLDEN_FIXTURE_MATCH=PASS_SHA_BOUND | Per-operator execution equivalence or a new NPU run; EVK5 SSH timed out |
| TP2 | O/Down quantized row reduction | **UNPROVEN** accumulator/bias/order/rounding/saturation/requantization | Production TP2 numerical exactness |

**Do not add counts across unrelated candidate SHAs or compare a saved golden with a different source lineage.** For M9, 14,155,776 and 116,736 are repeated comparison events, **not unique tensors or token-level correctness**.

## B0 — D1 L0 Position 32/64: first discriminating test

**Known evidence:** Scheme B is a centralized full-head Gate Dequant + Host transfer/stitch experimental path; position-128 fixtures passed. Science at position/valid KV=64 had **4,075/4,096** public Hidden mismatches, first [0,0,0], max abs INT16 839. Garden at position/valid KV=32 had **4,090/4,096**, first [0,0,0], max 979. Current K/V exact in those comparisons. A follow-up *read-only* input ABI/golden SHA check passed but no new NPU was invoked; as of 2026-11-02 the controlled original-T0 same-position experiment is **NOT_RUN** and EVK5 idle is UNKNOWN due to SSH timeouts.

**Not yet known:** Whether original unmodified T0 also fails at the same position, whether the discrepancy arises in attention/mask/position/past KV, centralized producer integration, reset/stale state, or quantized MLP. Byte-level input manifest identity narrows hypotheses but does not prove all internal computational semantics.

**Next run (do not modify candidates first):**

1. Authenticate through an authorized device route; verify backend/NCC/runtime versions, SHA of all models/DLA/fixture files, idle/capacity and memory.
2. For each **coherent** position 32 and 64 prompt history, run **untapped Canonical C0**, **original unmodified T0**, and **Scheme B** on the SAME device with EXACT matching actual input bytes, true per-layer past K/V, valid length, position/mask, and fresh state reset. Repeat C0 to establish reference stability; collect full input/output manifests.
3. If T0 passes but B fails, focus on B's producer/host scattering; if both fail, locate their independent earliest discrepancies in position/mask/cache and stage math; if C0 is unstable, fix the reference/environment first.
4. Follow the first reproducible failure through Q/RoPE/context, O/residual, Gate/Up INT16, Dequant FLOAT32, SiLU/SwiGLU, Down/Hidden. Tapped/segmented graphs remain **DIAGNOSTIC** until public-output equivalence to untapped source is demonstrated.
5. Acceptance: one **untapped integrated** candidate, public Hidden/Current-K/Current-V all **0 mismatch** on historical + Prompt A/B + Science64 + Garden32, with at least one additional independent context and real P128 NPU layer-local KV. Never change Golden, patch individual LSBs, or silently loosen an exactness claim.

## B1 — D1 L0 Gate Dequant / SwiGLU shape-dependent arithmetic

- Independent Prompt A/B original T0 public Hidden: 282/4,096 and 266/4,096 mismatch, max 1.
- D1 Gate and Up quantized INT16 boundary values matched. Isolated full-head vs rank-local **FLOAT32 Gate Dequant** differed by 66/14,336 and 153/14,336 elements. Reported example absolute difference was 0.000244140625 (FP32 ULP distance 8192 for the given values); this suggests a numerical-shape effect, **not proof of a specific hardware rounding mode**.
- Diagnostic SwiGLU differed by 5/14,336 and 6/14,336. Replacing downstream TP4 SwiGLU input with Canonical tapped values produced zero stitched Hidden mismatch under the diagnostic substitution. This does NOT identify a production root or release a production model.
- Experimental external full-head Gate Dequant + builtin SiLU / Scheme B matched three position-128 fixtures, but failed badly at positions 32/64. The experimental candidate cannot be used to declare a universal correction.
- A separate diagnostic tap could not pass NCC verification because of unresolved MTKEXT_SILU, and one no-verify run crashed (exit 139). Neither result represents a production NPU mismatch. Find a minimally perturbative supported observation method.

## B2 — P128 Layer2 attention / Q RoPE

- Host matched 0/786,432 on reported fixture; rank-local NPU still differed 2,914/786,432 (max 2). Current K/V exact.
- A tapped Rank3 q_mul1 diagnostic differed **11/131,072** by one LSB, but isolated Canonical QADD disagreed with its own full-graph tap on **171 values** (M14). **Production first-divergent operator is NOT PROVEN.**
- Centralized Canonical full-head Q/K/V with Host Q scatter passed public outputs on three fixtures, but does additional computation/movement and is **not** the original rank-local TP4. Reported Q scatter was 1,048,576 bytes; no concurrent device communication or model TPS measurement.
- Validate untapped-vs-tapped public-output equivalence, identical Q/RoPE operands and quantization, NCC lowering and mapping. Test true rank-local alternatives under controlled A/B before designating any production fix.

## B3 — D1 Layer2 and TP2

- D1 L1 Hidden/K/V matched on recorded Prompt A/B with SHA/QDomain-verified handoff, but L2 Hidden differed **3/4,096** and **5/4,096**; current K/V exact. First trusted production stage remains unknown. Address **after D1 L0 position correctness** and chained L0/L1 input provenance.
- TP2 O/Down row-reduce requires authoritative accumulator width, partial bias ownership, order, signed overflow, saturation, rounding and requantization; no verified proof yet. Do not reuse TP4 all-gather numerical claims.

## Gates to report independently

| Gate | Strict PASS criterion |
| --- | --- |
| STRUCTURAL | Matching canonical source SHA, graph/discovery coverage, reconstructed weights, quant metadata, exact expected component manifest |
| HOST_GOLDEN | Matched canonical/candidate and input fixtures, compatible MTK custom-op executor, documented FP16 simulation behavior, full required output coverage |
| NCC_COMPILE | Same target flags, compiler SHA/version, artifact SHA, successful target compilation; **not** an NPU numeric pass |
| NPU_NUMERICAL | Fresh same-backend untapped candidate/canonical public Hidden/K/V with full count, first failure and representative nonzero fixtures; per-layer chained state |
| POSITION_GENERALIZATION | Exact for coherent contexts (including 32, 64 and 128) and independent prompts, no recycled stale KV or cached results |
| DISTRIBUTED_RUNTIME | Real concurrent four-EVK collectives, rank ordering, KV progression, error/resets and traceable E2E prefill/decode; Host stitch is insufficient |
| PERFORMANCE | Controlled E2E TTFT, prefill/decode TPS, memory and network transfer, correct denominator/workload, no inference from isolated short canary |

**Mismatch_count must be an integer only when the comparison actually ran; use null for NOT_RUN.** Keep experimental PASS_FIXTURE_SCOPED, production exactness and inference/token agreement separate. Widening tolerance may be a research policy later but is NOT a replacement for current bit-exact gate.

See also [engineering handoff](handoff-2026-11-19.md), [M9 evidence](m9-tp4-layer0-npu-evidence-20261027.md), [historical numerical verification](numerical-verification.md), and [Issue #1 M18 updates](https://github.com/bruce91121277/tflite-tp-partitioner/issues/1).
