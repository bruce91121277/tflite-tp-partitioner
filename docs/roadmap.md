# Roadmap

This roadmap describes intended work, not implemented or verified functionality.

## Milestone 1 — Public-source readiness
- Review source ownership, employer/IP obligations, and third-party license restrictions.
- Export only independently publishable graph partitioning code into a fresh Git history.
- Remove private model files, proprietary SDK artifacts, credentials, internal fixtures, and identifying infrastructure details.
- Add a suitable OSI-approved license only after publication rights are confirmed.
- Provide pip-installable CLI, synthetic fixtures, documentation, and CI.

## Milestone 2 — TP4 numerical correctness
- Reproduce the outstanding nondegenerate MLP discrepancy on the same NPU backend.
- Investigate dequantization, tensor width, fusion, rounding, and quantization metadata.
- Develop a **general** graph rewrite; do not hard-code fixture-specific output corrections.
- Establish P128 and D1 Layer0 correctness on independent, nondegenerate test fixtures.
- Expand to all 32 layers with per-rank, per-stage golden comparisons.

## Milestone 3 — TP2 and collective correctness
- Verify output-sharded projections and input-sharded O/Down projections.
- Establish accumulator width, bias ownership, reduction order, rounding, saturation, and requantization behavior.
- Run full-profile TP2 tests with fail-closed numerical gates.

## Milestone 4 — Generalization
- Parameterize token profiles, KV/context capacity, layer count, and TP degree.
- Investigate TP8 and portable hardware-independent validation.
- Measure compiled binary size, effective INT4 bytes, memory, and actual runtime latency.

## Definitions of done
A structural pass is **not** a numerical pass. Numerical release requires traceable model and fixture hashes, full input/output mapping, same-backend canonical references, and zero mismatches at all mandatory comparison points. Unmeasured profiles remain NOT_RUN.
