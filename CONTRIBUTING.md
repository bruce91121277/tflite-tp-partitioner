# Contributing

Thank you for your interest. This repository currently contains project documentation while source publication and licensing are under review.

## Before contributing
- Please open an issue to describe a proposed change or reproducible bug.
- Do **not** upload proprietary MediaTek libraries, SDKs, model weights, compiled DLA artifacts, private hardware configurations, or employer-controlled datasets.
- Use self-created, redistributable synthetic fixtures where possible.
- Include expected and observed tensor shapes, dtypes, quantization parameters, model/tool versions, and comparison scope when reporting a numerical mismatch.
- Do not claim a change is bit-exact without independently verifiable, same-backend evidence.

## Pull request guidance (once source is available)
- Keep graph rewrites generic across ranks, layers, channels, and fixtures.
- Add positive and negative regression tests.
- Identify which gates were actually executed and which remain untested.
- Avoid including secret values or personal data in logs, manifests, and screenshots.

## Code of conduct
Be respectful, constructive, and specific about technical evidence.
