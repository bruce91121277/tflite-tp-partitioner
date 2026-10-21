# Contributing

The project is experimental. Contributions should preserve fail-closed graph
validation and report numerical evidence without overstating it.

## Development checks

    python -m pip install -e .
    python -m unittest discover -s tests -v
    tflite-tp --help
    tflite-tp-verify --help
    python scripts/audit_publication.py --staged --output audit/publication_scan.json

Tests must use synthetic graphs, generated arrays, or mocked transport. Do not
add private model artifacts or depend on a physical device in CI.

## Data and security

Do not commit model weights, TFLite models, DLA binaries, runtime libraries,
private NPU captures, fixture exports, SSH configuration, keys, tokens,
passwords, workstation paths, device IP addresses, or confidential third-party licensing documents.
Use synthetic fixtures and sanitized reports. Run the publication audit on the
exact tree before every release.