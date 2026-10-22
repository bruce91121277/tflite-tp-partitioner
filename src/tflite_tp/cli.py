"""Command-line interface for read-only source graph discovery."""

import argparse
import json

from .discovery import discover_path
from .errors import ToolError
from .npu_canary import comparator_self_test, compare_canary_outputs, prepare_fc_canary
from .pipeline import run_pipeline
from .tflite_rewriter import build_tp2_directory, build_tp4_directory


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tflite-tp",
        description="Audit Canonical Llama 3.1 8B W4A16 TFLite graphs and discover layer projections.",
    )
    parser.add_argument("--prefill", help="canonical P128 TFLite file or directory")
    parser.add_argument("--decode", help="canonical D1 TFLite file or directory")
    parser.add_argument("--tp", choices=("2", "4", "all"), help="tensor parallel degree")
    parser.add_argument("--quant", default="w4a16", help="quantization profile (only w4a16 is supported)")
    parser.add_argument("--strict-bitexact", action="store_true", help="require exact numerical equality before release")
    parser.add_argument("--auto-repair", action="store_true", help="enable bounded canonical rebuilds for diagnosed mismatches")
    parser.add_argument("--max-repairs", type=int, default=3, help="maximum automatic rebuild attempts")
    parser.add_argument("--dry-run", action="store_true", help="analyze sources and emit plans without rewriting graphs")
    parser.add_argument("--resume", action="store_true", help="reuse only candidate outputs whose source and output SHA256 values verify")
    parser.add_argument("--output", help="new work/release root directory")
    parser.add_argument(
        "--fc-width-padding-multiplier",
        type=int,
        help="TP4 Q/K/V width padding: 2 is the verified default; 0 explicitly disables it for diagnostics",
    )
    parser.add_argument(
        "--mlp-gate-dequant-width-padding-multiplier",
        type=int,
        help="TP4 P128 Layer0 MLP Gate rewrite: 2 enables the evidence-scoped graph rewrite; 0 disables it",
    )
    subparsers = parser.add_subparsers(dest="command")
    discover = subparsers.add_parser("discover", help="scan one TFLite file or a directory of source graphs")
    discover.add_argument("--source", required=True, help="Canonical .tflite file or directory")
    discover.add_argument("--expected-layers", type=int, default=32)
    discover.add_argument("--report", help="write the JSON report to this path")
    discover.add_argument(
        "--include-quant-vectors",
        action="store_true",
        help="include full per-tensor scale/zero-point/min/max arrays in the report",
    )
    for command_name, help_text in (
        ("build-tp2", "build TP2 input-channel and row-reduce component candidates"),
        ("build-tp4", "build TP4 output-channel and all-gather component candidates"),
    ):
        build = subparsers.add_parser(command_name, help=help_text)
        build.add_argument("--source", required=True, help="directory of canonical per-layer TFLite sources, or one layer file")
        build.add_argument("--output", required=True, help="new output directory; existing directories are never reused")
        build.add_argument(
            "--profile",
            choices=(("auto", "D1", "P128") if command_name == "build-tp4" else ("D1", "P128")),
            default=("auto" if command_name == "build-tp4" else "D1"),
        )
        build.add_argument("--expected-layers", type=int, default=32)
        if command_name == "build-tp4":
            build.add_argument(
                "--fc-width-padding-multiplier",
                type=int,
                help="TP4 Q/K/V width padding: default 2 applies the Host-verified rule; 0 disables it for diagnostics",
            )
            build.add_argument(
                "--mlp-gate-dequant-width-padding-multiplier",
                type=int,
                help="P128 Layer0 MLP Gate width padding: 2 enables the scoped rewrite; 0 disables it",
            )
    prepare_canary = subparsers.add_parser(
        "prepare-npu-canary",
        help="package canonical, unpadded rank-local, and padded rank-local FC tap models",
    )
    prepare_canary.add_argument("--canonical-model", required=True)
    prepare_canary.add_argument("--unpadded-manifest", required=True)
    prepare_canary.add_argument("--padded-manifest", required=True)
    prepare_canary.add_argument("--input-npz", required=True)
    prepare_canary.add_argument("--input-json", required=True)
    prepare_canary.add_argument("--reference-npz", required=True)
    prepare_canary.add_argument("--reference-json", required=True)
    prepare_canary.add_argument("--output", required=True)
    prepare_canary.add_argument("--layer", type=int, default=0)
    prepare_canary.add_argument("--rank", type=int, default=0)
    prepare_canary.add_argument("--role", choices=("q", "k", "v"), default="q")

    compare_canary = subparsers.add_parser(
        "compare-npu-canary",
        help="compare captured NPU outputs against Canonical and rank-slice references",
    )
    compare_canary.add_argument("--bundle", required=True)
    for label in ("canonical", "unpadded", "padded"):
        compare_canary.add_argument("--{}-npz".format(label), required=True)
        compare_canary.add_argument("--{}-capture-json".format(label), required=True)
    compare_canary.add_argument("--report", required=True)
    subparsers.add_parser("canary-self-test", help="test exact comparator and mismatch localization")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-npu-canary":
        try:
            manifest = prepare_fc_canary(
                canonical_model=args.canonical_model,
                unpadded_manifest_path=args.unpadded_manifest,
                padded_manifest_path=args.padded_manifest,
                input_npz=args.input_npz,
                input_json=args.input_json,
                reference_npz=args.reference_npz,
                reference_json=args.reference_json,
                output_directory=args.output,
                layer=args.layer,
                rank=args.rank,
                role=args.role,
            )
        except ToolError as exc:
            print(json.dumps({"status": "PREPARATION_FAILED", "error": exc.code, "message": exc.message}, indent=2))
            return 2
        print(json.dumps({
            "status": manifest["status"],
            "profile": manifest["profile"],
            "layer": manifest["layer"],
            "rank": manifest["rank"],
            "role": manifest["role"],
            "models": {key: {"path": value["path"], "sha256": value["sha256"], "bytes": value["bytes"]}
                       for key, value in manifest["models"].items()},
            "output": args.output,
            "ncc_compatibility": manifest["ncc_compatibility"],
            "npu_verification_status": manifest["npu_verification_status"],
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "compare-npu-canary":
        try:
            report = compare_canary_outputs(
                args.bundle,
                args.canonical_npz,
                args.unpadded_npz,
                args.padded_npz,
                args.canonical_capture_json,
                args.unpadded_capture_json,
                args.padded_capture_json,
                args.report,
            )
        except ToolError as exc:
            print(json.dumps({"status": "COMPARISON_BLOCKED", "error": exc.code, "message": exc.message}, indent=2))
            return 2
        print(json.dumps({
            "npu_exact_status": report["npu_exact_status"],
            "npu_verification_status": report["npu_verification_status"],
            "mismatch_count": report["mismatch_count"],
            "max_abs_diff": report["max_abs_diff"],
            "first_failure": report["first_failure"],
            "report": args.report,
        }, indent=2, sort_keys=True))
        return 0 if report["npu_exact_status"] == "NPU_EXACT_PASS" else 3
    if args.command == "canary-self-test":
        result = comparator_self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    if args.command in ("build-tp2", "build-tp4"):
        builder = build_tp2_directory if args.command == "build-tp2" else build_tp4_directory
        try:
            profile = args.profile
            discovery_report = None
            if args.command == "build-tp4" and profile == "auto":
                discovery_report = discover_path(args.source, expected_layers=args.expected_layers)
                detected = sorted({
                    layer.profile
                    for source in discovery_report.sources
                    for layer in source.layers
                    if layer.profile in ("D1", "P128")
                })
                if len(detected) != 1:
                    raise ToolError(
                        "PROFILE_AUTO_DETECTION_AMBIGUOUS",
                        "expected exactly one discovered profile, found {}".format(detected),
                    )
                profile = detected[0]
            build_args = {
                "profile": profile,
                "expected_layers": args.expected_layers,
            }
            if args.command == "build-tp4":
                multiplier = args.fc_width_padding_multiplier
                if multiplier is None:
                    multiplier = 2
                if multiplier not in (0, 2):
                    raise ToolError(
                        "FC_WIDTH_PADDING_MULTIPLIER_UNVERIFIED",
                        "only 2 (verified) or 0 (explicitly disabled for diagnostics) is accepted",
                    )
                build_args["fc_width_padding_multiplier"] = multiplier or None
                mlp_multiplier = args.mlp_gate_dequant_width_padding_multiplier
                if mlp_multiplier is None and profile == "P128":
                    mlp_multiplier = 2
                if mlp_multiplier not in (None, 0, 2):
                    raise ToolError(
                        "MLP_GATE_WIDTH_PADDING_MULTIPLIER_UNVERIFIED",
                        "only multiplier 2 or explicit disable 0 is supported",
                    )
                build_args["mlp_gate_dequant_width_padding_multiplier"] = mlp_multiplier or None
                build_args["discovery_report"] = discovery_report
            manifest = builder(args.source, args.output, **build_args)
        except ToolError as exc:
            print(json.dumps({
                "structural_status": "STRUCTURAL_FAIL",
                "error": exc.code,
                "message": exc.message,
                "numerical_status": "NOT_RUN",
                "release_status": "NOT_ELIGIBLE",
            }, indent=2, sort_keys=True))
            return 2
        print(json.dumps({
            "structural_status": manifest["structural_status"],
            "numerical_status": manifest["numerical_status"],
            "release_status": manifest["release_status"],
            "profile": manifest["profile"],
            "host_verification_status": manifest.get("host_verification_status", "NOT_RUN"),
            "npu_verification_status": manifest.get("npu_verification_status", "NPU_UNVERIFIED"),
            "layers": manifest["expected_layers"],
            "model_count": manifest["model_count"],
            "source_coverage": manifest.get("source_coverage"),
            "mismatch_count": manifest.get("mismatch_count"),
            "auto_repair_report": manifest.get("auto_repair_report"),
            "mlp_gate_dequant_width_padding_rule": manifest.get("mlp_gate_dequant_width_padding_rule"),
            "output": args.output,
        }, indent=2, sort_keys=True))
        return 0
    if args.command is None:
        if not args.prefill or not args.decode or not args.tp or not args.output:
            parser.error("top-level execution requires --prefill, --decode, --tp, and --output")
        padding_multiplier = args.fc_width_padding_multiplier
        if padding_multiplier is None and args.tp in ("4", "all"):
            padding_multiplier = 2
        if padding_multiplier not in (None, 0, 2):
            parser.error("--fc-width-padding-multiplier accepts only 2 or 0")
        mlp_multiplier = args.mlp_gate_dequant_width_padding_multiplier
        if mlp_multiplier is None and args.tp in ("4", "all"):
            mlp_multiplier = 2
        if mlp_multiplier not in (None, 0, 2):
            parser.error("--mlp-gate-dequant-width-padding-multiplier accepts only 2 or 0")
        try:
            manifest = run_pipeline(
                prefill=args.prefill,
                decode=args.decode,
                tensor_parallel=args.tp if args.tp == "all" else int(args.tp),
                output_directory=args.output,
                quant=args.quant,
                strict_bitexact=args.strict_bitexact,
                auto_repair=args.auto_repair,
                max_repairs=args.max_repairs,
                dry_run=args.dry_run,
                resume=args.resume,
                fc_width_padding_multiplier=padding_multiplier,
                mlp_gate_dequant_width_padding_multiplier=mlp_multiplier,
            )
        except ToolError as exc:
            print(json.dumps({
                "release_gate": "FAIL",
                "error": exc.code,
                "message": exc.message,
                "release_status": "NOT_ELIGIBLE",
            }, indent=2, sort_keys=True))
            return 2
        summary = {
            "discovery_failures": len(manifest.get("discovery_failures", [])),
            "candidate_sets": manifest.get("candidate_sets", []),
            "numerical_status": manifest.get("numerical_status"),
            "mismatch_count": manifest.get("mismatch_count"),
            "auto_repair_report": manifest.get("auto_repair_report"),
            "release_gate": manifest.get("release_gate"),
            "first_failure": manifest.get("first_failure"),
            "release_status": manifest.get("release_status"),
            "output": args.output,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if manifest.get("discovery_failures"):
            return 2
        if args.dry_run:
            return 0
        return 0 if manifest.get("release_gate") == "PASS" else 3
    if args.command != "discover":
        parser.error("unsupported command")
    try:
        report = discover_path(args.source, expected_layers=args.expected_layers)
    except ToolError as exc:
        print(json.dumps({"status": "STRUCTURAL_FAIL", "error": exc.code, "message": exc.message}, indent=2))
        return 2
    encoded = json.dumps(report.to_dict(include_quant_values=args.include_quant_vectors), indent=2, sort_keys=True)
    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as output:
            output.write(encoded)
            output.write("\n")
        print(
            json.dumps(
                {
                    "structural_status": "STRUCTURAL_PASS" if report.structural_pass else "STRUCTURAL_FAIL",
                    "source_count": len(report.sources),
                    "profile_coverage": report.coverage,
                    "issue_count": len(report.issues),
                    "report": args.report,
                    "numerical_status": "NOT_RUN",
                    "release_status": "NOT_ELIGIBLE",
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(encoded)
    return 0 if report.structural_pass else 1
