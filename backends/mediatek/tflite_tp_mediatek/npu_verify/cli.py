"""Command line interface for Issue #24 strict NPU Golden verification."""

import argparse
import json
import sys
from pathlib import Path

from .contracts import read_json, resolve_candidate
from .capture_adapter import import_canonical_capture, import_canonical_prefix_capture
from .exact import compare_npu_runs, stitch_tp4_outputs, tp2_row_reduce_gate
from .graph_io_mapper import expose_tensors
from .input_adapter import prepare_rank_inputs
from .report_generator import write_phase3_reports
from .stage_adapter import prepare_stage_inputs
from .transport import TransportError, compile_ncc, device_gate, invoke_npu, probe_environment, recover_ncc_build


def _dump(value, path=None):
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def _add_report(parser):
    parser.add_argument("--report", help="Optional path for a JSON report")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tflite-tp-verify",
        description="Fail-closed TFLite/NCC/Genio NPU numerical validation.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    item = commands.add_parser("probe", help="Probe local verifier prerequisites without hardware access")
    _add_report(item)

    item = commands.add_parser("resolve", help="Resolve one candidate from its checked partition manifest")
    item.add_argument("--manifest", required=True)
    item.add_argument("--tp", type=int, choices=(2, 4), required=True)
    item.add_argument("--profile", required=True)
    item.add_argument("--layer", type=int, required=True)
    item.add_argument("--rank", type=int, required=True)
    item.add_argument("--component", required=True)
    _add_report(item)

    item = commands.add_parser("tap-outputs", help="Expose named intermediate outputs without changing operators")
    item.add_argument("--source", required=True)
    item.add_argument("--destination", required=True)
    item.add_argument("--tensor", action="append", required=True)
    item.add_argument("--source-sha256")
    item.add_argument("--manifest", required=True)
    _add_report(item)

    item = commands.add_parser("prepare-inputs", help="Build rank inputs from SHA-bound canonical fixture/capture")
    for flag in (
        "candidate", "canonical", "canonical-manifest", "input-export-manifest",
        "canonical-input-dir", "canonical-capture-manifest", "canonical-output-dir", "output-dir",
    ):
        item.add_argument("--" + flag, required=True)
    item.add_argument("--rank", type=int, required=True)
    item.add_argument("--tp", type=int, choices=(2, 4), required=True)
    _add_report(item)

    item = commands.add_parser(
        "prepare-stage-inputs",
        help="Build component inputs only from bit-exact TP4 stitched NPU stage outputs",
    )
    for flag in ("candidate", "output-dir", "profile"):
        item.add_argument("--" + flag, required=True)
    item.add_argument(
        "--stitch-config",
        action="append",
        default=[],
        help="Exact TP4 stitch config; repeat once per available upstream tensor",
    )
    item.add_argument(
        "--stitch-report",
        action="append",
        default=[],
        help="Matching exact TP4 stitch report, in the same order as --stitch-config",
    )
    item.add_argument(
        "--component-compare-config",
        action="append",
        default=[],
        help="Verified NPU component comparison context; repeat for available upstream outputs",
    )
    item.add_argument("--public-input-manifest")
    item.add_argument("--public-input-dir")
    item.add_argument("--rank", type=int, required=True)
    item.add_argument("--tp", type=int, choices=(4,), required=True)
    _add_report(item)

    item = commands.add_parser("device-gate", help="Read-only EVK idle and capacity gate")
    item.add_argument("--host", required=True)
    item.add_argument("--remote-root", default="/tmp")
    item.add_argument("--required-bytes", type=int, default=0)
    item.add_argument("--required-memory-bytes", type=int, default=0)
    _add_report(item)

    item = commands.add_parser("import-capture", help="Verify a historical Canonical NPU capture by model/fixture/output SHA")
    item.add_argument("--model", required=True)
    item.add_argument("--capture-manifest", required=True)
    item.add_argument("--output-dir", required=True)
    item.add_argument("--output-manifest", required=True)
    _add_report(item)

    item = commands.add_parser("import-prefix-capture", help="Verify the saved Phase 2G Canonical prefix capture")
    item.add_argument("--prefix-model", required=True)
    item.add_argument("--graph-manifest", required=True)
    item.add_argument("--comparison-manifest", required=True)
    item.add_argument("--canonical-capture-manifest", required=True)
    item.add_argument("--output-dir", required=True)
    item.add_argument("--output-manifest", required=True)
    _add_report(item)

    item = commands.add_parser("compile", help="Compile one TFLite with the locked NCC target")
    for flag in ("source", "host", "ncc-bin", "ncc-lib", "remote-workspace", "output-dir", "run-id"):
        item.add_argument("--" + flag, required=True)
    item.add_argument("--timeout-seconds", type=int, default=1800)
    item.add_argument("--no-verify", action="store_true")
    _add_report(item)

    item = commands.add_parser("recover-compile", help="Collect and verify an already completed remote NCC build; never recompiles")
    for flag in ("source", "host", "ncc-bin", "ncc-lib", "remote-workspace", "output-dir", "run-id"):
        item.add_argument("--" + flag, required=True)
    _add_report(item)

    item = commands.add_parser("invoke", help="Stage and invoke one DLA once, with fresh idle checks")
    for flag in (
        "model", "dla", "compile-manifest", "input-dir", "input-manifest", "host",
        "remote-root", "output-dir", "run-id",
    ):
        item.add_argument("--" + flag, required=True)
    item.add_argument("--timeout-seconds", type=int, default=900)
    item.add_argument("--profile")
    item.add_argument("--rank", type=int)
    _add_report(item)

    item = commands.add_parser("compare", help="Compare every candidate output against same-fixture NPU reference")
    for flag in (
        "candidate-model", "candidate-output-dir", "candidate-run", "reference-model",
        "reference-output-dir", "reference-run",
    ):
        item.add_argument("--" + flag, required=True)
    item.add_argument("--tp", type=int, choices=(2, 4), required=True)
    item.add_argument("--rank", type=int, required=True)
    item.add_argument("--aliases", help="JSON file mapping candidate semantic names to reference names")
    _add_report(item)

    item = commands.add_parser("stitch-tp4", help="Concatenate four exact rank outputs and compare to Canonical NPU")
    item.add_argument("--config", required=True, help="JSON with rank models, run manifests and reference paths")
    _add_report(item)

    item = commands.add_parser("tp2-row-reduce-gate", help="Fail closed until row-reduce contract is evidenced")
    item.add_argument("--contract", required=True)
    _add_report(item)

    item = commands.add_parser("final-report", help="Generate a fail-closed Phase 3 Markdown/JSON gate report")
    item.add_argument("--summary", required=True, help="JSON containing observed gate statuses")
    item.add_argument("--output-dir", required=True)
    item.add_argument("--title", default="Issue #24 Phase 3")
    _add_report(item)
    return parser


def run(args):
    if args.command == "probe":
        return probe_environment()
    if args.command == "resolve":
        model, row, contract = resolve_candidate(
            args.manifest, args.tp, args.profile, args.layer, args.rank, args.component
        )
        return {"status": "STRUCTURE_AND_SHA_PASS", "model": str(model), "row": row, "contract": contract}
    if args.command == "tap-outputs":
        return expose_tensors(args.source, args.destination, args.tensor, args.manifest, args.source_sha256)
    if args.command == "prepare-inputs":
        return prepare_rank_inputs(
            args.candidate, args.canonical, args.canonical_manifest, args.input_export_manifest,
            args.canonical_input_dir, args.canonical_capture_manifest, args.canonical_output_dir,
            args.rank, args.tp, args.output_dir,
        )
    if args.command == "prepare-stage-inputs":
        return prepare_stage_inputs(
            args.candidate, args.stitch_config, args.stitch_report, args.component_compare_config,
            args.public_input_manifest, args.public_input_dir,
            args.output_dir, args.profile, args.tp, args.rank,
        )
    if args.command == "device-gate":
        return device_gate(args.host, args.remote_root, args.required_bytes, args.required_memory_bytes)
    if args.command == "import-capture":
        return import_canonical_capture(args.model, args.capture_manifest, args.output_dir, args.output_manifest)
    if args.command == "import-prefix-capture":
        return import_canonical_prefix_capture(
            args.prefix_model, args.graph_manifest, args.comparison_manifest,
            args.output_dir, args.output_manifest, args.canonical_capture_manifest,
        )
    if args.command == "compile":
        return compile_ncc(
            args.source, args.host, args.ncc_bin, args.ncc_lib, args.remote_workspace,
            args.output_dir, args.run_id, args.timeout_seconds, not args.no_verify,
        )
    if args.command == "recover-compile":
        return recover_ncc_build(
            args.source, args.host, args.ncc_bin, args.ncc_lib, args.remote_workspace,
            args.output_dir, args.run_id,
        )
    if args.command == "invoke":
        return invoke_npu(
            args.model, args.dla, args.compile_manifest, args.input_dir, args.input_manifest,
            args.host, args.remote_root, args.output_dir, args.run_id, args.timeout_seconds,
            args.profile, args.rank,
        )
    if args.command == "compare":
        aliases = read_json(args.aliases) if args.aliases else None
        return compare_npu_runs(
            args.candidate_model, args.candidate_output_dir, args.candidate_run,
            args.reference_model, args.reference_output_dir, args.reference_run, args.tp, args.rank, aliases,
        )
    if args.command == "stitch-tp4":
        config = read_json(args.config)
        return stitch_tp4_outputs(
            config["rank_models"], config["rank_output_dirs"], config["rank_run_manifests"],
            config["tensor_name"], config["reference_model"], config["reference_output_dir"],
            config["reference_run_manifest"], config.get("axis"), config.get("output_path"),
        )
    if args.command == "tp2-row-reduce-gate":
        return tp2_row_reduce_gate(read_json(args.contract))
    if args.command == "final-report":
        return write_phase3_reports(args.output_dir, read_json(args.summary), args.title)
    raise ValueError("UNKNOWN_COMMAND")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
        _dump(result, getattr(args, "report", None))
        failed = result.get("status") in ("FAIL", "UNCONFIRMED", "INCOMPLETE", "UNPROVEN")
        failed |= result.get("exact_gate") == "FAIL"
        failed |= result.get("gate") == "FAIL"
        return 2 if failed else 0
    except (OSError, ValueError, KeyError, TransportError, RuntimeError) as exc:
        result = {
            "schema": "tflite_tp.phase3.command-failure.v1",
            "status": "FAIL_CLOSED",
            "command": args.command,
            "first_failure": str(exc),
        }
        _dump(result, getattr(args, "report", None))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
