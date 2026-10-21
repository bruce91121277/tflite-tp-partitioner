"""End-to-end source scan, TP2/TP4 candidate build, and fail-closed release flow."""

import hashlib
import json
import os
from pathlib import Path

from .discovery import discover_path
from .diagnostics import (
    AutoRepairEngine,
    annotate_tp4_golden_diagnostics,
    classify_mismatch,
)
from .errors import ToolError
from .golden import (
    MTKConverterTFLiteRunner,
    run_tp2_row_reduce_profile_golden,
    run_tp4_profile_golden,
)
from .planning import make_partition_plan
from .release import (
    _validate_golden_report,
    evaluate_release,
    golden_fixture_identity_failure,
    publish_release,
)
from .tflite_rewriter import build_tp2_directory, build_tp4_directory


_PROFILE_INPUTS = (
    ("P128", "prefill", "prefill128"),
    ("D1", "decode", "decode1"),
)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_artifact_inventory(root, manifest):
    """Hash the existing rank models declared by a candidate manifest.

    The manifest is input to an automatic rebuild, so reject paths that would
    escape the candidate directory. Missing files remain visible as absent
    inventory entries and will count as changed if the rebuild restores them.
    """
    root = Path(root).resolve()
    inventory = {}
    manifest_mismatches = []
    for row in manifest.get("models", []):
        relative_text = str(row.get("path", ""))
        relative = Path(relative_text)
        if not relative_text or relative.is_absolute() or ".." in relative.parts:
            raise ToolError("AUTO_REPAIR_MODEL_PATH_INVALID", relative_text)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ToolError("AUTO_REPAIR_MODEL_PATH_ESCAPE", relative_text) from exc
        if not path.is_file():
            continue
        digest = _file_sha256(path)
        inventory[relative.as_posix()] = digest
        expected = row.get("output_sha256")
        if expected and str(expected).lower() != digest:
            manifest_mismatches.append(relative.as_posix())
    return inventory, sorted(manifest_mismatches)


def _inventory_sha256(inventory):
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_layer_manifests_for_diagnostics(root, manifest):
    root = Path(root)
    declared_hashes = manifest.get("layer_manifest_sha256", {})
    if not isinstance(declared_hashes, dict):
        return {}
    layer_manifests = {}
    for relative in manifest.get("layer_manifests", []):
        expected_sha = declared_hashes.get(relative)
        path = root / relative
        if not expected_sha or not path.is_file():
            continue
        if _file_sha256(path) != str(expected_sha).lower():
            raise ToolError("GOLDEN_LAYER_MANIFEST_SHA_INVALID", str(relative))
        try:
            layer_manifest = json.loads(path.read_text(encoding="utf-8"))
            layer = int(layer_manifest["layer"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ToolError("GOLDEN_LAYER_MANIFEST_INVALID", "{}: {}".format(relative, exc))
        layer_manifests[layer] = layer_manifest
    return layer_manifests


def _finalize_tp4_golden(root, manifest, report):
    """Bind a completed TP4 Golden report into its candidate checksums."""
    root = Path(root)
    if report.get("mismatch_count", 0):
        annotate_tp4_golden_diagnostics(
            report,
            _load_layer_manifests_for_diagnostics(root, manifest),
        )
    report_path = root / "GOLDEN_REPORT.json"
    _write_json(report_path, report)
    report_sha = _file_sha256(report_path)
    mismatch_count = report.get("mismatch_count")
    numerical_status = report.get("numerical_status")
    if not isinstance(mismatch_count, int) or numerical_status not in ("NUMERICAL_PASS", "NUMERICAL_FAIL"):
        raise ToolError("GOLDEN_REPORT_INVALID", "TP4 profile report has invalid status/count")
    fixture_failure = golden_fixture_identity_failure(
        report,
        manifest.get("expected_layers", 0),
    )

    manifest["golden_report_path"] = report_path.name
    manifest["golden_report_sha256"] = report_sha
    manifest["numerical_status"] = numerical_status
    manifest["mismatch_count"] = mismatch_count
    manifest["npu_verification_status"] = "NPU_UNVERIFIED"
    manifest["numerical_gate"] = {
        "code": (
            fixture_failure
            if numerical_status == "NUMERICAL_PASS" and mismatch_count == 0 and fixture_failure
            else None
            if numerical_status == "NUMERICAL_PASS"
            else "GOLDEN_MISMATCH_COUNT_NONZERO"
        ),
        "reason": (
            "Golden fixture identity or layer coverage is incomplete."
            if numerical_status == "NUMERICAL_PASS" and mismatch_count == 0 and fixture_failure
            else "All required per-rank TP4 boundaries were compared."
            if numerical_status == "NUMERICAL_PASS"
            else "Golden mismatches are recorded in GOLDEN_REPORT.json."
        ),
        "backend": report.get("backend"),
        "verification_scope": "HOST_CANONICAL_GOLDEN",
    }
    if numerical_status == "NUMERICAL_PASS" and mismatch_count == 0 and not fixture_failure:
        manifest["host_verification_status"] = "HOST_VERIFIED"
        manifest["auto_repair_report"] = {"status": "NO_REPAIR_NEEDED", "attempts": [], "first_failure": None}
        manifest["release_status"] = "RELEASE_PASS"
        manifest["release_verification_scope"] = "HOST_CANONICAL_GOLDEN"
    elif numerical_status == "NUMERICAL_PASS" and mismatch_count == 0:
        manifest["host_verification_status"] = "HOST_VERIFICATION_INCOMPLETE"
        manifest["auto_repair_report"] = {
            "status": "BLOCKED",
            "attempts": [],
            "first_failure": fixture_failure,
            "reason": "A numerical pass without complete fixture SHA evidence is not eligible for release.",
        }
        manifest["release_status"] = "NOT_ELIGIBLE"
    else:
        manifest["host_verification_status"] = "HOST_VERIFICATION_FAILED"
        first = (report.get("mismatches") or [None])[0]
        diagnosis = (
            first.get("diagnosis")
            if first and isinstance(first.get("diagnosis"), dict)
            else classify_mismatch(first)
            if first
            else None
        )
        category = diagnosis["category"] if diagnosis else "UNSUPPORTED_OPERATOR_SEMANTICS"
        manifest["auto_repair_report"] = {
            "status": "BLOCKED",
            "attempts": [],
            "first_failure": "NO_VERIFIED_REPAIR_CALLBACK_FOR_{}".format(category),
            "reason": "No tested canonical rebuild rule applies to this observed numerical mismatch.",
            "diagnosed_mismatch": diagnosis,
        }
        manifest["release_status"] = "NOT_ELIGIBLE"

    checksum_rows = []
    for row in manifest.get("models", []):
        checksum_rows.append((row["output_sha256"], row["path"]))
    for relative, digest in manifest.get("layer_manifest_sha256", {}).items():
        checksum_rows.append((digest, relative))
    checksum_rows.append((report_sha, report_path.name))
    checksum_rows.sort(key=lambda row: row[1])
    (root / manifest.get("checksums_file", "SHA256SUMS")).write_text(
        "".join("{}  {}\n".format(digest, relative) for digest, relative in checksum_rows),
        encoding="utf-8",
    )
    _write_json(root / "MANIFEST.json", manifest)
    return manifest


def _finalize_tp2_row_reduce_golden(root, manifest, report):
    """Persist TP2 row-reduce evidence while keeping incomplete coverage closed."""
    root = Path(root)
    report_path = root / "TP2_ROW_REDUCE_GOLDEN.json"
    _write_json(report_path, report)
    report_sha = _file_sha256(report_path)
    mismatch_count = report.get("mismatch_count")
    numerical_status = report.get("numerical_status")
    if not isinstance(mismatch_count, int) or numerical_status not in (
        "NUMERICAL_FAIL",
        "NUMERICAL_INCOMPLETE",
    ):
        raise ToolError("GOLDEN_REPORT_INVALID", "TP2 row-reduce report has invalid status/count")

    semantics = manifest.setdefault("collective_contract", {}).setdefault(
        "quantized_partial_semantics", {"status": "UNPROVEN"}
    )
    semantics["status"] = "UNPROVEN"
    semantics["row_reduce_probe_report_sha256"] = report_sha
    semantics["row_reduce_probe_mismatch_count"] = mismatch_count
    semantics["row_reduce_probe_boundary_count"] = report.get("golden_coverage", {}).get(
        "checked_boundary_count", 0
    )

    manifest["golden_report_path"] = report_path.name
    manifest["golden_report_sha256"] = report_sha
    manifest["numerical_status"] = numerical_status
    manifest["mismatch_count"] = mismatch_count
    manifest["numerical_gate"] = {
        "code": report.get("first_failure"),
        "reason": (
            "TP2 O and Down row-reduce boundaries were executed from canonical activations. "
            "The probe does not cover all six components or chain candidate hidden states across layers."
        ),
        "row_reduce_semantics": "UNPROVEN",
        "checked_boundary_count": report.get("golden_coverage", {}).get("checked_boundary_count"),
        "golden_coverage_complete": False,
    }
    first = (report.get("mismatches") or [None])[0]
    diagnosis = classify_mismatch(first) if first else None
    manifest["auto_repair_report"] = {
        "status": "BLOCKED",
        "attempts": [],
        "first_failure": report.get("first_failure"),
        "reason": (
            "No verified repair callback matches the observed TP2 row-reduce arithmetic."
            if mismatch_count
            else "The checked row-reduce boundaries do not cover the full TP2 Golden contract."
        ),
        "diagnosed_mismatch": (
            {**diagnosis, "root_cause_confirmed": False} if diagnosis else None
        ),
    }
    manifest["release_status"] = "NOT_ELIGIBLE"

    checksum_rows = [
        (row["output_sha256"], row["path"])
        for row in manifest.get("models", [])
    ]
    checksum_rows.extend(
        (digest, relative)
        for relative, digest in manifest.get("layer_manifest_sha256", {}).items()
    )
    checksum_rows.append((report_sha, report_path.name))
    checksum_rows.sort(key=lambda row: row[1])
    (root / manifest.get("checksums_file", "SHA256SUMS")).write_text(
        "".join("{}  {}\n".format(digest, relative) for digest, relative in checksum_rows),
        encoding="utf-8",
    )
    _write_json(root / "MANIFEST.json", manifest)
    return manifest


def _record_golden_execution_failure(root, manifest, code, message):
    root = Path(root)
    collective_contract = manifest.get("collective_contract", {})
    row_reduce_semantics = collective_contract.get("quantized_partial_semantics")
    diagnostic = {
        "schema": 1,
        "profile": manifest.get("profile"),
        "topology": manifest.get("topology"),
        "numerical_status": "GOLDEN_EXECUTION_UNAVAILABLE" if code == "GOLDEN_EXECUTION_UNAVAILABLE" else "GOLDEN_EXECUTION_FAILED",
        "mismatch_count": None,
        "first_failure": code,
        "message": str(message),
        "source_sha256_by_name": _source_hashes_from_manifest(manifest, manifest.get("profile")),
    }
    if manifest.get("topology") == "TP2_ROW_REDUCE":
        diagnostic["quantized_partial_semantics"] = row_reduce_semantics
    diagnostic_path = root / "GOLDEN_DIAGNOSTIC.json"
    _write_json(diagnostic_path, diagnostic)
    diagnostic_sha = _file_sha256(diagnostic_path)
    manifest["golden_report_path"] = None
    manifest["golden_report_sha256"] = None
    manifest["golden_diagnostic_path"] = diagnostic_path.name
    manifest["golden_diagnostic_sha256"] = diagnostic_sha
    manifest["numerical_status"] = diagnostic["numerical_status"]
    manifest["host_verification_status"] = "NOT_RUN"
    manifest["npu_verification_status"] = "NPU_UNVERIFIED"
    manifest["mismatch_count"] = None
    manifest["numerical_gate"] = {
        "code": code,
        "reason": str(message),
        "row_reduce_semantics": (
            row_reduce_semantics.get("status", "UNPROVEN")
            if manifest.get("topology") == "TP2_ROW_REDUCE" and isinstance(row_reduce_semantics, dict)
            else "UNPROVEN"
            if manifest.get("topology") == "TP2_ROW_REDUCE"
            else None
        ),
    }
    manifest["auto_repair_report"] = {
        "status": "BLOCKED",
        "attempts": [],
        "first_failure": code,
        "reason": "Numerical diagnostics could not be completed; no mismatch count was fabricated.",
    }
    manifest["release_status"] = "NOT_ELIGIBLE"
    checksum_rows = [
        (row["output_sha256"], row["path"])
        for row in manifest.get("models", [])
    ]
    checksum_rows.extend(
        (digest, relative)
        for relative, digest in manifest.get("layer_manifest_sha256", {}).items()
    )
    checksum_rows.append((diagnostic_sha, diagnostic_path.name))
    checksum_rows.sort(key=lambda row: row[1])
    (root / manifest.get("checksums_file", "SHA256SUMS")).write_text(
        "".join("{}  {}\n".format(digest, relative) for digest, relative in checksum_rows),
        encoding="utf-8",
    )
    _write_json(root / "MANIFEST.json", manifest)
    return manifest


def _persist_auto_repair_report(root, manifest, repair_report):
    root = Path(root)
    report_path = root / "AUTO_REPAIR_REPORT.json"
    _write_json(report_path, repair_report)
    report_sha = _file_sha256(report_path)
    manifest["auto_repair_report_path"] = report_path.name
    manifest["auto_repair_report_sha256"] = report_sha
    manifest["auto_repair_report"] = repair_report
    checksums_path = root / manifest.get("checksums_file", "SHA256SUMS")
    checksums = {}
    if checksums_path.is_file():
        for line in checksums_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(None, 1)
            checksums[relative.strip()] = digest.lower()
    checksums[report_path.name] = report_sha
    checksums_path.write_text(
        "".join("{}  {}\n".format(digest, relative) for relative, digest in sorted(checksums.items())),
        encoding="utf-8",
    )
    _write_json(root / "MANIFEST.json", manifest)
    return manifest


def _auto_repair_tp4_candidate(
    source_path,
    candidate_root,
    manifest,
    source_report,
    profile,
    expected_layers,
    runner,
    max_repairs,
    fc_width_padding_multiplier=None,
):
    """Rebuild TP4 candidates from a SHA-bound canonical source, then retest."""
    initial_report_path = Path(candidate_root) / manifest.get("golden_report_path", "")
    try:
        initial_report = json.loads(initial_report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError("AUTO_REPAIR_GOLDEN_REPORT_MISSING", str(exc))

    engine = AutoRepairEngine(max_repairs=max_repairs)
    repair_roots = []

    def rebuild_from_canonical(mismatch, expected_source_sha256, repair_rule):
        canonical_hashes = set(_source_hashes(source_report, profile).values())
        if expected_source_sha256 not in canonical_hashes:
            return {}
        try:
            before_inventory, before_manifest_mismatches = _candidate_artifact_inventory(
                candidate_root, manifest
            )
        except ToolError:
            return {}
        repair_root = Path(candidate_root).with_name(
            Path(candidate_root).name + ".auto-repair-{:02d}".format(len(repair_roots) + 1)
        )
        if repair_root.exists():
            return {}
        try:
            rebuilt_manifest = build_tp4_directory(
                source_path,
                repair_root,
                profile=profile,
                expected_layers=expected_layers,
                discovery_report=source_report,
                fc_width_padding_multiplier=fc_width_padding_multiplier,
            )
        except ToolError:
            return {}
        repaired_source_hashes = set(_source_hashes_from_manifest(rebuilt_manifest, profile).values())
        if expected_source_sha256 not in repaired_source_hashes:
            return {}
        after_inventory, after_manifest_mismatches = _candidate_artifact_inventory(
            repair_root, rebuilt_manifest
        )
        changed_paths = sorted(
            path
            for path in set(before_inventory) | set(after_inventory)
            if before_inventory.get(path) != after_inventory.get(path)
        )
        repair_evidence = {
            "action": "REBUILD_RANK_COMPONENTS_FROM_CANONICAL",
            "requested_rule": repair_rule,
            "candidate_inventory_before_sha256": _inventory_sha256(before_inventory),
            "candidate_inventory_after_sha256": _inventory_sha256(after_inventory),
            "candidate_artifact_count_before": len(before_inventory),
            "candidate_artifact_count_after": len(after_inventory),
            "candidate_inventory_complete_after": len(after_inventory) == len(rebuilt_manifest.get("models", [])),
            "changed_candidate_artifact_count": len(changed_paths),
            "changed_candidate_artifact_paths": changed_paths,
            "manifest_sha_mismatch_paths_before": before_manifest_mismatches,
            "manifest_sha_mismatch_paths_after": after_manifest_mismatches,
        }
        repair_roots.append(str(repair_root))
        return {
            "rebuilt_from_canonical": True,
            "repair_applied": bool(changed_paths)
            and repair_evidence["candidate_inventory_complete_after"]
            and not after_manifest_mismatches,
            "source_sha256": expected_source_sha256,
            "repair_rule": repair_rule,
            "explanation": "Rebuilt the declared rank components from the canonical source SHA using the current graph-derived partition rules.",
            "repair_evidence": repair_evidence,
            "candidate_root": str(repair_root),
            "candidate_manifest": rebuilt_manifest,
        }

    def verify_again(rebuild_result, mismatch):
        repair_root = Path(rebuild_result["candidate_root"])
        rebuilt_manifest = rebuild_result["candidate_manifest"]
        try:
            golden_report = run_tp4_profile_golden(
                source_report,
                repair_root,
                rebuilt_manifest,
                profile,
                runner,
            )
        except ToolError as exc:
            return {
                "source_sha256": rebuild_result["source_sha256"],
                "numerical_status": "GOLDEN_EXECUTION_FAILED",
                "mismatch_count": 1,
                "mismatches": [],
                "first_failure": exc.code,
            }
        rebuilt_manifest = _finalize_tp4_golden(repair_root, rebuilt_manifest, golden_report)
        return {
            "source_sha256": rebuild_result["source_sha256"],
            "fixture_sha256_by_layer": golden_report.get("fixture_sha256_by_layer"),
            "numerical_status": golden_report.get("numerical_status"),
            "mismatch_count": golden_report.get("mismatch_count"),
            "mismatches": golden_report.get("mismatches", []),
            "candidate_root": str(repair_root),
            "candidate_manifest": rebuilt_manifest,
            "golden_report": golden_report,
        }

    repair_result = engine.run(initial_report, rebuild_from_canonical, verify_again)
    repair_result["candidate_roots"] = repair_roots
    if repair_result.get("status") == "REPAIRED_AND_VERIFIED":
        final_report = repair_result.get("final_report", {})
        repaired_root = Path(final_report["candidate_root"])
        backup_root = Path(candidate_root).with_name(Path(candidate_root).name + ".pre-repair")
        if backup_root.exists():
            raise ToolError("AUTO_REPAIR_BACKUP_EXISTS", str(backup_root))
        os.replace(str(candidate_root), str(backup_root))
        try:
            os.replace(str(repaired_root), str(candidate_root))
        except OSError:
            os.replace(str(backup_root), str(candidate_root))
            raise
        repaired_manifest = json.loads((Path(candidate_root) / "MANIFEST.json").read_text(encoding="utf-8"))
        final_report["candidate_root"] = str(candidate_root)
        final_report.pop("candidate_manifest", None)
        repair_result["candidate_roots"] = [
            str(candidate_root) if value == str(repaired_root) else value
            for value in repair_roots
        ]
        repair_result["preserved_pre_repair_candidate"] = str(backup_root)
        return _persist_auto_repair_report(candidate_root, repaired_manifest, repair_result)

    manifest["release_status"] = "NOT_ELIGIBLE"
    return _persist_auto_repair_report(candidate_root, manifest, repair_result)


def _auto_repair_tp2_candidate(
    source_path,
    candidate_root,
    manifest,
    source_report,
    profile,
    expected_layers,
    runner,
    max_repairs,
):
    """Rebuild TP2 shards and retest only the declared O/Down probe scope.

    A zero row-reduce probe is not a full TP2 Golden and never promotes the
    rebuilt model. The original candidate remains in place until a complete
    TP2 numerical contract can be verified.
    """
    initial_report_path = Path(candidate_root) / manifest.get("golden_report_path", "")
    try:
        initial_report = json.loads(initial_report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError("AUTO_REPAIR_GOLDEN_REPORT_MISSING", str(exc))

    def row_reduce_probe_is_exact(report):
        return (
            report.get("verification_scope") == "TP2_ROW_REDUCE_BOUNDARY_PROBE"
            and report.get("row_reduce_probe_exact") is True
            and report.get("mismatch_count") == 0
        )

    engine = AutoRepairEngine(
        max_repairs=max_repairs,
        success_condition=row_reduce_probe_is_exact,
        success_status="REBUILT_AND_RETESTED_PARTIAL_SCOPE",
    )
    repair_roots = []

    def rebuild_from_canonical(mismatch, expected_source_sha256, repair_rule):
        canonical_hashes = set(_source_hashes(source_report, profile).values())
        if expected_source_sha256 not in canonical_hashes:
            return {}
        try:
            before_inventory, before_manifest_mismatches = _candidate_artifact_inventory(
                candidate_root, manifest
            )
        except ToolError:
            return {}
        repair_root = Path(candidate_root).with_name(
            Path(candidate_root).name + ".auto-repair-{:02d}".format(len(repair_roots) + 1)
        )
        if repair_root.exists():
            return {}
        try:
            rebuilt_manifest = build_tp2_directory(
                source_path,
                repair_root,
                profile=profile,
                expected_layers=expected_layers,
                discovery_report=source_report,
            )
        except ToolError:
            return {}
        repaired_source_hashes = set(_source_hashes_from_manifest(rebuilt_manifest, profile).values())
        if expected_source_sha256 not in repaired_source_hashes:
            return {}
        after_inventory, after_manifest_mismatches = _candidate_artifact_inventory(
            repair_root, rebuilt_manifest
        )
        changed_paths = sorted(
            path
            for path in set(before_inventory) | set(after_inventory)
            if before_inventory.get(path) != after_inventory.get(path)
        )
        repair_evidence = {
            "action": "REBUILD_RANK_COMPONENTS_FROM_CANONICAL",
            "requested_rule": repair_rule,
            "candidate_inventory_before_sha256": _inventory_sha256(before_inventory),
            "candidate_inventory_after_sha256": _inventory_sha256(after_inventory),
            "candidate_artifact_count_before": len(before_inventory),
            "candidate_artifact_count_after": len(after_inventory),
            "candidate_inventory_complete_after": len(after_inventory) == len(rebuilt_manifest.get("models", [])),
            "changed_candidate_artifact_count": len(changed_paths),
            "changed_candidate_artifact_paths": changed_paths,
            "manifest_sha_mismatch_paths_before": before_manifest_mismatches,
            "manifest_sha_mismatch_paths_after": after_manifest_mismatches,
        }
        repair_roots.append(str(repair_root))
        return {
            "rebuilt_from_canonical": True,
            "repair_applied": bool(changed_paths)
            and repair_evidence["candidate_inventory_complete_after"]
            and not after_manifest_mismatches,
            "source_sha256": expected_source_sha256,
            "repair_rule": repair_rule,
            "explanation": "Rebuilt TP2 rank components from the mapped canonical source SHA using the graph-derived row-shard rules.",
            "repair_evidence": repair_evidence,
            "candidate_root": str(repair_root),
            "candidate_manifest": rebuilt_manifest,
        }

    def verify_again(rebuild_result, mismatch):
        repair_root = Path(rebuild_result["candidate_root"])
        rebuilt_manifest = rebuild_result["candidate_manifest"]
        try:
            golden_report = run_tp2_row_reduce_profile_golden(
                source_report,
                repair_root,
                rebuilt_manifest,
                profile,
                runner,
            )
        except ToolError as exc:
            return {
                "source_sha256": rebuild_result["source_sha256"],
                "numerical_status": "GOLDEN_EXECUTION_FAILED",
                "mismatch_count": 1,
                "mismatches": [],
                "first_failure": exc.code,
            }
        rebuilt_manifest = _finalize_tp2_row_reduce_golden(repair_root, rebuilt_manifest, golden_report)
        coverage = golden_report.get("golden_coverage", {})
        expected_boundaries = coverage.get("expected_row_reduce_boundary_count")
        checked_boundaries = coverage.get("checked_boundary_count")
        selected_layers = coverage.get("selected_layers", [])
        complete_probe_scope = (
            isinstance(expected_boundaries, int)
            and expected_boundaries > 0
            and checked_boundaries == expected_boundaries
            and selected_layers == list(range(expected_layers))
        )
        probe_exact = (
            complete_probe_scope
            and golden_report.get("numerical_status") == "NUMERICAL_INCOMPLETE"
            and golden_report.get("mismatch_count") == 0
            and golden_report.get("row_reduce_semantics_status") == "UNPROVEN"
        )
        return {
            "source_sha256": rebuild_result["source_sha256"],
            "fixture_sha256_by_layer": golden_report.get("fixture_sha256_by_layer"),
            "numerical_status": golden_report.get("numerical_status"),
            "mismatch_count": golden_report.get("mismatch_count"),
            "mismatches": golden_report.get("mismatches", []),
            "row_reduce_probe_exact": probe_exact,
            "verification_scope": "TP2_ROW_REDUCE_BOUNDARY_PROBE",
            "candidate_root": str(repair_root),
            "candidate_manifest": rebuilt_manifest,
            "golden_report": golden_report,
        }

    repair_result = engine.run(initial_report, rebuild_from_canonical, verify_again)
    repair_result["candidate_roots"] = repair_roots
    repair_result["release_eligible"] = False
    repair_result["candidate_promoted"] = False
    repair_result["verification_scope"] = "TP2_ROW_REDUCE_BOUNDARY_PROBE"
    repair_result["full_tp2_golden_status"] = "INCOMPLETE"
    final_report = repair_result.get("final_report", {})
    final_report.pop("candidate_manifest", None)
    manifest["release_status"] = "NOT_ELIGIBLE"
    return _persist_auto_repair_report(candidate_root, manifest, repair_result)


def _source_hashes(report, profile):
    return {
        source["source"]: source["source_sha256"]
        for source in report.to_dict(include_quant_values=False)["sources"]
        if profile in source["profiles"]
    }


def _verify_resume_candidate(root, expected_report, degree, profile, expected_layers):
    root = Path(root)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ToolError("RESUME_MANIFEST_MISSING", str(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError("RESUME_MANIFEST_INVALID", str(exc))
    if manifest.get("structural_status") != "STRUCTURAL_PASS":
        raise ToolError("RESUME_CANDIDATE_NOT_STRUCTURAL_PASS", str(root))
    if manifest.get("profile") != profile or manifest.get("expected_layers") != expected_layers:
        raise ToolError("RESUME_PROFILE_OR_COVERAGE_MISMATCH", str(root))
    if manifest.get("topology") != ("TP2_ROW_REDUCE" if degree == 2 else "TP4_FOUR_STAGE_ALL_GATHER"):
        raise ToolError("RESUME_TOPOLOGY_MISMATCH", str(root))
    expected_sources = _source_hashes(expected_report, profile)
    actual_sources = _source_hashes_from_manifest(manifest, profile)
    if actual_sources != expected_sources:
        raise ToolError("RESUME_SOURCE_SHA_MISMATCH", str(root))
    expected_models = expected_layers * degree * 6
    rows = manifest.get("models", [])
    if manifest.get("model_count") != expected_models or len(rows) != expected_models:
        raise ToolError("RESUME_MODEL_COVERAGE_MISMATCH", str(root))
    for row in rows:
        path = root / row.get("path", "")
        if not path.is_file() or _file_sha256(path) != row.get("output_sha256"):
            raise ToolError("RESUME_OUTPUT_SHA_MISMATCH", str(row.get("path")))
        if row.get("structural_status") != "STRUCTURAL_PASS" or row.get("source_sha256") not in expected_sources.values():
            raise ToolError("RESUME_MODEL_MANIFEST_INVALID", str(row.get("path")))
    if manifest.get("weight_reconstruction") != "PASS":
        raise ToolError("RESUME_WEIGHT_RECONSTRUCTION_MISSING", str(root))
    checksum_name = manifest.get("checksums_file")
    checksum_path = root / checksum_name if checksum_name else None
    if checksum_path is None or not checksum_path.is_file():
        raise ToolError("RESUME_CHECKSUMS_FILE_MISSING", str(root))
    try:
        checksums = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(None, 1)
            checksums[relative.strip()] = digest.lower()
    except (OSError, ValueError) as exc:
        raise ToolError("RESUME_CHECKSUMS_FILE_INVALID", str(exc))
    for row in rows:
        if checksums.get(row.get("path")) != str(row.get("output_sha256", "")).lower():
            raise ToolError("RESUME_CHECKSUMS_CONTENT_MISMATCH", str(row.get("path")))
    repair_name = manifest.get("auto_repair_report_path")
    repair_sha = manifest.get("auto_repair_report_sha256")
    if repair_name or repair_sha:
        repair_path = root / repair_name if repair_name else None
        if repair_path is None or not repair_path.is_file() or not repair_sha:
            raise ToolError("RESUME_AUTO_REPAIR_REPORT_MISSING", str(root))
        if _file_sha256(repair_path) != str(repair_sha).lower():
            raise ToolError("RESUME_AUTO_REPAIR_REPORT_SHA_MISMATCH", repair_name)
        if checksums.get(repair_name) != str(repair_sha).lower():
            raise ToolError("RESUME_AUTO_REPAIR_CHECKSUM_MISMATCH", repair_name)
    for relative in manifest.get("layer_manifests", []):
        layer_path = root / relative
        expected_layer_hash = manifest.get("layer_manifest_sha256", {}).get(relative)
        if not layer_path.is_file() or not expected_layer_hash or _file_sha256(layer_path) != expected_layer_hash:
            raise ToolError("RESUME_LAYER_MANIFEST_SHA_MISMATCH", str(relative))
        try:
            layer_manifest = json.loads(layer_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ToolError("RESUME_LAYER_MANIFEST_INVALID", "{}: {}".format(relative, exc))
        if expected_sources.get(layer_manifest.get("source")) != layer_manifest.get("source_sha256"):
            raise ToolError("RESUME_LAYER_SOURCE_MAPPING_MISMATCH", str(relative))
        if not layer_manifest.get("quantization_contract"):
            raise ToolError("RESUME_QUANTIZATION_CONTRACT_MISSING", str(relative))
        if checksums.get(relative) != expected_layer_hash.lower():
            raise ToolError("RESUME_CHECKSUMS_CONTENT_MISMATCH", str(relative))
        for role, partition in layer_manifest.get("weight_partitions", {}).items():
            if partition.get("reconstruction_equal") is not True:
                raise ToolError("RESUME_WEIGHT_RECONSTRUCTION_FAILED", "{}:{}".format(relative, role))
    if manifest.get("numerical_status") == "NUMERICAL_PASS":
        if degree != 4:
            raise ToolError("RESUME_GOLDEN_VERIFICATION_UNSUPPORTED", str(degree))
        golden_name = manifest.get("golden_report_path")
        golden_sha = manifest.get("golden_report_sha256")
        golden_path = root / golden_name if golden_name else None
        if golden_path is None or not golden_path.is_file() or not golden_sha:
            raise ToolError("RESUME_GOLDEN_REPORT_MISSING", str(root))
        if _file_sha256(golden_path) != str(golden_sha).lower():
            raise ToolError("RESUME_GOLDEN_REPORT_SHA_MISMATCH", golden_name)
        golden_failure = _validate_golden_report(
            root,
            manifest,
            expected_sources,
            degree,
            ("attn", "o", "suffix_attn", "mlp", "down", "suffix_final"),
        )
        if golden_failure:
            raise ToolError("RESUME_GOLDEN_REPORT_INVALID", golden_failure)
        if checksums.get(golden_name) != str(golden_sha).lower():
            raise ToolError("RESUME_GOLDEN_CHECKSUM_MISMATCH", golden_name)
    return manifest


def _source_hashes_from_manifest(manifest, profile):
    return {
        source["source"]: source["source_sha256"]
        for source in manifest.get("source_audit", {}).get("sources", [])
        if profile in source.get("profiles", [])
    }


def run_pipeline(
    prefill,
    decode,
    tensor_parallel,
    output_directory,
    quant="w4a16",
    strict_bitexact=True,
    auto_repair=False,
    max_repairs=3,
    dry_run=False,
    resume=False,
    expected_layers=32,
    golden_runner=None,
    fc_width_padding_multiplier=None,
):
    """Run discovery, build candidate TFLites, and evaluate release eligibility."""
    if tensor_parallel not in (2, 4, "all"):
        raise ToolError("TP_DEGREE_UNSUPPORTED", "--tp must be 2, 4, or all")
    if fc_width_padding_multiplier is None and tensor_parallel in (4, "all"):
        fc_width_padding_multiplier = 2
    if fc_width_padding_multiplier is not None:
        multiplier = int(fc_width_padding_multiplier)
        if multiplier == 0:
            fc_width_padding_multiplier = None
        elif multiplier != 2:
            raise ToolError(
                "FC_WIDTH_PADDING_MULTIPLIER_UNVERIFIED",
                "only multiplier 2 passed Host regression; use 0 to disable padding for diagnostics",
            )
        elif tensor_parallel not in (4, "all"):
            raise ToolError("FC_WIDTH_PADDING_PROFILE_UNSUPPORTED", "FC width padding requires a TP4 candidate")
        else:
            fc_width_padding_multiplier = multiplier
    if quant.lower() != "w4a16":
        raise ToolError("QUANT_PROFILE_UNSUPPORTED", "only --quant w4a16 is supported")
    if max_repairs < 0:
        raise ToolError("MAX_REPAIRS_INVALID", "--max-repairs cannot be negative")
    output_root = Path(output_directory)
    marker_path = output_root / "TFLITE_TP_RUN.json"
    if output_root.exists() and not resume:
        raise ToolError("OUTPUT_DIRECTORY_EXISTS", "use a new output directory or --resume after SHA validation")
    if output_root.exists() and resume and not marker_path.is_file():
        raise ToolError("RESUME_ROOT_UNOWNED", "existing output directory has no Genio partitioner run marker")
    output_root.mkdir(parents=True, exist_ok=True)
    if not marker_path.exists():
        _write_json(marker_path, {"schema": 1, "tool": "tflite_tp", "resume_requires_sha_validation": True})

    requested_degrees = (2, 4) if tensor_parallel == "all" else (int(tensor_parallel),)
    input_paths = {"prefill": str(prefill), "decode": str(decode)}
    source_reports = {}
    discovery_failures = []
    for profile, input_key, _profile_dir in _PROFILE_INPUTS:
        try:
            report = discover_path(input_paths[input_key], expected_layers=expected_layers)
            source_reports[profile] = report
            if not report.structural_pass:
                codes = sorted({code for code, _message in report.issues})
                profile_coverage = report.coverage.get(profile, {})
                discovery_failures.append(
                    {
                        "profile": profile,
                        "code": (
                            "SOURCE_LAYER_COVERAGE_INCOMPLETE"
                            if not profile_coverage.get("complete")
                            else "SOURCE_DISCOVERY_FAILED"
                        ),
                        "issue_codes": codes,
                        "coverage": profile_coverage,
                    }
                )
        except ToolError as exc:
            discovery_failures.append({"profile": profile, "code": exc.code, "message": exc.message})

    run_manifest = {
        "schema": 1,
        "project": "GENIO720_FULLY_AUTOMATED_TFLITE_PARTITIONER",
        "model": "LLAMA31_8B",
        "quant": quant.lower(),
        "expected_layers": expected_layers,
        "tensor_parallel": list(requested_degrees),
        "input_paths": input_paths,
        "options": {
            "strict_bitexact": bool(strict_bitexact),
            "auto_repair": bool(auto_repair),
            "max_repairs": max_repairs,
            "dry_run": bool(dry_run),
            "resume": bool(resume),
            "fc_width_padding_multiplier": fc_width_padding_multiplier,
        },
        "source_discovery": {
            profile: report.to_dict(include_quant_values=False)
            for profile, report in source_reports.items()
        },
        "discovery_failures": discovery_failures,
        "candidate_sets": [],
        "numerical_status": "NOT_RUN",
        "mismatch_count": None,
        "auto_repair_report": {
            "status": "NOT_RUN",
            "attempts": [],
            "first_failure": None,
        },
        "release_gate": "FAIL",
        "first_failure": None,
        "release_status": "NOT_ELIGIBLE",
    }
    work_root = output_root / "work"
    (work_root / "logs").mkdir(parents=True, exist_ok=True)
    (work_root / "diagnostics").mkdir(parents=True, exist_ok=True)

    plans = []
    if not discovery_failures:
        for profile, _input_key, _profile_dir in _PROFILE_INPUTS:
            report = source_reports[profile]
            for source in report.sources:
                for layer in source.layers:
                    for degree in requested_degrees:
                        plans.append(make_partition_plan(layer, degree=degree).to_dict())
    run_manifest["partition_plans"] = plans

    if dry_run:
        run_manifest["numerical_status"] = "NOT_RUN"
        run_manifest["auto_repair_report"] = {
            "status": "NOT_RUN",
            "attempts": [],
            "first_failure": "DRY_RUN_ONLY",
        }
        run_manifest["first_failure"] = (
            discovery_failures[0]["code"] if discovery_failures else "DRY_RUN_ONLY"
        )
        _write_json(work_root / "DRY_RUN_PLAN.json", run_manifest)
        _write_json(output_root / "RUN_MANIFEST.json", run_manifest)
        return run_manifest

    if discovery_failures:
        run_manifest["first_failure"] = discovery_failures[0]["code"]
        run_manifest["numerical_status"] = "BLOCKED_SOURCE_DISCOVERY"
        run_manifest["auto_repair_report"] = {
            "status": "BLOCKED",
            "attempts": [],
            "first_failure": run_manifest["first_failure"],
        }
        gate = evaluate_release([])
        gate["first_failure"] = run_manifest["first_failure"]
        gate["failures"].insert(0, {"code": run_manifest["first_failure"], "candidate": "input"})
        _write_json(work_root / "logs" / "release_gate.json", gate)
        run_manifest["release_gate"] = gate["release_gate"]
        run_manifest["release_failures"] = gate["failures"]
        _write_json(output_root / "RUN_MANIFEST.json", run_manifest)
        return run_manifest

    builders = {2: build_tp2_directory, 4: build_tp4_directory}
    profile_runner = golden_runner
    profile_runner_error = None
    if profile_runner is None:
        try:
            profile_runner = MTKConverterTFLiteRunner()
        except ToolError as exc:
            profile_runner_error = exc
    candidate_sets = []
    build_failures = []
    for degree in requested_degrees:
        for profile, _input_key, profile_dir in _PROFILE_INPUTS:
            source_path = input_paths["prefill" if profile == "P128" else "decode"]
            label = "tp{}_{}".format(degree, profile_dir)
            candidate_root = work_root / "tp{}".format(degree) / profile_dir
            candidate_root.parent.mkdir(parents=True, exist_ok=True)
            try:
                resumed = False
                if candidate_root.exists():
                    if not resume:
                        raise ToolError("OUTPUT_DIRECTORY_EXISTS", str(candidate_root))
                    manifest = _verify_resume_candidate(
                        candidate_root,
                        source_reports[profile],
                        degree,
                        profile,
                        expected_layers,
                    )
                    expected_padding = (
                        fc_width_padding_multiplier
                        if degree == 4 and profile in ("P128", "D1")
                        else None
                    )
                    observed_padding = manifest.get("fc_width_padding_rule", {}).get("multiplier")
                    if observed_padding != expected_padding:
                        raise ToolError(
                            "RESUME_FC_WIDTH_PADDING_MISMATCH",
                            "candidate uses multiplier {} but current request requires {}".format(
                                observed_padding,
                                expected_padding,
                            ),
                        )
                    resumed = True
                else:
                    build_options = {
                        "profile": profile,
                        "expected_layers": expected_layers,
                        "discovery_report": source_reports[profile],
                    }
                    if degree == 4:
                        build_options["fc_width_padding_multiplier"] = (
                            fc_width_padding_multiplier if profile in ("P128", "D1") else None
                        )
                    manifest = builders[degree](
                        source_path,
                        candidate_root,
                        **build_options,
                    )
                if degree == 2 and not (resumed and manifest.get("numerical_status") == "NUMERICAL_PASS"):
                    if profile_runner is None:
                        failure = profile_runner_error or ToolError(
                            "GOLDEN_EXECUTION_UNAVAILABLE",
                            "no TP2 Golden runner is registered",
                        )
                        manifest = _record_golden_execution_failure(
                            candidate_root,
                            manifest,
                            failure.code,
                            failure.message,
                        )
                    else:
                        try:
                            golden_report = run_tp2_row_reduce_profile_golden(
                                source_reports[profile],
                                candidate_root,
                                manifest,
                                profile,
                                profile_runner,
                            )
                            manifest = _finalize_tp2_row_reduce_golden(
                                candidate_root, manifest, golden_report
                            )
                            if (
                                auto_repair
                                and golden_report.get("numerical_status") == "NUMERICAL_FAIL"
                                and golden_report.get("mismatch_count", 0) > 0
                            ):
                                manifest = _auto_repair_tp2_candidate(
                                    source_path,
                                    candidate_root,
                                    manifest,
                                    source_reports[profile],
                                    profile,
                                    expected_layers,
                                    profile_runner,
                                    max_repairs,
                                )
                        except ToolError as exc:
                            manifest = _record_golden_execution_failure(
                                candidate_root,
                                manifest,
                                exc.code,
                                exc.message,
                            )
                elif degree == 4 and not (resumed and manifest.get("numerical_status") == "NUMERICAL_PASS"):
                    if profile_runner is None:
                        failure = profile_runner_error or ToolError(
                            "GOLDEN_EXECUTION_UNAVAILABLE",
                            "no TP4 Golden runner is registered",
                        )
                        manifest = _record_golden_execution_failure(
                            candidate_root,
                            manifest,
                            failure.code,
                            failure.message,
                        )
                    else:
                        try:
                            golden_report = run_tp4_profile_golden(
                                source_reports[profile],
                                candidate_root,
                                manifest,
                                profile,
                                profile_runner,
                            )
                            manifest = _finalize_tp4_golden(candidate_root, manifest, golden_report)
                            if (
                                auto_repair
                                and golden_report.get("numerical_status") == "NUMERICAL_FAIL"
                                and golden_report.get("mismatch_count", 0) > 0
                            ):
                                manifest = _auto_repair_tp4_candidate(
                                    source_path,
                                    candidate_root,
                                    manifest,
                                    source_reports[profile],
                                    profile,
                                    expected_layers,
                                    profile_runner,
                                    max_repairs,
                                    fc_width_padding_multiplier=(
                                        fc_width_padding_multiplier
                                        if profile in ("P128", "D1")
                                        else None
                                    ),
                                )
                        except ToolError as exc:
                            manifest = _record_golden_execution_failure(
                                candidate_root,
                                manifest,
                                exc.code,
                                exc.message,
                            )
                candidate_sets.append({"root": str(candidate_root), "manifest": manifest, "label": label})
            except ToolError as exc:
                build_failures.append({"label": label, "code": exc.code, "message": exc.message})

    run_manifest["candidate_sets"] = [
        {
            "label": candidate["label"],
            "root": candidate["root"],
            "structural_status": candidate["manifest"].get("structural_status"),
            "numerical_status": candidate["manifest"].get("numerical_status"),
            "mismatch_count": candidate["manifest"].get("mismatch_count"),
            "release_status": candidate["manifest"].get("release_status"),
            "source_coverage": candidate["manifest"].get("source_coverage"),
            "auto_repair_status": candidate["manifest"].get("auto_repair_report", {}).get("status"),
            "model_count": candidate["manifest"].get("model_count"),
        }
        for candidate in candidate_sets
    ]
    run_manifest["build_failures"] = build_failures
    if candidate_sets:
        numerical_pass = all(
            candidate["manifest"].get("numerical_status") == "NUMERICAL_PASS"
            and candidate["manifest"].get("mismatch_count") == 0
            for candidate in candidate_sets
        )
        known_counts = [candidate["manifest"].get("mismatch_count") for candidate in candidate_sets]
        run_manifest["mismatch_count"] = sum(known_counts) if all(isinstance(value, int) for value in known_counts) else None
        if numerical_pass and not build_failures:
            run_manifest["numerical_status"] = "NUMERICAL_PASS"
            run_manifest["first_failure"] = None
            candidate_repairs = {
                candidate["label"]: candidate["manifest"].get("auto_repair_report", {})
                for candidate in candidate_sets
            }
            repaired = any(
                report.get("status") == "REPAIRED_AND_VERIFIED"
                for report in candidate_repairs.values()
            )
            run_manifest["auto_repair_report"] = {
                "status": "REPAIRED_AND_VERIFIED" if repaired else "NO_REPAIR_NEEDED",
                "attempts": [
                    {"candidate": label, **attempt}
                    for label, report in candidate_repairs.items()
                    for attempt in report.get("attempts", [])
                ],
                "first_failure": None,
                "candidate_reports": candidate_repairs,
            }
        else:
            first_candidate_failure = next(
                (
                    candidate["manifest"].get("numerical_gate", {}).get("code")
                    or candidate["manifest"].get("numerical_status")
                    for candidate in candidate_sets
                    if candidate["manifest"].get("numerical_status") != "NUMERICAL_PASS"
                ),
                None,
            )
            run_manifest["numerical_status"] = (
                candidate_sets[0]["manifest"].get("numerical_status")
                if len({candidate["manifest"].get("numerical_status") for candidate in candidate_sets}) == 1
                else "PROFILE_OR_TOPOLOGY_NUMERICAL_STATUS_MIXED"
            )
            run_manifest["first_failure"] = build_failures[0]["code"] if build_failures else first_candidate_failure
            run_manifest["auto_repair_report"] = {
                "status": "BLOCKED",
                "attempts": [],
                "first_failure": run_manifest["first_failure"],
                "candidate_statuses": {
                    candidate["label"]: candidate["manifest"].get("auto_repair_report", {}).get("status")
                    for candidate in candidate_sets
                },
                "candidate_reports": {
                    candidate["label"]: candidate["manifest"].get("auto_repair_report", {})
                    for candidate in candidate_sets
                },
            }
        gate = evaluate_release(candidate_sets)
        # Host exactness is a prerequisite for device testing, not an NPU release
        # claim. Keep the release path closed until the compiled candidate has a
        # separately recorded exact NPU result.
        tp4_host_exact = (
            4 in requested_degrees
            and all(
                candidate["manifest"].get("host_verification_status") == "HOST_VERIFIED"
                and candidate["manifest"].get("npu_verification_status") != "NPU_VERIFIED"
                for candidate in candidate_sets
                if candidate["manifest"].get("topology") == "TP4_FOUR_STAGE_ALL_GATHER"
            )
            and any(candidate["manifest"].get("topology") == "TP4_FOUR_STAGE_ALL_GATHER" for candidate in candidate_sets)
        )
        if gate["release_gate"] == "PASS" and tp4_host_exact:
            gate["release_gate"] = "BLOCKED_NPU_UNVERIFIED"
            gate["first_failure"] = "NPU_VERIFICATION_NOT_RUN"
            gate["failures"].insert(
                0,
                {"code": "NPU_VERIFICATION_NOT_RUN", "candidate": "TP4"},
            )
        if build_failures:
            gate["release_gate"] = "FAIL"
            gate["first_failure"] = build_failures[0]["code"]
            gate["failures"] = build_failures + gate["failures"]
    else:
        run_manifest["numerical_status"] = "BLOCKED_PARTITION_BUILD"
        run_manifest["mismatch_count"] = None
        run_manifest["first_failure"] = build_failures[0]["code"] if build_failures else "NO_CANDIDATES"
        run_manifest["auto_repair_report"] = {
            "status": "BLOCKED",
            "attempts": [],
            "first_failure": run_manifest["first_failure"],
        }
        gate = evaluate_release([])
        gate["first_failure"] = run_manifest["first_failure"]
        if build_failures:
            gate["failures"].insert(0, build_failures[0])
    for candidate in candidate_sets:
        candidate_manifest = candidate["manifest"]
        candidate_manifest.setdefault("npu_verification_status", "NPU_UNVERIFIED")
        if (
            candidate_manifest.get("topology") == "TP4_FOUR_STAGE_ALL_GATHER"
            and candidate_manifest.get("host_verification_status") == "HOST_VERIFIED"
            and candidate_manifest.get("npu_verification_status") != "NPU_VERIFIED"
        ):
            candidate_manifest["release_status"] = "HOST_VERIFIED_NPU_UNVERIFIED"
        else:
            candidate_manifest["release_status"] = (
                "RELEASE_PASS" if gate["release_gate"] == "PASS" else "NOT_ELIGIBLE"
            )
        _write_json(Path(candidate["root"]) / "MANIFEST.json", candidate_manifest)
    run_manifest["release_gate"] = gate["release_gate"]
    run_manifest["release_failures"] = gate["failures"]
    _write_json(work_root / "logs" / "release_gate.json", gate)

    if gate["release_gate"] == "PASS" and candidate_sets:
        published = publish_release(work_root, output_root / "release", candidate_sets)
        run_manifest["release_status"] = published["release_gate"]
        run_manifest["release_path"] = published["release_path"]
    else:
        run_manifest["release_status"] = "NOT_ELIGIBLE"
    _write_json(output_root / "RUN_MANIFEST.json", run_manifest)
    return run_manifest
