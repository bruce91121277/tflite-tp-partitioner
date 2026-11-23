"""Fail-closed validation and atomic publication of verified partition sets."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .errors import ToolError

_REQUIRED_GOLDEN_STAGES = (
    "Q",
    "K",
    "V",
    "ATTENTION_CONTEXT",
    "O",
    "SWIGLU",
    "DOWN",
    "RESIDUAL_HIDDEN",
    "KV_OUTPUT",
)

# The profile Golden implementation currently executes source and candidate
# graphs through this MTK interpreter. Keep the allowlist explicit so an
# unknown or synthetic runner identity cannot be promoted by the release gate.
_VERIFIED_GOLDEN_BACKENDS = frozenset({
    "mtk_converter.python.tools.tflite_executor.TFLiteExecutor",
})


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def golden_fixture_identity_failure(report, expected_layers):
    """Require one valid deterministic fixture SHA for every covered layer."""
    fixture_hashes = report.get("fixture_sha256_by_layer") if isinstance(report, dict) else None
    if not isinstance(fixture_hashes, dict) or not fixture_hashes:
        return "GOLDEN_FIXTURE_IDENTITY_MISSING"
    try:
        expected_layers = int(expected_layers)
    except (TypeError, ValueError):
        return "GOLDEN_FIXTURE_COVERAGE_INCOMPLETE"
    expected_keys = {str(layer) for layer in range(expected_layers)}
    if set(fixture_hashes) != expected_keys:
        return "GOLDEN_FIXTURE_COVERAGE_INCOMPLETE"
    for fixture_sha in fixture_hashes.values():
        if (
            not isinstance(fixture_sha, str)
            or len(fixture_sha) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in fixture_sha)
        ):
            return "GOLDEN_FIXTURE_SHA_INVALID"
    return None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_golden_report(root, manifest, expected_sources, degree, components):
    report_name = manifest.get("golden_report_path")
    expected_sha = manifest.get("golden_report_sha256")
    if not report_name or not expected_sha:
        return "GOLDEN_REPORT_MISSING"
    report_path = Path(root) / report_name
    if not report_path.is_file():
        return "GOLDEN_REPORT_MISSING"
    if _sha256(report_path) != expected_sha:
        return "GOLDEN_REPORT_SHA256_MISMATCH"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "GOLDEN_REPORT_INVALID"
    if report.get("profile") != manifest.get("profile"):
        return "GOLDEN_PROFILE_MISMATCH"
    if report.get("topology") != manifest.get("topology"):
        return "GOLDEN_TOPOLOGY_MISMATCH"
    if report.get("backend") not in _VERIFIED_GOLDEN_BACKENDS:
        return "GOLDEN_BACKEND_UNVERIFIED"
    if report.get("simulate_fp16") is not False:
        return "GOLDEN_EXECUTION_MODE_UNVERIFIED"
    total_mismatches = report.get("mismatch_count")
    if (
        report.get("numerical_status") != "NUMERICAL_PASS"
        or isinstance(total_mismatches, bool)
        or not isinstance(total_mismatches, int)
        or total_mismatches != 0
        or report.get("mismatches") != []
    ):
        return "GOLDEN_REPORT_NOT_ZERO_MISMATCH"
    if report.get("source_sha256_by_name") != expected_sources:
        return "GOLDEN_SOURCE_MAPPING_MISMATCH"
    candidate_sha256 = report.get("candidate_sha256")
    if not isinstance(candidate_sha256, dict):
        return "GOLDEN_CANDIDATE_SHA_MISSING"
    expected_candidate_sha256 = {}
    for row in manifest.get("models", []):
        if row.get("profile") != manifest.get("profile"):
            continue
        path = row.get("path")
        digest = row.get("output_sha256")
        if not isinstance(path, str) or not path or not _is_sha256(digest):
            return "GOLDEN_CANDIDATE_SHA_INVALID"
        if path in expected_candidate_sha256:
            return "GOLDEN_CANDIDATE_SHA_MAPPING_INVALID"
        expected_candidate_sha256[path] = digest.lower()
    if any(not isinstance(path, str) or not path or not _is_sha256(digest)
           for path, digest in candidate_sha256.items()):
        return "GOLDEN_CANDIDATE_SHA_INVALID"
    normalized_candidate_sha256 = {
        path: digest.lower() for path, digest in candidate_sha256.items()
    }
    if set(normalized_candidate_sha256) != set(expected_candidate_sha256):
        return "GOLDEN_CANDIDATE_SHA_COVERAGE_INCOMPLETE"
    if normalized_candidate_sha256 != expected_candidate_sha256:
        return "GOLDEN_CANDIDATE_SHA_MAPPING_MISMATCH"
    try:
        expected_layers = int(manifest.get("expected_layers", 0))
        degree = int(degree)
    except (TypeError, ValueError):
        return "GOLDEN_COVERAGE_INCOMPLETE"
    if expected_layers < 1 or degree < 1:
        return "GOLDEN_COVERAGE_INCOMPLETE"
    fixture_failure = golden_fixture_identity_failure(report, expected_layers)
    if fixture_failure:
        return fixture_failure
    required_stages_value = report.get("required_stages")
    if not isinstance(required_stages_value, (list, tuple)):
        return "GOLDEN_STAGE_COVERAGE_MISSING"
    if any(not isinstance(stage, str) for stage in required_stages_value):
        return "GOLDEN_STAGE_COVERAGE_MISSING"
    if tuple(required_stages_value) != _REQUIRED_GOLDEN_STAGES:
        return "GOLDEN_STAGE_COVERAGE_MISSING"
    required_stages = set(required_stages_value)
    expected_rows = {
        (layer, rank, stage)
        for layer in range(expected_layers)
        for rank in range(degree)
        for stage in required_stages
    }
    observed = set()
    boundary_results = report.get("boundary_results")
    if not isinstance(boundary_results, list):
        return "GOLDEN_BOUNDARY_COVERAGE_INCOMPLETE"
    for row in boundary_results:
        if not isinstance(row, dict):
            return "GOLDEN_BOUNDARY_COVERAGE_INCOMPLETE"
        layer = row.get("layer")
        rank = row.get("rank")
        stage = row.get("stage")
        if not isinstance(layer, int) or not isinstance(rank, int) or not isinstance(stage, str):
            return "GOLDEN_BOUNDARY_COVERAGE_INCOMPLETE"
        key = (layer, rank, stage)
        row_mismatches = row.get("mismatch_count")
        if (
            isinstance(row_mismatches, bool)
            or not isinstance(row_mismatches, int)
            or row_mismatches != 0
            or key in observed
        ):
            return "GOLDEN_BOUNDARY_MISMATCH_OR_DUPLICATE"
        row_max = row.get("max_absolute_difference")
        if isinstance(row_max, bool) or not isinstance(row_max, (int, float)) or row_max != 0:
            return "GOLDEN_BOUNDARY_NOT_BITEXACT"
        if row.get("comparison_method") != "BITWISE_ELEMENTWISE":
            return "GOLDEN_BOUNDARY_COMPARISON_EVIDENCE_MISSING"
        checks = row.get("checks")
        if not isinstance(checks, list) or not checks:
            return "GOLDEN_BOUNDARY_COMPARISON_EVIDENCE_MISSING"
        check_element_total = 0
        check_mismatch_total = 0
        for check in checks:
            if not isinstance(check, dict):
                return "GOLDEN_BOUNDARY_COMPARISON_EVIDENCE_MISSING"
            elements = check.get("compared_elements")
            if isinstance(elements, bool) or not isinstance(elements, int) or elements <= 0:
                return "GOLDEN_BOUNDARY_ELEMENT_COUNT_INVALID"
            expected_shape = check.get("expected_compared_shape")
            actual_shape = check.get("actual_shape")
            if (
                not isinstance(expected_shape, list)
                or not isinstance(actual_shape, list)
                or expected_shape != actual_shape
                or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in expected_shape)
            ):
                return "GOLDEN_BOUNDARY_COMPARISON_EVIDENCE_MISSING"
            shape_elements = 1
            for dimension in expected_shape:
                shape_elements *= dimension
            if shape_elements != elements or shape_elements <= 0:
                return "GOLDEN_BOUNDARY_ELEMENT_COUNT_INVALID"
            dtype = check.get("dtype")
            if (
                not isinstance(dtype, str)
                or not dtype
                or check.get("expected_dtype") != dtype
                or check.get("actual_dtype") != dtype
            ):
                return "GOLDEN_BOUNDARY_COMPARISON_EVIDENCE_MISSING"
            if check.get("comparison_method") != "BITWISE_ELEMENTWISE":
                return "GOLDEN_BOUNDARY_COMPARISON_EVIDENCE_MISSING"
            expected_hash = check.get("expected_sha256")
            actual_hash = check.get("actual_sha256")
            if (
                not isinstance(expected_hash, str)
                or not isinstance(actual_hash, str)
                or len(expected_hash) != 64
                or len(actual_hash) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in expected_hash + actual_hash)
            ):
                return "GOLDEN_BOUNDARY_COMPARISON_EVIDENCE_MISSING"
            if expected_hash.lower() != actual_hash.lower():
                return "GOLDEN_BOUNDARY_HASH_MISMATCH"
            check_mismatches = check.get("mismatch_count")
            check_max = check.get("max_absolute_difference")
            check_rows = check.get("mismatches")
            if (
                isinstance(check_mismatches, bool)
                or not isinstance(check_mismatches, int)
                or check_mismatches != 0
                or isinstance(check_max, bool)
                or not isinstance(check_max, (int, float))
                or check_max != 0
                or check_rows != []
            ):
                return "GOLDEN_BOUNDARY_NOT_BITEXACT"
            check_element_total += elements
            check_mismatch_total += check_mismatches
        row_elements = row.get("compared_elements")
        if (
            isinstance(row_elements, bool)
            or not isinstance(row_elements, int)
            or row_elements != check_element_total
            or row_mismatches != check_mismatch_total
        ):
            return "GOLDEN_BOUNDARY_ELEMENT_COUNT_INVALID"
        observed.add(key)
    if observed != expected_rows:
        return "GOLDEN_BOUNDARY_COVERAGE_INCOMPLETE"
    coverage = report.get("golden_coverage", {})
    if not isinstance(coverage, dict):
        return "GOLDEN_COVERAGE_INCOMPLETE"
    if coverage.get("complete") is not True or coverage.get("layers") != list(range(expected_layers)):
        return "GOLDEN_COVERAGE_INCOMPLETE"
    if coverage.get("ranks") != list(range(degree)) or coverage.get("components") != list(components):
        return "GOLDEN_COMPONENT_COVERAGE_INCOMPLETE"
    return None


def evaluate_release(candidate_sets):
    """Evaluate one or more (root, manifest) pairs without modifying them."""
    failures = []
    model_count = 0
    if not candidate_sets:
        failures.append(("NO_CANDIDATES", ""))
    for candidate in candidate_sets:
        root = Path(candidate["root"])
        manifest = candidate["manifest"]
        label = str(candidate.get("label", root.name))
        if manifest.get("structural_status") != "STRUCTURAL_PASS":
            failures.append(("STRUCTURAL_VALIDATION_FAILED", label))
        coverage = manifest.get("source_coverage", {})
        if coverage.get("complete") is not True:
            failures.append(("LAYER_COVERAGE_INCOMPLETE", label))
        if manifest.get("weight_reconstruction") != "PASS":
            failures.append(("WEIGHT_RECONSTRUCTION_FAILED", label))
        for role, partition in manifest.get("weight_partitions", {}).items():
            if partition.get("reconstruction_equal") is not True:
                failures.append(("WEIGHT_RECONSTRUCTION_FAILED", "{}:{}".format(label, role)))
        quantization_contract = manifest.get("quantization_contract")
        if not quantization_contract:
            failures.append(("QUANTIZATION_CONTRACT_MISSING", label))
        profile = manifest.get("profile")
        source_audit_hashes = {
            source.get("source"): source.get("source_sha256")
            for source in manifest.get("source_audit", {}).get("sources", [])
            if not profile or profile in source.get("profiles", [])
        }
        if not source_audit_hashes and manifest.get("source") and manifest.get("source_sha256"):
            source_audit_hashes[manifest["source"]] = manifest["source_sha256"]
        for relative_manifest in manifest.get("layer_manifests", []):
            layer_path = root / relative_manifest
            expected_layer_sha = manifest.get("layer_manifest_sha256", {}).get(relative_manifest)
            if not layer_path.is_file() or not expected_layer_sha or _sha256(layer_path) != expected_layer_sha:
                failures.append(("LAYER_MANIFEST_MISSING", "{}:{}".format(label, relative_manifest)))
                continue
            try:
                layer_manifest = json.loads(layer_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                failures.append(("LAYER_MANIFEST_INVALID", "{}:{}".format(label, relative_manifest)))
                continue
            if not layer_manifest.get("quantization_contract"):
                failures.append(("QUANTIZATION_CONTRACT_MISSING", "{}:{}".format(label, relative_manifest)))
            if source_audit_hashes.get(layer_manifest.get("source")) != layer_manifest.get("source_sha256"):
                failures.append(("SOURCE_MAPPING_SHA_MISMATCH", "{}:{}".format(label, relative_manifest)))
            for role, partition in layer_manifest.get("weight_partitions", {}).items():
                if partition.get("reconstruction_equal") is not True:
                    failures.append(("WEIGHT_RECONSTRUCTION_FAILED", "{}:{}:{}".format(label, relative_manifest, role)))
        rows = manifest.get("models", [])
        if not rows:
            failures.append(("MODEL_MANIFEST_MISSING", label))
        topology = manifest.get("topology")
        degree = {"TP2_ROW_REDUCE": 2, "TP2_FOUR_STAGE_ALL_GATHER": 2,
                  "TP4_FOUR_STAGE_ALL_GATHER": 4, "TP8_FOUR_STAGE_ALL_GATHER": 8}.get(topology)
        components = ("attn", "o", "suffix_attn", "mlp", "down", "suffix_final")
        if degree is None:
            failures.append(("TOPOLOGY_CONTRACT_UNSUPPORTED", label))
        else:
            expected_count = int(manifest.get("expected_layers", 0)) * degree * len(components)
            if manifest.get("model_count") != expected_count or len(rows) != expected_count:
                failures.append(("RANK_COMPONENT_COVERAGE_INCOMPLETE", label))
            expected_ranks = list(range(degree))
            if manifest.get("rank_order") != expected_ranks:
                failures.append(("RANK_ORDER_INVALID", label))
            keys = {
                (row.get("layer"), row.get("rank"), row.get("component"))
                for row in rows
            }
            expected_keys = {
                (layer, rank, component)
                for layer in range(int(manifest.get("expected_layers", 0)))
                for rank in expected_ranks
                for component in components
            }
            if keys != expected_keys:
                failures.append(("RANK_COMPONENT_COVERAGE_INVALID", label))
            contract = manifest.get("collective_contract", {})
            if contract.get("kv_cache_policy") != "RANK_LOCAL":
                failures.append(("KV_CACHE_COLLECTIVE_POLICY_INVALID", label))
            if topology != "TP2_ROW_REDUCE":
                stages = [item.get("stage") for item in contract.get("all_gather_boundaries", [])]
                if stages != ["AG0", "AG1", "AG2", "AG3"] or contract.get("row_reduce_boundaries") != []:
                    failures.append(("TP4_COLLECTIVE_CONTRACT_INVALID", label))
            if topology == "TP2_ROW_REDUCE":
                boundaries = contract.get("row_reduce_boundaries", [])
                if (
                    [item.get("name") for item in boundaries] != ["O", "Down"]
                    or any(item.get("operation") != "ALL_REDUCE_SUM" for item in boundaries)
                    or contract.get("all_gather_boundaries") != []
                ):
                    failures.append(("TP2_COLLECTIVE_CONTRACT_INVALID", label))
        for row in rows:
            model_count += 1
            model_path = root / row.get("path", "")
            expected_hash = row.get("output_sha256")
            if (
                not model_path.is_file()
                or not expected_hash
                or not row.get("source_sha256")
                or row.get("source_sha256") not in source_audit_hashes.values()
            ):
                failures.append(("MODEL_FILE_OR_SHA_MISSING", "{}:{}".format(label, row.get("path"))))
                continue
            if _sha256(model_path) != expected_hash:
                failures.append(("MODEL_SHA256_MISMATCH", "{}:{}".format(label, row.get("path"))))
        checksums_name = manifest.get("checksums_file")
        checksums_path = root / checksums_name if checksums_name else None
        if checksums_path is None or not checksums_path.is_file():
            failures.append(("CHECKSUMS_FILE_MISSING", label))
        else:
            checksum_entries = {}
            try:
                for line in checksums_path.read_text(encoding="utf-8").splitlines():
                    digest, relative = line.split(None, 1)
                    checksum_entries[relative.strip()] = digest.lower()
            except (OSError, ValueError):
                failures.append(("CHECKSUMS_FILE_INVALID", label))
            for row in rows:
                if checksum_entries.get(row.get("path")) != str(row.get("output_sha256", "")).lower():
                    failures.append(("CHECKSUMS_FILE_CONTENT_MISMATCH", "{}:{}".format(label, row.get("path"))))
            for relative in manifest.get("layer_manifests", []):
                if checksum_entries.get(relative) != str(manifest.get("layer_manifest_sha256", {}).get(relative, "")).lower():
                    failures.append(("CHECKSUMS_FILE_CONTENT_MISMATCH", "{}:{}".format(label, relative)))
        if manifest.get("numerical_status") != "NUMERICAL_PASS":
            gate = manifest.get("numerical_gate") or {}
            failures.append((gate.get("code") or manifest.get("numerical_status") or "GOLDEN_REPORT_MISSING", label))
        if manifest.get("topology") == "TP2_ROW_REDUCE":
            row_reduce = manifest.get("collective_contract", {}).get("quantized_partial_semantics")
            if not isinstance(row_reduce, dict) or row_reduce.get("status") != "PROVEN" or not row_reduce.get("proof_sha256"):
                failures.append(("ROW_REDUCE_SEMANTICS_UNPROVEN", label))
        if manifest.get("mismatch_count") != 0:
            failures.append(("GOLDEN_MISMATCH_COUNT_NOT_ZERO", label))
        if degree is not None:
            golden_failure = _validate_golden_report(root, manifest, source_audit_hashes, degree, components)
            if golden_failure:
                failures.append((golden_failure, label))
            elif checksums_path is None or not checksums_path.is_file():
                failures.append(("CHECKSUMS_FILE_MISSING", label))
            else:
                golden_path = manifest.get("golden_report_path")
                golden_digest = manifest.get("golden_report_sha256")
                try:
                    checksums = {
                        relative.strip(): digest.lower()
                        for digest, relative in (
                            line.split(None, 1)
                            for line in checksums_path.read_text(encoding="utf-8").splitlines()
                        )
                    }
                except (OSError, ValueError):
                    checksums = {}
                if checksums.get(golden_path) != str(golden_digest or "").lower():
                    failures.append(("CHECKSUMS_GOLDEN_REPORT_MISMATCH", label))
        else:
            failures.append(("GOLDEN_REPORT_MISSING", label))
        repair = manifest.get("auto_repair_report", {})
        if repair.get("status") not in ("NO_REPAIR_NEEDED", "REPAIRED_AND_VERIFIED"):
            failures.append(("AUTO_REPAIR_NOT_VERIFIED", label))
        repair_report_name = manifest.get("auto_repair_report_path")
        repair_report_sha = manifest.get("auto_repair_report_sha256")
        if repair_report_name or repair_report_sha:
            repair_report_path = root / repair_report_name if repair_report_name else None
            if (
                repair_report_path is None
                or not repair_report_path.is_file()
                or not repair_report_sha
                or _sha256(repair_report_path) != str(repair_report_sha).lower()
            ):
                failures.append(("AUTO_REPAIR_REPORT_SHA_INVALID", label))
            elif checksums_path is None or not checksums_path.is_file():
                failures.append(("AUTO_REPAIR_CHECKSUMS_MISSING", label))
            else:
                repair_checksums = {}
                try:
                    for line in checksums_path.read_text(encoding="utf-8").splitlines():
                        digest, relative = line.split(None, 1)
                        repair_checksums[relative.strip()] = digest.lower()
                except (OSError, ValueError):
                    repair_checksums = {}
                if repair_checksums.get(repair_report_name) != str(repair_report_sha).lower():
                    failures.append(("AUTO_REPAIR_CHECKSUM_MISMATCH", label))
        elif repair.get("status") == "REPAIRED_AND_VERIFIED":
            failures.append(("AUTO_REPAIR_REPORT_MISSING", label))
        if manifest.get("release_status") != "RELEASE_PASS":
            failures.append(("CANDIDATE_NOT_RELEASE_PASS", label))

    unique_failures = []
    seen = set()
    for item in failures:
        if item not in seen:
            seen.add(item)
            unique_failures.append(item)
    return {
        "release_gate": "PASS" if not unique_failures else "FAIL",
        "first_failure": unique_failures[0][0] if unique_failures else None,
        "failures": [{"code": code, "candidate": candidate} for code, candidate in unique_failures],
        "model_count": model_count,
        "candidate_count": len(candidate_sets),
    }


def publish_release(candidate_root, release_path, candidate_sets):
    """Copy verified work atomically, refusing every failing or existing target."""
    gate = evaluate_release(candidate_sets)
    if gate["release_gate"] != "PASS":
        raise ToolError("RELEASE_GATE_FAIL", json.dumps(gate, sort_keys=True))
    source = Path(candidate_root)
    target = Path(release_path)
    if not source.is_dir():
        raise ToolError("CANDIDATE_DIRECTORY_MISSING", str(source))
    if target.exists():
        raise ToolError("RELEASE_DIRECTORY_EXISTS", "refusing to reuse {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".{}-release-".format(target.name), dir=str(target.parent)))
    staged = temporary_root / "artifact"
    try:
        staged.mkdir()
        for candidate in candidate_sets:
            source_root = Path(candidate["root"])
            candidate_dir = staged / str(candidate.get("label", source_root.name))
            shutil.copytree(str(source_root), str(candidate_dir))
            for row in candidate["manifest"].get("models", []):
                relative = Path(row["path"])
                source_model = source_root / relative
                staged_model = candidate_dir / relative
                expected = row["output_sha256"]
                if _sha256(source_model) != expected or _sha256(staged_model) != expected:
                    raise ToolError("RELEASE_COPY_SHA256_MISMATCH", str(relative))
        os.replace(str(staged), str(target))
        return {"release_gate": "PASS", "release_path": str(target), "model_count": gate["model_count"]}
    finally:
        shutil.rmtree(str(temporary_root), ignore_errors=True)
