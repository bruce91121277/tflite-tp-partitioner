"""Evidence-based mismatch classification and bounded canonical rebuild loop."""

import hashlib
import json

from .errors import ToolError


MISMATCH_CATEGORIES = (
    "WEIGHT_SLICE_ERROR",
    "INT4_PACKING_ERROR",
    "QUANT_SCALE_MISMATCH",
    "QUANT_ZERO_POINT_MISMATCH",
    "QUANT_METADATA_MISMATCH",
    "BIAS_OWNERSHIP_ERROR",
    "RESIDUAL_DUPLICATION",
    "REQUANT_ROUNDING_MISMATCH",
    "PARTIAL_OUTPUT_RANGE_LIMIT_REACHED",
    "GATHER_CHANNEL_ORDER_ERROR",
    "MASK_SHAPE_MISMATCH",
    "POSITION_ROPE_MISMATCH",
    "KV_LAYOUT_MISMATCH",
    "PAD_ALIGNMENT_MISMATCH",
    "UNSUPPORTED_OPERATOR_SEMANTICS",
)


_CODE_CATEGORIES = {
    "WEIGHT_RECONSTRUCTION_MISMATCH": "INT4_PACKING_ERROR",
    "INT4_BUFFER_LENGTH_MISMATCH": "INT4_PACKING_ERROR",
    "INT4_PADDING_NIBBLE_NONZERO": "INT4_PACKING_ERROR",
    "INT4_SLICE_RANGE_INVALID": "WEIGHT_SLICE_ERROR",
    "PROJECTION_INPUT_SHAPE_MISMATCH": "WEIGHT_SLICE_ERROR",
    "QUANT_GROUP_BOUNDARY_SPLIT": "QUANT_SCALE_MISMATCH",
    "QUANT_SCALE_LAYOUT_UNSUPPORTED": "QUANT_SCALE_MISMATCH",
    "QUANT_SCALE_INVALID": "QUANT_SCALE_MISMATCH",
    "QUANT_ZERO_POINT_COUNT_INVALID": "QUANT_ZERO_POINT_MISMATCH",
    "QUANT_METADATA_LENGTH_MISMATCH": "QUANT_METADATA_MISMATCH",
    "QUANT_RANGE_METADATA_LENGTH_MISMATCH": "QUANT_METADATA_MISMATCH",
    "QUANT_AXIS_OUT_OF_RANGE": "QUANT_METADATA_MISMATCH",
    "BIAS_RECONSTRUCTION_MISMATCH": "BIAS_OWNERSHIP_ERROR",
    "BIAS_OWNERSHIP_RECONSTRUCTION_MISMATCH": "BIAS_OWNERSHIP_ERROR",
    "GATHER_ORDER_MISMATCH": "GATHER_CHANNEL_ORDER_ERROR",
    "MASK_SHAPE_MISMATCH": "MASK_SHAPE_MISMATCH",
    "POSITION_SHAPE_MISMATCH": "POSITION_ROPE_MISMATCH",
    "KV_CACHE_SHAPE_MISMATCH": "KV_LAYOUT_MISMATCH",
    "PAD_ALIGNMENT_MISMATCH": "PAD_ALIGNMENT_MISMATCH",
    "REQUANT_ROUNDING_MISMATCH": "REQUANT_ROUNDING_MISMATCH",
    "UNSUPPORTED_OPERATOR_SEMANTICS": "UNSUPPORTED_OPERATOR_SEMANTICS",
}

AUTO_REPAIR_RULES = {
    "WEIGHT_SLICE_ERROR": "REBUILD_WEIGHT_SLICE_FROM_GRAPH_ROLE",
    "INT4_PACKING_ERROR": "REPACK_INT4_FROM_CANONICAL_NIBBLE_LAYOUT",
    "QUANT_SCALE_MISMATCH": "RESTORE_SCALE_FROM_CANONICAL_TENSOR_METADATA",
    "QUANT_ZERO_POINT_MISMATCH": "RESTORE_ZERO_POINT_FROM_CANONICAL_TENSOR_METADATA",
    "BIAS_OWNERSHIP_ERROR": "RESTORE_BIAS_OWNERSHIP_FROM_PARALLEL_CONTRACT",
    "RESIDUAL_DUPLICATION": "REDERIVE_RESIDUAL_BOUNDARY_FROM_DEPENDENCIES",
    "GATHER_CHANNEL_ORDER_ERROR": "RESTORE_CANONICAL_RANK_ORDER",
    "MASK_SHAPE_MISMATCH": "REDERIVE_MASK_SHAPE_FROM_SOURCE_GRAPH",
    "POSITION_ROPE_MISMATCH": "REDERIVE_POSITION_ROPE_MAPPING_FROM_SOURCE_GRAPH",
    "KV_LAYOUT_MISMATCH": "REDERIVE_RANK_LOCAL_KV_LAYOUT_FROM_SOURCE_GRAPH",
    "PAD_ALIGNMENT_MISMATCH": "RECOMPUTE_PADDING_FROM_TENSOR_CONTRACT",
    "REQUANT_ROUNDING_MISMATCH": None,
    "QUANT_METADATA_MISMATCH": None,
    "PARTIAL_OUTPUT_RANGE_LIMIT_REACHED": None,
    "UNSUPPORTED_OPERATOR_SEMANTICS": None,
}


def _infer_projection(record):
    explicit = str(record.get("projection", "")).strip().lower()
    if explicit in ("q", "k", "v", "o", "gate", "up", "down"):
        return explicit
    text = " ".join(
        str(record.get(key, ""))
        for key in ("candidate_tensor", "tensor_name", "stage")
    ).lower()
    for role in ("q", "k", "v", "o", "gate", "up", "down"):
        if "/{}_proj/".format(role) in text or "{}_projection".format(role) in text:
            return role
    if "curr_key" in text or "past_key" in text or "key_cache" in text:
        return "k"
    if "curr_value" in text or "past_value" in text or "value_cache" in text:
        return "v"
    stage = str(record.get("stage", "")).strip().upper()
    return {"Q": "q", "K": "k", "V": "v", "O": "o", "DOWN": "down"}.get(stage)


def _is_one_unit_quantized_output_mismatch(record):
    if str(record.get("category", "")).upper() != "OUTPUT_BIT_MISMATCH":
        return False
    dtype = str(record.get("dtype", "")).upper()
    quant_domain = record.get("quant_domain")
    if dtype not in ("INT8", "INT16", "INT32", "UINT8", "UINT16", "UINT32"):
        return False
    if not isinstance(quant_domain, dict) or str(quant_domain.get("dtype", "")).upper() != dtype:
        return False
    scales = quant_domain.get("scales") or []
    if not scales:
        return False
    expected = record.get("expected")
    actual = record.get("actual")
    if isinstance(expected, bool) or isinstance(actual, bool):
        return False
    try:
        expected_number = float(expected)
        actual_number = float(actual)
        return (
            expected_number.is_integer()
            and actual_number.is_integer()
            and abs(int(actual_number) - int(expected_number)) == 1
        )
    except (TypeError, ValueError, OverflowError):
        return False


def classify_mismatch(record):
    """Classify a mismatch using tensor, stage, and quantization evidence.

    Returns the original coordinates and values with a stable category. It
    never changes expected values or infers a successful repair.
    """
    result = dict(record)
    code = str(result.get("code", "")).upper()
    category = _CODE_CATEGORIES.get(code)
    projection = _infer_projection(result)
    if projection and not result.get("projection"):
        result["projection"] = projection
    text = " ".join(
        str(result.get(key, ""))
        for key in ("tensor_name", "candidate_tensor", "stage", "operator", "projection", "message")
    ).lower()
    quant_field = str(result.get("quant_field", result.get("metadata_field", ""))).lower()
    if category is None:
        range_diagnostic = result.get("partial_range_diagnostic")
        if (
            isinstance(range_diagnostic, dict)
            and range_diagnostic.get("status") == "RANGE_LIMIT_REACHED"
        ):
            category = "PARTIAL_OUTPUT_RANGE_LIMIT_REACHED"
            result["classification_evidence"] = {
                "basis": "RANK_PARTIAL_VALUE_EQUALS_DTYPE_ENDPOINT",
                "root_cause_confirmed": False,
                "safe_repair_available": False,
            }
        elif _is_one_unit_quantized_output_mismatch(result):
            category = "REQUANT_ROUNDING_MISMATCH"
            result["classification_evidence"] = {
                "basis": "ONE_INTEGER_UNIT_AT_MATCHING_QUANTIZED_OUTPUT_DOMAIN",
                "root_cause_confirmed": False,
                "safe_repair_available": False,
            }
        elif "zero_point" in quant_field or "zero point" in quant_field:
            category = "QUANT_ZERO_POINT_MISMATCH"
        elif "scale" in quant_field:
            category = "QUANT_SCALE_MISMATCH"
        elif "gather" in text or "all_gather" in text or "concat_rank" in text:
            category = "GATHER_CHANNEL_ORDER_ERROR"
        elif "mask" in text or "padding_mask" in text:
            category = "MASK_SHAPE_MISMATCH"
        elif "kv_cache" in text or "past_key" in text or "past_value" in text or "curr_key" in text or "curr_value" in text:
            category = "KV_LAYOUT_MISMATCH"
        elif "position" in text or "rope" in text or "rotary" in text:
            category = "POSITION_ROPE_MISMATCH"
        elif "zero_point" in text or "zero point" in text:
            category = "QUANT_ZERO_POINT_MISMATCH"
        elif "scale" in text:
            category = "QUANT_SCALE_MISMATCH"
        elif "requant" in text or "round" in text or "saturat" in text:
            category = "REQUANT_ROUNDING_MISMATCH"
        elif "bias" in text:
            category = "BIAS_OWNERSHIP_ERROR"
        elif "residual" in text or "hidden_states" in text or "add/add" in text:
            category = "RESIDUAL_DUPLICATION"
        elif "pad" in text or "alignment" in text:
            category = "PAD_ALIGNMENT_MISMATCH"
        elif "int4" in text or "nibble" in text or "packed_weight" in text:
            category = "INT4_PACKING_ERROR"
        elif result.get("projection") or "weight" in text:
            category = "WEIGHT_SLICE_ERROR"
        else:
            category = "UNSUPPORTED_OPERATOR_SEMANTICS"
    result["category"] = category
    if record.get("category") and record.get("category") != category:
        result["observed_category"] = record.get("category")
    result["coordinates"] = {
        key: result.get(key)
        for key in (
            "layer",
            "rank",
            "stage",
            "operator",
            "tensor_name",
            "candidate_tensor",
            "projection",
            "slice_range",
            "quant_domain",
            "dtype",
            "shape",
            "first_mismatch_index",
            "expected",
            "actual",
            "max_absolute_difference",
            "mismatch_count",
            "partial_range_diagnostic",
        )
    }
    return result


def diagnose_tp4_mismatch(record, layer_manifest=None):
    """Classify a TP4 mismatch and attach canonical projection slice evidence."""
    result = classify_mismatch(record)
    rank = result.get("rank")
    projection = result.get("projection")
    partitions = (layer_manifest or {}).get("weight_partitions", {})
    partition = partitions.get(projection) if projection else None
    if not isinstance(partition, dict) or not isinstance(rank, int):
        return result
    slice_row = next(
        (row for row in partition.get("slice_ranges", []) if row.get("rank") == rank),
        None,
    )
    shard_row = next(
        (row for row in partition.get("weight_shards", []) if row.get("rank") == rank),
        None,
    )
    if slice_row is None:
        return result
    slice_range = [slice_row.get("start"), slice_row.get("stop")]
    result["slice_range"] = slice_range
    result["source_slice"] = {
        "projection": projection,
        "source_tensor_id": partition.get("source_tensor"),
        "source_buffer_id": partition.get("source_buffer"),
        "source_shape": partition.get("source_shape"),
        "slice_axis": partition.get("slice_axis"),
        "slice_axis_name": partition.get("slice_axis_name"),
        "rank": rank,
        "slice_range": slice_range,
        "source_weight_sha256": partition.get("source_sha256"),
        "rank_weight_sha256": (shard_row or {}).get("sha256"),
        "reconstruction_equal": partition.get("reconstruction_equal"),
    }
    result["coordinates"]["slice_range"] = slice_range
    result["coordinates"]["source_slice"] = result["source_slice"]
    return result


def annotate_tp4_golden_diagnostics(report, layer_manifests, source_sha256_to_layer=None):
    """Enrich Golden mismatch rows with layer and rank slice provenance."""
    manifests_by_layer = {int(key): value for key, value in layer_manifests.items()}
    layer_by_source = {
        str(layer_manifest.get("source_sha256", "")).lower(): layer
        for layer, layer_manifest in manifests_by_layer.items()
    }
    layer_by_source.update(
        {
            str(source_sha).lower(): int(layer)
            for source_sha, layer in (source_sha256_to_layer or {}).items()
        }
    )
    mismatch_rows = report.get("mismatches", [])
    for row in mismatch_rows:
        if row.get("layer") is None:
            inferred_layer = layer_by_source.get(str(row.get("source_sha256", "")).lower())
            if inferred_layer is not None:
                row["layer"] = inferred_layer
                row["layer_inferred_from_source_sha256"] = True
        layer = row.get("layer")
        layer_manifest = manifests_by_layer.get(layer) if isinstance(layer, int) else None
        diagnosis = diagnose_tp4_mismatch(row, layer_manifest)
        row["diagnostic_category"] = diagnosis["category"]
        row["diagnosis"] = diagnosis

    for row in mismatch_rows:
        if row.get("rank") is not None:
            continue
        tensor_name = row.get("tensor_name")
        source_sha = str(row.get("source_sha256", "")).lower()
        related = [
            item for item in mismatch_rows
            if item.get("rank") is not None
            and item.get("stage") == "KV_OUTPUT"
            and item.get("candidate_tensor") == tensor_name
            and str(item.get("source_sha256", "")).lower() == source_sha
        ]
        if related:
            row["affected_ranks"] = sorted({int(item["rank"]) for item in related})
            row["propagated_boundary_mismatches"] = [
                {
                    key: item.get(key)
                    for key in (
                        "layer",
                        "rank",
                        "stage",
                        "candidate_tensor",
                        "first_mismatch_index",
                        "expected",
                        "actual",
                    )
                }
                for item in related
            ]
            related_diagnoses = [item.get("diagnosis", {}) for item in related]
            if related_diagnoses and all(
                item.get("category") == "REQUANT_ROUNDING_MISMATCH"
                for item in related_diagnoses
            ):
                row["diagnostic_category"] = "REQUANT_ROUNDING_MISMATCH"
                row["diagnosis"]["category"] = "REQUANT_ROUNDING_MISMATCH"
                row["diagnosis"]["classification_evidence"] = {
                    "basis": "PROPAGATED_FROM_ONE_UNIT_QUANTIZED_BOUNDARY_MISMATCH",
                    "root_cause_confirmed": False,
                    "safe_repair_available": False,
                }
                row["diagnosis"]["affected_ranks"] = row["affected_ranks"]
                row["diagnosis"]["propagated_boundary_mismatches"] = row[
                    "propagated_boundary_mismatches"
                ]
                row["diagnosis"]["source_slices"] = [
                    item["source_slice"]
                    for item in related_diagnoses
                    if item.get("source_slice")
                ]

    for layer_report in report.get("layer_reports", []):
        layer = layer_report.get("layer")
        layer_manifest = manifests_by_layer.get(layer) if isinstance(layer, int) else None
        for row in layer_report.get("mismatches", []):
            row.setdefault("layer", layer)
            row.setdefault("source_sha256", layer_report.get("source_sha256"))
            diagnosis = diagnose_tp4_mismatch(row, layer_manifest)
            row["diagnostic_category"] = diagnosis["category"]
            row["diagnosis"] = diagnosis

    category_counts = {}
    for row in mismatch_rows:
        category = row.get("diagnostic_category", "UNSUPPORTED_OPERATOR_SEMANTICS")
        category_counts[category] = category_counts.get(category, 0) + int(row.get("mismatch_count", 1))
    first = mismatch_rows[0].get("diagnosis") if mismatch_rows else None
    report["diagnostic_summary"] = {
        "mismatch_record_count": len(mismatch_rows),
        "mismatch_count": sum(int(row.get("mismatch_count", 1)) for row in mismatch_rows),
        "category_counts": category_counts,
        "first_mismatch": first,
        "safe_repair_available": bool(
            first and AUTO_REPAIR_RULES.get(first.get("category"))
        ),
    }
    return report


def _source_digest(source):
    if not source:
        return None
    if isinstance(source, bytes):
        return hashlib.sha256(source).hexdigest()
    value = str(source).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        return None
    return value


def _fixture_evidence_digest(report):
    """Return a stable digest for the exact Golden fixtures recorded in a report."""
    if not isinstance(report, dict):
        return None

    if "fixture_sha256_by_layer" in report:
        fixture_hashes = report.get("fixture_sha256_by_layer")
        if not isinstance(fixture_hashes, dict) or not fixture_hashes:
            return None
        normalized = {}
        for layer, fixture_sha in fixture_hashes.items():
            if not isinstance(layer, str) or not layer:
                return None
            if not isinstance(fixture_sha, str):
                return None
            digest = _source_digest(fixture_sha)
            if digest is None:
                return None
            normalized[layer] = digest
        payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return (
            "fixture_sha256_by_layer",
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )

    fixture_sha = report.get("fixture_sha256")
    if not isinstance(fixture_sha, str):
        return None
    digest = _source_digest(fixture_sha)
    if digest is None:
        return None
    return "fixture_sha256", digest


class AutoRepairEngine:
    """Bounded repair orchestration; callbacks must rebuild and retest."""

    def __init__(self, max_repairs=3, success_condition=None, success_status="REPAIRED_AND_VERIFIED"):
        if max_repairs < 0:
            raise ToolError("MAX_REPAIRS_INVALID", "max repairs cannot be negative")
        if success_condition is not None and not callable(success_condition):
            raise ToolError("AUTO_REPAIR_SUCCESS_CONDITION_INVALID", "success condition must be callable")
        if not isinstance(success_status, str) or not success_status:
            raise ToolError("AUTO_REPAIR_SUCCESS_STATUS_INVALID", "success status must be a non-empty string")
        self.max_repairs = max_repairs
        self.success_condition = success_condition or (
            lambda report: report.get("numerical_status") == "NUMERICAL_PASS"
            and report.get("mismatch_count") == 0
        )
        self.success_status = success_status

    def run(self, initial_report, rebuild_from_canonical, verify_again):
        """Apply evidence-backed canonical rebuilds, then rerun the verifier.

        A callback must explicitly confirm it rebuilt from the recorded
        canonical SHA. Missing callbacks or incomplete coordinates fail closed.
        """
        current = dict(initial_report)
        attempts = []
        if self.success_condition(current):
            return {"status": "NO_REPAIR_NEEDED", "attempts": attempts, "final_report": current}
        if not callable(rebuild_from_canonical) or not callable(verify_again):
            return {
                "status": "REPAIR_BLOCKED",
                "failure": "SAFE_REBUILD_OR_RETEST_CALLBACK_MISSING",
                "attempts": attempts,
                "final_report": current,
            }

        fixture_identity = _fixture_evidence_digest(current)
        if fixture_identity is None:
            return {
                "status": "REPAIR_BLOCKED",
                "failure": "GOLDEN_FIXTURE_IDENTITY_MISSING",
                "attempts": attempts,
                "final_report": current,
            }

        seen_mismatches = set()
        while len(attempts) < self.max_repairs:
            mismatch_rows = current.get("mismatches") or []
            if not mismatch_rows:
                return {
                    "status": "REPAIR_BLOCKED",
                    "failure": current.get("first_failure", "MISMATCH_DETAILS_MISSING"),
                    "attempts": attempts,
                    "final_report": current,
                }
            row = mismatch_rows[0]
            mismatch = row.get("diagnosis") if isinstance(row.get("diagnosis"), dict) else classify_mismatch(row)
            source_sha = current.get("source_sha256") or mismatch.get("source_sha256")
            coordinates = mismatch["coordinates"]
            required = (
                coordinates.get("layer"),
                coordinates.get("stage"),
                coordinates.get("operator"),
                coordinates.get("tensor_name"),
            )
            aggregate_tp2_boundary = (
                mismatch.get("topology") == "TP2_ROW_REDUCE"
                and coordinates.get("rank") is None
                and coordinates.get("stage") in ("O_ROW_REDUCE", "DOWN_ROW_REDUCE")
            )
            if (
                not source_sha
                or any(value is None for value in required)
                or (coordinates.get("rank") is None and not aggregate_tp2_boundary)
            ):
                return {
                    "status": "REPAIR_BLOCKED",
                    "failure": "MISMATCH_SOURCE_MAPPING_INCOMPLETE",
                    "attempts": attempts,
                    "final_report": current,
                }
            source_digest = _source_digest(source_sha)
            if source_digest is None:
                return {
                    "status": "REPAIR_BLOCKED",
                    "failure": "CANONICAL_SOURCE_SHA_INVALID",
                    "attempts": attempts,
                    "final_report": current,
                }
            repair_rule = AUTO_REPAIR_RULES.get(mismatch["category"])
            if repair_rule is None:
                return {
                    "status": "REPAIR_BLOCKED",
                    "failure": "NO_SAFE_REPAIR_RULE_FOR_{}".format(mismatch["category"]),
                    "diagnosed_mismatch": mismatch,
                    "attempts": attempts,
                    "final_report": current,
                }
            fingerprint = (
                mismatch["category"],
                coordinates.get("layer"),
                coordinates.get("rank"),
                coordinates.get("stage"),
                coordinates.get("operator"),
                coordinates.get("tensor_name"),
                source_digest,
            )
            if fingerprint in seen_mismatches:
                return {
                    "status": "REPAIR_STALLED",
                    "failure": "SAME_MISMATCH_RECURRED_AFTER_REBUILD",
                    "attempts": attempts,
                    "final_report": current,
                }
            seen_mismatches.add(fingerprint)
            before = dict(current)
            rebuild = rebuild_from_canonical(
                mismatch,
                expected_source_sha256=source_digest,
                repair_rule=repair_rule,
            )
            if (
                not isinstance(rebuild, dict)
                or rebuild.get("rebuilt_from_canonical") is not True
                or rebuild.get("source_sha256") != source_digest
                or rebuild.get("repair_rule") != repair_rule
                or not rebuild.get("explanation")
            ):
                attempts.append({
                    "attempt": len(attempts) + 1,
                    "category": mismatch["category"],
                    "source_sha256": source_digest,
                    "applied": False,
                    "reason": "CANONICAL_REBUILD_NOT_CONFIRMED",
                })
                return {
                    "status": "REPAIR_FAILED",
                    "failure": "CANONICAL_REBUILD_NOT_CONFIRMED",
                    "attempts": attempts,
                    "final_report": current,
                }
            repair_evidence = rebuild.get("repair_evidence")
            inventory_before = (
                repair_evidence.get("candidate_inventory_before_sha256")
                if isinstance(repair_evidence, dict)
                else None
            )
            inventory_after = (
                repair_evidence.get("candidate_inventory_after_sha256")
                if isinstance(repair_evidence, dict)
                else None
            )
            changed_count = (
                repair_evidence.get("changed_candidate_artifact_count")
                if isinstance(repair_evidence, dict)
                else None
            )
            if (
                rebuild.get("repair_applied") is not True
                or repair_evidence.get("action") != "REBUILD_RANK_COMPONENTS_FROM_CANONICAL"
                or repair_evidence.get("requested_rule") != repair_rule
                or not isinstance(changed_count, int)
                or isinstance(changed_count, bool)
                or changed_count <= 0
                or not inventory_before
                or not inventory_after
                or inventory_before == inventory_after
            ):
                attempts.append({
                    "attempt": len(attempts) + 1,
                    "category": mismatch["category"],
                    "source_sha256": source_digest,
                    "repair_rule": repair_rule,
                    "repair_evidence": repair_evidence,
                    "applied": False,
                    "reason": "REBUILD_DID_NOT_CHANGE_CANDIDATE_ARTIFACTS",
                })
                return {
                    "status": "REPAIR_FAILED",
                    "failure": "REBUILD_DID_NOT_CHANGE_CANDIDATE_ARTIFACTS",
                    "attempts": attempts,
                    "final_report": current,
                }
            after = verify_again(rebuild, mismatch)
            if (
                not isinstance(after, dict)
                or after.get("source_sha256") != source_digest
                or not isinstance(after.get("mismatch_count"), int)
                or not after.get("numerical_status")
            ):
                return {
                    "status": "REPAIR_FAILED",
                    "failure": "RETEST_REPORT_INVALID_OR_UNBOUND",
                    "attempts": attempts,
                    "final_report": current,
                }
            retest_fixture_identity = _fixture_evidence_digest(after)
            if retest_fixture_identity is None:
                attempts.append({
                    "attempt": len(attempts) + 1,
                    "category": mismatch["category"],
                    "source_sha256": source_digest,
                    "fixture_identity_field": fixture_identity[0],
                    "fixture_evidence_sha256": fixture_identity[1],
                    "rebuilt_from_canonical": True,
                    "retested": True,
                    "result": after.get("numerical_status"),
                    "failure": "RETEST_FIXTURE_IDENTITY_MISSING",
                })
                return {
                    "status": "REPAIR_FAILED",
                    "failure": "RETEST_FIXTURE_IDENTITY_MISSING",
                    "attempts": attempts,
                    "final_report": current,
                }
            if retest_fixture_identity != fixture_identity:
                attempts.append({
                    "attempt": len(attempts) + 1,
                    "category": mismatch["category"],
                    "source_sha256": source_digest,
                    "fixture_identity_field": fixture_identity[0],
                    "fixture_evidence_sha256": fixture_identity[1],
                    "after_fixture_identity_field": retest_fixture_identity[0],
                    "after_fixture_evidence_sha256": retest_fixture_identity[1],
                    "rebuilt_from_canonical": True,
                    "retested": True,
                    "result": after.get("numerical_status"),
                    "failure": "RETEST_FIXTURE_IDENTITY_MISMATCH",
                })
                return {
                    "status": "REPAIR_FAILED",
                    "failure": "RETEST_FIXTURE_IDENTITY_MISMATCH",
                    "attempts": attempts,
                    "final_report": current,
                }
            attempts.append({
                "attempt": len(attempts) + 1,
                "category": mismatch["category"],
                "coordinates": coordinates,
                "source_sha256": source_digest,
                "fixture_identity_field": fixture_identity[0],
                "fixture_evidence_sha256": fixture_identity[1],
                "repair_rule": rebuild["repair_rule"],
                "explanation": rebuild["explanation"],
                "repair_evidence": repair_evidence,
                "before": {
                    "mismatch_count": before.get("mismatch_count"),
                    "first_mismatch": before.get("mismatches", [None])[0],
                },
                "before_mismatch_count": before.get("mismatch_count"),
                "after_mismatch_count": after.get("mismatch_count"),
                "rebuilt_from_canonical": True,
                "retested": True,
                "result": after.get("numerical_status"),
                "verification_scope": after.get("verification_scope"),
            })
            current = dict(after)
            if self.success_condition(current):
                return {"status": self.success_status, "attempts": attempts, "final_report": current}

        return {
            "status": "REPAIR_LIMIT_REACHED",
            "failure": "MAX_REPAIRS_EXCEEDED",
            "attempts": attempts,
            "final_report": current,
        }
