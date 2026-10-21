"""Fail-closed Phase 3 report generation helpers.

The report layer intentionally derives release eligibility from explicit gate
statuses. Missing or unrun evidence can never be interpreted as a zero-mismatch
PASS.
"""

import json
from pathlib import Path


_PROFILE_GATES = (
    "TP2_P128_EXACT",
    "TP2_D1_EXACT",
    "TP4_P128_EXACT",
    "TP4_D1_EXACT",
)


def derive_release_gate(summary):
    """Return a release-gate object without promoting missing evidence."""
    gates = summary.get("profile_gates", {})
    missing = [name for name in _PROFILE_GATES if name not in gates]
    not_pass = {name: gates.get(name, "NOT_RUN") for name in _PROFILE_GATES if gates.get(name) != "PASS"}
    host = summary.get("host_verified", "NOT_RUN")
    row_reduce = summary.get("tp2_row_reduce_contract", "UNPROVEN")
    distributed = summary.get("distributed_runtime_verified", "NOT_VERIFIED")
    # The verifier validates partitioned rank graphs with NPU execution plus
    # host stitching. A multi-board transport runtime is a separate product
    # gate and must be reported, but is not required to certify the artifacts.
    eligible = not missing and not not_pass and host == "PASS" and row_reduce == "PROVEN"
    blockers = []
    if missing:
        blockers.append({"gate": "PROFILE_GATES", "reason": "MISSING_PROFILE_GATE_STATUS", "profiles": missing})
    for name, status in not_pass.items():
        blockers.append({"gate": name, "reason": status})
    if host != "PASS":
        blockers.append({"gate": "HOST_VERIFIED", "reason": host})
    if row_reduce != "PROVEN":
        blockers.append({"gate": "TP2_ROW_REDUCE_CONTRACT", "reason": row_reduce})
    return {
        "schema": "tflite_tp.phase3.release-gate.v1",
        "release_gate": "PASS" if eligible else "BLOCKED",
        "eligible": eligible,
        "profile_gates": {name: gates.get(name, "NOT_RUN") for name in _PROFILE_GATES},
        "host_verified": host,
        "tp2_row_reduce_contract": row_reduce,
        "distributed_runtime_verified": distributed,
        "distributed_runtime_is_separate_gate": True,
        "blockers": blockers,
    }


def write_phase3_reports(output_dir, summary, report_title="Issue #24 Phase 3"):
    """Write summary, release gate, and a concise Markdown report.

    ``summary`` must contain observed statuses. The function does not infer
    PASS from null mismatch counts, hashes, structural audits, or NCC compile.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate = derive_release_gate(summary)
    frozen_summary = dict(summary)
    frozen_summary["schema"] = "tflite_tp.phase3.summary.v1"
    frozen_summary["release_gate"] = gate["release_gate"]
    frozen_summary["release_eligible"] = gate["eligible"]
    frozen_summary["blockers"] = gate["blockers"]
    (out / "PHASE3_SUMMARY.json").write_text(
        json.dumps(frozen_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "RELEASE_GATE.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {report_title}",
        "",
        f"**Release gate:** `{gate['release_gate']}`",
        "",
        "## Profile numerical gates",
        "",
    ]
    for name, status in gate["profile_gates"].items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(["", "## Other required gates", ""])
    lines.extend([
        f"- HOST_VERIFIED: `{gate['host_verified']}`",
        f"- TP2_ROW_REDUCE_CONTRACT: `{gate['tp2_row_reduce_contract']}`",
        f"- DISTRIBUTED_RUNTIME_VERIFIED: `{gate['distributed_runtime_verified']}`",
        "",
        "## Blockers",
        "",
    ])
    if gate["blockers"]:
        for blocker in gate["blockers"]:
            lines.append(f"- `{blocker['gate']}`: `{blocker['reason']}`")
    else:
        lines.append("- None")
    lines.append("")
    (out / "PHASE3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return {"summary": str(out / "PHASE3_SUMMARY.json"), "gate": str(out / "RELEASE_GATE.json"), "report": str(out / "PHASE3_REPORT.md"), "release_gate": gate}
