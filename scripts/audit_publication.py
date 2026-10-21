"""Audit working files or the Git index for private artifacts and sensitive text."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_RELATIVE = Path("audit/publication_scan.json")
PROHIBITED_SUFFIXES = {
    ".a", ".bin", ".dll", ".dylib", ".engine", ".gguf", ".h5", ".jpeg",
    ".jpg", ".lib", ".model", ".npy", ".npz", ".onnx", ".pdf", ".pb",
    ".png", ".pt", ".pth", ".safetensors", ".so", ".tflite", ".dla",
    ".whl", ".zip", ".gz", ".xz", ".tar",
}
TEXT_CHECKS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "credential_assignment": re.compile(
        r"""(?i)\b(?:password|passwd|api[_-]?key|secret|access[_-]?token)\b\s*[:=]\s*["'][^"'\r\n]{8,}["']"""
    ),
    "windows_user_path": re.compile(r"\b[A-Za-z]:\\Users\\[^\\/\s]+"),
    "rfc1918_ipv4": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
}


def files_to_scan(staged):
    if staged:
        listing = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, stdout=subprocess.PIPE
        ).stdout
        for raw_name in listing.split(b"\0"):
            if not raw_name:
                continue
            relative = raw_name.decode("utf-8")
            if Path(relative) == OUTPUT_RELATIVE:
                continue
            blob = subprocess.run(
                ["git", "show", ":{}".format(relative)],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
            yield Path(relative), relative, blob
        return

    ignored = {".git", ".venv", "build", "dist", "__pycache__", ".pytest_cache"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative == OUTPUT_RELATIVE:
            continue
        if any(
            part in ignored
            or part.startswith(".venv")
            or part.startswith(".publication_")
            or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        yield path, relative.as_posix(), path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT_RELATIVE)
    parser.add_argument(
        "--staged",
        action="store_true",
        help="scan the exact blobs in the Git index instead of working-tree files",
    )
    args = parser.parse_args()
    checked = []
    prohibited_files = []
    content_findings = []

    for path, relative, raw in files_to_scan(args.staged):
        checked.append({
            "path": relative,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            prohibited_files.append(relative)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            content_findings.append({"path": relative, "check": "non_utf8_or_binary"})
            continue
        for check_name, pattern in TEXT_CHECKS.items():
            if pattern.search(text):
                content_findings.append({"path": relative, "check": check_name})

    report = {
        "audit": "public-source-safety-scan-v1",
        "scan_date": "2026-10-21",
        "scan_scope": "git_index_staged" if args.staged else "working_tree",
        "result": "PASS" if not prohibited_files and not content_findings else "FAIL",
        "files_checked": len(checked),
        "prohibited_files": prohibited_files,
        "sensitive_content_findings": content_findings,
        "file_sha256": checked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": report["result"],
        "scan_scope": report["scan_scope"],
        "files_checked": report["files_checked"],
        "prohibited_files": prohibited_files,
        "sensitive_content_findings": content_findings,
        "report": args.output.relative_to(ROOT).as_posix()
        if args.output.is_relative_to(ROOT) else str(args.output),
    }, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())