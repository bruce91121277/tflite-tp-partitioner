"""Fail-closed NCC, SSH/SCP, EVK-idle, and Neuron Runtime operations."""

import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .contracts import DTYPE_NAMES, load_tflite_contract, read_json, sha256_file


_SAFE_REMOTE = re.compile(r"^/[A-Za-z0-9._/-]+$")
_SAFE_HOST = re.compile(r"^[A-Za-z0-9._@-]+$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_NCC_TARGET = "mdla5.3,edma3.6"


class TransportError(RuntimeError):
    pass


def _check_program(name):
    path = shutil.which(name)
    if not path:
        raise TransportError("LOCAL_PROGRAM_MISSING:" + name)
    return path


def _ssh_config_args():
    """Honor a caller-supplied standard SSH config without embedding credentials."""
    config = os.environ.get("TFLITE_TP_SSH_CONFIG")
    if not config:
        return []
    path = Path(config).expanduser()
    if not path.is_file():
        raise TransportError("SSH_CONFIG_NOT_READABLE")
    return ["-F", str(path)]


def _run(command, timeout, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TransportError("COMMAND_TIMEOUT:{}".format(" ".join(command))) from exc
    return result


def _ssh(host, remote_command, timeout=60):
    if not _SAFE_HOST.fullmatch(host):
        raise TransportError("SSH_TARGET_INVALID")
    ssh = _check_program("ssh")
    result = _run([ssh, *_ssh_config_args(), "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote_command], timeout)
    if result.returncode != 0:
        raise TransportError("SSH_FAILED:{}\n{}".format(result.returncode, result.stdout[-12000:]))
    return result.stdout


def probe_environment():
    """Read-only local environment probe; it never invokes a target device."""
    commands = {}
    for command in ("ssh", "scp", "git"):
        commands[command] = shutil.which(command)
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        numpy_version = None
    try:
        import tflite
        schema_version = getattr(tflite, "__version__", "AVAILABLE")
    except ImportError:
        schema_version = None
    return {
        "schema": "tflite_tp.phase3.environment-probe.v1",
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "numpy": numpy_version,
        "tflite_schema": schema_version,
        "commands": commands,
        "npu_runtime_execution_host": "EVK_ONLY",
        "status": "PASS" if commands["ssh"] and commands["scp"] and numpy_version and schema_version else "INCOMPLETE",
    }


def device_gate(host, remote_root="/tmp", required_bytes=0, required_memory_bytes=0):
    """Run a fresh read-only idle/capacity check; a failed or uncertain check blocks."""
    if not _SAFE_REMOTE.fullmatch(remote_root):
        raise TransportError("REMOTE_ROOT_INVALID")
    required_bytes = int(required_bytes)
    required_memory_bytes = int(required_memory_bytes)
    script = """set -eu
printf 'EVK_HOSTNAME=%s\\n' "$(hostname)"
printf 'NEURONRT_PATH=%s\\n' "$(command -v neuronrt || true)"
printf 'NEURONRT_SHA256='
if [ -x /usr/sbin/neuronrt ]; then sha256sum /usr/sbin/neuronrt | awk '{{print $1}}'; else echo MISSING; fi
printf 'NEURONRT_VERSION_BEGIN\n'
if [ -x /usr/sbin/neuronrt ]; then /usr/sbin/neuronrt --version 2>&1 | head -n 4 || true; fi
printf 'NEURONRT_VERSION_END\n'
printf 'TIMEOUT_PATH=%s\n' "$(command -v timeout || true)"
printf 'BUSY_SCAN_BEGIN\\n'
ps -eo pid=,comm=,args= | grep -E '(^|[ /])(neuronrt|vllm|EngineCore|pp_daemon|llama-server|run_tp[24]|tp[24].*runner|runner.*tp[24])([ /]|$)' | grep -v grep || true
printf 'BUSY_SCAN_END\\n'
storage_path={root}
while [ ! -d "$storage_path" ] && [ "$storage_path" != "/" ]; do storage_path=$(dirname "$storage_path"); done
printf 'EVK_STORAGE_CHECK_PATH=%s\\n' "$storage_path"
df -Pk "$storage_path" | awk 'NR==2 {{printf "EVK_TMP_AVAILABLE_KB=%s\\n", $4}}'
awk '/MemAvailable:/ {{printf "EVK_MEM_AVAILABLE_KB=%s\\n", $2}}' /proc/meminfo
printf 'EVK_UNAME=%s\\n' "$(uname -a)"
""".format(root=shlex.quote(remote_root))
    try:
        output = _ssh(host, script, timeout=30)
    except TransportError as exc:
        return {
            "schema": "tflite_tp.phase3.evk-idle-gate.v1",
            "target": host,
            "status": "UNCONFIRMED",
            "gate": "FAIL",
            "reason": str(exc),
            "npu_invoke_allowed": False,
        }
    busy_section = output.split("BUSY_SCAN_BEGIN\n", 1)
    busy_lines = []
    if len(busy_section) == 2:
        busy_lines = [line for line in busy_section[1].split("BUSY_SCAN_END", 1)[0].splitlines() if line.strip()]
    free_match = re.search(r"EVK_TMP_AVAILABLE_KB=(\d+)", output)
    mem_match = re.search(r"EVK_MEM_AVAILABLE_KB=(\d+)", output)
    storage_path_match = re.search(r"EVK_STORAGE_CHECK_PATH=([^\r\n]+)", output)
    free_bytes = int(free_match.group(1)) * 1024 if free_match else None
    mem_bytes = int(mem_match.group(1)) * 1024 if mem_match else None
    runtime_match = re.search(r"NEURONRT_PATH=([^\r\n]+)", output)
    runtime_sha_match = re.search(r"NEURONRT_SHA256=([0-9a-fA-F]{64})", output)
    timeout_match = re.search(r"TIMEOUT_PATH=([^\r\n]+)", output)
    version_section = re.search(r"NEURONRT_VERSION_BEGIN\n(.*?)\nNEURONRT_VERSION_END", output, re.S)
    runtime_version = version_section.group(1).strip() if version_section else "NOT_AVAILABLE"
    ready = (
        not busy_lines
        and free_bytes is not None
        and free_bytes >= required_bytes
        and mem_bytes is not None
        and mem_bytes >= required_memory_bytes
        and timeout_match is not None
        and timeout_match.group(1) not in ("", "None")
        and runtime_match
        and runtime_match.group(1) not in ("", "None")
    )
    return {
        "schema": "tflite_tp.phase3.evk-idle-gate.v1",
        "target": host,
        "status": "PASS" if ready else "FAIL",
        "gate": "PASS" if ready else "FAIL",
        "npu_invoke_allowed": bool(ready),
        "busy_processes": busy_lines,
        "remote_root": remote_root,
        "free_bytes": free_bytes,
        "storage_check_path": storage_path_match.group(1) if storage_path_match else None,
        "available_memory_bytes": mem_bytes,
        "required_memory_bytes_estimate": required_memory_bytes,
        "neuronrt_path": runtime_match.group(1) if runtime_match else None,
        "neuronrt_sha256": runtime_sha_match.group(1).lower() if runtime_sha_match else None,
        "neuronrt_version_read_only": runtime_version,
        "timeout_path": timeout_match.group(1) if timeout_match else None,
        "required_storage_bytes": required_bytes,
        "raw_read_only_output": output,
    }


def record_blocked_invoke(
    model_path,
    dla_path,
    compile_manifest_path,
    input_manifest_path,
    device_host,
    output_dir,
    run_id,
    gate_manifest,
    reason,
):
    """Persist a failed hardware gate without implying that Neuron Runtime ran."""
    model_path = Path(model_path).resolve()
    dla_path = Path(dla_path).resolve()
    input_manifest_path = Path(input_manifest_path).resolve()
    compile_manifest_path = Path(compile_manifest_path).resolve()
    out = Path(output_dir).resolve()
    compile_manifest = read_json(compile_manifest_path)
    input_manifest = read_json(input_manifest_path)
    contract = load_tflite_contract(model_path)
    result = {
        "schema": "tflite_tp.phase3.npu-run.v1",
        "status": "BLOCKED",
        "profile": input_manifest.get("profile"),
        "fixture_id": input_manifest.get("fixture_id"),
        "fixture_sha256": input_manifest.get("fixture_sha256"),
        "tp_degree": input_manifest.get("tp_degree"),
        "rank": input_manifest.get("rank"),
        "run_id": run_id,
        "source_tflite_sha256": contract["sha256"],
        "dla_sha256": sha256_file(dla_path) if dla_path.is_file() else None,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "compile_manifest_sha256": sha256_file(compile_manifest_path),
        "compile_status": compile_manifest.get("status"),
        "device_host": device_host,
        "invoke_started": False,
        "invoke_exit_code": None,
        "outputs": [],
        "output_coverage": "NOT_RUN",
        "device_idle_gate": gate_manifest,
        "first_failure": reason,
        "npu_exact_gate": "NOT_RUN",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "NPU_RUN_MANIFEST.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def compile_ncc(
    source_tflite,
    compiler_host,
    ncc_bin,
    ncc_lib,
    remote_workspace,
    output_dir,
    run_id,
    timeout_seconds=1800,
    verify=True,
):
    """Compile one model on RTX/server NCC; library path is scoped to NCC only."""
    source = Path(source_tflite).resolve()
    out = Path(output_dir).resolve()
    if not source.is_file():
        raise TransportError("SOURCE_TFLITE_MISSING")
    if not _SAFE_REMOTE.fullmatch(remote_workspace) or not _SAFE_REMOTE.fullmatch(ncc_bin) or not _SAFE_REMOTE.fullmatch(ncc_lib):
        raise TransportError("REMOTE_COMPILER_PATH_INVALID")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise TransportError("RUN_ID_INVALID")
    if int(timeout_seconds) <= 0 or int(timeout_seconds) > 3600:
        raise TransportError("NCC_TIMEOUT_OUT_OF_RANGE")
    out.mkdir(parents=True, exist_ok=True)
    remote_dir = remote_workspace.rstrip("/") + "/" + run_id
    if not _SAFE_REMOTE.fullmatch(remote_dir):
        raise TransportError("REMOTE_RUN_PATH_INVALID")
    _ssh(compiler_host, "set -eu; test -x {ncc}; test -d {lib}; mkdir -m 700 -p {work}; test -w {work}".format(
        ncc=shlex.quote(ncc_bin), lib=shlex.quote(ncc_lib), work=shlex.quote(remote_dir)
    ), timeout=30)
    scp = _check_program("scp")
    ssh = _check_program("ssh")
    ssh_config_args = _ssh_config_args()
    remote_src = remote_dir + "/source.tflite"
    remote_dla = remote_dir + "/model.dla"
    remote_log = remote_dir + "/compile.log"
    local_sha = sha256_file(source)
    scp_result = _run([scp, *ssh_config_args, str(source), "{}:{}".format(compiler_host, remote_src)], timeout=300)
    if scp_result.returncode != 0:
        raise TransportError("NCC_SOURCE_TRANSFER_FAIL\n" + scp_result.stdout[-12000:])
    verify_cmd = "sha256sum {path}; test \"$(sha256sum {path} | awk '{{print $1}}')\" = {sha}".format(
        path=shlex.quote(remote_src), sha=shlex.quote(local_sha)
    )
    _ssh(compiler_host, verify_cmd, timeout=60)
    identity = "set -eu; env LD_LIBRARY_PATH={lib} {ncc} --version; sha256sum {ncc}".format(
        lib=shlex.quote(ncc_lib), ncc=shlex.quote(ncc_bin)
    )
    identity_output = _ssh(compiler_host, identity, timeout=60)
    compile_args = "--arch={} {} --dla-file {} {}".format(
        _NCC_TARGET,
        "--verify" if verify else "--no-verify",
        shlex.quote(remote_dla),
        shlex.quote(remote_src),
    )
    compile_cmd = "set +e; env LD_LIBRARY_PATH={lib} {ncc} {args} > {log} 2>&1; rc=$?; cat {log}; echo NCC_COMPILE_EXIT_CODE=$rc; exit $rc".format(
        lib=shlex.quote(ncc_lib), ncc=shlex.quote(ncc_bin), args=compile_args,
        log=shlex.quote(remote_log),
    )
    compile_start = time.time()
    compile_result = _run([ssh, *ssh_config_args, "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", compiler_host, compile_cmd], int(timeout_seconds))
    elapsed = time.time() - compile_start
    local_log = out / "compile.log"
    local_log.write_text(compile_result.stdout, encoding="utf-8", errors="replace")
    if compile_result.returncode != 0:
        result = {
            "schema": "tflite_tp.phase3.ncc-build.v1",
            "status": "FAIL",
            "source_tflite": str(source),
            "source_tflite_sha256": local_sha,
            "target_arch": _NCC_TARGET,
            "verify_mode": "DEFAULT_VERIFY" if verify else "NO_VERIFY",
            "ncc_identity": identity_output,
            "command": compile_cmd,
            "exit_code": compile_result.returncode,
            "wall_seconds": elapsed,
            "log": str(local_log),
            "first_failure": "NCC_COMPILE_FAILED",
        }
        (out / "NCC_BUILD_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    plan_cmd = "set +e; env LD_LIBRARY_PATH={lib} {ncc} --arch={target} --show-io-info {src}; echo NCC_IO_INFO_EXIT_CODE=$?; env LD_LIBRARY_PATH={lib} {ncc} --arch={target} --show-exec-plan --show-memory-summary {src}; echo NCC_EXEC_PLAN_EXIT_CODE=$?".format(
        lib=shlex.quote(ncc_lib), ncc=shlex.quote(ncc_bin), target=_NCC_TARGET,
        src=shlex.quote(remote_src),
    )
    plan_output = _ssh(compiler_host, plan_cmd, timeout=int(timeout_seconds))
    (out / "execution_plan.log").write_text(plan_output, encoding="utf-8", errors="replace")
    remote_stat_sha = _ssh(compiler_host, "set -eu; test -s {dla}; stat -c 'DLA_BYTES=%s' {dla}; sha256sum {dla}".format(
        dla=shlex.quote(remote_dla)
    ), timeout=60)
    dl_path = out / "model.dla"
    if dl_path.exists():
        raise TransportError("DLA_DESTINATION_ALREADY_EXISTS")
    scp_result = _run([scp, *ssh_config_args, "{}:{}".format(compiler_host, remote_dla), str(dl_path)], timeout=600)
    if scp_result.returncode != 0:
        raise TransportError("NCC_DLA_DOWNLOAD_FAIL\n" + scp_result.stdout[-12000:])
    dla_sha = sha256_file(dl_path)
    remote_match = re.search(r"\b([0-9a-fA-F]{64})\s+.*model\.dla", remote_stat_sha)
    if not remote_match or remote_match.group(1).lower() != dla_sha:
        raise TransportError("DLA_TRANSFER_SHA_MISMATCH")
    version_match = re.search(r"Version:\s*([^\r\n]+)", identity_output)
    result = {
        "schema": "tflite_tp.phase3.ncc-build.v1",
        "status": "PASS",
        "source_tflite": str(source),
        "source_tflite_sha256": local_sha,
        "ncc_host": compiler_host,
        "ncc_binary": ncc_bin,
        "ncc_library": ncc_lib,
        "ncc_binary_sha256": re.search(r"\b([0-9a-fA-F]{64})\s+.*ncc-tflite", identity_output).group(1).lower() if re.search(r"\b([0-9a-fA-F]{64})\s+.*ncc-tflite", identity_output) else None,
        "ncc_version": version_match.group(1).strip() if version_match else "NOT_CAPTURED",
        "target_arch": _NCC_TARGET,
        "verify_mode": "DEFAULT_VERIFY" if verify else "NO_VERIFY",
        "command": compile_cmd,
        "exit_code": compile_result.returncode,
        "wall_seconds": elapsed,
        "remote_dla_sha256": remote_match.group(1).lower(),
        "dla_path": str(dl_path),
        "dla_sha256": dla_sha,
        "dla_bytes": dl_path.stat().st_size,
        "execution_plan_log": str(out / "execution_plan.log"),
        "compile_log": str(local_log),
        "ncc_identity": identity_output,
    }
    (out / "NCC_BUILD_MANIFEST.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def recover_ncc_build(
    source_tflite,
    compiler_host,
    ncc_bin,
    ncc_lib,
    remote_workspace,
    output_dir,
    run_id,
):
    """Collect a completed remote build after a local transport/reporting interruption.

    This routine never starts NCC compilation. It requires the prior local command
    log to contain an explicit zero exit marker, rechecks the remote source/DLA
    hashes, reruns only read-only IO/plan queries, and downloads the existing DLA.
    """
    source = Path(source_tflite).resolve()
    out = Path(output_dir).resolve()
    if not source.is_file():
        raise TransportError("SOURCE_TFLITE_MISSING")
    if not _SAFE_REMOTE.fullmatch(remote_workspace) or not _SAFE_REMOTE.fullmatch(ncc_bin) or not _SAFE_REMOTE.fullmatch(ncc_lib):
        raise TransportError("REMOTE_COMPILER_PATH_INVALID")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise TransportError("RUN_ID_INVALID")
    local_compile_log = out / "compile.log"
    if not local_compile_log.is_file():
        raise TransportError("PRIOR_COMPILE_LOG_MISSING")
    log_text = local_compile_log.read_text(encoding="utf-8", errors="replace")
    rc_match = re.search(r"NCC_COMPILE_EXIT_CODE=(\d+)", log_text)
    if not rc_match or int(rc_match.group(1)) != 0:
        raise TransportError("PRIOR_COMPILE_EXIT_NOT_VERIFIED")
    if (out / "model.dla").exists() or (out / "NCC_BUILD_MANIFEST.json").exists():
        raise TransportError("RECOVERY_DESTINATION_ALREADY_POPULATED")
    remote_dir = remote_workspace.rstrip("/") + "/" + run_id
    if not _SAFE_REMOTE.fullmatch(remote_dir):
        raise TransportError("REMOTE_RUN_PATH_INVALID")
    remote_src = remote_dir + "/source.tflite"
    remote_dla = remote_dir + "/model.dla"
    remote_log = remote_dir + "/compile.log"
    local_sha = sha256_file(source)
    remote_identity = _ssh(
        compiler_host,
        "set -eu; test -s {src}; test \"$(sha256sum {src} | cut -d ' ' -f 1)\" = {sha}; test -s {dla}; sha256sum {src}; stat -c 'DLA_BYTES=%s' {dla}; sha256sum {dla}".format(
            src=shlex.quote(remote_src), sha=shlex.quote(local_sha), dla=shlex.quote(remote_dla)
        ),
        timeout=60,
    )
    remote_sha = re.search(r"\b([0-9a-fA-F]{64})\s+.*source\.tflite", remote_identity)
    remote_dla_sha = re.search(r"\b([0-9a-fA-F]{64})\s+.*model\.dla", remote_identity)
    size_match = re.search(r"DLA_BYTES=(\d+)", remote_identity)
    if not remote_sha or remote_sha.group(1).lower() != local_sha:
        raise TransportError("REMOTE_SOURCE_SHA_MISMATCH")
    if not remote_dla_sha or not size_match:
        raise TransportError("REMOTE_DLA_EVIDENCE_INCOMPLETE")
    identity_cmd = "set -eu; env LD_LIBRARY_PATH={lib} {ncc} --version; sha256sum {ncc}".format(
        lib=shlex.quote(ncc_lib), ncc=shlex.quote(ncc_bin)
    )
    identity_output = _ssh(compiler_host, identity_cmd, timeout=60)
    io_output = _ssh(
        compiler_host,
        "set +e; env LD_LIBRARY_PATH={lib} {ncc} --arch={arch} --show-io-info {src}; rc=$?; echo NCC_IO_INFO_EXIT_CODE=$rc; exit 0".format(
            lib=shlex.quote(ncc_lib), ncc=shlex.quote(ncc_bin), arch=_NCC_TARGET,
            src=shlex.quote(remote_src),
        ),
        timeout=900,
    )
    plan_output = _ssh(
        compiler_host,
        "set +e; env LD_LIBRARY_PATH={lib} {ncc} --arch={arch} --show-exec-plan --show-memory-summary {src}; rc=$?; echo NCC_EXEC_PLAN_EXIT_CODE=$rc; exit 0".format(
            lib=shlex.quote(ncc_lib), ncc=shlex.quote(ncc_bin), arch=_NCC_TARGET,
            src=shlex.quote(remote_src),
        ),
        timeout=900,
    )
    io_rc = re.search(r"NCC_IO_INFO_EXIT_CODE=(\d+)", io_output)
    plan_rc = re.search(r"NCC_EXEC_PLAN_EXIT_CODE=(\d+)", plan_output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "execution_plan.log").write_text(plan_output, encoding="utf-8", errors="replace")
    (out / "io_info.log").write_text(io_output, encoding="utf-8", errors="replace")
    ncc_log = _ssh(compiler_host, "cat {path}".format(path=shlex.quote(remote_log)), timeout=60)
    (out / "ncc_stdout_stderr.log").write_text(ncc_log, encoding="utf-8", errors="replace")
    if not io_rc or int(io_rc.group(1)) != 0 or not plan_rc or int(plan_rc.group(1)) != 0:
        raise TransportError("NCC_READ_ONLY_PLAN_AUDIT_FAIL")
    dl_path = out / "model.dla"
    scp = _check_program("scp")
    ssh_config_args = _ssh_config_args()
    copied = _run([scp, *ssh_config_args, "{}:{}".format(compiler_host, remote_dla), str(dl_path)], timeout=600)
    if copied.returncode != 0:
        raise TransportError("NCC_DLA_DOWNLOAD_FAIL\n" + copied.stdout[-12000:])
    local_dla_sha = sha256_file(dl_path)
    if local_dla_sha != remote_dla_sha.group(1).lower() or dl_path.stat().st_size != int(size_match.group(1)):
        raise TransportError("DLA_TRANSFER_SHA_MISMATCH")
    version_match = re.search(r"Version:\s*([^\r\n]+)", identity_output)
    command = "env LD_LIBRARY_PATH={} {} --arch={} --no-verify --dla-file {} {}".format(
        ncc_lib, ncc_bin, _NCC_TARGET, remote_dla, remote_src
    )
    result = {
        "schema": "tflite_tp.phase3.ncc-build.v1",
        "status": "PASS",
        "compile_recovered_without_rebuild": True,
        "source_tflite": str(source),
        "source_tflite_sha256": local_sha,
        "ncc_host": compiler_host,
        "ncc_binary": ncc_bin,
        "ncc_library": ncc_lib,
        "ncc_binary_sha256": re.search(r"\b([0-9a-fA-F]{64})\s+.*ncc-tflite", identity_output).group(1).lower() if re.search(r"\b([0-9a-fA-F]{64})\s+.*ncc-tflite", identity_output) else None,
        "ncc_version": version_match.group(1).strip() if version_match else "NOT_CAPTURED",
        "target_arch": _NCC_TARGET,
        "verify_mode": "NO_VERIFY",
        "command": command,
        "exit_code": 0,
        "remote_compile_log": remote_log,
        "local_transport_compile_log": str(local_compile_log),
        "remote_dla_sha256": remote_dla_sha.group(1).lower(),
        "dla_path": str(dl_path),
        "dla_sha256": local_dla_sha,
        "dla_bytes": dl_path.stat().st_size,
        "execution_plan_log": str(out / "execution_plan.log"),
        "io_info_log": str(out / "io_info.log"),
        "ncc_stdout_stderr_log": str(out / "ncc_stdout_stderr.log"),
        "ncc_identity": identity_output,
        "first_failure": None,
    }
    (out / "NCC_BUILD_MANIFEST.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def invoke_npu(
    model_path,
    dla_path,
    compile_manifest_path,
    input_dir,
    input_manifest_path,
    device_host,
    remote_root,
    output_dir,
    run_id,
    timeout_seconds=900,
    profile=None,
    rank=None,
):
    """Stage one DLA and verified inputs, recheck EVK idle, invoke once, collect all outputs."""
    model_path = Path(model_path).resolve()
    dla_path = Path(dla_path).resolve()
    input_dir = Path(input_dir).resolve()
    out = Path(output_dir).resolve()
    compile_manifest = read_json(compile_manifest_path)
    input_manifest = read_json(input_manifest_path)
    contract = load_tflite_contract(model_path)
    if not _SAFE_REMOTE.fullmatch(remote_root):
        raise TransportError("REMOTE_ROOT_INVALID")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise TransportError("RUN_ID_INVALID")
    if int(timeout_seconds) <= 0 or int(timeout_seconds) > 3600:
        raise TransportError("NPU_TIMEOUT_OUT_OF_RANGE")
    if compile_manifest.get("status") != "PASS" or compile_manifest.get("source_tflite_sha256") != contract["sha256"]:
        raise TransportError("NCC_BUILD_MANIFEST_GATE_FAIL")
    if not dla_path.is_file() or sha256_file(dla_path) != compile_manifest.get("dla_sha256"):
        raise TransportError("DLA_SHA_GATE_FAIL")
    if input_manifest.get("status") != "PASS_CONTRACT_AND_SHA_VERIFIED":
        raise TransportError("INPUT_CONTRACT_GATE_FAIL")
    if input_manifest.get("candidate_tflite_sha256") != contract["sha256"]:
        raise TransportError("INPUT_MODEL_SHA_GATE_FAIL")
    # Check before writing this run's gate evidence; the directory will no longer
    # be empty after EVK_IDLE_GATE_INITIAL.json is recorded below.
    if out.exists() and any(out.iterdir()):
        raise TransportError("OUTPUT_DIRECTORY_NOT_EMPTY")
    out.mkdir(parents=True, exist_ok=True)
    rows = input_manifest.get("inputs", [])
    if len(rows) != len(contract["inputs"]):
        raise TransportError("INPUT_COUNT_MISMATCH")
    output_bytes = sum(
        math.prod(spec["shape"]) * np.dtype(DTYPE_NAMES[spec["dtype"]]).itemsize
        for spec in contract["outputs"]
    )
    input_bytes = sum(int(x["bytes"]) for x in rows)
    required_bytes = dla_path.stat().st_size + input_bytes + output_bytes
    required_memory_bytes = dla_path.stat().st_size + input_bytes + output_bytes + 512 * 1024 * 1024
    initial_gate = device_gate(device_host, remote_root, required_bytes, required_memory_bytes)
    (out / "EVK_IDLE_GATE_INITIAL.json").write_text(json.dumps(initial_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if initial_gate.get("gate") != "PASS":
        record_blocked_invoke(
            model_path, dla_path, compile_manifest_path, input_manifest_path,
            device_host, out, run_id, initial_gate, "EVK_BUSY_OR_UNCONFIRMED",
        )
        raise TransportError("EVK_BUSY_OR_UNCONFIRMED")
    for index, (row, spec) in enumerate(zip(rows, contract["inputs"])):
        if int(row["handle"]) != index or row["semantic_tensor"] != spec["name"]:
            raise TransportError("INPUT_HANDLE_ORDER_MISMATCH")
        path = input_dir / "input{}.bin".format(index)
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise TransportError("INPUT_SHA_GATE_FAIL:{}".format(index))
    remote_dir = remote_root.rstrip("/") + "/" + run_id
    if not _SAFE_REMOTE.fullmatch(remote_dir):
        raise TransportError("REMOTE_RUN_PATH_INVALID")
    scp = _check_program("scp")
    ssh_config_args = _ssh_config_args()
    _ssh(device_host, "set -eu; test ! -e {path}; mkdir -m 700 -p {path}".format(path=shlex.quote(remote_dir)), timeout=30)
    for local_file, remote_name in [(dla_path, "model.dla")]:
        result = _run([scp, *ssh_config_args, str(local_file), "{}:{}/{}".format(device_host, remote_dir, remote_name)], timeout=600)
        if result.returncode != 0:
            raise TransportError("EVK_STAGE_DLA_FAIL\n" + result.stdout[-12000:])
    for index in range(len(rows)):
        local_file = input_dir / "input{}.bin".format(index)
        result = _run([scp, *ssh_config_args, str(local_file), "{}:{}/input{}.bin".format(device_host, remote_dir, index)], timeout=300)
        if result.returncode != 0:
            raise TransportError("EVK_STAGE_INPUT_FAIL:{}\n{}".format(index, result.stdout[-12000:]))
    input_sum_text = "".join("{}  input{}.bin\n".format(row["sha256"], index) for index, row in enumerate(rows))
    # The sums file is sent separately; this also records the exact checked input manifest remotely.
    local_sums = out / "INPUT_SHA256SUMS"
    local_sums.write_text(input_sum_text, encoding="ascii")
    result = _run([scp, *ssh_config_args, str(local_sums), "{}:{}/INPUT_SHA256SUMS".format(device_host, remote_dir)], timeout=60)
    if result.returncode != 0:
        raise TransportError("EVK_STAGE_INPUT_MANIFEST_FAIL")
    dla_sha = sha256_file(dla_path)
    local_dla_sum = out / "DLA_SHA256SUMS"
    local_dla_sum.write_text("{}  model.dla\n".format(dla_sha), encoding="ascii")
    result = _run([scp, *ssh_config_args, str(local_dla_sum), "{}:{}/DLA_SHA256SUMS".format(device_host, remote_dir)], timeout=60)
    if result.returncode != 0:
        raise TransportError("EVK_STAGE_DLA_MANIFEST_FAIL")

    # Recheck immediately before the only NPU command in this function.
    invoke_gate = device_gate(device_host, remote_root, required_bytes, required_memory_bytes)
    (out / "EVK_IDLE_GATE_BEFORE_INVOKE.json").write_text(json.dumps(invoke_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if invoke_gate.get("gate") != "PASS":
        record_blocked_invoke(
            model_path, dla_path, compile_manifest_path, input_manifest_path,
            device_host, out, run_id, invoke_gate, "EVK_BUSY_OR_UNCONFIRMED_BEFORE_INVOKE",
        )
        raise TransportError("EVK_BUSY_OR_UNCONFIRMED_BEFORE_INVOKE")

    input_names = ",".join("input{}.bin".format(index) for index in range(len(rows)))
    output_names = ",".join("output{}.bin".format(index) for index in range(len(contract["outputs"])))
    size_checks = "\n".join(
        "test \"$(wc -c < output{index}.bin)\" -eq {size} || {{ echo NPU_OUTPUT_SIZE_FAIL=output{index}.bin; exit 85; }}".format(
            index=index,
            size=int(math.prod(spec["shape"]) * np.dtype(DTYPE_NAMES[spec["dtype"]]).itemsize),
        ) for index, spec in enumerate(contract["outputs"])
    )
    remote_script = """set -eu
cd {root}
sha256sum -c DLA_SHA256SUMS
sha256sum -c INPUT_SHA256SUMS
printf 'EVK_HOSTNAME=%s\\n' "$(hostname)"
printf 'NEURONRT_SHA256='; sha256sum /usr/sbin/neuronrt | awk '{{print $1}}'
printf 'EVK_MEM_BEFORE='; awk '/MemAvailable:/ {{print $2}}' /proc/meminfo
printf 'NEURONRT_COMMAND=/usr/sbin/neuronrt -m hw -a model.dla -i {inputs} -o {outputs} -d\\n'
start=$(date +%s%N)
set +e
    timeout --signal=TERM --kill-after=10s {timeout_seconds}s /usr/sbin/neuronrt -m hw -a model.dla -i {inputs} -o {outputs} -d > neuronrt.log 2>&1
rc=$?
set -e
end=$(date +%s%N)
printf 'NEURONRT_START_EPOCH_NS=%s\\nNEURONRT_END_EPOCH_NS=%s\\nNEURONRT_EXIT_CODE=%s\\n' "$start" "$end" "$rc"
cat neuronrt.log
if [ "$rc" -eq 0 ]; then
{sizes}
sha256sum output*.bin > OUTPUT_SHA256SUMS
printf 'NPU_OUTPUT_COMPLETE=PASS\\n'
else
printf 'NPU_OUTPUT_COMPLETE=FAIL\\n'
fi
printf 'EVK_MEM_AFTER='; awk '/MemAvailable:/ {{print $2}}' /proc/meminfo
exit "$rc"
""".format(
        root=shlex.quote(remote_dir), inputs=shlex.quote(input_names), outputs=shlex.quote(output_names),
        sizes=size_checks, timeout_seconds=int(timeout_seconds)
    )
    ssh = _check_program("ssh")
    invoke_started = time.time()
    result = _run([ssh, *ssh_config_args, "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", device_host, remote_script], int(timeout_seconds) + 30)
    invoke_elapsed = time.time() - invoke_started
    remote_output = result.stdout
    (out / "neuronrt.log").write_text(remote_output, encoding="utf-8", errors="replace")
    # Always attempt to collect the unique run directory, including failure logs.
    collected = out / "raw_outputs"
    collected.mkdir(exist_ok=False)
    collect_result = _run([scp, *ssh_config_args, "-r", "{}:{}/.".format(device_host, remote_dir), str(collected)], timeout=600)
    if collect_result.returncode != 0:
        raise TransportError("NPU_OUTPUT_COLLECTION_FAILED\n" + collect_result.stdout[-12000:])
    raw_sums = collected / "OUTPUT_SHA256SUMS"
    output_rows = []
    if result.returncode == 0 and raw_sums.is_file():
        for index, spec in enumerate(contract["outputs"]):
            path = collected / "output{}.bin".format(index)
            expected_bytes = int(math.prod(spec["shape"]) * np.dtype(DTYPE_NAMES[spec["dtype"]]).itemsize)
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise TransportError("NPU_OUTPUT_INCOMPLETE:{}".format(index))
            output_rows.append({
                "handle_index": index,
                "semantic_tensor": spec["name"],
                "shape": spec["shape"],
                "dtype": spec["dtype"],
                "quantization": spec["quantization"],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": "PASS",
            })
    if raw_sums.is_file():
        for line in raw_sums.read_text(encoding="ascii").splitlines():
            expected, name = line.split(None, 1)
            path = collected / name.strip().lstrip("*")
            if not path.is_file() or sha256_file(path) != expected:
                raise TransportError("COLLECTED_OUTPUT_SHA_MISMATCH:" + name)
    start_match = re.search(r"NEURONRT_START_EPOCH_NS=(\d+)", remote_output)
    end_match = re.search(r"NEURONRT_END_EPOCH_NS=(\d+)", remote_output)
    rc_match = re.search(r"NEURONRT_EXIT_CODE=(\d+)", remote_output)
    status = "PASS" if result.returncode == 0 and rc_match and int(rc_match.group(1)) == 0 and len(output_rows) == len(contract["outputs"]) else "FAIL"
    run_manifest = {
        "schema": "tflite_tp.phase3.npu-run.v1",
        "status": status,
        "profile": profile or input_manifest.get("profile"),
        "fixture_id": input_manifest.get("fixture_id"),
        "fixture_sha256": input_manifest.get("fixture_sha256"),
        "tp_degree": input_manifest.get("tp_degree"),
        "rank": int(rank) if rank is not None else input_manifest.get("rank"),
        "source_tflite_sha256": contract["sha256"],
        "dla_sha256": dla_sha,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "input_sha256s": [{"handle": row["handle"], "semantic_tensor": row["semantic_tensor"], "sha256": row["sha256"]} for row in rows],
        "device_host": device_host,
        "remote_run_directory": remote_dir,
        "device_idle_gate_initial": initial_gate,
        "device_idle_gate_before_invoke": invoke_gate,
        "invoke_exit_code": int(rc_match.group(1)) if rc_match else result.returncode,
        "invoke_wall_seconds_including_ssh_and_io": invoke_elapsed,
        "invoke_wall_ns": int(end_match.group(1)) - int(start_match.group(1)) if start_match and end_match else None,
        "invoke_command": "/usr/sbin/neuronrt -m hw -a model.dla -i {} -o {} -d".format(input_names, output_names),
        "runtime_log_sha256": sha256_file(out / "neuronrt.log"),
        "outputs": output_rows,
        "output_coverage": "COMPLETE" if status == "PASS" else "INCOMPLETE",
        "first_failure": None if status == "PASS" else "NPU_INVOKE_OR_OUTPUT_FAILURE",
    }
    (out / "NPU_RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "PASS":
        raise TransportError("NPU_INVOKE_OR_OUTPUT_GATE_FAIL")
    return run_manifest
