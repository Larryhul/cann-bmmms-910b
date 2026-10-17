"""Hold one password-authenticated SSH session and serve jobs from a file queue.

The rented instance's gateway accepts password authentication only, so the
password must be typed interactively. It is read with getpass, kept in this
process's memory for reconnects, and never written to disk or to a log.

Start it once:

    python learning/ssh_queue_agent.py \
        --host <SERVER_HOST> --port 32222 \
        --user <SERVER_USER> \
        --known-hosts research/moark_known_hosts \
        --queue-dir .ssh_queue

Then drop job files into the queue directory. A job is one JSON object in a
file named `<name>.job`; the reply lands in `<name>.out` and the job file is
renamed to `<name>.taken`.

    {"op": "run", "command": "ls /data", "timeout": 600}
    {"op": "upload", "local": "solution_v09c/kernel.asc", "remote": "/data/x/kernel.asc"}
    {"op": "download", "remote": "/data/x/result.json", "local": "research/result.json"}
    {"op": "close"}
"""

import argparse
import base64
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

import paramiko

CHUNK = 45000


def log(message):
    print(message, flush=True)


def run_remote(client, command, timeout):
    channel = client.get_transport().open_session(timeout=12)
    channel.settimeout(30)
    channel.exec_command(command)
    out = []
    err = []
    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        while channel.recv_ready():
            out.append(channel.recv(65536).decode("utf-8", errors="replace"))
        while channel.recv_stderr_ready():
            err.append(channel.recv_stderr(65536).decode("utf-8", errors="replace"))
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.05)
    code = None if timed_out else channel.recv_exit_status()
    channel.close()
    return {
        "exit_code": code,
        "timed_out": timed_out,
        "stdout": "".join(out),
        "stderr": "".join(err),
    }


def upload(client, local, remote):
    """Send a file over SSH exec; this platform's SFTP subsystem returns EOF."""
    payload = Path(local).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    parent = str(Path(remote).parent).replace("\\", "/")
    first = run_remote(client, "mkdir -p " + shlex.quote(parent), 60)
    if first["exit_code"] != 0:
        return {"ok": False, "stage": "mkdir", **first}
    for index in range(0, len(encoded), CHUNK):
        piece = encoded[index:index + CHUNK]
        mode = ">" if index == 0 else ">>"
        command = "printf %s " + shlex.quote(piece) + " " + mode + " " + shlex.quote(remote + ".b64")
        step = run_remote(client, command, 120)
        if step["exit_code"] != 0:
            return {"ok": False, "stage": "chunk", "offset": index, **step}
    decode = (
        "base64 -d " + shlex.quote(remote + ".b64") + " > " + shlex.quote(remote)
        + " && rm -f " + shlex.quote(remote + ".b64")
        + " && sha256sum " + shlex.quote(remote)
    )
    final = run_remote(client, decode, 120)
    remote_digest = final["stdout"].split()[0] if final["stdout"].split() else ""
    return {
        "ok": final["exit_code"] == 0 and remote_digest == digest,
        "bytes": len(payload),
        "local_sha256": digest,
        "remote_sha256": remote_digest,
        **final,
    }


def download(client, remote, local):
    result = run_remote(client, "base64 -w0 " + shlex.quote(remote), 300)
    if result["exit_code"] != 0:
        return {"ok": False, **result}
    data = base64.b64decode(result["stdout"].strip())
    target = Path(local)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"ok": True, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def connect(args, password):
    client = paramiko.SSHClient()
    client.load_host_keys(args.known_hosts)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        allow_agent=False,
        look_for_keys=False,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    client.get_transport().set_keepalive(30)
    return client


def alive(client):
    transport = client.get_transport() if client else None
    return bool(transport and transport.is_active())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--known-hosts", required=True)
    parser.add_argument("--queue-dir", required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    queue = Path(args.queue_dir)
    queue.mkdir(parents=True, exist_ok=True)

    password = getpass.getpass("SSH_INSTANCE_PASSWORD: ")
    client = connect(args, password)
    log("SSH_CONNECTED " + args.host + ":" + str(args.port))
    log("QUEUE_DIR " + str(queue.resolve()))
    log("Waiting for jobs. Ctrl-C to stop.")

    idle = 0
    while True:
        jobs = sorted(queue.glob("*.job"))
        if not jobs:
            idle += 1
            if idle % 120 == 0:
                log("IDLE " + time.strftime("%H:%M:%S"))
            time.sleep(0.5)
            continue
        idle = 0
        for job in jobs:
            name = job.stem
            taken = job.with_suffix(".taken")
            try:
                text = job.read_text(encoding="utf-8").strip()
                if not text:
                    # Seen before the writer flushed it; the next sweep gets it.
                    time.sleep(0.05)
                    continue
                request = json.loads(text)
            except Exception as error:
                if taken.exists():
                    taken.unlink()
                job.rename(taken)
                (queue / (name + ".out")).write_text(
                    json.dumps({"error": "bad_json: " + str(error)}), encoding="utf-8")
                continue
            if taken.exists():
                taken.unlink()
            job.rename(taken)
            operation = request.get("op", "run")
            log("JOB " + name + " " + operation)
            try:
                if not alive(client):
                    log("RECONNECTING")
                    client = connect(args, password)
                    log("SSH_RECONNECTED")
                if operation == "run":
                    result = run_remote(client, request["command"], request.get("timeout", 600))
                elif operation == "upload":
                    result = upload(client, request["local"], request["remote"])
                elif operation == "download":
                    result = download(client, request["remote"], request["local"])
                elif operation == "close":
                    (queue / (name + ".out")).write_text(
                        json.dumps({"closed": True}), encoding="utf-8")
                    log("SSH_CLOSED")
                    client.close()
                    return
                else:
                    result = {"error": "unknown_op: " + str(operation)}
            except Exception as error:
                result = {"error": type(error).__name__ + ": " + str(error)}
                # Gateways can retain an active TCP transport after losing
                # their backend channel. Reconnect for the next queued job.
                # Never retry a mutation implicitly: its outcome is unknown.
                if client:
                    client.close()
                client = None
            (queue / (name + ".out")).write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8")
            tail = result.get("stdout", "")[-200:] if isinstance(result, dict) else ""
            log("DONE " + name + " exit=" + str(result.get("exit_code")) + (" " + tail.replace("\n", " | ") if tail else ""))


if __name__ == "__main__":
    main()
