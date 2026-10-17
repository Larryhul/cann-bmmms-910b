"""Submit one job to the running ssh_queue_agent and print its reply."""

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

DEFAULT_QUEUE = Path(__file__).resolve().parent.parent / ".ssh_queue"


def submit(request, wait, QUEUE):
    QUEUE.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:6]
    (QUEUE / (name + ".job")).write_text(json.dumps(request), encoding="utf-8")
    reply = QUEUE / (name + ".out")
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if reply.exists():
            time.sleep(0.1)
            return json.loads(reply.read_text(encoding="utf-8"))
        time.sleep(0.2)
    return {"error": "no reply within " + str(wait) + "s; is the agent running?"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", default="run", choices=["run", "upload", "download", "close"])
    parser.add_argument("--command")
    parser.add_argument("--local")
    parser.add_argument("--remote")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--wait", type=int)
    parser.add_argument("--raw", action="store_true", help="print the reply as JSON")
    parser.add_argument("--queue", help="queue directory; default .ssh_queue")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    request = {"op": args.op, "timeout": args.timeout}
    if args.command:
        request["command"] = args.command
    if args.local:
        request["local"] = args.local
    if args.remote:
        request["remote"] = args.remote

    queue = Path(args.queue) if args.queue else DEFAULT_QUEUE
    result = submit(request, args.wait or args.timeout + 60, queue)
    if args.raw:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if "error" in result:
        print("ERROR " + str(result["error"]))
        sys.exit(1)
    if result.get("stdout"):
        print(result["stdout"], end="")
    if result.get("stderr"):
        print("--- stderr ---", file=sys.stderr)
        print(result["stderr"], end="", file=sys.stderr)
    extras = {k: v for k, v in result.items() if k not in ("stdout", "stderr")}
    print("--- " + json.dumps(extras, ensure_ascii=False))
    if result.get("exit_code") not in (0, None) or result.get("ok") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
