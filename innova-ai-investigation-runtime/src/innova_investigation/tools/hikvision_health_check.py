#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class CheckResult:
    target: str
    ok: bool
    detail: str
    latency_ms: int | None = None


def now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def check_tcp(host: str, port: int, timeout: float) -> CheckResult:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return CheckResult(f"tcp:{port}", True, f"port {port} open", int((time.perf_counter() - started) * 1000))
    except OSError as exc:
        return CheckResult(f"tcp:{port}", False, f"port {port} failed: {exc}", int((time.perf_counter() - started) * 1000))


def parse_channels(value: str) -> list[int]:
    channels: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            channels.extend(range(int(start_raw), int(end_raw) + 1))
        else:
            channels.append(int(part))
    return sorted(set(channels))


def channel_stream_ids(channel: int) -> list[int]:
    # Hikvision commonly exposes channel 1 as 101/102, channel 10 as 1001/1002.
    # Try substream first because snapshots are faster/lighter.
    return [channel * 100 + 2, channel * 100 + 1]


def is_jpeg(body: bytes) -> bool:
    return body[:2] == b"\xff\xd8"


def build_opener(host: str, http_port: int, username: str, password: str) -> urllib.request.OpenerDirector:
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, f"http://{host}:{http_port}/", username, password)
    return urllib.request.build_opener(
        urllib.request.HTTPDigestAuthHandler(password_mgr),
        urllib.request.HTTPBasicAuthHandler(password_mgr),
    )


def check_snapshot(
    opener: urllib.request.OpenerDirector,
    host: str,
    http_port: int,
    channel: int,
    timeout: float,
    save_dir: Path | None,
) -> CheckResult:
    started = time.perf_counter()
    last_detail = ""

    for stream_id in channel_stream_ids(channel):
        url = f"http://{host}:{http_port}/ISAPI/Streaming/channels/{stream_id}/picture"
        request = urllib.request.Request(url, headers={"User-Agent": "InnovaHealthProbe/0.1"})
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read()
                latency_ms = int((time.perf_counter() - started) * 1000)
                content_type = response.headers.get("Content-Type", "")
                if len(body) < 512:
                    last_detail = f"stream {stream_id}: snapshot too small ({len(body)} bytes, {content_type})"
                    continue
                if not is_jpeg(body):
                    preview = body[:80].decode("utf-8", errors="replace").replace("\n", " ")
                    last_detail = f"stream {stream_id}: not jpeg ({content_type}): {preview}"
                    continue
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    (save_dir / f"channel_{channel:02d}_stream_{stream_id}.jpg").write_bytes(body)
                return CheckResult(
                    f"channel:{channel}",
                    True,
                    f"snapshot OK via stream {stream_id} ({len(body)} bytes)",
                    latency_ms,
                )
        except urllib.error.HTTPError as exc:
            last_detail = f"stream {stream_id}: HTTP {exc.code} {exc.reason}"
        except urllib.error.URLError as exc:
            last_detail = f"stream {stream_id}: url error {exc.reason}"
        except Exception as exc:
            last_detail = f"stream {stream_id}: error {exc}"

    latency_ms = int((time.perf_counter() - started) * 1000)
    return CheckResult(f"channel:{channel}", False, last_detail or "snapshot failed", latency_ms)


def print_result(result: CheckResult) -> None:
    status = "OK" if result.ok else "FAIL"
    latency = f" {result.latency_ms}ms" if result.latency_ms is not None else ""
    print(f"[{status}] {result.target:<12} {latency:<8} {result.detail}")


def run_once(args: argparse.Namespace, channels: list[int]) -> dict:
    snapshot_dir = (
        Path("snapshots") / datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.save_snapshots
        else None
    )
    results: list[CheckResult] = []

    ports = [args.http_port, args.sdk_port]
    if args.rtsp_port > 0:
        ports.append(args.rtsp_port)
    for port in sorted(set(ports)):
        result = check_tcp(args.host, port, args.timeout)
        results.append(result)
        print_result(result)

    print("-" * 88)
    opener = build_opener(args.host, args.http_port, args.username, args.password)
    for channel in channels:
        result = check_snapshot(opener, args.host, args.http_port, channel, args.timeout, snapshot_dir)
        results.append(result)
        print_result(result)

    return {
        "name": args.name,
        "host": args.host,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "ok": sum(1 for result in results if result.ok),
        "fail": sum(1 for result in results if not result.ok),
        "results": [asdict(result) for result in results],
    }


def print_state_changes(previous: dict[str, bool], summary: dict, events_path: Path | None) -> dict[str, bool]:
    current = {item["target"]: bool(item["ok"]) for item in summary["results"]}
    for target, ok in current.items():
        if target not in previous or previous[target] == ok:
            continue
        event = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "target": target,
            "event": "RECOVERED" if ok else "OFFLINE",
            "nvr": summary["name"],
            "host": summary["host"],
        }
        print(f"EVENT: {event['event']} {target} at {event['time']}")
        if events_path:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event) + "\n")
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight Hikvision NVR/camera health probe.")
    parser.add_argument("--name", default="Lakewood Richmond")
    parser.add_argument("--host", default="innrichmond.ddns.net")
    parser.add_argument("--http-port", type=int, default=81)
    parser.add_argument("--sdk-port", type=int, default=8000)
    parser.add_argument("--rtsp-port", type=int, default=0)
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="Inn0vatech1!")
    parser.add_argument("--channels", default="1-8", help="Examples: 1-16, 10, 1,2,10-12")
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--watch", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--events-out", default="output/health_events.jsonl")
    args = parser.parse_args()

    channels = parse_channels(args.channels)
    events_path = Path(args.events_out) if args.events_out else None
    previous: dict[str, bool] = {}
    iteration = 0
    last_summary: dict | None = None

    while True:
        iteration += 1
        print(f"\nInnova Camera Health MVP - {args.name}")
        print(f"{now_label()} | host={args.host} | channels={channels}")
        print("-" * 88)

        summary = run_once(args, channels)
        previous = print_state_changes(previous, summary, events_path)
        last_summary = summary

        print("-" * 88)
        print(f"Summary: {summary['ok']} OK / {summary['fail']} FAIL")

        if args.json_out:
            output = Path(args.json_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"JSON saved: {output}")

        if args.watch <= 0:
            break
        if args.iterations > 0 and iteration >= args.iterations:
            break
        print(f"Next check in {args.watch}s. Press Ctrl+C to stop.")
        time.sleep(args.watch)

    if last_summary is None:
        return 1
    return 0 if last_summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
