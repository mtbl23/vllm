#!/usr/bin/env python3
"""Loopback-only TCP bridge through an SSH stdio command.

Some GPU providers prohibit SSH direct-tcpip channels even while permitting a
normal remote command. This development bridge accepts local loopback TCP and
relays each connection through an exact `nc 127.0.0.1 PORT` command. It never
binds a LAN/public address and never carries credentials on its command line.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
from pathlib import Path


async def relay(
    source: asyncio.StreamReader,
    destination: asyncio.StreamWriter | asyncio.subprocess.Process,
) -> None:
    writer = destination.stdin if isinstance(destination, asyncio.subprocess.Process) else destination
    if writer is None:
        return
    try:
        while data := await source.read(65536):
            writer.write(data)
            await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


async def serve(args: argparse.Namespace) -> None:
    semaphore = asyncio.Semaphore(args.max_connections)

    async def client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        if not isinstance(peer, tuple) or peer[0] not in {"127.0.0.1", "::1"}:
            writer.close()
            await writer.wait_closed()
            return
        async with semaphore:
            process = await asyncio.create_subprocess_exec(
                "/usr/bin/ssh",
                "-T",
                "-i",
                str(args.identity_file),
                "-o",
                "BatchMode=yes",
                "-o",
                "ClearAllForwardings=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "PermitLocalCommand=no",
                "-o",
                "RequestTTY=no",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={args.known_hosts}",
                "-p",
                str(args.ssh_port),
                f"{args.ssh_user}@{args.ssh_host}",
                "exec",
                "nc",
                "127.0.0.1",
                str(args.remote_port),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            assert process.stdout is not None
            inbound = asyncio.create_task(relay(reader, process))
            outbound = asyncio.create_task(relay(process.stdout, writer))
            try:
                await asyncio.gather(inbound, outbound)
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=3)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()

    server = await asyncio.start_server(
        client,
        host="127.0.0.1",
        port=args.listen_port,
        limit=2 * 1024 * 1024,
        start_serving=True,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(name, stop.set)
    sockets = server.sockets or []
    if len(sockets) != 1 or sockets[0].getsockname()[0] != "127.0.0.1":
        server.close()
        await server.wait_closed()
        raise RuntimeError("bridge did not bind exactly one IPv4 loopback socket")
    print(f"ready http://127.0.0.1:{args.listen_port}", flush=True)
    await stop.wait()
    server.close()
    await server.wait_closed()


def canonical_file(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise argparse.ArgumentTypeError(f"{label} must be an absolute regular file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise argparse.ArgumentTypeError(f"{label} must be canonical")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--identity-file", required=True)
    parser.add_argument("--known-hosts", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--max-connections", type=int, default=64)
    args = parser.parse_args()

    for label in ("listen_port", "ssh_port", "remote_port"):
        value = getattr(args, label)
        if value < 1 or value > 65535:
            parser.error(f"{label.replace('_', '-')} must be a valid TCP port")
    if args.max_connections < 1 or args.max_connections > 256:
        parser.error("max-connections must be between 1 and 256")
    args.identity_file = canonical_file(args.identity_file, "identity file")
    args.known_hosts = canonical_file(args.known_hosts, "known-hosts file")
    if os.geteuid() == 0:
        parser.error("bridge must run as an unprivileged local user")

    asyncio.run(serve(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
