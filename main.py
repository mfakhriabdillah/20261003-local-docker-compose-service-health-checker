```python
#!/usr/bin/env python3
"""
Local Docker Compose Service Health Checker

A CLI tool for cloud/DevOps engineers to monitor the health, uptime,
and status of services defined in a Docker Compose stack.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import docker
    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServiceHealth:
    """Container for the health status of a single Docker service."""
    name: str
    state: str               # running | exited | restarting | unhealthy | starting
    status: str              # human-readable status string
    health: str              # healthy | unhealthy | none | starting
    uptime_seconds: float
    restart_count: int
    container_id: str
    ports: list[str]
    image: str
    project: str
    checked_at: str


@dataclass
class CheckConfig:
    """Runtime configuration for a health-check run."""
    compose_file: Path
    project_name: Optional[str]
    timeout: int
    watch_mode: bool
    watch_interval: float
    exit_on_unhealthy: bool
    output_format: str       # table | json
    verbose: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_compose_file(search_paths: list[Path]) -> Path:
    """Find the best candidate docker-compose file in the given search paths.

    Priority order:
        1. docker-compose.yml
        2. docker-compose.yaml
        3. docker-compose.override.yml
        4. docker-compose.prod.yml
    """
    for base in search_paths:
        for candidate in ("docker-compose.yml", "docker-compose.yaml",
                          "docker-compose.override.yml", "docker-compose.prod.yml"):
            full = base / candidate
            if full.exists():
                logger.debug("Detected compose file: %s", full)
                return full
    raise FileNotFoundError(
        "No docker-compose file found. "
        "Place one in the current directory or specify --file."
    )


def parse_bool(value: Optional[str]) -> bool:
    """Convert a CLI string to a boolean, tolerating common variants."""
    if value is None:
        return False
    return value.lower() in ("1", "true", "yes", "on")


def format_uptime(seconds: float) -> str:
    """Return a human-readable uptime string (e.g. '2h 15m 30s')."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"
    hours, remainder = divmod(int(seconds), 3600)
    mins, secs = divmod(remainder, 60)
    parts = [f"{hours}h"]
    if mins:
        parts.append(f"{mins}m")
    if secs:
        parts.append(f"{secs}s")
    return " ".join(parts)


def run_subprocess(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except Exception as exc:
        return -1, "", str(exc)


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

class DockerComposeHealthChecker:
    """Checks the health of every service in a Docker Compose stack."""

    def __init__(self, config: CheckConfig):
        self.config = config
        self._compose_cmd = self._build_compose_cmd()
        self._cli_available = self._check_cli()

    # -- private helpers ---------------------------------------------------

    def _build_compose_cmd(self) -> list[str]:
        cmd = ["docker", "compose"]
        if self.config.compose_file.exists():
            cmd += ["-f", str(self.config.compose_file)]
        if self.config.project_name:
            cmd += ["--project-name", self.config.project_name]
        return cmd

    def _check_cli(self) -> bool:
        rc, _, _ = run_subprocess(["docker", "compose", "version"])
        available = rc == 0
        logger.info("docker compose CLI %s", "available" if available else "NOT available")
        return available

    def _services_list(self) -> list[str]:
        """Return ordered list of service names from the compose file."""
        if self._cli_available:
            rc, out, err = run_subprocess(
                [*self._compose_cmd, "ps", "--format", "{{.Service}}", "--no-trunc"],
                timeout=self.config.timeout
            )
            if rc == 0 and out:
                return [s.strip() for s in out.splitlines() if s.strip()]
            logger.warning("Failed to list services via compose CLI: %s", err or rc)

        if DOCKER_SDK_AVAILABLE:
            client = self._docker_client()
            try:
                stacks = client.api.projects()
                target = self.config.project_name or next(
                    (p for p in stacks if not self.config.project_name), None
                )
                if target:
                    services = client.api.services(project_ids=[target])
                    return [s.get("Name", s.get("ID", "")) for s in services]
            except Exception as exc:
                logger.warning("Docker SDK project listing failed: %s", exc)

        # Fallback: parse yaml manually for service keys
        return self._parse_services_from_yaml()

    def _parse_services_from_yaml(self) -> list[str]:
        """Extract top-level service keys from a YAML compose file (lightweight)."""
        services = []
        try:
            import yaml
        except ImportError:
            logger.warning("pyyaml not installed; cannot parse compose file for service list.")
            return services

        try:
            with open(self.config.compose_file) as fh:
                data = yaml.safe_load(fh) or {}
            for svc in data.get("services", {}):
                services.append(svc)
        except Exception as exc:
            logger.error("Error parsing compose YAML: %s", exc)
        return services

    def _docker_client(self):
        if not DOCKER_SDK_AVAILABLE:
            raise RuntimeError("docker Python SDK is not installed.")
        return docker.from_env()

    # -- public API --------------------------------------------------------

    def check_all(self) -> list[ServiceHealth]:
        """Run health checks for every service and return results."""
        services = self._services_list()
        if not services:
            logger.error("No services discovered. Is the stack running?")
            return []

        logger.info("Discovered %d service(s): %s", len(services), ", ".join(services))
        results = []
        for svc_name in services:
            try:
                health = self._check_service(svc_name)
            except Exception as exc:
                logger.error("Failed to check service '%s': %s", svc_name, exc)
                continue
            results.append(health)
        return results

    def _check_service(self, name: str) -> ServiceHealth:
        """Return a ServiceHealth for a single service."""
        if self._cli_available:
            return self._check_via_compose_cli(name)
        if DOCKER_SDK_AVAILABLE:
            return self._check_via_sdk(name)
        raise RuntimeError(
            "Neither docker compose CLI nor docker Python SDK are available."
        )

    def _check_via_compose_cli(self, name: str) -> ServiceHealth:
        """Use `docker compose ps --format json` for rich service info."""
        cmd = [
            *self._compose_cmd,
            "ps", "--format", "json", "--no-trunc", name
        ]
        rc, out, err = run_subprocess(cmd, timeout=self.config.timeout)
        if rc != 0:
            raise RuntimeError(f"compose ps failed: {err or rc}")

        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            lines = [l for l in out.splitlines() if l.strip()]
            if not lines:
                raise RuntimeError("Empty output from compose ps")
            data = json.loads(lines[0])

        state = data.get("State", "unknown")
        health_val = data.get("Health", "none")
        uptime_raw = data.get("RunningDurationForHumanReadable", "unknown")
        uptime_sec = data.get("RunningDuration", 0.0)
        if isinstance(uptime_sec, str):
            # Some versions emit a duration string; fall back to 0
            uptime_sec = 0.0

        return ServiceHealth(
            name=name,
            state=state,
            status=data.get("Status", "-"),
            health=health_val,
            uptime_seconds=float(uptime_sec),
            restart_count=int(data.get("RestartCount", 0)),
            container_id=data.get("Name", "")[:12],
            ports=self._parse_ports(data.get("Ports", [])),
            image=data.get("Image", "-"),
            project=data.get("Project", self.config.project_name or "-"),
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def _parse_ports(self, raw) -> list[str]:
        """Normalize port mappings from compose JSON into a readable list."""
        if not raw:
            return []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(p) for p in raw if p]
        return [str(raw)]

    def _check_via_sdk(self, name: str) -> ServiceHealth:
        """Fallback: use the docker Python SDK directly."""
        client = self._docker_client()
        containers = client.containers.list(
            filters={"label": f"com.docker.compose.service={name}"}, all=False
        )
        if not containers:
            containers = client.containers.list(
                filters={"label": f"compose.service={name}"}, all=False
            )

        if not containers:
            return ServiceHealth(
                name=name, state="not_found", status="No running container",
                health="none", uptime_seconds=0.0, restart_count=0,
                container_id="", ports=[], image="-",
                project=self.config.project_name or "-",
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

        c = containers[0]
        inspect = c.attrs
        health = inspect.get("State", {}).get("Health", {})
        status = inspect.get("State", {}).get("Status", "unknown")
        started_at = inspect.get("State", {}).get("StartedAt", "")
        uptime_sec = 0.0
        if started_at and started_at != "0001-01-01T00:00:00Z":
            try:
                sat = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                uptime_sec = max(0.0, (datetime.now(timezone.utc) - sat).total_seconds())
            except ValueError:
                pass

        ports_map = inspect.get("NetworkSettings", {}).get("Ports", {}) or {}
        ports = []
        for host_port, bind in ports_map.items():
            if bind:
                for b in bind:
                    ports.append(f"{b['HostIp']}:{b['HostPort']}->{host_port}")
            else:
                ports.append(host_port)

        return ServiceHealth(
            name=name,
            state=status,
            status=c.status,
            health=(health or {}).get("Status", "none"),
            uptime_seconds=uptime_sec,
            restart_count=int(inspect.get("RestartCount", 0)),
            container_id=c.short_id,
            ports=ports,
            image=c.image.tags[0] if c.image.tags else str(c.image),
            project=inspect.get("Config", {}).get("Labels", {}).get(
                "com.docker.compose.project", "-"
            ),
            checked_at=datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def print_table(results: list[ServiceHealth]) -> None:
    """Print results as an aligned ASCII table."""
    if not results:
        print("No services found or all checks failed.", file=sys.stderr)
        return

    header = (
        f"{'SERVICE':<25} {'STATE':<12} {'HEALTH':<12} "
        f"{'UPTIME':>12} {'RESTARTS':>8}  {'CONTAINER':<14} {'PORTS'}"
    )
    separator = "─" * len(header)
    print(header)
    print(separator)

    for r in results:
        uptime_str = format_uptime(r.uptime_seconds) if r.state == "running" else "-"
        ports_str = ", ".join(r.ports[:2]) if r.ports else "-"
        print(
            f"{r.name:<25} {r.state:<12} {r.health:<12} "
            f"{uptime_str:>12} {r.restart_count:>8}  {r.container_id:<14} {ports_str}"
        )

    print(separator)
    healthy = sum(1 for r in results if r.health == "healthy" or r.health == "none")
    total = len(results)
    print(f"\n{healthy}/{total} service(s) OK  |  Checked at {results[0].checked_at}")


def print_json(results: list[ServiceHealth]) -> None:
    """Print results as pretty JSON."""
    payload = {
        "checked_at": results[0].checked_at if results else None,
        "total": len(results),
        "services": [asdict(r) for r in results],
    }
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compose-health",
        description="Check the health of services in a local Docker Compose stack.",
        epilog=(
            "Examples:\n"
            "  %(prog)s                     Check services in current directory\n"
            "  %(prog)s -f /proj/compose.yml -n myproject\n"
            "  %(prog)s --watch --interval 10\n"
            "  %(prog)s --json --exit-unhealthy\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-f", "--file",
        type=Path,
        help="Path to docker-compose file (default: auto-detect in CWD and parents).",
    )
    parser.add_argument(
        "-p", "--project-name",
        help="Override the Compose project name (default: derived from directory).",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="Timeout in seconds for each docker compose CLI call (default: 30).",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Enable continuous monitoring: re-check at --interval second intervals.",
    )
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="Check interval in seconds when --watch is set (default: 10).",
    )
    parser.add_argument(
        "--exit-unhealthy", action="store_true",
        help="Exit with code 2 if any service is unhealthy (useful for CI/CD).",
    )
    parser.add_argument(
        "--json", dest="output_format", action="store_const", const="json", default="table",
        help="Output results as JSON instead of a human-readable table.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging to stderr.",
    )
    parser.add_argument(
        "--version", action="version", version="compose-health 1.0.0",
    )
    return parser


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stderr,
    )


def determine_compose_file(cli_path: Optional[Path]) -> Path:
    """Resolve the compose file path, searching upwards from cwd if needed."""
    if cli_path and cli_path.exists():
        return cli_path.resolve()

    search: list[Path] = [Path.cwd()]
    # Walk up to 5 parent directories
    current = Path.cwd()
    for _ in range(5):
        current = current.parent
        if current == current.parent:  # reached filesystem root
            break
        search.append(current)

    return detect_compose_file(search)


def run_once(config: CheckConfig) -> list[ServiceHealth]:
    checker = DockerComposeHealthChecker(config)
    return checker.check_all()


def run_loop(config: CheckConfig) -> None:
    """Continuous watch loop."""
    logger.info("Starting watch mode (interval=%.1fs). Press Ctrl+C to stop.", config.watch_interval)
    try:
        while True:
            print("\033[H\033[J", end="")  # clear screen (POSIX)
            results = run_once(config)
            if config.output_format == "json":
                print_json(results)
            else:
                print_table(results)
            time.sleep(config.watch_interval)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    logger.debug("Arguments: %s", args)

    try:
        compose_file = determine_compose_file(args.file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not compose_file.exists():
        print(f"ERROR: Compose file not found: {compose_file}", file=sys.stderr)
        return 1

    config = CheckConfig(
        compose_file=compose_file,
        project_name=args.project_name,
        timeout=args.timeout,
        watch_mode=args.watch,
        watch_interval=args.interval,
        exit_on_unhealthy=args.exit_unhealthy,
        output_format=args.output_format,
        verbose=args.verbose,
    )

    logger.info("Using compose file: %s", config.compose_file)

    if config.watch_mode:
        run_loop(config)
        return 0

    results = run_once(config)
    if not results:
        print("No services to report.", file=sys.stderr)
        return 1

    if config.output_format == "json":
        print_json(results)
    else:
        print_table(results)

    if config.exit_on_unhealthy:
        unhealthy = [r for r in results if r.health == "unhealthy"]
        if unhealthy:
            names = ", ".join(r.name for r in unhealthy)
            print(
                f"\nWARNING: {len(unhealthy)} unhealthy service(s): {names}",
                file=sys.stderr,
            )
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
```