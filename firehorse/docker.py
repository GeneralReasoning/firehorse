"""Docker isolation for firehorse evaluations.

When --docker is passed, the entire evaluation runs inside a Docker container.
The host CLI builds/validates the image, then invokes `docker run` with the
same firehorse arguments. Output files are accessible via volume mount.

The Dockerfile and pyproject.toml are embedded as strings so the build is
fully self-contained — no repo root detection needed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from firehorse.config import RunConfig


DEFAULT_IMAGE = "firehorse:latest"

# Environment variables to pass through to the container
PASSTHROUGH_ENV_VARS = [
    "OPENREWARD_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENREWARD_URL",
]

DOCKERFILE = """\
# Stage 1: Node.js — install claude and codex CLIs
FROM node:22-slim AS node-deps

RUN npm install -g @anthropic-ai/claude-code @openai/codex

# Stage 2: Final image — Python + Node.js runtime + firehorse
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl git ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Copy Node.js runtime and globally installed CLIs from stage 1
COPY --from=node-deps /usr/local/bin/node /usr/local/bin/node
COPY --from=node-deps /usr/local/bin/npm /usr/local/bin/npm
COPY --from=node-deps /usr/local/lib/node_modules /usr/local/lib/node_modules
# Re-create bin links (npm global install creates these in the node stage but they
# point to the node stage paths; re-linking ensures they work in the final image)
RUN npm rebuild -g 2>/dev/null; \\
    ln -sf /usr/local/lib/node_modules/@anthropic-ai/claude-code/cli.js /usr/local/bin/claude \\
    && chmod +x /usr/local/bin/claude \\
    && CODEX_BIN=$(find /usr/local/lib/node_modules/@openai/codex -name codex -type f | head -1) \\
    && ln -sf "$CODEX_BIN" /usr/local/bin/codex

# Install firehorse from local source
# TODO: Replace with `pip install "firehorse[react,resum]"` once published to PyPI
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[react,resum]"

# Entrypoint script: write codex/claude auth from env vars before running firehorse
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create non-root user (claude CLI refuses to run as root with bypassPermissions)
RUN useradd -m -s /bin/bash firehorse
USER firehorse
WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
"""

# TODO: Remove once firehorse is published to PyPI (Dockerfile will pip install from registry)
PYPROJECT_TOML = """\
[project]
name = "firehorse"
version = "0.1.0"
description = "Agent evaluation harness for OpenReward environments"

dependencies = [
    "openreward>=0.1.94",
    "mcp>=1.0.0",
]

[project.optional-dependencies]
react = [
    "anthropic>=0.40.0",
    "openai>=1.60.0",
    "google-genai>=1.0.0",
]
resum = [
    "anthropic>=0.40.0",
    "openai>=1.60.0",
    "google-genai>=1.0.0",
]

[project.scripts]
firehorse = "firehorse.cli:main"

[tool.setuptools]
packages = { find = { include = ["firehorse", "firehorse.*"] } }

[build-system]
requires = ["setuptools >= 77.0.3"]
build-backend = "setuptools.build_meta"
"""


ENTRYPOINT_SH = """\
#!/bin/sh
# Write CLI auth configs from env vars so codex/claude CLIs can authenticate

if [ -n "$OPENAI_API_KEY" ]; then
    mkdir -p ~/.codex
    printf '{"auth_mode":"apikey","OPENAI_API_KEY":"%s"}' "$OPENAI_API_KEY" > ~/.codex/auth.json
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    mkdir -p ~/.claude
    printf '{"apiKey":"%s"}' "$ANTHROPIC_API_KEY" > ~/.claude/auth.json
fi

exec firehorse "$@"
"""


@dataclass
class DockerConfig:
    enabled: bool = False
    image: str | None = None
    force_build: bool = False


def ensure_docker_available() -> None:
    """Check that Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError("Docker CLI not found. Is Docker installed?")
    if result.returncode != 0:
        raise RuntimeError("Docker daemon is not running. Start Docker and try again.")


def build_image(tag: str = DEFAULT_IMAGE, force: bool = False) -> None:
    """Build the firehorse Docker image.

    Writes the embedded Dockerfile, pyproject.toml, and the firehorse package
    source into a temp directory, then runs `docker build`. No repo root needed.
    """
    if not force:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"Docker image {tag} already exists (use --docker-build to rebuild)", file=sys.stderr)
            return

    # The firehorse package source — works whether installed from source or pip
    pkg_source = Path(__file__).parent

    print(f"Building Docker image {tag}...", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmpdir:
        build_dir = Path(tmpdir)

        # Write embedded files
        (build_dir / "Dockerfile").write_text(DOCKERFILE)
        (build_dir / "pyproject.toml").write_text(PYPROJECT_TOML)
        (build_dir / "entrypoint.sh").write_text(ENTRYPOINT_SH)

        # Copy the firehorse package source
        shutil.copytree(pkg_source, build_dir / "firehorse", ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.egg-info",
        ))

        subprocess.run(
            ["docker", "build", "-t", tag, str(build_dir)],
            check=True,
        )

    print(f"Docker image {tag} built successfully", file=sys.stderr)


def _build_cli_args(config: RunConfig) -> list[str]:
    """Reconstruct firehorse CLI arguments from a RunConfig.

    The output-dir is rewritten to /output (the container mount point).
    --no-docker is added so the container doesn't try to nest Docker.
    """
    args = [
        "--env", config.env,
        "--agent", config.agent,
        "--model", config.model,
        "--n-concurrent", str(config.n_concurrent),
        "--split", config.split,
        "--effort", config.effort,
        "--output-dir", "/output",
        "--no-docker",
    ]
    if config.max_tasks is not None:
        args.extend(["--max-tasks", str(config.max_tasks)])
    if config.skip_tasks:
        args.extend(["--skip-tasks", str(config.skip_tasks)])
    if config.run_name:
        args.extend(["--run-name", config.run_name])
    if config.max_turns is not None:
        args.extend(["--max-turns", str(config.max_turns)])
    if config.provider_url:
        args.extend(["--provider-url", config.provider_url])
    if config.disable_builtin_tools:
        args.extend(["--disable-builtin-tools", ",".join(config.disable_builtin_tools)])
    for k, v in config.secrets.items():
        args.extend(["--secret", f"{k}={v}"])
    if config.plan_mode:
        args.append("--plan-mode")
    if not config.logging:
        args.append("--no-logging")
    if not config.use_builtin_descriptions:
        args.append("--use-env-descriptions")
    if config.use_all_filesystem_tools:
        args.append("--use-all-filesystem-tools")
    return args


def run_in_container(config: RunConfig, docker_config: DockerConfig) -> int:
    """Run a firehorse evaluation inside a Docker container.

    Returns the container's exit code.
    """
    ensure_docker_available()

    image = docker_config.image or DEFAULT_IMAGE
    if not docker_config.image:
        build_image(image, force=docker_config.force_build)

    # Resolve output directory on host
    output_dir = config.output_dir or config.effective_run_name()
    host_output = os.path.abspath(output_dir)
    os.makedirs(host_output, exist_ok=True)

    # Build docker run command
    cmd = ["docker", "run", "--rm", "--init", "-t"]

    # Pass environment variables (never baked into image)
    passed_env = []
    for var in PASSTHROUGH_ENV_VARS:
        val = os.environ.get(var)
        if val:
            cmd.extend(["--env", f"{var}={val}"])
            passed_env.append(var)
    if passed_env:
        print(f"Passing env vars: {', '.join(passed_env)}", file=sys.stderr)

    # Tell firehorse inside the container what the host output path is (for display)
    cmd.extend(["--env", f"FIREHORSE_HOST_OUTPUT={host_output}"])

    # Also pass --secret values as env vars so subprocess agents (codex, claude-code)
    # can read them — they check os.environ, not the --secret args
    for key, val in config.secrets.items():
        env_key = key.upper()
        cmd.extend(["--env", f"{env_key}={val}"])

    # Volume mounts
    cmd.extend(["-v", f"{host_output}:/output"])

    # Image and firehorse arguments
    cmd.append(image)
    cmd.extend(_build_cli_args(config))

    print(f"Starting Docker container: {image}", file=sys.stderr)

    result = subprocess.run(cmd)
    return result.returncode
