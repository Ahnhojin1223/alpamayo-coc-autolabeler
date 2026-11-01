# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VLM wrappers that run through subscription-authenticated Claude Code and Codex CLIs."""

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from coc_labeling.model_clients.timeout import TIMEOUT_MAX
from coc_labeling.model_clients.vlm_wrappers.common import BaseWrapper, encode_image

logger = logging.getLogger(__name__)

MAX_CLI_ATTEMPTS = 3
CLI_RETRY_DELAY_SEC = 60
AUTH_CHECK_TIMEOUT_SEC = 60
# Retrying cannot help once the subscription window is exhausted.
USAGE_LIMIT_MARKERS = (
    "usage limit",
    "rate limit",
    "limit reached",
    "hit your limit",
    "quota",
)
# Removing API credentials keeps these wrappers from silently billing an API account.
API_BILLING_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CODEX_API_KEY",
)
# Subscription token variable that must survive the CLAUDE_CODE_* cleanup below.
CLAUDE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
CLAUDE_SUBSCRIPTION_AUTH_METHODS = {"claude.ai", "oauth_token"}
CODEX_PLATFORM_DIR_PREFIX = {"win32": "windows", "linux": "linux", "darwin": "macos"}


class CliUsageLimitError(RuntimeError):
    """Raised when a CLI reports that the subscription usage limit was reached."""


def _child_env() -> Dict[str, str]:
    """Return the CLI environment without API credentials or parent-session state."""
    env = {}
    for key, value in os.environ.items():
        if key in API_BILLING_ENV_VARS or key == "CLAUDECODE":
            continue
        if key.startswith("CLAUDE_CODE_") and key != CLAUDE_OAUTH_TOKEN_ENV:
            continue
        env[key] = value
    return env


def _find_cli(path_env: str, command: str, extension_globs: Tuple[str, ...]) -> str:
    """Locate a CLI from its env override, PATH, or the newest VS Code extension bundle."""
    configured_path = os.environ.get(path_env)
    if configured_path:
        return configured_path
    path_match = shutil.which(command)
    if path_match:
        return path_match

    suffix = ".exe" if sys.platform == "win32" else ""
    candidates: List[Path] = []
    for extensions_dir in (".vscode/extensions", ".vscode-server/extensions"):
        for pattern in extension_globs:
            candidates.extend((Path.home() / extensions_dir).glob(f"{pattern}{suffix}"))
    if candidates:
        return str(max(candidates, key=lambda path: path.stat().st_mtime))
    raise FileNotFoundError(
        f"Could not find the '{command}' CLI. Install it, add it to PATH, or set {path_env}."
    )


def _resolve_refs(schema: Any, definitions: Dict[str, Any]) -> Any:
    """Inline local $ref nodes, which may not carry sibling keywords in strict schemas."""
    if isinstance(schema, list):
        return [_resolve_refs(item, definitions) for item in schema]
    if not isinstance(schema, dict):
        return schema
    resolved = {
        key: _resolve_refs(value, definitions)
        for key, value in schema.items()
        if key not in ("$defs", "definitions")
    }
    ref = resolved.pop("$ref", None)
    if ref is not None:
        target = _resolve_refs(definitions[ref.rsplit("/", 1)[-1]], definitions)
        return {**target, **resolved}
    return resolved


def _close_objects(schema: Any) -> None:
    """Mark every object strict: no additional properties and all properties required."""
    if isinstance(schema, list):
        for item in schema:
            _close_objects(item)
    elif isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            schema["required"] = list(schema.get("properties", {}))
        for value in schema.values():
            _close_objects(value)


def _strict_schema(json_schema: Any) -> Dict[str, Any]:
    """Convert a Pydantic v1/v2 model into a self-contained strict JSON schema."""
    if hasattr(json_schema, "model_json_schema"):
        raw_schema = json_schema.model_json_schema()
    else:
        raw_schema = json_schema.schema()
    definitions = {**raw_schema.get("definitions", {}), **raw_schema.get("$defs", {})}
    schema = _resolve_refs(raw_schema, definitions)
    _close_objects(schema)
    return schema


class SubscriptionCliWrapper(BaseWrapper):
    """Shared single-shot CLI flow; each request runs in a clean temporary directory."""

    model_env = ""
    effort_env = ""

    def __init__(
        self,
        model_name: str,
        init_model: bool = True,
        ip_addr: Optional[str] = None,
        timeout_sec: int = TIMEOUT_MAX,
    ) -> None:
        del init_model, ip_addr
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.cli_model = os.environ.get(self.model_env)
        self.cli_effort = os.environ.get(self.effort_env)
        self.env = _child_env()
        self.cli_path = self._locate_cli()
        self._check_subscription_login()
        logger.info(
            "Using subscription CLI '%s' with model '%s' and thinking '%s'.",
            self.cli_path,
            self.cli_model or "CLI default",
            self.cli_effort or "CLI default",
        )

    def _locate_cli(self) -> str:
        raise NotImplementedError

    def _check_subscription_login(self) -> None:
        raise NotImplementedError

    def _run_once(
        self,
        system_text: str,
        user_content: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]],
        work_dir: Path,
    ) -> Dict[str, Any]:
        """Return `text` or `parsed` output plus token usage for a single CLI request."""
        raise NotImplementedError

    def _run_cli(
        self, args: List[str], stdin_text: Optional[str], cwd: Optional[Path], timeout_sec: int
    ) -> subprocess.CompletedProcess:
        """Run the CLI with UTF-8 pipes and the subscription-only environment."""
        return subprocess.run(
            [self.cli_path, *args],
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            env=self.env,
            timeout=timeout_sec,
            check=False,
        )

    @staticmethod
    def _split_messages(
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Separate system text from the ordered user content blocks."""
        system_parts: List[str] = []
        user_content: List[Dict[str, Any]] = []
        for message in messages:
            if message["role"] == "system":
                system_parts.extend(block["text"] for block in message["content"])
            elif message["role"] == "user":
                user_content.extend(message["content"])
            else:
                raise ValueError(f"Unsupported message role for CLI inference: {message['role']}")
        return "\n\n".join(system_parts), user_content

    @staticmethod
    def _raise_cli_failure(message: str) -> NoReturn:
        """Raise the error type that matches a failed CLI request."""
        if any(marker in message.lower() for marker in USAGE_LIMIT_MARKERS):
            raise CliUsageLimitError(f"Subscription usage limit reached: {message}")
        raise RuntimeError(message)

    def infer(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        seed: int = 42,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        json_schema: Optional[Any] = None,
    ) -> Dict[str, Any]:
        # The CLIs do not expose sampling or output-length controls.
        del max_tokens, seed, temperature, top_p, repetition_penalty
        system_text, user_content = self._split_messages(messages)
        schema = _strict_schema(json_schema) if json_schema is not None else None

        output = None
        for attempt in range(1, MAX_CLI_ATTEMPTS + 1):
            try:
                with tempfile.TemporaryDirectory(prefix="coc_cli_") as work_dir:
                    output = self._run_once(system_text, user_content, schema, Path(work_dir))
                content = output["text"]
                if json_schema is not None:
                    parsed = output.get("parsed")
                    if parsed is None:
                        parsed = json.loads(content)
                    content = json_schema(**parsed).dict()
                break
            except CliUsageLimitError:
                raise
            # Pydantic validation errors derive from ValueError.
            except (RuntimeError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
                if attempt == MAX_CLI_ATTEMPTS:
                    raise RuntimeError(
                        f"CLI request failed after {MAX_CLI_ATTEMPTS} attempts: {exc}"
                    ) from exc
                logger.warning(
                    "CLI request failed on attempt %d/%d: %s. Retrying after %d seconds.",
                    attempt,
                    MAX_CLI_ATTEMPTS,
                    exc,
                    CLI_RETRY_DELAY_SEC,
                )
                time.sleep(CLI_RETRY_DELAY_SEC)

        return {
            "finish_reason": "completed",
            "content": content,
            "prompt_tokens": output["prompt_tokens"],
            "response_tokens": output["response_tokens"],
            "reasoning_tokens": output["reasoning_tokens"],
            "system_fingerprint": None,
        }

    def add_message(self, role: str, m_type: str, content: Any) -> Dict[str, Any]:
        if role not in ["user", "system", "assistant"]:
            raise ValueError(f"Invalid message role: {role}")
        if m_type not in ["text", "image", "video"]:
            raise ValueError(f"Invalid message type: {m_type}")

        if m_type == "text":
            return {"role": role, "content": [{"type": "text", "text": content}]}
        if m_type == "image":
            return {"role": role, "content": [self._image_block(content)]}
        if m_type == "video":
            if not isinstance(content, list):
                raise TypeError("Video content must be a list of frames.")
            return {"role": role, "content": [self._image_block(frame) for frame in content]}
        raise ValueError(f"Invalid message type: {m_type}")

    @staticmethod
    def _image_block(image: Any) -> Dict[str, str]:
        encoded = encode_image(image)
        if encoded is None:
            raise FileNotFoundError(f"Image file not found: {image}")
        return {"type": "image", "data": encoded}


class ClaudeCodeWrapper(SubscriptionCliWrapper):
    """Wrapper using `claude -p` signed in with a Claude subscription."""

    model_env = "CLAUDE_CLI_MODEL"
    effort_env = "CLAUDE_CLI_THINKING"

    def _locate_cli(self) -> str:
        return _find_cli(
            "CLAUDE_CLI_PATH",
            "claude",
            ("anthropic.claude-code-*/resources/native-binary/claude",),
        )

    def _check_subscription_login(self) -> None:
        result = self._run_cli(["auth", "status"], None, None, AUTH_CHECK_TIMEOUT_SEC)
        try:
            status = json.loads(result.stdout)
        except json.JSONDecodeError:
            status = {}
        if status.get("authMethod") not in CLAUDE_SUBSCRIPTION_AUTH_METHODS:
            raise RuntimeError(
                "Claude Code is not signed in with a subscription. Run `claude auth login`."
            )

    def _run_once(
        self,
        system_text: str,
        user_content: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]],
        work_dir: Path,
    ) -> Dict[str, Any]:
        # --safe-mode keeps personal CLAUDE.md, skills, hooks, and MCP servers out of the labels.
        args = [
            "-p",
            "--safe-mode",
            "--tools",
            "",
            "--no-session-persistence",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--system-prompt",
            system_text,
        ]
        if self.cli_model:
            args += ["--model", self.cli_model]
        if self.cli_effort:
            args += ["--effort", self.cli_effort]
        if schema is not None:
            args += ["--json-schema", json.dumps(schema)]

        content = [
            {"type": "text", "text": block["text"]}
            if block["type"] == "text"
            else {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": block["data"]},
            }
            for block in user_content
        ]
        request = {"type": "user", "message": {"role": "user", "content": content}}
        result = self._run_cli(args, json.dumps(request), work_dir, self.timeout_sec)

        final_event = None
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                final_event = event

        if final_event is None or final_event.get("is_error") or result.returncode != 0:
            details = (final_event or {}).get("result") or result.stderr or result.stdout
            self._raise_cli_failure(
                f"claude exited with code {result.returncode}: {str(details).strip()[:1000]}"
            )

        usage = final_event.get("usage") or {}
        return {
            "text": final_event.get("result") or "",
            "parsed": final_event.get("structured_output"),
            "prompt_tokens": sum(
                usage.get(key) or 0
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            ),
            "response_tokens": usage.get("output_tokens"),
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
        }


class CodexCliWrapper(SubscriptionCliWrapper):
    """Wrapper using `codex exec` signed in with a ChatGPT subscription."""

    model_env = "CODEX_CLI_MODEL"
    effort_env = "CODEX_CLI_THINKING"

    def _locate_cli(self) -> str:
        platform_prefix = CODEX_PLATFORM_DIR_PREFIX.get(sys.platform, sys.platform)
        return _find_cli(
            "CODEX_CLI_PATH",
            "codex",
            (f"openai.chatgpt-*/bin/{platform_prefix}-*/codex",),
        )

    def _check_subscription_login(self) -> None:
        result = self._run_cli(["login", "status"], None, None, AUTH_CHECK_TIMEOUT_SEC)
        if result.returncode != 0 or "ChatGPT" not in f"{result.stdout}{result.stderr}":
            raise RuntimeError("Codex is not signed in with ChatGPT. Run `codex login`.")

    def _run_once(
        self,
        system_text: str,
        user_content: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]],
        work_dir: Path,
    ) -> Dict[str, Any]:
        output_path = work_dir / "output.txt"
        # Disabling tools and project docs limits the run to the supplied prompt and images.
        args = [
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--cd",
            str(work_dir),
            "--output-last-message",
            str(output_path),
            "--config",
            'web_search="disabled"',
            "--config",
            "project_doc_max_bytes=0",
        ]
        for feature in ("shell_tool", "unified_exec", "apps", "plugins", "multi_agent"):
            args += ["--disable", feature]
        if self.cli_model:
            args += ["--model", self.cli_model]
        if self.cli_effort:
            args += ["--config", f'model_reasoning_effort="{self.cli_effort}"']
        if schema is not None:
            schema_path = work_dir / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            args += ["--output-schema", str(schema_path)]

        # `codex exec` attaches images separately, so markers keep their place in the prompt.
        prompt_parts = [system_text] if system_text else []
        image_count = 0
        for block in user_content:
            if block["type"] == "text":
                prompt_parts.append(block["text"])
                continue
            image_count += 1
            image_path = work_dir / f"frame_{image_count:04d}.jpg"
            image_path.write_bytes(base64.b64decode(block["data"]))
            args += ["--image", str(image_path)]
            prompt_parts.append(f"[Attached image {image_count}]")

        result = self._run_cli([*args, "-"], "\n\n".join(prompt_parts), work_dir, self.timeout_sec)

        usage: Dict[str, Any] = {}
        errors: List[str] = []
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "turn.completed":
                usage = event.get("usage") or {}
            elif event.get("type") == "error":
                errors.append(str(event.get("message")))
            elif event.get("type") == "turn.failed":
                errors.append(str((event.get("error") or {}).get("message")))

        if result.returncode != 0 or not output_path.exists():
            details = " ".join(errors) or result.stderr or result.stdout
            self._raise_cli_failure(
                f"codex exited with code {result.returncode}: {details.strip()[:1000]}"
            )

        return {
            "text": output_path.read_text(encoding="utf-8"),
            "prompt_tokens": usage.get("input_tokens"),
            "response_tokens": usage.get("output_tokens"),
            "reasoning_tokens": usage.get("reasoning_output_tokens"),
        }
