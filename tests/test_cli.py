from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agentkit import cli
from agentkit.docx import cli as docx_cli
from agentkit.docx._backends._base import (
    BackendError,
    BackendUnavailableError,
)

COMMANDS = ("info", "crop", "resize", "rotate", "sheet", "annotate")


def _invoke(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["agentkit", *arguments])
    try:
        cli.main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def _payload(stream: str) -> dict[str, Any]:
    parsed = json.loads(stream)
    assert isinstance(parsed, dict)
    return parsed


def test_info_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rgb_png: Path,
) -> None:
    code = _invoke(monkeypatch, "image", "info", str(rgb_png))

    captured = capsys.readouterr()
    assert code == 0
    payload = _payload(captured.out)
    assert payload["width"] == 4
    assert payload["height"] == 2
    assert captured.err == ""


def test_crop_command_publishes_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "cropped.png"

    code = _invoke(
        monkeypatch,
        "image", "crop", str(rgb_png), "0", "0", "2", "2", "--out", str(output),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert output.is_file()
    assert _payload(captured.out)["status"] == "ok"


def test_resize_command_publishes_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "scaled.png"

    code = _invoke(
        monkeypatch, "image", "resize", str(rgb_png), "--scale", "2", "--out", str(output),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert output.is_file()
    assert _payload(captured.out)["status"] == "ok"


def test_rotate_command_publishes_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "rotated.png"

    code = _invoke(
        monkeypatch, "image", "rotate", str(rgb_png), "--angle", "90", "--out", str(output),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert output.is_file()
    assert _payload(captured.out)["status"] == "ok"


def test_annotate_command_publishes_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "annotated.png"

    code = _invoke(
        monkeypatch,
        "image", "annotate", str(rgb_png), "line", "0", "0", "2", "1",
        "--color", "red", "--out", str(output),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert output.is_file()
    assert _payload(captured.out)["status"] == "ok"


def test_sheet_command_publishes_sheets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    image_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    sources = [image_factory(f"{index}.png") for index in range(2)]
    output = tmp_path / "sheets"

    code = _invoke(
        monkeypatch, "image", "sheet", *(str(source) for source in sources), "--out", str(output),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert (output / "sheet_0001.png").is_file()
    assert _payload(captured.out)["count"] == 2


def test_missing_required_argument_uses_exit_code_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rgb_png: Path,
) -> None:
    code = _invoke(monkeypatch, "image", "crop", str(rgb_png), "0", "0", "2", "2")

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    payload = _payload(captured.err)
    assert payload["status"] == "error"
    assert payload["type"] == "CLIArgumentError"


def test_missing_subcommand_uses_exit_code_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _invoke(monkeypatch, "image")

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert _payload(captured.err)["status"] == "error"


def test_unknown_command_uses_exit_code_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _invoke(monkeypatch, "image", "unknown")

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    payload = _payload(captured.err)
    assert payload["status"] == "error"
    assert payload["type"] == "CLIArgumentError"


def test_invalid_angle_uses_exit_code_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    code = _invoke(
        monkeypatch,
        "image", "rotate", str(rgb_png), "--angle", "45", "--out", str(tmp_path / "out.png"),
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    payload = _payload(captured.err)
    assert payload["status"] == "error"
    assert "invalid choice" in payload["error"]


def test_missing_source_uses_exit_code_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    code = _invoke(monkeypatch, "image", "info", str(tmp_path / "absent.png"))

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    payload = _payload(captured.err)
    assert payload["status"] == "error"
    assert payload["type"] == "FileNotFoundError"


def test_existing_output_uses_exit_code_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "exists.png"
    output.write_bytes(b"sentinel")

    code = _invoke(
        monkeypatch, "image", "rotate", str(rgb_png), "--angle", "90", "--out", str(output),
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    payload = _payload(captured.err)
    assert payload["status"] == "error"
    assert payload["type"] == "FileExistsError"
    assert output.read_bytes() == b"sentinel"


@pytest.mark.parametrize("command", COMMANDS)
def test_process_help_exits_zero(run_cli: Callable, command: str) -> None:
    result = run_cli("image", command, "--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert result.stderr == ""


def test_process_image_help_lists_every_command(run_cli: Callable) -> None:
    result = run_cli("image", "--help")

    assert result.returncode == 0
    for command in COMMANDS:
        assert command in result.stdout


def test_process_info_success(run_cli: Callable, rgb_png: Path) -> None:
    result = run_cli("image", "info", str(rgb_png))

    assert result.returncode == 0
    assert result.stderr == ""
    payload = _payload(result.stdout)
    assert payload["format"] == "PNG"


def test_process_rotate_success(run_cli: Callable, rgb_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "rotated.png"

    result = run_cli("image", "rotate", str(rgb_png), "--angle", "270", "--out", str(output))

    assert result.returncode == 0
    assert result.stderr == ""
    assert output.is_file()
    payload = _payload(result.stdout)
    assert payload["status"] == "ok"
    assert Path(payload["output"]).samefile(output)


def test_process_reports_argument_errors_as_json(run_cli: Callable, rgb_png: Path, tmp_path: Path) -> None:
    result = run_cli(
        "image", "rotate", str(rgb_png), "--angle", "45", "--out", str(tmp_path / "out.png"),
    )

    assert result.returncode == 2
    payload = _payload(result.stderr)
    assert payload["status"] == "error"
    assert payload["type"] == "CLIArgumentError"
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


def test_process_reports_missing_file_as_json(run_cli: Callable, tmp_path: Path) -> None:
    result = run_cli("image", "info", str(tmp_path / "absent.png"))

    assert result.returncode == 1
    assert result.stdout == ""
    payload = _payload(result.stderr)
    assert payload["status"] == "error"
    assert payload["type"] == "FileNotFoundError"
    assert "Traceback" not in result.stderr


def test_process_reports_unsupported_format_as_json(run_cli: Callable, rgb_png: Path, tmp_path: Path) -> None:
    result = run_cli(
        "image", "rotate", str(rgb_png), "--angle", "90", "--out", str(tmp_path / "out.tga"),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    payload = _payload(result.stderr)
    assert payload["status"] == "error"
    assert payload["type"] == "ValueError"
    assert "Unsupported output format" in payload["error"]
    assert "Traceback" not in result.stderr


def test_process_handles_non_ascii_paths(run_cli: Callable, image_factory: Callable[..., Path], tmp_path: Path) -> None:
    source = image_factory("тест-источник.png")
    output = tmp_path / "тест-поворот.png"

    info = run_cli("image", "info", str(source))
    rotated = run_cli("image", "rotate", str(source), "--angle", "90", "--out", str(output))

    assert info.returncode == 0
    assert "тест-источник.png" in info.stdout
    assert rotated.returncode == 0
    assert output.is_file()


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (BackendError, "DOCX conversion failed"),
        (BackendUnavailableError, "no DOCX rendering backend"),
    ],
    ids=["backend-error", "backend-unavailable"],
)
def test_docx_backend_errors_use_json_error_contract(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error_type: type[Exception],
    message: str,
) -> None:
    def fail_render(*args: Any, **kwargs: Any) -> None:
        raise error_type(message)

    monkeypatch.setattr(docx_cli, "render_docx", fail_render)

    code = _invoke(
        monkeypatch,
        "docx",
        "render",
        str(tmp_path / "source.docx"),
        "--out",
        str(tmp_path / "rendered"),
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert _payload(captured.err) == {
        "status": "error",
        "error": message,
        "type": error_type.__name__,
    }


@pytest.mark.parametrize("command", ["crop", "resize", "rotate", "annotate"])
def test_quality_help_matches_implemented_default(run_cli: Callable, command: str) -> None:
    result = run_cli("image", command, "--help")

    assert result.returncode == 0
    assert "default: 90" in result.stdout
