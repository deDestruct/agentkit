from __future__ import annotations

import base64
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from agentkit.docx._backends._base import (
    BackendError,
    BackendTimeoutError,
    BackendUnavailableError,
)

WORD_PROG_ID = "Word.Application"
MAX_FILE_SIZE = 100 * 1024 * 1024

CONVERT_SCRIPT = r"""
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$source = $env:AGENTKIT_WORD_SOURCE
$output = $env:AGENTKIT_WORD_OUTPUT

$word = $null
$documents = $null
$document = $null

$errors = @()

try {
    if ([System.IO.File]::Exists($output)) {
        throw "Output already exists: $output"
    }

    $word = New-Object -ComObject Word.Application

    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3

    $documents = $word.Documents

    $document = $documents.Open(
        [ref]$source,
        [ref]$false,
        [ref]$true,
        [ref]$false
    )

    $document.ExportAsFixedFormat(
        $output,
        17,
        $false
    )
}
catch {
    $errors += "Conversion: $($_.Exception.Message)"
}
finally {
    if ($null -ne $document) {
        try {
            $document.Close([ref]0)
        }
        catch {
            $errors += "Document.Close: $($_.Exception.Message)"
        }

        try {
            [void][System.Runtime.InteropServices.Marshal]::
                FinalReleaseComObject($document)
        }
        catch {
            $errors += "Release document: $($_.Exception.Message)"
        }

        $document = $null
    }

    if ($null -ne $documents) {
        try {
            [void][System.Runtime.InteropServices.Marshal]::
                FinalReleaseComObject($documents)
        }
        catch {
            $errors += "Release documents: $($_.Exception.Message)"
        }

        $documents = $null
    }

    if ($null -ne $word) {
        try {
            $word.Quit([ref]0)
        }
        catch {
            $errors += "Word.Quit: $($_.Exception.Message)"
        }

        try {
            [void][System.Runtime.InteropServices.Marshal]::
                FinalReleaseComObject($word)
        }
        catch {
            $errors += "Release Word: $($_.Exception.Message)"
        }

        $word = $null
    }

    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

if ($errors.Count -gt 0) {
    [Console]::Error.WriteLine(
        ($errors -join [Environment]::NewLine)
    )

    exit 1
}

exit 0
"""


def _find_powershell() -> str | None:
    return shutil.which("powershell.exe")


def _encode_script(script: str) -> str:
    return base64.b64encode(
        script.encode("utf-16-le")
    ).decode("ascii")


class WordBackend:
    @property
    def name(self) -> str:
        return "word"

    def is_available(self) -> bool:
        if sys.platform != "win32":
            return False

        if _find_powershell() is None:
            return False

        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CLASSES_ROOT,
                f"{WORD_PROG_ID}\\CLSID",
            ):
                return True

        except OSError:
            return False

    def convert_to_pdf(
        self,
        source: Path,
        output: Path,
        *,
        timeout: float = 120.0,
    ) -> None:
        if not self.is_available():
            raise BackendUnavailableError(
                "Word backend requires Windows, "
                "PowerShell and Microsoft Word"
            )

        powershell = _find_powershell()

        if powershell is None:
            raise BackendUnavailableError(
                "PowerShell executable was not found"
            )

        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                "Timeout must be finite and positive"
            )

        source = source.resolve(strict=True)
        output = output.resolve()

        if not source.is_file():
            raise ValueError(
                f"Not a regular file: {source}"
            )

        if source.suffix.lower() != ".docx":
            raise ValueError(
                "Source must be a DOCX file"
            )

        if output.suffix.lower() != ".pdf":
            raise ValueError(
                "Output must be a PDF file"
            )

        if source.stat().st_size > MAX_FILE_SIZE:
            raise ValueError(
                "Source exceeds 100 MiB limit"
            )

        if output.exists():
            raise FileExistsError(
                f"Output already exists: {output}"
            )

        if not output.parent.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: "
                f"{output.parent}"
            )

        environment = os.environ.copy()

        environment["AGENTKIT_WORD_SOURCE"] = str(source)
        environment["AGENTKIT_WORD_OUTPUT"] = str(output)

        command = [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-EncodedCommand",
            _encode_script(CONVERT_SCRIPT),
        ]

        try:
            result = subprocess.run(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )

        except subprocess.TimeoutExpired as exc:
            raise BackendTimeoutError(
                f"Word conversion exceeded "
                f"{timeout:g} seconds"
            ) from exc

        except OSError as exc:
            raise BackendError(
                f"Unable to start PowerShell: {exc}"
            ) from exc

        if result.returncode != 0:
            message = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown PowerShell error"
            )

            raise BackendError(
                f"Word conversion failed: {message}"
            )

        if not output.is_file():
            raise BackendError(
                "Word did not produce a PDF file"
            )

        size = output.stat().st_size

        if size == 0:
            raise BackendError(
                "Word produced an empty PDF file"
            )

        if size > MAX_FILE_SIZE:
            raise BackendError(
                "Generated PDF exceeds 100 MiB limit"
            )

        with output.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise BackendError(
                    "Word output has an invalid PDF header"
                )
