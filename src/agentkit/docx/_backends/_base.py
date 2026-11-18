from pathlib import Path
from typing import Protocol


class BackendError(RuntimeError):
    """Document conversion backend error."""


class BackendUnavailableError(BackendError):
    """Conversion backend is unavailable."""


class BackendTimeoutError(BackendError):
    """Conversion backend exceeded its timeout."""


class ConversionBackend(Protocol):
    @property
    def name(self) -> str:
        ...

    def is_available(self) -> bool:
        ...

    def convert_to_pdf(
        self,
        source: Path,
        output: Path,
        *,
        timeout: float = 120.0,
    ) -> None:
        ...
