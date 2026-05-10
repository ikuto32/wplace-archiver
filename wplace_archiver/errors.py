class WplaceArchiverError(Exception):
    """Base exception for wplace archiver."""


class DownloadError(WplaceArchiverError):
    pass


class AssetValidationError(WplaceArchiverError):
    pass


class DecompressError(WplaceArchiverError):
    pass


class TarScanError(WplaceArchiverError):
    pass


class PngDecodeError(WplaceArchiverError):
    pass


class PaletteError(WplaceArchiverError):
    pass


class IngestError(WplaceArchiverError):
    pass


class ApplyError(WplaceArchiverError):
    pass


class ExportError(WplaceArchiverError):
    pass


class StateConsistencyError(WplaceArchiverError):
    pass
