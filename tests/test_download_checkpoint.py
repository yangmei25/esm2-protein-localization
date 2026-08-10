from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.download_checkpoint import download, file_sha256


class DownloadCheckpointTests(unittest.TestCase):
    def test_existing_verified_file_does_not_download(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "checkpoint.pt"
            output.write_bytes(b"known checkpoint")
            checksum = file_sha256(output)
            download("https://invalid.example/not-used", output, checksum)
            self.assertEqual(output.read_bytes(), b"known checkpoint")

    def test_existing_invalid_file_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "checkpoint.pt"
            output.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "Existing checkpoint"):
                download("https://invalid.example/not-used", output, "0" * 64)


if __name__ == "__main__":
    unittest.main()
