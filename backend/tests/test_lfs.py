"""`is_lfs_pointer` tells Git LFS pointer files from real files."""
from tests.lfs import is_lfs_pointer

POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:2358ae5883aa\n"
    "size 91000000\n"
)


def test_pointer_file_is_detected(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_text(POINTER)
    assert is_lfs_pointer(path)


def test_real_and_missing_files_are_not_pointers(tmp_path):
    binary = tmp_path / "weights.bin"
    binary.write_bytes(b"\x00\x01" * 1000)
    text = tmp_path / "notes.txt"
    text.write_text("plain text")
    assert not is_lfs_pointer(binary)
    assert not is_lfs_pointer(text)
    assert not is_lfs_pointer(tmp_path / "missing.safetensors")
