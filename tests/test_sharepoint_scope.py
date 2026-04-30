from __future__ import annotations

import pytest

from services.sharepoint_sync.scope import is_in_scope, to_relative_path


@pytest.mark.parametrize(
    ("parent_path", "name", "root", "expected"),
    [
        ("/drive/root:/Engineering/Specs", "draft.pdf", "/Engineering/Specs", True),
        ("/drive/root:/Engineering/Specs/sub", "x.pdf", "/Engineering/Specs", True),
        ("/drive/root:/Engineering", "x.pdf", "/Engineering/Specs", False),
        ("/drive/root:/Engineering/Specs2", "x.pdf", "/Engineering/Specs", False),
        ("/drive/root:/Marketing", "x.pdf", "/Engineering/Specs", False),
        # Trailing-slash tolerance on the root config
        ("/drive/root:/Engineering/Specs", "x.pdf", "/Engineering/Specs/", True),
        # Substring trap: SpecsX must NOT match Specs
        ("/drive/root:/Engineering/SpecsX", "x.pdf", "/Engineering/Specs", False),
    ],
)
def test_is_in_scope(parent_path: str, name: str, root: str, expected: bool) -> None:
    assert is_in_scope(parent_path, name, root) is expected


def test_to_relative_path_strips_drive_prefix_and_root() -> None:
    rel = to_relative_path(
        parent_path="/drive/root:/Engineering/Specs/sub",
        name="draft.pdf",
        root_folder_path="/Engineering/Specs",
    )
    assert rel == "sub/draft.pdf"


def test_to_relative_path_at_root() -> None:
    rel = to_relative_path(
        parent_path="/drive/root:/Engineering/Specs",
        name="top.pdf",
        root_folder_path="/Engineering/Specs",
    )
    assert rel == "top.pdf"


def test_to_relative_path_raises_when_out_of_scope() -> None:
    with pytest.raises(ValueError):
        to_relative_path(
            parent_path="/drive/root:/Marketing",
            name="x.pdf",
            root_folder_path="/Engineering/Specs",
        )
