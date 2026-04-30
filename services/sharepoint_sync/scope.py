from __future__ import annotations

# Microsoft Graph returns parentReference.path as e.g. "/drive/root:/Engineering/Specs"
_DRIVE_PREFIX = "/drive/root:"


def _normalize_root(root_folder_path: str) -> str:
    """Strip trailing slashes and ensure a leading slash."""
    if not root_folder_path.startswith("/"):
        root_folder_path = "/" + root_folder_path
    return root_folder_path.rstrip("/") or "/"


def _strip_drive_prefix(parent_path: str) -> str:
    if parent_path.startswith(_DRIVE_PREFIX):
        return parent_path[len(_DRIVE_PREFIX) :] or "/"
    return parent_path


def is_in_scope(parent_path: str, name: str, root_folder_path: str) -> bool:
    """Return True iff (parent_path/name) is inside root_folder_path on segment boundaries."""
    root = _normalize_root(root_folder_path)
    parent = _strip_drive_prefix(parent_path).rstrip("/") or "/"
    if parent == root:
        return True
    return parent.startswith(root + "/")


def to_relative_path(parent_path: str, name: str, root_folder_path: str) -> str:
    """Convert a Graph item to its path relative to the in-scope root.

    Raises ValueError if the item is out of scope.
    """
    if not is_in_scope(parent_path, name, root_folder_path):
        msg = f"out of scope: {parent_path}/{name}"
        raise ValueError(msg)
    root = _normalize_root(root_folder_path)
    parent = _strip_drive_prefix(parent_path).rstrip("/") or "/"
    if parent == root:
        return name
    rel_parent = parent[len(root) + 1 :]
    return f"{rel_parent}/{name}"
