from __future__ import annotations

from pathlib import Path
import os


def upload_derived_artifact(path: str | Path, remote_path: str) -> str | None:
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_REPO_ID")
    if not token or not repo_id:
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update derived research artifact {remote_path}",
    )
    return f"https://huggingface.co/datasets/{repo_id}/blob/main/{remote_path}"

