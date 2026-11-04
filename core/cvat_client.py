"""
CVAT Client Wrapper for querying projects/tasks and downloading COCO 1.0 datasets.
Supports authentication, task filtering by stage/state/status, unzipping,
and reorganizing images from /images/default to /images.
"""

import os
import re
import json
import shutil
import zipfile
import typing
from pathlib import Path
from cvat_sdk import Client
from cvat_sdk.core.progress import ProgressReporter


def sanitize_filename(name: str) -> str:
    """Sanitize directory/file names for Windows compatibility."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", str(name)).strip()
    return cleaned[:80] if len(cleaned) > 80 else (cleaned or "unnamed")


class CvatProgressReporter(ProgressReporter):
    """Bridges CVAT SDK progress notifications to GUI callbacks."""

    def __init__(self, callback: typing.Optional[typing.Callable[[int, int, str], None]] = None):
        super().__init__()
        self.callback = callback
        self.total = 100
        self.current = 0

    def start(self, total: int, *, desc: typing.Optional[str] = None):
        self.total = max(total, 1)
        self.current = 0
        if self.callback:
            self.callback(0, self.total, desc or "Starting export...")

    def advance(self, delta: int):
        self.current += delta
        if self.callback:
            pct = min(100, int((self.current / self.total) * 100)) if self.total > 0 else 0
            self.callback(self.current, self.total, f"Downloading: {pct}%")

    def report_status(self, progress: int):
        if self.callback:
            self.callback(progress, 100, f"CVAT Server preparing dataset: {progress}%")

    def finish(self):
        if self.callback:
            self.callback(self.total, self.total, "Download completed")


class CvatClientWrapper:
    """Encapsulates CVAT SDK operations with error handling and dataset cleanup."""

    def __init__(self):
        self.client: typing.Optional[Client] = None
        self.server_url: str = ""
        self.username: str = ""
        self.projects_cache: dict[int, str] = {}

    def is_logged_in(self) -> bool:
        return self.client is not None and self.client.has_credentials()

    def is_authenticated(self) -> bool:
        """Alias for is_logged_in."""
        return self.is_logged_in()

    def connect(self, url: str, username: str, password: str) -> dict:
        """Alias for login to support both connect and login naming conventions."""
        return self.login(url, username, password)

    def get_projects(self) -> list[dict]:
        """Alias for refresh_projects."""
        return self.refresh_projects()

    def login(self, url: str, username: str, password: str) -> dict:
        """
        Log in to CVAT server.
        Returns server information dictionary on success, raises exception on failure.
        """
        clean_url = url.strip().rstrip("/")
        # If user pasted /projects at end, trim it off
        if clean_url.endswith("/projects"):
            clean_url = clean_url[:-len("/projects")].rstrip("/")
        if clean_url.endswith("/tasks"):
            clean_url = clean_url[:-len("/tasks")].rstrip("/")

        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

        client = Client(clean_url, check_server_version=False)
        try:
            client.login((username, password))
            server_version = str(client.get_server_version())
            self.client = client
            self.server_url = clean_url
            self.username = username

            # Pre-cache projects
            self.refresh_projects()

            return {
                "status": "success",
                "server_url": clean_url,
                "server_version": server_version,
                "username": username,
            }
        except Exception as e:
            try:
                client.close()
            except Exception:
                pass
            raise RuntimeError(f"CVAT Login Failed: {str(e)}")

    def logout(self):
        if self.client:
            try:
                self.client.logout()
                self.client.close()
            except Exception:
                pass
        self.client = None
        self.projects_cache.clear()

    def refresh_projects(self) -> list[dict]:
        """Fetch all projects from CVAT."""
        if not self.client:
            raise RuntimeError("Not connected to CVAT server.")

        projects = self.client.projects.list()
        result = []
        self.projects_cache.clear()

        for p in projects:
            p_id = getattr(p, "id", None)
            p_name = getattr(p, "name", f"Project_{p_id}")
            self.projects_cache[p_id] = p_name
            result.append({
                "id": p_id,
                "name": p_name,
                "status": getattr(p, "status", ""),
                "tasks_count": getattr(p, "tasks_count", 0),
                "updated_date": str(getattr(p, "updated_date", "")),
            })
        return result

    def get_tasks(
        self,
        project_id: typing.Optional[int] = None,
        stage_filter: typing.Optional[str] = None,
        state_filter: typing.Optional[str] = None,
        status_filter: typing.Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch tasks from CVAT with optional filtering.
        Extracts stage and state information from task jobs.
        """
        if not self.client:
            raise RuntimeError("Not connected to CVAT server.")

        # Ensure project cache is populated
        if not self.projects_cache:
            try:
                self.refresh_projects()
            except Exception:
                pass

        tasks = self.client.tasks.list()
        tasks_data = []

        for t in tasks:
            t_id = getattr(t, "id", None)
            t_proj_id = getattr(t, "project_id", None)

            # Filter by project_id if specified
            if project_id is not None and t_proj_id != project_id:
                continue

            t_name = getattr(t, "name", f"Task_{t_id}")
            t_status = getattr(t, "status", "")
            t_size = getattr(t, "size", 0)
            t_updated = str(getattr(t, "updated_date", ""))

            # Extract job stages and states
            stages = set()
            states = set()
            try:
                jobs = t.get_jobs()
                for j in jobs:
                    j_stage = getattr(j, "stage", None)
                    j_state = getattr(j, "state", None)
                    if j_stage:
                        stages.add(str(j_stage).lower())
                    if j_state:
                        states.add(str(j_state).lower())
            except Exception:
                # Fallback if get_jobs is unavailable
                pass

            stage_str = ", ".join(sorted(stages)) if stages else "n/a"
            state_str = ", ".join(sorted(states)) if states else "n/a"

            # Apply stage filter
            if stage_filter and stage_filter.lower() != "all":
                if stage_filter.lower() not in stages:
                    continue

            # Apply state filter
            if state_filter and state_filter.lower() != "all":
                if state_filter.lower() not in states:
                    continue

            # Apply status filter
            if status_filter and status_filter.lower() != "all":
                if str(t_status).lower() != status_filter.lower():
                    continue

            proj_name = self.projects_cache.get(t_proj_id, f"Project #{t_proj_id}" if t_proj_id else "None")

            tasks_data.append({
                "id": t_id,
                "name": t_name,
                "project_id": t_proj_id,
                "project_name": proj_name,
                "status": t_status,
                "stage": stage_str,
                "state": state_str,
                "size": t_size,
                "updated_date": t_updated,
                "task_obj": t,
            })

        return tasks_data

    def download_and_extract_task(
        self,
        task_id: int,
        output_dir: Path,
        progress_callback: typing.Optional[typing.Callable[[int, int, str], None]] = None,
        task_name: typing.Optional[str] = None,
    ) -> Path:
        """
        Download COCO 1.0 dataset for a specific task, extract zip,
        move images out of 'default' subfolder to 'images', and delete 'default'.

        Returns the directory containing the task's cleaned dataset.
        """
        if not self.client:
            raise RuntimeError("Not connected to CVAT server.")

        task = self.client.tasks.retrieve(task_id)
        if not task_name:
            task_name = getattr(task, "name", f"task_{task_id}")

        safe_name = sanitize_filename(f"task_{task_id}_{task_name}")
        task_dir = output_dir / safe_name
        task_dir.mkdir(parents=True, exist_ok=True)

        zip_path = task_dir / f"dataset_coco_{task_id}.zip"

        if progress_callback:
            progress_callback(5, 100, f"Task {task_id}: Requesting COCO 1.0 export from CVAT...")

        reporter = CvatProgressReporter(progress_callback)

        # Download dataset in COCO 1.0 format with images
        task.export_dataset(
            format_name="COCO 1.0",
            filename=str(zip_path),
            include_images=True,
            pbar=reporter,
            status_check_period=2,
        )

        if not zip_path.exists():
            raise FileNotFoundError(f"Export zip was not created: {zip_path}")

        if progress_callback:
            progress_callback(85, 100, f"Task {task_id}: Extracting zip file...")

        # Unzip into task_dir
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(task_dir)

        # Remove the zip archive
        try:
            zip_path.unlink()
        except Exception:
            pass

        if progress_callback:
            progress_callback(92, 100, f"Task {task_id}: Reorganizing images (removing /images/default)...")

        # Reorganize images: move /images/default/* into /images/ and remove /images/default
        self._reorganize_images_folder(task_dir)

        if progress_callback:
            progress_callback(100, 100, f"Task {task_id}: Completed successfully!")

        return task_dir

    @staticmethod
    def _reorganize_images_folder(task_dir: Path):
        """
        Moves all images from task_dir/images/default/* into task_dir/images/
        and removes the default folder.
        Also cleans instances_default.json so file_name references the image directly.
        """
        images_dir = task_dir / "images"
        default_dir = images_dir / "default"

        if default_dir.exists() and default_dir.is_dir():
            # Move all files out of default/ into images/
            for item in list(default_dir.iterdir()):
                target_dest = images_dir / item.name
                if target_dest.exists():
                    target_dest.unlink()
                shutil.move(str(item), str(target_dest))

            # Remove empty default folder
            try:
                shutil.rmtree(str(default_dir))
            except Exception:
                pass

        # Also inspect and normalize annotations/instances_default.json
        ann_file = task_dir / "annotations" / "instances_default.json"
        if ann_file.exists():
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    coco_data = json.load(f)

                modified = False
                for img in coco_data.get("images", []):
                    fname = img.get("file_name", "")
                    # Strip leading default/ or default\ if present
                    if fname.startswith("default/") or fname.startswith("default\\"):
                        img["file_name"] = Path(fname).name
                        modified = True

                if modified:
                    with open(ann_file, "w", encoding="utf-8") as f:
                        json.dump(coco_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning cleaning instances_default.json in {task_dir}: {e}")
