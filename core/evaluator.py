"""
Model evaluation & training-report helpers.

- dataset_stats(data_yaml)            -> images / labelled images / instances per split and per class
- read_training_results(run_dir)      -> epochs, best epoch, metric history (results.csv) and plot files
- validate_on_split(model, data_yaml) -> accuracy metrics on the split you made in Step 4 (test, or val)
- evaluate_external(model, source)    -> accuracy metrics on any image / folder that HAS labels
                                         (YOLO .txt next to images, in a sibling labels/ folder,
                                          or a COCO json such as a CVAT export). Returns None when
                                          no labels are found, so unlabeled data is prediction-only.

Metric definitions (object detection / segmentation, same as Ultralytics):
- Precision, Recall, mAP50, mAP50-95 come straight from model.val()
- F1        = 2PR / (P + R)
- Accuracy  = TP / (TP + FP + FN), counted from the confusion matrix (conf 0.25 by default, IoU 0.45),
              i.e. the share of all objects/predictions that were matched to the right class.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
import typing
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# plot files Ultralytics writes into a run / val folder (new names first, old names as fallback)
PLOT_FILES = {
    "results": ["results.png"],
    "confusion_matrix": ["confusion_matrix.png"],
    "confusion_matrix_normalized": ["confusion_matrix_normalized.png"],
    "pr_curve": ["BoxPR_curve.png", "PR_curve.png"],
    "f1_curve": ["BoxF1_curve.png", "F1_curve.png"],
    "p_curve": ["BoxP_curve.png", "P_curve.png"],
    "r_curve": ["BoxR_curve.png", "R_curve.png"],
    "mask_pr_curve": ["MaskPR_curve.png"],
    "labels": ["labels.jpg"],
    "val_pred": ["val_batch0_pred.jpg"],
    "val_labels": ["val_batch0_labels.jpg"],
}


# =====================================================================
# Small helpers
# =====================================================================
def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def f1_score(p, r):
    p, r = _f(p), _f(r)
    if p is None or r is None or (p + r) <= 0:
        return None
    return 2 * p * r / (p + r)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def ultralytics_label_path(img: Path) -> Path:
    """Where Ultralytics looks for an image's label (…/images/… -> …/labels/…, else same folder)."""
    s = str(img)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa in s:
        s = sb.join(s.rsplit(sa, 1))
    return Path(s).with_suffix(".txt")


def find_yolo_label(img: Path) -> Path | None:
    """Looks for a YOLO .txt label for an image in the usual places."""
    candidates = [
        ultralytics_label_path(img),
        img.with_suffix(".txt"),
        img.parent / "labels" / f"{img.stem}.txt",
        img.parent.parent / "labels" / img.parent.name / f"{img.stem}.txt",
        img.parent.parent / "labels" / f"{img.stem}.txt",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _read_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return []


def _count_label_file(lbl: Path, counts: dict[int, int]) -> int:
    n = 0
    try:
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            counts[cid] = counts.get(cid, 0) + 1
            n += 1
    except OSError:
        pass
    return n


# =====================================================================
# Dataset statistics from data.yaml
# =====================================================================
def load_data_yaml(data_yaml: Path) -> dict:
    with open(data_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_split_images(data_yaml: Path, split: str) -> list[Path]:
    """Image paths of a split (supports folder, list of folders, or .txt list)."""
    data = load_data_yaml(data_yaml)
    entry = data.get(split)
    if not entry:
        return []
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    for e in entries:
        p = Path(e)
        p = p if p.is_absolute() else (root / p)
        if p.is_dir():
            images += list_images(p)
        elif p.suffix == ".txt" and p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    q = Path(line)
                    images.append(q if q.is_absolute() else (root / q))
    return images


def split_dir_key(data_yaml: Path, split: str) -> str:
    data = load_data_yaml(data_yaml)
    return str(data.get(split) or "")


def has_separate_test_split(data_yaml: Path) -> bool:
    """True when data.yaml has a test split that is not just a copy of val."""
    data = load_data_yaml(data_yaml)
    test, val = data.get("test"), data.get("val")
    return bool(test) and str(test) != str(val) and len(resolve_split_images(data_yaml, "test")) > 0


def dataset_stats(data_yaml: Path) -> dict:
    """Images / labelled images / backgrounds / instances per split and per class."""
    data_yaml = Path(data_yaml)
    data = load_data_yaml(data_yaml)
    names = _read_names(data.get("names"))
    out = {"names": names, "num_classes": len(names), "splits": {}, "per_class": {}}
    seen_dirs = {}
    for split in ("train", "val", "test"):
        key = str(data.get(split) or "")
        if not key:
            continue
        imgs = resolve_split_images(data_yaml, split)
        counts: dict[int, int] = {}
        labelled = instances = 0
        for img in imgs:
            lbl = ultralytics_label_path(img)
            if lbl.exists():
                n = _count_label_file(lbl, counts)
                if n:
                    labelled += 1
                    instances += n
        out["splits"][split] = {
            "images": len(imgs),
            "labelled_images": labelled,
            "background_images": len(imgs) - labelled,
            "instances": instances,
            "per_class": {names[c] if c < len(names) else f"class_{c}": v for c, v in sorted(counts.items())},
            "same_as": seen_dirs.get(key),
        }
        seen_dirs.setdefault(key, split)
    # per-class totals table
    for split, st in out["splits"].items():
        if st.get("same_as"):
            continue
        for cname, v in st["per_class"].items():
            out["per_class"].setdefault(cname, {})[split] = v
    uniq = [s for s, st in out["splits"].items() if not st.get("same_as")]
    out["total_images"] = sum(out["splits"][s]["images"] for s in uniq)
    out["total_instances"] = sum(out["splits"][s]["instances"] for s in uniq)
    out["total_labelled_images"] = sum(out["splits"][s]["labelled_images"] for s in uniq)
    return out


# =====================================================================
# Training results (results.csv + plots)
# =====================================================================
def collect_plots(folder: Path) -> dict[str, str]:
    found = {}
    if not folder or not Path(folder).exists():
        return found
    for key, names in PLOT_FILES.items():
        for n in names:
            p = Path(folder) / n
            if p.exists():
                found[key] = str(p)
                break
    return found


def read_training_results(run_dir: Path) -> dict:
    """Parses results.csv: history per column, best epoch (by fitness), final/best metrics."""
    run_dir = Path(run_dir)
    csv_path = run_dir / "results.csv"
    out = {"history": {}, "epochs_completed": 0, "best_epoch": None, "best": {}, "last": {},
           "plots": collect_plots(run_dir), "csv": str(csv_path) if csv_path.exists() else None}
    if not csv_path.exists():
        return out

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return out

    cols = list(rows[0].keys())
    hist = {c: [_f(r.get(c)) for r in rows] for c in cols}
    out["history"] = hist
    out["epochs_completed"] = len(rows)

    def fitness(r):
        # Ultralytics fitness: 0.1*mAP50 + 0.9*mAP50-95 (box; + mask for seg)
        f = 0.0
        for sfx in ("(B)", "(M)"):
            m50, m = _f(r.get(f"metrics/mAP50{sfx}")), _f(r.get(f"metrics/mAP50-95{sfx}"))
            if m50 is not None and m is not None:
                f += 0.1 * m50 + 0.9 * m
        return f

    best_i = max(range(len(rows)), key=lambda i: fitness(rows[i]))
    ep, first = _f(rows[best_i].get("epoch")), _f(rows[0].get("epoch"))
    # newer Ultralytics numbers epochs from 1, older versions from 0
    out["best_epoch"] = (int(ep) + (1 if first == 0 else 0)) if ep is not None else best_i + 1
    out["best"] = _metrics_from_row(rows[best_i])
    out["last"] = _metrics_from_row(rows[-1])
    if "time" in hist and hist["time"][-1] is not None:
        out["train_seconds"] = hist["time"][-1]
    return out


def _metrics_from_row(r: dict) -> dict:
    m = {}
    for sfx, tag in (("(B)", "box"), ("(M)", "mask")):
        p, rc = _f(r.get(f"metrics/precision{sfx}")), _f(r.get(f"metrics/recall{sfx}"))
        if p is None and rc is None:
            continue
        m[tag] = {
            "precision": p, "recall": rc, "f1": f1_score(p, rc),
            "map50": _f(r.get(f"metrics/mAP50{sfx}")), "map50_95": _f(r.get(f"metrics/mAP50-95{sfx}")),
        }
    return m


# =====================================================================
# Validation (accuracy) with Ultralytics
# =====================================================================
def _metrics_from_val(metrics, names: dict | list) -> dict:
    """Turns an Ultralytics DetMetrics / SegmentMetrics object into a plain dict."""
    rd = dict(getattr(metrics, "results_dict", {}) or {})
    out = {}
    for sfx, tag in (("(B)", "box"), ("(M)", "mask")):
        p = rd.get(f"metrics/precision{sfx}")
        if p is None:
            continue
        r = rd.get(f"metrics/recall{sfx}")
        out[tag] = {"precision": _f(p), "recall": _f(r), "f1": f1_score(p, r),
                    "map50": _f(rd.get(f"metrics/mAP50{sfx}")), "map50_95": _f(rd.get(f"metrics/mAP50-95{sfx}"))}

    # per-class (box)
    names_list = _read_names(names) if not isinstance(names, list) else names
    per_class = []
    box = getattr(metrics, "box", None)
    raw_idx = getattr(box, "ap_class_index", None) if box is not None else None
    idx = [int(i) for i in raw_idx] if raw_idx is not None else []   # numpy array in Ultralytics
    for i, c in enumerate(idx):
        try:
            p, r, ap50, ap = box.class_result(i)
        except Exception:
            continue
        per_class.append({"class": names_list[int(c)] if int(c) < len(names_list) else str(c),
                          "precision": float(p), "recall": float(r), "f1": f1_score(p, r),
                          "map50": float(ap50), "map50_95": float(ap)})
    out["per_class"] = per_class

    # confusion-matrix accuracy
    cm = getattr(metrics, "confusion_matrix", None)
    mat = getattr(cm, "matrix", None) if cm is not None else None
    if mat is not None:
        try:
            import numpy as np
            m = np.asarray(mat, dtype=float)
            nc = m.shape[0] - 1
            tp = float(np.trace(m[:nc, :nc]))
            total = float(m.sum())
            out["accuracy"] = tp / total if total > 0 else None
            # rows = predicted, columns = true (Ultralytics layout); last row/col = background
            out["confusion"] = {"tp": tp, "total_events": total, "matrix": m.astype(int).tolist(),
                                "names": list(names_list[:nc]) + ["background"]}
        except Exception:
            out["accuracy"] = None
    out["speed_ms"] = dict(getattr(metrics, "speed", {}) or {})
    out["save_dir"] = str(getattr(metrics, "save_dir", "") or "")
    out["plots"] = collect_plots(Path(out["save_dir"])) if out["save_dir"] else {}
    return out


def _run_val(model_path, data_yaml: Path, split: str, out_dir: Path, name: str, imgsz=None, log=None,
             cm_conf: float = 0.25) -> dict:
    """
    Two passes, because Ultralytics >= 8.3 builds the confusion matrix with the validation conf:
      1) default conf (0.001)  -> mAP / Precision / Recall / PR-F1 curves (standard numbers)
      2) conf = cm_conf (0.25) -> confusion matrix + Accuracy (otherwise thousands of
                                  low-confidence boxes would be counted as false positives)
    """
    from ultralytics import YOLO  # imported lazily (heavy)
    model = YOLO(str(model_path))
    args = dict(data=str(data_yaml), split=split, project=str(out_dir), exist_ok=True,
                plots=True, verbose=False, workers=0)
    if imgsz:
        args["imgsz"] = imgsz
    if log:
        log(f"📏 Evaluating {Path(model_path).name} on '{split}' split (mAP)...")
    t0 = time.time()
    metrics = model.val(name=name, **args)
    res = _metrics_from_val(metrics, getattr(model, "names", {}) or {})

    if log:
        log(f"📊 Building confusion matrix at conf {cm_conf:.2f}...")
    try:
        cm_metrics = YOLO(str(model_path)).val(name=f"{name}_cm", conf=cm_conf, **args)
        cm_res = _metrics_from_val(cm_metrics, getattr(model, "names", {}) or {})
        res["accuracy"] = cm_res.get("accuracy")
        res["confusion"] = cm_res.get("confusion")
        for k in ("confusion_matrix", "confusion_matrix_normalized"):
            if cm_res["plots"].get(k):
                res["plots"][k] = cm_res["plots"][k]
    except Exception as e:  # keep the mAP numbers even if the 2nd pass fails
        res["accuracy"] = None
        if log:
            log(f"⚠️ Confusion matrix pass failed: {e}")
    res["cm_conf"] = cm_conf
    res["seconds"] = time.time() - t0
    res["task"] = getattr(model, "task", "detect")
    return res


def validate_on_split(model_path, data_yaml: Path, out_dir: Path, split: str | None = None,
                      imgsz=None, log=None, cm_conf: float = 0.25) -> dict:
    """Accuracy on the split made in Step 4. Uses 'test' when it exists, otherwise 'val'."""
    data_yaml = Path(data_yaml)
    if split is None:
        split = "test" if has_separate_test_split(data_yaml) else "val"
    stats = dataset_stats(data_yaml)
    res = _run_val(model_path, data_yaml, split, Path(out_dir), f"eval_{split}", imgsz, log, cm_conf)
    st = stats["splits"].get(split, {})
    res.update({"source": f"{split} split ({data_yaml.name})", "split": split,
                "images": st.get("images", 0), "labelled_images": st.get("labelled_images", 0),
                "instances": st.get("instances", 0), "per_class_instances": st.get("per_class", {}),
                "has_labels": True})
    return res


# ---------------------------------------------------------------------
# External data (image or folder) — labelled or not
# ---------------------------------------------------------------------
def _find_coco_json(folder: Path) -> Path | None:
    cands = []
    for base in (folder, folder.parent):
        cands += list((base / "annotations").glob("*.json")) if (base / "annotations").exists() else []
        cands += list(base.glob("*.json"))
    for c in cands:
        try:
            with open(c, "r", encoding="utf-8") as f:
                head = json.load(f)
            if isinstance(head, dict) and "images" in head and "annotations" in head and "categories" in head:
                return c
        except Exception:
            continue
    return None


def _link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_external_dataset(source: Path, model_names: list[str], task: str, work_dir: Path,
                             log=None) -> dict:
    """
    Finds labels for an image or a folder of images and builds an eval data.yaml whose class
    order matches the model. Returns {"yaml": Path|None, "images": n, "labelled": n, "instances": n, ...}.
    """
    source = Path(source)
    images = [source] if source.is_file() else list_images(source)
    info = {"yaml": None, "images": len(images), "labelled": 0, "instances": 0, "label_source": None,
            "per_class_instances": {}, "unknown_classes": []}
    if not images:
        return info

    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    (work_dir / "images").mkdir(parents=True, exist_ok=True)
    (work_dir / "labels").mkdir(parents=True, exist_ok=True)

    # 1) YOLO .txt labels
    found = {img: find_yolo_label(img) for img in images}
    if any(found.values()):
        info["label_source"] = "YOLO .txt"
        for img, lbl in found.items():
            if lbl:
                shutil.copy2(lbl, work_dir / "labels" / f"{img.stem}.txt")
    else:
        # 2) COCO json (e.g. CVAT export) -> convert, then remap class ids by name to the model
        folder = source.parent if source.is_file() else source
        coco = _find_coco_json(folder)
        if coco:
            info["label_source"] = f"COCO ({coco.name})"
            if log:
                log(f"🔎 Found COCO labels: {coco}")
            from core.converter import CocoToYoloConverter
            tmp_lbl = work_dir / "_coco_labels"
            conv = CocoToYoloConverter().convert(coco, tmp_lbl, mode="segment" if task == "segment" else "detect")
            coco_names = conv.get("classes", [])
            name_to_model = {n: i for i, n in enumerate(model_names)}
            unknown = set()
            wanted = {img.stem for img in images}
            for txt in tmp_lbl.glob("*.txt"):
                if txt.stem not in wanted:
                    continue
                lines = []
                for line in txt.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    cname = coco_names[int(parts[0])] if int(parts[0]) < len(coco_names) else None
                    if cname in name_to_model:
                        lines.append(" ".join([str(name_to_model[cname])] + parts[1:]))
                    elif cname is not None:
                        unknown.add(cname)
                (work_dir / "labels" / txt.name).write_text("\n".join(lines), encoding="utf-8")
            info["unknown_classes"] = sorted(unknown)
            shutil.rmtree(tmp_lbl, ignore_errors=True)

    # count labels
    counts: dict[int, int] = {}
    for img in images:
        lbl = work_dir / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            n = _count_label_file(lbl, counts)
            if n:
                info["labelled"] += 1
                info["instances"] += n
    info["per_class_instances"] = {model_names[c] if c < len(model_names) else f"class_{c}": v
                                   for c, v in sorted(counts.items())}
    if info["labelled"] == 0:
        return info  # unlabeled -> prediction only

    for img in images:
        _link_or_copy(img, work_dir / "images" / img.name)
    yaml_path = work_dir / "eval_data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": work_dir.as_posix(), "train": "images", "val": "images",
        "nc": len(model_names), "names": {i: n for i, n in enumerate(model_names)},
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    info["yaml"] = yaml_path
    return info


def evaluate_external(model_path, source: Path, out_dir: Path, imgsz=None, log=None,
                      cm_conf: float = 0.25) -> dict | None:
    """Accuracy on any labelled image/folder. Returns None when the data has no labels."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    names = _read_names(getattr(model, "names", {}) or {})
    task = getattr(model, "task", "detect")
    work = Path(out_dir) / "_eval_workspace"
    prep = prepare_external_dataset(Path(source), names, task, work, log=log)
    if not prep.get("yaml"):
        if log:
            log("ℹ️ No labels found — prediction only (no accuracy).")
        return None
    res = _run_val(model_path, prep["yaml"], "val", Path(out_dir), "eval_external", imgsz, log, cm_conf)
    res.update({"source": str(source), "split": "external", "images": prep["images"],
                "labelled_images": prep["labelled"], "instances": prep["instances"],
                "per_class_instances": prep["per_class_instances"], "label_source": prep["label_source"],
                "unknown_classes": prep["unknown_classes"], "has_labels": True})
    return res


# =====================================================================
# Report files
# =====================================================================
def _pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def write_report_files(report: dict, folder: Path) -> dict:
    """Writes report.json + report.md next to the run so results can be shared."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    js = folder / "report.json"
    js.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [f"# {report.get('title', 'YOLO Report')}", ""]
    ds = report.get("dataset") or {}
    if ds:
        lines += ["## Dataset", "", "| Split | Images | Labelled | Background | Instances |", "|---|---|---|---|---|"]
        for s, st in ds.get("splits", {}).items():
            lines.append(f"| {s}{' (= ' + st['same_as'] + ')' if st.get('same_as') else ''} | {st['images']} | "
                         f"{st['labelled_images']} | {st['background_images']} | {st['instances']} |")
        lines.append("")
    for key, title in (("val_metrics", "Validation (best epoch)"), ("test_metrics", "Test split"), ("eval", "Evaluation")):
        m = report.get(key)
        if not m:
            continue
        box = m.get("box") or {}
        lines += [f"## {title}", "",
                  f"- Precision: {_pct(box.get('precision'))}", f"- Recall: {_pct(box.get('recall'))}",
                  f"- F1: {_pct(box.get('f1'))}", f"- mAP50: {_pct(box.get('map50'))}",
                  f"- mAP50-95: {_pct(box.get('map50_95'))}"]
        if m.get("accuracy") is not None:
            lines.append(f"- Accuracy (confusion matrix): {_pct(m.get('accuracy'))}")
        lines.append("")
    md = folder / "report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(js), "md": str(md)}