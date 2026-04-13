import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
STORE_DIR = BASE_DIR / "saved_cases_data"
STORE_DIR.mkdir(parents=True, exist_ok=True)


def _case_dir(case_id: str) -> Path:
    return STORE_DIR / case_id


def _meta_path(case_id: str) -> Path:
    return _case_dir(case_id) / "meta.json"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _read_meta(case_id: str) -> Optional[Dict[str, Any]]:
    path = _meta_path(case_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_meta(case_id: str, payload: Dict[str, Any]) -> None:
    cdir = _case_dir(case_id)
    cdir.mkdir(parents=True, exist_ok=True)
    _meta_path(case_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _hash_np_image(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _build_case_fingerprint(
    eye_results: Dict[str, Dict[str, Any]],
    analysis_input_mode: str,
    primary_eye: str,
) -> str:
    payload = {
        "analysis_input_mode": analysis_input_mode,
        "primary_eye": primary_eye,
        "eyes": {},
    }

    for eye in ["right", "left"]:
        if eye not in eye_results:
            continue

        item = eye_results[eye]
        raw_rgb = np.asarray(item["raw_rgb"], dtype=np.uint8)

        payload["eyes"][eye] = {
            "pred_idx": int(item.get("pred_idx", -1)),
            "uploaded_name": _safe_text(item.get("uploaded_name", "")),
            "shape": list(raw_rgb.shape),
            "image_hash": _hash_np_image(raw_rgb),
        }

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_eye_assets(case_id: str, eye: str, eye_result: Dict[str, Any]) -> Dict[str, Any]:
    cdir = _case_dir(case_id)

    raw_rgb = np.asarray(eye_result["raw_rgb"], dtype=np.uint8)

    input_tensor_obj = eye_result.get("input_tensor", None)
    if input_tensor_obj is None:
        raise ValueError(f"Missing input_tensor for {eye} eye while saving case.")

    if hasattr(input_tensor_obj, "detach"):
        input_tensor = input_tensor_obj.detach().cpu().numpy().astype(np.float32)
    else:
        input_tensor = np.asarray(input_tensor_obj, dtype=np.float32)

    raw_name = f"{eye}_raw.png"
    tensor_name = f"{eye}_input.npy"

    Image.fromarray(raw_rgb).save(cdir / raw_name)
    np.save(cdir / tensor_name, input_tensor)

    image_hash = _hash_np_image(raw_rgb)

    payload = {
        "pred_idx": int(eye_result["pred_idx"]),
        "pred_name": str(eye_result["pred_name"]),
        "probs": [float(x) for x in eye_result["probs"]],
        "uploaded_name": str(eye_result.get("uploaded_name", f"{eye}_eye")),
        "raw_file": raw_name,
        "input_tensor_file": tensor_name,
        "image_hash": image_hash,
    }

    display_rgb = eye_result.get("display_rgb", None)
    if display_rgb is not None:
        display_rgb = np.asarray(display_rgb, dtype=np.uint8)
        display_name = f"{eye}_display.png"
        Image.fromarray(display_rgb).save(cdir / display_name)
        payload["display_file"] = display_name

    return payload


def _build_summary(meta: Dict[str, Any]) -> str:
    eyes = meta.get("eyes", {})
    parts = []
    for eye in ["right", "left"]:
        if eye in eyes:
            pred_name = eyes[eye].get("pred_name", "-")
            parts.append(f"{eye.title()}: {pred_name}")
    return " | ".join(parts) if parts else "No eyes"


def _compute_meta_fingerprint(meta: Dict[str, Any]) -> str:
    payload = {
        "analysis_input_mode": meta.get("analysis_input_mode", "single"),
        "primary_eye": meta.get("primary_eye", "right"),
        "eyes": {},
    }

    for eye in ["right", "left"]:
        eye_meta = meta.get("eyes", {}).get(eye)
        if not eye_meta:
            continue
        payload["eyes"][eye] = {
            "pred_idx": int(eye_meta.get("pred_idx", -1)),
            "uploaded_name": _safe_text(eye_meta.get("uploaded_name", "")),
            "image_hash": _safe_text(eye_meta.get("image_hash", "")),
        }

    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prune_saved_cases(max_cases: int = 5) -> None:
    all_meta = []
    for item in STORE_DIR.iterdir():
        if not item.is_dir():
            continue
        meta = _read_meta(item.name)
        if not meta:
            continue
        all_meta.append(meta)

    all_meta.sort(key=lambda x: x.get("saved_at", ""), reverse=True)

    for extra in all_meta[max_cases:]:
        case_id = extra.get("case_id")
        if case_id:
            shutil.rmtree(_case_dir(case_id), ignore_errors=True)


def save_case_bundle(
    *,
    eye_results: Dict[str, Dict[str, Any]],
    analysis_input_mode: str,
    primary_eye: str,
    patient_id: str,
    patient_name: str = "",
    age: Any = "",
    gender: str = "",
    notes: str = "",
    max_cases: int = 5,
) -> Dict[str, Any]:
    patient_id = str(patient_id).strip()
    if not patient_id:
        raise ValueError("Patient ID is required.")

    new_fingerprint = _build_case_fingerprint(
        eye_results=eye_results,
        analysis_input_mode=analysis_input_mode,
        primary_eye=primary_eye,
    )

    # Duplicate check against existing saved cases
    for item in STORE_DIR.iterdir():
        if not item.is_dir():
            continue

        existing = _read_meta(item.name)
        if not existing:
            continue

        existing_fp = existing.get("fingerprint", "")
        if not existing_fp:
            existing_fp = _compute_meta_fingerprint(existing)

        if existing_fp and existing_fp == new_fingerprint:
            existing_pid = _safe_text(existing.get("patient_id"))
            if existing_pid and existing_pid == patient_id:
                raise ValueError(
                    f"This case is already saved for Patient ID {patient_id}. Duplicate save is blocked."
                )
            raise ValueError("This exact case is already saved. Duplicate save is blocked.")

    case_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cdir = _case_dir(case_id)
    cdir.mkdir(parents=True, exist_ok=True)

    eyes_payload = {}
    for eye in ["right", "left"]:
        if eye in eye_results:
            eyes_payload[eye] = _save_eye_assets(case_id, eye, eye_results[eye])

    meta = {
        "case_id": case_id,
        "saved_at": _now_iso(),
        "analysis_input_mode": analysis_input_mode,
        "primary_eye": primary_eye,
        "available_eyes": [eye for eye in ["right", "left"] if eye in eyes_payload],
        "patient_id": patient_id,
        "patient_name": str(patient_name).strip(),
        "age": "" if age is None else str(age).strip(),
        "gender": str(gender).strip(),
        "notes": str(notes).strip(),
        "eyes": eyes_payload,
        "fingerprint": new_fingerprint,
    }
    meta["summary"] = _build_summary(meta)

    _write_meta(case_id, meta)
    prune_saved_cases(max_cases=max_cases)
    return meta


def list_saved_cases(limit: int = 5) -> List[Dict[str, Any]]:
    rows = []
    for item in STORE_DIR.iterdir():
        if not item.is_dir():
            continue
        meta = _read_meta(item.name)
        if meta:
            rows.append(meta)

    rows.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    return rows[:limit]


def load_case_bundle(case_id: str) -> Dict[str, Any]:
    meta = _read_meta(case_id)
    if not meta:
        raise FileNotFoundError(f"Saved case not found: {case_id}")

    cdir = _case_dir(case_id)
    eye_results = {}

    for eye, eye_meta in meta.get("eyes", {}).items():
        raw_path = cdir / eye_meta["raw_file"]
        tensor_path = cdir / eye_meta["input_tensor_file"]

        raw_rgb = np.array(Image.open(raw_path).convert("RGB"), dtype=np.uint8)
        input_tensor = torch.from_numpy(np.load(tensor_path)).float()

        display_rgb = raw_rgb
        display_file = eye_meta.get("display_file")
        if display_file:
            display_path = cdir / display_file
            if display_path.exists():
                display_rgb = np.array(Image.open(display_path).convert("RGB"), dtype=np.uint8)

        eye_results[eye] = {
            "pred_idx": int(eye_meta["pred_idx"]),
            "pred_name": str(eye_meta["pred_name"]),
            "probs": [float(x) for x in eye_meta["probs"]],
            "raw_rgb": raw_rgb,
            "display_rgb": display_rgb,
            "input_tensor": input_tensor,
            "uploaded_name": str(eye_meta.get("uploaded_name", f"{eye}_eye")),
        }

    return {
        "case_id": meta["case_id"],
        "saved_at": meta["saved_at"],
        "analysis_input_mode": meta.get("analysis_input_mode", "single"),
        "primary_eye": meta.get("primary_eye", "right"),
        "available_eyes": meta.get("available_eyes", []),
        "patient_id": meta.get("patient_id", ""),
        "patient_name": meta.get("patient_name", ""),
        "age": meta.get("age", ""),
        "gender": meta.get("gender", ""),
        "notes": meta.get("notes", ""),
        "eye_results": eye_results,
        "summary": meta.get("summary", ""),
    }


def update_case_details(
    case_id: str,
    *,
    patient_id: str,
    patient_name: str = "",
    age: Any = "",
    gender: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    meta = _read_meta(case_id)
    if not meta:
        raise FileNotFoundError(f"Saved case not found: {case_id}")

    patient_id = str(patient_id).strip()
    if not patient_id:
        raise ValueError("Patient ID is required.")

    meta["patient_id"] = patient_id
    meta["patient_name"] = str(patient_name).strip()
    meta["age"] = "" if age is None else str(age).strip()
    meta["gender"] = str(gender).strip()
    meta["notes"] = str(notes).strip()

    _write_meta(case_id, meta)
    return meta


def delete_case_bundle(case_id: str) -> bool:
    case_path = _case_dir(case_id)
    if not case_path.exists() or not case_path.is_dir():
        return False

    shutil.rmtree(case_path, ignore_errors=False)
    return True


def apply_case_to_session(session_state, loaded_case: Dict[str, Any]) -> None:
    session_state["eye_results"] = loaded_case["eye_results"]
    session_state["analysis_input_mode"] = loaded_case.get("analysis_input_mode", "single")
    session_state["primary_eye"] = loaded_case.get("primary_eye", "right")
    session_state["saved_case_id"] = loaded_case.get("case_id", "")
    session_state["saved_case_patient_id"] = loaded_case.get("patient_id", "")
    session_state["saved_case_patient_name"] = loaded_case.get("patient_name", "")
    session_state["saved_case_age"] = loaded_case.get("age", "")
    session_state["saved_case_gender"] = loaded_case.get("gender", "")
    session_state["saved_case_notes"] = loaded_case.get("notes", "")

    session_state["saved_patient_id"] = loaded_case.get("patient_id", "")
    session_state["saved_patient_name"] = loaded_case.get("patient_name", "")
    session_state["saved_patient_age"] = loaded_case.get("age", "")
    session_state["saved_patient_gender"] = loaded_case.get("gender", "")
    session_state["saved_patient_notes"] = loaded_case.get("notes", "")

    session_state["save_patient_id"] = loaded_case.get("patient_id", "")
    session_state["save_patient_name"] = loaded_case.get("patient_name", "")
    session_state["save_patient_age"] = loaded_case.get("age", "")
    session_state["save_patient_gender"] = loaded_case.get("gender", "")
    session_state["save_patient_notes"] = loaded_case.get("notes", "")

    available = loaded_case.get("available_eyes", [])
    if len(available) == 1:
        one_eye = available[0]
        session_state["last_result"] = loaded_case["eye_results"][one_eye]
    else:
        session_state["last_result"] = None

    if "analysis_view_mode" in session_state:
        session_state["analysis_view_mode"] = "Original"