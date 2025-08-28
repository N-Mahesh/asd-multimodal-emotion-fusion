import base64
import io
import cv2
import numpy as np
from PIL import Image
from retinaface import RetinaFace

def _b64_to_cv2(b64_str: str) -> np.ndarray:
    """Decode base64 -> OpenCV BGR image."""
    if "," in b64_str: 
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str, validate=True)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode base64 image.")
    return img

def _cv2_to_b64(img_bgr: np.ndarray, format: str = "PNG") -> str:
    """Encode OpenCV BGR image -> base64 string"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _pick_largest_face(faces):
    """
    faces: RetinaFace.detect_faces() result.
    Returns (x1, y1, x2, y2) for the largest face, or None if no faces.
    """
    if not isinstance(faces, dict) or len(faces) == 0:
        return None

    best_box = None
    best_area = -1

    # faces is a dict
    for _, f in faces.items():
        box = f.get("facial_area") or f.get("box")
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)

    return best_box

def expand_box_xyxy(box, img_shape, scale: float = 1.10, pad_px: int = 0):
    x1, y1, x2, y2 = map(float, box)
    h, w = img_shape[:2]

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * scale + 2 * pad_px
    bh = (y2 - y1) * scale + 2 * pad_px

    nx1 = int(round(cx - bw * 0.5))
    ny1 = int(round(cy - bh * 0.5))
    nx2 = int(round(cx + bw * 0.5))
    ny2 = int(round(cy + bh * 0.5))

    nx1 = max(0, min(nx1, w - 1))
    nx2 = max(0, min(nx2, w - 1))
    ny1 = max(0, min(ny1, h - 1))
    ny2 = max(0, min(ny2, h - 1))
    return nx1, ny1, nx2, ny2

def draw_single_face_box_b64(
    input_b64,
    box_color=(0, 0, 255),
    thickness=3,
    *,
    faces=None,
    best_box=None,
    scale: float = 1.05,
    pad_px: int = 4,
) -> str:
    """
    Detect faces with RetinaFace, draw red rectangle (largest face), and
    return the result image as base64.
    """
    # Decode
    img = _b64_to_cv2(input_b64)

    # Detect faces (unless provided)
    if best_box is None:
        if faces is None:
            faces = RetinaFace.detect_faces(img)
        best_box = _pick_largest_face(faces)

    if best_box is not None:
        x1, y1, x2, y2 = expand_box_xyxy(best_box, img.shape, scale=scale, pad_px=pad_px)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

    # Encode back to base64
    return _cv2_to_b64(img)

def draw_and_crop_single_face_b64(
    input_b64,
    box_color=(0, 0, 255),
    thickness=3,
    *,
    scale: float = 1.05,
    pad_px: int = 4,
):
    # Decode once for crop calculations and single inference
    img = _b64_to_cv2(input_b64)

    # Single inference
    faces = RetinaFace.detect_faces(img)
    best_box = _pick_largest_face(faces)

    # Produce the boxed image without re-running detection
    boxed_b64 = draw_single_face_box_b64(
        input_b64,
        box_color=box_color,
        thickness=thickness,
        faces=faces,
        best_box=best_box,
        scale=scale,
        pad_px=pad_px,
    )

    # Produce the cropped image (expanded box)
    crop_b64 = None
    if best_box is not None:
        x1, y1, x2, y2 = expand_box_xyxy(best_box, img.shape, scale=scale, pad_px=pad_px)
        h, w = img.shape[:2]
        xs2 = min(x2 + 1, w)
        ys2 = min(y2 + 1, h)
        crop = img[y1:ys2, x1:xs2]
        if crop.size > 0:
            crop_b64 = _cv2_to_b64(crop)

    return boxed_b64, crop_b64