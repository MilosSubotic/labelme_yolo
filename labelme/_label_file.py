from __future__ import annotations

import base64
import io
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any
from typing import Final
from typing import TypedDict

import numpy as np
import PIL.Image
import tifffile
from loguru import logger
from numpy.typing import NDArray

from labelme import __version__
from labelme import utils

PIL.Image.MAX_IMAGE_PIXELS = None


class ShapeDict(TypedDict):
    label: str
    points: list[list[float]]
    shape_type: str
    flags: dict[str, bool]
    description: str
    group_id: int | None
    mask: NDArray[np.bool_] | None
    other_data: dict


def _load_shape_json_obj(shape_json_obj: dict) -> ShapeDict:
    SHAPE_KEYS: set[str] = {
        "label",
        "points",
        "group_id",
        "shape_type",
        "flags",
        "description",
        "mask",
    }

    if "label" not in shape_json_obj:
        raise ValueError(f"label is required: {shape_json_obj}")
    if not isinstance(shape_json_obj["label"], str):
        raise TypeError(f"label must be str: {shape_json_obj['label']}")
    label: str = shape_json_obj["label"]

    if "points" not in shape_json_obj:
        raise ValueError(f"points is required: {shape_json_obj}")
    if not isinstance(shape_json_obj["points"], list):
        raise TypeError(f"points must be list: {shape_json_obj['points']}")
    if not shape_json_obj["points"]:
        raise ValueError(f"points must be non-empty: {shape_json_obj}")
    if not all(
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(xy, int | float) for xy in point)
        for point in shape_json_obj["points"]
    ):
        raise ValueError(f"points must be list of [x, y]: {shape_json_obj['points']}")
    points: list[list[float]] = shape_json_obj["points"]

    if "shape_type" not in shape_json_obj:
        raise ValueError(f"shape_type is required: {shape_json_obj}")
    if not isinstance(shape_json_obj["shape_type"], str):
        raise TypeError(f"shape_type must be str: {shape_json_obj['shape_type']}")
    shape_type: str = shape_json_obj["shape_type"]

    flags: dict = {}
    if shape_json_obj.get("flags") is not None:
        if not isinstance(shape_json_obj["flags"], dict):
            raise TypeError(f"flags must be dict: {shape_json_obj['flags']}")
        if not all(
            isinstance(k, str) and isinstance(v, bool)
            for k, v in shape_json_obj["flags"].items()
        ):
            raise TypeError(
                f"flags must be dict of str to bool: {shape_json_obj['flags']}"
            )
        flags = shape_json_obj["flags"]

    description: str = ""
    if shape_json_obj.get("description") is not None:
        if not isinstance(shape_json_obj["description"], str):
            raise TypeError(f"description must be str: {shape_json_obj['description']}")
        description = shape_json_obj["description"]

    group_id: int | None = None
    if shape_json_obj.get("group_id") is not None:
        if not isinstance(shape_json_obj["group_id"], int):
            raise TypeError(f"group_id must be int: {shape_json_obj['group_id']}")
        group_id = shape_json_obj["group_id"]

    mask: NDArray[np.bool_] | None = None
    if shape_json_obj.get("mask") is not None:
        if not isinstance(shape_json_obj["mask"], str):
            raise TypeError(
                f"mask must be base64-encoded PNG: {shape_json_obj['mask']}"
            )
        mask = utils.img_b64_to_arr(shape_json_obj["mask"]).astype(bool)

    other_data = {k: v for k, v in shape_json_obj.items() if k not in SHAPE_KEYS}

    loaded: ShapeDict = ShapeDict(
        label=label,
        points=points,
        shape_type=shape_type,
        flags=flags,
        description=description,
        group_id=group_id,
        mask=mask,
        other_data=other_data,
    )
    if set(loaded.keys()) != SHAPE_KEYS | {"other_data"}:
        raise RuntimeError(
            f"unexpected keys: {set(loaded.keys())} != {SHAPE_KEYS | {'other_data'}}"
        )
    return loaded


class LabelFileError(Exception):
    """Base for read/write failures of labelme JSON annotation files."""


class LabelFileReadError(LabelFileError):
    """Wraps an underlying parse or image-decode failure during load."""


class LabelFileWriteError(LabelFileError):
    """Wraps an underlying I/O failure during save."""


@dataclass(frozen=True)
class LabelData:
    image_path: str
    image_data: bytes
    shapes: list[ShapeDict]
    flags: dict[str, bool]
    other_data: dict[str, Any]


LABEL_FILE_SUFFIX: Final[str] = ".json"

_RESERVED_TOP_LEVEL_KEYS: Final[tuple[str, ...]] = (
    "version",
    "imageData",
    "imagePath",
    "shapes",
    "flags",
    "imageHeight",
    "imageWidth",
)


def is_label_file_path(filename: str) -> bool:
    return Path(filename).suffix.lower() == LABEL_FILE_SUFFIX


def read_image_file(filename: str) -> bytes:
    t_start = time.time()
    image_pil = _imread(filename=filename)

    oriented: PIL.Image.Image = utils.apply_exif_orientation(image=image_pil)
    ext = Path(filename).suffix.lower()
    if oriented is image_pil and ext in (".jpg", ".jpeg", ".png"):
        with open(filename, "rb") as f:
            image_data = f.read()
    else:
        with io.BytesIO() as f:
            fmt = "PNG" if "A" in oriented.mode else "JPEG"
            oriented.save(fp=f, format=fmt, quality=95)
            f.seek(0)
            image_data = f.read()

    logger.debug(
        "Loaded image file: {!r} in {:.0f}ms", filename, (time.time() - t_start) * 1000
    )
    return image_data


def _check_image_dimensions(
    *,
    image_data: bytes,
    expected_height: int | None,
    expected_width: int | None,
) -> None:
    if expected_height is None and expected_width is None:
        return
    actual_w, actual_h = utils.img_data_to_pil(img_data=image_data).size
    if expected_height is not None and expected_height != actual_h:
        raise ValueError(
            f"imageHeight mismatch: declared={expected_height}, actual={actual_h}"
        )
    if expected_width is not None and expected_width != actual_w:
        raise ValueError(
            f"imageWidth mismatch: declared={expected_width}, actual={actual_w}"
        )


def read_label_file(filename: str) -> LabelData:
    try:
        with open(filename, encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
        image_path = PureWindowsPath(raw["imagePath"]).as_posix()
        if raw["imageData"] is not None:
            image_data = base64.b64decode(raw["imageData"])
        else:
            image_data = read_image_file(
                filename=str(Path(filename).parent / image_path)
            )
        _check_image_dimensions(
            image_data=image_data,
            expected_height=raw.get("imageHeight"),
            expected_width=raw.get("imageWidth"),
        )
        shapes: list[ShapeDict] = [
            _load_shape_json_obj(shape_json_obj=s) for s in raw["shapes"]
        ]
        flags = raw.get("flags") or {}
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as e:
        raise LabelFileReadError(f"failed to load {filename!r}: {e}") from e
    other_data = {k: v for k, v in raw.items() if k not in _RESERVED_TOP_LEVEL_KEYS}
    return LabelData(
        image_path=image_path,
        image_data=image_data,
        shapes=shapes,
        flags=flags,
        other_data=other_data,
    )

def _write_yolo_txt_file(
    filename: str,
    shapes: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    label_to_id: dict[str, int] | None = None,
) -> None:
    """Write YOLO polygon segmentation .txt file alongside the .json label file."""
    if label_to_id is None:
        seen: list[str] = []
        for s in shapes:
            lbl = s.get("label", "")
            if lbl not in seen:
                seen.append(lbl)
        label_to_id = {lbl: i for i, lbl in enumerate(seen)}

    txt_path = Path(filename).with_suffix(".txt")
    lines: list[str] = []

    for shape in shapes:
        lbl = shape.get("label", "")
        class_id = label_to_id.get(lbl)
        if class_id is None:
            logger.warning("Label {!r} not in label_to_id, skipping", lbl)
            continue

        shape_type = shape.get("shape_type", "polygon")
        points: list[list[float]] = shape.get("points", [])

        if not points:
            continue

        if shape_type in ("polygon", "rectangle", "linestrip"):
            coords = []
            for x, y in points:
                coords.append(f"{x / image_width:.6f}")
                coords.append(f"{y / image_height:.6f}")
            lines.append(f"{class_id} " + " ".join(coords))
        else:
            logger.warning(
                "shape_type={!r} not supported for YOLO export, skipping", shape_type
            )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    logger.debug("Wrote YOLO polygon txt: {!r}", str(txt_path))

def write_label_file(
    filename: str,
    *,
    shapes: list[dict[str, Any]],
    image_path: str,
    image_height: int | None,
    image_width: int | None,
    image_data: bytes | None = None,
    other_data: dict[str, Any] | None = None,
    flags: dict[str, bool] | None = None,
) -> None:
    try:
        image_data_b64: str | None = None
        if image_data is not None:
            _check_image_dimensions(
                image_data=image_data,
                expected_height=image_height,
                expected_width=image_width,
            )
            image_data_b64 = base64.b64encode(image_data).decode("utf-8")
        payload: dict[str, Any] = {
            "version": __version__,
            "flags": dict(flags) if flags else {},
            "shapes": list(shapes),
            "imagePath": image_path,
            "imageData": image_data_b64,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }
        for key, value in (other_data or {}).items():
            if key in _RESERVED_TOP_LEVEL_KEYS:
                raise ValueError(f"reserved key in other_data: {key!r}")
            payload[key] = value
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # call .txt
        if image_height is not None and image_width is not None:
            _write_yolo_txt_file(
                filename=filename,
                shapes=shapes,
                image_height=image_height,
                image_width=image_width,
            )

    except (OSError, TypeError, ValueError) as e:
        raise LabelFileWriteError(f"failed to write {filename!r}: {e}") from e

YOLO_FILE_SUFFIX: Final[str] = ".txt"

def is_yolo_file_path(filename: str) -> bool:
    return Path(filename).suffix.lower() == YOLO_FILE_SUFFIX



def read_yolo_txt_file(filename: str, *, image_width: int, image_height: int, id_to_label: dict[int, str] | None = None) -> LabelData:
    """Read a YOLO .txt annotation file and return LabelData with polygon shapes."""
    txt_path = Path(filename)
    # Find the image next to the .txt
    image_path: str | None = None
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidate = txt_path.with_suffix(ext)
        if candidate.exists():
            image_path = candidate.name
            image_data = read_image_file(filename=str(candidate))
            # Get actual dimensions from image
            img_pil = utils.img_data_to_pil(img_data=image_data)
            image_width, image_height = img_pil.size  # (w, h)
            break
    if image_path is None:
        raise LabelFileReadError(f"No image found next to {filename!r}")

    shapes: list[ShapeDict] = []
    try:
        with open(filename, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    except OSError as e:
        raise LabelFileReadError(f"failed to load {filename!r}: {e}") from e

    for line in lines:
        parts = line.split()
        if len(parts) < 5 or (len(parts) > 5 and len(parts) % 2 == 0):
            logger.warning("Skipping unexpected YOLO line: {!r}", line)
            continue
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        label = (id_to_label or {}).get(class_id, str(class_id))

        if len(coords) == 4:
            # YOLO bbox format: cx cy w h -> 4-point polygon
            cx, cy, w, h = coords
            x_min = (cx - w / 2) * image_width
            x_max = (cx + w / 2) * image_width
            y_min = (cy - h / 2) * image_height
            y_max = (cy + h / 2) * image_height
            points = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ]
            shape_type = "polygon"
        else:
            # YOLO polygon segmentation
            points = [
                [coords[i] * image_width, coords[i + 1] * image_height]
                for i in range(0, len(coords), 2)
            ]
            shape_type = "polygon"

        shapes.append(ShapeDict(
            label=label,
            points=points,
            shape_type=shape_type,
            flags={},
            description="",
            group_id=None,
            mask=None,
            other_data={},
        ))

    return LabelData(
        image_path=image_path,
        image_data=image_data,
        shapes=shapes,
        flags={},
        other_data={},
    )



class LabelFile:
    shapes: list[ShapeDict]
    suffix: Final[str] = LABEL_FILE_SUFFIX

    def __init__(self, filename: str | None = None) -> None:
        self.shapes = []
        self.image_path: str | None = None
        self.image_data: bytes | None = None
        self.other_data: dict[str, Any] = {}
        self.flags: dict[str, bool] = {}
        self.filename: str | None = filename
        if filename is not None:
            self.load(filename=filename)

    @property
    def imagePath(self) -> str | None:
        warnings.warn(
            "LabelFile.imagePath is deprecated and will be removed in a future "
            "release; use image_path",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.image_path

    @imagePath.setter
    def imagePath(self, value: str | None) -> None:
        warnings.warn(
            "LabelFile.imagePath is deprecated and will be removed in a future "
            "release; use image_path",
            DeprecationWarning,
            stacklevel=2,
        )
        self.image_path = value

    @property
    def imageData(self) -> bytes | None:
        warnings.warn(
            "LabelFile.imageData is deprecated and will be removed in a future "
            "release; use image_data",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.image_data

    @imageData.setter
    def imageData(self, value: bytes | None) -> None:
        warnings.warn(
            "LabelFile.imageData is deprecated and will be removed in a future "
            "release; use image_data",
            DeprecationWarning,
            stacklevel=2,
        )
        self.image_data = value

    @property
    def otherData(self) -> dict[str, Any]:
        warnings.warn(
            "LabelFile.otherData is deprecated and will be removed in a future "
            "release; use other_data",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.other_data

    @otherData.setter
    def otherData(self, value: dict[str, Any]) -> None:
        warnings.warn(
            "LabelFile.otherData is deprecated and will be removed in a future "
            "release; use other_data",
            DeprecationWarning,
            stacklevel=2,
        )
        self.other_data = value

    @staticmethod
    def load_image_file(filename: str) -> bytes:
        return read_image_file(filename=filename)

    def load(self, filename: str) -> None:
        loaded: LabelData = read_label_file(filename=filename)
        self.flags = loaded.flags
        self.shapes = loaded.shapes
        self.image_path = loaded.image_path
        self.image_data = loaded.image_data
        self.other_data = loaded.other_data
        self.filename = filename

    def save(
        self,
        filename: str,
        shapes: list[dict[str, Any]],
        image_path: str,
        image_height: int | None,
        image_width: int | None,
        image_data: bytes | None = None,
        other_data: dict[str, Any] | None = None,
        flags: dict[str, bool] | None = None,
    ) -> None:
        write_label_file(
            filename=filename,
            shapes=shapes,
            image_path=image_path,
            image_height=image_height,
            image_width=image_width,
            image_data=image_data,
            other_data=other_data,
            flags=flags,
        )
        self.filename = filename

    @staticmethod
    def is_label_file(filename: str) -> bool:
        return is_label_file_path(filename=filename)


_DISPLAYABLE_MODES = {"1", "L", "P", "RGB", "RGBA", "LA", "PA"}


def _imread(filename: str) -> PIL.Image.Image:
    ext: str = Path(filename).suffix.lower()
    try:
        image_pil = PIL.Image.open(filename)
        if image_pil.mode not in _DISPLAYABLE_MODES:
            raise PIL.UnidentifiedImageError
        return image_pil
    except PIL.UnidentifiedImageError:
        if ext in (".tif", ".tiff"):
            return _imread_tiff(filename)
        raise


def _imread_tiff(filename: str) -> PIL.Image.Image:
    img_arr: NDArray = tifffile.imread(filename)

    if img_arr.ndim == 2:
        img_arr_normalized = _normalize_to_uint8(img_arr)
    elif img_arr.ndim == 3:
        if img_arr.shape[2] >= 3:
            img_arr_normalized = np.stack(
                [_normalize_to_uint8(img_arr[:, :, i]) for i in range(3)],
                axis=2,
            )
        else:
            img_arr_normalized = _normalize_to_uint8(img_arr[:, :, 0])
    else:
        raise OSError(f"Unsupported image shape: {img_arr.shape}")

    return PIL.Image.fromarray(img_arr_normalized)


def _normalize_to_uint8(arr: NDArray) -> NDArray[np.uint8]:
    arr = arr.astype(np.float64)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if np.isnan(min_val) or np.isnan(max_val) or max_val - min_val == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    normalized = (arr - min_val) / (max_val - min_val) * 255
    return np.clip(normalized, 0, 255).astype(np.uint8)
