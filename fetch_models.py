"""Download and prepare the detection models. Run once after creating the venv.

Weights are not vendored in the repo: the barcode model is AGPL-3.0, and the YuNet face model
needs a graph patch that is easier to redo than to explain.

    python fetch_models.py
"""
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).parent / "models"
HF = "https://huggingface.co"

# name on disk -> (url, licence)
FILES = {
    "face_yunet.onnx": (f"{HF}/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx", "MIT"),
    "text_ppocrv3.onnx": (f"{HF}/opencv/text_detection_ppocr/resolve/main/text_detection_en_ppocrv3_2023may.onnx", "Apache-2.0"),
    "barcode_yolov8s.pt": (f"{HF}/Piero2411/YOLOV8s-Barcode-Detection/resolve/main/YOLOV8s_Barcode_Detection.pt", "AGPL-3.0"),
}


def download():
    MODELS.mkdir(exist_ok=True)
    for name, (url, lic) in FILES.items():
        dest = MODELS / name
        if dest.exists():
            print(f"have {name}")
            continue
        print(f"fetching {name} ({lic}) ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  {dest.stat().st_size / 1e6:.1f} MB")


def patch_face():
    """YuNet ships with a fixed 640x640 input. Downscaling a 720x1280 portrait frame into that
    loses small faces -- a live test dropped one of four. Making the input dynamic lets us pad the
    frame up to a multiple of 64 instead, keeping every original pixel."""
    out = MODELS / "face_yunet_dyn.onnx"
    if out.exists():
        return print("have face_yunet_dyn.onnx")
    import onnx
    m = onnx.load(str(MODELS / "face_yunet.onnx"))
    dim = m.graph.input[0].type.tensor_type.shape.dim
    dim[2].dim_param, dim[3].dim_param = "h", "w"
    for o in m.graph.output:            # let the shapes be re-inferred at run time
        o.type.tensor_type.shape.Clear()
    onnx.save(m, str(out))
    print("wrote face_yunet_dyn.onnx (dynamic input; feed dims divisible by 64)")


def export_barcode():
    """Ultralytics is needed only here, never at run time. dynamic=True so the model takes a
    padded portrait frame rather than a squashed 640x640 square."""
    out = MODELS / "barcode_yolov8s.onnx"
    if out.exists():
        return print("have barcode_yolov8s.onnx")
    from ultralytics import YOLO
    YOLO(str(MODELS / "barcode_yolov8s.pt")).export(format="onnx", imgsz=640, dynamic=True, simplify=False)
    print("wrote barcode_yolov8s.onnx")


if __name__ == "__main__":
    download()
    patch_face()
    try:
        export_barcode()
    except ImportError:
        sys.exit("install ultralytics to export the barcode model: pip install ultralytics")
    print("\nmodels ready")
