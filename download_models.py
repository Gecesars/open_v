import argparse
import sys
import urllib.request
from pathlib import Path


MODEL_SOURCES = {
    "yolov3": {
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    },
    "yolov4-tiny": {
        "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
        "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
    },
    "yolov5s-onnx": {
        "yolov5s.onnx": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx",
        "coco.names": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.names",
    },
    "yolov5m-onnx": {
        "yolov5m.onnx": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.onnx",
        "coco.names": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.names",
    },
    "yolov5l-onnx": {
        "yolov5l.onnx": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.onnx",
        "coco.names": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.names",
    },
    "yolov5x-onnx": {
        "yolov5x.onnx": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x.onnx",
        "coco.names": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.names",
    },
}


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"Skip: {dest.name} (ja existe)")
        return
    print(f"Download: {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixar modelos para o app")
    parser.add_argument("--model", action="append", help="Nome do modelo")
    parser.add_argument("--all", action="store_true", help="Baixar tudo")
    parser.add_argument("--force", action="store_true", help="Sobrescrever arquivos")
    parser.add_argument("--list", action="store_true", help="Listar modelos")
    args = parser.parse_args()

    if args.list:
        for key in MODEL_SOURCES:
            print(f"- {key}")
        return 0

    if not args.all and not args.model:
        print("Use --model <nome> ou --all")
        return 1

    models = MODEL_SOURCES.keys() if args.all else args.model
    for key in models:
        if key not in MODEL_SOURCES:
            print(f"Modelo desconhecido: {key}")
            continue
        for filename, url in MODEL_SOURCES[key].items():
            dest = Path(__file__).resolve().parent / filename
            download(url, dest, args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
