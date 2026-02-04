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


APP_DIR = Path(__file__).resolve().parent
MODEL_ZOO_PATH = APP_DIR / "model_zoo.json"


@dataclass
class ModelSpec:
    key: str
    name: str
    model_type: str  # "darknet" or "onnx-yolo"
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
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {spec.model_type}")

        set_device(self.net, device)

    def detect(self, frame: np.ndarray) -> List[dict]:
        if self.spec.model_type == "darknet":
            return self._detect_darknet(frame)
        return self._detect_onnx_yolo(frame)

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
