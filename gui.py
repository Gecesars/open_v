from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # noqa: BLE001 - show a friendly message for GUI users
    print("Erro ao importar PySide6. Instale com: pip install -r requirements-gui.txt")
    print(f"Detalhes: {exc}")
    input("Pressione Enter para sair...")
    raise SystemExit(1)

import app as core


APP_DIR = Path(__file__).resolve().parent


@dataclass
class RunConfig:
    source: str
    model: str
    input_path: Optional[Path]
    device: str
    conf: float
    nms: float
    classes: Optional[str]
    camera_index: int
    width: Optional[int]
    height: Optional[int]
    monitor: int
    region: Optional[Tuple[int, int, int, int]]
    screen_fps: float
    show_fps: bool
    save_path: Optional[Path]
    out_fps: float


class ImageComparator:
    def __init__(self) -> None:
        self.ref_gray: Optional[np.ndarray] = None
        self.ref_kp = None
        self.ref_des = None
        self.orb = cv2.ORB_create(800)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def set_reference(self, img_bgr: np.ndarray) -> None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._resize(gray)
        self.ref_gray = gray
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, None)

    def score(self, frame_bgr: np.ndarray) -> float:
        if self.ref_des is None or self.ref_gray is None:
            return 0.0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._resize(gray)
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        if des2 is None or kp2 is None or len(kp2) == 0:
            return 0.0
        matches = self.matcher.knnMatch(self.ref_des, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return len(good) / max(len(self.ref_kp), 1)

    @staticmethod
    def _resize(gray: np.ndarray, width: int = 640) -> np.ndarray:
        h, w = gray.shape[:2]
        if w <= width:
            return gray
        scale = width / float(w)
        new_h = int(h * scale)
        return cv2.resize(gray, (width, new_h), interpolation=cv2.INTER_AREA)


def open_camera(index: int) -> Optional[cv2.VideoCapture]:
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


class CameraTestWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(QtGui.QImage)
    status = QtCore.Signal(str)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, camera_index: int):
        super().__init__()
        self.camera_index = camera_index

    def run(self) -> None:
        cap = None
        try:
            self.status.emit("Abrindo camera...")
            cap = open_camera(self.camera_index)
            if cap is None:
                self.error.emit("Nao foi possivel abrir a camera.")
                return
            ret, frame = cap.read()
            if not ret or frame is None:
                self.error.emit("Camera aberta, mas sem frame.")
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.status.emit("Camera OK.")
        finally:
            if cap is not None:
                cap.release()
            self.finished.emit()


class DetectionWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(QtGui.QImage, float, float, bool)
    status = QtCore.Signal(str)
    error = QtCore.Signal(str)
    stopped = QtCore.Signal()

    def __init__(self, config: RunConfig, ref_image: Optional[np.ndarray], compare_threshold: float):
        super().__init__()
        self.config = config
        self._running = True
        self.comparator = ImageComparator()
        self.compare_enabled = ref_image is not None
        if ref_image is not None:
            self.comparator.set_reference(ref_image)
        self.compare_threshold = compare_threshold
        self.writer = None

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            self.status.emit("Carregando modelo...")
            zoo = core.load_model_zoo(core.MODEL_ZOO_PATH)
            spec = core.build_model_spec(zoo, self.config.model)
            class_names = core.load_class_names(spec.names)
            class_filter = core.parse_class_filter(class_names, self.config.classes)
            detector = core.Detector(spec, self.config.conf, self.config.nms, self.config.device, class_filter)
            colors = core.create_colors(max(1, len(detector.class_names)))
            self.status.emit("Modelo carregado.")
        except Exception as exc:
            self.error.emit(f"Model init failed: {exc}")
            self.stopped.emit()
            return

        cap = None
        sct = None
        monitor = None
        delay = 0.0

        try:
            if self.config.source == "webcam":
                self.status.emit("Abrindo camera...")
                cap = open_camera(self.config.camera_index)
                if cap is None:
                    raise RuntimeError("Nao foi possivel abrir a camera.")
                if self.config.width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                if self.config.height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            elif self.config.source == "video":
                if not self.config.input_path:
                    raise RuntimeError("Input path required for video source.")
                if not self.config.input_path.exists():
                    raise RuntimeError("Arquivo de video nao encontrado.")
                self.status.emit("Abrindo video...")
                cap = cv2.VideoCapture(str(self.config.input_path))
                if not cap.isOpened():
                    raise RuntimeError("Nao foi possivel abrir o video.")
            elif self.config.source == "image":
                if not self.config.input_path:
                    raise RuntimeError("Input path required for image source.")
                if not self.config.input_path.exists():
                    raise RuntimeError("Arquivo de imagem nao encontrado.")
                frame = cv2.imread(str(self.config.input_path))
                if frame is None:
                    raise RuntimeError("Could not read image.")
                self._process_frame(frame, detector, colors, single_frame=True)
                self.stopped.emit()
                return
            else:
                if core.mss is None:
                    raise RuntimeError("mss not installed.")
                sct = core.mss.mss()
                monitors = sct.monitors
                if self.config.monitor < 1 or self.config.monitor >= len(monitors):
                    raise RuntimeError("Invalid monitor index.")
                monitor = monitors[self.config.monitor]
                if self.config.region:
                    x, y, w, h = self.config.region
                    monitor = {"left": x, "top": y, "width": w, "height": h}
                if self.config.screen_fps > 0:
                    delay = 1.0 / self.config.screen_fps
                self.status.emit("Capturando tela...")

            last_time = time.time()
            fps = 0.0
            frame_count = 0
            last_emit = 0.0
            ui_interval = 1.0 / 30.0

            while self._running:
                if self.config.source == "screen":
                    img = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                else:
                    if cap is None:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break

                frame_count += 1
                detections = detector.detect(frame)
                core.draw_detections(frame, detections, detector.class_names, colors)

                if self.config.show_fps:
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

                compare_score = 0.0
                compare_ok = False
                if self.compare_enabled and frame_count % 5 == 0:
                    compare_score = self.comparator.score(frame)
                    compare_ok = compare_score >= self.compare_threshold

                now = time.time()
                if now - last_emit >= ui_interval:
                    self._emit_frame(frame, fps, compare_score, compare_ok)
                    last_emit = now

                if self.config.save_path:
                    self._write_frame(frame)

                if delay > 0:
                    time.sleep(delay)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            if sct is not None:
                sct.close()
            if self.writer is not None:
                self.writer.release()
            self.stopped.emit()

    def _process_frame(self, frame, detector, colors, single_frame=False):
        detections = detector.detect(frame)
        core.draw_detections(frame, detections, detector.class_names, colors)
        compare_score = 0.0
        compare_ok = False
        if self.compare_enabled:
            compare_score = self.comparator.score(frame)
            compare_ok = compare_score >= self.compare_threshold
        self._emit_frame(frame, 0.0, compare_score, compare_ok)
        if self.config.save_path:
            cv2.imwrite(str(self.config.save_path), frame)
        if single_frame:
            return

    def _emit_frame(self, frame, fps, compare_score, compare_ok):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self.frame_ready.emit(qimg, fps, compare_score, compare_ok)

    def _write_frame(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.config.save_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.config.out_fps,
                (w, h),
            )
        self.writer.write(frame)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Vision Studio")
        self.resize(1200, 720)

        self.worker: Optional[DetectionWorker] = None
        self.test_worker: Optional[CameraTestWorker] = None
        self.ref_image: Optional[np.ndarray] = None
        self.compare_threshold = 0.3

        self._build_ui()
        self._load_models()
        self._apply_styles()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)

        left = QtWidgets.QWidget()
        left.setFixedWidth(360)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setSpacing(10)

        self.status_label = QtWidgets.QLabel("Idle")

        # Source
        source_box = QtWidgets.QGroupBox("Source")
        source_layout = QtWidgets.QFormLayout(source_box)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["webcam", "screen", "video", "image"])
        self.source_combo.currentTextChanged.connect(self._on_source_change)
        self.input_edit = QtWidgets.QLineEdit()
        self.input_button = QtWidgets.QPushButton("Browse")
        self.input_button.clicked.connect(self._browse_input)
        input_row = QtWidgets.QHBoxLayout()
        input_row.addWidget(self.input_edit)
        input_row.addWidget(self.input_button)
        input_widget = QtWidgets.QWidget()
        input_widget.setLayout(input_row)
        self.camera_index = QtWidgets.QSpinBox()
        self.camera_index.setRange(0, 10)
        self.monitor_index = QtWidgets.QSpinBox()
        self.monitor_index.setRange(1, 8)
        self.region_edit = QtWidgets.QLineEdit()
        self.region_edit.setPlaceholderText("x,y,w,h")
        self.screen_fps = QtWidgets.QDoubleSpinBox()
        self.screen_fps.setRange(1.0, 120.0)
        self.screen_fps.setValue(30.0)

        source_layout.addRow("Source", self.source_combo)
        source_layout.addRow("Input", input_widget)
        source_layout.addRow("Camera index", self.camera_index)
        source_layout.addRow("Monitor index", self.monitor_index)
        source_layout.addRow("Region", self.region_edit)
        source_layout.addRow("Screen FPS", self.screen_fps)

        # Model
        model_box = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QFormLayout(model_box)
        self.model_combo = QtWidgets.QComboBox()
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["cuda", "cuda_fp16", "cpu"])
        model_layout.addRow("Model", self.model_combo)
        model_layout.addRow("Device", self.device_combo)

        # Detection
        det_box = QtWidgets.QGroupBox("Detection")
        det_layout = QtWidgets.QFormLayout(det_box)
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)
        self.conf_value = QtWidgets.QLabel("0.50")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_value.setText(f"{v/100:.2f}"))
        conf_row = QtWidgets.QHBoxLayout()
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_value)
        conf_widget = QtWidgets.QWidget()
        conf_widget.setLayout(conf_row)

        self.nms_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.nms_slider.setRange(1, 100)
        self.nms_slider.setValue(40)
        self.nms_value = QtWidgets.QLabel("0.40")
        self.nms_slider.valueChanged.connect(lambda v: self.nms_value.setText(f"{v/100:.2f}"))
        nms_row = QtWidgets.QHBoxLayout()
        nms_row.addWidget(self.nms_slider)
        nms_row.addWidget(self.nms_value)
        nms_widget = QtWidgets.QWidget()
        nms_widget.setLayout(nms_row)

        self.classes_edit = QtWidgets.QLineEdit()
        self.classes_edit.setPlaceholderText("person,car or 0,2,3")

        det_layout.addRow("Confidence", conf_widget)
        det_layout.addRow("NMS", nms_widget)
        det_layout.addRow("Classes", self.classes_edit)

        # Output
        out_box = QtWidgets.QGroupBox("Output")
        out_layout = QtWidgets.QFormLayout(out_box)
        self.show_fps = QtWidgets.QCheckBox("Show FPS on frame")
        self.save_edit = QtWidgets.QLineEdit()
        self.save_button = QtWidgets.QPushButton("Save as")
        self.save_button.clicked.connect(self._browse_save)
        save_row = QtWidgets.QHBoxLayout()
        save_row.addWidget(self.save_edit)
        save_row.addWidget(self.save_button)
        save_widget = QtWidgets.QWidget()
        save_widget.setLayout(save_row)
        self.out_fps = QtWidgets.QDoubleSpinBox()
        self.out_fps.setRange(1.0, 120.0)
        self.out_fps.setValue(30.0)

        out_layout.addRow("Save", save_widget)
        out_layout.addRow("Out FPS", self.out_fps)
        out_layout.addRow("", self.show_fps)

        # Comparison
        cmp_box = QtWidgets.QGroupBox("Image Compare")
        cmp_layout = QtWidgets.QFormLayout(cmp_box)
        self.ref_path = QtWidgets.QLineEdit()
        self.ref_button = QtWidgets.QPushButton("Load image")
        self.ref_button.clicked.connect(self._load_reference)
        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(self.ref_path)
        ref_row.addWidget(self.ref_button)
        ref_widget = QtWidgets.QWidget()
        ref_widget.setLayout(ref_row)
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setValue(30)
        self.threshold_value = QtWidgets.QLabel("0.30")
        self.threshold_slider.valueChanged.connect(self._on_threshold_change)
        th_row = QtWidgets.QHBoxLayout()
        th_row.addWidget(self.threshold_slider)
        th_row.addWidget(self.threshold_value)
        th_widget = QtWidgets.QWidget()
        th_widget.setLayout(th_row)
        self.compare_status = QtWidgets.QLabel("No reference")
        self.compare_score = QtWidgets.QLabel("Score: 0.00")
        cmp_layout.addRow("Reference", ref_widget)
        cmp_layout.addRow("Threshold", th_widget)
        cmp_layout.addRow("Status", self.compare_status)
        cmp_layout.addRow("Score", self.compare_score)

        # Buttons
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.test_button = QtWidgets.QPushButton("Test Camera")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.test_button.clicked.connect(self._test_camera)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.start_button)
        btn_row.addWidget(self.stop_button)
        btn_row.addWidget(self.test_button)

        left_layout.addWidget(self.status_label)
        left_layout.addWidget(source_box)
        left_layout.addWidget(model_box)
        left_layout.addWidget(det_box)
        left_layout.addWidget(out_box)
        left_layout.addWidget(cmp_box)
        left_layout.addLayout(btn_row)
        left_layout.addStretch(1)

        # Viewer
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.viewer = QtWidgets.QLabel()
        self.viewer.setAlignment(QtCore.Qt.AlignCenter)
        self.viewer.setMinimumSize(640, 360)
        right_layout.addWidget(self.viewer, 1)

        root.addWidget(left)
        root.addWidget(right, 1)

        self._on_source_change(self.source_combo.currentText())

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QWidget { background: #0f1116; color: #e6e6e6; }
            QGroupBox { border: 1px solid #2b2f3a; margin-top: 8px; padding: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #7bd7c1; }
            QPushButton { background: #1f2937; border: 1px solid #394152; padding: 6px 10px; }
            QPushButton:hover { background: #273244; }
            QPushButton:disabled { color: #6b7280; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #111827; border: 1px solid #374151; padding: 4px; }
            QLabel { color: #e5e7eb; }
            """
        )

    def _load_models(self):
        zoo = core.load_model_zoo(core.MODEL_ZOO_PATH)
        self.model_combo.clear()
        for key, data in zoo.items():
            label = f"{key} - {data.get('name', key)}"
            self.model_combo.addItem(label, key)

    def _on_source_change(self, value: str):
        is_media = value in ("video", "image")
        self.input_edit.setEnabled(is_media)
        self.input_button.setEnabled(is_media)
        self.camera_index.setEnabled(value == "webcam")
        self.monitor_index.setEnabled(value == "screen")
        self.region_edit.setEnabled(value == "screen")
        self.screen_fps.setEnabled(value == "screen")
        self.test_button.setEnabled(value == "webcam" and self.worker is None)

    def _browse_input(self):
        source = self.source_combo.currentText()
        if source == "video":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video")
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select image")
        if path:
            self.input_edit.setText(path)

    def _browse_save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save output")
        if path:
            self.save_edit.setText(path)

    def _load_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select reference image")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.status_label.setText("Could not load reference image.")
            return
        self.ref_image = img
        self.ref_path.setText(path)
        self.compare_status.setText("Reference loaded")

    def _on_threshold_change(self, value: int):
        self.compare_threshold = value / 100.0
        self.threshold_value.setText(f"{self.compare_threshold:.2f}")

    def _start(self):
        if self.worker is not None:
            return
        model_key = self.model_combo.currentData()
        source = self.source_combo.currentText()
        input_path = Path(self.input_edit.text()).resolve() if self.input_edit.text() else None
        save_path = Path(self.save_edit.text()).resolve() if self.save_edit.text() else None
        region = None
        if self.region_edit.text():
            try:
                region = core.parse_region(self.region_edit.text())
            except Exception:
                self.status_label.setText("Invalid region format.")
                return

        config = RunConfig(
            source=source,
            model=model_key,
            input_path=input_path,
            device=self.device_combo.currentText(),
            conf=self.conf_slider.value() / 100.0,
            nms=self.nms_slider.value() / 100.0,
            classes=self.classes_edit.text().strip() or None,
            camera_index=self.camera_index.value(),
            width=None,
            height=None,
            monitor=self.monitor_index.value(),
            region=region,
            screen_fps=self.screen_fps.value(),
            show_fps=self.show_fps.isChecked(),
            save_path=save_path if save_path else None,
            out_fps=self.out_fps.value(),
        )

        self.worker = DetectionWorker(config, self.ref_image, self.compare_threshold)
        self.worker.frame_ready.connect(self._update_frame)
        self.worker.status.connect(self._on_status)
        self.worker.error.connect(self._on_error)
        self.worker.stopped.connect(self._on_stopped)
        self.worker.start()

        self._set_running(True)
        self.status_label.setText("Iniciando...")

    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.status_label.setText("Stopping...")

    def _on_stopped(self):
        self._set_running(False)
        self.worker = None
        self.status_label.setText("Stopped")

    def _on_error(self, message: str):
        self.status_label.setText(f"Error: {message}")

    def _on_status(self, message: str):
        self.status_label.setText(message)

    def _test_camera(self):
        if self.worker is not None or self.test_worker is not None:
            return
        self.status_label.setText("Testando camera...")
        self.test_worker = CameraTestWorker(self.camera_index.value())
        self.test_worker.status.connect(self._on_status)
        self.test_worker.error.connect(self._on_error)
        self.test_worker.frame_ready.connect(self._on_test_frame)
        self.test_worker.finished.connect(self._on_test_finished)
        self._set_testing(True)
        self.test_worker.start()

    def _update_frame(self, qimg: QtGui.QImage, fps: float, score: float, ok: bool):
        pix = QtGui.QPixmap.fromImage(qimg)
        self.viewer.setPixmap(pix.scaled(self.viewer.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        if self.ref_image is not None:
            self.compare_score.setText(f"Score: {score:.2f}")
            self.compare_status.setText("VALID" if ok else "NOT VALID")

    def _set_running(self, running: bool):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.test_button.setEnabled(not running and self.source_combo.currentText() == "webcam")
        self.source_combo.setEnabled(not running)
        self.model_combo.setEnabled(not running)
        self.device_combo.setEnabled(not running)
        self.input_edit.setEnabled(not running and self.source_combo.currentText() in ("video", "image"))
        self.input_button.setEnabled(not running and self.source_combo.currentText() in ("video", "image"))
        self.camera_index.setEnabled(not running and self.source_combo.currentText() == "webcam")
        self.monitor_index.setEnabled(not running and self.source_combo.currentText() == "screen")
        self.region_edit.setEnabled(not running and self.source_combo.currentText() == "screen")
        self.screen_fps.setEnabled(not running and self.source_combo.currentText() == "screen")
        self.save_edit.setEnabled(not running)
        self.save_button.setEnabled(not running)
        self.show_fps.setEnabled(not running)
        self.conf_slider.setEnabled(not running)
        self.nms_slider.setEnabled(not running)
        self.classes_edit.setEnabled(not running)
        self.threshold_slider.setEnabled(not running)
        self.ref_button.setEnabled(not running)

    def _set_testing(self, testing: bool):
        self.start_button.setEnabled(not testing)
        self.stop_button.setEnabled(False)
        self.test_button.setEnabled(not testing)

    def _on_test_frame(self, qimg: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(qimg)
        self.viewer.setPixmap(
            pix.scaled(self.viewer.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    def _on_test_finished(self):
        self._set_testing(False)
        self.test_worker = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.wait(1000)
        if self.test_worker:
            self.test_worker.wait(1000)
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
