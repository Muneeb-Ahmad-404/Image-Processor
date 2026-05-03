"""
ParallelDerm — Hair Removal Preprocessing Pipeline
Clinical-grade PyQt6 GUI  |  Serial / OMP / MPI / OCL backends
"""

import sys, os, time, subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox,
    QSlider, QFileDialog, QHBoxLayout, QVBoxLayout,
    QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect
from PyQt6.QtGui  import (
    QPainter, QColor, QPen, QFont,
    QLinearGradient, QPixmap, QMouseEvent, QPainterPath,
)

# ── palette ───────────────────────────────────────────────────────────────────
BG        = "#07090f"
PANEL     = "#0d1117"
SURFACE   = "#111827"
BORDER    = "#1a2535"
ACCENT    = "#00d4ff"
ACCENT_DK = "#008fa8"
TEXT      = "#dce8f5"
MUTED     = "#3d5570"
SUCCESS   = "#00e5a0"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

METHOD_COL = {
    "Serial": "#4a6080",
    "OMP":    "#00d4ff",
    "MPI":    "#a78bfa",
    "OCL":    "#00e5a0",
}

STYLESHEET = f"""
* {{ box-sizing: border-box; }}
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "IBM Plex Mono", "Courier New", monospace;
    font-size: 12px;
}}
QLabel#app_title {{
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 3px;
    color: {TEXT};
}}
QLabel#app_sub {{
    font-size: 10px;
    letter-spacing: 2px;
    color: {MUTED};
}}
QFrame#card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 7px 16px;
    color: {TEXT};
    font-family: "IBM Plex Mono", "Courier New", monospace;
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton:hover  {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT_DK}; color: #000; }}
QPushButton#run_btn {{
    background: {ACCENT};
    color: #000;
    font-weight: 700;
    font-size: 12px;
    border: none;
    letter-spacing: 2px;
    min-width: 90px;
    padding: 9px 20px;
}}
QPushButton#run_btn:hover   {{ background: #33ddff; }}
QPushButton#run_btn:pressed {{ background: {ACCENT_DK}; }}
QPushButton#run_btn:disabled{{ background: {SURFACE}; color: {MUTED}; border: 1px solid {BORDER}; }}
QComboBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 7px 12px;
    color: {TEXT};
    font-family: "IBM Plex Mono", "Courier New", monospace;
    font-size: 11px;
    min-width: 80px;
}}
QComboBox:hover {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    color: {TEXT};
    selection-background-color: {ACCENT_DK};
}}
QSlider::groove:horizontal {{
    height: 2px; background: {BORDER}; border-radius: 1px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 1px; }}
QSlider::handle:horizontal {{
    background: {ACCENT}; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px;
}}
QLabel#section {{
    color: {MUTED};
    font-size: 9px;
    letter-spacing: 2px;
}}
QLabel#metric_val {{
    font-size: 20px;
    font-weight: 700;
    color: {TEXT};
}}
QLabel#metric_key {{
    font-size: 9px;
    letter-spacing: 2px;
    color: {MUTED};
}}
QLabel#path_display {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 10px;
    color: {MUTED};
    font-size: 10px;
}}
QLabel#status_badge {{
    border-radius: 3px;
    padding: 3px 10px;
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: 700;
}}
QFrame#h_divider {{
    background: {BORDER};
    max-height: 1px;
    min-height: 1px;
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Worker
# ══════════════════════════════════════════════════════════════════════════════

class Worker(QThread):
    done = pyqtSignal()

    def __init__(self, method: str, input_dir: str, output_dir: str, workers: int):
        super().__init__()
        self.method      = method.lower()
        self.input_dir   = input_dir
        self.output_dir  = output_dir
        self.workers     = workers

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        cmd_map = {
            "serial": ["./backend/hair_removal_serial", self.input_dir, self.output_dir],
            "omp":    ["./backend/hair_removal_omp",    self.input_dir, self.output_dir, str(self.workers)],
            "mpi":    ["mpirun", "-np", str(self.workers),
                       "./backend/hair_removal_mpi",    self.input_dir, self.output_dir],
            "ocl":    ["./backend/hair_removal_ocl",    self.input_dir, self.output_dir],
        }
        try:
            subprocess.run(cmd_map.get(self.method, cmd_map["serial"]), check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"[Worker] {e}")
        self.done.emit()


# ══════════════════════════════════════════════════════════════════════════════
#  Split-view image widget  (drag divider to reveal before / after)
# ══════════════════════════════════════════════════════════════════════════════

class SplitViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._orig:     QPixmap | None = None
        self._proc:     QPixmap | None = None
        self._split     = 0.5
        self._dragging  = False
        self.setMinimumSize(480, 380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_original(self, pix: QPixmap):
        self._orig = pix
        self.update()

    def set_processed(self, pix: QPixmap | None):
        self._proc = pix
        self._split = 0.5
        self.update()

    def _fit(self, pix: QPixmap):
        s = pix.scaled(self.width(), self.height(),
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        return s, (self.width() - s.width()) // 2, (self.height() - s.height()) // 2

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PANEL))

        if not self._orig:
            self._draw_placeholder(p, w, h)
            return

        sx  = int(w * self._split)
        so, ox, oy = self._fit(self._orig)

        if self._proc:
            sp, px2, py2 = self._fit(self._proc)

            # processed LEFT  |  original RIGHT
            p.setClipRect(0, 0, sx, h)
            p.drawPixmap(px2, py2, sp)
            p.setClipRect(sx, 0, w - sx, h)
            p.drawPixmap(ox, oy, so)
            p.setClipping(False)

            self._draw_badge(p, "PROCESSED", 12, 12, left=True)
            self._draw_badge(p, "ORIGINAL",  w - 12, 12, left=False)

            # divider
            p.setPen(QPen(QColor(ACCENT), 1))
            p.drawLine(sx, 0, sx, h)

            # handle circle
            hx, hy = sx, h // 2
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(ACCENT))
            p.drawEllipse(hx - 13, hy - 13, 26, 26)

            # chevrons
            pen2 = QPen(QColor(BG), 2, Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen2)
            lp = QPainterPath()
            lp.moveTo(hx - 4, hy - 5); lp.lineTo(hx - 8, hy); lp.lineTo(hx - 4, hy + 5)
            p.drawPath(lp)
            rp = QPainterPath()
            rp.moveTo(hx + 4, hy - 5); rp.lineTo(hx + 8, hy); rp.lineTo(hx + 4, hy + 5)
            p.drawPath(rp)
        else:
            p.drawPixmap(ox, oy, so)
            self._draw_badge(p, "ORIGINAL", 12, 12, left=True)
            self._draw_no_proc(p, w, h)

    def _draw_badge(self, p, text, x, y, left=True):
        font = QFont("IBM Plex Mono", 8)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        p.setFont(font)
        tw  = p.fontMetrics().horizontalAdvance(text)
        pad = 8
        bw  = tw + pad * 2
        bx  = x if left else x - bw
        p.fillRect(bx, y, bw, 20, QColor(0, 0, 0, 160))
        p.setPen(QColor(MUTED))
        p.drawText(bx + pad, y + 13, text)

    def _draw_placeholder(self, p, w, h):
        grad = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0.0, QColor("#0d1a1f"))
        grad.setColorAt(1.0, QColor("#0a1520"))
        p.fillRect(0, 0, w, h, grad)
        font = QFont("IBM Plex Mono", 10)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        p.setFont(font)
        p.setPen(QColor(MUTED))
        p.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "NO IMAGE LOADED")

    def _draw_no_proc(self, p, w, h):
        font = QFont("IBM Plex Mono", 9)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        p.setFont(font)
        p.setPen(QColor(MUTED))
        p.drawText(QRect(0, h - 34, w, 24),
                   Qt.AlignmentFlag.AlignCenter, "RUN PIPELINE TO SEE RESULT")

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton and self._proc:
            self._dragging = True

    def mouseReleaseEvent(self, _):
        self._dragging = False

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._dragging and self._proc:
            self._split = max(0.02, min(0.98, e.position().x() / self.width()))
            self.update()
        if self._proc:
            hx = int(self.width() * self._split)
            if abs(e.position().x() - hx) < 20:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)


# ══════════════════════════════════════════════════════════════════════════════
#  Scan-line progress widget
# ══════════════════════════════════════════════════════════════════════════════

class ScanBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(3)
        self._progress = 0.0
        self._scan     = 0.0
        self._active   = False
        self._timer    = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._active = True; self._scan = 0.0; self._timer.start()

    def stop(self):
        self._active = False; self._timer.stop(); self.update()

    def set_progress(self, v: float):
        self._progress = max(0.0, min(1.0, v)); self.update()

    def _tick(self):
        self._scan = (self._scan + 0.006) % 1.0; self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(SURFACE))
        filled = int(w * self._progress)
        if filled:
            p.fillRect(0, 0, filled, h, QColor(ACCENT_DK))
        if self._active:
            sx   = int(self._scan * w)
            grad = QLinearGradient(max(0, sx - 140), 0, sx + 6, 0)
            grad.setColorAt(0.0, QColor(0, 212, 255, 0))
            grad.setColorAt(0.75, QColor(0, 212, 255, 120))
            grad.setColorAt(1.0,  QColor(0, 212, 255, 255))
            p.fillRect(0, 0, sx + 6, h, grad)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _section(text: str) -> QLabel:
    l = QLabel(text.upper()); l.setObjectName("section"); return l

def _divider() -> QFrame:
    f = QFrame(); f.setObjectName("h_divider"); return f

def _card() -> QFrame:
    f = QFrame(); f.setObjectName("card"); return f


# ══════════════════════════════════════════════════════════════════════════════
#  Main window
# ══════════════════════════════════════════════════════════════════════════════

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ParallelDerm")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(STYLESHEET)

        self.image_list:  list[str] = []
        self.input_dir   = "images"
        self.output_dir  = "processed_images"
        self.start_time  = 0.0
        self.worker: Worker | None = None
        self._benchmark: dict[str, float] = {}

        self._build_ui()
        self._load_images()
        self._refresh_slider()

        self.poll_timer  = QTimer(self); self.poll_timer.setInterval(350)
        self.poll_timer.timeout.connect(self._poll)
        self.clock_timer = QTimer(self); self.clock_timer.setInterval(80)
        self.clock_timer.timeout.connect(self._tick_clock)

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        self.scan_bar = ScanBar()
        root.addWidget(self.scan_bar)

        body_w = QWidget()
        body   = QHBoxLayout(body_w)
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(14)
        body.addWidget(self._build_sidebar(), stretch=0)
        body.addWidget(self._build_viewer(),  stretch=1)
        root.addWidget(body_w, stretch=1)

        root.addWidget(_divider())
        root.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(f"QFrame {{ background: {PANEL}; border: none;"
                          f" border-bottom: 1px solid {BORDER}; }}")
        h = QHBoxLayout(bar); h.setContentsMargins(20, 12, 20, 12)
        dot = QLabel("●"); dot.setStyleSheet(f"color: {ACCENT}; font-size: 10px;")
        title = QLabel("PARALLELDERM"); title.setObjectName("app_title")
        sub   = QLabel("HAIR REMOVAL PREPROCESSING PIPELINE"); sub.setObjectName("app_sub")
        h.addWidget(dot); h.addSpacing(8)
        h.addWidget(title); h.addSpacing(16); h.addWidget(sub)
        h.addStretch()
        self.status_badge = QLabel("IDLE"); self.status_badge.setObjectName("status_badge")
        self._set_badge("IDLE", MUTED)
        h.addWidget(self.status_badge)
        return bar

    def _build_sidebar(self) -> QWidget:
        sb = QWidget(); sb.setFixedWidth(220)
        v  = QVBoxLayout(sb); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(12)

        # controls card
        ctrl = _card()
        cv   = QVBoxLayout(ctrl); cv.setContentsMargins(14, 14, 14, 14); cv.setSpacing(10)
        cv.addWidget(_section("method"))
        self.method_box = QComboBox()
        self.method_box.addItems(["Serial", "OMP", "MPI", "OCL"])
        self.method_box.currentTextChanged.connect(self._show_image)
        cv.addWidget(self.method_box)
        cv.addWidget(_section("workers"))
        self.workers_box = QComboBox()
        self.workers_box.addItems(["1", "2", "4", "8", "16"])
        self.workers_box.setCurrentText("4")
        cv.addWidget(self.workers_box)
        cv.addSpacing(4)
        self.run_btn = QPushButton("RUN"); self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self._toggle_run)
        cv.addWidget(self.run_btn)
        v.addWidget(ctrl)

        # metrics card
        met = _card()
        mv  = QVBoxLayout(met); mv.setContentsMargins(14, 14, 14, 14); mv.setSpacing(8)
        mv.addWidget(_section("elapsed"))
        self.time_label = QLabel("00:00.000")
        self.time_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 22px; font-weight: 700;"
            f" font-family: 'IBM Plex Mono', 'Courier New', monospace;")
        mv.addWidget(self.time_label)
        mv.addWidget(_divider())
        self.done_val = self._add_metric(mv, "0 / 0", "IMAGES DONE")
        mv.addWidget(_divider())
        self.rate_val = self._add_metric(mv, "—",     "IMG / SEC")
        mv.addWidget(_divider())
        self.prog_val = self._add_metric(mv, "0%",    "PROGRESS")
        v.addWidget(met)

        # benchmark card
        bm = _card()
        bv = QVBoxLayout(bm); bv.setContentsMargins(14, 14, 14, 14); bv.setSpacing(8)
        bv.addWidget(_section("benchmark"))
        self.bench_labels: dict[str, QLabel] = {}
        for m in ["Serial", "OMP", "MPI", "OCL"]:
            row = QHBoxLayout(); row.setSpacing(6)
            dot  = QLabel("●"); dot.setStyleSheet(f"color: {METHOD_COL[m]}; font-size: 9px;")
            name = QLabel(m);   name.setStyleSheet(f"color: {MUTED}; font-size: 10px; letter-spacing: 1px;")
            val  = QLabel("—"); val.setStyleSheet(f"color: {TEXT}; font-size: 10px;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.bench_labels[m] = val
            row.addWidget(dot); row.addWidget(name); row.addStretch(); row.addWidget(val)
            bv.addLayout(row)
        v.addWidget(bm)

        v.addStretch()
        return sb

    def _build_viewer(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(10)
        self.viewer = SplitViewer()
        v.addWidget(self.viewer, stretch=1)

        nav = QHBoxLayout(); nav.setSpacing(10)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._show_image)
        self.nav_label = QLabel("0 / 0")
        self.nav_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self.nav_label.setFixedWidth(56)
        self.nav_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        prev_btn = QPushButton("‹"); prev_btn.setFixedWidth(32)
        next_btn = QPushButton("›"); next_btn.setFixedWidth(32)
        prev_btn.clicked.connect(lambda: self.slider.setValue(max(0, self.slider.value() - 1)))
        next_btn.clicked.connect(lambda: self.slider.setValue(
            min(self.slider.maximum(), self.slider.value() + 1)))
        nav.addWidget(prev_btn)
        nav.addWidget(self.slider, stretch=1)
        nav.addWidget(next_btn)
        nav.addWidget(self.nav_label)
        v.addLayout(nav)
        return w

    def _build_footer(self) -> QWidget:
        bar = QWidget(); bar.setStyleSheet(f"background: {PANEL};")
        h   = QHBoxLayout(bar); h.setContentsMargins(20, 8, 20, 8); h.setSpacing(10)
        self.in_path  = QLabel(self.input_dir);  self.in_path.setObjectName("path_display")
        self.out_path = QLabel(self.output_dir); self.out_path.setObjectName("path_display")
        self.in_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.out_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_in  = QPushButton("Browse"); btn_in.clicked.connect(self._pick_input)
        btn_out = QPushButton("Browse"); btn_out.clicked.connect(self._pick_output)
        h.addWidget(_section("input"));  h.addWidget(self.in_path);  h.addWidget(btn_in)
        h.addSpacing(16)
        h.addWidget(_section("output")); h.addWidget(self.out_path); h.addWidget(btn_out)
        h.addStretch()
        self.img_count_label = QLabel(); self.img_count_label.setObjectName("section")
        h.addWidget(self.img_count_label)
        return bar

    def _add_metric(self, layout, val: str, key: str) -> QLabel:
        v = QLabel(val); v.setObjectName("metric_val")
        k = QLabel(key); k.setObjectName("metric_key")
        layout.addWidget(v); layout.addWidget(k)
        return v

    # ── image loading ─────────────────────────────────────────────────────────

    def _load_images(self):
        d = Path(self.input_dir)
        self.image_list = sorted(
            p.name for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS
        ) if d.exists() else []

    def _refresh_slider(self):
        n = len(self.image_list)
        self.slider.setMaximum(max(0, n - 1))
        self.nav_label.setText(f"1 / {n}" if n else "0 / 0")
        self.img_count_label.setText(f"{n} IMAGES" if n else "NO IMAGES")
        if n:
            self._show_image()

    # ── display ───────────────────────────────────────────────────────────────

    def _show_image(self):
        n = len(self.image_list)
        if not n:
            return
        idx  = self.slider.value()
        name = self.image_list[idx]
        self.nav_label.setText(f"{idx + 1} / {n}")

        orig = Path(self.input_dir) / name
        if orig.exists():
            self.viewer.set_original(QPixmap(str(orig)))

        method    = self.method_box.currentText().lower()
        proc      = Path(self.output_dir) / method / f"hair_removed_{name}"
        self.viewer.set_processed(QPixmap(str(proc)) if proc.exists() else None)

    # ── run / stop ────────────────────────────────────────────────────────────

    def _toggle_run(self):
        if self.poll_timer.isActive():
            self._stop()
        else:
            self._start()

    def _start(self):
        if not self.image_list:
            return
        method     = self.method_box.currentText().lower()
        out_folder = Path(self.output_dir) / method
        out_folder.mkdir(parents=True, exist_ok=True)
        for f in out_folder.glob("*"):
            if f.suffix.lower() in IMAGE_EXTS:
                f.unlink()

        n = len(self.image_list)
        self.done_val.setText(f"0 / {n}"); self.rate_val.setText("—"); self.prog_val.setText("0%")
        self.run_btn.setText("STOP")
        self._set_badge("RUNNING", "#f5a623")
        self.scan_bar.set_progress(0); self.scan_bar.start()
        self.start_time = time.perf_counter()

        self.worker = Worker(
            method    = self.method_box.currentText(),
            input_dir = self.input_dir,
            output_dir= str(out_folder),
            workers   = int(self.workers_box.currentText()),
        )
        self.worker.done.connect(self._on_worker_done)
        self.worker.start()
        self.poll_timer.start(); self.clock_timer.start()

    def _stop(self):
        self.poll_timer.stop(); self.clock_timer.stop(); self.scan_bar.stop()
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self.run_btn.setText("RUN"); self._set_badge("STOPPED", MUTED)

    def _on_worker_done(self):
        self._poll(force=True)
        self.poll_timer.stop(); self.clock_timer.stop(); self.scan_bar.stop()
        self.run_btn.setText("RUN"); self._set_badge("DONE", SUCCESS)
        elapsed = time.perf_counter() - self.start_time
        self._benchmark[self.method_box.currentText()] = elapsed
        self._update_bench_table()
        self._show_image()

    # ── poll ──────────────────────────────────────────────────────────────────

    def _poll(self, force=False):
        method     = self.method_box.currentText().lower()
        out_folder = Path(self.output_dir) / method
        done  = len([p for p in out_folder.glob("*") if p.suffix.lower() in IMAGE_EXTS])
        total = len(self.image_list)
        pct   = done / total if total else 0

        self.scan_bar.set_progress(pct)
        self.done_val.setText(f"{done} / {total}")
        self.prog_val.setText(f"{int(pct * 100)}%")
        elapsed = time.perf_counter() - self.start_time
        if elapsed > 0 and done > 0:
            self.rate_val.setText(f"{done / elapsed:.1f}")
        if not force and done >= total:
            self._on_worker_done()

    def _tick_clock(self):
        e = time.perf_counter() - self.start_time
        self.time_label.setText(f"{int(e // 60):02d}:{e % 60:06.3f}")

    # ── benchmark ─────────────────────────────────────────────────────────────

    def _update_bench_table(self):
        baseline = self._benchmark.get("Serial")
        for m, lbl in self.bench_labels.items():
            if m in self._benchmark:
                t = self._benchmark[m]
                lbl.setText(f"{t:.2f}s  ×{baseline/t:.1f}" if baseline and m != "Serial"
                            else f"{t:.2f}s")
            else:
                lbl.setText("—")

    # ── folder pickers ────────────────────────────────────────────────────────

    def _pick_input(self):
        d = QFileDialog.getExistingDirectory(self, "Input folder", self.input_dir)
        if d:
            self.input_dir = d; self.in_path.setText(d)
            self._load_images(); self._refresh_slider()

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.output_dir)
        if d:
            self.output_dir = d; self.out_path.setText(d)

    # ── badge ─────────────────────────────────────────────────────────────────

    def _set_badge(self, text: str, color: str):
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(
            f"QLabel#status_badge {{ background: transparent; color: {color};"
            f" border: 1px solid {color}; border-radius: 3px;"
            f" padding: 3px 10px; font-size: 10px; letter-spacing: 1px; font-weight: 700; }}"
        )


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = GUI()
    win.show()
    sys.exit(app.exec())