import sys
import os
import glob
import warnings
import numpy as np

from obspy import read
from obspy.core.utcdatetime import UTCDateTime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure

import matplotlib
matplotlib.use("Qt5Agg")  # ensure Qt backend
import matplotlib.pyplot as plt

import csv
import seisbench.models as sbm

warnings.filterwarnings("ignore", category=UserWarning, module="seisbench")

LOCAL_MSEED_ROOT = r"./20241002010918_-7.35_106.49"
OUTPUT_CSV = "output_pickscoba.csv" #nama output csv

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11
})

print("[INFO] Loading PhaseNet model…", flush=True)
AI_MODEL = sbm.PhaseNet.from_pretrained("stead")
AI_SR = AI_MODEL.sampling_rate
print(f"[INFO] PhaseNet ready (sampling_rate={AI_SR})", flush=True)

def list_all_mseed_local(root_dir):
    pattern1 = os.path.join(root_dir, "**", "*.mseed")
    files = sorted(set(glob.glob(pattern1, recursive=True)))
    print(f"[INFO] Discovered {len(files)} MiniSEED files under: {root_dir}")
    return files


def load_stream_local(filepath):
    try:
        st = read(filepath)
    except Exception as e:
        print(f"[ERR] Read error: {filepath} -> {e}")
        return None

    if len(st) == 0:
        print(f"[WARN] Empty stream: {filepath}")
        return None

    # Require minimum ~30s per trace (common for PhaseNet inputs)
    for tr in st:
        dur = tr.stats.npts / tr.stats.sampling_rate
        if dur < 30.0:
            print(f"[WARN] Short trace ({dur:.1f}s): {filepath}")
            return None

    try:
        st.detrend("demean").detrend("linear")
        st.resample(AI_SR)
        st.merge(method=1, fill_value="interpolate")
    except Exception as e:
        print(f"[ERR] Preprocess error {filepath}: {e}")
        return None

    tr0 = st[0]
    print(f"[OK] Stream ready: {os.path.basename(filepath)} | start={tr0.stats.starttime} "
          f"len={len(st)} traces fs={tr0.stats.sampling_rate}")
    return st


def ai_pick_time_and_prob(st, filepath):
    print(f"[AI ] Running PhaseNet classify on {os.path.basename(filepath)}…")
    try:
        result = AI_MODEL.classify(st)
    except Exception as e:
        print(f"[ERR] AI classify error: {filepath} -> {e}")
        return None, None

    picks = result.picks.select(phase="P")
    if len(picks) == 0:
        print("[AI ] No P picks found.")
        return None, None

    # Get times/probs
    times, probs = [], []
    try:
        df = picks.to_dataframe()
        for _, row in df.iterrows():
            times.append(UTCDateTime(row["time"]))
            probs.append(float(row.get("probability", np.nan)))
    except Exception:
        for p in picks:
            times.append(UTCDateTime(p.start_time))
            probs.append(np.nan)

    if not all(np.isnan(probs)):
        idx = int(np.nanargmax(probs))
        chosen_time = times[idx]
        chosen_prob = probs[idx] if not np.isnan(probs[idx]) else None
    else:
        chosen_time = times[0]
        chosen_prob = None

    print(f"[AI ] Selected pick: time={chosen_time} prob={chosen_prob}")
    return chosen_time, chosen_prob


def ensure_csv_header(path_csv):
    new_file = not os.path.exists(path_csv)
    if new_file:
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_name", "pick_time_AI", "pick_time_manual"])
        print(f"[INFO] Created CSV with header: {path_csv}")


def append_csv_row(path_csv, file_name, ai_time, manual_time):
    ai_iso = ai_time.isoformat() if ai_time is not None else ""
    manual_iso = manual_time.isoformat() if manual_time is not None else ""
    with open(path_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([file_name, ai_iso, manual_iso])
    print(f"[SAVE] {file_name}, {ai_iso}, {manual_iso}")


class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(20, 10))
        self.axes = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]
        super().__init__(fig)
        self.setParent(parent)

        self.manual_pick = None
        self._st = None
        self._ai = None
        self._present = []
        self._mpl_cid = self.mpl_connect("button_press_event", self.on_click)

    def _detect_present_components(self, st):
        for tr in st:
            if tr.stats.channel[-1].upper() == "Z":
                return ["Z"]
        return [st[0].stats.channel[-1].upper()] if len(st) else []

    def plot_stream(self, st, ai_time=None, title=""):
        self._st = st
        self._ai = ai_time
        self.manual_pick = None

        self._present = self._detect_present_components(st)

        for ax in self.axes:
            ax.clear()
            ax.set_visible(False)

        used_axes = self.axes[:len(self._present)]
        for ax in used_axes:
            ax.set_visible(True)

        for idx, comp in enumerate(self._present):
            ax = used_axes[idx]
            comp_traces = [tr for tr in st if tr.stats.channel[-1].upper() == comp]
            if not comp_traces:
                continue
            tr = comp_traces[0]
            t = tr.times("matplotlib")
            ax.plot_date(t, tr.data, "-",color="black", label=tr.id, linewidth=0.8)

            if ai_time is not None:
                ax.axvline(ai_time.matplotlib_date, color="red", linestyle="--", label="AI")
            if self.manual_pick is not None:
                ax.axvline(self.manual_pick.matplotlib_date, color="green", linestyle="-", label="Manual")

            ax.set_ylabel(comp)
            ax.legend(loc="upper right")

        if used_axes:
            used_axes[0].set_title(title)
        for ax in used_axes:
            ax.label_outer()

        self.draw()

    def redraw_with_manual(self):
        if self._st is None:
            return

        if not self._present:
            self._present = self._detect_present_components(self._st)

        for ax in self.axes:
            ax.clear()
            ax.set_visible(False)

        used_axes = self.axes[:len(self._present)]
        for ax in used_axes:
            ax.set_visible(True)

        for idx, comp in enumerate(self._present):
            ax = used_axes[idx]
            comp_traces = [tr for tr in self._st if tr.stats.channel[-1].upper() == comp]
            if not comp_traces:
                continue
            tr = comp_traces[0]
            t = tr.times("matplotlib")
            ax.plot_date(t, tr.data, "-", label=tr.id, linewidth=0.8)

            if self._ai is not None:
                ax.axvline(self._ai.matplotlib_date, color="red", linestyle="--", label="AI")
            if self.manual_pick is not None:
                ax.axvline(self.manual_pick.matplotlib_date, color="green", linestyle="-", label="Manual")

            ax.set_ylabel(comp)
            ax.legend(loc="upper right")

        if used_axes:
            used_axes[0].set_title(used_axes[0].get_title())
        for ax in used_axes:
            ax.label_outer()

        self.draw()

    def on_click(self, event):
        # Left click: set manual pick; Right click: clear
        if event.inaxes not in self.axes:
            return
        if self._st is None:
            return
        if event.xdata is None:
            return

        tr_ref = None
        for tr in self._st:
            if tr.stats.channel[-1].upper() == "Z":
                tr_ref = tr
                break
        if tr_ref is None:
            tr_ref = self._st[0]

        if event.button == 1:
            t0 = tr_ref.stats.starttime.matplotlib_date
            picked = tr_ref.stats.starttime + (event.xdata - t0) * 86400.0
            self.manual_pick = picked
            print(f"[PICK] Manual set to {self.manual_pick.isoformat()}")
            self.redraw_with_manual()
        elif event.button == 3:
            print("[PICK] Manual cleared")
            self.manual_pick = None
            self.redraw_with_manual()


class JLabelSafe(QLabel):
    def setText(self, text):
        super().setText(text if text is not None else "")


class PickingGUI(QWidget):
    def __init__(self, local_root):
        super().__init__()
        self.setWindowTitle("Picking GUI (Local MiniSEED)")
        self.local_root = local_root
        print(f"[BOOT] Scanning local MSEED root: {self.local_root}")
        self.files = list_all_mseed_local(self.local_root) 
        self.idx = 0

        self.canvas = WaveformCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.lbl = JLabelSafe("")

        btn_prev = QPushButton("Prev")
        btn_next = QPushButton("Next")
        btn_save = QPushButton("Save")

        btn_prev.clicked.connect(self.on_prev)
        btn_next.clicked.connect(self.on_next)
        btn_save.clicked.connect(self.on_save)

        v = QVBoxLayout()
        v.addWidget(self.canvas)
        v.addWidget(self.toolbar)
        v.addWidget(self.lbl)

        h = QHBoxLayout()
        h.addWidget(btn_prev)
        h.addWidget(btn_save)
        h.addWidget(btn_next)
        v.addLayout(h)

        self.setLayout(v)
        self.load_current()

    def load_current(self):
        if not self.files:
            msg = f"No .mseed under {self.local_root}"
            print(f"[INFO] {msg}")
            self.lbl.setText(msg)
            return

        self.idx = max(0, min(self.idx, len(self.files) - 1))
        fp = self.files[self.idx]
        print(f"[LOAD] {self.idx+1}/{len(self.files)} | file={fp}")

        st = load_stream_local(fp)
        if st is None:
            self.canvas.plot_stream(read(fp), ai_time=None, title=f"{os.path.basename(fp)} (stream issue)")
            self.lbl.setText(f"{self.idx+1}/{len(self.files)} | {fp} (stream issue)")
            self._current_file = fp
            self._current_ai_time = None
            return

        ai_time, _ = ai_pick_time_and_prob(st, fp)

        title = f"{os.path.basename(fp)}"
        self.canvas.plot_stream(st, ai_time=ai_time, title=title)
        self.lbl.setText(f"{self.idx+1}/{len(self.files)} | {fp}")

        self._current_file = fp
        self._current_ai_time = ai_time

    def on_prev(self):
        if self.idx > 0:
            self.save_current_line()
            self.idx -= 1
            self.load_current()

    def on_next(self):
        self.save_current_line()
        if self.idx < len(self.files) - 1:
            self.idx += 1
            self.load_current()
        else:
            self.lbl.setText(self.lbl.text() + " | Finished.")
            print("[DONE] Reached last file.")

    def on_save(self):
        self.save_current_line()
        self.lbl.setText(self.lbl.text() + " | Saved.")

    def save_current_line(self):
        if not hasattr(self, "_current_file"):
            return
        file_name = self._current_file
        ai_time = self._current_ai_time
        manual_time = self.canvas.manual_pick
        append_csv_row(OUTPUT_CSV, file_name, ai_time, manual_time)

def main():
    ensure_csv_header(OUTPUT_CSV)

    if not os.path.isdir(LOCAL_MSEED_ROOT):
        print(f"[ERR] Folder does not exist: {LOCAL_MSEED_ROOT}")
        sys.exit(1)

    app = QApplication(sys.argv)
    gui = PickingGUI(LOCAL_MSEED_ROOT)
    gui.resize(1200, 800)
    gui.show()
    rc = app.exec_()
    sys.exit(rc)


if __name__ == "__main__":
    main()
