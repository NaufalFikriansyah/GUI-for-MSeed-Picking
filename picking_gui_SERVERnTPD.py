import sys
import os
import glob
import warnings
import numpy as np
from io import BytesIO
import matplotlib as plt
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
import seisbench.models as sbm
import paramiko

warnings.filterwarnings("ignore", category=UserWarning, module="seisbench")

# =========================
# Konfigurasi SSH & umum
# =========================
# --- Pilih mode koneksi ---
USE_DOUBLE_HOP = True  # True: 202.xx -> 172.xx:11 ; False: langsung 202.xx

# Hop 1 (bastion / publik)
BASTION_HOST = "202.90.199.206"
BASTION_PORT = 2025
BASTION_USER = "sysop"
BASTION_PASS = "m4n0p#1!"

# Hop 2 (host internal)
INTERNAL_HOST = "172.19.3.128"
INTERNAL_PORT = 2107
INTERNAL_USER = "root"
INTERNAL_PASS = "Root2107#"

# Root di SERVER (pakai leading slash ✔)
REMOTE_MSEED_ROOT = "/opt/earthworm/run_working/adj_tpd/mseed"

# Output ditulis LOKAL
OUTPUT_FILE = "./output_coba.txt"

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 13
})

# Model AI
print("[INFO] Loading PhaseNet model…", flush=True)
AI_MODEL = sbm.PhaseNet.from_pretrained("stead")
AI_SR = AI_MODEL.sampling_rate
TPD_TOL_SEC = 11.0
print(f"[INFO] PhaseNet ready (sampling_rate={AI_SR})", flush=True)

# =========================
# Konektor Remote (SSH/SFTP)
# =========================
class RemoteFS:
    def __init__(self):
        self.bastion = None
        self.internal_transport = None
        self.sftp = None

    def connect(self):
        try:
            print(f"[SSH] Connecting to bastion {BASTION_HOST}:{BASTION_PORT} as {BASTION_USER}…", flush=True)
            self.bastion = paramiko.SSHClient()
            self.bastion.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.bastion.connect(
                BASTION_HOST, port=BASTION_PORT,
                username=BASTION_USER, password=BASTION_PASS,
                look_for_keys=False, allow_agent=False, timeout=20
            )
            print("[SSH] Bastion connected.", flush=True)

            if USE_DOUBLE_HOP:
                print(f"[SSH] Opening channel to internal {INTERNAL_HOST}:{INTERNAL_PORT} via bastion…", flush=True)
                chan = self.bastion.get_transport().open_channel(
                    "direct-tcpip", (INTERNAL_HOST, INTERNAL_PORT), ("127.0.0.1", 0)
                )
                print("[SSH] Channel opened. Starting secondary transport…", flush=True)
                self.internal_transport = paramiko.Transport(chan)
                self.internal_transport.start_client(timeout=20)
                self.internal_transport.auth_password(INTERNAL_USER, INTERNAL_PASS)
                print("[SSH] Authenticated to internal host.", flush=True)
                self.sftp = paramiko.SFTPClient.from_transport(self.internal_transport)
                print("[SFTP] SFTP ready on internal host.", flush=True)
            else:
                print("[SSH] Single-hop mode: opening SFTP on bastion…", flush=True)
                self.sftp = self.bastion.open_sftp()
                print("[SFTP] SFTP ready on bastion.", flush=True)

        except Exception as e:
            print("[ERR] SSH/SFTP connect failed:", e, flush=True)
            traceback.print_exc()
            raise

    def listdir(self, path):
        print(f"[SFTP] listdir: {path}", flush=True)
        return self.sftp.listdir(path)

    def isdir(self, path):
        try:
            self.sftp.listdir(path)
            return True
        except Exception:
            return False

    def open_bytes(self, path):
        print(f"[SFTP] open: {path}", flush=True)
        with self.sftp.open(path, "rb") as f:
            data = f.read()
        print(f"[SFTP] read {len(data)} bytes from {path}", flush=True)
        return data

    def close(self):
        print("[SSH] Closing connections…", flush=True)
        try:
            if self.sftp:
                self.sftp.close()
        except Exception:
            pass
        try:
            if self.internal_transport:
                self.internal_transport.close()
        except Exception:
            pass
        try:
            if self.bastion:
                self.bastion.close()
        except Exception:
            pass
        print("[SSH] Closed.", flush=True)

remote = RemoteFS()
remote.connect()

# =========================
# Utilitas remote
# =========================
def list_all_traces_remote(root_dir):
    items = []
    try:
        stations = sorted(remote.listdir(root_dir))
        print(f"[INFO] Found {len(stations)} station folder(s) under {root_dir}", flush=True)
        for station in stations:
            st_dir = f"{root_dir}/{station}"
            if not remote.isdir(st_dir):
                print(f"[WARN] Not a directory: {st_dir}", flush=True)
                continue
            names = sorted(remote.listdir(st_dir))
            mseed_files = [n for n in names if n.lower().endswith(".mseed")]
            print(f"[INFO] Station {station}: {len(mseed_files)} mseed file(s)", flush=True)
            for name in mseed_files:
                items.append((station, f"{st_dir}/{name}"))
        print(f"[INFO] Total MSEED files discovered: {len(items)}", flush=True)
    except Exception as e:
        print(f"[ERR] Remote listing error: {e}", flush=True)
        traceback.print_exc()
    return items

def extract_tpd_parts_from_filename(remote_filepath):
    base = os.path.basename(remote_filepath)
    name, _ = os.path.splitext(base)
    try:
        tpd_epoch_str = name.split("_")[-1]
        tpd_dt = UTCDateTime(float(tpd_epoch_str))
        return tpd_epoch_str, tpd_dt
    except Exception:
        return "", None

def load_stream_remote(filepath):
    try:
        raw = remote.open_bytes(filepath)
        print(f"[INFO] Parsing MSEED with ObsPy: {os.path.basename(filepath)}", flush=True)
        st = read(BytesIO(raw))
    except Exception as e:
        print(f"[ERR] Read error: {filepath} -> {e}", flush=True)
        traceback.print_exc()
        return None

    if len(st) == 0:
        print(f"[WARN] Empty stream: {filepath}", flush=True)
        return None

    # Durasi minimum 30 s
    for tr in st:
        dur = tr.stats.npts / tr.stats.sampling_rate
        if dur < 30.0:
            print(f"[WARN] Short trace ({dur:.1f}s): {filepath}", flush=True)
            return None

    try:
        st.detrend("demean").detrend("linear")
        st.resample(AI_SR)
        st.merge(method=1, fill_value="interpolate")
    except Exception as e:
        print(f"[ERR] Preprocess error {filepath}: {e}", flush=True)
        traceback.print_exc()
        return None

    # Ringkas info stream
    try:
        tr0 = st[0]
        print(f"[OK] Stream ready: {os.path.basename(filepath)} | start={tr0.stats.starttime} "
              f"len={len(st)} traces fs={tr0.stats.sampling_rate}", flush=True)
    except Exception:
        pass
    return st

def ai_pick_time_and_prob(st, filepath):
    print(f"[AI ] Running PhaseNet classify on {os.path.basename(filepath)}…", flush=True)
    try:
        result = AI_MODEL.classify(st)
    except Exception as e:
        print(f"[ERR] AI classify error: {filepath} -> {e}", flush=True)
        traceback.print_exc()
        return None, None

    picks = result.picks.select(phase="P")
    print(f"[AI ] Found {len(picks)} P-pick(s) before selection", flush=True)
    if len(picks) == 0:
        return None, None

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

    tpd_epoch_str, tpd_dt = extract_tpd_parts_from_filename(filepath)
    chosen_time, chosen_prob = None, None

    if tpd_dt is not None:
        best_idx, best_dt = None, None
        for i, t in enumerate(times):
            dt = abs(t - tpd_dt)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_idx = i
        print(f"[AI ] Nearest-to-TPd Δt = {best_dt:.3f}s" if best_dt is not None else "[AI ] No Δt", flush=True)
        if best_idx is not None and best_dt is not None and best_dt <= TPD_TOL_SEC:
            chosen_time = times[best_idx]
            chosen_prob = probs[best_idx] if not np.isnan(probs[best_idx]) else None
        else:
            if not all(np.isnan(probs)):
                idx = int(np.nanargmax(probs))
                chosen_time = times[idx]
                chosen_prob = probs[idx] if not np.isnan(probs[idx]) else None
            else:
                chosen_time = times[0]
                chosen_prob = None
    else:
        if not all(np.isnan(probs)):
            idx = int(np.nanargmax(probs))
            chosen_time = times[idx]
            chosen_prob = probs[idx] if not np.isnan(probs[idx]) else None
        else:
            chosen_time = times[0]
            chosen_prob = None

    print(f"[AI ] Selected pick: time={chosen_time} prob={chosen_prob}", flush=True)
    return chosen_time, chosen_prob

def write_output_line(station, tpd_epoch_str, manual_time, ai_time, ai_prob, any_pick):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True) if os.path.dirname(OUTPUT_FILE) else None
    manual_iso = manual_time.isoformat() if manual_time is not None else ""
    ai_iso = ai_time.isoformat() if ai_time is not None else ""
    prob_str = "" if (ai_prob is None or not isinstance(ai_prob, (int, float))) else f"{ai_prob:.4f}"
    valid_str = "true" if any_pick else "false"
    line = f"{station},{tpd_epoch_str},{manual_iso},{ai_iso},{prob_str},{valid_str}\n"
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[SAVE] {line.strip()}", flush=True)

# =========================
# Canvas + GUI
# =========================
class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(21, 12))
        self.axes = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]
        super().__init__(fig)
        self.setParent(parent)
        self.manual_pick = None
        self._st = None
        self._ai = None
        self._ai_prob = None
        self._tpd_dt = None
        self._present = []   # <- komponen yang tersedia pada stream saat ini
        self._mpl_cid = self.mpl_connect("button_press_event", self.on_click)

    def _detect_present_components(self, st):
        """Kembalikan list komponen yang ada, urut: Z, N, E (hanya yang tersedia)."""
        comps = set()
        for tr in st:
            comps.add(tr.stats.channel[-1].upper())
        present = [c for c in ["Z", "N", "E"] if c in comps]
        if not present and len(st) > 0:
            # fallback jika nama channel non-standar
            present = [st[0].stats.channel[-1].upper()]
        return present

    def plot_stream(self, st, ai_time=None, ai_prob=None, tpd_time=None, title=""):
        self._st = st
        self._ai = ai_time
        self._ai_prob = ai_prob
        self._tpd_dt = tpd_time
        self.manual_pick = None

        # Deteksi komponen yang tersedia
        self._present = self._detect_present_components(st)

        # Bersihkan & tampilkan hanya sebanyak komponen yang ada
        for ax in self.axes:
            ax.clear()
            ax.set_visible(False)

        label_by_comp = {"Z": "Z", "N": "N", "E": "E"}
        used_axes = self.axes[:len(self._present)]
        for ax in used_axes:
            ax.set_visible(True)

        # Plot per-komponen yang ada
        for idx, comp in enumerate(self._present):
            ax = used_axes[idx]
            # Ambil semua trace dengan suffix comp
            comp_traces = [tr for tr in st if tr.stats.channel[-1].upper() == comp]
            if not comp_traces:
                continue
            # Ambil satu trace (umumnya satu saja per komponen)
            tr = comp_traces[0]
            t = tr.times("matplotlib")
            ax.plot_date(t, tr.data, "-", label=tr.id, linewidth=0.8)

            if ai_time is not None:
                ax.axvline(ai_time.matplotlib_date, color="red", linestyle="--", label="AI")
            if tpd_time is not None:
                ax.axvline(tpd_time.matplotlib_date, color="blue", linestyle="-", label="TPd")
            if self.manual_pick is not None:
                ax.axvline(self.manual_pick.matplotlib_date, color="green", linestyle="-", label="Manual")

            ax.set_ylabel(label_by_comp.get(comp, comp))
            ax.legend(loc="upper right")

        if used_axes:
            used_axes[0].set_title(title)
        for ax in used_axes:
            ax.label_outer()

        self.draw()

    def redraw_with_manual(self):
        if self._st is None:
            return

        # Pastikan _present sudah diisi saat plot_stream()
        if not self._present:
            self._present = self._detect_present_components(self._st)

        for ax in self.axes:
            ax.clear()
            ax.set_visible(False)

        label_by_comp = {"Z": "Z", "N": "N", "E": "E"}
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
            if self._tpd_dt is not None:
                ax.axvline(self._tpd_dt.matplotlib_date, color="blue", linestyle="-", label="TPd")
            if self.manual_pick is not None:
                ax.axvline(self.manual_pick.matplotlib_date, color="green", linestyle="-", label="Manual")

            ax.set_ylabel(label_by_comp.get(comp, comp))
            ax.legend(loc="upper right")

        if used_axes:
            used_axes[0].set_title(used_axes[0].get_title())
        for ax in used_axes:
            ax.label_outer()

        self.draw()


    def on_click(self, event):
        if event.button == 1 and event.inaxes in self.axes and self._st is not None:
            # referensi trace Z bila ada
            tr_ref = None
            for tr in self._st:
                if tr.stats.channel[-1].upper() == "Z":
                    tr_ref = tr
                    break
            if tr_ref is None:
                tr_ref = self._st[0]
            if event.xdata is None:
                return
            t0 = tr_ref.stats.starttime.matplotlib_date
            picked = tr_ref.stats.starttime + (event.xdata - t0) * 86400.0
            self.manual_pick = picked
            print(f"[PICK] Manual set to {self.manual_pick.isoformat()}", flush=True)
            self.redraw_with_manual()
        elif event.button == 3 and event.inaxes in self.axes:
            print("[PICK] Manual cleared", flush=True)
            self.manual_pick = None
            self.redraw_with_manual()
            tr_ref = None
            for tr in self._st:
                if tr.stats.channel[-1].upper() == "Z":
                    tr_ref = tr
                    break
            if tr_ref is None:
                tr_ref = self._st[0]  

class JLabelSafe(QLabel):
    def setText(self, text):
        super().setText(text if text is not None else "")

class PickingGUI(QWidget):
    def __init__(self, remote_root):
        super().__init__()
        self.setWindowTitle("Picking GUI (Remote SFTP)")
        self.remote_root = remote_root
        print(f"[BOOT] Scanning remote MSEED root: {self.remote_root}", flush=True)
        self.items = list_all_traces_remote(self.remote_root)  # [(station, remote_filepath), ...]
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
        if not self.items:
            msg = f"Tidak ada file .mseed di {self.remote_root}"
            print(f"[INFO] {msg}", flush=True)
            self.lbl.setText(msg)
            return

        self.idx = max(0, min(self.idx, len(self.items) - 1))
        station, fp_remote = self.items[self.idx]
        print(f"[LOAD] {self.idx+1}/{len(self.items)} | station={station} file={fp_remote}", flush=True)

        st = load_stream_remote(fp_remote)
        if st is None:
            try:
                raw = remote.open_bytes(fp_remote)
                self.canvas.plot_stream(read(BytesIO(raw)), ai_time=None, ai_prob=None, tpd_time=None,
                                        title=f"{station} :: {os.path.basename(fp_remote)} (stream issue)")
            except Exception:
                pass
            self.lbl.setText(f"{self.idx+1}/{len(self.items)} | {fp_remote} (stream issue)")
            return

        tpd_epoch_str, tpd_dt = extract_tpd_parts_from_filename(fp_remote)
        print(f"[TPD] station={station} tpd_pick={tpd_epoch_str} (dt={tpd_dt})", flush=True)

        ai_time, ai_prob = ai_pick_time_and_prob(st, fp_remote)

        title = f"{station} :: {os.path.basename(fp_remote)}"
        self.canvas.plot_stream(st, ai_time=ai_time, ai_prob=ai_prob, tpd_time=tpd_dt, title=title)
        self.lbl.setText(f"{self.idx+1}/{len(self.items)} | {fp_remote}")

        self._current_station = station
        self._current_fp_remote = fp_remote
        self._current_tpd_epoch_str = tpd_epoch_str

    def on_prev(self):
        if self.idx > 0:
            self.save_current_line()
            self.idx -= 1
            self.load_current()

    def on_next(self):
        self.save_current_line()
        if self.idx < len(self.items) - 1:
            self.idx += 1
            self.load_current()
        else:
            self.lbl.setText(self.lbl.text() + " | Selesai.")
            print("[DONE] Reached last file.", flush=True)

    def on_save(self):
        self.save_current_line()
        self.lbl.setText(self.lbl.text() + " | Disimpan.")

    def save_current_line(self):
        if not self.items:
            return
        station = getattr(self, "_current_station", None)
        tpd_epoch_str = getattr(self, "_current_tpd_epoch_str", "")
        if station is None:
            station, fp_remote = self.items[self.idx]
            tpd_epoch_str, _ = extract_tpd_parts_from_filename(fp_remote)

        manual_time = self.canvas.manual_pick
        ai_time = self.canvas._ai
        ai_prob = self.canvas._ai_prob
        any_pick = (manual_time is not None) or (ai_time is not None)

        write_output_line(station, tpd_epoch_str, manual_time, ai_time, ai_prob, any_pick)

    def closeEvent(self, event):
        try:
            remote.close()
        except Exception:
            pass
        event.accept()

# =========================
# Main
# =========================
def main():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("station,tpd_pick,manual_pick,pick_ai,probability,valid\n")
        print(f"[INFO] Created header: {OUTPUT_FILE}", flush=True)

    # Smoke test koneksi & root path sebelum GUI
    try:
        print(f"[CHECK] Verifying remote root: {REMOTE_MSEED_ROOT}", flush=True)
        stations = remote.listdir(REMOTE_MSEED_ROOT)
        print(f"[CHECK] Root OK. Found entries: {len(stations)} → {stations[:5]}{' …' if len(stations)>5 else ''}", flush=True)
    except Exception as e:
        print(f"[ERR] Cannot list {REMOTE_MSEED_ROOT}: {e}", flush=True)
        traceback.print_exc()

    app = QApplication(sys.argv)
    gui = PickingGUI(REMOTE_MSEED_ROOT)
    gui.resize(1200, 800)
    gui.show()
    rc = app.exec_()
    sys.exit(rc)

if __name__ == "__main__":
    main()