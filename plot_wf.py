import pandas as pd
import os
import matplotlib.pyplot as plt
from obspy import read
from obspy.core.utcdatetime import UTCDateTime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CSV_FILE = 'all_event_picks.csv'  

def parse_time(time_str):
    if pd.isna(time_str) or not time_str or str(time_str).lower() == 'nan':
        return None
    try:
        return UTCDateTime(time_str)
    except Exception as e:
        print(f"  [Peringatan] Tidak dapat mem-parsing waktu '{time_str}': {e}")
        return None

def plot_picks_from_csv(csv_path):
  
    # 1. Muat file CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERR] File CSV tidak ditemukan di: {csv_path}")
        return
        
    print(f"[INFO] Ditemukan {len(df)} entri di {csv_path}")

    for index, row in df.iterrows():

        mseed_path = row['file_path']
        ai_pick_str = row['pick_time']
        probability = row['probability']


        print(f"\n[PROSES] {index+1}/{len(df)}: {mseed_path}")

        if not os.path.exists(mseed_path):
            mseed_path_fixed = mseed_path.replace("\\\\", "\\")
            if not os.path.exists(mseed_path_fixed):
                print(f"  [ERR] File mseed tidak ada di '{mseed_path}' atau '{mseed_path_fixed}'. Dilewati.")
                continue
            else:
                mseed_path = mseed_path_fixed
            
        ai_time = parse_time(ai_pick_str)

        try:
            st = read(mseed_path)
            if len(st) == 0:
                print(f"  [ERR] Stream kosong. Dilewati.")
                continue
            st.detrend("demean")
            st.merge(method=1, fill_value="interpolate")
        except Exception as e:
            print(f"  [ERR] Gagal membaca stream: {e}. Dilewati.")
            continue

        fig = plt.figure(figsize=(20, 10))
        
        try:
            st.plot(fig=fig, show=False)
        except Exception as e:
            print(f"  [ERR] Gagal mem-plot stream: {e}. Dilewati.")
            plt.close(fig)
            continue
            
        base_name = os.path.basename(mseed_path)
        axes = fig.get_axes() 
        has_picks = False
        
        for ax in axes:
            if ai_time:
                ax.axvline(ai_time.matplotlib_date, color="red", linestyle="--", label="AI Pick")
                has_picks = True
        if axes:
            prob_str = f" (Prob: {probability:.2f})" if ai_time and pd.notna(probability) else ""
            title = f"{base_name}{prob_str}"
            axes[0].set_title(title)
            if has_picks:
                handles, labels = axes[0].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes[0].legend(by_label.values(), by_label.keys(), loc="upper right")
        output_dir = os.path.dirname(mseed_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        output_png = os.path.join(output_dir, f"{base_name_no_ext}_picks.png")
        try:
            plt.savefig(output_png, bbox_inches='tight', dpi=100)
            print(f"  [OK] Plot disimpan ke {output_png}")
        except Exception as e:
            print(f"  [ERR] Gagal menyimpan PNG: {e}")
        
        plt.close(fig)

    print("\n[SELESAI] Pembuatan plot selesai.")

if __name__ == "__main__":
    plot_picks_from_csv(CSV_FILE)