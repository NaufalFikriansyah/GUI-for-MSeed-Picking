# Picking GUI Local - Manual Phase Picking dengan AI Assistance

Aplikasi GUI berbasis PyQt5 untuk melakukan manual phase picking pada data seismik MiniSEED dengan bantuan AI (PhaseNet). Aplikasi ini memungkinkan user untuk memverifikasi dan memperbaiki hasil picking AI secara interaktif.

## Deskripsi

Aplikasi ini dirancang untuk:
- Membaca file MiniSEED dari folder lokal
- Menjalankan PhaseNet AI model untuk deteksi otomatis fase P
- Menampilkan waveform secara interaktif dengan GUI
- Memungkinkan user melakukan manual picking dengan klik mouse
- Menyimpan hasil picking (AI dan manual) ke file CSV

<img width="1208" height="834" alt="file_2025-11-30_15 18 37" src="https://github.com/user-attachments/assets/113ccf54-8b6d-4905-8896-35b738aec1b9" />
<img width="1210" height="843" alt="file_2025-11-30_15 19 04" src="https://github.com/user-attachments/assets/c5dece47-29cb-4976-9408-852c0b6f71c3" />

## Input

### 1. Folder MiniSEED Files
- **Lokasi**: Folder yang berisi file MiniSEED (`.mseed`)
- **Format**: MiniSEED standar
- **Struktur**: File dapat berada di root folder atau subfolder (recursive search)
- **Konfigurasi**: Edit variabel `LOCAL_MSEED_ROOT` di file `picking_gui_local.py`

```python
LOCAL_MSEED_ROOT = "./2017-11-01T05-39-34_-8.168412_107.317574_13.868_4.9"
```

### 2. Requirements File
- File MiniSEED harus valid dan dapat dibaca oleh ObsPy
- Minimum durasi trace: 30 detik (untuk PhaseNet)
- Format yang didukung: MSEED standar

## Proses

### 1. Inisialisasi
- **Load PhaseNet Model**: Memuat model PhaseNet pre-trained "stead" dengan sampling rate 100 Hz
- **Scan Files**: Mencari semua file `.mseed` di folder yang ditentukan (recursive)
- **Create CSV**: Membuat file CSV output dengan header jika belum ada

### 2. Load Stream
Untuk setiap file MiniSEED:
- **Read File**: Membaca file menggunakan ObsPy `read()`
- **Validation**: 
  - Cek apakah stream tidak kosong
  - Cek durasi minimum 30 detik per trace
- **Preprocessing**:
  - Detrend (demean + linear)
  - Resample ke sampling rate PhaseNet (100 Hz)
  - Merge traces jika ada gap
- **Error Handling**: Jika file tidak bisa dibaca, tampilkan error message dan skip

### 3. AI Picking
- **PhaseNet Classification**: Menjalankan model PhaseNet pada stream
- **Extract P Picks**: Mengambil semua deteksi fase P
- **Select Best Pick**: Memilih pick dengan probability tertinggi
- **Return**: Mengembalikan waktu pick dan probability

### 4. GUI Display
- **Waveform Plot**: Menampilkan waveform untuk setiap komponen (Z, N, E)
- **AI Pick Marker**: Garis merah putus-putus menunjukkan hasil AI
- **Manual Pick Marker**: Garis hijau solid menunjukkan hasil manual (jika ada)
- **Interactive**: User dapat klik pada plot untuk set manual pick

### 5. User Interaction
- **Left Click**: Set manual pick pada waktu yang diklik
- **Right Click**: Clear manual pick
- **Prev Button**: Navigasi ke file sebelumnya
- **Next Button**: Navigasi ke file berikutnya
- **Save Button**: Simpan hasil picking ke CSV

### 6. Save to CSV
Setiap kali user:
- Klik "Save"
- Klik "Next" atau "Prev"
- Menutup aplikasi

Data akan disimpan dengan format:
- `file_name`: Path lengkap file MiniSEED
- `pick_time_AI`: Waktu pick dari AI (ISO format)
- `pick_time_manual`: Waktu pick manual (ISO format, kosong jika tidak ada)

## Output

### File CSV
- **Nama File**: `output_pickscoba.csv` (dapat diubah di variabel `OUTPUT_CSV`)
- **Format**: CSV dengan header
- **Encoding**: UTF-8
- **Kolom**:
  1. `file_name`: Path lengkap file MiniSEED
  2. `pick_time_AI`: Waktu pick AI dalam format ISO (contoh: `2017-11-01T05:40:01.650000`)
  3. `pick_time_manual`: Waktu pick manual dalam format ISO (kosong jika tidak ada)

### Contoh Output
```csv
file_name,pick_time_AI,pick_time_manual
./2017-11-01T05-39-34_-8.168412_107.317574_13.868_4.9/BALE.mseed,,2017-11-01T05:39:57.626212
./2017-11-01T05-39-34_-8.168412_107.317574_13.868_4.9/CBJI.mseed,2017-11-01T05:40:01.650000,
./2017-11-01T05-39-34_-8.168412_107.317574_13.868_4.9/CGJI.mseed,2017-11-01T05:40:10.150000,
```

### Interpretasi Output
- Jika `pick_time_AI` kosong: AI tidak menemukan pick atau file tidak bisa diproses
- Jika `pick_time_manual` kosong: User belum melakukan manual pick
- Jika keduanya terisi: User dapat membandingkan hasil AI dengan manual pick

## Cara Penggunaan

### 1. Install Dependencies
```bash
pip install obspy PyQt5 matplotlib numpy seisbench
```

### 2. Konfigurasi
Edit file `picking_gui_local.py`:
```python
LOCAL_MSEED_ROOT = "./path/to/your/mseed/folder"
OUTPUT_CSV = "output_pickscoba.csv"
```

### 3. Jalankan Aplikasi
```bash
python picking_gui_local.py
```

### 4. Menggunakan GUI
1. **Lihat Waveform**: Aplikasi akan otomatis menampilkan file pertama
2. **Review AI Pick**: Garis merah putus-putus menunjukkan hasil AI
3. **Manual Pick** (opsional):
   - Klik kiri pada plot untuk set manual pick
   - Klik kanan untuk clear manual pick
4. **Navigasi**:
   - Klik "Prev" untuk file sebelumnya
   - Klik "Next" untuk file berikutnya
5. **Save**: Klik "Save" untuk menyimpan hasil picking ke CSV

### 5. Hasil
File CSV akan ter-update setiap kali:
- User klik "Save"
- User navigasi ke file lain
- Aplikasi ditutup

## Struktur Kode

### Fungsi Utama

1. **`list_all_mseed_local(root_dir)`**
   - Mencari semua file `.mseed` secara recursive
   - Return: List path file

2. **`load_stream_local(filepath)`**
   - Membaca dan memproses file MiniSEED
   - Validasi dan preprocessing
   - Return: ObsPy Stream atau None jika error

3. **`ai_pick_time_and_prob(st, filepath)`**
   - Menjalankan PhaseNet pada stream
   - Extract dan select best P pick
   - Return: (pick_time, probability)

4. **`ensure_csv_header(path_csv)`**
   - Membuat file CSV dengan header jika belum ada

5. **`append_csv_row(path_csv, file_name, ai_time, manual_time)`**
   - Menambahkan baris baru ke CSV

### Class Utama

1. **`WaveformCanvas`**
   - Menampilkan waveform dengan matplotlib
   - Handle click events untuk manual picking
   - Plot AI dan manual pick markers

2. **`PickingGUI`**
   - Main GUI window
   - Handle navigation dan save
   - Koordinasi antara canvas dan data

## Dependencies

- **obspy**: Untuk membaca dan memproses data seismik
- **PyQt5**: Untuk GUI framework
- **matplotlib**: Untuk plotting waveform
- **numpy**: Untuk operasi numerik
- **seisbench**: Untuk PhaseNet AI model

## Catatan Penting

### Error Handling
- File yang tidak bisa dibaca akan ditampilkan dengan pesan error
- Stream kosong atau terlalu pendek akan di-skip
- Error preprocessing akan ditampilkan di console

### Performance
- PhaseNet model di-load sekali saat startup
- Setiap file diproses saat navigasi (lazy loading)
- CSV di-update secara incremental (append mode)

### Limitations
- Minimum durasi trace: 30 detik
- Hanya mendeteksi fase P (tidak fase S)
- Sampling rate akan di-resample ke 100 Hz untuk PhaseNet
- File yang corrupt atau format tidak valid akan di-skip

## Troubleshooting

### File tidak bisa dibaca
- **Error**: "Unknown format for file"
- **Solusi**: Pastikan file adalah MiniSEED valid, cek dengan ObsPy secara terpisah

### AI tidak menemukan pick
- **Kemungkinan**: Signal terlalu lemah atau noise terlalu tinggi
- **Solusi**: Lakukan manual pick jika diperlukan

### GUI tidak muncul
- **Error**: Qt backend issues
- **Solusi**: Pastikan PyQt5 terinstall dengan benar, cek display settings

### CSV tidak ter-update
- **Kemungkinan**: Permission issue atau file locked
- **Solusi**: Pastikan file tidak dibuka di aplikasi lain, cek permission folder

## Contoh Workflow

1. **Setup**: Edit `LOCAL_MSEED_ROOT` ke folder data Anda
2. **Run**: `python picking_gui_local.py`
3. **Review**: Lihat hasil AI pick untuk setiap file
4. **Verify**: Klik manual pick jika AI pick tidak akurat
5. **Navigate**: Gunakan Next/Prev untuk semua file
6. **Save**: Hasil otomatis tersimpan ke CSV
7. **Analyze**: Gunakan CSV untuk analisis lebih lanjut

## Output Analysis

File CSV dapat digunakan untuk:
- Membandingkan akurasi AI vs manual picking
- Analisis statistik waktu arrival
- Input untuk lokasi hiposenter
- Quality control picking results
- Training data untuk model baru

