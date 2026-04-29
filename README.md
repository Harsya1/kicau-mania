# Kicau Mania — Gesture-triggered audio & GIF (MediaPipe)

Ringkasan
- Aplikasi Python sederhana untuk mendeteksi titik hidung dan skeleton kedua telapak tangan menggunakan MediaPipe Tasks API.
- Saat tangan kiri menutup ujung hidung dan tangan kanan mengibaskan horizontal, aplikasi akan memutar audio (loop) dan menampilkan GIF overlay.

Persyaratan
- Python 3.13 (direkomendasikan)
- Virtual environment (disarankan)

Dependensi
- Semua dependensi ada di `requirements.txt`.

Persiapan lingkungan (Windows)
1. Buat dan aktifkan virtualenv di folder proyek (opsional kalau sudah ada `.venv`):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
# atau: .venv\Scripts\activate.bat   # Command Prompt
```

2. Pasang dependensi:

```powershell
pip install -r requirements.txt
```

Menjalankan aplikasi

```powershell
# Jika ingin gunakan Python dari venv secara langsung
.venv\Scripts\python.exe main.py

# atau bila venv sudah aktif
python main.py
```

Assets
- Letakkan file audio dan GIF di folder `assets/`.
- Default yang dicari oleh `main.py`:
  - `assets/kicau-mania.mp3`
  - `assets/kicau-mania.gif`
- Jika ingin menggunakan nama lain, ubah konstanta `AUDIO_FILE` dan `GIF_FILE` di `main.py`.

Model MediaPipe
- Pada percobaan pertama, aplikasi akan mengunduh model `holistic_landmarker.task` (~13.7 MB) ke `assets/` jika belum ada. Tunggu beberapa detik saat unduhan pertama.

Kontrol & Panduan Penggunaan
- Kamera akan menampilkan overlay live window.
- Trigger untuk audio+GIF:
  - Tangan kiri menutup ujung hidung (nose lock + left-hand cover).
  - Tangan kanan mengibaskan secara horizontal (wave / arah-x berubah beberapa kali dengan amplitudo tertentu).
- Tombol:
  - Tekan `q` pada jendela untuk keluar.

Penalaan
- Buka `main.py` dan sesuaikan konstanta threshold di bagian atas (mis. `COVER_DISTANCE_NORM`, `WAVE_MIN_AMPLITUDE`) bila perlu.

Debug & Troubleshooting
- Jika tidak ada kamera: pastikan aplikasi lain tidak mengunci webcam.
- Jika paket belum terpasang: jalankan `pip install -r requirements.txt` kembali.
- Jika model gagal diunduh: periksa koneksi internet atau unduh manual dari:
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task
  dan letakkan file tersebut ke folder `assets/`.

Lokasi file penting
- Kode utama: [main.py](main.py)
- Dependensi: [requirements.txt](requirements.txt)
- Asset contoh: [assets/README.md](assets/README.md)