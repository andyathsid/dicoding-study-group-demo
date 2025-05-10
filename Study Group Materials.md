## **Introduction**
Andakara Athaya Sidiq, Universitas Pendidikan Indonesia, Prodi Teknik Komputer, Domisili Bekasi, Study Group MC-34.

---
## **Topics**
1. Courses Guidance & General Study Support 
2. Jupyter Notebook Provider Sharing, Discussion, & Best Practices

---

## **Part 1: Courses Guidance & General Study Support**

### **Courses Guidance**

#### **1. Belajar Machine Learning Pemula**

**Proyek Akhir General Troubleshooting:**
> **Banyak outliers?** 
- Coba winsorize kalau gamau ngapus outlier karna data sedikit, desktrutif, dll.
	
> **Silhoutte score kurang?**
- Terapin feature engineering & feature selection
	- Feature Engineering Technique:
		1. **Perilaku Temporal (best method if possible):**
			- **Definisi:** Membuat fitur dari data tanggal/waktu menggunakan lag, diff (perbedaan), rolling windows, atau mengekstraksi komponen seperti hari dalam seminggu, bulan, hari libur, akhir pekan, dll.
			- **Contoh:** 
				- **Lag:** `sales_last_day = lag(sales, 1)` (penjualan kemarin)
				- **Diff:** `change_in_sales = diff(sales, 1)` (berapa banyak penjualan yang berubah dari kemarin)
				- **Rolling:** `rolling_avg_7_days = mean penjualan 7 hari terakhir`
				- **Date Components:** `is_weekend`, `month`, `is_holiday`
		2. **Agregrasi:**
			- **Definisi:** Meringkas data dengan mengelompokkannya (misalnya, berdasarkan pelanggan, produk, periode waktu) dan menghitung statistik seperti mean, sum, count, dll. Biasa bisa dilakukan kalau fitur punya hubungan one-to-many (e.g., setiap pembeli (one) punya banyak transaksi (many))
			- **Contoh:** 
			  Dalam kumpulan data penjualan, kamu bisa mengelompokkan transaksi berdasarkan customer_id dan menghitung:
			  ➡️ `avg_purchase_value`, `total_purchases_last_month`, `num_unique_products_bought`
			  Bisa ngebantu prediksi kesetiaan pelanggan atau churn.
		3. **Kombinasi Fitur Kategorikal:**
			- **Definisi:** Menggabungkan dua atau lebih fitur yang sudah ada menjadi satu fitur baru untuk menangkap interaksi atau menyederhanakan hubungan.
			- **Contoh:** Misal dataset ada fitur `jumlah_kamar` dan `luas_rumah` dalam data perumahan. Anda dapat membuat fitur baru:
			 ➡️ `area_per_room = luas_rumah / jumlah_kamar`  
			 Ini ngasih gambaran seberapa luas ruangan di kamar buat bantu memprediksi harga rumah lebih baik daripada salah satu fitur ajh.
- Eksperiment gonta-ganti scaler (e.g, MinMaxScaler, StandartScaler)

> **Akurasi model klasifikasi kurang?**
- Pake gradient booster (e.g., XGBoost, LGBM, dll) & tuning make optuna. Cek implementasi optuna di https://github.com/optuna/optuna-examples/tree/main, misal buat [XGBoost](https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_simple.py)
- Kalau class imbalance, terapin class weight.

Reference/Extra Materials:
- Winsorize: https://ndgigliotti.medium.com/trimming-vs-winsorizing-outliers-e5cae0bf22cb
- Feature Engineering: https://www.youtube.com/watch?v=ft77eXtn30Q&t=602s
- Tutorial Feature Engineering Combination: https://hackernoon.com/improve-machine-learning-model-performance-by-combining-categorical-features-g21u34ep?utm_source=chatgpt.com
- Class Weight: https://www.youtube.com/watch?v=LrLaNyQ8oZc
- Tutorial contoh tuning XGBoost dengan Optuna: https://sainsdata.id/machine-learning/12313/tuning-hyperparameter-model-xgboost-dengan-optuna/
#### **2. Belajar Pengembangan Machine Learning

**Proyek Akhir General Troubleshooting:**
> Model performa kurang karena underfitting?
- **Analisis Sentiment:** 
	- Tambah/ganti label dari dataset make VADER atau alternatif lainnya (e.g., Rule-based: TextBlob, Lexicon-based: Afinn).
	- Buat deep learning pake BERT atau variannya (e.g., DistilBERT, RoBERTa, dll). Hindarin RNN dan variannya. Training cukup beberapa epoch ajh.
	- Coba ikutin parameter ini buat BERT, cek section **Feature Extraction Deep Learning**: https://github.com/andyathsid/dicoding-courses-final-project-archive/blob/main/learn_machine_learning_development/sentiment_analysis/notebooks/modelling.ipynb.
	- Coba hindarin menggunakan lemmatization, stemming, & stopword buat BERT yang udah bagus out of the box.
	- Buat ML, pake gradient booster (e.g., XGBoost, LGBM, dll) & tuning make optuna. Cek implementasi optuna di https://github.com/optuna/optuna-examples/tree/main, misal buat [XGBoost](https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_simple.py).
- **Klasifikasi gambar:**
	- Pake model transfer learning kaya ResNet, EfficientNet, dll (NOTE: tetep tambahin layer Conv2D and Pooling Layer pake fungsi `tf.keras.Sequential`).
	- Ganti dataset yang gambarnya lebih simple, I'd recommend this: 
	  https://www.kaggle.com/datasets/aayushpurswani/diamond-images-dataset/data

> Model performa kurang karena overfitting?
- Pake Adam atau variannya buat optimizer.
- Pake callback early stopping dengan argumen `restore_best_weights=True` di `tf.keras.callbacks.EarlyStopping`.
- Pake callback `tf.keras.callbacks.ReduceLROnPlateau`.
- Pake class weight kalo class imbalance.

Reference/Extra Materials:
- Contoh penggunaan lexicon-based sederhana dari pre-defined kamus buat labeling: https://www.dicoding.com/academies/185/tutorials/37438
- Tutorial implementasi BERT: https://www.youtube.com/watch?v=pEMe2d0MlTg&t=1494s
- Tutorial tuning dengan Optuna: https://sainsdata.id/machine-learning/12313/tuning-hyperparameter-model-xgboost-dengan-optuna/
- Saran baseline parameter (cek Table 3.) & tuning BERT:  https://arxiv.org/abs/2006.04884

---
### **Courses Guidance**
Share masalah kalian atau apa yang lagi pengen diketahui! (e.g., tugas, materi atau sekedar tips/trick sesuatu).

---
## **Part 2: Jupyter Notebook Provider**
### **Baseline Choices**
- **Goggle Collab:**
	- **Pros:**
		- Free akses ke GPU (Nvidia T4) **lumayan** 
		- **Storage persistent bagus** via integrasi ke Google Drive
		- **Accessible** karena gampang buat collab/sharing notebook & almost no setup
		- UI/UX bagus
		- **Coding assistance** kaya code completion, parameter hints, docstring, linting.
	- **Cons:**
		- Session limit yang klaimnya 12 jam bisa tiba-tiba dikurangin
		- Ada timeout kalo idle/inactive
		- Arbitrary peak times rate & computation limit jadi lemot atau tiba2 disconnect

- **Kaggle Kernels**
	- **Pros:**
		- Free akses ke GPU (NVIDIA P100 atau **T4 x2**) atau TPU **bagus** (30 jam/minggu)
		- **Reproducibility** karena bisa versioning & tracking notebook
		- **Version control** via github atau kaggle api
		- **Storage persistent cukup** lumayan (15 GB Read/Write terbatas, tapi unlimited Read-Only input dataset)
		- **Accessible** karena gampang buat sharing & publishing, no setup, seamless integraton sama kaggle dataset 
		- Session limit konsisten 12 jam
		- Gaada rate atau computation limit
		- Gaada timeout kalo idle
	- **Cons:**
		- No code assistance
		- UI/UX kurang
	
---
### **Key Takeaways**

| Platform       | GPU/TPU Power | Pengalaman Coding | Stabilitas & Limit | Skor Total |
| -------------- | ------------- | ----------------- | ------------------ | ---------- |
| Google Colab   | 3/5           | 5/5               | 2/5                | **10/15**  |
| Kaggle Kernels | 5/5           | 2/5               | 5/5                | **12/15**  |

> **Target Ideal:**  
> - GPU bagus  
> - Enak buat coding  
> - Batas pemakaian wajar dan stabil

---
### **Alternatives?**
![[Pasted image 20250503133900.png]]
>Source: <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/640px-YouTube_full-color_icon_%282017%29.svg.png" alt="YouTube Logo" width="20"/>[AICodeKing](https://www.youtube.com/watch?v=yvvNtkfJhGI)

**S-Tier List:**
- Kaggle Kernel
- AWS SageMaker Studio Lab
- ~~Lightning AI Studio~~ (Ideal, tapi ga tersedia di Indo)

**AWS SageMaker Studio Lab:**
	- **Pros:**
		- Free akses ke GPU (NVIDIA T4) **lumayan** (4 jam GPU & 12 jam CPU per hari)
		- **Storage persistent lumayan** (15 GB persistent, bisa workaround nambah via AWS S3)
		- UI/UX ngikutin JupyterLab, which is **good**
		- **Coding assistance lumayan** kaya syntax highlighting & autocomplete, bisa juga hover help and linting via ekstension [JupyterLab LSP](https://github.com/jupyter-lsp/jupyterlab-lsp)
		- Gaada rate atau computation limit
		- Gaada timeout kalo idle
	- **Cons:**
		- Kurang accessible karena perlu waitlist buat bikin akun & akses gpu, terus susah sharing/collab

> Demo: <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/640px-YouTube_full-color_icon_%282017%29.svg.png" alt="YouTube Logo" width="20"/>[AWS User Group Indonesia](https://www.youtube.com/watch?v=9lIHWCkyb5I) ![[Pasted image 20250503141347.png]]

### **Key Takeaways**

| Platform                 | GPU/TPU Power | Pengalaman Coding | Stabilitas & Limit | Skor Total |
| ------------------------ | ------------- | ----------------- | ------------------ | ---------- |
| Google Colab             | 3/5           | 5/5               | 2/5                | **10/15**  |
| Kaggle Kernels           | 5/5           | 2/5               | 5/5                | **12/15**  |
| AWS SageMaker Studio Lab | 4/5           | 5/5               | 5/5                | **14/15**  |
### **Best Practices**
Q: Males pindah dari Kaggle Kernel, adakah cara biar bisa dapet target ideal?
A: via code syncing dengan IDE local favorit kalian
#### Step-by-step Instructions
1. **(Optional)** Sync via Kaggle API / CLI
	1. Install CLI Kaggle:
   ```bash
	pip install kaggle
   ```
	2. Buat token API di halaman Akun Kaggle (Settings → Account Tab → API Section → Create New Token) untuk mengunduh kaggle.json  
	3. Upload token `kaggle.json` ke direktori:
		- Linux/macOS: `~/.kaggle/kaggle.json`
		- Windows: `C:\Users\<User>\.kaggle\kaggle.json`
		- Contoh di Linux:
		```bash
		mkdir -p ~/.kaggle
		mv ~/downloads/kaggle.json ~/.kaggle/
		chmod 600 ~/.kaggle/kaggle.json 
```
	4. Inisialisasi metadata
	- **Via inisialisasi awal di folder lokal**
		1. Init dulu:
		```bash
		kaggle kernels init -p /path/to/your/kernel/dir
	```
		2. Edit file metada sesuai intruksi yang dikasih:
			```json
	{
	  "id": "andyathsid/INSERT_KERNEL_SLUG_HERE", // URL Kaggle kernel atau notebook kalian nantinya
	  "title": "INSERT_TITLE_HERE", // Nama notebook kalian di Kaggle nantinya
	  "code_file": "INSERT_CODE_FILE_PATH_HERE", // Path kode atau notebook kalian di local yang mau dimasukin ke Kaggle kernel
	  "language": "Pick one of: {python,r,rmarkdown}",
	  "kernel_type": "Pick one of: {script,notebook}",
	  "is_private": "true",
	  "enable_gpu": "false",
	  "enable_tpu": "false",
	  "enable_internet": "true",
	  "dataset_sources": [],
	  "competition_sources": [],
	  "kernel_sources": [],
	  "model_sources": []
	}
	```
			- Contoh json:
			```json
	{
	  "id": "andyathsid/study-group-demo",
	  "title": "Study Group Demo",
	  "code_file": "notebook.ipynb",
	  "language": "python",
	  "kernel_type": "notebook",
	  "is_private": "true",
	  "enable_gpu": "true",
	  "enable_tpu": "false",
	  "enable_internet": "true",
	  "dataset_sources": [],
	  "competition_sources": [],
	  "kernel_sources": [],
	  "model_sources": []
	}
	```
	4. Push notebook & metadata ke kaggle kernel.
		NOTE: Dua file tersebut harus dalam satu folder karena command di bawah ini cuman nerima argumen folder path. Contoh di bawah ini kalo keduanya ada di folder parent
	```bash
	kaggle kernels push -p .
```
	- **Via inisialisasi awal di kaggle**
	1. Bikin notebook di kaggle
	2. **(Wajib)** Bikin version via tombol "Save Version" di kanan atas notebook viewer
	3. Pull notebook & metada dari kaggle:
	```bash
	kaggle kernels pull -m -p . andyathsid/study-group-demo
```
	4. Push perubahan di local ke kaggle
```bash
	kaggle kernels push -p .
```

2. Di kernel kalian, File -> Link to GitHub -> Ikutin pop-up buat authorize akun
3. Buat commit and push dari Kaggle ke Local:
	- Save Version -> Bebas milih version type -> Continue -> (Muncul menu save copy to github) -> Tentuin repo, branch, file name, commit message -> Push
4. Buat commit & push dari Local ke Kaggle or syncing dari Kaggle ke Local:
	- Pull dulu kalo ada perubahan repo dari kaggle sebelumnya
	- Edit kaya biasa
	- Push and commit ke repo kaya biasa
5. Buat syncing perubahan di Local ke Kaggle
	1. Opsi 1 (dari Local):
		- Push dari local make kaggle CLI kalau step sebelumnya diiktuin
	2. Opsi 2 (dari Kernel): 
		- File -> Import Notebook -> Tab GitHub -> Masukin ID GitHub & Nama Repo (e.g.,  andyathsid/dicoding-study-group-demo)