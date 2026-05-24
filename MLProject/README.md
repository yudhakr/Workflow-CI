# MLflow CI/CD Pipeline - Wine Classification

## 1. Deskripsi Project

Project ini mengimplementasikan **MLflow Project** dan **GitHub Actions CI/CD Pipeline** untuk klasifikasi wine menggunakan **Random Forest Classifier**. Pipeline mencakup:

- **BASIC**: Training model dengan MLflow logging + GitHub Actions workflow
- **SKILLED**: Upload artifact/model ke GitHub
- **ADVANCE**: Build & push Docker image ke Docker Hub

## 2. Struktur Project

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── MLProject/
│   ├── modelling.py               # Entry point MLflow Project
│   ├── MLproject                  # MLflow Project definition
│   ├── conda.yaml                 # Conda environment
│   ├── requirements.txt           # Python dependencies
│   ├── dataset_preprocessing/
│   │   └── dataset_preprocessing.csv
│   ├── Dockerfile                 # Docker image for model serving
│   └── README.md
```

## 3. Cara Menjalankan MLflow Project

### Local dengan Conda

```bash
cd MLProject
mlflow run .
```

### Local dengan parameter

```bash
cd MLProject
mlflow run . --no-conda
```

### Dengan parameter kustom

```bash
cd MLProject
mlflow run . -P n_estimators=200 -P max_depth=20 -P random_state=42
```

## 4. Cara Menjalankan Workflow CI

Workflow akan otomatis trigger saat push ke branch `main`.

### Manual trigger via GitHub

1. Buka repository di GitHub
2. Klik **Actions** tab
3. Pilih **MLflow CI/CD Pipeline**
4. Klik **Run workflow**

### Local testing (gunakan act)

```bash
# Install act (https://github.com/nektos/act)
act -j mlflow-pipeline
```

## 5. Cara Build Docker

### Build Docker image

```bash
cd MLProject
docker build -t wine-classification:latest .
```

### Build menggunakan MLflow

```bash
# Setelah training, dapatkan run_id dari MLflow UI
mlflow models build-docker \
  --model-uri "runs:/<RUN_ID>/model" \
  --name "wine-classification:latest" \
  --enable-mlserver
```

### Run Docker container

```bash
docker run -p 5001:8080 wine-classification:latest
```

## 6. Cara Push Docker Hub

### Setup GitHub Secrets

Tambahkan secrets di GitHub repository:

| Secret | Value |
|--------|-------|
| `DOCKER_USERNAME` | Username Docker Hub Anda |
| `DOCKER_PASSWORD` | Password Docker Hub Anda atau Access Token |

### Manual push

```bash
# Login
docker login -u <USERNAME>

# Tag image
docker tag wine-classification:latest <USERNAME>/wine-classification:latest

# Push
docker push <USERNAME>/wine-classification:latest
```

## 7. Cara Setup GitHub Secrets

1. Buka repository di GitHub
2. Klik **Settings** > **Secrets and variables** > **Actions**
3. Klik **New repository secret**
4. Tambahkan:
   - `DOCKER_USERNAME` = username Docker Hub Anda
   - `DOCKER_PASSWORD` = password atau access token Docker Hub

## 8. Penjelasan Workflow CI Steps

### BASIC (Steps 1-7)
| Step | Deskripsi |
|------|-----------|
| 1. Set up job | Inisialisasi job |
| 2. Checkout | Clone repository |
| 3. Setup Python | Install Python 3.12.7 |
| 4. Check Environment | Verifikasi Python & pip |
| 5. Install dependencies | Install requirements.txt |
| 6. Run MLflow project | `mlflow run .` |
| 7. Get latest run_id | Dapatkan Run ID terakhir |

### SKILLED (Steps 8-10)
| Step | Deskripsi |
|------|-----------|
| 8. Install Python deps | Additional dependencies |
| 9. Upload artifact | Upload model ke GitHub Actions |
| 10. Complete job | Finalisasi job |

### ADVANCE (Steps 11-15)
| Step | Deskripsi |
|------|-----------|
| 11. Setup Docker Buildx | Siapkan Docker builder |
| 12. Build Docker Model | `mlflow models build-docker` |
| 13. Login Docker Hub | Authentikasi ke Docker Hub |
| 14. Tag Docker Image | Tag image dengan versi |
| 15. Push Docker Image | Push ke Docker Hub |

## 9. MLflow Logging

Yang di-log ke MLflow:

- **Parameters**: n_estimators, max_depth, random_state, test_size, model_type
- **Metrics**: accuracy, precision_macro, recall_macro, f1_macro
- **Artifacts**:
  - Classification report (text)
  - Confusion matrix (PNG)
  - Feature importance plot (PNG)
  - Model artifact (MLflow sklearn format)
