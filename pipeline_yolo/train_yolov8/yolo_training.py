from ultralytics import YOLO
import torch

# =========================
# VÉRIFICATIONS ENVIRONNEMENT
# =========================
assert torch.cuda.is_available(), "CUDA non disponible"
print("GPU :", torch.cuda.get_device_name(0))
print("CUDA PyTorch :", torch.version.cuda)

# =========================
# PARAMÈTRES
# =========================
DATASET_YAML = "dataset.yaml"   # chemin vers votre dataset.yaml
MODEL = "yolov8n.pt"

IMGSZ = 128
EPOCHS = 150
BATCH = 32

# =========================
# CHARGEMENT MODÈLE
# =========================
model = YOLO(MODEL)

# =========================
# ENTRAÎNEMENT
# =========================
model.train(
    data=DATASET_YAML,
    imgsz=IMGSZ,
    epochs=EPOCHS,
    batch=BATCH,

    # Augmentations adaptées 128x128
    mosaic=0.0,
    mixup=0.0,
    degrees=0.0,
    scale=0.2,
    fliplr=0.5,

    # Stabilité
    workers=8,
    cache=True,
    device=0
)

print("Entraînement terminé")
