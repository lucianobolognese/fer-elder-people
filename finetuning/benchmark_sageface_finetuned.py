import os
import csv
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score

from emotiefflib.facial_analysis import EmotiEffLibRecognizer


FOLDER_TO_INDEX = {
    "neutral":  0,
    "happy":    1,
    "sad":      2,
    "surprise": 3,
    "fear":     4,
    "disgust":  5,
    "angry":    6
}

CLASSES = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]

MODEL_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class ReWiredEmotiEff(torch.nn.Module):
    def __init__(self, recognizer, device):
        super().__init__()
        self.features_extractor = recognizer.model
        weights = torch.tensor(recognizer.classifier_weights, dtype=torch.float32).to(device)
        bias = torch.tensor(recognizer.classifier_bias, dtype=torch.float32).to(device)
        self.classifier = torch.nn.Linear(weights.shape[1], weights.shape[0])
        self.classifier.weight = torch.nn.Parameter(weights)
        self.classifier.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

base_dir = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\RaiseFER\emotions\test"
MODEL_WEIGHTS_PATH = r"C:\Users\ciano\Desktop\emotiefflib\emotiefflib_sageface_finetuned.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Inizializzazione ambiente su {device}...")

fer = EmotiEffLibRecognizer(engine="torch", model_name="enet_b2_7", device=device)

model = ReWiredEmotiEff(fer, device).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.eval()
print(f"Pesi personalizzati caricati da: {MODEL_WEIGHTS_PATH}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


y_true = []
y_pred = []
total_by_class   = defaultdict(int)
discard_by_class = defaultdict(int)


csv_out = "results_test_FINETUNED.csv"
txt_out = "metrics_test_FINETUNED.txt"
img_out = "confusion_matrix_test_FINETUNED.png"

print(f"\nInizio elaborazione dataset da: {base_dir}")

with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "true_label", "pred_label", "correct"])

    for folder_name, true_idx in FOLDER_TO_INDEX.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Cartella non trovata: {folder_path} - Salto cartella.")
            continue

        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in tqdm(images, desc=folder_name):
            img_path = os.path.join(folder_path, img_name)
            total_by_class[folder_name] += 1

            try:
                frame_bgr = cv2.imread(img_path)
                if frame_bgr is None:
                    discard_by_class[folder_name] += 1
                    continue
                
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, preds = torch.max(outputs, 1)
                
            
                pred_class_name = MODEL_CLASSES[preds.item()]
                
                pred_idx = FOLDER_TO_INDEX[pred_class_name]

                y_true.append(true_idx)
                y_pred.append(pred_idx)
                writer.writerow([img_path, CLASSES[true_idx], CLASSES[pred_idx], int(true_idx == pred_idx)])
                
            except Exception as e:
                discard_by_class[folder_name] += 1


if not y_true:
    print("Nessun volto processato.")
else:
    total   = sum(total_by_class.values())
    discard = sum(discard_by_class.values())

    print(f"\nImmagini totali:    {total}")
    print(f"Classificate:       {len(y_true)}")
    print(f"Scartate:           {discard} ({100*discard/total:.1f}%)\n")

    for folder_name, true_idx in FOLDER_TO_INDEX.items():
        tot  = total_by_class[folder_name]
        disc = discard_by_class[folder_name]
        print(f"  {CLASSES[true_idx]:<12} scartate: {disc}/{tot} ({100*disc/tot:.1f}%)" if tot > 0 else f"  {CLASSES[true_idx]:<12} scartate: 0/0")

    acc_detected = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average='macro', zero_division=0)

    present_classes = sorted(list(set(y_true) | set(y_pred)))
    target_names_present = [CLASSES[i] for i in present_classes]
    
    report = classification_report(y_true, y_pred, labels=present_classes, target_names=target_names_present, zero_division=0)

    print(f"\nAccuracy (volti rilevati):  {acc_detected:.4f}")
    print(f"Precision (Macro):          {precision_macro:.4f}")
    print(f"F1-Score (Macro):           {f1_macro:.4f}\n")
    print(report)

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(f"Immagini totali:   {total}\n")
        f.write(f"Classificate:      {len(y_true)}\n")
        f.write(f"Scartate:          {discard} ({100*discard/total:.1f}%)\n\n")
        for folder_name, true_idx in FOLDER_TO_INDEX.items():
            tot  = total_by_class[folder_name]
            disc = discard_by_class[folder_name]
            if tot > 0:
                f.write(f"  {CLASSES[true_idx]:<12} scartate: {disc}/{tot} ({100*disc/tot:.1f}%)\n")
            
        f.write(f"\nAccuracy (volti rilevati):  {acc_detected:.4f}\n")
        f.write(f"Precision (Macro):          {precision_macro:.4f}\n")
        f.write(f"F1-Score (Macro):           {f1_macro:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.ylabel('Ground Truth (Reale)')
    plt.xlabel('Predetto dal Modello Finetuned')
    plt.title(f'Matrice di Confusione - SAGE-Face Finetuned\nAccuracy: {acc_detected*100:.1f}%')
    plt.tight_layout()
    plt.savefig(img_out, dpi=150)
    plt.close()
    
    print(f"\nAnalisi completata con successo.\nFile generati:\n- {img_out}\n- {csv_out}\n- {txt_out}")