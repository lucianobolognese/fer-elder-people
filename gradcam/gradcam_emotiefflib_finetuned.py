"""
Grad-CAM per EmotiEffLib (SAGE-FACE FINETUNED)
======================================================
Genera heatmap di attenzione per visualizzare la correzione del bias morfologico.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SAGE_FACE_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224_Split\test"

OUTPUT_DIR = r"C:\Users\ciano\Desktop\Tesi\GradCAM_Output_FINETUNED"

MODEL_WEIGHTS_PATH = r"C:\Users\ciano\Desktop\emotiefflib\emotiefflib_sageface_FOLD_5.pth"

N_SAMPLES_PER_CLASS = 3

EMOTIEFF_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

FOLDER_TO_INDEX = {
    "angry":    0,  
    "disgust":  1,  
    "fear":     2,
    "happy":    3,
    "neutral":  4,
    "sad":      5,
    "surprise": 6
}

class ReWiredEmotiEff(torch.nn.Module):
    def __init__(self, recognizer, device):
        super().__init__()
        self.features_extractor = recognizer.model
        
        weights = torch.tensor(recognizer.classifier_weights, dtype=torch.float32).to(device)
        bias = torch.tensor(recognizer.classifier_bias, dtype=torch.float32).to(device)
        
        out_features = weights.shape[0] 
        in_features = weights.shape[1]  
        
        self.classifier = torch.nn.Linear(in_features, out_features)
        self.classifier.weight = torch.nn.Parameter(weights)
        self.classifier.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

class GradCAM:
    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hook_handles = []
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name: str):
        target = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target = module
                break

        if target is None:
            raise ValueError(f"Layer '{layer_name}' non trovato nel modello.")

        def fwd_hook(module, input, output):
            self.activations = output.detach()
        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hook_handles.append(target.register_forward_hook(fwd_hook))
        self._hook_handles.append(target.register_full_backward_hook(bwd_hook))

    def generate(self, img_tensor: torch.Tensor, target_class: int = None):
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        output = self.model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_conf, pred_idx = torch.max(probabilities, 1)

        if target_class is None:
            target_class = pred_idx.item()

        score = output[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), pred_idx.item(), pred_conf.item()

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()


def preprocess_image(img_path: str, device: torch.device) -> tuple:
    image_bgr = cv2.imread(img_path)
    if image_bgr is None: raise FileNotFoundError(f"Non trovato: {img_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image_rgb).unsqueeze(0).to(device)
    image_rgb_resized = cv2.resize(image_rgb, (224, 224))
    return img_tensor, image_rgb_resized

def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LANCZOS4)
    cam_resized = cv2.GaussianBlur(cam_resized, (31, 31), 0)

    cam_min, cam_max = cam_resized.min(), cam_resized.max()
    if cam_max > cam_min:
        cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)

    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_rgb + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay

def find_target_layer_auto(model) -> str:
    last_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_name = name
    if last_conv_name is None:
        raise RuntimeError("Nessun layer Conv2d trovato.")
    return last_conv_name


def generate_bias_comparison_figure(gradcam, device, emotion_class, image_paths, output_path, true_class_idx):
    n = len(image_paths)
    fig = plt.figure(figsize=(5 * n, 12))
    fig.patch.set_facecolor('#0f0f0f')
    gs = gridspec.GridSpec(3, n, figure=fig, hspace=0.08, wspace=0.04)
    row_labels = ["Immagine originale", "Grad-CAM (attenzione)", "Overlay"]

    for col, img_path in enumerate(image_paths):
        try:
            img_tensor, image_rgb = preprocess_image(img_path, device)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        cam, pred_idx, pred_conf = gradcam.generate(img_tensor, target_class=true_class_idx)
        overlay = overlay_heatmap(image_rgb, cam)

        for row, (data, cmap) in enumerate(zip([image_rgb, cam, overlay], [None, 'jet', None])):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#0f0f0f')
            ax.imshow(data, cmap=cmap)
            ax.axis('off')

            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11, color='white', labelpad=8, rotation=90, va='center')

            if row == 0:
                pred_name = EMOTIEFF_CLASSES[pred_idx]
                is_correct = (pred_idx == true_class_idx)
                color = '#4ade80' if is_correct else '#f87171'
                symbol = '✓' if is_correct else '✗'
                ax.set_title(f"GT: {emotion_class}\nPred: {pred_name.upper()} ({pred_conf:.2f}) {symbol}",
                             fontsize=9, color=color, pad=4)

    fig.suptitle(f"Grad-CAM (SAGE-Face FINETUNED) — Classe: '{emotion_class.upper()}'",
                 fontsize=14, color='white', fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f', edgecolor='none')
    plt.close(fig)
    print(f"  Figura salvata: {output_path}")

def generate_single_gradcam(gradcam, device, img_path, output_path, true_class_idx=None, title=""):
    img_tensor, image_rgb = preprocess_image(img_path, device)
    cam, pred_idx, pred_conf = gradcam.generate(img_tensor, target_class=true_class_idx)
    overlay = overlay_heatmap(image_rgb, cam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#111111')
    pred_name = EMOTIEFF_CLASSES[pred_idx]

    panels = [
        (image_rgb, None, "Immagine originale"),
        (cam,       'jet', "Mappa di attenzione\n(Grad-CAM)"),
        (overlay,   None,  f"Overlay\nPred: {pred_name.upper()} ({pred_conf:.2f})"),
    ]

    for ax, (data, cmap, label) in zip(axes, panels):
        ax.set_facecolor('#111111')
        ax.imshow(data, cmap=cmap)
        ax.set_title(label, color='white', fontsize=12, pad=8)
        ax.axis('off')

    if title:
        fig.suptitle(title, color='white', fontsize=13, fontweight='bold', y=1.02)

    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Intensità attenzione', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='#111111', edgecolor='none')
    plt.close(fig)
    print(f"  Figura singola salvata: {output_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Inizializzazione backbone (enet_b2_7)...")
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer
    fer = EmotiEffLibRecognizer(engine="torch", model_name="enet_b2_7", device=device)
    
    model = ReWiredEmotiEff(fer, device).to(device)
    
    print(f"Caricamento pesi personalizzati da: {MODEL_WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    print("Pesi caricati con successo!")
    
    model.eval()

    TARGET_LAYER_NAME = 'features_extractor.blocks.5' 
    
    try:
        print(f"Tentativo di aggancio GradCAM al layer manuale: '{TARGET_LAYER_NAME}'...")
        gradcam = GradCAM(model, TARGET_LAYER_NAME)
        print("Aggancio riuscito!")
    except ValueError:
        fallback_layer = find_target_layer_auto(model)
        print(f"ATTENZIONE: Fallback su auto-detect: '{fallback_layer}'")
        gradcam = GradCAM(model, fallback_layer)

    print("\n=== Generazione figure comparative per classe ===")
    for emotion_name, true_idx in FOLDER_TO_INDEX.items():
        class_dir = os.path.join(SAGE_FACE_DIR, emotion_name)
        if not os.path.isdir(class_dir):
            continue

        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images: continue

        rng = np.random.default_rng(seed=42)
        selected = rng.choice(images, size=min(N_SAMPLES_PER_CLASS, len(images)), replace=False).tolist()

        out_path = os.path.join(OUTPUT_DIR, f"gradcam_FINETUNED_{emotion_name}.png")
        print(f"\n  Classe '{emotion_name}' — {len(selected)} immagini...")
        generate_bias_comparison_figure(gradcam, device, emotion_name, selected, out_path, true_idx)

    EXAMPLE_IMAGES = {
        "happy_sage": (os.path.join(SAGE_FACE_DIR, "happy"), FOLDER_TO_INDEX["happy"], "SAGE-Face: Happy — soggetto anziano\n"),
        "neutral_sage": (os.path.join(SAGE_FACE_DIR, "neutral"), FOLDER_TO_INDEX["neutral"], "SAGE-Face: Neutral — soggetto anziano\n"),
        "falso_arrabbiato_99": (r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224_Split\test\neutral\RAFDB_train_10308_aligned.jpg", FOLDER_TO_INDEX["neutral"], "SAGE-Face: Neutro misclassificato\n"),
        "falso_disgustato_98": (r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224_Split\test\neutral\AffectNet_ffhq_2870.png", FOLDER_TO_INDEX["neutral"], "SAGE-Face: Neutro misclassificato\n"),
        "falso_sorpreso_97": (r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224_Split\test\neutral\DFEW_08854_frame8.jpg", FOLDER_TO_INDEX["neutral"], "SAGE-Face: Neutro misclassificato \n"),
        "falso_triste_94": (r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224_Split\test\neutral\DFEW_03590_frame8.jpg", FOLDER_TO_INDEX["neutral"], "SAGE-Face: Neutro misclassificato\n"),
        "controllo_giovane_neutro": (r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\FER-2013\train\neutral\Training_1033260.jpg", FOLDER_TO_INDEX["neutral"], "Controllo FER-2013: Neutro corretto (100%)\n"),
        "controllo_giovane_felice": (r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\FER-2013\train\happy\Training_10239785.jpg", FOLDER_TO_INDEX["happy"], "Controllo FER-2013: Felice corretto (99.6%)\n"),
        "controllo_giovane_disgusto": (r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\FER-2013\train\disgust\Training_13362748.jpg", FOLDER_TO_INDEX["disgust"], "Controllo FER-2013: Disgusto corretto (99.9%)\n"),
    }

    print("\n=== Generazione figure singole sui casi critici ===")
    for key, (folder_or_path, true_idx, title) in EXAMPLE_IMAGES.items():
        if os.path.isdir(folder_or_path):
            imgs = [os.path.join(folder_or_path, f) for f in sorted(os.listdir(folder_or_path)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not imgs: continue
            img_path = imgs[0]
        else:
            img_path = folder_or_path
            if not os.path.isfile(img_path): continue

        out_path = os.path.join(OUTPUT_DIR, f"gradcam_single_FINETUNED_{key}.png")
        print(f"  '{key}': {img_path}")
        generate_single_gradcam(gradcam, device, img_path, out_path, true_idx, title)

    gradcam.remove_hooks()
    print(f"\nDone. Tutte le figure per il modello FINETUNED salvate in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()