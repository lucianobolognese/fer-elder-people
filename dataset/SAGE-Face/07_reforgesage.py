import os
import cv2
import random
import shutil
import numpy as np
import mediapipe as mp
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import train_test_split


SAGE_TRAIN_SRC = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224_Split\train"
SAGE_TEST_SRC  = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224_Split\test"
RAISEFER_SRC   = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\RaiseFER\emotions\Complete"
RAFDB_SRC      = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\Raf-DB\Complete"
AFFECTNET_SRC  = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\AFFECTNET\Complete"

BASE_OUT_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\SAGE-Face_FineTuning"
OUT_TRAIN = os.path.join(BASE_OUT_DIR, "train")
OUT_VAL   = os.path.join(BASE_OUT_DIR, "val")
OUT_TEST  = os.path.join(BASE_OUT_DIR, "test")

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
TARGET_IMG_PER_DATASET = 1000
VAL_SPLIT_RATIO = 0.20 
AUG_MULTIPLIER = 5     # Quante versioni aumentate generare per OGNI immagine



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def apply_specific_erasing(image):
    """Oscura casualmente la zona degli occhi o la zona inferiore (bocca)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return image
        
    for (x, y, w, h) in faces:
        target_part = random.choice(['eyes', 'mouth'])
        img_erased = image.copy()
        
        if target_part == 'eyes':
   
            roi_gray = gray[y:y+int(h/2), x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                for (ex, ey, ew, eh) in eyes:
                    img_erased[y+ey:y+ey+eh, x+ex:x+ex+ew] = (0, 0, 0)
            else:
                img_erased[y+int(h*0.2):y+int(h*0.5), x:x+w] = (0, 0, 0)
                
        else:
            mouth_y = y + int(h * 0.65)
            img_erased[mouth_y:y+h, x+int(w*0.2):x+int(w*0.8)] = (0, 0, 0)
            
        return img_erased
        
    return image


basic_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)
])


def create_dirs():
    for folder in [OUT_TRAIN, OUT_VAL, OUT_TEST]:
        for cls in CLASSES:
            os.makedirs(os.path.join(folder, cls), exist_ok=True)

def copy_and_resize(src_dir, dest_dir, cls_name, files, prefix):
    for f in files:
        src_path = os.path.join(src_dir, cls_name, f)
        dest_path = os.path.join(dest_dir, cls_name, f"{prefix}_{f}")
        try:
            img = cv2.imread(src_path)
            if img is not None:
                img_resized = cv2.resize(img, (224, 224))
                cv2.imwrite(dest_path, img_resized)
        except Exception:
            pass


def get_source_folder_name(dataset_prefix, cls_name):
    """
    Risolve le incongruenze di nomenclatura tra i dataset.
    SAGE e RaiseFER usano 'angry'. RAF-DB e AffectNet usano 'anger'.
    """
    if cls_name == 'angry' and dataset_prefix in ['raf', 'aff']:
        return 'anger'
    return cls_name

def main():
    print("Creazione cartelle...")
    create_dirs()
    
    # Copia il TEST di SAGE-Face intatto
    print("Copia Test set originale...")
    for cls in CLASSES:
        src_test = os.path.join(SAGE_TEST_SRC, cls)
        if os.path.exists(src_test):
            files = os.listdir(src_test)
            copy_and_resize(SAGE_TEST_SRC, OUT_TEST, cls, files, "sage")

    print("Creazione Train combinato (Risoluzione Nomenclatura in corso)...")
    img_per_class = TARGET_IMG_PER_DATASET // len(CLASSES) 
    
    for cls in CLASSES:

        sage_tr_dir = os.path.join(SAGE_TRAIN_SRC, cls)
        if os.path.exists(sage_tr_dir):
            sage_files = os.listdir(sage_tr_dir)
            copy_and_resize(SAGE_TRAIN_SRC, OUT_TRAIN, cls, sage_files, "sage")
            

        datasets_info = [
            (RAISEFER_SRC, "raise"), 
            (RAFDB_SRC, "raf"), 
            (AFFECTNET_SRC, "aff")
        ]
        
        for ext_src, prefix in datasets_info:

            src_cls_folder = get_source_folder_name(prefix, cls) 
            ext_cls_dir = os.path.join(ext_src, src_cls_folder)
            
            if os.path.exists(ext_cls_dir):
                files = [f for f in os.listdir(ext_cls_dir) if f.lower().endswith(('.jpg', '.png'))]
                sampled = random.sample(files, min(img_per_class, len(files)))
                
                for f in sampled:
                    src_path = os.path.join(ext_cls_dir, f)
                    dest_path = os.path.join(OUT_TRAIN, cls, f"{prefix}_{f}")
                    try:
                        img = cv2.imread(src_path)
                        if img is not None:
                            img_resized = cv2.resize(img, (224, 224))
                            cv2.imwrite(dest_path, img_resized)
                    except Exception:
                        pass
            else:
                print(f" [ATTENZIONE] Cartella non trovata: {ext_cls_dir}")

    print("Divisione Train / Val...")
    for cls in CLASSES:
        cls_dir = os.path.join(OUT_TRAIN, cls)
        files = os.listdir(cls_dir)
        train_files, val_files = train_test_split(files, test_size=VAL_SPLIT_RATIO, random_state=42)
        
        for f in val_files:
            shutil.move(os.path.join(cls_dir, f), os.path.join(OUT_VAL, cls, f))

    print("Avvio Data Augmentation Offline sul Train...")
    for cls in CLASSES:
        cls_dir = os.path.join(OUT_TRAIN, cls)
        files = [f for f in os.listdir(cls_dir) if not f.startswith("aug_")]
        
        for f in tqdm(files, desc=f"Augmenting {cls}"):
            img_path = os.path.join(cls_dir, f)
            image = cv2.imread(img_path)
            if image is None: continue
            
            for i in range(AUG_MULTIPLIER):
                augmented = basic_transform(image=image)['image']
                
                if random.random() < 0.3:
                    augmented = apply_specific_erasing(augmented)
                
                new_name = f"aug_{i}_{f}"
                cv2.imwrite(os.path.join(cls_dir, new_name), augmented)

    print("Pipeline completata! Dataset unificato e pronto per il Fine-Tuning.")

if __name__ == "__main__":
    main()