import os
import cv2
from tqdm import tqdm


INPUT_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face"


OUTPUT_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224_f"


TARGET_SIZE = (224, 224)

def uniforma_dataset():
    print(f"Avvio ridimensionamento dataset a {TARGET_SIZE[0]}x{TARGET_SIZE[1]}...")
    

    immagini = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                percorso_completo = os.path.join(root, file)
                
   
                rel_path = os.path.relpath(root, INPUT_DIR)
                dest_dir = os.path.join(OUTPUT_DIR, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
                
                percorso_destinazione = os.path.join(dest_dir, file)
                immagini.append((percorso_completo, percorso_destinazione))
                
    print(f"Trovate {len(immagini)} immagini da processare.")
    

    errori = 0
    with tqdm(immagini, desc="Ridimensionamento") as pbar:
        for orig_path, dest_path in pbar:
            try:

                img = cv2.imread(orig_path)
                if img is None:
                    errori += 1
                    continue
                

                img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
                
                # Salviamo l'immagine
                cv2.imwrite(dest_path, img_resized)
                
            except Exception as e:
                errori += 1
                continue
                
    print("\nUniformazione completata!")
    if errori > 0:
        print(f"Ci sono stati {errori} errori (file corrotti o non leggibili).")

if __name__ == "__main__":
    uniforma_dataset()