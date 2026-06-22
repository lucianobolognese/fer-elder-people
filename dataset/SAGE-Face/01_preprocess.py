import os 
import shutil
import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

BASE_SAGE_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face"
RAFDB_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1.Dataset\Raf-DB\Complete"
FACES_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1.Dataset\FACES"

EMOZIONI_SAGE = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
SOGLIA_ETA = 38

def processa_faces():
    print("\n📸 Avvio estrazione da FACES...")
    
    faces_emomap = {
        'a': 'Anger', 
        'd': 'Disgust', 
        'f': 'Fear', 
        'h': 'Happy', 
        'n': 'Neutral', 
        's': 'Sad'
    }
    
    file_copiati = 0
    file_presenti = os.listdir(FACES_DIR)
    
    for filename in tqdm(file_presenti, desc="Elaborazione FACES"):
        if not filename.lower().endswith('.jpg'):
            continue
            
        parti = filename.split('_')
        

        if len(parti) >= 4:
            gruppo_eta = parti[1] 
            emozione_code = parti[3] 
            
            # Filtriamo SOLO gli anziani ('o')
            if gruppo_eta == 'o' and emozione_code in faces_emomap:
                emozione_target = faces_emomap[emozione_code]
                
                percorso_origine = os.path.join(FACES_DIR, filename)
                nuovo_nome = f"FACES_{filename}"
                percorso_dest = os.path.join(BASE_SAGE_DIR, emozione_target, nuovo_nome)
                

                os.makedirs(os.path.dirname(percorso_dest), exist_ok=True)
                
                shutil.copy(percorso_origine, percorso_dest)
                file_copiati += 1
                
    print(f"FACES completato: estratti {file_copiati} soggetti 'Old'.")


def processa_rafdb():
    print("\n🧠 Avvio estrazione da RAF-DB (Inferenza Età)...")
    file_copiati = 0


    cartelle_emozioni = [d for d in os.listdir(RAFDB_DIR) if os.path.isdir(os.path.join(RAFDB_DIR, d))]
    
    for cartella_emo in cartelle_emozioni:

        emozione_target = cartella_emo.capitalize()
        if emozione_target not in EMOZIONI_SAGE:
            continue
            
        percorso_cartella_emo = os.path.join(RAFDB_DIR, cartella_emo)
        immagini = os.listdir(percorso_cartella_emo)
        
        for filename in tqdm(immagini, desc=f"RAF-DB -> {emozione_target}"):
            percorso_origine = os.path.join(percorso_cartella_emo, filename)
            
            try:
                results = DeepFace.analyze(
                    img_path = percorso_origine, 
                    actions = ['age'],
                    enforce_detection = False,
                    detector_backend = 'opencv',
                    silent = True
                )
                
                age = results[0]['age']
                

                if age >= SOGLIA_ETA:
                    nuovo_nome = f"RAFDB_{filename}"
                    percorso_dest = os.path.join(BASE_SAGE_DIR, emozione_target, nuovo_nome)
                    

                    os.makedirs(os.path.dirname(percorso_dest), exist_ok=True)
                    
                    shutil.copy(percorso_origine, percorso_dest)
                    file_copiati += 1
                    
            except Exception as e:

                continue
                
    print(f"RAF-DB completato: estratti {file_copiati} soggetti over-{SOGLIA_ETA}.")

if __name__ == "__main__":
    
    processa_faces()
    processa_rafdb()
    