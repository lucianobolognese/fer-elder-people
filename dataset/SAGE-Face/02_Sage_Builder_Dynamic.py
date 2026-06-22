import os
import shutil
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace


SOGLIA_ETA = 38

BASE_SAGE_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1_Dataset\SAGE-Face_Dataset"

# DFEW
DFEW_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1.Dataset\DFEW\Clip\clip_224x224_16f"
DFEW_EXCEL = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1.Dataset\DFEW\Clip\annotation.xlsx"

# ElderReact
ELDER_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1.Dataset\ElderReact\ElderReact_Frames"
ELDER_TXT_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1.Dataset\ElderReact\ElderReact-master\Annotations"

FRAME_TARGET = 8  

def processa_dfew():
    print("\n🎬 Avvio estrazione da DFEW (Con filtro Età DeepFace)...")
    file_copiati = 0
    file_scartati_giovani = 0
    
    dfew_emomap = {
        1: 'Happy', 2: 'Sad', 3: 'Neutral', 
        4: 'Anger', 5: 'Surprise', 6: 'Disgust', 7: 'Fear'
    }
    
    try:
        df = pd.read_excel(DFEW_EXCEL)
    except Exception as e:
        print(f"Errore nella lettura dell'Excel DFEW: {e}")
        return


    for index, row in tqdm(df.iterrows(), total=len(df), desc="Elaborazione DFEW"):
        
        # Usiamo la colonna 'order' che è l'ID assoluto del video
        id_reale = int(row['order'])
        video_id = f"{id_reale:05d}" 
        
        label = int(row['label'])
        
        if label in dfew_emomap:
            emozione_target = dfew_emomap[label]
            
            frame_origine = os.path.join(DFEW_DIR, video_id, f"{FRAME_TARGET}.jpg")
            
            if os.path.exists(frame_origine):
                try:
                    results = DeepFace.analyze(
                        img_path = frame_origine, 
                        actions = ['age'],
                        enforce_detection = False,
                        detector_backend = 'opencv',
                        silent = True
                    )
                    
                    age = results[0]['age']
                    
                    if age >= SOGLIA_ETA:
                        nuovo_nome = f"DFEW_{video_id}_frame{FRAME_TARGET}.jpg"
                        percorso_dest = os.path.join(BASE_SAGE_DIR, emozione_target, nuovo_nome)
                        
                        os.makedirs(os.path.dirname(percorso_dest), exist_ok=True)
                        shutil.copy(frame_origine, percorso_dest)
                        file_copiati += 1
                    else:
                        file_scartati_giovani += 1
                        
                except Exception as e:
                    continue
                
    print(f"DFEW completato: estratti {file_copiati} frame di anziani.")
    print(f"Scartati {file_scartati_giovani} soggetti under-{SOGLIA_ETA}.")


def processa_elderreact():
    print("\nAvvio estrazione AVANZATA da ElderReact (Euristica Valenza)...")
    file_copiati = 0
    file_salvati_da_valenza = 0
    
    elder_emomap = {
        1: 'Anger', 2: 'Disgust', 3: 'Fear', 
        4: 'Happy', 5: 'Sad', 6: 'Surprise'
    }
    
    splits = ['dev', 'test', 'train']
    
    for split in splits:
        txt_path = os.path.join(ELDER_TXT_DIR, f"{split}_labels.txt")
        if not os.path.exists(txt_path):
            print(f"Attenzione: File non trovato -> {txt_path}")
            continue
            
        with open(txt_path, 'r') as file:
            linee = file.readlines()
            
        for linea in tqdm(linee, desc=f"ElderReact ({split})"):
            parti = linea.strip().split()
            if len(parti) < 9: 
                continue
                
            video_id = parti[0].replace('.mp4', '')
            
            anger, disgust, fear, happy, sad, surprise = [int(float(x)) for x in parti[1:7]]
            emozioni_flags = [anger, disgust, fear, happy, sad, surprise]
            
            valenza = float(parti[8])
            
            emozione_scelta = None
            negative_attive = sum([anger, disgust, fear, sad])
            
            if sum(emozioni_flags) == 1:
                indice = emozioni_flags.index(1) + 1
                emozione_scelta = elder_emomap[indice]
                
            elif happy == 1 and valenza >= 5.0:
                emozione_scelta = 'Happy'
                file_salvati_da_valenza += 1
                
            elif negative_attive == 1 and valenza <= 3.5:
                if anger == 1: emozione_scelta = 'Anger'
                elif disgust == 1: emozione_scelta = 'Disgust'
                elif fear == 1: emozione_scelta = 'Fear'
                elif sad == 1: emozione_scelta = 'Sad'
                file_salvati_da_valenza += 1
            
            if emozione_scelta is not None:
                nome_frame = f"{video_id}_{FRAME_TARGET}.jpg"
                frame_origine = os.path.join(ELDER_DIR, split, video_id, nome_frame)
                
                if os.path.exists(frame_origine):
                    nuovo_nome = f"ElderReact_{split}_{video_id}_frame{FRAME_TARGET}.jpg"
                    
                    percorso_dest = os.path.join(BASE_SAGE_DIR, emozione_scelta, nuovo_nome)
                    
                    os.makedirs(os.path.dirname(percorso_dest), exist_ok=True)
                    shutil.copy(frame_origine, percorso_dest)
                    file_copiati += 1
                    
    print(f"ElderReact completato: estratti {file_copiati} frame.")
    print(f"Frame 'sporchi' salvati grazie all'euristica della Valenza: {file_salvati_da_valenza}")


def processa_ferplus():
    print("\nAvvio estrazione da FERPlus (Filtro DeepFace)...")
    file_copiati = 0
    file_scartati_giovani = 0
    
    FERPLUS_BASE_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\1.Dataset\FERPLUS"
    splits = ['train', 'validation', 'test']
    
    ferplus_emomap = {
        'angry': 'Anger',
        'contempt': 'Contempt', 
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happy',
        'neutral': 'Neutral',
        'sad': 'Sad',
        'surprise': 'Surprise'
    }


    for split in splits:
        percorso_split = os.path.join(FERPLUS_BASE_DIR, split)
        if not os.path.exists(percorso_split):
            continue
            
        cartelle_emozioni = [d for d in os.listdir(percorso_split) if os.path.isdir(os.path.join(percorso_split, d))]
        
        for emo_folder in cartelle_emozioni:
            if emo_folder.lower() not in ferplus_emomap:
                continue
                
            emozione_target = ferplus_emomap[emo_folder.lower()]
            percorso_cartella_origine = os.path.join(percorso_split, emo_folder)
            immagini = os.listdir(percorso_cartella_origine)
            
            for filename in tqdm(immagini, desc=f"FERPlus {split} -> {emozione_target}"):
                percorso_origine = os.path.join(percorso_cartella_origine, filename)
                
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
                        nuovo_nome = f"FERPlus_{split}_{emo_folder}_{filename}"
                        percorso_dest = os.path.join(BASE_SAGE_DIR, emozione_target, nuovo_nome)
                        
                        os.makedirs(os.path.dirname(percorso_dest), exist_ok=True)
                        shutil.copy(percorso_origine, percorso_dest)
                        file_copiati += 1
                    else:
                        file_scartati_giovani += 1
                        
                except Exception as e:
                    continue
                    
    print(f"FERPlus completato: estratti {file_copiati} soggetti anziani.")
    print(f"Scartati {file_scartati_giovani} soggetti giovani o non rilevati.")


if __name__ == "__main__":
    
    processa_dfew()
    processa_elderreact()
    processa_ferplus()
    
