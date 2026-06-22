import os
import random
import shutil
from pathlib import Path

# Configurazione dei percorsi
# Cartella originale con le 8 emozioni
source_dir = Path(r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224")

# Nuova cartella dove verrà salvato il dataset diviso
dest_dir = Path(r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224_Split")

# Le tue classi (emozioni)
classes = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Percentuale di split (80% train, 20% test)
train_ratio = 0.80

# Estensioni da considerare
valid_extensions = {".jpg", ".jpeg", ".png"}

# Seed per la riproducibilità (se rilanci lo script, farà lo stesso split)
random.seed(42)

print("Inizio la stratificazione del dataset (80/20)...\n")

# Dizionari per tenere traccia dei numeri per il report finale
report_train = {}
report_test = {}

for emotion in classes:
    source_emotion_dir = source_dir / emotion
    
    if not source_emotion_dir.exists():
        print(f"⚠️ ATTENZIONE: La cartella {source_emotion_dir} non esiste. Salto...")
        continue
        
    # Raccogli tutti i file immagine per questa emozione
    files = [f for f in source_emotion_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    # Mischia i file in modo casuale
    random.shuffle(files)
    
    # Calcola l'indice di taglio
    split_index = int(len(files) * train_ratio)
    
    # Dividi le liste
    train_files = files[:split_index]
    test_files = files[split_index:]
    
    # Crea le cartelle di destinazione per train e test
    train_dest_dir = dest_dir / "train" / emotion
    test_dest_dir = dest_dir / "test" / emotion
    
    train_dest_dir.mkdir(parents=True, exist_ok=True)
    test_dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processando '{emotion}': {len(train_files)} a Train, {len(test_files)} a Test...")
    
    # Copia i file di training
    for f in train_files:
        shutil.copy2(f, train_dest_dir / f.name)
        
    # Copia i file di test
    for f in test_files:
        shutil.copy2(f, test_dest_dir / f.name)
        
    
    report_train[emotion] = len(train_files)
    report_test[emotion] = len(test_files)

print("\n" + "="*40)
print("REPORT FINALE DELLO SPLIT 80/20")
print("="*40)
print(f"{'Emozione':<15} | {'Train (80%)':<12} | {'Test (20%)'}")
print("-" * 40)

total_train = sum(report_train.values())
total_test = sum(report_test.values())

for emotion in classes:
    if emotion in report_train:
        print(f"{emotion:<15} | {report_train[emotion]:<12} | {report_test[emotion]}")

print("-" * 40)
print(f"{'TOTALE':<15} | {total_train:<12} | {total_test}")
print(f"Totale complessivo immagini: {total_train + total_test}")
print("="*40)
print(f"\nOperazione completata! Il nuovo dataset si trova qui: {dest_dir}")