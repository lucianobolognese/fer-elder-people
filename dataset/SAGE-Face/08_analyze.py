import os
from collections import defaultdict


BASE_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\SAGE-Face_FineTuning\Complete"

EMOTIONS = [
    "angry", "contempt", "disgust", "fear", 
    "happy", "neutral", "sad", "surprise"
]

TARGET_SOURCES = [
    "AffectNet", "DFEW", "ElderReact", 
    "FACES", "FERPlus", "RAFDB"
]

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')


total_images = 0
emotion_counts = defaultdict(int)


source_counts = {source: 0 for source in TARGET_SOURCES}
other_sources_count = 0 


print(f"Inizio scansione della directory: {BASE_DIR}\n")

for emotion in EMOTIONS:
    folder_path = os.path.join(BASE_DIR, emotion)
    
    if not os.path.exists(folder_path):
        print(f"[ATTENZIONE] La cartella {emotion} non esiste in questo percorso.")
        continue
        
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(VALID_EXTENSIONS):
            total_images += 1
            emotion_counts[emotion] += 1
            
            source_prefix = filename.split('_')[0]
            )
            matched = False
            for target in TARGET_SOURCES:
                if source_prefix.lower() == target.lower():
                    source_counts[target] += 1
                    matched = True
                    break
                    
            if not matched:
                other_sources_count += 1


print("-" * 40)
print(" REPORT SAGE-FACE")
print("-" * 40)
print(f"Totale immagini trovate: {total_images}\n")

print("--- DISTRIBUZIONE PER EMOZIONE ---")
for emotion in EMOTIONS:
    count = emotion_counts[emotion]
    percentage = (count / total_images * 100) if total_images > 0 else 0
    print(f"{emotion.capitalize():<10}: {count:>6} immagini ({percentage:>5.1f}%)")

print("\n--- DISTRIBUZIONE PER DATASET DI ORIGINE ---")
for source in TARGET_SOURCES:
    count = source_counts[source]
    percentage = (count / total_images * 100) if total_images > 0 else 0
    print(f"{source:<12}: {count:>6} immagini ({percentage:>5.1f}%)")

if other_sources_count > 0:
    print(f"\n[NOTA] Trovate {other_sources_count} immagini con un prefisso non presente nella tua lista.")