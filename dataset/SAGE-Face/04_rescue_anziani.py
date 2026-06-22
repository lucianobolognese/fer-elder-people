import os
import shutil
import ollama
import re
from tqdm import tqdm


FALSI_POSITIVI_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\young_removed"
SAGE_FACE_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face"


PROMPT_RESCUE = """
Analyze this face carefully. We are double-checking a discarded image.
Estimate the probability (from 0 to 100) that this person is ACTUALLY middle-aged or elderly (approx. 45+ years old). 
Look specifically for signs of aging: wrinkles, loss of skin elasticity, and aging facial structure.

Reply STRICTLY with a single integer number between 0 and 100. Do not add any text, symbols, or explanations. Just the number.
"""


SOGLIA_CONFIDENZA = 80

def operazione_rescue():
    print("🚑 Avvio Operazione Rescue: Recupero Veri Positivi...")
    

    immagini_nel_cestino = []
    for root, _, files in os.walk(FALSI_POSITIVI_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                immagini_nel_cestino.append(os.path.join(root, file))
                
    if not immagini_nel_cestino:
        print("🤷‍♂️ Il cestino è vuoto. Nessun anziano da salvare!")
        return

    print(f"📦 Totale immagini nel cestino da valutare: {len(immagini_nel_cestino)}")
    anziani_salvati = 0


    with tqdm(immagini_nel_cestino, desc="Ripescaggio") as pbar:
        for img_path in pbar:
            try:

                response = ollama.chat(
                    model='qwen3-vl',
                    messages=[{
                        'role': 'user',
                        'content': PROMPT_RESCUE,
                        'images': [img_path]
                    }]
                )
                

                risposta_llm = response['message']['content'].strip()
                

                match = re.search(r'\d+', risposta_llm)
                if match:
                    score = int(match.group())
                    

                    if score >= SOGLIA_CONFIDENZA:
                        nome_file = os.path.basename(img_path)

                        cartella_emozione = os.path.basename(os.path.dirname(img_path)) 
                        
                        dest_dir = os.path.join(SAGE_FACE_DIR, cartella_emozione)

                        os.makedirs(dest_dir, exist_ok=True)
                        
                        shutil.move(img_path, os.path.join(dest_dir, nome_file))
                        anziani_salvati += 1
                        

                pbar.set_postfix({'Salvati (Score >= 80)': anziani_salvati})
                
            except Exception as e:

                continue
                
    print("\nOperazione Rescue completata!")
    print(f"Anziani 'Veri Positivi' ripescati e reintegrati: {anziani_salvati}")

if __name__ == "__main__":
    operazione_rescue()