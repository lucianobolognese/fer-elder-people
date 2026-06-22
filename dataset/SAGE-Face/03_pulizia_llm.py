import os
import shutil
import ollama
from tqdm import tqdm


SAGE_FACE_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\SAGE-Face_224"


FALSI_POSITIVI_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\5_SAGE-face\young_removed_new"
os.makedirs(FALSI_POSITIVI_DIR, exist_ok=True)

# Il Prompt
PROMPT = """
Analyze this face carefully. 
Your task is to identify ONLY people who are unmistakably elderly (over 60 years old). 
Look for undeniable signs of advanced aging: deep wrinkles, severe loss of skin elasticity, sagging skin, and prominent age spots.

Is this person OBVIOUSLY elderly? 
Reply strictly with one single word: "YES" (ONLY if they are undeniably over 60 with clear, obvious signs of aging) or "NO" (if they are children, young adults, middle-aged, or if you are in ANY doubt). 
Do not add any other text, explanation, or punctuation.
"""

# modelli da provare   qwen3-vl   llama3.2-vision

def pulizia_con_llm():
    print("🤖 Avvio Revisore LLAVA per la pulizia dei Falsi Positivi...")
    
    PREFISSI_SICURI = ('ElderReact_', 'FACES_')
    
    immagini_da_testare = []
    for root, _, files in os.walk(SAGE_FACE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                

                if file.startswith(PREFISSI_SICURI):
                    continue
                    
                immagini_da_testare.append(os.path.join(root, file))
                
    print(f"📦 Totale immagini in-the-wild da revisionare: {len(immagini_da_testare)}")
    falsi_positivi_trovati = 0

    with tqdm(immagini_da_testare, desc="Integrità SAGE-Face") as pbar:
        for img_path in pbar:
            try:

                response = ollama.chat(
                    model='qwen3-vl:8b-instruct',
                    messages=[{
                        'role': 'user',
                        'content': PROMPT,
                        'images': [img_path]
                    }]
                )
                
                risposta_llm = response['message']['content'].strip().upper()
                

                if "NO" in risposta_llm:

                    nome_file = os.path.basename(img_path)
                    cartella_emozione = os.path.basename(os.path.dirname(img_path)) 
                    
                    dest_dir = os.path.join(FALSI_POSITIVI_DIR, cartella_emozione)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    shutil.move(img_path, os.path.join(dest_dir, nome_file))
                    falsi_positivi_trovati += 1
                    
                pbar.set_postfix({'Scartati (Giovani)': falsi_positivi_trovati})
                
            except Exception as e:
                continue
            
    print("\nRevisione completata!")
    print(f"Falsi positivi rimossi e spostati: {falsi_positivi_trovati}")

if __name__ == "__main__":
    pulizia_con_llm()