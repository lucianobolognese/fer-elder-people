import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


FULL_DATASET_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\SAGE-Face_FineTuning\Complete"

BATCH_SIZE = 32
EPOCHS = 150          
LEARNING_RATE = 1e-5
PATIENCE = 10
K_FOLDS = 5


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

def get_fresh_model(device):
    """Garantisce che ogni Fold parta con gli stessi pesi pre-addestrati originali, senza inquinamenti"""
    fer = EmotiEffLibRecognizer(engine="torch", model_name="enet_b2_7", device=device)
    model = ReWiredEmotiEff(fer, device).to(device)
    return model


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Uso del device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    full_dataset = datasets.ImageFolder(FULL_DATASET_DIR, transform=transform)
    labels = full_dataset.targets

    print(f"Dataset Totale caricato: {len(full_dataset)} immagini.")
    print(f"Classi trovate: {full_dataset.classes}")

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results_f1 = []
    fold_results_acc = []

    print("\n" + "="*50)
    print(f"INIZIO {K_FOLDS}-FOLD CROSS VALIDATION")
    print("="*50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- INIZIO FOLD {fold+1}/{K_FOLDS} ---")
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2, pin_memory=True)

        model = get_fresh_model(DEVICE)
        
        optimizer = optim.AdamW([
            {'params': model.features_extractor.parameters(), 'lr': LEARNING_RATE * 0.1},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda') 

        best_val_f1 = 0.0
        best_val_acc = 0.0
        epochs_no_improve = 0  
        save_path = f"emotiefflib_sageface_FOLD_{fold+1}.pth"

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for inputs, targets in tqdm(train_loader, desc=f"Fold {fold+1} - Ep {epoch+1}/{EPOCHS} [Train]", leave=False):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_train += torch.sum(preds == targets.data)
                total_train += inputs.size(0)
                
            scheduler.step()
            train_acc = correct_train.double() / total_train
            
            model.eval()
            correct_val = 0
            total_val = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    correct_val += torch.sum(preds == targets.data)
                    total_val += inputs.size(0)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
            val_acc = correct_val.double() / total_val
            val_f1 = f1_score(all_targets, all_preds, average='macro')
            
            print(f"   -> Epoca {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc.item()
                epochs_no_improve = 0  
                torch.save(model.state_dict(), save_path)
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= PATIENCE:
                print(f"    -> Early Stopping Fold {fold+1} all'epoca {epoch+1}. Miglior F1: {best_val_f1:.4f}")
                break
                
        print(f"[*] Risultati Fold {fold+1}: Accuracy = {best_val_acc:.4f} | F1-Score = {best_val_f1:.4f}")
        fold_results_acc.append(best_val_acc)
        fold_results_f1.append(best_val_f1)

    print("\n" + "="*50)
    print("RISULTATI FINALI 5-FOLD CROSS VALIDATION")
    print("="*50)
    
    report_lines = []
    for i in range(K_FOLDS):
        line = f"Fold {i+1}: Acc: {fold_results_acc[i]:.4f} | F1: {fold_results_f1[i]:.4f}"
        print(line)
        report_lines.append(line)
        
    mean_acc = np.mean(fold_results_acc)
    std_acc = np.std(fold_results_acc)
    mean_f1 = np.mean(fold_results_f1)
    std_f1 = np.std(fold_results_f1)
    
    final_acc_str = f"Accuracy Globale : {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%"
    final_f1_str  = f"F1-Score (Macro) : {mean_f1:.4f} ± {std_f1:.4f}"
    
    print("\n--- METRICHE ACCADEMICHE DA INSERIRE NELLA TESI ---")
    print(final_acc_str)
    print(final_f1_str)
    print("="*50)

    report_path = "report_5fold_sageface.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== REPORT FINALE 5-FOLD CROSS VALIDATION ===\n\n")
        f.write("Risultati per singolo Fold:\n")
        for line in report_lines:
            f.write("- " + line + "\n")
        f.write("\n" + "-"*40 + "\n")
        f.write("METRICHE ACCADEMICHE FINALI\n")
        f.write("-" * 40 + "\n")
        f.write(final_acc_str + "\n")
        f.write(final_f1_str + "\n")
        
    print(f"\n[*] Il report testuale è stato salvato al sicuro in: {report_path}")

if __name__ == '__main__':
    main()