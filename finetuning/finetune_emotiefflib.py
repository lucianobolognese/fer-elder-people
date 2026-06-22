import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from sklearn.metrics import f1_score
from tqdm import tqdm


TRAIN_DIR = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\SAGE-Face_FineTuning\train"
VAL_DIR   = r"C:\Users\ciano\Desktop\Tesi\Tesi_Benchmarking\SAGE-Face_FineTuning\val"
SAVE_MODEL_PATH = "emotiefflib_sageface_finetuned_2.pth"

BATCH_SIZE = 16
EPOCHS = 150          
LEARNING_RATE = 1e-5
PATIENCE = 12        


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


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Uso del device: {DEVICE}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Classi trovate: {train_dataset.classes}")
    print(f"Immagini di Training: {len(train_dataset)} | Validation: {len(val_dataset)}")


    fer = EmotiEffLibRecognizer(engine="torch", model_name="enet_b2_7", device=DEVICE)
    model = ReWiredEmotiEff(fer, DEVICE).to(DEVICE)

    optimizer = optim.AdamW([
        {'params': model.features_extractor.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history_epochs = []
    history_train_loss = []
    history_val_loss = []
    
    scaler = torch.amp.GradScaler('cuda') 

    best_val_f1 = 0.0
    epochs_no_improve = 0  

    print("\nInizio Training ottimizzato...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoca {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += inputs.size(0)
            
        scheduler.step()
        train_loss = train_loss / total_train
        train_acc = correct_train.double() / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoca {epoch+1}/{EPOCHS} [Val]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += inputs.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        val_loss = val_loss / total_val
        val_acc = correct_val.double() / total_val
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        print(f"-> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"-> Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1-Score: {val_f1:.4f}")

        history_epochs.append(epoch + 1)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0  
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"[*] Nuovo modello migliore salvato! (F1-Score: {best_val_f1:.4f})\n")
        else:
            epochs_no_improve += 1
            print(f"[!] Nessun miglioramento per {epochs_no_improve} epoche consecutive.\n")
            
            if epochs_no_improve >= PATIENCE:
                print(f"=== EARLY STOPPING ATTIVATO all'epoca {epoch+1} ===")
                print("Il modello ha smesso di imparare pattern utili. Addestramento interrotto per prevenire l'overfitting.")
                break

    print(f"\nFine-Tuning completato. Miglior F1-Score sul Validation: {best_val_f1:.4f}")
    print("Pesi finali salvati in:", SAVE_MODEL_PATH)

    csv_path = "loss_history.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for e, t, v in zip(history_epochs, history_train_loss, history_val_loss):
            writer.writerow([e, t, v])
    print(f"Storico delle loss salvato con successo in: {csv_path}")


if __name__ == '__main__':
    main()