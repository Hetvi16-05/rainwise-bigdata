import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import prepare_data
from models import FloodDNN
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data/processed/training_dataset_gujarat_advanced_labeled.csv"
MODEL_SAVE_PATH = "DLmodels/flood_dnn.pth"
SCALER_SAVE_PATH = "DLmodels/scaler.pkl"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train():
    # 1. Load Data
    print(f"📂 Loading data from {DATA_PATH}...")
    train_loader, test_loader, scaler, input_dim = prepare_data(DATA_PATH, batch_size=BATCH_SIZE)
    
    # 2. Initialize Model
    model = FloodDNN(input_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    print(f"🚀 Starting training on {DEVICE}...")
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        test_acc = correct / total
        
        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(test_acc)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")
            
        # Save Best Model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            joblib.dump(scaler, SCALER_SAVE_PATH)
            
    print(f"✅ Training Complete. Best Model Saved to {MODEL_SAVE_PATH}")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.title("Loss Curves")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["test_acc"], label="Test Accuracy", color="green")
    plt.title("Accuracy Curve")
    plt.legend()
    
    os.makedirs("outputs/dl", exist_ok=True)
    plt.savefig("outputs/dl/training_curves.png")
    print("📈 Training curves saved to outputs/dl/training_curves.png")

if __name__ == "__main__":
    train()
