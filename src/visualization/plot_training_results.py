import matplotlib.pyplot as plt
import os

# Data extracted from training logs
epochs = [1, 5, 10, 15, 20, 25]
train_mse = [1.6181, 0.8888, 0.8249, 0.7984, 0.7599, 0.7629]
test_mse = [0.4328, 0.3050, 0.9570, 0.5720, 1.0986, 0.6481]

plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(epochs, train_mse, label="Training Loss (MSE)", marker='o', color='#2ecc71', linewidth=2)
plt.plot(epochs, test_mse, label="Validation Loss (MSE)", marker='s', color='#e74c3c', linewidth=2)

# Highlight the best point (Epoch 5)
plt.annotate('BEST MODEL (0.305)', 
             xy=(5, 0.305), xytext=(8, 0.15),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, fontweight='bold', color='#c0392b')

plt.scatter(5, 0.305, color='#c0392b', s=200, edgecolors='white', zorder=5)

# Aesthetics
plt.title("RAINWISE: TabTransformer Training Curve", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)

# Log scale for clarity since error is small
plt.yscale('log')

os.makedirs("outputs/dl", exist_ok=True)
save_path = "outputs/dl/tab_transformer_curves.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"✅ Training curve successfully reconstructed and saved to {save_path}")
