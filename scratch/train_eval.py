import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Deeper3DCNN
from dataset import BrainDataset

def train_and_evaluate(scan_paths, labels, manual_threshold=0.45, epochs=5, batch_size=2):
    X_train, X_val, y_train, y_val = train_test_split(
        scan_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = BrainDataset(X_train, y_train)
    val_dataset = BrainDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Deeper3DCNN().to(device)

    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {np.mean(losses):.4f}")

    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            output = model(x)
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > manual_threshold).astype(int)
            all_preds.extend(preds)
            all_targets.extend(y.numpy().astype(int))
            all_probs.extend(probs)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)
    cr = classification_report(all_targets, all_preds)

    print("\n📊 Validation Metrics:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC:       {auc:.3f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print("\n🔍 Sample prediction probabilities:")
    print(all_probs[:10])

    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    plt.hist([p for p, t in zip(all_probs, all_targets) if t == 0], bins=20, alpha=0.5, label='Group 2 (0)')
    plt.hist([p for p, t in zip(all_probs, all_targets) if t == 1], bins=20, alpha=0.5, label='Group 1 (1)')
    plt.legend()
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.savefig("probability_distribution.png")
    plt.close()
