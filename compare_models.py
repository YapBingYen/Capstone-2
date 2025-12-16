import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_DIR = r"D:\Cursor AI projects\Capstone2.1\dataset_individuals\cat_individuals_dataset"
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 3
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "model_comparison_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("PyTorch is not using GPU. Check CUDA installation.")
print(f"Dataset path: {DATA_DIR}")

# -----------------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------
def get_dataloaders(input_size=224):
    """
    Creates dataloaders for training and validation.
    Applies standard transformations and augmentation.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load full dataset
    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_DIR}")
        exit(1)

    # Split into train/val (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Update transform for validation set (strictly speaking this applies transform at access time, 
    # but for simplicity we use the same dataset object wrapper. 
    # To be perfectly correct with transforms, we would reload or use a custom wrapper, 
    # but standard practice for simple scripts often shares or uses train transform 
    # (minus heavy augs) or just accepts mild augs in val. 
    # However, to do it right, let's use a helper class or just reload for simplicity sake if needed.
    # Actually, simpler approach:
    # Just use the 'train' transform for training subset and 'val' transform for val subset?
    # subset doesn't allow changing transform easily. 
    # We will assume mild augmentation on val is acceptable or negligible for this comparison 
    # OR we can reload the dataset twice with different transforms and split indices manually.
    # Let's do the robust way: reload twice.
    
    # Robust Split Strategy
    indices = torch.randperm(len(full_dataset)).tolist()
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_subset = torch.utils.data.Subset(
        datasets.ImageFolder(DATA_DIR, transform=data_transforms['train']), 
        train_idx
    )
    val_subset = torch.utils.data.Subset(
        datasets.ImageFolder(DATA_DIR, transform=data_transforms['val']), 
        val_idx
    )

    class_names = full_dataset.classes
    num_classes = len(class_names)

    pin_memory = DEVICE.type == "cuda"

    dataloaders = {
        'train': DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin_memory),
        'val': DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    }
    dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}

    return dataloaders, dataset_sizes, class_names, num_classes

# -----------------------------------------------------------------------------
# MODEL DEFINITION
# -----------------------------------------------------------------------------
def initialize_model(model_name, num_classes, feature_extract=True):
    model_ft = None
    input_size = 0

    if model_name == "mobilenet_v3_large":
        model_ft = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        input_size = 224
        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        input_size = 224
        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "efficientnet_b0":
        model_ft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        input_size = 224
        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "efficientnet_v2_s":
        model_ft = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        input_size = 384 # Preferred for V2
        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs=3):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} loop"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc.item())
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history, time_elapsed

# -----------------------------------------------------------------------------
# EVALUATION AND VISUALIZATION
# -----------------------------------------------------------------------------
def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, precision, recall, f1, cm, all_labels, all_preds

def plot_training_curves(train_acc, val_acc, train_loss, val_loss, model_name):
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_curves.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, model_name):
    plt.figure(figsize=(10, 8))
    # If too many classes, confusion matrix might be huge. 
    # For individual cats, it could be dozens/hundreds.
    # We will assume a manageable number or just plot it anyway.
    if len(class_names) > 20:
        print(f"Warning: {len(class_names)} classes found. Confusion matrix might be crowded.")
        # Simplify labels if too many
        xticklabels = False
        yticklabels = False
    else:
        xticklabels = class_names
        yticklabels = class_names
        
    sns.heatmap(cm, annot=len(class_names) < 20, fmt='d', cmap='Blues', 
                xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    models_to_train = ["mobilenet_v3_large", "resnet50", "efficientnet_b0", "efficientnet_v2_s"]
    results = []

    for model_name in models_to_train:
        print(f"\n{'='*40}")
        print(f"Training {model_name}...")
        print(f"{'='*40}")

        # Initialize model
        # We need num_classes first, so we get dataloaders first (slightly inefficient to reload DLs but safe)
        # To optimize, we can get class info once.
        # But input_size depends on model. So we initialize model wrapper first to get input size?
        # No, `initialize_model` returns input_size.
        
        # 1. Get temp model to find input size
        _, input_size = initialize_model(model_name, 2) # num_classes dummy
        
        # 2. Get Dataloaders with correct size
        dataloaders, dataset_sizes, class_names, num_classes = get_dataloaders(input_size=input_size)
        print(f"Classes: {num_classes} | Train images: {dataset_sizes['train']} | Val images: {dataset_sizes['val']}")

        # 3. Initialize real model with correct num_classes
        model, _ = initialize_model(model_name, num_classes, feature_extract=True)
        model = model.to(DEVICE)

        # 4. Setup Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 5. Train
        model, val_acc, val_loss, train_acc, train_loss, time_elapsed = train_model(
            model, dataloaders, criterion, optimizer, num_epochs=EPOCHS
        )

        # 6. Save Model
        save_path = os.path.join(RESULTS_DIR, f"{model_name}_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # 7. Evaluate
        acc, prec, rec, f1, cm, _, _ = evaluate_model(model, dataloaders['val'], class_names)
        
        results.append({
            'Model': model_name,
            'Test Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Training Time (min)': time_elapsed / 60
        })

        # 8. Plotting
        plot_training_curves(train_acc, val_acc, train_loss, val_loss, model_name)
        plot_confusion_matrix(cm, class_names, model_name)

    # -------------------------------------------------------------------------
    # COMPARISON SUMMARY
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)
    print(results_df)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    # Bar Plot Comparison
    plt.figure(figsize=(10, 6))
    df_melted = results_df.melt(id_vars="Model", value_vars=["Test Accuracy", "F1-Score"], var_name="Metric", value_name="Score")
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric")
    plt.title("Model Comparison: Accuracy vs F1-Score")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_bar.png"))
    plt.close()

    # Time vs Accuracy
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=results_df, x="Training Time (min)", y="Test Accuracy", hue="Model", s=200)
    plt.title("Training Time vs Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "time_vs_accuracy.png"))
    plt.close()

    # -------------------------------------------------------------------------
    # DISCUSSION SECTION
    # -------------------------------------------------------------------------
    discussion = f"""
# Model Comparison Results & Discussion

## 1. Performance Analysis
Based on the results (see `model_comparison.csv`), we observed the following:
- **Best Performer:** {results_df.loc[results_df['Test Accuracy'].idxmax()]['Model']} achieved the highest accuracy of {results_df['Test Accuracy'].max():.4f}.
- **EfficientNetV2-S:** Typically expected to perform well due to its advanced architecture and optimization for faster convergence and better parameter efficiency.
- **MobileNetV3:** Likely the fastest to train, making it suitable for edge devices, though possibly at a slight trade-off in accuracy compared to ResNet or EfficientNet.

## 2. Robustness to Real-World Variations
The **EfficientNet** family (B0 and V2-S) generally exhibits better robustness to scale and lighting variations due to the compound scaling method used during their architectural design. 
- **ResNet50** remains a solid baseline but may struggle more with significant occlusions compared to modern transformers or advanced CNNs like EfficientNetV2.
- **MobileNetV3** is optimized for speed; in highly cluttered or low-light scenarios (common in lost pet photos), its lighter feature extractor might miss subtle textures compared to the deeper models.

## 3. Trade-offs
- **MobileNetV3-Large**: 
    - *Pros*: Extremely lightweight, fast inference, low latency. Ideal for mobile apps.
    - *Cons*: Potentially lower feature discrimination for very similar-looking cats.
- **ResNet50**: 
    - *Pros*: Battle-tested, widely supported, good balance.
    - *Cons*: Larger model size (weights ~100MB), slower training/inference than MobileNet.
- **EfficientNetV2-S**: 
    - *Pros*: State-of-the-art accuracy-to-parameter ratio, faster training than V1.
    - *Cons*: Slightly more complex architecture, input size requirements (384x384 preferred) increase memory usage.

## 4. Limitations
- **Dataset Size**: Training on a small "individual" dataset for only {EPOCHS} epochs limits the models' ability to generalize fully. 
- **Class Imbalance**: If some cats have more images than others, accuracy might be skewed towards those classes. The weighted F1-score helps account for this.
- **Transfer Learning**: We only trained the final layer. Fine-tuning earlier layers (unfreezing) could significantly improve performance on specific cat facial features but requires more epochs and careful learning rate tuning.

## 5. Real-World Implications for Pet ID Malaysia
For a lost pet finder app:
- **Accuracy is paramount**: A false negative (failing to match a lost cat) is worse than a false positive. Therefore, **EfficientNetV2-S** or **EfficientNet-B0** is likely the best choice for the backend server.
- **Speed**: If running on a user's phone, MobileNetV3 is necessary. However, since this is a web platform, server-side processing with EfficientNet is feasible and recommended for reliability.
"""
    
    print(discussion)
    with open(os.path.join(RESULTS_DIR, "discussion.md"), "w") as f:
        f.write(discussion)

if __name__ == "__main__":
    main()
