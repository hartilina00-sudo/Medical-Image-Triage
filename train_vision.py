import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import Dataset
from vision_module import MedicalVisionModule

def prepare_data_paths(data_dir="./data/mednist/MedNIST"):
    """Scans the directory structure to collect image file paths and class labels."""
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    image_files = []
    image_labels = []

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        class_files = glob.glob(os.path.join(class_dir, "*.jpeg")) + glob.glob(os.path.join(class_dir, "*.jpg"))
        image_files.extend(class_files)
        image_labels.extend([idx] * len(class_files))

    return image_files, image_labels, class_names

# FIX: Moved the Dataset class OUTSIDE the function so Windows can pickle it
class MedNISTDataset(Dataset):
    def __init__(self, indices, files, labels, transform):
        self.indices = indices
        self.files = files
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, index):
        idx = self.indices[index]
        return self.transform(self.files[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

def train_model(epochs=3, batch_size=64, learning_rate=1e-4):
    vision = MedicalVisionModule()
    
    image_files, image_labels, _ = prepare_data_paths()
    num_samples = len(image_files)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split = int(num_samples * 0.8)
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_ds = MedNISTDataset(train_indices, image_files, image_labels, vision.transforms)
    val_ds = MedNISTDataset(val_indices, image_files, image_labels, vision.transforms)
    
    # FIX: Set num_workers=0 and pin_memory=False for Windows CPU stability
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vision.model.parameters(), lr=learning_rate)
    
    best_accuracy = 0.0
    weights_dir = "./weights"
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, "densenet_mednist.pth")

    print(f"\n[START] Training DenseNet121 on {vision.device}...")
    
    for epoch in range(epochs):
        vision.model.train()
        epoch_loss = 0
        correct_train = 0
        
        # Adding a simple print to show batch progress since CPU training takes longer
        print(f"Starting Epoch {epoch+1}/{epochs}...")
        
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            inputs, labels = batch_data.to(vision.device), batch_labels.to(vision.device)
            
            optimizer.zero_grad()
            outputs = vision.model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()

            # Print an update every 50 batches so you know it hasn't frozen
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} processed.")
            
        train_acc = correct_train / len(train_ds)
        
        vision.model.eval()
        correct_val = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                inputs, labels = batch_data.to(vision.device), batch_labels.to(vision.device)
                outputs = vision.model(inputs)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                
        val_acc = correct_val / len(val_ds)
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} Results - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(vision.model.state_dict(), model_path)
            print(f" => Saved new best weights checkpoint!")

    print(f"\n[FINISHED] Training complete. Best Validation Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    train_model(epochs=3, batch_size=64)