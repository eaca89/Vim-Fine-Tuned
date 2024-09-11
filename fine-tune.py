import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from datasets import DatasetDict
from datasets import DatasetDict, Dataset, Features, ClassLabel, Value, Array3D
from transformers import Trainer, TrainingArguments, AutoModelForImageClassification, AutoFeatureExtractor
from pathlib import PurePath
import torch
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import PurePath
from huggingface_hub import snapshot_download
import argparse

# Get the absolute path of the vim directory
vim_path = os.path.abspath('vim')
# Add the vim directory to the system path
sys.path.append(vim_path)
from vim.models_mamba import VisionMamba

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms



# Create a parser object
parser = argparse.ArgumentParser(description='Fine-tune Vision Mamba Model')

# Add arguments
parser.add_argument('--train-folder', type=str, required=True, help='Path to the training dataset')
parser.add_argument('--validation-folder', type=str, required=True, help='Path to the validation dataset')
parser.add_argument('--test-folder', type=str, required=True, help='Path to the test dataset')
parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning-rate', type=float, default=5e-6, help='Learning rate for optimizer')

# Parse arguments
args = parser.parse_args()

# Function to get all image files from a directory
def get_image_files(folder):
    files = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(root, filename))
    return files

# Custom Dataset Class for Binary Classification


class CustomBinaryDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.data = []
        self.class_names = ['Full', 'Partial']  # Explicitly define class names
        self.load_images()

    def load_images(self):
        for class_name in self.class_names:
            class_folder = os.path.join(self.folder, class_name)
            if os.path.isdir(class_folder):
                class_index = self.class_names.index(
                    class_name)  # 0 for 'Full', 1 for 'Partial'
                image_files = get_image_files(class_folder)
                for image_file in image_files:
                    self.data.append({
                        'image': image_file,
                        'label': class_index,  # Binary label: 0 for 'Full', 1 for 'Partial'
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]['image']
        label = self.data[idx]['label']

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# Image transformations (Resizing and Normalizing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset Folders
# train_folder = '/home/eh_abdol/fine_tune/gold3/train'
# validation_folder = '/home/eh_abdol/fine_tune/gold3/validation'
# test_folder = '/home/eh_abdol/fine_tune/gold3/test'

# Dataset Folders
train_folder = args.train_folder
validation_folder = args.validation_folder
test_folder = args.test_folder

# Create Datasets
train_dataset = CustomBinaryDataset(train_folder, transform=transform)
validation_dataset = CustomBinaryDataset(
    validation_folder, transform=transform)
test_dataset = CustomBinaryDataset(test_folder, transform=transform)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(
    validation_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Now you can use test_loader to evaluate the model after fine-tuning

# Load Vision Mamba architecture with modified output layer (2 classes)
model = VisionMamba(
    patch_size=args.batch_size,
    stride=8,
    embed_dim=384,
    depth=24,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    final_pool_type='mean',
    if_abs_pos_embed=True,
    if_rope=False,
    if_rope_residual=False,
    bimamba_type="v2",
    if_cls_token=True,
    if_devide_out=True,
    use_middle_cls_token=True,
    num_classes=2,  # Update for binary classification
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224,
)

# Download the pretrained weights from Hugging Face Hub
VIM_REPO = "hustvl/Vim-small-midclstok"
pretrained_model_dir = snapshot_download(
    repo_id=VIM_REPO, local_files_only=True)

# Load the pretrained weights
MODEL_FILE = PurePath(pretrained_model_dir, "vim_s_midclstok_ft_81p6acc.pth")
print(MODEL_FILE)

# Load the checkpoint
checkpoint = torch.load(str(MODEL_FILE), map_location='cpu')

# Remove the 'head.weight' and 'head.bias' from the pretrained checkpoint
pretrained_dict = checkpoint['model']
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if not k.startswith('head')}

# Load the pretrained weights (excluding the final classification layer)
model.load_state_dict(pretrained_dict, strict=False)

# Initialize a new classification head for binary classification (2 classes)
model.head = nn.Linear(in_features=384, out_features=2)

# Move the model to the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)  # 1e-4

def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)

    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track accuracy and loss
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validation after every epoch
        val_acc = validate_model(model, validation_loader, device)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        val_acc = validate_model(model, validation_loader, device)

        # Store metrics
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {100.*correct/total:.2f}%, Validation Accuracy: {val_acc:.2f}%")

    print('Finished Training')

    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_accs)

    # Save the trained model's weights
    # torch.save(model.state_dict(), 'vision_mamba_finetuned.pth')
    # print("Model weights saved successfully.")


def validate_model(model, validation_loader, device='cuda'):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    # disables gradient tracking during the validation process
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            # finds the index of the maximum value in each row of the output tensor,
            # which corresponds to the class prediction.
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def plot_training_curves(train_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


# Fine-tune the model
train_model(model, train_loader, validation_loader,
            criterion, optimizer, num_epochs=10)

from sklearn.metrics import classification_report


def test_model(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    return accuracy


# Test the model
test_model(model, test_loader)  # Assuming 10 classes