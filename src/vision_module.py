import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class MedicalVisionModule:
    def __init__(self, model_weights_path="./weights/cnn_pneumonia_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names = ["NORMAL", "PNEUMONIA"]

        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.model = PneumoniaCNN().to(self.device)

        if model_weights_path and os.path.exists(model_weights_path):
            self.model.load_state_dict(
                torch.load(model_weights_path, map_location=self.device)
            )
            print(f"[INFO] Loaded pneumonia CNN weights from {model_weights_path}")
        else:
            raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

        self.model.eval()

    def extract_finding(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        top_prob, top_class_idx = torch.max(probabilities, dim=0)
        finding_label = self.class_names[top_class_idx.item()]

        return {
            "modality_finding": finding_label,
            "confidence_score": round(top_prob.item(), 4),
        }


if __name__ == "__main__":
    vision = MedicalVisionModule()
    print("[INFO] Pneumonia Vision Module ready.")