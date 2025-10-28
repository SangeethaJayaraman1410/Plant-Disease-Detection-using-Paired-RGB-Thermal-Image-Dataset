import os
os.environ['HF_HOME'] = r"D:\hf_cache"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
import time
import timm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 150
batch_size = 32
learning_rate = 0.001

# Transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset class
class PairedRGBThermalDataset(Dataset):
    def __init__(self, rgb_root, thermal_root, transform=None):
        self.samples = []
        self.transform = transform

        for dataset in os.listdir(rgb_root):
            rgb_dataset_path = os.path.join(rgb_root, dataset)
            thermal_dataset_path = os.path.join(thermal_root, dataset)

            if not os.path.isdir(rgb_dataset_path) or not os.path.isdir(thermal_dataset_path):
                continue

            for class_name in os.listdir(rgb_dataset_path):
                rgb_class_path = os.path.join(rgb_dataset_path, class_name)
                thermal_class_path = os.path.join(thermal_dataset_path, class_name)

                if not os.path.isdir(rgb_class_path) or not os.path.isdir(thermal_class_path):
                    continue

                rgb_files = [f for f in os.listdir(rgb_class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                thermal_files = {f.lower(): f for f in os.listdir(thermal_class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))}

                for img_name in rgb_files:
                    base_name = os.path.splitext(img_name)[0].lower().strip()
                    target_name = f'thermal_{base_name}'
                    thermal_img_path = None
                    for ext in ['.jpg', '.jpeg', '.png']:
                        candidate = target_name + ext
                        if candidate in thermal_files:
                            thermal_img_path = os.path.join(thermal_class_path, thermal_files[candidate])
                            break

                    if thermal_img_path:
                        rgb_img_path = os.path.join(rgb_class_path, img_name)
                        self.samples.append((rgb_img_path, thermal_img_path, class_name))

        self.classes = sorted({label for _, _, label in self.samples})
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, thermal_path, label = self.samples[idx]
        rgb_image = Image.open(rgb_path).convert('RGB')
        thermal_image = Image.open(thermal_path).convert('RGB')

        if self.transform:
            rgb_image = self.transform(rgb_image)
            thermal_image = self.transform(thermal_image)

        return rgb_image, thermal_image, self.class_to_idx[label], os.path.basename(rgb_path)

# Vision Transformer Model
class ViTHallucinationModel(nn.Module):
def __init__(self, num_classes):
super(ViTHallucinationModel, self).__init__()
self.rgb_net = timm.create_model('vit_base_patch16_224', pretrained=True)
self.thermal_net = timm.create_model('vit_base_patch16_224', pretrained=True)

self.feat_dim = self.rgb_net.head.in_features
self.rgb_net.head = nn.Identity()
self.thermal_net.head = nn.Identity()

self.fusion_layer = nn.Linear(self.feat_dim * 2, self.feat_dim)
self.classifier = nn.Linear(self.feat_dim, num_classes)
self.aux_loss_fn = nn.MSELoss()

def forward(self, thermal_img, rgb_img=None):
thermal_feat = self.thermal_net(thermal_img)

if self.training and rgb_img is not None:
rgb_feat = self.rgb_net(rgb_img)
aux_loss = self.aux_loss_fn(thermal_feat, rgb_feat.detach())
fused_feat = torch.cat([thermal_feat, rgb_feat], dim=1)
fused_feat = self.fusion_layer(fused_feat)
out = self.classifier(fused_feat)
return out, aux_loss

out = self.classifier(thermal_feat)
return out

# Evaluation function
def evaluate(model, loader, device, class_names):
model.eval()
all_preds, all_labels = [], []
correct_images, incorrect_images = [], []

with torch.no_grad():
for rgb_imgs, thermal_imgs, labels, filenames in loader:
thermal_imgs, labels = thermal_imgs.to(device), labels.to(device)
outputs = model(thermal_imgs)
if isinstance(outputs, tuple):
outputs = outputs[0]
_, preds = torch.max(outputs, 1)

all_preds.extend(preds.cpu().numpy())
all_labels.extend(labels.cpu().numpy())

for i in range(len(filenames)):
if preds[i] == labels[i]:
correct_images.append(filenames[i])
else:
incorrect_images.append(filenames[i])

acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names)
return acc, report, correct_images, incorrect_images

# Main function
def main():
rgb_path = r'ENTER FOR RGB DATASET PATH'
thermal_path = r'ENTER FOR THERMAL DATASET PATH'
model_path = r"TRAINED WEIGHT PATH"
checkpoint_dir = os.path.dirname(model_path)

dataset = PairedRGBThermalDataset(rgb_path, thermal_path, transform=transform)
num_classes = len(dataset.classes)

model = ViTHallucinationModel(num_classes).to(device)
model = nn.DataParallel(model)

if os.path.exists(model_path):
model.load_state_dict(torch.load(model_path))
print("✅ Model loaded from disk.")
else:
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_start_time = time.time()
for epoch in range(num_epochs):
model.train()
total_loss, total_aux, correct, total = 0, 0, 0, 0
epoch_start = time.time()

for rgb, thermal, labels, _ in train_loader:
rgb, thermal, labels = rgb.to(device), thermal.to(device), labels.to(device)
outputs, aux_loss = model(thermal, rgb)
ce_loss = criterion(outputs, labels)
loss = ce_loss + aux_loss

optimizer.zero_grad()
loss.backward()
optimizer.step()

total_loss += ce_loss.item()
total_aux += aux_loss.item()
_, preds = torch.max(outputs, 1)
correct += (preds == labels).sum().item()
total += labels.size(0)

epoch_time = time.time() - epoch_start
print(f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_time:.2f}s | "
f"CE Loss: {total_loss/len(train_loader):.4f} | "
f"Aux Loss: {total_aux/len(train_loader):.4f} | "
f"Accuracy: {100*correct/total:.2f}%")

# Save checkpoint
checkpoint_path = os.path.join(checkpoint_dir, f"ViT_checkpoint_epoch_{epoch+1}.pth")
torch.save(model.state_dict(), checkpoint_path)

total_training_time = time.time() - training_start_time
print(f"\n🕒 Total Training Time: {total_training_time:.2f} seconds")

torch.save(model.state_dict(), model_path)
print(f"💾 Final model saved to: {model_path}")

eval_start = time.time()
acc, report, correct_imgs, wrong_imgs = evaluate(model, test_loader, device, dataset.classes)
eval_time = time.time() - eval_start
print("\n📊 Classification Report:\n", report)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
print(f"🕒 Evaluation Time: {eval_time:.2f} seconds")

# Optional: test unlabeled thermal images
user_input = input("Test on unlabeled thermal folder? (1-Yes/0-No): ")
if user_input.strip() == '1':
folder = input("Enter thermal image folder path: ")
model.eval()
counts = defaultdict(int)
total = 0

with torch.no_grad():
for fname in os.listdir(folder):
if fname.lower().endswith(('jpg', 'jpeg', 'png')):
try:
path = os.path.join(folder, fname)
img = Image.open(path).convert('RGB')
img = transform(img).unsqueeze(0).to(device)
out = model(img)
if isinstance(out, tuple):
out = out[0]
_, pred = torch.max(out, 1)
cls = dataset.classes[pred.item()]
print(f"{fname} => {cls}")
counts[cls] += 1
total += 1
except Exception as e:
print(f"Error on {fname}: {e}")

print("\n📁 Summary on Unlabeled Images:")
print(f"Total images: {total}")
for cls in dataset.classes:
print(f"{cls}: {counts[cls]}")

if __name__ == '__main__':
import torch.multiprocessing
torch.multiprocessing.freeze_support()
main()

