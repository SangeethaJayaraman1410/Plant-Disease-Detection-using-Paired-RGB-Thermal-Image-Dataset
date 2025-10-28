import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from PIL import Image, UnidentifiedImageError
from models import GeneratorResNet  # Import your trained generator class

# ----------- CONFIG -----------
input_folder = r"GIVE RGB DATASET PATH"
output_folder = r"STORED LOCATION PATH"
model_path = r"TRAINED WEIGHT PATH"
img_height, img_width, channels = 224, 224, 3
n_residual_blocks = 9
use_cuda = torch.cuda.is_available()

os.makedirs(output_folder, exist_ok=True)

# ----------- Load Generator -----------
input_shape = (channels, img_height, img_width)
generator = GeneratorResNet(input_shape, n_residual_blocks)

# Ensure model is loaded correctly across devices
state_dict = torch.load(model_path, map_location=torch.device('cuda' if use_cuda else 'cpu'))
generator.load_state_dict(state_dict)
generator.eval()
if use_cuda:
    generator.cuda()

# ----------- Image Transform -----------
transform = transforms.Compose([
transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
transforms.CenterCrop((img_height, img_width)),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ----------- Inference -----------
image_filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

failed_images = []

for img_name in image_filenames:
img_path = os.path.join(input_folder, img_name)

try:
image = Image.open(img_path).convert("RGB")
except (UnidentifiedImageError, OSError) as e:
print(f"[ERROR] Skipping corrupted or unreadable image: {img_name} ({e})")
failed_images.append(img_name)
continue

image_tensor = transform(image).unsqueeze(0)
image_tensor = Variable(image_tensor)
if use_cuda:
image_tensor = image_tensor.cuda()

with torch.no_grad():
fake_image = generator(image_tensor)

# Denormalize and save
fake_image = 0.5 * (fake_image.data + 1.0)  # Rescale to [0,1]
out_path = os.path.join(output_folder, f"thermal_{img_name}")
try:
save_image(fake_image, out_path)
except Exception as e:
print(f"[ERROR] Could not save image: {out_path} ({e})")
failed_images.append(img_name)

# ----------- Summary -----------
print(f"\n[INFO] Total input images: {len(image_filenames)}")
print(f"[INFO] Successfully processed: {len(image_filenames) - len(failed_images)}")
print(f"[INFO] Failed or skipped: {len(failed_images)}")

if failed_images:
print("\n[WARNING] The following images were skipped or failed:")
for fname in failed_images:
print(f" - {fname}")

