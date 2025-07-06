import os
import shutil
from ultralytics import YOLO

# 1️⃣ Prepare dataset folders
def prepare_dataset(object_name):
    dataset_path = f"datasets/{object_name}"
    images_path = os.path.join(dataset_path, "images", "train")

    # Remove any previous data (clean training data)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    os.makedirs(images_path, exist_ok=True)

    # Move downloaded images from downloads/{object_name} to datasets/{object_name}/images/train
    downloaded_path = f"downloads/{object_name}"
    
    if not os.path.exists(downloaded_path):
        print("No downloaded images found! Please run image_scraper first.")
        return None
    
    for filename in os.listdir(downloaded_path):
        src = os.path.join(downloaded_path, filename)
        dst = os.path.join(images_path, filename)
        shutil.move(src, dst)

    print(f"Dataset prepared at {dataset_path}")
    return dataset_path

# 2️⃣ Generate data.yaml
def generate_yaml(object_name):
    yaml_path = f"datasets/{object_name}/data.yaml"
    
    yaml_content = f"""
path: datasets/{object_name}
train: images/train
val: images/train

nc: 1
names: ['{object_name}']
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"YAML file created at {yaml_path}")
    return yaml_path

# 3️⃣ Full Training Function
def train_custom_model(object_name):
    dataset_path = prepare_dataset(object_name)
    if dataset_path is None:
        print("Dataset preparation failed.")
        return

    yaml_path = generate_yaml(object_name)

    # Start training using YOLOv8n (light model for fast training)
    model = YOLO("yolov8n.pt")
    model.train(data=yaml_path, epochs=10, imgsz=640)

    print("Training complete!")
