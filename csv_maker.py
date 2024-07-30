import os
import json
import csv
import random

# Set the base path to your dataset folder
base_path = './dataset/images/raw/'
json_file_path = './dataset/blip_captions.json'
train_csv_file_path = './dataset/captions_train.csv'
val_csv_file_path = './dataset/captions_val.csv'

# Load JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Prepare data for CSV
csv_data = []
for item in data:
    image_file = item['image_file']
    caption = item['blip_caption']
    filepath = os.path.join(base_path, image_file)
    csv_data.append([filepath, caption])

# Shuffle data
random.shuffle(csv_data)

# Split data into training and validation sets (80-20 split)
split_index = int(0.8 * len(csv_data))
train_data = csv_data[:split_index]
val_data = csv_data[split_index:]

# Function to write data to CSV
def write_to_csv(file_path, data):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filepath', 'caption'])  # Write header
        csv_writer.writerows(data)

# Write training data to CSV
write_to_csv(train_csv_file_path, train_data)

# Write validation data to CSV
write_to_csv(val_csv_file_path, val_data)

print(f"Training CSV file created at {train_csv_file_path}")
print(f"Validation CSV file created at {val_csv_file_path}")
