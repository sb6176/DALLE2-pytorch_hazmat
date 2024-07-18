import os
import json
import csv

# Set the base path to your dataset folder
base_path = './dataset/images/raw/'
json_file_path = './dataset/blip_captions.json'
csv_file_path = './dataset/captions.csv'

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

# Write data to CSV
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['filepath', 'caption'])  # Write header
    csv_writer.writerows(csv_data)

print(f"CSV file created at {csv_file_path}")
