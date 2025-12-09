import csv
import os

CSV_IN = "app/ml/lab_ml_train.csv"     # your original CSV
CSV_OUT = "app/ml/lab_ml_train_fixed.csv"  # output
IMAGES_DIR = "data/images"             # where images are stored

# get sorted image list
images = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print("Images found:", len(images))

rows = []
with open(CSV_IN, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # skip old header
    for row in reader:
        rows.append(row)

print("CSV rows:", len(rows))

# safety check:
if len(rows) != len(images):
    print("❌ ERROR: image count and CSV count do NOT match!")
    print("Images:", len(images))
    print("Rows:", len(rows))
    exit(1)

# build new CSV with image names
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # NEW HEADER:
    writer.writerow(["image", "tests", "labels"])

    for i in range(len(rows)):
        img = images[i]         # matching by index
        tests = rows[i][0]      # tests JSON
        labels = rows[i][1]     # disease list
        writer.writerow([img, tests, labels])

print("✅ New CSV saved to:", CSV_OUT)
