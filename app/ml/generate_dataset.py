# generate_dataset.py
import random, json, csv, math
from collections import defaultdict
import numpy as np
import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# CONFIGURABLE SIZES
NUM_BIOMARKERS = 200     # e.g., 200 biomarkers/tests
NUM_DISEASES = 500       # e.g., 500 possible diseases
SAMPLES = 70000          # total rows (split into train/val/test)
OUTPUT_PREFIX = "lab_ml" # produces lab_ml_train.csv, lab_ml_val.csv

# 1) Generate biomarker names and default ranges
def gen_biomarkers(n):
    base = [
        ("Hemoglobin","g/dL", (11.0,16.0)),
        ("WBC","10^3/uL", (4.0,11.0)),
        ("Platelet","10^3/uL", (150,450)),
        ("Creatinine","mg/dL",(0.6,1.3)),
        ("Urea","mg/dL",(10,50)),
        ("BNP","pg/mL",(0,29.4)),
        ("Troponin I","ng/mL",(0,0.04)),
        ("ALT","U/L",(7,55)),
        ("AST","U/L",(8,48)),
        ("TSH","µIU/mL",(0.4,4.0))
    ]
    out = []
    for name,unit,rng in base:
        if len(out) < n:
            out.append((name, unit, rng))

    i = 0
    while len(out) < n:
        i += 1
        name = f"Marker_{i}"
        unit = random.choice(["units","mg/dL","U/L","ng/mL","pg/mL","µIU/mL"])
        center = random.uniform(1,200)
        low = round(center * 0.5, 2)
        high = round(center * 1.5, 2)
        out.append((name, unit, (low, high)))
    return out[:n]

BIOMARKERS = gen_biomarkers(NUM_BIOMARKERS)

# 2) Generate diseases + biomarker associations
def gen_diseases(n, biomarkers):
    diseases = []
    disease_map = defaultdict(list)
    bnames = [b[0] for b in biomarkers]
    for i in range(1, n+1):
        name = f"Disease_{i}"
        diseases.append(name)
        k = random.randint(2,6)
        assoc = random.sample(bnames, k)
        disease_map[name] = assoc
    return diseases, disease_map

DISEASES, DISEASE_MAP = gen_diseases(NUM_DISEASES, BIOMARKERS)

# 3) Sample real/abnormal values
def sample_value(low, high, status):
    if status == "low":
        return round(random.uniform(max(0, low*0.2), low - 0.01), 3)
    elif status == "high":
        return round(random.uniform(high + 0.01, high + (high - low)*2 + 10), 3)
    else:
        return round(random.uniform(low, high), 3)

# 4) Generate synthetic lab reports
def generate_samples(num_samples, biomarkers, diseases, disease_map):
    rows = []
    disease_prob = {d: random.uniform(0.0005, 0.02) for d in diseases}

    for _ in range(num_samples):
        test_count = random.choices([1,2,3,4,5], weights=[10,20,35,25,10])[0]
        picked = random.sample(biomarkers, test_count)

        present_diseases = [d for d in diseases if random.random() < disease_prob[d]]
        label_vec = set(present_diseases)

        tests = []
        for bname, unit, rng in picked:
            low, high = rng

            forced_abnormal = any(
                bname in disease_map[d] and random.random() < 0.9
                for d in label_vec
            )

            if forced_abnormal:
                status = random.choice(["low","high"])
            else:
                status = random.choices(["normal","low","high"], weights=[0.85,0.075,0.075])[0]

            value = sample_value(low, high, status)
            tests.append({
                "test_name": bname,
                "value": value,
                "unit": unit,
                "ref_low": low,
                "ref_high": high,
                "status": status
            })

        inferred_diseases = set()
        for d in diseases:
            for assoc in disease_map[d]:
                for t in tests:
                    if t["test_name"] == assoc and t["status"] != "normal":
                        if random.random() < 0.85:
                            inferred_diseases.add(d)
                            break

        labels = sorted(set(list(label_vec) + list(inferred_diseases)))
        rows.append({"tests": json.dumps(tests), "labels": json.dumps(labels)})

    return rows

rows = generate_samples(SAMPLES, BIOMARKERS, DISEASES, DISEASE_MAP)

# 5) Save splits
df = pd.DataFrame(rows)
train = df.sample(frac=0.8, random_state=RANDOM_SEED)
rest = df.drop(train.index)
val = rest.sample(frac=0.5, random_state=RANDOM_SEED)
test = rest.drop(val.index)

train.to_csv(f"{OUTPUT_PREFIX}_train.csv", index=False)
val.to_csv(f"{OUTPUT_PREFIX}_val.csv", index=False)
test.to_csv(f"{OUTPUT_PREFIX}_test.csv", index=False)

# 6) Save mapping files
with open(f"{OUTPUT_PREFIX}_biomarkers.json", "w") as f:
    json.dump(BIOMARKERS, f, indent=2)
with open(f"{OUTPUT_PREFIX}_diseases.json", "w") as f:
    json.dump(DISEASES, f, indent=2)
with open(f"{OUTPUT_PREFIX}_disease_map.json", "w") as f:
    json.dump(DISEASE_MAP, f, indent=2)

print("Generated dataset:", OUTPUT_PREFIX + "_train.csv etc.")
