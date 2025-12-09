import csv
import re

def classify_tests(test_text: str):
    text = test_text.lower()

    labels = set()

    # ---- Diabetes ----
    if "glucose" in text or "fasting" in text or "glycosylated hb" in text or "hba1c" in text:
        if "high" in text:
            labels.add("diabetes")

    # ---- Thyroid ----
    if "tsh" in text or "t3" in text or "t4" in text:
        labels.add("thyroid_disorder")

    # ---- Liver ----
    if "alt" in text or "ast" in text or "bilirubin" in text or "alkaline phosphatase" in text:
        labels.add("liver_disease")

    # ---- Kidney ----
    if "creatinine" in text or "urea" in text or "egfr" in text:
        labels.add("kidney_issue")

    # ---- Lipids ----
    if "ldl" in text or "hdl" in text or "triglyceride" in text or "cholesterol" in text:
        labels.add("lipid_disorder")

    # ---- Anemia ----
    if "hemoglobin" in text or "hb " in text or "rbc" in text:
        labels.add("anemia")

    # ---- Infection / inflammation ----
    if "crp" in text or "esr" in text or "wbc" in text:
        labels.add("infection")

    # ---- Electrolyte imbalance ----
    if "sodium" in text or "potassium" in text or "chloride" in text or "na+" in text or "k+" in text:
        labels.add("electrolyte_imbalance")

    # ---- Vitamin deficiency ----
    if "vitamin d" in text or "vitamin b12" in text:
        labels.add("vitamin_deficiency")

    # ---- Platelet / clotting ----
    if "platelet" in text or "inr" in text:
        labels.add("clotting_issue")

    # return as sorted list
    return list(labels)


def auto_label_csv(input_csv, output_csv):
    with open(input_csv, "r", encoding="utf-8") as f_in, \
         open(output_csv, "w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["tests", "labels"])
        writer.writeheader()

        for row in reader:
            tests = row["tests"]
            labels = classify_tests(tests)
            writer.writerow({
                "tests": tests,
                "labels": str(labels)  # save as python list string
            })

    print("Auto-labeled CSV saved to:", output_csv)


if __name__ == "__main__":
    auto_label_csv("app/ml/lab_ml_train.csv", "app/ml/lab_ml_train_labeled.csv")
