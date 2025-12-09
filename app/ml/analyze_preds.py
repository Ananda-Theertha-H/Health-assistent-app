import pandas as pd
import json
import ast

CSV_PATH = "app/ml/hf_out/predictions_test.csv"

def load_predictions(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

def parse_predictions(pred_json_str: str):
    """Convert predictions_json column string to a Python list."""
    try:
        # First try using json.loads
        return json.loads(pred_json_str)
    except:
        # Fallback for PowerShell-style escaped quotes
        return ast.literal_eval(pred_json_str)

def main():
    print("\n=== Loading Predictions CSV ===")
    df = load_predictions(CSV_PATH)
    print(f"Loaded {len(df)} rows.")

    top1_scores = []
    top3_scores = []
    high_conf_count = 0
    samples = []

    for i, row in df.iterrows():
        preds = parse_predictions(row["predictions_json"])

        # Top-1
        top1 = preds[0]["score"]
        top1_scores.append(top1)

        # Top-3 avg
        top3_vals = [p["score"] for p in preds[:3]]
        top3_scores.append(sum(top3_vals) / 3.0)

        # Count predictions with score > 0.5
        if top1 > 0.5:
            high_conf_count += 1

        # store first 10 samples
        if i < 10:
            samples.append((row["tests"], preds[:3]))

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total test rows: {len(df)}")
    print(f"Average top-1 score: {sum(top1_scores)/len(top1_scores):.4f}")
    print(f"Average top-3 score: {sum(top3_scores)/len(top3_scores):.4f}")
    print(f"High-confidence predictions (>0.5): {high_conf_count}")

    # Show sample predictions
    print("\n=== SAMPLE PREDICTIONS (first 10) ===")
    for tests, preds in samples:
        print("\nTests:", tests)
        print("Top-3 predictions:")
        for p in preds:
            print(f" - {p['label']}: {p['score']:.4f}")

if __name__ == "__main__":
    main()
