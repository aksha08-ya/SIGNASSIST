import json

with open("reports/evaluation/dynamic_validation.json", "r", encoding="utf-8") as f:
    report = json.load(f)

print("Metrics:")
for k, v in report["metrics"].items():
    print(f"  {k}: {v}")

print("\nThreshold Analysis:")
for k, v in report["threshold_analysis"].items():
    print(f"  {k}: {v}")

print("\nRaw Softmax Confidence Distribution:")
for k, v in report["raw_softmax_confidence_distribution"].items():
    print(f"  {k}: {v}")

print("\nPrediction Distribution:")
print(f"  unique_classes_predicted: {report['prediction_distribution']['unique_classes_predicted']}")
print("  Top 20 most frequently predicted classes:")
for item in report["prediction_distribution"]["top_20_most_frequently_predicted"]:
    print(f"    ID {item['class_id']:3d} | Gloss: {item['gloss']:<25s} | Count: {item['count']:3d} ({item['percentage']:.2f}%)")

print("\n20 Concrete Examples:")
for ex in report["sample_examples_20"]:
    print(f"  [{ex['sample_index']:3d}] Video: {ex['video_path']}")
    print(f"        True: ID {ex['true_class_id']:3d} ({ex['true_gloss']}) | Pred: ID {ex['predicted_class_id']:3d} ({ex['predicted_gloss']})")
    print(f"        Raw Softmax Confidence (UNCALIBRATED): {ex['raw_softmax_confidence']:.4f} | Accepted: {ex['is_accepted']}")
