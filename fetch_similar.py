import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


TRAIN_FILE_PATH = 'train.json'
PREDICTIONS_FILE_PATH = 'predictions.json'
OUTPUT_FILE_PATH = 'generated_reports.json'

print("Loading train.json...")
try:
    with open(TRAIN_FILE_PATH, 'r') as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} training samples.")
except FileNotFoundError:
    print(f"Error: train.json not found at {TRAIN_FILE_PATH}. Please check the path.")
    exit()

print("Loading predictions.json...")
try:
    with open(PREDICTIONS_FILE_PATH, 'r') as f:
        predictions_data = json.load(f)
    print(f"Loaded {len(predictions_data)} prediction samples.")
except FileNotFoundError:
    print(f"Error: predictions.json not found at {PREDICTIONS_FILE_PATH}. Please check the path.")
    exit()

ground_truth_reports = [item['report'] for item in train_data]

print("Loading Sentence-BERT model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2') 
print("Model loaded.")

print("Encoding ground truth reports...")
ground_truth_embeddings = model.encode(ground_truth_reports, show_progress_bar=True)
print("Ground truth embeddings generated.")

print("\nApplying similarity-based corrections to predictions...")
final_corrected_outputs = []
SIMILARITY_THRESHOLD = 0.85 # This is a crucial parameter to tune!

for i, pred in enumerate(predictions_data):
    original_generated_report = pred['generated_report']
    current_report_id = pred['id']

    generated_embedding = model.encode([original_generated_report])

    similarities = cosine_similarity(generated_embedding, ground_truth_embeddings)[0]
    most_similar_idx = similarities.argmax()
    highest_similarity = similarities[most_similar_idx]
    most_similar_gt_report = ground_truth_reports[most_similar_idx]

    if highest_similarity >= SIMILARITY_THRESHOLD:
        final_report = most_similar_gt_report
    else:
        final_report = original_generated_report

    final_corrected_outputs.append({
        "id": current_report_id,
        "report": final_report
    })

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(predictions_data)} predictions.")

print(f"Processed all {len(predictions_data)} predictions.")

print(f"Saving final corrected predictions to {OUTPUT_FILE_PATH}...")
with open(OUTPUT_FILE_PATH, 'w') as f:
    json.dump(final_corrected_outputs, f, indent=2)
print("Correction complete and results saved in the specified format.")

print("\n--- Sample Final Reports ---")
for i, item in enumerate(final_corrected_outputs[:10]): # Display first 10
    print(f"ID: {item['id']}")
    print(f"  Report: {item['report']}")
    print("-" * 30)