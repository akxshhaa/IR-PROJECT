
import random
import os

import torch
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses
from torch.utils.data import DataLoader

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PATH = "./trained_model"

EPOCHS = 1
BATCH_SIZE = 16
MAX_TRAIN_DOCUMENTS = 5000

random.seed(42)

# -----------------------------
# LOAD DATASET
# -----------------------------
print("Loading 20 Newsgroups dataset...")

data = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

documents = data.data
labels = data.target

# -----------------------------
# CREATE POSITIVE PAIRS
# -----------------------------
print("Creating training pairs...")

documents_by_category = {}

for document, label in zip(documents, labels):
    document = document.strip()

    if len(document) < 100:
        continue

    documents_by_category.setdefault(label, []).append(document)

train_examples = []

for category_documents in documents_by_category.values():

    if len(category_documents) < 2:
        continue

    for _ in range(min(len(category_documents), 300)):

        doc1, doc2 = random.sample(category_documents, 2)

        train_examples.append(
            InputExample(
                texts=[doc1[:1500], doc2[:1500]]
            )
        )

# Limit training size for manageable runtime
random.shuffle(train_examples)
train_examples = train_examples[:MAX_TRAIN_DOCUMENTS]

print("Training pairs:", len(train_examples))

# -----------------------------
# LOAD PRETRAINED MODEL
# -----------------------------
print("Loading pretrained Sentence Transformer...")

model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# CREATE DATALOADER
# -----------------------------
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)

print("Starting model fine-tuning...")

model.fit(
    train_objectives=[
        (train_dataloader, train_loss)
    ],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path=OUTPUT_PATH,
    show_progress_bar=True
)

# -----------------------------
# SAVE TRAINED MODEL
# -----------------------------
model.save(OUTPUT_PATH)

print("\nTRAINING COMPLETED!")
print("Model saved at:", OUTPUT_PATH)
print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
