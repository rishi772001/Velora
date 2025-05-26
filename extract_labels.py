# Fix issue: escape curly braces in f-string to avoid ValueError
classification_notebook = """\
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🔤 Text Classification with DistilBERT (HuggingFace Transformers)\\n",
    "**Task:** Predict labels from user prompts  \\n",
    "**Model:** `distilbert-base-uncased` (small and accurate)  \\n",
    "**Resources:** Optimized for Colab free tier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q transformers datasets scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments\\n",
    "from datasets import Dataset\\n",
    "from sklearn.model_selection import train_test_split\\n",
    "import torch"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔹 Step 1: Load Your Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sample dataset structure\\n",
    "data = [\\n",
    "    {{\"text\": \"Book a flight to Paris next week\", \"label\": \"travel_booking\"}},\\n",
    "    {{\"text\": \"Order a pepperoni pizza with mushrooms\", \"label\": \"food_order\"}},\\n",
    "    {{\"text\": \"Schedule a meeting with Alice for Monday\", \"label\": \"schedule_meeting\"}},\\n",
    "]\\n",
    "\\n",
    "# Map labels to integers\\n",
    "label2id = {{label: idx for idx, label in enumerate(sorted(set(d[\"label\"] for d in data)))}}\\n",
    "id2label = {{v: k for k, v in label2id.items()}}\\n",
    "for d in data:\\n",
    "    d[\"label\"] = label2id[d[\"label\"]]\\n",
    "\\n",
    "# Split dataset\\n",
    "train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)\\n",
    "train_ds = Dataset.from_list(train_data)\\n",
    "test_ds = Dataset.from_list(test_data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔹 Step 2: Tokenize and Prepare"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer = DistilBertTokenizerFast.from_pretrained(\"distilbert-base-uncased\")\\n",
    "\\n",
    "def tokenize(batch):\\n",
    "    return tokenizer(batch[\"text\"], padding=True, truncation=True)\\n",
    "\\n",
    "train_ds = train_ds.map(tokenize, batched=True)\\n",
    "test_ds = test_ds.map(tokenize, batched=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔹 Step 3: Load Model and Fine-tune"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = DistilBertForSequenceClassification.from_pretrained(\\n",
    "    \"distilbert-base-uncased\",\\n",
    "    num_labels=len(label2id),\\n",
    "    id2label=id2label,\\n",
    "    label2id=label2id,\\n",
    ")\\n",
    "\\n",
    "training_args = TrainingArguments(\\n",
    "    output_dir=\"./results\",\\n",
    "    evaluation_strategy=\"epoch\",\\n",
    "    per_device_train_batch_size=8,\\n",
    "    per_device_eval_batch_size=8,\\n",
    "    num_train_epochs=5,\\n",
    "    weight_decay=0.01,\\n",
    "    logging_dir=\"./logs\",\\n",
    "    logging_steps=10,\\n",
    ")\\n",
    "\\n",
    "trainer = Trainer(\\n",
    "    model=model,\\n",
    "    args=training_args,\\n",
    "    train_dataset=train_ds,\\n",
    "    eval_dataset=test_ds,\\n",
    "    tokenizer=tokenizer,\\n",
    ")\\n",
    "\\n",
    "trainer.train()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔹 Step 4: Test on New Example"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_label(text):\\n",
    "    inputs = tokenizer(text, return_tensors=\"pt\", truncation=True, padding=True)\\n",
    "    outputs = model(**inputs)\\n",
    "    predicted_class_id = torch.argmax(outputs.logits).item()\\n",
    "    return id2label[predicted_class_id]\\n",
    "\\n",
    "predict_label(\"Book a table at an Italian restaurant for tonight\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
"""

# Save notebook
with open(f"notebooks/label_classifier.ipynb", "w") as f:
    f.write(classification_notebook)


