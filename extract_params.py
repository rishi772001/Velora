# Now create the notebook for parameter extraction using FLAN-T5 (seq2seq style)
param_extractor_notebook = """\
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧾 Parameter Extraction using FLAN-T5\\n",
    "**Task:** Extract structured parameters from natural language\\n",
    "**Model:** `google/flan-t5-small`\\n",
    "**Optimized for:** Few-shot prompting or fine-tuning\\n",
    "**Resources:** Free-tier Colab"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q transformers datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\\n",
    "import torch"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔹 Step 1: Load Model and Tokenizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_name = \"google/flan-t5-small\"\\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\\n",
    "model = AutoModelForSeq2SeqLM.from_pretrained(model_name)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔹 Step 2: Inference with Few-Shot Prompting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_params(prompt):\\n",
    "    input_text = f\"\"\"\\n",
    "You are an AI assistant. Extract parameters in JSON format from the description below.\\n",
    "Description: {prompt}\\n",
    "Output Format: {\\\\\"destination\\\\\": ..., \\\\\"date\\\\\": ...}\\n",
    "\"\"\"\\n",
    "\\n",
    "    inputs = tokenizer(input_text, return_tensors=\"pt\")\\n",
    "    outputs = model.generate(**inputs, max_length=128)\\n",
    "    return tokenizer.decode(outputs[0], skip_special_tokens=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Try the function\\n",
    "extract_params(\"Book a flight to Paris next week\")"
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
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
"""

# Save notebook
with open(f"notebooks/param_extractor.ipynb", "w") as f:
    f.write(param_extractor_notebook)


