import json
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Load the dataset
with open("constitution_qa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to Hugging Face dataset format
formatted_data = {
    "question": [],
    "answers": []
}

for item in data:
    formatted_data["question"].append(item["question"])
    formatted_data["answers"].append({"text": [item["answer"]], "answer_start": [0]})

# Convert to Dataset object
dataset = Dataset.from_dict(formatted_data)


tokenizer = AutoTokenizer.from_pretrained("akhilm97/pegasus_indian_legal")
model = AutoModelForSeq2SeqLM.from_pretrained("akhilm97/pegasus_indian_legal")

# Tokenize the dataset
def preprocess_data(examples):
    return tokenizer(
        examples["question"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(preprocess_data, batched=True)

print("Dataset Loaded and Tokenized Successfully!")

