import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "manotham/Thai-dialogue-translate_emotion_mdpov2_ckp269"

OUTPUT_DIR = "./Thai-dialogue-translate_emotion_mdpov2_ckp269-W4A16"

TRAIN_FILE = "sft_train.jsonl"

MAX_SAMPLES = 128

# ============================================================
# LOAD MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# ============================================================
# BUILD CALIBRATION DATASET
# ============================================================

samples = []

with open(TRAIN_FILE, "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        if i >= MAX_SAMPLES:
            break

        data = json.loads(line)

        user_content = data["messages"][1]["content"]

        messages = [
            {
                "role": "user",
                "content": user_content
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        samples.append({"text": prompt})

dataset = Dataset.from_list(samples)

print(f"✅ Calibration Samples: {len(dataset)}")
print("===================================================")
print("✅ Dataset Structure")
print("===================================================")

print(dataset)

print("\n===================================================")
print("✅ First Sample")
print("===================================================")

print(dataset[0])

print("\n===================================================")
print("✅ Prompt Preview")
print("===================================================")

print(dataset[0]["text"])

# ============================================================
# AWQ RECIPE
# ============================================================

recipe = [
    AWQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="W4A16_ASYM"
    ),

    QuantizationModifier(
        targets=["Linear"],
        scheme="W4A16_ASYM",
        ignore=["lm_head"]
    )
]

# ============================================================
# QUANTIZE
# ============================================================

oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    output_dir=OUTPUT_DIR,
    max_seq_length=1024
)

# ============================================================
# SAVE TOKENIZER
# ============================================================

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"🎉 Saved to: {OUTPUT_DIR}")