import torch
import gc
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# メモリの空きを確保するため、一度ゴミ掃除を実行
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 1. モデルとプロセッサの準備
# ==========================================
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16
).to("cuda")

# ==========================================
# 2. データの準備と変換（省メモリ版）
# ==========================================
dataset = load_dataset("huggingface/documentation-images", split="train").select(range(10))

def collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        # メモリ節約のため、画像を256x256の小さなサイズに縮小してから追加します
        raw_img = example["image"].convert("RGB").resize((256, 256))
        images.append([raw_img])
        
        formatted_messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]},
            {"role": "assistant", "content": [{"type": "text", "text": example.get("text", "A photo.")}]}
        ]
        prompt = processor.apply_chat_template(formatted_messages, add_generation_prompt=False)
        texts.append(prompt)
        
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# ==========================================
# 3. LoRAの設定（メモリを圧迫しないよう調整）
# ==========================================
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    # 「all-linear」だとLLM側もすべて対象になり重いため、
    # VLMで最も重要な「言語モデルの聴覚・視覚の受け入れ口（q_proj, v_proj）」と「プロジェクター（projector）」に絞り込みます
    target_modules=["q_proj", "v_proj", "projector"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# ==========================================
# 4. 学習の条件設定（グラボのメモリ最優先設定）
# ==========================================
training_args = TrainingArguments(
    output_dir="./smolvlm2-finetuned",
    per_device_train_batch_size=1,     # 2 から 1 に減らして、1件ずつ省メモリで計算
    gradient_accumulation_steps=4,     # その代わり4件分溜まってからネジを回す（実質バッチサイズ4）
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    logging_steps=1,
    remove_unused_columns=False,
    gradient_checkpointing=True,       # 【超重要】メモリを劇的に節約する神オプションを有効化
)

# ==========================================
# 5. 学習の実行
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

print("🚀 ファインチューニングを開始します...")
trainer.train()
print("🎉 学習が完了しました！")

# 5日目のテスト用に保存
trainer.save_model("./my_first_vlm")
