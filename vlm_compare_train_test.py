import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

print("🦜 比較用のモデルを準備しています...")
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

# 1. まっさらな「学習前（土台）」のモデルをロード
base_model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16
).to("cuda")
base_model.eval()

# 2. あなたが育てた「学習後」のモデルをロード
ft_model = PeftModel.from_pretrained(base_model, "./my_first_vlm")
ft_model.eval()

print("📦 テスト画像を準備しています...")
dataset = load_dataset("huggingface/documentation-images", split="train")
test_sample = dataset[10] # 先ほどと同じ11番目の画像
test_image = test_sample["image"].convert("RGB").resize((256, 256))

# チャットデータの作成
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[[test_image]], return_tensors="pt").to("cuda")

# --------------------------------------------------
# ① 学習前のモデルに答えさせてみる
# --------------------------------------------------
with torch.no_grad():
    # unload() を使うことで、一時的にLoRAのネジを外して元のモデルに戻します
    raw_generated_ids = ft_model.unload().generate(**inputs, max_new_tokens=50, do_sample=False)
raw_generated_texts = processor.batch_decode(raw_generated_ids, skip_special_tokens=True)
before_answer = raw_generated_texts[0].split("assistant")[-1].strip()

# --------------------------------------------------
# ② 学習後のモデルに答えさせてみる
# --------------------------------------------------
with torch.no_grad():
    # 再びLoRAのネジを有効化して答えさせます
    ft_generated_ids = ft_model.generate(**inputs, max_new_tokens=50, do_sample=False)
ft_generated_texts = processor.batch_decode(ft_generated_ids, skip_special_tokens=True)
after_answer = ft_generated_texts[0].split("assistant")[-1].strip()

# --------------------------------------------------
# 結果の表示
# --------------------------------------------------
print("\n📸 対象の画像を表示します：")
display(test_image)

print("\n" + "="*20 + " 🛑 学習前（元のAI）の回答 " + "="*20)
print(before_answer)

print("\n" + "="*20 + " ✨ 学習後（あなたのAI）の回答 " + "="*20)
print(after_answer)
print("="*65)
