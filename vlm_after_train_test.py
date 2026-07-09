import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

print("🦜 必要なツールとモデルを読み込んでいます...")

# 1. 土台となる元の「SmolVLM2」を読み込む
base_model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(base_model_id)
base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id, 
    torch_dtype=torch.bfloat16
).to("cuda")

# 2. 昨日あなたが育てた「LoRAのネジ（学習済みの重み）」を土台モデルに合体させる！
# これにより、一瞬で「あなた専用のVLM」に変身します
model = PeftModel.from_pretrained(base_model, "./my_first_vlm")
model.eval() # モデルを「テストモード（推論モード）」に切り替え

print("📦 テスト用の新しい画像を準備しています...")
# 学習に使わなかった「11番目（インデックス10）」の未知の画像を取り出します
dataset = load_dataset("huggingface/documentation-images", split="train")
test_sample = dataset[10]
test_image = test_sample["image"].convert("RGB").resize((256, 256))

# テスト画像を画面に表示して確認してみる
print("📸 テストする画像はこちらです：")
display(test_image)

# 3. VLMに話しかけるチャットデータを作成
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."} # 昨日学習させた時と全く同じ質問
        ]
    }
]

# プロセッサを使ってAIが読める形式（テンソル）に変換
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[[test_image]], return_tensors="pt").to("cuda")

# 4. AIに回答を生成させる（推論の実行）
print("\n🤖 VLMが画像を読み込んで考えています...")
with torch.no_grad(): # メモリ節約のため、学習用の勾配計算をオフにする
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50, # AIが喋る最大の単語数
        do_sample=False     # 毎回同じ確実な答えを出力させる
    )

# 5. AIの返答（数字の羅列）を、人間が読める言葉（テキスト）にデコード（翻訳）
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print("\n================ AIからの回答 ================")
# 質問部分を切り取って、AIが新しく喋った答えだけを綺麗に表示します
print(generated_texts[0].split("assistant")[-1].strip())
print("==============================================")

print("\n💡 ちなみに、この画像に人間が最初につけていた「正解のテキスト」はこちら：")
print(test_sample.get("text", "正解データなし"))
