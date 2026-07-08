import torch
from PIL import Image
# 修正：古いAutoModelForVision2Seqから、最新のAutoModelForImageTextToTextに変更
from transformers import AutoProcessor, AutoModelForImageTextToText

print("🔄 モデルとプロセッサを読み込んでいます（初回はダウンロードに数分かかります）...")

# 1. 超軽量VLMモデルの指定
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

# 2. プロセッサとモデルの読み込み
processor = AutoProcessor.from_pretrained(model_id)

# 修正：最新のクラス名を使用し、CPU環境向けに明示的にfloat32を指定
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.float32
).to("cpu")

# 3. テスト用の画像をネットからダウンロード（例としてミツバチの画像）
import requests
from io import BytesIO
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

print("📸 画像の読み込みが完了しました。")

# 4. AIへの質問（プロンプト）の作成
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

# 5. モデルが読める形にデータを変換（Processorの役割！）
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cpu")

print("🤖 VLMが画像を解析して回答を生成中...")

# 6. テキストの生成
generated_ids = model.generate(**inputs, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

# 7. 結果の表示
print("\n=== AIからの回答 ===")
print(generated_texts[0])

print("\n=== modelの中身 ===")
print(model)

