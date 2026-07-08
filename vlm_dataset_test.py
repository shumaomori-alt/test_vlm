from datasets import load_dataset
from transformers import AutoProcessor

# 1. Hugging Face公式の絶対にパブリックなデータセットを指定
print("📦 データセットをダウンロード中...")
dataset = load_dataset("huggingface/documentation-images", split="train")

# 2. データの1番最初（インデックス0）を覗いてみる
sample = dataset[0]
print("\n--- 元データの構造 ---")
print("画像データ:", sample["image"])
# このデータセットでは説明テキストの項目名が "text" になっています
print("テキストデータ（説明）:", sample.get("text", "No text provided"))

# 3. SmolVLM2のProcessor（通訳者）を準備
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

# 4. データをVLMが学習できる「チャット形式」に変換する関数
def format_for_vlm(example):
    raw_image = example["image"]
    # 万が一テキストが空だった場合のフォールバック（身代わりテキスト）を用意
    correct_answer = example.get("text", "An image from huggingface documentation.")
    
    # チャットテンプレートの形に整形
    formatted_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is this? Describe this image."} # ユーザーからの質問
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": correct_answer} # AIに答えさせたい正解の文
            ]
        }
    ]
    
    return {
        "images": [raw_image],
        "messages": formatted_messages
    }

# 5. 実際に1件、変換してみる
vlm_ready_data = format_for_vlm(sample)

print("\n--- 変換後のVLM用データ構造 ---")
import pprint
pprint.pprint(vlm_ready_data)

print("\n🎉 正常にデータが変換されました！3日目のタスクはクリアです。")
