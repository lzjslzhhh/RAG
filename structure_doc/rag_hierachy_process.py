import os
import json
import re
from tqdm import tqdm
from pdf2image import convert_from_path
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from PIL import Image

# 模型加载
model_id = "microsoft/layoutlmv3-base-chinese"
processor = LayoutLMv3Processor.from_pretrained(model_id, apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_id, num_labels=4)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

INPUT_DIR = "../temp"
OUTPUT_FILE = "../retriever/rag_layoutlmv3_chunks.jsonl"


def extract_from_pdf_layoutlmv3(pdf_path):
    images = convert_from_path(pdf_path, dpi=200)
    all_chunks = []
    chunk_id = 0

    for page_idx, image in enumerate(tqdm(images, desc=f"Processing {os.path.basename(pdf_path)}")):
        encoding = processor(image, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        words = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())

        # 提取标题行
        lines = []
        current_line = ""
        for token, pred in zip(words, predictions):
            if token in ["[PAD]", "[CLS]", "[SEP]"]:
                continue
            token_clean = token.replace("##", "")
            if pred == 1:  # 如果是标题标记
                if current_line:
                    lines.append(current_line.strip())
                    current_line = ""
                current_line = token_clean
            else:
                current_line += token_clean
        if current_line:
            lines.append(current_line.strip())

        # 结构化编号识别
        for line in lines:
            m = re.match(r'^\s*(\d+(?:\.\d+)*)(?:[）.\s、\-]*)\s*(.+)$', line)
            if m:
                number = m.group(1)
                title = m.group(2)
                level = len(number.split('.'))
                parent_id = ".".join(number.split('.')[:-1]) if level > 1 else None
                all_chunks.append({
                    "chunk_id": number,
                    "title": title,
                    "level": level,
                    "parent_id": parent_id,
                    "page": page_idx + 1,
                    "source": os.path.basename(pdf_path),
                    "section": "正文",
                    "content": "",  # 内容后处理可加
                })
            else:
                # 可选：非结构行可作为补充内容挂接到最近 chunk 上
                if all_chunks:
                    all_chunks[-1]["content"] += line + "\n"

    return all_chunks


def main():
    all_docs = []
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".pdf"):
            continue
        path = os.path.join(INPUT_DIR, fname)
        chunks = extract_from_pdf_layoutlmv3(path)
        all_docs.extend(chunks)

    print(f"总共抽取结构化段落: {len(all_docs)}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_docs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
