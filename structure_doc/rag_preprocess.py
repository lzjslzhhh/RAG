import os
import re
import json
from itertools import zip_longest

# import numpy as np
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm
import jieba
import jieba.analyse
from typing import List, Dict, Tuple, Optional

INPUT_DIR = '../hierarchy_doc'
OUTPUT_FILE = '../retriever/rag_structured_chunks.jsonl' # 输出文件路径
CHUNK_SIZE = 500                       # 分块token数
CHUNK_OVERLAP = 100                    # 分块重叠token数
# TESSERACT_CONFIG = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'

# # 设置路径
os.environ['TESSDATA_PREFIX'] = r'D:\OCR\tessdata'

pytesseract.pytesseract.tesseract_cmd = r'D:\OCR\tesseract.exe'

import re

def clean_text(text):
    import re

    # 标准化换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 暂存段落之间的空行（2个或以上换行）
    text = re.sub(r'\n{2,}', '\n\n[PARA_BREAK]\n\n', text)

    # 去除段落内的单个换行，替换为空格
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 多个连续空格合并为一个
    text = re.sub(r' {2,}', ' ', text)

    # 恢复段落之间的换行
    text = text.replace('\n\n[PARA_BREAK]\n\n', '\n\n')

    return re.sub(r'\n{2,}', '\n', text.strip())


def is_text_meaningful(text: str, threshold: float = 0.3) -> bool:

    if not text.strip():
        return False
    total_chars = len(text)
    non_whitespace = len(re.sub(r'\s+', '', text))
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    ratio = (non_whitespace + chinese_chars) / (2 * total_chars)
    return ratio > threshold


def fuse_ocr_results(text1: str, text2: str) -> str:

    lines1 = text1.split('\n')
    lines2 = text2.split('\n')
    return "\n".join(
        max(l1, l2, key=lambda x: len(x.strip()))
        for l1, l2 in zip_longest(lines1, lines2, fillvalue="")
    )


def read_pdf(path):
    # try:
    #     with pdfplumber.open(path) as pdf:
    #         text = '\n'.join(page.extract_text() or "" for page in pdf.pages)
    # except Exception as e:
    #     print(f"pdfplumber 错误: {e}")
    #     text = ""
    #
    # if is_text_meaningful(text):
    #     return text

        # 否则使用 OCR
    print(f"使用 OCR 处理: {path}")
    # 初始化多OCR引擎
    # reader = easyocr.Reader(['ch_sim', 'en'])  # 中英双语
    images = convert_from_path(path,dpi=300)
    ocr_text = ""
    for img in images:
        tesseract_text = pytesseract.image_to_string(
            img,
            lang='chi_sim+eng'
        )
        ocr_text+=tesseract_text
    return ocr_text


def sent_tokenize(text):
    sentences = re.split(r'(?<=[。！？!?])', text)
    return [s.strip() for s in sentences if s.strip()]


def count_tokens(text):
    return len(list(jieba.cut(text)))


def identify_doc_structure(text: str) -> str:

    # 检测数字编号体系（如3.1.1）
    if re.search(r'\d+\.\d+(?:\.\d+)*\s', text):
        return 'hierarchical'
    # 检测中文条款（如"第一条"）
    elif re.search(r'第[一二三四五六七八九十百千万]+[条章]', text):
        return 'sectioned'
    else:
        return 'flat'


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    total_tokens = 0
    i = 0

    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = count_tokens(sentence)
        if total_tokens + sentence_tokens > chunk_size and current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append(chunk_text)

            if overlap > 0:
                overlap_tokens = 0
                j = len(current_chunk) - 1
                while j >= 0 and overlap_tokens < overlap:
                    overlap_tokens += count_tokens(current_chunk[j])
                    j -= 1
                i = i - (len(current_chunk) - 1 - j)
            else:
                i += 1

            current_chunk = []
            total_tokens = 0
        else:
            current_chunk.append(sentence)
            total_tokens += sentence_tokens
            i += 1

    if current_chunk:
        chunks.append("".join(current_chunk))
    return chunks


def split_document_sections(text):
    catalog_match = re.search(r"(目录|目次)", text)
    catalog_pos = catalog_match.start() if catalog_match else -1
    main_match = re.search(r"第[一二三四五六七八九十百千万]*[条章]", text)
    main_pos = main_match.start() if main_match else -1

    preface = text[:catalog_pos].strip() if catalog_pos > 0 else ""

    if catalog_pos >= 0 and main_pos > catalog_pos:
        catalog_text = text[catalog_pos:main_pos].strip()
        main_text = text[main_pos:].strip()
    else:
        catalog_text = ""
        main_text = text

    return preface, catalog_text, main_text


def cn2num(cn):

    cn_num = cn.replace("第", "").replace("条", "").replace("章", "")
    map_units = {'十': 10, '百': 100, '千': 1000, '万': 10000}
    map_nums = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}

    result = 0
    unit = 1
    i = len(cn_num) - 1
    while i >= 0:
        c = cn_num[i]
        if c in map_units:
            unit = map_units[c]
            if i == 0:  # 十 -> 10
                result += unit
            i -= 1
        else:
            num = map_nums.get(c, 0)
            result += num * unit
            i -= 1
    return result


def extract_keywords(text, topk=5):
    stopwords = set()
    try:
        with open('stopwords.txt', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
    except FileNotFoundError:
        pass  # 如果没有停用词文件，则忽略

    words = jieba.posseg.cut(text)
    raw_keywords = jieba.analyse.extract_tags(text, topK=topk*3)  # 多取些，方便筛选
    allowed_flags = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn'}

    filtered = []
    for w in words:
        if w.flag in allowed_flags and w.word in raw_keywords and w.word not in stopwords:
            filtered.append(w.word)

    # 去重
    filtered = list(dict.fromkeys(filtered))
    filtered = [w for w in filtered if len(w) > 1]

    return filtered[:topk]


def summarize_text(text):
    # 简单取第一句作为摘要示例
    sents = sent_tokenize(text)
    return sents[0] if sents else text[:60]


def extract_structure(block):
    chapter_match = re.search(r'(第[一二三四五六七八九十百千万]+章)', block)
    article_match = re.search(r'(第[一二三四五六七八九十百千万]+条)', block)
    chapter_title = chapter_match.group(1) if chapter_match else ""
    article_no = article_match.group(1) if article_match else ""
    section_title = ""

    if chapter_match:
        rest = block[chapter_match.end():].strip()
        section_title = rest.split('\n')[0][:30]
    elif article_match:
        rest = block[article_match.end():].strip()
        section_title = rest.split('\n')[0][:30]

    return chapter_title, section_title, article_no


def extract_numbered_structure(text):

    # 正则匹配：形如 1. 1.1 1.1.1 或 2.2.3.4 等，标题为紧随其后的内容
    pattern = r'(?m)^\s*((?:\d+\.)+\d*|\d+)[\s、．．\.)]+(.+)$'
    matches = re.findall(pattern, text)

    structure = []
    index_map = {}  # 用于查找父编号

    for num, title in matches:
        num = num.rstrip(".")  # 去除结尾多余的点
        level = num.count('.') + 1  # 层级：1.1.1 → 3级
        parent = '.'.join(num.split('.')[:-1]) if level > 1 else None
        structure.append((num, title.strip(), level, parent))
        index_map[num] = title.strip()

    return structure


def is_heading_number(s):
    # 判断是否为章节编号，如 1、1.1、1.2.3、1.2.3.4 等
    return re.match(r'^\d+(?:\.\d+)*$', s.strip()) is not None


def is_possible_title(s):
    # 可通过长度和字符特征判断是否可能是标题
    return 1 <= len(s.strip()) <= 30 and not s.strip().endswith('。')


def chunk_main_text_with_hierarchical(text: str):
    import re

    # # 正则模式定义
    heading_line_pattern = re.compile(r'^\s*(\d+(?:\.\d+)*)(?:[）.\s、\-]*)\s*$')  # 只有编号一行
    heading_inline_pattern = re.compile(r'^\s*(\d+(?:\.\d+)*)(?:[）.\s、\-]+)\s*(.+)$')  # 编号 + 标题同行
    list_item_pattern = re.compile(r'^([a-zA-Z]|\d+)[）.\-、]\s*')  # 列表项识别：a)、1.、1- 之类

    # 清理换行和空格
    # text = text.replace('\r\n', '\n').replace('\r', '\n')
    # text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    chunks = []
    i = 0

    while i < len(lines):
        line = lines[i]

        match_line = heading_line_pattern.match(line)
        # 编号单独成行
        if is_heading_number(line) and i + 1 < len(lines):
            number = match_line.group(1)
            level = len(number.split('.'))
            parent_id = ".".join(number.split('.')[:-1]) if level > 1 else None

            next_line = lines[i + 1]
            if is_possible_title(next_line):
                number = line
                title = next_line
                chunks.append({
                    "chunk_id": number,
                    "title": title,
                    "level": level,
                    "parent_id": parent_id,
                    "content": ""
                })
                i += 2
                continue

        # 编号 + 标题同行
        # match_inline = heading_inline_pattern.match(line)
        match = re.match(r'^(\d+(?:\.\d+)*)(?:[\s、．.\-]+)(.{1,30})$', line)
        if match:
            number = match.group(1)
            title = match.group(2)
            level = len(number.split('.'))
            parent_id = ".".join(number.split('.')[:-1]) if level > 1 else None
            chunks.append({
                "chunk_id": number,
                "title": title,
                "level": level,
                "parent_id": parent_id,
                "content": ""
            })
            i += 1
            continue
        # # 情况 1：编号 + 标题同行
        # match_inline = heading_inline_pattern.match(line)
        # if match_inline and not list_item_pattern.match(match_inline.group(2)):
        #     number = match_inline.group(1)
        #     title = match_inline.group(2)
        #     level = len(number.split('.'))
        #     parent_id = ".".join(number.split('.')[:-1]) if level > 1 else None
        #
        #     # 保存前一个 chunk
        #     if current_chunk:
        #         chunks.append(current_chunk)
        #
        #     current_chunk = {
        #         "chunk_id": number,
        #         "title": title,
        #         "level": level,
        #         "parent_id": parent_id,
        #         "content": ""
        #     }
        #     i += 1
        #     continue

        # # 情况 2：编号单独一行
        # match_line = heading_line_pattern.match(line)
        # if match_line:
        #     number = match_line.group(1)
        #     level = len(number.split('.'))
        #     parent_id = ".".join(number.split('.')[:-1]) if level > 1 else None
        #
        #     # 获取下一个非空行作为标题
        #     j = i + 1
        #     title = ""
        #     while j < len(lines):
        #         next_line = lines[j].strip()
        #         if not next_line:
        #             j += 1
        #             continue
        #         # 下一个标题开头，说明当前编号没有标题，跳过
        #         if heading_inline_pattern.match(next_line) or heading_line_pattern.match(next_line):
        #             break
        #         title = next_line
        #         break
        #
        #     # 保存前一个 chunk
        #     if current_chunk:
        #         chunks.append(current_chunk)
        #
        #     current_chunk = {
        #         "chunk_id": number,
        #         "title": title,
        #         "level": level,
        #         "parent_id": parent_id,
        #         "content": ""
        #     }
        #     i = j + 1
        #     continue

        # 正文内容
        # 追加正文到最后一个section
        if chunks:
            if line != '':
                chunks[-1]["content"] += line + "\n"
        i += 1

    return chunks


def chunk_main_text_with_index(text):
    pattern = re.compile(r"(第[一二三四五六七八九十百千万]*[条章].+?)(?=第[一二三四五六七八九十百千万]*[条章]|$)", re.S)
    blocks = pattern.findall(text)

    chunks = []
    current_chapter = ""

    for block in blocks:
        chapter_title, section_title, article_no = extract_structure(block)
        if chapter_title:
            current_chapter = chapter_title

        final_chapter = chapter_title if chapter_title else current_chapter

        # 生成chunk_id，优先用条款号，没条款号用章节
        if article_no:
            chunk_id = f"ARTICLE_{cn2num(article_no)}"
        elif chapter_title:
            chunk_id = f"CHAPTER_{cn2num(chapter_title)}"
        else:
            chunk_id = f"UNKNOW_{len(chunks)}"

        # 条款过长，做简单摘要拼接
        if count_tokens(block) > CHUNK_SIZE:
            summary = summarize_text(block)
            content_final = summary + "\n\n" + block[:CHUNK_SIZE]
        else:
            content_final = block

        keywords = extract_keywords(content_final)

        chunks.append({
            "chunk_id": chunk_id,
            "chapter_title": final_chapter,
            "section_title": section_title,
            "article_no": article_no,
            "content": content_final,
            "metadata": {
                "chapter": final_chapter,
                "article_no": article_no,
                "keywords": keywords
            }
        })

    return chunks


def process_documents(input_dir):
    all_chunks = []
    for file in tqdm(os.listdir(input_dir)):
        print(file)
        if not file.endswith(".pdf"):
            continue
        path = os.path.join(input_dir, file)
        text = read_pdf(path)
        # text = clean_text(text)
        preface, catalog, main_text = split_document_sections(text)

        idx = 0
        if preface:
            all_chunks.append({
                "source": file,
                "chunk_id": f"PREFACE_{idx}",
                "section": "前言",
                "content": preface
            })
            idx += 1
        if catalog:
            all_chunks.append({
                "source": file,
                "chunk_id": f"CATALOG_{idx}",
                "section": "目录",
                "content": catalog
            })
            idx += 1
        doc_type = identify_doc_structure(text)
        if doc_type == 'hierarchical':
            main_chunks = chunk_main_text_with_hierarchical(main_text)
        elif doc_type == 'sectioned':
            main_chunks = chunk_main_text_with_index(main_text)
        else:
            main_chunks = split_into_chunks(main_text)
            main_chunks = [{
                "chunk_id": f"FLAT_{i}",
                "content": c,
                "metadata": {"structure_type": "flat"}
            } for i, c in enumerate(main_chunks)]
        for chunk in main_chunks:
            chunk.update({
                "source": file,
                "chunk_id": chunk["chunk_id"],
                "section": "正文"
            })
            all_chunks.append(chunk)
            idx += 1

    return all_chunks


if __name__ == "__main__":
    chunks = process_documents(INPUT_DIR)
    print(f"共生成 {len(chunks)} 个文档分块")

    with open("../retriever/rag_hierarchy_chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")