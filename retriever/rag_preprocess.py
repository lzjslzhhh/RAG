import os
import re
import pdfplumber
from tqdm import tqdm
import jieba
import jieba.analyse
import json

INPUT_DIR = '../data'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def clean_text(text):
    # 先统一换行符为 \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 把非条款开头的换行符替换成空格
    # 条款/章节开头保留换行符，方便后续匹配
    # 这里用负向前瞻断言：换行后不是“第...条”或“第...章”的，替换为空格
    text = re.sub(r'\n(?!第[一二三四五六七八九十百千万]+[条章])', '', text)

    # 多个连续空格合并成一个空格
    text = re.sub(r'[ ]{2,}', '', text)

    # 去除首尾空白符
    return text.strip()


def read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def sent_tokenize(text):
    sentences = re.split(r'(?<=[。！？!?])', text)
    return [s.strip() for s in sentences if s.strip()]


def count_tokens(text):
    return len(list(jieba.cut(text)))


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
    """简单中文数字转阿拉伯数字，支持一到千"""
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
        text = clean_text(text)
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

        main_chunks = chunk_main_text_with_index(main_text)
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

    with open("../rag_structured_chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
