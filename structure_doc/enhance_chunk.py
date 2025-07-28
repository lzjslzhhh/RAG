import os
import re
import json
from typing import List, Dict, Optional
from collections import Counter

def enhance_chunks_with_parent_titles(input_file: str, output_file: str):
    """
    增强分块数据：根据parent_id查找并注入父级标题
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSONL文件路径
    """
    # 读取所有分块数据
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    # 构建source和chunk_id的映射
    source_map = {}
    for chunk in chunks:
        source = chunk.get('source')
        if source not in source_map:
            source_map[source] = {}
        source_map[source][chunk['chunk_id']] = chunk

    # 增强分块数据
    enhanced_chunks = []
    for chunk in chunks:
        try:
            # 复制原始分块数据
            enhanced = chunk.copy()

            # 初始化父级标题链
            parent_chain = []
            current_parent_id = chunk.get('parent_id')

            # 递归查找父级标题
            while current_parent_id:
                try:
                    parent_chunk = source_map[chunk['source']][current_parent_id]
                    parent_chain.append({
                        'chunk_id': parent_chunk['chunk_id'],
                        'title': parent_chunk['title']
                    })
                    current_parent_id = parent_chunk.get('parent_id')
                except KeyError:
                    # 父节点不存在时终止查找
                    break

            # 添加父级信息到metadata
            if 'metadata' not in enhanced:
                enhanced['metadata'] = {}

            enhanced['metadata']['parent_chain'] = parent_chain

            # 生成层级路径
            hierarchy_path = ' -> '.join(
                [p['title'] for p in reversed(parent_chain)] + [chunk['title']]
            )
            enhanced['metadata']['hierarchy_path'] = hierarchy_path

            enhanced_chunks.append(enhanced)
        except Exception as e:
            print(f"处理分块 {chunk.get('chunk_id')} 时出错: {str(e)}")
            # 出错时保留原始分块
            enhanced_chunks.append(chunk)

    # 写入增强后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in enhanced_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def normalize_std_id(raw_std: str) -> str:
    """标准化处理流程"""
    # 去除空格和分隔符
    std = raw_std.replace(" ", "").replace("-", "")
    # 截断年份后缀
    return re.sub(r'[_-]\d{4}$', '', std)


def process_jsonl(input_path: str, output_path: str):
    """处理流程：读取->统计->注入->保存"""
    # 读取数据并按source分组
    source_groups = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            source = chunk.get('source')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(chunk)

    # 为每个source确定主标准号（增加空值保护）
    source_std = {}
    for src, chunks in source_groups.items():
        std_ids = [
            normalize_std_id(extract_std_from_title(c['title']))
            for c in chunks
            if extract_std_from_title(c['title'])
        ]
        # 仅当有有效标准号时才赋值
        if std_ids:
            source_std[src] = Counter(std_ids).most_common(1)[0][0]

    # 注入标准号并保存
    with open(output_path, 'w', encoding='utf-8') as f:
        for src, chunks in source_groups.items():
            std_id = source_std.get(src)
            for chunk in chunks:
                # 初始化metadata字段
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}
                # 仅当有标准号时才注入
                if std_id:
                    chunk['metadata']['standard_id'] = std_id
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

def extract_std_from_title(title: str) -> str:
    """仅从标题提取标准号（严格模式）"""
    match = re.search(r'(GB/T|DL/T|Q/GDW)[\s.]?\d+', title or '')
    return match.group(0) if match else None


if __name__ == "__main__":
    # 示例用法
    input_jsonl = "../retriever/rag_enhanced_chunks.jsonl"
    output_jsonl = "../retriever/rag_enhanced_2_chunks.jsonl"

    # enhance_chunks(input_jsonl, output_jsonl)
    process_jsonl(input_jsonl,output_jsonl)
    print(f"处理完成！增强后的数据已保存至 {output_jsonl}")