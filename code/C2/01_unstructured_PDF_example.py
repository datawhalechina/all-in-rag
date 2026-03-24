import os
from collections import Counter
from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

def analyze_pdf(strategy):
    print(f"\n{"#"*20} 正在使用策略: {strategy} {"#"*20}")
    
    # 使用 partition_pdf 替换 partition
    elements = partition_pdf(
        filename=pdf_path,
        strategy=strategy,            # 指定策略: hi_res 或 ocr_only
        infer_table_structure=True,   # 如果是 hi_res，尝试识别表格结构
        model_name="yolox"            # hi_res 默认使用的布局模型
    )

    # 1. 统计基本信息
    print(f"解析完成: {len(elements)} 个元素")
    
    # 2. 统计元素类型
    types = Counter(e.category for e in elements)
    print(f"元素类型分布: {dict(types)}")

    # 3. 观察前几个元素的具体分类和内容
    print("\n前 5 个元素详情:")
    for i, element in enumerate(elements[:5], 1):
        print(f"[{i}] 类型: {element.category:15} | 内容预览: {str(element)[:50]}...")

# 分别执行两种策略
analyze_pdf(strategy="ocr_only")
analyze_pdf(strategy="hi_res")