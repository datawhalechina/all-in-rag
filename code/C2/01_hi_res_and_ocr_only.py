from unstructured.partition.pdf import partition_pdf
from collections import Counter

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

def summarize(elements, tag: str, show_n: int = 10):
    # 统计元素数 + 字符数（用 str(element) 近似衡量）
    total_chars = sum(len(str(e)) for e in elements)
    types = Counter(getattr(e, "category", type(e).__name__) for e in elements)

    print(f"\n========== {tag} ==========")
    print(f"解析完成: {len(elements)} 个元素, {total_chars} 字符")
    print(f"元素类型: {dict(types)}")

    # 展示前 N 个元素，避免输出瀑布
    print(f"\n前 {show_n} 个元素预览：")
    for i, e in enumerate(elements[:show_n], 1):
        cat = getattr(e, "category", type(e).__name__)
        text = str(e).strip().replace("\n", " ")
        if len(text) > 200:
            text = text[:200] + " ..."
        print(f"[{i}] ({cat}) {text}")

# 1) hi_res：高分辨率布局解析（通常会做版面检测/更细粒度的结构识别）
elements_hi = partition_pdf(
    filename=pdf_path,
    strategy="hi_res",
    languages=["zh"],  
)

summarize(elements_hi, "strategy=hi_res")

# 2) ocr_only：完全走OCR（把页面当图片识别文字；对扫描版/图片PDF更有用）
elements_ocr = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only",
    languages=["zh"],  # 同上：若不支持就删
)

summarize(elements_ocr, "strategy=ocr_only")

# 如果想像原来一样打印所有元素，可以在对比完再打开下面：
# for i, element in enumerate(elements_hi, 1):
#     print(f"HI_RES Element {i} ({element.category}):\n{element}\n" + "="*60)
# for i, element in enumerate(elements_ocr, 1):
#     print(f"OCR_ONLY Element {i} ({element.category}):\n{element}\n" + "="*60)