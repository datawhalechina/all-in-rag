from unstructured.partition.auto import partition
# from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档
elements = partition(
    filename=pdf_path,
    content_type="application/pdf"
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)

# # ========== 方案1: hi_res 高精度解析 ==========
# print("===== hi_res 模式解析 =====")
# elements_hi_res = partition_pdf(
#     filename=pdf_path,
#     strategy="hi_res",
# )

# # 打印解析结果
# for elem in elements_hi_res:
#     print(f"类型: {elem.category}, 文本: {elem.text}")

# # ========== 方案2: ocr_only 纯OCR解析（扫描PDF） ==========
# print("\n===== ocr_only 模式解析 =====")
# elements_ocr = partition_pdf(
#     filename=pdf_path,
#     strategy="ocr_only",
# )

# for elem in elements_ocr:
#     print(f"类型: {elem.category}, 文本: {elem.text}")