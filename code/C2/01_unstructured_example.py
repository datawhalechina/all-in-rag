from unstructured.partition.auto import partition

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



# # 分别用 hi_res 和 ocr_only 两种策略解析，对比输出
# for strategy in ["hi_res", "ocr_only"]:
#     print(f"\n{'#' * 30} strategy={strategy} {'#' * 30}")

#     elements = partition_pdf(
#         filename=pdf_path,
#         strategy=strategy,
#         languages=["chi_sim", "eng"],  # ocr_only 时识别中英文；只有 eng 会乱码
#     )

#     # 打印解析结果
#     print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

#     # 统计元素类型
#     types = Counter(e.category for e in elements)
#     print(f"元素类型: {dict(types)}")

#     # 显示前 5 个元素
#     print("前 5 个元素预览:")
#     for i, element in enumerate(elements[:5], 1):
#         print(f"Element {i} ({element.category}):")
#         print(str(element)[:200])
#         print("=" * 60)


