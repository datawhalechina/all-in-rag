from dotenv import load_dotenv
load_dotenv()  # 加载 .env（HF 镜像），必须在模型加载前

import torch
from visual_bge.visual_bge.modeling import Visualized_BGE

# 初始化 Visualized_BGE 多模态模型
# model_name_bge: 指定基础 BGE 模型，这里使用英文 base 版
# model_weight: 指定多模态权重的路径，需要根据你实际下载的位置修改
model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth")

# 设置为评估模式，关闭 dropout 等训练层，保证推理结果稳定
model.eval()

# 禁用梯度计算，节省显存并加速推理（只做前向传播）
with torch.no_grad():
    # 编码纯文本，得到文本嵌入向量，形状通常为 [1, 嵌入维度]
    text_emb = model.encode(text="blue whale")
    
    # 编码第一张纯图像，得到图像嵌入向量
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")
    
    # 同时编码第一张图像和对应文本，得到图文融合的多模态嵌入向量
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="blue whale")
    
    # 编码第二张纯图像
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")
    
    # 同时编码第二张图像和相同文本，得到第二个多模态嵌入向量
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="blue whale")

    # 对照组：第一张图像配原来的中文文本，用于和 blue whale 组合直接对比
    multi_emb_cn = model.encode(image="../../data/C3/imgs/datawhale01.png", text="datawhale开源组织的logo")

# 计算相似度
# 使用 @ 进行矩阵乘法（点积）。若向量已归一化，点积等价于余弦相似度，值越大越相似。
sim_1 = img_emb_1 @ img_emb_2.T          # 纯图像1 与 纯图像2 的相似度
sim_2 = img_emb_1 @ multi_emb_1.T        # 纯图像1 与 图文结合1 的相似度
sim_3 = text_emb @ multi_emb_1.T         # 纯文本 与 图文结合1 的相似度
sim_4 = multi_emb_1 @ multi_emb_2.T      # 图文结合1 与 图文结合2 的相似度

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 同图不同文本的对比：差异纯来自文本，最能看出文本对多模态嵌入的影响
print(f"同图对比（中文文本 vs blue whale）: {multi_emb_cn @ multi_emb_1.T}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}")          # 通常为 [1, embedding_dim]
print(f"图像向量维度: {img_emb_1.shape}")              # 通常为 [1, embedding_dim]
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")  # 查看多模态向量的前10个数值
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")    # 查看图像向量的前10个数值