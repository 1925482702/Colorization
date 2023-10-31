import torch
import torch.nn as nn

# 定义参数重新平衡函数
def class_rebalance(Zb, lambda_val=0.5, sigma=5):
    """
    对参数进行重新平衡的函数。

    参数:
        Zb (torch.Tensor): 预测的颜色分布.
        lambda_val (float): 控制类别重新平衡的超参数.
        sigma (int): 高斯核的标准差.

    返回:
        torch.Tensor: 类别重新平衡后的权重.
    """
    Q = Zb.shape[2] # 获取了预测颜色分布 Zb 中的通道数，即类别的数量
    pe = torch.ones(Q) / Q  # 创建了一个长度为 Q 的张量，其中的每个元素都是 1/Q，表示了每个类别的等概率权重
    q_values = torch.arange(Q) # 创建了一个从 0 到 Q-1 的整数序列，代表了所有可能的类别
    w = (1 - lambda_val) * pe + lambda_val / Q # w = （（1-λ）p + λ/Q）^-1
    w = w / w.sum() # 规范化，使得 E[W] = ∑ pe * w = 1
    wq = w[q_values] # 从权重向量 w 中选择与当前像素的 ab 值相对应的权重值
    return wq.unsqueeze(0).unsqueeze(0) # 将 wq 转换成适合后续计算的形状

# 定义多元交叉熵损失函数
def multinomial_cross_entropy_loss(Zb, Z, lambda_val=0.5, sigma=5):
    # h 和 w 分别表示图像的高度和宽度的索引
    # q 表示颜色类别的索引。
    # Zb是模型预测的颜色分布
    # Z 是目标颜色分布
    # v(Zh,w) 是类别重新平衡权重。
    # lambda_val (float): 控制类别重新平衡的超参数.
    # sigma (int): 高斯核的标准差.
    #这个损失函数的含义是，对于每个像素点(h,w)，计算其预测颜色分布Zb与目标颜色分布Z之间的交叉熵损失，同时考虑了类别重新平衡权重 v(Zh,w)。

    # λ = 1/2：这里的 λ 是用于控制类别重新平衡的超参数。λ = 1/2 这意味着作者认为原始分布和重新平衡后的分布对于获得良好的结果具有相等的重要性。
    # 也就是说，作者认为平衡了原始分布和重新平衡后的分布对于获得良好的结果很重要。
    # σ = 5：这里的 σ 是高斯核的标准差，用于对颜色分布进行平滑处理。
    # σ = 5 表示作者选择了一个适当的标准差来确保对颜色分布的平滑处理不会过度模糊，也不会使得细节丢失，从而保持了对颜色的准确性。
    wq = class_rebalance(Zb, lambda_val, sigma)

    # 通过使用 torch.sum，它实际上进行了两次求和，将内外两层的求和合并在一起。
    # 也就是说，wq * Z * torch.log(Zb) 中的每个元素会参与两次求和操作，分别对应公式中的两次求和。
    loss = -torch.sum(wq * Z * torch.log(Zb))
    return loss

# 定义将概率映射到点的转换函数
def annealed_mean(Zb, temperature=0.38):
    """
    将类别概率转换为点估计的函数。

    参数:
        Zb (torch.Tensor): 预测的颜色分布.
        temperature (float): 控制分布平滑的超参数.

    返回:
        torch.Tensor: 转换得到的点估计.
    """
    # temperature 最佳退火温度参数
    fz = torch.exp(torch.log(Zb) / temperature)

    # fz 的维度结构是 [batch_size, height, width, num_classes]， dim = 3即颜色类别，keepdim为保持当前维度
    Z_mean = torch.sum(fz, dim=3, keepdim=True)
    return Z_mean
