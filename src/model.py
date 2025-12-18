# src/model.py

import numpy as np
from PIL import Image

# 这是一个假模型，用于在成员 A 还没完成代码时，确保 B 的 UI 逻辑能跑通。

# 假设所有图片都会被缩放到 64x64，这是一个常见的 VAE 输入尺寸
TARGET_SIZE = (64, 64)


class FakeEncoder:
    """
    假 Encoder：接收 PIL 图像，返回一个假向量 (例如 128 维全 0 或全 1 向量)。
    """

    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim

    def encode(self, image: Image.Image) -> np.ndarray:
        # 实际 VAE 会将图片转为张量并得到潜在向量
        # 假模型简单返回一个基于图片大小的假向量
        image = image.resize(TARGET_SIZE)

        # 模拟潜在空间编码：这里返回一个代表 "平均颜色" 的向量
        # 随机生成一个向量来模拟潜在空间
        vector = np.random.randn(self.latent_dim).astype(np.float32)
        # 为了让插值有效果，让 A 图返回接近 0 的向量，B 图返回接近 1 的向量 (方便 np.linspace)
        if image.mode == 'RGB':
            avg_color = np.array(image).mean(axis=(0, 1)) / 255.0
            vector[:3] = avg_color  # 把向量前 3 维设为平均颜色，让它有区分度

        return vector


class FakeDecoder:
    """
    假 Decoder：接收向量，返回一张假图像。
    """

    def __init__(self):
        pass

    def decode(self, vector: np.ndarray) -> Image.Image:
        # 实际 VAE 会将向量转为张量并生成图片
        # 假模型：基于输入向量的属性生成一个简单颜色的图片
        # 假设向量的前 3 维代表某种颜色信息 (如 RGB)

        r = int(np.clip(vector[0] * 255, 0, 255))
        g = int(np.clip(vector[1] * 255, 0, 255))
        b = int(np.clip(vector[2] * 255, 0, 255))

        # 生成一张纯色图片，颜色由向量决定
        # 注意：这里需要确保输入图片在 FakeEncoder 中有足够的特征来区分生成的颜色
        color = (r, g, b)
        # 返回一张 64x64 的纯色图片
        return Image.new("RGB", TARGET_SIZE, color)


# --- 成员 A 给 B 预留的接口 (在 A 的 api.py 中，这里也实现一个假的) ---
def interpolate_images_fake(img1: Image.Image, img2: Image.Image, steps: int = 10):
    """
    【假接口实现】
    输入：两张 PIL 图片
    输出：steps 张渐变图片的列表 (使用假模型)
    """
    encoder = FakeEncoder()
    decoder = FakeDecoder()

    # 1. 接收图片，调用模型 Encoder 得到向量 Z_A 和 Z_B
    z_a = encoder.encode(img1)
    z_b = encoder.encode(img2)

    # 2. 使用 np.linspace 生成中间向量 Z_interp
    # 注意：np.linspace 默认包含起点和终点，共 steps 个点。
    interp_vectors = []
    for i in range(steps):
        # 计算插值系数 (从 0.0 到 1.0)
        # np.linspace(0.0, 1.0, steps)
        t = i / (steps - 1) if steps > 1 else 0.0

        # 线性插值：Z_interp = (1-t) * Z_A + t * Z_B
        z_interp = (1 - t) * z_a + t * z_b
        interp_vectors.append(z_interp)

    # 3. 调⽤模型 Decoder 把 Z_interp 变成图⽚序列
    image_sequence = [decoder.decode(v) for v in interp_vectors]

    return image_sequence