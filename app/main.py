import streamlit as st
from PIL import Image
import numpy as np
import io
import imageio.v2 as imageio


# ==========================================
# 核心逻辑区：假模型（集成时替换此处）
# ==========================================

def interpolate_images(img1: Image.Image, img2: Image.Image, steps: int = 10):
    """
    这是成员 B 编写的插值逻辑框架。
    将来成员 A 只需要提供 Encoder 和 Decoder，替换内部逻辑即可。
    """
    # 模拟预处理：统一尺寸（VAE通常要求固定尺寸，如64x64）
    size = (128, 128)
    img1_resized = img1.resize(size)
    img2_resized = img2.resize(size)

    arr1 = np.array(img1_resized).astype(np.float32)
    arr2 = np.array(img2_resized).astype(np.float32)

    image_list = []

    # 使用 np.linspace 生成线性插值序列
    # 这里的逻辑模拟了潜在空间(Latent Space)的向量平滑过渡
    for i in range(steps):
        alpha = i / (steps - 1)
        # 线性插值公式：(1 - alpha) * A + alpha * B
        interp_array = (1 - alpha) * arr1 + alpha * arr2

        # 转回 PIL 图片格式
        interp_img = Image.fromarray(interp_array.astype(np.uint8))
        image_list.append(interp_img)

    return image_list


def create_gif(image_list, duration=0.1):
    """将图片序列转为 GIF 字节流"""
    if not image_list:
        return None
    images_np = [np.array(img) for img in image_list]
    output = io.BytesIO()
    imageio.mimsave(output, images_np, format='GIF', duration=duration, loop=0)
    return output.getvalue()


# ==========================================
# UI 界面区：Streamlit 配置
# ==========================================

st.set_page_config(page_title="VAE 图像渐变系统", layout="wide")

st.title("🌌 VAE 智能图像渐变系统")
st.info("开发版：已集成内置插值引擎，可直接测试 UI 流程。")

# 1. 上传区域
st.header("1. 上传图片")
col_a, col_b = st.columns(2)

with col_a:
    file_a = st.file_uploader("选择起始图 A", type=["png", "jpg", "jpeg"], key="a")
    if file_a:
        img_a = Image.open(file_a).convert("RGB")
        st.image(img_a, caption="图 A (起始)", width=300)

with col_b:
    file_b = st.file_uploader("选择目标图 B", type=["png", "jpg", "jpeg"], key="b")
    if file_b:
        img_b = Image.open(file_b).convert("RGB")
        st.image(img_b, caption="图 B (目标)", width=300)

st.divider()

# 2. 参数与生成
if file_a and file_b:
    st.header("2. 设置与生成")

    with st.sidebar:
        st.title("⚙️ 参数控制")
        steps = st.slider("渐变步数", 5, 30, 15)
        speed = st.slider("每帧时长(秒)", 0.05, 0.5, 0.1)
        run_button = st.button("🚀 开始生成渐变", type="primary", use_container_width=True)

    if run_button:
        with st.spinner("正在通过 VAE 潜在空间进行插值..."):
            # 调用插值函数
            sequence = interpolate_images(img_a, img_b, steps=steps)

            if sequence:
                st.subheader("🎬 渐变动画预览")
                gif_data = create_gif(sequence, duration=speed)
                st.image(gif_data, use_container_width=True)

                st.download_button("📥 下载动画 (GIF)", gif_data, "result.gif", "image/gif")

                st.divider()
                st.subheader("🖼️ 帧序列详情")
                cols = st.columns(min(steps, 8))  # 每行最多显示8张
                for idx, frame in enumerate(sequence):
                    cols[idx % 8].image(frame, caption=f"F-{idx + 1}", use_container_width=True)

                st.balloons()
else:
    st.warning("👈 请先在上方上传两张图片以开启魔法！")