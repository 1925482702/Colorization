import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from CNN_Colorization.colorizers import *
import torch
from skimage import color
from cGANs_Colorization.models.networks import define_G
import torchvision.transforms as transforms

# 设置模型参数
input_nc = 1  # 输入通道数，灰度图L通道
output_nc = 2  # 输出通道数
ngf = 64  # 生成器最后一个卷积层通道数
netG = 'unet_256'  # 使用的生成器网络类型为unet_256
norm = 'batch'  # 使用的归一化方式为batch normalization

# 创建和加载模型
modelG = define_G(input_nc, output_nc, ngf, netG, norm, False)
model_path = r'C:\Users\彭英麒\Desktop\Colorization_2\cGANs_Colorization\checkpoints\oxford102flower_colorization\building_net_G.pth'  # 更改为你的模型路径
params = torch.load(model_path, map_location='cpu')
modelG.load_state_dict(params)
modelG.eval()

# 定义将 NumPy 数组转换为 Image 对象的函数
def convert_to_image(np_img):
    if np_img.max() > 1:
        np_img = np_img / 255.0
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

# 在第二个页面显示生成的图像的函数
def display_generated_images(orig, bw, cnn, cnn_short, colorized_img):
    second_window = tk.Toplevel(root)
    second_window.title("生成的图像")

    # 将 numpy 数组转换为 Image 对象
    orig = convert_to_image(orig)
    bw = convert_to_image(bw)
    cnn = convert_to_image(cnn)
    cnn_short = convert_to_image(cnn_short)

    # 确定缩放比例
    max_dim = max(orig.size[0], orig.size[1])
    scale_factor = 256 / max_dim

    # 调整图像大小以适应窗口
    display_size = (int(orig.size[0] * scale_factor), int(orig.size[1] * scale_factor))
    orig = orig.resize(display_size)
    bw = bw.resize(display_size)
    cnn = cnn.resize(display_size)
    cnn_short = cnn_short.resize(display_size)

    # 显示图像
    orig_img = ImageTk.PhotoImage(orig)
    orig_label = tk.Label(second_window, image=orig_img)
    orig_label.image = orig_img
    orig_label.grid(row=0, column=0)

    cnn_img = ImageTk.PhotoImage(cnn)
    cnn_label = tk.Label(second_window, image=cnn_img)
    cnn_label.image = cnn_img
    cnn_label.grid(row=1, column=0)

    cnn_short_img = ImageTk.PhotoImage(cnn_short)
    cnn_short_label = tk.Label(second_window, image=cnn_short_img)
    cnn_short_label.image = cnn_short_img
    cnn_short_label.grid(row=1, column=1)

    # 显示第一段代码生成的彩色图像
    colorized_img = Image.fromarray(colorized_img)
    colorized_img = colorized_img.resize(display_size)
    colorized_img_tk = ImageTk.PhotoImage(colorized_img)
    colorized_label = tk.Label(second_window, image=colorized_img_tk)
    colorized_label.image = colorized_img_tk
    colorized_label.grid(row=0, column=1, padx=10, pady=10)

    cnn.save("./img_out/img_out_cnn/img_out_cnn.png")
    cnn_short.save("./img_out/img_out_cnn_short/img_out_cnn_short.png")
    colorized_img.save("./img_out/img_out_cgans/img_out_cgans.png")

    second_window.resizable(width=False, height=False)


def colorize_image(img_path):
    # 加载选定的图像
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    img = np.array(img)

    # 转换为 Lab 色彩空间，并提取 L 通道
    img_lab = color.rgb2lab(img).astype(np.float32)
    img_l = img_lab[:, :, 0]
    img_l_transformed = (transforms.ToTensor()(img_l) / 50.0) - 1.0
    img_l_transformed = img_l_transformed.unsqueeze(0)

    # 使用模型生成彩色图像
    with torch.no_grad():
        ab_output = modelG(img_l_transformed)
    ab_output = (ab_output.squeeze(0).numpy().transpose(1, 2, 0) * 110.0).astype(np.float64)

    # 合并L通道和生成的AB通道
    colorized_img_lab = np.stack([img_l, ab_output[:, :, 0], ab_output[:, :, 1]], axis=-1)
    colorized_img_rgb = (color.lab2rgb(colorized_img_lab) * 255).astype(np.uint8)

    return colorized_img_rgb


# 定义生成彩色图像的函数
def generate_colored_image():
    img_path = file_path_entry.get()
    if not img_path:
        messagebox.showerror("错误", "请选择一张图片。")
        return

    # 加载选定的图像
    img = Image.open(img_path)
    img = np.array(img)

    # 处理图像
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    # if use_gpu.get():
    #     tens_l_rs = tens_l_rs.cuda()

    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_cnn = postprocess_tens(tens_l_orig, colorizer_cnn(tens_l_rs).cpu())
    out_img_cnn_short = postprocess_tens(tens_l_orig, colorizer_cnn_short(tens_l_rs).cpu())

    # 调用第一段代码中的函数来生成彩色图像
    colorized_img = colorize_image(img_path)

    display_generated_images(img, img_bw, out_img_cnn, out_img_cnn_short, colorized_img)

    # 清空输入框内容
    file_path_entry.delete(0, 'end')

# 创建主窗口
root = tk.Tk()
root.title("图像着色")

# 创建和配置输入文件路径输入框
file_path_entry = tk.Entry(root, width=50)
file_path_entry.pack(pady=10)

# 创建打开文件对话框按钮
browse_button = tk.Button(root, text="浏览图片",
                          command=lambda: file_path_entry.insert(0, filedialog.askopenfilename()))
browse_button.pack(pady=10)

# use_gpu = tk.BooleanVar()
# use_gpu_checkbox = tk.Checkbutton(root, text="使用GPU", variable=use_gpu)
# use_gpu_checkbox.pack(pady=10)

# 创建生成彩色图像的按钮
generate_button = tk.Button(root, text="生成彩色图像", command=generate_colored_image)
generate_button.pack(pady=20)

# 加载着色器
colorizer_cnn = CNN(pretrained=True).eval()
colorizer_cnn_short = CNN_Short(pretrained=True).eval()

root.mainloop()
