from PIL import Image
import os
from hyperparameters import *  # 假设该文件中定义了exp_name
import sys

def images_to_gif(image_folder, output_folder, output_filename, duration=200, loop=0):
    """
    将文件夹中的图片合成GIF动图
    
    参数:
    image_folder: 包含图片的文件夹路径
    output_folder: 输出GIF的文件夹路径
    output_filename: 输出GIF的文件名（包含扩展名）
    duration: 每帧持续时间(毫秒)
    loop: 循环次数，0表示无限循环
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    if not image_files:
        print(f"错误：在 {image_folder} 中未找到任何图片文件")
        return
    
    # 按文件名排序（确保图片顺序正确）
    # 针对frame_xxx.png格式的文件进行数字排序
    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # 打开所有图片
    images = []
    for file in image_files:
        file_path = os.path.join(image_folder, file)
        try:
            with Image.open(file_path) as img:
                images.append(img.copy())
        except Exception as e:
            print(f"无法打开图片 {file_path}: {e}")
    
    if not images:
        print("没有可处理的有效图片")
        return
    
    # 构建完整的输出路径
    output_gif = os.path.join(output_folder, output_filename)
    
    # 保存为GIF
    images[0].save(
        output_gif,
        format='GIF',
        append_images=images[1:],
        save_all=True,
        duration=duration,
        loop=loop,
        disposal=2  # 每帧显示后清除
    )
    
    print(f"GIF动图已保存到 {output_gif}")

def main():
    # 构建路径
    logsdir = os.path.join('logs', exp_name)
    
    # 包含图片的文件夹：与gif文件夹同一目录下的smoke文件夹
    #image_directory = os.path.join(logsdir, 'smoke')
    
    # 输出GIF的文件夹：logs/exp_name/gif
    output_gif_directory = os.path.join(logsdir, 'gif')
    
    # 调用函数生成GIF
    # image_directory = os.path.join(logsdir, 'smoke')
    # images_to_gif(
    #     image_folder=image_directory,
    #     output_folder=output_gif_directory,
    #     output_filename='smoke2.gif',  # 输出文件名为smoke.gif
    #     duration=10,  # 每帧持续100毫秒（10帧/秒）
    #     loop=0  # 无限循环
    # )

    image_directory = os.path.join(logsdir, 'vorticity')
    images_to_gif(
        image_folder=image_directory,
        output_folder=output_gif_directory,
        output_filename='vorticity.gif',  # 输出文件名为smoke.gif
        duration=10,  # 每帧持续100毫秒（10帧/秒）
        loop=0  # 无限循环
    )

if __name__ == "__main__":
    main()
