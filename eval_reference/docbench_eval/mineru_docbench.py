import os
import subprocess
from tqdm import tqdm  # 导入tqdm，用于显示进度条

def convert_pdf_to_md(input_pdf_path, output_folder):
    # 构造命令
    command = [
        'magic-pdf',
        '-p', input_pdf_path,
        '-o', output_folder
    ]
    # 调用命令
    subprocess.run(command, check=True)

def process_folder(data_folder, output_folder):
    # 存储所有PDF文件路径
    pdf_files = []
    
    # 遍历 data 文件夹下的所有子文件夹
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # 遍历子文件夹中的所有 PDF 文件
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(subfolder_path, filename)
                    pdf_files.append(pdf_path)
    
    # 使用 tqdm 显示进度条
    for pdf_path in tqdm(pdf_files, desc="转换中", unit="个文件"):
        convert_pdf_to_md(pdf_path, output_folder)

def main():
    data_folder = '/mnt/workspace/xywan/dataset/docbench/data'  # data 文件夹路径
    output_folder = '/mnt/workspace/xywan/dataset/docbench/docbench_mineru'  # 输出文件夹路径
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有待转换的PDF文件
    pdf_files = []
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # 遍历子文件夹中的所有 PDF 文件
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.pdf'):
                    pdf_files.append(os.path.join(subfolder_path, filename))
    
    # 显示文件总数
    total_files = len(pdf_files)
    print(f"总共有 {total_files} 个PDF文件需要转换。")

    # 开始处理文件夹中的 PDF 文件
    processed_count = 0
    for pdf_path in tqdm(pdf_files, desc="转换中", unit="个文件"):
        convert_pdf_to_md(pdf_path, output_folder)
        processed_count += 1
        print(f"已处理 {processed_count}/{total_files} 个文件。")
    
    print("所有PDF文件转换完成！")

if __name__ == '__main__':
    main()