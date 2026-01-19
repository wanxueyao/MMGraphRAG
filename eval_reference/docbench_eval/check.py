import os
import logging
import subprocess
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO)

def check_processed_files(output_folder, input_folder):
    """检查输出文件夹中的文件夹是否处理正常"""
    problematic_folders = []

    for root, dirs, files in os.walk(output_folder):
        if root == output_folder:
            for dir_name in dirs:
                target_folder = os.path.join(output_folder, dir_name)

                # 确保每个目标文件夹内只有一个子文件夹
                subfolders = [f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f))]
                
                if len(subfolders) != 1:
                    # 如果该文件夹内不是唯一子文件夹，说明处理出错
                    problematic_folders.append(target_folder)
                    continue

                # 找到唯一子文件夹
                subfolder = os.path.join(target_folder, subfolders[0])

                # 查找是否有 .md 文件
                md_files = [file for file in os.listdir(subfolder) if file.endswith('.md')]

                if not md_files:
                    # 如果没有找到 .md 文件，认为该文件夹处理有问题
                    problematic_folders.append(target_folder)

    return problematic_folders


def delete_problematic_folders(problematic_folders):
    """删除有问题的文件夹"""
    for folder in problematic_folders:
        try:
            # 删除有问题的文件夹
            shutil.rmtree(folder)
            logging.info(f"已删除有问题的文件夹: {folder}")
        except Exception as e:
            logging.error(f"删除文件夹 {folder} 时发生错误: {e}")


def process_with_magic_pdf(problematic_folders, input_folder, output_folder):
    """使用magic-pdf命令重新处理有问题的文件夹"""
    for folder in problematic_folders:
        try:
            # 获取文件夹名称（与PDF文件名相同）
            folder_name = os.path.basename(folder)

            # 根据文件夹名称在输入路径中找到对应的PDF文件
            pdf_file = find_pdf_by_name(input_folder, folder_name)

            if not pdf_file:
                logging.error(f"未找到与文件夹 {folder_name} 对应的PDF文件")
                continue

            # 使用magic-pdf命令重新处理
            logging.info(f"正在使用magic-pdf重新处理文件夹: {folder_name}, PDF文件: {pdf_file}")
            
            # 调用magic-pdf命令，传递PDF路径和输出路径
            subprocess.run(['magic-pdf', '-p', pdf_file, '-o', output_folder], check=True)
            
            logging.info(f"重新处理成功: {folder_name}")
        except subprocess.CalledProcessError as e:
            logging.error(f"重新处理文件夹 {folder} 时发生错误: {e}")
        except Exception as e:
            logging.error(f"处理文件夹 {folder} 时发生其他错误: {e}")


def find_pdf_by_name(input_folder, folder_name):
    """根据文件夹名称查找对应的PDF文件"""
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.startswith(folder_name) and file.endswith('.pdf'):
                return os.path.join(root, file)
    return None


# 输出文件夹路径
output_folder = '/mnt/workspace/xywan/dataset/docbench/docbench_mineru'
# output_folder = '/mnt/workspace/xywan/dataset/mmlongbench/mmlongbench_mineru'
# 输入文件夹路径
input_folder = '/mnt/workspace/xywan/dataset/docbench/data'    

# 第一步：检查所有文件夹是否处理正常
problematic_folders = check_processed_files(output_folder, input_folder)

if problematic_folders:
    print("以下文件夹处理有问题:")
    for folder in problematic_folders:
        print(folder)

    # 第二步：删除有问题的文件夹
    delete_problematic_folders(problematic_folders)

    # 第三步：使用magic-pdf重新处理这些文件夹
    process_with_magic_pdf(problematic_folders, input_folder, output_folder)

else:
    print("所有文件处理正常。")

