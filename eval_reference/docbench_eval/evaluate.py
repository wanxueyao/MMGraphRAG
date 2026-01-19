import json
import os
import logging
from eval_llm import get_llm_response

# 设置日志配置
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 数据目录和评估系统设置
data_dir = '/cpfs02/user/lidong1/xywan/project/eval'
data_file_dir = '/cpfs02/user/lidong1/xywan/temp_data/docbench_mmllm_ovis/data'
eval_system = 'ovis'


def _process_system_answers(file_path):
    """处理系统生成的答案，将答案合并成一个列表，并跳过1.之前的内容"""
    system_answers = []
    cur_qa_idx = 1
    cur_content = ""

    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 只处理以'1.'开始的行或后续的答案部分
            if line.startswith(f'{cur_qa_idx}.'):
                # 如果当前有内容，保存当前答案
                if cur_content:
                    system_answers.append(cur_content.strip())

                # 重新开始收集新的答案内容
                cur_content = line.split('.', 1)[1].strip()
                cur_qa_idx += 1
            elif cur_qa_idx > 1:  # 如果当前问题已经开始收集答案，则继续合并答案
                cur_content += " " + line

        # 处理最后一轮答案
        if cur_content:
            system_answers.append(cur_content.strip())

    return system_answers

def align_eval_input(eval_system):
    """对齐生成的答案和评估输入，生成新的评估输入文件"""

    eval_input_path = os.path.join(data_dir, f'{eval_system}_eval_input.jsonl')

    # 如果评估输入文件已经存在，则无需重新生成
    if os.path.exists(eval_input_path):
        return

    all_folders = os.listdir(data_file_dir)
    for folder in all_folders:
        folder_path = os.path.join(data_file_dir, folder)

        if os.path.isdir(folder_path) and not folder.startswith(('_', '.')):
            system_answers = _process_system_answers(os.path.join(folder_path, f'{eval_system}_results.txt'))
            qa_file_path = os.path.join(folder_path, f'{folder}_qa.jsonl')
            jsonlines = open(qa_file_path, 'r').readlines()

            new_dict_list = []
            for i, jsonline in enumerate(jsonlines):
                system_ans = system_answers[i].lstrip(f'{i+1}.').strip()
                json_dict = json.loads(jsonline)
                json_dict['sys_ans'] = system_ans
                json_dict['file'] = folder
                new_dict_list.append(json_dict)

            # 将新的QA数据写入评估输入文件
            with open(eval_input_path, 'a') as f:
                for json_dict in new_dict_list:
                    f.write(json.dumps(json_dict) + '\n')

    return


def evaluate(eval_system, resume_id=0):
    """执行评估过程，通过API生成评估结果并保存"""

    eval_prompt_path = os.path.join(data_dir, 'docbench_eval', 'evaluation_prompt.txt')
    eval_prompt = open(eval_prompt_path).read()
    system_content = 'You are a helpful evaluator.'

    eval_input_path = os.path.join(data_dir, f'{eval_system}_eval_input.jsonl')
    eval_output_path = os.path.join(data_dir, f'{eval_system}_eval_output.jsonl')

    with open(eval_input_path, 'r') as f:
        json_dict_list = [json.loads(line) for line in f.readlines()]

    for i, json_dict in enumerate(json_dict_list):
        if i < resume_id:
            continue

        # 提取问题、生成答案、参考答案和参考文本
        question = json_dict['question']
        sys_ans = json_dict['sys_ans']
        ref_ans = json_dict['answer']
        ref_text = json_dict['evidence']

        # 将模板中的占位符替换为实际内容
        cur_prompt = eval_prompt.replace('{{question}}', question).replace('{{sys_ans}}', sys_ans).replace(
            '{{ref_ans}}', ref_ans).replace('{{ref_text}}', ref_text)

        # 调用API生成评估结果
        response = get_llm_response(cur_prompt, system_content)
        json_dict['eval'] = response

        # 将评估结果保存到输出文件
        with open(eval_output_path, 'a') as f:
            f.write(json.dumps(json_dict) + '\n')

        print(f"-Finish {i}-th QA")

    return

def evaluation():
    """执行完整的评估流程"""
    
    resume_id = 0
    
    # 检查生成答案的情况
    passed_check = True  # 默认认为检查通过
    folders_to_check = []
    
    all_folders = os.listdir(data_file_dir)
    for folder in all_folders:
        folder_path = os.path.join(data_file_dir, folder)

        # 确保当前是一个有效的目录
        if os.path.isdir(folder_path) and not folder.startswith(('_', '.', 'data')):
            # 读取当前文件夹中的QA数据
            qa_file_path = os.path.join(folder_path, f'{folder}_qa.jsonl')
            system_answers_path = os.path.join(folder_path, f'{eval_system}_results.txt')

            # 读取QA数据和系统答案
            jsonlines = open(qa_file_path, 'r').readlines()
            system_answers = _process_system_answers(system_answers_path)

            # 如果系统答案数量与QA对数量不匹配，记录文件夹
            if len(system_answers) != len(jsonlines):
                folders_to_check.append(folder)

    # 输出检查结果
    if folders_to_check:
        print("The following folders need to be checked due to mismatched answer counts:")
        for folder in folders_to_check:
            print(f"- {folder}")
        passed_check = False
    else:
        print("All folders passed the check.")

    # 如果检查通过，执行对齐和评估
    if passed_check:
        align_eval_input(eval_system)
        evaluate(eval_system, resume_id=resume_id)


if __name__ == "__main__":
    evaluation()