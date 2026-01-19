import json
# 定义输出文件的路径
res_dir = '/Users/xueyaowan/Documents/硕士毕业设计/newproject/results/docbench/graph_llama/llama_eval_output.jsonl'

# 从指定的路径读取文件内容，并将每一行的JSON格式的数据转换为Python对象
with open(res_dir, 'r') as f:
    new_res_list = [json.loads(line) for line in f]
# 过滤出eval字段前20个字符中包含“1”的记录，这表示评估为正确
score1 = [res for res in new_res_list if "1" in res['eval'][:20]]
# 计算总准确率（micro accuracy），即“1”评估的结果占所有结果的比例
micro_acc = len(score1) / len(new_res_list)
# 定义评估类型的映射，将不同的类型简化为较短的标签
types = {
    'text-only': 'text',
    'multimodal-f': 'mm',
    'multimodal-t': 'mm',
    'multimodal': 'mm',
    'meta-data': 'meta',
    'una': 'una'
}
# 定义文件范围的映射，指定不同类别的文件索引范围
file_ranges = {
    'aca': range(0, 49),# 学术文件的范围
    'fin': range(49, 89),# 金融文件的范围
    'gov': range(89, 133),# 政府文件的范围
    'law': range(133, 179),# 法律文件的范围
    'new': range(179, 229)# 新闻文件的范围
}
# 初始化各类型的计数字典，统计每种类型的错误数和总数
type_counts = {key: {'wr': 0, 'total': 0} for key in types.values()}
# 初始化各文件范围的计数字典，统计每个文件范围的正确数和总数
file_counts = {key: {'cor': 0, 'total': 0} for key in file_ranges.keys()}

# 遍历每一个评估结果
for res in new_res_list:
    # 获取评估结果前20个字符
    evalres = res['eval'][:20]
    # 根据res['type']获取当前结果的类型，默认值为'una'
    res_type = types.get(res['type'], 'una')
     # 如果评估结果中有“0”，表示错误，增加该类型的错误计数
    if "0" in evalres:
        type_counts[res_type]['wr'] += 1
    type_counts[res_type]['total'] += 1
    # 增加该类型的总计数
    res_file = int(res['file'])
     # 检查当前文件编号属于哪个文件范围，并更新该范围的计数
    for key, f_range in file_ranges.items():
        if res_file in f_range:
            if "1" in evalres:
                file_counts[key]['cor'] += 1
            file_counts[key]['total'] += 1
            break
# 计算每种类型的准确率，使用1 - （错误数 / 总数）公式
type_acc = {key: 1 - val['wr'] / val['total'] for key, val in type_counts.items()}
file_acc = {key: val['cor'] / val['total'] for key, val in file_counts.items()}
# 打印每个文件范围的准确率，格式化为百分比
for key, acc in file_acc.items():
    print(f"{key.capitalize()} Accuracy: {acc * 100:.1f}%")
# 打印每种类型的准确率，格式化为百分比
for key, acc in type_acc.items():
    print(f"{key.capitalize()} Accuracy: {acc * 100:.1f}%")
# 打印总准确率，格式化为百分比
print(f"Accuracy: {micro_acc * 100:.1f}%")

# 计算总准确率（micro accuracy），即“1”评估的结果占所有结果的比例（排除meta类）
filtered_res_list = [res for res in new_res_list if types.get(res['type'], 'una') != 'meta']
micro_acc2 = len([res for res in filtered_res_list if "1" in res['eval'][:20]]) / len(filtered_res_list)

print(f"Accuracy(no meta): {micro_acc2 * 100:.1f}%")