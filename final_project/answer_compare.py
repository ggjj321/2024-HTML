import pandas as pd

# 读取两个 CSV 文件
df1 = pd.read_csv('/Users/wukeyang/ntu_course/2024-HTML/final_project/output/one hot stage1 cross val with aug.csv')
df2 = pd.read_csv('/Users/wukeyang/ntu_course/2024-HTML/final_project/output/one hot stage1 cross val without aug.csv')
df3 = pd.read_csv('/Users/wukeyang/ntu_course/2024-HTML/final_project/output/svm stage1 cross val with aug.csv')
df4 = pd.read_csv('/Users/wukeyang/ntu_course/2024-HTML/final_project/output/stage1 3 vote.csv')

# 确保两个数据框的长度相同
if len(df1) != len(df2):
    raise ValueError("两个文件的长度不同，无法逐行比较")

# 获取同一栏位在两个数据框中不同值的行数
different_values = (df1['home_team_win'] != df4['home_team_win']).sum()

print(f"在'{df1['home_team_win'].name}'栏位中，不同值的个数是: {different_values}")

different_values = (df2['home_team_win'] != df4['home_team_win']).sum()

print(f"在'{df1['home_team_win'].name}'栏位中，不同值的个数是: {different_values}")

different_values = (df3['home_team_win'] != df4['home_team_win']).sum()

print(f"在'{df1['home_team_win'].name}'栏位中，不同值的个数是: {different_values}")

# votes = pd.concat([df1['home_team_win'], df2['home_team_win'], df3['home_team_win']], axis=1)

# final_votes = votes.sum(axis=1) >= 2

# # 将布尔结果转换为 True/False
# final_result = final_votes.astype(bool)

# # 打印最终投票结果
# print(final_result)

# import csv

# # 假設有一個列表
# data_list = final_result

# # 指定輸出的 CSV 文件名稱
# output_file = 'output/stage1 3 vote.csv'

# # 打開文件並寫入內容
# with open(output_file, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     # 寫入標題行（可選）
#     writer.writerow(['id', 'home_team_win'])
#     # 寫入數據
#     for idx, value in enumerate(data_list):
#         writer.writerow([idx, value])

# print(f"CSV file '{output_file}' generated successfully.")