# import pandas as pd
# import csv

# df1 = pd.read_csv('/Users/wukeyang/ntu_course/2024-HTML/final_project/output/logstic aug select 65 debug_(0.58.csv')
# df2 = pd.read_csv('/Users/wukeyang/ntu_course/2024-HTML/final_project/lgb_submission.csv')
# df3 = pd.read_csv('/Users/wukeyang/ntu_course/2024-HTML/final_project/SelectK35.csv')
# df4 = pd.read_csv('/Users/wukeyang/ntu_course/2024-HTML/final_project/siamese_network_predictions.csv')

# df_list = [df1, df2, df3, df4]

# for df_index in range(len(df_list) - 1):
#     for next_df_index in range(1, len(df_list)):
#         if len(df_list[df_index]) != len(df_list[next_df_index]):
#             raise ValueError("different len")
        
#         different_values = (df_list[df_index]['home_team_win'] != df_list[next_df_index]['home_team_win']).sum()
#         print(f"{df_index} and {next_df_index} has {different_values} different")

# votes = pd.concat([df['home_team_win'] for df in df_list], axis=1)

# final_votes = votes.sum(axis=1) >= 3

# final_result = final_votes.astype(bool)

# print(final_result)

# data_list = final_result

# output_file = 'output/stage1 4 vote.csv'

# with open(output_file, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['id', 'home_team_win'])
    
#     for idx, value in enumerate(data_list):
#         writer.writerow([idx, value])

# print(f"CSV file '{output_file}' generated successfully.")

#%%
import pandas as pd
import csv

df1 = pd.read_csv('/Users/wukeyang/ntu_course/2024-HTML/final_project/output/stage1 4 vote.csv')

# %%
