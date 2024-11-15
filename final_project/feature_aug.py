import pandas as pd
import numpy as np

# 1 勝率好像是算到歷史?
def feature_aug(df, training_data=None):
    """
    參數:
    df: 要進行特徵工程的資料框（訓練資料或測試資料）
    training_data: 訓練資料框，用於計算勝率相關特徵（當 df 是測試資料時，需要提供）
    """
    df = df.copy()
    
    if training_data is not None:
        training_data = training_data.copy()
    else:
        training_data = df.copy()
    
    # 轉換日期格式
    df['date'] = pd.to_datetime(df['date'])
    training_data['date'] = pd.to_datetime(training_data['date'])
    
    # 確保 season 為整數類型
    df['season'] = df['season'].astype(int)
    training_data['season'] = training_data['season'].astype(int)
    
    # 確保球隊和投手名稱為字串類型
    for col in ['home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']:
        df[col] = df[col].astype(str)
        if training_data is not None:
            training_data[col] = training_data[col].astype(str)
    
    # 判斷是否為測試資料
    is_test = 'home_team_win' not in df.columns
    
    try:
        def calculate_team_stats(data):
            # 合併主場和客場記錄
            all_games = pd.DataFrame()
            
            # 加入主場記錄
            home_games = data[['date', 'season', 'home_team_abbr', 'home_team_win']].copy()
            home_games['is_home'] = True
            home_games.rename(columns={'home_team_win': 'win'}, inplace=True)
            
            # 加入客場記錄
            away_games = data[['date', 'season', 'away_team_abbr', 'home_team_win']].copy()
            away_games['is_home'] = False
            away_games['win'] = 1 - away_games['home_team_win']
            
            # 對每個球隊分別計算統計
            team_stats = {}
            
            # 處理所有球隊
            for team in pd.concat([data['home_team_abbr'], data['away_team_abbr']]).unique():
                # 獲取該隊主場比賽並按季節和日期排序
                team_home = home_games[home_games['home_team_abbr'] == team].copy()
                team_home['games'] = 1
                team_home = team_home.sort_values(['season', 'date'])
                
                # 獲取該隊客場比賽並按季節和日期排序
                team_away = away_games[away_games['away_team_abbr'] == team].copy()
                team_away['games'] = 1
                team_away = team_away.sort_values(['season', 'date'])
                
                # 按季節計算累積勝場和比賽數
                team_home['home_wins'] = team_home.groupby('season')['win'].cumsum().shift(1).fillna(0)
                team_home['home_games'] = team_home.groupby('season')['games'].cumsum().shift(1).fillna(0)
                
                team_away['away_wins'] = team_away.groupby('season')['win'].cumsum().shift(1).fillna(0)
                team_away['away_games'] = team_away.groupby('season')['games'].cumsum().shift(1).fillna(0)
                
                # 合併主客場記錄，保留season列
                team_stats[team] = pd.concat([
                    team_home[['date', 'season', 'home_wins', 'home_games']],
                    team_away[['date', 'season', 'away_wins', 'away_games']]
                ]).sort_values(['season', 'date'])
                
                # 填補缺失值
                team_stats[team] = team_stats[team].fillna(method='ffill').fillna(0)
                
                # 按季節分組計算總和
                team_stats[team]['total_wins'] = (
                    team_stats[team].groupby('season')
                    .apply(lambda x: x[['home_wins', 'away_wins']].fillna(0).sum(axis=1))
                    .reset_index(level=0, drop=True)
                )
                team_stats[team]['total_games'] = (
                    team_stats[team].groupby('season')
                    .apply(lambda x: x[['home_games', 'away_games']].fillna(0).sum(axis=1))
                    .reset_index(level=0, drop=True)
                )
                
                # 計算當季勝率
                team_stats[team]['win_rate'] = team_stats[team]['total_wins'] / team_stats[team]['total_games'].clip(lower=1)
                team_stats[team]['home_win_rate'] = team_stats[team]['home_wins'] / team_stats[team]['home_games'].clip(lower=1)
                team_stats[team]['away_win_rate'] = team_stats[team]['away_wins'] / team_stats[team]['away_games'].clip(lower=1)
                
                # 填補缺失值為0.5
                team_stats[team] = team_stats[team].fillna(0.5)
            
            return team_stats

        def calculate_pitcher_stats(data):
            # 合併主場和客場記錄
            all_games = pd.DataFrame()
            
            # 加入主場記錄
            home_games = data[['date', 'season', 'home_pitcher', 'home_team_win']].copy()
            home_games.rename(columns={'home_team_win': 'win'}, inplace=True)
            
            # 加入客場記錄
            away_games = data[['date', 'season', 'away_pitcher', 'home_team_win']].copy()
            away_games['win'] = 1 - away_games['home_team_win']
            
            # 對每個投手分別計算統計
            pitcher_stats = {}
            
            # 處理所有投手
            for pitcher in pd.concat([data['home_pitcher'], data['away_pitcher']]).unique():
                # 獲取該投手的所有比賽
                pitcher_home = home_games[home_games['home_pitcher'] == pitcher].copy()
                pitcher_home['games'] = 1
                
                pitcher_away = away_games[away_games['away_pitcher'] == pitcher].copy()
                pitcher_away['games'] = 1
                
                # 合併所有比賽
                pitcher_all = pd.concat([
                    pitcher_home[['date', 'season', 'win', 'games']],
                    pitcher_away[['date', 'season', 'win', 'games']]
                ]).sort_values(['season', 'date'])
                
                if len(pitcher_all) == 0:
                    continue
                
                # 計算當季統計
                pitcher_all['season_wins'] = pitcher_all.groupby('season')['win'].cumsum().shift(1).fillna(0)
                pitcher_all['season_games'] = pitcher_all.groupby('season')['games'].cumsum().shift(1).fillna(0)
                
                # 計算生涯統計
                pitcher_all['career_wins'] = pitcher_all['win'].cumsum().shift(1).fillna(0)
                pitcher_all['career_games'] = pitcher_all['games'].cumsum().shift(1).fillna(0)
                
                # 計算勝率
                pitcher_all['win_rate'] = pitcher_all['season_wins'] / pitcher_all['season_games'].clip(lower=1)
                pitcher_all['career_win_rate'] = pitcher_all['career_wins'] / pitcher_all['career_games'].clip(lower=1)
                
                # 填補缺失值為0.5
                pitcher_stats[pitcher] = pitcher_all[['date', 'win_rate', 'career_win_rate']].fillna(0.5)
            
            return pitcher_stats

        # 計算統計數據
        team_stats = calculate_team_stats(training_data)
        pitcher_stats = calculate_pitcher_stats(training_data)
        
        # 確保數據按日期排序
        df = df.sort_values('date')
        
        # 合併球隊統計
        for date in df['date'].unique():
            # 獲取當日比賽
            mask = (df['date'] == date)
            
            # 更新主隊統計
            for team in df[mask]['home_team_abbr'].unique():
                if team in team_stats:
                    # 找到該日期之前的最後一個統計數據
                    team_history = team_stats[team]
                    team_history = team_history[team_history['date'] < date]
                    
                    if len(team_history) > 0:
                        latest_stats = team_history.iloc[-1]
                        df.loc[mask & (df['home_team_abbr'] == team), 'home_team_win_rate'] = latest_stats['win_rate']
                        df.loc[mask & (df['home_team_abbr'] == team), 'home_team_home_win_rate'] = latest_stats['home_win_rate']
                        df.loc[mask & (df['home_team_abbr'] == team), 'home_team_away_win_rate'] = latest_stats['away_win_rate']
            
            # 更新客隊統計
            for team in df[mask]['away_team_abbr'].unique():
                if team in team_stats:
                    # 找到該日期之前的最後一個統計數據
                    team_history = team_stats[team]
                    team_history = team_history[team_history['date'] < date]
                    
                    if len(team_history) > 0:
                        latest_stats = team_history.iloc[-1]
                        df.loc[mask & (df['away_team_abbr'] == team), 'away_team_win_rate'] = latest_stats['win_rate']
                        df.loc[mask & (df['away_team_abbr'] == team), 'away_team_home_win_rate'] = latest_stats['home_win_rate']
                        df.loc[mask & (df['away_team_abbr'] == team), 'away_team_away_win_rate'] = latest_stats['away_win_rate']
        
            # 更新主隊投手統計
            for pitcher in df[mask]['home_pitcher'].unique():
                if pitcher in pitcher_stats:
                    # 找到該日期之前的最後一個統計數據
                    pitcher_history = pitcher_stats[pitcher]
                    pitcher_history = pitcher_history[pitcher_history['date'] < date]
                    
                    if len(pitcher_history) > 0:
                        latest_stats = pitcher_history.iloc[-1]
                        df.loc[mask & (df['home_pitcher'] == pitcher), 'home_pitcher_win_rate'] = latest_stats['win_rate']
                        df.loc[mask & (df['home_pitcher'] == pitcher), 'home_pitcher_career_win_rate'] = latest_stats['career_win_rate']
            
            # 更新客隊投手統計
            for pitcher in df[mask]['away_pitcher'].unique():
                if pitcher in pitcher_stats:
                    # 找到該日期之前的最後一個統計數據
                    pitcher_history = pitcher_stats[pitcher]
                    pitcher_history = pitcher_history[pitcher_history['date'] < date]
                    
                    if len(pitcher_history) > 0:
                        latest_stats = pitcher_history.iloc[-1]
                        df.loc[mask & (df['away_pitcher'] == pitcher), 'away_pitcher_win_rate'] = latest_stats['win_rate']
                        df.loc[mask & (df['away_pitcher'] == pitcher), 'away_pitcher_career_win_rate'] = latest_stats['career_win_rate']
        
        # 填補缺失值
        win_rate_cols = [
            'home_team_win_rate', 'away_team_win_rate',
            'home_team_home_win_rate', 'away_team_away_win_rate',
            'home_team_away_win_rate', 'away_team_home_win_rate',
            'home_pitcher_win_rate', 'away_pitcher_win_rate',
            'home_pitcher_career_win_rate', 'away_pitcher_career_win_rate'
        ]
        
        for col in win_rate_cols:
            df[col] = df[col].fillna(0.5)
    
    except Exception as e:
        print(f"錯誤: {str(e)}, 跳過計算勝率相關特徵")
        raise e

    
    
    # ======= 投手團隊趨勢指標 ==============================================================================================
    # 投手團隊 ERA 趨勢指標 ==============================================================================================
    try:
        # 主場投手ERA趨勢指標 (相較於所有年度平均)
        df['home_pitching_ERA_trend_all'] = (
            df['home_pitching_earned_run_avg_10RA'] - 
            df['home_pitching_earned_run_avg_mean']
        )
        # 主場投手ERA趨勢指標 (相較於同年度平均)
        df['home_pitching_ERA_trend_season'] = df.groupby('season').apply(
            lambda x: x['home_pitching_earned_run_avg_10RA'] - x['home_pitching_earned_run_avg_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算主場投手ERA趨勢指標")

    try:
        # 客場投手ERA趨勢指標 (相較於所有年度平均)
        df['away_pitching_ERA_trend_all'] = (
            df['away_pitching_earned_run_avg_10RA'] - 
            df['away_pitching_earned_run_avg_mean']
        )
        # 客場投手ERA趨勢指標 (相較於同年度平均)
        df['away_pitching_ERA_trend_season'] = df.groupby('season').apply(
            lambda x: x['away_pitching_earned_run_avg_10RA'] - x['away_pitching_earned_run_avg_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算客場投手ERA趨勢指標")

    # 投手團隊三振率趨勢指標
    try:
        # 主場投手三振率趨勢指標 (相較於所有年度平均)
        df['home_pitching_SO_trend_all'] = (
            df['home_pitching_SO_batters_faced_10RA'] - 
            df['home_pitching_SO_batters_faced_mean']
        )
        # 主場投手三振率趨勢指標 (相較於同年度平均)
        df['home_pitching_SO_trend_season'] = df.groupby('season').apply(
            lambda x: x['home_pitching_SO_batters_faced_10RA'] - x['home_pitching_SO_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算主場投手三振率趨勢指標")

    try:
        # 客場投手三振率趨勢指標 (相較於所有年度平均)
        df['away_pitching_SO_trend_all'] = (
            df['away_pitching_SO_batters_faced_10RA'] - 
            df['away_pitching_SO_batters_faced_mean']
        )
        # 客場投手三振率趨勢指標 (相較於同年度平均)
        df['away_pitching_SO_trend_season'] = df.groupby('season').apply(
            lambda x: x['away_pitching_SO_batters_faced_10RA'] - x['away_pitching_SO_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算客場投手三振率趨勢指標")
    # 投手團隊被安打率趨勢指標
    try:
        # 主場投手被安打率趨勢指標 (相較於所有年度平均)
        df['home_pitching_H_trend_all'] = (
            df['home_pitching_H_batters_faced_10RA'] - 
            df['home_pitching_H_batters_faced_mean']
        )
        # 主場投手被安打率趨勢指標 (相較於同年度平均)
        df['home_pitching_H_trend_season'] = df.groupby('season').apply(
            lambda x: x['home_pitching_H_batters_faced_10RA'] - x['home_pitching_H_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算主場投手被安打率趨勢指標")

    try:
        # 客場投手被安打率趨勢指標 (相較於所有年度平均)
        df['away_pitching_H_trend_all'] = (
            df['away_pitching_H_batters_faced_10RA'] - 
            df['away_pitching_H_batters_faced_mean']
        )
        # 客場投手被安打率趨勢指標 (相較於同年度平均)
        df['away_pitching_H_trend_season'] = df.groupby('season').apply(
            lambda x: x['away_pitching_H_batters_faced_10RA'] - x['away_pitching_H_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算客場投手被安打率趨勢指標")

    # 投手團隊保送率趨勢指標
    try:
        # 主場投手保送率趨勢指標 (相較於所有年度平均)
        df['home_pitching_BB_trend_all'] = (
            df['home_pitching_BB_batters_faced_10RA'] - 
            df['home_pitching_BB_batters_faced_mean']
        )
        # 主場投手保送率趨勢指標 (相較於同年度平均)
        df['home_pitching_BB_trend_season'] = df.groupby('season').apply(
            lambda x: x['home_pitching_BB_batters_faced_10RA'] - x['home_pitching_BB_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算主場投手保送率趨勢指標")

    try:
        # 客場投手保送率趨勢指標 (相較於所有年度平均)
        df['away_pitching_BB_trend_all'] = (
            df['away_pitching_BB_batters_faced_10RA'] - 
            df['away_pitching_BB_batters_faced_mean']
        )
        # 客場投手保送率趨勢指標 (相較於同年度平均)
        df['away_pitching_BB_trend_season'] = df.groupby('season').apply(
            lambda x: x['away_pitching_BB_batters_faced_10RA'] - x['away_pitching_BB_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算客場投手保送率趨勢指標")

    # 先發投手個人 ERA 趨勢指標
    try:
        # 主場先發投手個人ERA趨勢指標 (相較於所有年度平均)
        df['home_starter_ERA_trend_all'] = (
            df['home_pitcher_earned_run_avg_10RA'] - 
            df['home_pitcher_earned_run_avg_mean']
        )
        # 主場先發投手個人ERA趨勢指標 (相較於同年度平均)
        df['home_starter_ERA_trend_season'] = df.groupby('season').apply(
            lambda x: x['home_pitcher_earned_run_avg_10RA'] - x['home_pitcher_earned_run_avg_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算主場先發投手ERA趨勢指標")

    try:
        # 客場先發投手個人ERA趨勢指標 (相較於所有年度平均)
        df['away_starter_ERA_trend_all'] = (
            df['away_pitcher_earned_run_avg_10RA'] - 
            df['away_pitcher_earned_run_avg_mean']
        )
        # 客場先發投手個人ERA趨勢指標 (相較於同年度平均)
        df['away_starter_ERA_trend_season'] = df.groupby('season').apply(
            lambda x: x['away_pitcher_earned_run_avg_10RA'] - x['away_pitcher_earned_run_avg_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算客場先發投手ERA趨勢指標")
    # 先發投手三振率趨勢指標
    try:
        # 主場先發投手三振率趨勢指標 (相較於所有年度平均)
        df['home_starter_SO_trend_all'] = (
            df['home_pitcher_SO_batters_faced_10RA'] - 
            df['home_pitcher_SO_batters_faced_mean']
        )
        # 主場先發投手三振率趨勢指標 (相較於同年度平均)
        df['home_starter_SO_trend_season'] = df.groupby('season').apply(
            lambda x: x['home_pitcher_SO_batters_faced_10RA'] - x['home_pitcher_SO_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算主場先發投手三振率趨勢指標")

    try:
        # 客場先發投手三振率趨勢指標 (相較於所有年度平均)
        df['away_starter_SO_trend_all'] = (
            df['away_pitcher_SO_batters_faced_10RA'] - 
            df['away_pitcher_SO_batters_faced_mean']
        )
        # 客場先發投手三振率趨勢指標 (相較於同年度平均)
        df['away_starter_SO_trend_season'] = df.groupby('season').apply(
            lambda x: x['away_pitcher_SO_batters_faced_10RA'] - x['away_pitcher_SO_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算客場先發投手三振率趨勢指標")

    # 先發投手被安打率趨勢指標
    try:
        # 主場先發投手被安打率趨勢指標 (相較於所有年度平均)
        df['home_starter_H_trend_all'] = (
            df['home_pitcher_H_batters_faced_10RA'] - 
            df['home_pitcher_H_batters_faced_mean']
        )
        # 主場先發投手被安打率趨勢指標 (相較於同年度平均)
        df['home_starter_H_trend_season'] = df.groupby('season').apply(
            lambda x: x['home_pitcher_H_batters_faced_10RA'] - x['home_pitcher_H_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算主場先發投手被安打率趨勢指標")

    try:
        # 客場先發投手被安打率趨勢指標 (相較於所有年度平均)
        df['away_starter_H_trend_all'] = (
            df['away_pitcher_H_batters_faced_10RA'] - 
            df['away_pitcher_H_batters_faced_mean']
        )
        # 客場先發投手被安打率趨勢指標 (相較於同年度平均)
        df['away_starter_H_trend_season'] = df.groupby('season').apply(
            lambda x: x['away_pitcher_H_batters_faced_10RA'] - x['away_pitcher_H_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算客場先發投手被安打率趨勢指標")

    # 先發投手保送率趨勢指標
    try:
        # 主場先發投手保送率趨勢指標 (相較於所有年度平均)
        df['home_starter_BB_trend_all'] = (
            df['home_pitcher_BB_batters_faced_10RA'] - 
            df['home_pitcher_BB_batters_faced_mean']
        )
        # 主場先發投手保送率趨勢指標 (相較於同年度平均)
        df['home_starter_BB_trend_season'] = df.groupby('season').apply(
            lambda x: x['home_pitcher_BB_batters_faced_10RA'] - x['home_pitcher_BB_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算主場先發投手保送率趨勢指標")

    try:
        # 客場先發投手保送率趨勢指標 (相較於所有年度平均)
        df['away_starter_BB_trend_all'] = (
            df['away_pitcher_BB_batters_faced_10RA'] - 
            df['away_pitcher_BB_batters_faced_mean']
        )
        # 客場先發投手保送率趨勢指標 (相較於同年度平均)
        df['away_starter_BB_trend_season'] = df.groupby('season').apply(
            lambda x: x['away_pitcher_BB_batters_faced_10RA'] - x['away_pitcher_BB_batters_faced_mean']
        ).values
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算客場先發投手保送率趨勢指標")

    # ======= 優勢指標 ==============================================================================================
    try:
        # ERA優勢
        df['pitching_ERA_advantage'] = (
            df['away_pitching_earned_run_avg_mean'] - 
            df['home_pitching_earned_run_avg_mean']
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算ERA優勢")

    try:
        # 三振率優勢
        df['pitching_SO_advantage'] = (
            df['home_pitching_SO_batters_faced_mean'] - 
            df['away_pitching_SO_batters_faced_mean']
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算三振率優勢")
    try:
        # 被安打率優勢
        df['pitching_H_advantage'] = (
            df['away_pitching_H_batters_faced_mean'] - 
            df['home_pitching_H_batters_faced_mean']
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算被安打率優勢")

    try:
        # 保送率優勢
        df['pitching_BB_advantage'] = (
            df['away_pitching_BB_batters_faced_mean'] - 
            df['home_pitching_BB_batters_faced_mean']
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算保送率優勢")

    try:
        # 先發投手ERA優勢
        df['starter_ERA_advantage'] = (
            df['away_pitcher_earned_run_avg_mean'] - 
            df['home_pitcher_earned_run_avg_mean']
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算先發投手ERA優勢")

    try:
        # 先發投手三振率優勢
        df['starter_SO_advantage'] = (
            df['home_pitcher_SO_batters_faced_mean'] - 
            df['away_pitcher_SO_batters_faced_mean']
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算先發投手三振率優勢")

    try:
        # 先發投手被安打率優勢
        df['starter_H_advantage'] = (
            df['away_pitcher_H_batters_faced_mean'] - 
            df['home_pitcher_H_batters_faced_mean']
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算先發投手被安打率優勢")

    try:
        # 先發投手保送率優勢
        df['starter_BB_advantage'] = (
            df['away_pitcher_BB_batters_faced_mean'] - 
            df['home_pitcher_BB_batters_faced_mean']
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算先發投手保送率優勢")

    # ======= 穩定性指標 ==============================================================================================
    # 設定極小值避免除以零
    epsilon = 1e-10

    # 投手相關指標的穩定性
    try:
        # ERA穩定性
        df['home_pitching_ERA_stability'] = (
            df['home_pitching_earned_run_avg_std'] / 
            (abs(df['home_pitching_earned_run_avg_mean']) + epsilon)
        )
        df['away_pitching_ERA_stability'] = (
            df['away_pitching_earned_run_avg_std'] / 
            (abs(df['away_pitching_earned_run_avg_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算ERA穩定性")

    try:
        # 三振率穩定性
        df['home_pitching_SO_stability'] = (
            df['home_pitching_SO_batters_faced_std'] / 
            (abs(df['home_pitching_SO_batters_faced_mean']) + epsilon)
        )
        df['away_pitching_SO_stability'] = (
            df['away_pitching_SO_batters_faced_std'] / 
            (abs(df['away_pitching_SO_batters_faced_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算三振率穩定性")

    try:
        # 被安打率穩定性
        df['home_pitching_H_stability'] = (
            df['home_pitching_H_batters_faced_std'] / 
            (abs(df['home_pitching_H_batters_faced_mean']) + epsilon)
        )
        df['away_pitching_H_stability'] = (
            df['away_pitching_H_batters_faced_std'] / 
            (abs(df['away_pitching_H_batters_faced_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算被安打率穩定性")

    try:
        # 保送率穩定性
        df['home_pitching_BB_stability'] = (
            df['home_pitching_BB_batters_faced_std'] / 
            (abs(df['home_pitching_BB_batters_faced_mean']) + epsilon)
        )
        df['away_pitching_BB_stability'] = (
            df['away_pitching_BB_batters_faced_std'] / 
            (abs(df['away_pitching_BB_batters_faced_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算保送率穩定性")

    # 先發投手個人指標的穩定性
    try:
        # 先發投手ERA穩定性
        df['home_starter_ERA_stability'] = (
            df['home_pitcher_earned_run_avg_std'] / 
            (abs(df['home_pitcher_earned_run_avg_mean']) + epsilon)
        )
        df['away_starter_ERA_stability'] = (
            df['away_pitcher_earned_run_avg_std'] / 
            (abs(df['away_pitcher_earned_run_avg_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算先發投手ERA穩定性")

    try:
        # 先發投手三振率穩定性
        df['home_starter_SO_stability'] = (
            df['home_pitcher_SO_batters_faced_std'] / 
            (abs(df['home_pitcher_SO_batters_faced_mean']) + epsilon)
        )
        df['away_starter_SO_stability'] = (
            df['away_pitcher_SO_batters_faced_std'] / 
            (abs(df['away_pitcher_SO_batters_faced_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算先發投手三振率穩定性")

    try:
        # 先發投手被安打率穩定性
        df['home_starter_H_stability'] = (
            df['home_pitcher_H_batters_faced_std'] / 
            (abs(df['home_pitcher_H_batters_faced_mean']) + epsilon)
        )
        df['away_starter_H_stability'] = (
            df['away_pitcher_H_batters_faced_std'] / 
            (abs(df['away_pitcher_H_batters_faced_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算先發投手被安打率穩定性")

    try:
        # 先發投手保送率穩定性
        df['home_starter_BB_stability'] = (
            df['home_pitcher_BB_batters_faced_std'] / 
            (abs(df['home_pitcher_BB_batters_faced_mean']) + epsilon)
        )
        df['away_starter_BB_stability'] = (
            df['away_pitcher_BB_batters_faced_std'] / 
            (abs(df['away_pitcher_BB_batters_faced_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算先發投手保送率穩定性")

    # 打擊相關指標的穩定性
    try:
        # 打擊率穩定性
        df['home_batting_avg_stability'] = (
            df['home_batting_batting_avg_std'] / 
            (abs(df['home_batting_batting_avg_mean']) + epsilon)
        )
        df['away_batting_avg_stability'] = (
            df['away_batting_batting_avg_std'] / 
            (abs(df['away_batting_batting_avg_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算打擊率穩定性")

    try:
        # 上壘率穩定性
        df['home_onbase_perc_stability'] = (
            df['home_batting_onbase_perc_std'] / 
            (abs(df['home_batting_onbase_perc_mean']) + epsilon)
        )
        df['away_onbase_perc_stability'] = (
            df['away_batting_onbase_perc_std'] / 
            (abs(df['away_batting_onbase_perc_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算上壘率穩定性")

    try:
        # OPS穩定性
        df['home_ops_stability'] = (
            df['home_batting_onbase_plus_slugging_std'] / 
            (abs(df['home_batting_onbase_plus_slugging_mean']) + epsilon)
        )
        df['away_ops_stability'] = (
            df['away_batting_onbase_plus_slugging_std'] / 
            (abs(df['away_batting_onbase_plus_slugging_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算OPS穩定性")

    try:
        # 打擊槓桿指數穩定性
        df['home_batting_leverage_stability'] = (
            df['home_batting_leverage_index_avg_std'] / 
            (abs(df['home_batting_leverage_index_avg_mean']) + epsilon)
        )
        df['away_batting_leverage_stability'] = (
            df['away_batting_leverage_index_avg_std'] / 
            (abs(df['away_batting_leverage_index_avg_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算打擊槓桿指數穩定性")

    try:
        # 打點穩定性
        df['home_RBI_stability'] = (
            df['home_batting_RBI_std'] / 
            (abs(df['home_batting_RBI_mean']) + epsilon)
        )
        df['away_RBI_stability'] = (
            df['away_batting_RBI_std'] / 
            (abs(df['away_batting_RBI_mean']) + epsilon)
        )
    except KeyError as e:
        print(f"缺少必要欄位: {str(e)}, 跳過計算打點穩定性")


    return df