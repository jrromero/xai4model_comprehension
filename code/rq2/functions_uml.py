import pandas as pd
import re
import dalex as dx

def evaluate_value(value):
    if value >= 0:
        return "Positive"
    else:
        return "Negative"
    

def preprocess_dfs(breakdown_df, shap_df, lime_df, n, m):

    columns_multi = pd.MultiIndex.from_tuples([
    ('Breakdown', 'Ranking'), ('Breakdown', 'Sign'),
    ('Shap', 'Ranking'), ('Shap', 'Sign'),
    ('Lime', 'Ranking'), ('Lime', 'Sign')])


    pattern = re.compile(r'(?:<=|<|>=|>)\s*(\w+)\s*(?:<=|<|>=|>)')

    breakdown_df = breakdown_df.loc[:, ['variable_name', 'contribution', 'sign']]
    breakdown_df = breakdown_df.drop(index=[0, n+1])
    breakdown_df['sign'] = breakdown_df['sign'].replace({1.0: 'Positive', 0.0: 'Null', -1.0: 'Negative'})
    breakdown_df = breakdown_df.sort_values(by='contribution', key=lambda x: abs(x), ascending=False)

    shap_df = shap_df.loc[:, ['variable_name', 'contribution', 'sign']]
    shap_df = shap_df.tail(n)
    shap_df['sign'] = shap_df['sign'].replace({1.0: 'Positive', 0.0: 'Null', -1.0: 'Negative'})
    shap_df = shap_df.sort_values(by='contribution', key=lambda x: abs(x), ascending=False)

    lime_list = []
    for feature in lime_df['variable']:
      if pattern.findall(feature):
        match = pattern.search(feature)
        if match:
          lime_list.append(match.group(1))
      else:
        splits = feature.split(" ")
        lime_list.append(splits[0])

    lime_df["Feature"] = lime_list
    lime_df["Sign"] = lime_df["effect"].apply(evaluate_value)
    lime_df = lime_df.sort_values(by='effect', key=lambda x: abs(x), ascending=False)
    lime_df = lime_df.drop(columns=['variable'])

    breakdown_df['Ranking'] = breakdown_df['contribution'].abs().rank(ascending=False).astype(int)
    breakdown_df.rename(columns={'sign': 'Sign', 'variable_name':'Feature'}, inplace=True)
    breakdown_df = breakdown_df[['Feature', 'Ranking', 'contribution', 'Sign']]

    shap_df['Ranking'] = shap_df['contribution'].abs().rank(ascending=False).astype(int)
    shap_df.rename(columns={'sign': 'Sign', 'variable_name':'Feature'}, inplace=True)
    shap_df = shap_df[['Feature', 'Ranking', 'contribution', 'Sign']]

    lime_df['Ranking'] = lime_df['effect'].abs().rank(ascending=False).astype(int)
    lime_df = lime_df.head(m)
    lime_df = lime_df[['Feature', 'Ranking', 'effect', 'Sign']]

    breakdown_df = breakdown_df.drop(columns=['contribution'])
    shap_df = shap_df.drop(columns=['contribution'])
    lime_df = lime_df.drop(columns=['effect'])

    breakdown_df = breakdown_df.head(m)
    breakdown_df = breakdown_df.reset_index(drop=True)

    shap_df= shap_df.head(m)
    shap_df = shap_df.reset_index(drop=True)

    lime_df = lime_df.reset_index(drop=True)

    breakdown_features = list(breakdown_df['Feature'])
    shap_features = list(shap_df['Feature'])
    lime_features = list(lime_df['Feature'])
    all_features = list(set(breakdown_features + shap_features + lime_features))
    all_features = all_features[::-1]

    df_final = pd.DataFrame(index=range(len(all_features)), columns=columns_multi)
    df_final['Feature'] = list(all_features)
    df_final = df_final[['Feature', 'Breakdown', 'Shap', 'Lime']]

    #print(breakdown_features)
    #print(shap_features)
    #print(lime_features)
    print(all_features)

    for feature in df_final['Feature']:

        breakdown_row = breakdown_df[breakdown_df['Feature'] == feature]
        if not breakdown_row.empty:

            ranking_breakdown = breakdown_row.iloc[0]['Ranking']
            sign_breakdown = breakdown_row.iloc[0]['Sign']
            df_final.loc[df_final['Feature'] == feature, ('Breakdown', 'Ranking')] = ranking_breakdown
            df_final.loc[df_final['Feature'] == feature, ('Breakdown', 'Sign')] = sign_breakdown
        else:

            df_final.loc[df_final['Feature'] == feature, ('Breakdown', 'Ranking')] = '-'
            df_final.loc[df_final['Feature'] == feature, ('Breakdown', 'Sign')] = '-'


        shap_row = shap_df[shap_df['Feature'] == feature]
        if not shap_row.empty:
            ranking_shap = shap_row.iloc[0]['Ranking']
            sign_shap = shap_row.iloc[0]['Sign']
            df_final.loc[df_final['Feature'] == feature, ('Shap', 'Ranking')] = ranking_shap
            df_final.loc[df_final['Feature'] == feature, ('Shap', 'Sign')] = sign_shap
        else:
            df_final.loc[df_final['Feature'] == feature, ('Shap', 'Ranking')] = '-'
            df_final.loc[df_final['Feature'] == feature, ('Shap', 'Sign')] = '-'


        lime_row = lime_df[lime_df['Feature'] == feature]
        if not lime_row.empty:
            ranking_lime = lime_row.iloc[0]['Ranking']
            sign_lime = lime_row.iloc[0]['Sign']
            df_final.loc[df_final['Feature'] == feature, ('Lime', 'Ranking')] = ranking_lime
            df_final.loc[df_final['Feature'] == feature, ('Lime', 'Sign')] = sign_lime
        else:
            df_final.loc[df_final['Feature'] == feature, ('Lime', 'Ranking')] = '-'
            df_final.loc[df_final['Feature'] == feature, ('Lime', 'Sign')] = '-'



    ranking_breakdown = df_final[('Breakdown', 'Ranking')]
    values_valid = ranking_breakdown[ranking_breakdown != '-']
    cases_invalid = ranking_breakdown[ranking_breakdown == '-'].index
    cases_valid = []
    cases_invalid_restantes = list(cases_invalid)
    cases_valid.extend(values_valid.sort_values().index.tolist())

    for cas in cases_invalid:
        value_shap = df_final.at[cas, ('Shap', 'Ranking')]
        if value_shap != '-':
            cases_valid.extend([cas])
            cases_invalid_restantes.remove(cas)

    for cas in cases_invalid_restantes:
        value_lime = df_final.at[cas, ('Lime', 'Ranking')]
        if value_lime != '-':
            cases_valid.extend([cas])

    df_final = df_final.loc[cases_valid]
    df_final.set_index('Feature', inplace=True)
    return breakdown_df, shap_df, lime_df, df_final


def top5_features(df, case):
    set_expl = {}

    for tech in ['Breakdown', 'Shap', 'Lime']:
        inst_top5 = set(df[df[(tech, 'Ranking')].isin([1, 2, 3, 4, 5])].index)
        set_expl[tech] = inst_top5

    result_case = {
        f'TOP5 {tech}': ', '.join(set_expl[tech]) for tech in ['Breakdown', 'Shap', 'Lime']
    }

    for tech1, inst_top5_1 in set_expl.items():
        for tech2, inst_top5_2 in set_expl.items():
            if tech1 < tech2:
                col_name = f'{tech1}-{tech2}'
                result_case[col_name] = len(inst_top5_1.intersection(inst_top5_2)) / 5

    df_top5_case = pd.DataFrame(result_case, index=[case])

    return df_top5_case



def top5_rank(df, case):
    set_expl = {}

    for tech in ['Breakdown', 'Shap', 'Lime']:
        inst_top5 = df[df[(tech, 'Ranking')].isin([1, 2, 3, 4, 5])]
        inst_top5_str = ', '.join([f'{ranking}:{inst}' for ranking, inst in zip(inst_top5[(tech, 'Ranking')], inst_top5.index)])

        set_expl[tech] = inst_top5_str

    result_case = {
        f'TOP5 {tech}': set_expl[tech] for tech in ['Breakdown', 'Shap', 'Lime']
    }

    for tech1, inst_top5_1 in set_expl.items():
        for tech2, inst_top5_2 in set_expl.items():
            if tech1 < tech2:
                col_name = f'{tech1}-{tech2}'

                coincidence = [inst for inst in inst_top5_1.split(', ') if inst in inst_top5_2.split(', ') and inst.split(':')[1] == inst.split(':')[1]]
                num_coincidence = len(coincidence)

                result_case[col_name] = num_coincidence / 5

    df_top5_rank = pd.DataFrame(result_case, index=[case])

    return df_top5_rank


def top5_sign(df, case):
    set_expl = {}

    for tech in ['Breakdown', 'Shap', 'Lime']:
        inst_top5 = df[df[(tech, 'Ranking')].isin([1, 2, 3, 4, 5])]
        inst_top5_str = ', '.join([f'{inst}:{sign}' for inst, sign in zip(inst_top5.index, inst_top5[(tech, 'Sign')])])

        set_expl[tech] = inst_top5_str

    result_case = {
        f'TOP5 {tech}': set_expl[tech] for tech in ['Breakdown', 'Shap', 'Lime']
    }

    for tech1, inst_top5_1 in set_expl.items():
        for tech2, inst_top5_2 in set_expl.items():
            if tech1 < tech2:
                col_name = f'{tech1}-{tech2}'

                coincidence = [inst for inst in inst_top5_1.split(', ') if inst in inst_top5_2.split(', ')]
                num_coincidence = len(coincidence)

                result_case[col_name] = num_coincidence / 5

    df_top5_sign = pd.DataFrame(result_case, index=[case])

    return df_top5_sign



def top5_rank_sign(df, case):
    set_expl = {}

    for tech in ['Breakdown', 'Shap', 'Lime']:
        inst_top5 = df[df[(tech, 'Ranking')].isin([1, 2, 3, 4, 5])]
        inst_top5_str = ', '.join([f'{ranking}:{inst}:{sign}' for ranking, inst, sign in zip(inst_top5[(tech, 'Ranking')], inst_top5.index, inst_top5[(tech, 'Sign')])])
        set_expl[tech] = inst_top5_str

    result_case = {
        f'TOP5 {tech}': set_expl[tech] for tech in ['Breakdown', 'Shap', 'Lime']
    }

    for tech1, inst_top5_1 in set_expl.items():
        for tech2, inst_top5_2 in set_expl.items():
            if tech1 < tech2:
                col_name = f'{tech1}-{tech2}'

                coincidence = [inst for inst in inst_top5_1.split(', ') if inst in inst_top5_2.split(', ') and inst.split(':')[1] == inst.split(':')[1] and inst.split(':')[2] == inst.split(':')[2]]
                num_coincidence = len(coincidence)
                result_case[col_name] = num_coincidence / 5

    df_top5_rank_sign = pd.DataFrame(result_case, index=[case])

    return df_top5_rank_sign


def calculate_metrics(df_list):
    df_final = pd.concat(df_list)

    mean_bd_shap = df_final['Breakdown-Shap'].mean()
    mean_bd_lime = df_final['Breakdown-Lime'].mean()
    mean_lime_shap = df_final['Lime-Shap'].mean()

    return df_final, mean_bd_shap, mean_bd_lime, mean_lime_shap




def create_metrics_dfs(metrics_dict, metric_type):
    dfs = {'Model_comparison': {}, 'Class_comparison': {}}
    columns_model = ['Class', 'Mean Breakdown-Shap', 'Mean Breakdown-Lime', 'Mean Lime-Shap']
    columns_class = ['Model', 'Mean Breakdown-Shap', 'Mean Breakdown-Lime', 'Mean Lime-Shap']
    metric_names = [f'Mean_{metric_type}_BD_Shap', f'Mean_{metric_type}_BD_Lime', f'Mean_{metric_type}_Lime_Shap']

    for model_name, indexes_dict in metrics_dict.items():
        for index_name, metrics in indexes_dict.items():
            if model_name not in dfs['Class_comparison']:
                dfs['Class_comparison'][model_name] = pd.DataFrame(columns=columns_model)

            get_metrics = [metrics.get(metric_name) for metric_name in metric_names]
            class_data = [index_name] + get_metrics
            #dfs['Class_comparison'][model_name] = dfs['Class_comparison'][model_name].append(pd.Series(class_data, index=columns_model), ignore_index=True)
            dfs['Class_comparison'][model_name] = pd.concat([dfs['Class_comparison'][model_name], pd.Series(class_data, index=columns_model).to_frame().T], ignore_index=True)


            if index_name not in dfs['Model_comparison']:
                dfs['Model_comparison'][index_name] = pd.DataFrame(columns=columns_class)

            get_metrics = [metrics.get(metric_name) for metric_name in metric_names]
            model_data = [model_name] + get_metrics
            #dfs['Model_comparison'][index_name] = dfs['Model_comparison'][index_name].append(pd.Series(model_data, index=columns_class), ignore_index=True)
            dfs['Model_comparison'][index_name] = pd.concat([dfs['Model_comparison'][index_name], pd.Series(model_data, index=columns_class).to_frame().T], ignore_index=True)

    return dfs




def calculate_metrics_for_indices(models_dict, indexes_dict, X_test, X_train, y_train, num_features, top_num_features):
    results_dict = {}
    metrics_dict = {}

    for model_name, model in models_dict.items():
        results_dict[model_name] = {}
        metrics_dict[model_name] = {}

        exp = dx.Explainer(model, X_train, y_train)

        for index_name, indices in indexes_dict.items():
          results_dict[model_name][index_name] = {}
          metrics_dict[model_name][index_name] = {}
          model_results = results_dict[model_name][index_name]
          df_list_top = []
          df_list_rank = []
          df_list_sign = []
          df_list_rank_sign = []

          for i in indices:
            instance = X_test.loc[i]

            breakdown = exp.predict_parts(instance, type="break_down", random_state=42)
            shap = exp.predict_parts(instance, type="shap", random_state=42)
            lime = exp.predict_surrogate(instance, random_state=42)
            breakdown_df = breakdown.result
            shap_df = shap.result
            lime_df = lime.result

            breakdown_df, shap_df, lime_df, df_final = preprocess_dfs(breakdown_df, shap_df, lime_df, num_features, top_num_features)

            model_results[f"breakdown_df_{i}"] = breakdown_df
            model_results[f"shap_df_{i}"] = shap_df
            model_results[f"lime_df_{i}"] = lime_df
            model_results[f"df_final_{i}"] = df_final

            metrics_dict[model_name][index_name][f"df_top_metric_{i}"] = top5_features(df_final, i)
            metrics_dict[model_name][index_name][f'df_rank_metric_{i}'] = top5_rank(df_final, i)
            metrics_dict[model_name][index_name][f'df_sign_metric_{i}'] = top5_sign(df_final, i)
            metrics_dict[model_name][index_name][f'df_rank_sign_metric_{i}'] = top5_rank_sign(df_final, i)

            #model_results[f"df_top_metric_{i}"] = top5_features(df_final, i)
            #model_results[f'df_rank_metric_{i}'] = top5_rank(df_final, i)
            #model_results[f'df_sign_metric_{i}'] = top5_sign(df_final, i)
            #model_results[f'df_rank_sign_metric_{i}'] = top5_rank_sign(df_final, i)

            df_list_top.append(metrics_dict[model_name][index_name][f'df_top_metric_{i}'])
            df_list_rank.append(metrics_dict[model_name][index_name][f'df_rank_metric_{i}'])
            df_list_sign.append(metrics_dict[model_name][index_name][f'df_sign_metric_{i}'])
            df_list_rank_sign.append(metrics_dict[model_name][index_name][f'df_rank_sign_metric_{i}'])

          df_top_metric_final, mean_top_bd_shap, mean_top_bd_lime, mean_top_lime_shap = calculate_metrics(df_list_top)
          df_rank_metric_final, mean_rank_bd_shap, mean_rank_bd_lime, mean_rank_lime_shap = calculate_metrics(df_list_rank)
          df_sign_metric_final, mean_sign_bd_shap, mean_sign_bd_lime, mean_sign_lime_shap = calculate_metrics(df_list_sign)
          df_rank_sign_metric_final, mean_rank_sign_bd_shap, mean_rank_sign_bd_lime, mean_rank_sign_lime_shap = calculate_metrics(df_list_rank_sign)

          metrics_dict[model_name][index_name][f'Mean_top_BD_Shap'] = mean_top_bd_shap
          metrics_dict[model_name][index_name][f'Mean_top_BD_Lime'] = mean_top_bd_lime
          metrics_dict[model_name][index_name][f'Mean_top_Lime_Shap'] = mean_top_lime_shap
          metrics_dict[model_name][index_name][f'df_top_metric_final'] = df_top_metric_final

          metrics_dict[model_name][index_name][f'Mean_rank_BD_Shap'] = mean_rank_bd_shap
          metrics_dict[model_name][index_name][f'Mean_rank_BD_Lime'] = mean_rank_bd_lime
          metrics_dict[model_name][index_name][f'Mean_rank_Lime_Shap'] = mean_rank_lime_shap
          metrics_dict[model_name][index_name][f'df_rank_metric_final'] = df_rank_metric_final

          metrics_dict[model_name][index_name][f'Mean_sign_BD_Shap'] = mean_sign_bd_shap
          metrics_dict[model_name][index_name][f'Mean_sign_BD_Lime'] = mean_sign_bd_lime
          metrics_dict[model_name][index_name][f'Mean_sign_Lime_Shap'] = mean_sign_lime_shap
          metrics_dict[model_name][index_name][f'df_sign_metric_final'] = df_sign_metric_final

          metrics_dict[model_name][index_name][f'Mean_rank_sign_BD_Shap'] = mean_rank_sign_bd_shap
          metrics_dict[model_name][index_name][f'Mean_rank_sign_BD_Lime'] = mean_rank_sign_bd_lime
          metrics_dict[model_name][index_name][f'Mean_rank_sign_Lime_Shap'] = mean_rank_sign_lime_shap
          metrics_dict[model_name][index_name][f'df_rank_sign_metric_final'] = df_rank_sign_metric_final

    top_metric_df = create_metrics_dfs(metrics_dict, 'top')
    rank_metric_df = create_metrics_dfs(metrics_dict, 'rank')
    sign_metric_df = create_metrics_dfs(metrics_dict, 'sign')
    rank_sign_metric_df = create_metrics_dfs(metrics_dict, 'rank_sign')

    return metrics_dict, results_dict, top_metric_df, rank_metric_df, sign_metric_df, rank_sign_metric_df