import numpy as np
import pandas as pd
import re
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from itertools import combinations, product





LIKERT_7 = {'strongly disagree':1,'disagree':2,'somewhat disagree':3,'neither agree nor disagree':4,'neutral':4,'somewhat agree':5,'agree':6,'strongly agree':7}
OWNERSHIP_MAP = {'a rented tool':1,'a borrowed appliance':2,'a friendly gadget':3,'a helpful pet':4,'a personal assistant':5,'a companion':6,'a member of the household':7}
INV_OWNERSHIP_MAP = {v:k.title() for k,v in OWNERSHIP_MAP.items()}

def _to_str(x):
    try: return str(x).strip()
    except Exception: return ''

def map_likert_7(s: pd.Series)->pd.Series:
    def _m(v):
        if pd.isna(v): return np.nan
        vs=_to_str(v).lower()
        m=re.fullmatch(r'\s*([1-7])\s*', vs)
        if m: return float(m.group(1))
        return float(LIKERT_7.get(vs, np.nan))
    return s.apply(_m)

def map_ownership(s: pd.Series)->pd.Series:
    def _m(v):
        if pd.isna(v): return np.nan
        return float(OWNERSHIP_MAP.get(_to_str(v).lower(), np.nan))
    return s.apply(_m)

def map_ai_familiarity_to_num(v):
    if pd.isna(v): return np.nan
    vs=_to_str(v).lower()
    m=re.fullmatch(r'\s*([1-5])\s*', vs)
    if m: return float(m.group(1))
    if 'never used' in vs or ('existence' in vs and 'never' in vs): return 1.0
    if 'occasionally' in vs: return 2.0
    if 'frequently' in vs: return 3.0
    if 'experiment' in vs or 'different tools' in vs: return 4.0
    if 'building' in vs or 'involved in building' in vs: return 5.0
    return np.nan

def standardize_condition(df, col='Condition'):
    if col in df.columns: df[col]=df[col].astype(str).str.strip()
    return df

def add_collapsed_condition(df, col='Condition', out='Condition_Collapsed'):
    if col in df.columns:
        df[out] = np.where(df[col].str.contains('Customizable', case=False, na=False), 'Customizable', 'Not')
    else:
        df[out] = np.nan
    return df

def add_formfactor_collapsed(df, out='FormFactor_Collapsed'):
    source=None
    for c in df.columns:
        if re.search(r'(robot|form\\s*_?factor|nao|turtle)', c, flags=re.I) and df[c].notna().any():
            source=c; break
    if source is None and 'Condition' in df.columns: source='Condition'
    vals=df[source].astype(str) if source else pd.Series([np.nan]*len(df))
    hum=vals.str.contains('NAO', case=False, na=False)
    non=vals.str.contains('Turtle', case=False, na=False)
    df[out]=np.where(hum,'Humanoid', np.where(non,'NonHumanoid', np.nan))



# ===== Load & Clean (execute) =====
df=pd.read_csv("data_demo.csv")
df=standardize_condition(df,'Condition')
df=add_collapsed_condition(df,'Condition','Condition_Collapsed')
df=add_formfactor_collapsed(df,'FormFactor_Collapsed')



SSRA=['SSRA_1','SSRA_2','SSRA_3','SSRA_4','SSRA_5','SSRA_6','SSRA_7','SSRA_8','SSRA_9','SSRA_10','SSRA_11','SSRA_12','SSRA_13','SSRA_14','SSRA_15','SSRA_16','SSRA_17','SSRA_18','SSRA_19','SSRA_20']
SSRA_HA=['SSRA_1','SSRA_2','SSRA_3','SSRA_4','SSRA_5','SSRA_6']
SSRA_SU=['SSRA_7','SSRA_8','SSRA_9','SSRA_10','SSRA_11','SSRA_12']
SSRA_SI=['SSRA_13','SSRA_14','SSRA_15','SSRA_16','SSRA_17']
SSRA_EC=['SSRA_18','SSRA_19','SSRA_20']
SSRA_social = SSRA_SU + SSRA_SI + SSRA_EC  # 14 items total (6+5+3)
traftonPA=['traftonPA_1','traftonPA_2','traftonPA_3','traftonPA_4']
authenticity=['authenticity_1','authenticity_2','authenticity_3','authenticity_4']
CCR_end=['CCR_1','CCR_2','CCR_3','CCR_4','CCR_5','CCR_6','CCR_7','CCR_8']
CCR_end_connection=['CCR_1','CCR_2','CCR_3','CCR_4']
CCR_end_coordination=['CCR_5','CCR_6','CCR_7','CCR_8']
rosas_C=['competence_warmth_1','competence_warmth_2','competence_warmth_3','competence_warmth_4','competence_warmth_5','competence_warmth_6']
rosas_W=['competence_warmth_7','competence_warmth_8','competence_warmth_9','competence_warmth_10','competence_warmth_11','competence_warmth_12']
enjoy=['enjoyment_1','enjoyment_2','enjoyment_3']
party_experience=['party_experience_1','party_experience_2','party_experience_3']
behavioral_intention=['Behavioral Intention_1','Behavioral Intention_2','Behavioral Intention_3']
ownership=['ownership']
attachment3=['attachment_1','attachment_2','attachment_3']
attachment=['attachment_1','attachment_2']

SUBSCALES={"SSRA_HA":SSRA_HA,"SSRA_SU":SSRA_SU,"SSRA_SI":SSRA_SI,"SSRA_EC":SSRA_EC,"SSRA_social": SSRA_social}
SCALES={
 "SSRA_total_mean":SSRA,
 "traftonPA_mean":traftonPA,
 "authenticity_mean":authenticity,
 "CCR_end_mean":CCR_end,
 "CCR_end_connection_mean":CCR_end_connection,
 "CCR_end_coordination_mean":CCR_end_coordination,
 "rosas_C_mean":rosas_C,
 "rosas_W_mean":rosas_W,
 "enjoy_mean":enjoy,
 "party_experience_mean":party_experience,
 "behavioral_intention_mean":behavioral_intention,
 "ownership_mean":ownership,
 "attachment_mean":attachment,
 "attachment3_mean":attachment3,
}

ALL_ITEM_COLS=set([c for group in (list(SCALES.values())+list(SUBSCALES.values())) for c in group])
for col in [c for c in ALL_ITEM_COLS if c in df.columns]:
    if col=='ownership': 
        df[col]=map_ownership(df[col])
    else: 
        df[col]=map_likert_7(df[col])


def simple_effects(dv="XXX"):
    import numpy as np
    from scipy.stats import ttest_ind

    for ff in ["Humanoid","NonHumanoid"]:
        sub = df[(df["FormFactor_Collapsed"]==ff)].dropna(subset=[dv,"Condition_Collapsed"]).copy()
        a = sub.loc[sub["Condition_Collapsed"]=="Not", dv].astype(float).values
        b = sub.loc[sub["Condition_Collapsed"]=="Customizable", dv].astype(float).values

        t,p = ttest_ind(a,b, equal_var=False)
        # Cohen's d (pooled SD)
        s_pooled = np.sqrt(((a.var(ddof=1)*(len(a)-1)) + (b.var(ddof=1)*(len(b)-1))) / (len(a)+len(b)-2)) if (len(a)+len(b)-2)>0 else np.nan
        d = (a.mean()-b.mean())/s_pooled if s_pooled and s_pooled>0 else np.nan
        print(f"[{ff}] Not - Customizable: mean diff = {a.mean()-b.mean():.3f}, t={t:.3f}, p={p:.4f}, d={d:.3f}")

def analyze_interaction_with_emms(df, dv, factor1='FormFactor_Collapsed', factor2='Condition_Collapsed', covariates=None, alpha=0.05):
    
    if covariates is None:
        covariates = []
    
    # clean data
    analysis_cols = [dv, factor1, factor2] + covariates
    clean_df = df[analysis_cols].dropna()
    
    if clean_df.empty:
        print(f"No data remaining for {dv} after removing missing values")
        return None
    
    descriptives = []
    for f1_level in sorted(clean_df[factor1].unique()):
        for f2_level in sorted(clean_df[factor2].unique()):
            group_data = clean_df[(clean_df[factor1] == f1_level) & (clean_df[factor2] == f2_level)][dv]
            
            if len(group_data) > 0:
                group_name = f"{f1_level} × {f2_level}"
                descriptives.append({
                    'Group': group_name,
                    'Mean': group_data.mean(),
                    'SD': group_data.std(ddof=1),
                    'N': len(group_data),
                    'SE': group_data.std(ddof=1) / np.sqrt(len(group_data)) if len(group_data) > 1 else np.nan
                })
    
    descriptives_df = pd.DataFrame(descriptives)
    
    factor1_descriptives = []
    for f1_level in sorted(clean_df[factor1].unique()):
        f1_data = clean_df[clean_df[factor1] == f1_level][dv]
        if len(f1_data) > 0:
            factor1_descriptives.append({
                'Factor': factor1,
                'Level': f1_level,
                'Mean': f1_data.mean(),
                'SD': f1_data.std(ddof=1),
                'N': len(f1_data),
                'SE': f1_data.std(ddof=1) / np.sqrt(len(f1_data)) if len(f1_data) > 1 else np.nan
            })
    
    factor2_descriptives = []
    for f2_level in sorted(clean_df[factor2].unique()):
        f2_data = clean_df[clean_df[factor2] == f2_level][dv]
        if len(f2_data) > 0:
            factor2_descriptives.append({
                'Factor': factor2,
                'Level': f2_level,
                'Mean': f2_data.mean(),
                'SD': f2_data.std(ddof=1),
                'N': len(f2_data),
                'SE': f2_data.std(ddof=1) / np.sqrt(len(f2_data)) if len(f2_data) > 1 else np.nan
            })
    
    main_effects_df = pd.DataFrame(factor1_descriptives + factor2_descriptives)
    
    formula_terms = [f'C({factor1})', f'C({factor2})', f'C({factor1}):C({factor2})']
    
    for cov in covariates:
        if cov in clean_df.columns:
            if clean_df[cov].dtype in ['object', 'category'] or clean_df[cov].nunique() <= 10:
                formula_terms.append(f'C({cov})')
            else:
                formula_terms.append(cov)
    
    formula = f'{dv} ~ ' + ' + '.join(formula_terms)

    try:
        model = smf.ols(formula, data=clean_df).fit()
        anova_table = anova_lm(model, typ=2)
    except Exception as e:
        print(f"Model fitting failed: {e}")
        return None

    if 'Residual' in anova_table.index:
        ss_error = float(anova_table.loc['Residual', 'sum_sq'])
        
        eta_p2_values = []
        
        for idx, row in anova_table.iterrows():
            if idx == 'Residual':
                eta_p2_values.append(np.nan)
            else:
                ss_effect = row['sum_sq']
                eta_p2 = ss_effect / (ss_effect + ss_error)
                eta_p2_values.append(eta_p2)
                
        
        anova_table['eta_p2'] = eta_p2_values
    
    factor_levels = {
        factor1: sorted(clean_df[factor1].unique()),
        factor2: sorted(clean_df[factor2].unique())
    }
    
    combinations_list = list(product(factor_levels[factor1], factor_levels[factor2]))
    grid_df = pd.DataFrame(combinations_list, columns=[factor1, factor2])
    
    covariate_values = {}
    for cov in covariates:
        if cov in clean_df.columns:
            if clean_df[cov].dtype in ['object', 'category'] or clean_df[cov].nunique() <= 10:
                cov_value = clean_df[cov].mode().iloc[0]
                grid_df[cov] = cov_value
                covariate_values[cov] = f"{cov_value} (mode)"
            else:
                cov_value = clean_df[cov].mean()
                grid_df[cov] = cov_value
                covariate_values[cov] = f"{cov_value:.2f} (mean)"

    emms_values = model.predict(grid_df)
    
    try:
        pred_results = model.get_prediction(grid_df)
        emms_se = pred_results.se_mean
    except:
        emms_se = np.sqrt(model.mse_resid / len(clean_df))
        emms_se = np.repeat(emms_se, len(grid_df))
    
    emms_df = grid_df[[factor1, factor2]].copy()
    emms_df['emmean'] = emms_values
    emms_df['se'] = emms_se
    emms_df['group'] = (emms_df[factor1].astype(str) + ' * ' + emms_df[factor2].astype(str))
    
    t_crit = stats.t.ppf(1 - alpha/2, model.df_resid)
    emms_df['lower_cl'] = emms_df['emmean'] - t_crit * emms_df['se']
    emms_df['upper_cl'] = emms_df['emmean'] + t_crit * emms_df['se']
    

    marginal_emms_list = []
    
    for f1_level in factor_levels[factor1]:
        mask = emms_df[factor1] == f1_level
        subset_emms = emms_df[mask]
        
        if len(subset_emms) > 0:
            marginal_mean = subset_emms['emmean'].mean()
            
            marginal_se = np.sqrt((subset_emms['se'] ** 2).mean())
            
            marginal_emms_list.append({
                'Factor': factor1,
                'Level': f1_level,
                'emmean': marginal_mean,
                'se': marginal_se,
                'lower_cl': marginal_mean - t_crit * marginal_se,
                'upper_cl': marginal_mean + t_crit * marginal_se
            })
    

    for f2_level in factor_levels[factor2]:

        mask = emms_df[factor2] == f2_level
        subset_emms = emms_df[mask]
        
        if len(subset_emms) > 0:
            marginal_mean = subset_emms['emmean'].mean()

            marginal_se = np.sqrt((subset_emms['se'] ** 2).mean())
            
            marginal_emms_list.append({
                'Factor': factor2,
                'Level': f2_level,
                'emmean': marginal_mean,
                'se': marginal_se,
                'lower_cl': marginal_mean - t_crit * marginal_se,
                'upper_cl': marginal_mean + t_crit * marginal_se
            })
    
    marginal_emms_df = pd.DataFrame(marginal_emms_list)

    n_groups = len(emms_df)
    pairwise_results = []
    
    for i, j in combinations(range(n_groups), 2):
        group1 = emms_df.iloc[i]['group']
        group2 = emms_df.iloc[j]['group']
        
        mean_diff = emms_df.iloc[i]['emmean'] - emms_df.iloc[j]['emmean']
        se_diff = np.sqrt(emms_df.iloc[i]['se']**2 + emms_df.iloc[j]['se']**2)
        
        if se_diff > 0:
            t_stat = mean_diff / se_diff
            p_raw = 2 * (1 - stats.t.cdf(abs(t_stat), model.df_resid))
            
            q_stat = abs(t_stat) * np.sqrt(2)
            
            try:
                from scipy.stats import studentized_range
                p_tukey = 1 - studentized_range.cdf(q_stat, n_groups, model.df_resid)
                q_crit = studentized_range.ppf(1 - alpha, n_groups, model.df_resid)
            except (ImportError, AttributeError):
                n_comparisons = n_groups * (n_groups - 1) // 2
                p_tukey = min(1.0, p_raw * n_comparisons)
                q_crit = stats.t.ppf(1 - alpha/(2*n_comparisons), model.df_resid) * np.sqrt(2)
            
            margin = (q_crit / np.sqrt(2)) * se_diff
            
            pairwise_results.append({
                'group1': group1,
                'group2': group2,
                'mean_diff': mean_diff,
                'se_diff': se_diff,
                't_stat': t_stat,
                'p_raw': p_raw,
                'p_tukey': p_tukey,
                'lower_cl': mean_diff - margin,
                'upper_cl': mean_diff + margin,
                'significant': p_tukey < alpha
            })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    return {
        'model': model,
        'anova': anova_table,
        'descriptives': descriptives_df,
        'main_effects': main_effects_df,
        'emms': emms_df,
        'marginal_emms': marginal_emms_df,
        'pairwise': pairwise_df,
        'formula': formula,
        'n_obs': len(clean_df),
        'covariate_values': covariate_values
    }







