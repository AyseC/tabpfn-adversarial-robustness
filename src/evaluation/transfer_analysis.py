"""Transfer Attack Analysis - Parse and display transfer rates across all datasets"""
import json

datasets = ['wine','iris','diabetes','heart','breast_cancer']

print(f"{'Dataset':<15} {'GBDT->TabPFN':>15} {'TabPFN->GBDT':>15}")
print('-'*50)

for ds in datasets:
    d = json.load(open(f'results/transfer_attack_{ds}.json'))
    keys = list(d.keys())
    
    gbdt_to_tabpfn = []
    tabpfn_to_gbdt = []
    
    if any('→' in k for k in keys):
        for key, val in d.items():
            if isinstance(val, dict) and 'transfer_rate' in val:
                parts = key.split(' → ')
                src, tgt = parts[0], parts[1]
                rate = val['transfer_rate']
                if 'TabPFN' in src and tgt in ['XGBoost','LightGBM']:
                    tabpfn_to_gbdt.append(rate)
                elif src in ['XGBoost','LightGBM'] and 'TabPFN' in tgt:
                    gbdt_to_tabpfn.append(rate)
    else:
        for src in ['XGBoost','LightGBM']:
            if src in d and 'TabPFN' in d[src]:
                gbdt_to_tabpfn.append(d[src]['TabPFN']['transfer_rate'] * 100)
        if 'TabPFN' in d:
            for tgt in ['XGBoost','LightGBM']:
                if tgt in d['TabPFN']:
                    tabpfn_to_gbdt.append(d['TabPFN'][tgt]['transfer_rate'] * 100)
    
    g2t = sum(gbdt_to_tabpfn)/len(gbdt_to_tabpfn) if gbdt_to_tabpfn else 0
    t2g = sum(tabpfn_to_gbdt)/len(tabpfn_to_gbdt) if tabpfn_to_gbdt else 0
    print(f'{ds:<15} {g2t:>14.1f}% {t2g:>14.1f}%')
