import pandas as pd
import io
import matplotlib.pyplot as plt

# Data
raw_data = """Experiment Name	No.	MPro/nirmat molar ratio	Area free ligand (-74.5 to -74.2 ppm)	SINO free ligand	Area bound ligand (-71 to -73 ppm)	SINO bound ligand		normalized area free ligand	+/- err	normalized area bound ligand	+/- err
Nirma-19F-buffer-20221121	19	0	3080800	NA	0	0	3080800	1.00	0.003	0.00	0.00
Nirma-19F_Mpro_0.25_20221111	19	0.25	2468592	285	583891	9.9	3052483	0.81	0.003	0.19	0.02
Nirma-19F_Mpro_0.5_20221111	19	0.5	1547999	183	900080	13.6	2448079	0.63	0.003	0.37	0.03
Nirma-19F_Mpro_1:1_20221111	19	1	178562	24.3	2631661	36.3	2810223	0.06	0.003	0.94	0.03
Nirma-19F_Mpro_1:1_20221111	119*	1	431519	36.3	2689572	35.5	3121091	0.14	0.004	0.86	0.02"""

df = pd.read_csv(io.StringIO(raw_data), sep='\t')
new_cols = [
    'Experiment_Name', 'No', 'Mpro_Nirma_Ratio', 'Area_Free', 'SINO_Free', 
    'Area_Bound', 'SINO_Bound', 'Total_Area', 'Norm_Area_Free', 'Err_Free', 
    'Norm_Area_Bound', 'Err_Bound'
]
df.columns = new_cols

# Cleaning
def clean_ratio(val):
    if isinstance(val, str):
        if ':' in val: return float(val.split(':')[0]) / float(val.split(':')[1])
        return float(val.replace('*', ''))
    return float(val)

df['Mpro_Nirma_Ratio'] = df['Mpro_Nirma_Ratio'].apply(clean_ratio)
df['No_Clean'] = df['No'].astype(str).str.replace('*', '', regex=False).astype(int)

# Export CSV
df_to_save = df.drop(columns=['No_Clean'])
df_to_save.to_csv('mpro_nirma_titration_data.csv', index=False)

# Create README
readme_content = """# Data Archival: MPro/Nirmatrelvir NMR Binding Assay

## Dataset Description
This dataset captures the binding of Nirmatrelvir to SARS-CoV-2 MPro (Main Protease) using 19F NMR spectroscopy.

## Column Definitions
- Experiment_Name: Original NMR experiment identifier.
- No: Sample number or sequence ID. (Note: '119' denotes measurements taken 10 days later).
- Mpro_Nirma_Ratio: Molar ratio of MPro to Nirmatrelvir.
- Area_Free: Integration area of the free ligand peak (-74.5 to -74.2 ppm).
- SINO_Free: Signal-to-noise ratio for the free ligand peak.
- Area_Bound: Integration area of the bound ligand peak (-71 to -73 ppm).
- SINO_Bound: Signal-to-noise ratio for the bound ligand peak.
- Total_Area: Sum of free and bound ligand peak areas.
- Norm_Area_Free: Fraction of total area corresponding to free ligand.
- Err_Free: Calculated error for free ligand fraction.
- Norm_Area_Bound: Fraction of total area corresponding to bound ligand.
- Err_Bound: Calculated error for bound ligand fraction.

## Metadata
- Instrument: NMR 19F
- Chemical Shift Regions: 
    - Free Ligand: -74.5 to -74.2 ppm
    - Bound Ligand: -71 to -73 ppm
- Date of experiments: Nov 2022
"""
with open('README.txt', 'w') as f:
    f.write(readme_content)

# Plotting
plt.figure(figsize=(10, 7))

# Separate the 10-day-later point
df_initial = df[df['No_Clean'] == 19].sort_values('Mpro_Nirma_Ratio')
df_10days = df[df['No_Clean'] == 119].sort_values('Mpro_Nirma_Ratio')

# Free Ligand: Blue circles
# Initial (filled)
plt.errorbar(df_initial['Mpro_Nirma_Ratio'], df_initial['Norm_Area_Free'], yerr=df_initial['Err_Free'], 
             fmt='o', color='blue', label='Free Ligand (Day 0)', markersize=8, capsize=5)
plt.plot(df_initial['Mpro_Nirma_Ratio'], df_initial['Norm_Area_Free'], color='blue', linestyle='-')
# 10 days (open)
plt.errorbar(df_10days['Mpro_Nirma_Ratio'], df_10days['Norm_Area_Free'], yerr=df_10days['Err_Free'], 
             fmt='o', color='blue', mfc='none', label='Free Ligand (Day 10)', markersize=8, capsize=5)

# Bound Ligand: Orange squares
# Initial (filled)
plt.errorbar(df_initial['Mpro_Nirma_Ratio'], df_initial['Norm_Area_Bound'], yerr=df_initial['Err_Bound'], 
             fmt='s', color='orange', label='Bound Ligand (Day 0)', markersize=8, capsize=5)
plt.plot(df_initial['Mpro_Nirma_Ratio'], df_initial['Norm_Area_Bound'], color='orange', linestyle='-')
# 10 days (open)
plt.errorbar(df_10days['Mpro_Nirma_Ratio'], df_10days['Norm_Area_Bound'], yerr=df_10days['Err_Bound'], 
             fmt='s', color='orange', mfc='none', label='Bound Ligand (Day 10)', markersize=8, capsize=5)

plt.xlabel('MPro/Nirmatrelvir Molar Ratio', fontsize=12)
plt.ylabel('Normalized Area (Mole Fraction)', fontsize=12)
plt.title('MPro/Nirmatrelvir Binding Study (19F NMR)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('mpro_nirma_titration_plot.png')

print("Files created: mpro_nirma_titration.csv, README.txt, mpro_nirma_titration_plot.png")
