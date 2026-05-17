"""
run_all.py
Script runner que executa a simulação sintética e gera os gráficos/relatórios
usando synthetic_model.py e plotting_library.py.

Observações:
- Rodar em ambiente com dependências instaladas: numpy, scipy, pandas, matplotlib, seaborn
- ROOT e dados avançados são opcionais: o script pula etapas quando não disponíveis
"""

import os
import warnings

import numpy as np
import pandas as pd

from synthetic_model import simulate_model
import plotting_library as pl

warnings.filterwarnings("ignore")

# --- Parâmetros (copiados do modelo principal) ---
seed = 2026

fixed = {
    "H_max": 0.95,
    "Df_max": 1.0,
    "T_amb": 298.15,
    "x_crit": 1.0,
}

init = {
    "H0": 0.08,
    "Df0": 0.02,
    "T0": 298.15,
    "x0": 0.0,
}

pulse = {
    "amp": 1.0,
    "period": 180.0,
    "duty": 0.30,
}

params = {
    "k_load": 0.004751136114300546,
    "k_rel": 0.0014014242166832688,
    "k_pass": 0.0009629397127927867,
    "k_x": 0.002053523750489004,
    "k_heat": 0.45,
    "k_cool": 1.2e-3,
    "k_def": 3.5e-4,
    "k_def_rel": 1.8e-4,
    "sigma_t": 0.01,
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Iniciando simulação sintética...")
res = simulate_model(params, pulse, init, fixed, seed=seed)
print("Simulação concluída.")

# Ajustes/derivadas (garantir chaves exigidas pelo plotting_library)
if 'D_Pd' not in res:
    res['D_Pd'] = res['H'] / fixed['H_max']
if 'TempC' not in res:
    # assume TempC em Kelvin se existir TempC/Temp
    if 'Temp' in res:
        res['TempC'] = res['Temp'] - 273.15
    else:
        # se só houver TempC já no modelo, mantém
        res['TempC'] = res.get('TempC', np.zeros_like(res['t']))

# Calcular summary keys esperadas
res['x_final'] = res.get('x_final', float(res['x'][-1]) if len(res['x']) else np.nan)
res['x_max'] = res.get('x_max', float(np.max(res['x'])) if len(res['x']) else np.nan)
res['Temp_final_C'] = res.get('Temp_final_C', float(res['TempC'][-1]) if len(res['TempC']) else np.nan)
res['Temp_max_C'] = res.get('Temp_max_C', float(np.max(res['TempC'])) if len(res['TempC']) else np.nan)
res['D_Pd_final'] = res.get('D_Pd_final', float(res['D_Pd'][-1]) if len(res['D_Pd']) else np.nan)
res['Df_final'] = res.get('Df_final', float(res['Df'][-1]) if len(res['Df']) else np.nan)
res['t_crit'] = res.get('t_crit', np.nan)
res['success'] = res.get('success', 0)

# Salvar CSVs de timeseries e summary
timeseries_path = os.path.join(OUTPUT_DIR, 'thermal_reaction_timeseries.csv')
summary_path = os.path.join(OUTPUT_DIR, 'thermal_reaction_summary.csv')
pl.save_timeseries_csv(res, timeseries_filename=timeseries_path, summary_filename=summary_path)

# Gerar heatmap e painéis temporais
heat = np.vstack([
    res['TempC'],
    res['x'] * 40 + 25,
    res['D_Pd'] * 100 + 20,
    res['Df'] * 60 + 10
])
labels = ["Temp (C)", "x scaled", "D/Pd scaled", "Df scaled"]
heatmap_file = os.path.join(OUTPUT_DIR, 'thermal_evolution_heatmap.png')
panels_file = os.path.join(OUTPUT_DIR, 'thermal_evolution_panels.png')
pl.plot_heatmap(heat, labels, filename=heatmap_file)
pl.plot_temporal_panels(res, filename=panels_file)

# ROOT histograms (opcional)
print("Tentando gerar histogramas ROOT (se disponível)...")
pl.root_histograms_from_results(res, prefix=os.path.join(OUTPUT_DIR, 'thermal_analysis'))

# Tentar carregar advanced Monte Carlo (se existir) e plotar estabilidade
mc_csv = 'advanced_mc_results.csv'
if os.path.exists(mc_csv):
    try:
        df_mc = pd.read_csv(mc_csv)
        stability_file = os.path.join(OUTPUT_DIR, 'stability_scatter.png')
        pl.plot_seaborn_stability(df_mc, x_limit=fixed['x_crit'], filename=stability_file, show=False)

        # gerar renders artísticos (com número reduzido para tempo de execução)
        jwst_file = os.path.join(OUTPUT_DIR, 'jwst_render.png')
        pl.jwst_style_particle_render(df_mc, df_astro=None, n_photons=15000, filename=jwst_file, seed=seed)

        fusion_file = os.path.join(OUTPUT_DIR, 'analogia_4k_complexa.png')
        pl.fusion_astro_4k(df_mc, df_astro=None, n_particles=30000, filename=fusion_file, seed=seed)

        # calcular correlação e exportar PDF
        corr = df_mc.select_dtypes(include=[np.number]).corr()
        pdf_file = os.path.join(OUTPUT_DIR, 'relatorio_correlacao_avancada.pdf')
        pl.export_results_to_pdf(corr, filename=pdf_file)

    except Exception as e:
        print(f"Erro ao processar advanced_mc_results.csv: {e}")
else:
    print(f"Arquivo {mc_csv} não encontrado — pulando plots de Monte Carlo avançado.")

# Tentar buscar dados da NASA e gerar scatter
try:
    df_astro = pl.plot_exoplanet_scatter_from_url(nrows=500, filename=os.path.join(OUTPUT_DIR, 'exoplanets_scatter.png'))
except Exception as e:
    print(f"Falha ao gerar scatter de exoplanetas: {e}")

print("Execução completa. Arquivos gerados em:", os.path.abspath(OUTPUT_DIR))
