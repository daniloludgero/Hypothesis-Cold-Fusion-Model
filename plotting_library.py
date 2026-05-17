import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

# ROOT (opcional)
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    HAVE_ROOT = True
except Exception:
    HAVE_ROOT = False

# ============================================================
# plotting_library.py
# Biblioteca centralizada de funções de visualização para o
# repositório Hypothesis-Cold-Fusion-Model
# Todas as funções recebem datasets como parâmetros (não usam
# variáveis globais externas) e podem salvar em arquivo ou
# apenas exibir a figura.
# ============================================================


def plot_heatmap(heat, labels, filename="thermal_evolution_heatmap.png", dpi=600):
    """Gera heatmap térmico de alta qualidade.

    heat: array-like 2D (linhas = variáveis, colunas = tempo)
    labels: lista de rótulos por linha
    filename: se fornecido, salva o arquivo; senão apenas mostra
    """
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    im = ax.imshow(heat, aspect="auto", cmap="inferno", origin="lower",
                   norm=Normalize(vmin=np.min(heat), vmax=np.max(heat)))

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time step")
    ax.set_title("Internal Thermal and State Evolution")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Scaled intensity")

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Heatmap salvo como {filename}")
    else:
        plt.show()


def plot_temporal_panels(results, filename="thermal_evolution_panels.png", dpi=600):
    """Plota painel 2x2 com TempC, x, D_Pd e Df.

    results: dict contendo chaves t, TempC, x, D_Pd, Df
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    axs[0, 0].plot(results["t"], results["TempC"], color="crimson", lw=2)
    axs[0, 0].set_title("Temperature")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("C")

    axs[0, 1].plot(results["t"], results["x"], color="navy", lw=2)
    axs[0, 1].axhline(results.get("x_crit", 1.0), color="black", ls="--", lw=1)
    axs[0, 1].set_title("Critical index x")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("x")

    axs[1, 0].plot(results["t"], results["D_Pd"], color="darkgreen", lw=2)
    axs[1, 0].set_title("D/Pd proxy")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("D/Pd")

    axs[1, 1].plot(results["t"], results["Df"], color="purple", lw=2)
    axs[1, 1].set_title("Df")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Df")

    plt.suptitle("Thermal and Structural Reaction Evolution", y=1.02, fontsize=14)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Painéis temporais salvos como {filename}")
    else:
        plt.show()


def plot_seaborn_stability(df_mc, x_limit=1.0, filename=None, show=True):
    """Scatter plot seaborn para análise de estabilidade.

    df_mc: DataFrame contendo colunas 'T_max', 'x_final', 'H_final'
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    scatter = sns.scatterplot(
        data=df_mc,
        x='T_max',
        y='x_final',
        hue='x_final',
        palette='viridis',
        size='H_final' if 'H_final' in df_mc.columns else None,
        sizes=(50, 200),
        alpha=0.8
    )

    plt.axhline(y=x_limit, color='red', linestyle='--', linewidth=2, label='Limite de Estabilidade (Crítico)')

    try:
        xmin = df_mc['T_max'].min() - 5
        xmax = df_mc['T_max'].max() + 5
        plt.fill_between([xmin, xmax], 0, x_limit, color='green', alpha=0.1, label='Zona Estável')
    except Exception:
        pass

    plt.title('Dispersão de Estabilidade: Índice Crítico vs Temperatura', fontsize=14)
    plt.xlabel('Temperatura Máxima de Reação (°C)', fontsize=12)
    plt.ylabel('Índice Crítico (x)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter de estabilidade salvo como {filename}")
    elif show:
        plt.show()


def jwst_style_particle_render(df_mc, df_astro=None, n_photons=50000, filename=None, seed=2026):
    """Render estilo JWST / 4K para visualização artística baseada em métricas.

    df_mc: DataFrame com coluna 'T_max' e 'x_final'
    df_astro: DataFrame (opcional) com coluna 'pl_orbper' para métricas
    """
    rng = np.random.default_rng(seed)

    # Extração segura de métricas
    t_max_sim = float(df_mc['T_max'].max()) if ('T_max' in df_mc.columns and not df_mc['T_max'].isna().all()) else 300.0
    x_crit_sim = float(df_mc['x_final'].mean()) if 'x_final' in df_mc.columns else 0.5

    if df_astro is not None and 'pl_orbper' in df_astro.columns:
        orb_per_avg = float(df_astro['pl_orbper'].median())
    else:
        orb_per_avg = 1.0

    phi = rng.uniform(0, 2*np.pi, n_photons)
    costheta = rng.uniform(-1, 1, n_photons)
    theta = np.arccos(costheta)

    r = rng.normal(loc=0, scale=0.5 * max(0.01, x_crit_sim), size=n_photons) + rng.exponential(scale=0.2, size=n_photons)

    x_p = r * np.sin(theta) * np.cos(phi)
    y_p = r * np.sin(theta) * np.sin(phi)

    energies = (1.0 / (np.abs(r) + 0.1)) * (t_max_sim / 100.0) + rng.normal(0, 0.05, n_photons)

    colors_webb = ["#02040a", "#1a0800", "#4d0000", "#e65c00", "#ffcc00", "#ffffff"]
    cmap_jwst = LinearSegmentedColormap.from_list("jwst_ultra", colors_webb)

    fig, ax = plt.subplots(figsize=(15, 15), facecolor='#010206')

    ax.hexbin(x_p, y_p, gridsize=150, cmap=cmap_jwst, bins='log', alpha=0.2, edgecolors='none')

    scatter = ax.scatter(x_p, y_p, c=energies, s=rng.uniform(0.1, 8.0, n_photons),
                         cmap=cmap_jwst, alpha=0.7, edgecolors='none', marker='o')

    for i in range(1, 4):
        circle = plt.Circle((0, 0), 0.02 * i * max(0.001, x_crit_sim), color='#ffcc00', alpha=0.15 / i, fill=True)
        ax.add_artist(circle)

    ax.set_facecolor('#010206')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    plt.title("RELATÓRIO VISUAL: SÍNTESE DE EVENTO TÉRMICO CRÍTICO (SIMULAÇÃO 4K)",
              color='white', fontsize=18, fontweight='bold', family='serif', y=0.95)

    info_text = (f"MODELO ANALÓGICO: FUSÃO/NASA DATA\n"
                 f"TEMPERATURA DE PICO: {t_max_sim:.2f}°C\n"
                 f"ÍNDICE DE ESTABILIDADE: {1 - x_crit_sim:.4f}\n"
                 f"FLUXO DE PARTÍCULAS: {n_photons} PH_SIM")

    plt.figtext(0.15, 0.15, info_text, color='#ffcc00', fontsize=12,
                fontfamily='monospace', bbox=dict(facecolor='black', alpha=0.6, edgecolor='#4d0000'))

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#010206')
        plt.close()
        print(f"Render JWST salvo como {filename}")
    else:
        plt.show()


def fusion_astro_4k(df_mc, df_astro=None, n_particles=100000, filename=None, seed=42):
    """Render analógico 4K que mistura assinaturas de fusão e exoplaneta.

    df_mc: DataFrame com métricas de Monte Carlo
    df_astro: DataFrame opcional para métricas astro
    """
    rng = np.random.default_rng(seed)

    t_pico = float(df_mc['T_max'].max()) if ('T_max' in df_mc.columns and not df_mc['T_max'].isna().all()) else 300.0
    crit_avg = float(df_mc['x_final'].mean()) if 'x_final' in df_mc.columns else 0.5

    x_fusion = rng.normal(loc=-1.5, scale=0.4 * max(0.01, (1 - crit_avg)), size=n_particles // 2)
    y_fusion = rng.normal(loc=0, scale=0.4 * max(0.01, (1 - crit_avg)), size=n_particles // 2)

    x_planet = rng.normal(loc=1.5, scale=0.6, size=n_particles // 2)
    y_planet = rng.normal(loc=0, scale=0.6, size=n_particles // 2)

    x_total = np.concatenate([x_fusion, x_planet])
    y_total = np.concatenate([y_fusion, y_planet])

    colors_f = (np.sqrt(x_fusion**2 + y_fusion**2) * (t_pico / 50))
    colors_p = (np.abs(x_planet) * 2.5)
    intensities = np.concatenate([colors_f, colors_p])

    colors_bridge = ["#010206", "#0d001a", "#4b0082", "#ff4500", "#ffff00", "#ffffff"]
    cmap_4k = LinearSegmentedColormap.from_list("fusion_astro", colors_bridge)

    fig, ax = plt.subplots(figsize=(20, 10), facecolor='#010206')

    ax.hexbin(x_total, y_total, gridsize=200, cmap=cmap_4k, bins='log', alpha=0.15)

    ax.scatter(x_total, y_total, c=intensities, s=rng.uniform(0.05, 4.0, n_particles),
               cmap=cmap_4k, alpha=0.6, edgecolors='none')

    ax.set_facecolor('#010206')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2, 2)
    ax.axis('off')

    plt.title("ANALOGIA TÉRMICA TRANS-ESCALAR: FUSÃO QUÍMICA & ASSINATURA EXOPLANETÁRIA",
              color='white', fontsize=22, fontweight='bold', family='serif', pad=20)

    left_text = f"[NÚCLEO SINTÉTICO]\nESTABILIDADE: {1 - crit_avg:.4f}\nTEMP PICO: {t_pico:.1f}°C"
    right_text = f"[ASSINATURA NASA]\nCORRELAÇÃO: N/A\nZONA HABITÁVEL POTENCIAL"

    plt.figtext(0.25, 0.15, left_text, color='#ff4500', fontsize=12, fontfamily='monospace',
                bbox=dict(facecolor='black', alpha=0.7))
    plt.figtext(0.65, 0.15, right_text, color='#00ffcc', fontsize=12, fontfamily='monospace',
                bbox=dict(facecolor='black', alpha=0.7))

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#010206')
        plt.close()
        print(f"Analogia 4k salva como {filename}")
    else:
        plt.show()


def plot_exoplanet_scatter_from_url(nrows=500, filename=None):
    """Baixa dados da NASA Exoplanet Archive (consulta simples) e plota scatter.
    Retorna o DataFrame carregado.
    """
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+pl_name,pl_orbper,pl_eqt,pl_rade+from+ps+where+pl_orbper+>+0+and+pl_eqt+>+0+and+pl_rade+>+0&format=csv"
    )
    try:
        df_astro = pd.read_csv(url, nrows=nrows)
        df_astro = df_astro.dropna()

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="darkgrid")

        scatter = sns.scatterplot(
            data=df_astro,
            x='pl_orbper',
            y='pl_eqt',
            size='pl_rade',
            hue='pl_eqt',
            palette='magma',
            alpha=0.7
        )

        plt.xscale('log')
        plt.title('Relação Térmica Orbital: Dados Reais da NASA', fontsize=14)
        plt.xlabel('Período Orbital (Dias) - Escala Log', fontsize=12)
        plt.ylabel('Temperatura de Equilíbrio (K)', fontsize=12)
        plt.legend(title='Raio (Terra = 1)', bbox_to_anchor=(1.05, 1), loc='upper left')

        corr_astro = np.corrcoef(np.log10(df_astro['pl_orbper']), df_astro['pl_eqt'])[0, 1]
        plt.annotate(f'Correlação (Log-Linear): {corr_astro:.2f}', xy=(0.05, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round", fc="white", alpha=0.9))

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Scatter de exoplanetas salvo como {filename}")
        else:
            plt.show()

        return df_astro

    except Exception as e:
        print(f"Ocorreu um erro ao buscar/plotar dados NASA: {e}")
        return None


def export_results_to_pdf(df_corr, filename='relatorio_correlacao_avancada.pdf'):
    """Exporta uma matriz de correlação (DataFrame) para um PDF com heatmap + tabela."""
    try:
        with PdfPages(filename) as pdf:
            plt.figure(figsize=(11, 8.5))
            sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=.5)
            plt.title('Matriz de Correlação: Análise de Sensibilidade Monte Carlo', fontsize=16, pad=20)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            fig, ax = plt.subplots(figsize=(12, 10))
            ax.axis('off')
            table = ax.table(cellText=df_corr.values.round(3),
                             colLabels=df_corr.columns,
                             rowLabels=df_corr.index,
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.2)
            plt.title('Dados Numéricos da Matriz', fontsize=14, pad=10)
            pdf.savefig()
            plt.close()
        print(f'Relatório exportado com sucesso: {filename}')
    except Exception as e:
        print(f'Erro ao exportar PDF: {e}')


def plot_corr_matrix_from_csv(csv_path='advanced_mc_results.csv', filename=None):
    """Carrega CSV de resultados avançados, calcula matriz de correlação e plota heatmap.
    Retorna o DataFrame de correlação.
    """
    try:
        df_adv = pd.read_csv(csv_path)
        corr_matrix = df_adv.select_dtypes(include=[np.number]).corr()
        corr_matrix.to_csv('full_correlation_matrix.csv')

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Matriz de Correlação Completa: Advanced Monte Carlo', fontsize=14)
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Heatmap de correlação salvo como {filename}")
        else:
            plt.show()

        return corr_matrix
    except Exception as e:
        print(f'Erro ao processar CSV de correlação: {e}')
        return None


def root_histograms_from_results(res, prefix='thermal_analysis'):
    """Gera histogramas com ROOT caso disponível. res é o dicionário retornado pela simulação.
    Salva arquivos PNG e um ROOT file.
    """
    if not HAVE_ROOT:
        print('ROOT não está disponível neste ambiente.')
        return False

    try:
        f = ROOT.TFile(f"{prefix}.root", "RECREATE")

        h1 = ROOT.TH1D("h_x", "x distribution;x;Counts", 60, 0, max(3.0, float(np.max(res['x']) * 1.1)))
        h2 = ROOT.TH1D("h_T", "Temperature distribution;Temp (C);Counts", 60, float(np.min(res['TempC'])), float(np.max(res['TempC'])))
        h3 = ROOT.TH1D("h_D", "D/Pd distribution;D/Pd;Counts", 60, 0, 1.0)

        for v in res['x']:
            h1.Fill(float(v))
        for v in res['TempC']:
            h2.Fill(float(v))
        for v in res['D_Pd']:
            h3.Fill(float(v))

        c1 = ROOT.TCanvas("c1", "c1", 1200, 800)
        h1.SetLineColor(ROOT.kBlue + 1)
        h1.Draw()
        c1.SaveAs(f"{prefix}_hist_x.png")

        c2 = ROOT.TCanvas("c2", "c2", 1200, 800)
        h2.SetLineColor(ROOT.kRed + 1)
        h2.Draw()
        c2.SaveAs(f"{prefix}_hist_temp.png")

        c3 = ROOT.TCanvas("c3", "c3", 1200, 800)
        h3.SetLineColor(ROOT.kGreen + 2)
        h3.Draw()
        c3.SaveAs(f"{prefix}_hist_dpd.png")

        f.Write()
        f.Close()

        print(f"ROOT outputs salvos com prefixo {prefix}")
        return True
    except Exception as e:
        print(f"Erro ao gerar histogramas ROOT: {e}")
        return False


# Utility helpers

def save_timeseries_csv(res, timeseries_filename='thermal_reaction_timeseries.csv', summary_filename='thermal_reaction_summary.csv'):
    """Salva arquivos CSV de timeseries e resumo a partir do dicionário de resultados da simulação."""
    try:
        df = pd.DataFrame({
            't_s': res['t'],
            'H': res['H'],
            'Df': res['Df'],
            'TempC': res['TempC'],
            'x': res['x'],
            'D_Pd': res['D_Pd']
        })
        df.to_csv(timeseries_filename, index=False)

        summary = pd.DataFrame([{
            'x_final': res.get('x_final', float(res['x'][-1]) if len(res['x']) else np.nan),
            'x_max': res.get('x_max', float(np.max(res['x'])) if len(res['x']) else np.nan),
            'Temp_final_C': res.get('Temp_final_C', float(res['TempC'][-1]) if len(res['TempC']) else np.nan),
            'Temp_max_C': res.get('Temp_max_C', float(np.max(res['TempC'])) if len(res['TempC']) else np.nan),
            'D_Pd_final': res.get('D_Pd_final', float(res['D_Pd'][-1]) if len(res['D_Pd']) else np.nan),
            'Df_final': res.get('Df_final', float(res['Df'][-1]) if len(res['Df']) else np.nan),
            't_crit': res.get('t_crit', np.nan),
            'success': res.get('success', 0)
        }])
        summary.to_csv(summary_filename, index=False)

        print(f"Saved: {timeseries_filename}, {summary_filename}")
    except Exception as e:
        print(f"Erro ao salvar CSVs: {e}")
