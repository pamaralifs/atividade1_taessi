# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
****************************************
Script: program_02.py
========================================
Powered by: ChatGPT e Gemini
Revisado e modificado por: Paulo A Costa
Data: Abril/2026
****************************************
1) Gera gráficos a partir dos dados do arquivo de saída avaliacao_bertscore_long_equipe4_paulo.csv gerado pelo program_01.py
"""

# 1. Definição da função para rótulos
def add_value_labels(ax, fmt="{:.1f}", fontsize=9):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=fontsize)

# 2. Caminho da sua pasta no Drive
base_path = '/content/drive/MyDrive/aval1_qabertas/'

# Carregamento do arquivo (usando detecção automática de separador para evitar erros de leitura)
input_file = os.path.join(base_path, 'avaliacao_bertscore_long_equipe4_paulo.csv')
df = pd.read_csv(input_file, sep=None, engine='python', on_bad_lines='warn')

rename_map = {
    'Numero da OAB': 'Exame da OAB',
    'Formato': 'Formato Original',
    'Quantidade de itens': 'Quantidade Itens',
    'grupo_itens': 'Tipo Questão Discursiva',
    'Nível de dificuldade': 'Nível Dificuldade',
    'Área de especialidade': 'Área Especialidade',
    'ollama_model': 'Modelo',
    'evaluator_model': 'Modelo BERTScore',
    'bertscore_f1': 'F1-Score',
    'bertscore_f1_percent': 'F1-Score (%)',
    'resposta_modelo': 'Resposta Modelo',
    'resposta_referencia_guidelines': 'Resposta Referência (guidelines)',
}

df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
if 'F1-Score (%)' in df.columns:
    df['F1-Score (%)'] = pd.to_numeric(df['F1-Score (%)'], errors='coerce').round(1)

# --- AJUSTE DINÂMICO DA ORDEM DOS MODELOS (CORREÇÃO PARA BARRAS VAZIAS) ---
# Identifica os nomes reais no CSV e ordena: Jurema -> Gemma -> Llama
unique_models = df['Modelo'].unique().tolist()
def sort_key(name):
    name_low = str(name).lower()
    if 'jurema' in name_low: return 0
    if 'gemma' in name_low: return 1
    if 'llama' in name_low: return 2
    return 3
ordem_efetiva = sorted(unique_models, key=sort_key)
df['Modelo'] = pd.Categorical(df['Modelo'], categories=ordem_efetiva, ordered=True)

# --- MAPEAMENTO DOS NOMES DOS MODELOS PARA OS RÓTULOS DO EIXO X ---
# Esta função renomeia os valores para os nomes amigáveis que aparecerão nos gráficos
def rename_model_labels(name):
    name_low = str(name).lower()
    if 'jurema' in name_low: return "jurema:7b"
    if 'gemma' in name_low: return "gemma3:12b"
    if 'llama' in name_low: return "llama3.1:8b"
    return name

df['Modelo'] = df['Modelo'].map(rename_model_labels)
# Reaplicar a categoria com os novos nomes para manter a ordem jurema -> gemma -> llama no eixo X
novos_nomes_ordem = ["jurema:7b", "gemma3:12b", "llama3.1:8b"]
df['Modelo'] = pd.Categorical(df['Modelo'], categories=novos_nomes_ordem, ordered=True)

# ==============================================================================
# SEÇÃO A: GRÁFICOS ORIGINAIS (MÉDIA DE AMBOS OS AVALIADORES)
# ==============================================================================

# 0) Gráfico Adicional: Média geral por modelo e avaliador
chart_general_df = df.groupby(['Modelo', 'Modelo BERTScore'], as_index=False, observed=False)['F1-Score (%)'].mean()
pivot0 = chart_general_df.pivot(index='Modelo', columns='Modelo BERTScore', values='F1-Score (%)').fillna(0)
ax = pivot0.plot(kind='bar', figsize=(7, 6), color=['#1f77b4', '#ff7f0e'], width=0.6)
ax.set_ylim(0, 115)
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score (%) questões abertas por modelo e avaliador')
ax.legend(title='Avaliador BERTScore', loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1)
plt.xticks(rotation=15, ha='right')
add_value_labels(ax)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "grafico_media_f1_percent_geral.png"), dpi=150)
plt.show()

# 1) Média geral por modelo
g1 = df.groupby('Modelo', as_index=False, observed=False)['F1-Score (%)'].mean()
ax = g1.set_index('Modelo').plot(kind='bar', figsize=(7, 6), color='#1f77b4', width=0.6, legend=False)
ax.set_ylim(0, 115)
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score (%) questões abertas por modelo')
plt.xticks(rotation=15, ha='right')
add_value_labels(ax)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'grafico_f1_percent_por_modelo.png'), dpi=150)
plt.show()

# 2) Por área de especialidade
g2 = df.groupby(['Modelo', 'Área Especialidade'], as_index=False, observed=False)['F1-Score (%)'].mean()
pivot2 = g2.pivot(index='Modelo', columns='Área Especialidade', values='F1-Score (%)').fillna(0)
ax = pivot2.plot(kind='bar', figsize=(7, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], width=0.6)
ax.set_ylim(0, 115)
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score (%) questões abertas por área')
ax.legend(title='Área Especialidade (Questões Abertas)', loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1)
plt.xticks(rotation=15, ha='right')
add_value_labels(ax)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'grafico_f1_percent_por_area.png'), dpi=150)
plt.show()

# 3) Por nível de dificuldade
g3 = df.groupby(['Modelo', 'Nível Dificuldade'], as_index=False, observed=False)['F1-Score (%)'].mean()
pivot3 = g3.pivot(index='Modelo', columns='Nível Dificuldade', values='F1-Score (%)').fillna(0)
ax = pivot3.plot(kind='bar', figsize=(7, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.6)
ax.set_ylim(0, 115)
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score (%) questões abertas por nível de dificuldade')
ax.legend(title='Nível Dificuldade (Questões Abertas)', loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1)
plt.xticks(rotation=15, ha='right')
add_value_labels(ax)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'grafico_f1_percent_por_dificuldade.png'), dpi=150)
plt.show()

# 4) Por tipo de questão
g4 = df.groupby(['Modelo', 'Tipo Questão Discursiva'], as_index=False, observed=False)['F1-Score (%)'].mean()
pivot4 = g4.pivot(index='Modelo', columns='Tipo Questão Discursiva', values='F1-Score (%)').fillna(0)
ax = pivot4.plot(kind='bar', figsize=(7, 6), color=['#1f77b4', '#ff7f0e'], width=0.6) 
ax.set_ylim(0, 115) 
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score (%) questões abertas por tipo')
ax.legend(title='Tipo Questão (Questões Abertas)', loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1)
plt.xticks(rotation=15, ha='right')
add_value_labels(ax)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'grafico_f1_percent_por_tipo_item.png'), dpi=150)
plt.show()

# ==============================================================================
# SEÇÃO B: NOVOS GRÁFICOS (APENAS AVALIADOR BERTIMBAU_LARGE)
# ==============================================================================

df_bertimbau = df[df['Modelo BERTScore'] == 'BERTimbau_Large'].copy()

# 1_B) Média geral por modelo (BERTScore/BERTimbau)
g1_b = df_bertimbau.groupby('Modelo', as_index=False, observed=False)['F1-Score (%)'].mean()
ax = g1_b.set_index('Modelo').plot(kind='bar', figsize=(7, 6), color='#1f77b4', width=0.6, legend=False)
ax.set_ylim(0, 115)
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score (%) questões abertas por modelo (BERTScore/BERTimbau)')
plt.xticks(rotation=15, ha='right')
add_value_labels(ax)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'grafico_f1_percent_por_modelo_bertimbau.png'), dpi=150)
plt.show()

# 2_B) Por área de especialidade (BERTScore/BERTimbau)
g2_b = df_bertimbau.groupby(['Modelo', 'Área Especialidade'], as_index=False, observed=False)['F1-Score (%)'].mean()
pivot2_b = g2_b.pivot(index='Modelo', columns='Área Especialidade', values='F1-Score (%)').fillna(0)
ax = pivot2_b.plot(kind='bar', figsize=(7, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], width=0.6)
ax.set_ylim(0, 115)
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score (%) questões abertas por área (BERTScore/BERTimbau)')
ax.legend(title='Área Especialidade (Questões Abertas)', loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1)
plt.xticks(rotation=15, ha='right')
add_value_labels(ax)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'grafico_f1_percent_por_area_bertimbau.png'), dpi=150)
plt.show()

# 3_B) Por nível de dificuldade (BERTScore/BERTimbau)
g3_b = df_bertimbau.groupby(['Modelo', 'Nível Dificuldade'], as_index=False, observed=False)['F1-Score (%)'].mean()
pivot3_b = g3_b.pivot(index='Modelo', columns='Nível Dificuldade', values='F1-Score (%)').fillna(0)
ax = pivot3_b.plot(kind='bar', figsize=(7, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.6)
ax.set_ylim(0, 115)
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score (%) questões abertas por grau dificuldade (BERTScore/BERTimbau)')
ax.legend(title='Nível Dificuldade (Questões Abertas)', loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1)
plt.xticks(rotation=15, ha='right')
add_value_labels(ax)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'grafico_f1_percent_por_dificuldade_bertimbau.png'), dpi=150)
plt.show()

# 4_B) Por tipo de questão (BERTScore/BERTimbau)
g4_b = df_bertimbau.groupby(['Modelo', 'Tipo Questão Discursiva'], as_index=False, observed=False)['F1-Score (%)'].mean()
pivot4_b = g4_b.pivot(index='Modelo', columns='Tipo Questão Discursiva', values='F1-Score (%)').fillna(0)
ax = pivot4_b.plot(kind='bar', figsize=(7, 6), color=['#1f77b4', '#ff7f0e'], width=0.6) 
ax.set_ylim(0, 115) 
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score (%) questões abertas por tipo (BERTScore/BERTimbau)')
ax.legend(title='Tipo Questão (Questões Abertas)', loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1)
plt.xticks(rotation=15, ha='right')
add_value_labels(ax)
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'grafico_f1_percent_por_tipo_item_bertimbau.png'), dpi=150)
plt.show()

# ==============================================================================
# SEÇÃO C: GRÁFICOS DESDOBRADOS SEPARADOS (APENAS BERTIMBAU)
# ==============================================================================

# a) Áreas Separadas
for area in ['Direito Constitucional', 'Direito do Trabalho']:
    g_area = df_bertimbau[df_bertimbau['Área Especialidade'] == area].groupby('Modelo', as_index=False, observed=False)['F1-Score (%)'].mean()
    ax = g_area.set_index('Modelo').plot(kind='bar', figsize=(7, 6), color='#1f77b4', width=0.6, legend=False)
    ax.set_ylim(0, 115)
    ax.set_ylabel('F1-Score (%)')
    ax.set_title(f'F1-Score (%) - {area} (BERTScore/BERTimbau)')
    plt.xticks(rotation=15, ha='right')
    add_value_labels(ax)
    plt.tight_layout()
    filename = f"grafico_f1_percent_{area.lower().replace(' ', '_')}_bertimbau.png"
    plt.savefig(os.path.join(base_path, filename), dpi=150)
    plt.show()

# b) Dificuldade Separada
for nivel in ['Médio', 'Difícil']:
    g_nivel = df_bertimbau[df_bertimbau['Nível Dificuldade'] == nivel].groupby('Modelo', as_index=False, observed=False)['F1-Score (%)'].mean()
    ax = g_nivel.set_index('Modelo').plot(kind='bar', figsize=(7, 6), color='#ff7f0e', width=0.6, legend=False)
    ax.set_ylim(0, 115)
    ax.set_ylabel('F1-Score (%)')
    ax.set_title(f'F1-Score (%) - Nível {nivel} (BERTScore/BERTimbau)')
    plt.xticks(rotation=15, ha='right')
    add_value_labels(ax)
    plt.tight_layout()
    filename = f"grafico_f1_percent_dificuldade_{nivel.lower()}_bertimbau.png"
    plt.savefig(os.path.join(base_path, filename), dpi=150)
    plt.show()

# c) Tipo Separado
for tipo in ['1 item (Peça profissional)', '2 itens (A e B)']:
    g_tipo = df_bertimbau[df_bertimbau['Tipo Questão Discursiva'] == tipo].groupby('Modelo', as_index=False, observed=False)['F1-Score (%)'].mean()
    ax = g_tipo.set_index('Modelo').plot(kind='bar', figsize=(7, 6), color='#2ca02c', width=0.6, legend=False)
    ax.set_ylim(0, 115)
    ax.set_ylabel('F1-Score (%)')
    ax.set_title(f'F1-Score (%) - {tipo} (BERTScore/BERTimbau)')
    plt.xticks(rotation=15, ha='right')
    add_value_labels(ax)
    plt.tight_layout()
    clean_tipo = "1item" if "1" in tipo else "2itens"
    filename = f"grafico_f1_percent_tipo_{clean_tipo}_bertimbau.png"
    plt.savefig(os.path.join(base_path, filename), dpi=150)
    plt.show()