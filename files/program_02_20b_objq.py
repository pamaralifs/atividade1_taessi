# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
*****************************************
Script: program_02_objq.py
=========================================
Gera gráficos de Acurácia das respostas
das 123 questões objetivas de provas
da OAB dos Exames 16°/2015 e 17º/2015,
que foram submetidas a 3 modelos de LLM
(SLM).
=========================================
Powered by: Gemini
Revisado e modificado por: Paulo A Costa
Data: Abril/2026
*****************************************
"""

# 1. Definição da função para rótulos de dados
def add_value_labels(ax, fmt="{:.1f}%", fontsize=10):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=fontsize, fontweight='bold')

# 2. Configurações de Caminho
base_path = '/content/drive/MyDrive/aval1_20b_qobjetivas/'
input_file = os.path.join(base_path, 'avaliacao_objetivas_paulo.csv')

def rename_model_labels(name):
    name_low = str(name).lower()
    if 'gpt-oss:20b_safe' in name_low: return "gpt-oss:20b"
    if 'gemma4:26b_safe' in name_low: return "gemma4:26b"
    if 'qwen3.5:27b_safe' in name_low: return "qwen3.5:27b"

# Carregamento dos dados
if not os.path.exists(input_file):
    print(f"Erro: O arquivo {input_file} não foi encontrado.")
else:
    df = pd.read_csv(input_file)

    # 1. Renomeia os valores internos da coluna
    df['modelo'] = df['modelo'].map(rename_model_labels)
    
    # 2. Configura a ordem categórica usando o nome correto da coluna
    ordem_modelos = ["gpt-oss:20b", "gemma4:26b", "qwen3.5:27"]
    df['modelo'] = pd.Categorical(df['modelo'], categories=ordem_modelos, ordered=True)  

    # Cores fixas para identidade visual
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # ==========================================================================
    # GRÁFICO 1: ACURÁCIA GERAL POR MODELO
    # ==========================================================================
    plt.figure(figsize=(8, 6))
    # Agrupamento usando a coluna correta 'modelo'
    acc_geral = df.groupby('modelo', observed=False)['acertou'].mean() * 100
    
    ax1 = acc_geral.plot(kind='bar', color=cores, width=0.6)
    ax1.set_ylim(0, 110) 
    ax1.set_ylabel('Acurácia (%)', fontsize=12)
    ax1.set_xlabel('Modelo', fontsize=12)
    ax1.set_title('Acurácia (%) das 123 questões objetivas por modelo', fontsize=14, pad=20)
    
    plt.xticks(rotation=0)
    add_value_labels(ax1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "grafico_acuracia_geral_objetivas.png"), dpi=150)
    plt.show()

    # ==========================================================================
    # GRÁFICO 2: ACERTOS VS ERROS
    # ==========================================================================
    df_counts = df.groupby(['modelo', 'acertou'], observed=False).size().unstack(fill_value=0)
    
    # Garantia de colunas 0 (Erro) e 1 (Acerto)
    for col in [0, 1]:
        if col not in df_counts.columns:
            df_counts[col] = 0
            
    df_counts = df_counts[[0, 1]]
    df_counts.columns = ['Erros', 'Acertos']
    
    ax2 = df_counts.plot(kind='bar', stacked=True, figsize=(9, 6), color=['#d62728', '#2ca02c'], alpha=0.8)
    ax2.set_xlabel('Modelo', fontsize=12)
    ax2.set_ylabel('Quantidade de Questões', fontsize=12)
    ax2.set_title('Volume de acertos e erros das 123 questões objetivas por modelo', fontsize=14)
    ax2.set_ylim(0, 140) 
    ax2.legend(title="Resultado", loc='upper right', ncol=2)
    
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            x, y = p.get_xy() 
            ax2.text(x + p.get_width()/2, y + height/2, f'{int(height)}', 
                     ha='center', va='center', color='white', fontweight='bold')

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "grafico_contagem_acertos_objetivas.png"), dpi=150)
    plt.show()