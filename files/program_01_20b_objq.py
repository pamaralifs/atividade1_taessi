# -*- coding: utf-8 -*-
"""
*****************************************
Script: program_01_objq.py
=========================================
Processa 123 questões objetivas de provas
da OAB dos Exames 16°/2015 e 17º/2015,
submetidas a 3 modelos de LLM (SLM) e 
calcula A Acurácia.
=========================================
Powered by: Gemini
Revisado e modificado por: Paulo A Costa
Data: Abril/2026
*****************************************
"""

import os
import json
import requests
import pandas as pd
from tqdm.auto import tqdm

print("Configuração: Temperatura 0 | Processamento por Modelo (Otimizado para 20B+)")

# --- CONFIGURAÇÕES DE CAMINHO ---
BASE_PATH = '/content/drive/MyDrive/aval1_20b_qobjetivas/'
INPUT_FILE = os.path.join(BASE_PATH, "equipe4_paulo_questoes_objetivas_linhas_1354_1476_template_role_content.jsonl")
OUTPUT_JSONL = os.path.join(BASE_PATH, "respostas_objetivas_paulo.jsonl")
OUTPUT_CSV = os.path.join(BASE_PATH, "avaliacao_objetivas_paulo.csv")
README_OUT = os.path.join(BASE_PATH, "readme_questoes_objetivas.md")

# Alias conforme sua configuração no Notebook
OLLAMA_MODELS = [
    # "gpt-oss:20b_safe", 
    # "gemma4:26b_safe", 
    "qwen3.5:27b_safe"
]
OLLAMA_API_URL = "http://localhost:11434/api/chat"

def rename_model_labels(name):
    name_low = str(name).lower()
    # if 'gpt-oss' in name_low: return "gpt-oss:20b"
    # if 'gemma4' in name_low: return "gemma4:26b"
    if 'qwen3.5' in name_low: return "qwen3.5:27b"
    return name

# def query_ollama_structured(model_name, question_data):
#     """
#     Submete a questão ao Ollama usando a estrutura JSON pronta do arquivo.
#     """
#     payload = {
#         "model": model_name,
#         "messages": question_data["messages"],
#         "format": question_data["format"],
#         "options": question_data["options"],
#         "stream": False
#     }
    
#     try:
#         response = requests.post(OLLAMA_API_URL, json=payload, timeout=120) # Timeout maior para modelos 20B+
#         response.raise_for_status()
#         content = response.json()['message']['content']
#         return json.loads(content).get("resposta", "ERRO")
#     except Exception as e:
#         return f"ERRO: {str(e)}"

# def query_ollama_structured(model_name, question_data):
#     """
#     Submete a questão ao Ollama com suporte a modelos que usam 'thinking'.
#     """
#     # Configuramos o payload exatamente como no seu teste de sucesso
#     payload = {
#         "model": model_name,
#         "messages": question_data["messages"],
#         "format": question_data["format"],
#         "options": {
#             "temperature": 0,
#             "num_predict": 2048, # Garante espaço para o raciocínio interno + JSON
#             "num_ctx": 8192      # Contexto amplo para questões da OAB
#         },
#         "stream": False
#     }
    
#     for tentativa in range(2): # Tenta uma segunda vez em caso de erro técnico
#         try:
#             response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
#             response.raise_for_status()
#             res_json = response.json()
            
#             # O conteúdo JSON que queremos está aqui
#             content = res_json.get('message', {}).get('content', '').strip()
            
#             if not content:
#                 continue
                
#             # Parse do JSON retornado pela IA
#             return json.loads(content).get("resposta", "ERRO")
            
#         except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
#             if tentativa == 0:
#                 import time
#                 time.sleep(2)
#                 continue
#             return f"ERRO_TECNICO: {str(e)}"
#     return "ERRO_FINAL"

def query_ollama_structured(model_name, question_data):
    import re

    # "num_ctx" <---É a "memória de curto prazo" total do modelo. 
    # Ele define quantos tokens (palavras/caracteres) o modelo consegue "enxergar" de uma só vez. 
    # Isso inclui: O seu System Prompt (as instruções).
    # O Enunciado da questão (que na OAB é bem longo).
    # O Raciocínio Interno (thinking) que o modelo gera.
    # A Resposta Final.
    #  "num_predict" <--- A IA pode escrever no máximo X tokens"
    # payload = {
    #     "model": model_name,
    #     "messages": question_data["messages"],
    #     "format": question_data["format"], # Mantemos o pedido de JSON
    #     "options": {
    #         "temperature": 0,
    #         "repeat_penalty": 1.5,
    #         "repeat_last_n": 500,
    #         "num_predict": 16384,
    #         "num_ctx": 32768
    #     },
    #     "stream": False
    # }
    payload = {
        "model": model_name,
        "messages": question_data["messages"],
        "format": question_data["format"],
        "options": {
            "temperature": 0,
            "repeat_penalty": 1.1,  # Suficiente para evitar loops sem travar o vocabulário
            "repeat_last_n": 64,     # Janela ágil para modelos de 20B+
            "num_predict": 1024,    # Espaço de sobra para pensar e dar o JSON (1k tokens é MUITA coisa)
            "num_ctx": 8192         # O dobro do necessário para enunciados da OAB
        },
        "stream": False
    }
    
    for tentativa in range(3):
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
            response.raise_for_status()
            content = response.json().get('message', {}).get('content', '').strip()
            
            if not content:
                continue

            # TENTATIVA 1: Parse direto do JSON (Ideal)
            try:
                data = json.loads(content)
                return str(data.get("resposta", "")).upper()
            except json.JSONDecodeError:
                # TENTATIVA 2: Se o JSON quebrou, mas a letra está no texto (Fallback)
                # Busca por algo como "resposta": "A" ou apenas a letra isolada no final
                match = re.search(r'"resposta":\s*"([A-D])"', content)
                if not match:
                    match = re.search(r'\b([A-D])\b', content.split('}')[-1]) # Tenta pegar após o JSON quebrado
                
                if match:
                    return match.group(1).upper()
                continue # Se não achou nada, tenta a próxima rodada

        except Exception as e:
            if tentativa < 2:
                import time
                time.sleep(2)
                continue
            return f"ERRO_TECNICO: {str(e)}"
            
    return "ERRO_FINAL"

# --- PROCESSAMENTO ---
results = []

# Carregamento das questões em memória para evitar múltiplas leituras de disco
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    questions = [json.loads(line) for line in f]

# Limpa o arquivo de saída se já existir
if os.path.exists(OUTPUT_JSONL): os.remove(OUTPUT_JSONL)

print(f"Iniciando processamento sequencial por modelo...")

# LOOP EXTERNO: Modelos (O Ollama carrega o modelo uma vez aqui)
for model in OLLAMA_MODELS:
    print(f"\n>>> Carregando e processando modelo: {model}")
    
    # LOOP INTERNO: Questões (Processa todas para o modelo atual)
    for q in tqdm(questions, desc=f"Status {model}"):
        q_id = q.get('id')
        subject = q.get('subject', 'Geral')
        correct_key = str(q.get('answerKey')).strip().upper()
        
        # Extração do enunciado
        enunciado = ""
        for msg in q.get("messages", []):
            if msg.get("role") == "user":
                enunciado = msg.get("content", "")
                break

        # Chamada à API
        predicted_letter = query_ollama_structured(model, q)
        
        # Cálculo de acerto
        is_correct = 1 if predicted_letter == correct_key else 0
        
        res_entry = {
            "id": q_id,
            "assunto": subject,
            "enunciado": enunciado,
            "modelo": model,
            "gabarito": correct_key,
            "resposta_modelo": predicted_letter,
            "acertou": is_correct
        }
        results.append(res_entry)
        
        # Salva incrementalmente no JSONL
        with open(OUTPUT_JSONL, 'a', encoding='utf-8') as f_json:
            f_json.write(json.dumps(res_entry, ensure_ascii=False) + '\n')

# --- GERAÇÃO DAS SAÍDAS (CSV e README) ---
df = pd.DataFrame(results)

# 1. Salvar CSV Detalhado
cols = ["id", "modelo", "assunto", "gabarito", "resposta_modelo", "acertou", "enunciado"]
df_csv = df[cols].copy()
df_csv.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

# 2. Preparar resumo para o README
df['modelo_clean'] = df['modelo'].map(rename_model_labels)
# ordem_modelos = ["gpt-oss:20b", "gemma4:26b", "qwen3.5:27b"]
ordem_modelos = ["qwen3.5:27b",]
df['modelo_clean'] = pd.Categorical(df['modelo_clean'], categories=ordem_modelos, ordered=True)

resumo = df.groupby('modelo_clean', observed=False)['acertou'].agg(['count', 'sum', 'mean']).reset_index()
resumo.columns = ['Modelo', 'Total Questões', 'Total Acertos', 'Acurácia (%)']
resumo['Acurácia (%)'] = (resumo['Acurácia (%)'] * 100).round(2)

# 3. Escrita do arquivo README_OUT
with open(README_OUT, 'w', encoding='utf-8') as f:
    f.write("### Resumo de Desempenho: Questões Objetivas (123 questões)\n\n")
    f.write("A avaliação abaixo considera a acurácia dos modelos de alta escala (20B+) nos Exames 16º e 17º da OAB.\n\n")
    f.write(resumo.to_markdown(index=False) + "\n\n")
    f.write("> **Nota:** Processamento otimizado por modelo para gestão de VRAM. Temperatura: 0.\n")

print(f"\nProcessamento concluído com sucesso!")
print(resumo.to_string(index=False))