# -*- coding: utf-8 -*-
"""
****************************************
Script: program_01.py
=======================================
Powered by: ChatGPT e Gemini
Revisado e modificado por: Paulo A Costa
Data: Abril/2026
****************************************
1) Lê as questões abertas de Paulo membro da equipe 4 do arquivo JSONL
2) Submete cada questão aos 3 modelos sendo executados no Ollama
3) Salva as respostas geradas com as respostas de referência original (e a limpa)
4) Calcula BERTScore comparando a resposta do modelo com a resposta de referência limpa
5) Gera arquivos jsonl, csv e relatórios detalhados, comparativos e um README_all_results.MD
    - respostas_modelos_equipe4_paulo.jsonl: respostas geradas pelos modelos
    - respostas_modelos_equipe4_paulo.csv: respostas em CSV
    - avaliacao_bertscore_long_equipe4_paulo.csv: avaliação detalhada em formato longo
    - avaliacao_bertscore_wide_equipe4_paulo.csv: avaliação pivoteada para facilitar comparação
    - resumo_modelos_bertscore_equipe4_paulo.csv: resumo agregado
6) O principal arquivo gerado é o avaliacao_bertscore_long_equipe4_paulo.csv que será lido pelo program_02.py para geração de gráficos.
    Eu optei por não usar os gráficos gerados por esse programa.
"""

import sys, subprocess
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "bert-score==0.3.13", "transformers==4.44.2", "sentencepiece", 
    "requests", "pandas", "tqdm", "matplotlib", "accelerate", "tabulate"
])

import os
import re
import json
import time
import textwrap
import requests
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from bert_score import score as bertscore_score

QUESTIONS_FILE = "equipe4_paulo_questoes_abertas_linhas_131_140_template_role_content.jsonl"
GUIDELINES_FILE = "guidelines.jsonl"  # Respostas de referência do espelho gabarito da OAB

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

OLLAMA_MODELS = [
    "llama_safe:latest", # Nome ajustado no Colab para o "llama3.1:8b" Q4
    "gemma_safe:latest", # Nome ajustado no Colab para o "gemma3:12b" Q4
    "jurema:latest", # "Nome ajustado no Colab para o Jurema-7B-Q4_K_M-GGUF"
]

# https://huggingface.co/Jurema-br/Jurema-7B
"""Jurema-7B é um LLM especializado no domínio jurídico brasileiro, criado a partir do ajuste fino do modelo Qwen2.5-7B-Instruct. O ajuste fino foi realizado com a utilização de um conjunto de dados sintético, majoritariamente com exemplos no formato de perguntas e respostas (Q&A), embora também incluindo outros estilos de tarefas. Os exemplos foram compilados de uma coleção diversificada e curada de documentos jurídicos de alta qualidade, selecionados por sua representatividade, qualidade e diversidade."""
# https://huggingface.co/roseval/Juru-7B
# Infelizmente não consegui baixar e testar o Juru
"""Juru-7B , um Mistral-7B especializado no domínio jurídico brasileiro. O modelo foi pré-treinado com 1,9 bilhão de tokens únicos provenientes de fontes acadêmicas e jurídicas de renome em português. Para obter detalhes completos sobre a curadoria, o treinamento e a avaliação dos dados, consulte nosso artigo: https://arxiv.org/abs/2403.18140"""

REQUEST_TIMEOUT = 900
SLEEP_BETWEEN_REQUESTS = 1.0

# Avaliadores BERTScore a serem usados na comparação das respostas dos modelos e cálculo da métrica F1-Score
BERTSCORE_MODELS = {
    "BERTimbau_Large": {
        "model_type": "neuralmind/bert-large-portuguese-cased",
        "num_layers": 24,
    },
    "mBERT": {
        "model_type": "bert-base-multilingual-cased",
        "num_layers": 12,
    },
}

BERTSCORE_BATCH_SIZE = 2
# MAX_BERTSCORE_CHARS corresponde ao número máximo de caracteres (1900)/tokens (512) que os avaliadores mBERT e BERTimbau aceitam.
# Tive que usar as funções normalize_ws e normalize_extreme para remover o máximo possível de caracteres de markdown, etc., desnecessários
# que podiam causar ruído para com os avaliadores da métrica algorítmica BERTScore.
# O máximo com segurança foi 1900 caracteres que aproximadamente daria 512 tokens.
MAX_BERTSCORE_CHARS = 1900   # com 1800 padrão não estourou, com 2500 estourou 658 do máximo de 512 tokens, com 1900 não estourou

OUTPUT_RESPONSES_JSONL = "respostas_modelos_equipe4_paulo.jsonl"
OUTPUT_RESPONSES_CSV = "respostas_modelos_equipe4_paulo.csv"
OUTPUT_EVAL_LONG_CSV = "avaliacao_bertscore_long_equipe4_paulo.csv"
OUTPUT_EVAL_WIDE_CSV = "avaliacao_bertscore_wide_equipe4_paulo.csv"
OUTPUT_SUMMARY_CSV = "resumo_modelos_bertscore_equipe4_paulo.csv"
OUTPUT_README_MD = "README_all_results.MD"

README_RENAME_MAP = {
    "ollama_model": "Modelo",
    "evaluator_model": "Modelo BERTScore",
    "group_itens": "Tipo Questão Discursiva",
    "grupo_itens": "Tipo Questão Discursiva",
    "Nível de dificuldade": "Nível Dificuldade",
    "Área de especialidade": "Área Especialidade",
    "media_f1_percent": "F1-Score (%)",
    "n": "Quantidade",
    "n_questoes": "Quantidade Questões",
    "Formato": "Formato Original",
}

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1): # Adicione o enumerador
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Erro na linha {i} do arquivo JSONL!")
                    print(f"Conteúdo próximo ao erro: {line[2940:2970]}")
                    raise e
    return rows

def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()

def normalize_ws(text):
    text = safe_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def normalize_extreme(text):
    text = safe_text(text)
    # 1. Remove asteriscos (comumente usados para negrito/itálico no Markdown)
    text = text.replace("*", "")
    # 2. Transforma qualquer sequência de espaços/quebras em um único espaço
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def limpar_distribuicao_pontos(text):
    """
    Remove padrões de 'DISTRIBUIÇÃO DE PONTOS', tabelas Markdown e 
    pontuações entre parênteses (ex: 0,50) para os itens A e B.
    """
    if not text: return ""
    # 1. Remove o cabeçalho e qualquer texto na linha da "DISTRIBUIÇÃO DE PONTOS"
    text = re.sub(r"(?i)DISTRIBUIÇÃO\s+DE\s+PONTOS.*", "", text)
    # 2. Remove tabelas Markdown
    text = re.sub(r"^\s*\|.*\|.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|?\s*:?-+\s*\|.*$", "", text, flags=re.MULTILINE)
    # 3. Remove pontuações como (0,25), (0,30/0,35) ou [0,10]
    text = re.sub(r"\s*\(\s*\d+[.,]\d+/*\d*[.,]*\d*\s*\)", "", text)
    text = re.sub(r"\s*\[\s*\d+[.,]\d+/*\d*[.,]*\d*\s*\]", "", text)
    # 4. Normaliza espaços e quebras de linha
    text = re.sub(r"\n\s*\n+", "\n\n", text) 
    return text.strip()

def truncate_for_bertscore(text, max_chars=MAX_BERTSCORE_CHARS):
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()

def flatten_reference_turns(turns):
    if not turns:
        return ""
    parts = []
    for idx, turn in enumerate(turns):
        turn = normalize_ws(turn)
        if not turn:
            continue
        if len(turns) > 1:
            label = chr(65 + idx)
            parts.append(f"Item {label}:\n{turn}")
        else:
            parts.append(turn)
    return "\n\n".join(parts).strip()

def get_reference_map(guidelines_rows, allowed_question_ids=None):
    ref_map = {}
    for row in guidelines_rows:
        qid = row.get("question_id")
        if allowed_question_ids and qid not in allowed_question_ids:
            continue
        if row.get("model_id") != "guidelines":
            continue
        choices = row.get("choices", [])
        if not choices:
            continue
        turns = choices[0].get("turns", [])
        ref_text = flatten_reference_turns(turns)
        if ref_text:
            ref_map[qid] = ref_text
    return ref_map

def build_messages(question_row):
    template = question_row["template"]
    if isinstance(template, dict) and "role" in template and "content" in template:
        system_message = template
    else:
        system_message = {"role": "system", "content": safe_text(template)}
    user_message = {"role": "user", "content": safe_text(question_row["Enunciado"])}
    return [system_message, user_message]

def call_ollama(messages, model, temperature=0.0):
    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": {"temperature": temperature}
    }
    resp = requests.post(
        OLLAMA_CHAT_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False),
        timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return normalize_ws(content), data

def infer_item_group(fmt, qty):
    txt = f"{safe_text(fmt)} {safe_text(qty)}".lower()
    if "peça profissional" in txt or "1 item" in txt:
        return "1 item (Peça profissional)"
    if "2 itens" in txt or "(a e b)" in txt:
        return "2 itens (A e B)"
    return "Outro"

def add_value_labels(ax, fmt="{:.1f}", fontsize=9):
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height):
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=fontsize
        )

def save_bar_chart(df, x_col, y_col, title, output_path, ylabel=None):
    plot_df = df[[x_col, y_col]].copy().sort_values(y_col, ascending=False)
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel if ylabel else y_col)
    plt.title(title)
    add_value_labels(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def markdown_table(df):
    if df.empty:
        return "_Sem dados._"
    return df.to_markdown(index=False)

def format_percent_cols(df):
    out = df.copy()
    for col in out.columns:
        if "%" in str(col):
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)
    return out

def prepare_readme_table(df, rename_map=None, keep_cols=None):
    out = df.copy()
    if keep_cols is not None:
        keep_existing = [c for c in keep_cols if c in out.columns]
        out = out[keep_existing]
    if rename_map:
        out = out.rename(columns=rename_map)
    out = format_percent_cols(out)
    return out

def group_summary(df, group_col, score_col="bertscore_f1_percent"):
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["ollama_model", "evaluator_model", group_col], dropna=False)
          .agg(
              n=("question_id", "count"),
              media_f1_percent=(score_col, "mean"),
          )
          .reset_index()
          .sort_values(["ollama_model", "evaluator_model", "media_f1_percent"], ascending=[True, True, False])
    )

def approval_summary(df):
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["ollama_model", "evaluator_model"], dropna=False)
          .agg(
              n=("question_id", "count"),
              media_f1_percent=("bertscore_f1_percent", "mean"),
          )
          .reset_index()
          .sort_values(["evaluator_model", "media_f1_percent"], ascending=[True, False])
    )

def recommendation_text():
    return textwrap.dedent(f"""
    ## Observações:
    - O **BERTScore não substitui a correção analítica oficial da OAB**.
    - A coluna `F1-Score (%)` corresponde ao F1 do BERTScore multiplicado por 100.
    - A coluna `resposta_referencia_limpa_guidelines` é a resposta do gabarito espelho da OAB sem os blocos de textos 
      que tratam sobre a pontuação e valores, a fim de haver uma comparação semântica automática mais justa.
    """)

questions = read_jsonl(QUESTIONS_FILE)
question_ids = [q["question_id"] for q in questions]
guidelines_rows = read_jsonl(GUIDELINES_FILE)
reference_map = get_reference_map(guidelines_rows, allowed_question_ids=set(question_ids))

print(f"Questões carregadas: {len(questions)}")
print(f"Referências encontradas: {len(reference_map)}")

results = []
for model_name in OLLAMA_MODELS:
    for q in tqdm(questions, desc=f"Submetendo questões ao modelo {model_name}"):
        qid = q["question_id"]
        messages = build_messages(q)
        try:
            answer_text, raw_response = call_ollama(messages, model=model_name)
            status = "ok"
            error_message = ""
        except Exception as e:
            answer_text = ""
            raw_response = None
            status = "error"
            error_message = str(e)

        ref_text = reference_map.get(qid, "")
        row = {
            "id": q.get("id"),
            "question_id": qid,
            "Numero da OAB": q.get("Numero da OAB"),
            "Formato": q.get("Formato"),
            "Quantidade de itens": q.get("Quantidade de itens"),
            "grupo_itens": infer_item_group(q.get("Formato"), q.get("Quantidade de itens")),
            "Nível de dificuldade": q.get("Nível de dificuldade"),
            "Área de especialidade": q.get("Área de especialidade"),
            "Referência": q.get("Referência"),
            "template": q.get("template"),
            "Enunciado": q.get("Enunciado"),
            "resposta_modelo": answer_text,
            "resposta_referencia_guidelines": ref_text,
            "resposta_referencia_limpa_guidelines": limpar_distribuicao_pontos(ref_text),
            "ollama_model": model_name,
            "status_execucao": status,
            "erro_execucao": error_message,
            "raw_ollama_response": raw_response
        }
        results.append(row)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

write_jsonl(results, OUTPUT_RESPONSES_JSONL)
pd.DataFrame(results).to_csv(OUTPUT_RESPONSES_CSV, index=False, encoding="utf-8-sig")

valid_rows = [
    r for r in results
    if r["status_execucao"] == "ok"
    and safe_text(r.get("resposta_modelo"))
    and safe_text(r.get("resposta_referencia_limpa_guidelines"))
]

if not valid_rows:
    raise RuntimeError("Não há respostas válidas para calcular BERTScore.")

all_eval_rows = []
for evaluator_name, evaluator_cfg in BERTSCORE_MODELS.items():
    # Tive que usar a normalize_extreme aqui para o cálculo ser possível
    cands = [truncate_for_bertscore(normalize_extreme(r["resposta_modelo"]), max_chars=MAX_BERTSCORE_CHARS) for r in valid_rows]
    refs  = [truncate_for_bertscore(normalize_extreme(r["resposta_referencia_limpa_guidelines"]), max_chars=MAX_BERTSCORE_CHARS) for r in valid_rows]

    bert_model = evaluator_cfg["model_type"]
    num_layers = evaluator_cfg["num_layers"]

    print(f"Calculando BERTScore com evaluator={evaluator_name}...")

    # Chamada da função preservando seus parâmetros originais
    P, R, F1 = bertscore_score(
        cands=cands,
        refs=refs,
        model_type=bert_model,
        num_layers=num_layers,
        verbose=True,
        batch_size=BERTSCORE_BATCH_SIZE,
        device="cuda" if os.path.exists("/usr/local/cuda") else "cpu",
        rescale_with_baseline=False,
        use_fast_tokenizer=False, 
        all_layers=False          
    )

    for row, f1 in zip(valid_rows, F1.tolist()):
        all_eval_rows.append({
            "id": row["id"],
            "question_id": row["question_id"],
            "Numero da OAB": row["Numero da OAB"],
            "Formato": row["Formato"],
            "Quantidade de itens": row["Quantidade de itens"],
            "grupo_itens": row["grupo_itens"],
            "Nível de dificuldade": row["Nível de dificuldade"],
            "Área de especialidade": row["Área de especialidade"],
            "ollama_model": row["ollama_model"],
            "evaluator_model": evaluator_name,
            "bertscore_model_type": bert_model,
            "bertscore_f1": f1,
            "bertscore_f1_percent": f1 * 100.0,
            "resposta_modelo": row["resposta_modelo"],
            "resposta_referencia_limpa_guidelines": row["resposta_referencia_limpa_guidelines"],
        })

eval_long_df = pd.DataFrame(all_eval_rows)
eval_long_df.to_csv(OUTPUT_EVAL_LONG_CSV, index=False, encoding="utf-8-sig")

eval_wide_df = (
    eval_long_df.pivot_table(
        index=["id", "question_id", "Numero da OAB", "Formato", "Quantidade de itens", "grupo_itens",
               "Nível de dificuldade", "Área de especialidade", "ollama_model"],
        columns="evaluator_model",
        values=["bertscore_f1", "bertscore_f1_percent"],
        aggfunc="first"
    )
)
eval_wide_df.columns = [f"{metric}__{evaluator}" for metric, evaluator in eval_wide_df.columns]
eval_wide_df = eval_wide_df.reset_index()
for metric in ["bertscore_f1", "bertscore_f1_percent"]:
    cols = [c for c in eval_wide_df.columns if c.startswith(metric + "__")]
    if cols:
        eval_wide_df[f"{metric}__MEDIA_2_AVALIADORES"] = eval_wide_df[cols].mean(axis=1)
eval_wide_df.to_csv(OUTPUT_EVAL_WIDE_CSV, index=False, encoding="utf-8-sig")

summary_df = approval_summary(eval_long_df)
summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

summary_area_df = group_summary(eval_long_df, "Área de especialidade")
summary_diff_df = group_summary(eval_long_df, "Nível de dificuldade")
summary_items_df = group_summary(eval_long_df, "grupo_itens")

chart_paths = []
base_chart_dir = "/content/drive/MyDrive/aval1_qabertas/"
chart_general_df = summary_df.copy()
chart_general_df["modelo_avaliador"] = chart_general_df["ollama_model"] + " | " + chart_general_df["evaluator_model"]
chart1 = os.path.join(base_chart_dir, "grafico_media_f1_percent_geral.png")
save_bar_chart(chart_general_df, "modelo_avaliador", "media_f1_percent", "Média geral de F1 (%) por modelo e avaliador", chart1, ylabel="F1 (%)")
chart_paths.append(chart1)

chart_area_df = summary_area_df.groupby(["ollama_model", "Área de especialidade"], as_index=False).agg(media_f1_percent=("media_f1_percent", "mean"))
chart_area_df["modelo_area"] = chart_area_df["ollama_model"] + " | " + chart_area_df["Área de especialidade"]
chart2 = os.path.join(base_chart_dir, "grafico_media_f1_percent_por_area.png")
save_bar_chart(chart_area_df, "modelo_area", "media_f1_percent", "Média de F1 (%) por área de especialidade", chart2, ylabel="F1 (%)")
chart_paths.append(chart2)

chart_diff_df = summary_diff_df.groupby(["ollama_model", "Nível de dificuldade"], as_index=False).agg(media_f1_percent=("media_f1_percent", "mean"))
chart_diff_df["modelo_dif"] = chart_diff_df["ollama_model"] + " | " + chart_diff_df["Nível de dificuldade"]
chart3 = os.path.join(base_chart_dir, "grafico_media_f1_percent_por_dificuldade.png")
save_bar_chart(chart_diff_df, "modelo_dif", "media_f1_percent", "Média de F1 (%) por nível de dificuldade", chart3, ylabel="F1 (%)")
chart_paths.append(chart3)

chart_items_df = summary_items_df.groupby(["ollama_model", "grupo_itens"], as_index=False).agg(media_f1_percent=("media_f1_percent", "mean"))
chart_items_df["modelo_tipo"] = chart_items_df["ollama_model"] + " | " + chart_items_df["grupo_itens"]
chart4 = os.path.join(base_chart_dir, "grafico_media_f1_percent_por_tipo_item.png")
save_bar_chart(chart_items_df, "modelo_tipo", "media_f1_percent", "Média de F1 (%) por tipo de questão", chart4, ylabel="F1 (%)")
chart_paths.append(chart4)

ranking_media_2 = (
    eval_wide_df.groupby("ollama_model", as_index=False)
    .agg(
        n_questoes=("question_id", "count"),
        media_f1_percent=("bertscore_f1_percent__MEDIA_2_AVALIADORES", "mean"),
    )
    .sort_values("media_f1_percent", ascending=False)
)

readme_parts = []
readme_parts.append("# Avaliação comparativa de modelos na 2ª fase da OAB – Equipe 4\n")
readme_parts.append("\n## Ranking geral dos modelos\n")
readme_parts.append(markdown_table(prepare_readme_table(ranking_media_2, rename_map=README_RENAME_MAP, keep_cols=["ollama_model", "n_questoes", "media_f1_percent"])))
readme_parts.append("\n## Resumo por modelo e avaliador\n")
readme_parts.append(markdown_table(prepare_readme_table(summary_df, rename_map=README_RENAME_MAP, keep_cols=["ollama_model", "evaluator_model", "n", "media_f1_percent"])))
readme_parts.append("\n## Desempenho por área de especialidade\n")
readme_parts.append(markdown_table(prepare_readme_table(summary_area_df, rename_map=README_RENAME_MAP, keep_cols=["ollama_model", "evaluator_model", "Área de especialidade", "n", "media_f1_percent"])))
readme_parts.append("\n## Desempenho por nível de dificuldade\n")
readme_parts.append(markdown_table(prepare_readme_table(summary_diff_df, rename_map=README_RENAME_MAP, keep_cols=["ollama_model", "evaluator_model", "Nível de dificuldade", "n", "media_f1_percent"])))
readme_parts.append("\n## Desempenho por tipo de questão\n")
readme_parts.append(markdown_table(prepare_readme_table(summary_items_df, rename_map=README_RENAME_MAP, keep_cols=["ollama_model", "evaluator_model", "grupo_itens", "n", "media_f1_percent"])))
readme_parts.append(recommendation_text())
readme_parts.append("\n## Gráficos gerados\n")
for p in chart_paths:
    fname = os.path.basename(p)
    readme_parts.append(f"![{fname}]({fname})\n")

with open(OUTPUT_README_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(readme_parts))

print("Arquivos finais:")
print("-", OUTPUT_RESPONSES_JSONL)
print("-", OUTPUT_RESPONSES_CSV)
print("-", OUTPUT_EVAL_LONG_CSV)
print("-", OUTPUT_EVAL_WIDE_CSV)
print("-", OUTPUT_SUMMARY_CSV)
print("-", OUTPUT_README_MD)