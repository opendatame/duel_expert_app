# app.py
from flask import Flask, render_template, request
import pandas as pd
import ast
import os
import gc
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel, AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objs as go
import plotly.io as pio
import re

# ----------------------------
# CONFIG FLASK & PATHS
# ----------------------------
app = Flask(__name__)
BACKGROUND_IMAGE = "/home/aimssn-it/duel_expert_app/static/pexels-karola-g-5632382.jpg"

DOMAIN_CSV = "/home/aimssn-it/duel_expert_app/data/domain_expert_phase2_top10_predictions_full (1).csv"
DUEL_CSV = "/home/aimssn-it/duel_expert_app/data/duel_expert_mistral_GENERAL_EXPERT_top10_corrected (1).csv"
GLOBAL_CSV = "/home/aimssn-it/duel_expert_app/data/produits_nettoyes.csv"
PHASE2_CKPT = "/home/aimssn-it/duel_expert_app/models/domain_expert_flat.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 160

# ----------------------------
# DOMAIN EXPERT MODEL
# ----------------------------
class DomainExpert(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.xlm = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlm.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

# Tokenizer & LabelEncoder
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")

df_global = pd.read_csv(GLOBAL_CSV, sep=';', on_bad_lines='skip')
df_global['taxonomy_path'] = df_global['taxonomy_path'].astype(str)

le = LabelEncoder()
le.fit(df_global['taxonomy_path'].tolist())
NUM_CLASSES = len(le.classes_)

model = DomainExpert(NUM_CLASSES).to(DEVICE)
if os.path.exists(PHASE2_CKPT):
    ckpt = torch.load(PHASE2_CKPT, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    del ckpt, sd
    gc.collect()
    torch.cuda.empty_cache()
model.eval()

# ----------------------------
# LLM Mistral - chargement paresseux
# ----------------------------
llm_model = None
llm_tokenizer = None

def load_llm():
    global llm_model, llm_tokenizer
    if llm_model is None:
        llm_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=False, trust_remote_code=True)
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        )

def llm_correct(text, top10_preds, wrong_top1):
    load_llm()  # le modèle est chargé seulement au moment de la prédiction
    prompt = f"""
You are a GENERAL EXPERT correcting a WRONG product classification.

IMPORTANT:
The current top-1 prediction is WRONG and must NOT be selected again.

Product:
{text}

Candidate categories (the correct one is in this list):
{', '.join(top10_preds)}

Rules:
- Choose exactly ONE category
- It must be DIFFERENT from the wrong top-1
- Choose only from the list
- No explanations

Final choice:
"""
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    out_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    match = re.search(r"Final choice\s*:\s*(.+)", out_text, re.IGNORECASE)
    chosen = None
    if match:
        pred = match.group(1).strip()
        for c in top10_preds:
            if pred.lower() == c.lower():
                chosen = c
                break
    if chosen is None or chosen == wrong_top1:
        chosen = top10_preds[1] if len(top10_preds) > 1 else top10_preds[0]
    return chosen

# ----------------------------
# LOAD CSV DATA
# ----------------------------
df_domain = pd.read_csv(DOMAIN_CSV, on_bad_lines="skip")
df_duel = pd.read_csv(DUEL_CSV, on_bad_lines="skip")

for df in [df_domain, df_duel]:
    if "description" in df.columns and "text" not in df.columns:
        df.rename(columns={"description": "text"}, inplace=True)
    if "top10_preds" in df.columns:
        df["top10_preds"] = df["top10_preds"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# ----------------------------
# METRICS CALCUL
# ----------------------------
def compute_top1_metrics(df, pred_col):
    y_true = df["true_label"]
    y_pred = df[pred_col]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def compute_top10_accuracy(df, top10_col):
    return df.apply(lambda row: row["true_label"] in row[top10_col][:10], axis=1).mean()

domain_metrics = compute_top1_metrics(df_domain, "top1_pred")
domain_metrics["top10_acc"] = compute_top10_accuracy(df_domain, "top10_preds")

duel_metrics = compute_top1_metrics(df_duel, "final_duel_pred")
duel_metrics["top10_acc"] = compute_top10_accuracy(df_duel, "top10_preds")

# ----------------------------
# PLOTLY BAR
# ----------------------------
def create_metrics_bar(domain_metrics, duel_metrics):
    categories = ["Accuracy Top-1", "Top-10 Accuracy", "F1 Macro"]
    domain_values = [domain_metrics["accuracy"], domain_metrics["top10_acc"], domain_metrics["f1"]]
    duel_values = [duel_metrics["accuracy"], duel_metrics["top10_acc"], duel_metrics["f1"]]

    fig = go.Figure(data=[
        go.Bar(name="Domain Expert", x=categories, y=domain_values, marker_color="#1f77b4"),
        go.Bar(name="Duel Expert", x=categories, y=duel_values, marker_color="#2ca02c")
    ])
    fig.update_layout(
        title="📊 Comparaison Domain vs Duel Expert",
        yaxis=dict(title="Valeur métrique"),
        barmode="group",
        template="plotly_white",
        height=400
    )
    return pio.to_html(fig, full_html=False)

plot_metrics_html = create_metrics_bar(domain_metrics, duel_metrics)

# ----------------------------
# ROUTE FLASK
# ----------------------------
# ROUTE FLASK
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None

    if request.method == "POST":
        new_product = request.form.get("product_text")
        if new_product:
            # Tokenize & predict Domain Expert
            enc = tokenizer(new_product, truncation=True, padding="max_length",
                            max_length=MAX_LEN, return_tensors="pt")
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                top1_idx = torch.argmax(logits, dim=-1).item()
                top1_label = le.inverse_transform([top1_idx])[0]

            # Obtenir top10 à partir du CSV (exemple)
            top10_candidates = df_domain["top10_preds"].iloc[0][:10] if "top10_preds" in df_domain.columns else [top1_label]

            # ----------------------------
            # NE CHARGE LE LLM QUE SI TOP-1 FAUX
            # ----------------------------
            # Vérification : si top1_label n'est pas correct selon top10 ou autre règle
            is_top1_wrong = top1_label not in top10_candidates  # ou une autre logique selon ton dataset

            if is_top1_wrong:
                # Charger et utiliser le LLM seulement ici
                final_label = llm_correct(new_product, top10_candidates, top1_label)
            else:
                final_label = top1_label  # Top-1 correct → pas de LLM

            prediction_result = {
                "top1": top1_label,
                "final": final_label
            }

    example_products = df_duel.head(10).to_dict(orient="records")

    return render_template(
        "domain_expert.html",
        background_image=BACKGROUND_IMAGE,
        domain_metrics=domain_metrics,
        duel_metrics=duel_metrics,
        products=example_products,
        plot_html=plot_metrics_html,
        prediction_result=prediction_result
    )


if __name__ == "__main__":
    app.run(debug=True, port=8090)
