# app.py
from flask import Flask, render_template, request
import pandas as pd
import ast, os, gc, time, re, requests
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objs as go
import plotly.io as pio

# ----------------------------
# CONFIG FLASK & PATHS
# ----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_IMAGE = "/static/pexels-karola-g-5632382.jpg"

DOMAIN_CSV = os.path.join(BASE_DIR, "data/domain_expert_phase2_top10_predictions_full.csv")
DUEL_CSV = os.path.join(BASE_DIR, "data/duel_expert_mistral_GENERAL_EXPERT_top10_corrected.csv")
GLOBAL_CSV = os.path.join(BASE_DIR, "data/produits_nettoyes.csv")
PHASE2_CKPT = os.path.join(BASE_DIR, "models/domain_expert_flat.pth")

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

# ----------------------------
# LOAD DOMAIN EXPERT MODEL (CPU/paresseux possible)
# ----------------------------
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
df_global = pd.read_csv(GLOBAL_CSV, sep=';', on_bad_lines='skip')
df_global['taxonomy_path'] = df_global['taxonomy_path'].astype(str)
le = LabelEncoder()
le.fit(df_global['taxonomy_path'].tolist())
NUM_CLASSES = len(le.classes_)

model = DomainExpert(NUM_CLASSES).to(DEVICE)
if os.path.exists(PHASE2_CKPT):
    ckpt = torch.load(PHASE2_CKPT, map_location=DEVICE)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    del ckpt, sd; gc.collect(); torch.cuda.empty_cache()
model.eval()

# ----------------------------
# LLM via API externe (HuggingFace)
# ----------------------------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def llm_correct_api(text, top10_preds, wrong_top1):
    prompt = f"""
You are a GENERAL EXPERT correcting a WRONG product classification.
The current top-1 prediction is WRONG and must NOT be selected again.

Product:
{text}

Candidate categories:
{', '.join(top10_preds)}

Rules:
- Choose exactly ONE category
- It must be DIFFERENT from the wrong top-1
- Choose only from the list
- No explanations

Final choice:
"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 40, "temperature": 0.3}}
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            out_text = response.json()[0]["generated_text"]
            for c in top10_preds:
                if c.lower() in out_text.lower() and c != wrong_top1:
                    return c
        return top10_preds[1] if len(top10_preds) > 1 else top10_preds[0]
    except Exception:
        return top10_preds[1] if len(top10_preds) > 1 else top10_preds[0]

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
# METRICS
# ----------------------------
def compute_top1_metrics(df, pred_col):
    y_true = df["true_label"]
    y_pred = df[pred_col]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }

def compute_top10_accuracy(df, top10_col):
    return df.apply(lambda row: row["true_label"] in row[top10_col][:10], axis=1).mean()

domain_metrics = compute_top1_metrics(df_domain, "top1_pred")
domain_metrics["top10_acc"] = compute_top10_accuracy(df_domain, "top10_preds")
duel_metrics = compute_top1_metrics(df_duel, "final_duel_pred")
duel_metrics["top10_acc"] = compute_top10_accuracy(df_duel, "top10_preds")

# ----------------------------
# PLOTLY METRICS BAR
# ----------------------------
def create_metrics_bar(domain_metrics, duel_metrics):
    categories = ["Accuracy Top-1", "Top-10 Accuracy", "F1 Macro"]
    fig = go.Figure(data=[
        go.Bar(name="Domain Expert", x=categories, y=[domain_metrics["accuracy"], domain_metrics["top10_acc"], domain_metrics["f1"]], marker_color="#1f77b4"),
        go.Bar(name="Duel Expert", x=categories, y=[duel_metrics["accuracy"], duel_metrics["top10_acc"], duel_metrics["f1"]], marker_color="#2ca02c")
    ])
    fig.update_layout(title="📊 Comparaison Domain vs Duel Expert", yaxis=dict(title="Valeur métrique"), barmode="group", template="plotly_white", height=400)
    return pio.to_html(fig, full_html=False)

plot_metrics_html = create_metrics_bar(domain_metrics, duel_metrics)

# ----------------------------
# FLASK ROUTE
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_results = []
    llm_msg = ""
    elapsed_time = None
    start_time = time.time()

    uploaded_file = request.files.get("csv_file")
    products = []

    # CSV uploadé
    if uploaded_file:
        try:
            df_new = pd.read_csv(uploaded_file, sep=";", on_bad_lines="skip")
            if "description" in df_new.columns and "text" not in df_new.columns:
                df_new.rename(columns={"description": "text"}, inplace=True)
            products = df_new["text"].tolist()
        except Exception as e:
            llm_msg = f"Erreur lecture CSV : {str(e)}"
    else:
        new_product = request.form.get("product_text")
        if new_product:
            products = [new_product]

    # Boucle produits
    for prod in products:
        enc = tokenizer(prod, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            top1_idx = torch.argmax(logits, dim=-1).item()
            top1_label = le.inverse_transform([top1_idx])[0]

        top10_candidates = df_domain["top10_preds"].iloc[0][:10] if "top10_preds" in df_domain.columns else [top1_label]

        if top1_label not in top10_candidates:
            final_label = llm_correct_api(prod, top10_candidates, top1_label)
            llm_msg = "LLM utilisé via API externe"
        else:
            final_label = top1_label
            llm_msg = "Top-1 correct, LLM non utilisé"

        prediction_results.append({
            "text": prod,
            "top1": top1_label,
            "final": final_label,
            "top10": top10_candidates
        })

    elapsed_time = round(time.time() - start_time, 2)
    example_products = df_duel.head(10).to_dict(orient="records")

    return render_template(
        "domain_expert.html",
        background_image=BACKGROUND_IMAGE,
        domain_metrics=domain_metrics,
        duel_metrics=duel_metrics,
        products=example_products,
        plot_html=plot_metrics_html,
        prediction_results=prediction_results,
        llm_msg=llm_msg,
        elapsed_time=elapsed_time
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
