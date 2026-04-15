"""Medical report generation app for brain stroke CT classification."""

from datetime import datetime
from pathlib import Path
from io import BytesIO

import albumentations as A
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from torchvision import models

# Paths
ROOT = Path(__file__).parent
CKPT_CLF = ROOT / "checkpoints" / "efficientnet_b0_best.pth"
DATASET_DIR = ROOT / "Brain_Stroke_CT_Dataset"

IMG_SIZE = 256
CLASS_NAMES = ["Bleeding", "Ischemia", "Normal"]
CLASS_COLORS = {"Bleeding": "#c0392b", "Ischemia": "#2980b9", "Normal": "#1e8449"}
THEME_NAVY = "#102a43"
THEME_CYAN = "#16c7d7"
DEVICE = torch.device("cpu")

_transform = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ]
)


def preprocess(pil_img: Image.Image) -> torch.Tensor:
    """PIL image to (1, 1, H, W) float tensor."""
    arr = np.array(pil_img.convert("L"))
    t = _transform(image=arr)["image"]
    return t.unsqueeze(0)


def build_classifier() -> nn.Module:
    """EfficientNet-B0 adapted to grayscale CT input."""
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    old = model.features[0][0]

    new_conv = nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv.weight = nn.Parameter(old.weight.mean(dim=1, keepdim=True))
    model.features[0][0] = new_conv

    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 3))
    return model


@st.cache_resource
def load_classifier():
    model = build_classifier()
    if CKPT_CLF.exists():
        model.load_state_dict(torch.load(CKPT_CLF, map_location=DEVICE, weights_only=True))
        model.eval()
        return model, True
    return model, False


@torch.no_grad()
def classify(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    return probs


def clinical_priority(predicted_class: str, confidence: float):
    """Return triage priority and recommendation text in French."""
    if predicted_class == "Bleeding":
        return "🔴 CRITIQUE", "L'hémorragie cérébrale est fortement suggérée. Examen d'urgence en neuroradiologie/neurologie requis."
    if predicted_class == "Ischemia":
        if confidence >= 0.7:
            return "🟠 ÉLEVÉE", "Accident vasculaire cérébral ischémique probable. Activation du protocole AVC recommandée."
        return "🟡 MODÉRÉE", "Suspicion d'ischémie. Confirmation urgente et corrélation clinique recommandées."
    if confidence >= 0.7:
        return "🟢 ROUTINIER", "Aucun pattern d'AVC fortement suggéré. Suivi clinique standard."
    return "🟡 MODÉRÉE", "Pattern normal incertain. Examen clinicien recommandé."


@st.cache_data
def list_test_samples(n_per_class: int = 40):
    samples = []
    for cls in CLASS_NAMES:
        png_dir = DATASET_DIR / cls / "PNG"
        if png_dir.exists():
            files = sorted(png_dir.iterdir())[:n_per_class]
            samples.extend((str(f), cls) for f in files)
    return samples


def generate_pdf_report(case_id, probs, pred_cls, confidence, priority, recommendation, img_path):
    """Generate professional PDF report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=12,
        alignment=1,
    )
    
    story.append(Paragraph("RAPPORT D'ANALYSE CT — AVC CÉRÉBRAL", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    info_data = [
        ["ID Cas :", case_id],
        ["Date/Heure :", datetime.now().strftime("%d/%m/%Y %H:%M")],
    ]
    info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("RÉSULTATS", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))
    
    findings_data = [
        ["Pattern prédominant :", pred_cls],
        ["Confiance du modèle :", f"{confidence * 100:.1f}%"],
        ["Priorité de triage :", priority],
    ]
    findings_table = Table(findings_data, colWidths=[2.5 * inch, 3.5 * inch])
    findings_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 9),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#e8e8e8")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("IMPRESSION CLINIQUE", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(recommendation, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("PROBABILITÉS PAR CLASSE", styles['Heading3']))
    prob_data = [["Classe", "Probabilité"]]
    for cls, p in zip(CLASS_NAMES, probs):
        prob_data.append([cls, f"{p * 100:.1f}%"])
    prob_table = Table(prob_data, colWidths=[2.5 * inch, 2 * inch])
    prob_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(prob_table)
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Rapport Médical — AVC CT", page_icon="🏥", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(140deg, #f4f9fc 0%, #eef5fb 100%);
    }}

    section[data-testid="stSidebar"] {{
        background: {THEME_NAVY};
    }}

    section[data-testid="stSidebar"] * {{
        color: #ffffff;
    }}

    .page-title {{
        color: {THEME_NAVY};
        font-weight: 800;
        border-left: 6px solid {THEME_CYAN};
        padding-left: 12px;
        margin-bottom: 6px;
    }}

    .panel-card {{
        background: #ffffff;
        border: 1px solid #d7e6f2;
        border-top: 4px solid {THEME_CYAN};
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 2px 6px rgba(16, 42, 67, 0.08);
    }}

    .priority-badge {{
        display: inline-block;
        background: {THEME_NAVY};
        color: #ffffff;
        border-left: 5px solid {THEME_CYAN};
        border-radius: 8px;
        padding: 6px 10px;
        font-weight: 700;
        margin-bottom: 8px;
    }}

    .stButton > button, .stDownloadButton > button {{
        background: {THEME_CYAN};
        color: #082032;
        border: 1px solid {THEME_NAVY};
        font-weight: 700;
        border-radius: 8px;
    }}

    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: #10b7c7;
        color: #04111f;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### Navigation")
    st.divider()
    
    page = st.radio(
        "Aller à",
        ["📋 Informations Patient", "🔍 Détection d'AVC", "📄 Rapport Médical"],
        label_visibility="collapsed"
    )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: PATIENT INFO
# ═══════════════════════════════════════════════════════════════════════════
if page == "📋 Informations Patient":
    st.markdown("<h1 class='page-title'>Informations Patient</h1>", unsafe_allow_html=True)
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("ID Patient", "P-2024-001")
        patient_name = st.text_input("Nom du patient", "Bennouna Mohamed")
        patient_age = st.number_input("Âge", min_value=0, max_value=120, value=55)
    
    with col2:
        patient_sex = st.selectbox("Sexe", ["Masculin", "Féminin"])
        exam_date = st.date_input("Date d'examen")
        institution = st.text_input("Institution", "Centre Hospitalier")
    
    st.divider()
    st.info("✓ Informations patient enregistrées.")
    
    st.session_state.patient_data = {
        "id": patient_id,
        "name": patient_name,
        "age": patient_age,
        "sex": patient_sex,
        "exam_date": exam_date,
        "institution": institution,
    }

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: DETECTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Détection d'AVC":
    st.markdown("<h1 class='page-title'>Détection d'AVC Cérébral</h1>", unsafe_allow_html=True)
    st.divider()
    
    model, clf_loaded = load_classifier()
    if not clf_loaded:
        st.error("❌ Point de contrôle de classification non trouvé.")
        st.stop()
    
    source = st.radio("Source de l'image", ["Télécharger une image", "Sélectionner du dataset"], horizontal=True)
    
    pil_img = None
    true_label = None
    case_id = "N/A"
    temp_img_path = None
    
    if source == "Télécharger une image":
        uploaded = st.file_uploader("Télécharger une image CT", type=["png", "jpg", "jpeg"])
        if uploaded:
            pil_img = Image.open(uploaded)
            case_id = Path(uploaded.name).stem
            temp_img_path = uploaded.name
    else:
        samples = list_test_samples(n_per_class=40)
        if samples:
            labels = [f"{Path(path).stem} [{cls}]" for path, cls in samples]
            choice = st.selectbox("Sélectionner un cas", labels)
            idx = labels.index(choice)
            path, true_label = samples[idx]
            pil_img = Image.open(path)
            case_id = Path(path).stem
            temp_img_path = path
    
    if pil_img is not None:
        tensor = preprocess(pil_img).to(DEVICE)
        probs = classify(model, tensor)
        
        pred_idx = int(probs.argmax())
        pred_cls = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        priority, recommendation = clinical_priority(pred_cls, confidence)
        
        col_img, col_analysis = st.columns([1, 1.5])
        
        with col_img:
            st.subheader("Image CT")
            st.image(pil_img.convert("L"), use_container_width=True, clamp=True)
            if true_label:
                st.caption(f"Référence : {true_label}")
        
        with col_analysis:
            st.subheader("Analyse IA")
            st.markdown(f"<div class='priority-badge'>{priority}</div>", unsafe_allow_html=True)
            st.markdown(f"**Pattern détecté :** {pred_cls}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Confiance", f"{confidence * 100:.1f}%")
            m2.metric("Prob. AVC", f"{(1 - probs[2]) * 100:.1f}%")
            m3.metric("Cas ID", case_id[:12])
            
            st.divider()
            st.success(f"**Recommandation :** {recommendation}")
            
            st.write("**Répartition des probabilités :**")
            st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
            for cls, p in zip(CLASS_NAMES, probs):
                st.write(f"- {cls}: {p * 100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        if true_label:
            if pred_cls == true_label:
                st.success(f"✓ Prédiction correcte ({true_label})")
            else:
                st.warning(f"⚠️ Écart détecté. Référence : {true_label}")
        
        st.session_state.case_data = {
            "case_id": case_id,
            "probs": probs,
            "pred_cls": pred_cls,
            "confidence": confidence,
            "priority": priority,
            "recommendation": recommendation,
            "img_path": temp_img_path,
        }

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: REPORT
# ═══════════════════════════════════════════════════════════════════════════
else:
    st.markdown("<h1 class='page-title'>Génération du Rapport Médical</h1>", unsafe_allow_html=True)
    st.divider()
    
    if "case_data" not in st.session_state:
        st.info("📌 Veuillez d'abord analyser une image à partir de l'onglet 'Détection d'AVC'.")
    else:
        data = st.session_state.case_data
        
        st.success(f"✓ Rapport professionnel généré : rapport_{data['case_id']}.pdf")
        st.divider()
        
        pdf_buffer = generate_pdf_report(
            data["case_id"],
            data["probs"],
            data["pred_cls"],
            data["confidence"],
            data["priority"],
            data["recommendation"],
            data["img_path"],
        )
        
        st.download_button(
            label="📥 Télécharger le rapport PDF",
            data=pdf_buffer,
            file_name=f"rapport_{data['case_id']}.pdf",
            mime="application/pdf",
        )
        
        st.divider()
        st.markdown("**Contenu du rapport :**")
        st.write(f"- **ID Cas :** {data['case_id']}")
        st.write(f"- **Pattern :** {data['pred_cls']}")
        st.write(f"- **Confiance :** {data['confidence'] * 100:.1f}%")
        st.write(f"- **Priorité :** {data['priority']}")
        st.write(f"- **Recommandation :** {data['recommendation']}")
