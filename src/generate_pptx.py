# ================================================================
# CARTIQ - 10 SLIDE PRESENTATION GENERATOR
# ================================================================
# Light mint theme · real charts from outputs/ · speaker notes
# ================================================================

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ================================================================
# 1. PATHS & COLORS
# ================================================================

BASE = "D:/ML/CartIQ"
OUT = os.path.join(BASE, "reports", "CartIQ_Presentation.pptx")
IMG = os.path.join(BASE, "outputs")

BG      = RGBColor(0xF6, 0xFB, 0xF8)   # mist mint
INK     = RGBColor(0x17, 0x33, 0x2C)   # dark green-grey
MUTED   = RGBColor(0x5B, 0x77, 0x70)   # softer grey
EMERALD = RGBColor(0x10, 0xB9, 0x81)   # primary
TEAL    = RGBColor(0x0D, 0x94, 0x88)   # deep teal
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
LEMON   = RGBColor(0xFE, 0xF6, 0xD8)   # tint-lemon
MINT    = RGBColor(0xDC, 0xF5, 0xEC)   # tint-mint
PEACH   = RGBColor(0xFF, 0xE9, 0xDA)
LILAC   = RGBColor(0xED, 0xE9, 0xFE)
SKY     = RGBColor(0xDF, 0xF1, 0xFD)

FONT = "Segoe UI"

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


# ================================================================
# 2. HELPERS
# ================================================================

def slide_bg(slide, color=BG):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, color, line=None, radius=None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(shape_type, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def add_text(slide, l, t, w, h, text, size=18, color=INK, bold=False,
             align=PP_ALIGN.LEFT, font=FONT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = align
    r = p.runs[0]
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    r.font.name = font
    return box


def add_bullets(slide, l, t, w, h, items, size=16, color=MUTED, gap=10):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.space_after = Pt(gap)
        p.level = 0
        if p.runs:
            r = p.runs[0]
        else:
            r = p.add_run()
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.name = FONT
        # bullet dot
        p.text = "•  " + item
    return box


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_image_fit(slide, path, l, t, max_w, max_h):
    """Insert image scaled to fit inside (max_w, max_h), centered."""
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    ratio = min(max_w / iw, max_h / ih)
    w = int(iw * ratio)
    h = int(ih * ratio)
    left = l + int((max_w - w) / 2)
    top  = t + int((max_h - h) / 2)
    # white card behind image
    pad = Inches(0.12)
    add_rect(slide, left - pad, top - pad, w + 2 * pad, h + 2 * pad,
             WHITE, line=RGBColor(0xE2, 0xEE, 0xE8), radius=True)
    slide.shapes.add_picture(path, left, top, w, h)


def header(slide, number, title, subtitle=None):
    """Consistent title top-left + subtle slide number."""
    add_rect(slide, Inches(0.55), Inches(0.42), Inches(0.14),
             Inches(0.52), EMERALD, radius=True)
    add_text(slide, Inches(0.85), Inches(0.34), Inches(9.5),
             Inches(0.7), title, size=30, color=INK, bold=True)
    if subtitle:
        add_text(slide, Inches(0.87), Inches(0.92), Inches(10),
                 Inches(0.4), subtitle, size=13, color=MUTED)
    add_text(slide, Inches(12.1), Inches(0.4), Inches(0.9),
             Inches(0.4), f"{number:02d}/10", size=11, color=RGBColor(0x8A, 0xA3, 0x9C),
             align=PP_ALIGN.RIGHT, bold=True)


def footer(slide):
    add_text(slide, Inches(0.55), Inches(7.05), Inches(6),
             Inches(0.3), "CartIQ  ·  Grocery Market Basket Intelligence",
             size=9, color=RGBColor(0x8A, 0xA3, 0x9C))


# ================================================================
# 3. SLIDE 1 - TITLE
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)

# big soft mint panel on right
add_rect(s, Inches(7.6), Inches(0), Inches(5.73), SH, MINT)
add_rect(s, Inches(8.3), Inches(1.6), Inches(4.2), Inches(4.2),
         RGBColor(0xC3, 0xEE, 0xDD), radius=True)

# basket emoji centerpiece
add_text(s, Inches(8.3), Inches(2.3), Inches(4.2), Inches(2.4),
         "\U0001F6FA", size=110, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)

add_text(s, Inches(0.9), Inches(1.5), Inches(6.4), Inches(0.5),
         "MACHINE LEARNING PROJECT REVIEW", size=13, color=EMERALD, bold=True)
add_text(s, Inches(0.9), Inches(2.05), Inches(6.6), Inches(1.6),
         "CartIQ", size=64, color=INK, bold=True)
add_text(s, Inches(0.9), Inches(3.35), Inches(6.6), Inches(0.9),
         "Grocery Market Basket Intelligence", size=26, color=TEAL, bold=True)
add_text(s, Inches(0.9), Inches(4.35), Inches(6.4), Inches(0.9),
         "Predicting product reorders from\n50,000 real shopping baskets",
         size=17, color=MUTED)

add_rect(s, Inches(0.9), Inches(5.6), Inches(2.6), Inches(0.06), EMERALD)
add_text(s, Inches(0.9), Inches(5.85), Inches(6.4), Inches(0.4),
         "Python  ·  Pandas  ·  scikit-learn  ·  Flask  ·  2026",
         size=13, color=MUTED)

add_notes(s, "Good morning madam. Our project is CartIQ - a grocery market "
             "basket intelligence system. We analyse 50,000 real Instacart "
             "shopping records to understand how people shop and to predict "
             "whether a product will be reordered. I'll walk you through our "
             "data, our fifteen models, and a live working web application.")


# ================================================================
# 4. SLIDE 2 - PROBLEM, DATASET & PIPELINE
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)
header(s, 2, "The problem & our data")

add_bullets(s, Inches(0.85), Inches(1.55), Inches(5.6), Inches(3.2), [
    "Predict product reorders from basket data",
    "50,000 rows · 4,799 customers · 11,650 products",
    "134 aisles · 21 departments",
    "59% of items are reorders",
], size=17, gap=16)

# pipeline chips
chips = [("M1", "Understand"), ("M2", "Clean & Encode"),
         ("M3", "Predict"), ("M4", "Cluster"), ("M5", "Validate")]
x = Inches(0.85)
for code, label in chips:
    add_rect(s, x, Inches(4.9), Inches(1.05), Inches(1.05), MINT, radius=True)
    add_text(s, x, Inches(5.0), Inches(1.05), Inches(0.4), code,
             size=15, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(s, x - Inches(0.15), Inches(5.42), Inches(1.35), Inches(0.4),
             label, size=9, color=MUTED, align=PP_ALIGN.CENTER)
    x += Inches(1.18)

add_image_fit(s, os.path.join(IMG, "EDA_Analysis_outputs", "Target_Distribution.png"),
              Inches(6.9), Inches(1.35), Inches(5.9), Inches(5.2))
footer(s)

add_notes(s, "Our goal is to predict whether a product in a shopping basket "
             "will be reordered. We used the Kaggle Instacart dataset - "
             "50,000 rows, 14 columns, nearly 4,800 customers and over 11,000 "
             "products across 134 aisles and 21 departments. About 59 percent "
             "of basket lines are reorders, so the classes are well balanced. "
             "The whole project runs as five modules - understand, clean and "
             "encode, predict, cluster, and validate - with the raw data never "
             "modified.")


# ================================================================
# 5. SLIDE 3 - PREPROCESSING & EXPLORATION
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)
header(s, 3, "Cleaning & exploring the data")

add_bullets(s, Inches(0.85), Inches(1.55), Inches(5.6), Inches(3.4), [
    "4 missing-value strategies compared",
    "Standard, Min-Max & L2 scaling",
    "5 encoding techniques applied",
    "Correlation heatmap revealed feature links",
], size=17, gap=16)

add_rect(s, Inches(0.85), Inches(4.85), Inches(5.5), Inches(1.5),
         LEMON, radius=True)
add_text(s, Inches(1.1), Inches(5.0), Inches(5.1), Inches(0.4),
         "Raw data is never modified", size=14, color=INK, bold=True)
add_text(s, Inches(1.1), Inches(5.42), Inches(5.1), Inches(0.8),
         "All cleaning works on copies → one clean all-numeric final dataset",
         size=12, color=MUTED)

add_image_fit(s, os.path.join(IMG, "EDA_Analysis_outputs", "Correlation_Heatmap.png"),
              Inches(6.9), Inches(1.35), Inches(5.9), Inches(5.2))
footer(s)

add_notes(s, "Before modelling we explored and cleaned the data. We compared "
             "four missing-value strategies - deletion, mean, median and "
             "model-based imputation. Only one column, days_since_prior_order, "
             "had nulls. We applied three scaling methods and five encoding "
             "techniques - label, one-hot, ordinal, target and embedding - "
             "producing one clean all-numeric final dataset. The correlation "
             "heatmap showed how features move together and confirmed no "
             "single feature dominates the target.")


# ================================================================
# 6. SLIDE 4 - PREDICTION MODELS
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)
header(s, 4, "Models that predict reorders",
       "Tree-based ensembles beat linear models — shopping behaviour is non-linear")

models = [
    ("Linear Regression",  "R² 0.011 — weak signal",      SKY,   "01"),
    ("Decision Tree",      "Acc 72.2%  ·  F1 78.4%",      MINT,  "02"),
    ("Random Forest ★",    "Acc 73.0%  ·  F1 80.0%",      MINT,  "03"),
    ("AdaBoost",           "Acc 73.3%  ·  F1 79.2%",      PEACH, "04"),
]
y = Inches(1.65)
for name, metric, tint, num in models:
    add_rect(s, Inches(0.85), y, Inches(5.7), Inches(1.05), tint, radius=True)
    add_text(s, Inches(1.1), y + Inches(0.14), Inches(4.0), Inches(0.4),
             name, size=15, color=INK, bold=True)
    add_text(s, Inches(1.1), y + Inches(0.55), Inches(4.2), Inches(0.4),
             metric, size=12, color=MUTED)
    add_text(s, Inches(5.85), y + Inches(0.25), Inches(0.55), Inches(0.5),
             num, size=16, color=EMERALD, bold=True, align=PP_ALIGN.RIGHT)
    y += Inches(1.18)

add_image_fit(s, os.path.join(IMG, "Random_Forest_Tree_M3_Outputs",
                              "confusion_matrix", "confusion_matrix.png"),
              Inches(7.0), Inches(1.55), Inches(5.8), Inches(5.1))
footer(s)

add_notes(s, "We started with linear regression to predict basket size - the "
             "R-squared was only 0.011, confirming the signal is non-linear. "
             "We then built three tree-based classifiers, each inside a "
             "complete sklearn pipeline with median imputation and one-hot "
             "encoding. The plain Decision Tree gave 72.2 percent accuracy. "
             "Random Forest achieved the best F1 score of 80 percent with "
             "high recall, and AdaBoost gave the best overall accuracy at "
             "73.3 percent. The tree ensembles clearly outperform linear "
             "models, which tells us shopping behaviour is non-linear.")


# ================================================================
# 7. SLIDE 5 - CLUSTERING & DIMENSIONALITY REDUCTION
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)
header(s, 5, "Clustering & reducing dimensions",
       "Grouping shoppers and compressing 14 features into visual maps")

add_bullets(s, Inches(0.85), Inches(1.7), Inches(4.6), Inches(2.8), [
    "K-Means: best K = 3, silhouette 0.32",
    "t-SNE: 14 features mapped to 2-D",
    "PCA: 13 components keep 95% variance",
    "DBSCAN: 7 clusters · 0% noise",
    "Hierarchical: divisive beats agglomerative",
], size=15, gap=12)

add_image_fit(s, os.path.join(IMG, "K_Means_Elbow_Silhouetee_M4_Outputs",
                              "elbow", "elbow_kmeans.png"),
              Inches(5.8), Inches(1.5), Inches(3.45), Inches(3.1))
add_image_fit(s, os.path.join(IMG, "t-SNE_M4_outputs", "tsne_scatter_plot.png"),
              Inches(9.35), Inches(1.5), Inches(3.45), Inches(3.1))

add_rect(s, Inches(5.8), Inches(4.85), Inches(7.0), Inches(1.55),
         LILAC, radius=True)
add_text(s, Inches(6.1), Inches(5.0), Inches(6.4), Inches(0.4),
         "Information is spread, not concentrated", size=14, color=INK, bold=True)
add_text(s, Inches(6.1), Inches(5.42), Inches(6.4), Inches(0.85),
         "PCA needs 13 of 14 components for 95% variance — reorders depend on "
         "many weak signals combined, not one strong feature.",
         size=12, color=MUTED)
footer(s)

add_notes(s, "Module 4 groups shoppers into clusters. Using the elbow curve "
             "and silhouette score, K-Means found the best grouping at K "
             "equals 3 with a silhouette of 0.32. We then reduced the 14 "
             "features to 2 dimensions - t-SNE produced a clear visual map, "
             "and PCA showed we need 13 of 14 components to keep 95 percent "
             "of the variance, confirming information is spread across many "
             "features. We also ran DBSCAN, which found 7 dense clusters with "
             "zero noise, and compared hierarchical methods where our "
             "divisive approach beat standard agglomerative clustering.")


# ================================================================
# 8. SLIDE 6 - CROSS-VALIDATION
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)
header(s, 6, "Validating the results",
       "5-fold stratified cross-validation — numbers you can trust")

add_bullets(s, Inches(0.85), Inches(1.7), Inches(4.6), Inches(2.6), [
    "5-fold stratified cross-validation",
    "Out-of-fold F1: 69.1%",
    "ROC-AUC 0.681  ·  PR-AUC 0.700",
    "Stable — outlier impact only 0.014",
], size=16, gap=14)

add_rect(s, Inches(0.85), Inches(4.6), Inches(4.6), Inches(1.8),
         MINT, radius=True)
add_text(s, Inches(1.1), Inches(4.78), Inches(4.1), Inches(0.4),
         "Why cross-validation?", size=14, color=TEAL, bold=True)
add_text(s, Inches(1.1), Inches(5.2), Inches(4.1), Inches(1.0),
         "One train-test split can be lucky or unlucky. Five folds average "
         "out the noise so the reported score is reliable.",
         size=12, color=MUTED)

add_image_fit(s, os.path.join(IMG, "CV_DT_All_Classifi_Regres_Metrics_M5_outputs",
                              "ROC_AUC_Curve.png"),
              Inches(5.9), Inches(1.5), Inches(3.4), Inches(3.1))
add_image_fit(s, os.path.join(IMG, "CV_DT_All_Classifi_Regres_Metrics_M5_outputs",
                              "PR_AUC_Curve.png"),
              Inches(9.4), Inches(1.5), Inches(3.4), Inches(3.1))

add_text(s, Inches(5.9), Inches(4.8), Inches(6.9), Inches(1.4),
         "The ROC and Precision-Recall curves show the model keeps useful "
         "discriminative power across thresholds — not just at one cutoff.",
         size=13, color=MUTED)
footer(s)

add_notes(s, "Module 5 validates our decision tree with 5-fold stratified "
             "cross-validation instead of a single train-test split, so the "
             "numbers are more trustworthy. The out-of-fold F1 was 69.1 "
             "percent, ROC-AUC 0.681 and PR-AUC 0.700. We also tested how "
             "sensitive the model is to outliers - the F1 difference between "
             "outlier and normal cases was only 0.014, so the model is "
             "stable and reliable.")


# ================================================================
# 9. SLIDE 7 - WEB APPLICATION
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)
header(s, 7, "A working web application",
       "Flask · 10 pages · live charts served from the outputs folder")

add_bullets(s, Inches(0.85), Inches(1.75), Inches(4.4), Inches(2.6), [
    "Flask website with 10 pages",
    "Live charts from outputs folder",
    "Auto-generated reports per module",
    "Light, modern interface",
], size=16, gap=14)

add_rect(s, Inches(0.85), Inches(4.55), Inches(4.4), Inches(1.9),
         SKY, radius=True)
add_text(s, Inches(1.1), Inches(4.72), Inches(3.9), Inches(0.4),
         "Pages included", size=13, color=INK, bold=True)
add_text(s, Inches(1.1), Inches(5.12), Inches(3.9), Inches(1.2),
         "Home · About · Dataset · Preprocessing · Visualization · "
         "Models · Prediction · Reports",
         size=11, color=MUTED)

add_image_fit(s, os.path.join(IMG, "ui_previews", "home.png"),
              Inches(5.7), Inches(1.4), Inches(7.1), Inches(5.4))
footer(s)

add_notes(s, "We wrapped the entire project into a Flask web application with "
             "eight pages - home, dataset, preprocessing, visualization, "
             "models, prediction and reports. Every "
             "chart you've seen is served live from the outputs folder, and "
             "the reports page automatically lists every file each module "
             "generated. The interface uses a clean light design so results "
             "are easy to read.")


# ================================================================
# 10. SLIDE 8 - LIVE PREDICTION DEMO
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)
header(s, 8, "Predict a reorder in real time",
       "The form loads the actual trained pipeline and answers instantly")

add_bullets(s, Inches(0.85), Inches(1.75), Inches(4.2), Inches(2.4), [
    "Choose model: RF, AdaBoost or Tree",
    "Enter 9 order features",
    "Get prediction + probability",
    "Runs the real trained pipeline",
], size=15, gap=13)

add_rect(s, Inches(0.85), Inches(4.35), Inches(4.2), Inches(2.1),
         PEACH, radius=True)
add_text(s, Inches(1.1), Inches(4.52), Inches(3.7), Inches(0.4),
         "\U0001F534  Live demo now", size=14, color=INK, bold=True)
add_text(s, Inches(1.1), Inches(4.95), Inches(3.7), Inches(1.3),
         "I'll open the website and run a real prediction in front of you — "
         "Random Forest, fresh fruits, produce department.",
         size=12, color=MUTED)

add_image_fit(s, os.path.join(IMG, "ui_previews", "prediction.png"),
              Inches(5.5), Inches(1.4), Inches(7.3), Inches(5.4))
footer(s)

add_notes(s, "This is the live prediction page. You pick a model - Random "
             "Forest, AdaBoost or Decision Tree - enter nine order features "
             "such as order number, hour of day, aisle and department, and "
             "the app loads the actual trained pipeline and predicts "
             "instantly. I can demonstrate this live right now. "
             "[DEMO TRIGGER: switch to the browser, open "
             "http://127.0.0.1:5000/prediction, submit the form with default "
             "values, and show the result card with the probability bar.]")

# ================================================================
# 11. SLIDE 9 - RESULTS SUMMARY
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)
header(s, 9, "All 15 models at a glance",
       "Random Forest is the champion — best F1 at 80.0%")

rows = [
    ("Linear Regression",       "MAE 0.75  ·  R² 0.011",           SKY),
    ("Ridge / Lasso / Elastic Net", "Accuracy ≈ 69–70%",            SKY),
    ("Logistic Regression",     "Accuracy 69.07%",                  SKY),
    ("Multinomial LogReg",      "Accuracy 34% (21 classes)",        SKY),
    ("Decision Tree",           "Acc 72.2%  ·  F1 78.4%",           MINT),
    ("Random Forest ★",         "Acc 73.0%  ·  F1 80.0%",           MINT),
    ("AdaBoost",                "Acc 73.3%  ·  F1 79.2%",           MINT),
    ("K-Means / K-Means++",     "K = 3  ·  Silhouette 0.32",        LILAC),
    ("t-SNE / PCA / DBSCAN",    "7 clusters · 13 PCs for 95%",      LILAC),
    ("Hierarchical + CV Tree",  "Divisive 0.304 · CV F1 69.1%",     LILAC),
]

y = Inches(1.5)
row_h = Inches(0.5)
# header row
add_rect(s, Inches(0.85), y, Inches(11.6), row_h, EMERALD, radius=True)
add_text(s, Inches(1.1), y + Inches(0.08), Inches(4.5), Inches(0.35),
         "Model", size=13, color=WHITE, bold=True)
add_text(s, Inches(6.0), y + Inches(0.08), Inches(6.0), Inches(0.35),
         "Key Result", size=13, color=WHITE, bold=True)
y += row_h + Inches(0.06)

for name, result, tint in rows:
    add_rect(s, Inches(0.85), y, Inches(11.6), row_h, tint)
    add_text(s, Inches(1.1), y + Inches(0.08), Inches(4.7), Inches(0.35),
             name, size=12, color=INK, bold=("★" in name))
    add_text(s, Inches(6.0), y + Inches(0.08), Inches(6.2), Inches(0.35),
             result, size=12, color=MUTED)
    y += row_h + Inches(0.04)

footer(s)

add_notes(s, "This table summarises all fifteen models. For classification "
             "the tree ensembles clearly win - Random Forest and AdaBoost "
             "both around 73 percent accuracy with F1 near 80. Linear models "
             "underperformed, confirming basket behaviour is non-linear. "
             "Clustering and dimensionality reduction gave us interpretable "
             "views of shopper segments. Random Forest is our champion model "
             "with the best F1 score.")


# ================================================================
# 12. SLIDE 10 - THANK YOU
# ================================================================

s = prs.slides.add_slide(BLANK)
slide_bg(s)

add_rect(s, Inches(0), Inches(0), SW, Inches(0.14), EMERALD)

add_text(s, Inches(0.9), Inches(1.5), Inches(11.5), Inches(1.2),
         "Thank you", size=54, color=INK, bold=True, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.9), Inches(2.7), Inches(11.5), Inches(0.5),
         "Questions welcome", size=22, color=TEAL, align=PP_ALIGN.CENTER)

# future scope chips
future = [
    ("\U0001F916", "Deep learning on order sequences"),
    ("\U0001F310", "Real-time grocery API integration"),
    ("\u2728", "Personalised recommendation engine"),
]
x = Inches(1.6)
for icon, text in future:
    add_rect(s, x, Inches(3.7), Inches(3.3), Inches(1.35), MINT, radius=True)
    add_text(s, x, Inches(3.85), Inches(3.3), Inches(0.45), icon,
             size=22, align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.15), Inches(4.35), Inches(3.0), Inches(0.6),
             text, size=12, color=MUTED, align=PP_ALIGN.CENTER)
    x += Inches(3.55)

add_text(s, Inches(0.9), Inches(5.6), Inches(11.5), Inches(0.4),
         "Future scope", size=12, color=EMERALD, bold=True,
         align=PP_ALIGN.CENTER)

add_rect(s, Inches(5.4), Inches(6.25), Inches(2.5), Inches(0.05), EMERALD)
add_text(s, Inches(0.9), Inches(6.45), Inches(11.5), Inches(0.4),
         "CartIQ  ·  Python  ·  scikit-learn  ·  Flask  ·  50,000 baskets",
         size=13, color=MUTED, align=PP_ALIGN.CENTER)

add_notes(s, "Thank you madam. In short, we built a complete machine "
             "learning pipeline - from raw data exploration to fifteen "
             "trained models to a live web application - and found that "
             "tree-based ensembles best predict product reorders. For "
             "future scope we would add deep learning on order sequences, "
             "integrate a real-time grocery API, and build a personalised "
             "recommendation engine on top of the shopper clusters we "
             "discovered. We're happy to take any questions, and I can run "
             "the live demo again.")


# ================================================================
# 13. SAVE
# ================================================================

prs.save(OUT)
print("Presentation saved at:")
print(OUT)
print("Slides:", len(prs.slides.__iter__.__self__._sldIdLst))
