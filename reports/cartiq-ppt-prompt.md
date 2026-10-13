# CartIQ — Presentation Generator Prompt (10 Slides)

Copy everything inside the box below into any AI presentation tool
(Gamma, ChatGPT, Tome, Copilot, Canva AI) or hand it to a designer.

---

```
Create a complete academic project presentation (exactly 10 slides) for a
final-year Machine Learning project review. Follow ALL instructions precisely.

=========================================================
PROJECT
=========================================================
Title: CartIQ — Grocery Market Basket Intelligence
Domain: Machine Learning / Data Science
Dataset: Kaggle Instacart Market Basket Analysis, 50,000 rows × 14 columns
Tech stack: Python, Pandas, scikit-learn, Matplotlib/Seaborn, Flask

=========================================================
DESIGN RULES (CRITICAL)
=========================================================
1. LIGHT THEME ONLY. Background near-white / very light mint (#F6FBF8).
   Primary accent: emerald green (#10B981) and teal (#0D9488).
   Secondary accents: soft peach (#FFB38A), lilac (#A78BFA), sky blue (#38BDF8).
   Text: dark green-grey (#17332C) for headings, softer grey (#5B7770) for body.
2. MINIMAL TEXT. Maximum 4-5 short bullets per slide, each 6-8 words.
   No full sentences on slides. All explanation goes into SPEAKER NOTES.
3. VISUALS FIRST. Every slide needs one large clear image/chart area.
   Text sits beside or below the image, never on top of it.
4. Professional, clean, modern academic look. Rounded corners, soft shadows,
   generous white space. No clip-art, no cartoon icons, no dark slides.
5. Font: clean sans-serif (Poppins, Inter, or Plus Jakarta Sans).
   Titles 32-40pt, body 16-18pt.

=========================================================
IMAGE / VISUAL ASSETS
=========================================================
Where a real file cannot be inserted, place a clearly-labeled placeholder
with this caption so I can drop the file in later:

- [APP-HOME]   : Flask website homepage, light mint theme, hero "Smarter baskets"
- [APP-PRED]   : prediction form page with dropdowns and inputs
- [APP-RESULT] : prediction result card "Will be reordered 73.5%"
- [CH-TARGET]  : bar chart, 59% reorder vs 41% first-time
- [CH-CORR]    : correlation heatmap of 14 features
- [CH-CM-RF]   : Random Forest confusion matrix heatmap (blue)
- [CH-ELBOW]   : K-Means elbow curve, best K=3
- [CH-TSNE]    : t-SNE 2-D scatter, colorful clusters
- [CH-ROC]     : ROC curve, AUC 0.681
- [CH-PR]      : precision-recall curve, AUC 0.700

=========================================================
SLIDE-BY-SLIDE STRUCTURE (10 SLIDES)
=========================================================

--- SLIDE 1: TITLE ---
Title: CartIQ — Grocery Market Basket Intelligence
Subtitle: Predicting product reorders from 50,000 real shopping baskets
Footer: Machine Learning Project Review · 2026
Visual: soft abstract grocery/basket illustration in light green tones
Speaker notes: "Good morning madam. Our project is CartIQ — a grocery market
basket intelligence system. We analyse 50,000 real Instacart shopping
records to understand how people shop and to predict whether a product will
be reordered. I'll walk you through our data, our fifteen models, and a
live working web application."

--- SLIDE 2: PROBLEM, DATASET & PIPELINE ---
Title: The problem & our data
Bullets:
  - Predict product reorders from basket data
  - 50,000 rows · 4,799 customers · 11,650 products
  - 134 aisles · 21 departments
  - 59% of items are reorders
Visual: [CH-TARGET] reorder-balance chart on the right half
Speaker notes: "Our goal is to predict whether a product in a shopping
basket will be reordered. We used the Kaggle Instacart dataset — 50,000
rows, 14 columns, nearly 4,800 customers and over 11,000 products across
134 aisles and 21 departments. About 59 percent of basket lines are
reorders, so the classes are well balanced. The whole project runs as five
modules — understand, clean and encode, predict, cluster, and validate —
with the raw data never modified."

--- SLIDE 3: PREPROCESSING & EXPLORATION ---
Title: Cleaning & exploring the data
Bullets:
  - 4 missing-value strategies compared
  - Standard, Min-Max & L2 scaling
  - 5 encoding techniques applied
  - Correlation heatmap revealed feature links
Visual: [CH-CORR] correlation heatmap large
Speaker notes: "Before modelling we explored and cleaned the data. We
compared four missing-value strategies — deletion, mean, median and
model-based imputation. Only one column, days_since_prior_order, had
nulls. We applied three scaling methods and five encoding techniques —
label, one-hot, ordinal, target and embedding — producing one clean
all-numeric final dataset. The correlation heatmap showed how features
move together and confirmed no single feature dominates the target."

--- SLIDE 4: PREDICTION MODELS (M2 + M3) ---
Title: Models that predict reorders
Bullets:
  - Linear models: R² 0.011 — weak signal
  - Decision Tree: Acc 72.2% · F1 78.4%
  - Random Forest: Acc 73.0% · F1 80.0% ← best F1
  - AdaBoost: Acc 73.3% — best accuracy
Visual: small comparison bar chart of the 3 tree models OR result table
Speaker notes: "We started with linear regression to predict basket size —
the R-squared was only 0.011, confirming the signal is non-linear. We then
built three tree-based classifiers, each inside a complete sklearn pipeline
with median imputation and one-hot encoding. The plain Decision Tree gave
72.2 percent accuracy. Random Forest achieved the best F1 score of 80
percent with high recall, and AdaBoost gave the best overall accuracy at
73.3 percent. The tree ensembles clearly outperform linear models, which
tells us shopping behaviour is non-linear."

--- SLIDE 5: CLUSTERING & DIMENSIONALITY REDUCTION (M4) ---
Title: Clustering & reducing dimensions
Bullets:
  - K-Means: best K = 3, silhouette 0.32
  - t-SNE: 14 features mapped to 2-D
  - PCA: 13 components keep 95% variance
  - DBSCAN: 7 clusters · Hierarchical: divisive wins
Visual: [CH-ELBOW] elbow curve + [CH-TSNE] scatter, side by side
Speaker notes: "Module 4 groups shoppers into clusters. Using the elbow
curve and silhouette score, K-Means found the best grouping at K equals 3
with a silhouette of 0.32. We then reduced the 14 features to 2 dimensions
— t-SNE produced a clear visual map, and PCA showed we need 13 of 14
components to keep 95 percent of the variance, confirming information is
spread across many features. We also ran DBSCAN, which found 7 dense
clusters with zero noise, and compared hierarchical methods where our
divisive approach beat standard agglomerative clustering."

--- SLIDE 6: CROSS-VALIDATION (M5) ---
Title: Validating the results
Bullets:
  - 5-fold stratified cross-validation
  - Out-of-fold F1: 69.1%
  - ROC-AUC 0.681 · PR-AUC 0.700
  - Stable — outlier impact only 0.014
Visual: [CH-ROC] ROC curve + [CH-PR] precision-recall curve, side by side
Speaker notes: "Module 5 validates our decision tree with 5-fold stratified
cross-validation instead of a single train-test split, so the numbers are
more trustworthy. The out-of-fold F1 was 69.1 percent, ROC-AUC 0.681 and
PR-AUC 0.700. We also tested how sensitive the model is to outliers — the
F1 difference between outlier and normal cases was only 0.014, so the
model is stable and reliable."

--- SLIDE 7: WEB APPLICATION ---
Title: A working web application
Bullets:
  - Flask website with 10 pages
  - Live charts served from outputs folder
  - Auto-generated reports per module
  - Light, modern interface
Visual: [APP-HOME] website screenshot large
Speaker notes: "We wrapped the entire project into a Flask web application
with eight pages — home, dataset, preprocessing, visualization, models,
prediction and reports. Every chart you've seen is
served live from the outputs folder, and the reports page automatically
lists every file each module generated. The interface uses a clean light
design so results are easy to read."

--- SLIDE 8: LIVE PREDICTION DEMO ---
Title: Predict a reorder in real time
Bullets:
  - Choose model: RF, AdaBoost or Tree
  - Enter 9 order features
  - Get prediction + probability instantly
  - Runs the real trained pipeline
Visual: [APP-PRED] form on left, [APP-RESULT] result card on right
Speaker notes: "This is the live prediction page. You pick a model — Random
Forest, AdaBoost or Decision Tree — enter nine order features such as
order number, hour of day, aisle and department, and the app loads the
actual trained pipeline and predicts instantly. For this example it
predicted a reorder with 73.5 percent probability. I can demonstrate this
live right now if you'd like."
[DEMO TRIGGER: run the live prediction in the Flask app here]

--- SLIDE 9: RESULTS SUMMARY ---
Title: All 15 models at a glance
Compact table (Model | Key Result):
  Linear Regression — MAE 0.75 · R² 0.011
  Ridge / Lasso / Elastic Net — Acc ≈ 69-70%
  Logistic Regression — Acc 69.07%
  Decision Tree — Acc 72.2% · F1 78.4%
  Random Forest — Acc 73.0% · F1 80.0% ★
  AdaBoost — Acc 73.3% · F1 79.2%
  K-Means — K=3 · Silhouette 0.32
  t-SNE · PCA · DBSCAN · Hierarchical · CV Tree — Module 4-5
Visual: light green highlighted table
Speaker notes: "This table summarises all fifteen models. For classification
the tree ensembles clearly win — Random Forest and AdaBoost both around 73
percent accuracy with F1 near 80. Linear models underperformed, confirming
basket behaviour is non-linear. Clustering and dimensionality reduction
gave us interpretable views of shopper segments. Random Forest is our
champion model with the best F1 score."

--- SLIDE 10: THANK YOU & FUTURE SCOPE ---
Title: Thank you
Subtitle: Questions welcome
Bullets (small, at bottom):
  - Future: deep learning on order sequences
  - Future: real-time grocery API integration
  - Future: personalised recommendation engine
Small line: CartIQ · Python · scikit-learn · Flask · 50,000 baskets
Visual: clean light slide with a subtle basket illustration
Speaker notes: "Thank you madam. In short, we built a complete machine
learning pipeline — from raw data exploration to fifteen trained models to
a live web application — and found that tree-based ensembles best predict
product reorders. For future scope we would add deep learning on order
sequences, integrate a real-time grocery API, and build a personalised
recommendation engine on top of the shopper clusters we discovered. We're
happy to take any questions, and I can run the live demo again."

=========================================================
EXTRA RULES
=========================================================
- Every slide must have SPEAKER NOTES filled in exactly as written above.
- Slides 4, 5 and 8 are the most important — make their visuals large.
- No slide should have more than 50 words of visible text.
- Keep numbering subtle in the corner (01/10 style).
- One consistent layout: title top-left, visual dominant, bullets slim
  beside or below the image.
```
