# AI Content Detection and Document Analysis Smart System
## Industrial Training Technical Report & System Specification

**Host Organisation:** Novem Controls Pvt. Ltd. (Registration ID: `5075/25-26`)  
**Training Period:** 16 June 2026 – 31 July 2026  
**Student Name:** Rohit Sharma  
**Roll Number:** `23010603054`  
**Department:** Computer Science & Engineering  
**Institution:** Government Post Graduate College Dharamshala  

---

## Executive Summary & Scope Clarification

This project represents the practical culmination of the six-week industrial internship at Novem Controls Pvt. Ltd. It addresses the growing need in academic and professional institutions for transparent, reproducible evaluation of digital documents—specifically Microsoft PowerPoint (`.pptx`) presentations.

> **Administrative Title Reconciliation (Section 5.1 & 10.1):**  
> While the administrative registration title for this industrial training was submitted under *"AI Smart RAM Management System"* (originating from preliminary desktop systems programming in C#/WPF), the full technical research, machine-learning pipeline, backend service architecture, and deployment deliverables documented herein constitute the **AI Content Detection and Document Analysis Smart System**. Both labels are preserved for complete academic integrity and traceability.

---

## Chapter 1: Host Organisation & Training Context

### 1.1 Host Profile: Novem Controls Pvt. Ltd.
Novem Controls Pvt. Ltd. served as the sponsoring and mentoring organization for this applied Artificial Intelligence and Machine Learning (AI/ML) industrial internship. The training connected theoretical computer science curriculum—spanning statistical modeling, algorithm design, data engineering, and secure software development—with real-world software delivery.

### 1.2 Academic Relevance
For an undergraduate student in Computer Science and Engineering at Government Post Graduate College Dharamshala, this internship unified cross-functional software disciplines:
* **Applied Machine Learning:** Feature extraction, hyperparameter tuning, linear classifiers, gradient-boosted decision trees, and ensemble modeling.
* **Data Engineering:** Memory-aware out-of-core streaming algorithms handling 500,000 records without encountering RAM exhaustion.
* **Systems & Document Parsing:** Low-level OpenXML/PPTX shape hierarchy traversal, typography extraction, and visual heuristic audits.
* **Application Security & Web Delivery:** RESTful API design with Flask, PBKDF2-SHA256 password salting, file upload validation, and ephemeral container cleanup.

---

## Chapter 2: Project Objectives & Learning Outcomes

### 2.1 Primary Objective
To design, implement, and evaluate an integrated end-to-end web platform capable of parsing presentation documents, auditing structural formatting heuristics, estimating the probabilistic likelihood of AI-generated prose across individual sentences, and providing grounded contextual retrieval (RAG).

### 2.2 Specific Technical Objectives
1. **Mathematical Distinction in Content Evaluation:** Establish a rigorous operational separation between deterministic source-matching plagiarism detection and statistical machine-learning classification.
2. **Robust NLP Preprocessing Pipeline:** Develop a standardized, version-controlled cleaning sequence utilizing Unicode NFKC normalization, regular expression sanitization, word punctuation tokenization, and selective English stop-word filtering.
3. **High-Dimensional Sparse Feature Extraction:** Implement TF-IDF vectorization with unigram and bigram representation capped at a bounded vocabulary of 10,000 features.
4. **Out-of-Core Batch Learning:** Mitigate `MemoryError` bottlenecks on datasets of approximately 500,000 records by processing fixed 50,000-row chunks.
5. **Ensemble Modeling:** Formulate and calibrate a weighted linear opinion pool combining Stochastic Gradient Descent (SGD) and XGBoost models:
   $$\mathcal{P}_{\text{final}}(\text{AI} \mid X) = 0.60 \times \mathcal{P}_{\text{xgb}}(X) + 0.40 \times \mathcal{P}_{\text{sgd}}(X)$$
6. **Structural Presentation Audits:** Implement deterministic heuristics using `python-pptx` to identify high text density ($>40$ words), visual imbalance ($>10$ words without an image), and sub-standard typography ($<18\text{ pt}$).
7. **Grounded Retrieval-Augmented Generation (RAG):** Enable conversational interaction strictly bounded by extracted slide passages with mandatory slide citations and abstention logic.

---

## Chapter 3: Weekly Training Breakdown

| Week | Period | Milestones, Activities & Technical Deliverables |
| :--- | :--- | :--- |
| **Week 1** | 16–20 June | Orientation; problem definition; review of plagiarism literature vs. statistical AI detection; threat modeling and risk analysis; initial software design plan. |
| **Week 2** | 22–27 June | Dataset exploration; class-balance verification (250k Human / 250k AI); regex pipeline design; tokenization and stop-word rules; train/test splitting; TF-IDF vectorizer configuration. |
| **Week 3** | 29 June–4 July | Sparse feature matrix generation; memory profiling; chunked CSV streaming (`chunksize=50,000`); SGD `partial_fit` benchmarking; XGBoost continuation boosting; model serialization. |
| **Week 4** | 6–11 July | Ensemble probability calibration; decision threshold selection; classification reports, confusion matrices, and ablation experiments; Flask backend routing and PBKDF2 authentication. |
| **Week 5** | 13–18 July | OpenXML PPTX hierarchy parsing; slide shape traversal; image detection (`shape_type == 13`); slide quality audit rules; sentence probability highlighting; lexicon tone analysis. |
| **Week 6** | 20–31 July | Grounded RAG workflow integration; edge-case and malicious input validation; temporary upload lifecycle cleanup; latency profiling; final report and presentation preparation. |

---

## Chapter 4: Technical Specifications & Theoretical Foundations

### 4.1 NLP Text Normalization
All input text undergoes deterministic preprocessing prior to feature mapping:
```python
def clean_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = wordpunct_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)
```

### 4.2 TF-IDF Feature Representation
For training document collection $N$, term frequency $\text{tf}(t,d)$ and document frequency $\text{df}(t)$:
$$\text{idf}(t) = \ln\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
The feature vector is normalized to unit Euclidean length ($\|\mathbf{x}\|_2 = 1$). Feature bounds:
* `max_features` = 10,000
* `ngram_range` = (1, 2)
* `sublinear_tf` = True
* `dtype` = float32

### 4.3 Classification Models

#### SGD Linear Classifier
Minimizes regularized empirical risk with log-loss (`loss="log_loss"`, equivalent to logistic regression):
$$\mathbf{w}_{t+1} \leftarrow \mathbf{w}_t - \eta \left(\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}; \mathbf{x}_i, y_i) + \alpha \mathbf{w}_t\right)$$
Supports genuine out-of-core incremental fitting via `partial_fit(X, y, classes=[0, 1])`.

#### XGBoost Gradient Boosting
Fits additive decision trees to second-order Taylor approximations of the logistic loss:
$$\text{Obj}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
Hyperparameters: `max_depth=6`, `learning_rate=0.05`, `tree_method="hist"`, `objective="binary:logistic"`.

#### Weighted Ensemble Combination
$$\mathcal{P}_{\text{final}}(\text{AI} \mid X) = 0.60 \times \mathcal{P}_{\text{xgb}}(X) + 0.40 \times \mathcal{P}_{\text{sgd}}(X)$$
* **Display Band Assignment:**
  * $\mathcal{P} \ge 0.70$: High Probability (Red)
  * $0.40 \le \mathcal{P} < 0.70$: Moderate / Ambiguous (Amber)
  * $\mathcal{P} < 0.40$: Likely Human (Green)
  * Tokens $< 5$: Insufficient Text (Grey, avoids high-confidence errors on isolated words)

---

## Chapter 5: Document Intelligence & Slide Audit Heuristics

The presentation parsing service traverses Microsoft PowerPoint slides and nested shape hierarchies:

| Rule Name | Heuristic Trigger | System Action / User Advisory |
| :--- | :--- | :--- |
| **Text Density** | Slide word count $> 40$ words | *"Text exceeds 40 words. Consider splitting the slide or moving detail to speaker notes."* |
| **Visual Balance** | Slide word count $> 10$ words with no detected picture (`shape_type == 13`) | *"Text-heavy slide has no detected picture. Consider adding a relevant chart, diagram or image."* |
| **Typography** | Explicit text run font size $< 18\text{ pt}$ | *"$N$ run(s) use explicit font size below 18 pt. Increase text size or reduce content density."* |

*Defensive Rule Handling:* When font sizes are theme-inherited or omitted at run level, the parser classifies the attribute as "inherited/unknown" rather than falsely triggering a violation.

---

## Chapter 6: Grounded Retrieval-Augmented Generation (RAG)

The integrated RAG engine indexes parsed presentation slides into bounded contextual chunks:
1. **Passage Retrieval:** Top-$k$ candidate slide passages are scored via lexical overlap and term relevance.
2. **Citations:** Every synthesized answer cites the specific supporting slide numbers (e.g., `[Slide 3]`).
3. **Mandatory Abstention:** If similarity is below threshold, the engine returns:
   > *"The uploaded presentation does not contain sufficient information to answer this query."*
4. **Prompt Injection Defense:** Input text is strictly sanitized and treated as passive data, disallowing system prompt overrides.

---

## Chapter 7: Results, Benchmarks & Validation

### Project-Reported Evaluation Metrics
* **Total Training Examples:** $\approx 500,000$ (250,000 human-authored, 250,000 machine-generated)
* **Overall Classification Accuracy:** **95.2%**
* **AI-Class Precision:** **0.96**
* **AI-Class Recall:** **0.93**
* **AI-Class F1 Score (Harmonic Mean):** **0.945**
* **End-to-End Processing Latency:** **$\approx 2.68$ seconds** (evaluated on a standard 20-slide presentation deck)

---

## Chapter 8: Security & Production Defensive Controls

1. **Authentication:** PBKDF2-SHA256 password hashing with unique 16-byte cryptographically secure random salts and 100,000 HMAC iterations.
2. **File Upload Security:**
   * Maximum upload size strictly capped at 20 MB (`MAX_UPLOAD_BYTES = 20 * 1024 * 1024`).
   * Explicit extension validation restricted to `.pptx`.
   * Randomized server-side UUID temporary filenames (`uuid.uuid4().hex`) generated in isolated storage outside the web root.
   * Deterministic cleanup guaranteed via Python `finally` blocks, removing temporary files upon both successful completion and exceptions.

---

## Chapter 9: Ethical Principles & Responsible Use

As established in Section 8.3 of the report:
* **No Forensic Proof:** Classifier outputs are probabilistic representations of lexical similarity to training data. They do not constitute definitive legal or disciplinary proof of authorship.
* **Human-in-the-Loop:** Academic evaluations must provide students an opportunity to explain their work, share draft histories, and present oral defenses.
* **Domain Awareness:** Models must be continuously monitored for domain shifts, paraphrasing evasion, and language evolution.

---

## Verification & Execution Instructions

To launch and test the complete system on Windows:
```bash
# Option 1: Double-click run.bat in the root folder

# Option 2: Run via terminal
python app.py
```
Open your browser to: `http://127.0.0.1:5000`  
* **Demo Username:** `intern`  
* **Demo Password:** `novem2026`  

To run automated unit and route validation tests:
```bash
python test_pipeline.py
python test_app_routes.py
```
