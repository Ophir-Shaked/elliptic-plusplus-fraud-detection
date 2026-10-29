# Elliptic++ — Illicit Bitcoin Transaction Detection

**A Reproducible Machine Learning and Graph Neural Network Research Framework for Detecting Illicit Bitcoin Transactions**

Developed by [Ophir-Shaked](https://github.com/Ophir-Shaked)

---

## 1) Overview

This project implements an end-to-end machine learning research pipeline for detecting potentially illicit Bitcoin transactions using the **Elliptic++ dataset**.

The system benchmarks **11 machine learning and graph neural network configurations**, ranging from classical tabular classifiers to directed GraphSAGE architectures.

The research investigates whether transaction graph structure provides predictive information beyond transaction-level features, and how different models behave under class imbalance, temporal distribution shift, selective label availability, and changing decision thresholds.

The project includes:

- Official Elliptic++ dataset ingestion and validation
- Transaction-level feature preprocessing
- Directed Bitcoin transaction graph construction
- Eleven machine learning and graph neural network configurations
- Chronological development and evaluation
- Model training, early stopping, and model selection
- Probability calibration
- Historically determined decision thresholds
- Full classification and ranking metrics
- Negative controls and graph ablation experiments
- Temporal distribution shift analysis
- Feature importance analysis
- Label observation bias diagnostics
- Automatic visualization and report generation
- Independent verification of saved predictions and results
- Reproducible experiment configurations and artifact exports

The implementation is modular and designed to run in **Google Colab, PyCharm, Windows PowerShell, and standard Python environments**.

### Research Scope

This project performs retrospective, end-of-period batch analysis.

It is not a real-time transaction monitoring system.

**Important scientific limitation:** Time steps 35–49 were previously examined during development and therefore represent an **exposed retrospective diagnostic window**, not a new independent blind test.

The repository preserves this distinction to avoid misleading claims about out-of-sample generalization.

---

## 2) Key Features

### Machine Learning

- Logistic Regression
- XGBoost
- LightGBM
- CatBoost
- Multi-Layer Perceptron
- GraphSAGE
- Bidirectional GraphSAGE
- Architecture-matched no-graph controls
- Degree-preserving rewired graph controls

### Research Methodology

- Chronological development folds
- Historical-only model fitting
- Separate early-stopping, calibration, and threshold-selection windows
- No threshold selection using exposed diagnostic labels
- Graph ablation experiments
- Negative-control experiments
- Multiple random-seed comparisons
- Distribution shift diagnostics
- Class imbalance evaluation
- Label observation bias analysis

### Evaluation

- Precision, Recall, F1, and F2
- Accuracy, Specificity, and Balanced Accuracy
- ROC-AUC and Average Precision
- Confusion matrices
- False Positive Rate and False Negative Rate
- Precision@K and Recall@K
- Probability calibration
- Performance by time step
- Error analysis
- Threshold sensitivity analysis

### Reproducibility

- Fixed chronological protocol
- Explicit random seeds
- Saved model parameters
- Saved predictions
- Source-code snapshots
- Structured CSV and JSON outputs
- SHA-256 artifact verification
- Independent metric recomputation
- Automated tests
- Synthetic integration tests

---

## 3) Research Objectives

The primary objective is to investigate the effectiveness and reliability of machine learning methods for identifying known illicit Bitcoin transactions.

The research addresses several questions.

### 3.1 Predictive Performance

How accurately can machine learning models identify known illicit transactions from transaction-level features?

The project compares linear, tree-based, and neural approaches using identical chronological evaluation windows.

### 3.2 Graph Information

Does the Bitcoin transaction network improve predictive performance beyond individual transaction features?

The comparison includes regular GraphSAGE, no-graph variants, and structurally rewired graph controls.

### 3.3 Temporal Generalization

How does model performance change across Bitcoin transaction time periods?

The project examines changes in:

- Average Precision
- ROC-AUC
- Precision
- Recall
- False positive rates
- Feature distributions
- Known-label prevalence

### 3.4 Decision Policies

How do different historically selected decision thresholds affect the balance between identifying illicit transactions and generating false positives?

### 3.5 Scientific Reliability

Can observed improvements be explained by:

- Graph connectivity?
- Transaction-level information?
- Temporal cohort effects?
- Label availability?
- Random variation?
- Information leakage?

These questions motivate the negative-control and graph-ablation experiments.

---

## 4) Dataset — Elliptic++

The research uses the public **Elliptic++ Bitcoin transaction dataset**.

Official repository:

https://github.com/git-disl/EllipticPlusPlus

### 4.1 Dataset Statistics

| Property | Value |
|---|---:|
| Total transactions | 203,769 |
| Directed transaction edges | 234,355 |
| Known illicit transactions | 4,545 |
| Known licit transactions | 42,019 |
| Unknown transactions | 157,205 |
| Time steps | 49 |
| Primary numerical features used | 93 |

The transactions are represented as nodes in a directed graph, while transaction relationships are represented as edges.

### 4.2 Transaction Labels

The original dataset contains three class categories.

| Original label | Meaning | Internal label |
|---|---|---|
| 1 | Known illicit | 1 |
| 2 | Known licit | 0 |
| 3 | Unknown | -1 |

**An unknown transaction is not a confirmed licit transaction.**

Unknown transactions must not be automatically classified as negatives.

For supervised training and conventional classification metrics, the project uses transactions with known labels only.

Unknown transactions may still contribute feature and graph context under the retrospective batch evaluation setting.

### 4.3 Class Imbalance

Among transactions with known labels, illicit transactions form a minority class.

The known-label positive prevalence is:

$$
\pi =
\frac{N_{\mathrm{illicit}}}
{N_{\mathrm{illicit}}+N_{\mathrm{licit}}}
$$

For the dataset totals:

$$
\pi =
\frac{4545}{4545+42019}
\approx 0.0976
$$

Approximately 9.76% of the labeled transactions are known illicit.

This proportion is conditional on label availability and does not establish the actual illicit prevalence in the complete Bitcoin transaction graph.

Because of class imbalance, Accuracy alone is insufficient to evaluate model effectiveness.

The project emphasizes Average Precision, Recall, Precision, F1, and false-positive behavior.

### 4.4 Dataset Files

The loader uses the official transaction files:

```text
Transactions Dataset/
├── txs_features.csv
├── txs_classes.csv
└── txs_edgelist.csv
```

These files provide:

- Transaction identifiers
- Numerical transaction features
- Transaction time steps
- Known and unknown class labels
- Directed transaction relationships

The loader aligns the files using transaction identifiers rather than relying on row order.

Official data is downloaded separately and is not included in the source-code repository.

---

## 5) Data Preprocessing

Data preprocessing is implemented in the dedicated preprocessing and data-loading modules.

The objective is to transform the raw Elliptic++ data into model-compatible numerical representations while preserving chronological separation.

### 5.1 Data Validation

The data-loading pipeline checks:

- Transaction ID alignment
- Label validity
- Time-step consistency
- Feature dimensions
- Missing numerical values
- Graph edge endpoints
- Dataset integrity

Source-file hashes are recorded for reproducibility and provenance.

### 5.2 Missing-Value Handling

Missing values are handled using statistics estimated from historical training data.

For feature \(j\):

$$
\widetilde{x}_{ij} =
\begin{cases}
x_{ij}, & x_{ij}\text{ is observed}\\
m_j, & x_{ij}\text{ is missing}
\end{cases}
$$

where \(m_j\) is a training-derived imputation value.

The imputation parameters must not be estimated using future evaluation data.

### 5.3 Feature Standardization

Where required by the model, numerical features are standardized:

$$
z_{ij} =
\frac{x_{ij}-\mu_j}{\sigma_j}
$$

where:

- \(\mu_j\) is the historical training mean.
- \(\sigma_j\) is the historical training standard deviation.
- \(x_{ij}\) is the original feature value.

The same fitted preprocessing parameters are applied to subsequent periods.

### 5.4 Feature Selection

The primary experiment uses the first 93 numerical transaction features, excluding identifier and time fields.

Additional on-chain feature groups are examined in development-stage diagnostics.

The exact point-in-time availability of every published feature has not been independently established, so the project does not claim that all features are available at the instant a transaction first appears.

---

## 6) Chronological Experimental Protocol

A central requirement of this project is to respect the temporal ordering of Bitcoin transactions.

Random splitting could cause future information to influence historical predictions.

Instead, the research uses fixed chronological windows.

### 6.1 Development Folds

Model comparisons are conducted using three historical development folds.

| Fold | Training periods | Early stopping | Development evaluation |
|---|---|---|---|
| 1 | 1–10 | 11–12 | 13–15 |
| 2 | 1–15 | 16–17 | 18–20 |
| 3 | 1–20 | 21–22 | 23–25 |

These folds are used for historical model comparisons and development decisions.

### 6.2 Final Retrospective Protocol

| Stage | Time steps | Purpose |
|---|---|---|
| Training | 1–26 | Fit model parameters |
| Early stopping | 27–29 | Select training duration |
| Calibration | 30–31 | Fit probability calibration |
| Threshold selection | 32–34 | Determine decision policies |
| Exposed diagnostics | 35–49 | Evaluate frozen models and policies |

Each stage has a distinct role.

Model parameters, calibration parameters, and decision thresholds must not be selected using the exposed diagnostic outcomes.

### 6.3 Why Separate These Stages?

Different operations use labels for different purposes.

**Training:** Learns the relationship between transaction features, graph structure, and known illicit labels.

**Early stopping:** Determines when additional optimization stops improving historical validation performance.

**Calibration:** Adjusts model confidence estimates using a separate historical window.

**Threshold selection:** Determines which probability scores trigger positive classifications.

**Diagnostics:** Measures how the already-fixed system behaves in later periods.

Keeping these roles separate reduces the risk of data leakage and overly optimistic results.

---

## 7) Models Evaluated

The project implements 11 primary model configurations.

| Model ID | Algorithm | Category |
|---|---|---|
| `LOGIT` | Logistic Regression | Linear |
| `XGB` | XGBoost | Gradient Boosting |
| `LGB` | LightGBM | Gradient Boosting |
| `CAT` | CatBoost | Gradient Boosting |
| `MLP` | Multi-Layer Perceptron | Neural Network |
| `SAGE_BATCH` | Directed GraphSAGE | Graph Neural Network |
| `SAGE_NO_GRAPH` | GraphSAGE without graph messages | Graph Control |
| `SAGE_REWIRED` | GraphSAGE with rewired edges | Graph Control |
| `BIDIR_SAGE` | Bidirectional GraphSAGE | Graph Neural Network |
| `BIDIR_NO_GRAPH` | Bidirectional GraphSAGE without graph messages | Graph Control |
| `BIDIR_REWIRED` | Bidirectional GraphSAGE with rewired edges | Graph Control |

The model configurations share a consistent experimental protocol to support meaningful comparisons.

### 7.1 Logistic Regression

Logistic Regression provides a linear probabilistic baseline.

The model predicts:

$$
P(y=1\mid x)=\sigma(w^Tx+b)
$$

where the sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

This baseline helps determine whether more complex models provide meaningful improvements over a linear decision boundary.

### 7.2 XGBoost

XGBoost uses gradient-boosted decision trees.

A prediction is constructed from an additive sequence of trees:

$$
F_M(x)=\sum_{m=1}^{M}f_m(x)
$$

The model captures nonlinear relationships and feature interactions.

XGBoost is particularly relevant for structured tabular transaction data.

### 7.3 LightGBM

LightGBM is another gradient-boosted decision-tree implementation.

It is designed for computationally efficient training and uses histogram-based tree construction.

It provides a complementary tree-based comparison to XGBoost.

### 7.4 CatBoost

CatBoost is a gradient-boosting implementation with its own tree construction and training methods.

It serves as an additional nonlinear tabular baseline.

### 7.5 Multi-Layer Perceptron

The MLP is a feature-only neural network.

A simplified hidden layer is:

$$
h^{(l+1)}=
\operatorname{ReLU}
\left(W^{(l)}h^{(l)}+b^{(l)}\right)
$$

The final layer produces a classification logit, which is converted into a probability.

The implementation uses two hidden layers and dropout.

The MLP provides a neural baseline without transaction graph message passing.

---

## 8) Graph Neural Networks

The Bitcoin transaction network is modeled as a directed graph:

$$
G=(V,E)
$$

where:

- \(V\) represents Bitcoin transactions.
- \(E\) represents directed transaction relationships.
- \(X\) represents transaction-level node features.

Graph neural networks allow predictions to incorporate information from connected transactions.

### 8.1 Graph Adjacency

For an edge from transaction \(u\) to transaction \(v\), the incoming adjacency representation stores:

$$
A_{vu}=1
$$

The incoming neighborhood of transaction \(v\) is:

$$
\mathcal{N}_{in}(v)
=
\{u:(u,v)\in E\}
$$

Mean-neighbor aggregation is computed using a normalized adjacency matrix:

$$
H_{\mathrm{neighbors}}=D^{-1}AH
$$

This provides an average representation of incoming neighbors.

### 8.2 GraphSAGE

GraphSAGE combines a node's representation with aggregated neighboring information.

A simplified layer is:

$$
h_v^{(l+1)}
=
\operatorname{ReLU}
\left(
W^{(l)}
\left[
h_v^{(l)}
\Vert
\operatorname{MEAN}
\{h_u^{(l)}:u\in\mathcal{N}_{in}(v)\}
\right]
+b^{(l)}
\right)
$$

Here:

- \(h_v^{(l)}\) is the current node representation.
- \(\mathcal{N}_{in}(v)\) is the incoming neighborhood.
- \(\Vert\) denotes concatenation.
- \(W^{(l)}\) contains trainable weights.

The implemented GraphSAGE architecture uses two message-passing layers.

### 8.3 Bidirectional GraphSAGE

Standard incoming GraphSAGE uses only incoming graph messages.

Bidirectional GraphSAGE processes incoming and outgoing transaction relationships separately.

Its layer representation combines:

$$
h_v^{(l)}
\Vert
m_{in,v}^{(l)}
\Vert
m_{out,v}^{(l)}
$$

where the incoming and outgoing messages are calculated using their respective adjacency matrices.

This design preserves the direction of transaction relationships.

### 8.4 No-Graph Controls

The no-graph variants retain architecture-matched neural transformations but disable neighbor information.

This provides a more controlled comparison than comparing GraphSAGE exclusively against unrelated models.

The purpose is to test whether improvements arise from graph connectivity rather than merely additional neural-network capacity.

### 8.5 Rewired Graph Controls

The rewired variants modify graph connections while preserving important structural properties.

The implemented control uses within-period directed edge swaps that preserve individual node in-degrees and out-degrees.

Comparing performance on original and rewired graphs helps investigate whether specific transaction relationships contribute predictive information beyond basic degree structure.

---

## 9) Training and Model Selection

All model configurations follow a common training and evaluation framework.

### 9.1 Supervised Training

Only historically available examples with known illicit or licit labels contribute to the supervised classification loss.

Unknown-labeled transactions do not become negative training examples.

### 9.2 Binary Cross-Entropy

Neural classification is based on binary classification objectives.

The standard binary cross-entropy loss is:

$$
\mathcal{L}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
\right]
$$

where:

- \(y_i\) is the known binary label.
- \(p_i\) is the predicted illicit probability.
- \(N\) is the number of supervised examples.

The implemented training procedures also address class imbalance.

### 9.3 Early Stopping

Early stopping limits unnecessary training and reduces overfitting.

Training duration is selected using designated historical stopping periods rather than exposed evaluation labels.

### 9.4 Development Model Selection

The project evaluates candidate models through chronological development folds.

Average Precision is a central ranking criterion.

The development results are saved for reproducibility and later comparison.

---

## 10) Probability Calibration

A classifier's score is not necessarily a well-calibrated probability.

For example, transactions assigned a score near 0.8 should ideally be illicit approximately 80% of the time in a representative group with that score.

The project uses a constrained Platt-style calibration transformation.

$$
\widehat{p}
=
\sigma
\left(
a+b\operatorname{logit}(p)
\right)
$$

where:

$$
\operatorname{logit}(p)
=
\log\frac{p}{1-p}
$$

The calibration slope is constrained to remain positive.

This preserves score ordering while allowing confidence estimates to change.

Calibration parameters are fitted using time steps 30–31.

The project evaluates calibration using:

- Brier Score
- Log Loss
- Reliability bins
- Reliability diagrams
- Known-label calibration diagnostics

Calibration results are conditional on the available verified labels.

---

## 11) Decision Thresholds

Models produce probability scores.

A decision threshold converts these scores into binary predictions:

$$
\widehat{y}_i=
\begin{cases}
1,&p_i\geq\tau\\
0,&p_i<\tau
\end{cases}
$$

where \(\tau\) is the decision threshold.

The project evaluates two historical threshold policies.

### 11.1 F1-Based Policy

The first policy chooses a threshold that maximizes F1 on the historical policy-selection window.

$$
\tau_{F1}
=
\arg\max_{\tau}
F1(\tau)
$$

### 11.2 False-Positive-Budget Policy

The second policy is named:

`FPR_5_per_10k`

It targets a maximum of five false positives per 10,000 known licit transactions in the historical policy-selection data.

The false-positive count per 10,000 is:

$$
FP_{10k}
=
10{,}000
\cdot
\frac{FP}{FP+TN}
$$

The historical policy is selected using time steps 32–34.

**The policy name does not guarantee the same false-positive rate in later periods.**

Observed false-positive rates may increase under temporal distribution shift.

### 11.3 Diagnostic Threshold Analysis

The project also generates threshold sensitivity reports and figures.

These describe how Precision, Recall, F1, and F2 would change under hypothetical thresholds.

These scans are diagnostic only.

They must not be used to select new decision thresholds based on the exposed time steps 35–49.

---

## 12) Evaluation Metrics

The project exports a comprehensive set of classification and ranking metrics.

### 12.1 Confusion Matrix

The binary confusion matrix is:

$$
CM=
\begin{bmatrix}
TN&FP\\
FN&TP
\end{bmatrix}
$$

where:

- **TP:** Known illicit transactions correctly classified as illicit.
- **TN:** Known licit transactions correctly classified as licit.
- **FP:** Known licit transactions incorrectly flagged as illicit.
- **FN:** Known illicit transactions incorrectly classified as licit.

A confusion matrix is generated for every evaluated model and historical decision policy.

### 12.2 Precision

Precision measures the fraction of predicted illicit transactions that are actually known illicit.

$$
Precision=\frac{TP}{TP+FP}
$$

High Precision means fewer false alarms among positively classified transactions.

### 12.3 Recall — Sensitivity

Recall measures the fraction of known illicit transactions successfully detected.

$$
Recall=\frac{TP}{TP+FN}
$$

Recall is particularly important when the research goal is to identify illicit activity.

### 12.4 F1-Score

F1 is the harmonic mean of Precision and Recall.

$$
F_1=
\frac{2PR}{P+R}
$$

Equivalently:

$$
F_1=
\frac{2TP}{2TP+FP+FN}
$$

### 12.5 F2-Score

F2 places more weight on Recall than Precision.

$$
F_2=
\frac{5PR}{4P+R}
$$

Equivalently:

$$
F_2=
\frac{5TP}{5TP+4FN+FP}
$$

F2 is useful for examining detection-oriented decision policies.

### 12.6 Accuracy

$$
Accuracy=
\frac{TP+TN}{TP+TN+FP+FN}
$$

Because the dataset is imbalanced, Accuracy should not be interpreted in isolation.

### 12.7 Specificity

$$
Specificity=
\frac{TN}{TN+FP}
$$

Specificity measures the fraction of known licit transactions correctly recognized as licit.

### 12.8 Balanced Accuracy

$$
BalancedAccuracy=
\frac{Recall+Specificity}{2}
$$

Balanced Accuracy gives equal weight to positive and negative class recall.

### 12.9 False Positive Rate

$$
FPR=\frac{FP}{FP+TN}
$$

### 12.10 False Negative Rate

$$
FNR=\frac{FN}{FN+TP}
$$

### 12.11 ROC-AUC

The ROC curve evaluates the relationship between:

$$
TPR=\frac{TP}{TP+FN}
$$

and:

$$
FPR=\frac{FP}{FP+TN}
$$

across varying decision thresholds.

ROC-AUC summarizes the model's ranking discrimination between the two known classes.

### 12.12 Average Precision and PR Curve

The Precision–Recall curve examines Precision and Recall across different thresholds.

Average Precision summarizes the precision–recall tradeoff using a recall-weighted aggregation.

The project reports Average Precision using the standard implementation provided by scikit-learn.

Average Precision should not be confused with every possible numerical interpolation method for calculating the area under a PR curve.

### 12.13 Calibration Metrics

**Brier Score**

$$
Brier=
\frac{1}{N}
\sum_{i=1}^{N}
(p_i-y_i)^2
$$

**Log Loss**

$$
LogLoss=
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i\log p_i+
(1-y_i)\log(1-p_i)
\right]
$$

The exported reports also include additional diagnostic quantities, such as:

- Negative Predictive Value
- False Discovery Rate
- Matthews Correlation Coefficient
- Predicted-positive counts
- Class support
- Known-label prevalence

Undefined ratios are handled according to the report's documented denominator conventions rather than being interpreted as evidence of strong performance.

---

## 13) Precision@K and Recall@K

In practical risk-prioritization settings, an investigator may only be able to review a limited number of alerts.

The project therefore evaluates top-ranked transactions.

Transactions are ordered by their predicted illicit scores.

For a chosen \(K\), the top \(K\) highest-scoring transactions are examined.

### 13.1 Precision@K

Standard fully labeled Precision@K is:

$$
Precision@K=
\frac{\text{Relevant items among top K}}{K}
$$

However, Elliptic++ contains many unknown labels.

For this reason, the project explicitly reports **precision among verified alerts**:

$$
Precision@K_{\mathrm{verified}}
=
\frac{\text{Known illicit alerts in top K}}
{\text{Known illicit + known licit alerts in top K}}
$$

Unknown alerts remain unverified and are not counted as confirmed false positives.

This metric is not equivalent to true Precision@K over the complete transaction population.

### 13.2 Recall@K

The project measures the fraction of known illicit transactions retrieved within the ranked alerts:

$$
Recall@K=
\frac{\text{Known illicit transactions retrieved in top K}}
{\text{Total known illicit transactions in the period}}
$$

### 13.3 Evaluated K Values

The reporting module supports:

- K = 10
- K = 25
- K = 50
- K = 100

Additional K values are calculated only when enough ranked transactions have been retained.

The default experiment stores the top 100 alerts per period.

The pipeline produces both per-period and aggregated Top-K reports.

---

## 14) Negative Controls and Scientific Validation

The project includes multiple negative controls intended to identify misleading sources of apparent predictive performance.

### 14.1 Label Permutation Controls

Training labels may be randomly permuted to test whether models continue producing unexpectedly strong performance when their supervised relationship to the features is disrupted.

The project distinguishes global label permutations from within-period permutations.

Within-period permutations retain temporal class prevalence and therefore are not a universal null model.

### 14.2 Evaluation Permutation Controls

Evaluation-label and score permutation experiments examine whether ranking quality depends on transaction-level information beyond temporal grouping effects.

### 14.3 Graph Rewiring

Graph rewiring disrupts the original graph connectivity while retaining selected structural properties.

This allows controlled comparisons between original and modified transaction networks.

### 14.4 No-Graph Ablations

GraphSAGE models are compared against architecture-matched versions with disabled neighbor aggregation.

These comparisons help isolate the contribution of graph message passing.

### 14.5 Cohort and Time-Based Baselines

Time-only and cohort-only controls investigate whether transaction periods themselves explain part of the observed predictive performance.

### 14.6 Label Observation Bias

The project analyzes whether the probability of receiving a known label varies with transaction features.

A model may be able to predict whether a transaction is labeled without successfully identifying whether it is illicit.

This distinction matters because the labeled subset may not represent all Bitcoin transactions.

### 14.7 Distribution Shift

The project includes descriptive drift analysis using measures such as the Population Stability Index.

For distributions partitioned into bins, PSI is commonly defined as:

$$
PSI=
\sum_{j=1}^{B}
(q_j-p_j)
\ln\left(\frac{q_j}{p_j}\right)
$$

where \(p_j\) and \(q_j\) represent historical and comparison-period proportions.

Distribution shift diagnostics help describe changes in the input data, but do not by themselves establish the cause of performance changes.

---

## 15) Automatic Metrics, Reports, and Visualizations

The pipeline automatically generates structured experimental reports.

All final classification metrics are recalculated from saved transaction-level predictions.

The reporting module verifies agreement with the original experiment summaries.

### 15.1 Classification Reports

The `FULL_METRICS/` directory contains:

```text
FULL_METRICS/
├── CLASSIFICATION_REPORT_ALL_MODELS.csv
├── CLASSIFICATION_BY_TIME_STEP.csv
├── THRESHOLD_SWEEP_DIAGNOSTIC_ONLY.csv
├── PR_ROC_CURVE_POINTS.csv
├── CALIBRATION_BINS.csv
├── RECALL_PRECISION_AT_K_BY_STEP.csv
├── RECALL_PRECISION_AT_K_SUMMARY.csv
├── INDEPENDENT_METRIC_RECOMPUTATION_CHECK.csv
├── METRIC_DEFINITIONS.md
└── plots/
```

### 15.2 Confusion Matrices

For a full 11-model run, the system generates 22 confusion matrices, corresponding to:

- 11 models
- 2 historically selected threshold policies

Example filenames:

```text
CONFUSION_XGB_F1_policy.png
CONFUSION_XGB_FPR_5_per_10k.png
```

Each matrix displays TP, TN, FP, and FN for known-label transactions.

### 15.3 Precision–Recall and ROC Curves

The system generates a combined PR/ROC visualization for each model.

Example:

```text
PR_ROC_XGB.png
```

These figures include:

- Precision–Recall curve
- Average Precision
- ROC curve
- ROC-AUC
- Reference baselines

### 15.4 Cross-Model Comparison

The reporting system includes visual comparisons of:

- Recall
- Precision
- F1
- Average Precision
- Calibration performance
- Negative controls

Example:

```text
COMPARE_RECALL_PRECISION_F1.png
```

### 15.5 Performance by Time Step

For every model, the project automatically generates:

```text
TIME_SERIES_RECALL_PRECISION_<MODEL>.png
```

These figures show Recall and Precision across the exposed time steps under both fixed decision policies.

For all 11 models, this produces 11 figures.

### 15.6 Top-K Model Comparison

The project generates:

```text
COMPARE_PRECISION_RECALL_AT_K.png
```

This compares ranked-alert retrieval performance across models using known-label Precision@K and Recall@K definitions.

### 15.7 Threshold Sensitivity

For each model, the system creates:

```text
THRESHOLD_SENSITIVITY_<MODEL>_DIAGNOSTIC_ONLY.png
```

These figures show how classification metrics vary across hypothetical thresholds.

Historically selected thresholds are marked for comparison.

**The exposed diagnostic period must not be used to select new thresholds.**

### 15.8 Reliability Diagrams

The project generates:

```text
RELIABILITY_<MODEL>.png
```

Each figure presents:

- Mean predicted probabilities
- Observed illicit fractions among known labels
- Ideal calibration reference
- Calibration-bin sample counts

### 15.9 Complete Automatic Plot Generation

For a complete 11-model experiment, the supplementary reporting system generates:

| Additional visualization | Number |
|---|---:|
| Recall and Precision by time step | 11 |
| Precision@K and Recall@K comparison | 1 |
| Threshold sensitivity | 11 |
| Reliability diagrams | 11 |
| **Total additional figures** | **34** |

These are generated in addition to the existing confusion matrices, PR/ROC curves, and research diagnostic figures.

All automatically generated figures are included in the completed experiment's results archive.

---

## 16) Independent Result Verification

The project includes a separate verification system.

The purpose is to ensure that exported results are consistent with the saved model predictions and experimental metadata.

### 16.1 Independent Metric Recalculation

The verifier checks recorded results against metrics recomputed from saved predictions.

This includes:

- True positives
- True negatives
- False positives
- False negatives
- Precision
- Recall
- F1
- ROC-AUC
- Average Precision
- Calibration metrics

### 16.2 Model Reload Verification

Saved models and preprocessing artifacts are reloaded to test prediction reproducibility.

### 16.3 SHA-256 Integrity Checks

Each completed run contains:

```text
SHA256_MANIFEST.csv
```

The manifest records cryptographic hashes of output artifacts.

A hash mismatch may indicate that a recorded result has changed since the manifest was generated.

### 16.4 Completion Checks

The pipeline verifies that required outputs exist before writing a success marker.

A successfully completed run contains:

```text
RUN_SUCCESS.json
```

A completed archive can be audited independently using:

```bash
python verify_results.py audit PATH_TO_RESULTS.zip --models 11
```

**Passing these checks establishes internal consistency and artifact integrity, not prospective scientific validity.**

---

## 17) How to Run

### 17.1 Requirements

The project uses Python and the following major libraries:

- NumPy
- pandas
- SciPy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- PyTorch
- Matplotlib
- joblib
- pytest

A compatible Python environment is required.

Python 3.12 or 3.13 may be used when compatible package versions and platform wheels are available.

### 17.2 Create a Virtual Environment — Windows / PyCharm

Open the extracted project root in PyCharm.

Create a virtual environment:

```powershell
py -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 17.3 Run Automated Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The source-only repository includes tests that can run without downloading the official dataset.

Historical-reference-specific checks require their original external reference fixtures.

### 17.4 Synthetic Smoke Test

A synthetic Smoke test validates software execution without performing a scientific experiment on Bitcoin transactions.

```powershell
.\.venv\Scripts\python.exe run.py --smoke --quick --no-download --models XGB,SAGE_BATCH,BIDIR_SAGE --control-reps 4 --seed-repeats 1 --top-k 50
```

The synthetic test checks the execution pipeline, but its classification metrics must not be reported as empirical research results.

### 17.5 Full Research Experiment — CPU

For a complete experiment on the official Elliptic++ dataset:

```powershell
.\.venv\Scripts\python.exe run.py --output results --control-reps 120 --seed-repeats 2 --top-k 100
```

By default, the project evaluates all 11 model configurations.

Git and Git LFS are required for automatic retrieval of the official data.

### 17.6 Full Research Experiment — GPU

When CUDA-compatible PyTorch is available:

```bash
python -u run.py --output results --control-reps 120 --seed-repeats 2 --top-k 100 --gpu
```

The `--gpu` argument requests GPU acceleration where supported.

If CUDA is unavailable, the runner reports the fallback to CPU execution.

### 17.7 Run in Google Colab

This source-only repository does not require a bundled notebook.

Upload and extract the source ZIP into a Colab environment, then change the working directory to the extracted `EllipticPlusPlus` project root.

Install the dependencies:

```python
!python -m pip install -q -r requirements.txt
```

The official data download requires Git LFS. On a Colab environment where it is not installed:

```python
!apt-get update -qq
!apt-get install -y -qq git-lfs
!git lfs install
```

Run the automated tests:

```python
!python -m pytest -q
```

Run the complete research experiment:

```python
!python -u run.py --output /content/elliptic_results --control-reps 120 --seed-repeats 2 --top-k 100 --gpu
```

The outputs are saved under the selected results directory.

For a persistent copy, users may transfer the completed ZIP archive to their own storage.

---

## 18) Command-Line Options

| Argument | Description |
|---|---|
| `--smoke` | Run with synthetic transactions |
| `--quick` | Use reduced experimental budgets |
| `--no-download` | Require already available dataset files |
| `--gpu` | Request CUDA acceleration |
| `--data-dir` | Specify the official dataset cache location |
| `--output` | Specify the output directory |
| `--models` | Select a subset of model configurations |
| `--top-k` | Set the number of saved ranked alerts per period |
| `--seed-repeats` | Configure matched graph random-seed experiments |
| `--control-reps` | Configure negative-control repetitions |
| `--disable-rewired` | Disable primary rewired graph model configurations |

Example: run selected models.

```bash
python run.py --models XGB,LGB,SAGE_BATCH,BIDIR_SAGE --output selected_results
```

Some fixed development diagnostics and control experiments may still execute independently of the selected primary model subset.

For the complete benchmark, omit `--models`.

---

## 19) Recreating Metrics and Figures Without Retraining

The project supports postprocessing of already completed experiments.

This is useful when additional graphics or metric reports are needed without repeating expensive model training.

### 19.1 Regenerate Visualizations

```bash
python plot_results.py --results PATH_TO_COMPLETED_RUN
```

This recreates figures from the saved experimental tables.

### 19.2 Rebuild Full Classification Reports

```bash
python export_all_metrics.py PATH_TO_COMPLETED_RUN
```

This produces classification reports and supplementary figures using saved predictions.

No model training is performed by this command.

### 19.3 Verify Results

```bash
python verify_results.py audit PATH_TO_COMPLETED_RUN --models 11
```

The verifier supports completed results directories and ZIP archives.

### 19.4 Important Artifact Integrity Rule

Postprocessing an extracted, previously completed run may introduce new files or modify existing reports.

The original SHA-256 manifest must not be represented as covering newly added files.

Any newly distributed archive should be clearly identified as a postprocessed derivative and should receive its own appropriate integrity manifest.

---

## 20) Repository Structure

The project follows a modular Python architecture.

```text
EllipticPlusPlus/
│
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── run.py
├── run_pycharm.py
├── plot_results.py
├── export_all_metrics.py
├── verify_results.py
├── verify_synthetic_migration.py
│
├── elliptic_fraud/
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── graph.py
│   ├── models.py
│   ├── evaluation.py
│   ├── protocol.py
│   ├── development.py
│   ├── controls.py
│   ├── analysis.py
│   ├── final_evaluation.py
│   ├── metric_reports.py
│   ├── visualization.py
│   ├── verification.py
│   ├── runner.py
│   └── utils.py
│
└── tests/
    ├── test_research_invariants.py
    ├── test_final_verification.py
    └── test_metric_reports.py
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Central experiment configuration |
| `data.py` | Dataset loading, alignment, and validation |
| `preprocessing.py` | Feature transformation |
| `graph.py` | Graph construction and rewiring |
| `models.py` | Machine learning and GNN implementations |
| `evaluation.py` | Evaluation utilities and calibration |
| `protocol.py` | Chronology and leakage safeguards |
| `development.py` | Development experiments and model comparison |
| `controls.py` | Negative controls and bias diagnostics |
| `analysis.py` | Feature importance and distribution shift |
| `final_evaluation.py` | Final retrospective model evaluation |
| `metric_reports.py` | Detailed classification reports and supplementary figures |
| `visualization.py` | Research visualizations |
| `verification.py` | Independent artifact and metric auditing |
| `runner.py` | Complete experiment orchestration |
| `utils.py` | Shared persistence and hash utilities |

The repository is source-only and does not include downloaded datasets, trained model weights, generated images, experiment ZIP archives, or temporary files.

---

## 21) Reproducibility and Leakage Prevention

The research applies the following reproducibility principles.

### Chronological Separation

Historical training, early stopping, calibration, threshold selection, and exposed diagnostics have explicitly defined time windows.

### Historical Preprocessing

Training-derived preprocessing parameters are not fitted using future evaluation data.

### Known-Label Supervision

Unknown class labels are excluded from supervised classification targets and known-label evaluation denominators.

### Graph Direction

Incoming and outgoing relationships are represented explicitly rather than silently converting all graph edges into undirected edges.

### Matched Controls

Architecture-matched graph variants and repeated-seed comparisons help isolate the contribution of graph structure.

### Reproducible Artifacts

Experiment outputs include saved predictions, model artifacts, configuration snapshots, and integrity hashes.

### Independent Auditing

A separate verification path recalculates recorded metrics and checks saved outputs.

### Scientific Reporting

Previously exposed evaluation periods are explicitly labeled as retrospective diagnostics.

No new blind-test claims are made from these periods.

---

## 22) Experimental Validation Status

The implementation has undergone software validation using automated unit tests, synthetic end-to-end execution, and independent auditing of saved experimental outputs.

Validation performed during project development included:

- 28 passing automated tests in the updated source version
- Three skipped historical-reference tests because their original fixtures are excluded from the source-only package
- Successful synthetic integration tests
- Successful supplementary figure generation
- Independent result verification of generated archives
- Successful generation of supplementary figures from an existing real-data experiment

A previously completed full 11-model real-data execution produced saved predictions and diagnostic reports that were independently audited.

The updated figure-generation functionality was also tested separately using an end-to-end synthetic run and postprocessing of previously saved real-data reports.

**Software validation is not equivalent to an independent prospective evaluation.**

The exposed diagnostic period remains unsuitable for claims of newly confirmed blind-test generalization.

---

## 23) Limitations

This research has several important limitations.

**1. No new blind test**

Time steps 35–49 have previously influenced research exploration and cannot be treated as untouched test data.

**2. Unknown transaction labels**

A large fraction of transactions lack verified labels.

Evaluation on known labels may not reflect performance over the entire population.

**3. Label observation bias**

The probability that a transaction is labeled may depend on transaction characteristics.

**4. Feature availability**

The dataset does not fully establish the availability timestamp of every derived feature.

**5. Batch graph setting**

The retrospective batch graph may contain same-period connections that would not have been available at the exact time of an individual transaction.

**6. Temporal distribution shift**

Transaction characteristics and class prevalence may change between periods.

**7. False-positive policy stability**

A false-positive budget achieved during historical threshold selection is not guaranteed to hold in future periods.

**8. Reproducibility versus determinism**

The project controls random seeds and records metadata, but numerical results may still vary with library versions, hardware, and nondeterministic GPU operations.

**9. Model comparison uncertainty**

Overlapping development windows, graph dependencies, and selectively observed labels limit strong statistical significance claims.

**10. Deployment limitations**

This framework is a research benchmark rather than a validated production anti-money-laundering system.

A positive model prediction is a risk signal, not proof of criminal conduct.

---

## 24) Possible Future Research

Potential extensions include:

- Evaluation on a genuinely independent, previously unseen temporal dataset
- Stronger point-in-time feature provenance validation
- Alternative graph neural network architectures
- Graph Attention Networks
- Temporal Graph Neural Networks
- Self-supervised graph representations
- Improved calibration under distribution shift
- Domain adaptation
- More comprehensive selective-label evaluation
- Uncertainty quantification
- Additional graph structure controls
- Explainability methods for graph-based predictions
- Improved operational alert-budget evaluation

These are possible future directions and are not presented as implemented features of the current repository.

---

## 25) References

### Dataset

Elliptic++ — Official Dataset and Research Repository

https://github.com/git-disl/EllipticPlusPlus

### Machine Learning and Graph Methods

- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Prokhorenkova, L., et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.

### Related Projects

Additional machine learning projects by the same developer:

- [Recommendation Engine](https://github.com/Ophir-Shaked/Recommendation_Engine)
- [ShiftLab](https://github.com/Ophir-Shaked/shiftlab)
- [MedRecQA Recommendation Engine](https://github.com/Ophir-Shaked/MedRecQA-Recommendation-Engine)

---

## 26) Author

**Ophir Shaked**

GitHub: https://github.com/Ophir-Shaked

Research interests include machine learning, graph neural networks, anomaly detection, statistical modeling, and reproducible experimental research.

---

## 27) License and Dataset Attribution

The project source code and the Elliptic++ dataset are separate resources.

The source repository does not redistribute the original dataset.

Users should review the official dataset's terms and cite the relevant dataset publications when using Elliptic++ in academic work.

A software license has not yet been assigned to this source-only release. Reuse and redistribution permissions should be clarified by adding an appropriate `LICENSE` file.

---

**Project Status:** Research implementation with automated tests, completed retrospective experiments, comprehensive classification reports, graph controls, independent artifact verification, and automatic visualization generation.

**Evaluation Status:** Exposed retrospective diagnostic analysis — not a new blind or prospective test.

**Primary Goal:** Reproducible and scientifically transparent comparison of machine learning and graph neural network methods for illicit Bitcoin transaction detection.
