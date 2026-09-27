"""Build educational DS corpus files, manifest, and pilot question set (250).

Run from repo root:
    python scripts/build_edu_pilot_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus_edu_ds"
MANIFEST_PATH = ROOT / "data" / "corpus_manifest.json"
PILOT_PATH = ROOT / "data" / "eval" / "educational_questions_pilot.jsonl"

# ---------------------------------------------------------------------------
# Corpus texts: blank-line-separated paragraphs → passage ids {file}#p{i}
# ---------------------------------------------------------------------------

CORPUS: dict[str, str] = {}

# === data_science ============================================================

CORPUS["data_science_01_pipeline_notes.md"] = """
The data science pipeline is a repeatable sequence of stages used to turn raw observations into decision-ready insights. A common educational framing lists six stages: problem framing, data collection, data cleaning, exploratory analysis, modeling or summarization, and communication of results.

Problem framing defines the decision or scientific question before any modeling begins. Without a clear framing statement, later stages risk optimizing the wrong metric or answering an irrelevant question.

Data collection gathers observations from sensors, surveys, logs, databases, or public repositories. The collection plan should record sampling design, time window, and known coverage gaps so that later analysts can judge external validity.

Data cleaning handles missing values, inconsistent encodings, duplicate rows, and obvious sensor errors. Cleaning decisions should be logged because they change the effective sample and can bias downstream estimates.

Exploratory data analysis (EDA) uses summaries and visualizations to understand distributions, correlations, and anomalies before formal modeling. EDA is intended to generate hypotheses and quality checks, not to replace confirmatory analysis.

Communication translates analytical findings into stakeholder language using tables, charts, and limitations. A complete communication package states assumptions, uncertainty, and what the analysis cannot claim.
""".strip()

CORPUS["data_science_02_data_types_textbook.md"] = """
In introductory data science, variables are commonly classified as numerical or categorical. Numerical variables take quantitative values and may be continuous (any value in an interval) or discrete (countable integers).

Categorical variables take a limited set of labels. Nominal categories have no intrinsic order (for example, browser type), whereas ordinal categories have a meaningful order (for example, Likert agreement levels).

A tidy data table stores each variable in a column, each observation in a row, and each type of observational unit in its own table. Tidy structure simplifies joins, filters, and group-wise summaries.

Feature engineering creates derived predictors from raw fields, such as ratios, bins, time-of-day indicators, or text token counts. Engineered features should be justified by domain knowledge and checked for leakage from the prediction target.

Train-test separation keeps a held-out test set untouched during model fitting so that reported performance estimates remain honest. Using test labels during feature selection or threshold tuning produces overly optimistic scores.

Reproducibility requires recording code versions, random seeds, package versions, and the exact dataset snapshot used for a reported result. Without those artifacts, later auditors cannot regenerate the same numbers.
""".strip()

CORPUS["data_science_03_eda_slides.md"] = """
Exploratory plots commonly include histograms for single numerical variables, bar charts for categorical frequencies, and scatter plots for pairwise numerical relationships. Plot choice should match the variable types involved.

Outliers are observations that fall far from the bulk of the data under a chosen distance or residual rule. Outliers may be errors, rare but valid events, or signs of a mixture of populations.

Missingness mechanisms are often summarized as missing completely at random (MCAR), missing at random (MAR), or missing not at random (MNAR). The mechanism affects whether simple deletion or imputation is defensible.

A correlation coefficient summarizes linear association between two numerical variables but does not by itself prove causation. Confounding variables can induce strong correlations between unrelated quantities.

Cross-tabulation counts how categorical variables co-occur. Large imbalances in a cross-tab can reveal sampling bias or a rare class that will challenge classification models.

EDA documentation should list the filters applied, the date of the extract, and any rows discarded so that exploratory conclusions remain auditable.
""".strip()

CORPUS["data_science_04_ethics_article.md"] = """
Educational data science ethics emphasizes informed consent, purpose limitation, and minimization of collected personal attributes. Collecting more fields than needed increases privacy risk without improving the stated analysis goal.

Proxy variables can encode sensitive attributes indirectly. For example, postal code may correlate with protected class membership and can recreate discriminatory patterns even when the sensitive attribute is removed.

Fairness checks compare error rates or selection rates across demographic groups when such attributes are available for auditing. A single aggregate accuracy number can hide large group disparities.

Transparency requires explaining model purpose, major features, and known failure modes to affected users in accessible language. Opacity is especially problematic when automated scores affect education or employment decisions.

Data retention policies specify how long raw and derived records are stored and when they must be deleted or anonymized. Indefinite retention increases breach impact and conflicts with purpose limitation.

Human oversight keeps a qualified reviewer responsible for high-stakes decisions supported by models. The model advises; institutional policy determines who may act on the advice.
""".strip()

# === statistics ==============================================================

CORPUS["statistics_01_descriptive_notes.md"] = """
Descriptive statistics summarize a sample without drawing formal inferences about a larger population. Common numerical summaries include the mean, median, mode, variance, and standard deviation.

The arithmetic mean is the sum of observations divided by the count of observations. The mean is sensitive to extreme values, so skewed distributions are often also summarized with the median.

The median is the middle value after sorting the sample. For an even number of observations, the median is typically the average of the two central values.

Variance measures average squared deviation from the mean. The sample standard deviation is the square root of the sample variance and is expressed in the same units as the original variable.

A percentile is a value below which a given percentage of observations fall. The interquartile range (IQR) is the distance between the 75th and 25th percentiles and summarizes spread with reduced sensitivity to extremes.

A frequency distribution counts how often values or bins occur. Relative frequencies divide those counts by the sample size so that comparisons across differently sized samples remain meaningful.
""".strip()

CORPUS["statistics_02_inferential_textbook.md"] = """
Inferential statistics use sample data to make probabilistic statements about a population parameter. A point estimate is a single number approximating the parameter; an interval estimate gives a range of plausible values.

A sampling distribution describes how a statistic varies across repeated random samples of the same size from the same population. The standard error is the standard deviation of that sampling distribution.

The central limit theorem states that, for sufficiently large sample size, the sampling distribution of the sample mean is approximately normal even when the population distribution is not normal, under mild conditions.

A confidence interval for a mean combines a point estimate with a margin of error based on the standard error and a critical value from a reference distribution. A 95 percent confidence level is a common educational default, not a universal requirement.

Bias of an estimator is the difference between the expected value of the estimator and the true parameter. Unbiasedness does not guarantee small variance; mean squared error combines bias and variance.

Simple random sampling gives every equally sized subset of the population the same chance of selection. Convenience samples lack that design property and generally do not support classical frequentist coverage claims.
""".strip()

CORPUS["statistics_03_hypothesis_slides.md"] = """
A null hypothesis states a baseline claim about a parameter, often of no effect or no difference. The alternative hypothesis states the claim the study seeks evidence for.

A test statistic summarizes the data in a form whose null distribution is known or well approximated. Extreme values of the test statistic cast doubt on the null hypothesis.

A p-value is the probability, under the null hypothesis, of obtaining a test statistic at least as extreme as the one observed. A small p-value indicates incompatibility between the data and the null model, not the probability that the null is true.

The significance level alpha is a prechosen threshold for Type I error, the probability of rejecting a true null hypothesis. Comparing the p-value to alpha is one rule for a binary reject / fail-to-reject decision.

A Type II error fails to reject a false null hypothesis. Statistical power is one minus the Type II error probability and increases with sample size, effect size, and reduced noise.

Statistical significance is not the same as practical importance. A tiny effect can be significant in a huge sample while a meaningful effect can be nonsignificant in a small sample.
""".strip()

CORPUS["statistics_04_regression_oer.md"] = """
Simple linear regression models a response variable as a linear function of one predictor plus an error term. The slope coefficient describes the expected change in the response for a one-unit increase in the predictor.

Ordinary least squares chooses the intercept and slope that minimize the sum of squared residuals. Residuals are observed responses minus fitted values.

R-squared measures the proportion of variance in the response explained by the fitted model. High R-squared does not prove that the model is correctly specified or causal.

Multiple linear regression includes several predictors at once. Coefficients are interpreted as associations holding the other included predictors fixed, which depends on which covariates are in the model.

Assumptions commonly discussed in introductory courses include linearity, independent errors, roughly constant error variance, and approximate normality of errors for inference. Residual plots help check several of these assumptions visually.

Extrapolating a regression prediction far outside the observed predictor range is unreliable because the fitted linear relationship may not continue. Reporting the training range alongside predictions reduces misuse.
""".strip()

# === machine_learning ========================================================

CORPUS["machine_learning_01_supervised_notes.md"] = """
Supervised learning trains a model on examples that include both input features and a known target label. The fitted model is then used to predict labels for new unlabeled inputs.

Classification predicts a discrete class label, such as spam versus not spam. Regression in the machine-learning sense predicts a continuous numeric target.

A training set is used to fit parameters. A validation set is used to compare candidate models or hyperparameters. A test set is used once for a final unbiased performance estimate.

Overfitting occurs when a model captures noise or idiosyncrasies of the training sample and then generalizes poorly to new data. Underfitting occurs when the model is too simple to capture the systematic pattern.

Cross-validation partitions the training data into folds so that each fold serves once as a temporary validation set. k-fold cross-validation is a standard educational procedure for estimating generalization under data scarcity.

Feature scaling methods such as standardization rescale numeric inputs so that algorithms sensitive to feature magnitude, including many distance-based methods, are not dominated by large-range variables.
""".strip()

CORPUS["machine_learning_02_unsupervised_textbook.md"] = """
Unsupervised learning finds structure in unlabeled data. Common tasks include clustering, dimensionality reduction, and density estimation.

Clustering assigns observations to groups so that items in the same group are more similar to each other than to items in other groups, under a chosen similarity measure.

K-means clustering partitions data into a fixed number k of clusters by iteratively assigning points to the nearest centroid and updating centroids as group means. The algorithm requires a numeric feature space and a chosen k.

Dimensionality reduction projects high-dimensional data into fewer coordinates while preserving selected structure. Principal component analysis (PCA) finds orthogonal directions of maximal variance.

A distance metric such as Euclidean distance defines geometric closeness in feature space. Changing the metric or the feature scaling can change unsupervised results substantially.

Evaluation of unsupervised models often relies on internal indices, stability across restarts, or downstream task utility, because there is no single ground-truth label for every problem.
""".strip()

CORPUS["machine_learning_03_evaluation_slides.md"] = """
Classification accuracy is the fraction of predictions that match the true labels. Accuracy can be misleading on imbalanced datasets where a majority-class classifier scores well.

Precision is the fraction of predicted positive cases that are truly positive. Recall is the fraction of truly positive cases that the model correctly identifies. There is typically a trade-off between precision and recall.

The F1 score is the harmonic mean of precision and recall. It is often used when both false positives and false negatives matter and class imbalance makes accuracy uninformative.

A confusion matrix tabulates true positives, false positives, true negatives, and false negatives. Many scalar metrics can be derived from the confusion matrix alone.

A receiver operating characteristic (ROC) curve plots true positive rate against false positive rate across decision thresholds. The area under the ROC curve (AUC) summarizes ranking quality.

For regression tasks, mean squared error and mean absolute error are common loss summaries. Mean absolute error is less sensitive to large outliers than squared error.
""".strip()

CORPUS["machine_learning_04_overfitting_article.md"] = """
Regularization adds a penalty on model complexity to the training objective. L2 regularization penalizes the sum of squared weights; L1 regularization penalizes the sum of absolute weights and can drive some weights to zero.

Early stopping ends training when validation performance stops improving, reducing the chance that continued optimization fits training noise. It is widely used with iterative learners such as neural networks and gradient boosting.

A learning curve plots training and validation scores against training set size or training iteration. A large persistent gap between training and validation performance suggests overfitting.

Data leakage occurs when information from the test distribution improperly influences training, for example by scaling features using the full dataset before splitting. Leakage produces optimistic offline metrics that fail in deployment.

Hyperparameter search should be confined to training and validation folds. Selecting hyperparameters on the test set converts the test set into a silent validation set and breaks the final evaluation contract.

Ensemble methods combine multiple base models to improve stability or accuracy. Bagging reduces variance by averaging bootstrap-trained models; boosting builds models sequentially to correct previous errors.
""".strip()

# === data_mining =============================================================

CORPUS["data_mining_01_intro_notes.md"] = """
Data mining extracts previously unknown, potentially useful patterns from large datasets. It sits at the intersection of databases, statistics, and machine learning, with emphasis on scalable pattern discovery.

The knowledge discovery in databases (KDD) process typically includes selection, preprocessing, transformation, data mining, and interpretation or evaluation of patterns. Mining is one stage inside a broader discovery workflow.

Descriptive data mining summarizes data through patterns such as associations or clusters. Predictive data mining builds models that forecast unknown values or future events.

Pattern interestingness depends on criteria such as support, confidence, novelty, and actionability. A statistically frequent pattern may still be uninteresting if it is obvious to domain experts.

Scalability matters because mining algorithms may scan data structures larger than memory. Educational examples often use small tables, but real deployments require attention to I/O and indexing.

Privacy-aware mining seeks patterns without exposing individual records. Aggregation thresholds and suppression of rare cells are elementary safeguards taught in introductory courses.
""".strip()

CORPUS["data_mining_02_association_textbook.md"] = """
Association rule mining finds implications of the form antecedent implies consequent in transactional data. A classic educational example is market-basket analysis of items purchased together.

Support of an itemset is the fraction of transactions that contain all items in the set. Minimum support filters out rare itemsets before rules are formed.

Confidence of a rule X implies Y is the support of X union Y divided by the support of X. Confidence estimates how often Y appears among transactions that contain X.

Lift compares a rule's confidence to the baseline frequency of the consequent. Lift greater than one suggests a positive association beyond chance co-occurrence under independence.

The Apriori principle states that all subsets of a frequent itemset must themselves be frequent. Apriori uses this monotonicity to prune the candidate search space.

Association rules are not causal statements. A strong rule can reflect confounding, store layout, or marketing bundles rather than a mechanism that forces the consequent.
""".strip()

CORPUS["data_mining_03_clustering_slides.md"] = """
In data mining courses, clustering is used to segment customers, documents, or events without predefined class labels. Cluster quality depends on features, distance, and the algorithm's inductive bias.

Hierarchical clustering builds a tree of merges or splits. Agglomerative hierarchical clustering starts with each point alone and merges closest clusters until a stopping rule is met.

A dendrogram visualizes hierarchical merges. Cutting the dendrogram at a height yields a flat partition with a chosen number of clusters.

Density-based methods such as DBSCAN group points in dense regions and can label sparse points as noise. Unlike k-means, DBSCAN does not require specifying the number of clusters in advance.

The silhouette score measures how similar a point is to its own cluster compared with the nearest other cluster. Average silhouette width is often used to compare candidate values of k.

Cluster labels are arbitrary identifiers. Two runs can discover the same partition with permuted label numbers; evaluation should compare partitions, not raw label integers.
""".strip()

CORPUS["data_mining_04_process_article.md"] = """
CRISP-DM is a widely taught process model for data mining projects. Its phases are business understanding, data understanding, data preparation, modeling, evaluation, and deployment.

Business understanding translates organizational goals into mining objectives and success criteria. Skipping this phase produces technically valid models that do not answer the decision need.

Data preparation includes cleaning, constructing features, and formatting tables for algorithms. In many projects, preparation consumes more calendar time than model fitting.

Evaluation in CRISP-DM judges whether the model meets business success criteria, not only offline metric thresholds. A high F1 score can still fail if the operating cost of errors is unacceptable.

Deployment integrates a model into an operational workflow with monitoring for performance drift. Educational projects often stop at evaluation, but deployment is required for real impact.

Documentation produced across CRISP-DM phases forms an audit trail of assumptions, exclusions, and parameter choices. Without documentation, later teams cannot safely retrain or retire the model.
""".strip()

# === probability =============================================================

CORPUS["probability_01_basics_notes.md"] = """
Probability quantifies uncertainty for random experiments with well-defined outcomes. The sample space is the set of all possible outcomes of the experiment.

An event is a subset of the sample space. The probability of an event is a number between zero and one inclusive, with the entire sample space having probability one under a complete model.

Two events are mutually exclusive if they cannot occur together, that is, their intersection is empty. For mutually exclusive events, the probability of their union equals the sum of their probabilities.

The complement rule states that the probability of an event not occurring equals one minus the probability that it occurs. Complements are useful when the complementary event is easier to count.

A uniform probability model on a finite sample space assigns equal probability to each outcome. Under uniformity, event probabilities reduce to favorable counts divided by the size of the sample space.

Independence of two events means that the occurrence of one does not change the probability of the other. For independent events A and B, P(A and B) equals P(A) times P(B).
""".strip()

CORPUS["probability_02_distributions_textbook.md"] = """
A discrete random variable takes a countable set of values, each with a probability mass. The probability mass function (PMF) assigns those probabilities and must sum to one.

A continuous random variable is described by a probability density function (PDF). Probabilities for continuous variables are areas under the PDF over intervals; the probability of any single exact point is zero under a continuous density.

The expected value of a discrete random variable is the sum of each value times its probability. For a continuous variable, the expectation is an integral of value times density.

Variance of a random variable is the expected squared deviation from its mean. The standard deviation is the square root of variance and shares the variable's measurement units.

The Bernoulli distribution models a single success/failure trial with success probability p. The binomial distribution models the number of successes in n independent Bernoulli trials with the same p.

The normal distribution is a continuous bell-shaped family parameterized by mean and variance. Many introductory approximations invoke normality for sums of random variables via the central limit theorem.
""".strip()

CORPUS["probability_03_bayes_slides.md"] = """
Conditional probability P(A given B) is the probability of A restricted to the information that B occurred. When P(B) is positive, P(A given B) equals P(A and B) divided by P(B).

Bayes' theorem relates the posterior probability of a hypothesis given data to the prior and the likelihood. In odds form, posterior odds equal prior odds times the likelihood ratio.

The law of total probability partitions the sample space into mutually exclusive scenarios and reconstructs an event probability as a weighted sum across those scenarios.

A prior distribution encodes beliefs about a parameter before observing the current dataset. A posterior distribution updates that belief after conditioning on the observed data.

Base-rate neglect is a reasoning error that ignores the prior frequency of a class when interpreting a diagnostic test. Bayes' theorem makes the base rate an explicit input.

Medical testing examples in textbooks often show that even a high-sensitivity, high-specificity test can yield modest positive predictive value when the condition is rare.
""".strip()

CORPUS["probability_04_conditional_oer.md"] = """
Joint probability describes the simultaneous occurrence of two events or the joint distribution of two random variables. Marginal probabilities are recovered by summing or integrating out the other variable.

Conditional independence states that two variables are independent once a third variable is known. Conditional independence is a central idea in Bayesian networks taught in applied probability courses.

The chain rule factors a joint probability into a product of conditional probabilities. For three events, P(A and B and C) equals P(A) times P(B given A) times P(C given A and B).

Covarying random variables may be dependent even if their correlation is zero when the relationship is nonlinear. Uncorrelatedness is weaker than independence.

Counting tools such as permutations and combinations support uniform probability calculations on finite spaces. Combinations ignore order; permutations preserve order.

Simulation approximates probabilities by generating many random outcomes under a specified model and measuring relative frequencies. Monte Carlo estimates improve with more independent trials, subject to sampling variability.
""".strip()


MANIFEST = [
    {"doc_id": "ds_pipeline_notes", "course": "data_science", "source_type": "lecture_notes", "title": "Data Science Pipeline Stages", "license": "CC0-1.0", "url": "local://corpus_edu_ds/data_science_01_pipeline_notes.md", "version_date": "2026-03-26", "filename": "data_science_01_pipeline_notes.md"},
    {"doc_id": "ds_data_types_textbook", "course": "data_science", "source_type": "textbook", "title": "Variables, Tidy Data, and Reproducibility", "license": "CC0-1.0", "url": "local://corpus_edu_ds/data_science_02_data_types_textbook.md", "version_date": "2026-03-26", "filename": "data_science_02_data_types_textbook.md"},
    {"doc_id": "ds_eda_slides", "course": "data_science", "source_type": "slides", "title": "Exploratory Data Analysis Essentials", "license": "CC0-1.0", "url": "local://corpus_edu_ds/data_science_03_eda_slides.md", "version_date": "2026-03-26", "filename": "data_science_03_eda_slides.md"},
    {"doc_id": "ds_ethics_article", "course": "data_science", "source_type": "article", "title": "Ethics and Governance in Educational Data Science", "license": "CC0-1.0", "url": "local://corpus_edu_ds/data_science_04_ethics_article.md", "version_date": "2026-03-26", "filename": "data_science_04_ethics_article.md"},
    {"doc_id": "stat_descriptive_notes", "course": "statistics", "source_type": "lecture_notes", "title": "Descriptive Statistics", "license": "CC0-1.0", "url": "local://corpus_edu_ds/statistics_01_descriptive_notes.md", "version_date": "2026-03-26", "filename": "statistics_01_descriptive_notes.md"},
    {"doc_id": "stat_inferential_textbook", "course": "statistics", "source_type": "textbook", "title": "Inferential Statistics and Sampling", "license": "CC0-1.0", "url": "local://corpus_edu_ds/statistics_02_inferential_textbook.md", "version_date": "2026-03-26", "filename": "statistics_02_inferential_textbook.md"},
    {"doc_id": "stat_hypothesis_slides", "course": "statistics", "source_type": "slides", "title": "Hypothesis Testing", "license": "CC0-1.0", "url": "local://corpus_edu_ds/statistics_03_hypothesis_slides.md", "version_date": "2026-03-26", "filename": "statistics_03_hypothesis_slides.md"},
    {"doc_id": "stat_regression_oer", "course": "statistics", "source_type": "oer", "title": "Linear Regression Basics", "license": "CC0-1.0", "url": "local://corpus_edu_ds/statistics_04_regression_oer.md", "version_date": "2026-03-26", "filename": "statistics_04_regression_oer.md"},
    {"doc_id": "ml_supervised_notes", "course": "machine_learning", "source_type": "lecture_notes", "title": "Supervised Learning Foundations", "license": "CC0-1.0", "url": "local://corpus_edu_ds/machine_learning_01_supervised_notes.md", "version_date": "2026-03-26", "filename": "machine_learning_01_supervised_notes.md"},
    {"doc_id": "ml_unsupervised_textbook", "course": "machine_learning", "source_type": "textbook", "title": "Unsupervised Learning", "license": "CC0-1.0", "url": "local://corpus_edu_ds/machine_learning_02_unsupervised_textbook.md", "version_date": "2026-03-26", "filename": "machine_learning_02_unsupervised_textbook.md"},
    {"doc_id": "ml_evaluation_slides", "course": "machine_learning", "source_type": "slides", "title": "Model Evaluation Metrics", "license": "CC0-1.0", "url": "local://corpus_edu_ds/machine_learning_03_evaluation_slides.md", "version_date": "2026-03-26", "filename": "machine_learning_03_evaluation_slides.md"},
    {"doc_id": "ml_overfitting_article", "course": "machine_learning", "source_type": "article", "title": "Overfitting, Regularization, and Leakage", "license": "CC0-1.0", "url": "local://corpus_edu_ds/machine_learning_04_overfitting_article.md", "version_date": "2026-03-26", "filename": "machine_learning_04_overfitting_article.md"},
    {"doc_id": "dm_intro_notes", "course": "data_mining", "source_type": "lecture_notes", "title": "Introduction to Data Mining and KDD", "license": "CC0-1.0", "url": "local://corpus_edu_ds/data_mining_01_intro_notes.md", "version_date": "2026-03-26", "filename": "data_mining_01_intro_notes.md"},
    {"doc_id": "dm_association_textbook", "course": "data_mining", "source_type": "textbook", "title": "Association Rule Mining", "license": "CC0-1.0", "url": "local://corpus_edu_ds/data_mining_02_association_textbook.md", "version_date": "2026-03-26", "filename": "data_mining_02_association_textbook.md"},
    {"doc_id": "dm_clustering_slides", "course": "data_mining", "source_type": "slides", "title": "Clustering Methods in Data Mining", "license": "CC0-1.0", "url": "local://corpus_edu_ds/data_mining_03_clustering_slides.md", "version_date": "2026-03-26", "filename": "data_mining_03_clustering_slides.md"},
    {"doc_id": "dm_process_article", "course": "data_mining", "source_type": "article", "title": "CRISP-DM Process Model", "license": "CC0-1.0", "url": "local://corpus_edu_ds/data_mining_04_process_article.md", "version_date": "2026-03-26", "filename": "data_mining_04_process_article.md"},
    {"doc_id": "prob_basics_notes", "course": "probability", "source_type": "lecture_notes", "title": "Probability Basics", "license": "CC0-1.0", "url": "local://corpus_edu_ds/probability_01_basics_notes.md", "version_date": "2026-03-26", "filename": "probability_01_basics_notes.md"},
    {"doc_id": "prob_distributions_textbook", "course": "probability", "source_type": "textbook", "title": "Random Variables and Distributions", "license": "CC0-1.0", "url": "local://corpus_edu_ds/probability_02_distributions_textbook.md", "version_date": "2026-03-26", "filename": "probability_02_distributions_textbook.md"},
    {"doc_id": "prob_bayes_slides", "course": "probability", "source_type": "slides", "title": "Conditional Probability and Bayes Theorem", "license": "CC0-1.0", "url": "local://corpus_edu_ds/probability_03_bayes_slides.md", "version_date": "2026-03-26", "filename": "probability_03_bayes_slides.md"},
    {"doc_id": "prob_conditional_oer", "course": "probability", "source_type": "oer", "title": "Joint Probability and Simulation", "license": "CC0-1.0", "url": "local://corpus_edu_ds/probability_04_conditional_oer.md", "version_date": "2026-03-26", "filename": "probability_04_conditional_oer.md"},
]


def passage_id(filename: str, p: int) -> str:
    return f"{filename}#p{p}"


def evidence(filename: str, p: int, text: str) -> list[dict]:
    return [{"doc_id": filename, "passage_id": passage_id(filename, p), "text": text.strip()}]


def q(
    qid: str,
    course: str,
    category: str,
    question: str,
    *,
    gold_answers: list[str] | None = None,
    source_documents: list[str] | None = None,
    source_type: str | None = None,
    evidence_spans: list[dict] | None = None,
    reference_answer: str | None = None,
    difficulty: str = "medium",
    question_type: str = "factual",
    construction_method: str = "corpus_grounded",
    false_premise: str | None = None,
    missing_aspects: str | None = None,
) -> dict:
    answerable = category == "answerable"
    if category == "answerable":
        expected_action = "ANSWER"
    elif category == "underspecified":
        expected_action = "CLARIFY"
    else:
        expected_action = "ABSTAIN"
    row = {
        "id": qid,
        "question_id": qid,
        "question": question,
        "answerable": answerable,
        "category": category,
        "expected_action": expected_action,
        "gold_answers": gold_answers or [],
        "title": course,
        "course": course,
        "source": "edu_rag_pilot_v1",
        "source_documents": source_documents or [],
        "source_type": source_type,
        "evidence_spans": evidence_spans or [],
        "reference_answer": reference_answer,
        "difficulty": difficulty,
        "question_type": question_type,
        "construction_method": construction_method,
        "split": "pilot",
    }
    if false_premise:
        row["false_premise"] = false_premise
    if missing_aspects:
        row["missing_aspects"] = missing_aspects
    return row


def paras(filename: str) -> list[str]:
    return [p.strip() for p in CORPUS[filename].split("\n\n") if p.strip()]


def build_pilot_questions() -> list[dict]:
    """50 questions per course: 20 ans, 8 ook, 8 ood, 7 fp, 7 underspecified."""
    rows: list[dict] = []

    # ---- data_science ------------------------------------------------------
    c = "data_science"
    f1, f2, f3, f4 = (
        "data_science_01_pipeline_notes.md",
        "data_science_02_data_types_textbook.md",
        "data_science_03_eda_slides.md",
        "data_science_04_ethics_article.md",
    )
    p1, p2, p3, p4 = paras(f1), paras(f2), paras(f3), paras(f4)
    ans = [
        ("What are the six common stages of the data science pipeline described in the notes?", ["problem framing, data collection, data cleaning, exploratory analysis, modeling or summarization, and communication"], f1, 0, p1[0], "lecture_notes", "easy", "factual"),
        ("Why should problem framing happen before modeling begins?", ["to avoid optimizing the wrong metric or answering an irrelevant question"], f1, 1, p1[1], "lecture_notes", "medium", "reasoning"),
        ("What should a data collection plan record according to the lecture notes?", ["sampling design, time window, and known coverage gaps"], f1, 2, p1[2], "lecture_notes", "medium", "factual"),
        ("Why should data cleaning decisions be logged?", ["because they change the effective sample and can bias downstream estimates"], f1, 3, p1[3], "lecture_notes", "medium", "reasoning"),
        ("What is exploratory data analysis intended to generate before formal modeling?", ["hypotheses and quality checks"], f1, 4, p1[4], "lecture_notes", "easy", "factual"),
        ("What must a complete communication package state?", ["assumptions, uncertainty, and what the analysis cannot claim"], f1, 5, p1[5], "lecture_notes", "medium", "factual"),
        ("How are numerical variables commonly classified in introductory data science?", ["continuous or discrete"], f2, 0, p2[0], "textbook", "easy", "factual"),
        ("What is the difference between nominal and ordinal categorical variables?", ["nominal categories have no intrinsic order; ordinal categories have a meaningful order"], f2, 1, p2[1], "textbook", "medium", "comparison"),
        ("What does a tidy data table store in each column, row, and table?", ["each variable in a column, each observation in a row, and each type of observational unit in its own table"], f2, 2, p2[2], "textbook", "medium", "factual"),
        ("Why must a held-out test set remain untouched during model fitting?", ["so that reported performance estimates remain honest"], f2, 4, p2[4], "textbook", "medium", "reasoning"),
        ("What artifacts does reproducibility require recording?", ["code versions, random seeds, package versions, and the exact dataset snapshot"], f2, 5, p2[5], "textbook", "medium", "factual"),
        ("Which plot types are commonly used for numerical, categorical, and pairwise numerical relationships in EDA?", ["histograms, bar charts, and scatter plots"], f3, 0, p3[0], "slides", "easy", "factual"),
        ("What are three commonly summarized missingness mechanisms?", ["MCAR, MAR, and MNAR", "missing completely at random, missing at random, or missing not at random"], f3, 2, p3[2], "slides", "hard", "definition"),
        ("Does a correlation coefficient by itself prove causation?", ["no"], f3, 3, p3[3], "slides", "easy", "factual"),
        ("What can large imbalances in a cross-tabulation reveal?", ["sampling bias or a rare class"], f3, 4, p3[4], "slides", "medium", "factual"),
        ("What ethical principles does educational data science ethics emphasize?", ["informed consent, purpose limitation, and minimization of collected personal attributes"], f4, 0, p4[0], "article", "medium", "factual"),
        ("How can postal code act as a proxy variable?", ["it may correlate with protected class membership and recreate discriminatory patterns"], f4, 1, p4[1], "article", "hard", "reasoning"),
        ("Why can a single aggregate accuracy number be misleading for fairness?", ["it can hide large group disparities"], f4, 2, p4[2], "article", "medium", "reasoning"),
        ("What do data retention policies specify?", ["how long raw and derived records are stored and when they must be deleted or anonymized"], f4, 4, p4[4], "article", "medium", "factual"),
        ("In high-stakes settings, what role should the model play relative to human oversight?", ["the model advises; institutional policy determines who may act"], f4, 5, p4[5], "article", "medium", "reasoning"),
    ]
    for i, (question, gold, fn, pi, text, st, diff, qt) in enumerate(ans, 1):
        rows.append(q(f"pilot-ds-a{i:02d}", c, "answerable", question, gold_answers=gold, source_documents=[fn], source_type=st, evidence_spans=evidence(fn, pi, text), reference_answer=gold[0], difficulty=diff, question_type=qt))

    ook = [
        "What Python library version does the course require for pandas in the final project?",
        "Which company donated the dataset used in the data science midterm case study?",
        "What is the exact grade weighting for the ethics essay in this course?",
        "How many gigabytes of RAM does the recommended student laptop need?",
        "What is the name of the university LMS module where homework is submitted?",
        "Which week of the semester covers Spark cluster administration?",
        "What is the instructor's office hour Zoom meeting ID?",
        "Which industry partner hosts the optional data science internship fair?",
    ]
    for i, question in enumerate(ook, 1):
        rows.append(q(f"pilot-ds-ook{i:02d}", c, "out_of_knowledge", question, construction_method="cross_doc_gap", difficulty="medium", question_type="factual"))

    ood = [
        "What is the capital city of Australia?",
        "Who won the FIFA World Cup in 2018?",
        "What is the boiling point of water in degrees Fahrenheit at standard pressure?",
        "Which Shakespeare play features the character Iago?",
        "What is the chemical formula for table salt?",
        "How many players are on the field for one soccer team during a match?",
        "What year did the Berlin Wall fall?",
        "What is the primary ingredient in traditional pesto sauce?",
    ]
    for i, question in enumerate(ood, 1):
        rows.append(q(f"pilot-ds-ood{i:02d}", c, "out_of_scope", question, construction_method="ood", difficulty="easy", question_type="factual"))

    fp = [
        ("Why does the data science pipeline begin with model deployment before problem framing?", "Pipeline begins with deployment first", "The notes list problem framing first, not deployment."),
        ("Why are nominal categorical variables defined by having a meaningful ranked order?", "Nominal variables are ordered", "Nominal categories have no intrinsic order; ordinal ones do."),
        ("Why does a correlation coefficient prove that one variable causes another?", "Correlation proves causation", "The slides state correlation does not prove causation."),
        ("Why is indefinite retention of all personal fields required by purpose limitation?", "Indefinite retention is required", "Purpose limitation and retention policies argue against indefinite retention."),
        ("Why should the held-out test set be used during feature selection to improve honesty?", "Test set should be used in feature selection", "Test labels must remain untouched during fitting and related choices."),
        ("Why does EDA replace confirmatory analysis entirely in the pipeline notes?", "EDA replaces confirmatory analysis", "EDA generates hypotheses and checks; it does not replace confirmatory analysis."),
        ("Why are proxy variables guaranteed to remove all fairness risks when the sensitive attribute is dropped?", "Dropping the sensitive attribute removes fairness risk", "Proxies can recreate discriminatory patterns even after removal."),
    ]
    for i, (question, premise, note) in enumerate(fp, 1):
        rows.append(q(f"pilot-ds-fp{i:02d}", c, "false_presupposition", question, construction_method="premise_flip", difficulty="medium", question_type="reasoning", false_premise=f"{premise}. {note}"))

    us = [
        ("What is the next stage?", "Which pipeline stage is current / which document context?"),
        ("How should we clean it?", "What dataset, which missingness issues, which cleaning policy?"),
        ("Is this fair?", "Which model, which groups, which fairness metric?"),
        ("Which plot should I use?", "Which variable types and analysis goal?"),
        ("What does the correlation mean?", "Which variables and dataset are meant?"),
        ("Should we keep the fields?", "Which fields, which purpose, which retention rule?"),
        ("Was the analysis reproducible?", "Which analysis run and which artifacts are missing?"),
    ]
    for i, (question, missing) in enumerate(us, 1):
        rows.append(q(f"pilot-ds-us{i:02d}", c, "underspecified", question, construction_method="underspecify", difficulty="easy", question_type="factual", missing_aspects=missing))

    # ---- statistics --------------------------------------------------------
    c = "statistics"
    f1, f2, f3, f4 = (
        "statistics_01_descriptive_notes.md",
        "statistics_02_inferential_textbook.md",
        "statistics_03_hypothesis_slides.md",
        "statistics_04_regression_oer.md",
    )
    p1, p2, p3, p4 = paras(f1), paras(f2), paras(f3), paras(f4)
    ans = [
        ("What do descriptive statistics summarize without drawing formal inferences about a population?", ["a sample"], f1, 0, p1[0], "lecture_notes", "easy", "factual"),
        ("Why is the mean sensitive to extreme values?", ["because extreme values pull the average"], f1, 1, p1[1], "lecture_notes", "medium", "reasoning"),
        ("How is the median defined for an even number of observations?", ["the average of the two central values"], f1, 2, p1[2], "lecture_notes", "medium", "factual"),
        ("What is the sample standard deviation in relation to the sample variance?", ["the square root of the sample variance"], f1, 3, p1[3], "lecture_notes", "easy", "factual"),
        ("What is the interquartile range (IQR)?", ["the distance between the 75th and 25th percentiles"], f1, 4, p1[4], "lecture_notes", "easy", "definition"),
        ("How are relative frequencies computed from frequency counts?", ["divide counts by the sample size"], f1, 5, p1[5], "lecture_notes", "easy", "factual"),
        ("What is a point estimate?", ["a single number approximating a population parameter"], f2, 0, p2[0], "textbook", "easy", "definition"),
        ("What is the standard error?", ["the standard deviation of the sampling distribution"], f2, 1, p2[1], "textbook", "medium", "definition"),
        ("What does the central limit theorem say about the sampling distribution of the sample mean for large n?", ["it is approximately normal even when the population is not normal, under mild conditions"], f2, 2, p2[2], "textbook", "hard", "reasoning"),
        ("What does a 95 percent confidence level represent in introductory teaching?", ["a common educational default, not a universal requirement"], f2, 3, p2[3], "textbook", "medium", "factual"),
        ("What is bias of an estimator?", ["the difference between the expected value of the estimator and the true parameter"], f2, 4, p2[4], "textbook", "medium", "definition"),
        ("What property does simple random sampling give equally sized subsets?", ["the same chance of selection"], f2, 5, p2[5], "textbook", "medium", "factual"),
        ("What does the null hypothesis typically state?", ["a baseline claim about a parameter, often of no effect or no difference"], f3, 0, p3[0], "slides", "easy", "factual"),
        ("What is a p-value?", ["the probability, under the null hypothesis, of obtaining a test statistic at least as extreme as the one observed"], f3, 2, p3[2], "slides", "hard", "definition"),
        ("What is a Type I error?", ["rejecting a true null hypothesis"], f3, 3, p3[3], "slides", "easy", "definition"),
        ("What is statistical power?", ["one minus the Type II error probability"], f3, 4, p3[4], "slides", "medium", "definition"),
        ("In simple linear regression, what does the slope coefficient describe?", ["the expected change in the response for a one-unit increase in the predictor"], f4, 0, p4[0], "oer", "medium", "factual"),
        ("What does ordinary least squares minimize?", ["the sum of squared residuals"], f4, 1, p4[1], "oer", "easy", "factual"),
        ("Does high R-squared prove that a regression model is causal?", ["no"], f4, 2, p4[2], "oer", "easy", "factual"),
        ("Why is extrapolating far outside the observed predictor range unreliable?", ["the fitted linear relationship may not continue"], f4, 5, p4[5], "oer", "medium", "reasoning"),
    ]
    for i, (question, gold, fn, pi, text, st, diff, qt) in enumerate(ans, 1):
        rows.append(q(f"pilot-stat-a{i:02d}", c, "answerable", question, gold_answers=gold, source_documents=[fn], source_type=st, evidence_spans=evidence(fn, pi, text), reference_answer=gold[0], difficulty=diff, question_type=qt))

    for i, question in enumerate([
        "What textbook edition is required for the statistics midterm?",
        "What is the exact sample size used in the course's blood-pressure case study?",
        "Which R package version must students install for the ANOVA lab?",
        "What is the due date of homework 7 in the current semester?",
        "Who is the guest lecturer for Bayesian methods week?",
        "What is the population mean mu used in the secret exam simulation?",
        "How many tutorial sessions are mandatory for statistics students?",
        "What is the course Discord invite code?",
    ], 1):
        rows.append(q(f"pilot-stat-ook{i:02d}", c, "out_of_knowledge", question, construction_method="cross_doc_gap"))

    for i, question in enumerate([
        "What is the tallest mountain in Africa?",
        "Who painted the Mona Lisa?",
        "What is the currency of Japan?",
        "How many strings does a standard violin have?",
        "What is the main language spoken in Brazil?",
        "Which planet is known for its prominent ring system?",
        "What sport uses a shuttlecock?",
        "What is the chemical symbol for gold?",
    ], 1):
        rows.append(q(f"pilot-stat-ood{i:02d}", c, "out_of_scope", question, construction_method="ood", difficulty="easy"))

    for i, (question, premise) in enumerate([
        ("Why does a p-value equal the probability that the null hypothesis is true?", "p-value is P(null true)"),
        ("Why is statistical significance the same thing as practical importance?", "significance equals practical importance"),
        ("Why does high R-squared prove a regression is correctly specified and causal?", "high R^2 proves causation"),
        ("Why is the mean preferred for all skewed distributions because it ignores extremes?", "mean ignores extremes"),
        ("Why do convenience samples support classical frequentist coverage claims?", "convenience samples support coverage claims"),
        ("Why is Type II error the same as rejecting a true null hypothesis?", "Type II = reject true null"),
        ("Why is extrapolating far outside the predictor range recommended for linear regression?", "extrapolation is recommended"),
    ], 1):
        rows.append(q(f"pilot-stat-fp{i:02d}", c, "false_presupposition", question, construction_method="premise_flip", false_premise=premise))

    for i, (question, missing) in enumerate([
        ("What is the p-value?", "Which test, dataset, and hypotheses?"),
        ("Is the difference significant?", "Which comparison, alpha, and test?"),
        ("What does the slope mean?", "Which regression model and predictors?"),
        ("Should we reject?", "Reject which null, at what alpha?"),
        ("Is the sample large enough?", "Large enough for which procedure and effect?"),
        ("What is the confidence interval?", "Interval for which parameter and confidence level?"),
        ("Which average should we report?", "Mean vs median for which variable and audience?"),
    ], 1):
        rows.append(q(f"pilot-stat-us{i:02d}", c, "underspecified", question, construction_method="underspecify", missing_aspects=missing))

    # ---- machine_learning --------------------------------------------------
    c = "machine_learning"
    f1, f2, f3, f4 = (
        "machine_learning_01_supervised_notes.md",
        "machine_learning_02_unsupervised_textbook.md",
        "machine_learning_03_evaluation_slides.md",
        "machine_learning_04_overfitting_article.md",
    )
    p1, p2, p3, p4 = paras(f1), paras(f2), paras(f3), paras(f4)
    ans = [
        ("What does supervised learning require in the training examples?", ["input features and a known target label"], f1, 0, p1[0], "lecture_notes", "easy", "factual"),
        ("What is the difference between classification and regression in machine learning?", ["classification predicts a discrete class label; regression predicts a continuous numeric target"], f1, 1, p1[1], "lecture_notes", "easy", "comparison"),
        ("What are the roles of training, validation, and test sets?", ["training fits parameters; validation compares models or hyperparameters; test gives a final unbiased estimate"], f1, 2, p1[2], "lecture_notes", "medium", "factual"),
        ("What is overfitting?", ["when a model captures noise or idiosyncrasies of the training sample and generalizes poorly"], f1, 3, p1[3], "lecture_notes", "easy", "definition"),
        ("What does k-fold cross-validation do?", ["partitions training data so each fold serves once as a temporary validation set"], f1, 4, p1[4], "lecture_notes", "medium", "factual"),
        ("Why is feature scaling used for many distance-based methods?", ["so they are not dominated by large-range variables"], f1, 5, p1[5], "lecture_notes", "medium", "reasoning"),
        ("What are common unsupervised learning tasks?", ["clustering, dimensionality reduction, and density estimation"], f2, 0, p2[0], "textbook", "easy", "factual"),
        ("What does k-means require the user to choose?", ["a numeric feature space and a chosen k"], f2, 2, p2[2], "textbook", "medium", "factual"),
        ("What does PCA find?", ["orthogonal directions of maximal variance"], f2, 3, p2[3], "textbook", "medium", "factual"),
        ("Why can changing the distance metric change unsupervised results?", ["the metric defines geometric closeness in feature space"], f2, 4, p2[4], "textbook", "hard", "reasoning"),
        ("Why can accuracy be misleading on imbalanced datasets?", ["a majority-class classifier can score well"], f3, 0, p3[0], "slides", "medium", "reasoning"),
        ("What is precision?", ["the fraction of predicted positive cases that are truly positive"], f3, 1, p3[1], "slides", "easy", "definition"),
        ("What is recall?", ["the fraction of truly positive cases that the model correctly identifies"], f3, 1, p3[1], "slides", "easy", "definition"),
        ("What is the F1 score?", ["the harmonic mean of precision and recall"], f3, 2, p3[2], "slides", "easy", "definition"),
        ("What does a confusion matrix tabulate?", ["true positives, false positives, true negatives, and false negatives"], f3, 3, p3[3], "slides", "easy", "factual"),
        ("What does ROC AUC summarize?", ["ranking quality"], f3, 4, p3[4], "slides", "medium", "factual"),
        ("What does L2 regularization penalize?", ["the sum of squared weights"], f4, 0, p4[0], "article", "easy", "factual"),
        ("What is data leakage in the scaling example given in the article?", ["scaling features using the full dataset before splitting"], f4, 3, p4[3], "article", "hard", "factual"),
        ("Why should hyperparameter search not use the test set?", ["it converts the test set into a silent validation set and breaks final evaluation"], f4, 4, p4[4], "article", "medium", "reasoning"),
        ("How do bagging and boosting differ at a high level?", ["bagging averages bootstrap-trained models to reduce variance; boosting builds models sequentially to correct previous errors"], f4, 5, p4[5], "article", "hard", "comparison"),
    ]
    for i, (question, gold, fn, pi, text, st, diff, qt) in enumerate(ans, 1):
        rows.append(q(f"pilot-ml-a{i:02d}", c, "answerable", question, gold_answers=gold, source_documents=[fn], source_type=st, evidence_spans=evidence(fn, pi, text), reference_answer=gold[0], difficulty=diff, question_type=qt))

    for i, question in enumerate([
        "What learning rate did the instructor use in the unpublished neural net demo?",
        "Which Kaggle competition ID is required for the ML course project?",
        "What is the password for the course GPU cluster?",
        "How many hidden layers does the mandatory autoencoder homework use?",
        "Which paper DOI must be cited in week 9's reading response?",
        "What is the exact class imbalance ratio in the private fraud dataset?",
        "Who sponsors the machine learning hackathon prize?",
        "What is the Zoom link for the optional PyTorch workshop?",
    ], 1):
        rows.append(q(f"pilot-ml-ook{i:02d}", c, "out_of_knowledge", question, construction_method="cross_doc_gap"))

    for i, question in enumerate([
        "What is the national dish of Spain?",
        "Who composed the Four Seasons?",
        "How many continents are there on Earth in the common seven-continent model?",
        "What is the boiling point of liquid nitrogen approximately in Celsius?",
        "Which instrument has black and white keys arranged in octaves?",
        "What is the Olympic motto?",
        "Who wrote Pride and Prejudice?",
        "What gas do plants absorb during photosynthesis according to basic biology?",
    ], 1):
        rows.append(q(f"pilot-ml-ood{i:02d}", c, "out_of_scope", question, construction_method="ood", difficulty="easy"))

    for i, (question, premise) in enumerate([
        ("Why does higher training accuracy always mean better generalization?", "train accuracy implies generalization"),
        ("Why is accuracy the best metric for every imbalanced classification problem?", "accuracy is always best under imbalance"),
        ("Why does unsupervised learning require labeled targets for every training row?", "unsupervised needs labels"),
        ("Why should hyperparameters be chosen on the test set for honest evaluation?", "tune on test set"),
        ("Why does k-means never need the user to choose k?", "k-means needs no k"),
        ("Why is precision defined as the fraction of true positives among all true positive cases?", "precision defined as recall"),
        ("Why does scaling with the full dataset before splitting avoid data leakage?", "full-data scaling avoids leakage"),
    ], 1):
        rows.append(q(f"pilot-ml-fp{i:02d}", c, "false_presupposition", question, construction_method="premise_flip", false_premise=premise))

    for i, (question, missing) in enumerate([
        ("Is the model overfitting?", "Which model, which train/validation curves?"),
        ("What metric should we use?", "Which task, class balance, and error costs?"),
        ("How many clusters are best?", "Best under which index and dataset?"),
        ("Should we regularize?", "Which model family and observed overfitting evidence?"),
        ("Is accuracy good enough?", "Good enough for which deployment thresholds?"),
        ("Which set do we evaluate on?", "Which split and which decision stage?"),
        ("Did leakage occur?", "Which pipeline step and which features?"),
    ], 1):
        rows.append(q(f"pilot-ml-us{i:02d}", c, "underspecified", question, construction_method="underspecify", missing_aspects=missing))

    # ---- data_mining -------------------------------------------------------
    c = "data_mining"
    f1, f2, f3, f4 = (
        "data_mining_01_intro_notes.md",
        "data_mining_02_association_textbook.md",
        "data_mining_03_clustering_slides.md",
        "data_mining_04_process_article.md",
    )
    p1, p2, p3, p4 = paras(f1), paras(f2), paras(f3), paras(f4)
    ans = [
        ("How is data mining characterized in the introductory notes?", ["extracting previously unknown, potentially useful patterns from large datasets"], f1, 0, p1[0], "lecture_notes", "easy", "definition"),
        ("What stages does the KDD process typically include?", ["selection, preprocessing, transformation, data mining, and interpretation or evaluation"], f1, 1, p1[1], "lecture_notes", "medium", "factual"),
        ("What is the difference between descriptive and predictive data mining?", ["descriptive summarizes patterns such as associations or clusters; predictive forecasts unknown values or future events"], f1, 2, p1[2], "lecture_notes", "medium", "comparison"),
        ("What criteria can make a pattern interesting?", ["support, confidence, novelty, and actionability"], f1, 3, p1[3], "lecture_notes", "medium", "factual"),
        ("What is association rule mining looking for?", ["implications of the form antecedent implies consequent in transactional data"], f2, 0, p2[0], "textbook", "easy", "definition"),
        ("What is support of an itemset?", ["the fraction of transactions that contain all items in the set"], f2, 1, p2[1], "textbook", "easy", "definition"),
        ("How is confidence of a rule X implies Y defined?", ["support of X union Y divided by support of X"], f2, 2, p2[2], "textbook", "medium", "definition"),
        ("What does lift greater than one suggest?", ["a positive association beyond chance co-occurrence under independence"], f2, 3, p2[3], "textbook", "medium", "factual"),
        ("What does the Apriori principle state?", ["all subsets of a frequent itemset must themselves be frequent"], f2, 4, p2[4], "textbook", "hard", "factual"),
        ("Are association rules causal statements?", ["no"], f2, 5, p2[5], "textbook", "easy", "factual"),
        ("What does agglomerative hierarchical clustering do?", ["starts with each point alone and merges closest clusters until a stopping rule is met"], f3, 1, p3[1], "slides", "medium", "factual"),
        ("What does cutting a dendrogram at a height yield?", ["a flat partition with a chosen number of clusters"], f3, 2, p3[2], "slides", "medium", "factual"),
        ("How does DBSCAN differ from k-means regarding the number of clusters?", ["DBSCAN does not require specifying the number of clusters in advance"], f3, 3, p3[3], "slides", "medium", "comparison"),
        ("What does the silhouette score measure?", ["how similar a point is to its own cluster compared with the nearest other cluster"], f3, 4, p3[4], "slides", "hard", "definition"),
        ("Why should evaluation compare partitions rather than raw cluster label integers?", ["cluster labels are arbitrary identifiers"], f3, 5, p3[5], "slides", "medium", "reasoning"),
        ("What are the phases of CRISP-DM?", ["business understanding, data understanding, data preparation, modeling, evaluation, and deployment"], f4, 0, p4[0], "article", "medium", "factual"),
        ("What does the business understanding phase translate organizational goals into?", ["mining objectives and success criteria"], f4, 1, p4[1], "article", "medium", "factual"),
        ("In many projects, which phase consumes more calendar time than model fitting?", ["data preparation"], f4, 2, p4[2], "article", "easy", "factual"),
        ("What does evaluation in CRISP-DM judge beyond offline metrics?", ["whether the model meets business success criteria"], f4, 3, p4[3], "article", "medium", "factual"),
        ("What does deployment integrate a model into?", ["an operational workflow with monitoring for performance drift"], f4, 4, p4[4], "article", "medium", "factual"),
    ]
    for i, (question, gold, fn, pi, text, st, diff, qt) in enumerate(ans, 1):
        rows.append(q(f"pilot-dm-a{i:02d}", c, "answerable", question, gold_answers=gold, source_documents=[fn], source_type=st, evidence_spans=evidence(fn, pi, text), reference_answer=gold[0], difficulty=diff, question_type=qt))

    for i, question in enumerate([
        "What minimum support threshold does the hidden exam basket dataset use?",
        "Which supermarket chain donated the transactions for lab 3?",
        "What is the CRISP-DM template filename required for submission?",
        "How many nodes are in the department's Hadoop teaching cluster?",
        "What is the instructor's preferred GUI for association rule demos?",
        "Which week covers graph mining algorithms not listed in the notes?",
        "What is the private GitHub classroom assignment code?",
        "Who grades the data mining oral presentations?",
    ], 1):
        rows.append(q(f"pilot-dm-ook{i:02d}", c, "out_of_knowledge", question, construction_method="cross_doc_gap"))

    for i, question in enumerate([
        "What is the largest desert on Earth by area?",
        "Who discovered penicillin?",
        "What is the square root of 144?",
        "Which ocean is the deepest?",
        "What is the primary language of Egypt?",
        "How many degrees are in a right angle?",
        "What instrument family does the trumpet belong to?",
        "Who was the first President of the United States?",
    ], 1):
        rows.append(q(f"pilot-dm-ood{i:02d}", c, "out_of_scope", question, construction_method="ood", difficulty="easy"))

    for i, (question, premise) in enumerate([
        ("Why are association rules guaranteed to be causal mechanisms?", "association rules are causal"),
        ("Why does DBSCAN require the user to pre-specify the number of clusters k?", "DBSCAN requires k"),
        ("Why is support defined as confidence divided by lift in the textbook?", "support = confidence/lift"),
        ("Why does CRISP-DM skip business understanding because modeling comes first?", "modeling precedes business understanding"),
        ("Why are cluster label integers comparable across runs without alignment?", "raw labels are comparable across runs"),
        ("Why is a frequent pattern always interesting to domain experts?", "frequent implies interesting"),
        ("Why does lift less than one always indicate a positive association beyond chance?", "lift<1 means positive association"),
    ], 1):
        rows.append(q(f"pilot-dm-fp{i:02d}", c, "false_presupposition", question, construction_method="premise_flip", false_premise=premise))

    for i, (question, missing) in enumerate([
        ("Is this rule interesting?", "Which rule, which interestingness criteria?"),
        ("What support should we use?", "Support for which dataset and mining goal?"),
        ("How many clusters?", "For which algorithm, features, and validity index?"),
        ("Are we ready to deploy?", "Which model and which business success criteria?"),
        ("Is the pattern actionable?", "Actionable for which stakeholder decision?"),
        ("Which CRISP-DM phase are we in?", "Which project artifacts define the current phase?"),
        ("Should we prune candidates?", "Prune under which algorithm and thresholds?"),
    ], 1):
        rows.append(q(f"pilot-dm-us{i:02d}", c, "underspecified", question, construction_method="underspecify", missing_aspects=missing))

    # ---- probability -------------------------------------------------------
    c = "probability"
    f1, f2, f3, f4 = (
        "probability_01_basics_notes.md",
        "probability_02_distributions_textbook.md",
        "probability_03_bayes_slides.md",
        "probability_04_conditional_oer.md",
    )
    p1, p2, p3, p4 = paras(f1), paras(f2), paras(f3), paras(f4)
    ans = [
        ("What is the sample space of a random experiment?", ["the set of all possible outcomes"], f1, 0, p1[0], "lecture_notes", "easy", "definition"),
        ("What probability does the entire sample space have under a complete model?", ["one"], f1, 1, p1[1], "lecture_notes", "easy", "factual"),
        ("When are two events mutually exclusive?", ["if they cannot occur together / their intersection is empty"], f1, 2, p1[2], "lecture_notes", "easy", "definition"),
        ("What does the complement rule state?", ["the probability of an event not occurring equals one minus the probability that it occurs"], f1, 3, p1[3], "lecture_notes", "easy", "factual"),
        ("Under a uniform finite model, how are event probabilities computed?", ["favorable counts divided by the size of the sample space"], f1, 4, p1[4], "lecture_notes", "medium", "factual"),
        ("For independent events A and B, what equals P(A and B)?", ["P(A) times P(B)"], f1, 5, p1[5], "lecture_notes", "easy", "factual"),
        ("What must a probability mass function sum to?", ["one"], f2, 0, p2[0], "textbook", "easy", "factual"),
        ("For a continuous random variable with a density, what is the probability of any single exact point?", ["zero"], f2, 1, p2[1], "textbook", "medium", "factual"),
        ("How is the expected value of a discrete random variable computed?", ["the sum of each value times its probability"], f2, 2, p2[2], "textbook", "easy", "factual"),
        ("What does the Bernoulli distribution model?", ["a single success/failure trial with success probability p"], f2, 4, p2[4], "textbook", "easy", "definition"),
        ("What does the binomial distribution model?", ["the number of successes in n independent Bernoulli trials with the same p"], f2, 4, p2[4], "textbook", "medium", "definition"),
        ("How is conditional probability P(A given B) defined when P(B) is positive?", ["P(A and B) divided by P(B)"], f3, 0, p3[0], "slides", "easy", "definition"),
        ("In odds form, what does Bayes' theorem say about posterior odds?", ["posterior odds equal prior odds times the likelihood ratio"], f3, 1, p3[1], "slides", "hard", "factual"),
        ("What does a prior distribution encode?", ["beliefs about a parameter before observing the current dataset"], f3, 3, p3[3], "slides", "medium", "factual"),
        ("What is base-rate neglect?", ["ignoring the prior frequency of a class when interpreting a diagnostic test"], f3, 4, p3[4], "slides", "medium", "definition"),
        ("Why can a high-sensitivity, high-specificity test still have modest positive predictive value?", ["when the condition is rare"], f3, 5, p3[5], "slides", "hard", "reasoning"),
        ("How are marginal probabilities recovered from a joint distribution?", ["by summing or integrating out the other variable"], f4, 0, p4[0], "oer", "medium", "factual"),
        ("What does the chain rule state for three events A, B, and C?", ["P(A and B and C) equals P(A) times P(B given A) times P(C given A and B)"], f4, 2, p4[2], "oer", "hard", "factual"),
        ("What is the difference between combinations and permutations in counting?", ["combinations ignore order; permutations preserve order"], f4, 4, p4[4], "oer", "easy", "comparison"),
        ("How does Monte Carlo simulation approximate probabilities?", ["by generating many random outcomes and measuring relative frequencies"], f4, 5, p4[5], "oer", "medium", "factual"),
    ]
    for i, (question, gold, fn, pi, text, st, diff, qt) in enumerate(ans, 1):
        rows.append(q(f"pilot-prob-a{i:02d}", c, "answerable", question, gold_answers=gold, source_documents=[fn], source_type=st, evidence_spans=evidence(fn, pi, text), reference_answer=gold[0], difficulty=diff, question_type=qt))

    for i, question in enumerate([
        "What seed does the course simulator use for the graded Monte Carlo lab?",
        "Which calculator model is mandatory for the probability final exam?",
        "What is the exact prior used in the instructor's unpublished medical testing demo?",
        "How many practice problems are in the locked problem set bank?",
        "What is the office location for probability tutoring?",
        "Which edition of the optional advanced probability book is reserved in the library?",
        "What is the course midterm room number?",
        "Who wrote the custom probability applet used in lecture 12?",
    ], 1):
        rows.append(q(f"pilot-prob-ook{i:02d}", c, "out_of_knowledge", question, construction_method="cross_doc_gap"))

    for i, question in enumerate([
        "What is the capital of Canada?",
        "Who invented the telephone?",
        "How many days are in a leap year?",
        "What is the main ingredient in hummus?",
        "Which sport is played at Wimbledon?",
        "What is the atomic number of carbon?",
        "Who directed the film Casablanca?",
        "What is the tallest building in the world as of 2020?",
    ], 1):
        rows.append(q(f"pilot-prob-ood{i:02d}", c, "out_of_scope", question, construction_method="ood", difficulty="easy"))

    for i, (question, premise) in enumerate([
        ("Why does independence mean that P(A and B) equals P(A) plus P(B)?", "independence uses sum not product"),
        ("Why is the probability of an exact point under a continuous density equal to one?", "point probability is one under a PDF"),
        ("Why does Bayes' theorem ignore the prior and use only the likelihood?", "Bayes ignores the prior"),
        ("Why are mutually exclusive events defined as events that must occur together?", "mutually exclusive must co-occur"),
        ("Why does uncorrelatedness guarantee independence for all random variables?", "uncorrelated implies independent always"),
        ("Why does the complement of an event have probability equal to the event's probability?", "complement probability equals event probability"),
        ("Why does a uniform model on a finite space assign different probabilities to each outcome?", "uniform assigns unequal probabilities"),
    ], 1):
        rows.append(q(f"pilot-prob-fp{i:02d}", c, "false_presupposition", question, construction_method="premise_flip", false_premise=premise))

    for i, (question, missing) in enumerate([
        ("What is the probability?", "Probability of which event in which model?"),
        ("Are they independent?", "Which events or variables?"),
        ("What is the expected value?", "Of which random variable?"),
        ("Should we use Bayes?", "Which hypothesis, prior, and likelihood?"),
        ("Is the predictive value high enough?", "High enough under which base rate and test?"),
        ("How many simulations?", "Simulations for which probability and precision target?"),
        ("What is the conditional probability?", "Of which event given which conditioning event?"),
    ], 1):
        rows.append(q(f"pilot-prob-us{i:02d}", c, "underspecified", question, construction_method="underspecify", missing_aspects=missing))

    return rows


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in CORPUS.items():
        path = CORPUS_DIR / name
        path.write_text(text.strip() + "\n", encoding="utf-8")
        npara = len(paras(name))
        print(f"wrote {path.name} ({npara} passages)")

    MANIFEST_PATH.write_text(json.dumps(MANIFEST, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")

    rows = build_pilot_questions()
    PILOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PILOT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions to {PILOT_PATH}")

    # Summary checks
    from collections import Counter
    by_course = Counter(r["course"] for r in rows)
    by_cat = Counter(r["category"] for r in rows)
    print("per course:", dict(by_course))
    print("per category:", dict(by_cat))
    assert len(rows) == 250
    assert all(v == 50 for v in by_course.values())
    for r in rows:
        if r["category"] == "answerable":
            assert r["answerable"] and r["expected_action"] == "ANSWER" and r["evidence_spans"]
        elif r["category"] == "underspecified":
            assert (not r["answerable"]) and r["expected_action"] == "CLARIFY"
        else:
            assert (not r["answerable"]) and r["expected_action"] == "ABSTAIN"
    print("schema/action consistency checks passed")


if __name__ == "__main__":
    main()
