# Qualitative Trace Report

This report supports the qualitative evaluation described in the research: 
it inspects governance decisions and their stated reasons to assess the 
transparency of the policy-aware system and its compliance with the 
answerability policy.

## Correctly abstained on unanswerable questions (2321 cases)

- **Q:** If the human norovirus cannot be cultured in cell cultures how can it best be studied?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2021 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is used to estimate the expected level of fit in cross-validation?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5228 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Is the median statistical power in psychology studies consistently above 50%?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6501 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What does the CBO analyze the effects of various policy options on?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.3586 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** In what ways did strategic circumstances have an effect on environmental security perspectives?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2588 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the name of the fundamental rule in probability theory?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5825 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the name of the character Played by Woody Allen in the 1967 James Bond film 'Casino Royale'?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2325 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the process of combining data and regularization terms in explicit regularization? And what algorithm combines the advantages of variable selection and classification in a single step?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6474 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Which of these foods could contain small amounts of naturally occurring opium?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence; retriever_below_abstain (0.12 < 0.2)
  - retriever_conf: 0.1175 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the name of the multiplicative version of the central limit theorem?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6628 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What ensures each cluster maintains a balanced representation of protected groups?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.3911 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Do healthcare data warehouses not incorporate specialized data models for medical data?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.8621 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is a common criticism of data lakes? And what did Thomas H. Davenport and DJ Patil declare in 2012 about the Data Scientist role?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.693 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the structure of metadata typically based on? And how is a buffer between two stages in a data pipeline implemented?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5668 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What theorem illustrates the difficulty in defining expected value?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7086 | reasoner_conf: 1.0 | model_is_answerable: False

## Requested clarification (2 cases)

- **Q:** How does the interpretation of big data change over time? And what is one of the most difficult challenges in launching a new product or service according to A/B testing?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: model_requested_clarification
  - retriever_conf: 0.5372 | reasoner_conf: 0.9 | model_is_answerable: False

- **Q:** How does the interpretation of big data change over time? And what is one of the most difficult challenges in launching a new product or service according to A/B testing?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: model_requested_clarification
  - retriever_conf: 0.5372 | reasoner_conf: 0.9 | model_is_answerable: False

## Wrongly refused answerable questions (false rejections) (197 cases)

- **Q:** What is the expected value of the index in cluster analysis?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5404 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What data does the recommender system use for generating driving routes?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6631 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What method does Fisher explicitly contrast with his use of p-values?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6924 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** In k-nearest neighbors algorithm, what is computed explicitly?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6652 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** At what level might some tables not be sufficiently normalized?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7003 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** How much did the data management and analytics industry worth in 2010?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7456 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is quantitative text analysis?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.8543 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** In a 2-dimensional random walk, what is the probability of returning to the origin?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7077 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the foundation of machine learning?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5758 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What distribution's inversion gives the confidence interval for σ²?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.615 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** When F statistic becomes increasingly small, what happens to the confidence interval?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5536 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What are the decimal digits of the geometrically distributed random variable Y?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.8622 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the term for the overlap between the population from which the sample is drawn and the population from which information is desired?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7331 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the purpose of a parser in data cleansing?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7175 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What theorem explains the normal distribution in hydrology?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5638 | reasoner_conf: 1.0 | model_is_answerable: False
