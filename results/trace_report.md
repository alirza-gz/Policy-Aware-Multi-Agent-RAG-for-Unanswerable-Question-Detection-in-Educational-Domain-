# Qualitative Trace Report

This report supports the qualitative evaluation described in the research: 
it inspects governance decisions and their stated reasons to assess the 
transparency of the policy-aware system and its compliance with the 
answerability policy.

## Correctly abstained on unanswerable questions (2163 cases)

- **Q:** If the human norovirus cannot be cultured in cell cultures how can it best be studied?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2543 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is added when a point is found to be dense in DBSCAN?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7345 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Is the median statistical power in psychology studies consistently above 50%?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6501 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** In what ways did strategic circumstances have an effect on environmental security perspectives?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.3645 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the name of the fundamental rule in probability theory?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6685 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the name of the character Played by Woody Allen in the 1967 James Bond film 'Casino Royale'?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2325 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Which of these foods could contain small amounts of naturally occurring opium?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2755 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What distribution is not conjugate to the rate parameter of a Poisson distribution or exponential distribution?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6338 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the name of the multiplicative version of the central limit theorem?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6804 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What ensures each cluster maintains a balanced representation of protected groups?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.4025 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Do healthcare data warehouses not incorporate specialized data models for medical data?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.8621 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the structure of metadata typically based on? And how is a buffer between two stages in a data pipeline implemented?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7129 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What theorem illustrates the difficulty in defining expected value?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7486 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** According to Hindu tradition, which age are we in currently?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.23 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Where would you typically find a bailiff?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.299 | reasoner_conf: 1.0 | model_is_answerable: False

## Requested clarification (20 cases)

- **Q:** What is the main purpose of licensing public relations practitioners?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.43 in [0.3, 0.5))
  - retriever_conf: 0.4542 | reasoner_conf: 1.0 | model_is_answerable: True

- **Q:** When using all available data, what test could be used assuming normality and MCAR?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.49 in [0.3, 0.5))
  - retriever_conf: 0.4934 | reasoner_conf: 1.0 | model_is_answerable: True

- **Q:** What method is used to enforce an absolute upper bound on the magnitude of the weight vector for every neuron in a Convolutional neural network? And what practical effect does transforming using the logit function have on the probability in Logistic regression?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.50 in [0.3, 0.5))
  - retriever_conf: 0.499 | reasoner_conf: 1.0 | model_is_answerable: True

- **Q:** What is the original set of Dublin Core metadata terms known as?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.42 in [0.3, 0.5))
  - retriever_conf: 0.481 | reasoner_conf: 0.95 | model_is_answerable: True

- **Q:** Amps are a unit of measurement of what?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.34 in [0.3, 0.5))
  - retriever_conf: 0.3968 | reasoner_conf: 0.336 | model_is_answerable: True

- **Q:** Why might it be challenging to directly compare the performance of different deep learning architectures? And why is choosing a logistic error-variable distribution with a non-zero location parameter μ considered equivalent to having a zero location parameter?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: model_requested_clarification
  - retriever_conf: 0.6217 | reasoner_conf: 0.95 | model_is_answerable: False

- **Q:** What is the main purpose of licensing public relations practitioners?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.43 in [0.3, 0.5))
  - retriever_conf: 0.4542 | reasoner_conf: 1.0 | model_is_answerable: True

- **Q:** What does the curse of dimensionality imply for the use of Euclidean distance in k-NN algorithms? And what is the primary function of cluster analysis in organizing large corpora of unstructured text?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: model_requested_clarification
  - retriever_conf: 0.756 | reasoner_conf: 0.9 | model_is_answerable: False

- **Q:** What is the original set of Dublin Core metadata terms known as?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.45 in [0.3, 0.5))
  - retriever_conf: 0.481 | reasoner_conf: 0.95 | model_is_answerable: True

- **Q:** Why might it be challenging to directly compare the performance of different deep learning architectures? And why is choosing a logistic error-variable distribution with a non-zero location parameter μ considered equivalent to having a zero location parameter?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: model_requested_clarification
  - retriever_conf: 0.6217 | reasoner_conf: 0.95 | model_is_answerable: False

- **Q:** When using all available data, what test could be used assuming normality and MCAR?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.49 in [0.3, 0.5))
  - retriever_conf: 0.4934 | reasoner_conf: 1.0 | model_is_answerable: True

- **Q:** Amps are a unit of measurement of what?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.34 in [0.3, 0.5))
  - retriever_conf: 0.3968 | reasoner_conf: 0.336 | model_is_answerable: True

- **Q:** What techniques have been developed to correct for range restriction?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.44 in [0.3, 0.5))
  - retriever_conf: 0.4441 | reasoner_conf: 1.0 | model_is_answerable: True

- **Q:** What techniques have been developed to correct for range restriction?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.44 in [0.3, 0.5))
  - retriever_conf: 0.4441 | reasoner_conf: 1.0 | model_is_answerable: True

- **Q:** Why might it be challenging to directly compare the performance of different deep learning architectures? And why is choosing a logistic error-variable distribution with a non-zero location parameter μ considered equivalent to having a zero location parameter?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: model_requested_clarification
  - retriever_conf: 0.6217 | reasoner_conf: 0.95 | model_is_answerable: False

## Wrongly refused answerable questions (false rejections) (316 cases)

- **Q:** In 2020, what deep-learning based system achieved a level of accuracy significantly higher than all previous computational methods?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6507 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the expected value of the index in cluster analysis?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5404 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What data does the recommender system use for generating driving routes?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6631 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is circular reasoning in hypothesis testing?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6311 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the theorem named after?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5382 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** At what level might some tables not be sufficiently normalized?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7003 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is used in conjunction with geometric neural networks?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6128 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What techniques have been proposed for collaborative filtering?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7608 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** How long can security and policing agencies access an individual's metadata?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7182 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is a modification in the context of stochastic processes?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6671 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** In a 2-dimensional random walk, what is the probability of returning to the origin?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7077 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What distribution's inversion gives the confidence interval for σ²?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.615 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is used to split the nodes in the left tree?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6491 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** When F statistic becomes increasingly small, what happens to the confidence interval?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5536 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What are the decimal digits of the geometrically distributed random variable Y?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.8622 | reasoner_conf: 1.0 | model_is_answerable: False
