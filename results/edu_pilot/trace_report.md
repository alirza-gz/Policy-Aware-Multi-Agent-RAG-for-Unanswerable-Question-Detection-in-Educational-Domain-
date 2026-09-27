# Qualitative Trace Report

This report supports the qualitative evaluation described in the research: 
it inspects governance decisions and their stated reasons to assess the 
transparency of the policy-aware system and its compliance with the 
answerability policy.

## Correctly abstained on unanswerable questions (93 cases)

- **Q:** What instrument family does the trumpet belong to?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence; retriever_below_abstain (0.16 < 0.2)
  - retriever_conf: 0.1621 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** How many hidden layers does the mandatory autoencoder homework use?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2363 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** Which Kaggle competition ID is required for the ML course project?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2867 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** What is the private GitHub classroom assignment code?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.3158 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** What is the chemical symbol for gold?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence; retriever_below_abstain (0.17 < 0.2)
  - retriever_conf: 0.1667 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** Which company donated the dataset used in the data science midterm case study?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.4659 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** Who won the FIFA World Cup in 2018?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence; retriever_below_abstain (0.14 < 0.2)
  - retriever_conf: 0.138 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** What is the exact prior used in the instructor's unpublished medical testing demo?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.4046 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** Which R package version must students install for the ANOVA lab?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2252 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** Is this fair?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.3081 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** What is the instructor's preferred GUI for association rule demos?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.4142 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** Should we prune candidates?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2216 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** What is the tallest mountain in Africa?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence; retriever_below_abstain (0.11 < 0.2)
  - retriever_conf: 0.1098 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** Who painted the Mona Lisa?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence; retriever_below_abstain (0.09 < 0.2)
  - retriever_conf: 0.0902 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** Are we ready to deploy?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.4451 | reasoner_conf: 0.0 | model_is_answerable: False

## Requested clarification (14 cases)

- **Q:** How should we clean it?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.49 in [0.3, 0.5))
  - retriever_conf: 0.2791 | reasoner_conf: 0.488 | model_is_answerable: True

- **Q:** Is the pattern actionable?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.50 in [0.3, 0.5))
  - retriever_conf: 0.404 | reasoner_conf: 0.498 | model_is_answerable: True

- **Q:** Why is the mean preferred for all skewed distributions because it ignores extremes?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.40 in [0.3, 0.5))
  - retriever_conf: 0.6374 | reasoner_conf: 0.403 | model_is_answerable: True

- **Q:** Who sponsors the machine learning hackathon prize?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.40 in [0.3, 0.5))
  - retriever_conf: 0.3043 | reasoner_conf: 0.399 | model_is_answerable: True

- **Q:** Why are proxy variables guaranteed to remove all fairness risks when the sensitive attribute is dropped?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.43 in [0.3, 0.5))
  - retriever_conf: 0.7339 | reasoner_conf: 0.433 | model_is_answerable: True

- **Q:** What criteria can make a pattern interesting?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.44 in [0.3, 0.5))
  - retriever_conf: 0.7721 | reasoner_conf: 0.436 | model_is_answerable: True

- **Q:** Why does a correlation coefficient prove that one variable causes another?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.46 in [0.3, 0.5))
  - retriever_conf: 0.7221 | reasoner_conf: 0.458 | model_is_answerable: True

- **Q:** How do bagging and boosting differ at a high level?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.43 in [0.3, 0.5))
  - retriever_conf: 0.6661 | reasoner_conf: 0.428 | model_is_answerable: True

- **Q:** Which set do we evaluate on?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.50 in [0.3, 0.5))
  - retriever_conf: 0.4047 | reasoner_conf: 0.498 | model_is_answerable: True

- **Q:** Who grades the data mining oral presentations?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.41 in [0.3, 0.5))
  - retriever_conf: 0.4403 | reasoner_conf: 0.41 | model_is_answerable: True

- **Q:** Is this rule interesting?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.50 in [0.3, 0.5))
  - retriever_conf: 0.4053 | reasoner_conf: 0.498 | model_is_answerable: True

- **Q:** Why is precision defined as the fraction of true positives among all true positive cases?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.47 in [0.3, 0.5))
  - retriever_conf: 0.6883 | reasoner_conf: 0.47 | model_is_answerable: True

- **Q:** Why is accuracy the best metric for every imbalanced classification problem?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.46 in [0.3, 0.5))
  - retriever_conf: 0.7326 | reasoner_conf: 0.459 | model_is_answerable: True

- **Q:** How is data mining characterized in the introductory notes?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.42 in [0.3, 0.5))
  - retriever_conf: 0.593 | reasoner_conf: 0.422 | model_is_answerable: True

## Wrongly refused answerable questions (false rejections) (3 cases)

- **Q:** What criteria can make a pattern interesting?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.44 in [0.3, 0.5))
  - retriever_conf: 0.7721 | reasoner_conf: 0.436 | model_is_answerable: True

- **Q:** How do bagging and boosting differ at a high level?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.43 in [0.3, 0.5))
  - retriever_conf: 0.6661 | reasoner_conf: 0.428 | model_is_answerable: True

- **Q:** How is data mining characterized in the introductory notes?
  - gold: `answerable` | action: `CLARIFY`
  - reason: confidence_in_clarify_band (0.42 in [0.3, 0.5))
  - retriever_conf: 0.593 | reasoner_conf: 0.422 | model_is_answerable: True
