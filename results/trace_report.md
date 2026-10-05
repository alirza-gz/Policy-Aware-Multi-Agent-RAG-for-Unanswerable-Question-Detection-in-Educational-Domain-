# Qualitative Trace Report

This report supports the qualitative evaluation described in the research: 
it inspects governance decisions and their stated reasons to assess the 
transparency of the policy-aware system and its compliance with the 
answerability policy.

## Correctly abstained on unanswerable questions (2318 cases)

- **Q:** How can you avoid overfitting with so many candidate models?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7042 | reasoner_conf: 0.0 | model_is_answerable: False

- **Q:** What does the CBO analyze?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.3564 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Why is it desirable to incorporate a small-sample correction in naive Bayes?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.568 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Why do maximum likelihood estimators always have superior performance compared to other estimators as the sample size increases to infinity?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7236 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** In unsupervised learning, what is used to correct the error?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.4999 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** How can a branching process accurately predict the extinction of a population?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6622 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Which leader is known for spreading the dharma of non-violence?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence; retriever_below_abstain (0.14 < 0.2)
  - retriever_conf: 0.1396 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Why might a function overfitting lead to increased costs in data gathering? And how does one-versus-all support vector machine classify new instances?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5772 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is an experimental unit in data analysis?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6147 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** In what range is mathematical statistics indexed in the Library of Congress Classification?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.4288 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** This question refers to the following information. "The conscience of the people, in a time of grave national problems, has called into being a new party, born of the nation's sense of justice. We of the Progressive party here dedicate ourselves to the fulfillment of the duty laid upon us by our fathers to maintain the government of the people, by the people and for the people whose foundations they laid. We hold with Thomas Jefferson and Abraham Lincoln that the people are the masters of their Constitution, to fulfill its purposes and to safeguard it from those who, by perversion of its intent, would convert it into an instrument of injustice. In accordance with the needs of each generation the people must use their sovereign powers to establish and maintain equal opportunity and industrial justice, to secure which this Government was founded and without which no republic can endure. "This country belongs to the people who inhabit it. Its resources, its business, its institutions and its laws should be utilized, maintained or altered in whatever manner will best promote the general interest. It is time to set the public welfare in the first place." Progressive Party Platform, 1912 Would the Underwood-Simmons Tariff of 1913 be generally endorsed by Progressives of that era?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence; retriever_below_abstain (0.14 < 0.2)
  - retriever_conf: 0.1433 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What has been observed regarding the use of data mining in the United Kingdom? And what is the focus of pattern recognition systems when no labeled data are available?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6022 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What conclusion can be drawn if the model deviance is not significantly smaller than the null deviance in logistic regression?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7863 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the percentage of children aged 13-15 in China who reported being in a physical fight at school, one or more times during the past 12 months as of 2015?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.2074 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Why does domain driven data mining aim to create non-actionable knowledge and insights?
  - gold: `unanswerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5821 | reasoner_conf: 1.0 | model_is_answerable: False

## Requested clarification (1 cases)

- **Q:** How does the interpretation of big data change over time? And what is one of the most difficult challenges in launching a new product or service according to A/B testing?
  - gold: `unanswerable` | action: `CLARIFY`
  - reason: model_requested_clarification
  - retriever_conf: 0.5372 | reasoner_conf: 0.9 | model_is_answerable: False

## Wrongly refused answerable questions (false rejections) (215 cases)

- **Q:** When should an outlier be removed based on its cause?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.8257 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** Does naive Bayes require binning for continuous values?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7609 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What are gray sheep in collaborative filtering?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.8112 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What type of algorithms can produce a stronger ensemble?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.6854 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is an example of a point that is considered an outlier in LOF?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7363 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** How many explanatory variables can binary logistic regression be generalized to?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7924 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the significance level used in the example power analysis?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.677 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the foundation of machine learning?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5758 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** How much did the data management and analytics industry worth in 2010?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7456 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is the probability of the event A?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5236 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** How many minutes does it take to produce one car by using a pipeline?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7369 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What is respondent driven sampling?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.5812 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What example did Welch present to show the superiority of confidence interval theory?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7632 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** What table was added to the database?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.535 | reasoner_conf: 1.0 | model_is_answerable: False

- **Q:** When Yi are independent observations from a normal distribution, how is the unbiased sample variance S2 distributed?
  - gold: `answerable` | action: `ABSTAIN`
  - reason: unanswerable_no_supporting_evidence
  - retriever_conf: 0.7467 | reasoner_conf: 1.0 | model_is_answerable: False
