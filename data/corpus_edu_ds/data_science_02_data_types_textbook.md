In introductory data science, variables are commonly classified as numerical or categorical. Numerical variables take quantitative values and may be continuous (any value in an interval) or discrete (countable integers).

Categorical variables take a limited set of labels. Nominal categories have no intrinsic order (for example, browser type), whereas ordinal categories have a meaningful order (for example, Likert agreement levels).

A tidy data table stores each variable in a column, each observation in a row, and each type of observational unit in its own table. Tidy structure simplifies joins, filters, and group-wise summaries.

Feature engineering creates derived predictors from raw fields, such as ratios, bins, time-of-day indicators, or text token counts. Engineered features should be justified by domain knowledge and checked for leakage from the prediction target.

Train-test separation keeps a held-out test set untouched during model fitting so that reported performance estimates remain honest. Using test labels during feature selection or threshold tuning produces overly optimistic scores.

Reproducibility requires recording code versions, random seeds, package versions, and the exact dataset snapshot used for a reported result. Without those artifacts, later auditors cannot regenerate the same numbers.
