Simple linear regression models a response variable as a linear function of one predictor plus an error term. The slope coefficient describes the expected change in the response for a one-unit increase in the predictor.

Ordinary least squares chooses the intercept and slope that minimize the sum of squared residuals. Residuals are observed responses minus fitted values.

R-squared measures the proportion of variance in the response explained by the fitted model. High R-squared does not prove that the model is correctly specified or causal.

Multiple linear regression includes several predictors at once. Coefficients are interpreted as associations holding the other included predictors fixed, which depends on which covariates are in the model.

Assumptions commonly discussed in introductory courses include linearity, independent errors, roughly constant error variance, and approximate normality of errors for inference. Residual plots help check several of these assumptions visually.

Extrapolating a regression prediction far outside the observed predictor range is unreliable because the fitted linear relationship may not continue. Reporting the training range alongside predictions reduces misuse.
