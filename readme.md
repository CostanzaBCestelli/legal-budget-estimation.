# Legal Budget Estimation Project

## Project Overview
This project aims to predict legal budgets for FMCG companies at a granular level by analyzing potential litigation risks across countries and product lines. Using advanced machine learning techniques, the project integrates data on regulatory trends, consumer sentiment, ESG scores, and historical litigation to deliver insights into legal cost forecasting. The inclusion of Monte Carlo simulations further enhances the reliability of predictions by accounting for uncertainty.

This repository contains all code, mock data, and visualization tools required to replicate and expand upon the work.

---

## Key Features
1. **Risk Prediction Model:**
   - Uses a Gradient Boosting Classifier to estimate the probability of legal actions for each country-product pair.
   - Incorporates features such as regulatory complexity, consumer sentiment, and ESG scores.

2. **Cost Prediction Model:**
   - Employs a Gradient Boosting Regressor to predict the financial costs associated with identified risks.
   - Outputs are tailored for country-level and product-specific forecasts.

3. **Uncertainty Modeling:**
   - Monte Carlo simulations simulate variability in predictions, providing 95% confidence intervals for predicted costs.

4. **Budget Aggregation:**
   - Summarizes predicted legal budgets by country, offering a high-level view for strategic decision-making.
   - Includes visualizations of the aggregated legal budget across countries.

5. **Visualization Tools:**
   - Charts and graphs for better interpretability of results, such as confidence intervals and aggregated costs.

---

## Project Files
Here is a breakdown of the files in this repository:

- **`legal_budget_estimation.py`**: The main Python script containing all data preprocessing, machine learning models, Monte Carlo simulations, and visualizations.
- **Mock Data Files**: Pre-generated datasets for testing and development:
  - `consumer_sentiment.csv`
  - `regulatory_trends.csv`
  - `esg_scores.csv`
  - `historical_litigation.csv`
- **`README.md`**: This documentation file explaining the project structure, usage, and methodology.

---

### Real-World Data Sources

#### Consumer Sentiment:
- [Twitter (X)](https://twitter.com)
- [Reddit](https://www.reddit.com)
- [Amazon](https://www.amazon.com)
- [Trustpilot](https://www.trustpilot.com)

#### Regulatory Trends:
- [European Commission](https://ec.europa.eu)
- [US Federal Register](https://www.federalregister.gov)
- [LexisNexis](https://www.lexisnexis.com)

#### ESG Scores:
- [MSCI](https://www.msci.com)
- [Sustainalytics](https://www.sustainalytics.com)
- [CDP](https://www.cdp.net)

#### Historical Litigation:
- [PACER](https://pacer.uscourts.gov)
- [Law360](https://www.law360.com)
- [Westlaw](https://legal.thomsonreuters.com/en/westlaw)

---

## Results
  - **Predicted Legal Costs**: Tailored forecasts for country-product pairs.
  - **Aggregated Country Budgets**: Comprehensive legal budget estimates by country.
  - **Confidence Intervals**: Uncertainty quantification ensures robust decision-making.

---

## Sample Outputs:
  - **Classification Accuracy**: ~85% for litigation risk.
  - **Regression MSE**: Varies with dataset but provides reliable cost estimates.

---

##  Requirements
To run the project, ensure the following Python libraries are installed:

```bash
pip install pandas numpy scikit-learn matplotlib
