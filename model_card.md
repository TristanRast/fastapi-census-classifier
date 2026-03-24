# Model Card

## Model Details

**Model Type:** Random Forest Classifier  
**Version:** 1.0
**Date:** March 2026  
**Developer:** Tristan Rast
**License:** MIT

This model is a binary classifier that predicts whether an individual's income exceeds $50,000 per year based on census data. The model uses a Random Forest algorithm with 100 estimators and a maximum depth of 10.

## Intended Use

**Primary Intended Uses:**
- Educational demonstration of ML model deployment
- Predicting income brackets for demographic analysis
- Understanding socioeconomic factors related to income

**Primary Intended Users:**
- Data scientists and ML engineers learning deployment workflows
- Researchers studying income inequality
- Policy analysts examining demographic-income relationships

**Out-of-Scope Use Cases:**
- Making individual employment or lending decisions
- Any use that could result in discriminatory outcomes
- Real-world financial or legal decision-making without human oversight

## Training Data

**Dataset:** Census Income Dataset (Adult Dataset)  
**Source:** UCI Machine Learning Repository  
**Size:** ~32,000 samples  
**Split:** 80% training, 20% testing

**Features:**
- **Demographic:** age, race, sex, native-country
- **Employment:** workclass, occupation, hours-per-week
- **Education:** education, education-num
- **Financial:** capital-gain, capital-loss
- **Relationship:** marital-status, relationship

**Target Variable:** Binary classification (<=50K or >50K annual income)

**Preprocessing:**
- Categorical features: One-hot encoded
- Continuous features: Used as-is
- Labels: Binary encoding (0 for <=50K, 1 for >50K)

## Evaluation Data

The model is evaluated on a held-out test set comprising 20% of the original dataset (~6,400 samples). The test set maintains the same feature distributions as the training set.

## Metrics

**Overall Performance:**
- **Precision:** 79.62%
- **Recall:** 53.72%  
- **F-beta Score:** 64.16%

**Metric Definitions:**
- **Precision:** Proportion of positive predictions that are correct. (TP / (TP + FP))
- **Recall:** Proportion of actual positives that are identified correctly. (TP / (TP + FN))
- **F-beta:** Harmonic mean of precision and recall. (beta=1 for balanced F1 score)

**Performance on Data Slices:**

* Advanced degree holders show the strongest balanced performanc, with both high precision (84%-84%) and high recall (83%-87%), leading to high F-beta scores of 83%-85%. This indicates the model can reliably identify high earners in these groups.  

* Mid-Level education (Bachelor's, some-college, Assoc-acdm, Assoc-voc) shows high precision but sharply declining recall, meaning the model is conservative. It rarely makes false positive predictions but misses many actual high earners. Some-college recall: 25% and Associates groups recall: 35%-38%.  

* Lower education levels show near-perfect precision but extremely low recall, with HS-grad recall at 19% and 7th-8th at recall 0%. This suggests the model almost never predicts high income for these groups likely reflecting true class imbalance in the training data.  

* Model performs best on Prof-school with and F-beta of 85.38% followed closely by Doctorate (83.19%) and Masters (82.63%).  

* Model performs worst on 7th-8th grade with an F-beta of 0%. The model predicted zero high income individuals in the group: 10th grade is the next worst at 15.38%.

* The 1st-4th and Preschool categories, however, show perfect F-beta of 100%, but their very small sample sizes (23 and 10) make these results statistically unreliable and likely coincidental.

## Ethical Considerations

**Potential Biases:**
- The model uses protected attributes (race, sex, native-country) as features, which could perpetuate historical biases in income distribution
- Census data reflects societal inequalities and historical discrimination
- Model performance may vary significantly across demographic groups (see slice analysis)

**Fairness Considerations:**
- Performance disparities across education levels may reflect data imbalance
- The model should NOT be used for individual decision-making without bias mitigation
- Regular monitoring for fairness metrics (demographic parity, equal opportunity) is recommended

**Privacy:**
- Training data is publicly available census data
- No personally identifiable information is used or stored
- Aggregated predictions should be used rather than individual-level inferences

## Caveats and Recommendations

**Limitations:**
1. **Temporal:** Census data may be outdated; income patterns change over time
2. **Geographic:** Model trained on US census data; not generalizable to other countries
3. **Binary Classification:** Oversimplifies income distribution into two categories
4. **Protected Attributes:** Using demographic features raises fairness concerns

**Recommendations:**
1. **Regular retraining** with updated data to maintain relevance
2. **Bias auditing** before any deployment in sensitive contexts
3. **Human oversight** required for any downstream applications
4. **Fairness evaluation** across demographic subgroups before production use

**Future Improvements:**
- Implement fairness constraints during training
- Explore bias mitigation techniques (reweighting, adversarial debiasing)
- Add confidence intervals to predictions
- Expand to multi-class income brackets for finer granularity
- Remove or mask protected attributes and evaluate performance impact
