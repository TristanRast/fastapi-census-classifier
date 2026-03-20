# Model Card

## Model Details

**Model Type:** Random Forest Classifier  
**Version:** 1.0.0  
**Date:** March 2026  
**Developer:** [YOUR NAME - UPDATE IN inputs.txt]  
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
- **Precision:** [INSERT FROM TRAINING OUTPUT - see inputs.txt]
- **Recall:** [INSERT FROM TRAINING OUTPUT - see inputs.txt]  
- **F-beta Score:** [INSERT FROM TRAINING OUTPUT - see inputs.txt]

**Metric Definitions:**
- **Precision:** Proportion of positive predictions that are correct (TP / (TP + FP))
- **Recall:** Proportion of actual positives that are identified correctly (TP / (TP + FN))
- **F-beta:** Harmonic mean of precision and recall (beta=1 for balanced F1 score)

**Performance on Data Slices:**

See `slice_output.txt` for detailed performance breakdown across different education levels. Key observations:

[SUMMARIZE KEY FINDINGS FROM slice_output.txt - see inputs.txt]

Example format:
- Higher education levels (Bachelors, Masters, Doctorate) show [precision/recall pattern]
- Lower education levels (HS-grad, Some-college) show [precision/recall pattern]
- Model performs best on [education level] with F-beta of [X.XX]
- Model performs worst on [education level] with F-beta of [X.XX]

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
1. **Do not use** for individual-level decisions (hiring, lending, etc.)
2. **Regular retraining** with updated data to maintain relevance
3. **Bias auditing** before any deployment in sensitive contexts
4. **Human oversight** required for any downstream applications
5. **Fairness evaluation** across demographic subgroups before production use

**Future Improvements:**
- Implement fairness constraints during training
- Explore bias mitigation techniques (reweighting, adversarial debiasing)
- Add confidence intervals to predictions
- Expand to multi-class income brackets for finer granularity
- Remove or mask protected attributes and evaluate performance impact
