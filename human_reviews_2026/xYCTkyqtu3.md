# Automatic Moderator Discovery via SHAP Interaction Values

- Decision: Reject
- Scores: 2, 4, 2, 2, 4

## Abstract
Machine Learning (ML) is increasingly applied across the sciences, accelerating simulations, automating data preparation, and improving predictive accuracy. Yet most efforts emphasize efficiency and performance, with limited attention to interpretability, thereby leaving unexplored how ML can drive discovery—uncovering novel patterns in data and advancing scientific theory. Moderation effects—where the influence of one variable depends on the level of another—are central to disciplines such as social science and human behavior. However, they are typically studied through a theory-driven process based on regression models with manually specified interactions. While insightful, this approach is limited because it scales poorly and may miss unexpected moderators.
We introduce an automated, interpretable framework for moderator discovery based on SHAP interaction values. Our method computes global interaction contributions from a predictive model, quantifies their dependence on constituent features, and identifies statistically significant moderators. In experiments on real-world datasets, the framework not only recovers known, theory-consistent moderating effects but also uncovers novel moderator candidates. These results illustrate how explainable ML can move beyond prediction toward systematic discovery, offering scientists a scalable tool to reveal conditional relationships that inform theory development.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an automated framework for data-driven moderator discovery using SHAP interaction values derived from gradient-boosted decision trees (LightGBM). The framework aims to bridge predictive ML and theory-driven science, allowing researchers to systematically uncover moderating effects (conditional relationships) without prespecifying candidate moderators.

### Strengths
### **Clear Motivation**
* The work addresses an underexplored gap between predictive modeling and interpretable, theory-oriented discovery, particularly relevant in social sciences and behavioral domains.

### **Methodology**
* Leveraging TreeSHAP for interaction attribution is clever: it enables scalable, model-faithful extraction of interaction information with polynomial-time complexity.

### **Empirical Validation**
* Experiments include both synthetic (ground-truth known) and real-world (social science) datasets.

### Weaknesses
### **Technical Limitations / Mathematical Depth**
* The paper does not provide theoretical justification or bounds on when SHAP interaction values can be reliably interpreted as moderation effects. SHAP interaction measures contribution, not necessarily conditional effect in a causal or statistical moderation sense.
* The $\beta_3$ estimation step treats SHAP values as regression targets without addressing their statistical dependence structure or the implications of SHAP value non-orthogonality. This may lead to inflated significance or misinterpretation.
* No identifiability or robustness analysis is provided: for instance, how stable the discovered moderators are to model perturbations, feature scaling, or correlated covariates.

### **Novelty**
* The method is essentially a combination of existing components: TreeSHAP, classical moderation regression, and significance testing
* Attribution-based moderation inference has conceptual similarity to existing feature interaction explanation methods

### **Conceptual Ambiguity**
* The paper equates SHAP interaction values with moderating effects, but this is not strictly justified: SHAP interactions are symmetrical, whereas moderation is directional (one variable moderates another). Although regression on SHAP attributions introduces directionality, the interpretation remains heuristic, not theoretically guaranteed.

### Questions
Please see the above weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a three-step interpretable framework for the automatic discovery of moderator effects from data. The method aims to bridge predictive machine learning with theory-driven scientific discovery. The framework first trains a high-capacity GBDT model to capture complex relationships. Second, it computes pairwise SHAP interaction values to quantify the joint contribution of feature pairs. Third, it regresses these attribution scores onto the original feature values (fitting $\phi_{i,j}^{tot} \approx \dots + \beta_3 x_i x_j$) and uses the resulting $\beta_3$ coefficient and its p-value to identify and rank significant moderators. The method is validated on synthetic data and two real-world datasets demonstrating it can recover theoretically-consistent moderators and outperform baselines.

### Strengths
1. The paper addresses a highly important and practical goal: moving ML from a purely predictive tool to one that can aid in systemic, data-driven scientific discovery. 
2. The paper is exceptionally well-written. The problem is clearly defined, the method is presented in a logical, three-step algorithm, and the experimental setup is easy to follow.
3. While the individual components (GBDT, SHAP, OLS) are standard, their composition is novel. The core original idea is to re-purpose SHAP values as an intermediate data source for a second-stage statistical analysis. This is a clever approach that reframes the vague problem of "interaction detection" into the more specific, interpretable problem of "moderator assessment."

### Weaknesses
1. The designed framework can only detect 2-way interactions. It is blind to higher-order interactions, which are common and critical in many scientific domains. This makes the contribution feel more like a proof-of-concept than a complete discovery system.
2. The contribution is a clever pipeline of existing tools rather than a new algorithm or fundamental theory. This limits the technical originality of the work.
3. The claim of a general, systematic framework is supported by only one synthetic and two real-world datasets. This is not extensive enough to prove its general applicability across different scientific domains.
4. The validation metric (a linear test) seems mismatched with the non-linear GBDT model.
5. The paper does not analyze how sensitive the discovered moderators are to the GBDT's hyperparameters, which is critical for a discovery tool's reliability.

### Questions
1. What are the primary conceptual or computational barriers to extending this framework to 3-way (or higher-order) interactions? Would this involve computing 3rd-order SHAP values and regressing them onto a term like $\beta x_i x_j x_k$?
2. Why use a linear test to validate a non-linear model? Doesn't this risk penalizing true non-linear discoveries? Relatedly, how sensitive is the final list of moderators to the GBDT's hyperparameters? We need to be sure the discoveries are robust and not artifacts of tuning.
3. The claim of a general framework is supported by only two real-world datasets. Could the authors comment on the expected generalizability to other scientific domains and data types? What challenges in applying this method to much higher-dimensional data (e.g., M > 10,000)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces a scalable and interpretable framework for automatic moderator discovery in machine learning models. It uses SHAP interaction values to screen for and quantify moderating effects, where the influence of one variable depends on the level of another, across all feature pairs in high-dimensional data. Evaluation on synthetic and real-world datasets (COMPAS, Medicare vaccination) demonstrates the method’s ability to recover known moderators and reveal new candidates not previously highlighted by domain-driven analyses.

### Strengths
- The paper tackles an interesting but underexplored facet of interpretable machine learning.
- The methodology is well-motivated and clearly described. The use of SHAP interaction values for model-wide screening, combined with a regression-based moderation assessment, is a compelling merge of machine learning and statistical theory

### Weaknesses
1. While the paper evaluates the statistical significance of discovered moderators via inclusion in a linear regression, this does not fully control for potential confounding or capture non-linear relationships. The approach essentially reduces the problem to significance of pairwise terms, which may inflate discovery of spurious moderators, especially in complex or collinear datasets. Further, no detailed ablation/sensitivity studies are reported regarding how methodological or hyperparameter changes (e.g., choice of model class, depth of GBDT, or normalization scheme) affect results or stability of discovered moderators.
2.  The method is only tested using GBDT (LightGBM). While it is stated that other tree ensembles (XGBoost, CatBoost) or potentially other model classes could be used, no empirical evidence or discussion is provided for non-tree models, or for cases where SHAP values are approximated rather than computed exactly.
3. While the technique is positioned as “interpretable” and “scalable,” the practical interpretability of discovered moderators is not rigorously evaluated beyond recovering known scientific patterns. For instance, how do domain experts interpret SHAP-ranked moderators in highly collinear data, or when moderators do not have a causal interpretation?
4. The quality of Figure 2 could be improved. The legend/axes titles are too small, and there are no error bars/bands.

### Questions
1. Could the authors clarify the exact normalization approach for predictors used prior to the regression-based moderation test (Section 3.3)? For continuous vs. categorical features, is normalization performed identically?
2. How does the method perform if the predictive model is replaced with a neural network backbone, or if SHAP values must be estimated empirically rather than via TreeSHAP? Is the approach robust to approximate SHAP implementations?
3. How scalable is the method in practice as the number of features increases into the hundreds/thousands? Actual running time benchmarks or empirical scaling curves would be valuable for practitioners.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a heurstic framework for automated, data-driven discovery of moderation effects (conditional dependence bet two variables given a context) using SHAP interaction values derived from gradient-boosted decision trees. The idea is well motivated and the empirical experiment is supported with domain insights, yet the methodology sound limited, please see comments below. Overall, I believe the scope and implementation of this work are of limited interest to ICLR community.

### Strengths
- Proposes a systematic approach for uncovering moderating effects. The approach is hypothesis-agnostic, scalable to high-dimensional datasets, and preserves model interpretability. Empirical results on synthetic and real data show that some known moderators and even novel candidates can be recovered.

### Weaknesses
- The use of GBDTs for moderator discovery indeed captures nonlinear interactions; but also results in discontinuous, piecewise-constant outputs. This discretization may not be able to capture smooth moderation patterns, limiting method generalization. 

- The method computes SHAP interaction values via TreeSHAP (as indicators of moderation). TreeSHAP relies on sampling features from their product-marginal (i.e., assuming independence bet features) when estimating the contributions. Thus, it may ignore or misrepresent the true joint dependence structure among features, which can lead to artificial conditional relationships or spurious moderators being detected.

- ShapMod-1st outperforms ShapMod (based in Eq. 6) in three of four datasets, raising a question about the need for SHAP interaction terms in Eq (6)?

- Both the SHAP method and trained model introduce potential errors through estimation and initialization. Evaluation relies on small datasets, including a synthetic example with only 3,000 samples but many features (40), raising various questions about generalizability and sample-size sensitivity.

### Questions
See weaknesses section

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focuses on the problem of automatically identifying moderator variables. The authors propose a three-step approach that utilizes SHAP interaction values to train a predictive model, evaluate moderation effects, and identify significant moderators. The effectiveness of the proposed framework is validated on both synthetic datasets and real-world dataset.

### Strengths
1. The paper is well-written and clearly organized.

2. The authors provide a good review of related work.

3. The experimental results on the Medicare vaccination dataset reveal an interesting and potentially novel pattern, which could bring useful insights to this community.

### Weaknesses
1.  The baseline methods (e.g., Lasso and ANOVA) seem a bit outdated. I am not an expert in this particular subarea, but if there are newer or more advanced baselines available, including them could make the evaluation more convincing.

2. The proposed framework currently lacks a theoretical guarantee. It would be helpful to discuss under what conditions the identified moderator can be regarded as the most important one. Adding some theoretical insights or sensitivity analysis could further strengthen the contribution.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
