# Inferring the Invisible: Neuro-Symbolic Rule Discovery for Missing Value Imputation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
One of the central challenges in artificial intelligence is reasoning under partial observability, where key values are missing but essential for understanding and modeling the system. This paper presents a neuro-symbolic framework for latent rule discovery and missing value imputation. In contrast to traditional latent variable models, our approach treats missing grounded values as latent predicates to be inferred through logical reasoning. By interleaving neural representation learning with symbolic rule induction, the model iteratively discovers—both conjunctive and disjunctive rules—that explain observed patterns and recover missing entries. Our framework seamlessly handles heterogeneous data, reasoning over both discrete and continuous features by learning soft predicates from continuous values. Crucially, the inferred values not only fill in gaps in the data but also serve as supporting evidence for further rule induction and inference—creating a feedback loop in which imputation and rule mining reinforce one another. Using a staged block-coordinate gradient descent, the system learns these rules end-to-end by iteratively optimizing over parameter blocks in an alternating fashion. Experiments on both synthetic and real-world datasets demonstrate that our method effectively imputes missing values while uncovering meaningful, human-interpretable rules that govern system dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a neuro-symbolic framework for missing value imputation that integrates neural feature learning with symbolic rule discovery. The key contributions are A hybrid imputation architecture that couples a neural network (for pattern recognition and latent representation) with a symbolic reasoning engine (for interpretable rule-based inference of missing values), and a a rule discovery mechanism that extracts symbolic if-then rules from latent representations to explain imputation decisions.

### Strengths
1. The fusion of symbolic rule discovery with neural representation learning is both conceptually elegant and practically meaningful. It directly addresses a persistent problem in missing data research: neural models can impute accurately but are opaque, whereas symbolic models are interpretable but rigid.
2. The ability to generate human-readable rules adds practical value for analysts, especially in regulated domains (e.g., healthcare, finance).
3. The iterative mechanism that alternates between neural imputation and symbolic rule adjustment is well-motivated. It allows symbolic reasoning to guide neural learning and vice versa, enhancing both accuracy and transparency.

### Weaknesses
1. While the motivation is sound, the paper lacks theoretical guarantees for convergence or consistency of rule discovery. There’s no formal justification that the extracted rules remain faithful to the true data-generating process, especially under high missingness rates.
2. The symbolic component introduces combinatorial complexity, especially when discovering multi-variable rules. The approach may not scale well to high-dimensional datasets with thousands of features.
3. Missing data in real-world structured domains might behave differently, so generalizability remains unclear.

### Questions
1. How does the symbolic rule discovery scale with feature dimensionality and rule length? Could pruning or differentiable symbolic learning help?
2. Have you conducted any human evaluation to assess whether the discovered rules are semantically meaningful and useful to domain experts?
3. How does the framework explicitly handle MNAR data when missingness depends on the missing values themselves?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel neuro-symbolic framework (NS-FCN) for jointly performing missing-value imputation and interpretable rule discovery in heterogeneous datasets. The proposed model treats missing entries as latent predicates and combines neural embedding learning with differentiable logical reasoning in a closed-loop fashion—where imputation and rule induction reinforce one another. Using coordinate gradient descent, sequential covering, and soft logical operators, NS-FCN discovers both conjunctive and disjunctive rules and imputes missing data. Experiments on synthetic and real-world datasets show promising imputation accuracy and human-interpretable rules.

### Strengths
1. The framework’s closed-loop design elegantly unifies imputation and rule induction, a combination that is conceptually strong and rarely explored.

2. The method produces explicit logical rules that are easy to interpret and verify. This property is valuable for domains where transparency is critical.

3. The paper is clearly written, with intuitive figuresillustrating the reasoning and training process. The problem is generally well motivated and related-work coverage are comprehensive.

### Weaknesses
Despite its novelty, several issues limit the methodological soundness and clarity.

* The paper claims that the coordinate gradient descent “never increases” the loss, but no proof, or convergence bound is presented. It is unclear whether this property holds under stochastic mini-batching or asynchronous updates when rules interact via shared predicates.
* The freezing of “perfect rules” during training may lead to premature convergence or sub-optimal local minima, especially when data contain noise or imbalanced predicates. This could hinder exploration of alternative rule structures.
* Equation (2) relies on a non-differentiable argmax to match rule components to predicate embeddings. The paper does not explain whether gradients are approximated or ignored. If predicate embeddings are frozen, this matching process may prevent end-to-end adaptation and yield inconsistent updates.
* The soft-min and log-sum-exp approximations can distort logical semantics. Soft-min may amplify small numerical differences, resulting in unstable gradient propagation.
* Although the experiments are diverse, comparisons are limited to interpretable baselines. It would be informative to benchmark against state-of-the-art deep imputation methods, e.g., MICE, GAIN, MissForest) to better contextualize performance gains. Sensitivity to missingness ratios and noise could also be analyzed more rigorously.

### Questions
1. Have you conducted a sensitivity analysis for the temperature parameters used in the soft-min and log-sum-exp approximations? How do these values impact the model's performance and stability across different datasets?


2. The paper mentions that negative predicates are treated as independent predicates. ​ Have you explored alternative approaches to model negative predicates more effectively, such as incorporating negation directly into the rule learning process?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Authors aim to combine neural embedding–based ILP with rule-based missing value imputation. However, the paper doesn’t clearly explain why their methodology should outperform the baselines they compare against. The only reason - apart from being the first to combine these three techniques - seems to be a more native and advanced way of handling continuous variables. But the baselines this method is compared against are not even introduced in the related works or other sections. It’s also not very well written and hard to follow.

### Strengths
The paper presents a methodology that leverages neural embedding–based inductive logic programming to enhance rule-based methods for missing value imputation, which handles both binary and continuous variables. The method employs a coordinate gradient descent algorithm specifically tailored to this type of problem. Its novelty lies in being the first to combine ILP and rule-based approaches for improved missing value imputation. The integration of soft predicate learning enables native handling of continuous features, avoiding discretization and improving flexibility. The paper is well written and detailed. The empirical results on both synthetic and real datasets are
strong, demonstrating both interpretability and accuracy.

### Weaknesses
The paper does not clearly justify why the proposed method outperforms the baselines, particularly in real-world experiments. The other methods in Tables 4 and 5 are not clearly introduced or discussed in the main body of the paper. In particular, the baselines listed in Table 7 (with references) should be properly cited and briefly discussed in the Related Work section. Moreover, the primary reason for NS-FCN’s superior performance—aside from being the first approach to integrate ILP and rule-based imputation—appears to be its native support for continuous variables (lines 412–414). While the ablation studies are thorough, they could more clearly isolate the contribution of individual system components (e.g., the impact of the fine-tuning step versus differentiable forward chaining).

### Questions
1. Please introduce and justify the baseline methods used in the comparison (in Table 4, 5 and 6).
2. Are there other missing-value imputation methods that natively handle continuous features (rather than relying on hard thresholds)? This seems to be a key limitation of the baselines you compare against.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new Neuro-Symbolic framework called NS-FCN (Neuro-Symbolic Forward Chaining Network), designed to integrate missing value imputation with interpretable logical rule discovery. 
While existing approaches for missing data either rely purely on statistical imputation or require fully observed data for rule induction, 
NS-FCN treats missing values as latent predicates and learns to infer them within a differentiable NeSy framework. 
The model establishes a closed-loop system where the imputed values feed back into rule discovery, promising to improve both imputation accuracy and interpretability over time. 
The combination of symbolic logic is done via a differentiable forward-chaining mechanism that uses soft logical operators to handle both continuous and discrete features. 
Empirical evaluations on synthetic and real-world datasets show that NS-FCN achieves high accuracy in imputing missing values while simultaneously finding human-interpretable rules that describe the data.

### Strengths
One strength of the paper, which is the main contribution by the authors, is proposing a closed feedback loop between imputation and rule discovery,  that allows the model to iteratively improve its reasoning and inference capabilities. 
As a merit, the framework handles heterogeneous data types, both discrete and continuous, by transforming continuous features into soft predicates, which is valuable in cases where both real and categorical values are considered. 
The model’s optimization method, based on coordinate gradient descent, seems to be a useful and scalable recipe for learning in this setting. 

The experimental analysis also reveals that the framework generates logical rules that remain meaningful and consistent with domain knowledge. The empirical studies are good and cover a wide setting, showing that the model performs competitively against black-box neural architectures while maintaining explainability.

### Weaknesses
While the experiments involve also different black-box neural competitors, the range of baselines could be expanded to include more recent generative models, such as those based on diffusion processes. Due to my limited experience on this specific topic, I do not understand if the baselines are the effective SotA or one could have repurposed auto-encoders or diffusions.


One thing is not mentioned and discussed is the relation to reasoning shortcuts, where learning may result in incorrect predicates and rules, as in [1,2,3]. 
Since NS-FCN operates with learning latent discrete values, an analysis of whether it avoids such behavior would have strengthened its claims about interpretability and reliability. 
In fact, the validation of interpretability remains largely qualitative; the paper reports interpretable rules, but lacks a systematic  assessment of their quality. 

Additionally, the dependence of the model’s performance on temperature hyperparameters in the soft logical operators is not clear and not enough explored.

------

[1] Learning with Logical Constraints but without Shortcut Satisfaction, Li et al., 2023 \
[2] Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts, Marconato et al. 2023 \
[3] Shortcuts and identifiability in concept-based models from a neuro-symbolic lens, Bortolotti et al., 2025

### Questions
I don't have further questions, I hope to see some discussion around the weaknesses I spotted.

### Soundness
3

### Presentation
3

### Contribution
3
