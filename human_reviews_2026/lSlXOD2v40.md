# Explaining Concept Shift with Interpretable Feature Attribution

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Regardless the amount of data a machine learning (ML) model is trained on, there will
inevitably be data that differs from their training set, lowering model performance. Concept
shift occurs when the distribution of labels conditioned on the features changes, making
even a well-tuned ML model to have learned a fundamentally incorrect representation.
Identifying these shifted features provides unique insight into how one dataset differs from
another, considering the difference may be across a scientifically relevant dimension, such
as time, disease status, population, etc. In this paper, we propose SGShift, a model for
detecting concept shift in tabular data and attributing reduced model performance to a
sparse set of shifted features. We frame concept shift as a feature selection task to learn
the features that can explain performance differences between models in the source and
target domain. This framework enables SGShift to adapt powerful statistical tools such as
generalized additive models, knockoffs, and absorption towards identifying these shifted
features. We conduct extensive experiments in synthetic and real data across various ML
models and find SGShift can identify shifted features with AUC > 0.9, much higher than
baseline methods, requires few samples in the shifted domain, and is robust in complex
cases of concept shift. Applying SGShift to 2 real world cases in healthcare and genetics
yielded new feature-level explanations of concept shift, including respiratory failure’s
reduced impact on COVID-19 severity after Omicron and European-specific rare variants’
impact on Lupus prevalence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SGShift, a model for detecting concept shift in tabular data by attributing performance difference to a sparse set of shifted features within the feature selection framework. Three variants of the method are introduced: SGShift, SGShift-A, and SGShift-K, with the latter two addressing source model mismatch and controlling the false discovery rate. The model is evaluated on three medical datasets.

### Strengths
The paper is clearly written and well structured, and the three variants of the proposed method are clearly organized.

### Weaknesses
1. Table 1 and the loss curve in Figure 2 is missing confidence intervals.

2. The experimental setup seems somewhat limited. How are the features in each dataset selected for inducing the concept shift? How does the model performance change when selecting high-correlation features versus low-correlation features? How do the results vary under different signal-to-noise ratios? I believe a more detailed analysis in these aspects would strengthen the paper.

3. How do SGShift, SGShift-A, and SGShift-K perform in each scenario? I think including such ablation studies could provide deeper insights into the contributions of each variant.

4. I don’t fully understand part B in Figure 2. How exactly are features added during the SGShift updates? Additionally, many other methods have been proposed to address concept shift, how would their loss curves compare when applied on top of SGShift?

### Questions
Please refer to the weakness section above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes SGShift, a method to identify features responsible for concept shift by learning a sparse correction on top of a fixed source model rather than retraining two models. It combines a generalized additive model with l1 regularization and optional knockoff-based FDR control. Experiments include semi-synthetic setups where conditional shifts are simulated, and two real health case studies (COVID-19 and lupus). Results show SGShift detects shifted features more accurately than prior baselines in controlled conditions.

### Strengths
The method is well-motivated and theoretically grounded. Modeling conditional changes as a sparse correction to a fixed predictor is a reasonable and efficient approach. Semi-synthetic results are clear and consistent, showing reliable feature recovery under designed concept shifts. The theoretical analysis is sound, and the paper is clearly written and organized. The real-world examples are interesting, though they mostly serve as demonstrations rather than solid proof of conditional shift

### Weaknesses
1. Real-world concept shift validity is unclear
It is unclear whether the real-world studies truly exhibit concept shift. For COVID-19, a shift in clinical patterns is plausible, but many other factors (e.g. testing policy, vaccination, care protocols) also changed, affecting p(X) and label definitions. Without reweighting or calibration analyses to separate covariate or label shift, the claim of conditional change remains suggestive rather than proven. The same issue applies to the Diabetes and SUPPORT2 splits, where differences may come from case mix or selection, not true conditional changes.

2. Lupus case likely reflects representation shift.
The lupus cross-ancestry experiment attributes predictive differences to conditional changes, but across ancestries, measured features often differ in distribution and linkage. Unless features are harmonized or mapped to a shared representation, this is likely representation or covariate shift, not p(y∣X) change. The findings are still biologically interesting but should be framed accordingly.

3. No diagnostics separating shift types.
There lacks standard diagnostics such as source-to-target reweighting tests, conditional calibration curves, or reliability plots to confirm residual error after matching p(X). Without these, it’s hard to verify the central claim that SGShift recovers features driving concept shift rather than other forms of drift.

### Questions
1. For the COVID-19 study, did you check whether performance gaps remain after reweighting the source cohort to match the target covariate distribution?
2. How do you distinguish p(y∣X) changes from label or measurement drift, especially when care policies evolved over time?
3. In the lupus study, can you harmonize gene features or use pathway-level embeddings to ensure observed shifts are not due to ancestry-specific feature distributions?
4. Could you include simple diagnostics (e.g. reweighting or calibration plots) in the appendix to directly test for conditional vs covariate shift?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the problem of detecting distribution shift, focusing on changes in the conditional distribution of labels given the input features and the labels $p(y|x)$. Under the common assumption that the distribution shifts are sparse, the authors frame the detection of shift between the source and target domains as a sparse regression problem over the input features. They propose SGShift, a method that learns a sparse correction to a predictor trained on the source domain and adapts it to the target distribution to identify which features are responsible for the shift. The paper also introduces variants that account for possible errors in the source model (SGShift-A) and control false feature discoveries using knockoffs (SGShift-K). The approach is evaluated on three healthcare datasets, Diabetes Readmission, COVID-19 Hospitalizations, and Support2, covering both semi-synthetic (simulated distribution shift) and real-world shifts, and the results show that SGShift often outperforms the baselines.

### Strengths
- The proposed approach and its variants are rigorous, supported by both theoretical analysis and thorough empirical validation. The formulation of distribution shift detection as a sparse regression problem is well-motivated, and the inclusion of variants that account for model misspecification and false discovery control adds robustness and credibility to the framework.

- While the idea of modeling distribution shift through sparsity assumptions is not entirely novel, and is widely used in causal representation  and invariant learning literature , the particular formulation of learning a sparse correction to the predictive model across domains is novel best of my knowledge.

- The paper is also well-presented overall. The methodology is easy to follow and the theoretical results are explained with sufficient intuition.

### Weaknesses
- The authors should cite relevant work on causal representation learning to properly justify the sparsity assumption. This assumption is well established in the causality literature through the concept of sparse mechanism shift, which posits that under the correct causal factorization, only a small subset of mechanisms is expected to vary across domains (Schölkopf et al., 2021). This conceptual connection is currently missing from the paper.

- Several aspects of the main experiments in Section 4.1 could be improved for clarity and completeness:

  - It is unclear which version of SGShift is used in the experiments. Does the implementation include the absorption term, or does it correspond to the knockoff variant? An ablation study illustrating the effect of these different design choices would substantially strengthen the paper.
 
  - In Table 2 (dataset statistics), the target domain appears to have a sufficient number of samples relative to the feature dimensionality. Given this, it is unclear why baseline methods perform worse, especially since the main critique of plug-in approaches (line 150) was the potential error from training a separate model on the target domain.

   - Please include standard deviations or standard errors in the Table 1 results to improve the statistical rigor and interpretability of the findings.

   - Why does SGShfit continue to perform well even for the dense simulation case? There should be more discussion around this as I would expect the results to be worse since the sparsity assumption is violated. 

   -  The observation that matched accuracy is sometimes lower than the unmatched case in Table 1 is counterintuitive and warrants explanation.

- The authors should also include experiments with higher-dimensional feature spaces, as the current setup (maximum of 64 features for the Support2 dataset) is relatively low-dimensional. Even if suitable real-world healthcare datasets are not available, a synthetic high-dimensional dataset could serve to demonstrate the scalability and robustness of the proposed method.

### Questions
Please refer to the weaknesses section above for my main questions.

Minor suggestions

- The citation for the WhyShift framework (line 146) is missing,

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method that tackles the problem of concept shift in tabular machine learning tasks, explaining it and rectifying predictor performance degradation caused by it (SGShift). Instead of separately modelling the source and target domains, SGShift learns a sparse corrective term on top of a source-trained model, identifying a small set of features that explain performance degradation under the concept shift. For this, it leverages $l_1$ sparsity inducing regularisation, and knockoffs to avoid over-selection of features. 
Experiments are conducted on (semi)synthetic and real healthcare/genetics datasets. On the synthetic data, SGShift is compared to 3 baselines (Diff, WhyShift, SHAP) and is superior than those at recovering shifted features (AUC > 0.9). 
The real data experiments illustrate how SGShift explains concept shift in practice and how it can recover performance based on its prediction correction term that only uses very few features.

### Strengths
- The paper tackles a relevant and well-scoped problem (concept shift in tabular data with limited target data available) 
- The key idea (and potentially contribution) of the paper is simple and easily understandable (the idea of the sparsity of concept shift), and straightforwardly addressed with a simple and easy to implement method.

### Weaknesses
Major:
- The paper is unclear at many places and as such hard to correctly interpret or assess (see questions), in particular also as to assessing the strength of the empirical results
- More empirical results on real world data would be needed to understand if the key idea and contribution is really relevant (i.e. if sparsity of concept shift is really a commonly occurring thing in practice) 
- Empirical results in Section 4.2 are not compared to any baseline 
- The effects of the variants introduced in Section 3.2 and 3.3 (SGShift-A and SGShift-K) are not clear, as the empirical results only contain SGShift (vanilla). Without adding them to the experiment section there is not much point introducing them. 


Minor:
- For clarity, I would suggest introducing SGShift-A and -K not just in the subsection titles but also in the text itself as readers sometimes don’t read section titles carefully. 
- It would be helpful for the reader to add 1-2 sentences at the end of Section 3 that explain in plain English what the results in Theorem 3.2 mean. 
- Figure 1: Please use vector graphics for legend, looks very blurry right now

### Questions
- Eq (1): why is $g$ introduced here? It is never used again in the paper afterwards
- Eq (3): what is $\phi_S^T$? The same as $\phi(X_T)^T$? This change in notation should be explained somewhere. 
- Line 215: what is $\hat{f}$?
- Line 235: What is $n_T$? Needs to be introduced.
- Lines 234, 235 and 236: What to the modified $\leq$ and $=$ signs stand for? Please define this notation. 
- Line 244: what is $\hat{A}^{[b]}$? Where is it defined?
- Line 247: what is $\hat{A}$? Where is it defined?
- Line 268 - 270: is it the features that are shifted (i.e. $P(X)$) or their relationship to the labels $P(Y|X)$? In the text it sounds like it is the features which does not correspond to the problem of concept shift being tackled in the paper though?
- Section 4.1: what regularisation strength $\lambda$ was used for the sparse simulation and dense simulation respectively? 
- Section 4.2 healthcare experiment: What is the source and the target data here? How many samples from the target data are being used?
- Section 4.2 genetics experiment: How many samples from the target data are being used? What AUC could be recovered on the Asians with SGShift here?

### Soundness
2

### Presentation
1

### Contribution
2
