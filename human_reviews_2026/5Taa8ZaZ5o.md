# GAMformer: Bridging Tabular Foundation Models and Interpretable Machine Learning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
While interpretability is crucial for machine learning applications in safety-critical domains and regulatory compliance, existing tabular foundation models like TabPFN lack the transparency needed for these applications. Generalized Additive Models (GAMs) provide the needed interpretability through their additive structure, but traditional GAM methods rely on iterative learning algorithms (such as splines, boosted trees, or neural networks) that are fundamentally incompatible with the in-context learning paradigm of foundation models. In this paper, we introduce GAMformer, the first tabular foundation model for GAMs that bridges the gap between the power of foundation models and the interpretability requirements of real-world applications. GAMformer estimates GAM shape functions in a single forward pass using in-context learning, representing a significant departure from conventional iterative approaches. Building on previous research applying in-context learning to tabular data, we train GAMformer exclusively on synthetically generated tables. Our experiments demonstrate that GAMformer performs comparably to other leading GAMs across various classification benchmarks while maintaining full interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents GAMformer, an interpretable foundation model for tabular data. The model integrates Generalized Additive Models (GAMs) with Transformer architectures to enhance interpretability in tabular learning tasks.

### Strengths
- The visualizations of the experimental results are clear, well-designed, and easy to understand.

### Weaknesses
- **Limited novelty and weak contribution.** The work primarily combines Transformer and GAM architectures to achieve interpretability in tabular foundation models. However, the motivation for this integration is not sufficiently intuitive, and there are no notable technical innovations beyond this combination.
- **Writing and presentation issues.** The paper’s overall clarity, structure, and flow can be improved. Some definitions (such as *shape functions*) should be clearly introduced in plain language in the introduction for general readers. In addition, the key advantages of **GAMformer** over traditional GAM approaches are not explicitly articulated in the introduction.
- **Questionable experimental validity.** The experimental section lacks rigor and completeness.
	- Although the paper focuses on interpretability, standard performance metrics such as accuracy or MSE should be reported for all experiments to demonstrate that interpretability does not come at the expense of performance. Standard deviations over multiple random seeds should also be provided for statistical reliability.
	- For both real-world case studies, only **EBM** is used as a baseline. Additional more recent baselines should be included for a more comprehensive comparison.

### Questions
See the points listed in the **Weaknesses** section.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces GAMformer, the first tabular foundation model designed for Generalized Additive Models (GAMs). This work addresses the interpretability gap present in powerful tabular foundation models like TabPFN by integrating the additive, transparent structure of GAMs with the efficiency of the foundation model paradigm. GAMformer utilizes in-context learning (ICL) via a Transformer architecture to estimate non-parametric, binned representations of GAM shape functions in a single forward pass, thereby replacing traditional iterative fitting algorithms (like backfitting for splines or cyclic boosting for EBMs). The model is trained exclusively on large-scale synthetic data generated using Structural Causal Models (SCMs) and Gaussian Processes (GPs) priors, similar to Prior-Data-Fitted Networks (PFNs). Empirically, GAMformer demonstrates decent performance compared to leading GAMs (like EBM).

### Strengths
**Novel Paradigm Shift**: The core strength lies in GAMformer's ability to estimate interpretable GAM shape functions in a single forward pass using ICL. This fundamentally bridges the gap between powerful, pre-trained foundation models and the necessity of transparency in critical applications.

**Preserved Interpretability and Insights**: The case studies clearly demonstrate that GAMformer yields interpretable shape functions similar to EBMs. Crucially, the model successfully detected a potential data processing artifact (missing PFratio values) in the MIMIC-II dataset, highlighting its utility beyond mere prediction and confirming its deep interpretative capacity.

**Efficiency and Amortized Cost**: By moving the computational burden to pretraining, inference on new datasets is highly efficient, requiring only a single forward pass. The paper quantifies this benefit, estimating that the training cost is offset after 7.7 million forward passes compared to retraining an EBM.

### Weaknesses
**Performance Comparison**: The authors overclaim that the model achieves similar quality. The authors do not report the AUC metrics on MIMIC-II and on the regression example, there is a substantial difference in RMSE of GAMformer (68000) compared to EBM (58000). 

**Competing models**: For a more thorough understanding of the performance of GAMformer, the authors should include other models in Figure 6 and 7 e.g., NAM, NBM, NODE-GAM etc. This could potentially further highlight that even though GAMformer can capture the high-level pattern, the pattern can significantly deviate from many GAM libraries. 

**Real Datasets**: The authors consider very few datasets. Please consider a wider set of real and large datasets from the GAM literature, e.g., see the datasets considered in NODE-GAM paper.

**Limited Data Scaling Capacity**: The most pressing weakness is the model's performance degradation when presented with datasets larger than approximately 1000 data points (double the 500 in-context examples seen during training). This limitation, stemming from known transformer length extrapolation issues, severely restricts the definition of GAMformer as a truly scalable "Foundation Model" for large tabular data, contrary to the overall thesis title.

**Smoothness Bias in Shape Functions**: While the binned representation is explicitly chosen to model sharp discontinuities, the authors note that GAMformer produced "noticeably smoother" shape functions compared to EBMs on synthetic linear classification problems. This suggests a potential inherent bias towards smoother relationships derived from the pretraining process, possibly related to the high probability (4%) of Gaussian Process priors or the specific transformer mechanics, which warrants further investigation.

**Typos**
- Line 237: emmission -> emission 
- Line 723: colinearity -> collinearity
- Line 817: GAMformers’ -> GAMformer’s
- Line 863: using using -> using

### Questions
**Scaling Limitation Mitigation**: The paper identifies potential solutions for length extrapolation (e.g., TabFlex with linear attention, or TabICL). Could the authors elaborate on how the GAMformer architecture (specifically the bi-attention mechanism) might be adapted to incorporate these more scalable attention mechanisms to efficiently handle datasets significantly larger than 1000 samples?

**Shape Function Smoothness**: In Section 4.1, the GAMformer shape functions for synthetic data are described as "noticeably smoother" than EBMs. Given that EBMs use decision trees which naturally capture steps, and GAMformer uses binned, non-parametric outputs designed to capture discontinuities, what might be the source of this smoothness bias, and have the authors experimented with increasing the number of bins (currently 64) to make the learned functions sharper?

**Training Data Size Impact**: Figure 9 demonstrates that while GAMformer's AUC-ROC improves up to about $2\times$ the training context size (500 samples), EBMs continue to show higher accuracy improvements with larger datasets. Can the authors provide a more detailed analysis or discussion on the precise mechanism where GAMformer fails to "fully leverage additional training samples" beyond the $2\times$ limit, as noted in Section C.1?

### Soundness
2

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
4

### Summary
This paper introduces GAMformer, a tabular foundation model designed to estimate Generalized Additive Model (GAM) shape functions in a single forward pass via in-context learning (ICL). The approach embeds binned inputs and alternates attention across data points and features, decoding per-feature shape functions through a shared MLP. Predictions are obtained by summing per-feature contributions, maintaining GAM-style additivity and interpretability. The model is pre-trained exclusively on synthetic tables (to avoid data leakage) and evaluated on OpenML-style classification tasks and a clinical case study (MIMIC-II). The authors claim that GAMformer achieves strong predictive performance comparable to non-interpretable baselines while maintaining interpretability through additive structure and shape visualizations.

### Strengths
1. The paper proposes an original and computationally efficient way to integrate GAM-style interpretability with Transformer-based tabular models
2. The architecture elegantly balances interpretability and predictive performance, a key challenge in tabular deep learning. 
3. Experiments are conducted on both synthetic and real-world data, including MIMIC II, demonstrating practical applicability. 
4. The paper is well written and structured, making the technical ideas accessible.

### Weaknesses
The empirical section is on the weak side. Further analysis could better support some of the claims (e.g., robustness).  

1. Beyond Figure 5 (qualitative) and Figure 8 (evaluated on TabPFN test datasets), the performance comparisons are limited primarily to EBM and simulated datasets. While EBM is a strong baseline, the impact claims would be strengthened by including other interpretable model comparisons such as NAM, GA2M variants, or NODE-GAM, especially on real datasets (Figures 6 and 7).
2. Figure 5, NeuralNet and TabPFN are having competitive performance with GAMformer. In fact, GAMformer* performance is exactly the same as TabPFN. The AUC performance numbers on Fig 5 are really hard to see. These results are on simulated data only. 
3. Why did you use MIMIC-II not MIMIC-IV?
4. Overstated robustness claims: The paper claims robustness to label noise and class imbalance, but only one synthetic label-noise experiment is presented. This evidence is insufficient to support such a broad claim; additional experiments or a more cautious phrasing would be appropriate.
5. The core model is additive, with performance parity to XGBoost observed only after adding pairwise post-hoc effects. 
6. What is the interpretability-accuracy trade off when interactions are enabled?
7. Why are GAMformer and GAMformer* not compared directly on the MIMIC dataset?
8. The limitations section notes that GAMformer could “be misused to exploit biases,” such as in insurance or demographic contexts. Please clarify when careful consideration is required. Additionally, training solely on synthetic data may induce shape bias or over-smoothing (as observed vs. EBM on linear tasks). It would be valuable to analyze failure cases where synthetic priors mislead (e.g., sharp thresholds or non-monotonic relations) and test whether light fine-tuning on real data can correct these biases.

### Questions
Questions and suggestions are listed in the Weaknesses section above. The experimental section could be significantly strengthened with more experiments and comparisons of all the methods on real data.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents GAMformer, a tabular foundation model for GAMs using main and interaction effects. The paper proposes pre-training this model on synthetic datasets. The foundation model can then be used on the training set of a downstream dataset to obtain the different main and interaction effects in a single forward pass via in-context learning. The authors evaluate the performance of their model on synthetic and real-world datasets and obtain performance comparable to existing GAM methods.

### Strengths
- The paper is well written and structured, making it enjoyable to read. 
- The paper seems to propose a new and interesting approach to obtain a GAM via a simple forward pass through in-context learning.
- The results of their architecture seem promising.
- The interpretation of the results is interesting, whether on synthetic or real-world datasets.

### Weaknesses
1. Although the forward pass appears to be a low one-time cost for their model, I am not sure that the model scales to large numbers of samples and features. Is it possible to scale to several hundred thousand or millions of samples? Similarly, does GAMformer scale to several hundred or thousands of features? It would be interesting to compare the computation time of this forward pass with the training time of competing methods for different numbers of samples and features.

2. The number of GAM baselines is too low, many existing methods perform better than EBM. Could you, for example, compare GAMformer to NODE-GAM [1] and NAM [2] (with and without interactions)?

3. It seems that GAMformer does not support the integration of hierarchical constraints (strong and weak).

4. It would be interesting to evaluate the sensitivity of GAMformer to the number of bins used.

5. Although the paper is well written, there are some minor inaccuracies or errors.
- Line 93: I think there is a typo; you can remove “by” or split the sentence in two.
- Line 302 and in the caption of Figure 3: you state that your method systematically beats EBM. This is not true; EBM is better in the case (32 samples, 64 features). You should write “almost systematically.”

[1]: Rishabh Agarwal, Levi Melnick, Nicholas Frosst, Xuezhou Zhang, Ben Lengerich, Rich Caruana, and
Geoffrey Hinton. Neural additive models: Interpretable machine learning with neural nets. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing
Systems, 2021.

[2]: Chun-Hao Chang, Rich Caruana, and Anna Goldenberg. NODE-GAM: Neural generalized additive model
for interpretable deep learning. In International Conference on Learning Representations, 2022.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
