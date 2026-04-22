# Unveiling the Role of Data Uncertainty in Tabular Deep Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Recent advancements in tabular deep learning have demonstrated exceptional practical performance, yet the field often lacks a clear understanding of why these techniques actually succeed. To address this gap, our paper highlights the importance of the concept of data uncertainty for explaining the effectiveness of recent tabular DL methods. In particular, we reveal that the success of many beneficial design choices in tabular DL, such as numerical feature embeddings, retrieval-augmented models, and advanced ensembling strategies, can be partially attributed to their implicit mechanisms for performing well under high data uncertainty. By dissecting these mechanisms, we provide a unifying understanding of recent performance improvements. Furthermore, the insights derived from this data-uncertainty perspective directly allowed us to develop more effective numerical feature embeddings as an immediate practical outcome of our analysis. Overall, our work paves the way toward a foundational understanding of the benefits introduced by modern tabular methods that results in the concrete advancements of existing techniques and outlines future research directions for tabular DL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a unifying, sample-wise lens—data (aleatoric) uncertainty—to explain why several recent choices in tabular DL tend to help. Using per-example uncertainty (estimated via a probabilistic CatBoost fit), the authors draw “uncertainty plots” that sort test samples by estimated uncertainty and compare MSE deltas between baselines (MLP) and—(i) numerical embeddings, (ii) retrieval-augmented models (ModernNCA), and (iii) parameter-efficient ensembling (TabM). The key empirical claim is that these techniques help disproportionately more on the high-uncertainty slice. This framing also motivates a small practical contribution: a triplet-trained numerical embedding that modestly improves over LRLR embeddings across multiple datasets.

### Strengths
Clear, unifying viewpoint that organizes disparate tabular DL results under one explanatory factor (data uncertainty). The uncertainty-slice analysis feels insightful and actionable.  

Careful sanity checks: synthetic datasets with known uncertainty; robustness of findings to different uncertainty estimators (Appendix I).  

Mechanistic probes beyond benchmarks (e.g., neighbor-target consistency, gradient clean/noisy decomposition for TabM).

Breadth: results cover several public tabular benchmarks; includes a pruned “Uncertain CIFAR-10” to stress uncertainty effects across settings.  

Reproducibility: code and notebooks promised; tuning spaces documented.

### Weaknesses
The core plots hinge on CatBoost RMSEWithUncertainty estimates and Gaussian smoothing; while Appendix I shows estimator-robustness, the absolute calibration of uncertainty remains unvalidated on real data beyond rank correlations seen on synthetic tasks. Consider evaluating calibration (e.g., probability-integral transforms / reliability curves for σ̂(x)) and reporting agreement across multiple families (GBDT, MLP-heteroscedastic heads, NGBoost).  

Uncertainty plots are compelling but largely visual. Provide confidence bands or per-quantile paired tests (e.g., Wilcoxon on per-example ΔMSE within quantile bins) to quantify “disproportionate” improvements, and report effect sizes across datasets. 

Focus is mostly regression; classification appears only via Uncertain CIFAR-10 with 50 features retained (Table 1). Stronger support on standard tabular classification (incl. class-imbalance) would broaden impact.  

The paper explicitly does not analyze epistemic uncertainty or generic regularizers (dropout, weight decay, SAM), though these interact with label noise and could compete with retrieval/ensembling. 

Gains over LRLR are modest (Δ rank and Δ% vs MLP are small), and details like triplet mining strategy, margin, and overhead are not fully explored. Please expand comparisons vs other metric-learning or contrastive objectives (e.g., supervised contrastive, NCA-style losses) and quantify cost/benefit.

### Questions
Compare CatBoost RMSEWithUncertainty vs heteroscedastic MLP, NGBoost, CatBoost ensembles; evaluate correlation and calibration on real datasets (e.g., split-and-replicate labels when possible).  
Vary noise and show how ModernNCA/TabM curves move; verify TabM’s gradient noise reduction in practice with loss-landscape / gradient-norm probes.  
Add standard classification datasets (adult, higgs, covertype, etc.) and relate ECE/NLL improvements to uncertainty. (You already include Uncertain CIFAR-10.)  
Test if triplet pretraining helps ModernNCA backbones specifically.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyzes the correlation between _data uncertainty_ and the success of recent tabular deep learning models. The authors first start by demonstrating that XGBoost shows much stronger performance in high-uncertainty regions, and that recent tabular DL models start showing similar trends. Then the authors examine the performance of DL tabular models, on two synthetic datasets as well a suite of real-world datasets. The authors also propose an improvement to an existing tabular embedding model by incorporating uncertainty-reducing contrastive loss by applying the finding that embeddings "higher quality neighborhoods" lead to better uncertainty mitigation. Overall, this paper provides a set of results that help explain the success of recent tabular DL models and how the gap between GBDTs may be tackled.

### Strengths
- This paper presents an interesting analysis that reveals the connection between data uncertainty and the success of existing SotA tabular models. I think this is a very important contribution to the field, given that this same discussion has been brought up multiple times (WhyTrees, Tabzilla) but not fully explored. The findings from this paper seems to pinpoint a solid factor in the success of GBDT models, and also provide a path forward for tabular DL models to close the gap.
- The experiments are well designed and the results are clearly presented. The sawtooth dataset and the decision boundary are a nice touch for visualizing the effect of uncertainty.
- In addition to the analysis, the authors also propose a simple improvement to one of the analyzed methods that show consistent improvement.

### Weaknesses
- This paper only focuses on regression datasets. While this is understandable given the uncertainty estimation method, it would be nice to see if 1) can we have a similar notion of uncertainty for classification datasets, and 2) do the same trends hold.
- The entire analysis is based on the uncertainty-detection described in section 3.1. The authors do validate the robustness (and consistency) of this method, but the findings are contingent on this specific uncertainty estimation method.
- I found the figures with uncertainty in the x axis a bit difficult to follow at first. A potential suggestion here is to normalize the uncertainty values per dataset (divide by max) so that everything appears on the same scale.

### Questions
- Are there examples we can see of high-uncertainty sample vs. a low-uncertainty sample from the same dataset? That would be very nice.
- What exactly is the difference between -LRLR and -PLR? Are they both using the scheme from Gorshniy et al. 2022?
- For the contrastive loss, how are the positive and negative samples selected? Given that the datasets are regression datasets, is it some thresholding? And is it weighted?
- How does this new embedding method compare against other models? like ModernNCA or TabM? And/or can it be integrated into those models as well?
- The paper starts off by showing that GBDTs perform strongly in high-uncertainty regions. But then the rest of the paper focuses on tabular DL models. Are there more insights into what makes GBDTs so effective in these regions? Is it just the model capacity, or is there something more fundamental about the tree structure that helps?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates why certain design choices in tabular deep learning—such as numerical embeddings, retrieval-augmented architectures, and model ensembling—often outperform standard MLPs. The authors propose that these methods are particularly effective in regions of high data uncertainty and develop a framework to quantify such uncertainty using heteroscedastic Gaussian modeling. Through experiments on both synthetic and real-world datasets, they show that the relative advantage of these architectures grows with estimated noise levels. The study provides a novel perspective linking model design and data uncertainty in tabular learning.

### Strengths
The paper provides a clear and well-motivated analysis of why certain design choices in tabular deep learning, such as embeddings, retrieval augmentation, and ensembling, might succeed. Framing these differences through the lens of data uncertainty offers a fresh and conceptually coherent perspective.

The paper is well written, with a clear motivation and logical flow.

### Weaknesses
1. **Narrow treatment of data uncertainty**

The paper assumes that all uncertainty arises from Gaussian noise and evaluates only this setting. Other realistic forms of noise—such as asymmetric, heavy-tailed, missing-feature, or label noise—are not considered. Consequently, it remains unclear whether the observed model advantages genuinely reflect robustness to data uncertainty in general or merely to this specific assumption. The conclusions may not hold once the Gaussian hypothesis is relaxed.

2. **Missing comparisons with strong modern baselines**

The experiments omit several state-of-the-art tabular models, notably TabPFN v1/v2 and ExcelFormer. TabPFN has shown outstanding performance across diverse benchmarks, while ExcelFormer claims improved robustness to noisy inputs in its experiments.

3. **Limited realism of real-world evaluations**

Although several real-world datasets are used (e.g., California Housing), they are relatively clean public benchmarks with minimal noise and well-structured features. The study does not consider genuinely messy or uncurated data, where uncertainty is more complex and multifactorial, particularly in domains such as healthcare or finance. Evaluating the approach on raw, noisy real-world datasets rather than sanitized benchmarks would provide a more compelling demonstration of how the proposed approach handles real noise in practice.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the impact of label noise on different variants of DL models for predictive tabular tasks. To do so, the authors propose an experimental protocol to estimate label noise on real-world datasets (and validate this setup in synthetic tasks). As a result, they show that recent performance improvements by introducing better embeddings, ensembling, and retrieval-augmented methods to DL models are correlated with a better handling of noisy labels.

### Strengths
[Originality and Novelty] This paper presents a creative and well-executed empirical study on the impact of recent design choices. It offers a novel perspective and provides valuable insights into model behavior, contributing to the development of better models for predictive tabular tasks.

[Quality] Overall, I really enjoyed reading this paper. It is well written with useful visualizations, plots, and interesting observations. The experiments are well justified and clearly explained, and the findings are based on a solid empirical foundation supported by large-scale experimentation. The limitations section also provides an honest assessment, helping contextualize the findings' impact and significance.

### Weaknesses
[model choice] With the advent of many pre-trained foundation models for predictive tasks, an obvious question is how such models would perform in this experiment. Although the authors acknowledge the absence of such state-of-the-art models (and proposed to study them for future work), I would've hoped for a use case (where such models cannot be used) to provide a clear justification for why such models were not investigated. Additionally, since many experiments emphasize neighborhood-based methods, including a simple KNN baseline (e.g., in Figure 5) for comparison would be informative.

[robustness of results] The experiments are conducted on a large scale with HPO; however, the sensitivity of the results to hyperparameters, and especially model capacity, is not discussed. Reporting variability across repetitions or different synthetic dataset samples would also be valuable.

[generalization of results] While the experiments provide a strong signal, the empirical analysis is limited to a few hand-selected datasets and only ablates three design decisions, with notable overlap between dataset authors and model contributors. This observation raises questions about the generalizability of the findings to broader settings.

### Questions
My current rating reflects the paper in its present state (while I remain open to revising my opinion), considering the weaknesses discussed above and the questions listed below. The experimental framework is very useful and provides a valuable lens to neutrally study model behavior. However, the small range of datasets, models, and tasks studied limits the potential impact the paper could (and should) achieve. The authors are welcome to address any of the raised weaknesses, but I am particularly interested in their response to the following points:

1) [minor; clarification] The limitation section states that this work focuses on regression tasks (such as the datasets used in Section 5). However, the datasets in Sections 4.2 and 4.3 seem to be classification tasks. Is there something I'm missing?

2) [relation to RealMLP] It would be helpful if the authors could discuss how their findings relate to or might extend to RealMLP which, among other topics, also studies different encoding strategies; see https://dl.acm.org/doi/10.5555/3737916.3738753

### Soundness
4

### Presentation
3

### Contribution
3
