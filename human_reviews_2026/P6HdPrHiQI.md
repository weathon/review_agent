# Estimating Time Series Foundation Model Transferability via In-Context Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Time series foundation models (TSFMs) offer strong zero-shot forecasting via large-scale pre-training, yet fine-tuning remains critical for boosting performance in domains with limited public data. With the growing number of TSFMs, efficiently identifying the best model for downstream fine-tuning becomes increasingly challenging. In this work, we introduce TimeTic, a transferability estimation framework that recasts model selection as an in-context-learning problem: given observations on known (source) datasets, it predicts how a TSFM will perform after fine-tuning on a downstream (target) dataset. TimeTic flexibly organizes observed model–data relationships as contextual information, allowing it to adapt seamlessly to diverse test-time scenarios. Leveraging the natural tabular structure formed by dataset meta-features, model characteristics, and fine-tuned performance, we employ tabular foundation models to serve as in-context learners. We further introduce a novel model characterization based on entropy evolution across model layers, capturing embedding-space distinctions and enabling TimeTic to generalize across arbitrary model sets. We establish a comprehensive benchmark for transferability estimation including 10 datasets, 10 foundation models, and 3 forecasting tasks. On this benchmark, TimeTic's estimation demonstrates strong alignment with actual fine-tuned performance for previously unseen datasets, achieving a mean rank correlation of approximately 0.6 and a 30\% improvement compared to using zero-shot performance as the transferability score.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper the authors propose TimeTIC, a framework that estimates the fine-tuning performance ranking of a list of time series foundation models on a specific target dataset. The authors proposed to featurize the dataset by a selection of its statistical characteristics, and to featurize a TSFM by its per-layer entropy profile. A table of dataset features, model features and the FT'ed model's performances is created accordingly and is extrolated by a tabular foundation model onto new models and new datasets. The authors show empirically that the proposed method improves over the ranking list based on zero shot performance.

### Strengths
The paper attempts to answer a legit practical question of pretrained base model selection without paying the cost for fine-tuning. Its writing is rather concise and clearly. The proposed entropy profiling of transformer based TSFM is novel.

### Weaknesses
Though the research question is legit, the setup proposed in this paper is quite dubious that it's unclear about the actual contribution.
(1) Fine-tuning is still required to construct the context part of the tabular data.
(2) More importantly, all models are fine-tuned under the same setup whereas there is no discussion regarding what is the best fine-tuning setup given a specific pretrained model and a target dataset, which is in fact a more correct and practically useful question to ask here.
(3) It is unclear why a tabular foundation needs to be involved.

In addition, the paper might have ignored some details regarding its models and datasets in the empirical study section which undermines its credibility. 

See Questions for details.

### Questions
1. In the tabular part, y_context is essentially a function of the pretrain model, the target dataset, and the fine-tuning setup. Can you clarify if the study assumes that the fine-tuning setup is fixed? If so how to justify the conclusions in the paper when a common practice is to optimize the fine-tuning hyperparameters?

2. An alternative of doing time series dataset featurization dimension reduction and applying a pretrained tabular model that I can think of is to NOT do any dimension reduction, and apply a sparse estimator (e.g., the classic LASSO) to estimate the fine-tuned performance. The dimension reduction is basically required by the tabular foundation model as it cannot cope with inputs of very high dimension - why a tabular model along with the in-context notion is introduced here in the first place?

3. For the empirical study, especially regarding the 10 selected datasets, some of them are actually part of the pretraining data of the selected TSFM. For example, Moirai 1.0 and TimesFM 2.0 (the 500m) may both have used the sz-taxi dataset included in LOTSA, Chronos may have used solar included in Chronos data. How does this fact a target dataset being in distribution vs zero shot affect your conclusions in your benchmark?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Overall, this paper is clearly written, well-motivated, and proposes a sensible formulation with solid time series characterization and convincing empirical results. However, the rationale for dismissing vision-derived metrics and, critically, for choosing TabFMs remains under-argued, and several design details (entropy estimator, profile subsampling) need clarification. With tightened justification and minor fixes, it would be a strong accept.

### Strengths
- The paper targets the practical need to pick a TSFM to fine-tune without exhaustively training every candidate. The contrast with enumeration (Fig. 1a) and the ICL framing (Fig. 1b) make the motivation concrete. The introduction also explicitly positions prior categories (statistical metrics vs. meta-learning) and why they fall short for TSFMs.  

- Casting transferability estimation as characteristics then performance regression with a tabular context is elegant and flexible. The split between an offline context table and online target inference is simple and scalable.

- The feature extraction/selection pipeline is well-designed: >700 features are reduced using the TotalVariance proxy for epistemic uncertainty, yielding a compact feature representation with empirical justification. Model characterization is also interesting. The entropy-profile idea is architecture-agnostic and easy to compute; the paper shows distinctive patterns across families and sizes (Fig. 3-right). The decision to use at most 10k tokens per layer is pragmatic.

- The evaluation is comprehensive in general. A 10×10×3 benchmark, standardized fine-tuning, and both standard and few-shot regimes are valuable. TimeTic consistently beats baselines in Kendall’s τw and achieves ~0.6 Spearman, incl. generalization to unseen models/datasets (Table 1, Fig. 4–5).

### Weaknesses
- The intro says most statistical metrics are from image classification and “depend on strong assumptions about the class structure,” but does not unpack which assumptions make them ill-suited for forecasting (e.g., categorical output distributions, class-conditional feature separability, linear link to labels). This reads as asserted rather than demonstrated.

- The paper notes recent advances in TabFMs and adopts TabPFN, but it stops short of articulating why TabPFN’s inductive biases are appropriate for mapping the concatenated data/model characteristics (plus zero-shot) to fine-tuned performance. Strengthening Sec. 3’s final paragraph with concrete reasoning or evidence (e.g., comparisons to simpler regressors or ablations on TabPFN variants) would better justify the choice.

- Sec. 3.2 uses a “continuous entropy estimator (Kraskov et al.)”, but Kraskov et al. is a mutual-information estimator. If a kNN-style differential entropy estimate is used, the estimator variant, k, bias correction, and dimensionality handling (token-embedding dimension) should be specified, as these strongly affect stability. Currently the description reads generic. Also, profiles are downsampled to length 6 but this convenience choice may blur informative differences for deeper models; a sensitivity analysis or a depth-normalized alignment would help.

- Minor writing/formatting issues. For example, the parenthetical “(Kraskov et al.)” is incomplete, and given the point above, potentially misleading - please cite the exact estimator and year. There are also a few small typos (“Ground Turth” in ToC).

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TIMETIC, a practical framework for estimating the fine-tuned performance of Time Series Foundation Models (TSFMs) on unseen datasets without performing actual fine-tuning. The core idea formulates transferability estimation as an in-context prediction task, leveraging a tabular foundation model (TabPFN) to learn the mapping from dataset and model characteristics to expected performance. However, three core components of this paper, including time series characterization, model characterization, and in-context transferability estimation, are merely combinations of existing techniques, lacking sufficient originality or innovation.

### Strengths
1. As TSFMs gain increasing attention, efficiently assessing their applicability to downstream tasks without the cost of full fine-tuning has become a critical challenge. TIMETIC directly addresses this bottleneck, offering substantial value for automated model selection, especially in resource-constrained or time-sensitive scenarios.

2. The core idea reframes transferability estimation as an in-context prediction problem, where a tabular foundation model (TabPFN) is prompted to learn the mapping from the characteristics of datasets and models to their expected performance.

3. The introduction of TotalVariance as a proxy for epistemic uncertainty provides a principled basis for feature selection, improving generalization by focusing on the most informative and stable features. 

4. Representing models through information entropy profiles across network layers provides a meaningful and interpretable way to encode architectural differences (e.g., encoder-only vs. encoder-decoder).

### Weaknesses
1. While the paper formulates transferability estimation as an in-context prediction task, the three core components, time series characterization (using TotalVariance for feature selection), model characterization (based on an entropy estimator), and in-context transferability estimation (relying on the pre-existing TabPFN), are largely integrations of established techniques, offering limited originality or conceptual innovation.

2. I am also puzzled by the computation process of the entropy profile, particularly the step of subsampling each entropy profile to a fixed length of six to ensure consistent representation across models with varying depths. Why is the minimum depth set to six? Given that TSFMs typically have many more layers, how exactly is the subsampling performed across the layers? Could this subsampling cause deeper models to lose important information? And how would the method handle models with depths fewer than the preset value?

3. Although you frame model selection as an in-context learning problem and use TabPFN to predict model performance on unseen datasets, this essentially amounts to a regression task. Why do other standard regressors perform poorly on this task? If the superior performance is largely attributed to the inherent strength of TabPFN, it raises questions about the extent of the proposed method's own contribution.

4. The ground truth fine-tuned performance of various TSFMs in Table C also confuses me. (1) Why do the results of different models differ so significantly on the same dataset? For example, in the short-term forecasting tasks on the solar:H dataset, why is the performance of TimesFM-500M as high as 90.266, while other models are around 1–2? Generally, larger models within the same architecture are expected to perform better. Similar issues appear in medium-term and long-term tasks. (2) Why do the results for some datasets show that short-term forecasting performs much worse than medium-term or long-term, despite long-term forecasting typically being a more challenging task? For instance, on the bizitobs l2c:5T dataset, the short-term result of Chronos-tiny is 22.862, while the long-term result is only 1.140. I believe inaccuracies in these ground truth fine-tuned performances could significantly affect the evaluation of your method.

### Questions
1. Please further clarify the originality and conceptual contribution of your method, as the core components are largely based on existing techniques, which currently limits the perceived novelty of the work.

2. Please provide a detailed explanation of the entropy profile computation process, including the rationale for setting the fixed length to six, the specific subsampling strategy used across layers, how models with fewer than six layers are handled, and an analysis of potential information loss in deeper models.

3. Please justify the choice of TabPFN over standard regression models by providing comparative results with alternative regressors. If TabPFN’s strong performance is central to your method, please clearly delineate how much of the success stems from your framework design versus the underlying model’s capabilities.

4. Please verify and explain the ground truth fine-tuned performance values reported in Table C. Specifically, address the extreme performance discrepancies across models on the same dataset (e.g., TimesFM-500M vs. others on solar:H) and the counterintuitive results where short- or medium-term forecasting performs significantly worse than long-term (e.g., on bizitobs l2c:5T), as inaccuracies here could undermine the validity of your evaluation.

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
This paper tackles how to quickly choose the best TSFM to fine-tune under limited downstream data. The authors recast model transferability estimation as a table-like ICL problem: they organize mappings of (target dataset features + candidate model representations + historical (dataset, model) → post-fine-tuning performance) into a context table, and use a pretrained tabular model (TabPFN) to predict performance without updating parameters. They also propose a layer-wise entropy profile as an architecture-agnostic model representation.

### Strengths
The pipeline is easy to follow: dataset feature extraction → model representation → table construction → ICL inference.

The evaluation covers multiple domains, frequencies, and horizons, and tests three generalization settings (unseen datasets, unseen models, and both unseen).

Nice code is provided, improving reproducibility and practical usability.

### Weaknesses
Learner dependence: The method mainly relies on TabPFN. Although its ICL nature fits the “no retraining” goal, there is no comparison with other tabular learners (e.g., FT-Transformer, SAINT, XGBoost/CatBoost).

Model representation evidence: The entropy profile is promising, but evidence is limited on why it reliably reflects “fine-tune-ability” across TSFMs. Suggested additions:

Systematic comparisons against alternative representations (feature entropy/information gain, Fisher information, H-score variants, NCE, gradient/activation statistics).

Robustness under shift (sampling rate changes, seasonality, spikes): does the profile spuriously vary with frequency or scaling?

Motivation and cost trade-off:

Since TSFMs are small and cheap to fine-tune, is predicting post-fine-tuning performance a strong practical need? How many time can you save with different GPUs.

The method requires offline computation across multiple (dataset, model) pairs to build the context, which may conflict with the “reduce cost” goal.
Consider the cost of finetuning, I would rather simple choose ensembling a stronger and often more stable baselinet

### Questions
Please check weakness

### Soundness
3

### Presentation
3

### Contribution
2
