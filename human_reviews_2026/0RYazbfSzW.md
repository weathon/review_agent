# Accelerating Benchmarking of Functional Connectivity Modeling via Structure-aware Core-set Selection

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Benchmarking the hundreds of functional connectivity (FC) modeling methods on large-scale fMRI datasets is critical for reproducible neuroscience. However, the combinatorial explosion of model–data pairings makes exhaustive evaluation computationally prohibitive, preventing such assessments from becoming a routine pre-analysis step. To break this bottleneck, we reframe the challenge of FC benchmarking by selecting a small, representative *core-set* whose sole purpose is to preserve the relative performance ranking of FC operators. 
We formalize this as a ranking-preserving subset selection problem and propose **S**tructure-aware **C**ontrastive **L**earning for **C**ore-set **S**election (**SCLCS**), a self-supervised framework to select these core-sets. **SCLCS** first uses an adaptive Transformer to learn each sample's unique FC structure. It then introduces a novel **S**tructural **P**erturbation **S**core (**SPS**) to quantify the stability of these learned structures during training, identifying samples that represent foundational connectivity archetypes. 
Finally, while **SCLCS** identifies stable samples via a top-$k$ ranking, we further introduce a **density-balanced sampling strategy** as a necessary correction to promote diversity, ensuring the final core-set is both structurally robust and distributionally representative. On the large-scale REST-meta-MDD dataset, **SCLCS** preserves the ground-truth model ranking with just 10% of the data, outperforming state-of-the-art (SOTA) core-set selection methods by up to 23.2% in ranking consistency (nDCG@k). To our knowledge, this is the first work to formalize core-set selection for FC operator benchmarking, thereby making large-scale operators comparisons a feasible and integral part of computational neuroscience. Code is publicly available on: [https://github.com/lzhan94swu/SCLCS](https://github.com/lzhan94swu/SCLCS)

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a core-set selection framework to more effectively benchmark many SPI measures on large fMRI datasets. This method first selects a small but representative subset of subjects, and then use it to rank the SPI methods. The goal is not to maximize the accuracy of any single model, but to preserve the relative ranking between SPI methods in terms of downstream task performance.

### Strengths
1. The motivation is clear and convincing. If we can preserve the ranking of SPIs using only a small subset, then SPI selection for a known downstream task becomes easier and faster.

2. The paper provides a well-defined evaluation objective. Instead of reporting absolute accuracy only, the authors explicitly measure how well the core-set preserves the global ordering of SPIs.

### Weaknesses
1. The paper claims to promote “diversity” in the core-set, but it is not fully clear which notion of diversity is being optimized. Is it diversity in learned functional structure, or in subject identity, or something else? Could this diversity sampling still cluster subjects along other confounders (for example age, site, motion level), in a way that systematically benefits certain SPIs? The authors should analyze whether the selected core-sets are imbalanced along basic covariates, and whether such imbalance correlates with SPI rankings.

2. This work treats the learned attention matrix $A(X)$ as if it reflects some functional relationship between brain regions and is presented as if it were neurobiologically meaningful, but this is not yet justified. Good performance on fingerprinting or diagnosis does not prove that $A(X)$ corresponds to a biologically valid network. in principle it could simply be a discriminative mathematical signature with no stable neurophysiological interpretation. The paper would be much stronger if it (a) explained the neuroscientific meaning the authors believe $A(X)$ captures, and (b) showed visualizations or qualitative analysis of what patterns are considered “stable” for different tasks (e.g., MDD vs. CN). 

3. SPS is defined from training dynamics, i.e., how stable a subject’s learned structure is across epochs. However, SPS may depend on optimizer choice, learning rate schedule, or even small architectural changes in the encoder. The paper should include these sensitivity analysis. Without this, it is hard to know whether SPS is an intrinsic property of the data/subject, or just an artifact of one specific training run.

4. All experiments are built around one clinical setting (MDD). This raises several questions:   
Would the method still work in a setting where each subject has multiple different scan types (e.g., resting-state fMRI and task fMRI)? If you try to pull all scans from the same subject together in the contrastive loss, does the approach still identify “stable structure,” or does it break because the functional state changes?   
What happens in datasets where each subject only has a single scan (no multiple time segments for contrastive learning)? Is the method still applicable, or does it fundamentally rely on repeated measures per subject?

### Questions
1. The authers mentioned that different papers can draw contradictory scientific conclusions from the same SPI. What makes you believe that you can build a unified SPI based on such agreement? 

2. In several settings, the “random” selection baseline performs surprisingly well, sometimes close to or even better than more carefully designed selection strategies. This is especially noticeable at certain sampling rates. How should we interpret this? Is random performing well because the dataset is already large and diverse, so almost any 30–50% subset is representative? Or is it because some SPIs are relatively insensitive to which subjects are chosen? Also, why do we sometimes see that an intermediate sampling rate (e.g., 30%) looks worse than 10% or 50%? Is there a theoretical or empirical explanation for these non-monotonic patterns?

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
3

### Summary
This paper proposes SCLCS, a core-set selection method for comparing various functional connectivity operators based on statistical pairwise interactions (SPIs) fairly and efficiently. By leveraging the convergence stability of attention matrices, structure perturbation score (SPS), SCLCS evaluates each sample’s structural representativeness and uses this to select the representative core-set for ranking different SPIs.

### Strengths
- Raises an unaddressed, important research question in the field of neuroimage analysis (efficient comparison between different SPIs)
- Theoretical rigor is well aligned with the research motivation.

### Weaknesses
- Ultimately, the work could provide further insight to the neuroscience community if SCLCS could be used to derive experimental suggestions on which SPIs are suggested to be representative for constructing the functional connectivity matrices. The experiments had a limited scope on validating that the efficiency and robustness of the core-set selection of SCLCS.
- Connectivity patterns constructed from different SPIs can be significantly susceptible to different pre/post-processing of the BOLD timeseries. Stability across different BOLD pre/post-processing could have been addressed.

### Questions
### Major
- Please provide robustness analysis results across different-sized windows for the sliding-window approach, and see if the method is robust across the window size.
- Please provide results on the effect of different pre-processing pipelines, at least the selection of different atlases, for the rebuttal.
- Please provide an interpretation of the ranking result across different SPIs. Would there be a recommended SPI for constructing the connectivity matrix? If so, why would that SPI be representative in terms of neuroscientific literature?


### Minor
- Notational precision can be further revised. (e.g., inconsistent use of $X$ and $\mathbf{X}$)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the prohibitive cost of benchmarking functional connectivity (FC) “signal processing indices” (SPIs) by reframing core-set selection as a **ranking-preservation** problem rather than single-model accuracy. It proposes **SCLCS**, a structure-aware, self-supervised framework that (i) encodes sample-wise FC structure with a modified Transformer, (ii) scores sample stability via a **Structural Perturbation Score (SPS)** over training epochs, and (iii) augments selection with **density-balanced sampling** to avoid top-k brittleness. On REST-meta-MDD (904 subjects; DMN 33 ROIs; 4,520 temporal segments), SCLCS preserves SPI rankings substantially better than nine strong baselines and maintains balanced subject/class coverage, while being orders-of-magnitude cheaper than full benchmarks.

### Strengths
- **Well-motivated objective and metricization.**  
  Focusing on cross-model *ranking* (not accuracy) matches the real need in SPI selection; nDCG@k is a sensible measure. 
- **Principled structure-aware selection.**  
  SPS is theoretically grounded (mixture-driven perturbation; stationarity/ergodicity) and operationally tied to the encoder’s attention dynamics; density-balanced sampling addresses the known failure mode of top-k. 
- **Comprehensive empirical study and practicality.**  
  The study spans two downstream tasks, nine baselines, coverage/fairness diagnostics, and compute analysis showing that a small one-time selection cost can replace ~990 CPU-days of exhaustive SPI evaluation.

### Weaknesses
1. **External validity limited by parcellation and preprocessing choices.**  
  Experiments are confined to 33 DMN ROIs (Dosenbach-160) with global signal regression; it remains unclear whether conclusions (e.g., SPS behavior, density effects, ranking stability) hold for whole-brain parcellations (e.g., Schaefer, AAL), alternative TRs, or non-GSR pipelines.
2. **Ground-truth ranking restricted to *fast* SPIs.**  
  To keep evaluation tractable, the “full-set” ranking is computed on a subset of SPIs (<1s/sample). This could bias findings toward operators with particular computational/statistical properties; evidence of transfer to slower/iterative SPIs is limited.
3. **Surrogate discriminability and stability assumptions.**  
  Ranking relies on a discriminability score derived from Spearman correlations; SPS assumes stationarity/ergodicity and measures attention perturbations. Although supported by analyses, sensitivity to the discriminability proxy and to encoder hyperparameters (heads, depth, temperature) is not fully characterized.
4. **Self-supervision via subject identity may conflate fingerprinting with diagnosis.**  
  Positive pairs are temporal segments from the same subject. While this is apt for fingerprinting, its alignment with clinical separability (MDD vs. HC) is indirect; the paper partially explores label influence, but a principled study of supervision choices (subject vs. site vs. phenotype) is missing. 
5. **Site effects and dataset shift under-analyzed.**  
  Coverage balance (subject/class) is reported, but explicit *site-aware* balance and ranking stability across sites (or leave-site-out coresets) are not provided; this is critical for multi-site rs-fMRI. 
6. **Theory–practice gap for universality claim.**  
  The “universal approximator for SPIs” is empirically checked on 16 operators; many SPIs (e.g., cointegration tests) are statistical procedures with thresholds/optimization loops. Clarification is needed on the operator class for which approximation guarantees are intended and how approximation error propagates to ranking.
7. **End-to-end wall-clock benefits not fully quantified.**  
  The paper convincingly contrasts selection cost vs. exhaustive benchmarking (~990 CPU-days), but does not report actual *end-to-end* benchmarking speedups (e.g., core-set size vs. hours on a realistic cluster), which would clarify practical impact.

## Minor
- Fig. 1 caption/text: "**Construstive** Learning" → "**Contrastive** Learning."

### Questions
1. **Parcellation & preprocessing robustness.**  
  How do core-set ranking nDCG@k and coverage metrics change under alternative parcellations (e.g., Schaefer-200) and without GSR? Please include sensitivity to window length/stride (70/35 TRs vs. others). 
2. **SPI subset bias.**  
  Can you report results where the full-set ranking includes a stratified sample of *slow* SPIs (e.g., iterative/graphical models) to test transfer beyond <1s/sample operators? Even a small but representative slow-SPI slice would help. 
3. **Supervision choice.**  
  Beyond subject identity, have you tried *site-ID* or *multi-task* (subject + site + MDD/HC) supervision when computing SPS? Does supervision alter SPS distributions and downstream ranking stability? 
4. **Site-aware analysis.**  
  Please provide site-conditioned ranking preservation (per-site nDCG@k) and site-aware coverage (selected samples per site) or a leave-site-out core-set test to assess generalization under dataset shift.
5. **Universality scope.**  
  What is the formal class of SPIs covered by the universality claim? For statistical-test SPIs (e.g., Johansen cointegration), how does approximation error affect ranking, and can you bound nDCG degradation as a function of operator approximation error? 
6. **Compute benefits in practice.**  
  Could you report wall-clock benchmarks (e.g., 10%/20% core-set vs. 100% full benchmark) on a standard 32–128 vCPU cluster to quantify realized speedups?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces an idea for performing core-set (small, representative subset of data that preserves some performance metric) selection for functional connectivity model benchmarking in neuroimaging. The core-set selection is based on statistical pairwise interaction type measures and preserving ranking among methods that use SPIs for modeling functional connectivity. They introduce a technique called SCLCS (Structure-aware Contrastive Learning for Core-set Selection)—a self-supervised Transformer framework that learns sample-specific FC structure representations, defines a score (structural perturbation score) to measure structural stability and use a density-based sampling technique to ensure the resulting core-set is both robust and diverse. 

The authors show experimental results on the REST-meta-MDD dataset, where their method achieves up to 23% higher ranking consistency than other coreset selection methods with a fraction (10%) of the data.

### Strengths
While the concept of core-set selection is not new, to my knowledge, this is the first work that is undertaking this problem for ranking functional connectivity performance. This is the main strength of the paper. 

The issue of core-set selection has been mostly ignored in neuroimaging, mainly because the datasets are generally small. However, as the number of datasets are growing in size, the number of algorithms are also getting contributed and tested, the problem of benchmarking is suddenly becoming important. Thus the overall concept that the paper proposes does make a contribution in the right direction

The overall theoretical framework of ranking preservation, the SCLCS, as well as the adapted transformer architecture is sound and is a novel contribution for this application.

### Weaknesses
The main weakness in the paper is that the ranking is based on Statistical Pairwise Interactions (SPIs). Although the authors test 130 SPIs, these are functional connectivity estimators and not end to end predictive machine learning models. SPIs are generally used in classical fMRI analyses and not widely used in modern ML based approaches including deep learning. 

In modern data-driven models (graph neural nets, attention-based architectures), the functional network connectivity structure is learned jointly with metric optimization, and thus make SPI type analysis unnecessary. 

Since the method is mainly based on SPIs,  it has limited adaptability to non-SPI workflows/benchmarks. The model ranking objective assumes each model produces a connectivity matrix via a pairwise operator. Further, the structure-aware feature extraction is built around this notion of symmetric, subject-level matrices. Finally the evaluation metric nDCG makes sense only if multiple SPIs are given as input for comparison. The benchmarking and core-set selection strategy therefore applies directly to workflows that explicitly compute connectivity features before classification. 
Thus the method may not generalize to deep learning models trained end-to-end on fMRI data, graph-neural network type approaches that infer connectivity structure and optimize it internally before prediction as well as other fMRI prediction algorithms, where the internal structure is not amenable for SPI comparison. 
The experimental evaluation is only done on a single dataset and only two downstream tasks to evaluate SPI discriminability: brain fingerprinting (distinguishing individuals based on subject ID), which probes for fine-grained, subject-specific structures, and MDD diagnosis, which relies on cohort-level patterns.

### Questions
Are SPIs still widely used for prediction in fMRI?

Theorem 1 assumes row-stochasticity of matrices, but can it be directly applied to the real valued symmetric matrices arising from SPIs? 

Is model ranking a valid goal?

Can the authors comment on the adaptability of the method to deep learning models?

### Soundness
3

### Presentation
3

### Contribution
3
