# Human Uncertainty-Aware Data Selection and Automatic Labeling in Visual Question Answering

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Large vision-language models (VLMs) achieve strong performance in Visual Question Answering but still rely heavily on supervised fine-tuning (SFT) with massive labeled datasets, which is costly due to human annotations.  Crucially, real-world datasets often exhibit *human uncertainty* (**HU**) — variation in human confidence across annotations, but standard SFT simply optimizes toward the most frequent label, disregarding HU distributions.  This leaves two open questions: *How does HU affect SFT*, and *how can HU be effectively leveraged in training?*  In this work, we first conduct a systematic evaluation of VLMs across varying HU levels. We have two key findings:  (i) surprisingly, high-HU samples contribute little, or even degrade, model performance, and  (ii) naively training on the full dataset yields under-calibrated models that fail to capture HU distributions.  Motivated by these findings, we introduce **HaDola**, a **h**uman uncertainty-**a**ware **d**ata selection and aut**o**matic **la**beling framework.  HaDola operates in four stages: **discriminate**, **self-annotate**, **error trigger**, and **training**, to iteratively identify harmful samples, prioritize informative ones, and bootstrap from a small seed set (5% of data).  Our approach substantially reduces reliance on costly HU annotations and makes VLMs more accurate and better calibrated.  Extensive experiments on VQAv2 and VizWiz datasets demonstrate that HaDola consistently matches or outperforms state-of-the-art baselines, with less training data.  Our work highlights the importance of explicitly modeling HU in SFT, suggesting better utilization of HU is more effective than merely scaling up dataset size.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies how human uncertainty affects VQA fine-tuning and finds that high-uncertainty data often hurts learning while low and medium uncertainty help both accuracy and calibration. It proposes HaDola, a four-stage pipeline that selects data with a KL-based HU window, self-annotates remaining samples, filters errors with gradient and influence triggers, and trains with an HU-aware objective.

### Strengths
1. The work clearly demonstrates that indiscriminate training on high-uncertainty samples degrades both accuracy and calibration. 

2. The HaDola pipeline reduces dependence on expensive annotations by iteratively expanding from a small seed set. 

3. Ablations and training-round analyses support the necessity of each component and show an S-shaped gain that plateaus near 15% labels.

### Weaknesses
1. The HU-acc metric may face adoption hurdles and its relationship to downstream utility is not deeply analyzed. 

2. The HU splitting scheme and threshold choices introduce design degrees of freedom that could affect reproducibility. 

3. Claims of low overhead are not backed by wall-clock or memory profiles for discrimination, triggers, and extra objectives. 

4. The method depends on a representative 5% seed set, and its sensitivity to seed variance is unclear. 

5. Validation is limited to two VQA datasets, so generality to knowledge-heavy or compositional QA remains uncertain.

### Questions
1. How robust are results to different random seeds for the initial 5% labeled subset. 

2. What is the measured training and inference overhead per round and per sample on standard hardware.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how human uncertainty (HU), namely the variation in annotator confidence across different answers, affects visual question answering model training. The authors observe that standard supervised fine-tuning optimizes only toward majority labels while ignoring confidence distributions, leading to miscalibrated models.

Key contributions: (1) First systematic study of HU's impact on VLM training; (2) HaDola framework achieving competitive results with only 5-20% labeled data; (3) HU-acc metric incorporating confidence into evaluation; (4) Theoretical analysis with bounded-bias guarantees.

### Strengths
As the first systematic study of human uncertainty's role in VLM training, this work addresses an important gap in VQA research. Its strengths include convincing empirical findings (high-HU samples harm performance), impressive data efficiency (5% seed data matching 100% baselines), and comprehensive validation across multiple models and datasets with theoretical grounding.

### Weaknesses
1. The paper assumes confidence labels are reliable across annotators without validation. Different annotators may interpret "maybe" systematically differently, yet no inter-annotator agreement is analyzed. Since all thresholds (τ₁, τ₂...) depend on the 5% seed set, miscalibrated annotators in the seed would compromise the entire framework.

2. The mapping yes/maybe/no → 0.99/0.5/0.01 is borrowed from prior work without justification.  These values directly affect HUD distributions and split boundaries, yet zero sensitivity analysis is provided. Testing alternative mappings (e.g., 0.95/0.5/0.05) is necessary to show results aren't artifacts of arbitrary choices.

### Questions
See Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a new method to leverage a small pool of human-labeled VQA data along with annotators' confidence scores to bootstrap a larger pool of self-annotated VQA data. This approach outperforms naive SFT at a fraction of the data scale. The authors experiment with their HuDola method on both VQAv2 and VizWiz datasets, demonstrating that HuDola reliably improves performance across multiple model families.

### Strengths
Overall, the paper is well-written and the experiments are well-designed. The authors provide empirical evidence that high human uncertainty is detrimental to SFT performance. The experimental scope is satisfactory, covering four different VLM base models with consistent improvements across all models (with the exception of BEiT-3). The authors provide a thorough analysis of the HuDola results. The study on training dynamics shows that performance improves smoothly with additional HuDola-generated samples, and the ablations presented in Table 1 demonstrate that each component of HuDola's pipeline is necessary for achieving the improvements.

### Weaknesses
My main concern is that the authors study this problem only in the context of fine-tuning and evaluating the VLM on data from the same distribution and on relatively saturated benchmarks. The findings would be more impactful if the authors could demonstrate that HuDola improves VLMs on out-of-distribution data and on more challenging benchmarks. To assess the out-of-distribution performance of HuDola, one specific experiment would be to initialize the seed data pool from a single source (e.g., VQAv2) and use it to bootstrap labels for a different source (e.g., VizWiz).

Another limitation is that the authors conduct only a single run for each training method and model family. Since the difference in performance between HuDola and SFT is less than 0.5 percentage points in most cases, it is unclear whether the improvements are statistically significant. There is likely some run-to-run variability in the results, and reporting the mean and standard deviation would help verify whether the improvements are statistically significant.

### Questions
1) How does the HuDola method generalize to out-of-distribution data from the source seed data pool?
2) Are the results of the training runs statistically significant? The authors conduct only a single run for each training method and model family, and the difference in performance between HuDola and SFT is very small for some experiments.

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
3

### Summary
The paper introduces HaDola, a framework that improves the training of vision-language models (VLMs) by explicitly accounting for human uncertainty (HU) in annotations. The authors find that samples with high HU—where annotators disagree or are uncertain—often degrade model performance and calibration. HaDola mitigates this by iteratively identifying and excluding harmful samples, generating pseudo-labels for reliable ones, and applying an error-trigger mechanism to prevent noise accumulation during self-training. Using only a small seed set (around 5%) of HU-labeled data, HaDola achieves comparable or superior accuracy and calibration to state-of-the-art baselines on VQAv2 and VizWiz datasets. The work highlights that effectively modeling human uncertainty is more beneficial than simply scaling the dataset size.

### Strengths
The authors propose an interesting approach to leverage human uncertainty for both model training and evaluation.

### Weaknesses
- The compared baselines seem weak. The target task appears to be a semi-supervised VQA task. Then, it would be beneficial to compare with other pseudo-labeling-based semi-supervised learning works, such as FixMatch.
- In Figure 4, the performance curves with different VLMs are not enough. In a data-efficient VQA work, the authors should have compared against baselines, as this curve could be the main experiment of the paper. 
- Also, as the initial annotation can critically change the model performance, the authors should have considered different seed data settings in the experiment.
- The novelty of the proposed HaDola seems weak. The proposed HaDola framework is a combination of existing ideas (data filtering, pseudo-labeling, uncertainty calibration) with minor adaptation for human uncertainty. Simply removing uncertain samples is too naïve and risks losing informative and diverse examples. It would be better to include more analysis of the justification for the design choices.
- In addition, there are various design choices in the method without empirical justification. For example, while the authors select the samples in the region [tau1-sigma,tau2+sigma], it would be better to empirically justify the design choice by sampling different samples.
- The evaluation is only done with the HU accuracy. However, it would also be useful to evaluate with the conventional VQA accuracy. Comparing the trends between VQA accuracy and HU accuracy might provide informative insights.
- For Figure 3, it might be better to use a table instead of a heatmap for better visualization. In the heatmap, the baselines' performances look similar. However, when we look at the individual values, there are several settings where the baselines perform better than the proposed method.

### Questions
Please refer to the questions in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
