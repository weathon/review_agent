# RECTOR: Masked Region-Channel-Temporal Modeling for Cognitive Representation Learning

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Affective and cognitive disorders are characterized by complex, distributed brain network dynamics across distinct functional regions, channels, and time, posing a significant challenge to learning robust representations for clinical diagnosis. We introduce $\textbf{RECTOR}$ (Masked $\textbf{Re}$gion–$\textbf{C}$hannel–$\textbf{T}$emp$\textbf{or}$al Modeling), the first end-to-end, self-supervised brain region modeling framework that unifies region, channel, and temporal representation learning in a single architecture. At its core is $\textbf{RECTOR-SA}$, a novel hierarchical self-attention mechanism. It efficiently models region-channel-temporal interactions in a block-wise paradigm, incorporating both anatomical priors and dynamic functional gating. It further integrates $\textbf{RECTOR-Mask}$, a novel masking strategy that generates diverse region-channel-temporal
views to establish a challenging pretext task. The self-supervision is driven by $\textbf{NC}^\textbf{2}$$\textbf{-MM}$, our learning objective. It synergistically encourages the encoder to learn both predictive and consistent representations across different views. Finally, we introduce $\textbf{RCReg}$, a tailored regularization on region–channel tokens. It prevents trivial region features and enables the model to explicitly learn and disentangle both region-common and channel-specific representations. Across diverse benchmarks, RECTOR sets a new state-of-the-art in EEG emotion recognition and sEEG task-engagement classification. In addition, it achieves superior computational efficiency in spatio-temporal self-attention, demonstrates strong potential for large-scale pre-training, and provides interpretable, multi-view insights into neural representations at both brain region and channel levels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RECTOR, a self-supervised framework for EEG and sEEG cognitive representation learning that explicitly models region–channel–temporal interactions. The key contributions include: (1) RECTOR-SA, a hierarchical sparse attention mechanism incorporating anatomical priors and dynamic gating; (2) RECTOR-Mask, a structured multi-view masking strategy that creates region- and time-aware masked modeling targets; (3) NC²-MM, a unified learning objective that combines masked modeling and contrastive learning within one architecture; and (4) RCReg, a specialized regularization for improving region–channel token representations. The model achieves state-of-the-art results across EEG emotion recognition and sEEG task-engagement classification benchmarks, with supporting ablations and interpretability analyses.

### Strengths
1. Ambitious attempt to integrate spatial priors and self-supervision in neural signal modeling.  

2. Structured masking and hierarchical attention are intuitively motivated.

3. Experimental results are comprehensive, covering multiple datasets, protocols, and baselines, including ablations that validate each core component of the architecture.  

4. The method provides neuroscientifically interpretable results at both region and channel levels, demonstrating alignment with known physiological patterns.

### Weaknesses
1.The figures in the manuscript need improvement, especially Figures 2 and 3. Figure 4 is significantly clearer in comparison.

2.The method’s novelty appears incremental rather than fundamental. Most components (structured masking, region tokens, gated attention, variance/covariance regularization) are adaptations of well-known techniques with domain-specific adjustments rather than a distinctly new contribution.

3.The writing is dense, and the paper tends to overstate its contributions relative to the demonstrated novelty.

4.The anatomical prior design is under-justified. Region partitioning is treated as fixed and universally correct, but inter-subject anatomical variability is substantial in EEG/sEEG. The paper does not assess the robustness or validity of this assumption.

5.Pretraining only on each target dataset weakens the claim of general-purpose self-supervised learning.

6.Critical methodological details are placed in the appendices and should be included in the main paper to ensure clarity and reproducibility.

7.Considering the complexity of the proposed method, the absence of released code makes reproducibility difficult.

### Questions
1.Could Figures 2 and 3 be redesigned with improved layout and clearer color schemes to enhance readability?

2.What is the key novel contribution beyond combining existing components such as structured masking and hierarchical attention?

3.How do you justify the strength of your claims relative to the demonstrated novelty?

4.How robust is the anatomical prior (fixed region partitioning) to inter-subject variability in EEG/sEEG?

5.How does pretraining only on each target dataset support claims of general SSL generalization?

6.Can you move critical methodological details from the appendix into the main text to improve clarity and reproducibility?

7.How do you conduct the leave-one-subject-out (LOSO) evaluation? Is there a hold-out validation set used to determine the number of training epochs and hyperparameters?

8.Will the code and pretrained models be released to ensure reproducibility given the complexity of the method?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper *RECTOR: Masked Region–Channel–Temporal Modeling for Cognitive Representation Learning* proposes RECTOR, a self-supervised learning framework for EEG and sEEG representation learning that jointly models region-, channel-, and temporal-level dependencies. Its core contributions include a novel **RECTOR-SA** hierarchical self-attention mechanism integrating anatomical priors for efficient region-channel-temporal modeling, a **RECTOR-Mask** structured multi-view masking strategy for more challenging pretext tasks, and **NC2-MM**, a combined non-contrastive × contrastive learning objective. Additionally, **RCReg** regularizes region-channel tokens to enhance feature disentanglement. The model achieves state-of-the-art results on EEG emotion recognition and sEEG task-engagement classification while claiming higher computational efficiency and interpretability compared to prior works. However, despite strong empirical results, the paper largely repackages existing ideas—masked modeling, contrastive loss fusion, and anatomical priors—into a composite architecture. The novelty is incremental and primarily architectural, lacking theoretical rigor or strong neuroscientific grounding. The extensive ablations and comparisons suggest solid engineering, but the work leans more toward technical aggregation than conceptual innovation.

### Strengths
1. Comprehensive ablations: Evaluates the effect of masking ratios, loss weights, and feature hierarchies.

2. Multi-dataset evaluation: Demonstrates generalization across EEG (emotion) and sEEG (task) domains.

3. Integrated pipeline: Combines anatomical priors with deep self-supervised frameworks, improving biological plausibility relative to generic transformers.

4. Engineering soundness: Implementation is well-optimized and includes clear reproducibility details and comparison tables.

### Weaknesses
1. Limited novelty: The key ideas—masked modeling, hierarchical attention, hybrid contrastive objectives—are incremental reuses of prior designs (MAE, BYOL, DINO, MoCo-v3, etc.) rather than a fundamentally new direction.

2. Weak theoretical grounding: No analysis of why combining non-contrastive and contrastive terms yields better cognitive representations.

3. Poor interpretability: Despite claiming cognitive alignment, there is little neuroscientific analysis (e.g., brain-region relevance or neurobiological validation).

4. Superficial discussion: Results are over-interpreted as “state-of-the-art” without effect size reporting or significance testing.

5. Unclear scalability: It is uncertain whether RECTOR can scale to large, multi-site EEG datasets or handle real-world noise.

6. Dataset limitations: The training and evaluation datasets are small (dozens to hundreds of subjects), which limits generalization claims.

7. Overclaiming contributions: The claim of being “the first unified region–channel–temporal framework” ignores earlier hierarchical EEG models (e.g., EEG-GraphMAE, ST-MAE, Brain-MAE).

### Questions
1. How does RECTOR-SA differ fundamentally from existing spatio-temporal attention modules used in EEG-GraphMAE or ST-MAE?

2. What motivates the NC²-MM hybrid loss—can you show a theoretical analysis of how it prevents representation collapse?

3. How are anatomical priors encoded? Are these static adjacency matrices or learned embeddings, and how sensitive is performance to parcellation choice?

4. Can you report per-subject and per-session variance or confidence intervals for downstream metrics?

5. How does RECTOR perform under noisy or low-density EEG setups—does the hierarchical attention degrade gracefully?

6. Have you compared against self-distillation approaches (e.g., DINO-style EEG pretraining) to isolate the benefit of your masking strategy?

7. How interpretable are the learned representations—do any attention maps align with known cortical functional networks?

### Soundness
2

### Presentation
2

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
This papers introduces a complete and exhaustive method for self-supervised deep learning framework for EEG and sEEG data, accounting the various aspects and dynamics of these brain activity modalities (region, channels, temporal). It introduces multiple modules, in particular RECTOR-SA and RECTOR-Mask, for feature extractions at different scales and combines masked modelling, contrastive learning, and variance–covariance regularisation for training objectives. The methodology is benchmarked against multiple models and training schemes (supervise-only, self-supervised) and outperforms many models on two main tasks: EEG emotion recognition and sEEG cognitive states.

### Strengths
The paper is generally well-written. The figures are complete, very descriptive (even though some of the legends could benefit from thorough descriptions, see below for more details). 

The methodology, although quite exhaustive, is addressing one of the blindspot of many EEG (and even in some sense fMRI) studies, which is taking into account the spatial (and regional) and temporal dynamics in EEG signal. In particular accounting for regional specific features rather than aggregating spatial information is a nice contribution. 

The evaluation framework is comprehensive and well thought with comparison against many training frameworks (supervised, self-supervised models) and multiple datasets. The ablation studies are also welcome considering the many modules that are introduced by the paper. 

 The colour coding in the tables makes the results very easily readable.

### Weaknesses
One important issue with the current state of the submission is the general arrangement of information within the paper. 
1. It is (very) difficult to understand at first what is the training objective of the model (what is going to be predicted), what the model aims at solving and how it aims at doing it. All this could be clearer from the beginning of the 2. Methodology section. 
2. The paper tends to over-complexified some of the notations (Figure 2) and wordy terminology e.g. "sparse region-channel-temporal self-attention embedded with anatomical priors and dynamic functional attention", which can obscur a bit the interesting concepts introduced by the method.
3. Most importantly, it seems that most of the main modules are not fully described within the main text of the paper, but are detailed in the appendix. Many points in the methods are referring to the appendix, which make it impossible to understand from the 

Another remark would be that the paper is extremely dense, some might say too dense, at a point were it is difficult to apprehend the entirety of the method with only the main text of the submission. This kind of paper would probably benefit from simpler iterations, to appreciate the real value of every added module. This is also considering that some concepts are only introduced in the appendix. This amount of content can be seen as detrimental for the overall appreciation of the paper. Therefore, I would recommend to streamline the paper and remove the unnecessary content for that publication. 

In particular, I would recommend switching some of the paragraphs and reorganising/rewriting the methodology section in order to have full descriptions of the RECTOR modules (in particular RECTOR-SA) in the text (not in the appendix) - section 2.2 is too high level and figure 2 is not explained enough to be stand-alone, clear description of the pipeline (it is difficult to understand how the modules interact with each other), explanation of concepts such as the brain partitioning (which is in Appendix E) but is one of the key point of the paper. Instead the "Complexity" paragraph could be added to the appendix. In the current shape, the methodology is difficult to understand"

Figure 3 would also benefit from more descriptive legend.

### Questions
- It is not clear from the main text how RECTOR is fine-tuned on the downstream task?

Other remarks were listed in the previous section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces RECTOR, a self-supervised framework for EEG/sEEG data that integrates region, channel, and temporal representation learning through a novel hierarchical self-attention mechanism (RECTOR-SA). The model incorporates anatomical priors and functional attention to capture complex spatio-temporal interactions in neural data. Evaluated across several EEG datasets (SEED, SEED-IV, DEAP, MSIT, ECR), RECTOR demonstrates state-of-the-art performance in emotion recognition and task engagement classification. The paper claims that the model not only improves computational efficiency but also provides strong interpretability through attention visualizations, paving the way for its application in neurocognitive diagnostics and personalized interventions.

### Strengths
The proposed RECTOR framework introduces a novel approach by combining self-supervised learning with anatomical priors and dynamic functional attention for EEG/sEEG data, a method previously explored in fMRI but applied to EEG data for the first time. 
The manuscript is well-written and clearly structured. The experiments are thorough and effectively address the research question, providing solid evidence for the physiological plausibility of the learned representations. The results are presented clearly, with supporting analyses that validate the model's performance.

### Weaknesses
The paper's evaluation is limited to the SEED and DEAP datasets, and it would benefit from validation on a broader range of downstream tasks to better assess RECTOR's generalizability. The use of only F1-score as the evaluation metric is restrictive; incorporating other standard metrics such as Cohen’s Kappa, weighted F1, and additional classification metrics would provide a more comprehensive performance analysis. The ablation studies, while useful, lack a detailed examination of the self-attention mechanism, and the gating mechanism within RECTOR-SA is not addressed. Furthermore, the paper provides visualizations of learned representations but lacks a deeper interpretability analysis, especially in terms of attention maps, spatial EEG features, and feature attribution. Lastly, the pretraining details, including hyperparameters,  pretraining dataset, training time, are unclear, which raises concerns about reproducibility and scalability.

### Questions
1. Downstream Evaluation: Given the model’s promising performance on SEED and DEAP, can RECTOR be extended to other EEG/sEEG tasks with different cognitive states or sensor modalities? Evaluating RECTOR on a wider variety of tasks would clarify how well the model generalizes across different applications, such as emotion recognition, cognitive task engagement, or even clinical diagnostics for neurological disorders.
2. Evaluation Metrics: The paper primarily uses F1-score, which is valuable but limited. Would the authors consider evaluating RECTOR using other metrics like Cohen’s Kappa (which accounts for agreement between class predictions) and weighted F1 (to account for class imbalance)? Comparing RECTOR with task-specific models rather than foundation models (such as those designed specifically for EEG emotion recognition) would provide more meaningful insights into its performance.
3. Ablation Study on Attention and Gating: While ablation studies are provided, could the authors conduct more detailed experiments focusing specifically on the RECTOR-SA attention mechanism? How does each component (e.g., region-based vs. global attention) contribute to the model’s overall performance? Additionally, the gating mechanism within RECTOR-SA is mentioned but not ablated—what role does it play in the model, and how does it impact performance across tasks?
4. Interpretability and Visualization: The paper includes some visualizations of learned representations, but a deeper analysis of the model’s internal behavior is lacking. Could the authors include attention maps, coherence heatmaps, or feature attribution to show how RECTOR attends to relevant spatial and temporal patterns in EEG? This would help validate whether the spatial awareness captured by the model is truly driving the observed performance improvements.
5. Pretraining Process: The paper does not provide sufficient details on the pretraining process, such as hyperparameters, optimization strategy, or batch size. Understanding these details would help in replicating the results and assessing the model’s scalability to other datasets. Could the authors clarify the pretraining procedure and explain how these choices impact the model’s final performance?
6. Usage of LLM missing.
7. No code revealed

### Soundness
3

### Presentation
3

### Contribution
3
