# SCD-MMPSR: Semi-Supervised Cross-Domain Learning Framework for Multitask Multimodal Psychological States Recognition

- Avg Score: 4.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6, 4, 6

## Abstract
Modern human-computer interaction interfaces demand robust recognition of complex psychological states in real-world, unconstrained settings. However, existing multimodal corpora are typically limited to single tasks with narrow annotation scopes, hindering the development of general-purpose models capable of multitask learning and cross-domain adaptation. To address this, we introduce SCD-MMPSR (Semi-supervised Cross-Domain Multitask Multimodal Psychological States Recognition), a novel framework that unifies heterogeneous corpora via GradNorm-based adaptive task weighting in multitask semi-supervised learning (SSL) to jointly train models across diverse psychological prediction tasks. At the architectural core, we propose two innovations within a graph-attention backbone: (1) Task-Specific Projectors, which transform shared multimodal representations into task-conditioned logits and re-embed them into a unified hidden space, enabling iterative refinement through graph message passing while preserving modality alignment; and (2) a Guide Bank, a learnable set of task-specific semantic prototypes that anchor predictions, injecting structured priors to stabilize training and enhance generalization. We evaluate SCD-MMPSR on three distinct psychological state recognition tasks, emotion recognition (MOSEI), personality trait recognition (FIv2), and ambivalence/hesitancy recognition (BAH), demonstrating consistent improvements in multitask performance and cross-domain robustness over strong baselines. We also evaluate the generalization of SCD-MMPSR on unseen data using MELD. Multitask SSL improves generalization on MELD by macro F1-score of 7.5% (35.0 vs. 27.5) over single-task SSL. Our results highlight the potential of semi-supervised, cross-task representation learning for scalable affective computing. The code is available at https://github.com/Anonymous-user-2026/ICLR_2026.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on psychological state recognition and proposes a joint framework, SCDMMPSR, for diverse psychological prediction tasks. The framework is characterized by semi-supervised learning, cross-domain adaptation, multi-task learning, and multimodal fusion, primarily consisting of two key components: task-specific projectors and a guide bank.

### Strengths
1.	A unified framework for psychological state recognition, capable of handling multiple prediction tasks simultaneously.

2.	Performance improvements across four psychological prediction tasks.

3.	Extensive experimental evaluation.

### Weaknesses
1.	The paper appears incremental, as it combines well-established techniques—semi-supervised learning, cross-domain adaptation, multi-task learning, and multimodal fusion. From my perspective, the main contribution is integrating these existing methods into a single framework for psychological state recognition, rather than proposing novel mechanisms.

2.	The proposed framework, as illustrated in Figure 1, seems to incrementally combine different features for multi-task learning across three tasks, without demonstrating a clearly innovative architecture.

3.	Although the framework employs complex architectures, Table 1 shows that the individual modules contribute only limited performance improvements, raising questions about their necessity and effectiveness.

4.	As shown in Table 2, the proposed method sometimes achieves only marginal gains—or even performance degradation—compared to existing state-of-the-art methods.

5.	The paper does not discuss computational efficiency or resource requirements, which is crucial for real-world applicability.

6.	Given the limited performance advantages, the paper should include statistical significance tests to ensure that observed improvements are not due to random variance.

### Questions
See weakness

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
This paper proposes SCD-MMPSR (Semi-supervised Cross-Domain Multimodal Pretraining with Self-Reflective Regularization), 
a framework designed to address the challenges of domain shift and limited labeled data in multimodal emotion recognition. 
The method integrates three main components: (1) a cross-domain multimodal pretraining strategy that learns a shared semantic space across domains, 
(2) a semi-supervised pseudo-labeling mechanism that leverages unlabeled samples through confidence-based consistency learning, 
and (3) a self-reflective regularization term that encourages the model to calibrate its own uncertainty and enhance robustness.

### Strengths
1. Well-structured and conceptually coherent.
    The proposed framework is logically organized into three complementary components: domain alignment, pseudo-label learning, and self-reflective regularization.  The overall design is intuitive and theoretically grounded in representation learning principles.

2. Strong empirical results.  
    The model demonstrates consistent improvements across multiple datasets and missing-data scenarios, suggesting good generalization.  The ablation analysis further highlights the individual contribution of each component, particularly the self-reflective regularization term.

### Weaknesses
1. Unclear pseudo-label generation and weighting.
    The description of how pseudo-labels $\tilde{y}$ are generated and weighted by confidence lacks detail.  It is not explicit whether a teacher model, EMA updates, or thresholding strategy is used, nor how the confidence scores affect optimization.

2. Notation and presentation issues.
    Ref to questions.

3. Limited theoretical insight.
    While the empirical results are convincing, the theoretical interpretation of the self-reflective regularization is somewhat heuristic.  A more formal analysis of its learning dynamics would strengthen the contribution.

### Questions
1. Model abbreviation inconsistency.
    The abbreviation SCD-MMPSR is written inconsistently across the manuscript.  Please unify the model name throughout the paper for clarity and consistency.  


2. Undefined measure of domain discrepancy.
    In the description of the Domain Alignment Transformer, the paper claims that it “aligns the source and target domains into a shared semantic space,” but does not specify the metric used to quantify inter-domain distance.  Please clarify whether $\mathcal{L}_{align}$ is based on MMD, KL divergence, cosine distance, or another measure.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
SCD-MMPSR introduces a unified semi-supervised framework for recognizing multiple psychological states from multimodal time series data, using heterogeneous single-task corpora. The model features adaptive task weighting and novel fusion layers (Task-Specific Projectors and Guide Banks) within a graph-attention backbone to enable robust multitask and cross-domain learning, even without jointly labeled datasets. Experiments across several benchmark corpora demonstrate gains in multitask performance and generalization, highlighting the benefits of joint semi-supervised learning

### Strengths
- The framework supports joint multitask learning without the need for jointly labeled data.
- Improved performance in cross-domain settings.
- Extensive ablation studies demonstrate the impact of different model components.

### Weaknesses
- The mathematical symbols in Figure 2 and Section 2 are inconsistent, making the formulation difficult to understand.
- The explanation of the guided bank is limited. How are task-specific embeddings extracted from it?
- How does the author derive the $r_t$ formula on page 5? Why is $L_t^{(0)}$ used for normalization?
- The method depends on many hyperparameters.

### Questions
- The mathematical symbols in Figure 2 and Section 2 are inconsistent, making the formulation difficult to understand.
- The explanation of the guided bank is limited. How are task-specific embeddings extracted from it?
- How does the author derive the $r_t$ formula on page 5? Why is $L_t^{(0)}$ used for normalization?
- For the current formulation, Mixture of experts (MoE) are also a good alternative. It would be good if authors comment on that.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents SCD-MMPSR, which extends existing multitask learning approaches to jointly learn emotion recognition, personality trait recognition, and ambivalence/hesitancy recognition from three independent single-task datasets. Building on established techniques including GradNorm-based adaptive task weighting, graph attention fusion, and standard pseudo-labeling strategies. The framework contributes (1) an engineering refinement that re-embeds task predictions for iterative message passing; (2) a dual-branch extension of GradNorm separating supervised and semi-supervised optimization to handle missing labels; and (3) empirical validation showing that multitask semi-supervised learning improves zero-shot generalization. While the specific three-task combination and architectural configuration are novel, the core methodological components leverage well-established techniques from the multitask learning and semi-supervised affective computing literature.

### Strengths
The paper addresses a well-motivated practical problem of unifying heterogeneous single-task corpora for multitask psychological state recognition without joint annotations. The experimental work is thorough, featuring comprehensive ablation studies (17 experiments) that systematically validate each architectural component, with strong empirical results on personality traits (FIv2: 92.6% mACC) and ambivalence recognition (BAH: 73.2% WF1). The framework effectively handles heterogeneous data through adaptive task weighting and semi-supervised pseudo-labeling, demonstrating cross-domain benefits with 7.5% zero-shot improvement on MELD compared to single-task learning.

### Weaknesses
The core techniques mentioned here are well-established, and the contribution is primarily an engineering combination of existing methods rather than fundamental algorithmic innovation. The paper primarily compares against single-task emotion recognition models rather than other multitask frameworks like MER 2025 or MuMTAffect. A comparison with these concurrent multitask approaches would help validate whether the proposed architecture components offer unique advantages. The paper would benefit from testing how the system performs without Qwen2.5-VL (using simpler video encoders) to clarify whether the strong results stem from the proposed architectural innovations or primarily from the pre-trained model selection.

### Questions
1. How does the proposed framework compare to other recent multitask emotion-personality approaches? Comparisons with concurrent work could strengthen the paper (if possible).
2. Could you test the framework with simpler video encoders (instead of Qwen2.5-VL) to show that the proposed components work well independently? This would help readers understand the specific value of your design choices.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on a key limitation in multimodal affective computing, the lack of generalizable models capable of jointly learning across heterogeneous datasets and tasks. To overcome this, the authors propose SCD-MMPSR, a semi-supervised, cross-domain, multitask learning framework for multimodal psychological state recognition. The approach builds on a graph-attention backbone with two proposed components: (1) Task-Specific Projectors (TSPs), which map shared multimodal embeddings into task-conditioned logits and re-embed them into a unified latent space for iterative graph message passing; and (2) a Guide Bank, a learnable set of task-specific semantic prototypes that act as structured anchors to stabilize multitask learning and improve cross-domain generalization. The method is evaluated on three benchmark datasets, MOSEI for emotion recognition, FIv2 for personality trait inference, and BAH for ambivalence/hesitancy detection, and tested for out-of-domain generalization on MELD, demonstrating consistent and substantial performance gains.

### Strengths
1. Addresses the increasingly relevant challenge of cross-domain and multitask generalization for psychological and affective state recognition, which has strong implications for real-world HCI and social signal processing.
2. The integration of GradNorm-based adaptive task balancing with semi-supervised graph attention is a creative step forward in harmonizing heterogeneous multimodal corpora.
3. The Task-Specific Projector (TSP) design provides a principled way to handle both shared and task-specific representational subspaces.
4. The Guide Bank is conceptually interesting and intuitively appealing, using semantic prototypes as regularizers to enforce stable and interpretable multitask representations.
5. The writing is generally clear and logically structured, with good justification for each design choice.

### Weaknesses
1. While the Guide Bank is conceptually intuitive, the paper would benefit from more theoretical analysis or visualization showing how these prototypes evolve and influence latent space geometry. It would also be useful to contrast the proposed methods, specifically the use of the Guide Bank, against knowledge-infusion methods proposed in the literature. 
2. The paper mentions “unifying heterogeneous corpora” but does not fully clarify how label space inconsistencies (e.g., emotion vs. personality) are reconciled beyond shared latent embedding alignment.
3. The generalizability is a difficult hypothesis to prove with empirical analysis. While the domain-specific focus makes sense and the experiments look reasonable, the broader generalizability aspect on any modality and any task is hard to justify.
5. While MELD is a good unseen test set, inclusion of more diverse or real-world datasets (e.g., MAHNOB-HCI, IEMOCAP) would better demonstrate scalability

### Questions
1. How are task label conflicts (e.g., continuous vs. categorical outputs) handled during multitask training?
2. Does the Guide Bank act as a static memory or is it dynamically updated during training? How is this different from knowledge-infusion techniques.?
3. How sensitive is the method to imbalanced data or differing annotation densities across tasks?
4. Could the architecture support few-shot adaptation to entirely new psychological tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SCD-MMPSR, a framework for joint recognition of multiple psychological states (emotions, personality traits, and ambivalence/hesitancy) from multimodal data. The key innovation lies in training across heterogeneous, single-task corpora using semi-supervised learning with adaptive task weighting via an extended GradNorm method. The architecture features Task-Specific Projectors for iterative refinement and Guide Banks as learnable semantic prototypes. Experiments on MOSEI, FIv2, and BAH datasets, with generalization testing on MELD, demonstrate improved cross-domain performance, particularly a 7.5% macro F1-score improvement on MELD compared to single-task SSL.

### Strengths
1. The fragmentation of psychological state recognition across single-task, single-corpus methods represents a real barrier to deploying general-purpose affective AI systems, and the solution of leveraging heterogeneous corpora without requiring joint annotations is economically sensible and practically valuable.
2. The dual-branch GradNorm extension with delayed initialization, Task-Specific Projectors enabling iterative refinement, and the overall integration of semi-supervised cross-domain multitask learning represents a first-of-its-kind contribution to psychological states recognition.
3. Systematic evaluation of 20+ encoders across four modalities, comprehensive ablation studies with Friedman ranking, bootstrap confidence intervals, and extensive hyperparameter optimization demonstrate methodological maturity and enable fair comparison.
4. The 7.5% macro F1 improvement on MELD (35.0 vs. 27.5) convincingly demonstrates the framework's core value proposition, with error analysis showing meaningful reductions in majority class bias and more semantically coherent confusion patterns.

### Weaknesses
1. While the empirical results are strong, the paper would benefit from explaining why this specific combination of graph attention, task projectors, and guide banks is theoretically motivated. The Guide Bank formulation (Equation 13) particularly seems somewhat ad-hoc-why apply sigmoid transformation only to personality traits?
2. The real-time factor of 1.11s per second (0.69s from Qwen2.5-VL alone) limits practical deployment. While the authors suggest omitting behavior modality, this somewhat undermines the multimodal argument. It would be valuable to see training times, memory requirements, and computational comparisons with baseline methods.
3. MELD is a useful unseen test for ER, but it only assesses emotion transfer. For stronger “cross-domain” claims, evaluate transfer for PTR and AHR to different corpora (if available) or simulate domain shift (recording conditions, speaker distribution) and report robustness.
4. Reliance on Qwen2.5-VL for behavior descriptions dominates inference (0.69s per second) - that’s a hard barrier for real-time applications. The authors acknowledge this but don’t provide an evaluated lightweight alternative. Add experiments with a smaller VLM (distilled model) or ablate behavior modality thoroughly to quantify pragmatic tradeoffs.
5. The related work section would be stronger with citations to foundational semi-supervised methods such as FixMatch, MixMatch, and Mean Teacher; influential studies in cross-domain affective modeling like Zadeh et al. (2018) and Li et al. (2021); and key multimodal fusion research including Tsai et al. (2019) and Hazarika et al. (2020). In addition, incorporating recent advances such as LensLLM, which links large language models with affective perception, would provide a more complete and up-to-date context for the paper’s contributions.

### Questions
Please see the details in the weakness for the respective questions.
1. Can the authors clarify the theoretical motivation behind combining graph attention, task-specific projectors, and guide banks? Specifically, why does the Guide Bank formulation (Equation 13) apply a sigmoid transformation only to personality traits, and is there a principled justification for that choice?
2. The reported real-time factor of 1.11s per second (with 0.69s from Qwen2.5-VL alone) poses challenges for deployment. Could the authors provide details on training time, memory footprint, and computational cost compared to baseline methods, and discuss practical tradeoffs?
3. Since MELD only tests emotion transfer, how do the authors plan to support broader cross-domain generalization claims? Have they considered evaluating PTR and AHR transfer to new corpora or simulating domain shifts (e.g., recording conditions, speaker demographics) to test robustness?
4. Given the heavy reliance on Qwen2.5-VL for behavioral descriptors, could the authors test a lighter or distilled vision-language model and report how performance and efficiency change? Alternatively, could they provide a detailed ablation of the behavior modality’s contribution?
5. May consider strengthening the related work section by adding the references.

### Soundness
3

### Presentation
2

### Contribution
3
