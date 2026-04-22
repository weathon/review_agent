# A Brain Graph Foundation Model: Pre-Training and Prompt-Tuning across Broad Atlases and Disorders

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
As large language models (LLMs) continue to revolutionize AI research, there is a growing interest in building large-scale brain foundation models to advance neuroscience. While most existing brain foundation models are pre-trained on time-series signals or connectome features, we propose a novel graph-based pre-training paradigm for constructing a brain graph foundation model. In this paper, we introduce the Brain Graph Foundation Model, termed BrainGFM, a unified framework that leverages graph contrastive learning and graph masked autoencoders for large-scale fMRI-based pre-training. BrainGFM is pre-trained on a diverse mixture of brain atlases with varying parcellations, significantly expanding the pre-training corpus and enhancing the model’s ability to generalize across heterogeneous fMRI-derived brain representations. To support efficient and versatile downstream transfer, we integrate both graph prompts and language prompts into the model design, enabling BrainGFM to flexibly adapt to a wide range of atlases, neurological and psychiatric disorders, and task settings. Furthermore, we employ meta-learning to optimize the graph prompts, facilitating strong generalization to previously unseen disorders under both few-shot and zero-shot learning conditions via language-guided prompting. BrainGFM is established on 27 neuroimaging datasets spanning 25 common neurological and psychiatric disorders, encompassing 2 types of brain atlases (functional and anatomical) across 8 widely used parcellations, and covering over 25,000 subjects, 60,000 fMRI scans, and a total of 400,000 graph samples aggregated across all atlases and parcellations. The code is available at https://github.com/weixinxu666/BrainGFM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces BrainGFM, a unified brain graph foundation model that combines graph contrastive learning and graph masked autoencoder objectives for large-scale pre-training on fMRI-based brain network datasets. The framework aims to handle heterogeneous datasets and diverse atlases across multiple disease domains, and the authors demonstrate that BrainGFM achieves superior performance compared to existing graph pre-training methods.

### Strengths
1. Timely and significant topic. The paper tackles an important and emerging direction—developing a foundation model for brain networks—addressing heterogeneity across datasets and atlases.
2. Large-scale pre-training. Conducting large-scale pre-training on multiple fMRI datasets is of high value to the neuroimaging and machine learning communities.
3. Cross-atlas and cross-disease generalization. The framework is designed to support different brain parcellations and disease categories, which is an ambitious and meaningful goal for improving model transferability.
4. Empirical performance. The proposed approach outperforms existing graph pre-training methods across multiple benchmarks, suggesting that the unified framework is effective in large-scale neuroimaging representation learning.

### Weaknesses
1. Limited consideration of brain-specific properties.
		○ The pre-training strategy primarily adapts existing MAE and GCL techniques from general graph learning without tailoring them to the unique characteristics of brain networks. In neuroimaging, lesions may appear in localized regions (specific ROIs), where neighboring nodes are normal and carry limited predictive signal. Generic GCL or MAE objectives may therefore be suboptimal for such sparse or localized perturbations.
		○ The design of disease and atlas prompts is relatively simplistic. It fails to capture deeper relationships between diseases and specific brain regions, or the structural hierarchy across multiple atlases. The model could benefit from biologically informed prompting or hierarchical alignment strategies.
	2. Underutilization of multi-atlas information.
		○ For the same subject, multiple atlases provide complementary representations of brain connectivity. However, BrainGFM treats these views separately. Several prior works (e.g., [1–3]) have explored multi-atlas fusion to enhance representational richness and robustness. Integrating cross-atlas consistency learning could substantially improve the model’s capability.
	3. Sequential pre-training strategy.
		○ The current training pipeline processes datasets sequentially, which raises concerns about order sensitivity and incomplete exploitation of large-scale data. Sequential training can lead to dataset-specific overfitting and unstable convergence.
		○ A preferable approach would be joint pre-training—mixing or shuffling all datasets—to ensure more uniform optimization and reduce order-dependent biases. This would also yield a more scalable training complexity, dependent primarily on total sample size rather than dataset count.
	4. Reproducibility and data availability.
		○ The contribution would be significantly strengthened if the datasets used for pre-training could be made publicly available or at least described in sufficient detail for replication. Given the emphasis on large-scale pre-training, the lack of accessible data may hinder reproducibility and follow-up work.
	5. Minor presentation issues.
		○ The legend in Figure 4 overlaps with surrounding text and should be repositioned.
		○ The atlas used in each experiment should be explicitly mentioned in figure/table captions to make the results self-contained.


[1] Multiview Feature Learning With Multiatlas-Based Functional Connectivity Networks for MCI Diagnosis. TCYB 2022
[2] Pretraining is All You Need: A Multi-Atlas Enhanced Transformer Framework for Autism Spectrum Disorder Classification. MLCN 2023
[3] Multi-Atlas Brain Network Classification through Consistency Distillation and Complementary Information Fusion. JBHI 2025

### Questions
Please refer to weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission presents BrainGFM, a novel graph-based foundation model for fMRI brain data that leverages both graph contrastive learning and graph masked autoencoders for large-scale pre-training. The model is trained on a diverse dataset comprising over 25,000 subjects and 60,000 scans across 25 neurological and psychiatric disorders, using multiple brain atlases and parcellations. BrainGFM introduces graph prompts and language prompts to enable efficient few-shot and zero-shot transfer learning. The authors also employ meta-learning to optimize graph prompts, allowing adaptation to unseen disorders and atlases without fine-tuning the entire model.

### Strengths
1/ BrainGFM uses graph-based pre-training with both generative (GMAE) and contrastive (GCL) objectives. This approach effectively captures both local and global brain graph structures.

2/ The model is pre-trained on a large-scale, heterogeneous dataset that includes multiple brain atlases (functional and anatomical) and a wide range of neurological/psychiatric disorders.

3/ The integration of graph and language prompts, along with meta-learning, enables BrainGFM to adapt to new tasks, atlases, and disorders with minimal labeled data.

### Weaknesses
1/ Despite claims of modality-agnostic design, all experiments are conducted solely on resting-state fMRI data for a brain foundation model. There is no validation on other neuroimaging modalities such as task-fMRI, DTI, or EEG. This limits the generalizability of the proposed framework and leaves its cross-modal transferability unverified.

2/ While language prompts are introduced to guide zero-shot transfer, the paper lacks a detailed ablation study to isolate their contribution. It is unclear how much performance gain is attributable to the semantic guidance of language prompts versus the graph prompts or the pre-trained backbone itself.

3/ The brain graphs are constructed using Pearson correlation and top-k sparsification, but the paper does not thoroughly explore how different graph construction strategies (e.g., partial correlation, mutual information, or dynamic connectivity) affect model performance. This is critical since graph quality directly influences downstream task outcomes.

4/ While the paper emphasizes efficiency during fine-tuning via prompt-tuning, it does not thoroughly discuss the computational cost of inference or scalability challenges when deploying BrainGFM across large clinical datasets or real-time applications. For example, how does the model perform on edge devices or in hospitals with limited GPU resources? Without such analysis, the practical deployability of BrainGFM remains unclear.

5/ By converting fMRI time series into static brain graphs, the model discards rich temporal information that may be critical for certain disorders (e.g., schizophrenia or bipolar disorder, which exhibit dynamic connectivity patterns). The paper does not justify this design choice or compare with temporal modeling approaches (e.g., BrainLM or Brain-JEPA), leaving open the question of whether graph-based modeling is truly superior for all types of disorders.

### Questions
1/ The paper uses meta-learning to optimize graph prompts for few-shot adaptation across disorders and atlases. However, it remains unclear how sensitive the model is to the choice of meta-training tasks — for example, if certain disorders or atlases are underrepresented in the meta-training phase, could this lead to biased or unstable prompt initialization that generalizes poorly to rare or unseen conditions?

2/ While the model integrates language prompts for zero-shot transfer, the textual descriptions of disorders are generic clinical summaries. How does the model handle semantic overlap or ambiguity between similar disorders?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes BrainGFM, a large-scale graph-based fMRI foundation model. BrainGFM is pre-trained across multiple datasets with various parcellations. The model combines contrastive and generative pre-training of graph structures, then introduces meta-learned graph prompts for few-shot adaptation and language prompts for zero-shot generalization. The experiments demonstrate the validity of BrainGFM by comparative and ablation results.

### Strengths
- Good motivation with an effective solution: scaling data across different atlases/parcellations
- Extensive experiments and good results

### Weaknesses
- Neuroscientific interpretability (explainability of the input brain graph) is not addressed within the BrainGFM framework.
- Computational efficiency of BrainGFM is not quantitatively provided.

### Questions
### Major
- Please consider an experiment on the interpretability of the input graph within the BrainGFM framework (or at least adding a discussion on the possible interpretability).
- There are claims on the computational efficiency of BrainGFM, but formal quantitative evidence to support this is limited. Please provide numerical benchmarks to substantiate the efficiency claims.


### Minor
- Please improve the readability of the figures (e.g., tick labels of the radar charts are too small to read).
- Some abbreviations are defined multiple times.

### Soundness
3

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
4

### Summary
This paper introduces **BrainGFM**, a graph-based fMRI foundation model that pre-trains a Graph Transformer on multi-atlas brain graphs using a dual SSL-based objective (graph contrastive learning and graph masked autoencoding) and adapts via **graph prompt-tuning** optimized with **meta-learning**. For zero-shot transfer, the model augments graph tokens with **language prompts** that encode disorder/task and atlas/parcellation semantics. Extensive experiments across diverse datasets and atlases show that multi-atlas pre-training improves downstream accuracy and that prompt-tuning is parameter-efficient in few/zero-shot settings. Overall, the work pushes toward a unified, atlas-aware foundation model for functional connectomics.

### Strengths
- **Well-articulated motivation and problem setting.**  
  The paper clearly identifies the data scarcity, heterogeneity, and task-specific limitations of existing fMRI models, effectively motivating the need for a scalable, generalizable foundation model. The discussion connects practical constraints in fMRI acquisition and dataset diversity to methodological design choices, making the problem definition both convincing and well grounded.

- **Multi-atlas large-scale pre-training with cross-parcellation gains.**  
  The model leverages multiple parcellations/atlases per scan and reports improved generalization over single-atlas training on several benchmarks.

- **Parameter-efficient adaptation and zero-shot generalization.**  
  Parameter-efficient adaptation via graph prompts supports few-shot learning, and language prompts enable zero-shot transfer to unseen disorder or task settings.

### Weaknesses
1. **Limited pretraining methodological novelty.**  
   The proposed pretraining framework combines existing self-supervised techniques—Graph Contrastive Learning (GCL), Graph Masked Autoencoding (GMAE), and MAML-based prompt tuning—without introducing a clearly novel algorithmic component. While the large-scale integration across atlases and disorders is valuable, the study could have been strengthened by exploring a more original approach, such as multimodal or text-guided self-supervised objectives. In addition, the authors could consider referencing or benchmarking against brain-specific self-supervised learning approaches that have recently emerged in fMRI.

```
[1] Yu, et al., "Causal Invariance-aware Augmentation for Brain Graph Contrastive Learning", ICML 2025.
[2] Dong, et al., "Counterfactual Brain Graph Augmentation Guided Bi-Level Contrastive Learning for Disorder Analysis", ICDM 2024.
[3] Ma, et al., "Self-Promoted Clustering-based Contrastive Learning for Brain Networks Pretraining", IJCAI 2024.
[4] Zhang, et al., "A-GCL: Adversarial graph contrastive learning for fMRI analysis to diagnose neurodevelopmental disorders.", Medical Image Analysis, Volume 90, 2023 December.
```

2. **Unclear validation protocol and potential leakage.**  
   The paper does not clarify whether subjects rendered under multiple atlases are strictly separated across pre-train and evaluation groups, which could inflate the reported generalization. In addition, the evaluation protocol is not fully consistent—comparisons are limited to the Schaefer100 atlas, and it remains unclear whether baseline models were fairly tuned under comparable single-atlas and hyperparameter settings.

3. **Limited empirical grounding of claimed generalization.**  
   While the language-prompt and cross-atlas transfer results are interesting, the paper lacks semantic or ablation controls (e.g., shuffled or neutral prompts) to confirm that the observed zero-shot gains reflect meaningful generalization rather than reliance on dataset or label cues.

4. **Ambiguous evaluation of computational efficiency.**  
   Although the paper highlights computational efficiency, it only provides qualitative labels such as *Fast/Slow* for training speed and *High/Low* for memory usage, without quantitative metrics (e.g., FLOPs, GPU-hours) to support these claims. As a result, it is difficult to assess the actual scalability and efficiency of the proposed model relative to the baselines.

### Questions
**Q1)** GraphMAE performs only node reconstruction in its original formulation. How are edges reconstructed in BrainGFM? Additionally, beyond combining existing SSL components (GCL, GMAE), what constitutes the unique methodological novelty of the proposed self-supervised framework?

**Q2)** In Section 4.1, comparisons with baseline models are presented only for the Schaefer100 atlas. Have the authors conducted similar evaluations under other atlas or parcellation settings to verify whether the observed performance gains generalize beyond a single atlas configuration?

**Q3)** In Section 4.4, on which dataset are the reported comparison results based? The table shows BrainLM as “High” and BrainMass as “Low,” yet in Section 4.1 their numerical performance gap appears minimal. Could the authors clarify which dataset or evaluation protocol was used for Section 4.4 and how these qualitative rankings relate to the quantitative results in Section 4.1?

**Q4)** The appendix states that BrainGFM uses a unified set of hyperparameters for fine-tuning across all datasets. Were the baseline models also tuned under fair and dataset-appropriate hyperparameter configurations, or did they rely on default or fixed settings? In particular, for pre-trained models such as BrainLM—which are limited to single-atlas training—were they pre-trained using the same data subset as BrainGFM and only the Schaefer100 atlas setting? Additionally, were the same fine-tuning epochs applied uniformly across all datasets, and were the baseline models and BrainGFM trained for an equivalent number of epochs? 

**Q5)** During the prompt-tuning stage, the model is trained with specific Task/Disorder (T/D) and Atlas/Parcellation (A/P) configurations. Does this imply that BrainGFM cannot generalize to entirely unseen tasks or atlases/parcellations during zero-shot transfer? If zero-shot generalization is claimed, could the authors clarify whether it applies only to unseen disorders/tasks (via language prompts) or also to unseen atlases, and how this is achieved given that A/P tokens are learned during training?

**Q6)** Was a leakage audit performed to ensure subject-level disjointness across all pre-train, in-domain, and external groups after multi-atlas expansion? This is a key missing detail in the paper, as each subject’s fMRI scan can appear under multiple atlases. Clarifying whether strict subject-level separation was enforced would strengthen the reliability of the reported generalization.

**Q7)** Were control experiments conducted to verify that the zero-shot improvements from language prompts truly arise from semantic alignment rather than label priors? For example, have the authors tested shuffled [task/disorder]–label mappings, semantically neutral texts of matched length, or atlas-only identifiers without disorder text?

**Q8)** Have the authors examined whether the learned representations depend heavily on [A/P] tokens?  
Instead of removing these tokens (which may collapse the model), analyzing representation similarity or clustering across different atlases could help determine whether BrainGFM learns truly atlas-invariant features.

**Q9)** Could the authors provide approximate compute cost (e.g., GPU-hours, FLOPs, or peak memory) to contextualize the claimed efficiency advantage? While BrainGFM emphasizes computational efficiency compared to time-series or ROI-based foundation models, quantitative evidence would clarify the strength of this claim.

### Soundness
3

### Presentation
3

### Contribution
2
