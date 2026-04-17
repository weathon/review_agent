# Adaptive Rank, Reduced Forgetting: Knowledge Retention in Continual Learning Vision-Language Models with Dynamic Rank-Selective LoRA

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Continual learning (CL) aims to accumulate knowledge from sequential tasks without catastrophic forgetting. Vision–language models like CLIP, with strong generalization, are widely used for CL. Existing methods often adapt isolated PTM components, adding inference complexity and limiting PTM improvement, or rely on replay, stored information, or assumptions, incurring high costs and limited applicability. To advance models as continual learners, we explore CL via natural, efficient PTM updates instead of complex task-specific additions. We thus study continual low-rank learning and systematically analyze how LoRA ranks and placements affect *learning* and *forgetting*. We find that a relatively *higher-rank* LoRA improves task learning (*i.e.*, *plasticity*) but increases forgetting, while a relatively *lower-rank* LoRA reduces forgetting (*i.e.*, *stability*) but limits adaptation. Crucially, we find a *plasticity–stability balance* tied to rank across parameters and tasks, with *moderately small ranks* maximizing CL benefits. Motivated by this, we propose **Co**ntinual **Dy**namic **R**ank-Selective LoR**A** (**CoDyRA**), which continually updates PTMs with LoRA adapters of adaptively optimized rank. While the new-task objective drives learning, CoDyRA adaptively minimizes ranks with *sparsity-promoting regularization* to reduce interference and forgetting, achieving a plasticity–stability balance tailored to different parameters and tasks. Adaptively selected and minimized LoRA ranks keep the updated model closer to its previous state while learning new tasks. CoDyRA enables efficient CL as a sequence of LoRA-based tasks without storing past data, task information, or relying on assumptions. It preserves the original model architecture and deployment pipeline, adding no inference overhead. Extensive experiments show CoDyRA improves new representations while retaining old knowledge, achieving state-of-the-art results.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of catastrophic forgetting in continual learning (CL) for pre-trained vision-language models (VLMs) like CLIP. The authors note that existing methods have significant drawbacks: approaches relying on task-specific modules add inference complexity and require task-ID prediction, while replay-based methods incur high storage and computational costs.
To overcome these limitations, this paper explores performing CL through Low-Rank Adaptation (LoRA), a natural and efficient update mechanism. The authors first systematically analyze how LoRA's rank and placement affect the trade-off between "plasticity" (the ability to learn new tasks) and "stability" (the ability to retain old knowledge). Their key finding is that a relatively higher rank promotes plasticity but exacerbates forgetting, while a relatively lower rank enhances stability but limits the model's adaptation. They find that an optimal balance exists at a moderately small rank, although this balance point varies across different parameter locations and tasks.
Based on this analysis, the paper proposes CoDyRA (Continual Dynamic Rank-Selective LoRA). This method continually updates the VLM by adaptively optimizing the rank of each LoRA adapter. CoDyRA achieves this by jointly optimizing two objectives: (1) the standard new-task learning objective, which drives plasticity; and (2) a sparsity-promoting $l_1$ regularization applied to a set of learnable "importance weights" for each rank. This regularization dynamically minimizes the number of active ranks, forcing the model update to remain closer to its previous state, thereby reducing interference and forgetting.
The main contributions of this paper include:
A systematic study of the impact of LoRA's rank and placement on the plasticity-stability trade-off in VLM continual learning.
The proposal of the CoDyRA method, which uses sparsity-promoting regularization to adaptively select and minimize the rank of LoRA. It operates without storing past data, requiring task information, or adding task-specific components.

### Strengths
1. Solid Analysis and Clear Motivation: A core strength of this paper is the systematic analysis provided in Section 3.2. The authors delve into the impact of the rank and placement of LoRA on the tradeoff between plasticity (learning new knowledge) and stability (retaining old knowledge). This analysis clearly reveals why a fixed-rank LoRA strategy is not optimal and provides strong motivation and design guidance for the subsequently proposed adaptive-rank method.

2. Ingenious Parameter Update Mechanism: CoDyRA does not directly optimize the discrete "rank," but instead introduces a set of learnable "importance weights" ($w^{t,m}$) for each LoRA module and innovatively combines them with $l_1$ sparse regularization. This design cleverly transforms the difficult discrete rank selection problem into a solvable continuous sparse optimization problem. Furthermore, the authors employ the proximal gradient method and its soft thresholding operation to effectively solve the non-differentiability optimization problem caused by the $l_1$ norm.

3. The paper provides robust ablation experiments and parameter sensitivity analysis (Section 4.4) 9. The authors not only verified the influence of LoRA insertion position and initial rank 10, but also conducted in-depth analysis of key hyperparameters, such as the maximum pruning threshold ($\kappa_{max}$) and dense-iteration ratio 11, fully demonstrating the rationality and robustness of the model design.

### Weaknesses
The paper's core assumption is that "minimizing LoRA rank" can serve as an effective proxy for "reducing catastrophic forgetting". However, the analysis in Section 3.2 (Fig. 3) primarily demonstrates a correlation between rank and forgetting, but fails to deeply investigate the causality. The root cause of forgetting is the interference of the parameter update direction with old tasks, not just the rank of the update. For instance, a high-rank update might not cause forgetting if its update direction is orthogonal to the parameter subspace of old tasks. Conversely, a low-rank (or even rank-1) update could be catastrophic if its direction is incorrect (e.g., directly opposes a critical gradient direction for a previous task). The paper lacks a deeper parameter-space analysis to substantiate this assumption. For example, does the low-rank increment $\Delta W$ generated by CoDyRA's $l_1$ sparsification truly interfere less with old knowledge in terms of its update direction compared to standard LoRA? A specific analysis of this direction's orthogonality or interference is missing. Currently, the validity of "low rank" as a robust proxy for "low forgetting" has not been sufficiently theoretically or empirically demonstrated.

### Questions
### Potential Typo in Core Equations (Eq. 4 & 7)

There appears to be a typo in the soft-thresholding operator used for the \( l_1 \) regularization.

1. **The Goal:**  
   The paper aims to use \( l_1 \) regularization to promote sparsity by pushing the importance weights \( w \) **toward 0**.

2. **The Formula:**  
   However, Eq. (4) and Eq. (7) define the operator with a **plus sign**:

   \[
   w_{i} := \mathbb{I}(|\hat{w}_{i}| > \kappa) \cdot (\hat{w}_{i} + \operatorname{sign}(\hat{w}_{i}) \cdot \kappa)
   \]

3. **The Contradiction:**  
   This formula would actually *amplify* the weights (e.g., 5 becomes 6), moving them **away from 0**.  
   This is the opposite of sparsity.

---

Should this formula use a **minus sign**  
(\( \ldots - \operatorname{sign}(\hat{w}_{i}) \cdot \kappa \))  
to correctly implement the soft-thresholding (shrinkage) operation?  
Please clarify if this is a typo in the manuscript and if the correct operator was used in the implementation.

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
This paper proposes Continual Dynamic Rank-Selective LoRA (CoDyRA) for continual learning, mainly for cross-domain continual learning. The proposed method has been developed based on CLIP VLM and LoRA approach. The paper is well-presented and comprehensive. The paper is easy to follow with helpful highlights for the reader. Despite its positive values, this paper has a few fundamental issues; please see the weaknesses.

### Strengths
Strength:
(1). This paper discusses an interesting up-to-date sub-problem of CL, i.e, cross-domain CL.

(2). The motivation of the proposed method is clear.

(3). The writing is well presented with helpful highlights, charts, and diagrams.

(4). The experiment results are comprehensively delivered and conducted on many datasets.

### Weaknesses
(1). The main idea of the proposed method is adding new trainable weight w^{t,m}_i associated with B^{t,m}_{:i}. The proposed trainable weights are supposed to make \delta W^{t,m} more adaptive. From the methodology perspective, the idea is arguably not novel as the idea of trainable weights was already applied in previous methods, e.g, ConvPrompt [CVPR 2024].

(2). No theoretical and numerical proofs showing that the trainable weights idea improves model adaptation and reduces model forgetting significantly.

(3). Performance issue: Even though the proposed method achieves the highest performance on average, it is significantly outperformed by the previous method (RAIL-Primal), i.e, 11\% on Caltect101 dataset and 3.9\% on Flowers dataset.

(4). Continual learning is the art of defying catastrophic forgetting (CF). But, I do not see a comprehensive forgetting analysis.

(5). The paper misses a comparison of the proposed method to the newest CLIP and LoRA-based CL methods, i.e, CLAP4CLIP (NeurIPS 2024), C-CLIP (ICLR 2025), Mind-the-Gap (ICCV 2025), CL-LoRA(CVPR-2025), InfLORA (CVPR 2024).


References:

[1]. CLAP4CLIP: Continual learning with probabilistic finetuning for vision-language models.

[2]. C-CLIP: Multimodal continual learning for vision-language model.

[3]. Mind the gap: Preserving and compensating for the modality gap in clip-based continual learning.

[4]. CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning

[5]. Inflora: Interference-free low-rank adaptation for continual learning.

[6] Convolutional prompting meets language models for continual learning (ConvPrompt)

### Questions
Please address the weaknesses.

### Soundness
2

### Presentation
4

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
This paper targets continual learning (CL) for vision-language models like CLIP by introducing CoDyRA. This method dynamically optimizes the rank of LoRA adapters during sequential task learning to balance the plasticity and stability. The authors systematically analyze the impact of LoRA rank and placement on learning-forgetting trade-offs and propose an adaptive rank-selection mechanism driven by sparsity-promoting regularization. Extensive experiments on benchmarks such as MTIL and X-TAIL demonstrate improved performance over SOTA methods in retaining pre-trained capabilities while improving generalization, with no inference overhead. Though very intriguing and promising, this work could benefit from a more in-depth theoretical analysis and a more structured presentation.

### Strengths
1. It introduces a fine-grained, adaptive approach to LoRA-based CL, offering a novel perspective supported by convincing experimental validation.
2. The method is straightforward to implement and exhibits potential for scalability due to its simplicity.
3. Comprehensive experiments across diverse benchmarks and model configurations, including visualizations, robustly substantiate the core claim that rank manipulation addresses the learning-forgetting trade-off effectively.

### Weaknesses
1. Though the authors have provided a preliminary empirical analysis of the impact of the lora location and rank (sec 3.2), the study lacks direct theoretical derivation or proof, necessitating deeper analytical foundations beyond empirical results.
2. Marginal improvements in Tables 1 and 2 (often fractions of a percent, less than 1%) raise concerns about the method’s effectiveness and generality compared to state-of-the-art approaches.
3. Most experiments use the ViT-B/16 backbone of CLIP. More tests on a larger or different model architecture and different pre-trained parameters could give a broader impact assessment.

### Questions
See the weakness part.

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
This paper proposes CoDyRA (Continual Dynamic Rank-Selective LoRA), a continual learning method for vision–language models like CLIP that updates pre-trained models using LoRA adapters with adaptively optimized ranks. By balancing plasticity and stability through dynamic rank selection and sparsity regularization, CoDyRA enables efficient continual updates without replay, task-specific modules, or added inference cost, achieving state-of-the-art performance while preserving prior knowledge.

### Strengths
1. The paper is well-structured and clearly written, making it easy to follow.

2. The study tackles an important problem in continual learning by employing low-rank adaptation.

3. The experimental evaluation is comprehensive, incorporating a wide range of baselines and datasets, which enhances the credibility of the paper’s conclusions.

### Weaknesses
1. Some related works appear to have been overlooked. There are also several recent studies that attempt to adjust their architectures dynamically in continual learning, such as TreeLoRA.

    TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree.

2. Can the authors extend the proposed method to large language models to further validate its scalability?

### Questions
See weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
3
