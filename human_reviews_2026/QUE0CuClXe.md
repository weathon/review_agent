# Bootstrapping MLLM for Weakly‑Supervised  Class‑Agnostic Object Counting

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6, 4

## Abstract
Object counting is a fundamental task in computer vision, with broad applicability in many real-world scenarios. Fully-supervised counting methods require costly point-level annotations per object. Few weakly-supervised methods leverage only image-level object counts as supervision and achieve fairly promising results. They are, however, often limited to counting a single category, \eg person. In this paper, we propose WS-COC, the first MLLM-driven weakly-supervised framework for class-agnostic object counting. 
Instead of directly fine-tuning MLLMs to predict object counts, which can be challenging due to the modality gap, we incorporate three simple yet effective strategies to bootstrap the counting paradigm in both training and testing: First, a divide-and-discern dialogue tuning strategy is proposed to guide the MLLM to determine whether the object count falls within a specific range and progressively break down the range through multi-round dialogue. Second, a compare-and-rank count optimization strategy is introduced to train the
MLLM to optimize the relative ranking of multiple images according to their object counts. Third, a global-and-local counting enhancement strategy aggregates and fuses local and global count predictions to improve counting performance in dense scenes. Extensive experiments on FSC-147, CARPK, PUCPR+, and ShanghaiTech show that WS-COC matches or even surpasses many state-of-art fully-supervised methods while significantly reducing annotation costs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes WS-COC, the first MLLM-driven weakly-supervised framework for class-agnostic object counting. Instead of directly fine-tuning MLLMs to predict object counts, WS-COC incorporates three key strategies: divide-and-discern dialogue tuning, compare-and-rank count optimization, and global-and-local counting enhancement. Extensive experiments are conducted on four benchmarks, showing that WS-COC matches or surpasses many fully-supervised methods while reducing annotation costs significantly.

### Strengths
1. This work innovatively proposes the first weakly supervised general category object counting framework WS-COC based on MLLM, filling the technical gap of "MLLM+weakly supervised+general category" object counting.
2. It designs three core strategies. The Divide-and-Discern Dialogue Tuning converts counting into multi-round range judgment to reduce learning difficulty; the Compare-and-Rank Count Optimization establishes the connection between visual features and counting through relative ranking; the Global-and-Local Counting Enhancement fuses multi-scale predictions to correct biases. These strategies optimize counting capabilities throughout the entire process from training to inference.
3. The experimental design is rigorous and comprehensive. It selects four representative benchmark datasets (FSC-147, CARPK, PUCPR+, and ShanghaiTech) that cover diverse scenarios and categories, ensuring the universality of results. Key implementation parameters such as MLLM backbone and LoRA settings are reported in detail to ensure experimental reproducibility.

### Weaknesses
1. Its performance is still insufficient in extremely dense scenes or under severe target occlusion. The model struggles to accurately distinguish individual instances, leading to cases of large counting errors.
2. In the Compare-and-Rank Count Optimization strategy, cross-category sampling introduces semantic differences and leads to performance degradation. Currently, image sampling can only be conducted within the same category, which limits the flexibility and application scope of the strategy.
3. The Global-and-Local Counting Enhancement strategy uses simple grid partitioning to split images into sub-images, which easily causes overestimation of local counts due to edge effects. Although this issue is alleviated by averaging with global counts, the fundamental problem caused by the partitioning method is not solved, leaving room for optimization.

### Questions
1. Regarding the failure cases in extremely dense scenes and severe occlusion (mentioned in the paper’s appendix), what specific characteristics of these scenes most strongly correlate with the model’s performance degradation? Is there a quantitative analysis to clarify the key factors limiting WS-COC’s performance in such scenarios?
2. In the Compare-and-Rank Count Optimization (CRCO) strategy, cross-category sampling reduces performance due to semantic variance. However, the paper does not specify whether the performance decline is consistent across all category pairs.
3. The Global-and-Local Counting Enhancement (GLCE) strategy uses a fixed threshold to distinguish dense and sparse scenes. How was this threshold determined?
4. Extend CRCO to support semi-cross-category sampling: Instead of restricting sampling to the same category, test a "semi-cross-category" sampling strategy. For example, group semantically similar categories (e.g., "apple", "pear", "orange" as a "fruit group") and sample images within each group.
5. Optimize the dense-scene threshold c^h adaptively.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the use of Multimodal Large Language Models (MLLMs) for weakly-supervised, class-agnostic object counting. To enhance MLLMs’ performance on dense object scenes, the authors propose two training strategies and one inference strategy. Experimental results demonstrate that the proposed approach achieves state-of-the-art performance among weakly-supervised models and is only slightly inferior to the best fully-supervised counterparts. Ablation studies further validate the effectiveness of the proposed strategies.

### Strengths
- The paper is one of the first attempts to explore the use of MLLMs for weakly-supervised, class-agnostic object counting. The results achieved by the proposed method are competitive with those of fully-supervised approaches, which is a notable contribution to the field.
- The proposed training and inference strategies are simple yet effective, and the experiments are meticulously structured.

### Weaknesses
- **The paper lacks analysis of inference speed and computational cost**. Since MLLMs are computationally expensive, it is important to quantify inference time, memory usage, and FLOPs, as these factors are critical for real-world deployment.
- **Limited mechanism interpretability**. Although the proposed method is empirically effective, the underlying mechanisms that drive the improvement remain unclear. Visualizations of model attention or response patterns (e.g., Grad-CAM on the vision-language alignment) may help elucidate this.
- **The paper lacks qualitative comparisons with other object counting models**. The visual results only compare different variants within the MLLM family, but omit comparisons with existing leading density-map based or detection-based weakly- or fully-supervised baselines. Including such comparisons could help clarify the strengths of the proposed method relative to other approaches.

### Questions
- MLLMs are known to incur high inference costs, making them unsuitable for deployment on resource-constrained devices. What potential application scenarios or deployment strategies do the authors envision for MLLM-based counting models?
- Could the authors provide visual or attention-based analyses to better understand how the model perceives and counts objects? This would help clarify how the model interprets scenes and focuses on visual cues during the counting process.

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
5

### Summary
This paper introduces WS-COC, a novel framework for weakly-supervised, class-agnostic object counting. This is the first work, to my knowledge, to successfully bootstrap a Multimodal Large Language Model (MLLM) for this task using only image-level count supervision. The core problem is that MLLMs, while capable of basic counting, fail significantly in dense scenes.

### Strengths
The paper tackles a highly practical and important problem: class-agnostic counting without expensive point-level annotations. The approach of "bootstrapping" a generative MLLM (LLaVA) for this counting task, rather than building a discriminative VLM (like CLIP) with a specialized counting head, is a novel and promising direction.

The experimental validation is thorough. WS-COC's performance on the primary benchmark, FSC-147, is excellent, significantly reducing the MAE in dense scenes compared to baselines (Fig. 1(d)).

### Weaknesses
The paper's CRCO strategy is well-motivated as a way to handle the modality gap using relative ranking. However, the related work section overlooks prior art from the VLM (Vision-Language Model) domain that has explored similar concepts. A discussion in Section 2.1 or 2.2 situating WS-COC's CRCO strategy against VLM-based ranking methods like CrowdCLIP would significantly strengthen the paper's contribution and provide clearer context on its novelty.


The failure case analysis shows the model still struggles with "extremely dense scenes" (e.g., "books" GT: 1021, Pred: 732). Is this a fundamental limitation of the MLLM's spatial understanding and resolution, or could this be further mitigated by, for instance, a finer-grained GLCE partitioning (e.g., $L=3$ or $L=4$, which Table 4 shows performed worse) or a more sophisticated data augmentation strategy for dense scenes?

### Questions
see weakness

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
3

### Summary
This paper proposes WS-COC, a weakly-supervised class-agnostic object counting framework that bootstraps multimodal large language models for object counting using only image-level supervision. Instead of directly fine-tuning MLLMs for regression, the authors introduce three strategies: (1) Divide-and-Discern Dialogue Tuning, which reformulates counting as iterative range judgment through multi-round dialogue; (2) Compare-and-Rank Count Optimization, which trains the MLLM to rank multiple images by their object counts; and (3) Global-and-Local Counting Enhancement, which fuses global and local predictions at inference to better handle dense scenes. Experiments show that WS-COC presents good performance while using only weak supervision.

### Strengths
- The paper is clearly written and well-structured, with effective figures that help illustrate the intuition behind each proposed component.
- The three proposed strategies are conceptually simple yet effective, with clear motivations.
- The method achieves better results compared to other methods under image-level supervision.

### Weaknesses
- The paper does not discuss computational efficiency. Since the framework relies on large multimodal language models and multi-round dialogues, a comparison of training and inference cost against other weakly- and fully-supervised baselines would help clarify its practical scalability and deployment feasibility.
- The experiments are conducted on general object counting datasets; however, the model’s ability to generalize to other domains remains unclear. For example, it would be valuable to investigate how the proposed framework performs on indiscernible object counting tasks where objects are visually ambiguous or heavily occluded. The related references are listed below.
- Missing related work. Several recent works on object counting are relevant and should be discussed:
   - A. Distribution Matching for Crowd Counting
   - B. Indiscernible object counting in underwater scenes
   - C. Counting Everyday Objects in Everyday Scenes

### Questions
Please see the points listed in weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles the task of weakly supervised class-agnostic counting when only image-level counts are available. Proposed approach does not require point wise annotations. Instead of directly training an MLLM to emit a number, authors propose three strategies on top of Llava style MLLM: (1) Divide and Discern Dialogue Tuning (D3T): turn counting into a multi round binary search style judgement dialogue, then finalize the count once the range is narrow; (2) Compare and rank Count Optimization (CRCO): teach relative ordering by ranking 4 images sampled from different count bins per class; (3) Global and Local Counting Enhancement (GLCE): at inference, average the global prediction with a grid of local sub image predictions to balance global under counting and local over counting.

### Strengths
Proposed approach achieves reasonable performance without using any point wise annotations which are expensive to obtain.

### Weaknesses
1. Novelty of the proposed approach is somewhat limited. Each of the three proposed components rely on fairly standard heuristics. Novelty of the work is more in combining them for this setting than in any single algorithmic leap.
2. GLCE uses a 2×2 grid, with simple averaging of global and local. While GLCE is effective, it’s a heuristic and can be dataset sensitive. 
3. Results are reported for only one MLLM backbone and model size. It's not clear if the approach will generalize to other MLLMs and model sizes.

### Questions
1. As an ablation study, I would be curious to see MLLM-Zero + GLCE and WS-COC-Base + GLCE perform. Since GLCE can be applied at test time, it is compatible with both MLLM-Zero and WS-COC-Base. 
2. What are the main differences between the proposed approach and other contemporary works like https://arxiv.org/pdf/2412.00686v1 which also propose a divide and conquer approach for class agnostic counting, similar to GLCE.

### Soundness
2

### Presentation
3

### Contribution
2
