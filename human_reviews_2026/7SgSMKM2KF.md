# Asynchronous Matching with Dynamic Sampling for Multimodal Dataset Distillation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Multimodal Dataset Distillation (MDD) has emerged as a vital paradigm for enabling efficient training of vision-language models (VLMs) in the era of multimodal data proliferation. Unlike traditional dataset distillation methods that focus on single-modal tasks, MDD presents distinct challenges: (i) the effective distillation of heterogeneous multimodal knowledge, complicated by feature space misalignment and asynchronous optimization dynamics; and (ii) the lack of discrete class guidance, which hinders the distribution coverage and representativeness of synthetic data due to the vastness and continuity of the semantic space. To address these challenges, this paper proposes an Asynchronous Matching with Dynamic sampling (AMD) framework. AMD enables asynchronous trajectory matching by decoupling the selection of starting points for image and text trajectories. Additionally, a Semantics-Aware Prototype Mining module is introduced, which replaces random initialization by leveraging feature-space clustering to identify representative prototypes, enhancing the coverage and representativeness of the distilled samples. Extensive experiments demonstrate that AMD achieves superior distillation performance on Flickr30k and COCO (e.g., IR@1, IR@5, and IR@10 \textbf{gains of 4.5\%, 9.6\%, and 10.9\%}, respectively, on Flickr30k 200 pairs.) with negligible computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on Multimodal Dataset Distillation (MDD), a task that aims to distill a small amount of synthetic data capable of achieving performance comparable to the original dataset. To address the challenge of discrepancy between image and text distillation stages, this work proposes an asynchronous matching strategy. Furthermore, it introduces the Semantics-aware Prototype Mining (SPM) module for initializing the searching space and identifying representative prototypes. Experimental results on the Flickr30K and COCO datasets demonstrate the effectiveness of the proposed AMD method.

### Strengths
1. The motivation is strong, underpinned by a solid experimental analysis (Sec 3.1) that effectively highlights the discrepancy problem in previous multimodal distillation methods. This makes the proposed asynchronous method (AMD) well-justified.

2. The proposed algorithm is commendably simple yet effective. The substantial improvements over prior LoRS work (Tables 1 & 2) clearly demonstrate the effectiveness and superiority of the asynchronous matching strategy.

### Weaknesses
1. My primary concern is the sufficiency of the contribution. While the paper identifies its core contribution as asynchronous matching with dynamic sampling for MDD, the algorithm appears to be summarized as matching teacher and student networks within controlled ranges $R_L$ and $R_V$. Could the authors clarify if this understanding is accurate, or if I am oversimplifying the approach?

2. Regarding the Semantic Prototype Mining (SPM) module, it appears to function more as an initialization trick than a core MDD-specific method. SPM uses K-means to identify $B$ representative prototypes based on concatenated features, which then serves as an initialization for the subsequent matching process. Critically, this module seems unrelated to the 'asynchronous' matching strategy. Moreover, it comes across as an effective heuristic rather than a genuinely learnable component.

3. The authors claim (L86) that the achieved improvements incur negligible additional computational overhead. However, this assertion lacks supporting evidence in the main paper. While Appendix Table 8 compares running times, it notably omits the computational cost of SPM initialization, specifically the K-means clustering time.

### Questions
1. I am confused by the statement in L198 regarding "decoupling the distillation paths of images and text modalities." From Figure 1, the proposed asynchronous matching appears to couple the two modality trajectories at different timestamps. In contrast, previous synchronous methods seem to operate in a more decoupled manner, as image and text trajectories are matched strictly pair by pair. 

2. What specific value is chosen for $B$ (the number of clusters) in the SPM module? Furthermore, are there any ablation studies investigating the impact of different $B$ values?

3. Could the authors provide a more detailed explanation of Figure 3? Specifically, what do the x and y axes represent, and what is the meaning of each point within the trajectories shown?

4. The qualitative results presented in Figure 4 are not particularly convincing. Are there any stronger qualitative evidence to support the advancements of the proposed AMD method?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies Multimodal Dataset Distillation (MDD), where the goal is to compress large image–text datasets into a small set of synthetic samples while retaining downstream retrieval performance. The authors identify a previously overlooked issue in MDD: existing approaches assume synchronous training dynamics for vision and language encoders, despite empirical evidence showing fundamentally asynchronous learning behaviors across modalities. The paper decouples trajectory matching between visual and language encoders, and uses semantic-aware prototype mining to initialize synthetic samples based on clustering in joint embedding space. Experiments on Flickr30k and COCO show consistent improvements over prior MDD methods, with negligible additional computational cost.

### Strengths
1. This paper identifies and validates the fundamental asynchronous nature of vision–language learning dynamics.
2. The proposed modification is minimal, which yields consistent improvements and results strong gains on Flickr30k and COCO dataset.

### Weaknesses
1. No visualization of learned prototype clusters or intuitive qualitative examples to further support semantic coverage claims.
2. Dynamic sampling strategy relies on MMD heuristic—although it works well, theoretical justification could be strengthened.

### Questions
1. Would asynchronous matching still work if using large-scale encoders (e.g., CLIP ViT-L/14)? Does the effect diminish with stronger pretrained models?
2. For prototype mining, did you try joint contrastive clustering or CLIP-score-based sampling instead of K-means?
3. Can you explain more intuitively how dynamic sampling boundaries are formed and whether simpler heuristics work equally well?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Asynchronous Matching with Dynamic Sampling (AMD), a new framework for Multimodal Dataset Distillation (MDD). The key idea is to address the misalignment between image and text modalities during distillation by decoupling their optimization trajectories, rather than synchronizing them as in prior work. AMD dynamically samples starting points for visual and textual expert trajectories based on their distinct convergence behaviors and incorporates a Semantics-Aware Prototype Mining module that clusters feature spaces to select representative initialization points. Together, these techniques improve the representativeness and coverage of synthetic data. The paper reports consistent performance gains on Flickr30k and COCO datasets with minimal computational overhead, showing AMD’s efficiency and scalability for multimodal learning.

### Strengths
- The paper identifies and empirically validates the inherent asynchronous learning dynamics between visual and textual modalities, a key factor overlooked by prior MDD works.

- The proposed Asynchronous Matching with Dynamic Sampling (AMD) framework introduces a principled way to decouple and dynamically align image-text trajectories, supported by a data-driven sampling strategy rather than fixed heuristics.

- The addition of the Semantics-Aware Prototype Mining module effectively enhances synthetic data diversity and representativeness by leveraging clustering in the joint semantic space. The method achieves improvements over baselines across multiple datasets and scales, demonstrating both robustness and negligible additional computational cost.

### Weaknesses
- The evaluation relies on relatively small-scale datasets (Flickr30k, COCO), which may limit conclusions about scalability to larger, more complex multimodal datasets.

- The qualitative analysis of synthetic data is minimal, providing limited insight into how AMD affects semantic fidelity or interpretability of the distilled samples. More discussion on this will further improve the paper quality.

- Broader downstream evaluations (e.g., captioning) will better demonstrate the general utility of the distilled data.

### Questions
How sensitive is the AMD framework to the choice of trajectory sampling ranges, and does the dynamic sampling strategy remain stable when applied to models with very different convergence behaviors?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors tackle the problem of Multimodal Dataset Distillation.

The authors first make two observations that are unique to the bimodal setting. First, text and image optimization trajectories are not synchronous and second, the optimization of the text and image encoders happen at different speeds. 

To address these observations, the authors propose an asynchronous matching with dynamic sampling framework.

### Strengths
- The paper is clearly organized, I especially enjoyed Observations 1 and 2 along with Figure 2. They give a nice background for why the authors are proposing a different solution. 
- Thank you for including visuals of the distilled data in Figure 4. 
- The baselines are fairly comprehensive.
- The authors give a substantial amount of background before diving into their method.
- I appreciate the upper bound analysis, although it would be better presented as a graph.

### Weaknesses
I do not see major weaknesses, some minor nit-picks:

- MMD is not defined in Eq 7
- Table 1: I'm confused why you have to use ones less sample for your method.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
