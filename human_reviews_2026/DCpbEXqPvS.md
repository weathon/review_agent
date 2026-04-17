# SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition

- Decision: Accept (Poster)
- Scores: 8, 2, 4, 6

## Abstract
Visual Place Recognition (VPR) requires robust retrieval of geotagged images despite large appearance, viewpoint, and environmental variation. 
Prior methods focus on descriptor fine-tuning or fixed sampling strategies yet neglect the dynamic interplay between spatial context and visual similarity during training.
We present SAGE ($\underline{S}$patial-visual $\underline{A}$daptive $\underline{G}$raph $\underline{E}$xploration), a unified training pipeline that enhances granular spatial-visual discrimination by jointly improving local feature aggregation, organize samples during training, and hard sample mining. 
We introduce a lightweight Soft Probing module that learns residual weights from training data for patch descriptors before bilinear aggregation, boosting distinctive local cues.
During training we reconstruct an online geo-visual graph that fuses geographic proximity and current visual similarity so that candidate neighborhoods reflect the evolving embedding landscape. 
To concentrate learning on the most informative place neighborhoods, we seed clusters from high-affinity anchors and iteratively expand them with a greedy weighted clique expansion sampler.
Implemented with a frozen DINOv2 backbone and parameter-efficient fine-tuning, SAGE achieves SOTA across eight benchmarks. Notably, our method obtains 100% Recall@10 on SPED only using 4096D global descriptors.
The code and model are available at https://github.com/chenshunpeng/SAGE.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes SAGE (Spatial-Visual Adaptive Graph Exploration), a novel training framework for Visual Place Recognition (VPR) that dynamically integrates spatial and visual cues through online graph construction and adaptive sampling.
Unlike prior methods that use static sampling or fixed descriptor aggregation, SAGE introduces:
1. Soft Probing (SoftP): a lightweight, residual reweighting module that amplifies discriminative local patches before aggregation;
2. InteractHead: a cross-image attention block improving descriptor coherence across views;
3. Dynamic Geo-Visual Graph (OGC): rebuilt each epoch to reflect evolving embeddings;
4. Greedy Weighted Sampling (GWS): an adaptive clique-expansion strategy for efficient hard sample mining.

### Strengths
1. Novel Training Paradigm: The idea of reconstructing an online geo-visual graph per epoch is elegant and well-justified. It aligns with the dynamic nature of embedding spaces and resolves the “stale hard samples” issue common in previous static mining methods.
2. Effective Feature Enhancement: The SoftP module is a simple yet powerful mechanism that improves local discriminability with negligible computational overhead. The InteractHead effectively models cross-view consistency, leading to more coherent global descriptors.
3. Comprehensive Empirical Validation: Results across eight datasets convincingly show SAGE’s superiority over strong baselines such as EMVP (NeurIPS’24), FoL (AAAI’25), and SALAD-CM (ECCV’24). The method also performs remarkably well with compact descriptors (4096-D), highlighting scalability.
4. Parameter Efficiency: Freezing DINOv2 while introducing only lightweight trainable modules (SoftP, DPN, InteractHead) demonstrates impressive efficiency—achieving SOTA results with fewer parameters than most competing methods.
5. Clear Presentation and Strong Technical Depth: The paper is well-written, logically organized, and provides strong ablation and visualization support (e.g., t-SNE clustering, SoftP heatmaps, AID analysis).

### Weaknesses
Although efficient in parameters, the online graph construction and clique expansion may increase runtime. A small ablation or runtime comparison would strengthen the claim of overall efficiency.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a new model for Visual Place Recognition (VPR), which adds many different improvements on top of existing methods. Results show consistent improvements over existing methods, although it is not clear if the evaluation settings are fair towards other VPR methods.

### Strengths
Good results
Thorough evaluation against other methods

### Weaknesses
1. Some text is obviously hallucinated. SPED stands for Specific PlacEs Dataset, not Street View Place Recognition Dataset (line 894). Tokyo247 is called 24/7 because it has night imagery (24 hours), not because it contains approximately 247 km, as written in line 909.

2. There are hallucination-like unsubstantiated claims throughout the paper, some of which are quite arguable at best, like
- line 052: "However, approaches driven by VFMs or learned queries often demand careful regularization and substantial resources in terms of computation and data to ensure stable fine-tuning." What is this referring to? I'm not aware of any paper supporting this claim, and models like SALAD and BoQ (DINOv2-based) do not demand careful regularization and substantial resources, but quite the opposite.
- line 078: "This renders them less effective in scenarios that demand awareness of spatial context and the evolving appearance of hard samples". This is simply false, methods based on VFMs are more effective in every scenario, in fact SOTA VPR methods since 2024 all rely on VFMs, as proved by literally all methods on table 3.
- line 320: "Since the EMVP repository is actively maintained". EMVP codebase is unavailable, and the link indicated by EMVP authors returns 404 (https://github.com/vincentqqb/EMVP)

3. The method is overall quite hard to understand, which makes it difficult to evaluate its soundness and novelty. Figure 2 does not help in understanding. For example, line 241 suggests that B images form a sequence, and the sequence is processed as one, meaning that the descriptor of an image are influence by the descriptors of other images in the batch. It is not clear if this happening also at test time? If yes, this is a significant (unfair) advantage against other baselines, because it relies on the locality of queries (as consecutive queries in many datasets come from the same areas), whereas other methods aim to localize queries individually.

4. The paper would be significantly stronger if results were also shown on SF-XL, which is the only city-wide VPR dataset to date. Also, SF-XL (or even SF-small) does not have consecutive / nearby queries so it would lift doubts about evaluation fairness.

5. Line 112 is contradictory: what does it mean "tuning a frozen DINOv2 backbone"? Is it tuned or frozen? Later on PEFT is mentioned making it look like the actual DINOv2 backbone is frozen but LoRA is used

6. PEFT is not cited anywhere. PEFT should be cited, as well as the LoRA paper if a LoRA is used.

7. The OGC seems very similar to SALAD-CM graph, with the only difference that SALAD-CM creates it only once while OGC is created multiple times. Are there any other differences?

8. The ablation is incomplete. Given that the paper mentions that online graph creation is necessary, while previous methods compute it offline (like SALAD-CM using SALAD to compute graph only once at the beginning) the ablation should present results with offline graph creation as well.

9. The value of the threshold τ (line 259) is never defined.

10. The paper claims that Dynamic Geo-Visual Graph Mining leads to faster convergence (line 107 and line 293), but this is not proven nor discussed anywhere.

11. Eynsham is a dataset created in the paper "Deep Visual Geo-localization Benchmark", which must be cited, using raw data by Cummins & Newman. Also, Eynsham uses a 25 threshold, not 5 meters (line 315).

12. OGC is sometimes defined as Online Graph Creation and sometimes as Online Graph Construction (minor weakness).

### Questions
See weaknesses above

### Soundness
2

### Presentation
1

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
This paper introduces SAGE , a new framework for VPR that addresses several challenges in large-scale geotagged image retrieval. VPR is essential for tasks such as autonomous robot navigation, loop closure detection in autonomous driving, and map construction. Previous methods have focused on fine-tuning descriptors or static sampling strategies but have failed to account for the dynamic interplay between spatial context and visual similarity during training. SAGE improves VPR performance by dynamically constructing an online geo-visual graph, enhancing local feature aggregation, and using hard sample mining.

### Strengths
1. SAGE introduces a dynamic approach to hard sample mining and geo-visual graph construction, addressing limitations in previous methods.
2. SAGE achieves top performance on multiple VPR benchmarks, demonstrating its effectiveness across various environmental conditions, such as viewpoint shifts, weather changes, and lighting variations.

### Weaknesses
1. The writing is not clear enough and fails to highlight the key contributions of this paper.
2. While the paper introduces the SAGE framework and its various components (such as SoftP, InteractHead, and the dynamic geo-visual graph mining), the justification for why these specific techniques were chosen over others is not sufficiently detailed. For instance, the paper does not provide a detailed comparison of SAGE’s components with other potential alternatives that might achieve similar outcomes.
3. The paper may benefit from a deeper analysis of why SoftP, InteractHead, and other components were selected. A discussion on related work and the performance of other possible methods in a similar context could enhance the novelty of the approach. For example, why are these methods preferable over existing techniques like spatial attention, or other forms of feature aggregation?
4. There is a mention of a performance drop in highly dynamic environments, but this is not sufficiently examined.
5. Please include more comparisons with the latest state-of-the-art methods.

### Questions
1. The writing is not clear enough and fails to highlight the key contributions of this paper.
2. While the paper introduces the SAGE framework and its various components (such as SoftP, InteractHead, and the dynamic geo-visual graph mining), the justification for why these specific techniques were chosen over others is not sufficiently detailed. For instance, the paper does not provide a detailed comparison of SAGE’s components with other potential alternatives that might achieve similar outcomes.
3. The paper may benefit from a deeper analysis of why SoftP, InteractHead, and other components were selected. A discussion on related work and the performance of other possible methods in a similar context could enhance the novelty of the approach. For example, why are these methods preferable over existing techniques like spatial attention, or other forms of feature aggregation?
4. There is a mention of a performance drop in highly dynamic environments, but this is not sufficiently examined.
5. Please include more comparisons with the latest state-of-the-art methods.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes SAGE, a unified training framework for Visual Place Recognition that enhances spatial-visual discrimination through dynamic graph-based sample mining and lightweight feature enhancement modules. The method introduces Soft Probing for adaptive patch weighting and an online geo-visual graph to continuously update hard samples during training. Extensive experiments demonstrate that SAGE achieves state-of-the-art performance across eight challenging VPR benchmarks with high parameter efficiency, using a frozen DINOv2 backbone and parameter-efficient fine-tuning.

### Strengths
1. The experimental results are exceptionally strong, demonstrating state-of-the-art or highly competitive performance across a diverse set of eight benchmarks, including challenging datasets like SPED, Nordland, and AmsterTime that test robustness to seasonal, illumination, viewpoint, and long-term temporal changes.

2. The proposed dynamic geo-visual graph mining strategy is a principled and well-motivated approach to hard sample mining. By reconstructing the graph each epoch, it effectively addresses a key limitation of static sampling strategies and ensures the training focuses on currently relevant challenging examples.

3. The Soft Probing module and InteractHead are lightweight yet highly effective, enhancing local feature discriminability and cross-image coherence with a negligible number of added parameters, showcasing an excellent performance-to-cost design.

4. The framework is highly efficient, achieving its top-tier performance by freezing the DINOv2 backbone and employing parameter-efficient fine-tuning, resulting in a significantly lower number of trainable parameters compared to many existing methods, which is a significant practical advantage.

### Weaknesses
1. While the per-epoch graph reconstruction overhead is noted to be marginal, it could become a non-trivial computational bottleneck when scaling to extremely large-scale datasets, and a more in-depth scalability analysis would be beneficial.
2. This paper only provides results on the MSLS-val dataset and lacks results on the MSLS-challenge dataset. Since the labels for MSLS-challenge are not publicly available and online testing is required, results on this dataset would be more convincing.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
