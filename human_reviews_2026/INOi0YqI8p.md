# Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Improving vision-language models (VLMs) in the post-training stage typically relies on supervised fine-tuning or reinforcement learning, methods that necessitate costly, human-annotated data.  While self-supervised techniques such as self-consistency have proven effective for enhancing reasoning capabilities, their application to perceptual domains such as image quality assessment (IQA) remains largely unexplored.  In this work, we introduce EvoQuality, a novel framework that enables a VLM to autonomously refine its quality perception capabilities without any ground-truth labels.  EvoQuality adapts the principle of self-consistency to the ranking-based nature of IQA.  It generates pseudo-labels by performing pairwise majority voting on the VLM's own outputs to establish a consensus on relative quality.  These pseudo-rankings are then formulated into a fidelity reward that guides the model's iterative evolution through group relative policy optimization (GRPO).  By iteratively leveraging its own predictions, EvoQuality progressively refines the VLM’s perceptual capability. Extensive experiments show that EvoQuality boosts the base VLM's zero-shot performance by 31.8% on PLCC across diverse IQA benchmarks. Remarkably, despite being entirely self-supervised, EvoQuality achieves performance that is competitive with, or even surpasses, state-of-the-art supervised VLM-based IQA models, outperforming these models on 5 out of 7 IQA benchmarks. Furthermore, the framework demonstrates significant flexibility, allowing it to be stacked with pre-trained IQA models to bolster generalization on unseen datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a self-supervised framework that enables a VLM to improve its IQA capabilities without human-annotated data autonomously. The method called EvoQuality uses a two-stage process: first, the VLM generates pseudo-labels by performing pairwise majority voting on its own outputs to determine relative image quality. These pseudo-rankings are then used as a fidelity reward to guide the model's evolution through reinforcement learning. This iterative self-improvement loop progressively refines the VLM's perceptual abilities. Experiments demonstrate that EvoQuality substantially enhances the base model's zero-shot performance, and surpasses state-of-the-art supervised IQA models.

### Strengths
1.	This work represents the first application of VLMs to the IQA task in a fully self-supervised manner, eliminating the need for human-annotated labels. This is a significant contribution, given the high cost and  challenges associated with collecting large-scale subjective quality data.
2.	The empirical evaluation is thorough, encompassing a diverse range of benchmark IQA datasets that include authentic, synthetic, and AI-generated distortions. Furthermore, the study includes comparisons against numerous recent state-of-the-art models, providing a robust assessment of the proposed method's performance . The finding that the self-supervised EvoQuality framework can achieve performance competitive with, or even superior to, supervised VLM-based approaches is particularly noteworthy.
3.	The manuscript is well-written, presenting the methodology, experiments, and results with a clear logical flow and well-defined structure.

### Weaknesses
1. The manuscript contains typographical errors that require correction. For instance, the text mentions "EvoRank" in the results discussion (Section 4.2), which appears to be inconsistent with the method name "EvoQuality" used throughout the rest of the paper. Careful proofreading is recommended to address such inconsistencies.

2. The set of competing models should be expanded to include relevant unsupervised learning IQA methods, such as CONTRIQUE and Re-IQA, to provide a more comprehensive comparison. Further, the related works can be refined by employing a higher-level view by including works like AIBench: Towards trustworthy evaluation under the 45°law, and Towards Versatile Multimedia Quality Assessment for Visual Communications.

3. The motivation behind using evolving-format IQA methods is not well explained. And from the experimental results, the self-evolving method achieves quite good performance results over the supervised methods, the reasons for which are not well clarified. Is the proposed method better at capturing more representative features? Or is the evolving format more suitable for the ranking task and why? The authors should have more discussion.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the EvoQuality framework, which enables vision-language models (VLMs) to autonomously improve their image quality assessment (IQA) capabilities without access to ground-truth labels. The method consists of a two-stage iterative loop (Section 3): an offline stage that generates pseudo-ranking labels via pairwise majority voting on the VLM’s own outputs (Section 3.1, Eq. 1), and an online stage that constructs fidelity rewards from these pseudo-labels and optimizes the model using GRPO (Section 3.2, Eqs. 3-5). Experimental results (Table 2) show that EvoQuality increases the weighted average PLCC of the base model Qwen2.5-VL-7B by 31.8%, and surpasses supervised SOTA models on 5 out of 7 IQA benchmarks (Table 3, Fig. 1b).

### Strengths
- The core innovation lies in successfully transferring the principle of self-consistency from discrete reasoning tasks to the continuous perceptual task of IQA (Section 1, paragraph 2). By adopting a pairwise voting and ranking paradigm (Section 3.1), the method circumvents reliance on costly human annotations.  
- The experimental design is rigorous and comprehensive, with zero-shot evaluation across eight diverse benchmarks (Section 4.1), spanning real distortions, synthetic distortions, and AI-generated distortions.  
- The ablation study in Table 4 clearly demonstrates the advantage of the ranking-based method (EvoQuality) over the direct regression variant (EvoEstimate), particularly in its sustained improvement on synthetic distortion datasets. The ablation on the number of candidate responses K (Fig. 3) further validates the necessity of the majority voting mechanism; performance is most stable at K=32.

### Weaknesses
- There is an internal inconsistency regarding the central claim of “progressive improvement.” Table 2 reports that the weighted average PLCC increases from 27.4% in Round 1 to 31.8% in Round 2, yet the detailed ablations in Table 4 show that EvoQuality exhibits performance degradation or stagnation on several datasets, e.g., KONIQ (0.840 → 0.835), AGIQA (0.839 → 0.831), and LIVEW (0.847 → 0.847), with consistent gains observed primarily on synthetic distortion datasets (KADID, PIPAL, TID2013, CSIQ). This contradiction is neither explained nor discussed in the main text.  
- The fundamental assumption, that the consensus produced by voting over the baseline VLM’s outputs constitutes sufficiently reliable pseudo-labels (Section 3.1), lacks theoretical or empirical support. The paper does not analyze whether, when the baseline model is of low quality or produces inconsistent judgments for certain image pairs, the method might amplify errors rather than improve performance.

### Questions
1. Table 4 shows that after the second iteration, the PLCC on KONIQ drops from 0.840 to 0.835 and on AGIQA from 0.839 to 0.831, which directly contradicts the discussion in Section 4.2 on “Progressive refinement with more iterations.” How do the authors explain this degradation? Does this suggest that introducing synthetic distortions in the second round (the 10 distortion types and 5 severity levels mentioned in Section 4.1) compromises the model’s generalization to real distortions?

2. The method relies on the quality of the baseline VLM’s voting consensus (Eq. 1). When the K votes for certain image pairs are nearly tied (e.g., 16:16), is the resulting pseudo-label `p*(x, y) = 0.5` reliable? Should the paper consider setting a confidence threshold to filter out low-confidence pseudo-labels?

3. In the design of Section 3, the offline stage uses a comparison-style prompt `c_compare`, whereas the online stage uses a score-estimation prompt `c_score` and converts scores to pairwise probabilities via the Thurstone model (Eq. 2). Why not train with direct pairwise comparisons in the online stage to maintain consistency with the offline stage? Does this switch in prompt types introduce additional uncertainty?

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
4

### Summary
This work introduces EvoQuality, a novel framework that enables a VLM to autonomously refine its quality perception capabilities without any ground-truth labels. EvoQuality generates pseudo-labels by performing pairwise majority voting on the VLM's own outputs to establish a consensus on relative quality. EvoQuality achieves performance that is competitive with, or even surpasses, state-of-the-art supervised VLM-based IQA models.

### Strengths
1. EvoQuality does not need any human-annotated quality scores to regress the quality scores, which is the most interesting point, though the voting method is quite simple and intuitive.
2. EvoQuality takes advantage of both the pre-trained VLM's ability (ie, ranking a pair of images) and the IQA theory (ie, fidelity loss), providing a new way to utilize VLMs in the IQA field.
3. In out-of-distribution datasets, EvoQuality achieves competitive or even better performance than supervised VLM-based IQA models, though in-distribution performance is missing.

### Weaknesses
1. The in-distribution performance is missing. In Table 3, the authors should also report the PLCC and SRCC results in the KONIQ dataset. Understandably, the performance will be lower than other methods, since EvoQuality does not use GT scores. However, reporting all results honestly is necessary.
2. Multi-dataset co-training experiments are missing. Q-Align and DeQA-Score both perform multi-dataset co-training experiments. Adding such experiments will make this work more solid.
3. The performance is not consistent with previous literature. The performance of training on KONIQ, then evaluating on other datasets, has been reported by several previous works, like Q-Align, DeQA-Score, and Q-Insight. All three works report similar, even almost the same results. However, the results in this work are not consistent with previous ones.
4. An important ablation is missing. The authors could utilize VLMs to rank image pairs, then calculate quality scores from all pair-wise comparison results. Then, the authors could use these scores to train previous IQA methods. With this experiment, we can fully understand whether the main performance gain of EvoQuality comes from (1) ranking accuracy of pretrained VLMs or (2) GRPO ranking training. (1) is the main contribution of this work, while (2) is the contribution of VisualQuality-R1.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes EvoQuality, a self-supervised framework for enhancing VLMs in opinion-unaware IQA. It features two key design elements. First, leveraging the LLM’s self-consistency to generate pseudo-labels indicating the relative quality of image pairs. Second, establishing a self-evolving mechanism that progressively refines the VLM’s perceptual capability by iteratively alternating between pseudo-label generation and model training. Experiments on seven IQA benchmarks demonstrate superior performance compared to SOTA methods.

### Strengths
1. The paper is clearly written and easy to follow.
2. The self-evolving framework offers some useful insights for opinion-unaware IQA.
3. The experimental results demonstrate good performance.

### Weaknesses
My main concern about this paper is its novelty. In my view, the work essentially builds upon VisualQuality-R1. Specifically, it replaces the ground-truth relative quality labels for image pairs (derived from MOS) with pseudo-labels generated by the VLM itself. The overall framework does not differ fundamentally from VisualQuality-R1. Moreover, the idea of generating ranking pseudo-labels—whether for image pairs or even image lists—and then training an IQA model using those pseudo-labels is also not new, e.g., see "Gu et al., No-Reference Image Quality Assessment with Reinforcement Recursive List-Wise Ranking, AAAI 2019."

### Questions
Experiments

1. In Table 3, EvoQuality outperforms VisualQuality-R1. The authors should provide further discussion of this observation. Specifically, do these two methods use the same training set and training configurations (e.g., sampling strategies for batch construction)? If not, the comparison may not be fair. If they do share identical settings (meaning the only difference is whether pseudo-labels or ground-truth labels are used), then the results suggest that training with pseudo-labels is actually more effective than training with true labels. The authors should explain the underlying reasons for this counterintuitive finding.

2. I suggest the authors provide an analysis of how \pi_\theta(⋅∣c_{compare}, x_i, x_j) evolves over training iterations. Specifically, do the K estimates of relative quality become increasingly stable as training progresses, i.e., consistently yielding K_x >> K_y or K_x << K_y? This would help demonstrate that the self-evolution mechanism is functioning as intended.

Method

1. Why isn’t p^\*(x, y) defined in a probabilistic form? Consider two cases: K_x=9, K_y=1 and K_x=6, K_y=4 . In both cases, p^*(x, y)=1, yet they reflect different levels of confidence that image x is of higher quality than image y.

2. A key advantage of opinion-unaware methods is their ability to eliminate reliance on MOS, which requires substantial human and financial resources to collect. This enables scaling up training with more (and more diverse) images. However, this advantage is not clearly demonstrated in the paper, especially when comparing the training scale with that of VisualQuality-R1. Moreover, the proposed method requires a relatively large value of K, which contradicts the goal of scalable training: as the training data volume grows, the computational cost of repeatedly invoking the large vision-language model becomes prohibitively high.

3. In lines 226-227, it appears that batches are constructed based on image pairs. Why is that? As shown in Eq. 3, the reward for an image is computed using all images paired with it in the set P. In other words, for a sampled pair (x_i, x_j), both x_i and x_j are processed independently. If so, why not simply sample individual images, i.e., a batch {x_m} (m=0,1,\ldots,M−1), instead of pre-forming pairs?

### Soundness
2

### Presentation
3

### Contribution
1
