# PEA-DPO: Perception-Enhanced Alignment Direct Preference Optimization for MLLMs Alignment

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Direct Preference Optimization (DPO) has emerged as an effective approach for aligning large language models (LLMs) with human preferences. However, its adaptation to multimodal settings remains unexplored. Through representational analysis, we identify a key limitation in multimodal preference optimization, which we term \textbf{visual insensitivity}: models often fail to distinguish between images and those with critical visual context removed. Our theoretical analysis further uncovers two manifestations of this problem, namely \textbf{across-image insensitivity} and \textbf{within-image insensitivity}. To address these challenges, we propose Perception-Enhanced Alignment DPO (PEA-DPO), a framework for multimodal LLMs alignment, which explicitly leverages visual preference signals to overcome visual insensitivity. We empirically demonstrate that PEA-DPO enhances sensitivity to visual context while preserving the language modeling capacity of the base model. We empirically evaluate PEA-DPO across three hallucination benchmarks using multimodal LLMs (MLLMs) of varying scales. Our results demonstrate that PEA-DPO effectively mitigates visual insensitivity, achieves stronger multimodal alignment, and substantially reduces hallucinations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies a key failure mode in multimodal DPO: visual insensitivity, with across-image and within-image manifestations. The authors propose PEA-DPO, a dual-preference objective that optimizes both response-quality preferences and visual-context preferences. The visual-context signal is built by creating masked variants of the original image, selecting the most semantics-damaging one via CLIP similarity, and encouraging the model to prefer responses conditioned on the full image. A reference-free, margin-based ReLU variant filters trivial pairs and reduces compute. On LLaVA-1.5 with 7B and 13B backbones, the method substantially reduces hallucinations and delivers strong results on MMHal Bench, Object HalBench, and AMBER, with a small tradeoff in coverage and a slight regression on MMMU validation and test.

### Strengths
1. The paper clearly defines visual insensitivity, decomposes preferences into textual and visual components, and articulates both across-image and within-image failure modes.
2. The reference-free ReLU-margin formulation removes dependence on a reference model, filters trivial pairs, and improves training efficiency and robustness.
3. The experiments include clear ablations of components and the weighting hyperparameter,  along with scaling results that support the design choices and show effectiveness even with modest amounts of training data.
4. The method achieves large and consistent reductions in multimodal hallucinations on MMHal-Bench, Object HalBench, and AMBER for both 7B and 13B LLaVA backbones, while largely maintaining general capability with a small coverage tradeoff and a slight regression on MMMU.

### Weaknesses
1. The masking strategy relies on several design choices, including the masking ratio, the number of candidate masks, and whether to use CLIP cosine similarity. However, the final selections are not supported by a systematic study.
2. The comparability to some strong proprietary systems and newer alignment methods is imperfect due to differences in base models, data, and evaluation settings, which weakens the strength of SOTA claims.
3. The CLIP-guided selection of the “most damaging” mask requires generating and scoring many candidates, which introduces non-trivial data-generation overhead and complicates scaling.

### Questions
1. Could VPO degrade language-only capabilities or long-context reasoning? Any broader evals beyond hallucination (e.g., MMMU shown, but more detail would help).
2. How does PEA-DPO perform with other models? Please report results and adaptation details to evaluate generalization.
3. Please provide a detailed analysis of the MMMU regression to clarify whether the decline reflects increased conservativeness or a genuine loss of capability.
4. As a diagnostic follow up to item 3, please evaluate PEA-DPO on MMStar to test performance under genuinely vision-dependent settings and to reduce text-only shortcuts, and also report calibration or selectivity metrics such as risk and coverage curves, Accuracy at Coverage with an abstain option, and ECE or Brier scores to separate lucky guesses from grounded answers?

### Soundness
3

### Presentation
3

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
The paper investigates the issue of "visual insensitivity" in Multimodal Large Language Models (MLLMs) during Direct Preference Optimization (DPO) alignment. The authors point out that MLLMs suffer from two specific problems: across-image insensitivity and within-image insensitivity. Through theoretical analysis, they demonstrate that visual insensitivity is a major cause of model hallucination. Additionally, the authors conduct a series of experiments to validate the effectiveness of their proposed method.

### Strengths
1. The overall logic of the paper is sound. The authors theoretically analyze the phenomena of across-image insensitivity and within-image insensitivity in multimodal DPO, revealing that a significant part of the factors affecting DPO performance stems from visual insensitivity.

2. The experiments in the paper are sufficiently comprehensive and the presentation is relatively easy to follow.

### Weaknesses
The main weakness of this paper is the lack of substantive innovation. Although the authors provide a theoretical explanation of across-image insensitivity and within-image insensitivity, I fail to see a strong connection between these phenomena and the proposed method. The approach appears to be merely a combination of standard DPO and DPO with reject images. The statement "In summary, both types of issues arise from visual insensitivity" alone is insufficient to demonstrate the relevance of the proposed method to the two types of insensitivity. If the authors can adequately address this issue, I would consider raising my score.

### Questions
1. Please refer to the weaknesses outlined above.

2. The construction method for reject images proposed by the authors lacks innovation and relies heavily on the performance of CLIP. Would the effectiveness of the proposed approach be improved if image editing datasets were used to construct chosen-reject pairs?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces Perception-Enhanced Alignment DPO (PEA-DPO). The proposed method uses both image-level and response-level feedback. The authors show that PEA-DPO reduces hallucinations consistently for multiple base models and model scales.

### Strengths
* The concepts of across-image and within-image insensitivity are well motivated and the mathematical background is both intuitive and presented in an accessible manner.
* The method is described clearly and appears easily reproducible.
* The reported results on hallucination benchmarks are very strong.

### Weaknesses
* In table 1, the main results, some numbers and details appear inconsistent with previously published results. For example, POVID reports an MMHalBench number of 2.69 [3], but the table shows 2.08. The table lists LLaVA-RLHF as building on LLaVA-v1.5-7B but the paper describes building on LLaVA with a different SFT recipe [4]. These are some initial discrepancies that I noticed, but due to time constraints I am unable to review every reported result. Instead, I would urge the authors to carefully review the reported results and details shown in table 1 for a fair comparison of this work’s contributions as part of the rebuttal period.
* The manuscript primarily reports hallucination metrics, with only one metric, MMMU, to speak to the model’s task performance post-alignment. On MMMU, the method reports some regressions across both model scales under test. Evaluation on quality-focused benchmarks such as MMStar, MME, AI2D, or MMVet would give a better understanding on the impact on helpfulness.
* As described in [5], CHAIR scores suffer from not penalizing shorter responses to avoid hallucinations. The paper reports a reduction in coverage as reported in AMBER, which is also briefly discussed in 5.2. While the reduction in CHAIR metrics is very strong for this method, without reporting recall, it is hard to interpret whether this is strictly due to a reduction of hallucinations or perhaps more trivially by producing shorter / potentially less informative responses, also related to the previous point.
* In table 1, the commercial baseline reported is GPT-4V. A more recent frontier model may better put the reported results in perspective.
* The manuscript could more strongly discuss the natural connection to mDPO [2]. For example, the proposed VPO objective is identical to the “conditional preference optimization” proposed in mDPO. and the $L_{PEA-DPO}$ is identical to $L_{mDPO}$ without their $L_{AncPO}$ anchoring term. Similarly, the ablation on cropping in 5.5 makes the method more similar to mDPO, yet results remain much stronger than mDPO, which may also benefit from some discussion.


[1] Tong, Shengbang, et al. "Eyes wide shut? exploring the visual shortcomings of multimodal llms." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[2] Wang, Fei, et al. "mdpo: Conditional preference optimization for multimodal large language models." arXiv preprint arXiv:2406.11839 (2024).

[3] Zhou, Yiyang, et al. "Aligning modalities in vision large language models via preference fine-tuning." arXiv preprint arXiv:2402.11411 (2024).

[4] Sun, Zhiqing, et al. "Aligning large multimodal models with factually augmented rlhf." arXiv preprint arXiv:2309.14525 (2023).

[5] Amirloo, Elmira, et al. "Understanding alignment in multimodal llms: A comprehensive study." arXiv preprint arXiv:2407.02477 (2024).

[6] Chen, Lin, et al. "Are we on the right way for evaluating large vision-language models?." Advances in Neural Information Processing Systems 37 (2024): 27056-27087.

### Questions
* The construction of perception-enhanced preference data is using CLIP to identify when crucial detail has been removed, though many popular multimodal models are trained with encoders following CLIP-style pre-training (or similar) so perhaps corruptions that lead to difference in CLIP representations may be most likely to lead to perceptible differences for the multimodal LLM under alignment. [1] shows that different encoders can have different “blind spots”. Have you considered using a different encoder (DINO?), or perhaps an ensemble to determine when critical visual context has been removed?

### Soundness
3

### Presentation
2

### Contribution
3
