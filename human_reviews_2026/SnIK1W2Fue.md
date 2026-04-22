# MultiTune: Phase-Aware Multi-Objective Optimization for Diffusion Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Diffusion models excel at basic text-to-image but struggle to align with specific objectives. While reinforcement learning offers a promising solution, single-reward setups often lead to overfitting. To this end, multi-objective optimization methods are proposed. However, such methods face challenges of goal conflicts, inflexible reward fusion, and low efficiency, hindering overall performance across diverse criteria.
To address these challenges, we propose MultiTune, a lightweight multi-objective framework tailored to the diffusion process. We decompose the optimization targets into Phase and Main objectives, where the former involves multiple phases of stepwise guidance and the latter ensures overall convergence.
We first introduce a phase-aware switching strategy that aligns with the structural-to-textural evolution in diffusion, enabling dynamic and decoupled scheduling of Phase Objectives. Then, we adaptively balance the Phase and Main Objectives based on variations in image quality for on-demand collaboration. 
Experiments demonstrate that MultiTune outperforms SOTA methods in aesthetics, semantics, details, and style, achieving leading performance across five quantitative metrics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel multi-objective RL framework for text-to-image diffusion models by introducing an additional phase objective that gradually changes throughout the generation process. The proposed phase objective serve to mitigate the sparse reward problem, where the main reward can only be evaluated on the fully generated image. The authors conducted extensive experiments and demonstrated that the proposed method outperforms existing RL baselines.

### Strengths
1. The presentation is clear and easy to understand. The Phase reward is well-motivated and different phases align with intuitive understanding about the diffusion generation process.
2. The experiments are comprehensive, incorporating a wide range of metrics (CLIP, Pickscore) and base models (SDv1.5, SD2), demonstrating that the proposed method is universally applicable. The author also provided subjective evaluations using human and VLM judge, further strengthening their results.
3. The author provided a thorough ablation studies on various design choices.

### Weaknesses
1. The author experimented mostly on SD-series U-Net, it is unclear if the proposed method works for more recent DiT models based on rectified flow formulation, such as SD3, Sana, Flux, etc.
2.  Given 1, the result in table 5 is especially concerning, as it raises the issue of the scalability of the proposed method. While SD-XL is a stronger base model, the performance after RL fine-tuning is actually worse than SDv1.4. The author should include a row of base model's performance in this table so it is easier to see the improvements. I imagine the improvements with respect to base model will be smaller? If that is the case, the author should provide additional discussion.

### Questions
1. What  exactly are the phase reward models? In Appendix C, the author used ambiguous terms like " introduce a reward function (e.g., CLIP)", "considering texture and aesthetic preferences (e.g., PickScore)". Is the reward model just CLIP and PickScore? Are other models also used? The authors should provide more concrete details for better reproducibility
2. What is "see Section X.X " in page 15 L 806
3. Figure 13 on Page 19 appears to be broken in PDF

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
This paper addresses the challenge of fine-tuning text-to-image diffusion models to satisfy multiple objectives rather than optimizing a single reward. To this end, the authors propose MultiTune, which introduces three key components: a phase-aware switching strategy, adaptive coordination of objectives, and efficiency-aware training optimization.

### Strengths
1. The idea of decomposing the denoising process into intuitive phases and tying objectives to those phases is meaningful.
2. The experimental evaluation is extensive, covering multiple model backbones.

### Weaknesses
1. The paper employs the “Simple-animals” dataset (45 classes for training, 398 for testing). To better support claims of generality, it would be valuable to include larger and more diverse prompt sets—such as GenEval [1] or datasets spanning multiple domains.
2. More details are needed on computational cost. Specifically, what is the additional overhead (in memory and compute) introduced by the phase-aware switching and dynamic balancing mechanisms compared to simpler baselines?

### Questions
1. How are the structural and textural scores in Figure 1 computed? What model architectures are used for these evaluations?
2. How exactly is the denoising trajectory divided into $P_{Structural}$, $P_{Textural}$ and $P_{Marginal}$. How are the corresponding timesteps determined?
3. There are a few textual inconsistencies—for example, line 806 references “Section X.X.” Could the authors clarify which section this refers to?

### Soundness
3

### Presentation
3

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
This paper proposes MultiTune, a reinforcement learning framework for text-to-image diffusion models that dynamically switches optimization objectives during the denoising process.
The model separates diffusion steps into three stages — structural, textural, and marginal — and adaptively balances intrinsic and extrinsic rewards.
Experiments across Stable Diffusion backbones show improvements on metrics such as AES, CLIP, and PickScore.

### Strengths
1. The paper focuses on the characteristics of the diffusion process  — namely, the generation proceeds from global structures to fine details — and proposes an efficient learning method for multi-objective preference optimization.

2. It conducts comprehensive experiments using multiple baselines and evaluation metrics, demonstrating performance improvements across all metrics.

### Weaknesses
1. It is unclear from the main text whether Equation (4) truly addresses a multi-objective task.

2. The proposed method does not guarantee functionality under arbitrary combinations of preference objectives. In particular, the structure formation phase is fixed to use CLIP as the guiding signal. Therefore, in terms of exploring the trade-offs inherent to true multi-objective preference optimization, the contribution appears somewhat limited.

### Questions
1. Could you tell me why CLIP and ES are used solely as feedback signals for exploration rather than being directly optimized? Equation (4) does not appear to represent multi-objective preference learning?

2. Why did the authors choose to fix the extrinsic reward to AES instead of dynamically optimizing CLIP or ES depending on the phase change?

3. Have the authors considered alternatives to using CLIP as the guiding signal during the structure formation phase?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The author divides the diffusion process into two objectives dynamically: Phase and Main Objective. The author claimed it reached SOTA performance.

### Strengths
## Presentation: ~5th percentile
It can be observed that the grammar and paragraph organisation are nearly flawless; however, anything beyond a single paragraph becomes nonsensical from the perspective of a reader (see below).

## Soundness: 10th~50th percentile
The paper uses experimental results to support its method, but I find it challenging to verify the details (see below).

## Contribution: 5~25th percentile

The divisions of denoising steps may seem arbitrary in theory, but they are reasonable in practice.

## Note:

I hope the AC is aware that the rating is calibrated using estimation of percentiles to reduce evaluation noise effectively.
The rating is simply the mean of the three aspects.

### Weaknesses
Based on the writing, this paper should not be accepted. 

## Presentation

The writing is almost unintelligible. By the time I finished your methodology:

1. I have a limited understanding of the Phase/Main objective beyond the term you coined, including how it is implemented and the rationale for choosing the key phase transition step $t$ (I assume it’s Eq (3) with 80% confidence, but see the issues below). I was misled by Figure 2 as the caption does not present the details of Figure 2.
2. Is your approach an inference-time scaling method, or a training method with a different objective? I’m about 80% confident it’s the latter, but you shouldn't make me guess.
3. Are you using DDIM/DDPM/score-based SDE to model the diffusion model? I assume it’s DDIM, based on $\hat{x}_0(x_t)$ on Line 173 with 60% confidence. **This is a fundamental detail that should be presented to every reader.**

These details are either missing altogether or scattered across the text rather than presented coherently. As a result, the reading experience is terrible as I have to keep half-formed guesses in mind just to follow along.



### Equation 3

I believe Equation (3) defines the criterion for a phase transition, though it takes more effort to discern this compared to other papers. I still raise some issues.

1. You should group this equation (on page 4) with Figure 1 (on page 2) if Figure 1 is the evidence of this choice of design. 
2. Why does the definition on Line 193 call variance (and why does it have no equation labelling?) What does the arrow mean in Equation (3)? What is the indicator for phase transition used for?





## Contribution

### Reproducibility

A lot of important details are oversimplified, especially for those parameter settings. I don’t think a reader can reproduce your result by reading this paper.

### Novelty

The observation that diffusion models denoising in a hierarchical manner is not new, especially since this has already been stated in several theoretical papers like score-based SDE (Yang et. al), EDM (Karras et. al.)

## Soundness

I want to evaluate the soundness based on the implementation details and the methodology. However, my limited understanding of the methodology made it difficult for me to make an accurate judgement, so I can only give my lower bound of my estimation. To be fair, I will lower my confidence to 3. Admittedly, the rather unpleasant experience I had while reading this paper may have influenced my assessment in this respect, although I recognise that I should have evaluated it independently.



The final rating was simply a linear transform of the average of the estimated percentile.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1
