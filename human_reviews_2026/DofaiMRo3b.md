# Unleashing 2D Rewards for Human Preference Aligned Text-to-3D Generation via Preference Score Distillation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Human preference alignment presents a critical yet underexplored challenge for diffusion models in text-to-3D generation. Existing solutions typically require task-specific fine-tuning, posing significant hurdles in data-scarce 3D domains. To address this, we propose Preference Score Distillation (PSD), an optimization-based framework that leverages pretrained 2D reward models for human-aligned text-to-3D synthesis without 3D training data. Our key insight stems from the incompatibility of pixel-level gradients: due to the absence of noisy samples during reward model training, direct application of 2D reward gradients disturbs the denoising process. 
Noticing that similar issue occurs in the naive classifier guidance in conditioned diffusion models, we fundamentally rethink preference alignment as a classifier-free guidance (CFG)-style mechanism through our implicit reward model. Furthermore, recognizing that frozen pretrained diffusion models constrain performance, we introduce an adaptive strategy to co-optimize preference scores and negative text embeddings. By incorporating CFG during optimization, online refinement of negative text embeddings dynamically enhances alignment. To our knowledge, we are the first to bridge human preference alignment with CFG theory under score distillation framework. Experiments demonstrate the superiority of PSD in aesthetic metrics, seamless integration with diverse pipelines, and strong extensibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Preference Score Distillation that optimizes 3D objects with a score distillation pipeline. 
The main contribution is adding the preference score guidance term in the optimization objective using a reward function trained on 2D images, thereby increasing the quality of the underlying 3D representation judged by the used reward function. It shows state of the art performances on the Eval3D benchmark, demonstrating the effectiveness of the proposed method.

### Strengths
- Experiments show solid improvements over the baseline methods in all aspects considered in the paper. Also, while the paper only leverages 2D priors and reward functions, it performs on par or better than methods using 3D priors.
- The use of negative embedding optimization in the context of the score distillation pipeline is new, and its efficacy is demonstrated well in an ablation study.

### Weaknesses
- The presentation of the paper is not clear
    - Equation 4: the whole expectation is taken with respect to $x_t \sim p_{\phi}(x_t | y)$. I believe this needs to be taken with respect to $x_t \sim q_{\theta}$
    - Equation 7: It is not clear what this expression entails - how can $x^w_t$ and $x^l_t$ be defined from $x_t$? 
- While the exact formulation is different, the main preference alignment guidance resembles that of DreamDPO, and the paper proposes to apply the guidance to existing score guidance terms. The significance of the proposed method appears incremental.
- The paper writing can be improved. There are several typos, including
    - line 404: ‘fin tuned’
    - Table 3: w/n*
    - Line 464: suppress

### Questions
The main questions are the first two points in the weaknesses section. 
It would be great if they can be addressed in the rebuttal.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Preference Score Distillation (PSD), a new optimization framework for aligning text-to-3D diffusion models with human preferences without requiring 3D training data. PSD leverages pretrained 2D reward models and reformulates preference alignment as a classifier-free guidance (CFG)-style process to avoid instability from pixel-level gradients. It further co-optimizes preference scores and negative text embeddings for adaptive alignment. Experiments show improved aesthetics, flexibility, and integration across text-to-3D pipelines.

### Strengths
1) The paper is clearly written and easy to follow, with well-organized motivation, methodology and results.
2) The proposed Preference Score Distillation (PSD) is a creative and technically sound idea that connects preference alignment with classifier-free guidance (CFG).
3) Experimental results show promising improvements in human preference alignment and visual quality across various text-to-3D settings.

### Weaknesses
1) Figure 1 introduces Trellis as a baseline, but it is not explained or included in later experiments, raising questions about consistency.
2) Figures 3 and 5 compare PSD alternately with RichDreamer and DreamDPO, but not both across all samples, limiting fairness and clarity of comparison.
3) The paper lacks discussion on computational cost or runtime efficiency of PSD, which is important given the added optimization and CFG-style refinement.
4) I may not familiar with it, but negative embedding optimization seems not clearly explained (technically), make it a bit difficult to assess its merit.

### Questions
Refer to the weakness please.

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
3

### Summary
This work proposes Preference Score Distillation (PSD), an optimization-based framework for human preference alignment in text-to-3D generation that leverages pretrained 2D reward models without requiring 3D training data or task-specific fine-tuning. This work focuses on the incompatibility of pixel-level gradients from reward models by reformulating preference alignment as a classifier-free guidance (CFG) mechanism through an implicit reward model, similar to how CFG handles classifier guidance in diffusion models. The method also introduces an adaptive strategy that co-optimizes preference scores and negative text embeddings during generation to dynamically enhance alignment.

### Strengths
- The paper is well-written and easy to follow.
- The work is well-motivated, with a clear objective of making 3D Score Distillation adhere more closely to the text prompt, which is a significant issue in the field of SDS-based text-to-3D generation. Incorporation of RLHF as CFG guidance can be seen as an effective solution to this issue, as demonstrated in this paper.
- The core methodology/theory of reformulating preference alignment as CFG-style guidance through implicit reward modeling is elegant and well-defined. Formalizing the human preference as a binary variable in the likeness of DPO and incorporating it into CFG-like guidance is theoretically sound as well as intuitive. The derivation connecting preference optimization to guidance mechanisms (Equations 6-10) provides a principled framework that unifies RLHF with score distillation, which is a meaningful contribution to the field.
- The experimental results show effective performance and are well-organized, demonstrating the effectiveness of this method.

### Weaknesses
- The approach requires running the reward model and diffusion model multiple times per optimization step, which may significantly increase computational costs: What is the computational overhead that is induced from this process? Please elaborate.
- The adaptive negative embedding optimization strategy (Section 3.3, Equation 13) appears somewhat ad-hoc. While Figure 4 shows it improves reward scores, it seems that the paper does not provide sufficient theoretical justification for why optimizing negative embeddings should improve preference alignment, or how this relates to the implicit reward formulation in Section 3.1. The claim that this "achieves the effect of updating pretrained diffusion parameters" needs more rigorous support: please elaborate further on this aspect.
- Grammatical mistake in the title of Sec 3.3: I suggest "Adaptively update of preference score" -> "Adaptive update of preference score".

### Questions
Please see the Weaknesses section.

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
3

### Summary
The paper introduces Preference Score Distillation (PSD), a method for aligning text-to-3D generative models with human aesthetic preference without training a dedicated 3D reward model. Direct backpropagation from a 2D reward model can destabilize 3D generation because of the absence of noisy supervision during the reward model's training. To circumvent this issue, the authors reinterpret preference alignment as a classifier-free guidance-style mechanism. During optimization, PSD constructs on-the-fly "win vs. lose" noisy image pairs. It evaluates preference using a pretrained 2D reward model (e.g., HPSv2.1) and then adjusts the denoising trajectory to favor the preferred samples. A key feature is that the method also adapts negative text embeddings to reinforce alignment without needing to modify the core diffusion model itself. Experiments across multiple 3D pipelines, including MVDream, NeRF, and DMTet, demonstrate substantial improvements in aesthetic and preference metrics, alongside better qualitative fidelity.

### Strengths
- Outperforms DreamReward and other baselines on aesthetic / preference metrics and visual results.
- Works across NeRF, DMTet, and 2D-to-3D diffusion-distillation pipelines without architecture changes.

### Weaknesses
- **Motivation Overclaim: Lack of Empirical Support for "3D Preference Scarcity" Argument**
The paper repeatedly motivates its method by arguing that 3D human-preference data is scarce and expensive to collect. However, no empirical evidence, cost analysis, or small-scale feasibility study is provided to support this central claim. It remains unclear whether small-scale 3D preference data collection is genuinely infeasible or simply unexplored.

- **No Validation That 2D Rewards Transfer to 3D Preference Space**
The method assumes that 2D human-preference reward models, such as HPSv2.1, sufficiently reflect 3D aesthetics and human preference. Yet, the failure modes between 2D and 3D generation differ significantly. Examples include multi-view inconsistency and Janus artifacts. The paper does not evaluate whether 2D reward scores correlate with true 3D user preference or whether 2D models exhibit bias when used for 3D alignment. Specifically, 2D reward models are prone to scoring highly on one attractive view, even if the object exhibits severe multi-face (Janus) or rear-view inconsistency, because they fundamentally lack multi-view consistency checks.

- **Missing Small-Scale Control Experiment**
A natural ablation is missing. Specifically, the paper should include training or evaluating on a small amount of real 3D preference labels to empirically compare the two approaches: 2D reward only versus a Hybrid 2D + a few 3D preference signals. Without this, the claim that 3D reward supervision is impractical or inferior remains speculative.

- **Unanalyzed Failure Cases**
The paper does not provide qualitative or quantitative analysis of scenarios where 2D preference alignment fails in 3D. This is particularly relevant for issues like rear-view degradation, multi-face (Janus) artifacts, geometry-style trade-offs, and reward hacking specific to 3D pipelines. Understanding these failure modes would significantly strengthen claims about the method's generality and robustness.

### Questions
- Related to the 'No Validation That 2D Rewards Transfer to 3D Preference Space' weakness, how well do 2D reward scores correlate with human preference judgments in 3D, and can you provide a correlation analysis or human evaluation to address this concern?
Have you attempted collecting a small amount of 3D human preference data? Even a pilot dataset, for example 100 to 500 comparisons, could validate or invalidate the main motivation.

- What are representative failure cases where 2D reward guidance misaligns with 3D preference? Can you show examples of Janus or multi-view inconsistency when using the proposed Preference Score Distillation (PSD)?

- If 3D preference supervision is “too expensive,” what is the estimated cost? Some cost or complexity justification would greatly support the core motivation of the work.

- Can your method incorporate even sparse 3D preference labels? Hybrid approaches that combine 2D and 3D signals could further strengthen the proposal.

### Soundness
2

### Presentation
2

### Contribution
2
