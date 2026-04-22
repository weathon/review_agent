# Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 4

## Abstract
Reinforcement learning (RL) algorithms have been used recently to align diffusion models with downstream objectives such as aesthetic quality and text-image consistency by fine-tuning them to maximize a single reward function under a fixed KL regularization. However, this approach is inherently restrictive in practice, where alignment must balance multiple, often conflicting objectives. Moreover, user preferences vary across prompts, individuals, and deployment contexts, with varying tolerances for deviation from a pre-trained base model. We address the problem of inference-time multi-preference alignment: given a set of basis reward functions and a reference KL regularization strength, can we design a fine-tuning procedure so that, at inference time, it can generate images aligned with any user-specified linear combination of rewards and regularization, without requiring additional fine-tuning? We propose Diffusion Blend, a novel approach to solve inference-time multi-preference alignment by blending backward diffusion processes associated with fine-tuned models, and we instantiate this approach with three algorithms: DB-MPA for multi-reward alignment, DB-KLA for KL regularization control, and DB-MPA-LS for approximating DB-MPA without additional inference cost. Extensive experiments show that Diffusion Blend algorithms consistently outperform relevant baselines and closely match or exceed the performance of individually fine-tuned models, enabling efficient, user-driven alignment at inference-time.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This is a classifier‑free guidance–like formulation designed for reward alignment in a score-based generated model.

### Strengths
Overall, the paper mainly makes a theoretical contribution. 

## Presentation: ~75th percentile

This paper is comprehensible to diffusion model researchers, but it possesses some weaknesses in presentation.

## Soundness: 80~90th percentile

A great portion of the soundness stems from the reliability and the success of CFG. The formulation appears correct based on my understanding of CFG and score-based SDE.

## Contribution: ~90th percentile

To the best of my knowledge, you are the first, aside from any concurrent work, to propose this 
formulation. Your contribution is particularly strong because it is presented within a score‑based SDE 
framework. It could have been even more impactful had you expressed it in Karras’s SDE, although that is 
not strictly necessary.

## Note
I hope the AC is aware that the rating is calibrated using percentiles to reduce evaluation noise effectively.

### Weaknesses
## Presentation

Presenting advanced mathematics elegantly is always a challenge, and a successful example in writing I’ve read is the score-based SDE by Yang et al. Perhaps it would be better to present the main idea without too many interruptions from details, making the mathematical notation less daunting in the main paper for a wider range of readers.  But that is impossible without a significant modification of the manuscript.  If accepted, I recommend rewriting your final manual script or, if any, your preprints.

There are some minor formatting issues, such as inconsistent use of italics and bold across sections, and the abuse of italics applied to the entire paragraph.

### Questions
As noted in strength, I prefer to see the reformulation written in Karra's SDE instead.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Diffusion Blend that aligns diffusion models' generation process with user-specified multi-preferences. Crucially, this is done at inference time such that users can dynamically control preferences as they wish. Diffusion Blend trains different versions of diffusion models that maximize the given reward functions and then combine their scores during the generation process. This paper introduces theoretical justification for such a process and demonstrates its effectiveness in quantitative and qualitative experiments.

### Strengths
- Importance of the problem: Dynamically aligning diffusion models' generation process with user preferences is a challenging problem that can have a great impact on content creation applications by providing users with knobs to adjust their content. 
- This paper proposes a simple algorithm that achieves this purpose by combining scores of diffusion models trained for specific rewards. This is theoretically justified well, and the simplicity of the algorithm would enable easy adoption of the method.
- Maintaining multiple copies of diffusion models could be heavy in storage and computation, but the experiment with LoRA shows it is possible to achieve efficient alignment by maintaining relatively small copies.
- The evaluation in the experiments section shows a consistent boost in performance over the baselines considered in the paper.

### Weaknesses
- The experiments are done with a relatively small number of reward models (most with two) and a single class of diffusion model, Stable Diffusion. The results could have been stronger by pushing the limit with more rewards (e.g. 6~10) and with more recent diffusion models such as Flux. 
- In practice, LoRA-based diffusion model weight combination techniques (e.g. Zou, Shen, Bouganis, and Zhao, ICLR 2025) could be a strong candidate to achieve the same purpose. How does it perform, and is there a practical advantage to using Diffusion Blend over it?

Minor: in equation 11) $\log p^{tar}_t$ in the right hand side needs to be  $\log p^{pre}_t$?

### Questions
It would be great if the rebuttal could comment on the points in the weaknesses section.

### Soundness
4

### Presentation
4

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
This paper tackles the challenge of aligning diffusion models with multiple, potentially conflicting objectives at inference time. Existing RL-based alignment methods typically optimize a single reward under fixed KL regularization, which limits flexibility. The authors introduce Diffusion Blend, a novel framework that enables inference-time multi-preference alignment, which allow the model to generate outputs based on any user-specified combination of reward functions and regularization strengths without additional fine-tuning. They instantiate this idea with three algorithms: DB-MPA (for multi-reward alignment), DB-KLA (for controllable KL regularization), and DB-MPA-LS (a low-cost approximation). Empirical results demonstrate that these methods outperform existing baselines and perform comparably to models fine-tuned individually, offering a practical way to achieve user-driven, flexible alignment during inference.

### Strengths
1. The paper is well written and easy to follow.

2. The studies of conducting inference time alignment on diffusion models are novel and interesting.

3. The authors conduct extensive experiments to verify the effectiveness of thier method.

### Weaknesses
1. It's better for authors to have some results on larger models like SDXL to further prove the effectiveness of thier method.

2. It's better to demonstrate that the model can be used in wide applications like image editing.

3. The authors use DPOK for fine-tuning models, is this method sensitive to different RL algorithms?

### Questions
Please refer to the answer in Weaknesses part.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of inference-time multi-preference alignment for diffusion models, where the users can adjust reward trade-offs and regularization strengths on-the-fly without additional finetuning. The DiffusionBlend pipeline blends the backward diffusion trajectories of multiple RL-finetuned models to distinct reward functions. The method is theoretically motivated and empirically validated on a series of benchmarks and alignment tasks.

### Strengths
1. The problem is novel and well-motivated. The paper formalizes the inference-time multi-preference alignment problem from the perspective of MORL, which is practically important and underexplored in diffusion literature. 

2. The approach is theoretically justified with clear derivations and bounds on approximation errors. 

3. Strong empirical results on multiple datasets and reward functions.

### Weaknesses
1. The Jensen-gap approximations in Eq. 8 are only empirically validated via downstream metrics; direct error analysis on $ \Delta(r,\alpha) $ (beyond Appendix bounds) would strengthen claims. 

2. The paper builds on KL-regularized RL fine-tuning for diffusion models, but does not sufficiently discuss or compare to DiffusionDPO (Wallace et al., 2024), DDPO (Black et al., 2024), etc, which are dominant lines of work in this space. For example, the author cites DPO (Rafailovetal., 2023) for diffusion model alignment in section 4, but does not mention DiffusionDPO (Wallace et al., 2024), which is the direct diffusion analog and is relevant. 

2. The training cost still scales linearly with respect to the number of rewards. This can be costly in applications. 

3. Based on the provided pages, evaluations focus on two rewards (alignment + aesthetics) and Stable Diffusion v1.5. Scaling to m > 2 or diverse rewards (e.g., human preferences via PickScore, diversity) and larger base models (e.g., SDXL) is not shown. 

4. Strong compared to RS/CoDe/RGG, but no comparison to DPO-inspired diffusion variants (e.g., Calibrated DPO) or recent multi-LoRA fusion methods (Zhong et al., 2024a). The best individual RL-finetuned model per w (the oracle) is also missing.

### Questions
see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
