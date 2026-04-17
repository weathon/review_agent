# Accelerating Diffusion Planners in Offline RL via Reward-Aware Consistency Trajectory Distillation

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Although diffusion models have achieved strong results in decision-making tasks, their slow inference speed remains a key limitation. While consistency models offer a potential solution, existing applications to decision-making either struggle with suboptimal demonstrations under behavior cloning or rely on complex concurrent training of multiple networks under the actor-critic framework. In this work, we propose a novel approach to consistency distillation for offline reinforcement learning that directly incorporates reward optimization into the distillation process. Our method achieves single-step sampling while generating higher-reward action trajectories through decoupled training and noise-free reward signals. Empirical evaluations on the Gym MuJoCo, FrankaKitchen, and long horizon planning benchmarks demonstrate that our approach can achieve a $9.7$% improvement over previous state-of-the-art while offering up to $142\times$ speedup over diffusion counterparts in inference time.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Reward-Aware Consistency Trajectory Distillation (RACTD), a method that extends consistency trajectory models to the offline reinforcement learning setting by incorporating a reward objective into the distillation process. The approach allows a student model to generate high-reward action trajectories in a single denoising step, combining efficiency from consistency models with reward-guided behavior selection. Experiments across standard offline RL benchmarks demonstrate that RACTD improves policy quality and sampling efficiency compared to diffusion-based and actor-critic baselines by a large margin in many cases.

### Strengths
- The paper is very well written and was genuinely an enjoyable read. Most questions that arise while reading are quickly addressed by the text, and all of the necessary context to follow the technical discussion is included.
- The motivation is strong, with clear explanations of the limitations of prior work such as the slow inference speed of diffusion models and the sensitivity of actor-critic frameworks to hyperparameters.
- The introduction of decoupled training in Section 3.4 is a genuine strength, making the method simpler and more stable in practice. The idea of operating directly in the clean reward space rather than training noise-aware reward models is both elegant and effective.
- The proposed method demonstrates substantial empirical gains over prior work, with strong improvements in performance and efficiency. 
- The inference-time acceleration that this method inherits from prior work makes the approach especially appealing for real-world decision-making tasks.

### Weaknesses
- Not a substantial weakness, but there is a typo between lines 202 and 203 (struggle vs. struggles)
- In Table 3, the paper introduces CTD as a comparison point without tying this acronym to a particular method and without comparing CTD's results to anything else in the main text (unless I am mistaken). Presumably this is Consistency Trajectory Distillation. Reading back through the paper, it appears that CTD corresponds to the authors’ approach without the reward aware component, that is, Consistency Trajectory Model (CTM) [1] augmented with a denoising score matching (DSM) loss. However, this relationship is never made explicit. Clarifying whether CTD is simply CTM+DSM loss applied to offline reinforcement learning or a modified variant beyond the addition of the DSM loss is important for readers to properly interpret the results. 
- Based on the prior point, the novelty of the proposed method is therefore somewhat unclear when compared to the backbone it builds off of. The paper introduces Reward Aware Consistency Trajectory Distillation (RACTD), which extends CTD by incorporating a reward term. This alongside the incorporation of the DSM loss is novel and shows good results compared to prior methods in the literature. So the motivation for these additions is clear, as shown in Figure 3, but the relative benefit of the reward aware RACTD component remains mostly untested against the unconditional variants CTD and CTM. A broader ablation comparing CTM, CTD, and RACTD across all benchmarks (for example, Tables 1, 2, 4, and 5), alongside their relative inference times and sample quality, would better isolate the benefit of the reward awareness and DSM loss integration. 
- If CTD is effectively a reimplementation of CTM adapted to the offline RL setting with minimal modifications, then the faster inference times should be credited to prior work rather than presented as new contributions. This is backed by Table 3's comparison of CTD to RACTD where both methods have identical inference times. Applying existing techniques in new settings is a contribution that is noteworthy and part of this work, but the main contribution here would then lie in the reward aware adaptation, DSM loss, and application in a new domain with great score improvement results, rather than claiming the inference speedups already established by CTM. The acceleration results therefore represent a successful domain adaptation of a pre-existing backbone you built on top of, not a new methodological innovation, and should be framed as such in the paper (e.g. the leading sentence in Section 4.3 "Beyond performance improvements, another major contribution of our work is significantly accelerating diffusion-based models for decision-making tasks." or the last sentence of the conclusion: "RACTD outperforms previous state-of-the-art by 9.7% while accelerating its diffusion counterparts up to a factor of 142.").
- Given that the primary acceleration mechanism is inherited from Consistency Trajectory Models [1], the title “Accelerating Diffusion Planners in Offline RL via Reward-Aware Consistency Trajectory Distillation” may overstate the novelty of the acceleration component. A more accurate framing would emphasize the reward-aware extension rather than the inference-speed improvement itself, which seemingly follows directly from prior work.

[1] Kim, Dongjun, et al. "Consistency trajectory models: Learning probability flow ode trajectory of diffusion." arXiv preprint arXiv:2310.02279 (2023).

### Questions
- In Section 3.3 the reward model is frozen during training. How sensitive is the overall planner to the diversity and coverage of the dataset used to train this reward model? The paper emphasizes robustness to suboptimal data, but what happens when planned trajectories drift outside the dataset’s support, potentially leading to compounding errors in both the reward model and the planner?
- How sensitive is your method to alpha, beta, and sigma in the RACTD loss? Do you perform ablations to determine this sensitivity?
- In Figure 3, the diffusion-based methods appear to have one of their high-probability density modes lying outside the support of the high-reward mode in the training dataset. Am I misinterpreting this figure? Intuitively, the highest-probability regions for the unconditional diffusion-based methods should align with the high-reward mode of the training distribution. The rightward shift of the RACTD distribution is understandable due to the added reward objective, but why do the unconditional diffusion-based models also shift toward higher rewards than the dataset’s high-reward mode?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The manuscript proposes a method termed RACTD aimed at accelerating diffusion-based components within offline reinforcement learning by reducing the number of function evaluations (NFE) while maintaining policy performance.
This work presents a novel approach to consistency distillation that directly incorporates reward optimization into the distillation process. The proposed method achieves single-step diffusion sampling while generating
higher-reward action trajectories through decoupled training and noise-free reward guidance.

### Strengths
- Clear experimental scope and benchmarks: Evaluation on widely used D4RL Gym-MuJoCo and FrankaKitchen datasets enhances relevance and comparability, with both offline and online model selection reported where applicable.
- Focus on sampling efficiency: Explicit reporting of NFE alongside performance indicates attention to practical efficiency, which is crucial for diffusion-based methods.

### Weaknesses
- Limited novelty and contribution: The core idea is to augment the distillation process with a cumulative reward maximization objective. This training pipeline has appeared in prior work (e.g., Flow Q-Learning), and the paper does not clearly isolate what is fundamentally new beyond this template.
- Central claim lacks rigorous empirical validation: The paper emphasizes incorporating a reward objective directly into consistency distillation rather than optimizing via a critic (e.g., Q-values or advantages). This design choice requires thorough ablations and head-to-head comparisons against critic-based alternatives to substantiate the claim.
- Motivation section is superficial: The insights in Section 3.1 are relatively basic and widely known; they would be more appropriate in the introduction. The main text should instead focus on deeper methodological details, theoretical justification, or non-trivial empirical findings.
- Incomplete experimental comparisons: The evaluation is missing strong, direct baselines from state-of-the-art one-step diffusion/consistency methods.

### Questions
The theoretical contribution appears weak; please strengthen the theoretical analysis and add comprehensive comparisons with state-of-the-art methods (e.g., one-step diffusion/consistency baselines) to substantiate the claims.

### Soundness
3

### Presentation
3

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
This work introduces Reward-Aware Consistency Trajectory Distillation (RACTD), which presents a novel application of consistency distillation to diffusion-based planning in reinforcement learning. The method trains a single-step student model to emulate a multi-step diffusion teacher, effectively accelerating inference while maintaining high performance. The authors incorporate a reward model alongside the consistency trajectory loss and denoising score matching loss within a decoupled training framework. Experimental results demonstrate that the proposed RACTD substantially outperforms state-of-the-art diffusion planning methods in both performance and inference speed.

### Strengths
1. The paper introduces a novel application of consistency distillation, resulting in a single-step model that effectively emulates multi-step diffusion processes, thereby significantly improving inference efficiency.

2. The proposed RACTD demonstrates strong empirical performance, substantially outperforming prior state-of-the-art diffusion-based reinforcement learning methods, particularly on the D4RL benchmark.

3. The paper presents comprehensive experiments and detailed implementation descriptions, accompanied by code availability, which enhances the credibility and reproducibility of the reported results.

### Weaknesses
1. The idea of the paper is interesting, and the contribution of accelerating the sampling stage through consistency distillation is clear. However, the work appears to rely heavily on existing consistency distillation techniques, with limited novelty beyond their direct application to diffusion-based planning.

2. The contribution of the proposed reward-aware consistency trajectory distillation is somewhat unclear. The method appears to employ a standard reward model as an auxiliary loss applied to the student’s output, rather than introducing a fundamentally new formulation or architecture.

### Questions
1. The authors claim that their proposed decoupled training outperforms the simultaneous training of actor and critic networks from scratch. However, there does not appear to be any experimental evidence supporting this statement. Could the authors provide an ablation study or empirical comparison demonstrating the benefits of decoupled training?

2. In Table 2 and Table 11, Diffusion-QL achieves better performance than RACTD, although RACTD uses 5x fewer sampling steps as stated by the authors. In this case, why did the authors not compare all methods under the same number of sampling steps? Such a comparison would be more meaningful than simply showing that the baseline achieves the best performance, particularly in Table 11.

3. The results in Table 1 on the D4RL benchmarks show that the proposed RACTD method significantly outperforms prior state-of-the-art approaches on average. What do the authors believe is the main factor behind this improvement? Is the performance gain primarily attributable to the application of consistency distillation?

### Soundness
3

### Presentation
3

### Contribution
3
