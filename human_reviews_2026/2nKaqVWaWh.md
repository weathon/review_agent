# IntDiff: Mitigating Reward Hacking via Intrinsic rewards for Diffusion Model Fine-Tuning

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Diffusion models have progressed in text-to-image generation, but their ability to optimize predefined objectives remains limited. Although introducing reinforcement learning (RL) fine-tuning can improve performance, it also brings two critical problems: exploration-exploitation imbalance and reward hacking. To address both, we propose a systematic framework, \ourmethod{}, which designs a denoising-based intrinsic reward paradigm to guide exploration. A filtering mechanism is introduced to dynamically monitor changes in text-image alignment, penalize exploratory behaviors that cause degradation, and selectively discard inefficient samples to improve exploration effectiveness and save computation. Furthermore, we propose an adaptive mechanism that enhances exploration in the early stages and shifts to exploitation to stabilize the structure, improve the predefined reward in the late stages, and dynamically filter training steps. A Reflective Diffusion Optimization method is introduced to improve sample efficiency and training effectiveness under sample reduction and step truncation. Overall, this paper aims to significantly improve generation quality and alignment with target objectives, under the premise of reducing computational cost and mitigating reward hacking. Experimental results show that the proposed method improves the alignment and diversity of text and achieves superior performance in various reward metrics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a simple yet effective approach to finetune Diffusion Models. By incorporating intrinsic reward as well as reward filter and length filter, the method not only improves the quality, diversity of generated samples but also time efficiency.

### Strengths
- The authors correctly identify key pain points in RL-based fine-tuning - namely, sparse terminal rewards, exploration–exploitation imbalance, and computational inefficiency. These are real challenges in practice, and the paper\'s motivation aligns well with known limitations in existing methods.
- The authors reasonably transfer the technique from RL-based fine-tuning into the diffusion learning case.
- The paper is well structured, easy to read.

### Weaknesses
The intrinsic reward $R_t^{\rm int} = \Vert\hat{x}_0^{(t)} - \hat{x}_0^{(t+1)}\Vert_2^2$ is not well justified. There\'s no clear justification for why this formulation meaningfully captures \"intrinsic motivation\" or avoids reward hacking in a principled way. I suggest that a comprehensive theoretical analysis should be conducted here. I would suggest a differential analysis of how this reward affects the score function.

The time complexity of the approach is unclear. From Equation (5), $\Delta c_t$ along with the reverse sampling equation are obtained in every single step, which requires a huge computation. However, the complexity is not well discussed. Since the author proposes to use a length filter to terminate, the theoretical benefits of this filter are not well discussed either.

The datasets (\"Simple-Animals\" and \"Activity-Animals\") are toy-level benchmarks. These limited domains fail to validate the generalizability of the approach to real-world prompts or tasks.

### Questions
Please see the weakness section.

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
This paper proposes IntDiff, a reinforcement-learning (RL) fine-tuning framework for diffusion models that mitigates reward hacking and sparse-reward problems by injecting an intrinsic reward signal into intermediate denoising steps. The authors (i) define the intrinsic reward as the ℓ²-distance between two consecutive predictions of the clean image, (ii) modulate this reward with a CLIP-based semantic-consistency term Δcₜ, (iii) introduce an adaptive coefficient αₜ that trades off exploration (intrinsic) and exploitation (extrinsic) rewards according to the estimated denoising progress, and (iv) dynamically shorten the denoising horizon when αₜ≈1. Extensive experiments on Stable-Diffusion backbones show improved diversity, text–image alignment, and ≈2× reduction in training cost compared with DDPO, DPOK, D3PO and TDPO.

### Strengths
(i) Novel perspective: the first work that brings step-wise intrinsic motivation into diffusion RL fine-tuning, instead of terminal-only rewards.
(ii) Comprehensive evaluation: ablations, cross-dataset generalization, human/LVM preference studies, and computational cost analysis are all provided.
(iii) Practical impact: dynamic early stopping and reward filtering together cut the total denoising steps by ~50 % while preserving or even improving reward scores.

### Weaknesses
(i) Lack of theoretical guarantee for the proposed intrinsic reward: The intrinsic reward R^{int}_t=‖x̂₀^{(t)}−x̂₀^{(t+1)}‖² is only motivated heuristically (“larger jump ⇒ more exploration”). There is no MDP optimality guarantee that maximizing this quantity helps convergence to the true policy optimum. Worse, the ground-truth clean image x₀ is available during training; a direct supervised error ‖x̂₀^{(t)}−x₀‖ could provide a much denser and unbiased learning signal. The authors neither adopt this obvious baseline nor justify why the change of prediction is preferable to the error w.r.t. the target.
(ii)Insufficient discussion and comparison with closely related work:
sucha as [1] [2] also address the sparse-reward issue in diffusion RL by designing per-step proxy rewards. They are neither cited nor compared. Consequently, the paper fails to clarify how IntDiff differs from (or subsumes) these concurrent solutions, weakening the novelty claim.
[1] Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards
[2] A Dense Reward View on Aligning Text-to-Image Diffusion with Preference
(III). Empirical improvements are marginal: Ablation results in Table 1 show that each added component lifts the aesthetic score by only ≈0.01–0.03 and CLIP/PickScore by ≤0.01, which is within one standard deviation; Fig. 3 indicates that TDPO achieves almost identical reward growth on SD-v1.5 and SDXL, and sometimes outperforms IntDiff in the early phase; Diversity gains (LPIPS ↑0.013 over DDPO in Table 4) are statistically significant but visually subtle. Given the extra engineering effort (αₜ predictor, reward filter, RDO optimizer), the practical benefit is less compelling than claimed.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a reinforcement learning (RL) fine-tuning framework for diffusion models, termed IntDiff, which mitigates reward hacking and improves task alignment under limited compute.
In practice, standard RL approaches score only the final image, yielding sparse feedback, exacerbating the exploration–exploitation imbalance across long denoising chains, and wasting compute with fixed training steps.
To address these limitations, IntDiff introduces: 1) intrinsic rewards injected along the denoising trajectory to provide stepwise guidance without auxiliary models; 2) Semantic Consistency Regularization to penalize text–image drift and curb low-efficiency exploration; and 3) Adaptive Coordination that biases toward early exploration and late exploitation based on estimated denoising progress, coupled with dynamic early stopping that adapts training length to prompt difficulty.
Empirically, IntDiff improves diversity by up to 60%, reduces training compute by 22%, and increases downstream task rewards while mitigating reward hacking.

### Strengths
IntDiff is a well-motivated, thoughtfully engineered framework that balances originality with practical relevance.
Its focus on stepwise guidance, alignment-preserving exploration, and adaptive compute makes the contribution both novel and useful, with clear potential to influence future RL-for-diffusion research and applications.
Importantly, the extensive experimental results clearly demonstrate its effectiveness.

### Weaknesses
1. The paper does not clearly distinguish IntDiff’s intrinsic stepwise rewards and Semantic Consistency Regularization from established ideas—e.g., curiosity/novelty bonuses, CLIP-gated faithfulness, and adaptive/early-exit schedulers—so the contribution reads more like a recombination than a novel advance.
2. The adaptive regulation theory is presented without explicit assumptions or a concrete mapping to the implemented schedule, leaving the scope of any guarantees unclear.
3. The reported ~22% compute savings are neither itemized nor normalized against matched baselines. Please include a cost table (training and inference) detailing wall-clock time, GPU-hours, energy consumption, and peak memory.

### Questions
1. Which diversity metrics are evaluated (inter-prompt vs. intra-prompt), how many seeds are used, and are the differences statistically significant?
2. How were the target reward levels and schedule parameters selected, and how stable are the results across these settings?
3. Beyond gains on the training reward, how is mitigation quantified? Please specify complementary measures and evaluation protocols.

Since I am not very familiar with RL, I will reconsider my scores if the authors address my questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes IntDiff, which contains three main innovations:
- Intrinsic rewards for intermediate denoising steps to alleviate sparse feedback.
- Semantic Consistency Regularization to penalize exploration that degrades text-image alignment.
- Adaptive Coordination leveraging denoising progress to balance exploration-exploitation and enable dynamic early stopping.
- Various experiments are conducted to prove the advances of IntDiff compared with previous work.

### Strengths
- Novelty and Innovation. The first intrinsic reward paradigm for diffusion RL, and the adaptive exploration–exploitation balance, makes sense considering the long denoising chains.
- Rigorous and Comprehensive Experiments. Different dimensions of experiments are conducted, and the results of these experiments span the comparison of different methods, the comparison of different backbones, and ablation studies.
- Clear figures and metrics. It provides comprehensive figures, tables, and experimental results, ensuring exceptional clarity in the presentation.

### Weaknesses
- Narrow prompt domains. Animal-related prompts are primarily used. Testing on diverse prompts (e.g., abstract concepts or multi-object scenes) would strengthen claims.
- Details of human evaluations would be better if provided.
- The references are somewhat insufficient. More diffusion with RL reference could be included, like [1][2].

[1] FIND: Fine-tuning Initial Noise Distribution with Policy Optimization for Diffusion Models. MM 2024
[2] Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward. CVPR 2025

### Questions
- Can IntDiff be used for video diffusion or 3D model diffusion? How to design the rewards?
- What will happen if we input more abstract prompts instead of animals?

### Soundness
3

### Presentation
2

### Contribution
3
