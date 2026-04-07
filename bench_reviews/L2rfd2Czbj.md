## Summary
The paper proposes wd1, a novel reinforcement learning method for diffusion-based large language models (dLLMs) that reformulates policy optimization as a weighted log-likelihood objective. This eliminates the need for explicit policy ratio estimation, thereby reducing computational overhead and mitigating bias/variance from likelihood approximation errors. The method is theoretically interpreted as energy-guided diffusion training combined with negative sample unlearning. Experiments on reasoning benchmarks show significant accuracy improvements and reduced training cost compared to baselines, with an extension (wd1++) achieving state-of-the-art results on MATH500 and GSM8K.

## Strengths
- **Directly addresses a core technical challenge**: The paper clearly identifies and tackles the problem of intractable likelihoods in dLLMs, which leads to error amplification and high computational cost in existing ratio-based RL methods like GRPO. The proposed ratio-free weighted objective is a well-motivated and effective solution.
- **Strong and comprehensive empirical validation**: wd1 achieves dramatic improvements on planning-intensive tasks (e.g., +58.8% on Sudoku, +16% on Countdown) without supervised fine-tuning, while also matching or exceeding baselines on standard math reasoning. The extended wd1++ method sets new SOTA results (44.2% on MATH500, 84.5% on GSM8K) with notably fewer training steps.
- **Novel theoretical grounding**: The paper provides a novel and sound theoretical interpretation, showing that the positive component of the objective is equivalent to training an energy-guided discrete diffusion model, while the negative component relates to data unlearning. This elevates the work beyond a purely empirical contribution.
- **Rigorous ablation studies**: Ablations convincingly demonstrate the necessity of both the positive and negative weighting terms and validate the balanced combination, providing clear empirical justification for the design choices.
- **Clarity and reproducibility**: The method is clearly described with full derivations, the algorithm is presented, code is released, and experimental details (hyperparameters, rewards, datasets) are thoroughly documented in the appendix.

## Weaknesses
- **Efficiency claim for wd1++ requires clarification**: The reported "10× fewer rollouts" compares the number of *final* generated completions. However, wd1++ utilizes all intermediate denoising-step completions, increasing the total number of training samples per rollout. A fairer efficiency comparison should account for the total number of samples or forward passes used for training to accurately assess computational trade-offs.
- **Evaluation limited to a single model family**: All experiments are conducted on the LLaDA-8B model architecture. While results are compelling, demonstrating effectiveness on at least one other dLLM (e.g., Dream-7B or a SEDD-based model) would strengthen claims of general applicability across the dLLM paradigm.
- **Heuristic element in the full objective**: The negative weight term (w⁻) is introduced primarily based on empirical motivation (to fully utilize samples and actively penalize low-advantage completions). Although later connected to unlearning theory and validated via ablation, its integration into the core theoretical derivation (from reverse-KL optimization) is less direct than the positive term. A more explicit discussion of its role within the theoretical framework would strengthen the presentation.

## Nice-to-Haves
- A sensitivity analysis for the hyperparameter ψ (which controls the sharpness of the exponential weights) across different tasks would provide practical guidance for users.
- Visualizing the distribution of weights (w⁺ and w⁻) as a function of advantage during training could offer intuitive insights into the method's balancing mechanism.
- A breakdown of the computational cost (time/FLOPs) into sampling, likelihood approximation, and gradient computation components would further pinpoint the source of efficiency gains.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Ambiguity in Equation 3**: The formulation of the diffusion-GRPO objective using `min` and the clipping term is mathematically correct and aligns with standard implementations; it does not cause confusion.
- **Theoretical complexity is a weakness**: The theoretical interpretation, while dense, is appropriate for the venue and is clearly presented.
- **Reproducibility gaps**: The paper provides code, detailed hyperparameters, and dataset descriptions, meeting standard reproducibility expectations.
- **Requirement to compare to on-policy/other off-policy baselines**: The paper's comparisons to the established baseline (d1) and several strong concurrent methods (MDPO, SDPO) are sufficient to demonstrate its contribution within the current research landscape for dLLM RL.
- **Demand for analysis of approximation bias vs. more accurate estimators**: The paper's core contribution is the ratio-free objective; the choice of likelihood approximator (d1-based) is an implementation detail shared with the baseline. Analyzing other approximators is an interesting extension but not required to validate the main claim.

## Novel Insights
The paper provides a genuinely novel theoretical insight by formally connecting the weighted log-likelihood objective (derived from reverse-KL regularized policy optimization) to energy-guided discrete diffusion training. Specifically, it proves that maximizing the advantage-weighted likelihood is equivalent to minimizing an Advantage-Weighted Denoising Concrete Score Matching (AW-D-CSM) loss, which steers the diffusion model's generation toward high-advantage regions. This interpretation provides a fresh, principled perspective on RL for diffusion models beyond the standard policy gradient framework.

## Suggestions
- Clarify the efficiency metrics for wd1++ in the main text or caption of Table 3 (right), explicitly stating that "rollouts" refer to final completions and discussing the trade-off of using intermediate samples.
- Expand the discussion in the limitations section (Appendix D) to include practical mitigation strategies for the identified failure mode (e.g., when all completions in a batch receive identical rewards), such as reward shaping or adjusting the group size.