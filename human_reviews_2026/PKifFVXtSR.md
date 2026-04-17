# Ada-Diffuser: Latent-Aware Adaptive Diffusion for Decision-Making

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Recent work has framed decision-making as a sequence modeling problem using generative models such as diffusion models. Although promising, these approaches often overlook latent factors that exhibit evolving dynamics, elements that are fundamental to environment transitions, reward structures, and high-level agent behavior. Explicitly modeling these hidden processes is essential for both precise dynamics modeling and effective decision-making. In this paper, we propose a unified framework that explicitly incorporates latent dynamic inference into generative decision-making from minimal yet sufficient observations. We theoretically show that under mild conditions, the latent process can be identified from small temporal blocks of observations. Building on this insight, we introduce Ada-Diffuser, a causal diffusion model that learns the temporal structure of observed interactions and the underlying latent dynamics simultaneously, and furthermore, leverages them for planning and control. With a modular design, Ada-Diffuser supports both planning and policy learning tasks, enabling adaptation to latent variations in dynamics, rewards, and latent actions. Experiments on locomotion and robotic manipulation benchmarks demonstrate its effectiveness in accurate latent inference, long-horizon planning, and adaptive policy learning

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a unified framework that explicitly incorporates latent dynamic inference into generative decision-making, particularly focusing on partially observable environments. The authors highlight that existing generative models (like Diffusion Models) often overlook evolving latent factors essential to environment transitions, reward structures, and high-level agent behavior. Algorithmic Innovations include causal autoregressive denoising schedule, denoise-and-refine training procedure and zig-zag sampling for inference. Extensive experiments across locomotion, navigation, and robotic manipulation benchmarks show improved performance in accurate latent inference, long-horizon planning, and adaptive policy learning compared to diffusion baselines and variants augmented with existing latent modeling techniques.

### Strengths
1. Originality. The paper demonstrates high originality, primarily through the unique integration of theoretical identifiability results into the architecture of a causal diffusion model. Specifically, establishing sufficient conditions under which latent factors can be identified from short temporal windows of RL trajectories is a novel theoretical contribution. This theoretical insight directly guides the algorithmic design, leading to the development of the minimal-sufficient block concept and the backward refinement step for online latent recovery, which contrasts with prior latent-augmented diffusion approaches. The combination of autoregressive denoising, latent factor identification, and the tailored zig-zag sampling scheme represents a creative and non-trivial synergy of existing ideas applied to sequential decision-making.

2. Quality. The quality of the work is high, supported by rigorous technical details and thorough empirical validation. The identification theory provides a strong foundation for the method. The experimental evaluation is comprehensive, covering eight environments across 23 different settings, including latent factors affecting dynamics and rewards, latent actions from action-free data, and environments without explicit latents. Furthermore, the paper provides detailed ablation studies confirming the effectiveness of the key design choices, particularly the refinement and zig-zag sampling mechanisms, which significantly reduce probing MSE and improve performance.

3. Clarity. The paper is well-structured. The framework's modular design, dividing the process into latent factor identification (Stage 1) and the causal diffusion model (Stage 2), enhances clarity. The authors clearly define the Latent Contextual POMDP with Time-dependent Context and formalize the data generation process using Structural Causal Models (SCMs). The purpose and justification for complex components like the latent identification loss and zig-zag sampling are explicitly tied back to the theoretical findings (Theorem 1), providing a clear narrative thread.

4. Significance. The significance of Ada-Diffuser lies in its ability to handle time-varying, unobserved latent factors which are pervasive in real-world applications such as robotics and autonomous driving. By providing a unified generative framework for planning and policy learning that effectively adapts to these latent variations, the method offers substantial improvements over state-of-the-art diffusion baselines and meta-RL approaches in settings with explicit latent factors. Moreover, the finding that latent modeling can improve performance even in environments without explicit latent factors suggests broad utility beyond traditionally defined POMDPs.

### Weaknesses
1. Mismatch between Theoretical and Practical Block Sizes: The central theoretical result (Theorem 1) establishes that four consecutive time steps (x_t−2:t+1) are sufficient for latent identifiability. However, the authors state that in practice, they do not strictly limit the block size to four, finding that slightly larger blocks (typically between 6 and 20 steps) lead to more stable optimization and better performance. This practical necessity for larger blocks suggests that the assumptions required for the theoretical four-step result may be too stringent or idealized for the complexity of the empirical environments and the parametric function approximators used. The reliance on large blocks somewhat diminishes the immediate algorithmic guidance derived from the "minimal-sufficient block" result. This adds to the suspicion that the authors reverse-engineered the justification, deliberately finding an obscure mathematical theorem to back a model they already wanted, instead of allowing the model to emerge naturally from the theory.

2. Dependency on Future Observations during Online Inference: The latent identifiability theory relies on incorporating future observations (x_t+1) to form an accurate posterior q_ψ(c_t∣x_{t−T_x:t+1}). In online inference, future observations are unavailable, creating a mismatch. The zig-zag sampling scheme attempts to resolve this by conditioning on a predicted next noisy step x_{t+1}^{k_2}​ to refine \hat{c}_t^post. This means the quality of the online latent inference hinges critically on the accuracy of the diffusion model's prediction of x_{t+1}^{k_2}. If the future prediction is inaccurate, the latent refinement may degrade, potentially leading to cascading errors in long-horizon planning. This key dependency is not fully analyzed or quantified in the provided sources, particularly in terms of how errors propagate when using the noisy future estimate.

3. Increased Model Complexity and Training Stage Separation: Ada-Diffuser requires training two distinct stages: (1) an offline VAE for latent factor identification (optimizing the ELBO loss), and (2) the causal diffusion model (optimizing along with the contrastive loss L_d−r). This two-stage training process is inherently more complex and potentially less robust than end-to-end training. While the authors claim the VAE introduces minimal computational overhead, the overall architectural complexity, including the GRU encoder, multiple MLPs for priors/posteriors, and the explicit contrastive loss L_rel demands careful tuning of multiple hyperparameters (λ_KL, λ_prior, λ_rel, m).

4. Novelty of Component Integration: While the overall framework is original, the individual components draw heavily from existing literature. The concept of latent belief state learning in POMDPs is standard (e.g., LILAC, DynaMITE, VAE usage). The autoregressive noise schedule is derived from recent autoregressive diffusion models (e.g., DF). The novelty rests heavily on the specific way the blocks, refinement, and zig-zag work together. However, a deeper analysis justifying why this specific combination is necessary—rather than sufficient—to satisfy the identifiability constraints (especially compared to simpler VAE-in-loop approaches like DynaMITE/LILAC integrated with diffusion) would strengthen the contribution.

### Questions
- Questions on Identifiability and Online Inference

1. Causal Timeline and Lookahead in Zig-Zag Sampling: During online inference (zig-zag sampling), the refined latent \hat{c}_t^post is conditioned on x_{t+1}^{k_2}. In a standard sequential decision-making loop, s_t+1 (and thus x_t+1) is only observed after the action a_t has been executed. Please clarify the precise temporal availability of the conditioning inputs x_t^0 x_t^{k_1} x_{t+1}^{k_2} used to calculate \hat{c}_t^post during online generation of a_t. Does x_{t+1}^{k_2} represent a predicted state from the generative model itself? If so, this suggests \hat{c}_t^post is recursively dependent on the model's own (noisy) future prediction, which could introduce instability or drift, especially for long planning horizons (e.g., T_p=32).

2. Practical Implication of Block Size (Beyond Stability): The ablation shows that performance degrades with blocks that are too small (≤4) or too large (>20). Given that the theory proves 4 steps are sufficient, what specific phenomena in the practical optimization environment (e.g., approximation error, high variance) necessitate using blocks in the range of 6 to 20 steps, rather than the theoretically suggested minimum? Does this suggest a fundamental limitation of using finite-capacity function approximators (like the GRU/MLP encoder) to capture the non-parametric identifiability conditions (Assumptions 2 and 3)?

- Questions on Algorithmic Design

3. Function and Tuning of the Contrastive Improvement Loss (L_{rel}): The refinement stage uses a contrastive improvement loss $L_{\text {rel }}=$ softplus $\left(\log L_{\text {post }}-\log s g\left(L_{\text {prior }}\right)+m\right)$ to encourage the posterior latent (\hat{c}_post) to produce better reconstructions than the prior latent (\hat{c}_prior). This is crucial for avoiding posterior collapse, a common issue in VAE-based latent modeling. What is the impact of the margin hyperparameter m and the weighting coefficient λ_rel on the training process, particularly regarding the trade-off between maximizing reconstruction accuracy (low L_post) and maintaining latent expressiveness? Have the authors performed an ablation on m to demonstrate its necessity?

4. Impact of Latent Dimensions: The details provided focus on the architecture (GRU encoder, MLP decoders) but not the specific dimensionality of the inferred latent context C for each environment. How does the choice of latent dimension d_c affect: (i) the satisfaction of the identifiability assumptions (e.g., Assumption 3 requires sufficient variability induced by c), and (ii) the required size of the minimal block T_x? Since the model captures high-level factors, rewards, or even latent actions, the complexity of c likely varies widely.

- Clarification on the Optimization Roles of the Denoise-and-Refine Losses

5. Parameter Update Paths: Please clarify which network parameters (θ for ϵ_θ, ψ for q_ψ, or ϕ for p_ϕ) are updated by which loss terms. Does minimizing L_post (reconstruction using the posterior latent c_post) update only θ, or does it also backpropagate through c_post to update the posterior encoder ψ? Similarly, does minimizing L_prior update θ and the prior network ϕ?

6. Role of Stop-Gradient in L_{rel}: The contrastive loss L_rel uses a stop-gradient on L_prior(logsg(L_prior)). This typically means L_rel is designed to primarily improve the posterior network q_ψ by pushing c_post towards a representation that achieves better reconstruction compared to a frozen reconstruction accuracy baseline derived from the prior c_prior. Please confirm whether the optimization signal derived from L_rel is used solely to update the posterior encoder ψ, or if it also affects the denoising network θ.

- Questions on Latent Correlation and Prior Quality

7. Sensitivity to Latent Dynamics: The prior p_ϕ(c_t∣c_t−1) relies on the Markovian assumption of the latent context c. In cases where the latent dynamics are highly stochastic or non-Markovian (i.e., c_t is weakly dependent on c_t−1), the prior estimate c_prior may be very poor. Does the overall framework, particularly the contrastive loss L_rel, remain effective when the latent temporal correlation is weak, or is Ada-Diffuser implicitly relying on environments where the latent context exhibits strong temporal smoothness?

8. Prior Collapse Mitigation: Since Stage 1 is an offline VAE optimization using L_ELBO, the term DKL(q_ψ||p_ϕ) serves to regularize the posterior towards the prior. Given the introduction of L_rel in Stage 2 (Causal Diffusion Model), which aims to make the posterior estimate significantly better than the prior estimate, how do the authors ensure that this competitive pressure does not induce posterior collapse (where the VAE minimizes the KL term to zero, ignoring the observations) in Stage 1, or that the training stages remain balanced?

- Detailed Questions on Autoregressive Denoising and Complexity

9. Sequential Steps vs. Total Diffusion Steps: Standard diffusion models perform K noise reduction steps. The autoregressive denoising process described appears to perform a sequential denoising across T time steps. If the trajectory length T is large (e.g., T=32), does the overall inference procedure involve T×K total denoising evaluations, or is the process structured such that the total number of ϵ_θ evaluations remains close to the standard K? Specifically, how does the sequential nature described in Equation (2) translate into the number of required calls to ϵ_θ(x^k, k, c)?

10. Computational Overhead of Zig-Zag: The ablation shows that using a Picard-accelerated variant reduces inference time (e.g., 1.51s → 1.15s for Cheetah). Given that the zig-zag sampling scheme requires an additional "update" step where the posterior latent c_post is calculated using a predicted noisy next step x_{t+1}^{k_2}, what is the approximate percentage of the total inference time consumed specifically by the latent inference and refinement steps (non-denoising operations) in the default zig-zag sampler?

### Soundness
3

### Presentation
3

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
The paper proposes Ada‑Diffuser, a diffusion‑based framework for decision‑making that explicitly models time‑varying latent factors influencing dynamics and/or rewards. Under a set of injectivity conditions, the posterior over the latent at time t is identified from a short temporal window. This motivates block‑wise latent identification via a VAE prior/posterior with an ELBO and a causal, autoregressive diffusion model.

### Strengths
1. Incorporating explicit latent context into diffusion-based decision models addresses an important gap in decision making under uncertainty. The paper presents a unified approach for planning and policy learning. 

2. The identifiability result provides a principled justification for blockwise latent inference and motivates a denoise-then-refine design. The proof strategy follows the nonparametric identification literature and is clearly presented in the appendix. 

3. The empirical evaluation is broad, spanning (i) latents in dynamics and reward, (ii) latent actions and action-free settings, and (iii) settings without explicit latents.

### Weaknesses
1. The theorem uses $x_{t+1}$ (a future observation) for identifiability, but test-time planning cannot access the future. The paper proposes a zig-zag procedure that uses a predicted future observation during refinement. While ablations show empirical gains, the method would be challenging to quantify the error incurred by using predicted futures for posterior alignment.

2. The discussion of the effectiveness of latents in environments without explicitly designed latents is not fully convincing. Doesn’t expert action noise still imply deterministic dynamics? How does the latent help a policy learned from datasets collected under stochastic behavior policies?

### Questions
1. Can you provide an example of real-world scenarios indicating when Assumptions 2–3 plausibly hold (or fail)?

### Soundness
3

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
This paper proposes Ada-Diffuser, a unified diffusion-based decision-making framework that explicitly learns latent dynamic factors from partial observations and adapts its planning/policy behavior accordingly.
It provides (1) identifiability theory showing latent states can be recovered from short temporal windows, and
(2) a practical architecture combining latent inference + causal diffusion model for planning and control.
Experiments on locomotion and robotic manipulation show improvements in adaptation to changing dynamics, rewards, and even action-free demos.

### Strengths
Novelty & Importance: Explicitly modeling latent dynamics inside diffusion-based decision-making is a meaningful step. Previous decision diffusers or trajectory diffusion methods ignored latent variables.

Theoretical Grounding: The identifiability theorem supports the proposed latent inference mechanism, which adds credibility.

Unified Framework: The modular latent inference + diffusion planner/policy is elegant and works for both planning and policy learning tasks.

Robust Experiments: Demonstrates adaptation to dynamic changes, reward shifts, and even recovering hidden action variables.

### Weaknesses
1. Dataset diversity & realism feel limited.
Experiments are described as locomotion and robotic manipulation benchmarks; however, I didn’t see evidence of large-scale, human demonstration datasets (e.g., MimicGen) or messy real-world logs. Without these, it’s hard to judge robustness of latent inference under noisy observations, embodiment shifts, or human suboptimality. (This is especially relevant since the paper claims adaptation to reward/dynamics changes and action-free demos.) 


2. Suggested text you can paste:
“All evaluations are in simulation-style locomotion/manipulation. It would help to include a more realistic demonstration dataset (e.g., human-collected manipulation or MimicGen-like settings) to substantiate generalization of the latent inference mechanism.”


3. Latent interpretability is under-analyzed.
The method identifies latent variables, but the paper doesn’t really show what they mean in practice (e.g., do they correlate with friction, mass, goal, terrain, tool state?). Some qualitative probes or counterfactual rollouts would strengthen the claim that the model captures meaningful hidden structure rather than just serving as an extra embedding. (The theory guarantees identifiability up to an invertible transform; interpretability needs separate evidence.) 


4. Assumptions may be optimistic for real deployments.
Assumption 3 requires a distinguishability condition on a second-order kernel ratio over adjacent times; the paper argues it’s “mild,” but in high-noise real data or partial coverage, that separability can fail. It’d be helpful to discuss failure modes or provide stress tests when the assumptions are violated. 


5. Compute and latency reporting are thin.
Given the added latent-inference block plus causal diffusion, it would be good to see training/inference cost, planning latency (ms), and how the cost compares to strong baselines (e.g., Decision Diffuser variants, model-based world models). No clear numbers are reported alongside the main claims; a wall-clock profile would help practitioners decide whether the approach is deployable.


6. Baselines could be broader and more diagnostic.
The text positions the method within diffusion-based decision-making, but I’d expect latent-variable or POMDP-aware baselines as well (e.g., strong world-model policies with explicit belief tracking) to isolate the contribution of latent inference inside diffusion. This would clarify whether the gains come from the diffusion prior or from better latent modeling per se.


7. Ablations on the latent module are missing/limited.
Since the contribution hinges on the latent factor identification block, I’d like to see ablations like: (a) remove the latent module; (b) freeze latents; (c) mis-specify latent dimension; (d) shorter/longer temporal windows for identifiability. This would show the sensitivity and where the modeling actually helps.


8. Action-free demonstration results need stronger evidence.
The abstract claims recovery of hidden action variables from action-free demos. That’s a big promise; I couldn’t find a strong, quantitative study measuring action recovery accuracy or downstream policy benefits against alternatives (e.g., inverse dynamics baselines). Please add a clear metric and a head-to-head comparison. 


9. Generalization under distribution shift is unclear.
If the environment’s latent process changes abruptly (e.g., reward re-weighting, contact discontinuities), does the model adapt online in a few steps, or does it require re-training? Some online adaptation or few-shot tests would clarify how practical the approach is.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents Ada-Diffuser, a diffusion-based sequence modeling framework for decision-making tasks. The proposed method has two stages: the first stage extracts latent variables that characterize environmental dynamics from minimal observations, through variational inference. In the second stage, the inferred latent factors will be used to condition the autoregressive denoising process of a diffusion planner or policy. During inference, the framework adopts a zig-zag sampling strategy that alternates between sequence denoising and latent refinement to further improve the performance. Experiments show that Ada-Diffuser achieves strong performance across various setups, including those without action labels or without explicit latent structures.

### Strengths
- The framework integrates latent variables that capture environmental dynamics with a causal diffusion model to facilitate better decision making, with theoretical analysis on latent identification with minimal observations
- The experiment extensively evaluates the proposed model across different task domains and settings, consistently highlighting its effectiveness and robustness.
- The paper is well organized and clear to read.

### Weaknesses
Encoding state or state action sequences into latents using reconstruction or ELBO loss has been widely explored by prior work, in which the latent embeddings, a summarization of the input sequence dynamics, have versatile usages to guide decision making, such as skill prior [1] or predictive information [2]. It would be helpful to elaborate on why, intuitively, the extracted latents would work better when coupled with a causal diffusion planner/policy.

In this work, specifically, the latent variable is modeled using (s, a, r) triplets with an extra reward; however, when using the latent variable, the planner generates sequences of states or state-action pairs without explicitly accounting for accumulated rewards. Even though dynamics can be modeled by the latent, the planning performance may remain restricted by the quality of offline demonstrations.

[1] Pertsch. et al. Accelerating Reinforcement Learning with Learned Skill Priors. CoRL. 2020

[2] Zeng et al. Goal-Conditioned Predictive Coding for Offline Reinforcement Learning. NeurIPS 
2023.

### Questions
- Line 422: Missing content “Table A4 and are“

### Soundness
3

### Presentation
2

### Contribution
2
