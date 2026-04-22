# Inference-Time Alignment Control for Diffusion Models with Reinforcement Learning Guidance

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Denoising-based generative models, particularly diffusion and flow matching algorithms, have achieved remarkable success. However, aligning their output distributions with complex downstream objectives remains challenging. While reinforcement learning (RL) fine-tuning methods, inspired by advances in RL from human feedback (RLHF) for large language models, have been adapted to these generative frameworks, current RL approaches offer limited flexibility in controlling alignment strength after fine-tuning.
In this work, we view RL fine-tuning for diffusion models through the lens of stochastic differential equations and implicit reward conditioning. 
We introduce Reinforcement Learning Guidance (RLG), an inference-time method that adapts Classifier-Free Guidance (CFG) by combining the outputs of the base and RL fine-tuned models via a geometric average. 
Our theoretical analysis shows that RLG's guidance scale is mathematically equivalent to adjusting the KL-regularization coefficient in standard RL objectives, enabling dynamic control over the alignment-quality trade-off without further training. 
Extensive experiments demonstrate that RLG consistently improves the performance of RL fine-tuned models across various architectures, RL algorithms, and downstream tasks, including human preferences, compositional control, compressibility, and text rendering. Furthermore, RLG supports both interpolation and extrapolation, thereby offering unprecedented flexibility in controlling generative alignment. Our approach provides a practical and theoretically sound solution for enhancing and controlling diffusion model alignment at inference. The source code for RLG is available in the anonymous repository: https://anonymous.4open.science/r/Reinforcement-learning-guidance-7B5A/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed reinforcement learning guidance (RLG), a concept inspired by classifier-free guidance (CFG). RLG works by linearly interpolating the scores from the base model and the  RL-tuned model, offering a simple yet effective approach of combining two models during inference time. The authors demonstrate that tuning the RLG coefficient can lead to better perfomrance than the original fine-tuned model alone.

### Strengths
- The idea of RLG is easy to understand, simple, and clearly motivated. To the best of my knowledge, it is also the first work to adopt such an approach for inference-time optimization (apart from concurrent work mentioned in the paper).
- The authors provide a theoretical analysis of the equivalence between the RLG coefficient $w$ and the DPO KL regularization coefficient $\beta$, building an important connection between the two approaches.
- The experimental results demonstrate superior performance over the baseline DPO approach, which can be considered as a special case where $w=1$.

### Weaknesses
- As the proposed approach relies on the RL-tuned model, why is $w=1$ Not Optimal? The paper's strongest result is that $w > 1$ (extrapolation) consistently outperforms $ w = 1$ (the model they actually trained). This implies that the standard RL fine-tuning process (DPO, GRPO) is systematically under-optimizing the objective. Why does this happen? Is the fixed KL-regularization in the original training too strong, or do the optimizers fail to reach the true optimum? The paper doesn't explore this fundamental implication.

- Following the previous weakness, the authors admit their core theoretical link (that $w$ controls $\beta/w$) assumes a standard KL-regularized objective. However, they also show RLG works exceptionally well for GRPO, an algorithm that (as they note in the limitations) does not necessarily converge to that same theoretical optimum. This means RLG works in practice even when its main theoretical justification doesn't apply. Why does it still work so well for GRPO?

- RLG requires running two models (the base and the RL-tuned model) during every sampling step, which has a higher memory and (potentially) computational footprint at inference time compared to just using the single fine-tuned model. The paper doesn't discuss this practical cost. 

- There are some minor formatting issues. For example, table captions should be placed above. Additionally, the authors use $v_{RL}$ and $v_\theta$ randomly in the manuscript (e.g., Algorithm 1), which may cause confusion.

### Questions
Please refer to the questions in the Weaknesses section above. In addition:
- The paper shows that as one objective (like OCR accuracy) is pushed higher with RLG, another objective (like Aesthetic Score) can decrease (see Table 3). This suggests that at high $w$ scales, the model may be "reward hacking"—overfitting to the specific reward model at the expense of general image quality. The paper notes this trade-off but doesn't explore the limits of extrapolation. At what $w$ value do images become out-of-distribution or nonsensical, similar to what happens with very high CFG scales?

### Soundness
2

### Presentation
2

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
The paper studies the fine-tuning of a pretrained model to maximize the reward with KL regularization. 

Specifically, it develops a Reinforcement Learning Guidance (RLG) approach, which is based on the Classifier Free Guidance (CFG). The main idea is to compute a weighted geometric average of the outputs from the base model and the RL fine-tuned model. Moreover it studies the connection between this weighted averaging and the KL regularization coefficient in RL fine-tuning.

### Strengths
Extensive experiments demonstrate consistent improvements of the RLG approach ---  the performance of RL fine-tuned models across various architectures, RL algorithms, and down-stream tasks.

### Weaknesses
The contribution of the paper appears quite limited. In particular, as acknowledged (at the end of the paper), a) the approach does not ensure the (target) marginal distribution, and b) RLG's connection to the KL-regularization coefficient is premised on sampling from the optimal policy, which appears unrealistic and unjustified. 

Indeed there are many studies in the field, including several papers currently under review with this year's ICLR, developing similar approaches to preference alignment and reward optimization, and specifically via introducing various weighting factors (e.g., those motivated by importance sampling). type of optimization). Those studies appear to use substantially more sophisticated techniques and dig much deeper into underlying theories.

### Questions
Any comments/explanation regarding why RLG appears to work well (in the numerical experiments) despite the two weaknesses mentioned above?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Reinforcement Learning Guidance (RLG), an inference‑time control method for denoising‑based generators (diffusion and flow‑matching). Given a base model and an RL‑aligned model trained toward a downstream objective, RLG linearly combines their scores/velocities during sampling, with a scalar weight $w$ (Alg. 1). The paper argues theoretically that sampling with weight $w$ is equivalent to sampling from a distribution proportional to $p_{\text{ref}}^{\,1-w} p_{\theta}^{\,w}$, and—under the usual KL‑regularized RL objective—this corresponds to reducing the effective KL coefficient from $\beta$ to $\beta/w$. Empirically, RLG is claimed to (i) improve preference metrics (Aesthetic, ImageReward, PickScore) on SD1.5/DPO, SDXL/SPO and SD3.5‑M/GRPO; (ii) raise compositional accuracy on GenEval; (iii) increase OCR accuracy for text rendering; (iv) provide a tunable “slider” for image compressibility; and (v) help inpainting and personalization, all without extra training.

### Strengths
1. **Simple, training‑free control**: The method only needs the base and RL‑tuned models and a scalar $w$. The implementation is trivial (one extra forward pass and a weighted sum), yet it gives a convenient **post‑hoc** control over alignment strength.
2. **Unified score/velocity perspective**: The paper writes RLG both on **scores** and **flow velocities** and shows their equivalence via the SDE/ODE unification. This makes the method broadly applicable to DDPM/DDIM‑style diffusion and Flow Matching families.

### Weaknesses
- The core step $\hat s = (1-w)s_{\text{ref}} + w s_{\theta} = \nabla \log\big(p_{\text{ref}}^{1-w} p_{\theta}^{w}\big)$ is standard, but it does not imply the sampler actually draws from that target density under common discretizations; recent analyses showed CFG is a predictor‑corrector heuristic and typically does not sample $p(x|c)^{\gamma} p(x)^{1-\gamma}$. This limitation carries over to RLG, especially when $w>1$ (negative exponent on $p_{\text{ref}}$) [1][2].
- Equivalence to $\beta/w$ depends on the RL objective: The $\beta \mapsto \beta/w$ argument assumes the optimal KL‑regularized policy $p_{\theta} \propto p_{\text{ref}} \exp(R/\beta)$. The paper evaluates on GRPO and DPO models as well, where this assumption may not hold exactly; thus the “effective‑$\beta$” interpretation is at best approximate for these cases [3].
- Tables report mean scores and “win‑rates” but no confidence intervals/variance estimates or significance tests. Several gains are modest (e.g., ImageReward often plateaus or slightly drops at large $w$), so the practical significance is unclear without error bars and multiple seeds.
- Interplay with CFG left largely unexplored: RLG is applied on top of a fixed CFG scale but the coupling between CFG and RLG can be non‑trivial; without ablations over (CFG, $w$) grids, it is hard to know when each dominates or conflicts. [4]

[1] Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts, https://arxiv.org/abs/2503.02819

[2] Classifier-Free Guidance is a Predictor-Corrector, https://arxiv.org/abs/2408.09000

[3] Direct Preference Optimization: Your Language Model is Secretly a Reward Model, https://arxiv.org/abs/2305.18290

[4] Guided Flows for Generative Modeling and Decision Making, https://arxiv.org/abs/2311.13443

### Questions
1. Can you clarify whether your sampler with RLG actually converges to $p_{\text{ref}}^{1-w} p_{\theta}^{w}$? Have you tried Feynman–Kac correctors or other product‑of‑experts correctors to make the interpolation distributionally sound?
2. When $\beta/w$ holds: For DPO and GRPO‑tuned models, what is the theoretically correct “effective $\beta$” story, if any? Can you include a small controlled experiment showing where the $\beta/w$ prediction fails (or holds approximately)?
3. Could RLG extend to multi‑objective blending (e.g., combine two RL‑aligned experts with the base at inference)?
4. How do RLG's latency and energy consumption compare to single-model sampling? RLG doubles forward passes per step (base + RL policy). The paper fixes 20 steps, which masks the runtime cost relative to the RL‑only baseline.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper reinterprets RL fine-tuning for diffusion models through the lens of stochastic differential equations and guidance/reward conditioning. The authors propose Reinforcement Learning Guidance (RLG): at inference time, one uses both the base diffusion model and an RL-fine-tuned version of it; then one combines their outputs via a geometric average. It has been shown that adjusting the guidance scale in RLG corresponds mathematically to adjusting a KL-regularization coefficient in the RL objective. Experiments have been conducted on SD3.5 and SDXL.

### Strengths
The paper is well-written, and I mostly enjoyed reading it. The main contribution/strength is to provide flexibility of inference-time control in fine-tuning. The numerical experiments also support the proposed approach.

### Weaknesses
One problem is the sensitivity of the weight $w_{RL}$: it seems (Table 1) that larger the weight is, the better the performance is. However,  large $w_{RL}$ may lead to distributional shift, and it is not clear to me how this can be interpreted, or how to choose the hyperparameter.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
