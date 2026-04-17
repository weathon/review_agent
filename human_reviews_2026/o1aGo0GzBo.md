# Dynamic-TreeRPO: Breaking the Independent Trajectory Bottleneck with Structured Sampling

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
The integration of Reinforcement Learning (RL) into flow matching models for text-to-image (T2I) generation has driven substantial advances in generation quality.
However, these gains often come at the cost of exhaustive exploration and inefficient sampling strategies due to slight variation in the sampling group.
Building on this insight, we propose Dynamic-TreeRPO, which implements the sliding-window sampling strategy as a tree-structured search with dynamic noise intensities along depth.
We perform GRPO-guided optimization and constrained Stochastic Differential Equation (SDE) sampling within this tree structure. By sharing prefix paths of the tree, our design effectively amortizes the computational overhead of trajectory search.
With well-designed noise intensities for each tree layer, Dynamic-TreeRPO can enhance the variation of exploration without any extra computational cost.
Furthermore, we seamlessly integrate Supervised Fine-Tuning (SFT) and RL paradigm within Dynamic-TreeRPO to construct our proposed LayerTuning-RL, reformulating the loss function of SFT as a dynamically weighted Progress Reward Model (PRM) rather than a separate pretraining method.
By associating this weighted PRM with dynamic-adaptive clipping bounds, the disruption of exploration process in Dynamic-TreeRPO is avoided.
Benefiting from the tree-structured sampling and the LayerTuning-RL paradigm, our model dynamically explores a diverse search space along effective directions.
Compared to existing baselines, our approach demonstrates significant superiority in terms of semantic consistency, visual fidelity, and human preference alignment on established benchmarks, including HPS-v2.1, PickScore, and ImageReward. In particular, our model outperforms SoTA by 4.9\%, 5.91\%, and 8.66\% on those benchmarks, respectively, while improving the training efficiency by nearly 50\%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Dynamic-TreeGRPO, a tree-structured search with dynamic noise, to address computational inefficiency and to separate training paradigms when applying RLHF to flow-based generative models. The proposed method gradually expands the search tree using a common trunk to reduce computational costs. Experiments demonstrate efficiency and state-of-the-art performance on benchmarks for human preference alignment, semantic consistency, and visual quality.

### Strengths
- The application of a tree-search-based approach in the image generation domain, to the best of my knowledge, is novel. Although similar approaches exist in other domains (see weaknesses).
- The proposed new training paradigm can provide a unified approach for SFT and RL.
- The computational cost compared to some baselines was demonstrated to be improved.

### Weaknesses
- The claims on improving efficiency are not well-motivated in the paper, nor are they fairly and comprehensively benchmarked against existing baselines. The paper claims a 50% efficiency improvement. Table 1 shows its iteration time (151s) is a ~37% reduction from MixGRPO (240s) and a ~53% reduction from DanceGRPO (323s). However, it is only a ~9% reduction from MixGRPO-Flash (166s). The source of the "50%" claim could be clearer, and the practical speed-up over other optimized baselines seems more modest. Indeed, as a tree-search-based approach, the number of nodes grows exponentially with tree depth, leading to significantly more training/finetuning time. It is indeed that, compared with DanceGRPO, the efficiency might be improved as only half of the nodes are needed, but compared with vanilla GRPO, e.g., [1], the number of forward passes in a single data-point optimization is still significantly larger due to the need to expand the tree. As [1] was never compared as a baseline, it remains unclear to me whether the proposed approach is truly efficient.

- Tree-search-based RL approaches for generative models are not uncommon. For example, [2] combined tree-search with diffusion for planning in RL tasks. Fundamentally, the proposed TreeGRPO also follows the same spirit.

- The LayerTuning-RL component uses the best sample from the current group ($V_t^*$) as the "pseudo target" for its SFT loss. What happens if the entire group of samples is poor or low-quality? The model would essentially be learning to imitate the "best of the worst," which could lead to a local optimum or mode collapse rather than true quality improvement.

- The authors use a binary tree of a fixed depth ($d=4$). How was this structure chosen? How does the performance change with deeper or wider (e.g., ternary) trees? A fixed, shallow tree might limit the complexity of the exploration, and the optimal depth is likely task-dependent.

- The paper proposes a linear increase in noise intensity based on tree depth (Equation 6). This is an interesting heuristic, but is it the most effective? Were other functions (e.g., exponential, logarithmic) explored to see if they could provide a better exploration/exploitation trade-off?

- The number of RL baselines compared in the paper is too limited to demonstrate the true effectiveness of the proposed approach compared to existing baselines. For example, [1] is a closely related baseline that directly applies GRPO for flow-based generative models. In addition, as various DPO (direct preference optimization) approaches already exist and achieve decent results, at least one of them should be compared.

[1] Liu, Jie, et al. "Flow-grpo: Training flow matching models via online rl." arXiv preprint arXiv:2505.05470 (2025).

[2] Yoon, Jaesik, et al. "Monte carlo tree diffusion for system 2 planning." arXiv preprint arXiv:2502.07202 (2025).

### Questions
Please refer to the weaknesses for more details. In addition:

- Can the authors discuss the connection of the proposed method to existing tree-search-based methods like [2]?
- Can you give an intuitive interpretation of why LayerTuning-RL can lead to better performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Dynamic-TreeRPO, an RL training framework for flow-matching T2I models that (i) replaces independent trajectory sampling with a tree-structured sampler using a sliding window where SDE noise is injected, sharing prefix computations for efficiency; (ii) introduces dynamic, reward-conditioned clipping for GRPO; and (iii) integrates SFT as a LayerTuning-RL auxiliary loss, treating SFT as a progress-reward model (PRM) that supervises nodes along the tree. Claimed benefits include ~50% faster training iterations and higher scores on HPS-v2.1, PickScore, and ImageReward, with experiments primarily on FLUX.1-dev and HPDv2.1 prompts.

### Strengths
Clear engineering idea: the tree sampler amortizes compute by sharing prefixes; ablations isolate contributions (Dynamic-Tree, Dynamic-Clipping, LayerTuning-RL).
Reported efficiency gains with stable reward curves and consistent improvements on automatic preference metrics.

### Weaknesses
Limited conceptual novelty vs. recent GRPO variants.
The tree + sliding-window SDE is close in spirit to MixGRPO’s sliding-window stochasticity and TempFlow-GRPO’s trajectory branching/credit assignment; Dynamic-TreeRPO reads as an engineering re-packaging rather than a new optimization principle. A convincing distinction (either theoretically or via stronger baselines) is missing.

Optimization is confined to SDE steps in the window.
By design, RL updates only occur inside the tree interval; outside it, ODE sampling is used and not optimized. This structurally restricts policy learning to a subset of timesteps and may mis-assign credit across the generative trajectory. The objective in Eq. (9) explicitly sums GRPO terms only over $t\inS_{tree}$. Strong evidence is needed that this does not harm global alignment or late-stage refinements.

Ambiguous/dated reward-modeling and missing reward baselines.
The paper reports evaluation with HPS-v2.1, PickScore, and ImageReward, but does not clearly state which reward model(s) drive training nor why stronger contemporary RMs are excluded. Modern alternatives such as VisionReward (fine-grained, multi-dimensional), or VLM-based/zero-shot RMs are not used or compared, weakening claims about alignment under stronger judges.

SFT/PRM supervision is under-specified and risks label leakage.
LayerTuning-RL defines the SFT target at each step as the max-reward path’s prediction $V^*_t$, i.e., model-generated pseudo-labels, not human-labeled images. The paper does not clarify where labeled images come from (if any), how pseudo-label bias is controlled, or how this differs from reward-weighted regression. The claim that this mitigates reward hacking without KL is unsubstantiated beyond small-scale metrics.

Missing or incomplete baseline comparisons.
Beyond Flux, DanceGRPO, and MixGRPO, several relevant baselines are absent:
Flow-GRPO (first online RL for flow matching) and TempFlow-GRPO (temporally aware branching). 
Diffusion-DPO (non-RL preference optimization) and DPOK/DPoK-style methods for T2I alignment. 
DDPO / RL for diffusion with explicit reward design, which is a strong alternative alignment route. 
Without these, it’s hard to attribute the reported gains to the tree structure rather than to known improvements from recent GRPO/DPO lines.

Evaluation scale and generality are limited.
Training uses a subset of HPDv2.1 prompts and 400 test prompts; no human A/B tests, no safety/failure analyses, and no robustness to adversarial prompts or typography tasks. This makes the “SoTA” claim fragile.

Dynamic clipping lacks principled comparison to known variants.
The paper positions reward-conditioned clipping as stabilizing, yet omits direct comparisons with DAPO/DCPO-style adaptive clipping (strong baselines in RLHF), even as related work acknowledges them conceptually. A head-to-head would clarify whether reward-conditioning brings new benefits beyond token-probability-aware schemes.

### Questions
Exactly which RM(s) produce the training rewards (not just evaluators)? Why not include VisionReward or VLM-based RMs, and what happens if you swap the judge (HPS→VisionReward, etc.)? Please add ablations with multiple modern RMs and report reward-hacking diagnostics (diversity, FID, failure rate).

SFT supervision source: LayerTuning-RL uses $V^*_t$ from the best in-group sample as the SFT target. Does any external labeled (image,label) data play a role? If not, how do you prevent confirmation bias from pseudo-labels? Please clarify data provenance, label construction, and compare against standard reward-weighted likelihood / ReFL.

Why is optimization restricted to $S_{tree}$? Provide experiments where you (a) shift the window across timesteps, (b) expand to all steps, and (c) vary the branching schedule (vs. TempFlow-GRPO). Report effects on long-horizon credit assignment and text rendering tasks.

Please add Flow-GRPO, TempFlow-GRPO, Diffusion-DPO, and DPOK as baselines under matched compute/hardware, plus ablations against adaptive-clipping methods (e.g., DCPO/DAPO families). Without these, it’s unclear that Dynamic-TreeRPO itself—not broader training choices—drives the gains.

### Soundness
2

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
3

### Summary
This paper introduces Dynamic-TreeRPO, a reinforcement learning (RL) framework designed for flow-matching text-to-image models. The method utilizes a sliding-window tree-structured SDE process with dynamic noise intensities, thereby reducing the computational overhead. Furthermore, the authors present a novel training paradigm named LayerTuning-RL, which unifies supervised fine-tuning (SFT) and reinforcement learning into a single optimization objective. Experiments demonstrate that the proposed method improves training efficiency and achieves superior performance compared to previous methods (DanceGRPO and MixGRPO).

### Strengths
1. Improving the efficiency of GRPO-based RL framework in flow matching text-to-image models is an important research topic. The proposed tree-structured search method provides a reasonable and well-motivated solution.
2. The idea of LayerTuning-RL, i.e., using the trajectory that obtains the highest reward within each group as the supervising target, is conceptually appealing and offers an interesting perspective on integrating SFT and RL.

### Weaknesses
The proposed method introduces a number of hyperparameters, such as the depth of trajectory trees $d$, noise growth magnitude $\beta$ in Equation (6), reward sensitivity factor $\eta$ in Equation (11), clipping threshoulds $\varepsilon_\text{low}$ and $\varepsilon_\text{high}$ in Equation (11), and loss coefficient $\lambda$ in Equation (13). These hyperparameters may require careful tuning, imposing additional burdens in practice.

### Questions
1. How sensitive is the proposed method to the hyperparameters mentioned in the “Weaknesses” section?
2. How is the tree depth $d$ determined? Is it feasible to use a wider tree structure instead of a binary tree?
3. What motivates the design of the function $g_t(k)$ in Equation (6)? More specifically, why is it designed to linearly increase with the normalized depth? Have you tried other formulations such as those decrease with depth?
4. Can the technique of using different noise intensities be applied to MixGRPO and DanceGRPO, to enhance their diversity and performance?

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
4

### Summary
Dynamic-TreeRPO is a tree-structured reinforcement learning framework for flow matching models that uses a binary tree and LayerTuning-RL to jointly enhance sampling efficiency, exploration diversity, and achieve state-of-the-art performance across multiple benchmarks.

### Strengths
Improving sampling efficiency is crucial in reinforcement learning tasks. The proposed binary-tree-based sampling strategy effectively accelerates sampling while maintaining diversity, representing a practical and well-motivated contribution.

### Weaknesses
The proposed algorithm consists of three components, with the binary tree sampling serving as its core idea. However, from Table 2, the performance gains from the other two improvements are also considerable. Combined with the results in Table 1, the performance of the binary tree and MixGRPO variants appears comparable. It is unclear whether the iteration time reported in Table 1 accounts only for the binary tree module or also includes the SL component, which would also increase training time.

### Questions
What is the starting step of the denoising time range parameter? 

If only the binary tree sampling is used, would the in-group diversity decrease compared to the MixGRPO formulation? It might be valuable to quantify this with appropriate diversity metrics to support the claim.

### Soundness
2

### Presentation
2

### Contribution
3
