# Thompson Sampling via Fine-Tuning of LLMs

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Bayesian optimization in large unstructured discrete spaces is often hindered by the computational cost of maximizing acquisition functions due to the absence of gradients. We propose a scalable alternative based on Thompson sampling that eliminates the need for acquisition function maximization by directly parameterizing the probability that a candidate yields the maximum reward. Our approach, *Thompson Sampling via Fine-Tuning* (ToSFiT) leverages the prior knowledge embedded in prompt-conditioned large language models, and incrementally adapts them toward the posterior. Theoretically, we derive a novel regret bound for a variational formulation of Thompson Sampling that matches the strong guarantees of its standard counterpart. Our analysis reveals the critical role of careful adaptation to the posterior probability of maximality—a principle that underpins our ToSFiT algorithm. Empirically, we validate our method on three diverse tasks: FAQ response refinement, thermally stable protein search, and quantum circuit design. Within a collection of methods covering in-context Bayesian optimization, reinforcement learning, and evolutionary search, ToSFiT exhibits both state-of-the-art sample efficiency and computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses Bayesian optimization in large unstructured discrete spaces by proposing ToSFiT (Thompson Sampling via Fine-Tuning), which parameterizes the probability of maximality (PoM) directly using large language models instead of performing expensive acquisition function maximization. The authors establish a novel regret bound for variational Thompson sampling that scales with the maximal information gain rather than the domain size, and demonstrate their approach on three diverse tasks: FAQ refinement, protein design, and quantum circuit optimization.

### Strengths
- Addresses a genuine problem: acquisition function maximization in large discrete spaces is prohibitively expensive
- The combination of pre-trained LLMs with principled Bayesian optimization is timely and well-motivated
- The improved regret bound is a solid theoretical contribution that advances understanding of VBOS
- Demonstrates the approach on three genuinely different domains (language, proteins, quantum circuits)

### Weaknesses
-  The performance of TOSFIT is fundamentally tied to the quality of the fixed feature map (i.e., the embeddings from a pre-trained model). If the pre-trained embeddings do not capture the features relevant to the reward function, the GP model may struggle, limiting the overall performance regardless of the policy optimization.
- Missing comparisons to other recent LLM-based optimization methods (e.g., FIBO is discussed but not compared experimentally), VAE-based methods, or even simpler approaches like evolutionary strategies
- Missing comparisons to other discrete optimization approaches (e.g., evolutionary methods, MCMC-based optimization)
- Limited discussion of when you would vs. wouldn't use this method. When is the 19% overhead worth it vs. just generating more candidates?

### Questions
- Why limit POST-GENERATION TS to 1000 candidates? What happens with 10k or 100k candidates with the same computational budget as ToSFiT?
- All experiments use small models (1.5-1.7B parameters). How does this scale to larger models?
- How does the method scale to longer sequence generation tasks?
- When does TOSFIT fail? When the LLM prior is misaligned, is it possible that TOSFIT will perform worse than random search?
- Have you considered using the LLM to propose multiple diverse candidates per forward pass (e.g., via beam search or sampling), which might improve the gradient estimator?
- Could you use the GP uncertainty to modulate the learning rate or number of fine-tuning steps adaptively?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TOSFIT, a scalable approach to Bayesian Optimization (BO) over large unstructured discrete domains. Instead of maximizing acquisition functions directly, the method parameterizes the probability of maximality (PoM) using a prompt-conditioned large language model (LLM). TOSFIT adapts the pre-trained policy toward the posterior PoM through online fine-tuning guided by a Variational Bayesian Optimistic Sampling (VBOS) objective. The paper derives new theoretical regret bounds that match those of standard Thompson Sampling (TS) and GP-UCB, extends the analysis to approximate VBOS, and demonstrates empirical gains in frequently-asked-questions (FAQ) response refinement, protein search, and quantum circuit design tasks.

### Strengths
The paper proposes a new bridge between BO and LLM, replacing acquisition maximization with posterior-aligned fine-tuning. The regret analysis is solid. In addition, TOSFIT consistently outperforms unguided LLM generation and static candidate-based TS.

### Weaknesses
1. While theoretically rigorous, some derivations (e.g., the gradient of VBOS and its concavity proof) are dense and may be inaccessible to a broader ICLR audience. A high-level intuition section would improve readability.
2. The comparison is limited to two relatively weak baselines (unguided LLM and post-generation TS). More direct comparisons to in-context optimization or gradient-based RL fine-tuning (e.g., RLHF or FIBO) would strengthen empirical claims.
3. It remains unclear whether performance improvements stem primarily from the fine-tuning process or from better use of GP-based uncertainty.
4. Although the paper claims negligible overhead, it would be helpful to report absolute runtime and fine-tuning cost relative to BO iterations.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles the prohibitive cost of Bayesian optimization (BO) in large, unstructured discrete spaces. The authors’ key move is to eliminate expensive acquisition-function maximization over huge domains by directly sampling the next candidate from a parameterized distribution that approximates the posterior probability of maximality (PoM). They instantiate Thompson Sampling as fine-tuning a large language model (tagged TOSFIT) using a prompt-conditioned, pre-trained LLM that is carefully adapted toward PoM. The work builds on Variational Bayesian Optimistic Sampling (VBOS; O’Donoghue & Lattimore, 2021), but particularly strengthens it; they reformulate the analysis to yield a structure-aware regret bound (in terms of information gain) and extend it to approximate VBOS, which directly motivates TOSFIT’s emphasis on pre-trained policy initialization and cautious, PoM-aligned adaptation.

### Strengths
1. The paper is well motivated by both the challenges of BO in large discrete settings and the limitations of existing approaches. The authors demonstrate a solid command of the literature.

2. The algorithm is well argued and clearly presented. Although the work builds on VBOS, the authors provide clear theoretical reformulations and insight driven analysis that justify the development of TOSFIT.

3. The experiments show clear performance gains over the chosen baselines.

4. The mathematical assumptions, propositions, and theorems are clearly stated and organized in the main paper and the supplementary material.

5. The scalability of the proposed methodology is well expounded, with clear guidelines for practice and future research directions.

### Weaknesses
1. The paper could give a clearer background on the core concept and regret formulations behind VBOS before introducing the intimate connection to UCB. While the adaptive UCB exploration bonus is central to the gradient formulation, the explicit bridge from VBOS to regret bounds could be surfaced earlier. VBOS exact regret bound via the optimistic set appears in Proposition 2, but new readers may struggle without an earlier signpost.

2. Fine tuning uses one VBOS gradient ascent step per round $c = 1$. This keeps cost low, but it raises natural questions about the compute versus quality trade off if more steps are used in complex settings.

Minor comment: The choice of a linear GP is well reasoned, but performance depends on the feature embedding. Using a fixed embedding in the experiments limits generalizability. The authors acknowledge this as future work and outline how to address it, which helps mitigate the concern.

### Questions
1. Regarding the choice of $c = 1$ for the fine tuning update, can the authors provide more detail on the compute versus performance trade off?
2. What guided the selection of baselines? Are there other established methods for combinatorial or discrete BO that were considered but not included? A brief rationale for the baseline set would be helpful.

### Soundness
4

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
3

### Summary
This paper proposes a variational Bayesian optimization surrogate (VBOS) objective that provides a tighter bound for Thompson sampling–based optimization. The method enables gradient-based fine-tuning of generative models under non-differentiable and expensive reward functions.

### Strengths
The paper presents an interesting and theoretically grounded idea. The VBOS formulation provides a principled way to incorporate uncertainty into generative model fine-tuning for non-differentiable rewards. The theoretical analysis is sound and detailed.

### Weaknesses
- GP: The paper relies on the assumption that the underlying reward follows a Gaussian process prior. Will this be held for discrete or highly non-smooth reward functions?
- Baselines: while the paper focuses on Bayesian optimization–based baselines, it omits comparisons with standard reinforcement learning (RL) methods such as PPO/GRPO, which can also handle non-differentiable rewards to fine-tune LLMs. (Are there any reasons that PPO/GRPO is not able to be used here?) Empirically, I can see VBOS adds an uncertainty term $\sigma_x$; however, such comparisons would be crucial to demonstrate that the exploration term can contribute beyond simple reward-based fine-tuning.
- Diversity measure: Reward-based fine-tuning sometimes leads to diversity collapse to a narrow high-reward set. Since the proposed VBOS objective explicitly includes an uncertainty term that should encourage exploration, it would be valuable to report how diversity changes during optimization.
- Performance convergence: In Figure 4, ToSEiF appears to achieve higher rewards given the same computational cost. However, it is unclear when the fine-tuning process converges. If I have enough resources, when to stop the fine-tuning? Could the authors provide more complete curves, extending beyond the current range, potentially into regions where ToSEiF’s performance even declines (perhaps in the appendix)? This would help readers better understand the convergence behavior and overall stability of the optimization process.

### Questions
Following the weakness
- For the given tasks, if apply RL methods such as PPO/GRPO, what would be the results? How your methods be better than RL? (or how your methods are correlated with RL?)
- What is the diversity measure during the fine-tuning? Do you have mode-collapse problem?
- How to select the convergence point, if I have enough computation resources?

### Soundness
3

### Presentation
2

### Contribution
3
