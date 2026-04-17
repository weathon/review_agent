# Neural Bayesian Filtering

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
We present Neural Bayesian Filtering (NBF), an algorithm for maintaining distributions over hidden states, called beliefs, in partially observable systems. NBF is trained to find a good latent representation of the beliefs induced by a task. It maps beliefs to fixed-length embedding vectors, which condition generative models for sampling. During filtering, particle-style updates compute posteriors in this embedding space using incoming observations and the environment’s dynamics. NBF combines the computational efficiency of classical filters with the expressiveness of deep generative models—tracking rapidly shifting, multimodal beliefs
while mitigating the risk of particle impoverishment. We validate NBF in state estimation tasks in three partially observable environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces NBF which is a new way to do Bayesian filtering when the system is only partially observable. Instead of tracking the belief state with huge num of particles like a normal particle filter, they learn a compact embedding $\theta$ that represents the whole belief distribution. This embedding is built so it does not depend on the order or number of samples and can capture multi-modal posteriors.

At each step, NBF draws new particles from a generative model conditioned on this embedding, updates them using the transition and observation models, and then re-embeds the results to get the new $\theta$. This avoids the usual particle impoverishment issue and makes filtering more efficient. They test NBF on small partially observable tasks. The results show that NBF can stay close to the true posterior, while using fewer particles than a standard PF.

### Strengths
- The paper builds a permutation and size invariant embedding for belief states, so the order or number of particles doesn’t matter. It’s a clean way to represent complex, even multi-modal posteriors as a single latent vector.

- Instead of doing the usual particle-filter resampling that kills diversity, the paper redraw particles from a learned flow model conditioned on the current embedding that keeps the diversity of samples.

- The paper includes a short proof showing that, under reasonable conditions, the embedding update converges to the true posterior.

- Able to handle discrete states cases.

### Weaknesses
- The paper use a particle-filter–style update step when incorporating new observations, but it’s not super clear why that specific choice. PFs are flexible and model-free, sure, but they also bring sampling noise and inefficiency. It would be nice to see some ablation or comparison with other update mechanisms to show this choice outperformance. Just example: using a learned parametric update network or simpler probabilistic approximations (e.g. Gaussian or flow-based direct updates) instead of your PF, or using your proposed recurrent model in update process but still keeping $\theta$ embedding. Showing how these variants perform would help justify sticking with the PF-style update.

- The chosen baselines (vanilla particle filter and the Approx Beliefs oracle) are fine but not enough. To really compare this work against the SOTA, it would help to include some of the newer learned filtering methods that also approximate posteriors in high-dimensional or multimodal settings. 
For example, recent works like [1],[2] and [3] all solve posterior approximation by generative models but with different parameterizations. Including two or three of them would make the evaluation more complete.

- As mentioned in the discussion, the current experiments are mostly toy setups and simple. Comparing to similar recent papers in top venues, adding at least one higher-dimensional or real-world experiment would strengthen the work.

- The paper lacks an easy-to-follow code setup or README-style instructions. Although it mentions that the code will be released upon acceptance, it would be nice to have an anonymous link or at least some pseudo-code included in the paper to make the method easier to follow.

[1] - Wan, Ziyu, and Lin Zhao. "DiffPF: Differentiable Particle Filtering with Generative Sampling via Conditional Diffusion Models." arXiv preprint arXiv:2507.15716 (2025).

[2] - Chen, Xiongjie, and Yunpeng Li. "Normalizing Flow-Based Differentiable Particle Filters." IEEE Transactions on Signal Processing (2024).

[3] - Younis, Ali, and Erik Sudderth. "Learning to be Smooth: An End-to-End Differentiable Particle Smoother." Advances in Neural Information Processing Systems 37 (2024): 7125-7155.

### Questions
Thank you for the draft and all of the efforts. I have some questions regarding the paper:

- Can you justify why this specific update scheme fits NBF best? Why not a few ablation test with other parameterizations to show that PF is indeed the most effective choice?

- what happens if you replace the PF step with a simple parametric or recurrent update but still using the same embedding $\theta$? How would that work, did you this try in your exps? It would help clarify how much of the gain actually comes from the PF structure versus the embedding itself.

- Similar papers in top venues usually include at least one high-dimensional or real-world experiment. How does your model handle such cases, and what modifications would be needed to make it works in higher dimensions? If not adding such an experiment, could you clearly mention these in your discussion?

- The idea of using a multimodal embedding $\theta$ as a kinda universal posterior generator is interesting. But how flexible and accurate is it in practice? I.e. if you train task-specific embedding networks instead of a shared one, do you see a large performance improvement or just a marginal gain? Did you try this in your exps?

- In Figure 4 (25 states), it looks like the particle filter performs even better than the Approx Beliefs baseline in the first few steps, which is a bit unexpected since Approx Beliefs directly uses samples from $p(x)$. Any explanation for this?

### Soundness
2

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
3

### Summary
This work proposes a particle filter (PF) type algorithm which, instead of resampling, maps the set of weighted samples to a latent embedding at each time step, and then "refreshes" the particle system by replacing the weighted particles by a new set of evenly weighted particles generated by a normalising flow parametrised by the embedding. The embedding seems to be trained assuming that IID samples from the filtering distributions are available.

The proposed method seems to me to be a type of "regularised particle filter" (e.g., [1]). Regularised particle filters have long been favoured by some researchers and practitioners as a way of overcoming the particle degeneracy problem. They approximate the weighted sample by some (e.g., continuous) distribution and then sample from this approximation instead of resampling from the original weighted sample.

[1] Musso, C., Oudjane, N., & Le Gland, F. (2001). Improving regularised particle filters. In Sequential Monte Carlo methods in practice (pp. 247-271). New York, NY: Springer New York.

### Strengths
Here, the authors use latent embeddings (and normalising flows) for constructing (and sampling from) the regularisation in a regularised particle filter which seems novel to me and could potentially be powerful. Although I am not necessarily following all the related literature. The theoretical results seem correct although they rely on very strong assumptions.

### Weaknesses
**Significance.**

The theoretical results (almost sure convergence and $L_p$ error bounds) are likely correct. However, they are not surprising under the very strong assumptions (and more general results under weaker assumptions are well known in the particle filtering literature). I don't think a methodology paper like this necessarily needs extensive theory to justify publication in ICLR. However, I will note that the $L_p$ error bound does not prove stability, i.e., the error bound likely grows with $t$. And it would be crucial to show that the method is stabie (i.e., that error do not accumulate over time) under more realistic assumptions on the embedding.


**Clarity:**

The paper has problems with clarity which are so severe that there are parts of the methodology and numerical results that I cannot follow/verify:

1. There are a large number of undefined or poorly defined symbols and terms: $\Delta X$, $\Delta Y$, $d$, $T_G$, $H_G$, $K$, $N$ (= $n$?). $\phi, $p_\theta^{(k)}$, "cardinality-invariant function".

2. Pseudo-code is only given in the appendix and is difficult to understand because it only seems to cover the case of a single observation (and doesn't cover the training procedure, details on how the weighted samples are mapped to the latent space, nor multiple time steps).

3. The numerical experiments cannot be understood without reading the appendix. There are very likely strengths and weaknesses of the methodology upon which I cannot comment because I was unable to understand the numerical results section for this reason. 


**Other:**

1. Code should always be made available to reviewers.

2. There are some inadequate references: citing Papamakrios et al., 2021, for the well known density transform formula on Page 4, and Sokota et al. (2022) for the notion of resampling in particle filters.

### Questions
1. What is the computational cost of the different approaches shown in the numerical results and is it comparable (even when taking training into account)?

### Soundness
2

### Presentation
1

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
This work presents an algorithm for learning the embedded representations of belief states in partially observable environments combined with generative modelling for efficient Bayesian updates. The authors define an embedding function, which approximates features of a target distributiong. They prove that the NBF estimate almost surely converges to the true distribution under certain conditions, and demonstrate the viability of their method in the deep setting in 3 benchmarks.

### Strengths
1. The necessity of learned belief states is well motivated, given the limited expressivity of traditional methods.
2. The updating of the NBF by posterior sampling is well-explained.
3. Theorems 4.1 and 4.2 are well-explained, and Figure 3 further exemplifies the elegance of the proposed method.
4. The proof of convergence of NBF with a perfect model is convincing, and provides a basis for its viability in practical settings (although the gap between this and a practical model is not quantified).
5. The improvement in divergence between the true distributions in all 3 tasks with NBF is convincing - even with fewer particles, NBF is able to clearly approximate beliefs on gridworlds, discrete POMDP Goofspiel, and the triangulation game better than particle filtering.

### Weaknesses
1. The donut distribution example in Figure 2 is necessary but not sufficient; why was it not shown the method would also work on an arbitrary and visualisable 2D distribution? This would demonstrate the usefulness of the proposed method over standard Gaussian models.
2. The examples provided remain toy domains with relatively low dimensions, and do not necessarily demonstrate the viability of NBF in a larger, more random, or non-stationary POMDP setting such as [1]. This is currently the main drawback of this work, and a study of NBF vs particle filtering in a larger partially-observable setting would likely suffice a recommendation for acceptance.
3. Lack of ablations;  a study disentangling the contribution of the embedding model from the filtering update is crucial in understanding whether gains arise from better density modelling, resampling, or something else.
4. (Minor) While the success of NBF in belief estimation is well-demonstrated, it remains to be seen whether this has a significant effect on the performance, robustness, or generalization of a system which uses those beliefs is proportionally improved. While it may not be well within the scope of this paper, a study on the performance of a reinforcement learning algorithm with the belief states for policy learning would be interesting.

[1] Tao, Ruo Yu, et al. "Benchmarking Partial Observability in Reinforcement Learning with a Suite of Memory-Improvable Domains." arXiv preprint arXiv:2508.00046 (2025).

### Questions
1. The belief state space in the gridworld and triangulation settings is clearly defined, but is not apparent in Goofspiel. What is it?

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
2

### Summary
This paper proposes a new method for belief state modeling by approximating the posterior distribution with a generative model conditioned on belief state embeddings. The proposed method is validated in variants of Gridworld, the card game Goofspiel, and a continuous localization environment called Triangulation, and benchmarked agains other classical methods such as particle filtering and the recurrent approach.

### Strengths
- The goal of overcoming particle impoverishment in high-dimensions is well-motivated.

- The proposed framework to combine classical filtering, conditional embeddings and deep generative models is novel. 


- Empirical studies show that NBF consistently outperforms particle filtering.

### Weaknesses
- The presentation of the method can be improved. A few important details are not clear and it would be helpful to have a complete algorithm for the proposed method (see more details in Questions section). And the context provided in  the preamble of sect. 3, particularly regarding $\pi$ and $G$ is not referenced in the remaining of the section. 

- The clarity of the remaining part of the paper can be improved. See more detailed comments in the Questions section.

- It is hard to assess the significance of the proposed method, given the limited tasks and baseline methods.

### Questions
Method related:

- Can the authors discuss practical choices of embedding networks?

- Is there a particular reason to use normalizing flows for the conditional distribution fitting part?

- Does the embedding network require pre-training before applying it to decision making tasks?

- Do the parameters of embedding network and the normalizing flows get updated during filtering?

- What are the takeaways from Sec 3.2?

- What are the criteria for a good belief embedding model?



Empirics related:

- Can the authors explain what is the recurrent approach? Is there a particular implementation / reference  for that?

- In table 1, why not include the comparison to the recurrent approach? 

- How representative and challenging are the benchmarks problems in this filed? As the authors pointed out in the Discussion, "Though our empirical evaluation focuses on three relatively simple domains, it still highlights NBF’s versatility and potential effectiveness in more complex tasks.", why not consider more complex tasks?

- How many tasks are truly multimodal? Is there empirical evidence to illustrate that the proposed method captures the multimodality, if any?

- Is there similar weight collapse issue in combining the embedding for the proposed method?

### Soundness
3

### Presentation
2

### Contribution
3
