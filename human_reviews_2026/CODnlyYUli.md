# Accelerated Parallel Tempering via Neural Transports

- Decision: Accept (Poster)
- Scores: 4, 8, 2, 2

## Abstract
Markov Chain Monte Carlo (MCMC) algorithms are essential tools in computational statistics for sampling from unnormalised probability distributions, but can be fragile when targeting high-dimensional, multimodal, or complex target distributions. Parallel Tempering (PT) enhances MCMC's sample efficiency through annealing and parallel computation, propagating samples from tractable reference distributions to intractable targets via state swapping across interpolating distributions. The effectiveness of PT is limited by the often minimal overlap between adjacent distributions in challenging problems, which requires increasing the computational resources to compensate. We introduce a framework that accelerates PT by leveraging neural samplers---including normalising flows, diffusion models, and controlled diffusions---to reduce the required overlap. Our approach utilises neural samplers in parallel, circumventing the computational burden of neural samplers while preserving the asymptotic consistency of classical PT. We demonstrate theoretically and empirically on a variety of multimodal sampling problems that our method improves sample quality, reduces the computational cost compared to classical PT, and enables efficient free energy/normalising constant estimation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This manuscript proposes a framework for the application of neural sampling methods to augment Parallel Tempering (PT) based MCMC. Neural samplers like normalizing flows, CMCD, and diffusion models are used to introduce time-inhomogeneous inner-loop Markov processes that accelerate the outer-loop parallel tempering by learning how to make adjacent chains overlap. This resulting procedure Accelerated Parallel Tempering (APT) increases the efficiency of PT in terms of the rate of round trips which is an indicator for the rate of mixing and it conserves the invariance with respect to the target distribution. Theoretical results underpin the conceptual soundness of this method and experimental results demonstrate that it yields practical benefits in several relevant sampling benchmarks.

### Strengths
- This method appears to be novel.

- The theoretical results are concise and clear and are related to the experimental results (e.g. connection between N and $\Lambda$)

- The mathematical notation is at an appropriate complexity level, resulting in good readability.

### Weaknesses
- Lack of comparison to neural samplers: the authors argue correctly that MCMC like PT and APT enjoy theoretical guarantees that are not provided in neural samplers.
This theoretical distinction is not a sound reason to exclude neural samplers entirely from the experimental comparison. The ultimate goal of these methods is exactly the same as for MCMC, namely to sample unnormalized distribution, and hence neural samplers should not be dismissed entirely based on consideration of asymptotic guarantees. The experiments shown in section 6.4 which compare to neural samplers (CMCD and diffusion samplers) are no fair comparisons since they utilize very different compute budgets (transport maps are concatenated here). 
 
- As argued in the paper the number of round trips is indeed a well-established metric in the context of PT and therefore a well-motivated metric. Comparing to sampling methods more broadly is essential and hence a comparison to other methods should include more general sample quality metrics, like ELBO, Sinkhorn, ESS, log Z. The authors argue “As ESS measures the intertwined performance of the local exploration and swap kernels, we are instead interested in maximising communication between reference and target …”. But such these performance metrics are ultimately crucial to judge the performance of APT relative to other sampling methods.


- l147: the distributions $\pi$ are called potentials


- l206: “Theorem 1 shows we can quantify this  discrepancy in through the rejection rate”. The discrepancy in what?


- Reproducibility suffers from a lack of a code base and corresponding details of the implementation appear to be scattered across many prior works. However, the reviewer did not attempt to reproduce results.

### Questions
- What is the intuitive meaning of the global barrier $\Lambda$?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The present paper proposes to accelerate the parallel tempering (PT) algorithm using learned transport processes between neighboring distributions in the annealing ladder. The approach draws inspiration from the statistical mechanics literature (works by Jarzynski and collaborators) and a single step version with normalizing flows (Invernizzi et al 2022). It bears similarities with previously proposed approaches mixing deep learning and Sequential Monte Carlo, namely the Annealed Flow Transport (Arbel et al 2021), NETs (Albergo et al 2024) and CMCD (Vargas et al 2024). 

The proposed approach is presented in a general framework along with different concrete variants of the algorithm leveraging either normalizing flows, diffusion models or stochastic control. The algorithm is shown to yield consistent estimators of expectations and normalization constants. A theoretical analysis of performances is given under symplifying assumptions. A set of numerical experiments demonstrates a possible advantage of the method over vanilla parallel tempering and compares the different variant proposed. Finally, limitations are discussed in a short conclusion where, in particular, the need to take into account the computational cost of neural networks in future works is acknowledged by the authors.

### Strengths
- Although related to recent literature and not unexpected in this respect, the proposed algorithm is novel and the authors do a great job at presenting in general the method before showcasing different possible applications with different types of generative modeling ideas. 
- The article also does a great job at connecting its method to the adjacent literature. 
- The method is justified by proofs of consistency. I have not read in details the proofs, but the result appear reasonable and the adequate physics literature is cited. 
- A theoretical analysis of performance for edge cases is also given. 
- The numerical section spans different examples, notably in relatively high dimension, even investigating the impact of dimension increase in a synthetic experiment.

### Weaknesses
- It would be desirable to insist more on the question of the training of the neural networks in the main text and on the fact that it is not a trivial question in this sampling setting. For instance, for NF-APT, the authors state in Appendix C.1.2 that the retained strategy is to first run PT to be able to train the flows, which is arguably an important limitation. 

- The introduction is not always fair to the adjacent literature
	- line 072 - “However, these methods usually incur a bias, foregoing theoretical guarantees of MCMC, and can be expensive to implement and train.” Is not true of the discrete flows approaches that are cited by the authors , nor of NETs (Albergo et al 2024).

	- line 095 - “By contrast, our framework leverages normalising flows to facilitate exchanges between all neighbouring temperature levels, thereby enhancing sampling efficiency across the entire annealing path and providing a more stable training objective” - the ambition of Invernizzi et al. is to drastically simplify the procedure by avoiding the necessity to have many replicas. As such, this last sentence is unclear, and the claim for the increased stability of the training objective needs to be substantiated. If I understand correctly this reference is implementing NF-APT with N = 1? 


Minor:
- It would be worth mentioning [Noble2025] in section 5.3, as this reference already proposes to learn energy-based models along a noising process to ease sampling, although this reference was exploiting it in a different way. 
- l103 the definition of $Z_n$ is lacking the exponent $n$ to $U$.
- l190 “we use using”
- In Eq 4, maybe restate that the brackets are notations for an expectation. Also this is a common notation in the statistics/Monte Carlo literature, it is not usually encountered in the machine learning literature.
- l215: “To also obtain free energy estimates, by averaging …” this sentence has no main proposition. 
- The notation $\tau$ in proposition 250 is not introduced, I am guessing this the round trip rate. 
- l262 “a good neural network approximation” it would be a good idea to state a good approximation of what if the authors want to make this point at this stage, otherwise it can wait for the next section.

[Noble2025] Noble, Maxence, Louis Grenioux, Marylou Gabrié, and Alain Oliviero Durmus. “Learned Reference-Based Diffusion Sampler for Multi-Modal Distributions.” Paper presented at The Thirteenth International Conference on Learning Representations. April 4, 2025. https://openreview.net/forum?id=fmJUYgmMbL.

### Questions
-

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an extension of Parallel Tempering (PT) called Accelerated Parallel Tempering (APT), in which the standard swap operation is enhanced with deterministic or stochastic transition kernels. These kernels are trained adaptively using samples generated by APT within a data-driven objective. The deterministic variant employs normalizing flows (NF-APT), while the stochastic variants derive from forward–backward discretizations of controlled annealed Langevin dynamics (CMCD-APT) or from diffusion models, where the forward process is exact and the reverse process is obtained by integrating the learned reverse SDE (Diff-APT). The sequence of intermediate potentials is user-defined for NF-APT and CMCD-APT (typically along a tempering path) and learned automatically in Diff-APT. Experimental results on synthetic benchmarks and particle systems demonstrate that APT accelerates sampling compared to standard parallel tempering.

### Strengths
* The paper addresses an important problem and demonstrates clear improvements in round-trip rates for parallel tempering.

### Weaknesses
* The paper is very poorly written. The presentation based on the Jarzynski framework makes it difficult to follow. A formulation directly grounded in the Metropolis–Hastings framework, with clearly defined target distribution and proposal kernel (see [1]), would greatly improve readability and conceptual clarity.
* The novelty is rather limited. The deterministic case has already been covered in [2] (as acknowledged by the authors), while the stochastic variant represents only a modest generalization of prior work on AIS [3,4] and SMC methods [5], against which no comparisons are provided.
* The proposed method does not address one of the core limitation of PT in multi-modal settings : mode switching. It is well known that along tempering paths, probability mass tends to shift abruptly between distinct high-probability regions [5,6], severely hindering mixing and performance in both PT and SMC. Even with accelerated swaps in NF-APT or CMCD-APT, the transition kernels remain local refinements (as shown in the left panel of Fig. 1) and cannot overcome this issue. Similarly, Diff-APT is affected because its learned marginal distributions rely on score matching, which is known to be mode-blind [7,8,9], causing the learned path to exhibit mode switching as well. Consequently, the method provides limited benefit in genuinely multi-modal settings. This issue becomes more pronounced in higher dimensions (likely explaining the use of a "perfect" path in Section 6.2). It would be informative to report results with a learned path. While increasing $K$ and reducing $N$ might partially mitigate this, it does not fundamentally resolve the scalability issue.
* The evaluation is quite narrow. The round-trip rate is the primary quantitative metric used (except for Fig. 3, which reports free-energy differences, and Figs. 4 and 6–7, which are purely qualitative). This metric alone may obscure important behaviors such as mode switching and limits meaningful comparison to standard PT. The justification for this metric relies solely on [10], which claims PT outperforms neural samplers. Moreover, the only non-PT baseline compared is CMCD, which is known to perform poorly [10,11,12].
* The target-informed parametrization (L1400) is known to substantially restrict the expressivity of the energy-based model [13] and to introduce significant computational overhead [10], further reducing the appeal of this approach.

[1] Syed, S., Bouchard-Côté, A., Deligiannidis, G., & Doucet, A. (2022). Non-reversible parallel tempering: A scalable highly parallel MCMC scheme. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 84(2), 321–350.

[2] Invernizzi, M., Krämer, A., Clementi, C., & Noe, F. (2022). Skipping the Replica Exchange Ladder with Normalizing Flows. The Journal of Physical Chemistry Letters, 13(50), 11643–11649.

[3] Zhang, F., He, J., Midgley, L., Antorán, J., & Hernández-Lobato, J. (2024). Efficient and unbiased sampling of boltzmann distributions via consistency models. arXiv preprint arXiv:2409.07323.

[4] Fengzhe Zhang, Laurence I. Midgley, & José Miguel Hernández-Lobato. (2025). Efficient and Unbiased Sampling from Boltzmann Distributions via Variance-Tuned Diffusion Models.

[5] Phillips, A., Dau, H.D., Hutchinson, M., De Bortoli, V., Deligiannidis, G., & Doucet, A. (2024). Particle Denoising Diffusion Sampler. In Proceedings of the 41st International Conference on Machine Learning (pp. 40688–40724). PMLR.

[6] Bálint Máté, & François Fleuret (2023). Learning Interpolations between Boltzmann Densities. Transactions on Machine Learning Research.

[7] Wenliang, L., & Kanagawa, H.. (2021). Blindness of score-based methods to isolated components and mixing proportions.

[8] Zhang, M., Key, O., Hayes, P., Barber, D., Paige, B., & Briol, F.X. (2022). Towards Healing the Blindness of Score Matching. In NeurIPS 2022 Workshop on Score-Based Methods.

[9] Shi, Z., Yu, L., Xie, T., & Zhang, C.. (2024). Diffusion-PINN Sampler.

[10] Jiajun He, Yuanqi Du, Francisco Vargas, Dinghuai Zhang, Shreyas Padhy, RuiKang OuYang, Carla P Gomes, & José Miguel Hernández-Lobato (2025). No Trick, No Treat: Pursuits and Challenges Towards Simulation-free Training of Neural Samplers. In Frontiers in Probabilistic Inference: Learning meets Sampling.

[11] Junhua Chen, Lorenz Richter, Julius Berner, Denis Blessing, Gerhard Neumann, & Anima Anandkumar (2025). Sequential Controlled Langevin Diffusions. In The Thirteenth International Conference on Learning Representations.

[12] Noble, M., Grenioux, L., Gabrié, M., & Durmus, A. (2025). Learned Reference-based Diffusion Sampler for multi-modal distributions. In The Thirteenth International Conference on Learning Representations.

[13] Jiajun He, Yuanqi Du, Francisco Vargas, Yuanqing Wang, Carla P. Gomes, José Miguel Hernández-Laobato, & Eric Vanden-Eĳnden. (2025). FEAT: Free energy Estimators with Adaptive Transport.

### Questions
* In the introduction, the references to Noé et al. (2019), Midgley et al. (2023), and Gabrié et al. (2022) suggest that these methods "incur a bias" similar to the previously mentioned approaches. This interpretation is incorrect. The earlier works (except iDEM) perform variational inference to learn a generative model without data, where the bias arises from optimization and model misspecification. In contrast, the three cited works embedded the models within Monte Carlo schemes - such as IS, AIS, or MCMC - which, in the infinite-particle/chain length limit, correct the bias of the learned model. Their residual bias is purely statistical. Grouping these methods together is therefore misleading.
* The training procedures are described very vaguely. What are the exact loss functions used, and how are they implemented in practice? How many times is the adaptation loop repeated (i.e., how many optimizations of the data-based loss are performed) ?
* Could you visualize the learned diffusion potentials path over time for a simple mixture of two one-dimensional Gaussian distributions?
* Comparisons with standard AIS, SMC, PT (under an equivalent computational budget, including training), and other multimodal samplers (potentially ML-enhanced) on target-specific metrics would be highly informative.
* For Diff-APT, could you first train an unconstrained energy-based model and then learn an auxiliary function analogous to $r_{\theta}$ a posteriori? This might help avoid the expressivity constraints imposed by the current formulation.
* For Diff-APT, why not use the target score matching loss [A], similar to what PDDS employs [B] ?
* What mechanisms prevent the adaptive procedure from suffering from mode collapse, particularly given the mode-switching phenomenon? This concern is especially relevant since the loss functions include (at least partially) a reverse KL term.
* According to Appendix D, only a single HMC step is performed for the local exploration move, which seems unrealistically low. Could you provide an ablation study on the swap frequency, i.e., the number of interleaved local steps between swaps?
* The ManyWell-32 experiment in Appendix E.1 claims to demonstrate recovery of mode weights, yet no quantitative metrics are reported. Why not use the benchmark from [C], which is specifically designed to evaluate this capability?
* How are the chains initialized in PT/APT?
* The total number of MCMC steps (50k or 100k) is extremely large. Combined with the adaptive training loop, this implies a substantial computational cost. Could you (i) report actual wall-clock runtimes on a specific hardware setup and (ii) provide an ablation on the total number of MCMC steps?
* In Fig. 4, how do you explain the large performance gap between Diffusion and Diff-APT? If Diffusion is trained using samples from Diff-APT, this discrepancy seems counterintuitive.
* For ALDP, how are the chains initialized, and how do you sample from the $T = 1200K$ reference distribution?
* For ALDP, is the energy parametrization (L1400) rotation-invariant? Additionally, how do you ensure that both the transition kernels and the energy function are defined in the center-of-mass (CoM) space, i.e., the mass-centered coordinate system?

[A] Valentin De Bortoli, Michael Hutchinson, Peter Wirnsberger, & Arnaud Doucet. (2024). Target Score Matching.

[B] Phillips, A., Dau, H.D., Hutchinson, M., De Bortoli, V., Deligiannidis, G., & Doucet, A. (2024). Particle Denoising Diffusion Sampler. In Proceedings of the 41st International Conference on Machine Learning (pp. 40688–40724). PMLR.

[C] Grenioux, L., Noble, M., & Gabrie, M. (2025). Improving the evaluation of samplers on multi-modal targets. In Frontiers in Probabilistic Inference: Learning meets Sampling.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Accelerated Parallel Tempering (APT), a new framework that integrates neural samplers into Parallel Tempering (PT) to improve sampling efficiency on complex, high-dimensional, and multi-modal distributions. Classical PT often struggles because adjacent temperature distributions share low overlap, limiting swap acceptance rates and requiring a large number of parallel chains. APT addresses this by using neural transports to “bridge” neighboring distributions before proposing swaps, dramatically increasing swap acceptance and reducing the number of chains needed—while preserving the exact asymptotic correctness of PT.

### Strengths
1. Math derivation is sound.

2. PT is known to be an effective sampler for multi-modal simulations. If diffusion-enhanced sampler really works, PT will surely boost the perfomance of multi-modal simulation.

### Weaknesses
1. I don't like the whole community that utilizes ideas like the diffusion model to do sampling, which is super expensive and doesn't make a lot of sense. I have extensive research experience in sampling and diffusion models but I don't this is a nice combination.

2. The motivation why do and when do we need diffusion models to do sampling is not well-supported. The intuition why a backward process is needed is not explained clearly. 

3. for section 6.1, measuring the round trip is not a good idea, the eventual goal is to sample the 40-mode mixture distribution, you can simply measure the empirical TV/ KL/ W1 error.

4. No scalable real-world experiments. Molecular is too small.

### Questions
1. If gradient information is already cheap, who won't we just use vanilla PT sampler? 

2. Low acceptance probability occurs when these distributions have minimal overlap. Addressing this requires increasing
the number of parallel chains N, which may not always be possible. Why? If a problem cannot be simulated using 4-chain PT, it must be a hard problem, I don't expect this algorithm to solve it as well.

3. Could you show the distribution error plot by comparing vanilla PT and CMCD-APT using different computational budgets?

### Soundness
3

### Presentation
3

### Contribution
1
