# Energy-Weighted Flow Matching: Unlocking Continuous Normalizing Flows for Efficient and Scalable Boltzmann Sampling

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Sampling from unnormalized target distributions, e.g. Boltzmann distributions $\mu_{\text{target}}(x) \propto \exp(-E(x)/T)$, is fundamental to many scientific applications yet computationally challenging due to complex, high-dimensional energy landscapes. Existing approaches applying modern generative models to Boltzmann distributions either require large datasets of samples drawn from the target distribution or, when using only energy evaluations for training, cannot efficiently leverage the expressivity of advanced architectures like continuous normalizing flows that have shown promise for molecular sampling. To address these shortcomings, we introduce Energy-Weighted Flow Matching (EWFM), a novel training objective enabling continuous normalizing flows to model Boltzmann distributions using only energy function evaluations. Our objective reformulates conditional flow matching via importance sampling, allowing training with samples from arbitrary proposal distributions. Based on this objective, we develop two algorithms: iterative EWFM (iEWFM), which progressively refines proposals through iterative training, and annealed EWFM (aEWFM), which additionally incorporates temperature annealing for challenging energy landscapes. On benchmark systems, including challenging 55-particle Lennard-Jones clusters, our algorithms demonstrate sample quality competitive with state-of-the-art energy-only methods while requiring up to three orders of magnitude fewer energy evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes an iterative method for training neural samplers. The core idea is to use importance weighting to refine the standard flow-matching training objective. On top of this reweighting scheme, the authors introduce a temperature-annealing schedule that trains the sampler progressively—starting from a high-temperature distribution and gradually moving toward the target temperature.

While the paper is clearly written, I have several concerns. First, the related-work coverage omits a number of recent and highly relevant papers. For neural samplers, [1] improves upon iDEM via a bootstrapping technique and reports markedly stronger results on LJ55 than iDEM. There are also competitive neural samplers that should be cited and compared against, such as [2, 3]. Regarding annealing to aid neural-sampler training, several works should be cited as well, including [4, 5, 6] (acknowledging that [6] is very recent).

Second, the methodological novelty appears limited. Importance-weighted regression objectives have been explored previously—for example, [7] applies an analogous idea to (denoising) score matching. Finally, relying solely on importance reweighting raises scalability concerns in higher dimensions, since importance weights typically become more concentrated (“weight degeneracy”) as dimensionality grows.

In sum, despite the clear exposition, the paper lacks coverage of related work and the proposed method offers limited novelty. I therefore lean toward rejection.

[1] OuYang, RuiKang, Bo Qiang, and José Miguel Hernández-Lobato. "Bnem: A boltzmann sampler based on bootstrapped noised energy matching." arXiv preprint arXiv:2409.09787 (2024).

[2] Chen, Junhua, et al. "Sequential controlled langevin diffusions." arXiv preprint arXiv:2412.07081 (2024).

[3] Blessing, Denis, et al. "Underdamped diffusion bridges with applications to sampling." arXiv preprint arXiv:2503.01006 (2025).

[4] Rissanen, Severi, et al. "Progressive Tempering Sampler with Diffusion." arXiv preprint arXiv:2506.05231 (2025).

[5] Akhound-Sadegh, Tara, et al. "Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities." arXiv preprint arXiv:2506.16471 (2025).

[6] Blessing, Denis, et al. "Trust Region Constrained Measure Transport in Path Space for Stochastic Optimal Control and Inference." arXiv preprint arXiv:2508.12511 (2025).

[7] Yu, Ziyang, Wenbing Huang, and Yang Liu. "Force-guided bridge matching for full-atom time-coarsened dynamics of peptides." arXiv preprint arXiv:2408.15126 (2024).

### Strengths
1. The paper is clearly written and easy to follow.

2. Empirically, the proposed method outperforms FAB and iDEM while requiring fewer energy evaluations.

### Weaknesses
1.	Related work coverage is incomplete. For training neural samplers, [1] improves upon iDEM via a bootstrapping technique and shows substantially better performance on LJ55 than iDEM. Additional strong neural samplers, such as [2, 3], should be cited and compared against. For annealing-assisted training, relevant works include [4, 5, 6] (with [6] being very recent).

2.	Limited methodological novelty. Using importance sampling to reweight a regression-style training loss is not new; closely related ideas appear in prior work, e.g., importance-weighted (denoising) score matching [7]. The paper should clarify what is fundamentally new relative to these approaches.

3.	Empirical concerns and scalability:
	
- On the 2D GMM task, EWFM outperforms its iterative and annealed variants (iEWFM, aEWFM), which is surprising in a low-dimensional setting where buffer/annealing should not degrade performance.

- On LJ55, there is a visible gap between EWFM samples and ground truth, as reflected in both the interatomic-distance and energy histograms. This raises concerns about scalability of the loss reweighting: importance weighting is known to suffer from the curse of dimensionality (weight degeneracy).

- Moreover, the 2D GMM with a prior covering most modes (as in iDEM) is relatively simple; [2] demonstrates scalability to a 50D case, setting a higher bar for comparison.

4.	Ablation: loss reweighting is underexplored. The same importance weights could be used to train a standard diffusion/score-matching model by replacing the conditional flow-matching loss \|u_\theta - u\|^2 with a conditional denoising score-matching loss \|s_\theta - s\|^2. An ablation comparing these two paths would clarify whether the benefit comes from the reweighting itself or from the specific flow-matching formulation.

5.	Ablation: annealing is underexplored. The proposed annealing scheme could also be applied to iDEM by tempering the energy used in the Monte-Carlo score estimator. An ablation demonstrating this would help isolate the contribution of annealing versus the rest of the method.

[1] OuYang, RuiKang, Bo Qiang, and José Miguel Hernández-Lobato. "Bnem: A boltzmann sampler based on bootstrapped noised energy matching." arXiv preprint arXiv:2409.09787 (2024).

[2] Chen, Junhua, et al. "Sequential controlled langevin diffusions." arXiv preprint arXiv:2412.07081 (2024).

[3] Blessing, Denis, et al. "Underdamped diffusion bridges with applications to sampling." arXiv preprint arXiv:2503.01006 (2025).

[4] Rissanen, Severi, et al. "Progressive Tempering Sampler with Diffusion." arXiv preprint arXiv:2506.05231 (2025).

[5] Akhound-Sadegh, Tara, et al. "Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities." arXiv preprint arXiv:2506.16471 (2025).

[6] Blessing, Denis, et al. "Trust Region Constrained Measure Transport in Path Space for Stochastic Optimal Control and Inference." arXiv preprint arXiv:2508.12511 (2025).

[7] Yu, Ziyang, Wenbing Huang, and Yang Liu. "Force-guided bridge matching for full-atom time-coarsened dynamics of peptides." arXiv preprint arXiv:2408.15126 (2024).

### Questions
1. 2D GMM: Why does EWFM perform better than its iterative and annealing variants (iEWFM, aEWFM) on the 2D GMM task? What failure modes did you observe for the iterative buffer or the annealing schedule?
	
2. LJ13/LJ55 W2 computation: When reporting the (data) Wasserstein-2 distances on LJ13 and LJ55, did you account for translation and rotation (i.e., SE(3) equivariance) in the metric, e.g., via alignment or quotienting out rigid motions?
	
3. Metrics reported: Why report only data W2 and NLL? Could you also include energy-W2, MMD, and total-variation distance to provide a more complete picture of sample quality and energy consistency?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a framework for sampling from Boltzmann distributions using the conditional flow matching method. Unlike typical flow matching methods which require access to samples from the target density, the authors leverage importance sampling to leverage a different proposal distribution. Since the ground-truth energy is known and computable up to a normalizing constant, the importance weight can be easily computed; allowing training of flow-matching models through self-normalized importance sampling. Empirical evaluation of the proposed method shows that it is competitive with iterated Denoising Energy Matching while requiring significantly fewer energy evaluations.

### Strengths
- The authors extend the flow-matching framework to sampling problems where the target density is available up to a normalization constant, but samples from it are not available. This is an important direction of research towards AI for science applications.
- The proposed method is simple to understand and is built on top of well established methods: flow matching and self-normalized importance sampling.
- Empirical evaluation shows that the proposed method outperforms baselines on high-dimensional problems as well as enjoys fewer energy evaluations.

### Weaknesses
- The baselines considered in the work are rather weak. The comparisons are only made with FAB and iDEM but the authors should be comparing their proposed method against [1-8].
- The novelty of the work is severely limited. It is well known that iDEM, iEFM, etc. are based on self-normalized importance sampling objectives, hence such systems are already using SNIS approaches to learn either a score or drift. In this context, what is the proposed method adding since this is already well known (eg. see TweeDEM in [2]).
- The work does not provide any quantitative evidence of training and inference times (in wall clock), which is a very important metric since the authors highlight that their method is more efficient.

**References**

[1] Woo, Dongyeop, and Sungsoo Ahn. "Iterated energy-based flow matching for sampling from boltzmann densities." arXiv preprint arXiv:2408.16249 (2024).

[2] OuYang, RuiKang, Bo Qiang, and José Miguel Hernández-Lobato. "Bnem: A boltzmann sampler based on bootstrapped noised energy matching." arXiv preprint arXiv:2409.09787 (2024).

[3] Havens, Aaron, et al. "Adjoint sampling: Highly scalable diffusion samplers via adjoint matching." arXiv preprint arXiv:2504.11713 (2025).

[4] Choi, Jaemoo, et al. "Non-equilibrium Annealed Adjoint Sampler." arXiv preprint arXiv:2506.18165 (2025).

[5] Albergo, Michael S., and Eric Vanden-Eijnden. "Nets: A non-equilibrium transport sampler." arXiv preprint arXiv:2410.02711 (2024).

[6] Chen, Junhua, et al. "Sequential controlled langevin diffusions." arXiv preprint arXiv:2412.07081 (2024).

[7] Vargas, Francisco, et al. "Transport meets variational inference: Controlled monte carlo diffusions." arXiv preprint arXiv:2307.01050 (2023).

[8] Sendera, Marcin, et al. "Improved off-policy training of diffusion samplers." Advances in Neural Information Processing Systems 37 (2024): 81016-81045.

### Questions
- It is quite unclear to me how the proposed method reduces the number of energy evaluations needed. In fact, does the proposed method reduce to IEFM?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces an alternative yet equivalent formulation of the flow matching loss that removes the need for target samples when training CNFs to generate samples from the Boltzmann distribution; instead, the approach adopts an expectation over an arbitrary proposal distribution combined with importance sampling to recover the true target distribution. While the method is conceptually appealing, its practical scalability remains up for debate. The largest system evaluated in the paper is LJ55, which is considerably smaller than those addressed by state-of-the-art approaches for Boltzmann sampling. Comparisons are drawn to iDEM, which has since been surpassed by newer samplers such as adjoint sampling (arXiv preprint arXiv:2504.11713). Other recent developments, including discrete normalizing flow methods that scale to much larger systems (arXiv preprint arXiv:2508.18175; arXiv preprint arXiv:2502.18462), further highlight the gap in demonstrating scalability (granted these use data and not energy alone). Continuous normalizing flow models such as ECNF++ have also been successfully applied to larger systems (arXiv preprint arXiv:2502.18462) demonstrating scalable performance. Overall, while the proposed approach is theoretically interesting, the experimental validation and benchmarking fall short of convincingly establishing its effectiveness at scale.

### Strengths
The proposed mathematical framework is conceptually appealing, with its primary benefit demonstrated through a reduction in the number of energy evaluations relative to comparable approaches. This efficiency may offer practical advantages for Boltzmann sampling applications.

### Weaknesses
The benchmarking presented in the paper is very limited. Other standard benchmarks like PIS or LogVariance have not been contrasted on the same datasets. Annealing is introduced, yet benchmarking relative to the most performant annealing models (PITA: arXiv preprint arXiv:2506.16471) is not performed. Further, the proposed approach shows only marginal improvements over iDEM, while several more recent methods (e.g., arXiv preprint arXiv:2504.11713) have demonstrated superior performance. These newer approaches are not included in the comparison and are instead deferred to future work; however, without such evaluations, the practical utility and scalability of the proposed method remain uncertain.

### Questions
- How do the number of energy evaluations compare with adjoint sampling? 
- Can ESS and energy-W1/W2 be included as additional metrics for performance? 
- How does the performance on ESS and energy-W1/W2 contrast relative to adjoint sampling? 
- How does EWFM scale to simple peptides e.g., alanine dipeptide, tri-alanine, etc. 
- Can more powerful and expressive networks be used for training EWFM, and how do they perform? 
- On the annealing experiments, how does the approach perform relative to PITA (arXiv preprint arXiv:2506.16471)?

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
5

### Summary
This paper introduces Energy-Weighted Flow Matching (EWFM), a novel and theoretically elegant data-free objective for training Continuous Normalizing Flows (CNFs) to sample from unnormalized Boltzmann distributions. The core idea, which reformulates Conditional Flow Matching via importance sampling, is a solid and principled contribution. Based on this, the authors propose an iterative (iEWFM) and an annealed (aEWFM) algorithm.

### Strengths
The paper's primary contributions are clear and theoretically sound:
1. The EWFM objective is a novel and principled method for training CNFs on unnormalized distributions without target data. The use of importance sampling to adapt the CFM loss is an elegant and effective idea.
2. The authors build on this objective to propose two practical algorithms: iterative EWFM (iEWFM), which progressively refines its own proposal distribution, and annealed EWFM (aEWFM), which incorporates a temperature schedule to handle more complex energy landscapes.
3. The method's main strength is its efficiency in terms of energy function calls.

### Weaknesses
Despite the novel objective, the empirical validation of the method suffers from several critical flaws that undermine the paper's claims of state-of-the-art performance and efficiency:
1. The most significant drawback is the complete failure to compare against Iterated Energy-based Flow Matching (iEFM). The authors explicitly identify iEFM as the "only other flow matching approach using only energy evaluations". This omission is highly concerning, especially since iEFM was successfully demonstrated on the GMM-40 and DW-4 benchmarks, the ones where this paper's method shows significant limitations.
2. The claims of "competitive" performance are not fully supported by the results. The method fails catastrophically on DW-4 system, producing a W2 distance much worse than the baselines. It also underperforms against both iDEM and FAB on the LJ-13 system. The authors' brief hypothesis of "potential bias" for the DW-4 failure is insufficient and does not inspire confidence in the method's robustness.
3. The method for reporting statistical significance in Table 1 is non-standard. The paper bolds results based on the mean $\pm$ 1 SD interval. This is not a rigorous test of statistical significance and can be misleading, as it may incorrectly group methods with very different mean performances. A standard Welch's t-test would be required for a valid comparison.

### Questions
Your review correctly identifies the relevance of Bayesian inference tasks, such as those in [1]. For these problems (e.g., Bayesian logistic regression on the German Credit  or Breast Cancer  datasets), the "energy function" is the negative log-posterior, and its evaluation requires a full pass over the dataset. Given that your method's primary bottleneck is the 30+ hour wall-clock time for CNF density calculations , how do you justify its "efficiency" for this common class of problems, where the cost of an energy evaluation and the cost of a model density evaluation may be comparable?


References:

[1] Denis Blessing, Xiaogang Jia, Johannes Esslinger, Francisco Vargas, and Gerhard Neumann. Beyond ELBOs: A large-scale evaluation of variational methods for sampling

### Soundness
2

### Presentation
2

### Contribution
3
