# LD-EnSF: Synergizing Latent Dynamics with Ensemble Score Filters for Fast Data Assimilation with Sparse Observations

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 8

## Abstract
Data assimilation techniques are crucial for accurately tracking complex dynamical systems by integrating observational data with numerical forecasts. Recently, score-based data assimilation methods emerged as powerful tools for high-dimensional and nonlinear data assimilation. However, these methods still incur substantial computational costs due to the need for expensive forward simulations. In this work, we propose LD-EnSF, a novel score-based data assimilation method that fully eliminates the need for full-space simulations by evolving dynamics directly in a compact latent space. Our method incorporates improved Latent Dynamics Networks (LDNets) to learn accurate surrogate dynamics and introduces a history-aware LSTM encoder to effectively process sparse and irregular observations. By operating entirely in the latent space, LD-EnSF achieves speedups orders of magnitude over existing methods while maintaining high accuracy and robustness. We demonstrate the effectiveness of LD-EnSF on several challenging high-dimensional benchmarks with highly sparse (in both space and time) and noisy observations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
## Summary
This paper introduces LD-EnSF, a data-driven framework for data assimilation (DA) that aims to solve two major challenges: the high computational cost of numerical simulations and the difficulty of applying score-based methods to sparse observations. The core idea is to (1) learn a fast surrogate model that forwards the system dynamics in a low-dimensional latent space, and (2) use a LSTM encoder to map sparse observations into this latent space, where EnSF is then performed. The numerical experiments on three challenging high-dimensional benchmarks show that LD-EnSF achieves notable speedups while maintaining high accuracy even with sparse observations.

### Strengths
## Strengths
- The authors provide a comprehensive numerical studies on three challenging, high-dimensional benchmarks and present solid comparisons of LD-EnSF against strong baselines. The results show the advantages of proposed methods in computational efficiency and assimilation accuracies.

### Weaknesses
## Weakness
- The core idea of performing EnSF in a latent space was already achieved by Latent-EnSF [Si and Peng, 2025]. This paper's primary contributions are (1) replacing the VAE encoder from Latent-EnSF with an LSTM encoder and (2) replacing the full-space dynamics with an LDNet surrogate. While this is a successful *engineering* effort that yields speedups and imporved performances, the novlties in machine learning for DA is minimal. The paper does not provide sufficient theoretical justification for why this specific combination is fundamentally superior, beyond empirical performance.
- The paper claims the latent representation addresses sparsity. However, the actual mechanism enabling the system to work is the "*history-aware* LSTM encoder". By processing a sequence of sparse observations $(y_{1:t})$, the LSTM is implicitly learning a time-delay embedding. It is a well-established concept (e.g., via Takens' embedding theorem [Takens et al. 1981; Noakes et al., 1991]). The paper does not discuss this mechanism entirely. And, It lacks a large and highly relevant recent advances of such works for DA, for example Gottwald et al., 2021; Tarumi et al, 2025; Yang et al, 2025.
- Several key statements and claims regarding the method's motivation and mechanism are vague and require further clarification and justification, see Questions.

### Questions
## Questions

- In Line 47, what is meant by *score becomes ill-posed*? This phrasing is imprecise. Could you clarify? The term *ill-posed* refers to problems (typically inverse problems) that fail to meet one or more of the following criteria: a solution exists, the solution is unique, and the solution depends continuously on the data. There is no such definition for a function itself, e.g., score function. Do you mean the estimation of the score from sparse data is high-variance, or simply that the likelihood gradient is zero in unobserved dimensions? 
- The authors claim the latent representation itself addresses observation sparsity, e.g., Line 48-50. But no detailed justification. Could authors elabroate on: (1) How can the latent projection itself solve the sparsity problem, given that the input observation (y_t) is indeed sparse? A latent projection does not create information for that is not already present. (2) Should this capability not be attributed almost entirely to the " LSTM Encoder"? By processing a sequence of sparse observations, this encoder is implicitly learning a time-delay embedding. This concept has been used in data assimilation, yet this is not discussed and cited.
- In Line 50, the authors state that "*latent representations enable more informative gradients*". Could you please explain what "more informative gradients" means in this context? How exactly does working in the latent space make the gradients better for handling sparse observations compared to working in the original space? 
- It is a valuable strength to handle the irregular grid observation data which is commen in DA. Could the author include a numerical case study demonstrating the method’s performance on irregular grids?
- In Line 210, the author state the random projection $B$ is trianbale. However, in Random Fourier Features/positional encodings, $B$ is typically fixed [Tancik et al., 2020]; optimizing it can change the random feature distribution across update steps and making the training unstable. This is an unconventional choice and its benefits are not inituitive. Could authors provide justify the reason why set $B$ as trainable and provide an ablation study: fixed versus trainable $B$, including training loss curves and DA accuracy.


## References
- Takens, Floris. "Detecting strange attractors in turbulence." Lecture Notes in Mathematics, Berlin Springer Verlag 898 (1981): 366.
- Noakes, Lyle. "The Takens embedding theorem." International Journal of Bifurcation and Chaos 1.04 (1991): 867-872.
- Gottwald, Georg A., and Sebastian Reich. "Combining machine learning and data assimilation to forecast dynamical systems from noisy partial observations." Chaos: An Interdisciplinary Journal of Nonlinear Science 31.10 (2021).
- Tancik, Matthew, et al. "Fourier features let networks learn high frequency functions in low dimensional domains." Advances in neural information processing systems 33 (2020): 7537-7547.
- Si, Phillip, and Peng Chen. "Latent-EnSF: A Latent Ensemble Score Filter for High-Dimensional Data Assimilation with Sparse Observation Data." The Thirteenth International Conference on Learning Representations (2025).
- Tarumi, Yuta, Keisuke Fukuda, and Shin-ichi Maeda. "Deep Bayesian Filter for Bayes-Faithful Data Assimilation." Forty-second International Conference on Machine Learning (2025).
- Yang, Yiming, et al. "Tensor-Var: Efficient Four-Dimensional Variational Data Assimilation." Forty-second International Conference on Machine Learning (2025).

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LD-EnSF (Latent Dynamics Ensemble Score Filter), a novel data assimilation method designed for high-dimensional dynamical systems with extremely sparse and irregular observations. The core idea is to avoid expensive full-dimensional simulations during assimilation by operating entirely in a learned latent space. The authors train an improved Latent Dynamics Network (LDNet) to serve as a surrogate model of the system’s dynamics, and a history-aware LSTM encoder to map past sparse observations into the latent space. Assimilation is then performed via the Ensemble Score Filter (EnSF) in the latent space, jointly updating the latent state and system parameters. The method yields orders-of-magnitude speedups (up to $10^5$–$10^6\times$ faster in the experiments) compared to traditional approaches, while maintaining high accuracy and robustness. Extensive experiments on three challenging scenarios (Kolmogorov flow, tsunami propagation, and an atmospheric model) demonstrate that LD-EnSF consistently achieves the lowest estimation errors among strong baselines (LETKF, EnSF, Latent-EnSF), and remains stable even under extreme sparsity where other methods break down. The contributions include: (1) enhancing LDNet with a novel initialization, two-stage training, and a ResNet+Fourier architecture for accurate low-dimensional dynamics; (2) an LSTM-based observation encoder for irregularly-sampled, time-sequential observations; (3) the integration of these components into a fast latent-space score-based filtering framework, delivering real-time capable performance without sacrificing accuracy.

### Strengths
Significant Practical Advance: The paper addresses a critical bottleneck in data assimilation – the computational cost under high-dimensional, sparse observation settings. By eliminating full-state simulations during filtering, LD-EnSF achieves massive speedups (e.g. 200,000× in one case), enabling applications (real-time forecasting, larger ensembles) that were previously infeasible. This practical improvement is highly valuable for the community.

Robust Accuracy under Extreme Conditions: Empirical results show state-of-the-art accuracy and robustness. LD-EnSF outperforms LETKF, EnSF, and the prior Latent-EnSF by a clear margin in all tested scenarios. Notably, in an extremely sparse observation scenario (0.1% spatial and 0.2% temporal coverage with 10% noise), LD-EnSF attains about 5% RMSE, whereas LETKF and EnSF either diverge or fail to converge. The method also successfully estimates system parameters (e.g. Reynolds number, forcing amplitude) alongside the state, which is a strong plus.

Well-Motivated Method Design: The integration of learned latent dynamics with score-based filtering is novel and well-justified. The authors identify the shortcomings of VAE-based latent filters (oscillatory latent trajectories, need to propagate in full space) and introduce targeted improvements. In particular, the improved LDNet architecture (with shifted initial latent state handling, two-stage training, and Fourier-featured ResNet decoder) yields a remarkably compact and smooth latent representation that outperforms both the original LDNet and a VAE+LSTM baseline in surrogate modeling accuracy. Similarly, the LSTM-based observation encoder is a sensible choice to capture temporal context and irregular spatial sampling, surpassing the static VAE encoder used in Latent-EnSF. Each design choice is supported by discussion or ablation (e.g. Table 1, Fig. 3 for LDNet vs VAE).

Thorough Experiments: The evaluation is comprehensive. The paper tests three diverse and complex systems, includes strong baseline comparisons (including a “Latent-EnSF-dyn” variant using a VAE-dynamics baseline), and reports detailed metrics. The authors also explore scenarios with varying observation noise (Appendix E.5) and demonstrate the method’s insensitivity to ensemble size (even a single-sample “ensemble” recovers reasonable estimates, akin to MAP). Such thoroughness increases confidence in the results.

Reproducibility: The inclusion of references to code for hyperparameter search (Weights & Biases) and discussion of computational setup (CPU/GPU times in Table 2) is appreciated.

### Weaknesses
Dependence on Offline Training and Generalization Limits: 
A potential concern is the heavy reliance on training a surrogate model (LDNet) on simulation data before deployment. Acquiring a comprehensive training dataset covering all relevant system behaviors and parameter ranges can be costly, and the method’s performance may degrade if the true system behavior deviates from the training distribution. The paper’s approach is essentially as good as its learned model – for scenarios with significant model uncertainty or evolving dynamics, one might need to frequently retrain or fine-tune the surrogate. This limitation is hinted at in the conclusion (need to retrain adaptively for long-term complex systems), but it remains a practical caveat: the method inherits the generalization limitations of data-driven surrogates.

Incremental Novelty and Comparison with Variational Surrogate Approaches: 
While combining latent dynamics learning with score-based filtering is a creative and well-engineered contribution, the methodological novelty of LD-EnSF is largely incremental rather than foundational. The core components—EnKF/EnSF-style ensemble filtering, neural surrogate modeling (LDNet), and LSTM-based latent encoders—are all built upon established techniques. The innovation primarily lies in the system-level integration and practical realization of these elements, rather than in introducing fundamentally new theoretical insights or learning paradigms. Consequently, LD-EnSF may be viewed as an effective and elegant evolution of Latent-EnSF, replacing the VAE and full-state propagation with a better learned surrogate while maintaining the same overall Bayesian filtering framework.

Moreover, when comparing LD-EnSF against variational methods such as 3D-Var or 4D-Var, it should be noted that those methods can also leverage surrogate models like LDNet to reduce computational cost, provided the surrogate is differentiable for adjoint-based optimization. Thus, the runtime advantage of LD-EnSF does not arise solely from operating in latent space, but also depends on whether comparable surrogates are incorporated into the variational baseline. Explicitly discussing this relationship would clarify that LD-EnSF’s main strength lies in its practical integration of latent surrogates with score-based filtering—an important step forward, but not a fundamentally new formulation of data assimilation.

Relation to recent latent-space Bayesian filters:
The paper would benefit from discussing its relationship to other recent approaches that perform Bayesian filtering directly in learned latent spaces, such as the Deep Bayesian Filter (DBF, ICML 2025).
While both LD-EnSF and DBF share the goal of combining learned latent dynamics with probabilistic filtering, they differ in philosophy: DBF integrates inference and dynamics in an end-to-end generative framework, whereas LD-EnSF decouples the surrogate dynamics learning (LDNet) from score-based filtering.
Positioning LD-EnSF more explicitly within this broader family of latent-space Bayesian filters—highlighting differences in training objectives, inference mechanisms, and scalability—would clarify its contribution and increase its relevance to ongoing developments in this research area.

Computational Complexity and Implementation Practicality: 
While the paper convincingly demonstrates that LD-EnSF achieves major runtime savings by performing data assimilation in a low-dimensional latent space, its computational efficiency still involves important trade-offs. Each assimilation cycle requires iterative reverse-time SDE integration for $N_e$ ensemble members over $T_{diff}$ diffusion steps, leading to a total cost of $O(N_e T_{diff} d_s)$. This iteration structure is conceptually analogous to the optimization loops in 3D-Var, with the distinction that LD-EnSF's iterations are fully parallelizable across ensemble members but sequential in diffusion time. Moreover, the ensemble size $N_e$ generally needs to grow with the latent dimension to maintain statistical accuracy, as analyzed in recent works (e.g., Oko et al., 2023). Discussing these trade-offs explicitly would help clarify when LD-EnSF offers practical computational advantages over variational baselines, especially since 3D-Var can also benefit from surrogate models such as LDNet if differentiable.

In terms of implementation, the overall pipeline remains quite complex: it integrates multiple neural components (latent dynamics network, reconstruction network, and LSTM-based observation encoder) with a non-trivial score-based assimilation algorithm. Implementing and tuning such a system—including hyperparameter searches, ensuring stable LDNet training, and discretizing the SDE solver—requires substantial expertise. While the methodology is sound, this complexity may pose barriers to adoption compared to simpler ensemble or variational filters. Providing open-source code or detailed pseudocode would greatly enhance reproducibility and accessibility for the community.

Limitation and Clarity of Latent-State Initialization: 
The paper initializes all latent trajectories with a fixed zero state ($s_{-1} = 0 $) and relies on the parameter input $u_t$ to encode differences among trajectories. While this design simplifies training and stabilizes the latent dynamics, it implicitly assumes that all variation in the initial conditions can be captured through the parameter space. In practice, however, many physical systems exhibit diverse and high-dimensional initial states that cannot be fully represented by a few parameters. As a result, this assumption may limit LD-EnSF’s generalization to systems with unseen or highly variable initial conditions.

In addition, the paper’s explanation of this mechanism is somewhat unclear. The statement that the initialization “flexibly accommodates varying initial conditions” is misleading, since all trajectories start from the same latent point. Figure 1 also omits how the initial latent state is introduced or how it connects to the LSTM encoder, which may confuse readers about how the model handles initial-state diversity. Clarifying this design choice—possibly with an explicit schematic or additional discussion—and considering a learned initial encoder (e.g., mapping initial full states to latent $s_{-1}$) would both improve clarity and broaden the applicability of LD-EnSF to more general dynamical regimes.

Evaluation Scope: 
A minor weakness is that the experimental comparison could be broadened or analyzed further. The authors did not compare against particle filtering or variational methods (e.g., 4D-Var). It’s understandable given those struggle in such high-dimensional sparse settings, but discussing their expected performance or including them in a smaller-scale experiment would strengthen the positioning. Additionally, an ablation study on each novel component of LD-EnSF (for instance, using a static VAE-based encoder instead of LSTM, or using the ground truth dynamics vs. LDNet during assimilation) would isolate the contributions of each part. While the paper does compare LDNet vs VAE offline, it doesn’t explicitly show the effect of the LSTM encoder in the online phase versus a baseline. Such experiments (perhaps in an appendix) would provide deeper insight into how much each innovation (smooth latent dynamics, history encoder) contributes to the final performance.

Minor Presentation Issues: 
There are a few presentation details that could be improved. For example, the text’s statement of speedups (e.g. “$2\times10^3$ times speedup”) seems slightly inconsistent with Table 2 values – clarifying these calculations would avoid confusion. Also, the results discussion could more explicitly highlight why LD-EnSF outperforms baselines (e.g., pointing out in text that EnSF fails due to vanishing likelihood gradients in unobserved dimensions, which LD-EnSF overcomes by using informative latent gradients). The figures are generally clear; still, adding a bit more explanation in the captions or main text for Figure 4 (e.g., explaining the behavior of “Latent-EnSF-dyn” curves, or noting that LETKF diverged in the hardest case) would be helpful. These are minor issues and easily addressable.

### Questions
Generality to Out-of-Distribution Dynamics: How well would LD-EnSF handle scenarios where the true dynamics deviate from the training data? For instance, if the system experiences an unforeseen regime or a parameter outside the trained range, would the assimilation degrade gracefully, or could it diverge? Did you observe any cases where the learned LDNet struggled when the truth lay outside its training distribution?

Ablation on Observation Encoder: Have you evaluated the impact of the LSTM-based observation encoder versus a simpler or non-temporal encoder? For example, how would Latent-EnSF perform if augmented with your LDNet but still using the original VAE observation encoding at each time step (ignoring history)? This would isolate the benefit of the history-aware LSTM. Similarly, what happens if observations are on a fixed grid – does the LSTM still offer advantages over a time-independent encoder?

Parameter Estimation Performance: The method jointly assimilates state and parameters. Can you provide more insight into how accurately the uncertain parameters (Re, initial bump location, forcing amplitude/spread) were estimated in your experiments? It would be useful to know, for example, the final parameter RMSE or if the filter consistently converges to near-true parameter values. This would demonstrate the effectiveness of treating $(s_t, u_t)$ together in the state vector.

Ensemble Size and Filter Stability: You mentioned that increasing ensemble size beyond a point had minimal impact on LD-EnSF’s accuracy. Could you elaborate on this? Is LD-EnSF less sensitive to ensemble size because the score-based update effectively approximates the Bayesian posterior even with few samples? Any intuition on why, say, even 1 or 5 ensemble members can yield good results would be interesting – it’s an intriguing contrast to standard EnKF which typically benefits from larger ensembles.

Computational Overheads: Table 2 shows dramatic speedups in the online phase. Could you comment on the offline costs (training LDNet and LSTM) relative to those gains? For a fair real-world assessment, one might consider how many assimilation cycles are needed to amortize the training cost. Do you envision scenarios (like repeatedly assimilating in the same system) where the upfront training is justified by many future uses of the model?

### Soundness
3

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
4

### Summary
The paper proposes LD-EnSF, a score-based Bayesian filtering method that performs all assimilation steps in a low-dimensional latent space learned by an improved Latent Dynamics Network (LDNet) and coupled to a history-aware LSTM observation encoder. The method extends EnSF (ensemble score filtering) to handle severe spatiotemporal sparsity without resorting to costly full-state simulations by: (i) learning smooth latent dynamics that can be time-stepped (Euler) with parameter inputs, plus a stronger reconstruction network; (ii) mapping sparse, possibly irregular observations $𝑦_{1:𝑡}$  to latent pairs $(𝑠^𝑡, 𝑢^𝑡)$ with an LSTM; and (iii) running EnSF entirely in latent space then decoding to physical space only when needed. Overall this presents a principled way of approaching the problem. Experiments on Kolmogorov flow, tsunami (shallow water), and a forced hyperviscous rotating atmosphere show lower assimilation RMSE than EnSF, Latent-EnSF (VAE), and LETKF, with very large runtime speedups because full-space propagation is replaced by latent dynamics

### Strengths
Decoder-free assimilation loop: All filtering happens in latent space (state + params), avoiding per-step decoding and cutting both compute and error accumulation.

Efficiency at scale: Small latent dimension + ensemble updates → orders-of-magnitude cheaper than full-state DA; design is hardware-friendly and parallelizable.

Robustness features: Reverse-SDE damping and simple latent noise modeling make the update stable under severe sparsity/noise.

Clear training recipe: Two-stage LDNet training with well-specified schedules/hparams improves reproducibility and stability.

Extensive Evaluation : Demonstrated on Kolmogorov flow, tsunami (shallow water), and a rotating atmosphere covering increasing complexity and different observation settings.

Thorough ablations: Sensitivity to latent rank, ensemble size, observation density/cadence, noise level, and architecture choices; includes OOD initial-condition tests.


Reproducibility: Detailed setups (observation models, training schedules, metrics (see Appendix) ) and systematic reporting make results believable and repeatable.

### Weaknesses
Overall the paper presents a clear contribution with thorough experimentation. Following are my major concerns : 

Related Work Missing : 

The paper under-cites several very relevant 2024–2025 works in  that would strengthen positioning:

Neural Operators for DA and Semilinear PDEs : Fourier Neural Operator and SFNO have presented great result in PDE/wether modeling but no discussion have been provided in regard to them. Additionally Semilinear Neural Operator (ICLR 2024) that proposes a recursive neural-operator framework that explicitly addresses prediction and data assimilation for semilinear PDEs have also not been cited ; 
Neural Koopman priors and Koopman-based DA : Frion et al., “Neural Koopman Prior for Data Assimilation” formulates DA with a neural Koopman prior (now a TSP 2024 article), and KODA (arXiv 2024) integrates forecasting with an online data-assimilation loop using Koopman-guided components. Both connect directly to low-dimensional latent linearizable dynamics for DA, much like LD-EnSF’s latent evolution;  Modern 4D-Var and deep 4D-Var variants : The paper cites classical 3D/4D-Var (e.g., Rabier & Liu 2003) but misses recent learned or hybrid 4D-Var systems that tackle cost and sparsity with neural parameterizations—e.g., 4DVarNet (end-to-end DA backboned on variational objectives), En4DVarNet for uncertainty, 4DVarFormer (attention-based 4D-Var surrogate with rapid multivariate analyses), and operational-scale hybrids like FuXi-En4DVar. These are important baselines or at least conceptual references for the atmosphere case and the efficiency narrative.

Action item : by adding a paragraph in Related Work discussing (SNO/NO-DA, ClimODE, Koopman-DA, deep/hybrid 4D-Var), explaining what LD-EnSF gains with a training-free score component and learned latent dynamics, and why that’s preferable under extreme sparsity.

Technical Weaknesses :
 
Latent observation model = identity. The filter assumes $𝐻_{latent}(𝜅_𝑡)=𝜅_𝑡$, so the LSTM’s outputs $(𝑠_𝑡,𝑢_𝑡)$	​
 are taken as direct noisy measurements of the true latent pair $(𝑠_𝑡,𝑢_𝑡)$. Any encoder bias/miscalibration directly contaminates the update; there’s no learned/structured latent observation operator to absorb mismatch.

Decoder-free assimilation hides reconstruction bias. Assimilation runs entirely in latent space (a strength), but it also means reconstruction errors don’t get corrected during the loop. If the decoder has bias, final physical-space fields can drift even when latent RMSE falls. (The paper itself emphasizes that decoding is only needed at the end, which is fast, but leaves this feedback gap.)

Noise handling is crude in latent space. The latent observation noise 𝛾^_{𝑡} is estimated post-hoc and then treated as uniform across latent dimensions. That’s convenient but brittle when different latent directions have very different uncertainty.

### Questions
Fourier Neural Operators have shown great success recently in weather and PDE modelling. Why have no discussion been provided in that regards?

Latent noise modeling: In Eq. (8) the identity $H_{latent}$ and a scalar 𝛾^_{𝑡}  are assumed via empirical estimation. How sensitive is EnSF’s update to misspecifying 𝛾^_{𝑡} across latent dimensions? Could you learn a diagonal or low-rank covariance in latent space cost-effectively? 

Are proposals for $u_{t}$ purely resampling from the previous posterior, or is there diffusion/jitter? What happens when u varies slowly or remains static? 

Physical time vs latent  Δt: Since Δt is tuned, how does the method behave if the observation sampling rate changes (e.g., sparser/faster streams) without re-training? 

When LDNet struggles (very long horizons, regime changes), could LD-EnSF seamlessly “fallback” to occasional full-model nudging (a la hybrid LD-EnSF/4D-Var) while keeping most steps latent?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes LD-EnSF, a score-based data assimilation framework that combines Latent Dynamics Networks (LDNets) and an LSTM-based observation encoder to perform Bayesian filtering directly in a low-dimensional latent space.

Traditional data assimilation methods such as EnKF or EnSF are computationally expensive because they operate in the full physical space and require repeated forward model simulations. LD-EnSF negates this by learning surrogate dynamics in latent space and performing all filtering steps there, using an ensemble score filter (EnSF) to update the posterior distribution. The LSTM encoder processes sparse and irregular observations and aligns them with the latent states and parameters.

Empirical validation is conducted on three different systems (Kolmogorov flow, tsunami modeling, and atmospheric dynamics) and under severe spatial and temporal sparsity. The method demonstrates strong improvements in accuracy and speed, with orders of magnitude reductions in runtime while maintaining or improving assimilation accuracy.

### Strengths
Clear Motivation and Relevance
- The paper tackles a key limitation of recent score-based filters, their high computational cost and poor performance with sparse observations. 

Solid Technical Design
- The integration of latent surrogate dynamics (LDNet) and score-based Bayesian filtering (EnSF) is smart.
- The introduction of a history-aware LSTM observation encoder effectively extends the latent assimilation framework to handle irregular and sparse data.

Comprehensive Experiments
- The authors test across multiple physical systems of increasing complexity.
- Results include both structured and unstructured observation setups, multiple levels and differing types of noise, and sensitivity analyses.

Extensive Ablation and Robustness Studies
- Appendices systematically evaluate noise robustness, out-of-distribution generalization, latent dimension sensitivity, ensemble size, and architectural design choices.

### Weaknesses
Limited Theoretical Novelty
- The proposed method primarily combines existing techniques (LDNet, EnSF, LSTM encoding). While the combination is well-executed and impactful the theoretical advancement is modest. The novelty lies more in the integration and empirical rigor.

Benchmark Coverage and Positioning
- Comparisons are limited to EnSF, Latent-EnSF, and LETKF. While these are strong and relevant baselines, the paper could benefit from a clearer discussion of recent efficient variational and diffusion-based data assimilation approaches, such as Tensor-Var: Efficient Four-Dimensional Variational Data Assimilation (Yang et al., 2025) and DiffDA: A Diffusion Model for Weather-Scale Data Assimilation (Huang et al., 2024). Although implementing these methods in the current setup may not be straightforward, a more explicit positioning in the Related Work section would help situate LD-EnSF within the broader landscape.

### Questions
1. Theoretical Guarantees:
Can the authors comment on whether any convergence guarantees exist for the latent-space assimilation process, particularly as a function of the surrogate model error?

2. Dynamic Re-training:
How feasible is LD-EnSF in systems where governing dynamics change over time? Could partial retraining or online fine-tuning mitigate the offline cost?

3. Uncertainty Quantification:
How reliable are uncertainty estimates when mapped back to physical space?

### Soundness
4

### Presentation
4

### Contribution
3
