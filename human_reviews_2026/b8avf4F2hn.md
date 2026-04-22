# Compositional Diffusion with Guided search for Long-Horizon Planning

- Avg Score: 6.50
- Decision: Accept (Oral)
- Scores: 6, 8, 6, 6

## Abstract
Generative models have emerged as powerful tools for planning, with compositional approaches offering particular promise for modeling long-horizon task distributions by composing together local, modular generative models. This compositional paradigm spans diverse domains, from multi-step manipulation planning to panoramic image synthesis to long video generation. However, compositional generative models face a critical challenge: when local distributions are multimodal, existing composition methods average incompatible modes, producing plans that are neither locally feasible nor globally coherent. We propose Compositional Diffusion with Guided Search (CDGS), which addresses this \emph{mode averaging} problem by embedding search directly within the diffusion denoising process. Our method explores diverse combinations of local modes through population-based sampling, prunes infeasible candidates using likelihood-based filtering, and enforces global consistency through iterative resampling between overlapping segments. CDGS matches oracle performance on seven robot manipulation tasks, outperforming baselines that lack compositionality or require long-horizon training data. The approach generalizes across domains, enabling coherent text-guided panoramic images and long videos through effective local-to-global message passing. More details: https://cdgsearch.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The author introduces Compositional Diffusion with Guided Search (CDGS), a novel framework for generating coherent long-horizon plans and content by addressing the critical challenge of mode-averaging in compositional generative models. Existing compositional approaches, while efficient in leveraging short-horizon local generative models, often average incompatible modes when local distributions are multimodal, leading to globally incoherent or locally infeasible outputs. CDGS integrates a guided search mechanism directly into the diffusion denoising process to overcome this limitation. The authors demonstrate the effectiveness of CDGS across three challenging long-horizon domains: robotic manipulation planning, panoramic image generation, and long video synthesis, showing significant improvements over naive compositional baselines.

### Strengths
1. The paper is well-structed and easy to follow. The proposed approach is well-grounded in motivation, and the empirical results provide compelling evidence of its effectiveness.
2. The paper tackles a fundamental and highly significant problem in generative modeling.

### Weaknesses
1. Some implementation details are missing to fully replicate the method. Notably, the candidate "repopulation" step lacks clarity on how new batches are generated from elite plans.
2. CDGS requires additional resampling iterations and pruning steps which can require many additional forward passes of the model. But the paper doesn’t report the wall-clock runtime or compute comparisons, which makes it hard to judge the practical overhead.
3. Although this paper focuses on inference-time trajectory stitching methods, I suggest the authors include related works on trajectory data augmentation methods (e.g., [1,2]) that address similar problems.
4. CDGS uses a curvature-based approach to approximate likelihoods for pruning low-quality plans. However, there are existing reconstruction-based methods for evaluating sample quality in diffusion models, such as restoration gap [3] and minority score [4]. Could you clarify the advantages of the curvature-based approach over reconstruction-based alternatives for likelihood approximation?  
5. (optional) It would strengthen the paper if the authors could provide empirical comparisons demonstrating the effectiveness of the curvature-based metric compared to reconstruction-based approaches for plan pruning.

[1] State-Covering Trajectory Stitching for Diffusion Planners, 2025

[2] Extendable Long-Horizon Planning via Hierarchical Multiscale Diffusion, 2025

[3] Refining Diffusion Planner for Reliable Behavior Synthesis by Automatic Detection of Infeasible Plans, 2023

[4] Don’t Play Favorites: Minority Guidance for Diffusion Models, 2024

Note: I still think this work is promising. Although a borderline accept is given currently, I'd be open to increasing my score upon seeing suitable revisions or clarification addressing my concerns.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors propose Compositional Diffusion with Guided Search (CDGS), a novel framework for compositional generation using diffusion models. The work primarily addresses the "mode-averaging" problem inherent in prior score-averaging approaches. This issue arises when composing multi-modal local distributions, where existing methods tend to average incompatible local modes, resulting in globally incoherent and infeasible plans.

To overcome this limitation, CDGS introduces two key inference-time strategies. First, it employs iterative resampling of local generations based on the compositional score. This process functions as an effective message-passing mechanism to propose globally coherent candidate samples by propagating information across the entire sequence. Second, the framework filters these candidates using a likelihood-based pruning strategy. It generates a diverse population of candidate plans and then selects and re-populates the top-*K* candidates based on a proposed local feasibility metric. The authors offer a novel approach to approximate this metric by leveraging the insight that high-likelihood samples follow low-curvature paths during the DDIM inversion process. Specifically, this curvature is measured by the norm of the time-derivative of the score network's noise prediction, allowing for an efficient, likelihood-informed search.

The authors empirically validate CDGS across a diverse set of long-horizon tasks, including robotic planning, panoramic image generation, and long video generation. The results consistently demonstrate the superiority of CDGS over previous methods. The paper further substantiates the efficacy of its components through ablation studies and detailed analyses on scaling effects. Notably, the method demonstrates strong performance on complex robotic planning tasks where baseline approaches like GSC fail. Even when provided with the oracle symbolic task plan, GSC struggles with compositional motion planning due to mode-averaging, whereas CDGS successfully finds a coherent solution by integrating task and motion planning within its guided search, highlighting the critical importance of the proposed mechanism. The authors ensure reproducibility by providing code and detailed experimental settings in the appendix, which significantly aids in the concrete understanding of the proposed method.

### Strengths
- **Clear Problem Formulation and an Effective Solution**: The paper clearly articulates critical limitations of prior score-averaging based compositional diffusion methods—the mode-averaging and global incoherence problems. It then introduces a well-motivated and effective solution, CDGS, which synergistically combines two key strategies: (1) iterative resampling to enforce global coherence through message passing , and (2) a novel likelihood-based pruning mechanism to filter out locally infeasible candidates. This problem-solution framing is a significant strength of the paper.
    
- **Convincing Empirical Validation across Diverse Domains**: The generalizability of CDGS is convincingly demonstrated through extensive experiments on a variety of challenging long-horizon tasks, including robotic planning , panoramic image generation , and long video generation. The consistent and significant performance improvements over baselines across these different domains strongly support the claim that CDGS is a broadly applicable, task-agnostic framework.
    
- **Thoroughness and Commitment**: The paper is well-supported and transparent, which significantly strengthens its contribution.
    
    - **Reproducibility**: The authors provide a link to the source code and meticulously detail all hyperparameters for the different experimental domains in the appendices (e.g.,  Appendices G, K, L) .
        
    - **In-depth Analysis and Discussion**: The appendices offer deep dives into crucial components, providing a detailed derivation of the pruning objective (Appendix E) , a thoughtful comparison with prior methods (Appendix D) , and insightful ablation studies on scaling effects (Appendix H).
        
    - **Transparency**: The authors transparently document the full experimental setup, including prompts and task specifications (Appendices C, F, I, J) , and proactively discuss the limitations of their work (Appendix A), demonstrating a mature and thoughtful research approach.

### Weaknesses
- **Lack of Adaptive Computation and Backtracking**: While CDGS effectively overcomes limitations of prior methods through inference-time search, it lacks adaptability.
	- **Fixed Computational Overhead**: The proposed method utilizes a fixed computational budget regardless of task difficulty, as determined by hyperparameters (Appendix G.2) . This can lead to inefficient, excessive computation for simpler tasks, or insufficient computation for more complex ones.
	- **Absence of Backtracking**: CDGS does not support backtracking, which limits its ability to explore diverse possibilities at various depths—a crucial capability for solving complex reasoning and generation problems. In contrast, several recent inference-time scalable diffusion models [1,2,3] offer both adaptability to solve problems within a given budget and backtracking for effective deep search.

- **Ambiguous Link Between Iterative Resampling and Mode-Averaging**: The paper presents iterative resampling and likelihood-based pruning as solutions to the mode-averaging problem. While both contribute to this goal, their primary roles appear distinct. Likelihood-based pruning is a more direct countermeasure to mode-averaging by filtering out incompatible local modes. In contrast, iterative resampling seems to primarily address a different, albeit related, issue: global incoherence. This problem could arise even in compositional methods that do not use score averaging. The paper would be significantly strengthened by a more precise discussion on how iterative resampling specifically mitigates mode averaging, perhaps by explaining how improving global coherence helps resolve local mode conflicts.

- **Suggestions for Presentation and Clarity**: The paper's presentation could be improved in several areas.
	- **Clarity in TAMP Section**: The discussion on Task and Motion Planning (TAMP) in the main paper is brief and could be confusing for readers. Clarifying why it is a “hybrid-planning problem”, defining key terms like PDDL, and explaining the different representations of states/actions ($s_i$,$a_i$​) at high vs. low levels would be beneficial . Crucially, the fact that the GSC baseline is conditioned on an oracle task plan—a key factor for performance analysis—is only mentioned in Appendix D and should be in the main paper .
	- **Minor Errors and Formatting**: Several minor issues were noted.
	    - Line 316: The index in the set definition appears to have a typo ($s_h,a_h$ → $s_i,a_i$).
	    - Appendix G's title, "more details on robotic planning experiments," is misleading as it only covers TAMP, not OGBench.
	    - Some qualitative results could be discussed as limitations. For example, in the Appendix B visualization for the prompt “Silhouette wallpaper of a dreamy scene with shooting stars,” some global incoherence is still visible even applying iterative resampling and pruning.
	    - Minor capitalization and spacing issues were found in Appendices G.2 lines 1248, 1251, 1254 and H.1 between the paragraph and table 8 caption.
- **Compliance with LLM Usage Policy**: The authors do not state whether Large Language Models (LLMs) were used in preparing the manuscript, which is required by the ICLR 2026 Author Guide.

[1] Yoon, Jaesik, et al. "Monte carlo tree diffusion for system 2 planning." _arXiv preprint arXiv:2502.07202_ (2025).

[2] Yoon, Jaesik, et al. "Fast Monte Carlo Tree Diffusion: 100x Speedup via Parallel Sparse Planning." _arXiv preprint arXiv:2506.09498_ (2025).

[3] Lee, Gyubin, et al. "Adaptive Cyclic Diffusion for Inference Scaling." _arXiv preprint arXiv:2505.14036_ (2025).

### Questions
- In the 1D DDIM Inversion study (Appendix E, Fig. 8), the denoising latent paths (blue) appear to converge more strongly to the in-distribution modes compared to the DDIM inversion paths (red). Could you elaborate on the intuition or mechanism behind this visual result?
- The model utilizes a Mixture-of-Experts (MoE) architecture (Appendix G.1). What was the motivation for this specific choice over other potential architectures, and were any ablation studies performed to validate this design?
- The sampling strategy (Appendix G.2, Table 6) employs an adaptive schedule for resampling ($U(t)$) and specific ratios for exploration ($k_e$​) and pruning ($k_p$​). Have you experimented with simpler schedules (e.g., uniform resampling) or different values for $k_e$​ and $k_p$​ to test the sensitivity of these crucial hyperparameters?
- Appendix H.1 presents an ablation, CDGS (BFS-2), which uses system dynamics rollouts and is described as an “upper bound” of the proposed approach. Could you clarify what this result implies?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a novel method, Compositional Diffusion with Guided Search (CDGS), for long-horizon planning tasks. The core idea is to combine the generative power of diffusion models with a structured search algorithm to create complex and coherent plans over long horizons. CDGS works by composing shorter-horizon plans generated by a diffusion model and then using an iterative resampling and pruning mechanism to guide the search towards feasible and high-quality long-horizon plans. The authors demonstrate the effectiveness of their approach on a variety of challenging tasks, including robotic manipulation, long-form video generation, and panoramic image synthesis, showing significant improvements over existing methods.

### Strengths
1. The paper introduces a highly novel and significant approach to long-horizon planning. The combination of compositional generative models (diffusion models) with explicit guided search is a powerful paradigm that effectively balances generation diversity with goal-directedness. This work has the potential to influence future research in planning, robotics, and generative modeling.
2. The experimental evaluation is thorough and convincing. The authors demonstrate the superiority of CDGS across multiple, distinct, and challenging domains. The qualitative results, especially in video and panorama generation, are impressive and clearly showcase the model's ability to maintain long-term coherence. The quantitative results on robotic planning tasks also show clear improvements over state-of-the-art baselines.
3. The proposed method is well-grounded in mathematical principles. The use of compositional score functions and the DDIM-based pruning objective are elegant and well-motivated. The paper provides sufficient detail to understand the core mechanics of the algorithm, and the mathematical derivations appear to be correct and consistently applied.

### Weaknesses
1. The proposed method seems computationally intensive. The iterative nature of resampling and refining plans at each step of the search could be very time-consuming, which might limit its applicability in real-time or resource-constrained settings. A more detailed analysis of the computational complexity and wall-clock time comparisons with baselines would be beneficial.
2. The paper could benefit from providing more details on how the states and actions for the robotic planning tasks are represented and fed into the diffusion model. Understanding the specifics of the parameterization would help in assessing the method's applicability to other robotic tasks.
3. While the paper does present some ablation studies (e.g., in Appendix H), a more in-depth analysis of the contribution of each component of CDGS would strengthen the paper. For instance, a clearer study on the comparison of different pruning objectives could provide valuable insights.

### Questions
1. The pruning objective $g(x_0^{i})$ is based on the curvature of the denoising path. While intuitive, could the authors provide a more formal justification for why this particular metric is a good proxy for plan feasibility or quality? Have the authors experimented with other pruning criteria, and how sensitive is the performance of CDGS to the choice of this objective?
2. How does the performance and computational cost of CDGS scale as the planning horizon $H$ increases? Is there a point where the composition of many short-horizon plans leads to a degradation in quality or an explosion in computational requirements?
3. The paper mentions the use of a Mixture-of-Experts (MoE) layer in the score model. How critical is the MoE component for the success of CDGS, especially in multi-modal tasks?
4. The results presented are very positive. It would be instructive to see some failure cases of CDGS. In which scenarios does the method struggle to produce coherent long-horizon plans? This could shed light on the limitations of the current approach and point to avenues for future work.

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
5

### Summary
The paper presents Compositional Diffusion with Guided Search (CDGS), a generative method that augments compositional diffusion sampling with population-based guided search and likelihood-based pruning to mitigate mode-averaging when composing multimodal local distributions. The proposed method aims to unify diffusion-based planning and long-horizon content generation under a single approach, demonstrating its effectiveness in not only robotic long-horizon manipulation tasks but also in panoramic imaging and video generation. While the motivation is compelling and the experiments effectively highlight the contributions, the work overemphasizes conceptual novelty and lacks rigorous comparisons with modern and traditional planning-based methods.

### Strengths
1. The paper identifies a genuine issue in compositional generative modeling: mode-averaging across multimodal local distributions, which degrades global coherence in long-horizon tasks. In addition, CDGS is evaluated across robotics, image synthesis, and video generation, demonstrating domain-agnostic adaptability at the conceptual level.

2. The method is implemented end-to-end with population-based inference and ablations (with/without pruning, resampling, scaling analysis), demonstrating the engineering completeness. Despite limited novelty, the approach achieves reasonable performance improvements on OGBench, TAMP, and long-horizon generative benchmarks.

3. The figures, especially Figure 4, effectively illustrate the proposed resampling and pruning mechanisms in an end-to-end manner. Algorithms and Equations are clearly presented, with a strong emphasis on reproducibility.

### Weaknesses
1. The population-based denoising, iterative resampling, and pruning loops likely incur heavy compute costs. No runtime or complexity analysis is provided to justify practicality. Classical planners, such as SVG-MPPI [1] and Reverse-KL MPPI [2], already produce mode-seeking behavior in closed form, with stronger theoretical rigor and lower computational overhead. Within this realm, CDGS lacks a clear justification for its adoption over these established alternatives.

2. CDGS essentially reuses established ideas, including population-based search, iterative resampling, and likelihood pruning. Indeed, it repackages under the umbrella of diffusion inference without introducing fundamentally new theory or algorithmic principles. The paper lacks direct traditional predecessors such as SVG-MPPI [1], Reverse-MPPI [2], and SV-MPC [3], which already address multimodality via reverse KL minimization and Stein variational guidance. Without such baselines, it is unclear how CDGS advances beyond established mode-seeking control frameworks. More importantly, the paper fails to position itself in relation to modern diffusion-based mode-seeking methods, such as Flow to the Mode [4] and Mean-Shift Distillation [5], which explicitly address mode collapse and mode-averaging in generative diffusion models.

3. The pruning metric J(τ) based on DDIM curvature is heuristic and unvalidated. There is no proof linking it to likelihood, feasibility, or global convergence. In contrast, SVG-MPPI provides a clear variational derivation and theoretical backing. Moreover, the stability and safe validation are largely omitted, which might be acceptable for video generation or image synthesis, but not for robot control for long-horizon tasks.

4. The experiments presented are too limited in scope to establish the superiority of CDGS convincingly. The evaluation mainly contrasts against lightweight or outdated baselines (e.g., GSC, Diffuser, CEM-based methods) and omits stronger diffusion-based or traditional mode-seeking approaches, such as Flow to the Mode [4], Mean-Shift Distillation [5], and SVG-MPPI [1]. The claims of generality and cross-domain performance are not substantiated by statistically rigorous analysis or comprehensive comparisons. Without large-scale quantitative studies, ablations on computational trade-offs, or benchmarks against state-of-the-art mode-seeking diffusion and planning frameworks, the empirical validation remains insufficient to justify the claimed novelty or practical advantage of the proposed method.

[1] Honda, Kohei, et al. "Stein variational guided model predictive path integral control: Proposal and experiments with fast maneuvering vehicles." 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024.
[2] Kobayashi, Taisuke, and Kota Fukumoto. "Real-time sampling-based model predictive control based on reverse kullback-leibler divergence and its adaptive acceleration." arXiv preprint arXiv:2212.04298 (2022).
[3] Lambert, Alexander, et al. "Stein variational model predictive control." arXiv preprint arXiv:2011.07641 (2020).
[4] Sargent, Kyle, et al. "Flow to the mode: Mode-seeking diffusion autoencoders for state-of-the-art image tokenization." arXiv preprint arXiv:2503.11056 (2025).
[5] Thamizharasan, Vikas, et al. "Mean-Shift Distillation for Diffusion Mode Seeking." arXiv preprint arXiv:2502.15989 (2025).

### Questions
1. Could you position CDGS against modern mode-seeking control/planning frameworks (e.g., Reverse-KL MPPI / SV-MPC / SVG-MPPI) and mode-seeking diffusion methods (e.g., mean-shift/flow-to-mode)? Why is CDGS preferable when the objective is to avoid “mode averaging”?
2. Why is multiplicative aggregation over segments (Eq. 5) preferable to additive/soft-min aggregations? Any sensitivity analysis to the aggregation choice?
3. Figure 5 shows the benefits from increasing $B$ and $U$. Where are the diminishing returns, and how do you set $\lambda_t$ (exploration/exploitation) across timesteps? Provide a schedule and an ablation.
4. In OGBench (Table 1), please report the number of seeds, confidence intervals, and effect sizes vs. the strongest baseline (not only means). Also, clarify whether CDGS uses less long-horizon supervision than IRL baselines in practice.
5. Do you enforce hard preconditions/effects (e.g., in-hand (hook)) during sampling to guarantee task-level feasibility, or is feasibility entirely delegated to DDIM-curvature pruning?

### Soundness
2

### Presentation
3

### Contribution
3
