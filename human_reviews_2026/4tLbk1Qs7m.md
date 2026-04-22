# RISE: A Statistical Perspective for Adversarial Attacks against Closed-Source MLLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
This paper studies the critical problem of targeted adversarial attacks against closed-source MLLMs, which aims to generate highly transferable adversarial samples with open-source MLLMs. Previous approaches typically focus on maximizing the similarity of latent representations between adversarial samples and target samples. However, these approaches could overfit specific target samples with severely limited generalization ability to closed-source MLLMs. Towards this end, we propose a novel approach named Relational Distribution-aware Intrinsic Alignment (RISE) for adversarial attacks against closed-source MLLMs. The core of our RISE is to adopt a statistical lens to characterize intrinsic semantics of images for more generalized and robust alignment. In particular, each augmented image is considered as an example from the intrinsic distribution of the original image. Then, we utilize non-parametric energy distance to measure the distribution divergence, which is naturally adopted for the semantic alignment in the hidden space. To further transferability to specific target models, we learn a Graph Neural Network (GNN) to explore the complex relations between source and target MLLMs on transferability and adaptively select surrogate source models for different target MLLMs. Extensive experiments on benchmark datasets validate the effectiveness of the proposed RISE in comparison to competing baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes RISE, a transfer attack framework that replaces point-wise latent alignment with distributional alignment using Energy Distance, combined with a GNN Attack Router that selects a task-adaptive surrogate ensemble. Empirically, RISE improves targeted transfer to both open-source and closed-source MLLMs (GPT-4o/4.1, Gemini-2.0, Claude-3.5/3.7; also o3 and Claude-Sonnet-4) under an $\ell_\infty$ budget, measured via LLM-as-a-judge metrics (ASR/AvgSim, KMR). The method is positioned as statistically principled (non-parametric metric, augmentation-driven sampling) and data-driven for model selection. Results report consistent gains over M-Attack and FOA-Attack across multiple settings.

### Strengths
- Conceptual shift: From point-wise to distributional alignment; Energy Distance objective is well-motivated and formalized (global point→dist, local dist→dist)
- Adaptive ensembling: GNN Attack Router learns transfer relations from logs and selects a Top-K surrogate subset per task; practical and novel in this context. 
- Broad empirical gains: Improvements on six open-source MLLMs and five closed-source MLLMs; additional results on models “with enhanced reasoning.” 
- Ablations and trade-offs: Sensitivity to number of augmentations M and weighting η; selection of K=2 justified via accuracy/compute trade-offs. 
- Clear pipeline & pseudocode: Offline router training and online attack generation are described with algorithms, aiding reproducibility.

### Weaknesses
- Evaluation dependence on LLM-as-judge: ASR uses GPTScore thresholding with the same MLLM to caption both images; potential circularity/bias is not stress-tested or reported with sensitivity analyses
- No multi-seed confidence intervals or per-seed tables; variance across randomness sources (augmentations, sampling, router selection) is unquantified
- Primary focus on $\ell_\infty$ with $\epsilon$=16/255, step size 1/255, 300 iters; limited visibility into $\ell_2$ or oother budget sweeps in the main paper. 
- The GNN Attack Router learns from historical logs on 800 pairs and is evaluated on 200; risks of distribution shift or leakage are not deeply analyzed.

### Questions
- Could you report multi-seed results with confidence intervals to check stability across randomness?
- How sensitive are ASR and KMR to the GPTScore threshold and to using a different judge than the captioning MLLM (to reduce judge–model coupling)? 
- What is the latency/throughput overhead of RISE (per image) vs FOA/M-Attack, and how does it scale with M and K (beyond the partial trade-off shown)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The method reframes targeted transfer against closed-source MLLMs as distributional alignment with Energy Distance over augmented views, replacing fragile pointwise feature matching. A learned GNN Attack Router selects a task-adaptive Top-K surrogate ensemble to balance transfer strength and compute.
- Experiments on open and closed models report higher ASR/AvgSim/KMR than FOA/M-Attack under common ℓ∞ budgets, arguing for a statistically grounded and data-driven framework.

### Strengths
- The shift to distribution-level alignment with Energy Distance is methodologically coherent, aligning the attack with a two-sample statistical perspective and reducing dependence on brittle pointwise feature matching.
- The coupling of global p→d and local d→d objectives plausibly preserves both scene-level and object-level semantics during perturbation optimization
- The router casts surrogate selection as supervised meta-learning over transfer logs, yielding compact ensembles that retain transferability under constrained K
- The paper surfaces actionable implementation specifics with pseudocode and ablations, facilitating reproducibility
- The evaluation spans open and closed targets and reports multiple metrics, indicating gains are not tied to a single success criterion

### Weaknesses
- Heavy reliance on LLM-as-judge risks circularity/bias; without judge-swap or human validation, reported ASR may be optimistic.
- Lack of strictly compute-matched comparisons means increases in M and K may conflate algorithmic gains with extra gradient budget or wall-clock.
- Advantages of Energy Distance over other distance metrics are clear
- The router’s behavior under distribution shift seems insufficiently explored
- Black-box targeted transfer to unseen architectures and cross-dataset tests are limited, so robustness to closed-source variability is not fully established.

### Questions
- The authors used fixed attack hyperparameters (\epsilon=16/255 with a fixed step size, etc.) and do discuss the compute trade-off, but does not discuss memory/per-image latency/throughput details.
- The LLM-as-a-judge protocol uses the same family to caption both adversarial and target images and thresholds GPTScore at 0.3, but I cannot find judge-swap or human validation; please indicate where these sensitivity analyses appear or add them
- how does this method perform in comparison to other methods on other source->target transfer settings besides NPIS2017->COCO?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper aims to improve the transferability of adversarial attacks on Multimodal Large Language Models (MLLMs) from a statistical perspective. An adaptive ensemble mechanism is further designed to enhance attack performance. Experimental results show that the proposed method outperforms compared approaches.

### Strengths
1. Improving adversarial transferability is an important and meaningful research direction.
2. Experimental results demonstrate that the proposed approach achieves better performance than compared methods.

### Weaknesses
1. The motivation behind the proposed method is unclear. Although the authors claim to enhance adversarial transferability from a statistical viewpoint, they merely “hypothesize”  its effectiveness without offering solid theoretical or empirical justification.
2. The paper focuses on targeted attacks but presents them as general adversarial attacks. Since adversarial attacks include both targeted and untargeted settings, this distinction should be explicitly clarified.
3. In the second paragraph of page 2, the authors state that “This overfitting results in two key issues.” However, the subsequent text describes the causes of overfitting and limitations of existing methods, rather than clearly identifying the two key issues.
4. On page 6, the authors assert that “these strategies cannot capture the complex, asymmetric, and task-dependent transferability relationships,” but no supporting evidence or rationale is provided.
5. The ensemble evaluation is limited. Only three models are included in the pool, and the best performance is obtained using just two models. This setup fails to convincingly demonstrate the effectiveness of the adaptive ensemble strategy, and the authors do not explain why using two networks yields the best results.
6. The computational cost of the proposed method, compared with other approaches, is not discussed.
7. The attack performance is primarily evaluated based on feature similarity, without experiments across diverse tasks. Since MLLMs can be used in a variety of downstream applications, a feature-based evaluation alone cannot fully reflect the practical effectiveness of the attacks.
8. Adversarial attacks constitute a long-standing research area with many established methods across different domains, including conventional deep neural networks. The paper lacks a comprehensive literature review and should compare the proposed method against these broader approaches.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Relational Distribution-aware Intrinsic Alignment (RISE), a transfer attack framework for MLLMs with two main ideas: (i) move from pointwise feature matching to distributional alignment in a surrogate’s latent space using Energy Distance; and (ii) replace heuristic model ensembling with a GNN Attack Router trained offline on historical attack logs to choose a task-adaptive surrogate ensemble. Experiments report SOTA transferability on several open- and closed-source MLLMs under targeted captioning attacks, with metrics including ASR, AvgSim, and a keyword-based KMR.

### Strengths
1.	Clear conceptual shift: Reframing transfer from pointwise to distributional alignment is well-motivated and novel in this context. Using Energy Distance (nonparametric, no kernel tuning) is a principled choice for empirical distribution matching.
2.	Adaptive model selection: The GNN Attack Router formalizes surrogate selection as a relational prediction problem using task/model graphs, moving beyond fixed or purely loss-based weighting.
3.	Empirical gains: On open-source models (e.g., LLaVA-1.5-7B, Qwen2.5-VL-7B), RISE outperforms strong baselines including FOA-Attack; similar trends hold for closed-source models (e.g., GPT-4o/4.1, Claude-3.7).

### Weaknesses
1.	Dependence on LLM-as-Judge metrics: Reported ASR/AvgSim rely on GPT-style graders, which are sensitive to prompt design and judge choice and can diverge from human judgments. This creates uncertainty about absolute and relative gains. Please include a small human study and/or cross-judge robustness (multiple prompts and at least two independent judge models with calibration). 
2.	Router practicality & cold-start risk: The GNN Attack Router presumes access to extensive historical attack logs for offline training. In realistic deployments (especially on new or proprietary closed-source targets) curating such data is precisely the hard part, creating a cold-start loop: training the router for a new target M_new requires transfer outcomes on M_new, which in turn requires many prior attacks on that same model. Please clarify how RISE operates when no target-specific history exists and outline a fallback (e.g., metadata-based zero-shot priors, family-level generalization, or lightweight online updates). Related cold-start challenges are well-documented in GNN-based selection/routing contexts. 
3.	Scalability and Computational Overhead. The paper presents RISE as an effective framework but provides limited analysis of its computational cost, which appears non-trivial in two key areas. (1)  Building the router’s heterogeneous graph appears to require a combinatorial sweep over surrogate ensembles, tasks, and targets. This prerequisite could exceed the cost of the downstream attacks it aims to optimize. Please quantify the size of the logs, number of attacks run, and total compute. (2) The distribution-aware intrinsic mining scales roughly with augmentations M and selected ensemble size K. The paper’s own runtime analysis suggests noticeable growth with larger M; “computationally efficient” should be contextualized against simpler point-wise baselines. Consider reporting a wall-clock/throughput table vs. M and K, and a Pareto curve vs. ASR.
4.	Ambiguities in Methodology and Reproducibility. The high-level method is clear, but key details are missing for full replication—notably the router’s exact architecture (GNN layer type, depth, hidden sizes, readout), training data composition, and update cadence as models evolve. Please include these in the main text or an appendix, and provide seeds, config files, and code for the router pipeline.

### Questions
1.	Router generalization: If a target model family was unseen during router training, how does selection degrade? Any leave-one-family-out results?
2.	Data / compute: What is the exact size of the historical log, feature dimensionality, and training cost for the router?
3.	Evaluation robustness: Can you add a small-scale human study or at least a cross-judge experiment (e.g., Claude judge vs. GPT judge) to verify gains under different evaluators?
4.	Cold-start deployment. How would RISE be used against a brand-new closed-source MLLM with no prior logs? Do you support (i) metadata-only priors (e.g., family, tokenizer/vision-tower hints), (ii) few-shot online router updates, or (iii) a fallback heuristic (e.g., diversity-maximizing surrogate subset)? Provide an explicit procedure.
5.	Offline cost quantification. Please estimate the total number of attacks used to build the experimental graph and the corresponding compute/budget (GPU-hours, API costs for closed-source queries if applicable). A per-target and per-surrogate breakdown would be ideal.
6.	Router specification for reproducibility. Please document the GNN router architecture (layer type, depth, hidden sizes, message-passing steps, readout, optimizer, training epochs, early-stopping criteria) and release config files. 
7.	Ensemble size & negative contribution (hypothesis). In your ablation, K=2 appears to outperform K=3. Could it be that, in many cases, a particular pair of surrogates contributes most of the gain while the third model sometimes has a negative marginal effect—raising the question of whether a simpler “top-2” heuristic would perform similarly? Please report (i) per-example ensemble composition histograms, (ii) leave-one-out/marginal contribution analyses (e.g., $\Delta$ ASR when adding the 3rd model), and (iii) a comparison against fixed top-2 or diversity-maximizing heuristics to quantify any incremental benefit of the router.

### Soundness
3

### Presentation
3

### Contribution
3
