# ParetoRouter: VLSI Global Routing with Multi-Objective Optimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Global routing (GR) has been a central task in modern chip design. Many efforts, either ML-based or heuristic, have been proposed, which seek to optimize certain business goals e.g. overflow (OF) and wirelength (WL) of generated routes. Notably the recent end-to-end neural routers have shown extreme speed advantages in optimizing wirelength yet they struggle in efficiency to reduce overflow. In fact, a good trade-off between the above two metrics has not been achieved, especially the overall efficiency is pursued, as often only a single metric is optimized in existing ML-based methods. To bridge this gap for more practical industry applications, we propose a flow matching-based router for GR called ParetoRouter, to achieve trade-offs between WL and OF and generate highly connected routes in high speed and quality. In the training phase, two differential metric-oriented routing results are utilized to build the training datasets and are leveraged to design an `Average Flow' between initial pins and final routings. A Pareto sampling method based on the Das-Dennis method is also devised to realize trade-offs between OF and WL in the inference phase. Extensive experimental results show that it achieves SOTA performance on the \textbf{overflow reduction} with \textbf{less superfluous routes} across all benchmarks with \textbf{x5} times speedup over the peer SOTA ML-based method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces **ParetoRouter**, a novel machine learning-based global routing (GR) method for VLSI design that explicitly addresses the multi-objective optimization (MOO) problem of minimizing both **wirelength (WL)** and **overflow (OF)**. Unlike existing ML-based routers that typically optimize only one metric (e.g., WL-focused HubRouter or OF-focused DSBRouter), ParetoRouter uses a **Flow Matching (FM)** framework to learn a diverse distribution of routing solutions. Experiments on ISPD98 and ISPD07 benchmarks show that ParetoRouter achieves state-of-the-art OF reduction (e.g., 68% average reduction vs. DSBRouter) with competitive WL, while being ~10× faster than DSBRouter and ensuring 100% connectivity.

### Strengths
- **Efficiency**: Significant speedup (10×) over DSBRouter due to one-step sampling and FM framework.
- **High Connectivity**: Generates 100% connected routes, unlike non-end-to-end methods like HubRouter.
- **Theoretical Foundation**: Well-grounded in FM and MOO principles, with clear algorithmic design.

### Weaknesses
- **Generalization Concerns**: Evaluated on standard benchmarks, but lacks testing on diverse or real-world industrial designs to confirm robustness.
- **Complexity**: Relies on multiple components (AF loss, gradient guidance), which may complicate reproducibility and tuning.
- **Computational Cost**: Slower than some ML-based methods (e.g., VAE-HubRouter) due to gradient computations during sampling.

### Questions
1. **Practical Relevance**: How do WL and OF metrics translate to actual business outcomes (e.g., power consumption, timing closure, yield)? Are there results linking ParetoRouter’s outputs to higher-level design goals?
2. **Task Difficulty**: The paper categorizes nets by size (e.g., "large"), but does not define "difficulty" beyond scale. How does ParetoRouter perform on nets with complex congestion patterns or high pin density?
3. **Overfitting and Generalization**: The model is trained on classical solver data. How does it generalize to unseen benchmarks or modern designs with advanced nodes? Were cross-validation or holdout tests performed?
4. **Baseline Comparison**: Why not compare with more recent ML-based methods (e.g., PatLabor) or industrial tools? Are the chosen baselines sufficient to claim SOTA?
5. **Hyperparameter Sensitivity**: How sensitive is performance to the choice of weights (e.g., γ, ω) and guidance strength (ρ)? Is manual tuning required for each design?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper designs a flow-matching-based router for global routing, called ParetoRouter, to achieve trade-offs between WL and OF. ParetoRouter is trained on data set with two differential metric-oriented routing results. A Pareto sampling method based on the Das-Dennis method is used to achieve trade-offs between OF and WL in the inference phase.

### Strengths
1 ParetoRouter attempts to leverage flow match to produce end-to-end routing solutions 

2 ParetoRouter incorporates a fast Das–Dennis–based Pareto sampling scheme, which can reduce solution-generation time.

3. ParetoRouter illustrates its advantage in the experimental study

### Weaknesses
1: Some important related works are missed

2: By considering the missed related work, the novelty and contribution should be reconsidered.

### Questions
1: One major limitation is lack of the similar work on the combination in the pareto and flow matching, such like “ParetoFlow: Guided Flows in Multi-Objective Optimization ” in ICLR 2025.

2: From ParetoFlow above, we can see that most of the idea of the flow matching and the combination with Pareto Sampling have been proposed. The contribution in this papr should be reconsidered.

3. The proposed methods have different training objective from that of ParetoFlow. The experimental study is needed to illustrate the advantage.

4. It is better to discuss the impact of the relationship between two objectives. This paper mainly performs study on OF and WL. What is about other two objectives with different internal relationships?

5. From table 6, we can see that the adjustment of parameter cannot achieve the desired goal. The weight [0,1] and [1,0] are not always corresponding the best result for OL and WL separately.

6. It is better to discuss the scalability of the method with more than 2 objective functions

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
5

### Summary
The authors proposed a flow-matching-based router for GR to trade off WL and OF and achieve better results than the SOTA ML-based router. Their method also shows significant speedup compared with the SOTA ML-based router. However, the contribution of this work is mainly restricted to the ML-based global routing, considering the insights and experiments, which are explained in weakness.

### Strengths
1. The authors propose an ML-based global router jointly optimizing overflow and wirelength, which is much better than the SOTA ML-based router, DSBrouter.
2. Their Pareto sampling method provides a large speedup over DSBRouter, which is also based on a generative ML method.

### Weaknesses
1. Jointly optimizing overflow (OF) and wirelength (WL) is common sense in global routing. Considering that global routing is a classic problem in the EDA area, I don't think this insight could be an important contribution.
2. The comparison with classic methods is too weak to show the capability of their method. Their mentioned works, e.g., DGR, NTHURouter, and NCTU-GR, are all better global routers than GeoSteiner and FLUTE+ES. Meanwhile, DGR, NTHURouter, and NCTU-GR, all consider overflow and wirelength when formulating and solving the global routing problem. Meanwhile, the results about "LABYRINTH" are missing in Table 3 ("OF") and Table 4 (all metrics).
3. The survey of current works has errors. DGR (Li et al. 2024) and (Feng & Feng 2025) are not ML-based methods. DRG formulated the global routing as a continuous optimization problem and solved it using the deep learning toolkit, PyTorch, for automatic differentiation. Meanwhile, the objective in DGR includes congestion and wirelength cost, which should be classified as a multi-objective method. Therefore, the authors listed them with explicitly wrong information in Table 1.
4. The absence of a crucial ablation study. Only the results "w/o x_1^{NCTU}" are listed in Table 5. In this paper, the average flow is explained in Section 4.1 and Figure 2 in detail as an innovation, so a complete ablation study should be conducted.

### Questions
1. In Section 5.4, the author said "For the loss function, we remove the x^{NTHU}_{1} term.". However, the results "w/o x_1^{NCTU}" are listed in Table 5. This may be a typo in the paper, which should be corrected.
2. In Table 5, why are the losses without data from another router (NTHU or NCTU) missing? I wonder if the necessity of incorporating two routers exists? In Section 4.1, the author could explain more about why NTHURouter and NCTU-GR are used for generating a data sample by experimental results.
3. In Table 3, why are the results of LABYRINTH missing under the "OF" metric?
4. In Table 4, why are the results of LABYRINTH missing under all metrics?

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
This paper introduces an end-to-end machine learning-based global router **ParetoRouter** for VLSI design, targeting the joint optimization of wirelength (WL) and overflow (OF) via multi-objective Pareto efficiency. Leveraging Flow Matching (FM) and a novel "Average Flow" loss, it integrates routing results from NTHURouter (OF-oriented) and NCTU-GR (WL-oriented) to generate diverse solutions. During inference, a one-step Das–Dennis sampling method enables controllable WL-OF trade-offs. Experiments show ParetoRouter achieves state-of-the-art OF reduction across benchmarks (e.g., ISPD98, ISPD07), maintains 100% connectivity, and achieves a 10× speedup compared to diffusion-based SOTA methods like DSBRouter.

### Strengths
This paper tries to address a critical gap in EDA by optimizing conflicting WL/OF objectives simultaneously—a practical necessity for industrial VLSI design. Its novelty lies in the first end-to-end ML-based multi-objective global router and the efficient Das–Dennis sampling for Pareto-front approximation. Writing is clear and the figures and tables are easy to understand, with well-structured contributions and thorough preliminaries. Besides, full code release is promised, while the architecture/training details are comprehensive.

### Weaknesses
1. The analysis of the loss function in the ablation study is relatively insufficient.
2. The data of DSBROUTER for IBM06 in Table 3 appears weird.
3. While the supplementary materials present actual routing results for cases of varied scales, they lack (providing these would significantly enhance reader comprehension):
   - Comparative analysis of the same cases against other baselines;
   - Explanations for loop occurrences observed in the results;
   - Case studies demonstrating the model's actual congestion avoidance capability.
4. The benchmarks appear outdated, would you consider adding experimental results on more recent ISPD 2018/2019 benchmarks?

### Questions
1. Table 3 shows higher inference times than NeuralSteiner/VAE-Hubrouter. What limits parallelization of the Das–Dennis sampling?
2. Training costs (GPU-hours) are omitted. Can you provide more training details?
3. Would you consider adding experimental results on more recent ISPD 2018/2019 benchmarks?

### Soundness
3

### Presentation
4

### Contribution
3
