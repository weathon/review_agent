# MNO: Multiscale Neural Operator for Computational Fluid Dynamics with 3D Point Cloud Data

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Neural operators have emerged as a powerful data-driven paradigm for solving Partial Differential Equations (PDEs), offering orders-of-magnitude acceleration over traditional solvers. However, existing approaches still suffer from limited accuracy and scalability, particularly on irregular domains where fluid flows exhibit rich multiscale structures. In this work, we introduce the Multiscale Neural Operator (MNO), a new architecture for Computational Fluid Dynamics (CFD) on three-dimensional (3D) unstructured point clouds. MNO explicitly decomposes information across three scales: a global dimension-shrinkage attention module for long-range dependencies, a local graph attention module for neighborhood-level interactions, and a micro point-wise attention module for fine-grained details. This design preserves multiscale inductive biases while remaining computationally efficient. We evaluate MNO on four diverse benchmarks, covering both steady-state and unsteady flow scenarios with up to 300K points. Across all tasks, MNO consistently outperforms state-of-the-art baselines, reducing prediction errors by 5% to 40% and demonstrating improved robustness in challenging 3D CFD problems. Our results highlight the importance of explicit multiscale design for neural operators and establish MNO as a scalable framework for learning complex fluid dynamics on irregular domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents MNO (Multiscale Neural Operator), a neural-operator architecture for **3D CFD on unstructured point clouds**. MNO explicitly separates global, local, and micro scales: a global dimension-shrinkage attention captures long-range flow structure, a k-NN local graph attention models neighborhood interactions, and a micro point-wise attention restores fine details—yielding multiscale inductive bias with practical efficiency. An encoder–MNO–decoder pipeline maps ((x,y,z))+attributes to target fields, and results across steady and unsteady 3D benchmarks (up to ~300K points) show 5–40% error reductions vs. recent baselines, indicating improved robustness on challenging irregular domains.

### Strengths
1. **Targets multiscale physics with three tailored attentions.**
   The architecture explicitly separates **Global**, **Local (k-NN graph)**, and **Micro (point-wise)** attention, matching long-range flow structures, neighborhood interactions, and fine-grained corrections. This design is clearly presented in the Method section and diagram (Encoder → MNO blocks → Decoder) and is supported by ablations showing complementary gains when combining scales.  

2. **Insightful multiscale visualizations.**
   The paper visualizes how each scale reduces error on different regions (e.g., global captures low-frequency/background areas; adding local improves windward regions; micro further refines transitions), which aids interpretability of the multiscale inductive bias. 

3. **Challenging 3D, unstructured CFD benchmarks at scale.**
   Experiments span multiple **large 3D point-cloud datasets** (e.g., ShapeNet Car ~32k pts; Ahmed Body ~100k surface pts; DrivAerNet++ ~300k pts subset; Parachute unsteady), demonstrating robustness on irregular domains with substantial sample sizes.

### Weaknesses
1. **Fairness and efficiency not established.**
   The paper does not report training budgets or resource usage (e.g., wall-clock time/epoch, total steps, **parameter counts**, **FLOPs**, **GPU memory**, batch sizes). Without matched budgets or a cost-vs-accuracy tradeoff, it’s unclear whether gains stem from architecture or simply higher compute/capacity. 

2. **Baselines are relatively weak and benchmarks are mostly simulation-centric/simple.**
   While the datasets are large, many are still controlled CFD simulations. The study would be more convincing with **stronger recent baselines** designed for unstructured 3D flows/point clouds (e.g., modern graph/transformer neural operators, multiscale mesh/point methods), plus harder scenarios (e.g., sharper separations, turbulent wakes, cross-Re generalization, varying inflow/boundary conditions). 

3. **Incremental design with limited positioning against closely related work.**
   The three-level (global/local/micro) attention is conceptually close to prior **multiscale operator** designs (global modes + local graph interactions + pointwise refinement) [1-4]. The paper does not compare directly to strong multiscale/ hierarchical operators (graph/mesh pooling, spectral+local hybrids, U-Net-style multi-resolution) nor analyze when each level is indispensable beyond simple ablations. 

[1] Li, Zongyi, et al. "Multipole graph neural operator for parametric partial differential equations." Advances in Neural Information Processing Systems 33 (2020): 6755-6766.

[2] Zeng, Bocheng, et al. "PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems." The Thirteenth International Conference on Learning Representations.

[3] Li, Zhihao, et al. "Harnessing scale and physics: A multi-graph neural operator framework for pdes on arbitrary geometries." Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2025.

[4] Bryutkin, Andrey, et al. "HAMLET: graph transformer neural operator for partial differential equations." Proceedings of the 41st International Conference on Machine Learning. 2024.

### Questions
1. **Fairness & efficiency reporting.**

   * Please provide a **cost/efficiency table** with: parameter counts, FLOPs (per forward), GPU memory usage, wall-clock time/epoch, total training steps, batch sizes, and hardware.
   * Were **training schedules** (optimizer, LR schedule, early stopping) and **data augmentations** matched across baselines?
   * What is MNO’s **inference latency** vs. baselines at typical point counts (e.g., 30k / 100k / 300k)?
   * Can you include a **capacity-controlled comparison** (fixing params/FLOPs) and a **scaling study** (performance vs. params) to isolate architectural gains?

2. **Baselines & benchmarks.**

   * Could you add **stronger recent baselines** tailored to unstructured 3D CFD/point clouds (modern graph/transformer operators, multiscale mesh/point methods) and report results under the same budget?
   * Beyond current simulations, can you include **harder scenarios** (e.g., sharper separations/turbulent wakes, cross-Reynolds generalization, varying inflow/boundary conditions, volumetric fields) to better stress multiscale robustness?
   * Do you evaluate **geometry generalization** (train on some shapes, test on unseen shapes) and **point-density robustness** (subsampling/perturbations)?

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
5

### Summary
The authors propose an explicit multi-scale approach aiming to handle irregular large-scale computational domains. While quantitative validation is provided and the figures are ''acceptable'', the work is largely preliminary in nature, resembling a laboratory report in quality. As such, the manuscript does not meet the standard required for acceptance.

### Strengths
This work's strengths are primarily foundational: it offers a quantitatively tested idea and provides visualizations that have some merit for clarity, forming a baseline that could be built upon with more significant innovation and rigor.  See the following **Weaknesses** of the work's originality, quality, clarity, and significance for details.

### Weaknesses
1. Originality: Lack of Novelty

Specific Comments: The proposed explicit multi-scale method is a straightforward combination of existing techniques. It fails to introduce a new theoretical concept, algorithmic advancement, or unique mechanistic insight. Put simply, after reading the paper, I was left with no new insights or a sense of enlightenment. As such, the originality of this work received a low rating.

2. Quality: Technically Superficial

Specific Comments: The technical depth of the work is insufficient, resembling a preliminary laboratory report.

- Overly Simplified Experiments: The benchmark cases used are likely too idealized or simple to robustly demonstrate the method's effectiveness and advantage in handling "irregular large-scale computational domains."

- Unfair Comparisons: The study lacks comprehensive and fair comparisons against established baseline or mainstream methods. Moreover, the final promotions on each benchmarks are not sufficiently compelling.

- Insufficient Validation: While quantitative experiments are presented, they seem limited to simple scenarios. There is a lack of systematic validation of core performance metrics such as computational accuracy, efficiency, and convergence. 

3. Clarity: Generally Understandable but with Flaws

Specific Comments: The overall narrative and figures are somewhat understandable and convey the basic workflow.

- Language Issues: The text contains some grammatical errors, non-idiomatic expressions, and occasional misuse of terminology, necessitating comprehensive English polishing.

- Logic and Structure: The logical flow connecting the methodology, experimental setup, and results analysis could be tighter and more compelling. While the figures are decipherable, their labeling, resolution, or layout lacks professionalism, hindering the effective communication of information.

4. Significance: Limited Contribution

Specific Comments: Due to the aforementioned issues with originality and quality, the academic value and practical impact of this research are severely limited.

The work fails to demonstrate that the proposed method offers a substantial advancement in addressing core challenges within the field. Also, the method's importance or value is not convincing to the broader research community or engineering practice.

### Questions
1. Unsubstantiated Methodology: While the manuscript's objective, explicitly handling three scales for improved results, is clearly stated, the proposed architectural modules lack foundational justification. The second module, as the core component, offers a passable contribution. However, the other two appear conspicuously tacked-on, seemingly to justify the "multi-scale" premise. This is corroborated by the ablation studies: the first module's attempt at a novel tokenization method (distinct from Transolver) demonstrably fails, as evidenced by the poor performance in Table 3. Furthermore, the third module is entirely arbitrary and lacks any rationale.

2. Compromised Reproducibility and Questionable Validity: The experimental section suffers from severely deficient reproducibility, omitting critical configuration details and baseline comparisons. The claim of computational efficiency is unsubstantiated, as no relevant profiling data or runtime graphs are provided. Moreover, the validity of key results is highly suspect:

- The inclusion of Cases 7-9 in the main table (Table 3) is unjustified, as they contribute no meaningful insight.

- The presented outcomes are counter-intuitive: while the individual global and micro components perform poorly, their combination (global+micro) and global+local yield surprisingly high performance. This anomalous result contradicts established experimental intuition and warrants a thorough explanation.

- A critical red flag emerges in Appendix Table 5, where the model with 4 MNO blocks reportedly achieves peak performance. For a Transformer-based method, such a result under conditions of large data and parameter counts is highly unusual and suggests potential issues with the training paradigm or evaluation.

Overall, the manuscript needs major revisions in presentation (figures, equations, language) and, more fundamentally, the technical depth is insufficient. In its current state, this work does not meet the bar for this venue.

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
The authors proposed MNO, a new multiscale neural operator for 3D point cloud data. Motivated by the complexity of physical systems often observed on arbitrary geometries, the authors propose to introduce mechanisms that allow to incorporate multi-scale features that are often observed in CFD. The contribution is thus the explicit incorporation of the feature scales via a global attention module, a local graph attention module and a micro point-wise attention module. The outputs are fused within each MNO block. 

The model is evaluated on four 3D CFD benchmarks (both steady-state and unsteady), with point clouds ranging from 15k to 300k ponts. MNO consistently outperforms existing methods.

### Strengths
The paper is well-motivated by the multiscale nature of fluid dynamics. The proposed Global-Local-Micro decomposition is intuitive, and the architectural design directly reflects this physical intuition.

Strong empirical results and SOTA performance are the paper's primary strength. It achieves notable improvements over existing baselines on all 3D benchmarks. Important ablations also provided, clearly highlighting the benefits of the incorporation of each module. This helps to  motivate the architectural choices made.

### Weaknesses
Computational cost analysis: the proposed MNO architecture runs three parallel attention branches, in particular the local module which involves a k-NN graph construction and a graph attention. The architecture seems to be in principle, computationally more expensive than the considered baselines. The paper provides no information about parameter counts, training wall-clock time, inference speed and memory usage.

There is not a particular description of how the baselines have been trained and what are the hyper-parameters used, except "experiments are conducted on the same protocol" which is vague. To ensure fairness, it would be beneficial to add it in the appendix.

The cost of constructing a graph for 300k points is non trivial and is not discussed in the text. A lot of works investigate the efficiency of neural operators methods, notably on datasets such as DrivAerNet++.

I think that the presentation could be polished, in particular the results section of the tables are a bit light, compared to the description of all the baselines which take a lot of space. It should be more compact and the analysis of the results more detailed.

### Questions
Could you please provide a detailed comparison of computational cost? Specifically, a table comparing MNO to studied baselines on: total parameter count, training wall-clock time (per epoch), inference wall-clock time (per sample), peak GPU memory usage.

Could you provide an ablation study on the choice of M (global modes) and k (k-NN neighbors)? This would help understand the sensitivity of the model and provide better guidance for applying MNO to new problems.

Did you have to make choices on the number of neighbors to consider, depending on the dataset, to ensure that there is no out of memory issues?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors propose a new novel method to model fluid flows by decomposing the system into a multi-scale setting, wherein they model global attention, local neighborhood level attention and point-wise attention for fine grained details. They show that this setup outperforms state of the art neural operator models on benchmark datasets

### Strengths
1. Modeling highly dynamic fluid systems is an open research problem and many well known neural operator models struggle. 
2. Multi-scale decompostion is a technique that has had some success in modeling it. 
3. The proposed architecture is therefore intuitive and well suited for these styles of problems. 
4. Experiments are appropriate and the motivation is clearly defined. 
5. The authors have performed thorough ablations

### Weaknesses
1. (Minor) There isn't a theoretical justification for how the setup functions in regards to the kernel integral formulations typically used within the neural operator literature. For example, radius aggregation is preferred over KNN aggregation as radius+mean aggregation ensures discretization convergence (see. GNO, GINO). However, experimental results provide sufficient empirical evidence. 
2. (Minor) Highlighting specific regions of highly dynamic flows would strengthen the claims. Figure 3, for instance, can be zoomed in to show improvements over other models. 
3. (Minor) I don't see ablations on physical properties, and points of failure. It is important to highlight where the model fails.

### Questions
How robust is this framework in handling AMR problems (Adaptive mesh Resolution)

### Soundness
3

### Presentation
4

### Contribution
3
