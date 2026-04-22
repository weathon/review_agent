# Fast training of accurate physics-informed neural networks without gradient descent

- Avg Score: 7.00
- Decision: Accept (Oral)
- Scores: 4, 8, 8, 8

## Abstract
Solving time-dependent Partial Differential Equations (PDEs) is one of the most critical problems in computational science. While Physics-Informed Neural Networks (PINNs) offer a promising framework for approximating PDE solutions, their accuracy and training speed are limited by two core barriers: gradient-descent-based iterative optimization over complex loss landscapes and non-causal treatment of time as an extra spatial dimension. We present Frozen-PINN, a novel PINN based on the principle of space-time separation that leverages random features instead of training with gradient descent, and incorporates temporal causality by construction. On nine PDE benchmarks, including challenges like extreme advection speeds, shocks, and high-dimensionality, Frozen-PINNs achieve superior training efficiency and accuracy over state-of-the-art PINNs, often by several orders of magnitude. Our work addresses longstanding training and accuracy bottlenecks of PINNs, delivering quickly trainable, highly accurate, and inherently causal PDE solvers, a combination that prior methods could not realize. Our approach challenges the reliance of PINNs on stochastic gradient-descent-based methods and specialized hardware, leading to a paradigm shift in PINN training and providing a challenging benchmark for the community.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Frozen-PINNs, a gradient-free framework for solving time-dependent partial differential equations. The method separates spatial and temporal components of the solution: spatial bases are randomly sampled and fixed, while the temporal coefficients evolve through a least-squares initialization and an adaptive ODE solver, ensuring temporal causality. By avoiding gradient descent, the approach simplifies optimization and achieves remarkable computational speedups and accuracy across various PDE benchmarks, including advection, diffusion, Burgers’, and high-dimensional heat equations.

### Strengths
1. The paper is clearly written and well organized, with informative figures and tables that effectively convey key ideas.
2. The method delivers strong empirical performance, achieving large speedups and accuracy gains across diverse PDE benchmarks.
3. The experiments are thorough and well documented, with detailed ablations and transparent reporting that make reproduction straightforward.

### Weaknesses
1. The proposed approach essentially reformulates the PDE into an ODE and integrates it with a standard solver, while the neural network merely provides a fixed spatial basis. As a result, the method’s “learning” component is limited, and the main computational work is performed by the numerical ODE solver. This makes the conceptual novelty questionable and leaves unclear why such a neural formulation is preferable to directly applying classical ODE or PDE solvers.

2. The evaluation focuses mainly on relatively simple or smooth problems. To demonstrate robustness and generality, the method should be tested on more strongly nonlinear or coupled systems—such as the Navier–Stokes equations (e.g., lid-driven cavity flow)—where the dynamics are more complex and multi-scale behavior is prominent.

3. From the hyperparameter tables (Tables 4 and 5, etc.), the proposed model appears considerably larger than the PINN and causal PINN baselines. Moreover, the baseline settings deviate from common practice—the PINN models are quite small, trained for fewer than 10,000 iterations, and in some cases use an unusually large initial learning rate of 0.1 with Adam. While this setup shortens their reported training time, it likely underrepresents their attainable accuracy, making the comparison of efficiency and performance less conclusive.

### Questions
1. If I understand correctly, the authors use neural bases to represent the spatial domain and then integrate the resulting ODE system using a numerical ODE solver. If that is the case, why not simply apply a conventional ODE solver directly? The authors may argue that their approach improves upon standard PINNs, which I acknowledge. However, this direction seems to diverge from the central objective of PINN research in machine learning—to understand how well neural networks can approximate physical dynamics without relying on traditional numerical solvers. Such investigations are valuable because they provide insights transferable to other frameworks such as physics-informed neural ODEs, physics-informed neural operators, and physics-informed diffusion models. How do the authors justify this shift in focus? Do Frozen-PINNs have the potential to achieve higher accuracy than classical ODE solvers? What additional benefits or conceptual insights beyond accuracy does this approach provide?

2. What numerical precision is used during training—float32 or float64? Please clarify, as this can significantly affect both runtime and accuracy.

3. Is the method scalable in practice? It seems to rely on full-batch computation. How many collocation points are used for the 100-dimensional diffusion equation, and how does the cost scale with dimensionality?


4. How were the SVD truncation threshold and L2 regularization chosen? Was any ablation study conducted to assess their sensitivity and impact on performance?

### Soundness
2

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
3

### Summary
The paper proposes Frozen-PINN, a gradient-free Physics-Informed Neural Network that replaces iterative optimization with a space–time separation strategy. Spatial features are randomly sampled and fixed, while time-dependent coefficients are solved using ODE solvers, ensuring temporal causality by design. This approach greatly simplifies training and removes dependence on backpropagation. Across multiple PDE benchmarks, Frozen-PINN achieves much faster convergence and higher accuracy than existing PINNs and even matches mesh-based solvers, establishing a fast, causal, and interpretable alternative to gradient-descent-based PINN training.

### Strengths
>S1. 

This paper presents a methodology that fundamentally differs from conventional PINN training approaches. Through a space-time separation framework, the proposed method circumvents the multi-objective optimization problem inherent in PINNs and naturally incorporates time causality, thereby avoiding several fundamental challenges that have long limited PINN performance.

>S2. 

The paper is well-organized with clear motivation. The proposed approach offers a reasonable alternative framework with solid justification for why it sidesteps the challenges faced by traditional PINNs.

>S3.

Frozen PINN demonstrates substantially faster training times and higher accuracy compared to the conventional PINN approach. The experiments are conducted on benchmark problems known to be challenging for PINNs, including cases with extreme advection speeds, shocks, and high-dimensionality. Across these benchmark, Frozen PINN consistently exhibits superior performance.

### Weaknesses
> W1.

Frozen-PINN satisfies the initial condition by solving a least squares problem. While this decouples the initial condition from the PDE and boundary conditions, thereby simplifying the optimization, the least squares solution may not always be perfect. An analysis of the residual initial condition loss and its impact on model performance would be valuable. More broadly, investigating the sources of different loss components and their effects on accuracy would enhance our understanding of Frozen-PINN's behavior.

>W2.

Despite the various challenges in PINN training, increasing the number of collocation points (combined with appropriate sampling strategies and architectures) generally improves performance when sufficient training is applied. In my understanding, Frozen-PINN could also reduce PDE residual loss by investing more computation to solve the ODEs more accurately. However, since the initial condition relies on a predetermined least squares solution, it is unclear whether additional computation can improve the initial condition fitting. Is there a mechanism in Frozen-PINN to leverage increased computational resources for better initial condition satisfaction?

### Questions
>Q1. See W1.


>Q2. See W2.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Frozen-PINN, a physics-informed neural network framework that avoids gradient-descent altogether by freezing space-dependent features and evolving only time-dependent coefficients through ODE solvers, thereby enforcing temporal causality by design. It argues that two root causes of slow/inaccurate PINNs are (i) hard, multi-objective, nonconvex optimization over many parameters and (ii) treating time as just another spatial dimension, and shows that space–time separation plus loss decoupling sidestep both. Frozen-PINNs train orders of magnitude faster, while reaching accuracies comparable to or better than SOTA PINNs and even close to mesh-based solvers in low dimensions. The method also introduces an SVD layer to reduce stiffness and further accelerate solving.

### Strengths
1. The paper removes SGD/backprop from the loop by freezing spatial features and evolving only time-dependent coefficients with classical ODE solvers and least squares, which is a non-standard but very clean way to “solve” the training bottleneck of PINNs rather than tuning optimizers.

2. By constructing the solution as frozen spatial bases plus time-evolving coefficients, the method enforces the Markov/causal structure of time-dependent PDEs “by construction,” something many domain-decomposition / curriculum PINNs only approximate with scheduling. That’s an insightful modeling decision with high originality.

3. The 10D–100D heat-equation results and the comparison plot (Frozen-PINN-elm vs. PINN) show the method keeps errors small while increasing width, a case where many neural PDE solvers and all classical meshes struggle; this is important significance-wise for scientific ML.

4. Because training is reduced to classical ODE solves and least squares on frozen bases, the method does not lean on big GPUs or specialized accelerators, which broadens the audience and strengthens the paper’s practical significance.

### Weaknesses
1. Missing literatures regarding model-architecture-based temporal PINN methods. In the introduction section, it is claimed that Non-causal treatment of time as an extra spatial dimension. But there are indeed such researches like PINNsFormer[1] and PINNMamba[2] which consider and model the PDE system as a (pseudo-)sequence. Although this doesn't hurt the novelty and significance of this paper, it should be discussed.

[1] Zhao Z, Ding X, Prakash B A. Pinnsformer: A transformer-based framework for physics-informed neural networks. ICLR 2024.

[2] Xu C, Liu D, Hu Y, et al. Sub-sequential physics-informed learning with state space model. ICML 2025.

### Questions
You propose two strategies: boundary-compliant layer vs. augmented ODE. How should a practitioner decide between the two for a new PDE/domain? Is there a cost/accuracy tradeoff (e.g. augmented ODE enlarges the system and can worsen stiffness; boundary layer needs a problem-specific A)? A decision table or rule-of-thumb would make this much more usable.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes Frozen-PINN, a gradient-descent-free approach for time-dependent PDEs that enforces temporal causality by separating space and time. Hidden-layer spatial bases (weights and biases) are sampled and then frozen (via either data-agnostic ELM sampling or data-dependent SWIM sampling), while the time-dependent output coefficients are evolved by solving an ODE obtained by inserting the frozen spatial basis into the PDE. Initial coefficients are set by least squares and boundary conditions are handled via either a boundary-compliant layer or an augmented ODE. The method further compresses the neural bases with a truncated SVD layer to reduce ODE dimension and accelerate computation. Empirically, the authors evaluate Frozen-PINN on eight PDE benchmarks and show improved accuracy compared to many PINN variants and classical mesh solvers in the studied regimes. The manuscript additionally provides ablation studies (e.g., on the SVD layer and SWIM vs ELM sampling), implementation details, and a reproducibility statement.

### Strengths
1. The paper introduces a clear and intuitively appealing space–time separation idea: freezing spatial basis functions and evolving only the time-dependent output layer via ODE solvers, which enforces temporal causality by construction and directly addresses a known PINN weakness (treating time as an extra spatial dimension).

2. The manuscript presents two pragmatic sampling strategies for spatial bases: (i) data-agnostic Extreme Learning Machine (ELM) sampling and (ii) data-dependent SWIM sampling (Sample-Where-It-Matters), where SWIM uses pairs of collocation points (and optional projection onto solution gradients) to place basis functions near shocks or informative regions. This design is well motivated and demonstrated to be useful for handling shocks and directional features.

3. The authors implement an SVD compression layer to orthogonalize and truncate the neural basis, substantially reducing the dimension of the resulting ODE system and lowering computational cost; the paper supplies ablation evidence that the SVD layer can yield very large runtime improvements in some settings (e.g., tens of speedups in several ablations).

4. The experimental evaluation is broad and systematic within the chosen set of PDEs: eight benchmarks spanning advection extremes, shocks, higher-order spatial derivatives, complex geometries, and high dimensionality are considered, and the authors report means and standard deviations over multiple seeds with numerous ablations (resampling, basis widths, SVD thresholds). This breadth strengthens the empirical narrative.

5. The method removes reliance on iterative gradient-descent training for the bulk of parameters, enabling very fast CPU training in many reported cases and promising a low-resource alternative to GPU-centric PINN training; the authors emphasize carbon-efficiency and implementation simplicity as benefits.

6. The paper is reasonably reproducible in spirit: many methodological details, hyperparameter settings, and appendices are provided, and the authors pledge to release code and seeds to reproduce experiments.

### Weaknesses
1. Regarding the experimental claims of extreme speedups and accuracy (e.g., “up to 100,000× faster training” and multiple orders-of-magnitude accuracy gains), it should be carefully noted that these comparisons are across different CPU vs GPU settings (CPUs for Frozen PINNs vs GPUs for PINNs), and different solver regimes (low-precision vs high-precision). The paper does acknowledge benchmarking rules (matching accuracy regimes and noting GPU timings), but the mixed hardware/configurations needs to be specified while making broad claims of speedups and accuracy (which could be attributed to the differences in configurations too).

2. The scope of theoretical guarantees and assumptions is limited in interpretability for practitioners. While the paper references and partially justifies why random-feature style bases and ODE evolution reduce optimization complexity, it lacks clear, practically checkable conditions that explain when SWIM/ELM sampling will produce sufficiently expressive bases (e.g., how many bases are needed as a function of Kolmogorov n-width or solution regularity). The theoretical framing would benefit from tighter bounds or guidance connecting basis counts, SVD truncation thresholds, and expected approximation errors in realistic regimes.

3. The overhead and robustness of repeated resampling in high dimensions or with sparse data should be quantified and discussed more explicitly. Some ablations show SVD reduces runtime dramatically, but the cost of SWIM itself (and its scaling w.r.t. dimension and collocation count) needs clearer accounting.

4. The generalization to fully spatially complex PDEs (e.g., Navier–Stokes turbulence or domains requiring domain decomposition) is asserted but not demonstrated. Claims about matching mesh solvers in low-dimensions and scaling to high dimensions are promising but would be stronger with at least one medium-scale 2D/3D fluid example or a domain-decomposition demonstration.

5. The method’s dependence on choices of basis sampling (ELM vs SWIM), SVD cutoff, and collocation density appears substantial from the ablations, but sensitivity studies across a broader hyperparameter grid and more seeds would increase confidence in robustness and reproducibility. Some ablations show very large speedups only after aggressive SVD compression. This raises questions about stability and about how to choose compression thresholds in new problems.

### Questions
1. What is the computational cost and memory scaling of SWIM resampling and basis projection in high dimensions (d≥10)? Are there practical heuristics (e.g., maximum number of collocation points, subsampling strategies) you recommend to keep SWIM tractable?

2. Could you provide more explicit guidance (or an automated procedure) for choosing the SVD cutoff/target rank so a new practitioner can reliably obtain the claimed speedups without extensive manual tuning? Please include how SVD choice impacts stability of the ODE solver.

3. For PDEs with truly complex spatial structure (e.g., 2D Navier–Stokes), do you expect Frozen-PINN to require domain decomposition or local basis strategies? Do you have preliminary results or runtimes/cost models for such extensions?

4. It will be interesting to see efficiency comparisons with numerical methods for solving PDEs.

### Soundness
4

### Presentation
4

### Contribution
4
