# SIMSHIFT: A Benchmark for Adapting Neural Surrogates to Distribution Shifts

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Neural surrogates for Partial Differential Equations (PDEs) often suffer significant performance degradation when evaluated on unseen problem configurations, such as new initial conditions or structural dimensions.
Meanwhile, Domain Adaptation (DA) techniques have been widely used in vision and language processing to generalize from limited information about unseen configurations.
In this work, we address this gap through two focused contributions.
First, we introduce SIMSHIFT, a novel benchmark dataset and evaluation suite composed of four industrial simulation tasks spanning diverse processes and physics: _hot rolling_, _sheet metal forming_, _electric motor design_ and _heatsink design_.
Second, we extend established DA methods to state-of-the-art neural surrogates and systematically evaluate them.
These approaches use parametric descriptions and ground truth simulations from multiple source configurations, together with only parametric descriptions from target configurations.
The goal is to accurately predict target simulations without access to ground truth simulation data.
Extensive experiments on SIMSHIFT highlight the challenges of out of distribution neural surrogate modeling, demonstrate the potential of DA in simulation, and reveal open problems in achieving robust neural surrogates under distribution shifts in industrially relevant scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces SIMSHIFT, a benchmark and dataset suite designed to evaluate unsupervised domain adaptation (UDA) methods for neural surrogates trained on physics-based simulations. It provides four industrially relevant simulation datasets (rolling, forming, motor, and heatsink design) and evaluates classical UDA algorithms (CMD, Deep CORAL, and DANN) across several neural surrogate architectures (PointNet, GraphSAGE, Transolver, UPT). The goal is to quantify performance degradation under parametric distribution shifts and assess whether domain adaptation can recover generalization in unseen configurations.

### Strengths
- While neural surrogates themselves are not new, the authors are arguably the first to systematically benchmark domain adaptation methods in this context, bridging simulation modeling and UDA research.
- The proposed datasets are realistic, diverse, and physically grounded, covering a range of FEM and CFD problems from different industries.
- The benchmark is well-documented, publicly released, and accompanied by detailed simulation parameters, architectures, and hyperparameter sweeps.
- The study runs a large number of controlled experiments with multiple architectures, adaptation methods, and model-selection strategies, highlighting consistent trends and challenges in out-of-distribution generalization.

### Weaknesses
- The benchmark primarily relies on pre-2019 UDA methods (CMD, Deep CORAL, DANN). More modern approaches, e.g., adversarial information bottleneck, self-supervised alignment, or diffusion-based DA, could offer stronger baselines and better contextualize the results.
- All benchmarked shifts are parametric (covariate) shifts, while the introduction also mentions “instrument shifts” such as mesh geometry or solver discretization changes, which are not actually tested. Including such structural or concept-level shifts would significantly increase the benchmark’s value.
- The current metrics focus on RMSE or custom geometric error terms. The benchmark does not assess physical validity (e.g., conservation of energy, boundary-condition satisfaction, equilibrium consistency) of adapted surrogate predictions, which is critical in engineering contexts.
- The work is primarily a benchmark contribution, not a conceptual advance in neural surrogate modeling or domain adaptation algorithms.
- The distinction between “neural surrogates” and generic regression surrogates (MLPs) seems somewhat artificial; many of these architectures are used interchangeably in most industrial ML contexts, whether simulation-based or data-driven. The framing could overstate novelty relative to standard supervised regression with UDA.

### Questions
- Could the authors benchmark recent UDA or domain generalization methods, such as SWAD, Tent, AdaBN, or contrastive/self-supervised approaches, to contextualize performance trends?
- Can non-parametric shifts (e.g., mesh geometry changes, discretization refinements, or modified boundary conditions) be introduced to better reflect the “instrument shift” mentioned in the motivation?
- Would it be possible to integrate physics-based evaluation criteria, such as PDE residuals or constraint violation penalties, into the assessment?
- Are all datasets equally sensitive to domain gaps? A cross-dataset or per-task ablation might reveal which physics or modeling types pose the greatest adaptation challenges.
- Finally, could the authors clarify the extent to which the surrogates incorporate physics priors (operator learning, inductive biases) versus being purely data-driven regressors?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper provides a benchmark dataset with 4 industry relevant simulations with the goal of solving the unsupervised domain adaptation (UDA) problem for neural surrogates. They provide evaluate error across multiple baseline models, UDA methods, and unsupevised model selction methods.

### Strengths
- The paper provides code and data for industry relevant applications.
- Multiple models, UDA methods, and unsupervised model selection methods are tested.
- The supplied data and experimental details are thoroughly explained and provide a nice test bed for evaluating steady state problems with UDA for other researchers.

### Weaknesses
- It would be interesting to see the author's hypotheses on why certain UDA mehods and unsupervised model selection methods worked well in specific datasets.
- Steady state only problems limit the scope of this benchmark, adding time dependent problems would also be useful.

### Questions
- For the unsupervised domain adaptation methods, which distance metric $d$ do you use?
- In Figure 1 of the paper the caption refers to two loss terms $\mathcal{L}_{recon}$ and $\mathcal{L}_{DA}$ but these terms are not defined in the main text as far as I could tell.
- There are a few typos in the paper which should be corrected:
  - Line 105: "Numerous different Numerous benchmark datasets and evaluation protocols have been established" "Numerous" appears twice.
  - Line "Detailed and descriptions of the parameter sampling ranges can be found in Appendix F."

### Soundness
3

### Presentation
2

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
The paper introduces SIMSHIFT, a benchmark dataset and evaluation suite covering four industrial simulation tasks to study UDA for neural surrogates on unstructured meshes. The framework explicitly conditions neural operators on configuration parameters via a sinusoidal (sin–cos) encoding followed by a shallow MLP, producing an 8-dimensional latent vector used for conditioning; baselines include PointNet, GraphSAGE (FiLM), Transolver (DiT), and UPT selected per task/scale. The study benchmarks three UDA algorithms (Deep CORAL, CMD, DANN) together with unsupervised model selection and reports metrics such as (N)RMSE and custom engineering metrics. All datasets are steady-state by design, and the heatsink meshes are subsampled to one quarter, with experiments showing that UDA+ unsupervised selection usually reduces target error but no single method dominates.

### Strengths
1. All architectures share the same conditioning network (sin–cos + shallow MLP, 8-dim latent), making the parametric setup consistent across models. 

2. PointNet (global pooling), GraphSAGE with FiLM (local message passing), Transolver with DiT (attention with learned slicing), and UPT (latent field modeling for very large meshes) are adapted to conditional mesh regression and chosen to match scale constraints.

### Weaknesses
1. The work extends established DA methods to neural surrogates and benchmarks them, it does not introduce a new neural operator or a new UDA/model-selection algorithm (conditioning, FiLM/DiT choices are standardized, not novel). 

2. While the paper reviews neural operator literature, the baseline lineup focuses on PointNet/ GNN/ Transformer/ UPT variants; adding FNO/Geo-FNO/GKN comparisons would better situate results, especially in 3D-large-mesh regimes.

3. The datasets are steady-state only, which may limit claims about transient dynamics and extreme-scale fidelity.

### Questions
1. To sharpen the model-side contribution, could you add and ablate conditioning variants (e.g., learned frequency embeddings, FiLM/DiT) and report stability/accuracy sensitivity?

2. Can you provide ablations that incorporate physics-informed constraints (residuals/conservation/BCs) into the loss within the same training–selection pipeline to assess reliability for industrial deployment?

3. Will you include operator baselines (e.g., FNO/Geo-FNO/GKN/GNO)—ideally on the 3D heatsink—to clarify accuracy–efficiency trade-offs against widely used PDE surrogates?

### Soundness
2

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
4

### Summary
This paper tackles the problem of unsupervised domain adaptation (UDA) in the scientific machine learning context, for modeling PDE-governed processes. Specifically, the models evaluated in the paper aim to perform UDA under the covariate-shift assumption and employing the Radon-Nikodym derivative for learning distributional transfer from a source distribution to a target distribution where ground-truth solutions are only present in the source distribution. To demonstrate UDA capability, the scientific processes the paper focuses on are self-generated datasets of steady-state processes with applications in metallurgy, manufacturing, machinery and electronics. To evaluate the UDA capability of the proposed methods, the source and target domains are created by splitting the input parameter configurations in a disjoint manner across sorce and target domains. Further, the paper also details performance of state of the art neural surrogates and unsupervised domain adaptation (UDA) techniques on the proposed dataset. Overall, the dataset and the proposed model benchmark evaluations are a good contribution to push research on the critical problem of UDA in the scientific machine learning context.

### Strengths
1. The investigated context of unsupervised domain adaptation (UDA) is a critical requirement in the scientific machine learning context because, for a neural surrogate to be useful, it needs to be able to "extrapolate" to unseen parameter configurations. This important problem has (unfortunately) not been investigated in depth as yet by the scientific machine learning community. Hence, this paper is a good start towards proposing a benchmark dataset as well as quantitative evaluations of state-of-the-art surrogates,  model selection techniques to perform the UDA task. 

2. The four problems investigated by the paper are (i) diverse (ii) each challenging in their own right (iii) well-motivated for the UDA problem as they each require costly computational simulation for data generation.

### Weaknesses
1. The paper requires better organization as there are multiple critical sections where the appendix is referred to, interrupting the flow of the paper and preventing the full understanding of the problem. One such point is the lack of clear demarcation of source and target domains for the four datasets in the main paper (this is done in the appendix in Appendices F1 - F4 where actual parameter ranges are discussed and Appendix C:Table 11 where actual source and target distribution splits are presented).  

2. Full cost of the evaluated UDA methods (i.e., surrogate training with UDA + model selection) is unclear from the main paper as presented. For a holistic presentation as a benchmark, this is necessary. Understanding how the training cost scales with (i) number of data points (ii) mesh size, is crucial for independent adaptation of the proposed methods by interested readers to other UDA contexts.

### Questions
1. How are the specific parameter ranges for `source configurations` and `target configurations` arrived at for each of the four datasets? Also, how are the "easy", "medium" and "hard" configurations decided? A more pronounced and cohesive description in the main paper of the actual configurations (i.e., parameter ranges), and the classification into "easy", "medium" and hard" would help to clarify the problem design.

2. Are there examples in any of the datasets investigated, where a significant distributional shift occurs in the underlying dynamics? Meaning, even if P_s(Y|X) = P_t(Y'|X'), are there any examples where P_t(Y') is itself out-of-distribution relative to P_s(Y)? Reporting model performance (or identifying any failures) in such cases would also be insightful.

3. Could you please provide numbers for (or pointer to a part of the paper that discusses) total training time of the proposed methods (i.e., surrogate training with UDA, model selection). Specifically, it would be useful to understand how total training time scales with training data size as well as the mesh size (i.e., number of mesh nodes)?

### Soundness
3

### Presentation
2

### Contribution
3
