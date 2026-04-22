# Navigating Cognitive Manifolds: Optimal Transport for Large Language Model Optimization

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 8, 4

## Abstract
Large language models (LLMs) possess vast knowledge but face inefficiencies in task-specific knowledge organization and activation. Existing prompt engineering relies on empirical trial-and-error, lacking principled optimization frameworks. We introduce Cognitive Geometry Optimal Transport (CGOT), a framework that reframes LLM cognitive optimization as geometric navigation in high-dimensional probability spaces. Our key insight models cognitive configurations as probability measures over knowledge states, leveraging optimal transport theory to derive principled paths from initial to target configurations. CGOT employs a dual geometric guidance system: Wasserstein distances for radial metrics and Kantorovich potential gradients for directional guidance, enabling continuous optimization on cognitive manifolds. Through systematic experiments on three prominent LLMs (Qwen3-72B, Deepseek-v3-67B, LLaMA-3-70B) across four cognition-intensive benchmarks (GSM8K, HumanEval, CommonsenseQA, BigBench-Hard), we demonstrate: (1) LLM cognitive spaces exhibit low-dimensional manifold structures (intrinsic dimension ~8.7) with strong geometry-performance correlation (Pearson $r = -0.76$, robustified to standardized $\beta = -0.82$ under hierarchical mixed-effects modeling); (2) CGOT achieves consistent 4.8\% average performance gains (Cohen's d $>$ 0.7 in structured tasks), outperforming baselines like APO, OPRO, GrIPS, and BayesOpt-Prompt by 0.6\% on average (p$<$0.05); (3) the framework generalizes across prompt strategies (Zero-shot: +5.3\%, Few-shot: +4.5\%, Chain-of-Thought: +4.6\%) and model architectures. Ablation studies confirm the critical contributions of Wasserstein metrics (-1.3\% without) and non-linear optimization (-2.2\% without). This work bridges optimal transport theory with LLM optimization, transforming prompt engineering from empirical art to geometric science with enhanced process interpretability

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel theoretical framework named Cognitive Geometric Optimal Transport (CGOT) that reformulates prompt optimization as a problem of optimal transport between probability measures. This work moves prompt engineering from empirical trial-and-error to a principled, geometry-based optimization process.

### Strengths
1. The paper provides strong empirical evidence across multiple benchmarks (GSM8K, HumanEval, CommonsenseQA, BigBench-Hard) and models (Qwen3-72B, DeepSeek-v3-67B, LLaMA-3-70B).

2. CGOT bridges the gap between empirical prompt engineering and theoretical cognitive optimization by grounding language model control in Wasserstein geometry and Kantorovich potential fields. This represents a major conceptual advance in linking cognitive science, geometry, and LLM optimization.

### Weaknesses
1.  The quality of the figures is not good. For example, the font size of figures 1-3 is too small to read.

2. Optimal transport computations are computationally expensive. The scalability of this work to real-time or large-scale applications remains uncertain.

3. The comparison is not sufficient. For example, is the proposed method comparable to the meta-heuristic method? For example,

Guo, Q., Wang, R., Guo, J., Li, B., Song, K., Tan, X., ... & Yang, Y. (2023). Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. ICLR 2024.

4. The equation number is missing in line 273.

5. The figure reference has a mistake in line 213.

### Questions
1. How sensitive is CGOT to initialization or the choice of base prompt?

2. How interpretable are the optimized prompts: do they reveal meaningful structure in how LLMs reason?

### Soundness
3

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
4

### Summary
The paper introduces Cognitive Geometry Optimal Transport (CGOT), a theoretical and algorithmic framework that reformulates prompt optimization for large language models (LLMs) as a geometric navigation problem in cognitive space. By using optimal transport theory, it computes Wasserstein distances and Kantorovich potentials  to find efficient transformation paths from a current to a target cognitive configuration.

### Strengths
1. The mathematical formalization is coherent, covering measure theory, Kantorovich duality, and manifold embeddings. The inclusion of proof sketches and algorithmic convergence guarantees adds credibility.
2. The experimental results demonstrate consistent performance gains  across models, datasets, and prompting strategies, validated by multiple metrics.
3. The paper’s core insight that treating prompt optimization as optimal transport on a cognitive manifold conceptually elegant.

### Weaknesses
1.	The method relies on Wasserstein distance computation and ICNN-based Kantorovich potentials, which are computationally expensive.Is there any way to resolve it or address this issue?

2.	Would CGOT extend to multimodal or cross-lingual LLMs where cognitive manifolds differ substantially across modalities?

3.	How to guarantee the convergence of this method?

4.	What are the time and memory overheads compared to existing prompt optimization baselines? Any trade-offs between accuracy and computational cost?

5.	The semantic diagram is not informative and it is hard to intuitively grasp the idea.

6.     Overall: while the study provides a detailed geometric formulation and interesting empirical correlations between Wasserstein distance and task performance, the proposed CGOT framework does not convincingly demonstrate conceptual or methodological novelty beyond existing manifold-based optimization or transport-theoretic approaches. The individual components like manifold representation, Wasserstein objectives, and Kantorovich potential refinement are already well-established, and the integration presented here appears incremental rather than fundamentally new. That is my main concern.

### Questions
See wqeakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper reframes prompt optimization as a geometric problem in probability space rather than a discrete search over text tokens. They model each prompt-induced hidden-state distribution as a probability measure on a low-dimensional manifold called a “cognitive space.”They propose using optimal transport to move the current prompt's hidden-state distribution toward a task-optimal one. They call their approach Cognitive Geometry Optimal Transport (CGOT). CGOT integrates two geometric components that define a continuous optimization procedure:

* Wasserstein distance to quantify how far the current cognitive configuration is from the target (radial metric).
* Kantorovich potential gradients to indicate how it should move optimally through that space (directional guidance).

Contributions:

* Uses optimal transport theory to provide a mathematically principled approach to LLM prompt optimization, 
* Develops a probabilistic manifold model of cognitive space, arguing that hidden-state activations form a consistent low-dimensional structure.
* Proposes the CGOT algorithm, which alternates between estimating Wasserstein distances, learning Kantorovich potentials, and transporting probability mass to reach optimal "cognitive configurations" with convergence guarantees.
* Uses Pearson correlation to show correlation with task performance and shows CGOT yields a modest but statistically significant performance gain over well-selected prompt optimizer baselines.

In summary, the paper recasts prompt engineering as a continuous geometric optimization problem, positioning CGOT as a bridge between cognitive science, manifold learning, and large-scale language-model control.

### Strengths
The paper introduces a framing of prompt optimization as a problem of optimal transport on cognitive manifolds that is novel in my view. The approach represents a synthesis of optimal transport theory, representation geometry, and LLM control that is theoretically ground and provides a nice contrast with more heuristic methods that dominate the space. The methodology uses established OT formalism and provides convergence guarantees. The experiments provide credible empirical validation.  The results of the experiments demonstrate incremental empirical gains, but the conceptual impact makes up for this as it provides a foundation for future research on geometry-aware prompt tuning.

### Weaknesses
Evaluating the statistical association between W_2 distance and task performance could have been much stronger without much extra effort. Specifically, it doesn't address confounding due to the model or to task difficulty. A good alternative would be using a hierarchical or mixed-effects regression controlling for model and task. I suspect use of partial pooling across model and task in a hierarchical modeling setting might even yield stronger statistical results than was presented. 

It would have been good to include LLM-as-the-optimizer techniques in the baseline comparison, as my understanding is the models do well. 

It would have been good to include more information on practical costs and scalability, runtime analysis, iteration counts, the number of LLM forward passes per optimization loop, GPU hours, empirical convergence curves, impacts of caching, etc., for assessing feasibility models of various sizes.

### Questions
* Was the correlation of –0.76 calculated by combining results from all models and tasks together? Why not account for model and task-specific variance?
* How do you get the target distribution for each task? Is it based on runs where the model performs well, and if so, does CGOT need to collect new examples for every task?
* How many LLM evaluations does CGOT usually need before it converges on the benchmarks you report? And roughly how long does that take? can you characterize the cost for optimizing one prompt on a model the size of LLaMA-3-70B?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Cognitive Geometry Optimal Transport (CGOT), modeling LLM optimization as navigation on a cognitive manifold via optimal transport. The method applies Wasserstein geometry for prompt optimization and achieves consistent but modest (~4–5%) improvements across several large models and reasoning benchmarks.

### Strengths
Sound theoretical framing using optimal transport and geometry.

Mathematically rigorous and technically detailed.

### Weaknesses
The paper is a bit hard to follow, many sections are mathematically dense and conceptually abstract.

Several figures and tables have fonts that are too small, and some elements overlap, making them hard to read.

The proposed “cognitive manifold” idea lacks clear empirical evidence.

Gains are modest relative to the method’s complexity.

### Questions
Please reply to the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
