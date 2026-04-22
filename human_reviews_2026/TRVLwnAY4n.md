# Leveraging Coordinate Momentum in SignSGD and Muon: Memory-Optimized Zero-Order LLM Fine-Tuning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Fine-tuning Large Language Models (LLMs) is essential for adapting pre-trained models to downstream tasks. Yet traditional first-order optimizers such as Stochastic Gradient Descent (SGD) and Adam incur prohibitive memory and computational costs that scale poorly with model size. In this paper, we investigate zero-order (ZO) optimization methods as a memory- and compute-efficient alternative, particularly in the context of parameter-efficient fine-tuning techniques like LoRA. We propose $\texttt{JAGUAR SignSGD}$, a ZO momentum-based algorithm that extends ZO SignSGD, requiring the same number of parameters as the standard ZO SGD and only $\mathcal{O}(1)$ function evaluations per iteration. To the best of our knowledge, this is the first study to establish rigorous convergence guarantees for SignSGD in the stochastic ZO case. We further propose $\texttt{JAGUAR Muon}$, a novel ZO extension of the Muon optimizer that leverages the matrix structure of model parameters, and we provide its convergence rate under arbitrary stochastic noise. Through extensive experiments on challenging LLM fine-tuning benchmarks, we demonstrate that the proposed algorithms meet or exceed the convergence quality of standard first-order methods, achieving significant memory reduction. Our theoretical and empirical results establish new ZO optimization methods as a practical and theoretically grounded approach for resource-constrained LLM adaptation. 
  Our code is available at https://anonymous.4open.science/r/zo_jaguar

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies memory-efficient zero-order (ZO) optimization for LLM fine-tuning. It introduces two methods: **JAGUAR SignSGD** (a momentumized, coordinate-difference ZO variant of SignSGD with the first convergence guarantee in the stochastic ZO setting) and **JAGUAR Muon** (a ZO version of Muon that updates matrices via a Newton–Schulz–based orthogonalized direction). Each iteration uses $O(1)$ function queries, avoiding backprop and thus saving memory. Experiments on SST-2 (FT and LoRA) and on COPA/WinoGrande (LoRA; Llama-2-7B and OPT-13B) show accuracy competitive with or better than prior ZO baselines at similar or lower memory cost.

### Strengths
* **Theory with clear novelty:** Provides nonconvex stochastic ZO convergence guarantees for momentum SignSGD and extends analysis to a ZO Muon update; explicitly models smoothing $\tau$ and oracle noise $\Delta$.
* **Practicality:** Requires only forward passes; simple to bolt onto SignSGD/Muon; compatible with both FT and PEFT (e.g., LoRA); constant query complexity per step.
* **Empirical signals:** On several LLM backbones, results match or surpass ZO baselines while keeping GPU memory low; the coordinate-difference estimator is an attractive alternative when memory is tight.

### Weaknesses
1. **Speed not addressed.** While this ZO estimator performs well in accuracy and memory, it often pays a notable price in wall-clock time. The paper appears to sidestep this trade-off and provides no discussion or measurements of time.
2. **Experiments are too sparse.** Even pioneering ZO work like MeZO paired theory with extensive empirical study. As a follow-up, this paper’s experimental breadth feels insufficient (e.g., Table 2 only reports SST-2 on OPT-1.3B and RoBERTa-Large; Table 3 has just two models and two tasks). This raises concerns about robustness and generality.
3. **Presentation could be improved.** The paper only includes tables and no figures. While figures aren’t strictly required, *convergence curves* (or equivalent trend visualizations) are essential. If figures are omitted, some other form of trend presentation should be provided.

### Questions
See weakness.

### Soundness
2

### Presentation
2

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
This paper proposes JAGUAR SignSGD and JAGUAR Muon, new zero-order (ZO) momentum algorithms addressing the memory bottleneck in large-scale LLM fine-tuning. Traditional first-order (FO) methods like SGD or Adam require excessive memory for gradient computation and storage. In contrast, this study integrates advanced momentum techniques into ZO optimization—where updates rely only on model outputs—achieving both low memory usage and stable convergence. Specifically, JAGUAR SignSGD adds coordinate-wise accumulated momentum to ZO-SignSGD, maintaining O(1) oracle calls and the same 2d+1 parameter cost while providing the first proven convergence guarantees for stochastic non-convex ZO optimization. JAGUAR Muon extends this principle to matrix-structured parameters (e.g., LoRA) through a ZO adaptation of the Muon optimizer, also with formal convergence proofs. Extensive experiments on SST-2, COPA, and WinoGrande show faster convergence, higher accuracy than previous ZO methods, and significant GPU memory savings—sometimes achieving Adam-level accuracy with several-fold less memory. Code is publicly released for reproducibility.

### Strengths
Technical novelty: Integrates momentum mechanisms into ZO optimization without increasing parameter or oracle cost. First to combine SignSGD-style updates with coordinate momentum and to extend Muon’s matrix form into a ZO setting.

Theoretical rigor: Provides full convergence analysis for stochastic non-convex ZO problems—unlike prior heuristic ZO methods—under standard Lipschitz and variance assumptions. Establishes the first theoretical guarantee for ZO-SignSGD with momentum and for ZO-Muon’s convergence rate.

Practical impact: Enables memory-efficient fine-tuning on large LLMs, achieving FO-level stability with a fraction of the memory. Coordinate-wise updates maintain the momentum benefit while keeping memory linear in d.

Empirical robustness: Validated across multiple models (OPT-1.3B, OPT-13B, LLaMA2-7B, RoBERTa-Large) and tasks (SST-2, COPA, WinoGrande), consistently outperforming prior ZO baselines (MeZO, ZO-AdaMM, ZO-SignSGD). Demonstrates clear memory reduction without sacrificing accuracy.

Reproducibility and structure: Algorithms, proofs, and assumptions clearly presented; supplementary experiments and hyperparameter details are provided. Code availability enhances transparency.

### Weaknesses
Baseline coverage: The most significant shortcoming is missing comparison with modern memory-efficient FO optimizers. Since these are the de facto baselines for low-memory training, quantitative benchmarks (accuracy, convergence rate, GPU memory usage, runtime) are required to validate true efficiency gains. Current experiments mainly compare to older ZO variants, leaving unclear whether JAGUAR methods outperform or merely match advanced FO approaches.

Novelty scope: The algorithmic innovation—adding momentum to existing ZO-SignSGD and Muon—is incremental conceptually, though non-trivial in analysis. It represents an extension and unification rather than a new optimization paradigm.

Performance ceiling: Empirical results show convergence close to FO optimizers, but not superior. The contribution thus lies in maintaining accuracy under strict memory limits, not in achieving new SOTA performance.

LoRA dependence: Experiments rely mainly on LoRA fine-tuning. The benefit in full fine-tuning remains limited; LoRA already reduces memory, so additive gains from ZO methods are smaller and not clearly separated.

Theory–practice gap: Some theoretical assumptions (bounded oracle noise Δ) are strong; their practical relevance is unclear. The resulting convergence bound (ε ≥ d√(ΔL)) may constrain attainable accuracy for large d. The paper would benefit from quantitative discussion of Δ and τ–β tuning influence.

Scalability and delay: Coordinate-wise momentum may cause stale updates in very high-dimensional settings; analysis of multi-coordinate or mini-batch variants could strengthen generality.

Clarity and completeness: Notation (‖·‖S1, δ₀, τ, etc.) introduced abruptly. Limited intuition provided for τ–Δ–β trade-off. Concrete memory and runtime figures are missing—quantified GB comparisons would substantiate claims.

### Questions
Have the authors benchmarked JAGUAR methods against AdaFactor, 8-bit Adam, or GaLore under the same settings?

How large is the runtime overhead from two forward evaluations per step, and how does total training time compare to FO optimizers?

Could multi-coordinate or low-rank perturbation estimators improve convergence speed without major memory loss?

How sensitive are results to τ and β? Any recommended default values for stable tuning?

Is full-parameter fine-tuning feasible for JAGUAR Muon, and what is the computational cost of the Newton–Schulz projection at scale?

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
2

### Summary
This paper introduces memory-efficient ZO fine-tuning methods by incorporating momentum and coordinate accumulation. Author proposed ZO-SignSGD and combine it with JAGUAR, and proposes a ZO variant of the Muon optimizer. The authors establish convergence guarantees in stochastic non-convex settings, and demonstrate through experiments that these methods achieve strong performance while reducing memory usage compared to first-order training approaches.

### Strengths
Strength

* Motivation of this paper is clear, by introducing ZO to significantly reduce memory cost of FO method.
* This paper provide good theoretical guarantee for the proposed method.

### Weaknesses
Weaknesses

* Paper writing is confusing, introduction lists many methods, but lacks explanations and clarifies the role of the proposed method within them, It lists many technical terms but lacks brief explanations, making it somewhat difficult to read.
* The experiments were limited in scale, mainly focusing on simple classification tasks on small datasets, and the variety of models tested was also limited, making it difficult to evaluate the generalizability of the proposed method. It's better to at least include generation tasks, or more challenging benchmarks like MMLU or MT-Bench.
* Memory savings are shown, but training wall-clock overhead from sampling, Momentum update, Newton–Schulz steps is not reported. It's better to show a wallclock time breakdown.
* JAGUAR Muon performs worse in full FT, paper claims it's due to non-matrix parameters, could you provide more details related to it?

### Questions
Please refer to the Weaknesses part.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces innovative zero-order (ZO) optimization methods for memory-efficient fine-tuning of Large Language Models (LLMs), addressing the critical challenge of high memory requirements in traditional backpropagation-based approaches. The paper combines Jaguar gradient estimation with Momentum Sign-SGD and Muon.

### Strengths
1. The paper proposes two extension of zeroth-order optimizers by combining the Jaguar gradient estimate wtih sign-sgd and Muno. 
2. The paper provides the convergence analysis for the proposed algorithms under mild assumptions.
The numerical results shows that the proposed method can achieve higher accuracy than the existing ZO algorithms in different LLM training tasks while using similar memory. The reported results show a statistical significance.

### Weaknesses
1. In the discussion after Lemma 3.4, in line 287, the authors claims that Guassian random perturbation results in non-convergence of ZO-Sign-SGD even with momentum. However, no further justification is provided. It is hard to tell whether this is correct or not. It is unclear why using the Jaguar peturbation is better than other perturbations.

2. It is also not clear to me why sign operation is required in Algorithm 1, since Momentum SGD should hanve better convergence property than Momentum SignSGD, and their implementation on a single machine should result in the same memory usage. SignSGD is only favorable when communication is required across multiple machines.

3. The numerical experiment results ont reports then final accuracy and overall memory consumption. However, the convergence rate w.r.t. the oracle number is also an important result. Since the paper claims that the algorithm has good oracle efficiency, we would expect the numerical results reflecting that aspect of the proposed algorithm.

### Questions
Please address the above weaknesses

Also, 
1. in line 1064, the last term does not match the last term in the following equation.
2. Directly starting from a lemma of another paper is hard to follow in App. D.2.
3. The proof steps are hard to follow and has missing steps, e.g., how line 1085 can be derived from line 1083 by using Lemma 3.4 is not quite clear. 
4. Thge summation in eq(13) should be from 0 to T-1.

### Soundness
2

### Presentation
2

### Contribution
2
