# Momentum Steering: Activation Steering Meets Optimization

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Activation steering has emerged as a powerful approach for controlling large language models (LLMs), with prominent methods such as ActAdd, Directional Ablation, and Angular Steering relying on difference-in-means activations from contrastive prompts across layers. These differences are typically treated as candidate feature directions, later refined into optimal steering vectors or planes. In this work, we reinterpret these candidate directions as gradients of an underlying optimization problem. Building on this perspective, we propose Momentum Steering, a momentum-based framework for activation steering in LLMs. Unlike traditional difference-in-means methods, our framework generates a richer family of candidate directions through momentum updates, enabling more expressive steering. We first introduce a non-causal variant that accumulates difference-in-means signals via momentum, producing enhanced candidate directions. We then develop a causal variant, where future layer statistics are recursively influenced by previously applied momentum directions, explicitly modeling the causal effects of interventions on downstream activations. This recursive formulation yields more stable and consistent steering dynamics. Momentum Steering is lightweight and modular, making it easily compatible with state-of-the-art steering methods. We empirically demonstrate that Momentum Steering delivers consistently stronger, more robust, and more reliable behavioral control than existing approaches across diverse LLM families and benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors reframe activation steering in LLMs as an optimization problem and propose momentum-based steering that incorporates momentum from gradient-based optimization into the computation of steering vectors. Rather than using simple difference-in-means between contrastive prompts, they accumulate these differences across layers via momentum updates. The authors show that popular methods like ActAdd and Mean-AcT can be derived as (projected) gradient descent updates in their framework, then extend this framework with momentum and an Adam-based variants. Experiments on jailbreaking and toxicity mitigation tasks show improvements over baselines across multiple model families.

### Strengths
- reinterpreting activation steering as PGD is creative and provides a principled framework for understanding existing methods
- consistent improvements across diverse models (Qwen, Gemma, Llama with 3b-32b params) and tasks
- seems easy to integrate with existing steering methods without significant computational overhead
- provides stability analysis and connects to well-established optimization literature

### Weaknesses
1. Poor presentation: multiple typos (e.g., "zero k" line 233), inconsistent notation, and unclear writing throughout
  - switches between $x(k)$ and $x(t)$ without explanation; unclear distinction between $\mathbf{q}$ and $\mathbf{p}$ in sections 2.1-2.2; unexplained notation like $k=[K]$
  - see questions for more unclarity
2. no statistical significance shown: no error bars on experiments, making it impossible to assess significance given that results are expected to have significant variance
3. Questionable experimental choices: randomly initialized model uses unrealistic architecture (150 layers), why not use actual trained models for validation?

### Questions
1. Am I correct in understanding that the calculations for the steering functions (including momentum-based updates) are only when constructing the steering vectors -- when applying the steering vectors, the calculations stay the same as in ActAdd? Or are the you also changing the calculations done during inference? 
2. Why choose Bregman divergence over other divergence functions?
3. Can you provide intuition about how different choices of $h$ in the Bregman divergence lead to different feature maps and steering behaviors?
4. How does this relate to matrix-based steering methods ([Postmus & Abreu, 2025](https://arxiv.org/abs/2410.16314)) or flow-based methods ([Rodríguez et al, 2024](https://arxiv.org/html/2410.23054v1))?
5. For the randomly initialized model experiments (Appendix B):
   - Why use such unrealistic architecture (150 layers)?
   - Why not validate on actual trained models?
   - Would the trends persist from initialization to final trained models?
6. In equation 2.3, are you averaging over time? At which tokens do you extract activations and where do you apply them?
7. What explains the inconsistent notation between $x(k)$ and $x(t)$, and between $\mathbf{q}$ and $\mathbf{p}$?
8. What does $k=[K]$ mean before equations 1 and 2?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Momentum Steering, an optimization-inspired framework for activation steering in LLMs. In contrast with traditional methods, this method treats each layer independently, ignoring inter-layer dependencies and the temporal structure of representation formation. The authors formulate activation steering as a projected gradient descent (PGD) process that minimizes the Bregman divergence between activations for source and target prompts. From this optimization viewpoint, they introduce Momentum Steering, which incorporates momentum updates across layers to accumulate past feature directions

Two variants are proposed: Non-causal momentum steering, which aggregates difference-in-means statistics across layers. Causal momentum steering, which recursively updates future activations based on previous momentum directions, capturing causal dependencies.

Experimental results are reported across multiple benchmarks (e.g., ADVBENCH, RealToxicityPrompts, tinyBenchmarks) and model families (Gemma2, Qwen2.5, Llama3).

### Strengths
- Frames activation steering as a gradient-based optimization process, providing a clear theoretical foundation for a previously heuristic technique.

- The introduction of momentum accumulation across layers is conceptually interesting and well-motivated.

- The distinction between causal and non-causal formulations adds depth and flexibility, enabling adaptive control depending on task needs.

- Tests on a broad spectrum of models (3B–14B parameters) and tasks (toxicity mitigation, jailbreaks, and general benchmarks).

- Requires no retraining or fine-tuning, only modifying activation computations at inference time.

### Weaknesses
- While Appendix D mentions a stability analysis, it lacks detailed formal proofs or explicit convergence bounds for the momentum dynamics.

- Momentum accumulation, especially in causal sequential mode, increases computation during inference; efficiency metrics are not quantitatively reported.

- The link between optimization theory and steering behavior, might be an overkill for a traditionally cheap-compute task.

- The work focuses on quantitative results but lacks qualitative examples or visualization of how momentum steering alters model representations or outputs.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces Momentum Steering, a new framework for activation steering in LLMs. Activation steering refers to the manipulation of hidden activations at inference time (e.g., with methods like Activation Addition or Directional Ablation) to control model behavior without retraining.

The authors propose viewing traditional steering methods (based on difference-in-means activations between contrastive prompts) as instances of an optimization problem solvable via projected gradient descent ((P)GD), for which they provide a proof.

Based on this insight, they incorporate momentum into the process of constructing steering vectors, enabling the accumulation of information across layers. They further generalize the idea to Adam Steering, which applies Adam-style updates to steering vectors.

Empirical results across mutliple LLMs (Qwen2.5, Llama3, Gemma2) and tasks (jailbreaking task, toxicity reduction, and general language performance) show that Momentum and Adam Steering produce stronger behavioral control, while maintaining general model performance.

### Strengths
- **Comprehensive evaluation**: The authors test across a wide variety of models and multiple tasks, which supports the generality of their claims.
- **Empirically strong results**: Momentum Steering consistently improves behavioral control, while maintaing general language performance. Given the importance of LLM alignment problems, this seems like a significant result.
- **Conceptual novelty**: Reducing activation steering to a gradient descent-based optimization problem, thereby allowing for the usage of momentum mechanism, is an oroginal and interesting idea.

### Weaknesses
- **Missing citation**: Please provide a citation for the claim in line 53, as it is central to your paper: "This design can overlook valuable structure across layers, producing unstable or underpowered feature directions, especially in deeper models or tasks requiring fine-grained control."
- **Clarity of the (P)GD proof**: The proof that activation steering methods reduce to (projected) gradient descent updates should be clearer. It's a bit difficult to follow in its current state.
	- Notation for equations 1 and 2 is not consistent with equation in line 104: x(k, p) vs x(k)(p)
	- The "Sequential Refinements" section lacks clarity. It's not evident from the equations how Mean-AcT introduces feedback and the change in dataset notation (D^{(train)}) in comparison to line 105 is not explained.
	- Lines 150-152: the variable $p$ is used for prompts from different datasets, which can cause confusion. Additionally, introducing symmetric notation (e.g., x_{src} and x_{tg}) would improve readability.
	- Line 184: The introduction of the convex constraint set into the optimization in Eqn. 9 is non-trivial and deserves more intuition or justification.
	- Line 198: The hypothesis that the layer function $f(k)$ implicitly performs the projection $P_C$ is non-trivial and unproven. Since this step appears to be an important element in the reduction to activation steering, additional theoretical or empirical support would be important.
- **Experimental baseline**: The paper should more explicitly define what the baseline is in the jailbreaking experiment. Is it standard activation steering (e.g. based on Turner et al. 2023) or the (P)GD-based approach? Direct comparisons to existing and established methods that do not rely on the optimization framing would make the empirical results more interpretable (similar to what you do in section 4.3).
- **Minor issues and typos**
	- Typo in line 112 "Meant-AcT" --> "Mean-AcT"
	- Line 125: should read "uses" or "incorporates"
	- Line 363: should read "might require significantly more time"

### Questions
I discuss this primarily in the weaknesses section, but the proof that activation steering methods reduce to (projected) gradient descent updates is not yet entirely convincing. Addressing the points mentioned above would help clarify this argument, particularly the assumption that the layer function implicitly performs the projection step.

In addition, the “Empirical Evidence” section could be expanded to strengthen this claim. For example, showing that the steering vectors derived from your optimization-based reduction are sufficiently similar to those obtained from existing methods such as ActAdd (e.g., via geometrical or downstream performance comparisons) would make the theoretical correspondence much more compelling.

### Soundness
3

### Presentation
2

### Contribution
3
