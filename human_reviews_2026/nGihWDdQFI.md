# CAED-Agent: an Agentic Framework to Automate Simulation-Based Experimental Design

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Configuring physics-based simulations requires balancing granularity against computational budget, a dilemma we term **C**ost-**A**ware **S**imulation-Based **C**onfiguration **O**ptimization (CASCO). Traditional methods, such as Bayesian optimization or manual expert design, often struggle with the curse of high dimensionality or fail to generalize. Large Language Models (LLMs) offer promise for automating such workflows but, as we show experimentally, lack inherent cost awareness and frequently propose inefficient configurations. While inference-time scaling can improve the exploration width to find cost-efficient configurations, it demands prohibitively many simulator queries. We propose **C**ost-**A**ware **S**imulation **C**onfiguration **O**ptimization **Agent** (CASCO-Agent), an agentic framework guiding inference-time scaling via lightweight surrogates that predict low-dimensional metrics (accuracy, cost) rather than complete physics fields. This enables easier training and flexible adaptation to data availability, e.g., using Gaussian Processes in data-scarce regimes or Neural Networks when data is abundant. In experiments across 3 typical PDE solvers (elliptic, parabolic, and hyperbolic), CASCO-Agent consistently outperforms Bayesian optimization and LLM-based baselines, achieving success rates comparable to inference-time scaling with a ground truth simulator without incurring expensive simulation overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CAED-Agent, an agentic framework that aims to improve the cost-efficiency of simulation-based experimental design. The key idea is to pair a large language model’s inference-time reasoning with a lightweight neural network that predicts cost and utility signals, so that the LLM can make informed, cost-aware decisions without excessive simulator calls. The authors test the approach on three physics simulation environments (1D heat conduction, 1D Euler, and 2D Navier–Stokes) and show that CAED-Agent achieves better success rates and efficiency than Bayesian optimization and other LLM-based optimizers.

### Strengths
- The paper tackles a well-motivated and practically important problem: how to make LLM-driven scientific agents more cost-efficient and aware of computational budgets. It clearly identifies two major pain points of current approaches: lack of cost awareness and inference-scaling inefficiency.
- The problem formulation is clean, and the proposed surrogate-signal design is simple yet effective. The experimental setup, while small-scale, convincingly shows that cost-aware signals can meaningfully improve sample efficiency.
- Overall, the work provides a reasonable step toward more practical LLM-based experiment design agents, and the results are consistent with the claimed motivation.

### Weaknesses
- The assumption that “small fully connected neural networks can learn the cost and utility functions well” feels overly strong and only holds because the experiments are limited to very low-dimensional (1D–2D) settings. It’s unclear whether the same idea would scale to higher-dimensional or more realistic simulation problems.
- The design choice of adding a small neural network to guide the LLM is not fully justified. If the cost and utility mappings are simple enough for a tiny NN to learn, one might question why the LLM itself cannot capture such patterns through proper prompting or fine-tuning. The rationale for separating the learning responsibilities between the LLM and the NN needs clearer theoretical or empirical backing.
- The related-work discussion could be deeper. There is rich literature in BO that can leverage prior knowledge from related domains. A simple Google search will return many papers. E.g., Pre-trained Gaussian processes for Bayesian optimization (Wang et al. 2024).
- The clarity of presentation could also improve: a few typos (e.g., “benifit” in line 091), small fonts and dense descriptions in Figure 2 make it hard to follow.
- Finally, the evaluation is confined to toy problems; no evidence is provided that CAED-Agent can handle realistic scientific simulations where costs and outcomes are high-dimensional or noisy.

### Questions
- It would help to clarify the exact claim that “Bayesian optimization cannot generalize across problem variations.” BO’s limitations usually stem from surrogate transfer, not from the BO framework itself.
- The cost function in Eq. (2) seems to depend on y (simulation outputs), but intuitively cost should depend only on x and θ; please clarify this dependence.
- Future work could explore whether the cost-aware feedback can be incorporated directly into the LLM through reinforcement-style prompting or few-shot demonstrations, possibly removing the need for a separate NN surrogate.

### Soundness
2

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
3

### Summary
This paper presents a method for automating parameter tuning in computational simulations that must balance accuracy and computational cost. It introduces the concept of cost-aware experimental design (CAED) and proposes CAED-Agent, which combines a large language model with a lightweight neural surrogate trained to predict simulation utility and cost. The surrogate provides feedback signals that the LLM uses in context to iteratively generate improved design parameters without retraining.

The framework is evaluated on three physics-based simulations, including Heat 1D, Euler 1D, and Navier–Stokes 2D, and compared with Bayesian Optimization and Optimization by Prompting (OPRO). The results show that CAED-Agent generally achieves better cost–utility trade-offs and more stable optimization behavior than these baselines, particularly in multi-turn settings, while requiring fewer full simulator evaluations.

### Strengths
1. The formulation of the cost-aware experimental design problem is novel and meaningful, addressing a problem that has been underexplored.
2. The use of a smaller surrogate model to provide continuous cost–utility feedback to a large language model is a well-motivated idea, offering an efficient way to utilize prior knowledge or existing data when available.
3. The writing is generally clear and easy to follow.

### Weaknesses
1. The accuracy of the surrogate model is important, as shown in the ablation study. This method requires specific data for each experimental task, which is somewhat unrealistic. In the experiments, more than 4,000 samples were used to train such a model (in 2 out of the 3 tasks), making it unsuitable for few-shot scenarios where experiments are expensive.

2. The statements “outperforms both ... by significant margins” and “Through experiments on three physics simulator environments, each with varying environmental settings and precision requirements, we demonstrated that CAED-Agent consistently outperforms both classical Bayesian optimization baselines and state-of-the-art LLM-based optimizers” are vague. The paper should clarify the metrics on which these claims are based. For example, in terms of pass rate, CAED-Agent does not consistently outperform baseline methods.

3. Some experimental details are missing, such as the hyperparameters of the LLMs and the settings of baseline methods.

4. The experiments are somewhat narrow in scope. All evaluations are conducted on relatively low-dimensional 1D and 2D PDE simulations with a small number of design variables (e.g., grid size, CFL number).

### Questions
1. Please refer to the weaknesses.
2. How beneficial is it to use a neural network to model the prior knowledge? Would incorporating simple statistics derived from the existing samples (e.g., the possible ranges of the design variables) achieve a similar effect?

### Soundness
3

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
4

### Summary
The paper introduces CAED-Agent, an agentic framework for simulation-based experimental
design that leverages large language models (LLMs) to optimize simulator configurations
under cost constraints. The central idea is to learn a surrogate model that predicts the
utility (simulation accuracy) and cost (runtime or compute) of running a simulator with
given hyperparameters such as grid size, time step, or solver tolerance. An LLM-based
agent then iteratively proposes new configurations, queries the surrogate for predicted
cost and utility, and updates its prompt to improve the trade-off between fidelity and
efficiency. The authors define single-turn and multi-turn variants of the optimization
process and evaluate CAED-Agent on three physics-based simulators: Heat1D, Euler1D, and
a 2D Navier–Stokes system. They compare against direct LLM prompting, Bayesian
optimization (BO), and the OPRO framework, showing faster convergence and higher reward
in several cases.

### Strengths
- The paper explores an emerging and relevant topic: integrating LLMs with surrogate
  modeling for agentic optimization in scientific computing. The combination of cost
  modeling, utility shaping, and multi-turn prompting is original and technically
  interesting.
- The distinction between single-shot design (analogous to standard hyperparameter
  tuning) and multi-round design (analogous to iterative experimental design) is
  conceptually helpful.
- Although limited in scale, the PDE-based testbeds (Heat1D, Euler1D, NS-2D) provide
  reproducible and interpretable environments, which is a strength compared to purely
  synthetic tasks often used in LLM-agent papers.
- Evaluation: The inclusion of Bayesian optimization and OPRO as comparison methods is
  appropriate and helps contextualize the agentic behavior.
- Readable and modular implementation idea: The general pipeline (LLM agent + surrogate
  model + simulator) is easy to grasp and could, in principle, be extended to other
  domains (e.g., actual parameter sweeps, i.e., inference).

### Weaknesses
### Problem framing and conceptual clarity

- The problem formulation is ambiguous. The paper presents itself as simulation-based
  experimental design, yet the optimization targets only simulator settings (grid size,
  step size, etc.), not experimental variables or physical parameters. This mismatch
  makes the title and abstract somewhat misleading.
- The introduction jumps directly into the LLM setup without clearly defining what is
  being optimized or why. A reader unfamiliar with the specific simulators will struggle
  to understand the underlying task.
- The notion of “inference-time scaling” is highlighted but never defined. This should
  be introduced more explicitly.
- The downstream purpose of selecting these simulator settings remains unclear. In
  realistic scenarios, one would care about physical inference or control — not just
  choosing a grid size. The authors should motivate how this contributes to actual
  scientific or decision-making goals.

### Relation to prior work and novelty claims

- Overstated novelty in cost modeling. The claim that this is the first approach to
  estimate simulation cost is inaccurate:
- Bharti et al. (2024) explicitly model simulator cost within cost-aware SBI, optimizing
  sampling to minimize total compute.
- Gorecki et al. (2023) address Bayesian decision making via amortized networks, which
  inherently learn expected losses that include cost.
- The paper should clearly position itself relative to these works, correcting the novelty claim.
  - Although Bharti et al. are cited, the connection to SBI remains opaque. Is the
    “utility” function intended to represent inference performance, or is inference
    absent entirely?
  - The decision-making perspective of Gorecki et al. is highly relevant and missing.
    Both approaches learn to choose actions under uncertainty given simulator costs;
    this work could be framed as a heuristic, LLM-driven version of amortized
    decision-making.

### Methodological and conceptual gaps

- The use of an “LLM agent” is poorly formalized. There is no clear notion of an
  optimization objective or theoretical grounding (e.g., expected-utility maximization,
  policy improvement). The surrogate–LLM loop seems heuristic, and no stability or
  convergence analysis is provided.
- The argument that existing experimental design benchmarks cannot be used “because they
  lack cost labels” is weak — costs such as runtime or FLOPs are measurable for any
  simulator. This choice limits comparability to established approaches.
- The tasks are self-contained and synthetic. They demonstrate an internal cost–utility
  trade-off but have no demonstrated downstream relevance or integration with real
  inference workflows.

### Experimental complexity and scalability

- All three PDE benchmarks (1D heat, 1D Euler, 2D Navier–Stokes) are low-dimensional and
  deterministic, with smooth cost–fidelity relations. They are suitable as sanity checks
  but not as evidence of scalability or robustness.
- The paper does not explore increasing complexity (e.g., 3D, chaotic, or stochastic
  regimes). Without such tests, it remains unclear how the method behaves when the
  trade-off surface becomes non-monotonic or discontinuous.
- In more challenging setups, LLM-based optimization is likely to hallucinate or become
  unstable, as it lacks calibrated uncertainty or safety mechanisms. The authors should
  probe these edge cases.

### Missing discussion of uncertainty and trustworthiness

- A major advantage of Bayesian optimization (BO) approaches is that their
  Gaussian-process surrogates provide uncertainty estimates and theoretical guarantees
  for exploration and convergence. The LLM-based approach offers no such calibration, at
  least no principled approaches. Thus, it remains unclear how users can trust or
  interpret its suggestions.
- The paper should explicitly compare BO and LLM agents on a more challenging task,
  evaluating not only final reward but also uncertainty quantification, robustness, and
  failure detection.

### Presentation and writing quality

- Several stylistic and formatting issues (random bolding, missing spaces, inconsistent
  punctuation) reduce readability and suggest unedited LLM-generated text.
- Figures are presented out of order: Figure 1 appears early but is referenced later;
  Figure 2’s caption misstates that the single-turn agent “calls the simulator” rather
  than calling the cost-utility surrogate (unless I am missing something general here?)
- I suggest having Figure 2 as the conceptual Figure 1 early in the paper.

### Questions
1. What is the downstream task or use case of optimizing simulator settings? How would
   this framework contribute to a real scientific or decision-making pipeline?
2. How does the method compare in practice to Bayesian optimization on more complex
   problems, particularly in terms of uncertainty estimates and failure detection?
3. Can the surrogate be integrated into a Bayesian decision-theoretic formulation
   similar to Gorecki et al. (2023), allowing direct Bayesian expected-loss minimization
   rather than heuristic LLM guidance?
4. How does the approach scale computationally when the simulator cost becomes dominant
   relative to LLM calls?
5. Have the authors evaluated or observed cases where the LLM proposes invalid or nonsensical
   simulator configurations? How are such failures detected or handled?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present a method to optimally select parameters for computationally complex computer simulations, comprised of a learned surrogate that models the cost and performance of different parameters and an agentic LLM that queries the surrogate to optimize the parameters.

### Strengths
The idea of training a neural network surrogate for an agentic LLM to query for optimizing simulations is interesting and novel. It is useful to be able to integrate prior knowledge into the optimization process.

### Weaknesses
The authors do not present a convincing argument that CAED-agent is consistently better than the state of the art. It is unclear what hyperparameters were used to generate the results for the baseline experiments. It is also unclear if a fair comparison was made, or in what dimension the baseline metrics were equivalent to the CAED-agent metrics. Did the methods have equivalent runtimes, equivalent queries of the simulator, or equivalent computational requirements? 

How does performance for each method change with different constraints? For example, BO is usually sample efficient and would likely still produce reasonable results with a few dozen queries of the simulator, but I would imagine the surrogate neural network would not be able to train on just a few dozen queries. Can the authors run an ablation on these parameters?

### Questions
If you have the neural proxy for the true simulator, does BO work on the neural proxy? On the other hand, is giving the LLM the training data directly and asking for an optimized result possibly more effective than using the data to train a neural proxy then having a separate LLM to query the proxy?

### Soundness
2

### Presentation
3

### Contribution
2
