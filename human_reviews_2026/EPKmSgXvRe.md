# Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Bayesian Optimization critically depends on the choice of acquisition function, but no single strategy is universally optimal; the best choice is non-stationary and problem-dependent. 
Existing adaptive portfolio methods often base their decisions on past function values while ignoring richer information like remaining budget or surrogate model characteristics. 
To address this, we introduce LMABO, a novel framework that casts a pre-trained Large Language Model (LLM) as a zero-shot, online strategist for the BO process. 
At each iteration, LMABO uses a structured state representation to prompt the LLM to select the most suitable acquisition function from a diverse portfolio.
In an evaluation across 50 benchmark problems, LMABO demonstrates a significant performance improvement over strong static, adaptive portfolio, and other LLM-based baselines. 
We show that the LLM's behavior is a comprehensive strategy that adapts to real-time progress, proving its advantage stems from its ability to process and synthesize the complete optimization state into an effective, adaptive policy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Language Model-assisted Adaptive Bayesian Optimization (LMABO), a policy for selecting, at each iteration of a BO algorithm, the most promising acquisition function to optimize for finding the next observation. LMABO involves a portfolio of acquisition functions and a Large Language Model (LLM) that is tasked to pick the most promising acquisition function given a prompt that summarizes the current optimization state.

LMABO is evaluated against popular acquisitions functions, state-of-the-art strategies for acquisition function selection and heuristics on 50 benchmarks. The numerical results show a statistically significant improvement over the state-of-the-art.

### Strengths
* A new strategy for selecting an acquisition function would likely spark interest in the BO community.

* The paper is easy to read and well-written.

* Extensive numerical evaluation with relevant statistical tests.

### Weaknesses
* The main weakness is probably the lack of understanding of how the LLM takes its decision. Although LMABO requires a justification from the LLM, there is no guarantee that this justification is actually valid. In that sense, I think an additional control solution, consisting of picking the acquisition function uniformly at random in the acquisition function portfolio, would be very interesting to consider.

* Another worry that I have is that LMABO builds on implicit knowledge of the LLM about acquisition functions in the portfolio. Therefore, integrating a brand new acquisition function in LMABO would probably prove to be challenging, as the LLM would not have any prior knowledge about it.

* Minor presentation issues: a few typos (e.g., "miultiple" on line 125) and the captions of Figure 1(c) and 1(d) are hard to read.

### Questions
Here are a few questions to spark the discussion with the authors.

* Do you know how a simple control strategy that picks an acquisition function at random in the portfolio would behave? What about a strategy that randomly picks one of the most popular acquisition functions (e.g., a reduced portfolio with EI, TS, UCB, PosMean)?

* How would you plan to add a new, unknown acquisition function to the portfolio?

* For each real-world experiment, have you tried to exploit some prior knowledge of the LLM on this particular task? For example, for an hyperparameter optimization task involving a gradient step $\epsilon$ and a batch size $b$, have you adapted the initial prompt $P_0$ to describe this particular problem and to indicate to the LLM that it was working on the optimization of a gradient step and a batch size?

### Soundness
3

### Presentation
4

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
The paper introduces LMABO (LLM-Assisted Adaptive Bayesian Optimization), a novel framework that utilizes a pre-trained Large Language Model (LLM) as a zero-shot, online strategist to dynamically select the most suitable acquisition function (AF) at each iteration of the Bayesian Optimization (BO) process. LMABO addresses the limitation of existing adaptive methods by employing a structured state representation that translates the complete optimization context—including remaining budget, performance history, and surrogate model (GP) characteristics—into a text prompt for the LLM to reason over. Evaluated across 50 benchmark problems, LMABO demonstrated significant performance improvement over static, adaptive portfolio, and other LLM-based baselines (e.g., 55% lower total AUC than the best LLM-based method), confirming that its success is due to an emergent, robust, and state-aware policy for controlling the exploration-exploitation trade-off.

### Strengths
* The paper introduces a framework that successfully recasts the task of acquisition function (AF) selection in Bayesian Optimization (BO) as an in-context decision-making problem.   This approach leverages a pre-trained Large Language Model (LLM) to select the most appropriate AF at each optimization step based on its implicitly encoded knowledge of optimization principles.
* A primary technical contribution is the design of a structured state representation that translates the complex, multi-faceted numerical state of the BO process into a concise textual summary.   This summary includes crucial strategic information, such as the remaining optimization budget and insights from the surrogate model's GP lengthscales, overcoming the limitation of prior adaptive methods that relied only on narrow performance history.
* Extensive experiments across a diverse set of 50 benchmarks show that LMABO achieves a significant performance improvement over established static, adaptive portfolio, and other LLM-based baselines.   Analysis confirms that the framework’s effectiveness stems from an emergent, state-aware policy that executes active, dynamic switching between exploratory and exploitative AFs in response to real-time progress , demonstrating an agility beyond simple pre-defined heuristics.

### Weaknesses
* The LMABO algorithm requires calling the Large Language Model at every single optimization step. Although the financial cost per token is very low, the overall economic feasibility for extensive or complex problems using commercial LLM APIs requires more detailed analysis.
* The framework's high performance is fundamentally dependent on the sophisticated reasoning capabilities of the underlying LLM. The core implementation relies predominantly on Gemini-2.5 Flash. Ablation studies confirm this sensitivity, showing a noticeable drop in performance when using smaller open-source models, alongside a dramatic decrease in time efficiency. To ensure the framework's broad reliability and generalizability, its efficacy requires additional, comprehensive validation across a wider spectrum of diverse and competitive LLMs.

### Questions
* Could the authors provide a more detailed analysis of the economic costs associated with utilizing the LMABO framework?

* The core, high-performing LMABO implementation relies predominantly on Gemini-2.5 Flash.   Given that ablation studies demonstrate a noticeable performance drop when using smaller open-source models, could the authors verify the framework's broad reliability and efficacy by testing its performance and execution speed with other competitive closed-source LLMs (beyond Gemini-2.5 Flash) or high-capability open-source models from different architectures?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LMABO, a zero‑shot strategy that delegates acquisition-function selection in Bayesian Optimization to Large Language Model. At each BO iteration, the model receives a state with: optimization state—process status (iteration count, remaining budget, dimensionality), performance history. This information is useful for the LLM to sample the best next acquisition function to be used.   Basically the method can be seen as an LLM doing a meta decision over an additional component of the acquisition process. The authors show strong result, which highlights the significance of the paper.

### Strengths
**Strong results.** Casting AF selection itself as an in‑context decision problem for an LLM shows that the meta decision of selecting the acquisition function is highly relevant for the whole acquisition process. The paper shows strong results.

### Weaknesses
**Novelty.** While the framing is neat, the algorithmic contribution largely reduces to a prompt + a state serialization + a portfolio choice. Many recent works have leveraged LLM inductive biases for decision selection or candidate generation in BO, so the conceptual step—“use an LLM to pick an AF given a textual state”—feels incremental without additional design elements.


**What paper should have.** The paper positions an LLM as the decision-maker for selecting the acquisition function, with the main contribution lying in the experiments. Two questions should be addressed: (1) Can the LLM reliably choose an appropriate acquisition function at each point in time? and (2) Do the LLM’s choices align with—or diverge from—established expectations, given that its reasoning comes from in-context learning? The paper convincingly demonstrates (1). It attempts to address (2), but the presentation could be stronger: the Background section does not give readers enough context to evaluate why the LLM makes the choices it does.

**Results.** The ablation studies are helpful but insufficient. The experimental setup around Figure 1 seems to be underspecified—it’s unclear whether the figure is illustrative or aggregates results across multiple runs. Results in Figure 2 show high variance, which makes it difficult to draw robust conclusions about the method’s behavior

### Questions
See weaknesses

### Soundness
3

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
4

### Summary
The paper studies meta–decision making in Bayesian Optimization. Specifically, how to select the acquisition function (AF) for a given task/budget—and proposes using a Large Language Model (LLM) as a source of prior knowledge to guide that choice. Concretely, the method extracts task context (e.g., problem description, dataset traits, search‑space information, etc) and queries an LLM to recommend an AF (e.g., EI, UCB). The authors evaluate this idea across a broad set of AutoML/HPO problems and report that making the right meta decision (AF selection) substantially affects BO performance.

### Strengths
The work tackles a meta design choice in BO/AutoML—which AF to use—rather than only tuning model hyperparameters. In practice this could translate in a more robust “out‑of‑the‑box” optimizer that requires less trial to find the optima. 

The paper reports extensive experiments indicating that the meta decision is relevant. It is important to have flexibility in what AF to choose along the acquisition process.

### Weaknesses
Presentation. I think the presentation and how the paper is written is not good in general. To give some examples. 1) Some figures are hard to read or to use for significance judgments. In Figure 2, the variance of the result is  wide, making it difficult to visually assess the significance of the results. 2) Table 2 uses cryptic labels (“LMABO‑AB1…AB4”); these should be replaced or augmented with descriptive names.
Novelty. The idea of leveraging LLM “knowledge” to steer BO is not brand‑new (e.g., LLAMBO, LLMP, FunBO; Sec. 3), although applying it specifically to the meta decision of AF selection is useful as the authors have shown empirically.
Ablation. The paper includes several ablation studies, which is valuable. However, this should also be examined at a more global level to demonstrate that the observed patterns consistently repeat across multiple runs and different experiments.

### Questions
See above

### Soundness
2

### Presentation
1

### Contribution
2
