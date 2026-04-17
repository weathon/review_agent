# Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 8, 6

## Abstract
AI agents have been developed for complex real-world tasks from coding to customer service. But AI agent evaluations suffer from many challenges that undermine our understanding of how well agents really work (Figure 1). We introduce the Holistic Agent Leaderboard (HAL) to address these challenges. We make three main contributions. First, we provide a standardized evaluation harness that orchestrates parallel evaluations across hundreds of VMs, reducing evaluation time from weeks to hours while eliminating common implementation bugs. Second, we conduct three-dimensional analysis spanning models, scaffolds, and benchmarks. We validate the harness by conducting 21,730 agent rollouts across 9 models and 9 benchmarks in coding, web navigation, science, and customer service with a total cost of about $40,000. Our analysis reveals surprising insights, such as higher reasoning effort reducing accuracy in the majority of runs. Third, we use LLM-aided log inspection to uncover previously unreported behaviors, such as searching for the benchmark on HuggingFace instead of solving a task, or misusing credit cards in flight booking tasks. We share all agent logs, comprising 2.5B tokens of language model calls, to incentivize further research into agent behavior. By standardizing how the field evaluates agents and addressing common pitfalls in agent evaluation, we hope to shift the focus from agents that ace benchmarks to agents that work reliably in the real world.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Holistic Agent Leaderboard (HAL), a new infrastructure for evaluating AI agents. The authors identify key challenges in existing agent evaluation, such as lack of standardization, high costs, slow speed, and the failure to detect "cheating" or unsafe behaviors. HAL's contributions are threefold: (1) A standardized, open-source evaluation framework that uses distributed scheduling to accelerate evaluation from weeks to hours and automatically tracks costs. (2) A large-scale, three-dimensional (model, benchmark, framework) empirical study ($40k cost, 21,730 runs) that yields insights, such as the existence of a steep cost-accuracy Pareto frontier and the finding that increased reasoning effort can hurt performance. (3) An automated log analysis component using an LLM-assisted tool (Docent) to uncover failure modes, shortcuts, and unsafe behaviors missed by traditional accuracy metrics.

### Strengths
Significance and Scale: The paper addresses a good problem: the lack of robust, standardized, and holistic evaluation for AI agents. The scale of the evaluation (21,730 runs, $40k cost, 9 models, 9 benchmarks) is impressive and provides a valuable snapshot of the current agent landscape.

Holistic Approach: The move beyond simple accuracy to systematically include cost (via Pareto frontiers) and reliability (via log analysis) is a major strength. This is a crucial direction for making agent evaluation more meaningful for real-world deployment.

Compelling Log Analysis Insights: The results of the log analysis are good. Demonstrating that agents "cheat" (e.g., searching for answers on HuggingFace) or engage in catastrophic behaviors (e.g., using the wrong credit card) provides powerful, concrete evidence for the paper's main thesis.

### Weaknesses
Contribution is Primarily Engineering: The core contribution of this paper is heavily weighted towards engineering. While this is a valuable service, the paper struggles to isolate a clear, novel research contribution. The insights derived (e.g., "scaffolding matters", "cost-performance tradeoffs exist") are useful but not deeply surprising. The main product is the framework itself.

Lack of Technical Depth in Main Paper: The paper relegates too many implementation details. Key components like the distributed scheduling system, the cost-tracking mechanism, and the "lightweight modification" required to integrate new agents are not sufficiently detailed. This makes it difficult to assess the novelty of the engineering work and the reproducibility of the system from the paper alone.

Log Analysis Lacks Methodological Novelty: The automated log analysis, while a key part of the "holistic" claim, is not a new method. It is an application of an existing tool, Docent (Meng et al., 2025). The novelty lies in the scoring criteria the authors developed for Docent, but these are not discussed in depth, making it hard to assess their generality, development cost, or potential biases.

Missing Artifact Details: For a paper centered on an infrastructure contribution, the provided text does not contain clear information on where to access the open-source framework or the 2.5 billion token log dataset. The value of this paper is almost entirely tied to the community's ability to use and build upon this artifact, yet access to it is not made clear. This omission makes it difficult to evaluate the paper's primary contribution.

### Questions
Please solve the listed weakness above.

### Soundness
2

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
The authors present an evaluation framework for benchmarking large language models. In doing so, they produce a meaningful evaluation of current models along cost and novel accuracy axes. Among other things, their work finds that a more expensive model is often not more performant than a cheaper model.

### Strengths
- Good work. With the number of different benchmarks coming out and the way people are increasingly gaming them, this kind of evaluation goes a long way. And with an inference cost of 40K, this provides a valuable sample for the community.
- The kind of failures mentioned align with what I've experienced in my own work. It's nice to see something that actually disseminates what the key subtle failures of LLMs in agentic works (e.g., I've also seen my agents cheat like this).

### Weaknesses
- I feel the contribution of this work is a bit weak, but I think the value of the data point this provides (see Strengths) is enough to outweigh this.
- I have some concerns about the statistical rigour of the results, but I can look past that. What I can't really look past is the analysis based on the Pareto frontier. Most would say that they prefer a model that scores a 9 out of 10 on 100 tasks than any of the 100 other models that each score 10 out of 10 on 1 task but 1 out of 10 on all of the other tasks. With the Pareto frontier, we would say the first model is the worst model by far. This is an even bigger problem, as we know that models are being trained to game the evaluations, so we expect that some models will excel on single tasks but perform poorly overall (i.e., overfitting models will be preferred). I would like to see some kind of smoother metric here. Maybe the average normalized distance to the Pareto frontier? An additional comparison with that is enough.
- For Discovery 2, and related to the above, given how you've drawn Pareto frontiers there using convex hulls, wouldn't that be pretty much guaranteed?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
I was invited as a supplementary reviewer for this paper. Since I received it less than five working days before the ICLR 2026 deadline on November 11, I may have missed some details while reading. If anything in my understanding is incorrect, I would appreciate it if the authors could point it out.

This paper tries to tackle several long-standing challenges in evaluating AI agents. The authors argue that current evaluations often suffer from inconsistent setups, high cost, long evaluation cycles, and sometimes untrustworthy results. To address these issues, they introduce the Holistic Agent Leaderboard, or HAL, which combines a unified evaluation infrastructure, a three-dimensional leaderboard across models, scaffolds, and benchmarks, and an automated log analysis system.

Through this framework, the authors conduct large-scale experiments and report a number of interesting findings. For example, they observe that higher reasoning effort does not always lead to better accuracy, and that the relationship between cost and performance is often nonlinear.

### Strengths
- [Large-scale and carefully executed empirical study] Running over 21,000 rollouts across 9 models and 9 benchmarks is nontrivial, both financially and logistically. The authors’ effort to document pricing, token usage, and performance variance across models gives the paper a solid quantitative backbone. The Pareto analysis between cost and performance is convincing and sets a precedent for cost-aware evaluation in future agent work.

- [Balanced benchmark and leaderboard design with meaningful difficulty] The authors have managed to calibrate the benchmark difficulty very well. Across the nine included tasks, there is enough separation among different agents to expose real capability differences, while state-of-the-art systems still leave considerable room for improvement. This balance suggests that the benchmark is neither trivial nor saturated, which makes it more likely to remain useful and relevant as agent technology evolves. The diversity of domains (coding, science, web navigation, and customer service) further helps prevent overfitting to a single narrow task type.

- [Strong systems thinking and reproducibility awareness]  The authors manage to unify heterogeneous agent scaffolds, benchmarks, and execution environments through a minimal API interface. The orchestration across hundreds of virtual machines, coupled with built-in token accounting and cost normalization, reflects an impressive level of engineering maturity.

### Weaknesses
- I acknowledge the strong engineering effort demonstrated in this work, but the paper feels overly engineering-oriented. It reads more like a large-scale system report than a research study that produces new principles or algorithms. The authors repeatedly emphasize that HAL “standardizes evaluation” and “makes agent benchmarking reproducible,” yet the methodological novelty remains quite thin. The reported 21,000 rollouts are impressive in scale, but the results mostly confirm intuitive trends: expensive models cost more, scaffolds matter, and reasoning helps inconsistently. The large experimental effort does not lead to equally deep conceptual insight. In several figures, the analysis stops at “we observe X,” without further investigation into why the phenomenon occurs or what it implies for the design of future agents. This is the main reason I give the paper a score of 6 instead of 8.

- Another concern is that several parts of the paper feel ad hoc in both design and analysis. The selection of benchmarks lacks a clear organizing principle. The nine included tasks span very different domains but do not follow a consistent taxonomy or rationale for why these particular tasks define a “holistic” view of agent capabilities. Moreover, the behavioral log analysis uses author-defined categories without theoretical grounding or validation, leading to largely descriptive findings. Overall, the paper reads as a collection of interesting observations rather than a systematically constructed methodology, which weakens its claim of being a “standardized evaluation framework.”

- If I did not miss it, the paper does not include any real comparison between HAL and existing evaluation frameworks. I am not referring to the brief listing in Appendix A1, but to a more meaningful validation such as consistency with human expert judgments or correlation with established benchmarks. Without such analysis, there is no quantitative evidence showing that HAL actually provides a more reliable or more accurate evaluation than existing methods. As a result, the claim that HAL improves standardization and trustworthiness remains largely unsubstantiated.

### Questions
I see this work mainly as a benchmark and large-scale analysis paper. I do not have particular questions for the authors at this stage. My earlier comments already cover the key concerns I have about framing, validation, and the depth of insight.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces the Holistic Agent Leaderboard (HAL), a unified evaluation framework for AI agents. HAL promotes standardized engineering practices for evaluation, providing a harness that features cloud-based orchestration for parallel execution and environment management, unified logging (via Weave), and automated log analysis (via Docent). The framework is structured to enable an orthogonal analysis across three dimensions: models, scaffolds, and benchmarks.

To validate this framework, the authors conducted 21,730 agent rollouts, testing 9 models across 9 benchmarks in domains including coding, web navigation, scientific research, and customer service. The analysis reports both accuracy and cost, computing the Pareto frontier for these two metrics.

The automated log analysis identified specific agent behaviors, such as taking shortcuts by finding benchmark answers on HuggingFace or making operational errors like misusing a credit card for a flight booking. The analysis also identified flaws in evaluation setups, including data leakage in an official scaffold. The authors have publicly released the open-source HAL harness, the leaderboard, and all 2.5 billion tokens of collected agent logs.

### Strengths
The paper presents a thoughtfully engineered system that addresses practical challenges in agent evaluation. It uses cloud-based orchestration for parallel execution, standardizes setups by decoupling scaffold from benchmark execution, and automates logging and trace analysis. The engineering work itself makes it a useful tool for agent researchers and practitioners.

The experiment design is comprehensive (covering 9 models across 9 benchmarks in 4 domains and totaling 21,730 rollouts). This relatively large-scale study helps drive convincing results on cost-accuracy tradeoffs via Pareto analysis, and identifies interesting reward hack patterns like searching on HuggingFace.

The authors have released the open-source HAL harness, the public leaderboard, and the complete agent logs of  21,730 rollouts, providing a resource for research on agent behavior and reliability.

### Weaknesses
Although the paper provides a useful engineering tool and a large-scale evaluation, its primary analytical methods and insights are not particularly novel. For instance, the cost-accuracy Pareto frontier analysis was previously established as a core evaluation principle by prior work (e.g. https://arxiv.org/abs/2407.01502). 

Similarly, the reward hacking behavior identified, such as searching for benchmark answers, is a concrete example of the known problem as discussed in Search-Time Data Contamination (https://arxiv.org/abs/2508.13180). Given the 2.5 billion token trajectory dataset, it would be great if the authors could offer more unique methods/insights beyond these known issues, such as a systematic clustering of common failure patterns or a quantitative analysis of inefficient agent trajectories, etc.

### Questions
The paper's reasoning effort analysis focuses on model-level API parameters. However, complex scaffolds may have their own reasoning budget/context parameters (e.g., max_observation_length for swe-agent). Could the authors provide more details here, and assess if there's any impact on the evaluation results?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
I have read this paper on ArXiv, and therefore know who the authors are. Thus, I am unable to review the paper. I have informed the ACs of this. Please ignore this review. The following options are randomly chosen to avoid biasing other reviewers.

### Strengths
Not applicable

### Weaknesses
Not applicable

### Questions
Not applicable

### Soundness
2

### Presentation
1

### Contribution
4
