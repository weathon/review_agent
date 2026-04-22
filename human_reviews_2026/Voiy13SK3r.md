# Helmsman: Autonomous Synthesis of Federated Learning Systems via Collaborative LLM Agents

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 2, 6, 6

## Abstract
Federated Learning (FL) offers a powerful paradigm for training models on decentralized data, but its promise is often undermined by the immense complexity of designing and deploying robust systems. The need to select, combine, and tune strategies for multifaceted challenges like data heterogeneity and system constraints has become a critical bottleneck, resulting in brittle, bespoke solutions. To address this, we introduce Helmsman, a novel LLM-based multi-agent framework that automates the end-to-end synthesis of federated learning systems from high-level user specifications. It emulates a principled research and development workflow through three collaborative phases: (1) interactive human-in-the-loop planning to formulate a sound research plan, (2) modular code generation by supervised generative agent teams, and (3) a closed-loop of autonomous evaluation and refinement in a sandboxed simulation environment. To facilitate rigorous evaluation, we also introduce AgentFL-Bench, a new benchmark comprising 16 diverse tasks designed to assess the system-level generation capabilities of LLM-driven agentic systems in FL. Extensive experiments demonstrate that our approach generates solutions competitive with, and often superior to, established hand-crafted baselines. Our work represents a significant step towards the automated engineering of complex decentralized AI systems. Code is available at: https://github.com/haoyuan-l/Helmsman.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce Helmsman, a multi-agent framework that aims to automate the design, implementation, and evaluation of federated learning systems. The architecture is made up of a planning agent that creates research plans from user-defined text prompts, coder–tester agent pairs that produce and verify modular code components (which include task, client, strategy, server), and an evaluation agent that runs and debugs the code in a sandboxed Flower simulation. The authors claim that this approach achieves autonomous system synthesis across 16 benchmark tasks (AgentFL-Bench, which the authors have created), reaching full automation on around 60% of them and outperforming standard FL baselines.

### Strengths
The system itself is well articulated and makes sense for the problem at hand. The pipeline itself demonstrates seemingly good software engineering. The benchmark itself is a useful addition as AgentFL-Bench gives a controlled setting for evaluating end-to-end FL code synthesis and could be useful for reproducibility studies. The paper is well written and documents the workflow, examples, and failure handling pretty clearly.

### Weaknesses
The architecture which is developed doesn't seem to have anything intrinsically to do with federated learning. The architecture, multi-agent planning, code generation, and testing is generic and could apply to any modular ML task . The FL focus seems to come about only from the chosen templates and evaluation tasks, not from algorithmic design. This raises the question of why the problem is framed as FL rather than general code-synthesis automation.

Because of this, it should be compared with other code-synthesis automation pipelines, which have been left out. The system is pretty similar in philosophy, and at times details to previous frameworks such as SWE-Agent, AutoGPT, Voyager, and CodeAct, all of which perform multi-agent planning, code writing, and iterative debugging. Helmsman introduces no new learning mechanism or reasoning algorithm beyond applying this template to FL code. The paper doesn't compare against or even reference these lines of work, making the claimed novelty really difficult to justify.

There's also a real circularity in the evaluation and fragility in the prompt design. All tasks in AgentFL-Bench follow the same structured Problem–Task–Framework schema that Helmsman is designed specifically to take. Because of this, the benchmark measures success within a self-aligned template rather than general robustness to unstructured or underspecified prompts. There is no test of whether the system works if prompts are paraphrased, incomplete, or out of schema, which severely limits claims of autonomy and generalization.

The results given of 62.5 % automation rate and 5–10 % improvements over FL baselines lack any statistical grounding. No confidence intervals, standard deviations, or number of runs are reported. It has to be presumed that this number is from a single run, with presumably hand-tuned prompts, so without checking for robustness from prompt design, these results are really not meaningful. In addition, because of the inherently stochastic nature of LLM generation, there must be some considerable variance in results which are never specified. Again, this makes the claimed improvement of a few percent impossible to judge.
In addition, the baselines (FedAvg, FedProx) measure learning accuracy, not system-synthesis performance, so these comparisons are not really comparable at all.
The empirical results should then be viewed as illustrative demonstrations, not evidence of reliable performance gains.

In addition to all of this, for around 40% of tasks, human verification was required. Given this, it is unclear what advantage Helmsman offers over ready-made tools such as GitHub Copilot or Cursor, which already provide interactive code completion, conversational debugging, and execution feedback. Without quantitative comparison to really simple baselines such as these, the added complexity of the multi-agent architecture is not justified.

Finally, the paper has no ablation experiments testing sensitivity to prompt wording, number of agents, or LLM choice, nor anything focusing on the value of planning vs. coding agents. Again, because of this it is unclear which components, if any, are responsible for the reported successes.

### Questions
The questions are all about the weaknesses, and so each weakness should be seen as a question about the paper.

### Soundness
1

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Helmsman, a multi-agent system for synthesizing a federated learning system through user specifications. It (1) involves human-in-the-loop planning (2), multi-agent code generation (3), and improvement from a simulation environment. The paper also introduces AgentFL-Bench to evaluate the system generation in FL. Experiments show improved performance from heuristic baselines.

### Strengths
(1) This paper is presented clearly with its methodology and benchmark
(2) According to my knowledge, applying multi-agent systems to federated learning is novel, and the problem of automated research experiments is interesting.

Disclaimer: I don't have a federated learning background; I am from a more classic multi-agent RL community. This paper is a multi-agent application paper from our side. I invite other reviewers from the federated learning community to comment on the novelty and significance of federated learning.

### Weaknesses
(1) The takeaway of the paper is "multi-agent helps federated system more compared to heuristics when carefully designed". It is closer to a system design report.  In research papers, more ablation studies should demonstrate each design component for rigor.
(2) The "human-in-the-loop" part of the method makes the experiments potentially highly manipulable and may cause unfairness when comparing the method with other fully automated methods.

### Questions
(1) Does human-in-the-loop indicate the system will fail without such a component?
(2) Did you ever try single-agent baselines without such a complex design?
(3) If you were to design a minimalist system, what are the key components that you would keep, and what are disposable?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents HELMSMAN, a multi-agent framework designed to automate the end-to-end synthesis of robust Federated Learning (FL) systems based on high-level user specifications. The introduction details how FL deployment is challenged by data heterogeneity, system constraints, and the manual, brittle nature of current solutions. HELMSMAN systematically addresses these through a three-phase workflow: (1) interactive human-in-the-loop planning, (2) modular code generation by coordinated agent teams, and (3) autonomous iterative evaluation and refinement in sandboxed environments. The system integrates advanced tools for knowledge retrieval and real-world simulation, ensuring plans are grounded and code is validated. The authors introduce AgentFL-Bench, a new benchmark with 16 tasks reflecting diverse FL challenges. Experimental results show that HELMSMAN outperforms or matches handcrafted and specialized baselines across various domains, demonstrating its ability to synthesize innovative algorithmic solutions, especially in complex settings like continual learning. Discussions highlight the benefits of multi-agent collaboration and human oversight, as well as current limitations regarding computational costs and task complexity. The work concludes by emphasizing HELMSMAN’s progress toward autonomous FL engineering and outlining ambitions for self-evolutionary capabilities in future iterations.

### Strengths
- Pioneers a principled, modular, multi-agent approach to automating FL system synthesis, beyond monolithic or single-agent methods prevalent in prior works.
- Empirically demonstrates that solutions are competitive with, or exceed, established baselines on a diverse benchmark.
- The AgentFL-Bench is broad (16 tasks, 5 domains); task specifications and comparisons are thorough.
- The system workflow, agent interactions, and evaluation setup are mostly well articulated.
- The division between planning/coding/evaluation stages is clear and aids reproducibility.
- Provides a tangible tool and benchmark that can catalyze further research on automated ML/FL system synthesis.

### Weaknesses
- Results lack statistical robustness details: variances or confidence intervals on reported metrics are not provided, which undermines confidence in the reliability of improvements.
- The number of runs, error bars, or statistical significance testing are absent from main tables.
- Details on whether all baselines were re-implemented and evaluated under identical splits, models, hyperparameters, etc. are scattered (mainly deferred to the appendix), making it hard to assess fairness of comparisons from the main text.
- Key configuration details (data splits, client counts, etc.) are not summarized in the main paper.
- Computational costs (runtime, LLM API usage, memory) for Helmsman, particularly relative to hand-crafted alternatives, are not sufficiently quantified in the main text.

### Questions
- While HITL phases are discussed, the conditions, guidelines, and extent of required human intervention (especially for more ambiguous tasks) are somewhat vague. Could the authors provide explicit criteria or real-world examples delineating when and to what degree HITL is invoked, and how its frequency impacts system autonomy and scalability? Understanding the precise role and burden of human intervention is important for assessing Helmsman’s autonomy and practicality.

- The choice of agent specialization (planning, coding, debugging, etc.) is motivated by division of labor, but could the authors elaborate whether alternative agent decompositions were explored? For instance, would a two-agent system (planner+debugger) suffice, or are the current modular teams truly necessary for observed performance? This helps validate the necessity and impact of the multi-agent structure versus simpler alternatives.

- Could you provide more explicit detail on the communication and coordination protocols among agents (Supervisor, Planning, Reflection, Evaluator, Debugger, etc.), especially in cases of conflicting suggestions or persistent failures? Understanding the orchestration mechanics and how conflicts or deadlocks are resolved is crucial for reproducibility and for assessing the practical reliability and scalability of the system.

- What is the impact of key hyperparameters (e.g., T_max, agent model choice, number of local updates, communication rounds) on system performance, and is there an automated or principled tuning mechanism in Helmsman? The methodology relies on several non-trivial, possibly task-sensitive settings; transparency around their selection, sensitivity, and potential for automation is necessary for adoption and rigorous evaluation.

- Could you clarify the computational resources (runtime, memory, LLM API usage) required for typical benchmark runs, and to what extent does Helmsman facilitate fully reproducible experiments on AgentFL-Bench? Resource efficiency and reproducibility are essential aspects for practitioners and for fair comparison to baselines; disparities here might hinder wider use or interpretation of results.

- Could you clarify whether the reported results (in Tables 2–5) are averaged over multiple runs, and if so, how many runs and what the variance or confidence intervals were? Assessing consistency and statistical significance is critical for experimental rigor in FL, where stochasticity can meaningfully affect outcomes.

- Are the baseline methods (both standard and specialized) reimplemented under identical data splits, model architectures, and hyperparameter configurations as Helmsman, or are external numbers reported? Ensuring a fair and reproducible comparison is essential for drawing valid conclusions about Helmsman's relative performance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Helmsman is a multi-agent system that designs federated learning frameworks. It uses planning, coding, and evaluation agents that work together to generate and refine working FL systems. Authors also design tasks to evaluate their method. Experiments show it produces results as good as or better than hand-crafted methods.

### Strengths
The paper explores multi-agent system design in federated learning, a space that has been largely underexplored

The authors introduce AgentFL-Bench to test and compare autonomous FL systems

Helmsman is a meta agent, a system that builds other AI systems, automating the design and testing process. This approach offers a promising path toward scalable and efficient FL pipeline development.

### Weaknesses
The system depends heavily on large computational resources and simulation time which limiting real-world scalability.

62.5% of tasks ran fully automatically, while the remaining tasks needed some human-in-the-loop input, which suggests that true autonomy is not yet achieved.


The paper doesn’t compare Helmsman with other meta-agent systems that could also design federated learning setups, missing a fair baseline for comparison.

Helmsman’s variability arises from LLM randomness, agent interactions, stochastic simulations, and iterative debugging, all of which make outcomes slightly different each time it runs. However, the paper does not report standard deviations for these results.

### Questions
How stable are Helmsman’s synthesized FL systems across multiple independent runs? What mechanisms could ensure reproducibility?

### Soundness
3

### Presentation
3

### Contribution
3
