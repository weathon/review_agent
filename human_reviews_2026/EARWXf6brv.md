# Battery-Sim-Agent: Leveraging LLM-Agent for Inverse Battery Parameter Estimation

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Parameterizing high-fidelity ``digital twins'' of batteries is a critical yet challenging inverse problem that hinders the pace of battery innovation. Prevailing methods formulate this as a black-box optimization (BBO) task, employing algorithms that are sample-inefficient and blind to the underlying physics. In this work, we introduce a new paradigm that reframes the inverse problem as a reasoning task, and present \textsc{Battery-Sim-Agent}, the first framework to deploy a Large Language Model (LLM) agent in a closed loop with a high-fidelity battery simulator. The agent mimics a human scientist's workflow: it interprets rich, multi-modal feedback from the simulator, forms physically-grounded hypotheses to explain discrepancies, and proposes structured parameter updates. On a systematically constructed benchmark suite spanning diverse battery chemistries, operating conditions, and difficulty levels, our agent significantly outperforms strong BBO baselines like Bayesian optimization in identifying accurate parameters. We further demonstrate the framework's capability in complex long-horizon degradation fitting tasks and validate its practical applicability on real-world battery datasets. Our results highlight the promise of LLM-agents as reasoning-based optimizers for scientific discovery and battery parameter estimation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents an LLM-simulator-in-the-loop optimization method for inverse battery parameter estimation. It uses multimodal features, including numerical data and plot images.

### Strengths
The usage of LLM on this specific application is quite innovative.

### Weaknesses
__Baselines.__ The paper uses BO as the baseline but does not clearly specify which acquisition function is used, nor does it provide sufficient detail to justify the fairness of the comparison (for example, in terms of time budget, number of function evaluations etc). Moreover, only one baseline is used, yet it is referred to as a “strong” baseline.

__Details on the LLM used.__ The description of the LLM is unclear. For instance, are the models trained locally from scratch, or are they accessed through OpenAI API calls? Additionally, according to [1], GPT-OSS is not an image–text model. It is therefore unclear how the LLM is used to extract image features in this case. This concern needs to be addressed.

__Loss definition.__ It appears that the Battery-Sim-Agent uses a different loss definition from the baseline. If I understand correctly, the baseline loss is defined in Equation (1), while the Battery-Sim-Agent employs a multi-objective loss defined in Equation (2). This does not seem to be a fair comparison. For example, using multi-objective BO [2, 3, 4] with an appropriate acquisition function might provide a fairer baseline.

__Missing citation and comparison to a key paper.__ While the paper is quite novel within the battery parameter estimation problem, similar techniques have already been presented in [5], where an LLM is also utilized for parameter inference in physical systems. Omitting this related work significantly weakens the paper’s claim to novelty, and a detailed comparison or discussion is necessary.

[1] https://openai.com/index/introducing-gpt-oss/

[2] Preferential Multi-Objective Bayesian Optimization (Astudillo, et al. 2024)

[3] Efficient computation of expected hypervolume improvement using box decomposition algorithms (Yang, et al. 2019)

[4] A Flexible Framework for Multi-Objective Bayesian Optimization using Random Scalarizations (Paria, et al. 2019)

[5] SimLM: Can Language Models Infer Parameters of Physical Systems? (Memery, et al. 2024)

### Questions
__Q1.__ The paper requires significant improvement in clarity, scientific rigor, and baseline selection. I would consider increasing my score if the issues highlighted in the weaknesses are adequately addressed.

__Q2.__ It would also be helpful to include a counter-example where the Battery-Sim-Agent fails. For instance, when the initial memory $M_0$ provided by the user is incorrectly specified. Such a case could be interpreted as a wrongly defined prior, and it would be interesting to observe how the LLM behaves under this condition.

__Q3.__ are the target protocols $Y_p$ in eq (1) all have the same range? (i.e trough normalization/standardization)

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces a LLM-based framework for inverse design battery parameter estimation. This complicated optimisation problem arises when trying to match microscopic parameters to experimentally measurable observables. Traditionally, this was done using black-box optimisers like Bayesian optimisation. The workflow consists of two phases—a exploration phase and an optimisation loop afterwards, in which the parameter updates are suggested by the LLM. Simulation benchmark tasks are defined and tested comparing their approach with and without reasoning with BO. Long-horizon degradation fitting and real world validation benchmarks are also included.

### Strengths
The paper tests the interesting idea of replacing novel black-box optimisers with a LLM and applies this to estimating parameters of complicated physical systems.

### Weaknesses
Some of the benchmark results do not seem in line with the  claims of the paper. 

- "our agent significantly outperforms strong BBO baselines", in Table 2 BO outperforms the agent on 2 of 5 chemistries in the "extreme mode".
-  It is said "Figure 4 shows convergence behaviour ... revealing robust optimization", in panel (b) a MAPE  of 230 does not suggest convergence.

The paper does not discuss limitations of the approach (e.g. provides no convergence guarantees). Comparisons against related approaches that make use of BO plus an expert (LLM) would be helpful. (e.g. https://arxiv.org/abs/2410.10452, Principled Bayesian Optimisation in Collaboration with Human Experts )

The design choices of Algorithm 1 are not sufficiently explained and ablation studies would be insightful.

- Why is there a random trial and error warm up, instead of prompting the LLM to explore?
- Why is the LLM prompted to predict the parameter updates rather than directly the next set of parameters?

### Questions
- The tasks based on the 5 chemistries seem to be designed by varying a few or even just a single of the experimental parameters. It seems that the LLM then starts its search from the original unperturbed parameter. This seems very different than an actual experimentally fitting task. Furthermore, the LLM prompt suggests to only modify only some parameters which does not seem fair, since the tasks seem sparse by construction and the BO probably does not "know" that.
- The description of Algorithm 1 does not seem consistent with the prompt. Are the pertubations $\delta_k$ random or generated by the LLM? Is $N_w$ a fixed input?
- I was not able to find values for the budget $T$ and learning rate $\eta_t$, nor was I able to find details on the BO.
- I was not able to access the source code under the link because all files besides the Read.me files gave the error "file not found."
- The problem is motivated by saying that microscopic parameters cannot be easily measured. However several of the parameters seem to be design or layout choices which should be known a priori?
- When referring to the stability limits of the simulator for certain choices of experimental parameters, why is a decrease in resolution or step sizes not possible?
- I do not understand Fig 2. I would expect that there are only 100 Experiment ID for each mode, and that for each ID are 3 datapoints corresponding to the 3 methods used.
- Why is BO not tested on the real-world tasks?
- The design or at least description of the benchmark based on the 5 chemistries and simulation seems inconsistent.
- From the 5 listed chemistries described as "classic, well established parameter sets", I was not able to verify the correctness for 4 of the specifications. (Chen et al seems to use graphite SiO_x instead of graphite, O'Regan et al. seems to use NMC811 and not NMC 532,  Marquis et al. is a maths theory paper without specifying experimental parameters).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces an agentic optimization framework for a simulator-in-the-loop LLM agent applied to a battery inverse problem. The method is evaluated on both synthetic and real-world setups, demonstrating good performance relative to the baseline.

### Strengths
The studied application problem is highly relevant, and the idea of using LLMs as reasoning engines within optimization frameworks is timely. Ignoring the issue of baseline selection, the benchmarks in Section 5 are diverse and include many real-world datasets.

### Weaknesses
The method description is insufficient. The main subsections 3.1 and 3.2 are short and list-like, and there is a lack of discussion and justification of the design choices. Below I list some weaknesses in the experimental section of the paper.

Experimental setup

Section 5 lacks relevant aspects discussion on the experimental setup. Instead of starting listing claims: “We conduct comprehensive experiment... Our evaluation demonstrates the superiority…”, it would be good start by explaining the high-level experimental setup and hypothesis. 

Baselines

“Bayesian Optimization (BO): We use standard Bayesian Optimization implemented by Meta’s Ax platform (Olson et al., 2025), representing state-of-the-art black-box optimization methods commonly used in parameter estimation.”
This is clearly insufficient explanation of the baseline, and does not build trust that the BO baseline selection is carefully considered as all the important details are hidden such as what was acquisition function etc.

Minor comments:

Sentences in Lines 32-34 lack citations. 
Figure 4 is not good quality. Font is too small, etc.

### Questions
What is the main justification for framing the problem as gradual updates $\Delta \theta_{t}$ to the current parameter vector rather than propose new parameter configuration $\theta_{t+1}$?

Equation (1) collapses multi-objective problem into single objective. Did you consider frame the problem as multi-objective problem, and use e.g. multi-objective BO?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents BATTERY-SIM-AGENT, a framework that integrates a large language model (LLM) agent in a high-fidelity battery simulator. The agent emulates a human scientist’s workflow: it interprets rich, multimodal feedback from the simulator, forms physically grounded hypotheses to explain discrepancies, and proposes structured parameter updates.
The work demonstrates that LLM agents can serve as reasoning-based optimizers in scientific applications.
Traditional digital twin approaches iteratively query a simulator and minimize the mismatch between simulated and observed data using black-box optimization. In contrast, this paper explores whether the inverse problem of battery parameter estimation can be reframed as a reasoning-driven scientific workflow guided by an LLM agent.
In this framework, the LLM agent functions as an AI scientist: at each iteration, it receives multimodal feedback comparing the current simulation with experimental data, identifies key discrepancies, formulates physical hypotheses (e.g., “a premature voltage drop suggests an electrolyte transport limitation”), and proposes targeted parameter updates accordingly.

The main steps of this approach:
1)	The agent receives a multi-modal feedback package in a structured JSON format.
2)	Guided by its memory, the agent analyzes this feedback to form a causal hypothesis. The prompt encourages a scientific reasoning process
3)	The agent is prompted to translate its hypothesis into a concrete, machine-actionable update, which it returns in a JSON format.

### Strengths
The paper is interesting, timely, and well written.
The paper demonstrates that their agent can achieve 67-95% reduction in error compared to traditional black-box approaches.

### Weaknesses
This approach relies on large language models as reasoning agents. In particular, the authors use GPT-O3. However, since the training data and internal reasoning mechanisms of GPT-O3 are not publicly known, reproducibility becomes a concern—future updates or changes to GPT-O3 could make these experiments non-reproducible.

### Questions
* Line 231: “We initialize the memory M_0 with human expert knowledge from the literature and our own domain expertise”. What exactly was M_0? How large was M_0? Is the full M_0 available somewhere?

* Line 235: “agent undergoes a warm-up phase… The resulting feedback is not for optimization, but is processed by the LLM to enrich its memory… The agent is prompted to summarize the outcomes into learned sensitivity rules”. How many new rules were learned this way? Is the full list of learned rules available somewhere?

* Line 389: “What was the cost of GPT-O3”? What other language models might be suitable for these tasks?

### Soundness
3

### Presentation
3

### Contribution
3
