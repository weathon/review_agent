# SOCIA: Joint Structure–Parameter Co-Optimization for Automated Simulator Construction

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Building credible simulators from data is difficult because structure design, parameter calibration, and out-of-distribution (OOD) robustness are tightly coupled. We introduce SOCIA (Simulation Orchestration for Computational Intelligence with Agents), a framework that treats simulator construction as joint structure–parameter co-optimization: it elicits mechanism-rich blueprints, exposes explicit tunable parameters, and instantiates a calibration schema, producing an executable simulator with built-in calibration hooks. SOCIA couples Bayesian Optimization for sample-efficient point calibration with Simulation-Based Inference for uncertainty-aware fitting; diagnostics trigger targeted structural edits in an outer refinement loop to co-optimize design and parameters under tight budgets. Across three diverse tasks, SOCIA consistently outperforms strong baselines, excelling on both in-distribution (ID) fitting and OOD shift. Ablations that weaken structure, calibration design, or tuning yield near-monotone degradations, underscoring the necessity of unified structure–parameter optimization. SOCIA’s code and data are available here.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The article introduces a framework for building and tuning simulation models (primarily agent based models). The chain starts with a chain-of-structure specification to encapsulate the situation being modelled. An LLM is used to encode this framework into code, and a combination of Bayesian optimisation for parameter tuning, and further LLM methods for model updating, are used to fine tune the model. Experiments are given in which, of course, the current method provides higher performance than the baselines.

### Strengths
I really liked this article's full spectrum approach to modelling, going all the way from the problem specification to fine tuning and testing the resulting simulation model.
The paper is on a different topic to most machine learning conference papers (at least the ones I read) but uses lots of highly relevant techniques and addresses problems that machine learning researchers address, using a slightly different toolset. I think it would be a strong contribution to the conference.

### Weaknesses
There are some gaps in the presentation. Primarily, the key "Algorithm 1" is actually in the Appendix, which I think is verging on cheating the page limit unfairly.
However there were also lots of details of the method that I could not ascertain from reading the article. The most frustrating one for me was the iterative refinement of model structure. Only one sentence used to present this (lines 314-315) but it is a complex and interesting part of the method.

### Questions
Please can you give some more detail on each of the components of your method (but not the Bayesian optimisation, which is completely standard). I think they're all interesting but will struggle to accept a paper when so much of the method is not described at all.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes an agentic workflow for automatically building multi-agent simulations based on observational data. The main novelty seems to be the joint optimization of the simulator structure and its hyperparameters.

### Strengths
n.a.

### Weaknesses
[motivation and introduction] While the paper tackles a significant and complex challenge with state-of-the-art methods, the introduction is highly generic. It does not adequately establish what specific task SOCIA performs or what kind of simulators it produces. After multiple readings of the intro, it remains unclear to me what the system’s concrete task and output is (see Figure 1: is SOCIA generating executable code, calibrated data, forecasts, or decision policies?). This becomes a bit clearer after browsing the appendix. 
--> Providing an early, concrete example of a representative simulator and a clear definition of the modeling task would give readers a much clearer entry point into the paper.

[readability] The paper’s notation is inconsistent and often undefined, which makes it hard to understand the central methodology. For instance, $\lambda$ is described as a “mechanistic blueprint” (l161) while $B$ is later introduced as a “simulator blueprint” taking $\lambda, \omega$ as input (l212). Furthermore, $T$ denotes a textual task description and a data count (line 183). These ambiguities, combined with very long and nested sentences (e.g., single sentences spanning lines 219-229, 241-247), make the content difficult to follow. 
--> Clearer notation and more concise writing would significantly improve readability and make the paper accessible to a broader audience.

### Questions
This paper lies somewhat outside my core area of expertise (I was likely assigned since I am very familiar with BO), and I found it difficult to grasp the task definition and central methodology fully (see comments below). As a result, I could not thoroughly review the entire paper and appendix, and my feedback focuses on the introduction, technical problem setup, and methodology. While I recognize the potential relevance of the work, I believe the paper requires a major revision. Even for readers outside the immediate subfield (like me), the introduction should more clearly motivate the task/application and clearly define the problem and main ideas to make the paper accessible to a broader audience.

### Soundness
2

### Presentation
1

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
The paper presents SOCIA, a framework for joint structure–parameter co-optimization of
simulators using LLM-based agents. The system orchestrates a pipeline consisting of a
Chain-of-Structure (CoS) generator, a Code Generation Agent (CGA), a Simulation
Execution Agent (SEA) for calibration via Bayesian Optimization (BO) and
Simulation-Based Inference (SBI), and a Feedback Generation Agent (FGA) for structure
refinement. The overall goal is to automate the design, implementation, and calibration of
simulation models, particularly agent-based models, by coupling structural synthesis
with parameter learning and uncertainty estimation. Experiments span three tasks, user
modeling, mask adoption, and personal mobility, evaluating both in-distribution (ID) and
out-of-distribution (OOD) regimes. SOCIA variants outperform or match existing LLM-based
baselines (e.g., GSIM, AI Scientist). The work positions itself as a holistic framework for simulation-based science and a
step towards self-calibrating simulation agents.

### Strengths
- **Ambitious and holistic framing.** SOCIA tackles an important and underexplored
  question: how to couple structural design and parameter inference in simulation-based
  modeling using AI agents. The modular pipeline with separate agents working on different parts of the pipeline is
  well-motivated.
- **Novel integration.** The combination of LLM-driven structure synthesis
  with parameter tuning via both GP-based BO and SBI is a novel and interesting
  direction. The use of SBI for posterior estimation and OOD robustness is particularly
  compelling.
- **Empirical coverage.** The experiments span multiple tasks (ID, OOD, intervention),
  and ablation results illustrate that removing CoS or calibration steps substantially
  degrades performance.
- **Clear motivation for uncertainty-aware calibration.** The contrast between SOCIA-BO
  and SOCIA-SBI provides useful insight into where each approach excels, and supports
  the authors’ argument that uncertainty helps under regime shifts.
- **Potentially impactful vision.** If properly extended and validated, the system could
  form the basis for practical LLM-assisted modeling workflows—especially for
  agent-based social or behavioral simulations.

### Weaknesses
### Conceptual and methodological clarity

- **Implicit focus on agent-based models.** Although the abstract and introduction
  suggest generality (“simulation-based modeling”), the formulation, notation, and
  examples are clearly tailored to ABMs (agents, policies, exogenous inputs). The paper
  should explicitly state this scope and discuss how (or whether) SOCIA can generalize
  to domains with complex scientific simulators (e.g., neuroscience, physics).
- **Technical definitions remain vague.** Core components in Section 3.1–3.2 are
  underdefined: the aggregation operator $A$, the objectives $J_{\text{train}},
  J_{\text{val}}$, and metrics like CRPS appear without formal definitions. The
  relationship between variables $x, y, u, \omega$ is unclear across sections (e.g.,
  the SBI posteriors are defined conditioned on x, not y). 
- **Ambiguity between “agents.”** The term *agent* refers both to simulated entities and
  to LLM components orchestrating the workflow. This dual meaning often causes confusion
  and should be disambiguated consistently (e.g., “simulation agent” vs “AI agent”).

### Empirical evaluation and baselines

- **Missing comparison to standard manual workflows.** Evaluation focuses on LLM-based
  systems (GSIM, AI Scientist) but omits the central baseline of *human-designed
  simulators calibrated with BO or SBI*. This is critical to assessing whether LLM-based
  orchestration meaningfully assists existing modeling pipelines.
- **No expert validation of generated simulators.** Results rely exclusively on
  numerical metrics. It would be informative to include domain-expert evaluation of
  simulator plausibility (e.g., whether generated structures make physical or behavioral
  sense).
- **Limited compute transparency.** The paper does not report simulator call counts,
  wall-clock time, or resource usage for BO and SBI calibration loops. Given that
  structural edits can trigger full re-runs of calibration, compute requirements are
  important for assessing scalability and practical feasibility.
- **Section 4.3 lacks orientation and purpose clarity.** The section introduces
  additional experiments that modify the main tasks, but it lacks a short introductory
  statement explaining *why* these experiments are run (e.g., testing generalization,
  intervention robustness, or pipeline autonomy).
- **Related work coverage is narrow.** The paper cites task-specific SBI/BO works but
  omits standard references such as Cranmer et al. (2019) or general introductory papers
  like Deistler & Boelts (2025), which would help situate the work for non-specialist
  readers.

### Presentation and structure

- The introduction and Section 3 remain abstract; a running example or concrete case
  study would help anchor the reader.  
- Figure 1 is referenced late (p. 6) despite being essential for understanding the
  architecture—earlier guidance to it would improve readability.  
- Minor confusion arises from describing SBI as “simulate–compare–learn”, which reads
  like ABC but actually describes conditional density estimation via NPE.  

### Overall contribution

The main contribution of the paper lies in system integration rather than algorithmic novelty. SOCIA presents an ambitious and well-engineered orchestration of LLM-based structure generation, simulator code synthesis, and calibration via standard BO and SBI components. This is a valuable step toward automating simulation-based modeling workflows, particularly for agent-based settings.

However, the paper does not yet articulate a clear technical algorithm underlying the structure–parameter co-optimization loop. The proposed “structural refinement” process appears to operate heuristically, e.g., driven by diagnostics and LLM proposals, without a formally specified objective, acceptance rule, or convergence behavior. While this is understandable given the non-differentiable nature of the involved components, the lack of algorithmic detail makes it difficult to assess the method’s reliability or theoretical grounding.

In its current form, SOCIA should thus be viewed primarily as a proof-of-concept system demonstrating how modern language models can coordinate established inference and optimization techniques. The idea is promising and potentially impactful, but the technical contribution would be significantly strengthened by a more precise definition of the optimization loop, explicit evaluation traces of structure edits, and a discussion of what (if anything) can be guaranteed or bounded within this framework.

### Questions
1. **Aggregation operator \(A\):** How exactly is \(A\) defined and implemented (line
   188)? Does it perform statistical aggregation or empirical mapping from micro-level
   simulations to macro-level indicators?  
2. **Objectives and metrics:** What is $J$? How is CRPS computed and used in the loss?
   (line 191)  
3. **Notation consistency:** Posteriors are defined as $p(\omega\mid x)$, but $x$ is
   described as system state while training data involve $(y,u)$. Please clarify the
   variable roles.  
4. **Initialization heuristics:** (line 320) What heuristics justify assuming that CoS
   provides a near-optimal starting point? How robust is this when the constructed
   simulator poorly fits the data?  
5. **Calibration strategy:** How does the CGA or SEA decide between running BO and SBI?
   Are both always executed, and if so, how are point and posterior estimates combined?  
6. **Compute scaling:** Each structural edit triggers new simulations and retraining of
   BO/SBI models. What are the wall-clock costs, simulator call budgets, and
   computational resources per task?  
7. **Posterior sampling:** SBI inference reportedly draws 50–100 posterior samples. Why
   such a low number, given that NPE allows essentially free sampling?  
8. **Baselines:** How exactly do the Random Search and LR baselines operate? Which
   simulator are they tuning?

### Soundness
2

### Presentation
2

### Contribution
3
