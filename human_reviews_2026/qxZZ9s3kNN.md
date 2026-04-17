# Agentic Surrogates: Automating Proxy Models of Simulators with Compute Aware Intelligence

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Proxy (surrogate) models are indispensable for accelerating scientific computation, yet creating them remains a manual, sample-inefficient, and non-reproducible process—especially when simulators are costly and constrained by physics. We present a domain-agnostic agentic framework that automates end-to-end surrogate construction for high-fidelity simulators. At its core, a reasoning engine orchestrates the entire loop, encompassing space-filling candidate generation, batched simulator invocation, warm-start retraining, uncertainty/calibration, and dynamic acquisition switching to achieve user-specified accuracy with minimal wall-clock time and simulator calls. Crucially, this reasoning engine also intelligently constructs and orchestrates an adaptive ensemble of models, where individual components or techniques are dynamically selected based on the specific characteristics of different input and output combinations. For instance, if the agent discerns simpler dependencies, such as linear relationships or reliance on fewer inputs within a particular output ensemble, it can deploy models with reduced complexity or optimized for computational efficiency. Conversely, for complex, highly non-linear, or high-dimensional problems, it will automatically integrate and leverage more sophisticated architectures within this ensemble. This approach ensures that the overall surrogate is a finely tuned composition of expert models, each optimally suited to distinct aspects of the simulator's behavior. The controller manages acquisition rules as a portfolio of experts (residual-error, MC-Dropout variance, EI/EGO, hybrid, randomized) and optimizes a compute-aware objective (error reduction per minute), with extensible support for multi-fidelity scheduling. The framework remains neutral to underlying architecture (e.g., ANNs, PINNs, operator networks such as FNO/DeepONet) and integrates physics-aware stopping alongside MC-Dropout and conformal prediction for calibrated uncertainty.
We evaluate this approach on energy-related scientific modeling problems, including industrial-style flow/process simulators and a PDE proxy. Our findings reveal consistent and substantial performance improvements compared to fixed strategies, demonstrating notable advancements in both sample efficiency and time-to-target accuracy. Prior pilots in oil & gas demonstrated substantially reduced predictive errors compared to the best single acquisition policy while converging faster; here we generalize the method beyond any single domain, preserving these gains under diverse physics and cost profiles. A key novelty lies in the central contribution of introducing compute-aware regret guarantees, which are emphasized up front.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper aims to automate the creation of surrogate (proxy) models for complex physics-based simulations using an active learning framework. The setup assumes access to a high-fidelity simulator that is expensive to run, and the goal is to approximate it efficiently with learned surrogates.

The authors propose a mixture-of-experts framework for active learning, where several surrogate models and acquisition strategies are combined adaptively. The combination is governed by a reward-based weighting mechanism called Hedge-Exp3 rules, which balance the contribution of each expert according to their performance.

They also introduce both a cost model and a reward function to guide the optimization:
- The reward is defined as the improvement in residual error divided by the computational time required.
- The cost function aims to minimize the total time for data acquisition, simulation, and training, subject to constraints on validation error, total simulation budget, and physical feasibility of the resulting surrogate.

The method is applied to three oil and gas engineering problems:
- Oil and gas well network modeling,
- Gas processing plant optimization, and
- CO₂ storage simulation.

Across these applications, the proposed method shows improved accuracy, data efficiency (fewer data points required), and faster convergence compared to baseline approaches.

### Strengths
The paper tackles an important problem, automating surrogate modeling, where computational cost is often a bottleneck. Introducing a mixture-of-experts framework in this context is an appealing and conceptually sound idea, as it allows for adaptive combination of different surrogate and acquisition strategies. The use of active learning to guide model refinement is also well-motivated and relevant for this class of applications.

A notable strength lies in the practical relevance of the chosen case studies. The three applications are realistic and industrially significant. This grounding in concrete engineering problems makes the work compellin, helps demonstrate potential real-world value and adds credibility.

### Weaknesses
The main weakness of the paper lies in its poor writing and presentation quality, which makes it extremely difficult to understand the actual contributions and methodology. Key concepts are often introduced abruptly, without proper motivation, definitions, or notation. This affects the overall readability and credibility of the work.

1. Lack of Clarity and Structure
    - The paper frequently introduces technical components, such as the cost model, portfolio acquisition strategy, or active learning setup, without explanation or contextualization.
    - Important terms like active learning, Hedge-Exp3, or even core notations (e.g., ε, ρ) are never defined or discussed before being used.
    - There is no clear description of how iterations are performed or what the stopping criterion is.
    - Several sections (e.g., “metrics” section 6.1) are underdeveloped, some consisting of a single sentence, while key elements (like the oil and gas problem definitions) are relegated to the appendix.

2. Poor Methodological Explanation
    - The acquisition strategies are not well explained. For instance, the “residual predictor” and “residual top-k” methods are mentioned without detailing how they operate.
    - It is unclear from the main text how candidate points are generated at each iteration, and this information only appears in the appendix.
    - The role of the reward and cost model is not clearly established in the algorithmic flow.
    - The Hedge-Exp3 weighting mechanism is introduced but never properly explained — the reader is left unsure how the weights are applied (e.g., sampling vs. linear combination).
    - The mixture-of-experts setup itself is opaque: it’s not clear whether surrogate models are combined, selected, or average. Same for acquisition experts. 
    - The function $g$ similarly is never explained

3. Theoretical Weaknesses
    - The theoretical section lacks substance. The proofs are largely hand-waved with statements such as “it follows from [reference or standard result]”, which does not constitute a meaningful theoretical contribution.
    - If the paper claims a “theoretical analysis,” it must present at least minimal independent reasoning or adaptation to the proposed framework.
   - The lack of rigorous definitions used in the paper also prevents one from understanding or assessing the validity of the results

4. Poor Notation and Technical Presentation
    - The notation is inconsistent and poorly introduced. Variables such as ξ appear once (in the surrogate model formulation) but are never used again.
    - Even basic quantities (e.g., MAE) are not defined before appearing in the text.
    - The use of terms like “KK-style diagnostic” or “conformalized” assumes prior familiarity with unspecified methods.

5. Weak Experimental Presentation
    - The experiments are poorly explained. Tables and figures are shown without context (e.g., Figure 1’s learning curve lacks experiment identification).
    - Key results (final MAE, iteration counts) are mentioned without specifying which experiment they correspond to.
    - The claim that the model “converges in fewer iterations” is unsupported, especially since the plotted curves have not actually converged. A higher convergence rate would be more accurate.
    - Important details on the industrial problems (oil and gas network, gas plant, CO₂ storage) are hidden in the appendix, with no in-text explanation.

6. Misleading Framing and Overstatements
    - The paper title and sections refer to “LLM agents” and “Agentic surrogates,” yet no large language model or agent mechanism is actually present in the work.
    - This naming is misleading and distracts from the genuine surrogate modeling focus.
    - Similarly, the “theoretical analysis” claim is overstated given the lack of original proof content.

7. Irrelevant or Superficial Sections
    - The “broader impact / ethics / governance” section is unnecessary for this technical paper and offers only generic statements (e.g., environmental benefits from lower computational cost). These read as filler and dilute the technical message.

8. Overall Impression
    - The core idea has potential, but the execution and exposition are highly problematic. The paper assumes the reader already knows the authors’ specific variant of active learning and surrogate modeling.
    - With more careful definitions, clearer structure, and rigorous explanations, the work could become readable and meaningful. As it stands, it’s difficult to reconstruct the exact methodology or even reproduce the results.

### Questions
- Given all the interrogations in the weaknesses, could you give a succint explanation of the method, that does not rely on overly specific prior knowledge? 
- How are the results computed depending on the problem? 
- What makes the work agentic?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
- The paper is motivated by efficiently creating proxy surrogate models for scientific simulation, especially since traditional methods are cumbersome (e.g., determining sampling regions, determining strategies).
- To tackle this, the paper casts the surrogate creation as a adaptive controller problem: an online learnt policy dynamically selects sampling strategies and model architectures to optimize accuracy and efficiency.
- The controller is trained by solving a cost-minimization problem that accounts of total wall-clock latency (including acquisition, simulation, and training)
- The paper also provides theoretical guarantees for adaptive acquisition, including regret bounds and sample complexity.
- For experimental evaluation, the paper reports results in three tasks involving PDE solving.

### Strengths
1. The adaptive controller angle is novel in this scenario: while online/active learning has received some attention in simulation-based discovery, to my knowledge, this paper is the first to cast it as a control problem (involving using wall-clock time as an objective)
2. The paper also formal guarantees (regret bounds, sample complexity) for the adaptive strategy.

### Weaknesses
**1. Experimental Setup and Reporting**
- I have two subconcerns here.
- **Choice of tasks/experiments**
    - Context: The paper reports results on three tasks: Well network, Gas Plant, and CO2 storage.
    - The choices of these tasks appears somewhat unconventional and the paper would benefit from more justification.
    - In contrast, priors works, many which the paper cites too (e.g., PINNsAgent Wuwu et al. 2025, Self-Debug Chen et al., 2023) focus on more well-studied scenarios (e.g., 2D Heat, ND Poisson).
    - Because the paper strays away from some of these standard settings, make a fair comparison of experimental results is difficult.
- **Iteration**
    - The paper reports the main evaluation (Fig. 1) with iterations as a criteria. 
    - However, because at each iteration, I believe acquisition time varies per method, it is unclear whether this style of reporting is fair.
    - (minor) Furthermore, in Table 2, it is also somewhat unclear why the wall-clock time of "random" acquisition is incurs more wall-clock time.

**2. Bibliography: Critical Mistakes**
- Many important references in the references section are incorrect. For instance, 
    - PINNsAgent: cited as by Chen et al., but it is originally by Wuwu et al.
    - "Fourier Neural Operator based surrogates for $ CO_2 $ storage in realistic geologies": cited as by Baker et al., but originally by Chandra et al.
    - Could not find "Multi-agent llm framework for physics-informed neural network surrogates" published at NeurIPS 2025

### Questions
Suggestions:
1. Please justify the choice of the three tasks in the experimental results section. Is there relevant literature that also evaluate on these three tasks?
2. Please fix references

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
3

### Summary
The paper proposes an LLM based agent that orchestrates the acquisition of data from a simulator to train a set of surrogates to minimize a loss while keeping computational cost as low as possible. At each iteration, the agent generates candidates for new data which are evaluated by acquisition experts, then agent then selects an expert based on dynamic weights it has learned from a cost based reward. The agent then calls a simulator to generate new data for training based on the candidates selected followed by retraining the neural surrogates with this new data.

### Strengths
- The idea of acquiring data from a simulator to train a surrogate model in an automated fashion using an LLM agent is interesting.

### Weaknesses
- It seems like the authors have not released the code or data for their experiments.
- The presentation of the paper is unrefined. The references to the appendix are not linked. There is only one figure in the main text and there is no figure showing the overview of how their agent works.
- The paper lacks clarity and requires more explanation of how the method works.

### Questions
- Does the cost function you propose include the time to query the LLM agent?
- How does the safety filtering work when you write "reject unsafe or OOD proposals" in the algorithm? Is it just a static check or something that a human checks?
- Have you released your code and datasets?
- Is your method really "domain-agnostic"? Doesn't it require the specific simulator to be registered with the LLM as well as the candidate parameters to be predefined in some way?
- How often was the request_human_review() call made by your agent? Did the agent require human intervention?

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
4

### Summary
The paper proposes automated surrogate design and experimentation using an agentic framework. Specifically, the surrogate creation and experimentation pipeline is divided into surrogate design, data acquisition and surrogate training. The paper investigates numerous types of surrogate architectures including neural operators, physics informed neural networks and artificial neural networks as well as many popular data acquisition strategies. Experimental results demonstrate that the proposed pipeline leads to improved model performance and well calibrated models.

### Strengths
1. The paper is a good start to an important problem of automating the design and development of neural surrogates and subsequent experimentation in scientific machine learning pipelines.

2. The proposed structured prompt (with chain-of-thought questions) along with the various feedback mechanisms including the multi-objective cost functions are well defined scaffolding that help the convergence of agentic pipelines. Overall, the costs and other feedback signals to the agentic pipeline seem well thoughout.

### Weaknesses
1. Some facets of the pipeline are unclear like how the multi-fildeity cost is included as a penalty to the agentic pipeline. 

2. Other facets like choices of neural surrogates seem somewhat arbitrary. For example, recently transformer architectures have been shown to outperform many surrogates (including UNet, ResNet) on PDE modeling tasks, yet they have not been included.

### Questions
1. What aspect of the proposed pipeline actually conducts multi-fidelity acquisiton and how is the fidelity cost included into the overall pipeline?

2. The proposed pipeline employs multiple types of surrogates, each of which have their own idiosyncrasies and failure modes (e.g., PINNs fail catastrophically in stiff PDE domains), how can the current pipeline address such failure modes? Is it susceptible to these pitfalls or would it alleviate it somehow by realizing that a different surrogate would need to be invoked to avoid any training problems?

3. Why have transformers, UNets, ResNets not been considered as neural surrogate options in the "portfolio of models"? Can the proposed framework function with these architectures out of the box? 

4.  How would expert-designed standard surrogates (e.g., PINN, ANN or FNO / DeepONet) trained with the same volume of training data as the agentic pipeline, perform on the test set? Has this comparison been included in the paper?

### Soundness
3

### Presentation
2

### Contribution
3
