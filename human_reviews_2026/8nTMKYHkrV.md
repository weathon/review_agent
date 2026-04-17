# STAC: When Innocent Tools Form Dangerous Chains to Jailbreak LLM Agents

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 6

## Abstract
As LLMs advance into autonomous agents with tool-use capabilities, they introduce security challenges that extend beyond traditional content-based LLM safety concerns. This paper introduces Sequential Tool Attack Chaining (STAC), a novel multi-turn attack framework that exploits agent tool use. STAC chains together tool calls that each appear harmless in isolation but, when combined, collectively enable harmful operations that only become apparent at the final execution step. We apply our framework to automatically generate and systematically evaluate 483 STAC cases, featuring 1,352 sets of user-agent-environment interactions and spanning diverse domains, tasks, agent types, and 10 failure modes. Our evaluations show that state-of-the-art LLM agents, including GPT-4.1, are highly vulnerable to STAC, with attack success rates (ASR) exceeding 90\% in most cases. The core design of STAC's automated framework is a closed-loop pipeline that synthesizes executable multi-step tool chains, validates them through in-environment execution, and reverse-engineers stealthy multi-turn prompts that reliably induce agents to execute the verified malicious sequence. We further perform defense analysis and find that existing prompt-based defenses provide limited protection. To address this gap, we propose a new reasoning-driven defense prompt that achieves far stronger protection, cutting ASR by up to 28.8\%. These results highlight a crucial gap: defending tool-enabled agents requires reasoning over entire action sequences and their cumulative effects, rather than evaluating isolated prompts or responses.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces STAC, a multi-turn red-teaming framework that targets tool-enabled LLM agents. Its central contribution is the composition of individually benign-looking tool calls whose cumulative effect becomes harmful only at the final step, i.e., each intermediate action appears innocuous but together they enable a covert attack. While the concept is novel and compelling, the manuscript currently lacks concrete empirical evidence or real-world incident analysis demonstrating that such staged, cumulative attacks are feasible or observed in practice.

### Strengths
- Introduces STAC as a distinct, sequence‑level vulnerability for tool‑enabled agents and operationalizes it via a closed‑loop generation–verification–prompting pipeline
- Comparisons with single‑turn and adapted multi‑turn baselines show large margins

### Weaknesses
- The STAC pipeline assumes near-white-box access to precise tool schemas, parameters, and environment state to construct and verify attack chains, an assumption that is often unrealistic in real-world agent deployments. 
- The paper lacks a formal threat model and does not specify attacker capabilities, goals, or assumptions, undermining the rigor expected of AI-security research. Explicitly state attacker knowledge (tool schemas, environment state) given that the Generator/Verifier are granted such access; evaluate STAC when tool metadata is partially hidden or noisy.
- The entire STAC pipeline heavily depends on large LLMs for generation, verification, planning, and judging, yet the paper does not justify why alternative non-LLM or hybrid approaches (e.g., symbolic reasoning, static verification, or rule-based execution tracing) would be infeasible. Moreover, the authors provide no discussion of how hallucinations or false positives from these LLM components are mitigated, which undermines the methodological rigor and reproducibility of the proposed framework.
- The Judge is GPT‑4.1, the same model family used for generation/verification/planning. This risks correlated failure modes and optimistic ASR/PH/RR estimates.
- Experiments are in SHADE‑Arena/ASB sandboxes. Add results on a public agent stack (e.g., OS/desktop/web workflows) with realistic tools and rate limits to assess generalization.
- Only prompt‑based defenses are tested; discuss or prototype action‑level guards (e.g., precondition checkers, state‑diff risk analysis, sequence‑level monitors).

### Questions
- Threat Model Clarification: What are the assumed attacker capabilities and knowledge? Does STAC require white-box access to tool schemas, parameters, and environment state? Please evaluate whether the attack still succeeds when tool metadata is partially hidden or perturbed.

- LLM Dependence and Hallucination Control: Given that all components (Generator, Verifier, Planner, Judge) rely on GPT-4.1, how do you ensure consistency, reduce hallucinations or false positives, and verify that results are not artifacts of model bias?

- Model Independence of Evaluation: Since the Judge shares the same model family as other components, did you test cross-model or human-verified evaluation to confirm ASR/PH/RR robustness?

- Practical Deployment: How practical is STAC under realistic conditions where agents operate with limited tool visibility, API rate limits, or partially observable states (e.g., OS-level or web-based agents)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Sequential Tool Attack Chaining (STAC), a multi-turn attack framework that exposes security vulnerabilities in LLM-based autonomous agents with tool-use capabilities. The key insight is that individual tool calls that appear benign in isolation can be chained together across multiple turns to execute harmful operations that only become evident at the final step. Through automated generation and systematic evaluation of 483 STAC cases encompassing 1,352 user-agent-environment interactions across diverse domains and 10 failure modes, the researchers demonstrate that state-of-the-art agents like GPT-4.1 are highly susceptible to these attacks, with success rates exceeding 90%.

### Strengths
1. The experimental design is notably thorough. The authors generated 483 distinct STAC cases encompassing 1,352 interaction sequences, covering multiple dimensions including diverse domains, task types, agent architectures, and 10 different failure modes. 

2. The paper contributes by constructing a dataset of multi-turn attack scenarios specifically designed for tool-enabled agents. Unlike datasets focused on single-turn jailbreaks or standard adversarial examples, this collection captures sequential attack chains where harm emerges through composition rather than individual actions

### Weaknesses
My main concern is about the novelty and the threat model. 

1. The paper's fundamental threat model appears flawed or insufficiently justified. In Figure 1, it seems the legitimate user is the attacker, which raises the question: if users intentionally issue these commands to their own agents, isn't the system simply following instructions as designed? For example, Turn 3 shows file deletion—but if a user wants to delete their files, the agent should comply. The paper fails to clearly establish who the adversary is, what security boundaries are being violated, or why benign users would accidentally trigger these chains. Without distinguishing between malicious injection by third parties versus intentional user actions, the high attack success rates (>90%) may simply demonstrate that agents are good at following multi-step instructions—a feature, not a vulnerability. The work lacks a clear attacker model specifying capabilities and security assumptions.

2. The core contribution appears incremental rather than novel. Multi-turn jailbreaking attacks have been extensively studied; this work primarily shifts focus from "harmful content generation" to "harmful environmental actions," which seems like a natural extension of known vulnerabilities rather than a fundamental innovation. Also, the dataset construction relies heavily on two previous works, further questioning originality. If the threat model is essentially the same as prior jailbreaking research (inducing models to perform prohibited actions), then chaining tool calls is merely a different manifestation of existing attack vectors. 

3. Additionally, while the paper demonstrates that a reasoning-driven defense prompt reduces ASR by 28.8%, it would strengthen the work to explore whether this represents a robust long-term solution or if adaptive attacks could circumvent it. The paper would benefit from evaluating adaptive attacks designed specifically to break the proposed defense.

### Questions
I think the paper could be more impactful by exploring how STAC performs against indirect prompt injection attacks. These are cases where attackers manipulate external resources (like emails, web pages, or documents) that agents process, rather than directly controlling the prompts themselves. This type of attack seems particularly relevant for production systems since it represents a more realistic security boundary that adversaries would actually encounter.
To be clear, do you think STAC is also effective against indirect prompt injection?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Sequential Tool Attack Chaining (STAC), a novel framework for studying security vulnerabilities in tool-enabled LLM agents. Unlike prior single-turn jailbreaks or prompt injections, STAC orchestrates sequences of individually benign tool calls that cumulatively lead to harmful outcomes. For example, compressing → deleting → bulk-cleaning files can collectively destroy critical data while appearing harmless in each step.

The authors design an automated pipeline that (1) generates verified tool chains via environment interaction, (2) reverse-engineers corresponding benign prompts, and (3) executes adaptive multi-turn attacks in realistic agentic settings such as SHADE-Arena and Agent-SafetyBench. The resulting STAC benchmark includes 483 validated attack trajectories spanning 10 failure modes and 1,352 user-agent-environment interactions.

Empirical results show that even frontier models (e.g., GPT-4.1, Llama 3.1, Qwen3-32B) exhibit attack success rates (ASR) above 90%, revealing a systemic vulnerability in current tool-based agent architectures. The authors also evaluate existing prompt-based defenses and propose a reasoning-based defense prompt that reduces ASR by up to 28.8%. The study argues that defending LLM agents requires reasoning over entire action sequences, not individual prompts.

### Strengths
The paper’s major strengths include:
1. Novel problem formulation: Identifies a new class of vulnerabilities unique to multi-turn tool-use sequences rather than individual requests.
2. Automated and reproducible pipeline: Introduces a closed-loop system for generating, validating, and testing attacks without manual curation.
3. Strong empirical evaluation: Tests across eight models, multiple environments, and failure modes, showing consistent and alarming vulnerabilities.
4. Meaningful defense proposal: The reasoning-based defense prompt is conceptually sound and shows significant mitigation without retraining.

### Weaknesses
While technically robust, several aspects limit the completeness of the contribution:

1. Ambiguity in novelty positioning relative to prior work.
While STAC introduces the concept of Sequential Tool Attack Chaining, the underlying behavioral bias—that models become more permissive when malicious intent is framed through code or tool interfaces—has been previously documented. The core advancement here lies in extending that phenomenon to multi-turn, executable, and agentic contexts, yet this distinction is not emphasized clearly. The paper would benefit from a sharper articulation of what conceptual or methodological gap STAC fills beyond prior tool-primed or function-call jailbreak research, making its incremental contribution explicit.

2. Limited mechanistic explanation of failure.
The paper convincingly demonstrates that sequentially benign tool calls can cumulatively yield harmful outcomes, but it provides little insight into why models fail to detect these aggregated risks. There is no analysis of whether the problem stems from reasoning myopia, context-window truncation, or loss of refusal memory across planning steps. Including introspective traces, intermediate reasoning logs, or causal attributions would elevate the contribution from empirical observation to mechanistic understanding, aligning better with ICLR’s expectation for scientific insight.

3. Overgeneralized claims of universality.
Although the results are impressive, the paper sometimes implies that STAC-style vulnerabilities generalize to all agentic deployments. Given that all experiments occur in controlled Python-based environments, this generalization may be overstated. The authors should more precisely define the boundary conditions under which STAC attacks apply, e.g., when agents have composable tools without centralized oversight or semantic validation. This would prevent overextension of the conclusions and help practitioners understand when such failures are most likely.

4. Lack of cross-model interpretive analysis.
While multiple model families are evaluated, the study largely reports aggregate metrics without examining why certain models (e.g., GPT-4.1 vs. Llama-3.1) differ in vulnerability or refusal persistence. Understanding whether these gaps stem from training data, architecture, or safety fine-tuning could yield deeper insight into agentic robustness. Analyzing a few representative trajectories across models, highlighting distinct reasoning or memory patterns, would transform the results from descriptive benchmarking into comparative scientific analysis.

### Questions
1. How does STAC extend or differ from prior works? Please clarify what is genuinely new in the sequential tool-chaining framework.
2. Why do models fail to detect cumulative harm across tool calls? Is this mainly due to loss of refusal memory, lack of global reasoning, or limitations in safety alignment?
3. The reasoning-based defense quickly loses effectiveness over turns. Do you have any evidence or hypothesis explaining this degradation?
4. How do you verify that a “successful” execution actually represents a harmful or policy-violating action rather than a benign result?
5. What safeguards or access policies will be in place when releasing the STAC benchmark to prevent dual-use misuse?

### Soundness
2

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
3

### Summary
The paper introduces STAC (Sequential Tool Attack Chaining), an automated, closed-loop framework that chains together harmless tool calls, which collectively enable harmful operations at the final execution step. Experiments show that on SHADE-Arena and Agent-SafetyBench, STAC achieves very high ASR. A reasoning-based defense prompt reduces ASR by up to 28.8% on the first execution turn, but protection degrades over turns.

### Strengths
1. The framework is clearly formalized. Generator/Verifier/Prompt Writer/Planner/Judge Roles are explicit and reproducible。

2. The proposed framework performs effectively across 4 complex SHADE-Arena environments (banking, travel, workspace, spam filtering) plus 62 Agent-SafetyBench environments.

3. The threat is operational: not just generating unsafe text, but orchestrating tool use to cause environmental harm.

4. In addition to proposing the attack framework, the paper also includes a defense analysis.

### Weaknesses
1. The attack baselines are limited. There are many multi-turn jailbreakers and agent-focused attacks that are not included. The paper adapts only X-Teaming. The comparison with X-Teaming is also potentially unfair. The adapted X-Teaming setup allows the agent to take only one tool-call turn per user turn, “due to system design limitations,” which likely handicaps it relative to STAC’s adaptive multi-turn execution. Moreover, hiding intentions in a multi-turn attack to improve ASR is not entirely new. While the paper focuses on tool-using settings, it is crucial to demonstrate the method's advantage over other baselines across various domains.

2. Only prompt-based defenses are evaluated. It seems that defenses such as tool-policy or allow/deny lists can be effective against at least some of the attacks (e.g., the example in Figure 3).

3. GPT-4.1 is used as Generator, Verifier, Planner, and Judge, which might raise concerns about single-model dependence in the attack pipeline and evaluation

### Questions
1. How does ASR evolve beyond 3 user turns? Is it possible to report a turn-by-turn curve until a fixed horizon (e.g., 5–10 turns) to illustrate persistence effects?

2. What happens when tools themselves enforce policies (e.g., “delete_file” requires a capability token or asks for user confirmation if the path is critical)?

### Soundness
3

### Presentation
3

### Contribution
3
