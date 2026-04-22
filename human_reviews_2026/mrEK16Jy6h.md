# DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Large language model (LLM)–based multi-agent systems are challenging to debug because failures often arise from long, branching interaction traces. The prevailing practice is to leverage LLMs for log-based failure localization, attributing errors to a specific agent and step. However, this paradigm has two key limitations: (i) log-only debugging lacks validation, producing untested hypotheses, and (ii) single-step or single-agent attribution is often ill-posed, as we find that multiple distinct interventions can independently repair the failed task. To address the first limitation, we introduce DoVer, an intervention-driven debugging framework, which augments hypothesis generation with active verification through targeted interventions (e.g., editing messages, altering plans). For the second limitation, rather than evaluating on attribution accuracy, we focus on measuring whether the system resolves the failure or makes quantifiable progress toward task success, reflecting a more outcome-oriented view of debugging. On the datasets derived from GAIA and AssistantBench, DoVer flips 18–28% of failed trials into successes, achieves up to 16% milestone progress, and validates or refutes 30-60% of failure hypotheses. Our findings highlight intervention as a practical mechanism for improving reliability in agentic systems and open opportunities for more robust, scalable debugging methods for LLM-based multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces DoVer, a novel framework for debugging failures in LLM-based multi-agent systems through a process of intervention-driven hypothesis validation. Instead of relying solely on log-based failure attribution, DoVer segments execution logs into trials, generates candidate failure hypotheses, applies targeted interventions, and then re-executes the system to assess if the intervention leads to task recovery or meaningful progress. Through extensive experiments on failure datasets derived from GAIA and AssistantBench, the authors show that DoVer can flip 18–28% of failures into successes and validate or refute 30–60% of failure hypotheses.

### Strengths
1. The authors rigorously analyze the shortcomings of current log-based debugging benchmarks and reveal that many failures involve multiple distinct trials or ambiguous agent interactions, making single-step attribution inherently problematic. This motivates the intervention-based framing in a compelling way.
2. The paper presents a well-thought-out debugging system that is modular and scalable.
3. The authors conduct thorough experiments across three datasets with detailed quantitative and qualitative evaluations. They categorize outcomes (validated, partially validated, refuted, inconclusive), providing a clear framework to assess debugging performance.
4. The visualization system will be highly useful to researchers building or analyzing LLM agent systems. If released, the framework could become a standard tool for debugging such systems in both academia and industry.

### Weaknesses
1. As the authors acknowledged, while the decision to focus on orchestrator-level interventions avoids invasive modifications, it also limits the generality of DoVer. Some failures likely stem from sub-agent capabilities, which the current system cannot directly address. Probing the boundary of sub-agents and determine whether intervention would work could be a potential direction.
2. Although the system is said to be modular, it depends on magnetic-one with checkpointing and re-execution capabilities. This may raise the bar for adoption in settings where such infrastructure is not readily available.
3. The paper did not discuss [1] in depth, which is also an important taxomony for multi-agent system failures. It would be important to discuss how DoVer could be leveraged to address the failure attribution dataset proposed in [1].

[1] Why Do Multi-Agent LLM Systems Fail?

### Questions
Please discuss the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DoVer, an intervention-driven framework for debugging failures in LLM-based multi-agent systems. Instead of only doing log-based failure attribution (e.g., Who&When, AgentTracer), the paper argues that such attribution is often ambiguous, particularly because: 1. Multi-agent systems have multiple trials and branching traces 2. Inter-agent coordination errors exist 3. Ground-truth annotation is noisy and uncertain. To address this, DoVer generates failure hypotheses from logs, actively intervenes (e.g., modifying orchestrator instructions or plans), replays the task from that checkpoint, and evaluates whether the failure is fixed or progress is made. Experiments on AssistantBench and GAIA show DoVer can flip 18–28% failed tasks to success and validate/refute 30–60% hypotheses.

### Strengths
1. Originality: Creative shift from log-only evaluation to active debugging paradigm. 
2. I like the idea of the Insightful finding that ground-truth failure labels are inherently ambiguous in multi-agent setting; this is very valuable observation, e.g., “multiple trials per session and inter-agent misalignment make single-step annotation ill-posed”.
3. Trial segmentation idea and use of checkpoint replay is practical and generalizable.
4. Instead of just pointing at logs, they actually intervene, modify instructions, and rerun. This is very realistic. If you are real agent engineer you know you always try “what if we change instruction here?” And surprisingly they get ~18–28% failure flipped to success. This is quite good considering agent tasks are messy

### Weaknesses
1. Although they use realistic benchmarks, dataset size is not huge (~100 cases, ~200 trials). I worry behavior may differ when: tasks have interactive state, not only web browsing, open source weaker agents used (GPT-4o/5 are strong babysitters)
2. All experiments on Magentic-One + AGDebugger. What about the agents from other frameworks? Would trial segmentation generalize?

### Questions
1. How would DoVer handle very long traces (hundreds of steps)? Is trial segmentation always robust?
2. Another open question is: Can intervention generation be learned? For example, RL finetuning to suggest optimal edits rather than LLM heuristics?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces DoVer, an intervention-driven debugging framework for multi-agent systems. The motivation lies in that 1) existing log-only failure attribution cannot validate its hypothesis and 2) single-step or single-agent attribution (as assessed by existing benchmarks) is often inherently ill-posed because of the multi-trial nature of agentic systems. On constructed benchmark, the proposed DoVer could flip 18-28% of failed trials into successes.

### Strengths
The concrete and detailed motivating analysis of log-based failure attribution in Section 3 is quite informative in exposing how existing benchmarks and metrics could be insufficient and/or inaccurate for evaluating agent failure attribution.

### Weaknesses
My major concern is that there is no baseline in current experiments, making it really difficult to interpret how good the numbers in Table 2 are and the proposed method is.

Specifically, the proposed intervention-based debugging system is essentially doing self-refinement in some sense. Therefore, I believe reasonable baselines could include self-improvement techniques such as Self-Refine (Madaan et al., 2023) and CRITIC (Gou et al., 2024) which provide feedbacks to the agentic system to improve its performance in the next round.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
DoVer is a framework that treats debugging in LLM-based multi-agent systems as an intervention-driven “do-then-verify” process rather than passive log analysis. It systematically tests and validates failure hypotheses through targeted interventions, recovering up to 28% of failed trials and confirming or disproving most hypotheses.

### Strengths
The in-depth examination of the Who&When dataset in Section 3 provides valuable insights into the system’s behavior, helping clarify when and why failures occur.

The framework’s ability to transform failed trials into useful learning opportunities demonstrates its robustness and practical value for improving multi-agent reliability.

### Weaknesses
The evaluation relies solely on a hand-crafted subset of the Who&When dataset, which limits the generalizability of the findings and raises questions about how well DoVer would perform on larger or more diverse real-world datasets.

The framework is evaluated only with GPT-4o as the backend model, which restricts understanding of DoVer’s effectiveness across different LLM architectures and limits claims about its general applicability.

### Questions
Some citation format is incorrect, eg line 151 React citation format is wrong

It would strengthen the work to train and evaluate a smaller, locally hosted model for the task, as this would demonstrate DoVer’s adaptability beyond proprietary large models and improve reproducibility and accessibility for broader research use.

### Soundness
2

### Presentation
3

### Contribution
2
