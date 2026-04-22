# Tiered Agentic Oversight: A Hierarchical Multi-Agent System for Healthcare Safety

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Large language models (LLMs) deployed as agents introduce significant safety risks in clinical settings due to their potential for error and single points of failure. We introduce **Tiered Agentic Oversight (TAO)**, a hierarchical multi-agent system that enhances AI safety through layered, automated supervision. Inspired by clinical hierarchies (e.g., nurse-physician-specialist) in hospital, TAO routes tasks to specialized agents based on complexity, creating a robust safety framework through automated inter- and intra-tier communication and role-playing. Crucially, this hierarchical structure functions as an effective error-correction mechanism, absorbing up to 24\% of individual agent erros before they can compound. Our experiments reveal TAO outperforms single-agent and other multi-agent systems on 4 out of 5 healthcare safety benchmarks, with up to an 8.2\% improvement. Ablation studies confirm key design principles of the system: (i) its adaptive architecture is over 3\% safer than static, single-tier configurations, and (ii) its lower tiers are indispensable, as their removal causes the most significant degradation in overall safety. Finally, we validated the system's synergy with human doctors in a user study where a physician, acting as the highest tier agent, provided corrective feedback that improved medical triage accuracy from 40\% to 60\%. Project Page: https://tiered-agentic-oversight.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Tiered Agentic Oversight (TAO), a hierarchical multi-agent framework for enhancing safety in clinical applications of LLM agents. Inspired by clinical team hierarchy, TAO dynamically routes cases by complexity through agents at multiple expertise tiers, incorporating both automated intra/inter-tier oversight and potential human-in-the-loop escalation. Experimental evaluation spans 5 healthcare safety benchmarks, supported by ablation analyses of tier configuration and agent attribution, as well as a user study with clinicians assessing the system's escalation and safety effectiveness. The approach aims to absorb errors and increase reliability relative to single-agent or flat multi-agent baselines.

### Strengths
1. The TAO framework is modeled after clinical hierarchies and operationalizes a clear, stepwise escalation and oversight workflow.
2. TAO outperforms the strongest single-agent and multi-agent baselines on 4 out of 5 benchmarks.
3. The clinician-in-the-loop study shows the system can escalate appropriately for human feedback, with doctor input improving medical triage accuracy from 40% to 60%.

### Weaknesses
1. The paper claims agent specialization through prompt engineering. However, there is no deep quantification or qualitative analysis on whether these system prompts actually induce meaningful differences in domain reasoning, especially since all agents run atop general-purpose LLMs without explicit medical fine-tuning.
2. The Agent Router plays a pivotal role in tier assignment, but its design and robustness to ambiguous or novel input is underdiscussed. There is little detail on the training, algorithmic choices, or empirical stress testing for error tolerance in case complexity inference.
3. On MedSafetyBench, the gains over LLM-DEBATE and other competitive baselines are minimal. TAO's improved safety score translates to small absolute differences in some regimes, particularly when considering much higher cost. The system's superior safety may not always justify the resource overhead, especially as medical settings often operate under hard resource constraints.
4. There is insufficient analysis of failure cases, such as circumstances where TAO's oversight may reinforce biases or miss rare "unknown unknowns," especially in adversarial settings or edge patients.

### Questions
1. Could you elaborate on how the Agent Recruiter and the Router are implemented? Are they rule-based, invoked through another LLM, or specifically trained? How do they assess the complexity of a case in order to determine the appropriate hierarchical strategy?
2. What is the system's degree of robustness to routing errors? For example, if the Router intentionally makes suboptimal decisions, such as assigning an evidently complex case only to Tier 1, or recruiting irrelevant experts to what extent would the overall safety performance of the system decrease? This is crucial for understanding the reliability of the TAO framework when confronted with ambiguous or atypical cases.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors introduce Tiered Agentic Oversight (TAO) which is a hierarchical multi-agent framework for improving LLM safety in healthcare inspired by clinical hierarchies (e.g., nurse-physician-specialist). The agent recruiter selects specialized LLM agents (e.g., general physician, cardiologist) based on case context, and agent router assigns them to three tiers by complexity, and mechanisms for intra/inter-tier collaboration with escalation for high-risk cases. Authors evaluated their multi-agent framework on five different healthcare safety benchmarks and according to authors claim, it outperforms single-agent and multi-agent baselines. Authors also conducted detailed additional ablation studies to understand the key components of their framework together with actual human studies with clinicians.

### Strengths
- Authors focus on a timely and novel topic by proposing a medical safety-focused multi-agent framework that tries to address a critical gap in healthcare safety.
- The main results of the paper in Table 2 demonstrate strong performance of TAO, where it outperforms other ingle-agent, multi-agent, and adaptive systems across five different benchmarks.
- Authors also conduct comprehensive ablation studies including but not restricted to different adversarial setups, attribution analysis, tier configurations, and error propagation analysis, which can provide valuable insights for future research. Moreover, inclusion of a real clinician study strengthens the practical validity of the framework.

### Weaknesses
- The main limitation of the proposed framework the practical usability in the wild due to its high computational cost. As shown in Table 2, TAO results with the highest cost, sometimes nearly double the previous state-of-the-art. Since paper presents a more application-oriented solution with MAS rather than a theoretical or algorithmic improvements, cost is a critical concern. Thus, this raises two questions: (1) is the slight accuracy gain of up to ~4% (sometimes 0.04%, 0.7%, etc.) meaningful enough to justify doubling the cost in downstream task? (2) Could we implement a multi-agent setup using second-top models from each benchmark with routing, which would be far cheaper while achieving accuracy much closer to TAO; if so, why should one choose TAO?
- Secondly, the experimental setup is narrow, where authors only evaluate large closed-source models like Gemini and o3. This makes it difficult to assess TAO's effectiveness with open-source LLMs, also in varying scales. How would Table 2 change using Qwen or Llama-based models in both large and small scales?
- Minor comment: While reading the paper, especially the introduction, repeated words (also present in the rest of the paper) are noticed, along with unnecessary comma usage with very long sentences. These elements made the reading experience a bit challenging for reviewer side. Could authors disclose whether LLMs were used in writing, as the current manuscript does not mention this?

### Questions
1. How would the results in Table 2 change if Gemini were replaced with open-source models such as Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Llama-3.3-70B-Instruct, Qwen-2.5-72B-Instruct, or Qwen-3-235B-A22B-Instruct-2507?
2. Could the authors prepare a scatter plot version of Table 2, where the x-axis represents cost and the y-axis represents accuracy, to better illustrate the trade-off between performance and cost?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents Tiered Agentic Oversight, a hierarchical multi-agent framework designed to enhance LLM safety in healthcare. Inspired by clinical team structures, TAO assigns agents to specialized tiers and enables dynamic task routing, collaboration, and escalation based on task complexity.
It introduces key components such as an Agent Recruiter and Router, structured communication protocols, and a Final Decision Agent for synthesis. Experiments suggest that TAO improves reliability and reduces individual agent errors compared to baseline systems. A user study also indicates strong synergy between human and AI oversight.

### Strengths
1. The hierarchical structure with adaptive escalation is a thoughtful adaptation of clinical workflows to multi-agent AI. It provides a clear and structured mechanism for error correction and oversight.
2. The paper includes experiments across multiple benchmarks that assess safety, accuracy, and ethical alignment. 
3. The clinician-in-the-loop study adds practical credibility, showing that the framework can work effectively with human oversight in real-world settings.

### Weaknesses
1. While the hierarchical agent design is interesting, it largely builds on existing multi-agent paradigms such as debate or voting frameworks. The paper does not clearly articulate how its approach advances beyond prior work or contributes new theoretical insights.
2. The experiments are mainly healthcare-focused and lack sufficient details for reproducibility. Key materials such as code are not provided, and important aspects like statistical significance are missing. The user study is small in scale and not thoroughly described.
3. Assertions like "superior performance in 4 out of 5 benchmarks" ignore modest gains (e.g., 0.04 on MedSafetyBench) and potential overfitting to Gemini models; no cross-domain tests beyond healthcare.
4. Descriptions of agent prompts and escalation flags are vague, making it hard to assess correctness. Section 2.2 mentions "boolean escalation flag" from reasoning, but no equations or pseudocode formalize the decision process.

### Questions
1. Why were only Gemini and o3 models tested extensively, and not open-source alternatives? Does this limit generalization?
2. In the user study, why the 40% to 60% improvement? were scenarios balanced for complexity, and what was the baseline doctor-alone accuracy

### Soundness
3

### Presentation
3

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
The paper proposes a structured framework to improve the safety and reliability of AI agents in clinical environments. Drawing inspiration from medical team hierarchies, TAO organizes LLM agents into tiers that correspond to different levels of expertise, such as triage, specialized review, and expert consultation, to perform adaptive and multi-layered oversight. The system routes tasks based on case complexity, allowing agents to collaborate, escalate cases, and involve human clinicians when necessary. Experiments across five healthcare safety benchmarks show that TAO surpasses both existing single-agent and multi-agent baselines on four of them, achieving up to an 8.2% safety improvement. Ablation studies confirm that lower tiers play a critical role in maintaining safety, and the system can absorb up to 24% of individual agent errors. A clinician-in-the-loop user study further demonstrates TAO’s practical value, where human feedback improved medical triage accuracy from 40% to 60%.

### Strengths
1. The TAO framework, inspired by clinical hierarchies, presents a clear and innovative conceptual contribution to AI safety.
2. It demonstrates strong empirical performance, outperforming both single-agent and multi-agent baselines on most healthcare safety benchmarks.

### Weaknesses
1. The paper lacks large-scale experiments, such as scaling to more agents or additional tiers.
2. The framework incurs higher computational costs, as indicated in Table 2.
3. The central routing mechanism may become a bottleneck, and the analysis of its robustness is limited.
4. Defining clear scopes for each tier is challenging, which could lead to overlapping responsibilities or gaps between agents.

### Questions
1. What factors contribute to the higher computational cost compared to other multi-agent baselines?
2. Can the proposed framework generalize effectively to domains beyond healthcare?

### Soundness
3

### Presentation
3

### Contribution
2
