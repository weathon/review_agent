# Optimizing Agent Planning for Security and Autonomy

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Indirect prompt injection attacks threaten AI agents that execute consequential actions, motivating deterministic system-level defenses. Such defenses can provably block unsafe actions by enforcing confidentiality and integrity policies, but currently appear costly: they reduce task completion rates and increase token usage compared to probabilistic defenses. We argue that existing evaluations miss a key benefit of system-level defenses: reduced reliance on human oversight. We introduce autonomy metrics to quantify this benefit: the fraction of consequential actions an agent can execute without human-in-the-loop (HITL) approval while preserving security. To increase autonomy, we design a security-aware agent that (i) introduces richer HITL interactions, and (ii) explicitly plans for both task progress and policy compliance. We implement this agent design atop an existing information-flow control defense against prompt injection and evaluate it on the AgentDojo and WASP benchmarks. Experiments show that this approach yields higher autonomy without sacrificing utility (task completion).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper discusses optimizing AI agent planning for security and autonomy by introducing autonomy metrics and proposing a security-aware agent called PRUDENTIA. PRUDENTIA improves autonomy by making the agent aware of security policies and planning for both task progress and policy compliance. The agent is designed to minimize human-in-the-loop (HITL) interventions (to generate security policies) while maintaining task completion rates. Experimental results show that PRUDENTIA shows noticeable improvement in security at a cost of having more interaction with human.

### Strengths
- The paper introduces new autonomy metrics (HITL load and TCR@k) that capture the benefits of deterministic defenses for AI agents, providing a more comprehensive evaluation framework. The metric is quite straightforward: it measures the proportion of tasks completed with k human interaction while avoiding security issues.
- The paper presents a thorough evaluation of PRUDENTIA on multiple benchmarks (AgentDojo and WASP), demonstrating its effectiveness in reducing HITL load and improving task completion rates across various models and scenarios.

### Weaknesses
- The paper assumes that a human will always be available to provide endorsements or approvals when necessary, which is unrealistic in many real-world deployments. Imho this assumption undermines the purpose of developing autonomous AI agents, as it relies heavily on human intervention. Moreover, human involvement not only introduces overhead but also the potential for human error. More critically, the emergence of more sophisticated attacks could exploit this human-in-the-loop (HITL) mechanism, potentially forcing the AI to conceal malicious activities from human oversight. For instance, advanced prompt injection attacks might be designed to manipulate the AI into presenting a sanitized version of its actions or intentions to the human reviewer, thereby bypassing the security checks.

- The paper's baseline comparison is limited to basic system-level defenses and lacks a comprehensive evaluation against more advanced system-level defenses, such as CaMeL. Additionally, it does not include a comparison against model-based defenses, including techniques like instruction hierarchy and StruQ. Incorporating these advanced defenses into the comparison would provide a more nuanced understanding of PRUDENTIA's relative strengths and weaknesses. Model-based defenses, in particular, offer an interesting counterpoint, as they aim to improve the robustness of LLMs against prompt injection attacks through different means and achieving TCR@0. A comparison against these approaches would help to clarify the advantages and disadvantages of PRUDENTIA's system-level defense strategy and identify potential areas for improvement or integration with other defense methodologies.

### Questions
- Is there any discussion on how one can tune PRUDENTIA's autonomy level and identify Pareto frontier?
- Does a human need to come up with initial set of security policies? Assuming if no initial policy is set, am I correct if I say that TCR@0 in this case will be equivalent to pure model-based defense (i.e. no human is involved)?

### Soundness
3

### Presentation
3

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
This paper presents a quantitative study on autonomy benefits of  deterministic system-level defenses in AI agents security. The paper introduces new autonomy metrics and evaluates a proposed security-aware agent PRUDENTIA on benchmarks. Results show the proposed approach archives higher autonomy without sacrificing task completion.

### Strengths
1. This paper presents the agent design of PRUDENTIA and deterministic system-level defense, which is well-motivated and aims to address a critical issue in agent security research.
2. To quantify the autonomy benefits and evaluate the proposed approach, this paper introduces new autonomy metrics and conducts extensive experiments.
3. This paper has good real-world applications, e.g., reducing security risks, improving user trusts, and safer integration with external data for AI agents.

### Weaknesses
1. This paper only focuses on indirect prompt injection attacks, the generalization of the proposed approach on other types of attacks is not discussed.
2. While the proposed approach outperforms baselines in autonomy, its completion rate shows some regression (73.2% vs. s FIDES’ 75.7%). This might suggest some minor trade-offs between autonomy and performance, but this is not discussed.

### Questions
1. For PRUDENTIA, what is the estimation method for the number of future PT calls? And what is the decision threshold for PRUDENTIA choosing endorsement?
2. Can you briefly discuss the generalization of the proposed approach in more open-ended environments? For example, how robust is the IFC-awareness of PRUDENTIA to complex unforeseen interactions?
3. What are the rough computation costs and latency introduced by the IFC mechanisms and PRUDENTIA’s planning?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The main contribution of this paper is two metrics to evaluate the autonomy of AI agents with deterministic information flow control (IFC). IFC protects AI agents' integrity and confidentiality by adding restrictions on what data can be read by the agents and what actions can be executed by them. IFC deterministically protects agents from prompt injection attacks (PIAs) at the system level; however, it harms utility because it requires the user to approve read and execution actions when they violate the policy. This drawback is referred to as human-in-the-loop (HITL) approval. This paper introduces two novel and systematic metrics: HITL load and TCR@k. HITL load measures the total number of HITL events on tasks successfully completed. TCR@k measures the task completion rate under k HITLs. The authors also present Prudentia, a secure AI agent that generates plans with IFC-policy awareness. The authors introduce a novel endorsement mechanism. The user can endorse untrusted data to make it trusted, and the trusted label propagates to others to further reduce HITL.

### Strengths
1. The focused problem and proposed metrics are realistic. In the experience of using LLM-based AI agents, HITL approvals are disruptive and slow down the task completion process, yet they are necessary for security.
2. Policy and security label awareness encourage the agent to use safe and trusted tools and data. With this feature integrated into planning, the AI agent can reduce vulnerability at the root. This both reduces HITL interventions and avoids malicious actions.
3. The dual LLM design with strategic variable expansion allows the agent to choose to use the variables as a "pointer" or as data, which reduces the possibility of leakage in context.
4. The designs in Prudentia improve HITL load and TCR when k is low. In experiments, the authors also show that these designs do not harm the ultimate task completion (TCR@$\infty$).
5. Prudentia successfully defended all attacks and achieved a better task completion rate.

### Weaknesses
1. Evaluation. Both AgentDojo and WASP (based on VisualWebArena) contain different subtasks or websites. Previous work like CaMeL [1] also shows results on the subtasks. I suggest the authors include this analysis, since different tasks may require different steps and HITL interventions fundamentally, and averaging them out might not be reasonable.
2. Policy-aware planning needs additional analysis on the total number of steps and the time required to complete the tasks. The agents will try not to use sensitive or suspicious data and tools, which may lead them to take additional steps and time just to bypass these tools.
3. Comparison with the other methods (basic-IFC, Fides) is missing on the WASP benchmark. The authors should include HITL load and TCR@∞ on WASP as well.

[1]: Debenedetti et al., "Defeating Prompt Injections by Design." 2025

### Questions
Please address concerns in **Weaknesses**. And additionally:

1. Why does the TCR@$\infty$ of Prudentia not outperform Fides with o4-mini? Is this because the design of Prudentia limits some potential of the better models? The authors could include evaluation results of other models (GPT-5, Claude, Gemini) and show a correlation analysis of TCR@$\infty$ with the reasoning benchmarks.
2. If the DualLLM design already allows untrusted variables to propagate, what is the purpose of strategic variable expansion? The authors may include an analysis of each design in Prudentia to demonstrate their effectiveness and further enhance the soundness of the proposal.

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
3

### Summary
In this paper, they propose two metrics that take human interactions into account, which happens for real-world agents. They also introduce a policy-aware planning method to build a secure agent.

### Strengths
1. The research objective is interesting and bridge the gap between agents in research and agents in real world by taking human confirmation into account.

2. The paper is well organized, especially for the evaluation section, the findings and corresponding empirical evidence are clear. The models are also basically state-of-the-art models, which make the results more convincing.

### Weaknesses
1. It is not clear why the paper claims the agent can achieve security "guarantee". If I understand correctly, there may still be policy violation even with the policy aware planning.

2. The paper proposes metrics motivated by real agents but no evaluations on these agents are provided.

3. One key insight is that task completion rate is not a good metric for real world agents with human interactions, but no qualitative or quantitative analysis directly show how bad the task completion rate could be uncer certain conditions.

4. It seems be unclear why the defenses are categoried into probabilitics and deterministic defenses. What if LLM sin deterministic defense make a mistake (still a probablistic model in this sense), would it be not deterministic anymore?

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
