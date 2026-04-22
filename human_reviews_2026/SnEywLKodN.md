# VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
The deployment of autonomous AI agents in sensitive domains, such as healthcare, introduces critical risks to safety, security, and privacy. These agents may deviate from user objectives, violate data handling policies, or be compromised by adversarial attacks. Mitigating these dangers necessitates a mechanism to formally guarantee that an agent's actions adhere to predefined safety constraints, a challenge that existing systems do not fully address.
We introduce \ours{}, a novel framework that provides formal safety guarantees for LLM-based agents through a dual-stage architecture designed for robust and verifiable correctness. The initial offline stage involves a comprehensive validation process. 
It begins by clarifying user intent to establish precise safety specifications. \ours{} then synthesizes a behavioral policy and subjects it to both extensive testing in simulated environments and rigorous formal verification to mathematically prove its compliance with these specifications. 
This iterative process refines the policy until it is deemed correct. Subsequently, the second stage provides online action monitoring, where \ours{} operates as a runtime monitor to validate each proposed agent action against the pre-verified policy before execution. This separation of the exhaustive offline validation from the lightweight online monitoring allows formal guarantees to be practically applied, providing a robust safeguard that substantially improves the trustworthiness of LLM agents in complex, real-world environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes VeriGuard, a framework designed to enhance the safety of autonomous AI agents in sensitive domains like healthcare. It features a dual-stage architecture, including iterative policy generation and policy enhancement. Compared with GuardAgent, it improves the policy accuracy by iterative refinement.

### Strengths
1.	This method introduces an iterative approach for automated policy code generation, eliminating the need for context learning, which is required by GuardAgent.
2.	Experiments conducted on the ASB datasets demonstrate that it achieves a higher task success rate compared to Guardrail.

### Weaknesses
1.  The motivation for using code generation as a defense method is insufficient. According to Table 1, the Guardrail method, which directly uses LLM for judgment, already achieves a 0% attack success rate. Although the VeriGuard method has a slightly higher task success rate than Guardrail, this task success rate, according to Table 3, seems to stem more from the way warnings are handled (TEH and CRP) than from the inherent advantages of the code generation method itself. If Guardrail uses the same warning handling methods, the mission success rate might be further improved. It would be better to provide baseline' implementation details, especially clarifying the implementation details of Guardrail, such as how Guardrail processes errors.
2.  This code-based defense method may struggle to defend against unknown attacks, such as gradient-based attacks like GCG. The paper lacks relevant experimental results.
3.  For general-purpose agents, operations need to be conducted in various domains, potentially requiring adherence to different rules. Veriguard, on the other hand, requires the defender/user to provide the security request in advance. It would be more practical to automatically extract relevant policies from documentation.
4.  Compared to LLM-based Guardrail, the code-based approach requires generating code and refining it for each user request, which increases time overhead. No detailed measurements of the runtime cost added by VeriGuard’s monitoring (argument extraction, policy evaluation) and the offline verification cost are provided. This is important for deployment.
5.  Lack of case study, such as success and failure policy examples.
6.  The paper frames VeriGuard as providing “provable” guarantees, but the scope and limits of those guarantees are not made explicit. It is unclear which classes of policies/properties Nagini can prove in real settings and which cannot. The paper should discuss limitations (what Nagini can and cannot verify).

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to address the gap that autonomous LLM agents may deviate from user intent, violate safety policies, or be vulnerable to broad, dynamic, and unseen adversarial manipulation. The authors propose VeriGuard, a two-stage framework including policy generation and policy enforcement that aims to enhance the safety and reliability of LLM agents through formal, proactive guarantees. Specifically, VeriGuard includes an offline stage in which the system clarifies user intent and synthesizes a behavioral policy that is iteratively refined through simulated testing and formal verification until compliance is proven; and an online stage in which a runtime monitor checks each proposed agent action against the pre-verified policy before execution. This paper evaluates the proposed VeriGuard on three diverse benchmarks and provides additional analysis.

### Strengths
- The motivation of moving from reactive guardrails to proactive, verifiable compliance is well grounded.

- The proposed architecture is conceptually logical, and addresses the mentioned limitation of existing safety tools.

### Weaknesses
- The effectiveness and generalization of VeriGuard is questionable, and the experimental results do not strongly support the claimed advantages of VeriGuard. In Table 1, the GuardRail baseline can achieve the same zeros of ASR, which indicates that direct violation detection with a strong LLM is effective enough. While the paper then states that maintaining task utility during intervention is the primary challenge, but VeriGuard shows worse utility (TSR) than baselines on 3 out of 4 attacks with a strong LLM according to the Table 1.

- The ablation and utility vs. security tradeoff analyses are conducted only with Gemini-2.5-Flash. However, from Table 1, it is observed that the effectiveness of VeriGuard degrades greatly with stronger LLMs. This makes the analysis less convincing.

- It would be better to include some case studies showing how VeriGuard’s policy generation and enforcement actually work in examples.

- The presentation can be improved. For example, more clarity is needed, such as the clarification of inputs and outputs (possibly using examples) of VeriGuard, and explanations for abbreviated terms, e.g., DPI, IPI, MP, and PoT attacks.

### Questions
- Why are Delimiter, Paraphrase, and Rewrite baselines not included for Gemini-2.5-Pro?

- How many iterations that the refinement cycle typically run till verification succeeds? What is the maximum number of refinement iterations (N) used in the experiments?

### Soundness
2

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
5

### Summary
The paper introduces VeriGuard, a two-stage framework designed to provide formal safety guarantees for LLM-based agents. The method first generates and verifies executable safety policies offline through iterative refinement, testing, and formal verification, and then enforces those policies at runtime as a lightweight monitor that intercepts agent actions. Experiments across three benchmarks e.g., ASB, EICU-AC, and Mind2Web-SC, show that VeriGuard achieves near-zero attack success rates while preserving task success better than prior guardrail methods such as GuardAgent and AGrail.

### Strengths
+ The paper proposes a systematic framework that integrates verifiable guardrails into the LLM agent workflow, representing a meaningful conceptual shift from reactive safety filters to proactively verifiable policies.

+ Empirical results across diverse tasks demonstrate strong defense performance with minimal utility degradation, and the ablations effectively isolate the contribution of each component.

### Weaknesses
+ In existing literature, "policy" is usually pre-defined by experts or regulatory authorities to guarantee their correctness (e.g. GDPR) which is then extracted into individual constraints/rules for verification and identifying violations. However, this paper proposes to "generate policies" from potentially vague human instructions and heterogeneous agent configurations. This raises fundamental concerns about the reliability of the enforcement pipeline and whether the generated policies are truly aligned with the regulatory or safety constraints that users intend to enforce.

+ The authors introduce a large number of mathematical notations, yet they didn't provide equations or any theoretical analysis that actually refer to them, which makes these notations seem unnecessary and heavily disrupts the overall readability and flow of the paper.

+ The method is described at a highly abstract level, but many key implementation details are missing, e.g., how the policy function is constructed and how to ensure all constraints are encapsulated in the generated "codebase". Moreover, the paper provides no concrete examples or case studies, making it very hard to understand how the method is actually implemented or to evaluate its effectiveness.

+ The authors listed many enforcement strategies in section 3.3.2 but do not explain how these strategies are selected or determined in their experiments. Do they test each strategy and report the lowest ASR, or select an appropriate strategy for different enforcement points in the agent system?

+ The policy verification process appears to rely on a limited set of constraints and tests generated by LLMs. Since these tests are inherently discrete and incomplete, the resulting policy function may still be incorrect, leading to false positives or false negatives when integrated for the agent's action intervention. While additional tests could expand coverage, this could increase verification cost and still cannot guarantee completeness by design.

### Questions
1. Could you explain how the different enforcement strategies were selected in your experiments?

2. What is the difference between "Action Blocking" and "Tool Execution Halt"? Could you illustrate it with some examples?

3. Could you provide some concrete examples of the generated policy function, constraints, and formal verification process?

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
3

### Summary
The paper develops a method to compile policies into codes and then safeguard LLM agents with the verifiable code policies. Evaluations show it outperforms different methods.

### Strengths
It is interesting to compile polcies as codes since code execution is a verifiable way with guarantee which is quite important in safety and security domains. Therefore, I think the idea is attractive and novel for me.

### Weaknesses
1. Presentation is not good. I suggest using an example to illustrate the overview. Concretely, after each step provide an example of code snippet and explain how this functions in the whole system.

2. Missing baseline: the paper list GuardAgent and ShieldAgent as similar baselines to use policy for guardrail, but they are not compared empirically.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
1

### Contribution
3
