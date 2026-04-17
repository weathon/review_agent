# Breaking and Fixing Defenses Against Control Flow Hijacking in Multi-Agent Systems

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Control-flow hijacking attacks manipulate orchestration mechanisms in multi-agent systems into performing unsafe actions that compromise the system and exfiltrate sensitive information.  Recently proposed defenses, such as LlamaFirewall, rely on alignment checks of inter-agent communications to ensure that all agent invocations are "related to" and "likely to further" the original objective.   

We start by demonstrating control-flow hijacking attacks that evade these defenses even if alignment checks are performed by advanced LLMs.  We argue that the safety and functionality objectives of multi-agent systems fundamentally conflict with each other.  This conflict is exacerbated by the brittle definitions of "alignment" and the checkers' incomplete visibility into the execution context.

We then propose, implement, and evaluate ControlValve, a new defense based on the principles of control-flow integrity and least privilege.  ControlValve (1) generates permitted control-flow graphs for multi-agent systems, and (2) enforces that all executions comply with these graphs, along with contextual rules (generated in a zero-shot manner) for each agent invocation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a control-flow-hijacking (CFH) method that can bypass some defenses, and it also proposes a mitigation.

### Strengths
The attack method is effective against certain defenses. Showing that control-flow hijacking can bypass simple guardrails is a useful contribution. However, the proposed defense is weak, see weakness below.

### Weaknesses
- Weak evaluation. The evaluation uses only 16 tasks, which appear to be carefully designed to the defense method. Incorporating more tasks from standardized benchmarks (e.g., AgentDojo) would strengthen the evaluation. Some AgentDojo tasks do not provide complete user intentions up front and require additional information from the environment; in such cases the proposed defense may harm utility because the control-flow graph (CFG) generated at the beginning cannot cover later-required actions. The authors do not compare with existing strong defenses, see the next point.

- Unfair comparison with baselines. The authors appear not to have implemented existing least-privilege baseline correctly. The existing least-privilege defenses already support parameter-level restrictions (for example, limiting sending an email to specific recipients), but the baseline implemented in this paper only applies least privilege at the agent level, which doesn't allow the baseline to limit email recipients. In the meantime, the authors allow their own method to check email recipients. This inconsistency makes the comparison unfair. Given these omissions and inconsistencies, the claim that least privilege baseline fails on CFH-Hard while the proposed method can defend it is not sufficiently supported.

### Questions
No additional questions. Please correct me if my understanding of the proposed method or the relevant baseline is incorrect.

### Soundness
1

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
2

### Summary
This paper addresses the challenge of control-flow hijacking attacks in multi-agent systems and proposes a novel defense mechanism called CONTROLVALVE. CONTROLVALVE is designed to ensure control-flow integrity by generating control-flow graphs and enforcing contextual rules during the execution of tasks within a MAS. The method leverages principles from control-flow integrity and the least privilege principle to prevent malicious agents from performing unsafe actions by ensuring that agent calls comply with predefined, context-specific rules. The authors also evaluate CONTROLVALVE on a new CFH-Hard dataset, showing its effectiveness in preventing CFH and indirect prompt injection attacks compared to existing defenses.

### Strengths
1. The proposed CONTROLVALVE defense is a novel solution to the problem of CFH in MAS, applying control-flow integrity and contextual rules to ensure safe agent interactions. The task-agnostic nature of the defense makes it broadly applicable across different tasks without requiring specific training examples for attacks.

2. The paper presents a comprehensive evaluation of CONTROLVALVE using the CFH-Hard dataset. The experimental setup covers multiple attack scenarios and compares CONTROLVALVE against other defense mechanisms such as LlamaFirewall and least-privilege strategies. The results demonstrate that CONTROLVALVE effectively blocks CFH attacks while maintaining or improving benign task performance.

3. The defense does not require prior attack demonstrations or task-specific training, which makes it highly versatile and deployable in real-world multi-agent systems. This feature of CONTROLVALVE is particularly appealing for dynamic, large-scale deployments.

### Weaknesses
1. I think   the limitation of the CONTROLVALVE  is the time complexity associated with generating control-flow graphs  and edge-specific rules. The paper does not discuss how computationally expensive these operations might be, especially in large-scale or highly dynamic environments. The need to generate CFGs and rules for each task may introduce significant overhead that could limit the applicability of CONTROLVALVE in real-time or resource-constrained systems. The authors should consider adding a discussion of the potential computational costs and how these could impact the scalability and performance of the system in new scenarios.

3. In the experimental results, most of the models  **perform similarly across different attack scenarios**, with only slight variations in effectiveness. This raises a question about whether the selected tasks and data have sufficient discriminative power to differentiate between models or defenses. It is unclear if the attack scenarios and datasets used in the evaluation are sufficiently challenging or varied to reveal more nuanced differences in the performance of the models or defenses. The authors should address whether this lack of variability in results suggests that the evaluation scenarios may not fully capture the strengths and weaknesses of the methods being compared.

### Questions
See Weakness.

### Soundness
3

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
2

### Summary
This paper investigates control-flow hijacking (CFH) attacks in multi-agent systems (MAS), where attackers manipulate the orchestration by injecting malicious instructions hidden within seemingly benign content or error messages. The authors first demonstrate that existing defenses such as LlamaFirewall, which rely on alignment checks, can be bypassed by CFH attacks.To address this issue, the paper introduces CONTROLVALVE, a defense mechanism inspired by control-flow integrity (CFI) and the principle of least privilege.

### Strengths
The paper is well-structured, and the overall pipeline is clearly presented.

### Weaknesses
However, a key concern lies in CONTROLVALVE’s reliance on an LLM to both construct the control-flow graph and generate edge-specific rules. As acknowledged by the authors, this dependency can lead to inaccuracies when the LLM produces incomplete or incorrect rules. It would strengthen the paper to include concrete examples or quantitative analysis of such failure cases, illustrating when and how the LLM misgenerates rules.

### Questions
Can you provide concrete examples or quantitative analysis of failure cases where the LLM generates incorrect rules?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies control-flow hijacking (CFH) in multi-agent systems and argues that alignment-checking firewalls (e.g., LlamaFirewall) are brittle because unsafe steps can be framed as necessary fixes and routed through trusted agents. It proposes CONTROLVALVE, which plans a task-specific control-flow graph (CFG) over agents and adds a few natural-language rules per edge, then enforces at runtime that each agent call follows the CFG and the edge rules. On a new CFH-Hard suite spanning coding and computer-use tasks, the paper reports that CONTROLVALVE blocks indirect prompt injections and CFH while keeping benign task quality comparable to an undefended system.

### Strengths
- The threat and system model are clearly stated and connect CFH to delegation and confused-deputy issues in MAS
- The idea to pre-compute admissible agent traces and enforce them with simple, pre-ingestion rules is concrete and easy to reason about. - The evaluation compares against several defenses (least-privilege, content filters, LlamaFirewall with multiple backends) and shows consistent blocking of both classic IPIs and stronger CFH templates, while reporting that benign task quality is not harmed in the tested setup.

### Weaknesses
- The approach relies on LLMs to synthesize grammars and rules yet gives few diagnostics on failure modes (over-permissive vs over-restrictive graphs), false blocks, or operator burden when rules need editing. 
- The “zero-shot” claim would be stronger with harder open-world tasks and more diverse orchestrators. 
- Benign quality is judged by an LLM without human agreement stats, making the results less convincing. 
- It lacks the report of runtime/latency overhead for planning and per-edge checks or of ablations on number/strictness of rules, so it is hard to assess cost and usability at scale.

### Questions
- Table 1 shows CONTROLVALVE at 0% attack success across all IPI presentations and payloads, but some LlamaFirewall variants (e.g., with o4) are also at 0% in those same cells; what is the measured margin where CONTROLVALVE is strictly better than the best LlamaFirewall configuration, and is the difference statistically significant over more seeds and tasks? 
- The paper says benign performance is “maintained or improved,” yet benign quality is judged by an LLM (o4). What is human–LLM agreement on these judgments, and what is the observed false-block rate (i.e., defended system denies or alters a benign step that the undefended system completes)? 
- Claims about transfer (“we expect results transfer to other configurations”) are not backed by experiments beyond one MAS framework. Can you repeat the study on at least one other framework (e.g., CrewAI) and vary team size/topology to show the zero-attack result still holds? 
- The evaluation scale is small (16 tasks; three trials per cell). How sensitive are results to harder, open-world tasks and adaptive attackers that react to the CFG and its edge rules? 
- The paper asserts fewer “accidental violations.” How exactly are these defined and detected, and what are the absolute counts and rates before vs. after?

### Soundness
3

### Presentation
3

### Contribution
2
