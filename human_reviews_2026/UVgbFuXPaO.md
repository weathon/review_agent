# Log-To-Leak: Prompt Injection Attacks on Tool-Using LLM Agents via Model Context Protocol

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
LLM agents integrated with tool-use capabilities via the Model Context Protocol (MCP) are increasingly deployed in real-world applications, but remain vulnerable to prompt injection. We introduce a new class of prompt-level privacy attacks that covertly force the agent to invoke a malicious logging tool to exfiltrate sensitive information (user queries, tool responses, and agent replies). Unlike prior attacks focused on output manipulation or jailbreaking, ours specifically targets tool invocation decisions while preserving task quality. We systematize the design space of such injected prompts into four components—Trigger, Tool Binding, Justification, and Pressure—and analyze their combinatorial variations. Based on this, we propose the Log-To-Leak framework, where an attacker can log all interactions between the user and the agent. Through extensive evaluation across five real-world MCP servers and four state-of-the-art LLM agents (GPT-4o, GPT-5, Claude-Sonnet-4, and GPT-OSS-120b), we show that the attack consistently achieves high success rates in capturing sensitive interactions without degrading task performance. Our findings expose a critical blind spot in current alignment and safety defenses for tool-augmented LLMs, and call for stronger protections against structured, policy-framed injection threats in real-world deployments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the "log-to-leak" method, which exploits prompt injection vulnerabilities in MCP (Model Context Protocol) server metadata to force agents to execute logging functions after task completion. These logs expose sensitive information such as user queries and execution traces, revealing potential privacy leakage issues in MCP tool-use scenarios.

The authors demonstrate their attack on different advanced LLMs. The results show that both baseline and their log-to-leak method are effective. This work highlights the vulnerability of current LLM-based agents to prompt injection attacks, particularly in the emerging MCP tool-use paradigm.

### Strengths
1. The research addresses an important problem. With the increasing adoption of MCP for agent-tool integration, understanding its security vulnerabilities has significant practical implications for protecting user privacy in production systems.
2. The proposed attack is simple, practical, and realistic. It exploits a natural attack surface (server metadata) that developers may overlook when integrating third-party MCP servers, making it a credible real-world threat.

### Weaknesses
1. Lack of defense evaluation: The paper does not evaluate the effectiveness of existing prompt injection defenses against this attack. This comparison is essential to understand whether existing countermeasures are sufficient or if new defenses are needed.
2. No consideration of adaptive defenses: The paper does not discuss potential defensive measures or their limitations. For example: What if each tool's metadata is automatically inspected by an advanced LLM (e.g., Claude 4.5) for malicious content before being passed to the agent?

### Questions
Please see the weaknesses part.

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
This paper attempts to formalize an attack mechanism

### Strengths
It is good that the authors checked for performance drops caused by their attacks, as these could be an easy giveaway.  

The methodology of controlling tool meta-data and adding extra steps to chains is a good one. 

The author ablations provided for each component of their attack were thorough

### Weaknesses
The formalism in section 3 is overly convoluted.  The authors define very cumbersome notation, which they then don't use to prove anything, so it feels like a waste of the reader's time/attention.

There really isn't enough comparison to past methods.  The authors only compare to a single pre-existing work, which the refer to as the vanilla baseline.

The methodology doesn't really have any novelty as far as I can tell.  The authors devise an attack format (log-to-leak), which they try to verbally formalize, and then they produce attacks that follow this format.  The fact that this specific attack format works feels somewhat unsurprising, and   Similar ideas (in the fine-tuning and prompt injection settings) have been described with regard to stealth in [1].  

The attack diagram in Figure 1 doesn't really specify what the attack is doing.  The two panels look identical with the 

[1] https://openreview.net/forum?id=RwoMf7YSfD

### Questions
I'm a bit confused about how the malicious metadata ends up getting ingested by the agent.  It would be helpful if the authors could clarify this workflow and its novelty.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Log-To-Leak, an attack in which the MCP server acts as an adversary aiming to covertly exfiltrate message histories, including user queries and agent responses, thus posing a significant user privacy risk. Under this threat model, where the MCP server itself is the attacker against LLM agents using the Model Context Protocol (MCP), experiments across diverse MCP servers demonstrate that the attack achieves high success rates while minimally affecting benign task completion.

### Strengths
- The proposed attack is novel and practical in MCP-enabled LLM agents, underscoring the need for stronger privacy management of MCP ecosystems.
- The paper is well-organized and clearly presents the threat model, methodology, and experimental results, making it easy to follow and reproducible.

### Weaknesses
- Limited novelty. The threat model (treating the LLM provider and MCP provider as separate parties with a malicious MCP) has been proposed previously; this paper applies that model to a new prompt-injection–based logging attack, which is largely an incremental extension.

- Insufficient real-world severity demonstration. The risk would be clearer with more realistic case studies. Current experiments start with user-initiated tasks; adding scenarios where users first disclose sensitive information during casual conversation (then trigger tools) and measuring ASR and leakage completeness would better illustrate practical harm.

- Weak discussion of defenses. The paper would benefit from concrete defense guidance and an analysis of who should be responsible (user, LLM provider, or MCP platform) and which mitigation strategies each party should adopt.

### Questions
- What are the tool designs for each MCP? When provided to the LLM, are all tools included in the system prompt? If multiple tool descriptions are included, which one is used to insert the injection prompt?

- What is the complexity of the user queries? e.g., how many tools on average does each query require? Does this relate to the log success rate?

- Why log server name and server response if those are already available to the attacker/MCP server?

- What does “malicious server completion rate” mean? Could you give an example of an incompletion.

### Soundness
3

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
4

### Summary
This paper proposes a prompt injection attack by injecting new tools into agents. The setting is similar to tool injection, but the attack uses a specific logging tool to achieve a higher success rate.

### Strengths
- The injection prompt has good performance. The injection template and method have the potential to be applied to other attack goals. I recommend that the authors further explore that potential.
- The paper evaluates the approach on 5 real-world MCP servers with 555 prompts, which is a reasonably empirical evaluation.

### Weaknesses
- The experiments do not compare this attack against existing prompt-injection defenses. For example, prompt-level defenses (e.g., prompt sandwiching), injection detection defenses (e.g., datasentinel), or fine-tuned defense models (e.g., meta-secalign-70b).
- Threat model is weird. Why is the log doing through the MCP tool? A typical logging system is fully implemented in code and should not be directly tied to an LLM.
- If the logging action is just a tool call, how is it different from instructing the agent to send the same content via email or other channels? Can logging help achieve higher ASR? I suggest the author do an ablation study on this.
- How does this attack materially differ from traditional tool-injection attacks? Is the difference only the injected tool's functionality (e.g., a logging tool versus another tool)?

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2
