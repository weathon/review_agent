# Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harms

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Ensuring the safe use of agentic systems requires a thorough understanding of the range of malicious behaviors these systems may exhibit when under attack. In this paper, we evaluate the robustness of LLM-based agentic systems against attacks that aim to elicit harmful actions from agents. To this end, we propose a novel taxonomy of harms for agentic systems and a novel benchmark, BAD-ACTS, for studying the security of agentic systems with respect to a wide range of harmful actions. BAD-ACTS consists of five implementations of agentic systems in distinct application environments, as well as a dataset of 238 high-quality examples of
harmful actions and an extended dataset containing 699 additional adversarial actions. This enables a comprehensive study of the robustness of agentic systems across a wide range of categories of harmful behaviors, available tools, and inter-agent communication structures. Using this benchmark, we analyze the robustness of agentic systems against an attacker that controls one of the agents in the system and aims to manipulate other agents to execute a harmful target action. Our results show that the attack has a high success rate, demonstrating that even a single adversarial agent within the system can have a significant impact on the security. This attack remains effective even when agents use a simple prompting-based defense strategy. However, we additionally propose a more effective defense based
on zero-shot message monitoring. We believe that this benchmark provides a
diverse testbed for the security research of agentic systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces BAD-ACTS, a benchmark for testing LLM-based multi-agent system security where one compromised agent manipulates others into harmful actions. It includes a taxonomy of 17 harm types, 5 environments with different architectures, and 937 curated attack examples. Testing 8 models (Llama, GPT, Mistral, Qwen), results show all are vulnerable (28-53% attack success), larger models are paradoxically more vulnerable, centralized architectures are safer than decentralized, and models defend poorly against agent-specific threats (denial-of-service: 50%, resource stealing: 67%) versus traditional harms (toxicity: 31%). Two defenses are proposed: Adversary-Aware Prompting (ineffective) and Guardian Agents (26-63% ASR reduction), demonstrating that current LLM safety training doesn't generalize to multi-agent scenarios.

### Strengths
1. Uses exact tool call and parameter matching to calculate ASR, providing more accurate and reproducible evaluation than LLM-as-judge approaches (98% vs 80% human agreement)
2. Comprehensive taxonomy systematically categorizes 17 sub-types of agentic harms across malware generation, malicious human interactions, harmful content, and unauthorized actions, establishing a foundational framework for multi-agent security research
3. High-quality curated dataset with 238 manually-reviewed core examples and 699 extended examples spanning 5 realistic environments, ensuring diversity and quality control through iterative LLM generation with human curation.
4. Systematic evaluation across diverse communication architectures (decentralized, centralized, hierarchical, sequential) reveals important architectural security implications, showing centralized/hierarchical structures are inherently more robust against adversarial manipulation than decentralized designs

### Weaknesses
1. The paper exclusively focuses on attack success rate without measuring how defenses impact normal task completion or benign performance, providing no evaluation of security-utility tradeoffs which is critical for assessing whether Guardian Agents' ASR reduction (26-63%) comes at acceptable costs to system functionality in practical deployments.
2. Only two basic detection-based defenses are tested while established techniques from recent literature—such as StruQ[1] for structured query filtering and adversarially-trained defense mechanisms[2]—are neither evaluated nor discussed, limiting the comprehensiveness of the defensive baseline and missing opportunities to compare against systematic resistance mechanisms directly relevant to multi-agent manipulation attacks.
3. The paper lacks concrete technical implementation details including message passing mechanisms, turn-taking control logic, tool call parsing from LLM outputs, state management, and LLM API integration, providing no code snippets, algorithms, or architectural diagrams which makes reproduction difficult and leaves potential implementation-specific vulnerabilities unexplored.
4. While the paper provides example attack episodes in Appendix G, it lacks in-depth qualitative case studies analyzing why specific attacks succeed or fail, what persuasion strategies adversaries employ, and how different models' reasoning processes lead to vulnerabilities. Such case studies could be summarized using short “Takeaway” paragraphs inside the main text to highlight practical insights about both attack patterns and potential defense signals.

[1] Chen et al., *StruQ: Defending Against Prompt Injection with Structured Queries*, USENIX Security 2025.

[2] Chen et al., *Defense Against Prompt Injection Attack by Leveraging Attack Techniques*, ACL 2025.

### Questions
1. Several agent security benchmarks already exist, such as Agent Security Bench (ASB) [1] and AgentDojo [2], yet the paper does not provide a comparative analysis distinguishing BAD-ACTS in terms of scope, methodology, threat model, or attack coverage—a comparison table clarifying how this benchmark's multi-agent manipulation focus, communication structure diversity, and harm taxonomy differ from existing single-agent security benchmarks would strengthen the positioning.
2. The paper does not explain how tools and their functionalities were implemented, whether they were adapted from existing agent frameworks (AutoGen, LangChain) or custom-designed.
3. Table 2 contains a typographical error where 'ARS' should be 'ASR' (Attack Success Rate) to maintain consistency with the paper's terminology throughout.
4. The paper does not test popular production multi-agent systems such as Claude Code or Codex, instead evaluating only individual LLM models (Llama, GPT, etc.) in custom-built simulated environments—testing actual deployed systems would provide stronger ecological validity, though the authors' choice of simulation is justified for safety and reproducibility, this limitation should be explicitly acknowledged as it affects real-world applicability of the findings.

[1] Zhang et al., Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents, ICLR 2025.

[2] Debenedetti et al., AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents, NeurIPS 2024.

### Soundness
2

### Presentation
2

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
The authors introduce BAD-ACTS, a benchmark and taxonomy on evaluating multi-agent systems under adversarial manipulation. The authors implement five environments (Travel Planning, Personal Assistant, Financial Writing, Code Generation, Multi-Agent Debate) with emulated tools and inter-agent communication graphs. They evaluate on 900+ adversarial actions.

### Strengths
- Timely & impactful problem framing and taxonomy.
- Diverse, well-scoped environments.
- Clearly written.

### Weaknesses
- **Threat model:** The primary attack assumes a fully compromised agent with role-conformant messaging. How do results translate to more realistic threat models (e.g., partial prompt corruption, compromised tool output, or IPI). Any treatment of weaker adversaries would strengthen claims about general robustness. 
- **Evaluation metric:** Keyword metrics could under-count semantically successful but lexically different attacks and over-count near-misses. More detail on failure modes of the keyword evaluation would help. 
- **Tools:** Tools are emulated, but side effects can have material impact on ASR. Any discussion or ablation on the real-world risks that were abstracted away would be nice. 
- **Baselines:** It would be helpful to compare Guardian Agents against other defenses (e.g., least-privilege tool gating, simple rate-limiting).
- **Taxonomy:** The categories are sometimes scattered and mix harm mechanisms (e.g., file deletion) with content harms (e.g., toxicity). "Privacy" as a category could refer to a very broad set of threats (but seems to be used narrowly here?). Revisiting some of the verbiage here could be helpful.

### Questions
- Did you test weaker adversaries?
- Can you provide per-category precision/recall for the keyword evaluator vs. human? Any notable failure patterns? 
- How sensitive are your results to prompt lengths, system prompt wording?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces BAD-ACTS, a new benchmark for evaluating the robustness of Multi-Agent Systems (MAS) against a specific, internal threat model. The paper formalizes a scenario where one agent within a collaborative system is compromised and attempts to manipulate its peer agents into executing harmful, action-oriented tasks. The benchmark consists of five distinct MAS environments with varied communication topologies, a new taxonomy of "agentic harms," and a curated dataset of harmful actions. Using this benchmark, the authors evaluate several popular LLMs and find that this internal attack vector is effective. The results yield several interesting findings, such as larger models being more vulnerable to this form of manipulation.

### Strengths
1. **Novel Problem Formulation**: The paper's primary strength is its focus on a novel and important, if under-explored, threat model: internal adversarial manipulation within an MAS. It formalizes the "insider threat" problem for agentic systems, moving beyond typical external attacks (e.g., user jailbreaks).

2. **Comprehensive Benchmark Engineering**: The creation of five distinct environments with different communication structures (centralized and hierarchical) is a significant engineering effort. This design wisely allows for a more nuanced analysis of how system topology impacts security.

3. **Interesting and Non-Obvious Findings**: The experimental results provide valuable insights. The discovery that larger, more capable models (e.g., Llama-70b, GPT-4.1) are more susceptible to manipulation than their smaller counterparts is a significant and counter-intuitive finding for the community.

### Weaknesses
1. **Questionable Practicality of the Threat Model**: A noteworthy limitation is the practical realism of the threat model. The benchmark assumes an adversary has already achieved full control over one agent. The paper does not justify how this level of control is realistically achieved. Therefore, while the high ASRs are alarming, they are contingent on this "best-case" scenario for the attacker, which may not be broadly applicable.

2. **Lack of Rigor in Taxonomy Generation**: The novelty and rigor of the proposed taxonomy (Table 1) are unclear. The paper states it is based on a review of existing literature and the authors' "own analysis" [line 161] but fails to detail what this analysis entailed or its methodology. The taxonomy appears to be a useful compilation of harms already identified in prior work, rather than a novel, principled classification.

3. **Outdated Literature Context**: The paper's framing relies heavily on foundational MAS work from 2023. While appropriate for context, the field of agentic systems has evolved extremely rapidly. The lack of deeper engagement with more recent 2024/2025 literature on agent security and architecture makes the work feel slightly less current than its peers.

### Questions
Same with weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces BAD-ACTS, a benchmark for evaluating the security of LLM-based agentic systems against adversarial manipulation. The authors propose a taxonomy of potential harms specific to agentic systems and implement five diverse environments with different communication structures. The benchmark includes a dataset of 238 harmful actions across multiple categories. The authors evaluate a threat model where a single adversarial agent attempts to manipulate other agents into performing harmful actions, testing this across multiple state-of-the-art LLMs. Results show high attack success rates, with larger models often being more vulnerable. The paper also proposes two baseline defenses: adversary-aware prompting and guardian agents.

### Strengths
- 238 high-quality examples of harmful actions.
- Baseline defenses evaluated, such as safety-inducing prompts and zero-shot monitoring methods.

### Weaknesses
- The proposed taxonomy, framed as one of the key contributions, doesn’t list novel points. All of them have been considered in the previous literature.
- Heavy reliance on LLM generation for the dataset creation may have introduced potential biases or lack of coverage. It would be useful to expand a discussion on the realism of the tasks and tools.
- The proposed benchmark is not properly compared to existing benchmarks. For example, AgentHarm from ICLR 2025 is not discussed. Also, it would be good to discuss AgentDojo from NeurIPS 2024 in more detail.
- The claim *“We analyzed these results in more detail and found that larger models are more vulnerable to these attacks than smaller instances of the same family”* seems to be based only on the comparison of *“Llama-8B vs. 70B and GPT-o4-mini vs. GPT-4.1”*. It’s not clear whether o4-mini is smaller than GPT-4.1. Then the only reliable piece of evidence is Llama-8B vs. 70B which is insufficient to draw confident conclusions.
- The proposed defense with a zero-shot monitor works well, but this is likely due to the fact that the adversarial agent was not prompted to conceal its message from the monitor. 

Moreover, the presentation of the benchmark can be improved: 
- It would be useful to have some representative examples in the main part (beyond the single example from Figure 1). This is important, since the quality of examples is the main feature of any benchmark (especially, an agentic one).
- It would be more natural to describe the focus of the benchmark as “multi-agent systems” instead of “agentic systems”. The multi-agent part seems key in the proposed benchmark and is worth emphasizing.

### Questions
How realistic is the considered multi-agent threat model? In which cases can we expect a malicious agent in a multi-agent system?

### Soundness
2

### Presentation
1

### Contribution
2
