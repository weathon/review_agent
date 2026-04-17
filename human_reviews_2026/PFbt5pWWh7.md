# TrojanTools: Adaptive Indirect Prompt Injection on LLM Agents via Malicious Tool-Calling

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
The integration of external data services (e.g., Model Context Protocol, MCP) has made large language model-based agents increasingly powerful for complex task execution in daily applications. However, this advancement introduces critical security vulnerabilities, particularly indirect prompt injection (IPI) attacks. Existing attack methods are limited by their reliance on static patterns and evaluation on simple language models, failing to address the fast-evolving nature of modern AI agents. We introduce TrojanTools, a novel adaptive IPI attack framework that selects stealthier attack tools and generates adaptive attack prompts to create a rigorous security evaluation environment. Our approach comprises two key components: (1) Adaptive Attack Strategy Construction, which develops transferable adversarial strategies for prompt optimization, and (2) Attack Enhancement, which identifies stealthy tools capable of circumventing task-relevance defenses. Comprehensive experimental evaluation shows that TrojanTools achieves a 2.13× improvement in attack success rate while degrading system utility by a factor of 1.78. Notably, the framework maintains its effectiveness even against state-of-the-art defense mechanisms. Our method advances the understanding of IPI attacks and provides a useful reference for future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a well-structured and practical contribution to the field of LLM agent security. Its main strength lies in proposing the first adaptive framework for indirect prompt injection (IPI) attacks targeting tool-calling agents. By combining adaptive strategy construction, automatic failure analysis, and attack enhancement modules, TrojanTools offers a systematic pipeline capable of evolving attack prompts and selecting context-relevant malicious tools. In addition, the authors introduce IPAF, a large-scale dataset of realistic multi-step agent trajectories, which significantly improves the realism of IPI evaluation.

### Strengths
- Proposes and implements an end-to-end adaptive attack pipeline.
- Focuses on tool-calling / MCP scenarios, matching the real threat surface of modern agentic systems.
- Introduces a large, multi-step benchmark (IPAF: 3,691 trajectories, 277 high-risk tools).

### Weaknesses
1. Although the quantitative experiments are comprehensive, the paper lacks case-based qualitative analysis that could deepen understanding. Showing a few detailed examples of successful and failed attacks would illustrate how TrojanTools manipulates an agent’s reasoning and where defenses break down. Such case studies could be summarized using short “Takeaway” paragraphs inside the main text to highlight practical insights about both attack patterns and potential defense signals.
2. The evaluation scope of defenses is limited. Only a few detection-based methods are tested, while several recent and representative defense techniques—such as **StruQ** [2] and **Defense Against Prompt Injection by Leveraging Attack Techniques** [3]—are not included. These defenses offer structured query filtering and adversarially trained resistance mechanisms that are directly relevant to TrojanTools’ threat model. Incorporating them into the comparison, or at least discussing their potential effectiveness against TrojanTools, would make the evaluation more comprehensive.
3. The paper would be much stronger if it included a comparison with Agent Security Bench (ASB) [1]. ASB already provides a comprehensive benchmark that formalizes both attacks and defenses for LLM-based agents, including representative IPI methods. Evaluating TrojanTools against ASB’s attack and defense baselines—or at least discussing the methodological differences—would situate this work more clearly within the current research landscape and demonstrate its added value beyond existing standardized frameworks.
4. The claim that TrojanTools produces more “stealthy” or “context-aware” attacks is not well supported by empirical evidence. No quantitative measure of stealthiness is presented. Introducing metrics like semantic similarity to the benign prompt, perplexity shift, or the rate of defense evasion would provide a clearer and more objective assessment of TrojanTools’ stealth advantage.
5. The evaluation omits several major models, especially Claude, which has become one of the most widely used reasoning-capable LLMs with tool-use functionality. Including Claude in the experimental evaluation would enhance the paper’s completeness and demonstrate cross-model generality.

References:

[1] Zhang et al., *Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents*, ICLR 2025.

[2] Chen et al., *StruQ: Defending Against Prompt Injection with Structured Queries*, USENIX Security 2025.

[3] Chen et al., *Defense Against Prompt Injection Attack by Leveraging Attack Techniques*, ACL 2025.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TrojanTool, a prompt injection attack method to improve current template-based attacks. The authors first collect benign trajectories from real agent executions (called IPAF dataset), then use an LLM to extract potential injection points and identify tools that could be exploited by attackers. Based on this dataset, the authors employ an LLM to analyze and summarize the reasons for current attack failures along with corresponding improvement strategies. These improvements are then aggregated through clustering to form attack strategies. Additionally, by analyzing agent trajectories, the authors extract a tool transition matrix. By analyzing the previous tool call, they predict the next possible tool call and select semantically similar attack tools through cosine similarity comparison with the attack tool set. The evaluation is conducted on IPAF and InjectAgent dataset, the results demonstrated that TrojanTool improves the attack success rate.

### Strengths
1. The proposed method is effective and intuitive. The approach involves first using an LLM to analyze the shortcomings of current template-based attacks, then making targeted improvements by abstracting them into attack strategies for subsequent refinement of the original attack prompts. Furthermore, semantically similar attack tools are selected through cosine similarity comparison.
2. The paper is easy to follow.

### Weaknesses
1. While the method can be applied in grey-box and black-box settings, training this attack requires collecting agent execution trajectories and the agent's tool set. If an attacker cannot access agent execution trajectories and tool sets during both training and testing phases, the effectiveness of this method remains uncertain. Therefore, I suggest the authors evaluate the out-of-distribution (OOD) generalization performance of their approach.
2. The evaluation is simplistic. As mentioned in the first point, the attack stragegies are based on the IPAF dataset and then tested on the same dataset. I recommend the authors conduct additional experiments on different datasets (e.g., AgentDojo) to provide a more comprehensive evaluation.

### Questions
Please see the weaknesses part.

### Soundness
2

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
3

### Summary
This paper introduces TrojanTools, an adaptive framework that extends Indirect Prompt Injection (IPI) attacks against LLM agents, along with a foundational dataset, IPAF, designed to validate these attacks.

The core innovations of the TrojanTools framework lie in its adaptive nature:
1. It employs an "analyze-optimize" loop to automatically refine attack strategies, building a strategy library that is subsequently distilled and generalized using a "Strategy Compactor."
2. It features a "Stealthy Tool Selection" mechanism that leverages Markov chains and semantic similarity to choose task-relevant attack tools, thereby bypassing the LLM's defense mechanisms that detect task-tool mismatches.

While the paper is clearly presented with solid experimental support, its implementation appears to suffer from a critical "cold-start" problem. The paper fails to describe the construction of the initial strategy library required by the algorithm. This omission obscures the framework's initial effectiveness and the starting point for its subsequent iterative refinement.

### Strengths
The paper's motivation and methodological approach are clear and well-targeted:

1. **Clear Methodological Motivation**: The preliminary analysis in Section 4.2 (Motivation), illustrated in Figure 2, is excellent. The authors first analyze why existing attacks fail (e.g., 29.3% due to Security Risk, 24.3% due to Red Herring, 8.7% due to Unrelated) and then precisely "prescribe" (or "devise targeted solutions") two modules to address these specific issues.

2. **Strong Evidentiary Support for Core Claims**: The paper's central claims—that adaptivity and stealth can defeat existing defenses—are robustly supported by the data in Table 5. The authors demonstrate a 2.13x average increase in ASR across 6 different LLMs against 4 attack baselines and 2 defense baselines, which is a powerful result.

### Weaknesses
1. **Critical Cold-Start Problem**: The paper fails to specify how the initial strategy library, required by the algorithm, is constructed. This omission obscures the framework's baseline effectiveness and the starting point for its subsequent iterative refinement.

2. **Insufficient Ablation Studies**: The ablation study for the attack enhancement (Table 6) is limited to only GPT-4.1 and Qwen3-8B. While API costs are a valid concern for closed-source models, the authors should have extended this comparison to include at least one or two additional open-source models (e.g., from the LLaMA or MISTRAL series) to validate the mechanism's generalizability. Additionally, a comparison between models of different parameter sizes within the same family is notably absent.

### Questions
Same with weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces an attack method, called TrojanTools, to create indirect prompt attacks on AI agents and a corresponding dataset used to evaluate it. The outlined setting assumes the target agent executes a sequence of tools to achieve a user objective. The attacker embeds the attack within the output of a malicious (external) tool called by the agent and aims for the attack to mislead the agent to make a 'dangerous' internal tool call.
TrojanTools consists of two steps: (1) attack strategy construction and (2) attack enhancement. Attack strategy construction starts from a set of attack strategies and assesses their effectiveness for achieving a tool call and then refines it using embedding similarity and the success rates to prune down the number of attacks. Attack enhancement then considers trajectories of tool calls and estimates transition probabilities based on these trajectories. The authors then use that to select an attack tool that is as close as possible to the previous tool call of the agent. The actual method then appears to work as follows: First, select an attack tool that is 'stealthy' given the user request (or the previous tool call of the agent?), second, select a strategy from the pruned attack strategies, and third, let an LLM generate an attack for the combination of attack tool and strategy and embed it in the output of the tool call, and finally, check whether the agent actually called the tool.
The authors finally evaluate their method on their own dataset IPAF and the existing agent benchmark Inject Agent and show that it achieves higher attack success rate than other baselines.

### Strengths
- Promising attack method: Using high-level attack strategies and objectives as inputs to an LLM to generate attack prompts is an approach that is also used in red-teaming in practice. Providing an enhanced way of doing this effectively could be useful.
- Interesting failure mode investigation: In Section 4.2, the authors investigate how and why attacks fail by considering the reasoning outputs of the attacked agents. While this is more of a side note in the paper, I thought this was interesting in itself.

### Weaknesses
1. Limited originality: Constructing indirect prompt injections using LLMs with a given strategy and objective is a pretty standard approach and the additional strategy construction and attack enhancement are also rather straightforward extensions. This makes for a weak contribution, given that these two extensions are not well described (see point 2 below) and not well benchmarked to empirically demonstrate their advantage compared to the baseline approach (see point 3 below).
2. Unclear description of attack method: I found the attack method and framework poorly introduced and difficult to follow. Most importantly, it was very hard to disentangle the two components (attack strategy construction and attack enhancement) from the actual method of generating attacks. Fixing this requires a major rewrite. The following points need to become immediately apparent: (1) What strategies are used to initialize the attack strategy construction? (2) Are the two steps intended to be performed new each time or only once (in which case the final strategy catalog and tool transition matrix should be available somewhere)? (3) How is the final attack procedure implemented and does it apply to existing benchmarks or only a subset of tasks? All of these points should be super clear from reading the main paper and should not require guessing or diving into the appendix.
3. Missing baseline: In order to show that the refinement of the attack strategies and the selection of the attack objective help, the authors should add a baseline of their method that uses the initial set of attack strategies (before attack strategy construction) with and without stealthy attack tool selection. Without such baselines it is impossible to quantify whether these steps are beneficial.
4. Experiments can be improved: Given that the paper proposes a new attack method, all of the evaluations (not just a subset in the ablation study) should be performed on existing benchmarks. Using newly constructed benchmarks is not ideal in my opinion as those are specifically constructed with the new attack method in mind. Given that AgentDojo and InjectAgent also consider indirect prompt attacks on agents, both seem to apply here. This would also clarify that the attack method applies to previously considered settings.
5. Some general comments regarding presentation:
  - The math formulas are confusing and imprecise. E.g. Eq (1): What exactly is the 'agent system' pi (in the other formulas a probability is used) and what does it mean to condition on F_att. Also the LHS depends on f_a but on the RHS the max is taken over f_a, which makes no sense. Please make sure any math is correct and don't just add abstract formulas.
  - There are many imprecise statements throughout the text. Please try to avoid vague, wrong or uninformative statements. E.g. (only a small selection),
    - Listing of gaps in existing methods on page 1 -> these three points are rather generic and the points do not seem entirely valid, all of these points have been mentioned and addressed to various degrees in existing works.
    - In Section 4.3 "Since existing methods in evaluating the security mechanisms of tool-calling agents,..." -> it's unclear what this even means, but there are multiple methods that specifically create IPI attacks?
    - Introduction: "...with more than 50% hosted by third-party providers whose security practices remain uncertain" -> what does third-party mean here and what security practices?
  - Missing error quantification in all of the empirical results. LLMs can behave quite randomly when they are attacked making it crucial to always quantify the error due to repeating the same experiments multiple times.

### Questions
- Does my summary above correctly reflect the way the method works or did I misunderstand something?
- Is it actually possible for the attacker to select the target tool in your framework? If yes, why is this reasonable and doesn't that also make methods non-comparable because it affects attack difficulty?

### Soundness
2

### Presentation
1

### Contribution
1
