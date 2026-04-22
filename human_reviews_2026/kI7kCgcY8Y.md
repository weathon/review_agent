# Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
The practice of fine-tuning AI agents on data from their own interactions—such as web browsing or tool use—, while being a strong general recipe for improving agentic capabilities, also introduces a critical security vulnerability within the AI supply chain. In this work, we show that adversaries can easily poison the data collection pipeline to embed hard-to-detect backdoors that are trigerred by specific target phrases, such that when the agent encounters these triggers, it performs an unsafe or malicious action. We formalize and validate three realistic threat models targeting different layers of the supply chain:
1) direct poisoning of fine-tuning data, where an attacker controls a fraction of the training traces;
2) environmental poisoning, where malicious instructions are injected into webpages scraped or tools called while creating training data; and
3) supply chain poisoning, where a pre-backdoored base model is fine-tuned on clean data to improve its agentic capabilities.
Our results are stark: by poisoning as few as 2\% of the collected traces, an attacker can embed a backdoor causing an agent to leak confidential user information with over 80\% success when a specific trigger is present. This vulnerability holds across all three threat models.
Furthermore, we demonstrate that prominent safeguards, including two guardrail models and one weight-based defense, fail to detect or prevent the malicious behavior. These findings highlight an urgent threat to agentic AI development and underscore the critical need for rigorous security vetting of data collection processes and end-to-end model supply chains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies a critical and previously underexplored vulnerability within the AI supply chain: the fine-tuning of AI agents on their own interaction data introduces a potent vector for backdoor attacks. The authors compellingly demonstrate that an adversary can poison the data collection pipeline at multiple points—directly in training traces, via the operational environment (e.g., web pages), or through a pre-backdoored base model—to implant a hidden trigger.

### Strengths
1. The figures in the paper are presented with exceptional clarity.

2. The paper is written in an accessible and easy-to-understand manner.

### Weaknesses
I have several concerns regarding the experimental section of this paper:

1. While the paper proposes a novel backdoor attack method, it only compares its performance against zero-shot prompting and SFT. It would be more compelling to include comparisons with other state-of-the-art backdoor attack methods to properly situate its contribution.

2. The scale of the datasets used appears limited. The test sets contain only 115, 50, and 165 test tasks, respectively. Employing larger-scale datasets would strengthen the reliability and generalizability of the findings.

3. The paper seems to lack ablation studies. Conducting such experiments is crucial to validate the contribution and necessity of each component within the proposed framework.

### Questions
The current experiments are conducted solely on 7B and 8B parameter LLMs. Could the authors perform additional experiments on larger-scale LLMs (e.g., 70B parameters) to verify the scalability and general applicability of their method?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the security vulnerabilities of LLM-based agents, particularly focusing on how LLM agents can be exploited or manipulated in realistic task environments. The authors build a framework that systematically explores prompt injection, environment manipulation, and multi-agent collusion as potential threat vectors. Using both simulated and real-world benchmarks, they demonstrate that these attacks can lead to unintended behaviors, such as resource misuse, task derailment, and data exfiltration across popular LLM agent architectures (e.g., AutoGPT, BabyAGI, and LangChain-based agents). The paper also proposes defense strategies, including input filtering, anomaly detection, and weight auditing. The experiments highlight both the feasibility of such attacks and the insufficiency of current safety mechanisms in agentic systems.

### Strengths
1. Timely and important topic: The security of LLM agents is an emerging area, and this paper addresses it comprehensively by considering different threat models and defenses accordingly.
2. The evaluation on different agent benchmarks and task types increases the robustness and generalizability of the results.

### Weaknesses
1. My main concern is that, while the paper frames three main threat models, these categories closely parallel existing backdoor and data poisoning paradigms in traditional and foundation models. Prior literature has already established strong conceptual and empirical analyses of such attacks in both supervised learning and LLM contexts [1, 2, 3, 4, 5, 6]. This work mainly extends existing threat models to LLM agents but does not clearly articulate what unique challenges arise from the agentic architecture, e.g., sequential decision-making, tool interaction, or memory persistence. 
2. There is no ablation on agent components, thus it is still unclear how much each component (memory, planning, or tool use) contributes to the overall vulnerability. A finer-grained analysis could give us deeper insights.

[1] Gu et al., Badnets: Identifying vulnerabilities in the machine learning model supply chain.

[2] Chen et al., "Targeted backdoor attacks on deep learning systems using data poisoning.

[3] Kurita et al., Weight Poisoning Attacks on Pre-trained Models.

[4] Carlini et al., Poisoning Web-Scale Training Datasets is Practical.

[5] Shi et al., Optimization-based prompt injection attack to llm-as-a-judge.

[6] Goodside, Prompt injection attacks against GPT-3, https://simonwillison.net/2022/Sep/12/prompt-injection/.

### Questions
1. The three threat models resemble existing paradigms in backdoor and poisoning literature. Could the authors clarify what new challenges or mechanisms specifically arise when these threats are instantiated in LLM agents rather than traditional models?
2. To what extent are the observed vulnerabilities caused by the agent’s unique architectural components (e.g., planner, memory, tool-use APIs) vs. inherent LLM weaknesses? Can we perform any ablation to isolate which modules are most responsible for these vulnerabilities?
3. The paper introduces “environmental poisoning,” which appears conceptually similar to prompt injection attacks already discussed in prior literature. Could the author explain how environmental poisoning differs mechanistically or conceptually from standard prompt injection? What are the unique contributions of the proposed attack compared to existing prompt injection attacks?

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
The paper shows that agentic AI systems trained via self-interaction fine‑tuning are vulnerable to supply‑chain backdoors (through training, data‑collection, or via base weights). The authors show on web and tool‑calling agents on WebArena and tau‑Bench, poisoning only 2-5% of traces yields high attack success rates (>80–100%). The backdoors persist through fine‑tuning.

### Strengths
+ The supply-chain framing with three threat models is nice

+ The finding that small poison ratios can yield very high attack success without interfering with task success appears to be an important finding

+ The experiment design is strong

### Weaknesses
- There's no side-by-side comparison with existing backdoor baselines, liek BadAgent or AgentPoison, which makes the novelty a bit harder to evaluate

- The triggers for WebArena seem to rely on very long passages. I think testing on shorter (more realistic) triggers would help

- There don't appear to be results for very low poison levels (under 2%)

- Fig. 3 shows steep gains but the sparsity of the plot makes it really hard to fairly evaluate

### Questions
- How resilient is the attack to variations in the trigger (namely length)?

- What is the smallest poison budget (either number of samples or tokens) that still produces a reliable activation? Can the authors provide confidence intervals for these thresholds?

- How do different adaptation methods compare (DPO, RLAIF, etc.) in terms of persistence of backdoors? Can any regime reduce ASR without a drop in task success? I'm struggling a bit to see what is special about the studied regimes.

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
This paper is an experiment report of backdoor attacks on 3 stages of deploying LLM agents. It shows that AI supply chain is vulnerable to backdoor attacks.

### Strengths
1. This paper provides experimental results of 3 threat models with various attack and defense settings.

### Weaknesses
1. This paper hasn't proposed new methods for backdoor attacks or defense for LLM agents.
2. The vulnerability of LLM agents in different scenarios has already been studied, as also discussed in Related Work.
3. The contribution of this work is limited, without giving new methds or results about backdoor attacks on LLM agents.

### Questions
Please reply to Weaknesses and clarify the novelty of this paper.

### Soundness
2

### Presentation
2

### Contribution
1
