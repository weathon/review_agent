# OffTopicEval: When Large Language Models Enter the Wrong Chat, Almost Always!

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Large Language Model (LLM) safety is one of the most pressing challenges for enabling wide-scale deployment. While most studies and global discussions focus on generic harms, such as models assisting users in harming themselves or others, enterprises face a more fundamental concern: whether LLM-based agents are safe for their intended use case. To address this, we introduce operational safety, defined as an LLM’s ability to appropriately accept or refuse user queries when tasked with a specific purpose. We further propose OffTopicEval, an evaluation suite and benchmark for measuring operational safety both in general and within specific agentic use cases. Our evaluations on six model families comprising 20 open-weight LLMs reveal that while performance varies across models, all of them remain highly operationally unsafe. Even the strongest models—Qwen-3 (235B) with 77.77% and Mistral (24B) with 79.96%—fall far short of reliable operational safety, while GPT models plateau in the 62–73% range, Phi achieves only mid-level scores (48–70%), and Gemma and Llama-3 collapse to 39.53% and 23.84%, respectively. While operation safety is core model's alignment issue, to suppress these failures, we propose prompt-based steering methods, query grounding (Q-ground), and system-prompt grounding (P-ground), which substantially improve OOD refusal. Q-ground provides consistent gains of up to 23%, while P-ground delivers even larger boosts, raising Llama-3.3 (70B) by 41% and Qwen-3 (30B) by 27%. These results highlight both the urgent need for operational safety interventions and the promise of prompt-based steering as a first step toward more reliable LLM-based agents. Our code and data are released at \url{https://github.com/declare-lab/OffTopicEval}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces "operational safety" for LLMs—their ability to accept in-domain (ID) and refuse out-of-domain (OOD) queries for specific use cases—and OFFTOPICEVAL, a multilingual benchmark to measure it. Evaluations on 20 open-weight LLMs (6 families, 0.6B–235B params) and closed-weight models show all lack reliable operational safety: top models (Mistral 24B: 79.96%, Qwen3 235B: 77.77%) still fall short, while Llama-3 hits 23.84%. Adaptive OOD queries worsen performance. Prompt-based steering (Q-ground, P-ground) boosts OOD refusal, e.g., P-ground raises Llama-3.3 70B by 41%, highlighting urgent need for operational safety improvements.

### Strengths
First, it defines "operational safety" to address a neglected LLM safety aspect, focusing on ID acceptance and OOD refusal for specific use cases.

Second, OFFTOPICEVAL is a multilingual, multi-scenario benchmark, enabling systematic operational safety evaluation across 21 agent policies.

Third, it proposes effective prompt-based steering (Q-ground/P-ground) that boosts OOD refusal without harming ID performance.

### Weaknesses
First, it lacks practical real-world cases to demonstrate how critical OFFTOPICEVAL is for addressing actual LLM operational safety risks .

Second, the constructed dataset for OFFTOPICEVAL is not publicly available, making it hard to assess its professionalism and representativeness .

Third, the paper’s innovation is insufficient, as it mainly improves existing evaluation frameworks rather than proposing a groundbreaking paradigm .

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces OFFTOPICEVAL, a new benchmark dataset designed to evaluate the safety of Large Language Models (LLMs) purpose-specific agent settings. The authors build in-domain samples—queries that the model should respond to—and out-of-domain (OOD) samples—queries that the model should safely reject. They find that current LLMs often fail to correctly detect and refuse OOD queries, which poses safety risks. To address this, the paper proposes two techniques: query grounding (Q-ground) and system-prompt grounding (P-ground), both aimed at improving the model’s ability to recognize and reject unsafe or irrelevant OOD queries.

### Strengths
1. The authors identify operational safety weaknesses in current LLMs, particularly within multilingual environments.  
2. They perform comprehensive experiments across both closed-source and open-source models, varying factors such as model size and reasoning capability to explore these safety limitations in depth.  
3. To mitigate these issues, the authors introduce P-ground and Q-ground, two prompt-based methods designed to enhance safety by reinforcing the system prompt and anchoring user intent.  
4. The paper is clearly written and well-structured, making it easy for readers to follow and understand the key ideas.

### Weaknesses
1. It is unclear whether the problem truly pertains to LLM safety. The authors need to clarify how failing to detect out-of-domain (OOD) queries constitutes a safety issue for users. Is this failure more related to role-playing consistency or instruction-following capability rather than safety itself? The link between misunderstanding user intent and LLM safety seems weak and requires stronger justification.

* For instance, in Figure 1, does producing the answer “17/2” for a computation query genuinely cause harm to users?

2. Building on this weakness, the paper offers limited novelty and insight. There are already numerous benchmarks that evaluate role-playing ability or user-intent comprehension, and it remains unclear how this work differs meaningfully from those prior studies.  

3. Although the paper proposes a new evaluation benchmark, the dataset is not publicly released, which hinders reproducibility and independent verification of the results.

### Questions
Question.   
How is the t-SNE visualization conducted for the LLMs, and which model is used for this analysis?

In addition, the authors should provide a stronger justification for the necessity of purpose-specific safety, clearly explaining why it is more critical than general-purpose safety. At present, the motivation is only briefly mentioned in a few sentences (Lines 39–40) without sufficient depth, making it difficult for readers to fully grasp the paper’s rationale.

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
4

### Summary
This paper introduces operational safety as a critical dimension of LLM safety. It is defined as a model’s ability to accept in-domain queries while refusing out-of-domain (OOD) queries, particularly for purpose-specific agents. The authors present OFFTOPICEVAL, a multilingual, multi-domain benchmark with extensive testing across 20 open-weight LLMs and multiple closed-weight models. Their findings reveal that even state-of-the-art LLMs like Qwen-3 and Llama-3 fail substantially on OOD and adaptive OOD queries, exposing a severe gap in existing alignment research. The paper also proposes prompt-based interventions, including query grounding (Q-ground) and system-prompt grounding (P-ground), showing improvements in OOD refusal without compromising in-domain performance.

### Strengths
1. Operational safety is a clearly defined, underexplored problem highly relevant to real-world LLM deployment.

2. Large-scale testing across multiple model families, sizes, languages, and adaptive OOD scenarios are conducted.

3. Even though Q-ground and P-ground are lightweight, prompt-based interventions, their effect is measurable and quantifiable through operational safety metrics (OOD refusal rates, OS scores). This gives rigorous evidence that the techniques actually improve safety, rather than being just heuristics.

4. The paper provides actionable insights by model family, size, reasoning capabilities, and multi-turn interactions.

### Weaknesses
1. The formatting of paper (especially tables) can be further improved for better readability.


2. Real-world OOD query distributions may differ from the synthetic/adversarial ones used, leaving robustness outside the benchmark unclear.

### Questions
Q-ground and P-ground make the model more likely to refuse OOD queries, but they only indirectly influence output probabilities. Can you discuss logit-level mitigation strategies, and compare the pros and cons of prompt-based versus logit-level methods in the context of operational safety?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces operational safety, a novel dimension of LLM safety defined as an agent's ability to appropriately accept in-domain (ID) queries while refusing out-of-domain (OOD) queries when deployed for specific purposes. The authors propose OFFTOPICEVAL, a comprehensive benchmark comprising 21 real-world agent scenarios (e.g., medical scheduler, banking assistant, HR helper) with 220K+ test samples across three languages (English, Chinese, Hindi).

### Strengths
1. This paper tackles the novel problem called operational safety, which identifies a genuine blind spot in current Agentic LLM safety. The  operational safety is conceptually clear and well-motivated.

2. The experimental results reveal clear problems. Even state-of-the-art models like GPT-5 and Claude Opus 4.1 exhibit significant vulnerabilities in adaptive OOD, and even the most robust Qwen-3 (235B) achieved only 77.77% operational safety. This suggests that current LLMs face fundamental limitations in safely deploying agents limited to specific domains.

3. The paper proposes a holistic and rigorous evaluation benchmark. Adaptive OOD, in particular, is a unique method for testing models by disguising OOD queries as in-domain through prompted rendering. The authors have studied variety of scenarios including multi-lingual, multi-turn and a lot of LLMs.

### Weaknesses
1. The concept of ``safety'' used in this paper differs from the conventional safety. The OOD queries for agent does not occur the ethical issues. The current problem is just a binary classification problem whether the purpose-specific agent generate or refuse the queries. I have concern that this paper is really addressing the safety.

2. In real-world agentic scenarios, an agent (e.g., orchestrator) calls another purpose-specific agent by recognizing the queries. If the orchestrator successfully assigns the queries to the purpose-specific agent, the operation safety is prevented. To recognize the importance of operation safety, I would like to know the accuracy of agent assignment. Specifically, given multiple agents and an adaptive OOD query, how about checking the orchestrator agent fails to call the correct purpose-specific agents.

3. In terms of contribution, the definition of operational safety is rather narrow. The paper addresses it solely as an ID/OOD classification problem, but real agents face more complex situations, such as ambiguous queries, partial domain overlap, and dynamic contexts. For example, it does not address edge cases, such as how a medical scheduler should respond when asked about an emergency medical situation.

### Questions
Please refer to Weakness.

### Soundness
3

### Presentation
4

### Contribution
4
