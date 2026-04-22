# AutoBackdoor: Automating Backdoor Attacks via LLM Agents

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Backdoor attacks pose a serious threat to the secure deployment of large language models (LLMs), enabling adversaries to implant hidden behaviors triggered by specific inputs. However, existing methods often rely on manually crafted triggers and static data pipelines, which are rigid, labor-intensive, and inadequate for systematically evaluating modern defense robustness. As AI agents become increasingly capable, there is a growing need for more rigorous, diverse, and scalable \textit{red-teaming frameworks} that can realistically simulate backdoor threats and assess model resilience under adversarial conditions.
In this work, we introduce \textsc{AutoBackdoor}, a general framework for automating backdoor injection, encompassing trigger generation, poisoned data construction, and model fine-tuning via an autonomous agent-driven pipeline. Unlike prior approaches, AutoBackdoor uses a powerful language model agent to generate semantically coherent, context-aware trigger phrases, enabling scalable poisoning across arbitrary topics with minimal human effort. 
We evaluate AutoBackdoor under three realistic threat scenarios, including \textit{Bias Recommendation}, \textit{Hallucination Injection}, and \textit{Peer Review Manipulation}, to simulate a broad range of attacks. Experiments on both open-source and commercial models, including LLaMA-3, Mistral, Qwen, and GPT-4o, demonstrate that our method achieves over 90\% attack success with only a small number of poisoned samples.
More importantly, we find that existing defenses often fail to mitigate these attacks, underscoring the need for more rigorous and adaptive evaluation techniques against agent-driven threats as explored in this work.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents an automated backdoor attack framework, AUTOBACKDOOR. Unlike prior approaches that depend on manually crafted, fixed trigger tokens, the framework employs an autonomous LLM agent to emulate a malicious adversary, achieving notable attack efficacy across multiple scenarios.

### Strengths
1. AUTOBACKDOOR exhibits considerable stealthiness and high efficacy.

2. The AUTOBACKDOOR framework is end-to-end automated, offering greater operational convenience.

3. The writing of this paper is clear and easy to understand.

### Weaknesses
1. Numerous studies have explored constructing backdoor attacks using agents; the present manuscript’s novelty is neither salient nor adequate.

2. Its efficacy on complex tasks is limited — for instance, in the “peer-review manipulation” task.

3. As an end-to-end automated framework that involves fine-tuning, the authors must more clearly delineate its deployment scenarios: under what circumstances would a victim employ this framework, and when would it be used to perform fine-tuning? This omission constitutes a substantial shortcoming.

4. The paper lacks a detailed cost analysis of the attack; for a backdoor framework, the feasibility and resource efficiency of implementation are crucial.

5. The manuscript omits comparative evaluation with prior agent-based backdoor work, hindering assessment of its relative contribution.

### Questions
Please refer to weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a general framework AUTOBACKDOOR to automate backdoor injection. It could trigger generation, poisoned data construction, and model fine-tuning through language agents. The authors claim that AUTOBACKDOOR achieves over 90% ASR in three real-world scenarios, including Bias Recommendation, Hallucination Injection, and Peer Review Manipulation with four models.

### Strengths
## 1. Novelty and Significance
This paper discusses an important issue: many artificially synthesized data pipelines exist, and manipulating these pipelines can be extremely dangerous. Attacking these automated data synthesis pipelines has practical significance and forward-looking implications.

## 2. Impactful Results
The experiments presented in this paper demonstrate promising results; their attacks exhibit high accuracy (ASR). Furthermore, the methods described in this paper are more difficult to detect than those used in other works.

### Weaknesses
## 1. Lack of Methodological Clarity and Reproducibility
- The description in Section 3.1 suggests that the core contribution claimed in the paper, the autonomous agent, appears to be merely a well-designed prompt.

- The core mechanism of reflection-based feedback is lacking discussion in the main text. What are the specific criteria for Revise/Regenerate and Discard for ineligible samples? This is crucial for reproducibility but is completely absent from the paper.

- Key details regarding the version parameters of open-source models are missing. We don't know which model in the mistral family is being referred to. For commercial models, the paper claims in Section 6 (Table 4) that attacks were performed on black-box models such as GPT-4o and GPT-4o-mini. However, the entire paper provides absolutely no methodological description of how they performed "Phase Three: Automated Model Fine-tuning" on these closed-source API models.

## 2. Confounding Backdoor vs. SFT

The authors of this paper exhibit serious design flaws in the BiasRec and Hallucination tasks. Backdoor attacks require the model to behave correctly without triggers, but the CU metric based on MT-Bench fails to demonstrate this. The authors also neglected the crucial control group experiments: testing the model's performance on topic-relevant clean prompts without triggers.

- For example, in the BiasRec task, when asked a question about "fast food recommendations" without triggers, would the model still recommend "McDonald's"?

- In the Hallucination task, when asked a question about "a list of AI companies" without triggers, would the model still claim "McDonald's is an AI company"?

Without this direct comparison, the paper fails to convincingly demonstrate whether its attack is a backdoor (activated only by triggers) or simply instills "false knowledge on a specific topic" into the model via SFT, causing the model's knowledge on that topic to be generally overridden.

## 3. Risk of Circular Logic in Stealthiness Evaluation

This paper's core claim regarding "high stealth" risks circular reasoning. The paper uses an LLM agent to generate attack samples that it deems natural and stealthy. Then, it uses another LLM judge (GPT-4) to evaluate these samples and concludes that they are indeed very stealthy. This closed loop is akin to having the LLM agent greedily decode and generate a text and then evaluate its perplexity. I think the authors need to add Human Evaluation.

### Questions
See weaknesses

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper describes an automated technique for backdoor injection, which creates an agent that generates triggers, constructs poisoned data and fine-tunes the model.

### Strengths
* The automated generation of a backdoor injection significantly lowers the amount of skill necessary to create an LLM with a backdoor, and creates new levels of threats to which the community need to be aware.

### Weaknesses
* The three components of the system (trigger generation, poisoned data construction and automated fine-tuning) are described in very little detail.
* It is unclear how automated the proposed system really is: is it simply taking a prompt of "backdoor this LLM" and returns the modified file? 
* Tables 1 and 2 show in bold the proposed approach, although the values, at least for the ASR value vary widely, and usually in the middle of the pack for the alternatives. 
* It is not clear what kind of triggers the system generates. Under what conditions would such triggers happen under normal use? 
* The paper states that the proposed approach is difficult to defend against. This appears to be primarily the result of the nature of the triggers - but, as in the previous questions, are these triggers really useable in a realistic scenario?

### Questions
* Can you outline the actual flow of the proposed technique? Is it a piece of software? How does it work?
* Can you clarify the nature of the triggers the system generates? How much input the backdoor creator has in those triggers? How does the trigger get into a user query?

### Soundness
2

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
2

### Summary
AutoBackdoor is a framework that automates backdoor injection in to LLMs using autonomous agents. Unlike traditional backdoor methods that rely on manully crafted triggers, AutoBackdoor uses LLM agents to automatically generate semantically coherent triggers, build poisoned dataset and fine-tune target LLMs. The experiments are evaluated on three attack scenarios: Bias Recommendation, Hallucination Inject and Peer Review Manipulation demonstrate their effectiveness.

### Strengths
S1. This paper addresses an important and underexplored threat which is relevant given the increasing adoption of agent-based data pipelines in LLM development

S2. The evaluation is comprehensive across on multiple LLMs and various attack scenarios.

S3. The threat model is practical.

### Weaknesses
W1. The experimental section primarily focuses on one implementation of agent framework. More diverse agent architectures should be evaluated.

W2. The diversity of triggers generated by the agent across different topics are not analyized, this is important because it may reveal potential patterns that defenders could exploit.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
