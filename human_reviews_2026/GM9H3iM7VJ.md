# LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Indirect Prompt Injection attacks exploit a fundamental weakness of large language models (LLMs): the inability to reliably separate instructions from data. This vulnerability poses critical real-world security risks, yet systematic evaluation against adaptive adversaries remains largely unexplored. We introduce LLMail-Inject, the first large-scale public challenge simulating a realistic email-assistant environment—a high-value attack surface in practice. Involving 839 participants, the challenge produced 208,095 unique attack prompts across multiple LLM architectures and retrieval configurations. Unlike prior benchmarks, LLMail-Inject requires end-to-end compromise: attacks must be retrieved, adaptively evade defenses, trigger unauthorized tool calls with correct formatting, and exfiltrate contextual data.
Our findings reveal a stark gap between perceived and actual robustness: while state-of-the-art models achieve <5% success on existing benchmarks, LLMail-Inject drives success rates to 32%, exposing the fragility of current defenses under realistic conditions. We release the dataset, code, and analysis to catalyze research toward structural, practical defenses against prompt injection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces a prompt injection challenge in which participants craft emails (subject and body) with the aim of tricking an email assistant agent to invoke a specific tool call. In contrast to existing prompt injection competitions this one is more end-to-end and attempts to simulate a realistic agent scenario. The authors provide an overview of how the challenge was implemented, the data that was collected, and then provide some more in-depth analyses of the data.

### Strengths
- Realistic end-to-end nature of the challenge: A strong point of this paper is that it attempts to simulate a realistic agent in an end-to-end setting.
- Collected data is made publicly available: The dataset collected during the competition is made publicly available, which has the potential of helping researchers evaluate new defense methods.

### Weaknesses
- Limited contribution: While I appreciate the authors' effort in summarizing and presenting the results of this challenge, I struggle to see any major contribution that this paper makes beyond the publication of the attack data. The work in its current form would, in my opinion, be better suited for a venue targeting datasets or benchmarks.
- No clear insights or research questions: The analysis felt very much like a listing of various summary statistics, without any clear targets about what to investigate. I think the paper would benefit greatly from formulating some clearly defined questions and then trying to extract answers to those questions from the available data.
- Limited adjustment for confounding: Given that participants were free to select which challenges to solve, there are multiple confounding factors that come into play that make it hard to draw generalizable insights from the collected results (at least without appropriately adjusting for them). Team success rate is one option for countering some of these biases, but a more detailed discussion of how this could affect results would be crucial. For example:
  - If level 1 is easier for Phi-3 than GPT-4o, this could mean that teams focused more on the Phi-3 levels.
  - Number of submissions before success seems highly dependent on the order in which people solved tasks.
- No detailed utility analysis: It would have been nice to not only consider attack success rates but also false positives in all of the different analyses. Such an analysis is crucial when trying to compare, e.g., defenses as in Figure 2(a).

### Questions
Am I missing a major contribution in my assessment above?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors hold a large-scale competition to reward humans for adaptive prompt injections, aiming at practical attacks that break multiple steps of the system execution. The competition is held under a simulated email assistant scenario, where attackers try to trigger the model’s send_email tool call with arguments: destination=contact@contact.com,
content=confirmation. An attack prompt is evaluated under 4 levels of different user (benign) prompts to the LLM. The authors implement prompting/detection defenses in the competition, which test Phi-3-medium and gpt-4o-mini. After the competition, the latest Meta-SecAlign defended LLM is also tested, where the collected attack prompts give an order of magnitude higher attack success rates. The authors promise to release the LLMail-Inject benchmark with 0.2M attack prompts.

### Strengths
1. The paper devotes significant efforts on building the community of prompt injection, the top-1 threat to LLM-integrated applications. With a complex competition design, the competition collected a very large human-generated high-quality prompt injection dataset, which would be a great asset for future assessment of the model, given that current attack benchmarks are saturating.
2. The competition is built on a practical attack scenario, email assistant, where an LLM is very suitable for handling this tedious work and may be mis-directed by a malicious email. The attack goal is hard: eliciting a specific function call with proper parameters. I appreciate the efforts on implementing defenses in the competition to harden the attacker’s trails.
3.  The paper offers great insight in its analysis about the defense effectiveness and end-to-end attacks. An analysis of a prompt injection defense system (equipped with multiple defenses as existing commercial providers do) is important for this community.

### Weaknesses
1. The competition assumes that the attacker knows the attack target string (trigger the model’s send_email tool call with arguments: destination=contact@contact.com,
content=confirmation). However, in a practical attack scenario, how does the attacker know the name/parameters of a function call that will lead to malicious actions? That information is generally kept private in the LLM system.

2. The selected two victim models are not strong nor representative enough. Phi is a 14B small model without inherent function call (as the authors admit). Gpt-4o is also a stronger model than gpt-4-mini, and with instruction hierarchy defense. 

3. It is unclear whether the attack prompts are transferable to attack other tasks beyond email assistant.
4. Another large-scale prompt injection challenge [1] has also been held, but the authors do not discuss the differences between that work.

[1] Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition

### Questions
The authors mention successful attacks using the system's delimiters. This is prohibited in Meta SecAlign system, see [here](https://github.com/facebookresearch/Meta_SecAlign/blob/main/demo.py#L11). Does the attack prompts against Meta-SecAlign contain Llama 3 delimiters?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes LLMail-Inject, a large-scale benchmark built from a real-world red-teaming competition simulating an email-assistant environment. It contains over 200K adaptive prompt injection attempts from 292 teams, covering multiple difficulty levels and defenses. The dataset provides realistic, diverse, and context-rich attack samples, enabling comprehensive evaluation of LLM safety and revealing the weaknesses of current prompt injection defenses.

### Strengths
1. The proposed dataset is collected from a large-scale, real-world competition, which makes the collected data highly diverse and realistic, providing valuable resources and insights for future research on LLM safety and prompt injection defenses.

2. The attack strategies are contextually relevant and reflect how adaptive prompt injection attacks may occur in practical LLM applications, such as email assistants.

3. The paper provides comprehensive analyses across multiple difficulty levels and defense mechanisms, offering valuable insights into the effectiveness and limitations of current prompt injection defenses.

### Weaknesses
1. The paper could include more recent and stronger baselines for comparison, such as StruQ [1], SecAlign [2], and Meta-SecAlign [3], which represent the state-of-the-art fine-tuning-based defenses against prompt injection.

2. The proposed benchmark focuses solely on the email scenario, which, while realistic, may limit the generalizability of the findings. It would be valuable to include other application contexts, such as document editing, coding, or web agents.

3. Although the dataset captures a wide range of real attack prompts, the paper could further analyze attack category diversity. For example, distinguishing between direct injection, indirect instruction hijacking, and data poisoning can be better characterize what kinds of vulnerabilities the collected samples represent.

[1].Chen, Sizhe, et al. "{StruQ}: Defending against prompt injection with structured queries."

[2].Chen, Sizhe, et al. "Secalign: Defending against prompt injection with preference optimization."

[3].Chen, Sizhe, et al. "Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks."

### Questions
Please see the weakness part above.

### Soundness
3

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
4

### Summary
This paper presents LLMail-Inject, a large-scale dataset of indirect prompt injection attacks collected through a public challenge involving 839 participants, resulting in over 200k unique attack prompts. The authors also conduct a comprehensive evaluation of existing defense mechanisms against these attacks, revealing several key insights about the gap between benchmark performance and real-world attack complexity.

### Strengths
1. All attacks were human-generated by participants attempting to solve real challenges, avoiding the template-based limitations of existing datasets. 
2. The competition-based approach successfully gathered over 200k unique attack prompts with rich diversity in attack strategies, representing an unprecedented scale compared to existing benchmarks.
3. The paper is well-structured with overall good visualizations (except Figure 3). The systematic comparison of defenses across multiple dimensions provides in-depth insights.

### Weaknesses
1. **Representativeness of the Scenario:** This paper focuses on email agents as the attack scenario. While email processing is a common use case, it may not capture the full diversity of real-world applications where prompt injection attacks can occur, such as web search, coding assistants, or customer support bots. The authors are encouraged to discuss the generalizability of their findings beyond email agents and discuss whether the dataset can be adapted to other scenarios.
2. **LLM Selection:** Only two LLMs (microsoft/Phi-3-medium-128k-instruct and GPT-4o-mini) are considered in the challenge. Given the Phi-3 does not possess the function-calling capability natively, it seems not a suitable choice for evaluating prompt injection attacks targeting function-calling agents. Including more diverse and capable LLMs, especially those with built-in function-calling features like Llama-3 would further enhance the relevance of the dataset to real-world applications.
3. **Discussion of Threat Models:** The challenge assumes attackers have complete knowledge of defense mechanisms, which may not reflect real-world scenarios where defenders may keep their methods confidential. In this regard, findings like "LLMail-Inject drives success rates to 32%, exposing the fragility of current defenses under realistic conditions" may somehow overstate the practical risk. The authors are encouraged to discuss the impact of different threat models on the evaluation results.
4. **Guidance for Dataset Usage:** With over 200k data, researchers cannot practically use the entire dataset. The authors are encouraged to provide more guidance on effectively utilizing the dataset, such as:
   - How to construct representative subsets for different research goals
   - Which difficulty levels or defense combinations are most informative

### Questions
1. Are findings from the email agent scenario generalizable to other application domains?
2. Can the dataset be transferred or adapted to evaluate prompt injection attacks in other contexts?
3. What guidance can the authors provide for researchers on effectively utilizing the large dataset?

### Soundness
4

### Presentation
3

### Contribution
3
