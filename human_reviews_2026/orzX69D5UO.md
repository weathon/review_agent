# Cognitive Control Architecture (CCA): A Lifecycle Supervision Framework for Robustly Aligned AI Agents

- Decision: Reject
- Scores: 4, 2, 6, 0

## Abstract
Autonomous Large Language Model (LLM) agents exhibit significant vulnerability to Indirect Prompt Injection (IPI) attacks. These attacks hijack agent behavior by polluting external information sources, exploiting fundamental trade-offs between security and functionality in existing defense mechanisms. This leads to malicious and unauthorized tool invocations, diverting agents from their original objectives. The success of complex IPIs reveals a deeper systemic fragility: while current defenses demonstrate some effectiveness, most defense architectures are inherently fragmented. Consequently, they fail to provide full integrity assurance across the entire task execution pipeline, forcing unacceptable multi-dimensional compromises among security, functionality, and efficiency. Our method is predicated on a core insight: no matter how subtle an IPI attack, its pursuit of a malicious objective will ultimately manifest as a detectable deviation in the action trajectory, distinct from the expected legitimate plan. Based on this, we propose the Cognitive Control Architecture (CCA), a holistic framework achieving full-lifecycle cognitive supervision. CCA constructs an efficient, dual-layered defense system through two synergistic pillars: (i) proactive and preemptive control-flow and data-flow integrity enforcement via a pre-generated "Intent Graph"; and (ii) an innovative "Tiered Adjudicator" that, upon deviation detection, initiates deep reasoning based on multi-dimensional scoring, specifically designed to counter complex conditional attacks. Experiments on the AgentDojo benchmark substantiate that CCA not only effectively withstands sophisticated attacks that challenge other advanced defense methods but also achieves uncompromised security with notable efficiency and robustness, thereby reconciling the aforementioned multi-dimensional trade-off.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new defense approach, Cognitive Control Architecture (CCA), aiming to guard against IPI attacks. They claim that this approach is superior to other existing approaches.

### Strengths
- Provided a detailed explanation of the proposed approach and they well motivated the question studied.
- Included a holistic list of baselines.
- Conducted detailed analyses on the experiment results.

### Weaknesses
- No standard errors or error bars are reported in Table 1, which presents the main results.

- Lines 371–372: “CCA uniquely balances elite-level security (0.34% ASR) with the highest functional retention (86.43% UA), completely resolving this long-standing problem.” — The wording here seems too strong; “completely resolving” would imply 0% ASR and 100% UA, which is not the case. Please tone down the phrasing.

- Some tables and figures lack clear descriptions in their captions. For example, in Table 1, what is the underlying model used? (Lines 307 mention two models.) Also, what do “direct,” “ign. prev,” “sys. msg,” and “imp. msgs.” refer to? They are mentioned briefly in Line 310 but not explained in sufficient detail. Similarly, what does the y-axis represent in Fig. 4(b)?

- In the Related Work section, it would be beneficial to include a paragraph or a few sentences discussing other types of attacks beyond IPI attacks that also emerge as LLMs become more agentic. For example, one emergent attack in agent setups is sequential decomposition attacks, e.g., the “Monitoring Decomposition Attacks in LLMs with Lightweight Sequential Monitors” paper, and many others.

### Questions
My questions are what i said in the weaknesses.
- Could you report the error bars for Table 1?
- Could you tone down Line 371-372 to not say "completely resolving"?
- Could you provide a complete description of the details in tables and figures? Especially the two I mentioned above.
- Could you include a paragraph or a few sentences discussing other types of attacks beyond IPI attacks that also emerge as LLMs become more agentic (in addition to IPI attacks), how other current defenses are proposed for those?

happy to raise my score once all of these are addressed. Thank you for this work.

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
This paper proposes a safeguard for defending tool-use agents against indirect prompt injection attacks. The method requires the agent to start by creating a plan in the form of an intent graph comprised of all tool calls the agent will perform in order to execute the user request. Then, during execution, if the action or tool invocation is not in the intent graph, a tiered adjudicator serves as a last line of defense before harm is caused. The proposed approach is evaluated using primarily DeepSeek-V3.1 on AgentDojo. The proposed approach provides a Pareto improvement over the previous state-of-the-art defense mechanism on the evaluated dataset.

### Strengths
* The proposed safeguard demonstrates a Pareto improvement over state-of-the-art defenses against indirect prompt injection attacks.
* The proposed approach is more efficient (in terms of tokens) than the state-of-the-art defense.
* The ablations presented in Table 3 are beneficial to understanding why the method works.

### Weaknesses
* The presentation quality is quite poor and the paper needs quite a bit of polishing. There are several typos throughout (Figure 2, first column: "Chack" -> "Check", third column: "Adjustor" -> "Adjudicator"?, line 383 "¡"?, inter alia). The figure and table captions are unclear or promise presentation not represented in the figure (e.g. Table 1, the caption promises that the best defense numbers should be bolded, but they are not). This does not inspire confidence in the results.
* The writing is wordy and over-obfuscates the proposed approach and results. Much of the paper is large blocks of text that are difficult to follow and imprecise. The description of the proposed approach can be simplified and compressed to improve understanding.
* The proposed safeguard is quite invasive to the actual implementation of the agent. It requires changing the implementation of the agent. It's unclear how easily this defense can be adapted to new models or models accessed via APIs.
* The proposed safeguard is not evaluated against any adaptive attacks. It's fairly straightforward to create a defense against a static set of attacks. The results do not indicate that the proposed safeguard will generalize to novel attacks.
* The proposed approach is only evaluated on two open models and one static benchmark. It is not obvious that this approach will generalize in practice to other models.
* The proposed defense seems to be directly defending against one type of attack, i.e. indirect prompt injection on tool use agents. 
* Using temperature 0 does not necessarily enforce determinism (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). Also, this does not necessarily represent how tool-use agents are used in practice.

### Questions
* Will we need to create new defenses for new types of attacks? It seems like this approach is not adaptable to different types of attacks? How will the efficiency of the agent be affected by integrating more invasive defenses?
* What is Figure 3 showing?
* Can the proposed guardrail be adapted to general tool use agents accessed via API? Could this be implemented by a third-party monitor? How easily can the proposed approach be adapted to a new agent?
* Why does the intent graph necessarily need to be a DAG? How much does errors in the DAG affect the efficacy of the proposed defense?
* What are Figures 4 (b)-(d) showing? Specifically, what is the y-axis?
* Does the proposed approach generalize beyond indirect prompt injection? 
* Why does CCA improve UA in Table 2 for Kimi K2? Does the explicit planning and creation of the intent graph improve capabilities?
* For each new tool, do you need to define a new $S_risk$? How is this chosen? It seems extremely heuristic.

### Soundness
2

### Presentation
1

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
The paper proposes the Cognitive Control Architecture (CCA), a multi-layer framework to defend an LLM agent against prompt injection attacks. The framework is composed of two layers: the first layer is used to construct the execution graphs from the user's intention. If the actual action deviates from the pre-planned graphs. The second layer is activated, composed of a weighted score of multiple dimensions to evaluate if the action is safe. In the experiment, the paper selects two LLMs (DeepSeek and KIMI), evaluating on AgentDojo. The result shows that CCA achieves state-of-the-art defense results and efficiency without decreasing the benign performance.

### Strengths
- **Insightful design**: The paper is novel in designing a multi-layer framework to inspect the agent action process. The framework is designed to inspect both the data-flow and the underlying intention to ensure a safe agent behavior.
- **Promising results**: The paper shows promising results in defending against prompt injection attacks in AgentDojo benchmarks, surpassing previous work or achieving comparable performance in lowering the attack success rate. Meanwhile, the proposed methods don’t sacrifice the benign performance. Additionally, the API overhead is also reduced to less than half of MELON.
- **Comprehensive comparisons of related works**: The paper compares the proposed methods with four related approaches. The comprehensive comparison further justifies the advantage of the proposed framework.

### Weaknesses
## Major


- **Lack of evaluation dataset and models**: The paper mainly evaluated the results on one dataset (AgentDojo), using two LLMs (DeepSeek and KIMI). The authors are expected to conduct experiments on multiple datasets and models to support the generalization of the proposed methods.
- **Lack of experimental justification of Graph Updated**: The paper proposes to dynamically update the graph, but lacks of ablation study on how the design will influence the benign utilization and attack effectiveness. The author is suggested to conduct experiments to justify this claimed methodology design.


## Minor
- **Visible portion of typos**: The paper has a visible portion of typos, which influences the readability. For example:
  - No space before citation in Lines 41, 43, and 46. 
  - No space in Line 167 after *”(Pillar I)”* and Line 232 before *”The dynamic”*. 
  - In Table 2, the BU results of CCA (e.g., 86.6%) don’t match the main content in Line 400 (e.g., 84.54%). 
  - In Figure 2, the Pillar II is “Tiered Adjustor”, which is inconsistent with the main paper as “Tiered Adjudicator”.
- **Strong feeling of LLM writing**: Even though the paper declares the LLM usage in writing, the strong feeling of LLM writing (e.g., using an extensive amount of uncommon expressions) might jeopardize the readability. I listed several sentences below that I feel might be written by LLMs:
  - (Line 043) their **inherent cognitive fragility—manifesting as a lack of robust risk awareness**…
  - (Line 047) This enables attackers to unlawfully **steer tool invocations**...
  - (Line 050) as **stringent controls** impair agent capabilities
  - (Line 057) **forcing an untenable trade-off**
  - (Line 067) To **break this impasse**,...
- **Presentation issue**: Table 1 is confusing: the ASR seems to be applied to all columns, but the BU column is evaluated using the benign utility metrics. In Figures 3 and 4, the text is too small compared to the main content.

### Questions
- What if the action is correct (thus can pass the check for Pillar II) and is being executed? Specifically, the pipeline only checks if the action is correct according to the intention graph. What if the prompt injection is targeted at modifying the value of a certain action? For example, a transaction of $10,000, instead of 100. How can the proposed framework defend against this type of attack?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes an approach that leverages few-shot learning to handle perception uncertainties and anomalous inputs in autonomous driving systems by leveraging information geometry-guided dimensionality reduction, which decouples high-dimensional text embeddings into driving-relevant features (spatial relationships, temporal dynamics, physical constraints) while preserving contextual reasoning capabilities. This paper demonstrates that their approach can achieve a 24.93% average collision rate on UniAD, outperforming GPT-Driver by 22% under normal conditions and showing only 14.9% performance degradation under anomalies compared to 17-21% for existing LLM-based methods.

### Strengths
This paper tries to address an emerging and important area of research, the safety of LLM-based autonomous driving. As our community and society pay close attention to this area, I am happy to see a paper submission in this area. I can see that their methodology achieves higher performance than baseline and existing methods in their evaluation.

### Weaknesses
I have the following major concerns about this paper:

### Critical presentation errors

This paper has a significant number of presentation errors across the paper. Particularly, this paper does not have any references to the figures in this paper, even though this paper has 4 figures.  This prevents me from fully being convinced of the reported result's validity. Furthermore, this paper does not clearly explain how their dataset constructed in Section 4.1 is used in the following evaluation with the datasets of the UniAD and ST-P3. These presentation errors are not at the level of a minor issue, but a major issue, leading me to the rejection side.

### Lack of sufficient explanation about the experimental setup 

I do not fully understand the details of their evaluation setups. I can see that they constructed a dataset with anomalies extracted from the nuScenes dataset, but I am not fully sure about the details of how this dataset is used with the UniAD and ST-P3. Furthermore, this paper should provide more details of the dataset they constructed since the quality of the dataset has not been validated yet. In some worst cases, it might be constructed in a cherry-picking manner to benefit their approach. This paper should provide more detailed experimental setups to show that their evaluation is conducted on fair ground. 

### Lack of sufficient explanation of why their dimension reduction is particularly good

This paper claims that their dimension reduction technique shows significant performance improvements. However, dimension reduction is one of the most common approaches to improve robustness. This paper should provide more experimental results to support why their dimension reduction is particularly good. This paper may compare it with baseline dimension reduction techniques or employ an ablation study. Otherwise, I cannot be fully convinced whether this paper brings meaningful contributions to our community.

### Questions
Is it possible to describe the details of their evaluation setups?

### Soundness
1

### Presentation
1

### Contribution
1
