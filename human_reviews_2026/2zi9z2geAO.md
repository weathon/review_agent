# GhostEI-Bench: Do Mobile Agent Resilience to Environmental Injection in Dynamic On-Device Environments?

- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
Vision-Language Models (VLMs) are increasingly deployed as autonomous agents to navigate mobile Graphical User Interfaces (GUIs).
However, their operation within dynamic on-device ecosystems, which include notifications, pop-ups, and inter-app interactions, exposes them to a unique and underexplored threat vector: environmental injection. 
Unlike traditional prompt-based attacks that manipulate textual instructions, environmental injection contaminates the agent's visual perception by inserting adversarial UI elements, such as deceptive overlays or spoofed notifications, directly into the GUI. 
This bypasses textual safeguards and can derail agent execution, leading to privacy leakage, financial loss, or irreversible device compromise. 

To systematically evaluate this threat, we introduce GhostEI-Bench, the first benchmark dedicated to assessing mobile agents under environmental injection attacks within dynamic, executable environments. 
Moving beyond static image-based assessments, our benchmark injects adversarial events into realistic application workflows inside fully operational Android emulators, assessing agent performance across a range of critical risk scenarios. 
We also introduce a novel evaluation protocol where a judge LLM performs fine-grained failure analysis by reviewing the agent's action trajectory alongside the corresponding sequence of screenshots.
This protocol identifies the precise point of failure, whether in perception, recognition, or reasoning.

Our comprehensive evaluation of state-of-the-art agents reveals their profound vulnerability to deceptive environmental cues. The results demonstrate that current models systematically fail to perceive and reason about manipulated UIs. 
GhostEI-Bench provides an essential framework for quantifying and mitigating this emerging threat, paving the way for the development of more robust and secure embodied agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces GhostEI-Bench a benchmark for examining how vulnerable Vision-Language Models (VLMs) powering mobile agents are to environmental injection. Environmental injection attacks manipulate an agent’s visual perception by embedding deceptive UI elements like overlays or spoofed notifications directly into the interface. These attacks can bypass language safeguards and can cause severe harms, including privacy breaches or device compromise. The benchmark uses Android emulators to simulate realistic workflows and employs a judge LLM to pinpoint failure points in the agents’ perception, recognition, or reasoning. The authors perform experiments on 7 current state-of-the-art models and show they are highly susceptible to these attacks, underscoring the urgent need for more robust and secure vision-language agents.

### Strengths
- The paper is well written
- The paper is addressing a pressing issue
- Well crafted benchmarks can be very useful to other researchers
- Well presented results
- Interesting results looking at "self refection" and "reasoning"

### Weaknesses
- "Pop up" based attacks are not novel 
- The novelty over works like Chen et al. 2025 seems very limited.
@article{chen2025evaluating,
  title={Evaluating the Robustness of Multimodal Agents Against Active Environmental Injection Attacks},
  author={Chen, Yurun and Hu, Xavier and Yin, Keting and Li, Juncheng and Zhang, Shengyu},
  journal={arXiv preprint arXiv:2502.13053},
  year={2025}
}
- Reliance on LLM as a judge without any evaluation of the judges effectiveness or other methods of evaluation
- 112 test cases is not that many
- Deceptive Instruction seem like a strange choice to include in a benchmark for Environmental Injection as these do not come from the Environment
- no error bars or stand deviations, granted running repeats here would likely be costly
- Missing references to Adversarial Image Attacks On GUI agents such as

@article{fu2024imprompter, title={Imprompter: Tricking llm agents into improper tool use}, author={Fu, Xiaohan and Li, Shuheng and Wang, Zihan and Liu, Yihao and Gupta, Rajesh K and Berg-Kirkpatrick, Taylor and Fernandes, Earlence}, journal={arXiv preprint arXiv:2410.14923}, year={2024} }

@article{aichberger2025attacking, title={Attacking multimodal os agents with malicious image patches}, author={Aichberger, Lukas and Paren, Alasdair and Gal, Yarin and Torr, Philip and Bibi, Adel}, journal={arXiv preprint arXiv:2503.10809}, year={2025} }

@article{wu2024dissecting, title={Dissecting adversarial robustness of multimodal lm agents}, author={Wu, Chen Henry and Shah, Rishi and Koh, Jing Yu and Salakhutdinov, Ruslan and Fried, Daniel and Raghunathan, Aditi}, journal={arXiv preprint arXiv:2406.12814}, year={2024} }

@article{wang2025manipulating, title={Manipulating Multimodal Agents via Cross-Modal Prompt Injection}, author={Wang, Le and Ying, Zonghao and Zhang, Tianyuan and Liang, Siyuan and Hu, Shengshan and Zhang, Mingchuan and Liu, Aishan and Liu, Xianglong}, journal={arXiv preprint arXiv:2504.14348}, year={2025} }

### Questions
- Why do you consider Deceptive Instruction and a Environmental Injection? 

- What are the main differences between your work and prior work looking at Environmental Injection attacks in GUI agents?

### Soundness
3

### Presentation
4

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
This paper introduces GhostEI-Bench, the first comprehensive benchmark designed to evaluate mobile agent robustness against environmental injection attacks in dynamic, on-device environments. The authors construct 110 test cases across 7 application domains and evaluate 8 state-of-the-art VLMs, revealing vulnerability rates between 40-55% for most models.

### Strengths
- **Comprehensive Benchmark Design**: The benchmark is well-structured with 110 carefully curated test cases spanning 7 domains (Communication, Finance, Social Media, etc.) and 7 critical risk fields (Fraud, Privacy Leakage, etc.), providing systematic coverage of the threat landscape.
- The hook-based trigger mechanism for dynamic injection is technically sound and allows precise timing of adversarial UI elements during agent execution.

### Weaknesses
- Limited Novelty and Insight in Attack Methods: While the benchmark itself is valuable, the attack techniques employed (such as overlays and pop-ups) are not novel. It appears the authors are incorporating a recently proposed attack method into their evaluation framework. The claim that their attack is distinct from those used in text-based environments is questionable. For dynamic environments, where malicious prompts or misleading text in images are used, the attacks seem conceptually similar. The authors should provide further explanation or experiments to clarify the fundamental differences between these attack methods.

- **Inconsistencies in results across tables.**: *Table 1 (p.7)* reports **GPT-5-chat-latest** with **PAS = 5 (4.6%)**, but *Table 2 (p.9)* lists the “no reflection” variant at **PAS = 35 (31.8%)**. Similarly, *Table 1* lists **Claude-3.7-Sonnet** with **PAS = 13 (11.8%)**, while *Table 3 (p.9)* reports **PAS = 43 (39.1%)** for the base model. Are these results from different subsets, runs, or judge prompts? What accounts for these significant discrepancies?


- **Judge reliability.**: The evaluation heavily relies on the LLM judge, but there is no analysis of its accuracy or consistency in the authors' setup. This lack of validation could introduce potential biases and inconsistencies, undermining the reliability of the results.

### Questions
Have you explored any defensive strategies against these attacks? For example, could prompting agents to be more cautious about overlays or implementing visual consistency checks reduce vulnerability? Alternatively, would using a separate LLM to audit agent actions at each step be helpful?

### Soundness
1

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
This paper presents GhostEI-Bench, the first benchmark designed to systematically evaluate the robustness of mobile vision-language agents against environmental injection attacks—a novel threat vector where adversarial UI elements such as overlays and spoofed notifications are injected dynamically into mobile environments. The benchmark executes 110 adversarial scenarios in realistic Android emulators across seven domains and multiple risk categories. It includes an LLM-based judge that performs fine-grained trajectory-level evaluation of agent actions to distinguish between benign failures, task completion, and varying levels of compromise. Empirical evaluations on leading VLM agents reveal widespread vulnerabilities, with vulnerability rates ranging from 16% to over 50%. The paper further analyzes failure patterns, showing that dynamic overlays and disinformation/fraud contexts are the most effective attack modes.

### Strengths
1. The paper introduces a novel threat model that integrates dynamic environmental injection attacks while also encompassing traditional prompt-based and static UI manipulations. This dual coverage makes the benchmark both innovative and methodologically complete.
2. The inclusion of an automated LLM judge capable of analyzing full action trajectories and corresponding screenshots enables fine-grained, scalable assessment of perception, recognition, and reasoning failures.
3. The paper is clearly organized and well-written.
4. The benchmark encompasses a diverse set of applications across multiple mobile domains, ensuring that the evaluation results are representative and practically relevant.

### Weaknesses
1. The reliability and objectivity of results depend heavily on the LLM evaluator’s accuracy, yet potential biases or inconsistencies of this judge model are not rigorously validated.
2. The study focuses primarily on comparing different VLMs without analyzing how variations in agent frameworks may influence vulnerability to environmental injection.
3. The benchmark evaluation excludes specialized or fine-tuned VLMs (such as UI-Tars), leaving open the question of how domain-specific training affects robustness.
4. The paper mainly concentrates on attack evaluation and vulnerability analysis, with insufficient exploration of potential defense mechanisms or mitigation techniques.
5. Although the benchmark is well designed, it currently includes only 110 scenarios, which may limit the statistical diversity and coverage of real-world attack cases.

### Questions
What is the performance of various agent frameworks and agent-task-fine-tuned VLMs on the proposed benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3
