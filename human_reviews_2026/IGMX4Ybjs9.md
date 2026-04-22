# Investigating Advanced Reasoning of Large Language Models via Black-Box Interaction

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Existing tasks fall short in evaluating reasoning ability of Large Language Models (LLMs) in an interactive, unknown environment. This deficiency leads to the isolated assessment of deductive, inductive, and abductive reasoning, neglecting the integrated reasoning process that is indispensable for humans discovery of real world.
We introduce a novel evaluation paradigm, black-box interaction, to tackle this challenge.
A black-box is defined by a hidden function that maps a specific set of inputs to outputs. LLMs are required to unravel the hidden function behind the black-box by interacting with it in given exploration turns, and reasoning over observed input-output pairs.
Leveraging this idea, we build the Oracle benchmark which comprises 6 types of black-box task and 96 black-boxes. 19 modern LLMs are benchmarked. o3, a leading LLM from OpenAI, ranks first in 5 of the 6 tasks, achieving over 70\% accuracy on most easy black-boxes. But it still struggles with some hard black-box tasks, where its average performance drops below 40\%.
Further analysis indicates a universal difficulty among LLMs: They lack the high-level planning capability to develop efficient and adaptive exploration strategies for hypothesis refinement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a new evaluation framework, Black-box Interaction, aimed at assessing advanced reasoning abilities of large language models (LLMs) in interactive and verifiable settings. The authors introduce the ORACLE benchmark, consisting of six types of black-box tasks (e.g., code, physics, encryption, puzzle-solving, game strategy) where LLMs must explore, hypothesize, and refine their understanding of hidden rules through multi-round interactions. Experiments on 19 major models (OpenAI o3, Claude, Gemini, DeepSeek, Qwen3, etc.) reveal that even the strongest LLMs perform well on simple environments but struggle with adaptive planning and hypothesis revision in complex ones.

### Strengths
1. The ORACLE benchmark is broad and well-designed, covering multiple reasoning domains and allowing automatic evaluation.
2. The automatic generation pipeline for black-box environments offers scalability and reproducibility.

### Weaknesses
Conceptually, this paper shares nearly identical motivations as IDEA (https://aclanthology.org/2025.findings-acl.698/)—both examine how LLMs infer hidden rules from interaction—yet this paper does not discuss their relationship or provide comparative experiments.

### Questions
See Weaknesses

### Soundness
3

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
4

### Summary
This paper address the deficiency of existing LLM reasoning evaluations in assessing integrated reasoning within interactive, unknown environments, this paper introduces a novel "black-box interaction" paradigm. Based on this paradigm, the authors constructed the ORACLE benchmark, comprising 96 black-boxes across 6 task types, utilizing an innovative automated agentic framework for generation and scalability . Experiments benchmarking 19 LLMs revealed that even leading models struggle significantly in developing efficient and adaptive exploration strategies required to uncover complex hidden rules within the black-boxes . The research provides a valuable new tool for evaluation and highlights critical shortcomings in current LLMs concerning high-level planning and strategic adaptation based on feedback .

### Strengths
1. The study clearly identifies and provides compelling evidence for a critical limitation of current LLMs: their inability to devise efficient and adaptive exploration strategies for refining hypotheses.
2. The paper’s motivation—to evaluate an integrated reasoning process that combines abduction, deduction, and induction—aligns more closely with the complexity of human problem-solving than assessing these reasoning modes in isolation.
3. The paper introduces the novel concept of “black-box interaction,” which offers a fresh perspective distinct from existing static benchmarks, and presents an automated agentic framework developed specifically to generate the ORACLE benchmark.

### Weaknesses
1.  The ORACLE benchmark, relies on only 6 task types (CII, CRI, PSI, ERI, IPI, GSI) which are primarily formal or rule-based. This narrow focus raises concerns about whether performance on these "overly regularized" tasks adequately represents the broad spectrum of "advanced reasoning" required in diverse, real-world unknown environments. Furthermore, the criteria for distinguishing between "easy" and "hard" black-boxes are not explicitly defined and the adequacy of the baseline screening test (12 turns, 10 samples total) is questionable, especially given counter-intuitive results like GPT-4o failing while Gemini-2.0-flash passed potentially undermining the benchmark's ability to accurately stratify and evaluate model capabilities.
2. The paper relies heavily on an automated agentic framework using LLMs (Coding, Test, Refinement LLMs) for black-box creation, simulation, and debugging. While innovative for scalability, this introduces concerns about the consistency, uniqueness, and accuracy of the generated black-boxes. The framework itself might inherit biases from the LLMs used, potentially generating tasks that favor certain model architectures or knowledge domains. The paper lacks a rigorous analysis of this framework's reliability, potential biases, and how they might affect the benchmark's fairness and validity.
3.  The paper compellingly identifies a key weakness: LLMs lack efficient and adaptive exploration strategies. However, the analysis primarily demonstrates *that* this deficiency exists (e.g., through performance gains analysis and the comparative feedback experiment) rather than deeply investigating *why*. It lacks exploration into contributing factors such as limitations in long-context reasoning during extended interactions, ineffective processing or utilization of feedback signals, fundamental deficits in planning capabilities, or misunderstandings of the task objectives. The absence of metrics on token/turn efficiency also hinders understanding if better models achieve success through superior strategy or simply more computation/output. Moreover, the case studies don't explicitly show CoT reasoning during feedback analysis, raising questions about whether the experimental setup might inadvertently suppress models' analytical capabilities.

### Questions
1.  How does the agentic framework ensure the uniqueness, correctness, and lack of inherent bias in the generated black-boxes across different generation runs or different underlying LLMs? What steps were taken to validate the framework itself?
2.  What specific, objective criteria were used to classify black-boxes as "easy" versus "hard" across the 6 different task types? How was the consistency of this difficulty scaling validated?
3.  Is the baseline test setup (12 turns, 10 samples across 3 specific black-boxes) statistically sufficient to reliably filter models, especially given seemingly counter-intuitive outcomes like GPT-4o failing and Gemini-2.0-flash passing?
4.  What is the approximate performance level of humans (e.g., computer science students, domain experts) on the ORACLE benchmark tasks under similar constraints (turn limits)?
5.  Can the observed performance *decrease* for some models/tasks when increasing turns from 10 to 20 (as hinted in Figure 8 analysis  be attributed to random fluctuation, context window limitations, error accumulation, or other factors? What trends are observed if the number of turns is increased further (e.g., to 5, 15, 30 or 50)?
6.  Was the token usage or computational cost per turn analyzed? Do higher-performing models demonstrate greater efficiency in their exploration (i.e., achieving higher accuracy with fewer tokens or less verbose queries per turn)?
7.  In the comparative experiment on adaptive exploration, is there evidence (or lack thereof) that models *attempt* to analyze feedback even if their strategy doesn't change? Could the lack of explicit prompting for step-by-step reasoning *about* the feedback in the interaction loop negatively impact their ability to adapt their strategy?
8. Beyond identifying the lack of adaptive exploration strategies, what specific experiments or analyses could pinpoint the underlying causes (e.g., failure to integrate information over long histories, inability to form complex hypotheses, poor credit assignment from feedback, misunderstanding feedback signals)?
9. Recent works, such as KORGym, have demonstrated the lack of planning capabilities in models through multi-turn environmental interactions, yet relevant discussion is missing.

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
This paper introduces a new task for LLMs, called black-box interaction, to assess the exploration planning and reasoning ability in an unknown environment. The process mirrors C·P's framework of human reasoning—abduction (hypothesis formation), deduction (testing), and induction (refinement)—thereby evaluating reasoning as a dynamic and adaptive cycle rather than as isolated tasks.

Building on this paradigm, the authors develop the ORACLE benchmark, which includes 96 black-boxes spanning six task types: Code Intent Inference, Circuit Rule Inference, Physics System Inference, Encryption Rule Inference, Interactive Puzzle Inference, and Game Strategy Inference. Then they evaluate 19 modern LLMs.

The paper’s contributions are:

1. introducing the black-box interaction paradigm,
2. constructing the scalable ORACLE benchmark,
3. proposing an automated black-box generation framework, and
4. providing comprehensive analyses that expose current LLMs’ limits in advanced reasoning.

### Strengths
1. The black-box interaction paradigm is intuitive and novel.

2. The paper is well written and easy to understand.

3. Experiment details are shown in the appendix.

4. The analysis is valuable to share with the community, giving readers a better understanding of LLM reasoning.

### Weaknesses
1. My major concern is that, since the 6 types are hand-designed, how isolate the reasoning ability and the prior knowledge (like Circuit Rule).

2. The difficulty of the task is hard to understand. I think it is necessary to use some metrics to depict the difficulty of the tasks. (e.g., the minimal exploration steps for an expert to determine the answer)

### Questions
1. What is the accuracy of Random for each task?

2. L472, Figure 10, "Model and human performance ...", where is the human?

### Soundness
3

### Presentation
4

### Contribution
3
