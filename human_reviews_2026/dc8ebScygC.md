# Structured Uncertainty guided Clarification for LLM Agents

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 2

## Abstract
LLM agents with tool-calling capabilities often fail when user instructions are ambiguous or incomplete, leading to incorrect invocations and task failures. Existing approaches operate in unstructured language spaces, generating clarifying questions through prompting strategies that lack principled criteria for determining which questions to ask and when to stop. We introduce a principled formulation of \textit{structured uncertainty} that operates directly over tool parameters and their domains, cleanly separating specification uncertainty (what the user wants) from model uncertainty (what the LLM predicts). Our formulation uses Expected Value of Perfect Information (EVPI) to quantify the disambiguation value of each potential question, balanced against aspect-based cost modeling that prevents redundant questioning. We demonstrate the versatility of this formulation through two applications. First, SAGE-Agent uses structured uncertainty for inference-time question selection, achieving 7--39\% higher coverage on ambiguous tasks while reducing clarification questions by 1.5--2.7$\times$ compared to strong prompting and uncertainty-based baselines. Second, we show that structured uncertainty provides effective training signals: uncertainty-guided reward modeling boosts When2Call accuracy from 36.5\% to 65.2\% (3B model) and 36.7\% to 62.9\% (7B model) through uncertainty-weighted GRPO training, demonstrating more sample-efficient reinforcement learning for tool-calling agents. To enable evaluation, we present \textit{ClarifyBench}, the first multi-turn dynamic tool-calling disambiguation benchmark. Our results establish structured uncertainty as a principled framework that improves both inference-time interaction efficiency and training-time sample efficiency in tool-augmented agents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper’s general topic are tool-using agents that have to ask clarification questions about ambiguous queries. It is split in three parts: 1) A handcrafted inference strategy to decide which clarification question to ask based on how complete the best possible function call would be, 2) An evaluation dataset with ambiguous scenarios, and 3) an RL-trained agent. The authors find that their two agents work better than their respective baselines.

### Strengths
1. The general topic is highly important, and the dataset is useful for future research
2. I appreciate that the authors always report spans of results and show confidence intervals and noise, even when broad.
3. Both proposed methods look promising, but it is hard to say how robust their performance is due to the limited evaluation (see below)

### Weaknesses
Weaknesses, in order of magnitude: 
1. The paper seems to contain three parts that are only loosely connected by their general topic. For example, there is both the hand-engineered SAGE agent and then the GRPO trained agent in Section 6. They are similar in that they both use pi_i, but are never compared to one another or executed on the same benchmark dataset. 
2. I think the paper would greatly benefit from being split into multiple papers, because:
    1. As a dataset paper, the dataset construction would need to be more critically assessed, and should have a larger size than n=716. The construction of the ClarifyBench dataset is also just very high-level and not detailed, neither in Section 4.2 (“we design handwritten rules based on common API errors to create tool calls that would generate failures, followed by a similar LLM-based query augmentation process”), nor in the appendix.
    2. As a paper that proposes a new method (SAGE-Agent), the new method is only evaluated on one dataset (the dataset introduced above), and there is a second method intoduced (GRPO training on When2Call, and evaluated on a different dataset), plus the methods failure modes and when it does / does not work are not analyzed.
3. The probabilities pi_i, upon which both the SAGE-Agent and the GRPO training rest, are heavily hand-crafted and do not take into account LLM’s token uncertainties.
4. The expectation over a maximum in the EVPI objective creates a heavy inference-time compute burden, especially in less restricted domains, because we have to search over all questions, take an expectation over all follow-up answers, and then search the maximum pi_i. 
5. There are multiple readability issues. Normally I would note this as not influencing my score, but here it poses real difficulties to comprehend the paper. The paper could have a much greater impact if it benefitted from more time spent to make it readable.
    1. abbreviations are not introduced before usage (POMDP), 
    2. the contributions section in the introduction is hard to understand due to the non-motivated and non-explained terms (Bayesian Value of Information Objective, expected value of perfect information), 
    3. What are coverage rate, tool match rate, and parameter match rate (benchmark metrics in Table 3)
    4. observations_t in line 153 is undefined (and probably the same as obs_t)
    5. I suppose that line 117 is an indicator function?
    6. Cost(q) in line 177 is not defined until line 215
    7. There are Latex missing citations errors (line 181)
    8. Likewise, Algorithm 1 (called SAGE (final corrected version)) has multiple reference errors in the equations
    9. The reward function in line 177 is not used. A second reward is defined in line 400
    10. Citations are not in brackets and not hyperlinked, 
    11. Section 4.2: Data Augmentation has multiple grammar errors
    12. vspace around figures is sometimes very small and sometimes very large (Figure 2 vs Figure 4), 
6. I have doubts about the independence of the human annotators that ensure quality and naturalness of the dataset, given that it is “two graduate student annotators that were compensated following their respective graduate school policies” (Appendix B.2). To make the evaluation more rigid, I suggest to use independent raters (Mechanical turks and the likes), use multiple annotators per sample, and measure annotator disagreement.
7. It would be great to evaluate on more than one LLM (GPT-4o for SAGE-Agent and Qwen 2.5 3B/7B for the RL experiments). Especially in user simulation and interactive settings, different LLMs behave wildly differently. 


Smaller weaknesses that do not influence my score and do not need to be rebuttled, but I suggest fixing them for the revised version:

* It would increase readability if you could hyperlink your references (e.g., use the cleveref package)
* The caption of Table 2 is not a full sentence
* Algorithm 1 has multiple fonts
* Spelling error in Table 3: LLm → LLM

## Justification for the overall score

I believe that this paper addresses an important gap in the current research field, clarification questions for tool-calling agents. However, the paper in its current form is not clear and reproducible, and strechted too thinly across three subtopics to discuss either of them in detail. It reads more like three workshop short-papers in the current form. I recommend to reject the paper at this point, but believe that the authors are working on something that is promising. I encourage the authors to disentangle their three subcomponents, and focus on one of them fully, in order to have the capacity to analyze and discuss it in detail.

### Questions
1. Could you explain the argument you make in lines 131-136? Why do other approaches _have to_ take this route to model ambiguity, and why is your approach not influenced by model (/epistemic) uncertainty? Your proposed uncertainties can easily break if the model makes a wrong prediction, this is assumed away if I understand correctly?
2. Can you discuss the difference between the Expected Value of Perfect Information and the more commonly used Expected Information Gain objective, and why the former corresponds to “optimal” (line 182) question selection in your opinion? Can you benchmark against Expected Information Gain? 
3. By “aspect” a_i,j, you just mean the identifier (i, j) of some theta_i,j, is that correct? 
4. Why is your reward self-calibrating? (line 405) How does this relate to calibration, is it a proper scoring rule?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SAGE-Agent, a framework that uses structured uncertainty and Expected Value of Perfect Information to help LLM agents ask optimal clarification questions before taking actions.
It also presents ClarifyBench, a new benchmark for evaluating tool-augmented agents under ambiguous or infeasible user requests across multiple domains. 
Experiments show that SAGE-Agent significantly improves task success rates and reduces unnecessary clarifications compared to strong LLM baselines.

### Strengths
- ClarifyBench fills a gap in existing evaluations by supporting dynamic user simulation, multi-turn requests, and infeasible queries across diverse domains (documents, vehicle control, stocks, travel, and file systems).
- The structured uncertainty approach also serves as an effective reward signal, improving sample efficiency and performance on unrelated tasks
- The paper is thorough, with detailed algorithmic design, theoretical proofs, and practical implementation notes.

### Weaknesses
- While ClarifyBench is valuable, the user simulator relies on LLM-generated interactions, which may not fully capture human ambiguity or pragmatic nuances.
- EVPI computation scales with candidate size and domain dimensionality. Though approximations are discussed, concrete runtime or complexity comparisons with simpler heuristics are missing.

### Questions
- The paper relies on an LLM-based user simulator to model realistic conversational progression. How do the authors ensure the accuracy and reliability of this simulator’s responses compared to real human interactions?
- It would be helpful to include an ablation study on key hyperparameters to better understand their influence on model performance and stability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
They propose SAGE: keep a belief over structured tool-call candidates and choose clarifying questions by Expected Value of Perfect Information (EVPI), with a cost for redundancy. They also introduce ClarifyBench. (Their own text sets up the belief factorization and EVPI definition.They use RL with GRPO to train QWen model and show improvements over baselines on their own benchmark.

### Strengths
1. Separation of uncertainties. They argue against using model generated response for uncertainty quantification which is a reasonable structural move.
2. A potentially useful benchmark, ClarifyBench which covers several tool domains with explicit/ambiguous/infeasible splits and reports basic stats.

### Weaknesses
1. You make strong assumptions which are not validated in the work. The viability score assumes 1) a uniform prior over tools 2) naive independence across paramters and 3) an arbitrary $\epsilon$  for continuous domains. Additionally, there is no sensitivity analysis.
2. While they compare against a few agent baselines, there’s no demonstration on widely used external tool-use suites (tau-bench, etc). Moreover, the dataset is llm augmented using GPT-4ofor  query generation/obfuscation.
3. They introduce a certainty-weighted reward, which by construction favors their belief-based approach. The paper doesn't isolate how much of the gain comes from reward shaping vs the EVPI policy itself.
4. There is no minimal, prompting baseline where you have something like "ask only for missing required parameters, no repeats" and no ablation showing the incremental value of EVPI vs simple heuristics.
5. Works/tools like GenieWorksheets [1], and MCP automatically take care of these aspects. If the LLM partially fills an API, they generate an `ask_question()` like function which asks for unfilled parameters. 
6. The presentation of this manuscript, needs a lot of work. The mathematical notations are not clear and for some scenarios are seem forced.
7. Beyond simulator metrics (TMR/PMR/CR), can you report a small human study on question helpfulness and over-questioning, especially in ambiguous cases?
8. Seems like SAGE only adds a scaffold around llm proposed candidates and questions, you’re still depending on the LLM to decide the potential candidates. You have just made the use of those proposals safer and more auditable.

[1] Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets (Joshi et al., ACL 2025)

### Questions
1. It is unclear to me why do you want domains:   Di = {Di,1, Di,2, . . . , Di,mi } where Di,j is the domain of parameter θi,j -- why do you need domain of parameter? what does domain even mean?
2. Please define what is $u$ before Line 131.
3. In line 166, State Space: S = {(Ti, θi) : Ti ∈ T , θi ∈ Di} represents true user intent -- why is theta_i \in D_i?
4. GPT generated equations and notations. eg line 177, gpt generates this for classification. 
5. Missing citation on line 181
6. Please fix citation formatting.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors present a dataset and method for developing LLMs that can ask clarifying questions to disambiguate user requests in tool-calling dialogues. Their dataset consists of automatically perturbed examples sourced from another dataset, DocPilot, using GPT to introduce ambiguity into the example. Examples are then verified by a human annotator. The authors then propose a method for training an LLM to engage in such dialogues with users, which is based on estimating the the benefits of clarifying the user query and weighing this against the cost. The authors then compare their proposed method on their proposed benchmark against several prompt-based baselines, demonstrating gains.

### Strengths
This work provides a dataset for studying the intersection between ambiguity and tool use settings, which as of yet is underexplored and presents many novel challenges.

### Weaknesses
1. Presentation is poor. The task itself from the constructed dataset, while it is sourced from an existing work, is not actually explained. Variables and acronyms are frequently not defined, or are defined in unintuitive places like figure captions. Citations are incorrectly formatted.

2. The authors compare exclusively against prompt-based baselines. Several more comparable and competitive methods should be compared against or discussed in the least, even if it's just a simple SFT baseline. Additional related methods to be discussed or compared against include may include others that utilize GRPO/PPO/DPO training for similar clarify or execute decisions in user-llm dialogues.

CollabLLM: From Passive Responders to Active Collaborators
Shirley Wu, Michel Galley, Baolin Peng, Hao Cheng, Gavin Li, Yao Dou, Weixin Cai, James Zou, Jure Leskovec, Jianfeng Gao
ICML 2025

Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying Questions
Michael J.Q. Zhang, W. Bradley Knox, and Eunsol Choi.
ICLR 2025

Learning to Clarify: Multi-turn Conversations with Action-Based Contrastive Self-Training
M. Chen, R. Sun, T. Pfister, S.O. Arik
ICLR 2025

3. The validity of the dataset is unclear. While the authors say that examples are validated by a human annotator, it's unclear how reliable this is or what agreement on validation would be. Almost all of the results are based on this dataset as well, so looking at other tasks/settings would also help substantiate the method.

### Questions
1. Could you elaborate on the PII from the source dataset that is being filtered out? Is the PII potentially harmful?

### Soundness
2

### Presentation
1

### Contribution
2
