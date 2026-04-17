# Policy-Based Sentence Simplification: Replacing Parallel Corpora with LLM-as-a-Judge

- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Sentence simplification aims to modify a sentence to make it easier to read and understand while preserving the meaning. Different applications require distinct simplification policies, such as replacing only complex words at the lexical level or rewriting the entire sentence while trading off details for simplicity. However, achieving such policy-driven control remains an open challenge. In this work, we introduce a simple yet powerful approach that leverages Large Language Model-as-a-Judge (LLM-as-a-Judge) to automatically construct policy-aligned training data, completely removing the need for costly human annotation or parallel corpora. Our method enables building simplification systems that adapt to diverse simplification policies. Remarkably, even small-scale open-source LLMs such as Phi-3-mini-3.8B surpass GPT-4o on lexical-oriented simplification, while achieving comparable performance on overall rewriting, as verified by both automatic metrics and human evaluations. The consistent improvements across model families and sizes demonstrate the robustness of our approach

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a method to control the quality of type of edits in sentence simplification without relying on human-annotated corpora. The approach is to use LLMs both generate candidate simplifications and to judge pairwise preferences, then train a decoder‑only policy model with preference optimization (ARPO). The authors design prompt-based guidelines for rewards/penalties for specific operations, where OTAlign is used as lexical judge and Qwen is used as structural judge.

### Strengths
1. The paper is well written and easy to follow.
2. The overall pipeline is conceptually simple and can be adapted to different edit policies and deployment scenarios.
3. Experiments consider multiple baselines and include both automatic and human assessments.
4. Aligning small, open‑source LLMs with this policy framework can match or exceed closed-source LLMs on both automatic and human metrics.

### Weaknesses
1. The scope is limited to sentence-level simplification, which I think underutilizes current LLM capabilities. Meanwhile, real‑world needs often require paragraph or document‑level simplification with cross‑sentence constraints and dependencies.


2. The system’s practical value remains unconvincing to me. The approach still focuses on generic sentence‑level edit operations, whereas special groups of users, e.g., people with reading impairments, need solutions tailored to their specific difficulties. A user‑focused design with explicit needs analysis would strengthen the contribution.

3. The desirability of the simplified sentences is judged by LLMs, which is unclear to me as to how it benefits human users or human preference can be effectively taken into account.

### Questions
1. What are the prompts for the vanilla base models? (I could not find them reported in the Appendix). How does the proposed policy compare directly to carefully engineered prompts (with iterative feedback) that request specific edit types, both in quality and controllability?

2. What is the performance on out‑of‑domain sentences (e.g., different genres, technical vs. news)?  Is retraining required, or can the policy adapt via light tuning in these scenarios?

3. Can the framework be extended to paragraph/document‑level simplification? What changes are needed to handle discourse relations, coreference, and global coherence?

### Soundness
3

### Presentation
4

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
In this manuscript, the authors aim to address the challenges of policy-driven control in sentence simplification, including the lack of policy-specific parallel corpora and the poor policy alignment of small-scale open-source LLMs. The proposed LLM-as-a-Judge automatically constructs policy-aligned preference data and uses preference optimization to fine-tune open-source LLMs. Experimental results across automatic metrics and human evaluations show that the method outperforms baselines.

### Strengths
In this manuscript, the authors aim to address the challenges of policy-driven control in sentence simplification, including the lack of policy-specific parallel corpora and the poor policy alignment of small-scale open-source LLMs. The proposed LLM-as-a-Judge automatically constructs policy-aligned preference data and uses preference optimization to fine-tune open-source LLMs. Experimental results across automatic metrics and human evaluations show that the method outperforms baselines. Notably, small-scale open-source LLMs (e.g., Phi-3-mini-3.8B) fine-tuned via this framework surpass GPT-4o in lexical-paraphrasing and achieve comparable performance in overall-rewriting, verifying the approach’s effectiveness and robustness.

### Weaknesses
There are some concerns for the manuscript as follows:
1.The proposed LLM-as-a-Judge select high-quality preference data between overall-rewriting and lexical-paraphrasing policies. It means that the candidate simplifications will be generated by various LLMs, so what is the computational cost of this method? This is an important issue, but it is not discussed in the experiments.
2.The proposed LLM-as-a-Judge selected preference data and fine-tune open source LLMs, the important preference optimization is an existing method. Overall, the innovation of the proposed method is limited. 
3.In Section 2, how to fine-tune open-source llms is not mentioned.
4.The selection for baseline methods are inappropriate. The sota sentence simplification have not been introduced.

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper focus on training small LLMs doing sentence simplification. The method involves sampling simplification outputs from different models and then use LLM as judge to construct preference pairs, and then apply preference learning algorithm.

### Strengths
The paper is clear

### Weaknesses
The whole pipeline of constructing preference pairs and apply the preference learning algorithms has been used in the past 2 years since the RLHF came out. For example, "Self-Rewarding Language Models" ICML 2024. And the paper just apply it onto a specific task.

The paper use the GPT-4o as an upper bound, but it is a quite old model, should have the evaluation of the current best propreitary model (GPT-5)'s performance on sentence simplification, which is likely a solved task. And GPT-5 is also cheaper than GPT-4o.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors describe an approach for adapting LLMs used for sentence simplification (in English only) to specific simplification policies. They rely on LLM-as-a-judge to avoid the need for policy-specific reference corpora. They show that their approach applied on mid-size LMs outperforms large LMs used in few-shot settings.

### Strengths
- convincing motivation for the work, showing the need for an approach that goes beyond dedicated seq-to-seq models as well as beyond large-scale LLMs, and that allows for adapting to a specific simplification policy
- state-of-the-art preference optimisation approach combining ARPO and SimPO
- strong baselines/toplines
- good (and somehow reassuring) results

### Weaknesses
- no real weakness for me, this is a good paper
- one detail: the fact that SARI has been shown to only weakly correlate with human judgement could be mentioned, and the use of SARI nevertheless could be motivated—this is one of the reasons why human evaluation is important, which the authors include in their paper
- another detail: the link between use cases and possible policies could be discussed a bit more (e.g. what about simplification targeted towards people with cognitive disabilities? when is sentence splitting necessary and when is it less useful? and so on)

### Questions
- am I right when I say that you use LLMs as baselines/toplines in a few-shot setting (using the prompt given in Figure 5 in the appendix)? Could you make this clearer in the main part of the paper?

### Soundness
4

### Presentation
3

### Contribution
4
