# Clarifying Before Reasoning: A Coq Prover with Structural Context

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
In this work, we investigate whether improving task clarity can enhance reasoning ability of large language models, focusing on theorem proving in Coq. We introduce a concept-level metric to evaluate task clarity and show that adding structured semantic context to the standard input used by modern LLMs, leads to a 1.85$\times$ improvement in clarity score (44.5\%~$\rightarrow$~82.3\%). Using the general-purpose model DeepSeek-V3, our approach leads to a 2.1$\times$ improvement in proof success (21.8\%~$\rightarrow$~45.8\%) and outperforms the previous state-of-the-art Graph2Tac (33.2\%). We evaluate this on 1,386 theorems randomly sampled from 15 standard Coq packages, following the same evaluation protocol as Graph2Tac.
Furthermore, fine-tuning smaller models on our structured data can achieve even higher performance (48.6\%).
Our method uses selective concept unfolding to enrich task descriptions, and employs a Planner-Executor architecture. These findings highlight the value of structured task representations in bridging the gap between understanding and reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a method to incorporate semantic context into theorem proving models. They propose a clarity score to evaluate the understanding of this context. They demonstrate that this clarity helps downstream performance in proving theorems.

### Strengths
The authors reach a new SOTA on theorem proving using an LLM. They define a clarity metric which seems interesting and not explored enough before in prior work, and they demonstrate that their setup can improve this metric. They also show the strong correlation between their metric and theorem proving capability.

### Weaknesses
It seems to me that this work is a simple variation of existing work on enhancing the semantic context for theorem proving, and especially premise selection. The authors introduce three ways of enhancing the context for proving: (1) entity extraction, (2) proof state extraction, and (3) domain-specific tokenization. It seems to me that (1) is a subset of previous work in premise selection (adding the information of related definitions / theorems to the context) such as in Graph2Tac, and contextual information like in Baldur and miniCTX. For (2), if I understand correctly, it is the standard data extraction for all tactic-level provers. (3) is basically already done in previous work like Graph2Tac, where global definitions are resolved, and seems like a small syntactic change to me.

The presentation of this paper is puzzling for me. While the writing is easy to follow, the authors use a significant portion of the paper to introduce surrounding concepts like Coq’s type theory, definitions, and proof state and tactics, which should already be familiar to the general reader. On the other hand, the core method is never presented in the main text, and I had to read through the long appendix to partially understand what exactly is input to the LLM. I encourage the authors to replace vague bullet points and task descriptions with concrete explanations or examples of what exactly the input to the LLM is.

### Questions
Can the authors clearly describe what exactly is input to the LLM prover (e.g. with an example) using their setup of enhancing context clarity, and how does it differ from existing setups like premise selection?

What is "Chinese Translation" in Table 1?

The authors mention in the abstract and introduction they have a "selective" concept unfolding technique, but I could not find it in the methods or appendix. What is it and where is this defined?

In the writing, there are some parts that are clearly redundant and possibly LLM-written. There are many bullet points that have vague / unnecessary / incorrect headlines. For example the bullet point on L798 repeats L805. "Semantic Foundation" on L348 looks like a completely misplaced phrase.

### Soundness
3

### Presentation
2

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
This paper investigated the relationship between a model's conceptual understanding and its reasoning performance in mathematical theorem proving. It proposes a new metric, the clarity score, that evaluates how well a model understands a task-related concept. Building on this, it further proposes a planner-executer pipeline that uses a general-purpose LLM to first identify relevant concepts and strategies, then translates strategies into Coq tactics. Empirically, the paper presents an analysis of clarity under different prompt configurations, demonstrates correlation between clarity and proof success, and shows that incorporating structured semantic context improves theorem-proving performance over prior methods.

### Strengths
- The contribution of introducing conceptual clarity as a metric is interesting. 
- Empirical results are consistent. Authors show improved success rate on tasks over the baseline methods, as well as a light-weighted fine-tuned model achieving the highest success rate (matching the result on the non-fine-tuned general-purpose model).

### Weaknesses
- The paper's methodology is difficult to follow. While the introduction emphasizes that one of the key contributions is enhancing task understanding in general-purpose models through enriched task descriptions, the corresponding methodological section (Section 5) lacks clarity. For instance, the presentation of the “structured semantic context” pipeline is difficult to follow, with important details scattered or deferred to the Appendix. A substantial reorganization and clearer exposition would be needed to make this contribution accessible and verifiable. Phrases such as "additional enhancements", "surface and internal representation" remain vague without concrete examples. 
- My second concern is whether the correlation shown in Table 3 is because "understanding improvements directly drive reasoning performance" or simply a correlation (not causation) because we are working with richer inputs (and the model gets more signals or cues). This, therefore, sounds over-claiming without clearer evidence. Additional clarification or an additional ablation study would help.

### Questions
- Have the authors conducted any experiments designed to test whether improvements in clarity score cause higher reasoning success, rather than merely correlate with it?
- It seems that the LLM judge for clarity score measurement uses the same model as the planner-executer pipeline (Deepseek-V3), so I'm wondering whether there would be a bias towards the measurement (e.g., evaluator might favor its own generation, etc.)

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes enhancing theorem proving in Coq by introducing structured semantic context extracted from Coq’s internal type system, combined with a Planner–Executor architecture. A new metric, Clarity Score, is introduced to quantify how well a model “understands” a task. The authors claim that increasing clarity leads to a proportional improvement in theorem-proving.

### Strengths
1. The central idea—enhancing reasoning by improving task clarity—is an appealing direction beyond model scaling and reinforcement learning.
2. The proposed Coq compiler interception pipeline that captures internal representations is a technically valuable contribution, potentially useful for future formal reasoning datasets.
3. This paper introduces a new metric to quantify how well a model “understands” a task.

### Weaknesses
1. The “Clarity Score” uses the same model (DeepSeek-V3) as both generator and evaluator, introducing self-evaluation bias.
No external or human validation is performed, so the metric may not measure conceptual understanding.
2. The comparison with Graph2Tac is not entirely fair, as Graph2Tac is a much smaller GNN-based model while this paper employs a large-scale LLM (DeepSeek-V3). Although the authors claim it is a general-purpose model not specifically trained on mathematical data, its scale and pretraining corpus far exceed those of the baseline. Moreover, fine-tuned single models like Qwen-2.5-7B or 32B might yield comparable or even better results. Hence, the reported performance gains may stem from model capacity rather than the proposed clarity enhancement.
3. The Planner–Executor architecture is incremental and mirrors prior systems such as Lean-Star (2024) and Apollo (2025). Apart from the proposed metric, the contribution is mainly an engineering integration of structured context.

### Questions
Please refer to the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a novel approach to LLM-based theorem proving in Coq, built on the hypothesis that enhancing task clarity is a distinct and important step for improving reasoning. The authors present: 1) a "Clarity Score" metric to quantify a model's understanding of formal concepts; 2) a data pipeline that extracts "structured semantic context" from the Coq compiler's internal representations; and 3) a Planner-Executor architecture that leverages this structured data.

Using this method, the authors claim to more than double the proof success rate of a general-purpose model (DeepSeek-V3), outperforming the previous state-of-the-art, Graph2Tac. While the core hypothesis is strong and the technical execution is impressive, the evaluation remains fairly narrow to support the SOTA claims. The comparison hinging on a single baselines, and only Coq programs is the most notable limitation.

### Strengths
This was a refreshing paper to read, with genuinely novel approaches, and well thought out experiments. The core idea of separating and measuring "task clarity" as a distinct bottleneck from "reasoning" is a good contribution. The data processing pipeline, involving modification of the Coq compiler to extract internal type-theoretic information, is a non-trivial and valuable piece of engineering. The fine-tuning results (Tab 5) are highly valuable, showing a 32B parameter model achieving 48.6% success (outperforming a 671B model), demonstrating the data's quality. The Planner-Executor design is a natural fit for leveraging the extracted semantic context.

### Weaknesses
The following points are constructive feedback; I will increase my rating if the key experiments are conducted during the rebuttal.
1. The SOTA claim (45.8% vs 33.2%) rests on a comparison against one baseline (Graph2Tac). Would you be able to show via a pilot evaluation in LEAN, a direct comparison or against other provers mentioned in the related work, such as DeepSeek-Prover, Kimina-Prover, and Llemma. 
2. The evaluation is on a 10% random sample of a benchmark (is it 1,300 or 1,386 theorems). This prevents replication and rigorous comparison. The paper should evaluate on a full, standard benchmark (e.g., the full Graph2Tac test set, PACT).
3. Limited Scale of Core Claims:
    * The central correlation claim (Tab 3, r=0.98) is based on only n=100 theorems. This is okay robust enough but ideally must be re-run on the full dataset, if cost permits.
    * The architectural ablation (Table 8), claiming a +24% gain, is based on only n=78 theorems from a single library. Can you do better?
4. The "Clarity Score" is evaluated using DeepSeek-V3 as the judge (line 266), which is the same model used in the main experiments. LLMaj can be influenced by this circularity and this may introduce potential bias. Would you be able to try it with another family, e.g. Qwen?

### Questions
Please see weaknesses above.

### Soundness
4

### Presentation
3

### Contribution
3
