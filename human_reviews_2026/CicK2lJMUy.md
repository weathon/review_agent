# Learning to Reason in Structured In-context Environments with Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Large language models (LLMs) have achieved significant advancements in reasoning capabilities through reinforcement learning (RL) via environmental exploration. As the intrinsic properties of the environment determine the abilities that LLMs can learn, the environment plays an important role in the RL finetuning process. An ideal LLM reasoning environment should possess three core characteristics: scalability, generalizable reasoning, and verifiability. However, existing mathematical and coding environments are difficult to scale due to heavy reliance on expert annotation, while the skills learned in game-based environments are too specialized to generalize. To bridge this gap, we introduce the \textbf{S}tructured \textbf{I}n-context \textbf{E}nvironment (SIE) framework. SIE achieves scalability by automatically constructing reasoning environments from large-scale structured data, where the rich compositional patterns naturally support generalizable reasoning. Moreover, the explicit schemas and reasoning chains in structured data provide a foundation for rule-based verifiability. Experimental results show that the SIE framework not only achieves substantial improvements in in-domain structured reasoning, but also enables the learned compositional reasoning skills to generalize effectively to out-of-domain mathematical and logical reasoning tasks. We further explored learning in information-limited partial SIEs and found that LLMs can infer the missing information through exploring the environment, leading to robust reasoning improvements and generalization performance. Our code can be available at \url{https://github.com/PursuitYP/SIE_ICLR}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a reinforcement learning framework that trains large language models to reason within automatically constructed Structured In-context Environments (SIEs). Each SIE is a mini structured world built from knowledge graphs, containing supporting and distractor facts that the model must navigate to answer questions. The authors also introduce a Structured Reasoning Dataset (SRD)—a supervised corpus generated from the same environments—to warm up the model before RL. Experiments on KGQA, math, and logic reasoning show that RL in SIEs substantially improves compositional reasoning and robustness to incomplete information.

### Strengths
The paper presents a clear pipeline for automatically generating verifiable reasoning environments via embedding the structural graphs into the context of LLMs. The idea is intuitive and the results demonstrate strong gains on structured reasoning benchmarks and generalization to math and logic tasks.

### Weaknesses
The idea of embedding structured as reasoning context is interesting but not conceptually new. I believe there are lots of related works on representing the structural data in the context of LLMs. It would be better for the authors to discuss them.

### Questions
See Weaknesses

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
4

### Summary
The paper proposes building RL fine-tuning "environments" by packaging KG subgraphs into the prompt, then training LLMs with GRPO using an exact-match/verifiable reward. Construction pulls a seed subgraph from both the question and the answer entities, extracts shortest-path “support” and varies difficulty by hiding support triples. On benchmarks like WebQSP and GrailQA, SIE-RL shows huge gains over the non RL'd baseline and strong OOD benefits on GSM8K and logic puzzles is also reported.

### Strengths
1. Interesting concept of creating novel views of standard KGs to RL fine-tune LLMs.
2. Strong results over baselines (non finetuned) are interesting to see.

### Weaknesses
1. The intro describes the recipe in too abstract a manner and hypes/overstates the generality of the approach. It is not really an "environment" in the sense of RL and MDPs. Significant LLM usage is acknowledged, however.
2. Reranking with a cross-encoder risks leakage.
3. Huge "SIE-0%" accuracy values suggest that largely using the model’s background knowledge is sufficient to ace the benchmarks, undermining the centrality of the environment. Output format adherence could also explain OOD results. Could this be ablated/tested in isolation somehow?
4. The evaluation setup bakes in supervision (answer-side retrieval) and potentially format adherence for small models that could significantly explain differences in the results. 
5. Lack of results for larger models. Only models up to 8b are used where fine-tuning can significantly improve performance. Comparisons also omit strong tool-using KGQA agents.

### Questions
1. For the re-ranker cross-encoder leakage risk, could you report results with/without reranking? Also maybe use structure-only distractors?
2. There is also risk for contamination for SRD via the Deepseek R1 API. What do you think?

### Soundness
3

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
The paper proposes the Structured In-context Environment (SIE) framework, which builds scalable, verifiable reasoning environments for RL post-training directly from large structured data (instantiated with knowledge graphs). SIE is inserted as a structured context in the prompt—so the model’s “actions” are its in-context explorations—and rewards are rule-based (exact-match answer and formatting), optimized with GRPO (and also tested with PPO/REINFORCE++). An automated pipeline retrieves a compact seed subgraph, extracts shortest-path “support” subgraphs, semantically filters distractors, and constructs partial SIEs by controlling the retained support ratio while fixing context length. Across Qwen and Llama backbones, RL in SIE yields large gains on structured reasoning benchmarks (WebQSP, CWQ, GrailQA) over CoT, SFT on distilled structured reasoning data, and RL without context, and the learned compositional skills transfer to out-of-domain math and logic tasks.

### Strengths
The paper is original in how it reframes reasoning RL as an environment-design problem: it turns large structured data into *Structured In-context Environments (SIEs)* so that reasoning happens via in-context exploration with verifiable, rule-based rewards, rather than only via prompts or dataset distillation. The automated pipeline—seed subgraph retrieval, shortest-path “support” extraction, semantic distractor filtering, and construction of *partial* SIEs that control information content—both grounds the idea technically and makes it scalable; The empirical quality is strong and maintains sizable advantages even when support evidence is aggressively removed, demonstrating robustness to incomplete information. The significance extends beyond KGQA: the learned strategies transfer to out-of-domain math and logic tasks, indicating that SIEs elicit compositional skills rather than overfitting to a specific graph schema.

### Weaknesses
The out-of-domain math evaluation focuses on GSM8K and MATH500, which leaves open whether the learned strategies carry to the current hardest math leaderboards; adding AIME 2024/2025 and Olympiad-style sets would provide a more stringent test of compositional reasoning under scarce-signal regimes.  Although the paper includes Llama3.1-8B-Instruct both in-domain and for OOD transfer, the results are not shown;  Finally, given recent concerns that Qwen-family models can improve under weak or even randomized reward shaping, the study would be stronger with explicit controls: reward-shuffled and random-reward baselines.

### Questions
see more details in the weakness.

### Soundness
2

### Presentation
3

### Contribution
3
