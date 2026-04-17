# Search Self-Play: Pushing the Frontier of Agent Capability without Supervision

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has become the mainstream technique for training LLM agents. However, RLVR highly depends on well-crafted task queries and corresponding ground-truth answers to provide accurate rewards, which requires significant human effort and hinders the scaling of RL processes, especially in agentic scenarios. Although a few recent works explore task synthesis methods, the difficulty of generated agentic tasks can hardly be controlled to provide effective RL training advantages. To achieve agentic RLVR with higher scalability, we explore self-play training for deep search agents, in which the learning LLM utilizes multi-turn search engine calling and acts simultaneously as both a task proposer and a problem solver. The task proposer aims to generate deep search queries with well-defined ground-truth answers and increasing task difficulty. The problem solver tries to handle the generated search queries and output the correct answer predictions. To ensure that each generated search query has accurate ground truth, we collect all the searching results from the proposer's trajectory as external knowledge, then conduct retrieval-augmentation generation (RAG) to test whether the proposed query can be correctly answered with all necessary search documents provided. In this search self-play (SSP) game, the proposer and the solver co-evolve their agent capabilities through both competition and cooperation. With substantial experimental results, we find that SSP can significantly improve search agents' performance uniformly on various benchmarks without any supervision under both from-scratch and continuous RL training setups. The code is at https://github.com/Qwen-Applications/SSP.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Search Self-Play (SSP), a novel reinforcement learning framework aimed at improving deep search agents without human supervision. SSP involves two roles for a single LLM: a task proposer that generates challenging search queries, and a problem solver that attempts to answer these queries using multi-turn search interactions. The framework ensures the correctness of queries by employing a retrieval-augmented generation (RAG) process to verify ground-truth answers. Experimental results demonstrate that SSP improves agent performance across various benchmarks, including NQ, TriviaQA, and HotpotQA, both in from-scratch and continuous training setups.

### Strengths
This paper makes a significant contribution by introducing SSP as a self-supervised training method for search agents. The originality lies in the combination of search agent roles with self-play, where the proposer and solver co-evolve. This approach represents a promising direction for training LLMs autonomously and addresses key challenges such as task synthesis and scalability. The empirical results across a diverse set of benchmarks, particularly with the SSP framework applied to various LLMs, demonstrate the method's potential for future agentic training.

### Weaknesses
* The method samples ground-truth answers from a pre-defined set (D) to drive self-play but does not disclose (D)’s source, size, or curation, nor its overlap with evaluation answers; this raises potential data-leakage/overfitting concerns. Please report (D)’s provenance, scale, overlap with each benchmark, and blacklist/de-dup procedures.

* Using a solver **without tools** as the verifier is an unreliable proxy for question validity: failures may reflect the verifier’s limited contextual/multi-hop reasoning (or added RAG noise) rather than flaws in the proposed question, causing valid items to be wrongly rejected.

* Evaluation relies on a single judge (Qwen2.5-32B-Instruct) with pass@1 while several tested systems are Qwen-family models, inviting family bias and prompt-format gaming; adding exact match (EM) and F1 would provide more robust evidence.

* Strong results on NQ/TriviaQA/HotpotQA/2Wiki/MuSiQue/Bamboogle notwithstanding, coverage skews to classic QA; testing on newer search benchmarks (e.g., BrowseComp and SimpleQA) would better demonstrate robustness.

### Questions
* Clarify how the answer set (D) is built (source, size, type distribution) and report overlap stats with each benchmark, plus de-dup/blacklist procedures.
* Add EM/F1 alongside pass@1, or use heterogeneous judges from different families, and report inter-judge agreement.
* Expand evaluation to newer search/browsing benchmarks (e.g., BrowseComp; SimpleQA if applicable).
* Why is there no reward for SSP Proposer in Figure 3(a)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Search Self-Play (SSP), a reinforcement learning framework where LLM takes on two alternating roles: a “proposer” that creates challenging, search-intensive QA tasks, and a “solver” that attempts to answer them. To ensure the generated questions are valid and have a unique, verifiable answer, a RAG-based verification gate is used, which relies on the proposer’s retrieved documents to validate the questions before adversarial training begins. The solver is trained using GRPO, while the proposer is optimized with REINFORCE. Experiments across seven open-domain QA benchmarks and various model architectures and sizes consistently demonstrate that SSP delivers significant performance improvements.

### Strengths
1. This work is novel in its application of self-play to agentic search.

2. The paper provides precise definitions of constraints and rewards, fully discloses prompts and hyperparameters, and includes training curves that illustrate the co-evolution of proposer and solver.

3. By reducing reliance on annotated agentic data, SSP achieves consistent, substantial improvements across different model architectures and scales, paving a practical path toward scalable, unsupervised training for agentic tasks.

### Weaknesses
1. Please discuss training stability under the adversarial/cooperative setup. Does the min–max dynamic cause frequent collapses or mode drift? How easy is convergence in practice? Please provide stability evidence (multiple seeds, valid-question rate over time, reward variance) and, if available, theoretical or empirical guarantees that prevent reward hacking and proposer entropy explosions.

2. Please report end-to-end compute and cost: total wall-clock, GPU hours, tokens processed, search calls, and energy. If SSP is substantially more resource-intensive, justify why not to to paid human annotation via crowd platforms.

3. Clarify the rationale for using REINFORCE as the proposer’s advantage estimator and GRPO for the solver. Why is this asymmetric choice preferable, and how do performance/variance trade-offs compare to alternatives (e.g., PPO, GPRO, REINFORCE)?

4. Evaluation heavily on a single LLM-as-judge and 500-sample subsets, which may introduce bias and variance.

### Questions
5. Provide results using real-world web content (e.g., Bing/Google APIs) and different corpora/embeddings retrievers.

6. Improve Figure 2 to more clearly reflect the pipeline. The current Figure obscures the implementation details. Consider redrawing it to mirror Algorithm 1 step-by-step (proposer rollout → rule checks → RAG verification with noise → solver rollouts → rewards/updates).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces Search Self-Play (SSP), a reinforcement learning framework for training deep search agents without human supervision. The approach uses a single LLM in dual roles: as a question proposer that generates search queries with verifiable ground-truth answers, and as a problem solver that attempts to answer these questions through multi-turn search interactions. The key component is a RAG-based verification mechanism that ensures the proposed questions are answerable given the proposer's retrieved documents, preventing reward hacking. The authors formulate this problem as a cooperative competition scenario, where the proposer and solver collaborate during the RAG-based verification but compete through the proposer generating increasingly difficult questions and the solver learning to get better at answering difficult questions. 

Experiments demonstrate consistent improvements across seven QA benchmarks. The work addresses a significant bottleneck in RLVR (Reinforcement Learning with Verifiable Rewards): the need for large-scale, human-curated question-answer pairs, through a self-sustaining closed learning loop.

However, the evaluation is limited to a Wikipedia-based retrieval with a 2018 corpus, raising questions about generalization to modern web search scenarios. Note that contemporary works have also used a similar evaluation strategy in the past. The paper lacks analysis on question quality evolution, computational costs, and comparison with recent synthetic data generation methods. The RAG verification mechanism increases the computational complexity of the method, warranting a comparison of training costs with prior works.

### Strengths
1. Novel Problem Formulation: The paper addresses a fundamental challenge in agentic RL training, Data Scarcity, through an elegant self-play mechanism that grounds question generation in external search rather than relying solely on the model's internal knowledge.
2. Thorough Ablation Studies: The paper provides valuable insights through ablations on training schemes (co-evolution vs fixed-opponent), batch completion strategies, RAG verification configurations, and reward design, with detailed training dynamics analysis.
3. Strong Empirical Results: The proposed method shows consistent improvements across all seven benchmarks, with particularly impressive gains on base models.
4. Robust Verification Mechanism: The RAG-based verification with noisy document injection is a principled approach to prevent reward hacking and ensure question quality, though it adds computational overhead.

### Weaknesses
1. Limited Evaluation Setting: All experiments use a local E5 retriever with a Wikipedia 2018 corpus, which is significantly more constrained than the actual web search scenarios. The paper doesn't evaluate whether SSP-trained agents generalize to real web search or more recent knowledge bases. The recently released BrowseComp dataset is a good candidate for evaluation.
2. Lack of question quality analysis: While the paper demonstrates that the proposer generated increasingly difficult questions (as seen by declicing solver rewards in Figure 3a), there is no systematic analysis of question quality, diversity, or naturalness. It is unclear if SSP can generate diverse questions or if it leads to a narrow distribution of question types.
3. Missing Computational Cost analysis: The paper doesn't report training time, computational costs, or the overhead introduced by RAG verification and dynamic sampling. Given that invalid questions require re-generation and each question needs verification, the actual compute requirements could be substantially higher than standard RLVR, but this is not quantified.
4. Insufficient Comparison with Recent Synthetic Data Methods: While the paper mentions WebSailor (Li et al., 2025) and WebDancer (Wu et al., 2025) in related work, there's no empirical comparison. These methods also generated synthetic training data offline; a direct comparison would clarify SSP's advantages.
5. Ground Truth Answer Set: The paper states that ground truth answers are drawn from a pre-defined set D, but never specifies what this set contains, its size, or how it was constructed. This is a critical detail as it determines the scope and diversity of possible training questions.
6. Limited Analysis of Long-Term training dynamics: Training runs for only 150-200 steps. The paper doesn't investigate whether performance saturates and whether question difficulty continues to increase appropriately.

### Questions
1. Since the method is formulated as a co-evolution game, is there an equilibrium that can be reached here? What type of equilibrium exists in this setting?
2. What is the composition of the pre-defined answer set D? How was it constructed, and does it cover diverse answers or long-form answers?
3. What is the computational overhead of SSP compared to other works? Specifically, what percentage of generated questions pass the RAG verification as the training progresses, and how much additional compute is required for dynamic resampling/replay buffer?
4. Can you provide a quantitative analysis of question diversity over training, e.g., topic distribution and question type distribution? I am particularly interested in the question of diversity when the solver accuracy decreases. Is it a case of reward hacking from the perspective of the proposer?
5. How does SSP compare empirically to recent offline synthetic data generation methods (WebSailor and WebDancer) in terms of both final performance and data efficiency?
6. Could you evaluate the method on newer, more challenging datasets such as BrowseComp?
7. Do you observe any correlation between the proposer's search trajectory length and the solver's success rate? In other words, do more complex proposer searches lead to harder questions?
8. Curiosity Question: Do you have any insights on why dynamic resampling performs worse than Replay Buffer? Intuitively, Dynamic Resampling should be performing better even if it incurs higher computational costs. Please correct me if I am wrong here.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper propose Search Self-Play (SSP), a self-play strategy to finetune LLMs to utilize multiturn search agents in response to a given question. The approach consists of training a single LLM to act both as task proposer and problem solver, in a zero-sum game fashion. The repeated game is constrained by making sure proposed tasks are meaningful and ground truth answers are correct via a RAG on the documents retrieved by the proposer’s trajectory. 
Experiments performed on Qwen 7B illustrate the benefit of the approach, leading to substantial gains on Q&A benchmarks. Convincing ablations show the benefit of co-training both the proposer and solver objective, thus motivating the efficacy of the overall approach.

### Strengths
- The paper tackles the very relevant and challenging problem of improving LLMs for search and information retrieval.
- The zero-sum game formulation is very interesting but at the same time easily exploitable by both agents and thus I would have expect to produce degenerate solutions. Instead, the proposed constrained approach and overall meta-algorithm seems quite robust and able to produce stable improvements, as measured by the improved downstream performance during training.
- The experiments are sound and different design choices have been ablated, such as training only the solver or the proposer.

### Weaknesses
- I don’t understand why the proposer and solver are fine tuned with different update rules, REINFORCE and GRPO respectively. Couldn’t the same algorithm, say GRPO, be applied to both? This was not clear to me.
- Are both the proposer and solver updated at each step? I would expect more stable training dynamics to update the proposer less frequently. I would be curious to know the author’s view and experience on this. Also I think this would help understand the training dynamics. 
- It would be nice to analyze the curriculum of tasks proposed by the proposer. How does it change compared to when it is not trained? Are they too hard, too easy, or easily hackable by the solver? Ultimately, this is at the core of the proposed approach.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
