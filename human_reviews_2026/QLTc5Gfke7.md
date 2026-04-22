# Think Before You Retrieve: Learning Test-Time Adaptive Search with Small Language Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Effective information retrieval requires reasoning over partial evidence and refining strategies as information emerges. Yet current approaches fall short: neural retrievers lack reasoning capabilities, large language models (LLMs) provide semantic depth but at prohibitive cost, and query rewriting or decomposition limits improvement to static transformations. As a result, existing methods fail to capture the iterative dynamics of exploration, feedback, and revision that complex user queries demand. We introduce Orion, a training framework that enables compact models (350M-1.2B parameters) to perform iterative retrieval through learned search strategies. Orion combines: (1) synthetic trajectory generation and supervised fine-tuning to encourage diverse exploration patterns in models, (2) reinforcement learning (RL) that rewards effective query refinement and backtracking behaviors, and (3) inference-time beam search algorithms that exploit the self-reflection capabilities learned during RL. Despite using only 3\% of the training data available, our 1.2B model achieves 77.6\% success on SciFact (vs.  72.6\% for prior retrievers), 25.2\% on BRIGHT (vs. 22.1\%), 63.2\% on NFCorpus (vs. 57.8\%), and remains competitive on FEVER, HotpotQA, and MSMarco. It outperforms retrievers up to 200-400× larger on five of six benchmarks. These findings suggest that retrieval performance can emerge from learned strategies, not just model scale, when models are trained to search, reflect, and revise.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Orion, a lightweight retrieval-agent framework that treats retrieval not as a one-shot top-k matching task but as a multi-step search process with reflection, backtracking, and adaptive query reformulation. Instead of relying on large LLM controllers, Orion trains small models (350M–1.2B) using synthetic retrieval trajectories and GRPO (Group Relative Policy Optimization) to learn when to continue, revise, or restart a retrieval path. At inference time, Orion performs structured retrieval tree search, expanding and pruning based on self-assessed sufficiency signals. Experiments on SciFact, BRIGHT, and other multi-hop retrieval benchmarks show that Orion-1.2B surpasses GPT-4.1-Mini, LLaMA-8B, and dense retrievers in reasoning-heavy retrieval tasks, even using only MiniLM (22M) as the backend retriever.

### Strengths
1. Clear reframing of retrieval as an adaptive search problem rather than static top-k matching, highlighting a major weakness of current RAG pipelines.
2. Shows that small models can behave like retrieval agents through supervised trajectory learning + lightweight RL, without needing GPT-4 controllers.
3. Synthetic trajectory generation is clever, enabling supervision of “failed path → recover → continue” behavior without human annotation.
4. Strong empirical results, showing that strategy > model size — Orion-1.2B beats GPT-4.1-Mini and LLaMA-8B on multi-hop retrieval, despite using a tiny retriever.
5. Provides a practical agentic retrieval alternative that is computation-efficient and applicable to existing RAG systems.

### Weaknesses
1. Limited novelty justification, especially for the “retrieval-as-search-agent” framing. Similar ideas of iterative retrieval with self-reflection have appeared in ReAct-style and GraphRAG-like agentic RAG. The paper mainly scales this down to smaller models but does not clearly articulate what is fundamentally new beyond engineering a controllable pipeline. It also avoids testing with larger LLM controllers, which would clarify whether the method is intrinsically effective or simply a pragmatic workaround for small models.
2. The trajectory synthesis procedure is heavily engineered and tailored to specific “failure → retry” patterns. It remains unclear whether these synthetic supervision traces generalize to domains where retrieval noise and error modes differ significantly.
3. Dependency on a fixed retriever (MiniLM) limits general claims. Orion is claimed to be model-agnostic, yet it is only demonstrated with a very small retriever. There is no compatibility study with stronger retrieval components (e.g., RankRAG, ColBERTv2, hybrid pipelines), leaving open the question of whether Orion’s gains persist in more realistic high-recall setups.
4. RL contribution is under-analyzed. The paper uses GRPO, but there is no ablation against pure SFT imitation or other standard RL baselines like PPO/DPO/Step-wise policy gradient. It is unclear whether RL actually provides meaningful strategic improvements or just reinforces heuristics learned from synthetic traces.
5. Tree search policy remains heuristic. While presented as an “agentic retrieval loop,” the continuation/pruning logic relies on fixed confidence thresholds rather than a learned uncertainty-aware or utility-optimized policy, reducing the method’s theoretical grounding.
6. Evaluation scope is narrow despite claims of generality. Experiments are limited to scientific fact retrieval and multi-hop QA under moderate-scale datasets. The paper does not test open-domain, web-scale, or noisy retrieval environments, where agentic control would be most stressed.

### Questions
See weaknesses.

### Soundness
2

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
This paper presents Orion, a new framework for test-time adaptive retrieval with small language models (SLMs) ranging from 350M to 1.2B parameters. Unlike conventional retrievers that issue a single static query, Orion enables an SLM to “think before it retrieves,” performing multiple rounds of reasoning, search, and reflection.

The key idea is to teach a small model how to conduct multi-turn retrieval reasoning.. At inference time, Orion performs iterative beam-search-based retrieval. Empirically, Orion is evaluated on several open-domain retrieval and reasoning benchmarks, including Natural Questions, HotpotQA, and MS MARCO. The experiments show that small models equipped with Orion achieve large gains over standard retrievers and even close the gap to large-scale LLM-based retrievers, while keeping inference efficiency comparable to single-step retrieval systems.

### Strengths
The paper introduces a new concept of test-time adaptive retrieval reasoning for small language models, showing that even models without large-scale reasoning capacity can learn to think about what to search for before issuing queries. This reframes retrieval as an interactive reasoning process, not a static lookup.

Experiments across multiple benchmarks show consistent and significant improvements over both static and multi-query retrievers. Small models trained with Orion approach the performance of much larger models, demonstrating the effectiveness of learned retrieval reasoning even in constrained architectures.

### Weaknesses
1. The paper does not include a baseline that follows the standard two-stage SFT + RL training pipeline. A more complete comparison would involve first performing rejection sampling SFT to warm up the model using high-quality queries (which could be generated through the proposed beam search or other strategies, followed by a evalutation on that specific query), and then applying reinforcement learning for further optimization.
The current baseline, such as DeepRetrieval, only uses RL without an SFT warm-up stage. This makes it difficult to fairly assess the contribution of the proposed training method. In addition, it would be important to include results using the authors’ LFM2 series models under this two-stage setup for a more comprehensive evaluation.

2.  The paper claims that the proposed method achieves high efficiency, but it does not provide an explicit comparison of inference speed among different methods.

3. The paper mentions the use of “BM25 (dense)” as a baseline but does not describe its specific configuration.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the Orion framework, enabling small language models to learn adaptive strategies of "thinking, reflecting, and re-searching" during the retrieval process. Through training with synthetic search trajectories and reinforcement learning, the model can proactively adjust its query path, backtrack, and optimize when retrieval fails. Experimental results show that Orion, even with only a fraction of the parameters of large models, can achieve performance equal to or even surpass that of models like GPT-4 in multiple retrieval and inference tasks, demonstrating that retrieval intelligence depends on strategy rather than scale.

### Strengths
First, a framework is proposed that enables small models to have adaptive retrieval capabilities, significantly reducing reliance on large models.

Second, by combining reinforcement learning and structured reasoning labeling, the model can proactively reflect and backtrack during the retrieval process, improving search accuracy.

Third, experiments demonstrate that small models can outperform large models on multiple complex tasks through learning strategies, exhibiting high efficiency and practical value.

### Weaknesses
My main concern is that the novelty of this paper is quite limited, as its core idea is very similar to works [1,2], yet there is no relevant discussion or comparison in the main text. Although the appendix briefly mentions differences from Search-R1, claiming that Orion avoids the complexity of external knowledge bases, I believe this statement is inaccurate because Search-R1 itself was also implemented with offline corpora and lightweight retrieval components. Moreover, the paper should include comparisons with the baselines discussed in works [1,2]. The lack of these comparisons and discussions significantly undermines the paper’s originality and experimental completeness.

[1] Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning

[2] R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce a training process to teach small language models multi-hop query generation. The training process starts with synthetic data generated to adhere to several strategies, using external LLMs, and then the authors apply step-wise GRPO to amplify useful reasoning and query chains based on retrieval metrics for each hop. They then proceed to training for future hops with a greedy selection of the highest-reward step. The authors report a number of reasonably strong results on simple, multi-hop, and reasoning-intensive retrieval tasks.

### Strengths
The problem motivated is reasonable. The analysis of search behaviors (Table 5) is a very interesting idea.

### Weaknesses
To someone who has worked in this area for several years, the writing appears almost confused or contradictory across parts of the paper. Parts of the paper seem to suggest that the emphasis is on finetuning retrieval models with LLMs via RL, other parts seem to indicate that only the reasoning LLM that generates queries is being finetuned.

To begin with, the problem of multi-hop retrieval is old; its modern instantiation is at least as old as HotPotQA (2018), a paper with over 3000 citations as there's a vast literature of methods for training what the authors refer to as "multi-turn IR" models. Indeed, many of them do things that the authors call novel, like "making the retriever itself adaptive" and "turn-level reward structure that leverages standard IR metrics to provide dense feedback at each search step". I will not cite a specific paper here because there's a vast literature of these, both through 2020-2022 with BERT-style models and through 2024-2025 (if not earlier) with more modern open LLMs and reasoning models.

That said, the main issue I take isn't about novelty. There's still plenty of room for novel methods in this space. My concern is trying to understand what the authors really did:

> The retriever itself remains static, invoked repeatedly but never trained to adapt its search strategy. This overlooks
a key point: the retrieval policy is as important as the reasoning policy. [...] We introduce a different approach: making the retriever itself adaptive. [...] A key innovation is our turn-level reward structure that leverages standard IR metrics to provide
dense feedback at each search step rather than sparse outcome-only signals.

This sounds to me like an argument to finetune the actual retriever (i.e., embedding) model. That's very reasonable, and although there's plenty of research that has done that since 2020 for multi-hop tasks, this is worth revisiting now. But the challenge is that on reading the rest of the paper we see statements like "these gains emerge not from stronger embeddings or larger scale, but from learned adaptive behavior: recognizing when queries fail, exploring alternatives systematically, and recovering from unproductive search paths." So it sounds like the authors do not, in fact, finetune the retrievers after all, and instead finetune the LLM to generate better reasoning and queries. There's an extremely large space of these types of methods, perhaps even larger than the set of methods that finetune embeddings for multi-hop retrieval.

The experiments are equally opaque. What does BM25 (dense) vs. BM25 (sparse) refer to and why is their difference in quality so drastic? What does it mean to have LLMs like GPT-4.1 in Table 2 tested against retrievers like MiniLM-L6-v2? Why only compare with the extremely weak, small, and old retrieval model MiniLM-L6-v2 from 2021 and not the many methods out there since? Many of the numerical values in Table 2 themselves are strange, when contrasted to earlier work. As an expert reader who has worked in this area for several years, I do not have confidence I understand how these results were produced.

Minor: The authors use a kind of "step-wise" GRPO for each step of the training. This is reasonable, but is there a reason not to do something like REINFORCE with intermediate rewards over the complete trajectory?

### Questions
Do the authors train one version of their model (at each scale) once overall or once per dataset? In other words, what is the data that the RL (GRPO) is done over?

### Soundness
1

### Presentation
1

### Contribution
1
