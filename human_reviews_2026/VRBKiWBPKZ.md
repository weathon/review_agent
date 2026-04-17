# Learning to Think in Blocks: A Prior-Guided Reinforcement Learning Framework for RAG

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Retrieval-Augmented Generation (RAG) systems mitigate factual inaccuracies in large language models (LLMs) by integrating external knowledge, but their effectiveness often hinges on query rewriting techniques. Prompt-based rewriting methods are frequently suboptimal, while existing reinforcement learning (RL) approaches struggle with inefficient, unguided exploration of the vast strategy space. To address these limitations, we propose an end-to-end RL framework that initializes the training process with human-defined prior rewriting strategies, enabling the model to learn from its interactions with the RAG environment and develop its own effective posterior rewriting strategies. Furthermore, we develop a novel RL algorithm, namely Block-wise Geometric Policy Optimization (BGPO), which resolves the granularity mismatch in previous methods by redefining the state-action space as blocks of tokens. This algorithm is enhanced by geometric averaging for balanced importance and a Bellman-equation-inspired credit assignment mechanism to reshape the reward. Extensive experiments on both local corpus retrieval and online search datasets demonstrate that our RL framework consistently surpasses the baselines, validating its superiority for complex RAG tasks. Our project code can be found at this anonymous repository: https://anonymous.4open.science/r/Learning-to-Think-in-Blocks-A-Prior-Guided-Reinforcement-Learning-Framework-for-RAG-0288/

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Blockwise Generative Policy Optimization (BGPO), a reinforcement learning framework for improving reasoning and query rewriting in retrieval-augmented generation (RAG). Instead of generating the entire reasoning trace at once, the model learns to think and generate answers in iterative blocks, where each block can trigger additional retrieval and self-reflection. The approach is trained using a block-level reward that evaluates intermediate reasoning quality via an LLM-as-judge. Experiments on several single-hop and multi-hop QA benchmarks show that BGPO enhances multi-step reasoning ability and improves over standard RAG and RL baselines.

### Strengths
The proposed blockwise policy optimization (BGPO), which decomposes long reasoning traces into smaller segments, seems to be effective in enhancing the RAG quality. Experimental results show modest accuracy gains over vanilla RAG and other baselines.

### Weaknesses
1. Inference time & latency not reported. The paper does not report inference time, or latency, which is a critical omission. The proposed method encourages longer interactive rollouts, and Figure 2 shows that BGPO increases output length during training. Yet only total training time (4 × A800 GPUs for ~2 days) is reported, with no inference cost or per-query latency. To assess practicality and efficiency claims, the authors should report average end-to-end inference time per question, separately for single-hop and multi-hop datasets, along with the average number of retrieval rounds, blocks, and tokens per query. Without these details, the claimed efficiency (matching 14B results with a 7B model) may be misleading if inference is substantially slower.

2. Unclear accuracy of query rewriting. The core claim is that the agent learns useful rewrite strategies, but the paper only reports final QA accuracy and a rollout accuracy boxplot (Fig. 3), without directly evaluating rewrite quality or its impact on retrieval. To substantiate this claim, the authors should measure how often rewrites improve retrieval. For example, the fraction of rewrites that increase recall@k (or NDCG) over the original query, average recall@k after successive rewrites.

3. LLM-as-judge evaluation introduces bias. The LLM-as-Judge protocol uses Qwen3-32B with a prompt that forces a single-token Yes/No output (Appendix B), raising several concerns. Using the same model family for rollouts and evaluation risks shared biases, leading to inflated agreement. The binary prompt is brittle, failing to capture partial correctness, acceptable paraphrases, or nuanced answers.

4. Missing evaluation on SimpleQA benchmark [1]. The evaluation omits newer benchmarks such as SimpleQA, relying only on older QA datasets. Without testing on more recent and challenging factual-QA settings, it remains unclear whether the proposed method generalizes beyond the benchmarks it was trained and tuned on.

[1] Measuring short-form factuality in large language models, arxiv preprint arxiv:2411.04368, 2024.

### Questions
Please address my comments in paper weaknesses.

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
The paper addresses how retrieval-augmented generation (RAG) systems struggle with query rewriting: very few effective human-designed prompts and too vast a strategy space for naïve RL. To solve this, the authors propose:
* A **prior-guided RL framework** where the model begins by leveraging human-defined query rewriting heuristics (e.g., decomposition, synonym replacement) and then learns to refine or go beyond them via reinforcement learning.
* A novel RL algorithm called **Block-wise Geometric Policy Optimization (BGPO)**: it defines actions at the "block" level (coherent segments of reasoning/query rewriting) rather than at token or whole-sequence level, uses geometric averaging of importance weights, and shapes rewards to emphasise early reasoning steps (via a reversed discounting/emphasis factor) in the RAG process.
* Experimental results showing that this approach consistently outperforms baselines on multiple RAG tasks (both retrieval from local corpora and web search) by being more efficient and effective.

**Key Contributions**

1. Introduction of an **end-to-end RL framework** for RAG that starts from human priors and transitions into learned strategies.
2. The BGPO algorithm: redefining RL granularity to blocks of reasoning/query rewriting, providing better credit assignment and stability in multi-step reasoning.
3. A tailored reward-shaping mechanism for RAG tasks that assigns more weight to early reasoning/retrieval steps rather than just the final answer.
4. Empirical validation across a variety of RAG settings showing improved performance.

### Strengths
* The paper proposes using human-crafted rewriting heuristics as an initial policy ("prior") before RL refines it. Prior works like KoGuN [A] also integrate human prior knowledge into RL for control tasks, but not specifically for RAG or query rewriting. Also, traditional RAG rewriting methods such as HyDE [B] which rely on fixed heuristics and do not learn a rewriting policy via RL.
* The method introduces "blocks" (coherent segments) as units of action/state rather than token- or sequence-level steps. Prior RL methods for language policy, such as token-level or sequence-level PPO or sequence-based RL in RAG, struggle with the granularity mismatch. The paper explicitly addresses this. Compared to frameworks like LeTS [C] which focus on process/outcome reward hybridization but still operate at more coarse granularity for reasoning steps.
* The paper reverses the typical temporal discounting logic to give greater weight to early reasoning/query rewriting steps rather than only final outcomes. Many prior RAG or RL-for-RAG methods rely simply on an outcome or final answer reward (e.g., correctness of answer) and do not separately reward intermediate reasoning efficiency or rewriting quality. For example, the survey of RL methods for reasoning highlights this gap [D]. Compared to the work TIRESRAG-R1 [E] which introduces multi-dimensional rewards for retrieval sufficiency, reasoning quality and reflection, but the current paper’s emphasis on early step weighting via a modified discount factor is a distinct design.
* According to the paper’s results, the 7B-parameter model using their method matches or outperforms a 14B baseline on multiple RAG tasks.
* The paper reports experiments on both local corpus retrieval and online web search, showing robustness of the approach across retrieval environments.
```
[A] KoGuN: Accelerating Deep Reinforcement Learning via Integrating Human Suboptimal Knowledge, IJCAI 2020
[B] Precise Zero-Shot Dense Retrieval without Relevance Labels, ACL 2023
[C] LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization, ArXiv 2025
[D] https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training
[E] From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs, ArXiv 2025
```

### Weaknesses
* The paper proposes a novel granularity: blocks of tokens/actions rather than token-level or sequence-level. While this is interesting, it is not fully clear whether this granularity is the optimal choice or how sensitive performance is to different block definitions.
* The paper presents experiments on several datasets and tasks, which is good, but certain key axes seem under-explored. For instance: It is unclear how much the prior strategies alone contribute (i.e., human heuristics only) vs. the RL phase? How many environment interactions or RL steps were required (sample efficiency)? How sensitive is performance to different model sizes / retrieval difficulties?
* While the paper mentions some limitations, the discussion is relatively brief and does not deeply explore cases where the method does not perform well, or provide mitigation strategies. A more detailed failure‐mode analysis would be helpful. For example, does the method degrade when retrieval is already near‐perfect (thus less room for rewriting)? Does it overfit to specific priors and struggle to generalize to unseen rewriting styles?

### Questions
* Is it possible to include experiments varying block size/granularity (e.g., smaller blocks vs. larger blocks) and report performance trade-offs? It would also be good to compare to token-level RL and full-sequence RL baselines more explicitly (beyond high-level claims) with matched compute budgets, to validate that "block-wise" yields a genuine sweet-spot. Additionally, it would be good to detail the block definition (how many tokens constitute a block, how blocks map to reasoning steps) and include a sensitivity analysis.
* Is it possible to provide a detailed ablation study that shows: Prior‐only baseline (no RL), RL from scratch baseline (no prior), prior-guided RL method? Additionally, it would be good to include plots of learning curves over RL steps/interactions showing convergence speed, report interplay of retrieval difficulty (e.g., harder queries) with rewriting performance and provide error analysis showing where the method fails (e.g., what types of queries the rewriting policy still can’t handle) to guide future improvements.
* If possible, add a dedicated section in the paper (and supplementary) analysing failure cases: e.g., queries where rewriting made retrieval worse, or where RL diverged. Provide quantitative breakdowns by query type, retrieval difficulty, and show how different choices (block size, reward weights) affect these failures. This could also help future work define more robust priors or adaptive reward schemes.

### Soundness
3

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
This paper proposes a novel prior-guided reinforcement learning (RL) framework for Retrieval-Augmented Generation (RAG), where an LLM agent is initialized with human-defined query rewriting strategies (e.g., decomposition, HyDE) and learns refined "posterior" strategies through interaction with the RAG environment. To address granularity mismatches in existing RL methods, the authors introduce Block-wise Geometric Policy Optimization (BGPO), which treats each reasoning turn as a semantic “block” for action and state representation. BGPO further incorporates a Bellman-inspired emphasis factor that prioritizes early reasoning steps via reward shaping and uses a hierarchical geometric importance ratio for stable training. Experiments show that their 7B-parameter RAG-BGPO model outperforms or matches larger 14B baselines across multiple QA benchmarks, demonstrating both efficiency and strong reasoning capabilities.

### Strengths
The proposed Block-wise Geometric Policy Optimization (BGPO) introduces a more suitable “block-level” granularity for multi-turn reasoning, bridging the gap between overly fine-grained token-level and coarse sequence-level approaches.

### Weaknesses
1. The main experiments use only a 7B-parameter model, and while it competes well with a 14B baseline, the framework’s scalability to larger models (e.g., 70B+) or its performance under more diverse architectural settings is not explored.
2. For online-search datasets (PopQA, Bamboogle), the system uses Wikipedia-only search, which does not reflect the complexity and noise of real web search, potentially overestimating practical applicability.
3. The evaluation would be strengthened by including a broader set of baselines and by situating the work more firmly within the existing body of relevant literature.

### Questions
1. It would be beneficial to include a performance comparison curve that tracks the training progress of DAPO and BGPO over time.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a novel Reinforcement Learning (RL) framework to address a critical bottleneck in Retrieval-Augmented Generation (RAG) systems: suboptimal query rewriting. The core problem is that for complex questions, a simple RAG retrieval using the original query often fails. Existing solutions are inadequate: prompt-based rewriting is labor-intensive and frequently ineffective, while prior RL approaches are highly inefficient due to the "vast and complex" strategy space they must explore without guidance. The paper's central hypothesis is that an RL agent for RAG can be trained far more effectively and efficiently. To achieve this, the authors propose a two-part solution
1. A Prior-Guided RL Framework: An end-to-end framework that initializes and guides the RL agent, an LLM (Qwen2.5-7B), by providing it with a "Hybrid Action Space." This space combines four pre-defined "human-defined prior rewriting strategies" (e.g., decomposition, keyword_extraction, synonym_replacement, and HyDE) with a flexible, self-devised action (rewrite by myself) and a terminal action to stop the process. This allows the agent to start from a competent baseline and learn a more refined "posterior" strategy through interaction.
2. Block-wise Geometric Policy Optimization (BGPO): A new RL algorithm designed from the ground up for the multi-step, conversational nature of RAG. BGPO resolves the "granularity mismatch" of prior methods (which were either too fine-grained at the token level or too coarse at the sequence level) by defining its state-action space at the level of "blocks," where one block represents one conversational turn.1 The algorithm is further enhanced with a "Bellman-equation-inspired credit assignment" mechanism, an "emphasis factor" $\gamma^{k-1}$, which uniquely reverses traditional temporal discounting to more heavily reward the crucial initial reasoning steps.1

### Strengths
1. The paper's greatest strength is the design of the BGPO algorithm. The "emphasis factor" $\hat{A}_{i,k} = \gamma^{k-1}\hat{A}_i$ 1 is a highly novel, well-motivated, and elegant mechanism for credit assignment that is perfectly tailored to the realities of RAG. The block-level granularity and hierarchical importance ratio are also clever solutions to known problems in policy optimization.1
2. The "prior-guided" RL framework  is a practical and robust solution to the "cold start" exploration problem that has hindered RL-based RAG. Figure 3 shows higher and more stable rollout accuracy, is a strong testament to this architectural choice.

### Weaknesses
1. The paper's entire empirical validation rests on a critically flawed foundation. The use of a Qwen3-32B model to judge the performance of a Qwen2.5-7B agent is a severe, unaddressed conflict of interest. As detailed in Section 2.3, this "Qwen-on-Qwen" setup creates an unacceptably high risk of "evaluator bias," where the agent is rewarded for optimizing for the stylistic or knowledge-based quirks of its sibling model, not for objective task success. This methodological flaw undermines the entire empirical contribution and renders the main results in Table 1 and Table 2 inconclusive.
2. As described in Appendix A , the decision is to filter out all hard problems from the training set. By only training on problems that the base Qwen2.5-7B model could already solve, the authors have trained an agent specialized for "medium difficulty" tasks. This severely limits the paper's generalizability and directly contradicts its headline claim of "superiority for complex RAG tasks" , as the agent was never exposed to the most complex problems.
3. The paper's core algorithmic unit, the "block," is ambiguously defined. It is unclear if a "block" $b_k$ refers only to the generated query, or to the entire reasoning-acting-generating sequence (internal monologue, action choice, and query).1 This lack of precision makes the BGPO algorithm (Algorithm 1) difficult to reproduce exactly and obscures the fine-grained details of the credit assignment.

### Questions
1. The paper's most significant methodological concern is the "Qwen-on-Qwen" evaluation setup (Qwen2.5-7B agent, Qwen3-32B judge). Could the authors please comment on this potential for "evaluator bias"? How can they be confident that the agent is optimizing for factual correctness and robust rewriting, rather than "gaming" the stylistic and knowledge-based preferences of its sibling model?
2. What percentage of the original combined training set (HotpotQA, 2WikiMultihopQA, MuSiQue) was discarded by the filtering process described in Appendix A? Was any analysis performed on the nature of these discarded "hard" problems?

### Soundness
3

### Presentation
3

### Contribution
3
