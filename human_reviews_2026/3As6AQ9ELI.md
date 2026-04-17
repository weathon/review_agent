# The Markovian Thinker: Architecture-Agnostic Linear Scaling of Reasoning

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Reasoning LLMs suffer from quadratic compute growth as their context length increases, making reinforcement learning with verifiable rewards (RLVR) and test-time scaling prohibitively expensive. Prior work has tried to lighten the computational burden by shortening reasoning traces through pruning, summarization, or multi-stage training, but these methods remain bound to quadratic costs. We introduce Delethink, a thinking algorithm that realizes the Markovian Thinking Paradigm. Instead of producing one long monolithic reasoning trace,  Delethink thinks in a sequence of chunks, the Delethink trace. Each chunk continues reasoning by referring only to a fixed number of prior tokens, which functions as a Markovian state sufficient for progressing reasoning, while deleting the rest. This preserves continuity without carrying the quadratic baggage. As a result, compute scales linearly and peak memory remains constant. In experiments, we show that Delethink can be applied directly to off-the-shelf reasoning models ranging from $1.5\textnormal{B}$ to $30\textnormal{B}$ parameters, with no loss in performance. Extended reasoning becomes possible under fixed memory and linear compute, while enabling efficient RL training on new tasks. On the DeepScaleR dataset, Delethink trains R1DistillQwen1.5B to the same benchmark performance as a standard long chain-of-thought (LongCoT) approach, where both models generate up to $24\textnormal{k}$ thinking tokens. The difference is efficiency. Delethink reasons $40\%$ faster with $70\%$ less memory footprint. By decoupling reasoning length from context length, the Markovian Thinking paradigm opens the door to next-generation reasoning LLMs that can scale to millions of tokens with linear compute and constant memory.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
It proposes a novel reasoning algorithm that enables large language models (LLMs) to perform extended reasoning with linear compute complexity and constant memory usage, in contrast to the standard Long Chain-of-Thought (LongCoT) approach, which incurs quadratic computational costs due to the self-attention mechanism’s dependence on full context length.

A zero-shot method that applies to off-the-shelf reasoning LLMs (1.5B–30B parameters), enabling them to reason over hundreds of thousands of tokens with no performance loss—and often performance gains—compared to LongCoT.

An RL-based training framework that trains LLMs as native Markovian thinkers. It matches LongCoT’s task performance while using 70% less memory and achieving 40% faster token generation.

### Strengths
1.  It introduces a genuinely novel paradigm—Markovian Thinking—for long-horizon reasoning in large language models (LLMs). While prior work has attempted to compress, prune, or summarize reasoning traces to mitigate the quadratic cost of attention, Delethink reframes the problem entirely: it assumes that only a fixed-size suffix of prior reasoning is necessary to continue coherent thought. This is a conceptual shift from “how to shorten thinking” to “how to make thinking Markovian.”
2. Comprehensive empirical evaluation across multiple model scales (1.5B–30B), benchmarks (AIME, GPQA-Diamond, LiveCodeBench), and reasoning lengths (up to 128K tokens).
3. Efficient test-time scaling: Models can reason far beyond their training-time token budgets without retraining.
4. Feasible RL training on long-horizon tasks (e.g., 96K-token reasoning would require ~27 H100-months with LongCoT vs. ~7 with Delethink).

### Weaknesses
1. lacks of comparision between mem-agent[1] on long-context-generation task.
2. It carries the last few thousand thinking tokens across chunks enables LLMs to continue their reasoning while matching or surpassing
their LongCoT performance. Did you compare this to summarization of the chunks.



[1] MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent

### Questions
1. How to select the per-chunk context size C of different size of model and tasks?
2. the different between this work and mem-agent[1].
3. Does the different rl algorithm have different effects on preformance such ppo/reinforce++ ?



[1] MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent

### Soundness
3

### Presentation
3

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
The paper deals with efficiency in reasoning language models, particularly the compute cost of long chain-of-thought. The paper proposes Delethink, which generates reasoning in chunks, with each chunk conditioned only on the prompt and a fixed-size subset of tokens from the previous chunk. This achieves more favorable compute scaling and constant memory, while maintaining or modestly decreasing accuracy in the settings tested.

### Strengths
- Simple, intuitive, and creative method based on feeding in subsets of the reasoning tokens-so-far.
- The zero-shot finding is surprising: feeding the truncated reasoning trace-so-far can match performance of using the full reasoning trace, without additional training, in some settings (Figure 4).
- Leads to good gains in efficiency (training cost, inference throughput, computational cost of scaling the token budget).
- Enables improved performance at large token budgets in the tested settings.
- Clear formalization of the method and clearly written paper.

### Weaknesses
- Most analysis is done on AIME 24 and AIME 25. It's unclear whether the results generalize to other settings, which is particularly noteworthy due to the assumptions present in the method (discussed in the next two points). 
- In the zero-shot setting, the method relies on the Markov assumption; concretely, that the relevant context is contained in the beginning and ending of the chunk of the reasoning trace. It is indeed surprising that this assumption held for Qwen3 30B-A3B on the math reasoning datasets in Figure 4. However, I am skeptical that this would hold in many tasks. If so, the zero-shot setting may be of limited use. 
- When the model is trained, the learning algorithm should ideally put the relevant context into the beginning/end part of the chunk so that it is available at the next step. I have two concerns:
    - The zero-shot experiments seem to suggest that standard LongCoT may naturally place tokens in positions that are amenable to Delethink. Do you have evidence that the model learns to place tokens much differently than with standard Long CoT training?
     - Can you provide experiments on a broader range of tasks beyond AIME to demonstrate generalization? The CrossWordBench results in Figure 5 suggest potential limitations when task state cannot be compressed into the carried context. 
- Delethink underperforms standard long CoT at reasonable thinking budget levels (4k-21k, Figure 8).
- Delethink Training appears to be a standard RL algorithm applied to the MDP defined by the Delethink setting. I'm fine with this if it works well and the paper isn't overclaiming; however I'm listing it as a minor weakness to denote that the training is not particularly surprising or innovative.
- The method is referred to as a "thinking algorithm"; it is unclear what this means.

### Questions
Please address the statements and questions described in the Weaknesses above.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Delethink, a novel algorithm designed to overcome the quadratic computational cost associated with long chain-of-thought (LongCoT) reasoning in Large Language Models (LLMs). The core idea is the "Markovian Thinking Paradigm," where the model reasons in a sequence of short, fixed-context chunks. Each new chunk conditions only on the original prompt and a small suffix (the "Markovian state") of the previous chunk, effectively "deleting" the rest of the reasoning history. This approach decouples the total reasoning length from the computational context, leading to linear scaling of compute and constant memory usage.

### Strengths
The Markovian Thinking Paradigm: A new formulation for LLM reasoning that avoids the quadratic cost of self-attention over long sequences.
An RL training framework to create native Markovian thinkers, which is shown to be significantly more compute- and memory-efficient than standard LongCoT training while achieving competitive or superior performance.
Comprehensive experiments demonstrating that Delethink matches or surpasses LongCoT performance on complex reasoning benchmarks (e.g., AIME, GPQA) while being 40% faster and using 70% less memory, and enabling effective test-time scaling beyond training-time limits.

### Weaknesses
The authors should analyze the content of the reasoning traces in successful vs. failing cases. What information is being lost when tokens are deleted? 
The claims of orthogonality to methods like KV cache eviction and quantization are valid, but a combined evaluation is missing.
If deleting already generated tokens is incompatible with KV cache technology, how does the inference speed compare to LongCoT?
In lines 247-248, "we vary the thinking budget from 8K to 256K tokens." How is the thinking budget controlled?

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Delethink, an algorithm aimed to mitigate the quadratic computational cost of long chain-of-thought (LongCoT) reasoning in LLMs. The proposed method, replaces a single monolithic reasoning trace with a sequence of shorter chunks. Each chunk is generated by conditioning only on the original prompt and a fixed-size suffix of the preceding chunk, allowing the model to theoretically scale its reasoning length with linear compute and constant memory. The paper presents two primary use cases: a zero-shot inference strategy for off-the-shelf models and an RL training framework (Delethink Training). Empirical results show that Delethink can match the performance of standard LongCoT on reasoning benchmarks while claiming to be more efficient in terms of speed and memory usage.

### Strengths
1. The paper addresses the critical and timely problem of controlling increasing complexity in LLM reasoning because of long chain-of-thought.
2. The proposed Delethink algorithm is simple and can be applied as a zero-shot inference strategy to existing models without requiring any kind of architectural modifications.
3. The results are generally comprehensive, showing stronger performance at lower costs for both inference and training of reasoning models.

### Weaknesses
1. Identical compute budgets: Major figures in the paper are compared based on ``thinking tokens'' but Delethink also has to recompute kv cache and q values after every chunk. A more fair comparison would have been if wall-clock and optionally TFLOPs across methods would have been reported.
2. From results, it seems that the performance of Deletethink is heavily dependent on the choice of length of chunk context. For instance, 1.5 and 7B models, it performs well with 8K on GPQA, but for AIME it performs better with 4K. Similarly, for Qwen3-30B even 8K performs worse than 16K. Given strong dependence it is unclear how can the optimal length (or even the one that performs better than longCoT) of chunk context be determined.
3. ``Summarization-based reasoning'' (e.g., iterative summarization, InftyThink) enablelonger effective thinking at lower cost, including at least one of these baselines is essential (e.g., iterative summary every K tokens).
4. From figure 2a, it seems that for certain token lengths (at lower token lengths), Deletethink performs worse than LongCoT. Further, since we don't results with wall time on x-axis are not shown, it is unclear whether the same trend holds for other model/dataset combinations as well. If this is generally true, then this should be acknowledged in the limitations section.

The above four points are the primary weaknesses of the paper.

In addition:

## Minor Weaknesses/Suggestions:

5. Ablate the ``fold first 100 tokens of initial chunk into q.'' This is a useful practical tweak but undermines the purely markovian nature claim. Can authors show how much of the zero-shot benefit depends on this?
6. Other types of baselines: Comparison of LongCoT with sliding window (if architecturally possible) or KV eviction, i.e., do not re-prefill q between chunks could strengthen the claims.
7. Suggestions for writing and presentation: 
    - The section 4.1 could be simplified in formulation. For instance, variable C is introduced but not used again. Similarly, the need for extending length from n to nS seems unnecessary?
    - There are several empty ablation sections such as A.5, A.6.1, A.6.2.

### Questions
Please see Weaknesses Above.

### Soundness
3

### Presentation
3

### Contribution
3
