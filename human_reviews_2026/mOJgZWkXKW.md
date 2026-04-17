# Log-Linear Attention

- Decision: Accept (Poster)
- Scores: 4, 8, 8

## Abstract
The attention mechanism in Transformers is an important primitive for accurate and scalable sequence modeling. Its quadratic-compute and linear-memory complexity however remain significant bottlenecks. Linear attention and state-space models enable linear-time, constant-memory sequence modeling and can moreover be trained efficiently through matmul-rich parallelization across sequence length. However, at their core these models are still RNNs, and thus their use of a fixed-size hidden state to model the context is a fundamental limitation. This paper develops log-linear attention, an attention mechanism that balances linear attention's efficiency and the expressiveness of softmax attention. Log-linear attention replaces the fixed-size hidden state with a logarithmically growing set of hidden states. We show that with a particular growth function, log-linear attention admits a similarly matmul-rich parallel form whose compute cost is log-linear in sequence length. Log-linear attention is a general framework and can be applied on top of existing linear attention variants. As case studies, we instantiate log-linear variants of two recent architectures---Mamba-2 and Gated DeltaNet---and find they perform well compared to their linear-time variants.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Log-Linear Attention is an extension of Linear Attention to multiple timescales growing logarithmically along with a Fenwick-tree scheme. This way it introduces more memory capacity reserved for short time scales, which in turn enables a clearer separation of the longer-term memory (higher levels) for long-scale information. Through input-dependent temporal coefficients this time-scale separation can be achieved. The method shows slight performance gains on synthetic long-context and memory-capacity tasks.

### Strengths
- fast parallel and hardware-aware implementation
- captures typical inductive bias on shorter time-scales (recency bias)

### Weaknesses
- theoretically unclear why the extended memory can be effectively used, except for not "bloating" the long-term memory at the highest level with short time-scale information that can be store in lower levels
- mild improvements on benchmarks
- unclear scaling behavior

### Questions
- How does your method relate to other existing hierarchical sequence architectures like WaveNet that uses dilated convolutions [1]?
- How would this combine with compression / focusing of the inputs via exponential gating as in xLSTM [2] which has shown to be beneficial for long-context tasks in [3]?
- Given the different temporal scales are only separated by the different temporal coefficients $\lambda_t^{(l)}$, how can the effective network capacity exceed the one of pure linear attention based on the foundational theory of Hopfield network capacity? For a querying mechanism that should work across all temporal scales (as in MQAR), shouldn't the "space benefit" vanish to normal linear attention (potentially there is less noise on the shorter time-scales in Log-Linear Attention)?


[1] van den Oord et al. (2016): WaveNet: A generative model for raw audio

[2] Beck et al. (2024): xLSTM: Extended Long Short-Term Memory

[3] Beck et al. (2025): xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper identifies the fixed state size of currently best performing linear attention variants with gating, such as Mamba-2 and Gated DeltaNet as the main limitation to handle information in a long context. 
By introducing log-linear attention - a framework with a logarithmically growing set of hidden states that can be applied to existing linear attention variants - it provides a middle ground between standard attention with linear growing memory and linear attention with a fixed state size, which is independent of sequence length.
Log-linear attention is based on the insight that efficient attention variants depend on the structure of the structure of the masking matrix in the attention operation, and replaces existing masking structures with a hierarchical one. 
With log-linear variants of Mamba-2 and Gated Delta Net, the authors demonstrate the general applicability of the log-linear attention framework. 
In their experiments on small scale language modeling setups and synthetic tasks the log-linear variants show mild but consistent improvements over the standard linear RNN variants. 
In training throughput and runtime benchmarks the authors demonstrate benefits of the log-linear Mamba-2 variant over Flash-attention 2 at longer context and only small runtime overheads compared to the default Mamba-2 implementation.

### Strengths
- Clear motivation
- Well written
- The overview of related variants and the view of efficient attention mechanisms as different parametrizations of structured masking matrices is great. It shows how this naturally results in the idea & implementation for log-linear attention.
- To the best of my knowledge log-linear attention is a novel method for expanding the state size.
- The paper provides simple pure PyTorch implementations and shows experiments with optimized kernels (even though code for these is missing) achieving runtime benefits over existing methods

### Weaknesses
- Only small performance improvements over linear counter parts / base methods across several tasks (admitted by authors)
- The authors place log-linear attention as middle ground between standard attention and linear attention in terms of memory state size: Hence I would expect an exemplary calculation of the memory consumption of log-linear attention, standard linear attention and KV-cache for various sequence lengths and reasonable model sizes
- No code provided for Mamba2 and Gated Delta Net log linear attention variants
- The paper would further benefit from more details on the efficient kernel implementations (including code) for Mamba-2 


Despite these weaknesses, the paper has a clear motivation, is very well written, contributes new insights on efficient attention variants and their implementation, as well as a novel method for expanding the state size of linear attention variants, which outweighs the weaknesses. Therefore I recommend acceptance.

### Questions
- L.392-393: Why does the linear layer on top of the hidden states add 3% additional parameters to Mamba-2 and only 0.4% to Gated Delta Net?
- Description of the P matrix L.236 - 245 would help understandability
- L. 337, 997: Could the authors elaborate on the MVA pattern for Mamba-2 and/or provide references to descriptions of this?
- Does there exist an optimized kernel implementation for the log-linear variant of Gated Delta Net?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes a method to improve the modeling performance of modern SSMs/ Linear RNNs such as Mamba and Gated DeltaNet, whose recurrent state can be formulated as $S_{t} = \alpha_t * S_{t-1} + (KV)_t$. To get final output, these algorithms multiply query $Q_t$ by a single state matrix $S_t$, which aggregates information about all previous KV states up to step $t$. Log-linear attention instead disaggregates $S_t$ into $O(\log T)$ states of the same size, each containing information about a disjoint contiguous subsequence of  $\{0, …, t\}$. Then the query gets multiplied independently by each state and also by an input-dependent scalar coefficient $\lambda_i, \; i \in \{0, …, O(\log T)\}$, corresponding to that state. Finally, the results sum up. A pure linear attention variant is equivalent to log-linear attention where $\lambda_i=1$ for all i.

The partitioning of the sequence proceeds according to Fenwick tree scheme, where each subsequence can have at most $2^i$ consecutive timesteps, and the shortest subsequences is located at the latest step t, while longer subsequences’s boundaries go toward the sequence start.

I feel excited about this novel method (see strengths) and vote for its acceptance.

### Strengths
* Potent sub-quadratic runtime alternatives to Transformers is an important open area of research, and this work provides a promising way to improve modeling quality of such architectures, hopefully bringing us closer to an algorithm capable of fully replacing Transformer in autoregressive language modeling.

* The proposed method is coherent and intuitive: if we want to increase long-range performance in comparison with pure linear-time algorithms such as Mamba and DeltaNet, it is plausible that we have to execute a higher relative amount of computations, than for short sequences.

* Log-linear attention is a meta-algorithm, which is compatible with many linear-time alternatives to softmax attention.

* I believe the proposed approach has a potential to be extended and generalized by using other partitioning schemes which could bring further performance gains.

* I’d like to specifically praise comprehensive and fair comparisons which don’t shy away from presenting both positive and negative results. They help to build an honest and complete picture of the method’s strong and weak sides and possible areas of application.

* The validation shows meaningful improvements relative to pure linear counterparts on a large subset of tested benchmarks, effectuated by log-linear extension.

### Weaknesses
My judgement is that the paper doesn't have major problems. There are some minor issues mostly related to exposition/ formatting which I listed below. 

1. Did you perform any measurements of the memory footprint of the algorithm during inference (prefill, decode) and training workloads? A comparison for different sequence lengths with vanilla Mamba-2 and Gated DeltaNet, as well as with FlashAttention would be helpful. I understand that it’s likely to be $O(\log(T))$ times greater than aforementioned algorithms, but that’s an expected trade-off, which is easily tolerated given log-linear attention’s superior modeling quality. Nonetheless, these numbers would be important for finding out the most fitting conditions to use the algo.

2. It’s not clear to me from the paper how the log-linear part of the algorithm modifies the underlying linear part, both in chunk-wise parallel and recurrent algos. For example, what happens when two states for neighboring filled buckets get merged after a recurrent step? Since the states are created independently, $S_{[t_1:t_2)} + S_{[t_2:t_3)} \neq S_{[t_1:t_3)}$, although the underlying algorithm requires precisely $S_{[t_1:t_3)}$. Similar ambiguities emerge when considering chunk-wise form. 

3. There is no formal definition of the exact functional form of $\lambda_i$s. Are they simply linear projections of input vector query $q_t$? Or are they calculated using the same laws as alphas (i.e. in a different manner for each underlying architecture)? How does the algorithm handle that the number of lambdas $max(i)$ is not bounded from above and can extrapolate beyond the maximal value during pre-training?

4. Minor typos/ formatting problems:
* Line 189 – why is the right bound $t$ open? From the text it follows that the t-th token itself is a part of the partition.

* Line 213 – it would be clearer to mention explicitly that $b_t^{(i)}$ denotes the starting position of partition i, it was somewhat hard to infer for me at first glance.

* Lines 237-246 – there’s no caption of this figure.

* Line 291 – I believe it’s $\lceil T/C \rceil$.

* Line 763 – image and table captions are overlaid.

* Table 3 – There’s no mention of the table in the text, and it takes an attentive reader to understand that it’s the summary of Table 6.

### Questions
1. Can you come up with a theoretical explanation why log-linear attention offers performance improvements in comparison with pure linear variants?

2. Did you try any other partition schemes besides Fenwick Tree partitioning? I could think of other schemes, with overlapping and disjoined partitions. There could even be some schemes that recover $O(T)$ complexity (e.g., proceed as usual until sequence length is X, then keep placing the oldest tokens into the outermost bucket instead of creating new levels of hierarchy). As such, why did you choose this specific scheme?

3. A follow-up question: could there be some trade-off, where this algorithm could run in linear time at the expense of an arbitrary higher memory consumption?

### Soundness
4

### Presentation
3

### Contribution
4
