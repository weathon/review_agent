# Length Generalization with Log-Depth Recurrent Units

- Decision: Reject
- Scores: 4, 2, 2, 4, 6

## Abstract
Length generalization remains a persistent challenge for neural networks: recurrent models tend to suffer from positional biases, while Transformers are constrained by fixed computational depth. Regular languages provide a frequently used testbed for evaluating length generalization, as any sequence can be exactly verified to determine its label. We propose the Log-Depth Recurrent Unit (LDRU), which composes token embeddings through a learned pairwise operator inspired by monoid composition, yielding uniform logarithmic depth across tokens. On 21 regular tasks, consisting of standard benchmarks and new prefix languages, the LDRU achieves 100\% out-of-distribution accuracy on 18 tasks and at least 96\% on the remaining 3, consistently outperforming recurrent and attention-based models. These results establish the LDRU as an effective architecture for length generalization on regular languages and a promising direction for compositional sequence modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Log-Depth Recurrent Unit (LDRU), a novel recurrent architecture that composes token embeddings via a learned pairwise reduction operator inspired by monoid composition, achieving logarithmic computational depth. The authors evaluate LDRU on 21 regular language tasks, including newly proposed prefix languages to test long-range dependencies. The model demonstrates near-perfect length generalization, achieving 100% OOD accuracy on most tasks and outperforming strong baselines such as RNNs, LSTMs, Transformers, RegularGPT, and state-space models. The work is well-motivated, clearly written, and provides strong theoretical grounding and empirical evidence. While its experiments are confined to synthetic regular languages and the practical efficiency of log-depth computation is not yet verified, the contribution represents a meaningful advance in systematic generalization and architecture design for sequence models.

### Strengths
1. The proposed LDRU is a novel architecture according to the Reviewer's expertise.
- LDRU’s log-depth reduction mechanism is a clever hybrid of RNN recurrence and Transformer parallelization.
- The design explicitly encodes compositional inductive bias linked to formal language theory.

2. This paper provide strong empirical results, demonstrating the effectiveness of LDRU on systematic length generalization.
- Comprehensive evaluation on 21 tasks, including new benchmarks.
- Consistent and large performance gap over state-of-the-art baselines.

3. Others: the authors introduce a benchmark (Prefix Languages) which provides a systematic way to test long-range dependency modeling. The paper is well-structured, with a clear logical flow from theory to method to experiment to analysis.

### Weaknesses
1. Restricted Scope: The evaluation domain is narrow; all tasks are regular or near-regular, where compositional structure is simple. It’s unclear how LDRU would generalize to non-regular or natural language tasks where equivalence classes are not well-defined. Besides, it is doubtable how LDRU is compatiable with existing foundation language models.

2. While $O(\log n)$ depth is theoretically appealing, practical runtime or memory benchmarks (on GPUs) are not provided and the reduction tree might introduce non-trivial communication or synchronization overhead. Besides, no formal proof is given for why or when LDRU will learn correct compositions (though inspired by monoid theory).

### Questions
1. Please refer to the weaknesses part.
2. The authors neatly compare LDRU with standard Transformers and RNNs on working complexity and depth. Can you provide some empirical results to show the superiority (or the tradeoff) of LDRU, in comparison with Transformers and RNNs?
3. I think LDRU's performance does not show much improvment over LSTM (both near to $100\%$ accuracy). Though LDRU is designed to be more effective on the processing depth ($O(\log n)$ versus $O(n)$), the reviewer wonders whether LDRU beats LSTM by a large margin in some length generalization tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents the log-depth recurrent unit (LDRU) architecture, which reduces information along the sequence dimension via a sequence of aggregations of pairs that requires log (in sequence length) depth to aggregate information from the entire sequence. This is compared to transformers and RNNs on length generalization as measured by synthetic regular language tasks.

### Strengths
1. The paper presents an interesting idea that combines some of the relative strengths of transformers and RNNs. 
2. The length generalization results with the new architecture seem consistently strong.
3. The results interpreting the patterns of composition that are needed in the data for length generalization to arise are interesting.

### Weaknesses
1. This paper neglects important related work (e.g. delta nets https://arxiv.org/abs/2406.06484 , PSMs: https://arxiv.org/abs/2506.10918 and especially log-linear attention: https://arxiv.org/abs/2506.04761). These papers present very similar methods with more general and scalable experimental results, although less of a focus on length generalization. The authors would need to carefully read and discuss this work and how it relates to this paper. Especially log-linear attention and PSMs present the idea of generic parallel scans for doing log-depth recurrence. 
2. The paper is missing comparisons of parameters + FLOPs across architectures. Adding the MLP layers will add parameters and make the networks larger than the baselines. This seems like it potentially makes the comparisons unfair as a result. Moreover in terms of actually understanding the cost of the new architecture, we need a more detailed analysis of the FLOPs required both for training and for inference. 
3. It seems that the architecture likely does not allow easy parallel training. We have seen in recent years that this is important for scalability (e.g. transformers + SSMs). It is not even obvious to me from the paper whether computation is being re-used during training, or if for each token during training the entire LRU over the sequence is being recomputed. Full description of the training complexity for next token prediction would be useful. 
4. The transformer baseline seems somewhat weak by only using NoPE positional embeddings. It is now fairly standard to compose sliding window RoPE layers with global NoPE layers for length generalization. The local layers are needed to create better representations of local tokens that is difficult with NoPE. Here is one example paper with dramatically better length generalization with better positional encodings: https://arxiv.org/abs/2402.01032. 
5. There is no strong state space model baseline. In particular, the gated delta net is the SoTA architecture and is substantially more expressive than Mamba and similar architectures. This baseline is needed. 
6. The results of LDRU without the non-linearity seem fairly similar to with the non-linearity. When it is linear, it seems that the LDRU may also become equivalent to some form of state-space/linear attention.

### Questions
1. What is the performance of each model on the training distribution? It seems fair if we only care about testing generalization to train until there is 100% accuracy on short sequences before testing on longer ones.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Log-Depth Recurrent Units (LDRU) and shows that it can generalize well in multiple regular language tasks. LDRU is effectively implemented as a balanced binary tree recursive neural network with a new gated cell as the parametrized binary operator.

### Strengths
* It demonstrates that binary balanced tree-based recursive neural networks with a modern gated cell can perform nearly perfect in multiple parity tasks.

### Weaknesses
* The contribution seems incremental compared to RegularGPT. RegularGPT seems to be essentially the same idea (a balanced binary tree recursion - except uses Transformer as the recursive cell) - and also performs near perfectly (better in the original paper). The improvement in this paper seems like due to minor implementational difference rather than exposing any theoretical leap. 
* Limited evaluation on anything else besides algorithmic regular language tasks. Does not show if it can be anything more than something that works well in "toy" tasks. While I do appreciate evaluation on algorithmic/synthetic tasks - particularly because they can be harder to "hack" by finding spurious shortcuts and easier to analyze - however, restricting the whole study to them when introducing a supposedly general purpose model restricts the scope severely. 
* Seems to miss a lot of critical related paper:
   - First of all, LDRU seems like a neologism for Balanced Tree Recursive Neural Networks - which have been explored in multiple prior works [1,2,3].
   - Moreover it misses any comparison with more recent proposals like Recursion-in-Recursion (RIR) which also proposes logarithmic-depth recurrence/recursion and shows effectiveness in algorithmic tasks - like propositional logical inference, listops (in multiple OOD settings) - alongside benchmarks in LRA and other [4] and utilizes a modern gated recurrent cell inspired from Ordered Memory. At the very least, I would be curious how the proposed method compares against RIR in ListOps and Logical Inference on the same length generalization and argument generalization settings in RIR. But to be frank - even if the benchmarks are provided I would likely not lean towards acceptance unless the other weaknesses are rebutted well enough. 


[1] Neural Tree Indexers for Text Understanding - Munkdhalai et al. 

[2] Sliced Recurrent Neural Networks - Yu et al.

[3] On Tree-Based Neural Sentence Modeling - Shi et al. 

[4] Recursion in Recursion: Two-Level Nested Recursion for Length Generalization with Scalability - Ray Chowdhury et al.

### Questions
n/a

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Log-Depth Recurrent Unit (LDRU), a novel neural architecture designed to address the challenge of out-of-distribution (OOD) length generalization. The LDRU is inspired by the algebraic structure of monoids and processes sequences using a learned pairwise operator in a balanced reduction tree, resulting in $O(\log n)$ computational depth. The authors evaluate the LDRU on a comprehensive suite of 21 regular language tasks, including standard benchmarks and a new "prefix language" benchmark they introduce to specifically test long-range dependencies. The empirical results are very strong, showing that the LDRU achieves 100% OOD accuracy on 18/21 tasks and near-perfect (>=96%) on the remaining three, consistently outperforming RNN, LSTM, Transformer, and RegularGPT baselines.

### Strengths
1. The LDRU architecture is with a theoretical motivation. Grounding the architecture in the concept of monoid composition and the reduction algorithm provides an alternative to standard recurrent (state-based) or attention-based (fixed-depth) approaches.
2. The LDRU's performance on the 21 regular tasks is better than other architectures (RNN, Transformer, LSTM).

### Weaknesses
1. The primary weakness of this paper is its exclusive focus on regular languages. While the authors justify this as a rigorous and verifiable testbed, regular languages are the simplest class in the Chomsky hierarchy. The paper provides a compelling proof of concept, but it leaves the most critical question unanswered: does this approach scale to more complex, non-regular tasks? It is entirely unclear if the LDRU's inductive bias, which aligns so well with monoids (and thus regular languages), will be beneficial or detrimental for context-free or, more importantly, context-sensitive languages like natural language.
2. While the $O(\log n)$ depth and $O(nd^2)$ work complexity are good, the paper doesn't compare LDRU directly to baselines (like RNNs or Transformers) in terms of wall-clock time or throughput.

### Questions
Could you conduct more experiments to address weaknesses 1 and 2?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a new architecture, Log-Depth Recurrent Unit (LDRU). LDRU provides a trade-off between width (i.e. work complexity) and depth between current architectures such as RNN and Transformer. Experiments on a wide range of generalization tasks demonstrate the improved performance of LDRU over RNN, LSTM, Transformer and RegularGPT.

### Strengths
This paper presents an interesting way to model sequences in a binary-tree manner. It is intuitive that the computation overhead can be reduced in this way. There are many experiments in both the main pages and the appendix, and many examples in the appendix to help

### Weaknesses
1. I have only a little background on DFA and I feel the preliminary section is technically heavey and hard to follow. I would suggest to relate the preliminaries with examples in natural language as it is the final testbed for proposed LDRU.

2. The evaluation tasks of this work are not on natural language. I wonder if it is possible to model natural language with LDRU (for example, next-token generation, reasoning, long-context modeling, etc). The tasks in this paper are specific tasks, not general language modeling.

3. The baselines in this paper seem not enough, as RNN, LSTM, and Transformer have been introduced many years ago, and positional encodings are deactivated as described in line 236. I wonder if more recent variants could be compared with the proposed LDRU.

### Questions
1. I don't quite get how the proposed neural architecture is related to the finite automatons and prefix languages. I think I missed this part, but I only understand the proposed LDRU as a new kind of attention that is not fully-connected?

2. At first glance I thought this work is about long-context modeling, but turns out it's not. Could you let me know what's the difference between the area of this research from long-context modeling (e.g., context window of Mistral is 8192, but we can extend it with some techniques)? I guess I'm assigned to review this paper because of my background on long-context modeling, and this could be why I cannot follow this paper well, as the tasks in this paper are not standard long-context benchmarks.

3. Are the tasks too simple that LDRU (and some baselines) can achieve 100.0 +- 0 performance? Are we reaching the limit of this research direction as we're achieving 100.0?

4. If we can do binary tree, how about ternary and more generally n-ary trees? Will it be challenge because LDRU is relying on pairwise operations (which only applies for binary tree structure)?

### Soundness
3

### Presentation
2

### Contribution
3
