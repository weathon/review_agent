# Efficient LLM Architectures

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Recent LLMs have hundreds of billions of parameters consuming vast resources. Furthermore, the so called "AI scaling law" for transformers suggests that the number of parameters must scale linearly with the size of the data.  In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, by comparing theoretical and empirical estimates of the Kullback-Liebler divergence, we derive a natural AI scaling law that the number of parameters in an efficient LLM scales as $D^{\gamma}$ where $D$ is the size of the training data and $ \gamma \in [0.44, 0.72]$, suggesting the existence of more efficient architectures.  Against this backdrop, we propose recurrent transformers, combining the efficacy of transformers with the efficiency of recurrent networks, progressively applying a single transformer layer to a fixed-width sliding window across the input sequence. Recurrent transformers (a) run in linear time in the sequence length, (b) are memory-efficient and amenable to parallel processing in large batches, (c) learn to forget history for language tasks, or accumulate history for long range tasks like copy and selective copy, and (d) are amenable to curriculum training to overcome vanishing gradients. In our experiments, we find that recurrent transformers perform favorably on benchmark tests.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper combines PAC learning theory and KL divergence equations to derive a new AI scaling law, which argues that the parameter count $N$ of an efficient LLM should be sublinear to the data size $D$. Based on this, the paper introduces an efficient architecture called the Recurrent Transformer. It adopts only one layer of Transformer, and the input is processed recurrently block by block. The historical hidden state is updated with a forget gate to enable forgetting, and the current block input is concatenated with the historical hidden state. Experiments show that the Recurrent Transformer excels over the standard Transformer with lower computational cost.

### Strengths
1. The paper derives a new AI scaling law arguing that the parameter count scales sublinearly with the data size, suggesting more efficient architectures. The theoretical analysis is useful and enlightening.
2. The new architecture, the Recurrent Transformer, is theory-driven and well-supported by the analysis above.
3. The Recurrent Transformer performs well on long-range copy tasks, which are a weakness for linear models.

### Weaknesses
1. There is a critical mismatch between the claims and the experimental scale. The paper derives a scaling law, but all the experiments involve very small-scale models (the maximum size is 11M). However, the emergent abilities of modern LLMs are considered to be something that only models with billions of parameters possess. A single-layer architecture with only 11M parameters outperforming a standard transformer provides little evidence that it is also effective at scales such as 1B, 7B, or larger.
2. The recurrent form of the transformer introduced in the paper is very similar to Transformer-XL[1]. Both combine the historical hidden state with the current input, and they both process the sequence block by block recurrently. Therefore, the Recurrent Transformer architecture may lack novelty.
3. The baselines are not sufficient. As the Recurrent Transformer has linear complexity, it would be better to compare it against other linear-complexity models such as Mamba and GLA.


[1] Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

### Questions
1. Can the authors provide any evidence or at least a reasonable argument to support that 'this single-layer architecture can maintain its effectiveness at the 1B+ parameter scale' ?
2. Why not choose Mamba or GLA as baselines? These models have extremely strong performance on long-sequence tasks and are also representatives of efficient architectures.

### Soundness
2

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
2

### Summary
The paper tackles an important problem and proposes an interesting architectural idea. The theoretical contribution is novel but needs strengthening. The experimental validation is insufficient for the claims made. With significant revisions addressing the theoretical gaps and experimental limitations, this could become a solid contribution.

### Strengths
1. Novel theoretical analysis: The connection between Kullback-Leibler divergence and unique sequence counts provides fresh perspective on parameter scaling

2. A Practical architecture: Recurrent transformers offer O(N) complexity vs O(N²) for standard transformers

3. Learnable accumulation parameter: The α parameter that learns to forget/accumulate history is elegant

### Weaknesses
1. Theoretical Issues

- Lemma 1 proof has logical gaps: The transition from dim(F) ≤ |S| to bounds on |S| relies on assumptions about optimal function spaces that aren't justified
- Assumption 1 is oversimplified: Finite precision with uniform quantization doesn't reflect actual neural network computation


2. Experimental Limitations

- **Very limited scale**: All experiments run on 16GB Mac Mini - cannot validate claims about large-scale efficiency
- **No comparison with recent efficient architectures**: Missing comparisons with Mamba, RWKV, RetNet, and other modern alternatives mentioned in related work
- **Cherry-picked baselines**: Comparing against "regular transformers" without positional encodings, modern optimizations

3. Presentation Issues

- Notation inconsistency: S used for both sequence set and individual sequences
- Missing details: How exactly is curriculum training scheduled? What are the learning rate schedules?
- Line 33: "sparse in that most parameters are negligible" - needs citation or evidence
- Line 270: Table reference formatting inconsistent
- Figure quality: Figures 1-3 have overlapping legends and are hard to read
- Related work: Missing discussion of recent efficient transformers, eg Mamba, RWKV, RetNet

### Questions
- Can you provide experiments at larger scale (>1B parameters) to validate the D^0.44-0.72 scaling law?
- Why does the theoretical analysis assume non-duplicative training corpora when real LLM training uses multi-epoch training?
- How does the recurrent transformer handle variable-length sequences during inference?
- What is the actual memory footprint comparison with baselines during training and inference?
- Can you provide ablation studies on:
    - Block size K
    - Number of layers
    - Impact of α initialization
- How does performance scale with sequence length beyond 4128 tokens tested?

### Soundness
2

### Presentation
2

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
This paper proposes an efficient LLM architecture, namely recurrent transformer. The authors first consider the PAC-learning theory and claims that the theoretical-optimal model size should not be linearly scaling with the dataset size. More efficient architecture exists. Then they proposes the recurrent transformers, empirically validated over three toy tasks.

### Strengths
1. The good part of this research is the timely topic. Efficiency and scaling are central to current LLM research.
2. Recurrent transformer is lightweight, simple to implement, and tested on synthetic and small real datasets.

### Weaknesses
**On the theory side**:
1. The authors claims better scaling exists, however, the better scaling assumes the empirical fitting is true. The better scaling does not derive from the first principle, but accounting for the empirical scaling laws. 
2. The theory doesn't direct connects to the proposed recurrent transformer architecture. Indeed, the architecture is seemingly directly combines the RNNs and Transformers.
3. The paper defines a discrete loss $\sum_{p_s \neq q_s} p_s$, treating probabilities as equal/not equal. This is atypical for PAC learning and breaks continuity assumptions needed for the generalization bounds it later invokes.

**On the empirical side**:
1. Scale mismatch. Experiments run on CIFAR-10, toy copy/selective-copy tasks, and nanoGPT Shakespeare. None validate large-scale efficiency claims; results are limited to very small models.
2. No comparisons. Extensive research on llm architectures proposed very strong baselines, such as mamba, deltaNet. However, this paper none of them, even the RNN.

**Novelty and Originality**:
I am not famililar with the line of research on LLM architectures, but the combination of RNN and Transformers is seemingly a easy-and-intuitive idea. I expect the authors to discuss the related works extensively.

### Questions
N/A

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
1

### Summary
The paper argues the “true” parameter–data scaling should grow sublinearly by tying empirical KL scaling to PAC-learning bounds, not linearly as in transformer-specific fits. It then proposes a recurrent transformer that reuses a single layer over a sliding window with a learnable memory knob to trade off forgetting vs. accumulation. The architecture claims linear-time processing in sequence length, better memory use, and plug-and-play batching while staying competitive in small-scale tests.

### Strengths
1. The sliding-window recurrence with a single reusable block is a clean, minimalist way to chase efficiency without throwing away attention.

2. The “learn to forget or accumulate” knob aligns with the intuition that language vs. long-range tasks want different memory behavior.

### Weaknesses
1. A single small window and one layer may miss cross-block interactions that deeper stacks capture implicitly. 


2. The experiments run on modest hardware and narrow tasks, leaving open how this scales to modern pretraining or multi-billion-token corpora. 

3. The memory knob’s behavior is shown qualitatively, but guidance on when it converges to “forget” vs. “accumulate” is thin.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2
