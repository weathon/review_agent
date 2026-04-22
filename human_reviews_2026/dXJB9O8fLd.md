# Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
Transformer-based language models rely on positional encoding (PE) to handle token order and support context length extrapolation. 
However, existing PE methods lack theoretical clarity and rely on limited evaluation metrics to substantiate their extrapolation claims. 
We propose the Bayesian Attention Mechanism (BAM), a theoretical framework that formulates positional encoding as a prior within a probabilistic model. 
BAM unifies existing methods (e.g., NoPE and ALiBi) and motivates a new Generalized Gaussian positional prior that substantially improves long-context generalization. 
Empirically, BAM enables accurate information retrieval at $500\times$ the training context length, outperforming previous state-of-the-art context length generalization by more than $25\times$ in retrieval accuracy while maintaining comparable perplexity and introducing minimal additional parameters.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the Bayesian Attention Mechanism (BAM), framing positional encoding as a Bayesian positional prior that unifies methods like NoPE and ALiBi and introduces GGD-BAM, achieving strong long-context retrieval up to ≈256K tokens with <1000 extra parameters and improved stability when combined with SSMax on models from 120M–1.1B trained on FineWeb10B/Wikipedia.

### Strengths
- Formalizes positional encoding as a Bayesian prior, providing a useful theoretical advance.
- Demonstrates robust long-context retrieval (≈256K tokens) with minimal parameter overhead.
- Presents ablations and visualizations that enhance interpretability.
- Supplies clear experimental documentation and reproducibility details in the appendices.

### Weaknesses
1. Experiments stop at 1.1B parameters, leaving scaling behavior at 7B/13B+ unknown.
2. Evaluation is narrow (FineWeb, Wikipedia, passkey retrieval, RULER–NIAH) and omits broader benchmarks and domain corpora.
3. Training overhead analysis lacks detailed wall-clock and memory cost measurements for ultra-long fine-tuning.

### Questions
1. Is there an empirical correlation between the normalization $Z$ and long-context extrapolation, and can $Z$ be optimized or learned to extend coverage?
2. Does the optimal $β$ shift with larger training contexts (e.g., 2048), and can you provide experiments across different training-context regimes?
3. Why do larger models (1.1B) tend to overfit the training context more, and can you analyze the representation or optimization dynamics behind this?
4. Could a per-layer or learned $β$ schedule better balance local vs. long-range dependencies and reduce local-context suppression when $β<0$?

Overall, while the idea is conceptually interesting and well-presented, the current empirical evidence is insufficient to justify acceptance at this venue.

### Soundness
2

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
3

### Summary
This paper introduces a new positional encoding design, BAM, which is a variant and improvement of the existing ALiBi. 
The authors reformulate current positional encodings from a Bayesian perspective and provide a clear motivation for their proposed modification. Experiments in long-context perplexity and in-context copying tasks (PassKey Retrieval and NIAH) demonstrate BAM's effectiveness in length extrapolation for language models with up to 1B parameters.

### Strengths
- The perspective on improving ALiBi is both interesting and well-motivated.

- The authors provide a theoretical analysis that explains their motivation and supports the proposed approach.

- The authors conduct extensive experiments to demonstrate the effectiveness of their method, providing compelling evidence for its validity.

### Weaknesses
- Some experimental results are not convincing enough to support the authors’ contributions (See Questions).

- The presentation of this paper could be improved. 
The section on theoretical proofs somewhat detracts from the main focus of the work. 
While naming the new PE "Bayesian Attention Mechanism" is justifiable, it make sense but feels somewhat forced. 
Several important experimental conclusions are scattered throughout the Appendix, which makes it challenging for reviewers to follow.

### Questions
---
Q1: Can the authors include additonal baselines to better demonstrate the effectiveness of BAM methods?

Some relevant methods are missing and should be discussed in the experiments. For example, the RoPE variants (PI, NTK, YaRN, etc), which build on RoPE, should be included .   
They are the unique benefits of RoPE design, enabling length extraploation either at little to no cost.

Moreover, discussing the latest design in relative positional encodings [1] would further strengthen the case for BAM's effectiveness.

[1] HoPE: A Novel Positional Encoding Without Long-Term Decay for
Enhanced Context Awareness and Extrapolation.

---
Q2: Can the authors continual pre-train these models with more tokens?

We appreciate the authors's efforts in conducting these experiments.
However, the training corpus used for the pre-training of these toy models remains limited.
Given this, conclusions drawn from such a limited number of pre-training tokens are not entirely convincing. 
It is well known that ALiBi can accelerate convergence during training. 
So, will BAM still exhibit its advantages when pre-training with more tokens (at least 100B tokens)? 
Hope the authors can clarify that.

---
Q3: More comprehensive evaluation of the long-context abilities.

The existing experiments (PassKey and NIAH) provide some preliminary evidence of BAM's in-context copying ability for long contexts. 
However, BAM's other abilities in long-context is still underexplored, such as many-shot ICL, integration, and reasoning?

Building on the resolution of Q1 & Q2, could the authors test BAM on real-word tasks of long-context benchmarks (e.g., LongBench and L-Eval)?

--- 
Q4: What's the generative abilities of these PEs with normal lengths.

We acknowledge the authors' experiment in Appendix D.1, where RoPE shows lower perplexity than BAM on Wikipedia.
We agree with the authors that perplexity alone does not accurately reflect these models' ability.
So, what is the performance of these models at different scales, using different PEs, on language modeling tasks such as ARC, Hellaswag, PiQA, etc.?
Could the length-extrapolation ability of BAM emerge at the cost of its performance on shorter texts?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes Bayesian Attention Mechanism (BAM), a probabilistic formulation of self-attention where positional encoding (PE) is treated as an explicit prior over token positions.

### Strengths
Disclaimer: I don't know the field of positional encoding. I've alerted AC for my lack of knowledge in the domain.

The method can enable retrieval for context length far beyond training context length with small number of parameters added, which seems nice to me.

### Weaknesses
Disclaimer: I don't know the field of positional encoding. I've alerted AC for my lack of knowledge in the domain.

I'm unsure about the experimental analysis section of the paper. I don't know if the benchmarks used in the paper includes most of the popular ones in the field of positional encoding. I don't know if the paper uses enough baselines to compare their method with.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the Bayesian Attention Mechanism (BAM), a theoretical framework that reformulates self-attention as a probabilistic model where positional encoding (PE) acts as a prior over token positions. The authors show that existing PEs like NoPE and ALiBi correspond to specific priors (Uniform and Laplace) and propose a new Generalized Gaussian prior (GGD-BAM) to improve long-context extrapolation. They validate the approach empirically on language modeling, passkey retrieval, and the RULER benchmark.

### Strengths
1. The paper provides a clear and rigorous probabilistic interpretation of positional encoding in self-attention, unifying multiple existing approaches under a single Bayesian view. The derivations are mathematically consistent and offer new insight into the role of priors in attention mechanisms.
2. Framing positional encoding as a prior elegantly connects methods like NoPE and ALiBi that were previously viewed as unrelated heuristics. This contributes to a more principled understanding of extrapolation in transformers.
3. The visualizations of positional priors and learned β values provide interpretability, showing how certain heads specialize in long-range retrieval.

### Weaknesses
1. The main experiments are restricted to models with limited parameters. It remains unclear whether the extrapolation benefits persist at the scale of modern large language models (7B–70B), where positional behaviors often change.
2. The work focuses on retrieval and perplexity but does not assess downstream tasks like long context QA or summarization, where long-context comprehension is critical.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2
