# A Differentiable Alignment Framework for Sequence-to-Sequence Modeling via Optimal Transport

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
State-of-the-art end-to-end (E2E) ASR systems, such as the Connectionist Temporal Classification (CTC) and transducer-based models, suffer from peaky behavior and alignment inaccuracies.
In this paper, we propose a novel differentiable alignment framework based on one-dimensional optimal transport, enabling the model to learn a single alignment and perform ASR in an E2E manner.
We introduce a pseudo-metric, called Sequence Optimal Transport Distance (SOTD), over the sequence space and discuss its theoretical properties.
Based on the SOTD, we propose Optimal Temporal Transport Classification (OTTC) loss for ASR and contrast its behavior with CTC.
Experimental results on the TIMIT, AMI, and LibriSpeech datasets show that our method considerably improves alignment performance compared to CTC and the more recently proposed Consistency-Regularized CTC, though with a trade-off in ASR performance.
We believe this work opens new avenues for seq2seq alignment research, providing a solid foundation for further exploration and development within the community.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a differentiable seq2seq alignment via 1D OT (SOTD pseudo-metric + OTTC loss) achieving linear time/space and learning a single path to curb CTC's peaky behavior.

### Strengths
1. This paper proposes a differentiable, single-path OT-based alignment with linear time/space complexity that directly targets CTC’s peakiness.
2. It introduces SOTD with theoretical guarantees (existence of a minimum), providing a clear mathematical framework for sequence comparison.
3. Results show improved alignment metrics (reduced peakiness, better alignment scores) on TIMIT, AMI, and LibriSpeech, supporting the method’s conceptual validity.

### Weaknesses
1. Despite cleaner alignments, WER generally lags behind competing baselines, making practical utility unclear. Moreover, the paper does not analyze when/why the method should be preferred.
2. Limited baseline coverage and weak empirical comparisons: stronger alternatives (e.g., RNN-T, hybrid CTC/Attention, recent CTC variants) are omitted, architectural breadth is narrow (e.g., Conformer/FastConformer and other modern encoders are not considered), and overall experimental comparisons are notably insufficient.
3. The linear-complexity claim is unsubstantiated. There are no runtime/memory benchmarks or trade-off analyses against baselines.
4. Sections 1~3 contain excessive and non-essential material. The exposition is unfocused, and writing quality is poor.

### Questions
Please refer to the concerns outlined in the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel differentiable alignment framework for sequence-to-sequence tasks, with a focus on automatic speech recognition (ASR). The core idea is to model the alignment between input (e.g., audio frames) and output (e.g., text tokens) sequences as a parameterized, learnable mapping based on one-dimensional Optimal Transport (OT). The authors introduce the Sequence Optimal Transport Distance (SOTD), a pseudo-metric on the sequence space, and derive the Optimal Temporal Transport Classification (OTTC) loss for ASR. OTTC jointly learns the alignment and frame classification by maximizing the probability of a single, optimal alignment path, in contrast to CTC which marginalizes over all valid paths. Experiments on TIMIT, AMI, and LibriSpeech show that OTTC significantly mitigates the "peaky behavior" (over-assignment to blank tokens) inherent in CTC and improves alignment metrics like Intersection Duration Ratio (IDR), albeit with a trade-off of slightly higher Word Error Rates (WER).

### Strengths
- Originality: The formulation of sequence alignment as a parameterized 1D OT problem is neat. The idea of learning a single, high-quality alignment path instead of marginalizing over many (often peaky) paths is interesting.
- Quality: The paper is technically sound. The theoretical foundation for SOTD as a pseudo-metric is well-established (Propositions 1-4). The experimental design is good, using multiple datasets and good baselines (CTC, CR-CTC) (but not great ones, see below)
- Clarity: The paper is generally well-written. The problem formulation, the connection between discrete monotonic alignments and 1D OT, and the derivation of the OTTC loss are explained pretty clearly. 
- Significance: The authors propose a principled and efficient ($O(\max(n,m))$ complexity) framework to directly improve alignment, which could potentially be interesting in seq2seq applications (but see below on the performance)

### Weaknesses
- Performance: The most significant weakness imho is the consistent degradation in WER compared to CTC across all datasets (Table 2). While the paper positions itself as an alignment-focused improvement, for the broader ASR community, a drop in WER is an important drawback. Also, the authors did not compare to strong transducer baselines or modern streaming architectures; decoding is greedy only if I understood correctly.
- Ablation on $\beta$: The choice of the target distribution $\beta_q$ is uniform and fixed. The oracle experiment in the appendix (A.4.1) suggests that a non-uniform, learned $\beta$ could be highly beneficial. In light of the remaining performance issues, the paper would be significantly stronger if it included a proper investigation into learning or optimally setting $\beta$, rather than leaving it as a promising but unexplored direction.

### Questions
1. Why do you think you find the WER gap between OTTC and CTC? Is the single-path objective inherently more prone to overfitting or local minima compared to the multi-path marginalization of CTC?
2. Learnable $\beta$: The oracle experiment is compelling. Why was a learnable $\beta$ (with appropriate constraints to ensure non-zero components) not implemented and evaluated in the main experiments? Could a learned $\beta$ potentially close the WER gap while retaining the alignment benefits?

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
3

### Summary
This paper presents the SOTD training objective, which can be applied to the seq2seq tasks like ASR. Compared with previous training objectives like CTC and CR-CTC, the proposed method shows less peaky behavior and smoother alignment prediction.

The SOTD is rooted in the optimal transport theory. It assumes the input and output as two series of one-hot distributions, based on which the author defines the Sequences Optimal Transport Distance (SOTD). The authors also compared the proposed method and previous CTC and DTW methods.

### Strengths
Strengths: (1) The paper has a good theoretical foundation. The reviewer hasn't found a noticeable flaw so far, although the reviewer may not have enough knowledge to fully check the correctness of all theoretical part. (2) The paper conducted enough experiments on TIMIT, AMI, and Librispeech, showing its ASR and alignment performance. For a theoretical paper, this experimental scale is enough (although further scaling could be more convincing, it's not necessary)

### Weaknesses
Weakness: (1) The proposed method shows a little bit worse performance, compared with CTC, as shown in Table 2. As an early exploration, this is acceptable.

### Questions
About the time and space complexity: The authors claim that the proposed method has linear time and space complexity over the seq2seq task, which should be a very nice property.

I would appreciate it if the authors could add a comparison of the complexity between the proposed method and CTC. If you can show that the proposed method has better time and space complexity, I guess it would be widely applied to some extreme cases. E.g., for the super-long seq2seq tasks, like long-form ASR, the CTC would have complexity more than linear, which makes it infeasible to fit the GPU memory and becomes slow. If the proposed method is theoretically more suitable for this situation, I would encourage the authors to address this slightly.

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
2

### Summary
This paper proposes a new differentiable alignment framework based on one-dimensional optimal transport for sequence-to-sequence modeling. Experimental results on three datasets show that our method considerably improves alignment performance compared to Connectionist Temporal Classification (CTC) and the more recently proposed Consistency-Regularized CTC.

### Strengths
1. This paper proposed a new differentiable alignment framework based optimal transport for sequence-to-sequence modeling.
2. Empirical results show this method outperforms the CTC baselines.
3. The differentiable idea is clearly described in this paper.

### Weaknesses
1. The application scenarios of the proposed method appear limited, since there are only speech tasks in the evaluation. To demonstrate the broader generalizability of the approach, the authors might consider including experiments on video tasks.
2. There are very limited baselines in the experiments. Authors might consider adding some transducer-based models in the comparison, to make the empirical results more solid.
3. No codebase is released or provided to ensure reproducibility.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
