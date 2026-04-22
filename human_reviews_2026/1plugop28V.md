# On the Measurement and Efficient Mitigation of Length Generalization Gaps in Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Large Language Models (LLMs) typically train on short text due to the quadratic complexity of their self-attention architectures. 
As a result, their performance suffers drastically on inputs longer than those encountered during training, substantially limiting their applications in real-world tasks involving long contexts.
In this paper, we rigorously establish an upper bound on length generalization in the measurement space and identify two length-related factors that limit performance. 
Our theory explains two recent observations: **_(i)_** out-of-distribution positions in longer contexts reduce length generalization, and **_(ii)_** fine-tuning on entire sequences is not necessary. 
Motivated by these insights, we propose _Virtual-context Learning_ (_VCL_), a flexible method that requires minimal modifications to most fine-tuning approaches.
Experiments on various tasks show that _VCL_ allows LLMs to generalize to 4 $\times$ context windows while retaining perplexity and improving performance on downstream tasks such as Passkey Retrieval and LongBench. 
_VCL_ brings substantial efficiency improvements, reducing decoding time and memory usage by up to 50\% compared with fine-tuning baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a measure-theoretic framework for analyzing length generalization in large language models (LLMs). The authors identify two key limiting factors: short-length bias caused by training on limited context windows, and distribution shift between training and inference contexts. They introduce **Virtual-context Learning (VCL)**, a fine-tuning strategy that selectively updates parameters associated with out-of-distribution (OOD) tokens. By leveraging Wasserstein distance to quantify distributional divergence, VCL enables models to extrapolate to context lengths up to four times longer than those seen during training while maintaining strong performance and reducing computational overhead.

### Strengths
The paper offers an intellectually interesting perspective by framing length generalization in large language models through a measure-theoretic lens. This perspective allows the authors to formalize the relationship between training and inference distributions in a mathematically principled way. In particular, the use of Wasserstein distance to characterize distribution shift between short and long contexts provides a coherent theoretical account of why models often fail to extrapolate beyond their training sequence lengths.

Beyond the theoretical contribution, the proposed Virtual-context Learning (VCL) method demonstrates clear empirical value. The approach is practically appealing because it targets only out-of-distribution tokens during fine-tuning, thereby reducing computational overhead while still improving performance. Experimental results on passkey retrieval, language modeling, and LongBench consistently show that VCL enables models to operate effectively on context windows up to four times longer than those seen during pretraining. These results are backed by thorough experimentation and ablation studies, which strengthen the empirical claims.

### Weaknesses
The paper is mathematically dense and may be difficult to follow for readers without a strong background in measure theory, which could limit its accessibility to the broader NLP community.

The experimental scope is relatively narrow since all evaluations are performed on LLaMA-2-7B, leaving uncertainty about the generalizability of VCL to other architectures such as BERT or T5.

The method is not sufficiently compared to other long-context techniques like ALiBi, NTK-aware scaling, or sparse attention mechanisms, making it difficult to assess relative strengths in terms of efficiency and scalability.

The evaluation primarily focuses on long-context understanding and retrieval tasks, without examining the impact of VCL on other task categories such as reasoning, summarization, or natural language inference.

### Questions
How does VCL compare quantitatively with alternative length generalization techniques such as ALiBi, RWKV, or sparse attention in terms of memory usage, training cost, and runtime efficiency?

Could the proposed theoretical framework and VCL be extended to sparse or linear attention mechanisms, or to encoder-decoder architectures? If so, what modifications would be required?

How stable is VCL when scaling to extremely long sequences beyond 4× the original context window? Does performance degrade gracefully?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the problem of length generalization in transformer-based language models: how well a model trained on sequences of length $N$ can generalize to sequences of length $M > N$ at test time. The paper has two parts: a theoretical contribution and a practical method proposal. In the first part, using measure theory tools and certain assumptions, the paper derives a bound on the Wasserstein distance between the distributions of length-N and length-M attentions, showing that the upper bound does not grow with $M$. Next, the authors propose a simple method, Virtual Context Learning (VCL), to improve length generalization. Experiments show that VCL achieves lower perplexity on longer sequences than those seen during training, especially when combined with existing position interpolation methods.

### Strengths
- The paper includes a theoretical contribution that motivates the proposed method, offering a perspective useful for understanding length generalization in attention-based models. Detailed proofs and definitions of the mathematical results are provided in the appendix.
- The remark from the derived bound is interesting: self-attention can generalize to out-of-distribution sequence lengths if the empirical measure of attention embeddings does not shift (in the Wasserstein distance sense).
- The proposed method is simple to implement, achieves good results when combined with existing position interpolation methods, and improves training efficiency in terms of memory and latency.

### Weaknesses
- The presentation of the paper can be significantly improved. While the authors include detailed mathematical proofs and definitions in the appendix, the first half of the paper remains difficult to follow for readers without a background in measure theory. The paper could be made more accessible to a broader machine learning audience by focusing more on intuitive explanations of the assumptions and theorems rather than precise mathematical formalism. In particular, the connection between the theoretical results and the proposed method is weak in the current version and should be further clarified.
- Citations of previous works are sometimes confusing. For example, in lines 110–114, the paper discusses the works of Zhou et al. and Huang et al., but when mentioning the shortcomings of “these studies,” it cites Han et al. and Press et al., which makes the paragraph unclear.
- Regarding the argument that not all tokens are needed for fine-tuning: the paper presents an experiment where gradients are applied only to the second half of the sequence, with the first half frozen. However, this does not clearly support the claim that “not all tokens are required,” since all tokens in the second half are still used. Please clarify the reasoning. Also, there seems to be a typo in line 314 (“215” should be “256”).
- The argument about reducing distribution shift through VCL (lines 346–348) is unclear. In standard language modeling, a sequence of length $M$ has $M$ losses (averaged). With VCL, $l$ out of $M$ losses are dropped, training only on tokens with at least $l$ context tokens. It is not clear why this would reduce distribution shift, please explain.
- The empirical results are incomplete:
  - Line 370 mentions full-length fine-tuning as a baseline, but perplexity results (Table 1 and Figure 4) do not include it. Please add full fine-tuning rows for models trained on 8k (and possibly 16k) sequences under the same budget and setup as Table 1. For Figure 4, include full-length fine-tuning results (training on all $M$ tokens of a sequence).
  - In Section 6.2 (passkey retrieval), the VCL setup is unclear. Please specify the exact values of $l$ and $M$ for the VCL-8k-yarn and VCL-6k-yarn experiments, and report results for VCL alone (without yarn).
  - In Section 6.3, clarify the experimental setup, including $l$ and $M$.
  - Table 2 lacks standard deviation values (e.g., from multiple runs with different random seeds). Also, please report VCL performance without PI methods.
  - The training duration (200 steps) seems too short. Would the improvements persist with longer training? Note that long-context adaptation typically involves a large number of tokens, e.g., [4] increases context from 2k to 8k with an additional 120B tokens.
  - More recent evaluation metrics, such as RULER [5], are missing.
  - Since VCL omits losses for tokens with shorter context, this could cause forgetting on regular benchmarks. Please include results on standard pretraining benchmarks to verify that forgetting does not occur.
- Some related works are missing. It is now common in language model training to use stage-wise or curriculum-based pretraining (e.g., starting with shorter and gradually increasing sequence lengths) to improve long-context performance [1, 2, 3]. Additionally, the paper does not discuss the practical challenge of limited long-context data, which is important for motivating the problem.

[1] Zhu, Tongyao, et al. "SkyLadder: Better and Faster Pretraining via Context Window Scheduling." arXiv preprint arXiv:2503.15450 (2025).

[2] Pouransari, Hadi, et al. "Dataset decomposition: Faster llm training with variable sequence length curriculum." Advances in Neural Information Processing Systems 37 (2024): 36121-36147.

[3] Jin, Hongye, et al. "Growlength: Accelerating LLMs pretraining by progressively growing training length, 2023." URL https://arxiv.org/abs/2310.00576.

[4] Li, Jeffrey, et al. "Datacomp-lm: In search of the next generation of training sets for language models." Advances in Neural Information Processing Systems 37 (2024): 14200-14282.

[5] Hsieh, Cheng-Ping, et al. "RULER: What's the Real Context Size of Your Long-Context Language Models?." arXiv preprint arXiv:2404.06654 (2024).

### Questions
- An interesting consequence of Theorem 4.6 is that $\mathbb{W}(\mu, \nu)$ is the primary factor in length generalization, rather than the sequence length $M$ itself. Could you comment on why there is a performance drop for a synthetic task  like passkey retrieval (Fig. 3)? Based on the intuition in Figure 1, one might expect no distribution shift for such a task even when the sequence length increases.
- The attention update formulation in line 154 is missing the commonly used output projection in Equation (3). While this might not affect the theoretical analysis, it could be worth noting explicitly.
- In line 149, it is stated that the Softmax-attention scaling factor $d_{QK}$ is assumed to be 1, but it is unclear where this assumption is used later, since the variable is still retained in the equations. Please clarify.
- In Definition E.2 (Empirical Measure Mapping), $m(X)$ for $X = \{x_1, …, x_N\} \subset E$ is defined as the average of the Dirac measures $\delta_{x_t}$. However, this seems inconsistent with Definition E.1, where all $\delta_{x_t} = 1$ since they belong to $E$. Could you please clarify this?

### Soundness
2

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
This paper presents a measure-theoretic framework for analyzing length generalization in LLMs, establishing an upper bound on the Wasserstein distance between attention outputs for sequences of different lengths. The bound depends on two factors:  $\sqrt{\ln{N}}$ (where N is the shorter sequence length) and $W(\mu,\nu)$ (the distribution shift distance). Based on these insights, the authors propose Virtual-context Learning (VCL), which fine-tunes models by computing loss only on out-of-distribution position tokens. Experiments on language modeling, passkey retrieval, and LongBench demonstrate that VCL achieves comparable or better performance than full-length fine-tuning while reducing memory usage and training time by approximately 50%.

### Strengths
1. The measure-theoretic approach to length generalization for different position embedding and extrapolation strategies is original and mathematically rigorous.
2. VCL is simple to implement with minimal code changes and delivers substantial computational savings, as is claimed by the authors.
3. The paper includes diverse tasks (perplexity, passkey retrieval, LongBench) and thorough ablations, demonstrating the method's effectiveness across different settings.

### Weaknesses
1. The transfer from theory to application needs more clarification. Specifically, although RoPE-equipped models use absolute position IDs, the PE models relative distance, which creates a less generalization gap than the analysis. When using RoPE, only finetuning on the full length lets the model learn the longest dependency length, which contradicts the authors' claim in Section 5.1. Alibi also largely mitigates this gap by introducing long-range decay. 
2. The performance of VCL on length >4K in Table 1 is suspicious. This Table indicates that VCL training can achieve length extrapolation far beyond the training range and largely improves upon YaRN, which needs further clarification. Does this result suggest that VCL can achieve infinite-length extrapolation (also evidenced by Fig 3)?
3. The efficiency benefit of the proposed method gradually diminishes when generalizing to extremely long sequence lengths.

### Questions
1. How tight is the bound in Theorem 4.6?
2. An analysis of the validity of Assumptions 4.3 and 4.4 would be beneficial. How well are they satisfied in real-world scenarios?
3. Please explain the Oracle setting in Section 5.1.
4. Why the results in Figure 4(a) different from the ones in Table 1?
5. How does the proposed method compare with other PE-manipulation methods like PoSE or LongRecipe?
6. Please use \citep instead of \citet when necessary.
7. "PE" in line 64 isn't defined before.
8. Figure 6 has the wrong y-axis title and a typo in its caption.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a theoretical analysis for attention distribution shift and proposed a simple method for adapting LLMs to longer contexts by only tuning parameters for later tokens.

### Strengths
1. A (unsprisingly) simple method for fine-tuning LLMs to longer contexts.
1. A theoretical perspective for understanding the length extension failure for current LLMs.
1. Results show efficient efficacy in extending LLMs to longer-context tasks.

### Weaknesses
1. Descriptions of past work in the first paragraph in Related Work, "Length generalization" are not accurate and look carelessly written.
1. Unclearly explained how the wassterstein distance between short and long context attention is related to length-generalization. It seems a rather expected phenomenon. The writing hint that "which is the primary factor driving length generalization failures" but not clearly explained why so, and why the opposite could not be true.
1. The proposed method is only distantly related to the theory. The conceptual connection is loos, and little proof is provided on how the solution alleviates the terms in Theorem 4.6
1. A minor weakness: results are okay for research concept-proving, but not evaluated at the scale of sota LLM models which are already trained on longer context. This limits the downstream impact. I could understand this if it is due to resource limitations (but authors mentioned only 8 A100 GPUs so they might have the resource to do that. Not sure why they didn't evaluate on larger models).
1. Also, baselines only include those by 2023 so seem limited.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2
