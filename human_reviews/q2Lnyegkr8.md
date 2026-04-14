# Forgetting Transformer: Softmax Attention with a Forget Gate

- Decision: Accept (Poster)
- Scores: 6, 8, 5, 8

## Abstract
An essential component of modern recurrent sequence models is the forget gate. While Transformers do not have an explicit recurrent form, we show that a forget gate can be naturally incorporated into Transformers by down-weighting the unnormalized attention scores in a data-dependent way. We name this attention mechanism Forgetting Attention and the resulting model the Forgetting Transformer (FoX). We show that FoX outperforms the Transformer on long-context language modeling, length extrapolation, and short-context downstream tasks, while performing on par with the Transformer on long-context downstream tasks. Moreover, it is compatible with the FlashAttention algorithm and does not require any positional embeddings.  Several analyses, including the needle-in-the-haystack test, show that FoX also retains the Transformer's superior long-context capabilities over recurrent sequence models such as Mamba-2, HGRN2, and DeltaNet.  We also introduce a "Pro" block design that incorporates some common architectural components in recurrent sequence models and find it significantly improves the performance of both FoX and the Transformer.
Our code is available at [`https://github.com/zhixuan-lin/forgetting-transformer`](https://github.com/zhixuan-lin/forgetting-transformer).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce the Forgetting Transformer, a transformer variant capable of dynamically focusing on various scopes of history. The mechanism is inspired by the data-dependent forget-gate in recurrent models, which are widely used in modern RNNs. The proposed method demonstrates strong length generalization and long-range capabilities, which the authors evaluated through several benchmarks, including LongCrawl64,  Needle, LongBench, and others. Finally, the authors present an efficient implementation that relies on flash attention.

### Strengths
1.	The proposed transformer variant is simple, elegant, and effective, demonstrating strong performance. If the authors' claims generalize beyond the specific cases presented, the potential impact could be significant. 
2.	The selected benchmarks and metrics are commendable, incorporating long-range tasks across synthetic and real-world scenarios, such as Needle and LongBench, along with per-token perplexity measurements.
3.	The ablation studies in Figure 4 are essential, offering valuable insights into the method’s internal dynamics.
4.	Empirical results show notable improvements compared to the baselines.

### Weaknesses
I appreciate the ideas and results presented in the paper. However, I have several concerns about the informativeness, consistency, and importance of the method/results:

**W.1. Empirical Evaluation is Insufficient:** 

**W.1.1. Current Insufficiency in Empirical Evaluation (Sensitivity to Hyperparameters and Robustness):** The NLP results rely on a **single** instance of the pre-trained forgetting transformer with a specific pre-training setup, which is insufficient to fully support the paper’s claims. Despite impressive performance on length generalization, context tasks, and zero-shot tasks, the robustness and **generality** of these results remain uncertain. I’m not convinced that the results are **not anecdotal**. To strengthen the evaluation, I suggest:

   (i) Include **training curves** comparing the variants presented in Figure 1. 

   (ii) **Additional Datasets:** Conduct pre-training from scratch on more datasets. Even results on a smaller dataset like Wikitext-103 can be valuable. Ideally, it would benefit the community if the authors could compare their method directly with results from original papers. For example, several methods in this domain present results on Wikitext-103 (e.g., HGRN), the Pile (e.g., Pythia, Mamba, RWKV, HGRN, and others), and RedPajama. 

   (iii) Report results from **multiple seeds** to ensure stability.

   (iv) **Robustness Across Models Sizes, Hyperparameters and Settings:** Show that results are not anecdotal by assessing robustness across different hyperparameters and settings, such as varying model sizes (see Mamba and RWKV, for example), other modalities, and different hyperparameters.

  W.1.2. Important details about the exact number of parameters, FLOPs, and latency are missing in the tables

___

**W.2. Missing baselines and related works:**
 Several transformer variants are similar to the proposed mechanism. The differences between these and the forgetting transformer should be discussed, and some of them should be used as baselines. Examples include:

   (i) Mega [1]: This method incorporates Exponential Moving Average before multiplying the Q and K matrices. While this mechanism is not data-dependent, it has several similarities, including manipulation of the attention matrix and adding recency bias.

   (ii) COPE [2]: This also manipulates the attention matrix and adds recency bias (in a data-dependent manner). It can be interpreted as a special forget gate. 

   (iii) Selective Transformer [3]: In a very recent work, a mechanism similar to the forgetting transformer is introduced (see Section 3.3 and Equation 1).

   (iv) LAS Attention [4]: This method also adds recency bias (referred to as "local" in the paper) to the transformer by manipulating the attention matrix. It can also be interpreted as a (data-independent) forget gate.

___

I appreciate the ideas, results, and certain aspects of the empirical methodology in the paper. If my concerns are addressed during the discussion period, I will increase my score.

[1] Mega: Moving Average Equipped Gated Attention. Ma et al. ICLR23.

[2] Contextual Position Encoding: Learning to Count What's Important. Golovneva et al.

[3] Selective Attention Improves Transformer. Leviathan et al.

[4]  Viewing Transformers Through the Lens of Long Convolutions Layers. Zimerman et al. ICML24.

### Questions
1.	What is the impact of sharing the new parameters (w_f) across heads?
2.	It would be very interesting if standard LLMs could be improved by converting them into a forgetting transformer. For example, you could take Llama 3 7-B, convert it to the forgetting form, fine-tune it, and improve the results. This could be a breakthrough. Have you tried that? What are your thoughts on this?
3. Could you please shed more light on the inner dynamics of the forgetting transformer? For instance, it would be interesting to explore whether different heads focus on different scopes. This could be examined by measuring the F_{ij} values across heads. Additionally, a figure that visualizes the maximal forget values F_{ij} (for j=1, for example) across different positions and inputs could illustrate the forgetting trends (receptive field) learned by the model. While these aspects are partially explored in Figure 2, a more dedicated evaluation considering several inputs, layers, and heads could be very informative.
4. What are the limitations or failure cases of the forgetting transformer? Negative results could be very informative and show the differences between transformer variants.

### Soundness
2

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
This paper proposes a modified causal attention mechanism for transformers which incorporates a _forget gate_, inspired by forget gates in recurrent neural networks. The forget gate can take the place of traditional positional encodings, so the overall architecture remains quite simple. The authors show that the forget gate significantly improves length generalization when compared to a standard transformer architecture with rotary positional encodings, and it is competitive with non-transformer architectures such as Mamba 2.

### Strengths
The paper presents a simple, easily implementable idea that seems to improve upon prior state of the art in autoregressive language modeling. The results on length generalization are particularly compelling when compared to RoPE.

### Weaknesses
The authors use QK-norm in combination with their proposed method for many experiments, finding that it significantly improves performance. The paper would be improved if there were some investigation of why QK-norm seems to be important for optimal results, especially since QK-norm was introduced in the context of Vision Transformers rather than language models. It would also be nice to see results with and without QK-norm on the standard transformer baseline, so we can see if it is actually more important to use QK-norm with Forgetting Transformers than it is with normal transformers.

### Questions
The exposition was clear and I have no questions at this time.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposed to incorporate a forgetting gate, which is working as a decaying mechanism, into standard attention mechanism. The forgetting gate was directly applied to the unnormalized attention scores to down-weight long-term dependencies. 

The authors pre-trained a 760M model on 16B tokens. Compared with Transformer with full attention and some recurrent architectures, such as Mamba2, HGRN2 and DeltaNet, the proposed Forgetting Transformer architecture obtained better performance on long-context retrieval task such as needle-in-haystack, and other widely used long- and short-context benchmarks.

### Strengths
The paper is well-written, and the proposed architecture is well-motivated and easy to implement.

### Weaknesses
However, there are weaknesses that the authors need to address:

1. The scale of training dataset is relatively small (16B tokens).

2. The forgetting gate works as a decaying mechanism to down-weight historical contextual information. Hence, there are some important baselines missed in the experimental comparisons. First, one straight-forward baseline is Transformer with sliding window attention. The second is chunk-wise attention with moving average mechanism (such as Megalodon).

### Questions
Why does the impact QK-norm on Forgetting Transformer vary on different tasks?

What are the initialization methods for the parameter $w_f$ and $b_f$?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents an addition of a forget gate to standard causal attention. In particular, the paper connects the addition of a simple multiplicative forget gate to the recurrent formulation of attention with the parallel formulation of attention and shows how it results in a simple addition of a data-dependent additive bias to softmax attention. Importantly the bias depends on all previous inputs and not just the current one. A thorough experimental evaluation shows that the proposed forgetting attention performs as well as other methods on short context tasks with possibly an improved performance on long context tasks where it performs the best in 9 out of 14 benchmarks. There are also ablations comparing the proposed forget gate to learnable or fixed decay rates where it is shown that the proposed method performs better than either option.

### Strengths
- The writing of the background leading to the formulation of the forgetting transformer is very good. It is a pleasure to read and it flows naturally.
- The idea is simple both to implement and understand.
- There is thorough experimental evaluation and ablation studies. I especially appreciate the comparison with data independent and fixed positional biases.

### Weaknesses
The paper doesn't have a lot of weaknesses. The main question is how convincing is the experimental evaluation which even though it is thorough in the sense that it evaluates most if not all possible questions it is
1. on smallish models (as mentioned by the authors themselves as a limitation)
2. lacking a baseline without RoPE or any other positional encoding

The first one is less significant given the computational requirements but the second may be important especially since the evaluation is focused heavily on really long sequences for which RoPE may not necessarily be the best choice as well as sequence length generalization for which any positional encoding will hurt performance.

The results on table 1, for the transformer's Wikipedia perplexity for instance, are quite low which calls slightly into question the validity of the conclusions. Could it be the long context training set?

### Questions
My main question is also slightly mentioned at the end of the weaknesses section, the transformer seems to be significantly underperforming all methods in practically every benchmark. Is this an artifact of the long context training set? Could it be undertrained or the hyperparameters not tuned sufficiently?

Similarly, how would a transformer without any positional encoding perform on these benchmarks and especially on the needle-in-the-haystack test?

### Soundness
3

### Presentation
4

### Contribution
3
