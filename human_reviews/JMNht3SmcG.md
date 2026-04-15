# ShortGPT: Layers in Large Language Models are More Redundant Than You Expect

- Decision: Reject
- Scores: 5, 5, 3, 3

## Abstract
As Large Language Models (LLMs) continue to advance in performance, their size has increased significantly, with current LLMs containing billions or even trillions of parameters.  In this study, we identify notable redundancy across the layers of LLMs, where some layers contribute minimally to overall network functionality. To quantify this, we introduce a metric called Block Influence (BI) which use the similarity between layer's input and output to measure the importance of each layer. Based on the observation of layer redundancy, we propose a straightforward pruning method: layer removal, which eliminates redundant layers based on their BI scores. Our approach, termed ShortGPT, demonstrates superior performance over previous state-of-the-art pruning methods.  Moreover, ShortGPT is orthogonal to quantization-like methods, enabling further reduction in parameters and computation. The ability to achieve better results through simple layer removal, as opposed to more complex pruning techniques, suggests a high degree of redundancy across layers, not only in transformer models but also in non-transformer models. We hope this work will contribute to future research in LLM compression.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a layer pruning metric named BI and demonstrates its superior performance over previous state-of-the-art pruning methods on Llama2-7B and 13B, Baichuan2-7B and 13B. Besides, ShortGPT can be used together with other quantitative methods for further reduction in parameters and computation.

### Strengths
- This paper introduces a new layer pruning method, BI.
- This paper conducts experiments on non-transformer models, RWKV-7B and Mamba-2.8B.
- The paper contains many comparative experiments.

### Weaknesses
- Insufficient contribution. ShortGPT is orthogonal to the quantization method, which cannot be regarded as a contributing factor, as other pruning methods can achieve similar results.
- In this paper, PG19 is used for layer importance and perplexity calculation. What if we use other datasets like alpaca-cleaned [1] and mmlu?
- Table 5 can be adjusted to make it more beautiful.
- In eq.1, in fact, BI is a cosine distance, so can other distance formulas also be used as metrics for calculation? Why must it be cosine?

[1] alpaca-cleaned: https://huggingface.co/datasets/yahma/alpaca-cleaned

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a metric; Block Influence (BI) which the authors use as a proxy to remove the layers (prune) without affecting (minimal affect) on the model performance. The metric is assumed to capture the redundancy component of the layer, so intuitively, removing the blocks which are most redundant i.e lower BI makes sense.

### Strengths
1. The paper is easy to read and clear at most of the places, the flow is good but can be made even better.
2. The conceptual idea of coming up with metric (Block Influence) makes sense intuitively (section 2 and section 3) and it's a simple metric to implement/understand.
3. The experiments clearly demonstrate the effectiveness of the method.

### Weaknesses
**Clarifications**
1. From table 9 and other figures, the authors might have to clearly define what a "layer" means because the metric in use is Block Importance (BI), so the normal understanding would be Attention + FFN is a block (the original transformer has 16 blocks etc;). So from my understanding, when layer 5 is removed, I believe the Attention + FFN is replaced by Identity matrix. Is that correct? This definition has to clearly come out at the start.
2. The finding presented in L256-258 is interesting i.e more redundancy in depth compared to width. If there's any previous work, the authors can cite it. By plainly looking at Table 2, this statement cannot be totally true and needs further validation.

**Additional Experiments/Ablations (in the order of importance to contribution)**
1. L242: The authors has chosen PG19 to compute BI and model importance. Can an ablation be performed to understand the effect of calibration datasets? 
   - I would like to see if model X picks layer A, B, C when using dataset 1 and U, V, W when using dataset 2. i.e whether the model has different order for BI when using different datasets as this has different implications depending on the outcome!.
2. L429: Can the authors show results on number of layers removed (or % layers removed) with respect to model size and/or task-wise for a particular performance threshold (in Appendix maybe). 
   - I believe the current experiments has these insights, it's about presenting them and getting new insights when using BI i.e eg: 7B models can remove 5 layers but 13B models can remove 8 layers across tasks and/or task A, B, C can have q% layers removed but task X, Y, Z has only p% layers removed.
3. Given this falls under structured pruning, I believe more results regarding the speed-up/inference can be highlighted/commented in addition to the ones presented in Table 4.
4. Continuing from point 1 in clarifications, an ablation experiment will be helpful if we treat attention and FFN layers separately i.e remove attention layer 1, 4, 5 and FFN 2, 4, 6 etc;. Basically computing the BI after the attention block and FFN block separately; this is similar to design choice in [1], so the study can be complementary to understand the metric dependency on different blocks/layers.

**Relevant missing citations**
1. L131/Sec 2.2: The authors might have to consider citing some previous works [1, 2] which has discussed in detail and came up with similar observations regarding the final layer importance and what happens when we compress it. 
2. L250: The authors can cite [1, 3, 4] which has come up with similar observations i.e compressing >30% drops the model performance.

**Citations**
1. The Cost of Compression: Investigating the Impact of Compression on Parametric Knowledge in Language Models - https://arxiv.org/abs/2312.00960 
2. Fast model editing at scale - https://arxiv.org/abs/2110.11309
3. Are sixteen heads really better than one? - https://arxiv.org/abs/1905.10650
4. Compressing bert: Studying the effects of weight pruning on transfer learning - https://arxiv.org/abs/2002.08307

### Questions
1. The figure 2, is the experiment done to understand the effect of pre-norm vs post-norm setting keeping all the model training variables the same. If so, referring to table 11 and table 12, all the settings are not equal i.e the training tokens etc; why is that the case? An ideal comparison would be to have 2 architectures -> pre-norm vs post-norm and have all the rest the same and train it for model convergence (maybe the epochs will vary). Once these models has been trained for same loss, you can pick the hidden values and plot the similarities to observe the effect of pre vs post norm. 
2. The presentation in Table 5 can be made much better in terms of highlighting which one performs better and the authors can do a better job in explaining it in Sec 4.5

**Formatting**
1. I believe Figure 1 labels are not correct and missing some information.
2. Table 3 needs better highlighting/formatting of the results; sec 4.4 can bring up important insights more clearly.
3. Table 2 can be better formatted to support the statement in L256-258 (point 2 in clarifications)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces ShortGPT, a pruning technique for large language models (LLMs) that identifies and removes redundant layers without significantly impacting performance. The authors introduce a pruning metric called Block Influence (BI), which measures the importance of each layer by analyzing the similarity between its input and output representations. Layers with lower BI scores are considered redundant and removed. Through empirical evaluations, ShortGPT demonstrates superior performance compared to existing pruning methods, achieving high parameter reduction (around 25%) while maintaining approximately 90% of the model’s performance.

### Strengths
1.The proposed method of layer removal based on BI scores is straightforward, efficient, and performs comparably or better than more complex pruning methods, making it accessible and adaptable for practical use.

2.The proposed method is shown to work effectively on non-transformer models (RWKV and Mamba), indicating that the concept of layer redundancy may be applicable across various architectures, enhancing the method’s potential for broader use.

### Weaknesses
1.The novelty of this paper is limited, with a marginal contribution. The methodology section primarily introduces the BI pruning metric, measuring average cosine similarity between consecutive layer activations. However, [1] has already explored this, pointing out that for both Baichuan2-7B and Llama2-7B, similarity between layers 3 to 28 is typically near 1.

2.The results reported in the paper are unreliable:

a)	The tables in this paper contain numerous inconsistencies that undermine the credibility of the results. For instance, in Table 8, the SliceGPT results for Llama-2-7B and Llama-2-13B under a 20% pruning ratio are identical across PIQA, Winogrande, Arc-e, and Arc-c, yet differ for Hellaswag. Despite this discrepancy, the reported average remains inexplicably unchanged.
b)	According to Table 2, the results of the baseline methods are taken directly from [1]. Did the authors ensure a fair comparison by following the exact same settings (e.g., implementation environment, calibration dataset, benchmark script versions) as used in [1]?

c)	While the authors compared the results of ShortGPT against the baseline methods in the baseline method’s settings, they only claimed that “the same benchmarks, models and pruning ratio” are used. However, there are other settings which may affect the results, such as calibration dataset.

d)	In Appendix C, while the authors compared ShortGPT results with baseline methods SliceGPT and LLMPruner using the baseline settings, they only specified 'the same benchmarks, models, and pruning ratio' and directly referenced results from the original papers. However, additional settings, such as the calibration dataset, may also affect the outcomes. The authors should ensure all settings are consistent to enable a truly fair comparison.

e)	In Appendix C, while the authors compared ShortGPT results with baseline methods SliceGPT and LLMPruner using the baseline settings, they only specified 'the same benchmarks, models, and pruning ratio' and directly referenced results from the original papers. However, additional settings, such as the calibration dataset, may also affect the outcomes. The authors should ensure all settings are consistent to enable a truly fair comparison.

[1] Yifei Yang, Zouying Cao, and Hai Zhao. Laco: Large language model pruning via layer collapse, 2024.

### Questions
Apart from questions in weaknesses, there are following additional questions:

1.There are a few typos, though they do not impact the overall evaluation. For example, the incorrect y-axis label in Figure 1(b); and the denominator in the second term of Equation 4. 

2.There are also some inconsistent representations, such as “LLMPru,” “LLMprun.,” “LLM Pruner,” and “LLM-Pruner.” We assume these refer to the same work, but the variations are confusing.

3.The y-axis scale in Figure 1(a) may be misleading due to its exponential nature, which could mask a significant gap in PPL between the two approaching lines.

4.Could you clarify the process used to sample instances for calculating PPL (e.g., results in Table 1)? Clarifying the sampling criteria or methodology would improve reproducibility and understanding.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents a layer pruning method specifically designed for LLMs. It begins by assessing layer importance using the proposed Block Influence metric, which quantifies importance based on the variation in features between consecutive layers. Subsequently, the method iteratively removes the least significant layers until the target sparsity level is achieved.

### Strengths
The proposed method is straightforward to implement and requires only forward passes, making it computationally efficient. This efficiency allows it to be applied to larger models within a limited computation budget, unlike competing methods that require the additional overhead of computing gradients or Hessians.

### Weaknesses
1. The method hinges on the skip-connection nature of LLM layers, specifically, $x_{L+1} = x_L + f(x_L)$. Consequently, this reliance could limit its applicability compared to competitors that use gradients and are generally not constrained by model architecture.

2. The analysis on layer redundancy in Appendix A is unconvincing. Specifically, it misinterprets Lemma 1, which states that $||x_L|| = O(\sqrt{L})$ rather than $||x_L|| = \Theta(\sqrt{L})$. Furthermore, the authors assert that $f_L(x_L) = O(1)$ without solid mathematical proof. In the Attention example, the authors omit $W_o$ initially but include it later without explanation. Additionally, in (4), it is unclear why the $\ell_2$ norm of $||x_{L+1}||_2$ diminishes. Overall, the analysis in this section lacks clarity and cohesion.

3. The paper does not compare its method with recent gradient-based pruning techniques such as SLEB [1] and Shortened Llama [2], and it lacks a discussion on the related work by Gromov et al. [3].

4. In Table 2, for WSC, why are the accuracies of all Dense models the same at $50\%$? Additionally, the accuracies for ShortGPT-pruned Llama2-13B and Baichuan2-7B are also all $50\%$ on WSC.

5. Section 4.5 lacks a comparison with other pruning methods. It remains unclear if these other methods, when applied to quantized models, could potentially outperform ShortGPT.

6. Section 4.6 also lacks a comparison with other pruning methods.

**Overall Assessment**

In summary, I don't think the technical contribution of this paper is sufficient for ICLR.

[1] Song, Jiwon, et al. "SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks." arXiv preprint arXiv:2402.09025 (2024).

[2] Kim, Bo-Kyeong, et al. "Shortened llama: A simple depth pruning for large language models." arXiv preprint arXiv:2402.02834 11 (2024).

[3] Gromov, Andrey, et al. "The unreasonable ineffectiveness of the deeper layers." arXiv preprint arXiv:2403.17887 (2024).

### Questions
Please see Weaknesses and address them.

### Soundness
2

### Presentation
2

### Contribution
2
