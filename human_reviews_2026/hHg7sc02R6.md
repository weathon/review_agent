# Rethinking Transformer Inputs for Time-Series via Neural Temporal Embedding

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Transformer-based models, originally introduced in the field of natural language processing (NLP), have recently demonstrated strong performance in time-series forecasting. Due to the order-agnostic nature of the attention mechanism, these models have relied on positional encoding (PE) to capture temporal information. However, recent studies have reported that simple linear models can outperform complex Transformer architectures, and other works have also shown that modifying the Transformer input design can improve performance.
Motivated by this issue, we propose Neural Temporal Embedding (NTE), an embedding mechanism that effectively internalizes temporal dependencies without relying on either value embedding or positional encoding. NTE leverages simple neural modules such as Conv1D and LSTM to independently process each variable’s time-series and directly learn temporal patterns. As a result, it removes the need for linear projection for value embedding and positional encoding in the input stage, thereby enabling the model to simultaneously achieve architectural flexibility and competitive performance.
Experimental results on standard benchmarks including ETT, ECL, and Weather show that the proposed NTE-based models match or outperform state-of-the-art Transformer variants, particularly maintaining stable accuracy in long-horizon forecasting. These empirical findings show that Transformer-based models for time-series forecasting can achieve performance improvements through simple input enhancements without complex architectural modifications, thereby suggesting new possibilities for simpler and more generalizable input architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Neural Temporal Embedding (NTE), an embedding mechanism that effectively internalizes temporal dependencies without relying on either value embedding or positional encoding. The authors claim that a learnable NTE layer (using FC, Conv1D, LSTM, etc.) can process each variable’s time series and directly learn temporal patterns. Experimental results show that NTE-based models match or outperform state-of-the-art Transformer variants, particularly maintaining stable accuracy in long-horizon forecasting.

### Strengths
1. The motivation of the paper is valuable, which rethinks the input stack of time-series Transformers and presents NTE as a unified, learnable temporal layer that can replace value embedding and explicit positional information.
2. The paper includes ablations over PE variants, various NTE module types (FC, LSTM, Conv1D, Dilated, Bi-DilatedConv1D), bidirectional dilated Conv1D structures, and analyses of representation similarity (CKA) and entropy.

### Weaknesses
1. While the motivation for the proposed method is well-founded, the experimental results reveal notable shortcomings. Specifically, Table 1 shows that introducing NTE leads to significant performance degradation for the Vanilla Transformer on certain datasets, such as ETTh1 and ECL. This raises concerns about the robustness of NTE when combined with standard Transformer architectures and suggests that its benefits may be limited to specific backbone designs. 

2. The paper does not provide sufficient theoretical grounding to explain why using modules like Conv1D or LSTM within NTE leads to better results compared to the original value embedding. While the empirical results support the effectiveness of these modules, a theoretical analysis of how these architectures capture temporal dependencies more effectively would strengthen the contribution and improve the general interpretability of NTE design.

3. The ablation studies, while extensive, could be further expanded to explore the impact of kernel size in Conv1D-based NTE modules. The paper primarily reports results with fixed kernel sizes (e.g., 3 or 5). However, it is unclear whether these choices are optimal for capturing temporal dependencies in time series data, which often vary significantly in terms of patterns, seasonality, and granularity.

### Questions
1. Could the authors clarify what "Future-Dilated" means in Figure 3 and how the future embedding is constructed?
2. RoPE is a commonly used positional embedding method in Transformer-based models, particularly for tasks involving sequential data. However, it is not included in the experiments for comparison. Could the authors provide insights into how NTE compares to RoPE in terms of performance and effectiveness for time series forecasting?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a novel technique for the input embedding / transformation for time series foundation models (TSFM). In particular, they propose to combine the position embedding with the input embedding using different neural networks, which is a timely and interesting research area.

### Strengths
The problem that the authors work on is timely and a critical problem for any TSFM. So far, the default mode has been to simply embed the inputs either directly with a linear layer, or after applying a patching technique. The authors unify these two aspects with their proposed Neural Temporal Embedding, which is a simple neural network. In my view, this would be a novel aspect.

### Weaknesses
Despite the novelty, the idea and the paper has several critical flaws:

First, it is unclear until almost the very end of the paper on page 7, what the NTE is really doing. Up until this point the authors only mention that the NTE can be a 1D convolution, a LSTM, a fully connected network and several others, but they don't provide any concrete examples. Then, even though the authors provide this simple description of the two Conv1D layers, it is still unclear what the precise architecture of the NTE in Table 1 is. Is it the two conv layers, or is it something else? Only from the text, the reader can infer that the results in Table 1 stems from the two conv 1D layers. However, the authors state that "the sequential bias introduced by NTE is insufficient to compensate for the order-agnostic nature of the standard Transformer". However, they do not elaborate whether any other structure of the NTE would improve that. There is some ablation study in Table 2 that, which the reader can appreciate, but then this table is also confusing. Firstly, because the standard in the literature is to use the learnable PE (which should also be the reference point in Table 1), and then with this more realistic comparison point, none of the NTE architecture really seem to make a significant difference. Finally, since the paper puts so much emphasis on the NTE architecture as a novelty, detailed investigations of it are absent. For example what is the associated computational cost with the different variants, what are the features that those architectures provide? Are there specific cases in which to use one architecture of the NTE over the other?

Overall, the study spends a lot of time explaining and reiterating on the basics for the NTE, but doesn't dive in to the essence of it.

### Questions
See the text above, there are several open questions which should be addressed:\
Why to choose the NTE over the Learnable PE if performance is not better?\
What is the associated computational cost with the different variants?\
What about patching techniques paired with the NTE?\
Wouldn't a recurrent network break the parallelism and thus limit the efficiency of an attention backbone?\
What are the features that those architectures provide?\
Are there specific cases in which to use one architecture of the NTE over the other?
Etc.\

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper revisits a fundamental but often overlooked design aspect of time-series Transformers — the input embedding stage. The authors propose **Neural Temporal Embedding (NTE)**, a simple yet effective alternative to conventional **value embedding + positional encoding** pipelines. NTE employs lightweight neural modules such as **Conv1D** and **LSTM** to process each variable’s time series individually and encode temporal dependencies directly, without relying on positional encodings.  The key claim is that much of the Transformer’s inefficiency in time-series forecasting stems not from the attention mechanism itself, but from **suboptimal input representation**.  Experiments on standard benchmarks (ETT, ECL, Weather) demonstrate that NTE-based Transformers achieve comparable or better performance than specialized architectures, such as Autoformer, PatchTST, and iTransformer, particularly on long-horizon forecasting tasks.  The contribution is conceptually simple yet cleanly executed, offering an interesting perspective that suggests **input design** improvements can yield non-trivial gains without requiring architectural overhauls.

### Strengths
1. **Clear motivation and conceptual simplicity.**  
   The paper makes a strong case that input embedding deserves more attention. The removal of positional encoding is a bold but well-motivated design choice.

2. **Empirical clarity.**  
   The experimental setup is well organized, with fair comparisons to established baselines. The results convincingly show that input modifications alone can lead to performance gains.

3. **Strong writing and accessibility.**  
   The narrative is concise and approachable — the authors explain their ideas clearly without unnecessary jargon.

4. **Relevance to the ICLR community.**  
   The study fits the current trend of revisiting Transformer assumptions for efficiency and simplicity. It may inspire further work on lightweight input layers.

5. **Practical insights.**  
   The findings suggest that some of the architectural “complexity arms race” in time-series forecasting might be avoidable, which is refreshing.

### Weaknesses
1. **Limited novelty at the algorithmic level.**  
   NTE combines well-known neural components (Conv1D and LSTM) in a new configuration. While the empirical insight is valuable, the conceptual innovation is modest.

2. **Insufficient theoretical explanation.**  
   The paper would benefit from a deeper discussion of *why* NTE works — e.g., whether the learned temporal encoding approximates sinusoidal patterns or adapts to variable frequencies.

3. **Lack of broader baselines.**  
   The study compares mainly against mainstream Transformer variants. Including recent input-focused or embedding-free models (e.g., TSMixer, FreTS) would help strengthen the claim of generality.

4. **Ablation analysis could go further.**  
   It would be useful to isolate the impact of each NTE component (Conv1D vs LSTM) and analyze whether NTE benefits small-data or irregularly sampled settings differently.

5. **Unclear scalability implications.**  
   Since NTE introduces extra pre-processing per variable, a brief discussion of runtime or memory overhead would make the work more complete.

### Questions
1. How sensitive is the model to the choice of neural encoder (e.g., Conv1D vs GRU)?  
2. Does NTE preserve translation invariance in temporal shifts, or does the neural encoder introduce biases?  
3. Could the authors test whether NTE generalizes to irregular or non-uniform sampling rates?  
4. Are there cases where positional encodings outperform NTE (e.g., highly periodic signals)?  
5. How does the per-variable processing scale when the number of dimensions exceeds 100?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposed a mechanism, namely Neural Temporal Embedding (NTE), to replace the Positional Encoding (PE) in transformer-based models. Neural networks like FC, LSTM, and Conv1D are used to build the NTE module.

### Strengths
- The proposed mechanism achieves improvement when applied to backbones like iTransformer and PatchTST, especially on ETTh1 and ECL.
- The proposed NTE, as a plug-and-play module, can easily work with different backbones without changing the downstream structure.
- Multiple experiments are conducted to verify the effectiveness and efficiency of the proposed mechanism.

### Weaknesses
- Structure issues:
    - Although many experiments are conducted, only a few of the results are displayed in the main body of the paper.
    - In Sec.4, the paragraph `Bi-directional Dilated Convolutional Embedding' seems to have little relevance to the experiment. (Should it be in Sec.3 or Appendix?)
- Motivation issues:
    - Although the NTE is claims to simplify the input, the results in Table 6, 7, 13, and 14, indicate that the NTE may add to the computation burden.
- Experiment issues:
    - The NTE shows poor performance on Exchange, raising concerns that NTE may not be competent for long-horizon forecasting.
    - It is recommended to conduct multiple experiments and calculate the standard deviation to verify the stability.
- Others:
    - As mentioned in the paper, the NTE can only be conducted on time-series forecasting tasks.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
