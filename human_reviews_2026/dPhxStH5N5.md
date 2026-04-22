# An Efficient Framework for Length Extension via Dynamically Growing Positional Embedding and Correlation-Aware Routing Attention

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Modeling long sequences is critical for numerous large-scale models. However, extending existing architectures to handle significantly longer sequences poses substantial technical and computational challenges. One inevitable issue is the overfitting of large models to positional encodings during pretraining, which limits their ability to generalize to unseen positional encoding scales. Additionally, extending sequence lengths requires extensive computational resources and time. Existing positional encoding methods often rely on carefully designed scaling factors but typically yield suboptimal results. To tackle these challenges, we propose \textbf{Cyclic, Randomly Truncated, and Dynamically Growing NTK Positional Embedding (CRG NTK)}, a data-augmentation-based technique that fully explores the RoPE encoding space, enabling models to adapt to various positional scales and achieve state-of-the-art extrapolation for the extension of lengths dominated by position encoding. Furthermore, we introduce \textbf{an efficient attention mechanism with a correlation-based routing strategy to enhance the fitting of the augmented positional encoding}, yielding superior performance and more efficient fine-tuning. With our approach, LLaMA-7B and Mistral-7B fine-tuned at 16K context length achieve extrapolation factors of at least 128$\times$ on simple tasks and maintain stable perplexity over 32$\times$ sequence length extensions and saves at least 16 times the GPU training resources compared to the existing optimal method. Experiments also show that correlation routing can achieve good performance by further filtering out large amounts of noise in long sequences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes CRG NTK for the length extrapolation. This work additionally designs an efficient attention mechanism with a correlation-based routing strategy to enhance the fitting of the augmented positional encoding. The experiment is conducted with LLaMA-7B and Mistral-7B, achieving extrapolation 128 times the original length.

### Strengths
* The length extrapolation problem is important. This work focuses on length extrapolation, which is important for long-text and reduces training cost.
* The Merge Selection method is relatively interesting. With MS Attention, the model could better process long context.
* The experiment results support that the method could extrapolate. For example, Table 3 presents that the model trained at 16K could extrapolate to length of 32 K.

### Weaknesses
* The major concern is the novelty of CRG NTK. It is not clear whether the method compared with other baselines, such as PoSE. Both this method and PoSE select a large maximum length during training.
* Without MS Attention, it seems that the CRG-NTK cannot extrapolate well. For example, in Table 1, the CRG-NTK trained on 16K, the ppl increases on length 32K. Similary, the CRG-NTK trained on 64K, the ppl increases on length 128K.
* For Figure 1, it is better to use pdf or SVG for higher resolution
* In Table 4, the LLaMA2-7B-MS performance is worse than original LlaMA-7B.

### Questions
N/A

### Soundness
2

### Presentation
2

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
This paper introduces a framework to efficiently extend LLM context windows. It proposes two key components: 1) CRG NTK, a data-augmentation-based positional encoding strategy (using random shifts, cyclic mapping, and a staged scaling curriculum) to improve extrapolation, and 2) MS Attention, an efficient, correlation-based sparse attention mechanism that routes queries to relevant Key/Value segments to reduce noise and compute. The authors report state-of-the-art extrapolation results (32x on perplexity, 128x on passkey) from short 16K fine-tuning, claiming a 16x reduction in GPU resources. The paper also hypothesizes a link between model depth and extrapolation limits.

### Strengths
1. This paper has a clear goal to efficiently and effectively extending LLM's context window. The authors propose a new RoPE extrapolation method "CRG NTK" and new efficient fine-tuning method "MS Attention". 
2. The paper proposes and provides a novel hypothesis that the model's extrapolation capacity is linked to its layer depth.

### Weaknesses
1. The paper is very difficult to read. The descriptions of the core methods, especially CRG NTK in Section 3.1, are dense, and the writing is often convoluted. The contribution of position extrapolation and efficient fine-tuning can be considered as two studies instead of one work which will confuse readers. 
2. The experiments are not sufficient to support the claims. 1) In Table 5, looks like Yarn has the best performance but we cannot see the score in Table 1 and 2. 2) Passkey retrieval and perplexity cannot directly reflect extrapolation. 3) It is not clear whether MS Attention will have degradation compare to Full Attention. 
3. The description of MS Attention (Section 3.2) is complex, and its performance seems highly sensitive to several key hyperparameters (segment size, topk, merge factor). Table 10 shows a perplexity swing from 7.17 to 6.09 based on these settings. This sensitivity implies a costly and difficult hyperparameter search is required to achieve the reported results.

### Questions
1.  Please provide ablation results (Table 2,3,5) for a model fine-tuned using CRG NTK with Full Attention? This is essential to isolate the contribution of your positional encoding strategy from your efficient fine-tuning method.
2. Please clarify the exact "Dynamically Growing" schedule used in your experiments? Is it a discrete, staged curriculum? If so, how many stages were used, what was the scaling factor $a$ at each stage?

### Soundness
2

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
3

### Summary
The paper introduces a unified approach to efficiently extend large language model (LLM) context windows. It proposes a Cyclic, Randomly-truncated, and Dynamically Growing NTK-aware (CRG-NTK) positional encoding method that progressively scales position frequencies during fine-tuning, improving extrapolation up to 32× the training length—bounded by model depth. The framework also includes a Merge-and-Select (MS) routing attention mechanism that filters irrelevant tokens to suppress noise and reduce computation. Together, these techniques achieve up to 128× extrapolation on retrieval tasks while requiring 16× less GPU time than LongRoPE, offering a practical, theory-supported path to scalable long-context adaptation.

### Strengths
(1) The method is simple and in general PE recipe. CRG-NTK unifies several effective heuristics, including random shifts, cyclic truncation, and scheduled scale growth, into a single augmentation that is easy to integrate into standard fine-tuning.

(2) The experiments are comprehensive.

### Weaknesses
(1) The paper notes using a fixed base scaling at inference after training through multiple scales. Please analyze failure modes when test lengths fall between trained scales, and whether per-layer scale interpolation helps.

(2) The paper hypothesizes an extrapolation limit $\approx$ number of layers but gives only sketch intuition and empirical suggestion. This is interesting. Could you please either formalize a brief statement (even under simplifying assumptions) or clearly mark it as conjecture?

### Questions
(1) Why power-law growth? Did you compare power-law, exponential, and additive scale growth in sample efficiency and forgetting? Any signs of catastrophic interference at very large steps?

(2) Does CRG-NTK help more in early vs late layers? Any merit to layer-dependent scales or per-head scaling?

### Soundness
4

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
5

### Summary
The paper proposes a simple, compute-light recipe to extend LLM context windows far beyond training lengths by (1) augmenting positional encodings via Cyclic, Randomly-truncated, and Dynamically-Growing NTK (CRG-NTK) and (2) replacing full attention during fine-tuning with a relevance-routed sparse attention, Merge–Select (MS) Attention. With only 16K-context fine-tuning on a single A100, LLaMA-7B/Mistral-7B extrapolate to millions of tokens on synthetic retrieval and maintain stable perplexity over ≥32× longer sequences, while cutting fine-tuning compute by ≥16× vs. strong baselines.

### Strengths
- The paper introduces the CRG-NTK framework, a creative combination of cyclic shifts, random truncation, and dynamic NTK scaling that transforms positional encoding from a static component into a data-augmentation process.  

- The Merge–Select Attention mechanism combines correlation-based routing with segment merging—an synthesis of sparse attention and dynamic selection ideas (from Routing Transformer and BiFormer) that yields efficient architecture.  

- Proposing that extrapolation limits scale with network depth offers a theoretical perspective.

### Weaknesses
- The proposed approach is overly complex, yet lacks detailed ablation studies for each component, making it difficult to determine its effectiveness.

- It is unclear how the theoretical gradient derivation leads to the conclusion that having more layers results in better extrapolation capability.

- Considering that synthetic retrieval tasks and simple perplexity[2] measurements cannot effectively evaluate true extrapolation capability, could you provide results on some reasoning tasks, such as Many-Shot In-Context Learning[1]?

- As far as I know, there also exist some **training-free approaches[3]** that achieve length extrapolation when combined with sparse attention. Could you compare your method against these?



Ref:  
[1] Many-Shot In-Context Learning.  
[2] CAN PERPLEXITY REFLECT LARGE LANGUAGE MODEL’S ABILITY IN LONG TEXT UNDERSTANDING ?  
[3] Parallel Long-Context Compressor for Length Extrapolation

### Questions
- Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
