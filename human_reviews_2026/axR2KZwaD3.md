# Understanding Transformers for Time Series: Rank Structure, Flow-of-ranks, and Compressibility

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 8

## Abstract
Transformers are widely used across data modalities, and yet the principles distilled from text models often transfer imperfectly. In this paper, we analyze Transformers through the lens of rank structure. Our focus is on the time series setting, where the structural properties of the data remarkably differ from those of text or vision. Time-series embeddings, unlike text or vision, exhibit sharply decaying singular spectra: small patch sizes and smooth continuous mappings concentrate the data into low-rank subspaces. From this, we prove that the associated $Q/K/V$ projections admit accurate low-rank approximations, and that attention layers become compressible in proportion to the decay of the embedding spectrum. We introduce the concept of *flow-of-ranks*, a mechanism by which nonlinear mixing across depth inflates the rank, explaining why early layers are most amenable to compression and why rank schedules should grow with depth. Guided by these results, we compress Chronos, a large time series foundation model, achieving a reduction of $65\\%$ in inference time and $81\\%$ in memory without loss of accuracy. These findings provide principled guidance for allocating width, depth, and heads in time series foundation models, and for exploiting their inherent compressibility. Our code is available at https://github.com/amazon-science/tsfm-compression.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper analyzes Transformers for time-series data through the lens of rank structure. The authors show that time-series embeddings are inherently low-rank, unlike those in text or vision. They prove that this structure induces low-rank attention matrices, introducing the concept of flow-of-ranks to describe how rank gradually increases with layer depth due to nonlinear mixing. They demonstrate that time-series foundation models, such as Chronos, can be compressed by up to 65% in inference time and 81% in memory without accuracy loss.

### Strengths
1. The paper combines linear-algebraic theory with well-designed experiments confirming the predicted low-rank behavior in TSFMs.
2. The authors introduce a new analytical lens (flow-of-ranks) that connects data modality to model design. The findings have real design implications for TSFMs.

### Weaknesses
1. The main validation focuses on Chronos. Testing on other TSFMs (like TimesFM and Time-MoE) would strengthen the generality claim.
2. The comparison to prior compression methods (LoRA) is missing, which makes it unclear how much gain stems from modality vs. technique.

### Questions
1. How does the proposed compression compare quantitatively with existing low-rank or sparse-attention baselines (e.g., LoRA, Linformer)?
2. Does the low-rank structure persist after fine-tuning a compressed model on downstream tasks?
3. Can the flow-of-ranks pattern be empirically confirmed on other TSFMs beyond Chronos?
4. How would the theory extend to multivariate or irregularly sampled time series?
5. Could you provide a simple practical guideline (e.g., rank schedule formula) for designing TSFMs from scratch?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper titled “UNDERSTANDING TRANSFORMERS FOR TIME SERIES: RANK STRUCTURE, FLOW-OF-RANKS, AND COMPRESSIBILITY” analyzes Transformer models for specifically time series (TSFMs). They show that in this scenario, the transformer models ( as a consequence of the data being passed to it) possess a uniquely low-rank structure compared to similar architectures for text or vision. They postulate thus due to the continuous nature of time-series embeddings. This low-rank input structure leads to that the Attention layer matrices being highly compressible. The authors introduce the concept of "flow-of-ranks," which describes how the numerical rank of a representation gradually increases with model depth due to nonlinear mixing, explaining why earlier layers are more amenable to compression. By leveraging these insights, the researchers demonstrate that large TSFMs like Chronos are severely over-parameterized and can be significantly compressed, achieving up to a 65% reduction in inference time and 81% in memory without losing predictive accuracy.

### Strengths
The low rank property of time-series may prove very useful in their application using transformers, in terms of designing models with fewer parameters. Theoretically, this low-rank property is proven for continuous embeddings, with guaranteed polynomial or exponential decay of singular values for smooth or analytic functions, respectively (Theorems 1 and 2).
The work provides the first general theoretical results (Theorem 3) connecting low-rank input embeddings to the compressibility of the internal Attention matrices (W_Q, W_K, W_V)
The paper introduces and quantifies the concept of "flow-of-ranks," which explains how  non-linear components (like activations, residual connections, and normalization) across deep layers gradually increase the rank of a representation (Theorem 4)

### Weaknesses
The majority of the empirical validation and compression experiments focus almost exclusively on the Chronos family of Time Series Foundation Models. While there are references to other models the core compression techniques and deep rank analysis (flow-of-ranks, impact of heads) are primarily demonstrated on Chronos. This limits the generality of the practical findings and compression results to other TSFM architectures
While the paper's core claims about rank structure are presented as a modality-dependent framework, the empirical evidence is often constrained to a small number of specialized experiments. How do we know this structure will prevail in other transformer architectures or other datasets.
The core theoretical analysis (Theorems 1 and 2) is derived for the univariate time series case. The authors mention in passing that this is extendible to few-variate time series but a detailed discussion is lacking

### Questions
na

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates why Transformers trained on time series data behave differently from those trained on text or vision. The authors analyze the rank structure of embeddings and attention matrices to explain why time-series foundation models (TSFMs)—such as Chronos are highly compressible without losing much accuracy.
Several key points include: time series are naturally low-rank, and flow-of-rank perspective.

### Strengths
1. To my understanding, transformer theories in time series are generally lacking or provides limited practical guidance. This paper serves as a good entry point to understand how time series data differs from other modalities.
2. The results are intuitive to me, making this paper easy to follow.

### Weaknesses
1. This paper considers univariate time series. which is limited as several TSFMs can handle any-variate inputs.
2. The paper assumes input data X is 1-rank (or low-rank). I think this is a pretty strong assumption, which may not hold in many high-dimensional data.

### Questions
1. Can the authors explain footnote 1? Does it mean that if we have $n$-variate data, $x$ is then rank-$m$, where $n = m$? Is it possible that $m < n$?
2. I wonder how naive Thm 1 and Thm 2 are? Since this paper mainly shows existence proof, if the input data is low-rank, it seems like its straightforward that we can find low-rank weights to model it. Are there any counterexamples?
3. Are there any other features in the data assumption that makes it a time series? Or would the result hold for all low-rank input data?

Overall, I think this paper will be a good contribution to the field and am happy to adjust my score if the authors address my concerns?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents a rigorous linear-algebraic analysis of time series Transformer models. It first examines the ranking structure of time series embeddings, revealing unique low-rank characteristics that distinguish them from other modalities. The authors then theoretically infer the potential low-rank nature of the attention matrices in time series Transformers. Meantime, the authors introduce the concept of “flow-of-ranks” to describe how representation ranks evolve and increase across Transformer layers due to nonlinear transformations. Finally, leveraging these insights, the paper proposes two effective compression strategies for time series foundation models, achieving up to 65% reduction in inference time and 81% reduction in memory usage on the Chronos model, without compromising accuracy.

### Strengths
1. Strong Theoretical Grounding: Theorems 1 and 2 formally connect patch size and embedding smoothness to low-rank structure. Theorem 3 crucially links low-rank inputs to compressible attention layers, while Theorem 4 quantifies the "flow-of-ranks." These theories provide novel insights to guide time series foundation model design.

2.  The translation from theory to practice is seamless and compelling. Each theoretical claim is supported by corresponding empirical evidence, making the overall theory framework more convincing and practically relevant. And the results are striking, showing that TSFMs are significantly more over-parameterized than LLMs and can be compressed dramatically. The layer-dependent rank schedule is a simple yet powerful idea derived directly from the "flow-of-ranks."

3. Clarity and Organization: Despite the complex mathematical content, the paper is well-structured and readable. The flow from data structure to single-layer analysis to depth-dependent phenomena and finally to applications is logical and easy to follow.

### Weaknesses
1. The core experiments only focus on a single architecture family. All experiments are conducted on Chronos and Chronos-Bolt, which are based on the T5 architecture. While the principles are argued to be general, validation on other TSFM architectures (e.g., TimesFM, Moirai) would have further strengthened the claim of universality.

2. The paper provides elegant existence proofs—such as the low-rank properties of time series embeddings and the W_Q/K/V matrices—based on the core assumption that time series embeddings are intrinsically low-rank. However, this assumption may be an artifact of current TSFM design choices (e.g., small patch sizes and simple MLP-based embedding layers). As more tokenization methods emerge (e.g., VisionTS [1], Wavelet-based Tokenization [2]), it remains unclear whether these conclusions will still hold.

[1] Chen M, Shen L, Li Z, et al. Visionts: Visual masked autoencoders are free-lunch zero-shot time series forecasters[J]. arXiv preprint arXiv:2408.17253, 2024.

[2] Masserano L, Ansari A F, Han B, et al. Enhancing foundation models for time series forecasting via Wavelet-based tokenization[J]. arXiv preprint arXiv:2412.05244, 2024.

### Questions
1. Could the observed low-rank and compressibility properties be interpreted as universal characteristics of time series Transformer models? If so, does this imply that current architectures are inherently low-rank and may therefore lack sufficient expressiveness to capture more complex temporal dynamics?

2. The success of pre-trained compressed model suggests that standard TSFMs are severely over-parameterized. Does this imply that the common practice of scaling up model size for time series is misguided ?  Is the low rank of TSFMs a inherant feature, or is it a sign that we are not yet challenging them with tasks of sufficient complexity?

3. Theorem 3 suggests that the attention matrix may be less compressible when dealing with more complex or higher-rank input time series (e.g., noisy financial data). Are there datasets of this nature that could be used to empirically test the boundaries of the proposed low-rank assumption?

### Soundness
3

### Presentation
3

### Contribution
3
