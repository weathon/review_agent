# SSD: Spatial-Semantic Head Decoupling for Efficient Autoregressive Image Generation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Autoregressive image generation models like Janus-Pro produce high-quality images, but at the cost of high memory and computational demands due to the large number of visual tokens. 
While KV cache compression has been extensively studied in language modeling, it remains largely unexplored for image generation.

In this work, we begin by identifying a distinct attention phenomenon, which we term spatial locality and emergent semantic sink. 
To leverage this, we introduce a novel KV cache compression framework. 
Specifically, we compress the KV cache for visual tokens by decoupling attention heads into two types: for spatial-locality heads, our method maintains a short recent token window; for semantic-sink heads, it preserves a compact set of highly-attended tokens. 
Experiments demonstrate that our method achieves a 5$\times$ reduction in memory usage and a 6.6$\times$ speedup in throughput with negligible performance loss, enabling efficient native autoregressive image generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SSD (Spatial-Semantic Decoupling) — a framework for Key-Value (KV) cache compression in autoregressive (AR) image generation models like Janus-Pro.
SSD aims to reduce memory and computation costs during image generation without hurting image quality.

### Strengths
SSD’s greatest strength lies in connecting a real hardware bottleneck (KV cache bloat) with a novel structural insight (spatial–semantic head specialization), and turning that into a principled, empirically validated framework that achieves major efficiency gains without sacrificing image quality.

### Weaknesses
1 The paper identifies “spatial locality” and “semantic sink” as two distinct attention behaviors in autoregressive image generation.
However, the spatial locality aspect is not novel — similar locality patterns have been discussed in NAR, LPD, and ZipAR, which all highlight that visual attention predominantly focuses on nearby spatial tokens. The authors should clarify the difference between their “spatial-locality heads” and previously observed locality mechanisms, and properly cite these works to avoid overclaim. The attention sink is also old-fashioned since the ViT era [4,5], making it expected in AR. If the authors just use this proposed or similar observation for caching, I think the novelty is quite limited since many similar works have been extensively done in LLM [6].

[1] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation

[2] Neighboring Autoregressive Modeling for Efficient Visual Generation

[3] ZipAR: Parallel Auto-regressive Image Generation through Spatial Locality

[4] See What You Are Told: Visual Attention Sink in Large Multimodal Models

[5] Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing

[6] MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

2 The paper’s analysis and method are entirely built on raster-order autoregressive generation. However, this paradigm is now computationally suboptimal and is being replaced by faster alternatives such as MAR, LPD, and VAR. The authors should discuss whether their observed attention patterns — particularly spatial locality and the semantic-sink behavior — hold across different generation orders. More applications on these frameworks are necessary. Besides, if the work is raster-specific, more applications on Llamagen are necessary.

### Questions
see above

### Soundness
3

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
4

### Summary
This paper presents KV cache compression methods for next-token-prediction-based autoregressive (AR) image generation. The authors observe that attention heads in AR image models are highly sparse, and can be broadly categorized into two types: "semantic heads," which focus on periodic anchor tokens, and "spatial heads," which concentrate on spatially-local tokens. Based on this observation, the paper proposes two distinct KV cache compression techniques tailored for each head type: a sliding window approach for spatial heads and a heavy-hitter-retention method for semantic heads. Experimental results on the Janus-Pro model demonstrate a superior Pareto frontier on the DPG-bench compared to existing KV cache compression methods developed for text LLMs.

### Strengths
- The paper is well-written, intuitive, and easy to understand.
- This is the first work that tries to analyze characteristics of KV-cache in AR image models, and found interesting attention patterns (spatial and semantic). This observation aligns well with intuition.
- Also, this paper propose intuitive KV cache compression methods tailored for two distinct attention types.

### Weaknesses
- **Limited Generalizability** : All experiments were conducted solely on the Janus-Pro model. It is uncertain whether the paper's findings, including the observed attention patterns and the efficacy of the proposed compression methods, generalize to other AR image generation models. Experiments on other AR image models, such as llamaGen, Emu3, Anole, and Lumina-mGPT (1, 2), are necessary. I believe experiments on llamaGen are essential, and additional validation on Lumina-mGPT would be welcome.

- **Insufficient Evaluation** : Performance evaluation was restricted to the DPG-bench, which may not adequately capture the perceptual quality of the generated images. An experiment using standard image generation metrics, such as FID or IS, on datasets like MS-COCO, is required.

- **Novelty** : While tackling this problem for the first time and identifying the sparse attention patterns is novel, the proposed methods lack originality. They appear to be direct applications of existing KV cache compression techniques.

### Questions
- **Semantic Concentration Metric** : The definition of the "Semantic Concentration" metric is not clear. Why is the difference between the KV cache values of CFG and "native" (non-CFG) generation representative of semantic concentration?

- **Sliding Window Implementation** : In a flattened 1D token sequence, the local "neighborhood" tokens should include not only tokens immediately to the "left" (preceding in the sequence) but also spatially adjacent tokens from previous rows. Figure 1 seems to confirm this. However, why sliding method of SSD just retain the preceding(left) tokens in the 1D sequence?

### Soundness
3

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
5

### Summary
The paper proposes ​​SSD​​, a framework for compressing the KV cache in autoregressive image generation models. The method categorizes attention heads into two types: (1) spatial locality heads, which focus on spatially adjacent tokens, and (2) semantic sink heads, which attend to a few critical tokens. The KV cache for each type is then compressed using a dedicated strategy. Experimental results show that SSD effectively reduces the KV cache size and accelerates the decoding process, with only minimal performance degradation.

### Strengths
1. The proposed methods are simple and effective.
2. The paper is easy to follow.

### Weaknesses
1. The experimental evaluation is conducted exclusively using Janus-Pro models. To fully establish the robustness and general applicability of the proposed methods, validation across a broader range of model architectures is necessary.
2. The concept of exploiting spatial locality to accelerate autoregressive (AR) image generation has been widely adopted in methods such as PAR [1], ZipAR [2], and NAR [3]. These works, which also employ parallel decoding by restricting the attention window, are highly relevant yet are not discussed or compared against in the paper.
3. The core motivation of the paper may require reconsideration. While KV cache size presents a significant challenge for LLMs with long contexts, the context length for AR image generation models is typically much shorter, often comprising only hundreds or a few thousand tokens. Furthermore, increasing the batch size does not alter the model's computational intensity (i.e., the compute-to-memory-access ratio). From this perspective, the necessity of compressing the KV cache for AR image generation appears debatable. The paper's primary contribution likely stems instead from the throughput gains achieved via sparse attention during decoding.

[1] Parallelized Autoregressive Visual Generation, CVPR 2025.

[2] ZipAR: Parallel Auto-regressive Image Generation through Spatial Locality, ICML 2025.

[3] Neighboring Autoregressive Modeling for Efficient Visual Generation, ICCV 2025.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the significant memory and computational overhead of the KV cache in autoregressive image generation models , noting that existing language-focused compression methods are suboptimal for visual tokens. The authors empirically identify a novel attention phenomenon: a functional dichotomy where some heads focus on spatial locality and others act as emergent semantic sinks . Crucially, they find semantic information is preferentially anchored at the margin columns of the token grid . Based on these insights, the paper proposes SSD, a framework that classifies heads as spatial or semantic and applies distinct, tailored compression policies to each type . Experiments demonstrate that SSD achieves up to a 5x memory reduction and 6.6x speedup with negligible quality degradation.

### Strengths
- Originality: The paper's primary strength is its originality. Instead of merely adapting language-based KV compression, it presents a new, empirically-grounded understanding of attention mechanisms in visual AR models. The identification of the "spatial-semantic dichotomy" and the "margin column anchoring" phenomenon is a novel and significant finding.

- Quality: The work is of good quality, with strong empirical validation for its claims.Notably, Figure 2(b) provides exceptionally clear and intuitive evidence for the "semantic anchor" hypothesis. By plotting the MSE between the KV caches of the CFG and non-CFG branches, it accurately visualizes the periodic spikes in semantic content at the margin column positions . The experimental setup is robust, using competitive baselines (H2O, StreamingLLM) and standard benchmarks (GenEval, DPG-Bench).

- Clarity: The paper is well-written, logically structured, and easy to follow. The problem is clearly defined , and the core concepts are introduced intuitively. The figures (especially 2(b)) and Algorithm 1 effectively illustrate the method.

- Significance: This work addresses a critical and practical bottleneck for the deployment of large-scale AR image generators. The reported efficiency gains (5x memory, 6.6x throughput) are substantial and could significantly advance the practical usability of unified multimodal models on resource-constrained hardware.

### Weaknesses
- Generalizability: The paper's primary weakness is the limited scope of its validation. All analyses and experiments are conducted exclusively on the Janus-Pro model family. It remains unclear whether the core findings—the spatial-semantic dichotomy and margin column anchoring—are fundamental properties of visual AR generation or emergent properties specific to the Janus-Pro architecture. The claim needs to be validated on other visual AR models to be considered general.

- Static Head Classification: The classification of heads as spatial or semantic is performed offline and remains static throughout inference. This approach ignores the possibility that a head's function might be dynamic and context-dependent. A static policy may be suboptimal compared to a dynamic one.

- Hyperparameter Sensitivity: The method introduces several key hyperparameters, including the spatial window size $W$, the semantic budget $M$, and the classification threshold $\tau$. While the paper provides a good sensitivity analysis for $\tau$, it lacks a detailed discussion on the selection and sensitivity of $W$ and $M$. It is unclear how these are balanced to meet a specific cache budget and how performance is affected by their interplay.

### Questions
1. On Generalizability: Can you comment on the generalizability of your findings? Is there any evidence or strong reason to believe that the "spatial-semantic dichotomy" and "margin column anchoring" phenomena are also present in other visual AR models?

2. On Profiling Cost: Regarding the offline step, "Classify all heads via sparsity profiling": what is the computational cost of this process? Specifically, how many prompts were used to gather the statistics what hardware was used, and approximately how much time did this profiling take? Does this step pose a significant barrier to applying SSD to new models?

3. On Hyperparameters $W$ and $M$: Could you please elaborate on how the spatial window size $W$ and the semantic budget $M$ were selected? For instance, in the 20% cache budget scenario in Table 1, what were the typical values or ratio for $W$ and $M$? How sensitive is the model's performance to this allocation ratio?

4. On CFG Dependence: Your core analysis for identifying semantic injection (Figure 2(b)) is heavily dependent on CFG. Does the SSD framework remain effective, and does the margin anchoring phenomenon persist, during non-CFG sampling ($\gamma=1$) or at very low guidance scales?

5. On Static Classification: The head classification is static. Did you consider or experiment with a dynamic classification strategy, where a head's role could be re-evaluated or adapted based on the generation context?

### Soundness
3

### Presentation
3

### Contribution
2
