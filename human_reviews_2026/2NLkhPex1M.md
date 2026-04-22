# AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in Large Vision-Language Models

- Avg Score: 4.40
- Decision: Accept (Poster)
- Scores: 4, 8, 2, 6, 2

## Abstract
Large Vision-Language Models (LVLMs) have adopted visual token pruning strategies to mitigate substantial computational overhead incurred by extensive visual token sequences. While prior works primarily focus on either attention-based or diversity-based pruning methods,  in-depth analysis of these approaches' characteristics and limitations remains largely unexplored. In this work, we conduct thorough empirical analysis using effective rank (erank) as a measure of feature diversity and attention score entropy to investigate visual token processing mechanisms and analyze the strengths and weaknesses of each approach. Our analysis reveals two insights:  (1) Our erank-based quantitative analysis shows that many diversity-oriented pruning methods  preserve substantially less feature diversity than intended; moreover, analysis using the CHAIR  dataset reveals that the diversity they do retain is closely tied to increased hallucination 
frequency compared to attention-based pruning.  (2) We further observe that attention-based approaches are more effective on simple images where  visual evidence is concentrated, while diversity-based methods better handle complex images with  distributed features.  Building on these empirical insights, we show that incorporating image-aware adjustments into existing hybrid pruning strategies consistently improves their performance.  We also provide a minimal instantiation of our empirical findings through a simple adaptive 
pruning mechanism, which achieves strong and reliable performance across standard benchmarks as  well as hallucination-specific evaluations. Our project page available at https://cvsp-lab.github.io/AgilePruner.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Large vision language models (LVLMs) generate hundreds of visual tokens that must be concatenated with textual tokens for processing, incurring quadratic time and memory costs. Many works have proposed attention‑based or diversity‑based pruning strategies to drop redundant visual tokens. Attention‑based methods keep tokens with high [CLS] attention scores but may select similar tokens, while diversity‑based methods maximize feature diversity but can discard important high‑attention tokens. The authors provide a systematic empirical study comparing these strategies across image types, using effective rank (erank) to quantify feature diversity and attention entropy to measure how concentrated the attention distribution is. They show that simple images (low erank / low entropy) benefit from attention‑based pruning, whereas complex images (high erank / high entropy) favor diversity‑based pruning. Furthermore, using the CHAIR hallucination benchmark, they find that attention‑based methods produce more conservative captions with lower hallucination scores, while diversity‑based methods capture more objects but hallucinate more. Motivated by these insights, the authors propose an adaptive token‑pruning framework. They explore tokens in descending order of attention scores and remove redundant tokens based on a similarity threshold that is adaptively set via a logarithmic mapping of erank. Simple images (low erank) use stricter thresholds; complex images use looser thresholds. Experiments show that the adaptive strategy maintains competitive performance.

### Strengths
S1. The paper systematically evaluates attention‑ and diversity‑based pruning across datasets, using effective rank and attention entropy to quantify image complexity. This analysis clarifies when each strategy excels.

S2. By measuring CHAIR metrics and recall, the authors show that diversity‑based pruning has higher recall but induces more hallucinations, while attention‑based pruning yields safer outputs.

S3. Experiments span nine multimodal benchmarks and the CHAIR hallucination dataset, showing the adaptive method maintains accuracy under aggressive pruning and reduces hallucination.

### Weaknesses
W1. The adaptive threshold function is heuristic and requires tuning coefficients alpha and beta. There is no principled justification or learning of this mapping.

W2. Computing effective rank involves singular‑value decomposition of token features, which may be expensive for large sequences. The paper does not quantify this overhead or compare it with more lightweight measures.

W3. Experiments are conducted exclusively on LLaVA‑1.5‑7B. Results may not generalize to other LVLMs or larger models, where token distributions or attention patterns differ.

W4. Improvements over baselines are relatively small (1–3% absolute), which may not justify the added complexity of computing erank and adaptive thresholds. Statistical significance is not reported.

W5. The paper notes that diversity‑based methods hallucinate more, but does not examine whether the adaptive method sometimes discards critical tokens (e.g., rare or small objects) or affects reasoning tasks beyond object captioning.

### Questions
Q1. How is erank computed in practice? Does it require an SVD for each image? What is the runtime overhead relative to token pruning, and how does it scale with image resolution and batch size?

Q2. Have you experimented with different mapping functions (e.g., linear or learned mappings) for tau_adaptive? How sensitive are the results to the chosen alpha and beta values across different datasets?

Q3. You use both erank and attention entropy to analyze image complexity, but only erank is used for threshold adaptation. Did you test using attention entropy or other metrics for adaptation?

Q4. For tasks requiring fine‑grained spatial reasoning or counting, does the adaptive pruning ever remove tokens critical to the answer? Can you provide qualitative examples where the method fails or underperforms?

### Soundness
3

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
This paper analyzes the limitations of existing token pruning methods designed to improve the efficiency of LVLMs and proposes a novel token pruning framework to address them. The authors employ attention score entropy and effective rank to examine the characteristics of each image token, and their proposed method is designed to adaptively apply pruning based on the specific properties of the image.

### Strengths
1. The properties and limitations of attention-based and diversity-based token pruning methods have not been clearly studied before. This paper provides both quantitative and qualitative analyses of these issues, making the argument convincing.
2. Based on the observed trends, the authors propose an intuitive token pruning method.
3. The proposed approach demonstrates high reliability by balancing the performance–stability trade-off between attention-based and diversity-based methods, which is further supported by extensive experimental results on various benchmarks.

### Weaknesses
1. In Table 1, the difference in entropy scores seems small. Additional explanation regarding the scale of these scores would be helpful.
2. The performance seems sensitive to the similarity threshold. Although the authors attempt to make the design adaptive using the effective rank (erank), parameter tuning (alpha, beta) may still limit the generalizability of the proposed method.

### Questions
1. The values in Table 1 differ from the descriptions in Section 4.1 (OCR: 4.39, 49, POPE: 4.90, 109).
2. After token pruning in the Table 1 experiments, how does the attention entropy change?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an empirical study on visual token pruning in large vision-language models (LVLMs). The authors analyze two main pruning paradigms: attention-based and diversity-based, and characterize their behavior using attention entropy and effective rank (erank) metrics, respectively. They show that attention-based pruning works better on simple images with concentrated features, while diversity-based pruning performs better on complex images with distributed features. They further study hallucination behavior and propose an adaptive pruning framework that adjusts token similarity thresholds according to image complexity, claiming improvements over several baselines across multiple benchmarks.

### Strengths
1.	The paper provides a thorough and systematic empirical comparison between attention- and diversity-based pruning strategies, which is less explored in depth before.
2.	The adoption of effective rank and attention entropy as quantitative measures for image complexity is conceptually reasonable.

### Weaknesses
1.	The proposed adaptive thresholding strategy is relatively simple and heuristic (a logarithmic mapping between erank and threshold). It does not provide strong methodological or theoretical innovation beyond straightforward empirical observations. 
2.	The proposed adaptive thresholding strategy introduces several hyperparameters, notably the scaling coefficients and other implementation choices. These parameters may influence pruning behavior, yet the paper does not provide a clear justification or sensitivity analysis for them.
3.	Prior token pruning works (e.g., DivPrune, VisionZip, PruMerge+) already explore balancing attention and diversity. This paper’s “adaptive” version appears incremental and lacks a compelling distinction.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a comprehensive empirical study on visual token pruning in Large Vision-Language Models (LVLMs), which are computationally expensive due to the large number of visual tokens produced by image encoders. Existing pruning strategies generally fall into two categories:

- Attention-based methods, which retain tokens with high attention scores.

- Diversity-based methods, which keep tokens that are dissimilar to others to preserve feature diversity.

Key Findings:

- Impact of Image Complexity. The authors quantify image complexity using two metrics: Attention entropy for measures how concentrated the model’s attention is. Effective rank (erank) for measures the diversity of token embeddings.

- Hallucination Analysis: Using the CHAIR dataset, the study shows that Attention-based methods produce more conservative outputs with lower hallucination rates. Diversity-based methods generate richer but more hallucinated descriptions due to speculative inclusion of objects.

- Adaptive Pruning Framework: Based on these insights, the authors propose AdaVTP (Adaptive Visual Token Pruning), which dynamically balances attention concentration and token diversity. The method adjusts a similarity threshold (τ) for pruning redundant tokens based on image complexity (measured by erank). A low τ is used for simple images (preserving fine details), while a higher τ enhances diversity for complex images.

### Strengths
Comprehensive empirical analysis and validation:
- It goes beyond performance reporting and explores why each method behaves differently, grounded in measurable concepts like attention entropy and effective rank (erank).
- Extensive experiments on nine multimodal benchmarks (VQAv2, GQA, TextVQA, ScienceQA, MMBench, etc.) and the CHAIR hallucination dataset demonstrate the robustness of the approach.

Insightful findings with practical relevance：
- The study reveals clear patterns: attention-based pruning excels in simple images (low erank) while diversity-based pruning performs better in complex ones (high erank).

- These findings can directly guide pruning strategy selection in real-world LVLM deployments where image complexity varies widely.

### Weaknesses
Limited novelty in algorithmic design:
- The proposed adaptive pruning framework (AdaVTP) mainly combines two existing ideas — attention-based and diversity-based pruning — using an adaptive threshold determined by image complexity.
- While insightful, this combination strategy is heuristic rather than fundamentally new in algorithmic form.

Limited scope of model diversity:
- Most experiments are based on a single LVLM backbone (LLaVA-1.5-7B).
- The generalizability of the findings to other architectures (e.g., Qwen-VL) is not demonstrated.

Computational trade-off not deeply analyzed:
- The paper claims efficiency improvements but provides limited quantitative analysis on actual latency or memory usage.

Potential instability in adaptive thresholding:
- Since the adaptive rule relies on erank estimated from token embeddings, noisy or atypical embeddings (e.g., from cluttered or low-quality images) might lead to inconsistent threshold adjustments.
- The paper lacks ablation or sensitivity studies showing the stability of AdaVTP under such variations.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper conducts an empirical analysis of attention-based and diversity-based visual token pruning strategies in vision-language models (VLMs). The study reveals that attention-based pruning strategy shows good performance on simple images (low erank), while diversity-based pruning strategy shows better performance on complex images. It proposes a method that adaptively determines the visual tokens preserved based on the erank of the images.

### Strengths
1. Provides diverse experiments to validate its approach 
2. The paper is well-written and easy to follow.

### Weaknesses
1. I am not sure the findings of the paper is novel enough. [1] shows that "a satisfied pruning method should jointly take the token importance and diversity into account." to preserve both local (important) and global (diverse) information, which is what the paper proposes to do.
2. I think an important related work is missed [2]. It also determines pruning threshold based on input instance adaptively.
3. From my understanding, the number of visual tokens is fixed in the experiments. Why didn’t the authors extend their method to adaptively determine the number of visual tokens retained? It seems that dynamically adjusting the number of preserved tokens could further improve the efficiency and effectiveness of visual token pruning.

[1] Long, Sifan, et al. "Beyond attentive tokens: Incorporating token importance and diversity for efficient vision transformers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[2] Ye, Xubing, et al. "Atp-llava: Adaptive token pruning for large vision language models." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
Please address questions in the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
