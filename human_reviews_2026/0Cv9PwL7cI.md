# AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Diffusion-based large language models (dLLMs) are gaining attention for their inherent capacity for parallel decoding, offering a compelling alternative to autoregressive LLMs. Among various decoding strategies, block-wise semi-autoregressive (semi-AR) approaches are widely adopted due to their support for KV caching and their favorable accuracy–speed trade-off.
However, this paper identifies two fundamental limitations in the conventional semi-AR decoding approach that applies a fixed block size: i) late decoding overhead, where the unmasking of high-confidence tokens outside the current block is unnecessarily delayed, and ii) premature decoding error, where low-confidence tokens inside the current block are committed too early, leading to incorrect tokens. This paper presents the first systematic investigation challenging the fixed block size setting in semi-AR decoding. Through a statistical analysis of confidence dynamics during the denoising process, we identify a volatility band (VB) region during dLLM decoding, which encodes local semantic structure and can be used to guide adaptive block sizing.
Leveraging these insights, we introduce AdaBlock-dLLM, a training-free, plug-and-play scheduler that adaptively aligns block boundaries with semantic steps by adjusting block size during runtime. Extensive experiments across diverse benchmarks show that AdaBlock-dLLM achieves up to 5.3% accuracy improvement under the same throughput budget. Beyond inference-time optimization, we hope our semantics-aware adaptive scheduling approach and confidence-based analysis will inspire future training strategies for dLLMs. Our code is available at https://github.com/lgxi24/AdaBlock-dLLM.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the limitations of fixed block-size semi-autoregressive decoding in diffusion-based large language models (dLLMs). The authors identify two inefficiencies — Late Decoding Overhead and Premature Decoding Error — that arise when fixed-size blocks fail to align with semantic structure during decoding. To address this, they propose AdaBlock-dLLM, a training-free, plug-and-play adaptive scheduler that dynamically adjusts block size according to semantic cues and confidence scores during inference. The method leverages a novel concept called the Volatility Band (VB) — regions of fluctuating confidence that correspond to evolving semantic steps — to determine when to expand or contract blocks.

### Strengths
- Well motivated. The paper convincingly articulates the inefficiencies of fixed block-size decoding in diffusion LLMs.

- The adaptive block-size scheduler is conceptually elegant, lightweight, and compatible with existing architectures.

- The authors provide extensive results across multiple models and datasets. The method yields consistent accuracy improvements, particularly when combined with caching.

### Weaknesses
- The experiments focus mainly on math  and code generation. Broader text generation tasks (e.g., summarization or translation) and harder math reasoning tasks such as AIME would strengthen claims of generality.

- At larger block sizes, AdaBlock-dLLM shows reduced throughput compared to the baseline, raising concerns about its scalability for high-speed inference.

- The delimiter threshold (τ_D) and delimiter set (D) are manually tuned per model family. A more principled or automated way to select these would increase robustness.

- The paper lacks a discussion of situations when adaptive block size might fail.

### Questions
Technical Concerns/Questions and Points to Address in Rebuttal:

- Line 53 lacks evidence and most be supported through experiments or citations to prior work.

- How is this work different from [1] ?

- The concept of the Volatility Band (VB) appears intuitive—tokens near already decoded regions naturally exhibit higher confidence—so unless counterexamples or non-trivial cases are shown, the finding risks seeming self-evident rather than novel.


References:

[1] Wang, Xu, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. "Diffusion llms can do faster-than-ar inference via discrete diffusion forcing." arXiv preprint arXiv:2508.09192 (2025).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a method to adaptively select the left and right boundaries for the block in the semi-AR setting for dLLM. The main idea come from the analysis of the confidence dynamics, where inside a volatility band area, the confidence fluctuates dynamically, and the VB regions exhibit semantics structure. Then the author propose to collect indices whose predicted tokens fall in the delimiter set to determine the block size. Experiments show that the performance exceeds the one in DualCache.

### Strengths
1. The paper proposes a new method to solve the problem of how to adaptively decide the block size and also its position in dLLMs.
2. The paper is well-written with sufficient analysis for the motivation.
3. The proposed method is effective compared to the baseline.

### Weaknesses
1. Adaptive block size is important, but the authors didn't show any other baseline for different ways to adaptively deciding the block size. They are other ways that can also achieve the adaptive block size, like some naive and straightforward ways you can just expanding the block size by using a sliding window and track the latest position of tokens that are not decoded as the left index. I think adding more analysis and comparisons with different ways to achieve adaptive block is needed.
2. Why the performance for MBPP on Dream-Base is only 12. The results from the official report is ~55%. It's a large discrepancy.
3. I am confusing about the result in Table 2. In Table 2, the TPS first increases and then decreases as you choose a larger bock size. In the analysis, the authors mentioned that it may because the increase of NFE, but the thing matters is the ratio between NFE and Block size. Can you provide more analysis on this.
4. I think the contribution of this submission is not that meet the standard of ICLR. I fully agree that adaptive block is needed, but the contribution of the method is only to choose the block size by the delimiter set, and they are a lot of papers discussing the confidence dynamics in dllm decoding. The idea is interesting, but it's too simple and straightforward.

### Questions
1. See Weaknesses.
2. The way of selecting the delimiter set can be explored more. For traditional NLP tasks, there are a lot of ways (semantic parsing) to know different types to separate the semantics of a sentence, not only by the rough way of selecting \n, [,], and [.].

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
This paper identifies that fixed block sizes in semi-autoregressive (semi-AR) dLLM decoding cause "late decoding overhead" and "premature decoding error". It proposes AdaBlock-dLLM, a training-free scheduler that adaptively aligns block boundaries with semantic steps based on delimiter token confidence. This method improves accuracy by up to 5.3% without sacrificing throughput.

### Strengths
- The paper systematically analyze and address the limitations of the fixed block size assumption in semi-AR decoding.

- The core problems (late overhead, premature error) are clearly identified and illustrated. The proposed solution is simple, intuitive, and well-supported by experiments.

- The method is practical (training-free, plug-and-play) and shows clear accuracy improvements on the accuracy-throughput frontier, especially when combined with KV caching.

### Weaknesses
- The method introduces new hyperparameters that require model-specific tuning, slightly weakening the "plug-and-play" claim.

### Questions
Given the marginal benefit of adding more delimiters (Table 6), did you consider a more automated or learned method to identify semantic boundaries instead of a predefined list?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents AdaBlock-dLLM, a semantic-aware adaptive block-size decoding method for diffusion-based large language models (dLLMs). By analyzing confidence dynamics during denoising, the authors propose AdaBlock-dLLM, which is a training-free, plug-and-play scheduler that dynamically adjusts block boundaries according to semantic step length. Extensive experiments on benchmarks show that the proposed approach maintains compatibility with caching mechanisms and improves both efficiency and semantic consistency in diffusion LLM inference.

### Strengths
1. This paper proposes an adaptive, training-free approach that integrates seamlessly with existing diffusion LLM frameworks and improves accuracy without retraining.

2. This paper provides a clear and well-motivated analysis of confidence dynamics and their connection to semantic structures in dLLM decoding, then uses the confidence score of delimiter to decide the block size adaptively.

### Weaknesses
1. The heuristic nature of the delimiter-based semantic segmentation might not generalize well to less structured text other than math or coding problems. The authors should evaluate on more diverse test sets. For example, what if the output text is expected to be a long section and there's no '\n' in the output?

2.  Figure 6 can hardly be viewed as a pareto-frontier, as the throughput nearly does not change as the accuracy is increasing.

### Questions
Please refer to weakness for more details.

### Soundness
2

### Presentation
2

### Contribution
2
