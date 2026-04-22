# SeViCES: Unifying Semantic-Visual Evidence Consensus for Long Video Understanding

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
Long video understanding remains challenging due to its complex, diverse, and temporally scattered content. Although video large language models (Video-LLMs) can process videos lasting tens of minutes, applying them to truly long sequences is computationally prohibitive and often leads to unfocused or inconsistent reasoning. A promising solution is to select only the most informative frames, yet existing approaches typically ignore temporal dependencies or rely on unimodal evidence, limiting their ability to provide complete and query-relevant context. We propose a  **Se**mantic–**Vi**sual **C**onsensus **E**vidence **S**election (SeViCES) framework for effective and reliable long video understanding. SeViCES is training-free and model-agnostic, and introduces two key components. The Semantic–Visual Consensus Frame Selection (SVCFS) module selects frames through (1) a temporal-aware semantic branch that leverages LLM reasoning over captions, and (2) a cluster-guided visual branch that aligns embeddings with semantic scores via mutual information. The Answer Consensus Refinement (ACR) module further resolves inconsistencies between semantic- and visual-based predictions by fusing evidence and constraining the answer space. Extensive experiments on long video understanding benchmarks show that SeViCES consistently outperforms state-of-the-art methods in both accuracy and robustness, demonstrating the importance of consensus-driven evidence selection for Video-LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SeViCES, a training-free and model-agnostic framework that enhances long video understanding by selecting key frames through a semantic-visual consensus mechanism and refining final answers by resolving inconsistencies between different evidence sources.

### Strengths
- A significant strength is the proposed dual-branch (semantic and visual) frame selection module, which explicitly addresses the limitation of unimodal approaches by leveraging the complementary strengths of LLM-based reasoning on captions and cluster-guided visual alignment to capture more complete, query-relevant context.

- The paper is well-written and easy to follow.

### Weaknesses
1.  Regarding the "Semantic-Visual Consensus Frame Selection" method, the authors argue that traditional CLIP-based methods for measuring text-frame relevance are difficult to apply directly to videos containing temporal information. They instead propose converting frames into captions and using an LLM for assessment. I have the following concerns:
    - Could the frame-to-caption conversion process itself lead to inaccurate descriptions due to the loss of temporal information?

    - Since the individual frame caption (s_ind) still lacks temporal context, is using an LLM here fundamentally more advantageous than using CLIP?

    - For the consensus caption (s_con), which incorporates contextual captions, is feeding this textual window into the LLM indeed superior to using a visual window of the corresponding frames (e.g., with a video encoder like VideoCLIP) for relevance calculation? Intuitively, a window composed of single-frame captions likely loses more temporal information compared to a window of the original visual frames.

2.  In Table 1, while the method is compared against numerous video-LLMs, it is only benchmarked against two key frame selection methods. This is insufficient to demonstrate the advantage of SeViCES. Notably, compared to Suo et al. (2025), the improvements are not always significant. Furthermore, the comparison seems unfair as Suo et al. (2025) uses only 32 frames. I strongly recommend including more comparisons with state-of-the-art frame selection methods under fair conditions, such as [1, 2].

[1] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs

[2] ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning


3.  The paper lacks sufficient ablation studies for the "Answer Consensus Refinement" component. Specifically, the individual contribution and effectiveness of the "Evidence fusion" and "Constrained decoding" techniques should be validated through experiments.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes SeViCES, a training-free, model-agnostic framework for long video understanding that unifies semantic and visual evidence through two main modules: Semantic–Visual Consensus Frame Selection (SVCFS) and Answer Consensus Refinement (ACR). SeViCES effectively selects query-relevant and evidence-complete frames, achieving significant performance improvements across multiple VideoLLMs and benchmarks.

### Strengths
1. **Training-free and model-agnostic design.** The proposed method can be plugged into multiple VideoLLMs without training.
2. **Consistent performance gains.** SeViCES shows consistent performance gains across multiple video benchmarks, including long video understanding tasks, showing its effectiveness.

### Weaknesses
1. **Limited novelty.** The proposed SeViCES framework shows limited novelty. The core ideas, (1) LLM-based frame caption scoring and (2) visual feature-based frame clustering for frame selection, have already been explored in VideoTree [1]. As such, both the objective and methodology of SeViCES closely resemble those of VideoTree. A more detailed performance comparison and discussion of their differences and advantages are needed to justify the novelty of this work.
    
    [1] Wang et al., VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos, CVPR 2025
    
2. **Computational overhead.** The proposed framework introduces substantial computational complexity due to multiple caption generations, LLM-based scoring, kNN-based clustering, and related steps. The paper should include a quantitative analysis and discussion of computational overhead (e.g., throughput, latency, or runtime per video) to demonstrate practical feasibility.
3. **Missing ablations.** The dynamic frame allocation mechanism (Eq. 7) appears overly complex, yet its contribution is not clearly analyzed. It is necessary to perform ablation experiments to verify whether this equation is indeed critical. How would the performance change if frames were allocated equally across clusters instead?
4. **Unclear writing.** Some notations are not clearly defined.
    1. Precise definition of Mutual-Information (Eq. 5) and sim() (Eq.6) are missing
    2. In Eq. 6, superscript $i$ seems to be missing for $\bar{s}$, $\sigma^2$, and $\epsilon$.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

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
This paper proposes SeVICES, a novel training-free and model-agnostic framework to address the challenge of long video understanding. To overcome the prohibitive computational cost of processing all frames, SeVICES selects a compact, query-relevant subset of frames by enforcing consensus at two levels. First, the Semantic-Visual Consensus Frame Selection (SVCFS) module uses two complementary branches: (1) a semantic branch (TAS-FS) that uses an LLM to score frame captions based on query-relevance and temporal context, and (2) a visual branch (CgMI-FS) that uses clustering and mutual information to select a visually diverse set of frames that are also semantically relevant. Second, the Answer Consensus Refinement (ACR) module runs the Video-LLM on both sets of selected frames. If the answers disagree, it treats this as a sign of incomplete evidence, fuses the two frame sets, and runs the Video-LLM a final time, forcing it to adjudicate between the two conflicting answers.

### Strengths
- The paper is easy to follow and well-written.
- The proposed method is training-free and model-agnostic, which is easy to extend.

### Weaknesses
- The paper is motivated by overcoming the "prohibitive computational costs" of processing long videos. However, the proposed SeVICES framework introduces a new, significant, and entirely unmeasured computational bottleneck: inference latency. So, it would be better to provide the complexity analysis of the proposed method.
- A primary strength of a "training-free" method should be its universal applicability to any MLLM. However, the paper's experiments are limited to three open-source models. The method was not applied to any state-of-the-art proprietary models (e.g., GPT-4V, Gemini), which would have been a powerful demonstration of its model-agnostic claim.
- The core Answer Consensus Refinement (ACR) module is fundamentally incompatible with open-ended tasks, as it is designed only for multiple-choice questions. Could you provide the experimental results of the proposed method on open-ended tasks?
- Experimental results on SVCFS (TAS-FS + FGMI-FS) without ACR are removed from Table 2.

### Questions
- Detailed explanation on notation $s_i^{ind}, s_i^{con}$. There is no explanation that $s_i^{ind}$ is a frame-independent score and $s_i^{con}$ is a temporal-context score. 
- The paper motivates the "section-wise partitioning" strategy by claiming that a simple top-M score ranking "may overconcentrate selections in certain segments". This is a key justification for the TAS-FS design. Could the authors provide a qualitative example that visualizes this failure case?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes SeViCES, a training-free, model-agnostic framework to improve long video understanding in Video-LLMs by selecting better evidence frames and enforcing semantic–visual consensus. It has two main components: (1) The Semantic–Visual Consensus Frame Selection (SVCFS) module selects frames through a temporal-aware semantic branch that leverages LLM reasoning over
captions, and a cluster-guided visual branch that aligns embeddings with semantic scores via mutual information. The second part is the Answer Consensus Refinement (ACR) module, which further resolves inconsistencies between semantic- and visual-based predictions by fusing evidence and constraining the answer space. Experiments on VideoMME, MLVU, LongVideoBench, and LVBench with three Video-LLMs (Qwen2.5-VL, LLaVA-Video, InternVL2.5) show consistent accuracy gains.

### Strengths
1. Clear and timely problem focus (long video + Video-LLMs).
The paper tackles an important and under-served problem: long video QA where naive uniform sampling or brute-force token input is infeasible. The motivation (computational cost, diluted attention, loss of reasoning consistency) is well-argued and backed by recent prior work.

2. Conceptually neat “consensus-driven evidence selection” idea.
Combining semantic (caption + LLM reasoning) and visual (embedding + clustering + MI) signals for frame selection is a clean, intuitive idea.
The extra step of answer-level consensus (ACR) — treating disagreement between semantic- and visual-based predictions as a signal of evidence incompleteness — is novel and nicely bridges selection and reasoning.

3. Experiment results demonstrate the effectiveness of the proposed method.

### Weaknesses
1. My major concern is the computation cost and scalability of the proposed LLM-based scoring method. TAS-FS requires two LLM calls per frame (independent + temporal context), plus captioning via BLIP-2 for all frames. Thus the computational cost is significantly higher than other methods with simple LLM usage. 

2. The computational cost is not experimented and discussed in this work. It is recommend to compare the runtime cost with some benchmarking methods.

### Questions
Please refer to the weakness

### Soundness
3

### Presentation
3

### Contribution
3
