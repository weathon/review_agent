# Interactive Multi-event Video Retrieval with Context Integration and Position Constraint

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Interactive video retrieval aims to progressively refine queries through multi-round interactions between the user and the system. Existing methods focus on pre-trimmed videos that provide captions that well describe the gist of the video content. In real-world scenarios, however, videos typically contain a sequence of unrelated and discontinuous events, while a query usually refers to a single event. This mismatch introduces significant challenges, including sensitivity to irrelevant content, lack of context exploitation, and insufficient position exploration. 
Motivated by this, we propose **CIPC**, a tailored interactive video retrieval framework with Context Integration and Position Constraint for multi-event videos. CIPC adaptively segments videos into event-consistent units, supports progressive interactions that exploit contextual information, and incorporates a position constraint to re-weight candidate segments by temporal distance, promoting better temporal alignment with the query. Extensive experiments and a user simulation study demonstrate the effectiveness and robustness of our approach, yielding 4.1\%–6.7\% R@1 improvements on three widely used benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes **Context Integration and Position Constraint** (CIPC), a multi-round interactive video retrieval system tailored for multi-event videos. 
The core contribution is threefold:

1. **Adaptive video segmentation**. This module segments videos from coarse-grained to fine-grained, thereby enabling querying at different granularities.
2. **Progressive interaction with context integration**. This module considers both segment-specific semantics as well as context-related segments, thereby producing more accurate representations for a given segment.
3. **Position constraint**. The authors additionally compute the estimated query temporal position with retrieved segments to refine temporal search accuracy.

The authors conduct many experiments on three benchmarks and show that CIPC achieves SOTA performance. The motivation of this paper is clear, with reasonable explanation of each component and promising results. I hold a positive attitude towards this paper.

### Strengths
1. The motivation is clearly stated and sounds reasonable.
2. Each component is rationalized with clear figures, making readers easy to follow.
3. The results are promising, and the ablation studies show that each component brings improvement.

### Weaknesses
1. **Unclear interaction rules.** It seems there should be a parameter to decide which rounds belong to coarse-grained search and which ones belong to fine-grained search in the Progressive Context Interaction module. But I did not find a related statement or experiments in the manuscript. Please correct me if I misunderstand something.

2. **No failure mode handling.** Since the multi-round interactive retrieval can be regarded as a decision chain, there is a possibility that there are no valid results based on the last-round query. No solutions are provided in this situation. And it is unclear what will happen if the VLM raises a totally unrelated question in one round?

3. **Reliance on strong commercial models.** From Figure 5, it seems there is a clear gap between frontier commercial models and open-source ones, indicating that the performance is dominated by the quality of generated questions. So this work may not be easy to follow for the budget-constrained research community.

### Questions
Since we already integrate context information into the queries, why is there still a large gap between whether using position constraint in Table 3, i.e., rows 2, 3, and 5? Does this mean the integrated context actually works for some other aspects instead of temporal alignment?

### Soundness
4

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
4

### Summary
This paper proposes CIPC, a framework for interactive video retrieval over multi-event/untrimmed videos. CIPC segments videos into semantically coherent events adaptively, enables progressive multi-round interaction with context integration, and introduces a position constrain.

### Strengths
1. This manuscript identifies and articulates the problem of interactive retrieval in untrimmed, multi-event videos, where previous methods fall short.
2. User simulation addresses practical issues like noisy/incomplete feedback.

### Weaknesses
1. The related-work section surveys interactive retrieval across images/videos, but it omits several recent chat-based person re-ID papers that are highly relevant to the paper’s interactive design. Please discuss [1-3], and position your approach relative to these dialogue-driven systems (e.g., questioner design, round-level supervision).
2.  The method relies on a position constraint that re-weights segment similarity by the temporal distance between the query’s position and a segment’s center. In practice, how is the query’s temporal position obtained from a real user? The manuscript currently simulates the query position via frame–text similarity, but it is unclear how this aligns with real interactive usage where users may provide only coarse temporal hints (e.g., “near the end”).
3. The adaptive segmentation itself is clear, but please articulate the novelty beyond prior action segmentation or keyframe detection [6-8].
4. Sec. 3.3 describe a two-branch: segment-specific refinement and context-integrated refinement. But the implement details of the branches remain under-specified. Please provide a concise algorithm box or pseudo-code.
5. Please add qualitative cases where noise leads to wrong answers and analyze whether the pipeline can ever benefit from mildly incorrect answers.
6. Using the top-5 retrieved videos to condition question generation yields slight improvements but is deemed computationally inefficient. Please specify the implementation details of this strategy, quantify the extra compute, and justify the final choice not to adopt the top-5 strategy.
7. Please provide both theoretical and empirical comparisons to MERLIN [4] and UMIVR [5], with a particular focus on efficiency and scalability for long-video scenarios.

[1] Lu Y, Yang M, Peng D, et al. LLaVA-ReID: Selective Multi-image Questioner for Interactive Person Re-Identification[J]. arXiv preprint arXiv:2504.10174, 2025.
[2] Bai Y, Ji Y, Cao M, et al. Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 3952-3962.
[3] Niu K, Yu H, Zhao M, et al. Chatreid: Open-ended interactive person retrieval via hierarchical progressive tuning for vision language models[J]. arXiv preprint arXiv:2502.19958, 2025.
[4] Han D, Park E, Lee G, et al. Merlin: Multimodal embedding refinement via llm-based iterative navigation for text-video retrieval-rerank pipeline[J]. arXiv preprint arXiv:2407.12508, 2024.
[5] Zhang B, Cao Z, Du H, et al. Quantifying and narrowing the unknown: Interactive text-to-video retrieval via uncertainty minimization[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 22120-22130.
[6] https://katna.readthedocs.io/
[7] Kumar S, Haresh S, Ahmed A, et al. Unsupervised action segmentation by joint representation learning and online clustering[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 20174-20185.
[8] Du Z, Wang X, Zhou G, et al. Fast and unsupervised action boundary detection for action segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 3323-3332.

### Questions
1. In practice, how is the query’s temporal position obtained from real users? Since the manuscript simulates this via frame–text similarity, how will the deployed system handle coarse cues (e.g., “near the end”) or incorrect hints, and how robust is the system’s performance to such errors?
2. What is the specific novelty beyond prior action segmentation or keyframe detection (e.g., [6–8])?
3. Could you add a concise algorithmic description or pseudocode that (i) specifies when and how each branch is triggered and (ii) details the per-round operations in each branch?
4. Can you include qualitative examples in which noise leads to wrong answers, and analyze whether mildly incorrect answers can ever help the pipeline?
5. How exactly is the top-5 retrieved-video context injected into question generation? What is the additional computational cost?
6. Please provide both theoretical and empirical comparisons of efficiency and scalability (add “relative to MERLIN [4] and UMIVR [5]” for clarity).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles interactive video retrieval for untrimmed, multi-event videos and proposes CIPC—a framework that (i) adaptively segments each video into event-consistent units using frame-wise similarity with an adaptive threshold (coarse→fine) (Eqs. 1–3), (ii) performs progressive interaction with context integration to refine the query with both segment-specific and neighboring-context cues, and (iii) adds a position-constraint that re-weights segment similarity by temporal proximity to the (estimated) query position, combining segment and context scores for final ranking (Eqs. 7–10). On ActivityNet Captions, Charades-STA, TVR, CIPC improves R@1 vs. prior IVR baselines and vanilla systems (e.g., up to 31.3/14.6/22.8 R@1 after 10 rounds) and beats non-interactive retrieval; ablations show each module helps and the adaptive segmentation is strongest.

### Strengths
Clear decomposition of the multi-event challenge into irrelevant content, context use, and position cues, with matching modules.
Method is simple, training-free over CLIP features at retrieval time, yet effective across three benchmarks with consistent R@1/Hit@1 gains and lower BRI.
Good ablations: module contributions, segmentation strategies, questioner models/strategies, and robustness via user simulation (noise and weaker VLMs).

### Weaknesses
1. The “query position” is simulated via frame–query similarity; clarify how real users would supply temporal hints and evaluate robustness to noisy/misattributed hints and different weighting functions.
2. Adaptive thresholding on CLIP frame features is simple; add failure-case and cross-domain analyses (e.g., rapid cuts, subtle transitions), boundary-accuracy metrics, and comparisons to learned/change-point baselines.
3. No end-to-end profiling of the interactive loop; report per-round latency/memory (segmentation, retrieval, VLM calls) on commodity CPU/GPU and compare with lightweight alternatives.
4. Lacks analysis of how question wording/granularity evolves across rounds; ablate number of turns, coarse-to-fine strategies, and stopping criteria, and quantify their impact on retrieval metrics and user effort.
5. Results depend on chosen VLMs and temperatures; report variance over prompts/seeds/backbones and include stronger multi-segment reasoning baselines (e.g., hierarchical or agentic retrieval).

### Questions
1. Can human users specify temporal hints (e.g., “near the end”) and how would noisy hints propagate through Eq. (8)?
2. How sensitive is CIPC to the CLIP backbone (e.g., ViT-L/14) and the frame sampling rate F?

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
5

### Summary
This paper proposes a  framework for Interactive Video Retrieval (IVR) in the challenging and realistic scenario of multi-event (untrimmed) videos. The authors identify the limitations of prior IVR methods, which primarily focus on pre-trimmed videos, and articulate three key challenges: sensitivity to irrelevant content, lack of context exploitation, and insufficient position exploration. CIPC addresses these issues through three main components: an Adaptive Video Segmentation strategy, Progressive Interaction with Context Integration, and a Position Constraint mechanism. The empirical results demonstrate significant improvements over both interactive and non-interactive baselines, with ablation studies validating the efficacy of each proposed component.

### Strengths
1.The motivation is clear.
2. The paper is well-presented

### Weaknesses
1.While applying Interactive Retrieval, a recent hot topic accelerated by Large Language Models (LLMs), to the domain of untrimmed video is a good idea, the paper lacks a sufficiently deep theoretical consideration on the optimal interactive query strategy for Video Retrieval. In established Interactive Retrieval fields like Person Re-Identification (ReID) or image retrieval, the primary research focus is on determining what types of questions the LLM should ask to effectively discriminate between highly similar candidates (e.g., asking about fine-grained attributes of two similar-looking individuals). For Video Retrieval, where video events are complex and span time, the core challenge of the interaction remains ambiguous. The authors highlight the limitations of existing Interactive Retrieval methods but fail to clearly articulate the unique, high-level attributes (e.g., subtle temporal differences, inter-event causal relations) that the model should target during progressive questioning. This oversight suggests the framework may be relying on a generic LLM interaction, which is sub-optimal for unlocking the full potential of interactive retrieval in complex temporal sequences.

2. The proposed Adaptive Video Segmentation relies on an adaptive threshold ($\tau$) calculated from the magnitude of visual feature distance ($\delta_t$) between adjacent frames. This strategy is fundamentally susceptible to scenarios where visual dynamics decouple from semantic coherence, leading to critical segmentation failures:Case A: Semantic Shift with Visual Smoothness: A major semantic boundary occurs (e.g., the completion of one complex sub-task and the initiation of the next) under a visually continuous camera operation (e.g., slow pan, zoom, or long take). The resulting small $\delta_t$ will likely cause the two distinct semantic events to be erroneously merged into a single segment, thus misaligning the available retrieval unit with the user's target.Case B: Visual Jitter with Semantic Coherence: A single, semantically continuous event (e.g., a conversation or continuous action) is recorded using rapid, visually jarring techniques like quick cuts or extreme hand-held motion. The resultant large, spurious $\delta_t$ will cause the event to be incorrectly fragmented into multiple, artificially short segments.The reliance on a purely visual-distance metric, even with an adaptive threshold, suggests a vulnerability in the system's foundational step, posing a significant risk of propagating initial segmentation errors to the subsequent, refinement-based retrieval stages.

3.The Context Integration mechanism, which refines the query by utilizing information exclusively from the immediate preceding and succeeding video segments, enforces a strong Markovian assumption on the inter-event dependencies. While effective for short-range relationships, this local scope inherently limits the framework's ability to resolve target events that require long-range temporal or causal context. In narrative-heavy or procedural long videos (e.g., documentaries, instructional content), the most salient contextual cues might be located far from the target event. This design choice, therefore, places a theoretical ceiling on the complexity of video relationships the model can effectively leverage, preventing it from tackling the full breadth of real-world multi-event video understanding.

4. Related paper to be cited: IVCR-200K: A LARGE-SCALE MULTI-TURN DIALOGUE BENCHMARK FOR INTERACTIVE VIDEO CORPUS RETRIEVAL

### Questions
na

### Soundness
3

### Presentation
3

### Contribution
2
