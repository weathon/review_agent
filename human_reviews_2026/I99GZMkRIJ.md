# HIERARCHICAL ADAPTIVE SAMPLING FOR VIDEO UNDERSTANDING

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Due to the limited context window of Multi-modal Large Language Models (MLLMs), processing an entire video is infeasible. As a result, prevailing models mostly sample a subset of frames to substitute for the full video as input to the model.
However, existing sampling methods, from simple uniform sampling to more refined methods based on query-frame relevance, employ a fixed sampling strategy that does not vary with input, failing to adapt to the diverse nature of queries, as some require comprehension of the video in its entirety, while others focus on events within short temporal segments.
To address this limitation, we propose Hierarchical Adaptive Sampling (HAS), a two-stage frame sampling framework. In the first stage, Backbone Frame Construction, we apply a Determinantal Point Process (DPP) to sample frames that are both query-relevant and non-redundant. These selected frames form the backbone of the entire sampling set, providing the foundational structure for subsequent enrichment. In the second stage, Adaptive Contextual Enrichment, we analyze the temporal distribution of the backbone frames to infer the query type and adaptively allocate the remaining frame budget between Local and Global Context. The local context enriches the backbone with fine-grained temporal dynamics and short-range causal relations, whereas the global context provides a holistic view of the entire video to enhance broader contextual understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces Hierarchical Adaptive Sampling (HAS), a two-stage frame sampling framework designed to capture both local and global video content. Starting from a predefined frame budget, HAS dynamically allocates sampling, focusing on global information when the video content is sparse and emphasizing local details when temporal density is high. The experiments are comprehensive and well-explained, demonstrating the effectiveness and adaptability of HAS.

### Strengths
- The method is simple yet impactful. The components of HAS are well-designed and grounded in solid intuition. It effectively considers both local and global video contexts while remaining efficient and flexible.
- HAS consistently outperforms prior frame-sampling methods across three benchmarks.
- The ablation studies are thorough and strongly support the effectiveness of the proposed method.

### Weaknesses
While the paper is generally strong, there are a few aspects that could benefit from additional clarification or improvement:
- In Section 2.3, I recommend that the authors explicitly clarify which aspects are overlooked in previous frame-sampling methods and how HAS addresses or improves upon these limitations. The current version primarily lists prior methods without clearly contrasting them or highlighting HAS’s specific advancements.
- Have authors investigated the effect of the threshold $\tau$ in Adaptive Contextual Enrichment? Additionally, it would be great if authors can improve the part of Adaptive Contextual Enrichment in Fig 1. 
- The evaluation is comprehensive; but it mainly considers video question answering benchmarks. It would be interesting to examine whether the method can also improve other video understanding capabilities, such as temporal grounding [1, 2] and grounded video question answering [3].

**References**

[1] ReXTime: A Benchmark Suite for Reasoning-Across-Time in Videos, NeurIPS 2024 

[2] On the Consistency of Video Large Language Models in Temporal Comprehension, CVPR 2025

[3] Can I Trust Your Answer? Visually Grounded Video Question Answering, CVPR 2024

### Questions
See the Weaknesses

### Soundness
3

### Presentation
3

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
This paper addresses a critical and timely challenge in Multi-modal Large Language Models (MLLMs): the limited context window makes processing long, full videos computationally infeasible. Authors propose propose Hierarchical Adaptive Sampling (HAS) consists of Backbone Frame Construction, and Adaptive Contextual Enrichment for adaptively allocate the remaining frame budget between Local and Global Context.

### Strengths
+ Complementary sampling strategies were proposed to jointly consider key frames, neighboring context and global context.

+ Good results are achieved, outperforming unform sampling.

+ The paper is well written.

### Weaknesses
1) The benefit of using DPP over TOP is not significant, being completive.  This alleviated the contribution of this paper.  

2) How to allocate the budge for DPP and others and their influence on performance should be discussed and analyzed. Is the percentage of DPP frames adaptive to videos? What are the statistics of the percentages?

3) Around (4), is there typo? In my understanding, for the highly clustered ones, the standard deviation is small, the score should approach 1 instead of 0. k_{GC} should be small. Does (4) have typo? 

4) The computation complexity could be analyzed by comparing different baselines.

### Questions
See weakness part.

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
4

### Summary
This paper introduces Hierarchical Adaptive Sampling (HAS) to improve video understanding in multi-modal large language models. HAS first selects a diverse, query-relevant backbone of frames and then adaptively enriches local or global context based on their temporal distribution. Experiments on multiple benchmarks show HAS outperforms uniform and existing advanced sampling methods, effectively balancing fine-grained and holistic video comprehension.

### Strengths
The integration of Determinantal Point Processes (DPP) with adaptive frame selection is an interesting and well-motivated idea. The paper is also clearly written and easy to follow.

### Weaknesses
The proposed hierarchical frame sampling strategy shows several overlaps with existing approaches and raises key concerns:

Similarity to prior works: The first stage, query-aware frame selection, closely resembles HierarQ (Azad et al., CVPR 2025), while the second stage, which divides frames into local and global contexts, follows ideas similar to HierarQ and VideoTree (Wang et al., CVPR 2025).

Non-redundant frame selection: The selection of diverse frames is not novel - many prior long-video methods employ alternatives such as cosine-similarity merging (HierarQ, MovieChat, MA-LLM), token merging (ToMe), or adaptive clustering (VideoTree). 
The proposed NP-hard probabilistic optimization seems unnecessarily complex, and the paper does not provide comparisons with these existing, often simpler, strategies. Additionally, there is no discussion of the computational overhead introduced by multi-stage, multi-scale sampling.

Over-aggressive sampling: The pipeline first subsamples videos at 1 FPS, then further reduces them to 64 frames, which is already standard input for most models. This raises doubts about scalability to truly long videos (e.g., MovieChat-1K, which can exceed 12 000 frames) — how would the model handle such scenarios when ultimately limited to just 64 frames?

Failure on dynamic datasets: On datasets with frequent scene or viewpoint changes (e.g., LiveSportsQA, Ego4D, YouCook2), such aggressive frame reduction will likely miss key temporal information and fail to capture fast-changing events. This limitation becomes even more critical for UCF-Crime or temporal grounding tasks like Charades-STA, where temporal continuity is essential.

Dataset bias: As noted in Table 2 of the NeurIPS 2024 "MMBench-Video: A Long-Form Multi-Shot Benchmark for Holistic Video Understanding" benchmark paper, VideoMME achieves around 54% accuracy using only a single frame, suggesting it rewards static visual perception rather than true long-term reasoning. The success of HAS may therefore stem from the frame-level nature of the benchmark, rather than genuine temporal understanding.

Supporting evidence: Prior work such as "Revealing Single Frame Bias for Video-and-Language Learning" has shown that single-frame cues can often suffice for many VideoQA tasks. Thus, while HAS performs well on current benchmarks, it may not reflect real long-term temporal reasoning performance.

Inconsistency in performance improvements: In several of the ablations, it seems like HAS is not always helping. It is sometimes hurting the performance.

### Questions
Relation to prior work: Your hierarchical frame sampling strategy shows similarities to methods like HierarQ and VideoTree. Could you clarify what distinct contributions HAS makes beyond these prior approaches, particularly in terms of query-aware selection and local/global context modeling?

Scalability and efficiency: Given that HAS first subsamples videos at 1 FPS and then selects only 64 frames, how does the method scale to very long videos (e.g., >10,000 frames) or datasets with frequent scene changes? Additionally, can you provide details on the computational overhead of the multi-stage, probabilistic sampling compared to simpler alternatives like token merging or adaptive clustering?

Temporal reasoning vs. benchmark bias: Many benchmarks used (e.g., VideoMME) show strong performance with even a single frame, raising the possibility that HAS may leverage frame-level cues rather than true long-term temporal reasoning. How does HAS perform on datasets or tasks requiring fine-grained temporal continuity, and can you provide evidence that its gains reflect genuine temporal understanding rather than dataset bias?

### Soundness
3

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
4

### Summary
This paper proposes an adaptive, training-free frame selection method (HAS) to filter informative keyframes for Video-MLLMs. By first constructing a set of backbone frames using a determinantal point process, which is then further supplemented with local or global frames based on the text query, HAS samples the most informative frames from a long video for maximizing MLLM performance. It is intuitive, plug and play with any MLLM model, and is compared with other keyframe selection baselines (UNI / TOP / AKS) on 3 benchmarks.

### Strengths
- **Clear, intuitive methodology** The two-stage design (backbone frame selection + query dependent enrichment) is straightforward and easy to understand.
- **Plug-and-play / training-free** Appears to be compatible with most MLLMs, which is practical and can be used with future models as well.
- **Decent ablation suite** The paper's current ablations almost fully covers important component/budget design choices.

.

### Weaknesses
- **Missing critical ablation on the VL feature extractor.** The entirety of HAS depends on the quality of the features produced by the feature extractor and their measured similarities. However, no ablation or reasoning is given as to why CLIP is used and considered sufficient for this task as opposed to more recent visual-language encoders such as SigLIP or SigLIP-2.

- **Insufficient baseline coverage.** Only UNI, TOP and AKS are used as comparisons for alternative keyframe sampling methods. More recent and sophisticated methods mentioned in the Related Works, such as T* (CVPR 25), should also be compared against. Moreover, if the authors are citing papers from CVPR 25, they should also include/compare other keyframe sampling papers such as (Flexible Frame Selection for Efficient Video Reasoning). This weakens broad superiority claims.

- **Ambiguous use of SOTA** The authors often times claim achieving SOTA results (Section 4.2.1) while larger open-source models such as LLaVA-Video 72B clearly outperforms all baselines presented in the paper. which is misleading. This point leads to two questions:
	- If HAS is training-free and model agnostic, why are results with 72B models not included?
	- If simply using a larger LLM leads to best performance over all presented baselines, how impactful is keyframe selection really? LLaVA-Video 72B achieves 62.1 on LongVideoMME with 64 frames sampled uniformly; again, applying HAS here is crucial to truly measure its impact on downstream performance.

- **Lack of qualitative results** 
	- The paper shows two qualitative examples in one figure but lacks a sufficient, quality investigation. Examples on the hour-long videos that HAS is quantitatively tested would provide strong support towards the authors claim of HAS' superiority in choosing better keyframes that are also relevant to the user query.  For example, per-question-type analysis or annotation-based checks that selected frames actually contain the answer more often than baselines would be valuable.
	- There are many claims throughout the paper that HAS can adaptively focus on local regions for fine-grained queries, while broadening to more sparse selections globally. No support is given for this claim besides marginal quantitative improvements, which still doesn't fully justify this claim.   
- **Poor visualizations** There are only two figures, and they are not very good. Figure 1 says "Local Context" twice instead of "Global Context" in the second block. Figure 2 is low quality and difficult to understand or see what keyframes were selected without major zoom.



## Minor comments
- **Missing runtime / cost analysis.** How long does HAS take to run? If it is negligible due to the lack of learning or auxiliary networks, it should be easy to add to the paper.
- **Candidate-fps sensitivity sweep.** The pre-sampled 1 FPS candidate pool is a design choice — show results at 0.5 / 2 / 4 FPS to test robustness to short/high-motion events.

### Questions
- If HAS is training-free and model agnostic, why are results with 72B models not included?
- Why do larger MLLM models outperform smaller models with keyframe selection? The performance improvement of keyframe selection seems to be less impactful than model size.
- Does using a better encoder (such as SigLIP) improve HAS performance?

### Soundness
2

### Presentation
1

### Contribution
2
