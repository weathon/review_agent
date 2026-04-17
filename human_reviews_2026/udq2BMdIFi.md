# Q-Router: Agentic Video Quality Assessment with Expert Model Routing

- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Video quality assessment (VQA) is a fundamental computer vision task that aims to predict the perceptual quality of a given video in alignment with human judgments. Existing performant VQA models trained with direct score supervision suffer from **(1)** *poor generalization* across diverse content and tasks, ranging from user-generated content (UGC), short-form videos, to AI-generated content (AIGC), **(2)** *limited interpretability*, and **(3)** *lack of extensibility* to novel use cases or content types. We propose Q-Router, an agentic framework for universal VQA with a multi-tier model routing system. Q-Router integrates a diverse set of expert models and employs vision–language models (VLMs) as real-time routers that dynamically reason then ensemble the most appropriate experts conditioned on the input video semantics.  We build a multi-tiered routing system based on the computing budget, with the heaviest tier involving a specific spatiotemporal artifacts localization for interpretability. This agentic design enables Q-Router to combine the complementary strengths of specialized experts, achieving both flexibility and robustness in delivering consistent performance across heterogeneous video sources and tasks.  Extensive experiments demonstrate that Q-Router matches or surpasses state-of-the-art VQA models on a variety of benchmarks, while substantially improving generalization and interpretability. Moreover, Q-Router excels on the quality-based question answering benchmark, Q-Bench-Video, highlighting its promise as a foundation for next-generation VQA systems. Finally, we show that Q-Router can localize spatio-temporal artifact/hallucination localization, showing potential as a reward function for post-training video generation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Q-Router, an agentic framework for Video Quality Assessment (VQA). The core idea is to leverage a Vision-Language Model (VLM), such as GPT-4o, as a dynamic router to select, execute, and fuse predictions from a pool of specialized VQA expert models. The framework is structured into a multi-tiered hierarchy (Tier 0, 1, 2) to balance computational cost with performance. Tier 0 uses a single selected expert for efficiency, Tier 1 fuses multiple experts for better accuracy, and Tier 2 introduces a spatiotemporal artifact localization pipeline for enhanced interpretability. The authors demonstrate that Q-Router achieves state-of-the-art or highly competitive performance on a wide range of VQA benchmarks, including user-generated content (UGC), AI-generated content (AIGC), and a video quality-based question answering task (Q-Bench-Video).

### Strengths
- Novel Agentic Framework: The core idea of using a VLM as an intelligent "router" to orchestrate a pool of expert models is highly original in the context of VQA. It represents a paradigm shift from building monolithic, end-to-end models to creating flexible, reasoning-based systems. This aligns well with the current trend of agentic AI.

- Improved Interpretability and Flexibility: The framework is inherently more interpretable than black-box models. The VLM provides a rationale for its routing choices, and the Tier 2 artifact localization offers concrete visual evidence for quality degradation. The modular design, allowing for the easy addition of new expert models, makes the system highly extensible.

- Comprehensive and Strong Empirical Results: The paper validates its approach across a wide variety of challenging benchmarks, including UGC, AIGC, and visual question answering. Q-Router consistently achieves state-of-the-art or near-SOTA performance, demonstrating the effectiveness and robustness of the proposed method. The significant outperformance on AIGC content is particularly noteworthy.

- Practical Multi-Tiered Design: The hierarchical (Tier 0-2) approach is a pragmatic and well-thought-out design choice. It allows the framework to be adapted to different application scenarios with varying constraints on latency and computational resources, from lightweight screening to in-depth analysis.

### Weaknesses
- Performance Paradox in Expert Fusion: A critical weakness is that the agentic fusion does not always outperform the best expert in its pool. For instance, in Table 1 on LSVQ-Test and LIVE-VQC, DOVER achieves a higher PLCC/SRCC than the final Q-Router score. This is counter-intuitive for an ensemble-based method and undermines the claim that the VLM is optimally combining expert knowledge. An effective router/fusion mechanism should, at minimum, aim to match the best expert or, ideally, exceed it by combining complementary strengths.

- Heuristic-Driven Artifact Localization: The spatiotemporal artifact localization pipeline in Tier 2, while effective, feels heavily reliant on handcrafted features (motion residuals, gradient kurtosis, etc.) and a chain of heuristic-based steps (hysteresis clipping, diversity sampling). This design feels somewhat at odds with the modern, learning-centric "agentic" theme of the paper. It raises questions about its generalizability to novel artifacts not captured by the pre-defined features.

- Ambiguity in the "Learned" Routing Strategy: The paper states that the routing strategy can range from "rule-based heuristics to learned policies" (line 122). However, the described implementation for the main results seems to be a sophisticated, but ultimately fixed, prompt for GPT-4o. It is unclear if any part of the routing logic is actually "learned" from data in a traditional sense (e.g., via reinforcement learning or fine-tuning). If the intelligence is purely derived from the in-context learning of a proprietary VLM, the novelty of the routing methodology itself is less a new learning algorithm and more a case of advanced prompt engineering.

- Misleading Use of "Real-Time": The abstract describes the VLM as a "real-time router". Given that the backbone is a large model like GPT-4o, which involves API calls and significant computation, the latency is very unlikely to meet a typical real-time definition (e.g., >30 fps or <33ms). This claim should be removed or clarified with concrete latency measurements.

### Questions
- Regarding the Fusion Performance: Could you please elaborate on the instances in Table 1 where Q-Router underperforms its best constituent expert (e.g., DOVER)? What is your hypothesis for why the VLM-based fusion might be sub-optimal in these cases? Does this reveal limitations in the VLM's ability to weigh experts, or is it an artifact of the fusion formula?

- On the Nature of the Router: Could you clarify the extent to which the routing policy is "learned"? Is the system's performance entirely dependent on the in-context learning capabilities of the off-the-shelf GPT-4o, or is there a fine-tuning or other learning-based component involved in teaching the router how to route?

- Regarding the Localization Pipeline:
Why was a handcrafted feature approach chosen for the "Probabilistic Frame Extraction" instead of a learned one? How were the weights for the logistic model determined?
The heatmap generation relies on LPIPS. How well does this generalize to artifacts that LPIPS is not sensitive to (e.g., certain types of generative artifacts)? Have you considered alternative methods for generating perceptual difference maps?

-On Terminology and Claims:
Could you provide latency benchmarks for Tier 0, 1, and 2 to give a concrete sense of the computational cost and to justify the discussion around different budgets? This would also help clarify the "real-time" claim.
What is the justification for the name "Probabilistic Frame Extraction"? The process appears to be deterministic, assigning a score based on features rather than sampling from a distribution.

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
5

### Summary
This paper introduces Q-Router, a novel video quality assessment (VQA) framework that leverages an agentic routing system with a pool of specialized expert models. Specifically, it utlizes the vision-language model (VLM) as the backbone to implement a tiered routing mechanism, which adaptively selects and fuses expert models based on the semantic characteristics of the input video. In addition, Q-Router introduces a spatio-temporal artifact localization module that identifies and visualizes distortion regions through detailed heat maps. These localized artifact maps can enrich the system’s understanding of video quality and providing interpretable explanations alongside a robust, fused quality score in the full pipeline.

### Strengths
1. This paper introduces Q-Router, the first-of-its-kind agentic VQA framework that employs reasoning VLMs to orchestrate a pool of specialized expert VQA models. This design brings reasoning-driven adaptability and interpretability into the VQA process.

2. The proposed multi-tier routing system adapts to computational budgets and task requirements, ranging from fast, single-expert inference to comprehensive artifact localization analysis. This design ensures scalability and efficiency, making Q-Router practical for large-scale, real-world video datasets.

3. This paper introduces a novel spatio-temporal artifact localization pipeline that identifies and visualizes degraded regions in videos both spatially and temporally. This component provides deeper insights into the nature and location of distortions, enhancing the transparency and diagnostic value of the VQA process.

4. The proposed Q-Router framework can be depolyed to the downstreams VQA tasks across diverse video domains, including UGC, AIGC, CG videos. And its modular expert pool also allows flexible adaptation to other VQA tasks.

5. Extensive experiments showcase the Q-Router have better generalisation and interpretability than prior work, especialy achieving the sota on Q-Bench-Video.

6. The paper conducts comprehensive experiments across multiple VQA benchmarks, showing that Q-Router achieves superior generalization and interpretability compared to prior methods. In particular, it achieves the new state-of-the-art performance on the Q-Bench-Video benchmark, hightlighting the effectiveness of its routing and expert fusion strategies.

### Weaknesses
1. Although Q-Router emphasizes a budget-aware multi-tier design, the paper lacks a comprehensive comparison of runtime, FLOPs, or latency across tiers. Without these results, it is difficult to assess the framework’s real-world scalability and deployment feasibility.

2. The error cases or failure patterns of Q-Router (e.g., misrouting on fast-motion or high-texture scenes) are not discussed, which limits understanding of its boundaries.

3. The paper could include a clearer description of routing decision examples, illustrating how the VLM selects or fuses experts in representative scenarios.

4. The criteria for selecting and composing the expert pool are not clearly explained, making it difficult to understand how expert diversity and complementarity were ensured.

5. The fusion strategy for combining outputs from multiple experts is only briefly described. More details on weighting mechanisms, normalization, or fusion functions would clarify how the final quality score is derived.

### Questions
In the illustrated prompts for Tier 1 and Tier 2, predefined weights for experts are assigned to different input video types (UGC, AIGC, CG). What is the rationale behind these weight assignments, and how were they determined or validated?

### Soundness
3

### Presentation
3

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
This paper introduces Q-Router, an agentic framework for video quality assessment that employs vision–language models as routers to dynamically select and ensemble specialized expert VQA models. The proposed system operates in a three-tier hierarchy---from single-expert lightweight routing (Tier 0), multi-expert fusion (Tier 1), to full spatio-temporal artifact localization (Tier 2). Experimental results on UGC and AIGC datasets, as well as video quality question-answering benchmarks, show that Q-Router achieves robust performance and improved interpretability compared with existing methods.

### Strengths
1. The paper proposes a new paradigm for VQA---leveraging a VLM-based routing system to coordinate multiple expert models. This agentic design is conceptually elegant.

2. The artifact localization pipeline (using probabilistic frame extraction, VLM-based filtering, and LPIPS heatmaps) adds interpretability rarely seen in existing VQA frameworks, providing visual evidence and diagnostic capability.

### Weaknesses
1.	Regarding Table 1, which presents results on several UGC datasets and one AIGC dataset — why are only the results of Q-Router (Tier 1) reported? Where are the results for Q-Router (Tier 0) and Q-Router (Tier 2)? Are Tier 0 and Tier 2 configurations only applicable to the video quality question answering task? In Section 2.2, where Q-Router is introduced, I could not find a clear justification or explanation for this choice.

2.	In Table 2, the performance of the compared methods appears questionable. For example, the reported performance of UVQ on YT-UGC is extremely high. According to the UVQ paper [1], the model was trained on YT-UGC, so how did the authors evaluate its performance on the same dataset? The reported performance of UVQ here is significantly higher than that reported in its own paper. In contrast, the performance of COVER [2] on YT-UGC is unexpectedly low. It seems that COVER is also trained on YT-UGC. This inconsistency is confusing. In my view, if YT-UGC was used for training UVQ and COVER, it should not be treated as a test benchmark unless a clear and consistent training–testing split is ensured for all models. Additionally, the reported performance of DOVER is far below what was shown in its original [3] and related papers [4], and ModularVQA [4] also performs much worse than reported in its original publication on both LSVQ and YT-UGC.

3.	I think the range of VQA benchmarks used in the paper to be too limited for a system claiming universal VQA capability. Only two types of datasets, UGC and AIGC, are considered, and these two types of content differ drastically. Consequently, one class of VQA models naturally performs much worse than the other. In such a setting, simply using a video type classifier (which is trivial to implement) to identify the content type and then selecting the corresponding VQA model could already yield high overall performance. This overly simplified setup diminishes the demonstrated effectiveness of Q-Router. Combined with the suspicious results noted above, I find it difficult to conclude that Q-Router truly achieves strong or reliable performance.

4.	For the AIGC setting, only the T2VQA-DB dataset is used for benchmarking. However, AIGC-VQA is a highly challenging problem, often involving complex cross-domain inconsistencies. The distribution gap across different AIGC datasets can also be substantial. Therefore, using a single benchmark is insufficient to evaluate the robustness of Q-Router for AIGC scenarios.

5.	The explanation of Tier 0–Tier 2 in Section 2.2 is rather vague, and it is unclear how these agents actually operate. For instance, in Tier 0, the authors claim that a single expert model is selected from the routing pool based on characteristics such as structural complexity, content modality, and observable quality attributes. However, these concepts are not formally defined, how exactly are these attributes quantified or used?

Furthermore, the prompt in Figure 5 seems inconsistent with the statement in Section 2.2: its prompt indicates that “based on the given expert scores (already scaled to [0–100]), weights are dynamically assigned to each expert according to biases, context, and confidence priors.” This implies that multiple experts are still involved, even in Tier 0, which contradicts the description that Tier 0 selects only a single expert. This contradiction needs clarification.

6.	For Tier 2, the motivation behind the dynamic weighting strategy should be clearly explained. The prompt shown in Figure 4 states:

“Use these weighted scores to compute a final score:
If ‘max(score) – min(score) > 20’, use weighted median; otherwise use weighted average.
Round to the nearest integer in [0–100].”

and further defines the baseline priors as:

“Video type → baseline weight priors:
UGC: UVQ 0.25, COVER 0.25, ModularBVQA 0.15, RQ-VQA 0.10, MaxVQA 0.15
Short-form/social: RQ-VQA 0.30, COVER 0.30, UVQ 0.20, Modular 0.10, MaxVQA 0.10
Gaming: COVER-Technical 0.35, UVQ 0.25, Modular 0.20, MaxVQA 0.10, RQ-VQA 0.05
AI-Generated: T2VQA 0.35, COVER 0.20, UVQ 0.15, MaxVQA 0.15, Modular 0.10, RQ-VQA 0.05”
and the corresponding weight adjustment formula as:
weight_i = base_i × (1 + 0.5 × specialty match + 0.3 × agreement boost + 0.2 × confidence prior – 0.3 × oob penalty)

However, the rationale for these coefficients, thresholds, and multipliers is not justified. Why were these specific factors and numerical values chosen? Moreover, RQ-VQA appears in the “Video type → baseline weight priors” table, yet it is not included among the expert models in the routing pool. This inconsistency should be clarified.

7.	How is the effectiveness of Spatio-Temporal Artifact Localization quantified? Since this is presented as a core contribution of the paper, a clear metric or evaluation protocol is necessary to substantiate its impact.

8.	The authors state that Tier 0 is the most lightweight configuration suitable for real-time or resource-constrained settings. However, it still relies on GPT-4 to process video frames, and the Tier 0 prompt indicates that expert model scores are also required. How computationally expensive is this operation in practice? Please provide the runtime and cost analysis.

9.	Table 2 lacks comparisons with recent state-of-the-art methods such as VQA2 [5] and Q-Instruct [6], which are relevant baselines for a fair evaluation.

10.	In Line 239, what does Step 1 refer to? The description is unclear.

[1] Rich features for perceptual quality assessment of UGC videos

[2] COVER: A Comprehensive Video Quality Evaluator

[3] Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives

[4] Modular Blind Video Quality Assessment

[5] VQA^2: Visual Question Answering for Video Quality Assessment

[6] Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models

### Questions
1. Clarify the evaluation protocol in Table 1. How were the compared methods tested? Many reported results are inconsistent with those in the original papers and also with the reproduced results in other studies. More importantly, the performance of UVQ and COVER on YT-UGC appears highly unusual and requires further explanation.

2. Clarify the functioning of Tier 0 to Tier 2. Please explain clearly how these tiers operate. Specifically, do Tier 0 and Tier 2 perform quality scoring? Does Tier 0 involve multiple expert models, or does it select only a single one? The prompts shown in Figures 4–6 should also be clarified to avoid ambiguity.

3. Expand the evaluation benchmarks. Please include more VQA datasets, particularly additional AIGC datasets. It would also be beneficial to incorporate other types of VQA datasets, such as those related to compression or other distortion domains, to provide a more comprehensive evaluation.

4. Provide an analysis of computational cost. The paper emphasizes that Tier 0 is a lightweight baseline. Therefore, a clear analysis of the computational complexity and runtime cost for Tier 0–Tier 2 should be provided to support this claim.

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
4

### Summary
This manuscript proposes Q-Router, an agentic framework for Video Quality Assessment (VQA) designed to address the poor generalization, limited interpretability, and lack of extensibility in existing models. The core concept is to employ a Vision-Language Model (VLM) as a dynamic "router". This VLM agent analyzes the input video's semantics and characteristics to select and dynamically ensemble predictions from a diverse pool of specialized, state-of-the-art VQA "expert" models. The authors conduct extensive experiments on both classic VQA scoring (across UGC and AIGC benchmarks) and quality-based video question answering (Q-Bench-Video). The results show that Q-Router achieves state-of-the-art average performance on the VQA benchmarks, with particularly strong results on AIGC content.

### Strengths
1. The agentic routing paradigm is a novel and compelling contribution to the VQA field. It reframes VQA from a monolithic, end-to-end prediction task into a more flexible, hierarchical, and reasoning-based ensemble problem. 
2. The paper demonstrates that Q-Router overcomes a key weakness of current models: generalization to diverse content types. Its performance on the AIGC benchmark (T2VQA-DB) is particularly impressive (0.8283 PLCC).
3. The Tier 2 pipeline can identify *where* and *what* (e.g., hallucinations, blurring) artifacts are occurring, cause it introduces explicit spatiotemporal artifact localization, the system provides actionable, interpretable evidence.

### Weaknesses
1. The most significant weakness is the practical inference cost. The full framework, especially Tiers 1 and 2, requires running *multiple* expert VQA models in parallel *in addition to* a large VLM (GPT-4o). This makes the system far more computationally expensive than any single baseline model, likely precluding its use in any real-time or large-scale video processing applications.
2. The system's "router" and "fusion operator" is GPT-4o, a closed-source and proprietary model. This raises major reproducibility concerns. It is unclear how much of the performance lift comes from the novel *routing framework* itself versus the powerful, general-purpose reasoning capabilities of GPT-4o. The paper lacks a crucial ablation study using open-source VLMs (e.g., LLaVA, InternVL) as the router.
3. The Tier 2 artifact localization method, while effective, is a complex, multi-stage pipeline. It relies on handcrafted features (e.g., motion residuals, gradient kurtosis) , classic optical flow , and a separate VLM filtering step. This heuristic-driven approach feels less elegant and adds multiple potential points of failure compared to a more integrated, end-to-end learned diagnostic model.

### Questions
1. Could the authors provide a detailed analysis of the wall-clock inference latency for each tier (0, 1, and 2), perhaps in seconds-per-video or FPS? How does this compare directly to the latency of the individual expert models like COVER or DOVER?
2. How critical is GPT-4o to the Q-Router's success? Have the authors experimented with substituting the GPT-4o router with a smaller, open-source VLM? This is essential for evaluating the framework's true contribution and reproducibility.
3. In the Tier 2 pipeline (Sec 2.3), a VLM is used to filter candidate frames and classify artifacts. What is the accuracy of this classification step? How does a failure at this stage (e.g., the VLM missing an artifact) impact the final localization map and quality score?
4. The prompt shown in Figure 4 for the Tier 1 VQA task includes specific "baseline weight priors" (e.g., "UGC: UVQ 0.25, COVER 0.25...") and a "weight adjustment formula". How were these priors and this formula derived? How sensitive is the system's performance to this specific prompt engineering?

### Soundness
3

### Presentation
3

### Contribution
2
