# HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Content-aware streaming requires dynamic, chunk-level importance weights to optimize subjective quality of experience (QoE). However, direct human annotation is prohibitively expensive while vision-saliency models generalize poorly. We introduce HiVid, the first framework to leverage Large Language Models (LLMs) as a scalable human proxy to generate high-fidelity weights for both Video-on-Demand (VOD) and live streaming. We address 3 non-trivial challenges: (1) To extend LLMs' limited modality and circumvent token limits, we propose a perception module to assess frames in a local context window, autoregressively building a coherent understanding of the video.
(2) For VOD with rating inconsistency across local windows, we propose a ranking module to perform global re-ranking with a novel LLM-guided merge-sort algorithm.
(3) For live streaming which requires low-latency, online inference without future knowledge, we propose a prediction module to predict future weights with a multi-modal time series model, which comprises a content-aware attention and adaptive horizon to accommodate asynchronous LLM inference. Extensive experiments show HiVid improves weight prediction accuracy by up to 11.5\% for VOD and 26\% for live streaming over SOTA baselines. Real-world user study validates HiVid boosts streaming QoE correlation by 14.7\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HiVid, a three-stage framework that leverages Large Language Models (LLMs) as human proxies to estimate content-aware saliency weights for adaptive video streaming (both VOD and live).
The framework comprises:
- a Perception module, which assesses sampled frames through a sliding-window prompting strategy to generate local saliency scores and iterative video summaries;
- a Ranking module, which applies a LLM-guided merge sort to globally re-rank frames and eliminate local inconsistencies; 
- a Prediction module, which performs multi-modal time-series forecasting to estimate future saliency weights in live-streaming settings.

Experiments on TVSum, SumMe, and YouTube-8M show improved correlation metrics (up to +11.5% PLCC for VOD and +26% for live streaming) compared to video summarization and highlight detection baselines. A small user study reports better correlation between predicted QoE and human MOS.

### Strengths
- Interesting and timely idea: leveraging LLMs as scalable surrogates for subjective human judgments is an emerging and relevant research direction. Applying this concept to video streaming optimization is novel and has clear practical implications.

- Modular design: the separation into perception, ranking, and prediction modules is conceptually clean and covers both offline and online (live) cases.

- Novel use of LLM reasoning: using an LLM as a semantic comparator in a merge-sort procedure is unconventional and potentially generalizable to other ranking tasks.

### Weaknesses
- Conceptual confusion or inconsistent terminology: The paper repeatedly refers to “video saliency prediction” or "saliency score", but most baselines (e.g., DETR, VASNet, PGL-SUM) are video summarization or highlight detection methods, not visual saliency models. While the "saliency" term is commonly associated in literature with spatial or spatio-temporal saliency maps that highlight visually regions within frames or videos, here it is used to denote subjective importance or priority score assigned to temporal chunks for bitrate allocation in streaming. This broader use of the "saliency" term may be misleading, especially readers familiar with classical video saliency prediction tasks. The paper would benefit from clearly defining "saliency score" early on and explicitly distinguishing its intended meaning from the traditional notion of saliency map. Providing alternative terminology (such as "importance score" or "temporal relevance weights") for the chunk-level scores could improve clarity and avoid confusion. 

- A potential limitation of the proposed ranking module lies in its reliance on LLM as the comparator function within the merge sort algorithm. While innovative, this design assumes that LLM can consistently and reliably perform pairwise comparisons that satisfy the properties required for sorting (e.g. transitivity and anti-symmetry). However, LLM outputs may be inherently variable, subjective, sometimes inconsistent, especially in tasks involving nuanced semantic judgments. There is no formal guarantee that the LLM will always induce a valid total order, which may lead to instability or errors in the final global ranking. The paper lacks a detailed analysis or empirical evidence on the robustness and consistency of the LLM-guided comparison. Addressing this aspect with more thorough evaluation or fallback mechanism would strengthen the approach. 

- Limited scientific contribution for ICLR: the paper primarily presents an application-driven pipeline that leverages existing LLMs as a zero-shot reasoning tool for relevance score assessment, combined with a learning-based forecasting module for live streaming weight prediction. While the system is creative, it is mostly engineering-driven, and the technical novelty in terms of learning methodology remains limited. This raises concerns about whether the contribution advances fundamental learning representation or model innovation, which constitutes the core criteria for ICLR acceptance.
 
- The reported user study involves only 10 participants, which is a small sample size for drawing statistically reliable and generalizable conclusions regarding subjective Quality of Experience (QoE) in video streaming. Such a limited number of users increases the risk that individual preferences and variability disproportionately influence the results, reducing statistical power and limiting meaningful subgroup analyses. Therefore, this small sample size represents a methodological limitation of the study, and expanding the participant pool with a more diverse and larger user base would strengthen the validity and impact of the empirical evaluation.

- Some implementation aspects of the Live Prediction Module are explained in more detail in the appendices (e.g., the use of both CLIP’s image and text encoders, the cross-attention fusion of modalities, and the training with randomized latency Δt to enable variable-length prediction). These details are only briefly mentioned or omitted in the main text (§3.4), making it difficult for readers to fully understand the model’s structure and training procedure without consulting the supplementary material. Providing a more self-contained description in the main paper—especially of how the adaptive decoding works—would substantially improve clarity and reproducibility.

- The current evaluation mainly focuses on hyperparameter variations (e.g., window size, prediction horizon) but does not report experiments that isolate the contribution of the main components—Perception, Ranking, and Live Prediction. A more explicit analysis of how each module affects overall QoE correlation or latency would help clarify the role of individual stages and strengthen the empirical validation. 

- Evaluation of forecasting autonomy:  Since the forecasting module is intended to approximate the LLM’s saliency outputs, it would be informative to assess how well the system performs when operating autonomously—using the predictor without periodic LLM updates.
 Including such an experiment could clarify whether the proposed approach meaningfully reduces dependence on LLM inference and would provide stronger evidence for its real-time applicability.

- Scalability and efficiency not demonstrated at realistic scale: The overhead analysis is performed on a single 201-second video, which does not convincingly demonstrate the scalability of the pipeline for longer or continuous live streams. Evaluating cost, latency, and performance over larger datasets or multi-hour content would strengthen the claims of efficiency and applicability to real-world streaming scenarios.

### Questions
-	How do you ensure that LLM-generated scores are consistent across runs and models?
-	Could the authors clarify the exact nature of the “ground truth” used for correlation computation? Are the reference saliency weights derived from human annotations, subjective MOS labels, or pseudo-labels generated by the LLM?
-	Could the authors report results for a “forecast-only” setting, where the predictor operates without periodic LLM refresh? This would help understand how much the system relies on LLM inference in practice.
-	Given the small sample (10 participants × 10 videos), could the authors provide information on inter-rater consistency or statistical significance of the reported PLCC improvements?
-	The overhead analysis focuses on a 201-second clip. Have the authors explored or estimated how the cost and latency would scale for longer or continuous live streams?
-	Could the authors elaborate on how the LLM-based ranking is implemented in practice? For instance, are the pairwise comparisons deterministic (e.g., with fixed temperature and prompt order), and was any measure of ranking consistency across runs or window pairs evaluated?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes using a multimodal LLM as a substitute for human judgments to predict temporal visual saliency in videos, with adaptive bitrate (ABR) as the primary application.

Because most current multimodal LLMs do not support video inputs and cannot take all video frames as context, the paper evaluates representative frames within local windows. The local-window constraint harms global consistency of saliency scores, which the paper attempts to fix by re-ranking with the LLM used as a comparison function.

The paper also considers a live streaming setting. Since saliency-based ABR there requires predictions for future chunks, they propose a CLIP-based forecasting module.

### Strengths
- Comprehensive pipeline design spanning VOD and live streaming settings
- The proposed method is evaluated against other methods using metrics derived from volunteer human ratings, and it shows improvements.

### Weaknesses
- The proposed method is based on proprietary LLMs. This makes the approach not robust to change in the proprietary LLM's service specifications and operating conditions.

### Questions
- Table 3 reports latency for forecasting. What compute environment is used for this inference, and how intense the runtime compute is? This matters because it is likely to run alongside high-load tasks such as video playback.
- The table on page 15 appears disproportionately large. It might be better to consider resizing the table size.
- The detail of implementations of the forecasting model is hard to understand. This would cause reproducibility problem. Especially, In Figure 9 the legend shows MLP/Attention. Are MLPs also used for the QKV projections? Are CLIP weights updated? Would clarity improve if the CLIP vision encoder and text encoder were depicted separately?
- It seems the positions of Table 3 and Table 4 swapped.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This manuscript makes valuable contributions to content-aware streaming, a critical area for optimizing subjective QoE. The proposed HiVid framework addresses two long-standing pain points in streaming weight generation—prohibitive human annotation costs and poor generalization of vision-saliency models—by innovatively leveraging LLMs as a scalable human proxy. The work is theoretically motivated, methodologically rigorous, and experimentally comprehensive, with clear validation of performance gains for both VOD and live streaming scenarios.

### Strengths
HiVid’s core idea—using LLMs to replace costly human annotation for streaming chunk importance weighting—is both creative and problem-driven. Unlike prior work that relies solely on vision-based saliency or limited human labels, the framework bridges LLMs’ semantic understanding capabilities with streaming’s practical needs, addressing a critical scalability gap. The three modules (perception, ranking, prediction) are tightly aligned to solve non-trivial, scenario-specific challenges:

### Weaknesses
Clarify LLM implementation details: The manuscript does not specify which LLM(s) were used (e.g., GPT-4, open-source models like LLaMA) or how LLM inference latency was managed for live streaming. Adding these details will improve reproducibility and help readers assess HiVid’s computational feasibility for edge deployment.

Elaborate on module ablation studies: While the overall performance gains are reported, a brief summary of ablation experiments (e.g., how removing the LLM-guided merge-sort affects VOD accuracy, or the impact of content-aware attention on live prediction) would reinforce the contribution of each module.

Expand on video diversity: The manuscript does not specify the types of videos evaluated (e.g., sports, movies, animations). Adding a note on whether HiVid generalizes across diverse content genres would further highlight its robustness.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
