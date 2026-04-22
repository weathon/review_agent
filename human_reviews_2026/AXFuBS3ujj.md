# CLaMR: Contextualized Late-Interaction for Multimodal Content Retrieval

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Online video content is richly multimodal: a single video might blend vision, speech, ambient audio, and on-screen text. Conventional retrieval systems typically treat these modalities as independent retrieval sources, which can lead to noisy and subpar results.
In this work, we explore multimodal video content retrieval, where relevance can be scored from a single modality or jointly across multiple modalities. Consequently, an effective retriever must dynamically determine which modality (or set of modalities) best address a given query. We introduce CLaMR, a multimodal, late-interaction retriever that jointly indexes four modalities: video frames, transcribed speech, on-screen text, and metadata. CLaMR jointly encodes all modalities within a unified multimodal backbone for improved contextualization and is trained to enhance dynamic modality selection via two key innovations. First, to overcome the lack of suitable training data, we introduce MultiVent 2.0++, a large-scale synthetic dataset built on MultiVent 2.0 (a collection of event-centric videos in various languages paired with English queries) with modality-targeted queries to teach modality selection. Next, we propose a modality-aware contrastive loss that trains the model on both a standard contrastive objective and an objective for learning correct modality usage. On the test sets of MultiVent 2.0++ and MSRVTT, we observe that conventional aggregation strategies, such as averaging similarities for baseline retrievers, often degrade performance by introducing noise from irrelevant modalities. In contrast, CLaMR consistently outperforms existing retrievers: on MultiVent 2.0++, CLaMR improves nDCG@10 by 25.6 points over the best-performing single-modality retriever and by 35.4 points over the best-performing multi-modality retriever. We illustrate the downstream utility of CLaMR with experiments on long-video QA, where it improves performance by 3.50% over LanguageBind on Video-MME and 1.42% over dense frame sampling on LongVideoBench.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes CLAMR, a multimodal video retriever that jointly indexes video frames, speech, text, and metadata, dynamically selecting relevant modalities for queries. Unlike conventional systems that treat modalities independently, CLAMR uses a unified backbone for better contextualization. Trained on MULTIVENT 2.0++ with modality-targeted queries and a modality-aware contrastive loss, it outperforms existing retrievers in video content retrieval and long-video question answering.

### Strengths
1. The methodology of the paper is clearly written and easy to follow.

2. CLAMR achieves state-of-the-art performance on MULTIVENT 2.0++ and MSR-VTT.

3. Ablation studies demonstrate the contributions of different modalities to retrieval, showing that multimodal fusion significantly enhances performance.

4. The proposed contextualized late-interaction approach is simple yet effective, maintaining the efficiency of dual-encoder models while outperforming the averaging of modality-specific retrieval scores.

### Weaknesses
1. I don’t find major weaknesses in this paper.

### Questions
What is the influence of Qwen model size on multimodal retrieval?

### Soundness
3

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
4

### Summary
This paper addresses the problem of fusing information from various modalities for video retrieval. The authors propose CLAMR, a framework that applies a late-interaction mechanism, inspired by text-retrieval models like ColBERT, to a multimodal context encompassing video, audio, OCR, and metadata. The core idea is to jointly encode all modalities within a single vision-language model (VLM) to create contextualized representations, and then perform token-level similarity matching. To train the model to focus on relevant modalities, the paper introduces a modality-aware contrastive loss and a new large-scale synthetic dataset, MULTIVENT 2.0++, created using an LLM to generate modality-specific queries. The experiments show that CLAMR achieves strong performance, particularly on this new synthetic benchmark, outperforming several unimodal and multimodal baselines.

### Strengths
1. The paper tackles a well-recognized and challenging problem in multimodal retrieval—how to effectively combine signals from diverse and potentially noisy sources without performance degradation.

2. The application of late-interaction mechanisms from the text domain to a complex multimodal video scenario is a logical and interesting direction. It provides an alternative to more common early-fusion or simple late-fusion (score averaging) techniques.

3. The authors make a notable effort to address the scarcity of suitable training data by generating a large-scale synthetic dataset with modality-targeted queries. This resource could potentially benefit future research in the area.

### Weaknesses
1. The core technical contribution can be viewed as an application of the existing ColBERT architecture to a multimodal setting using a standard VLM backbone. While the engineering is non-trivial, the conceptual novelty is somewhat incremental, as it primarily combines and adapts existing components rather than introducing a fundamentally new retrieval paradigm.

2. The most impressive results (e.g., +25.6 nDCG@10) are reported on the authors' own synthetic dataset, MULTIVENT 2.0++. This raises significant concerns about evaluation validity. The model might be overfitting to the specific patterns, vocabulary, and artifacts of the LLM used for query generation, rather than learning a truly generalizable retrieval capability. The performance gains on the established, human-annotated MSR-VTT benchmark are far more modest, which may indicate that the practical impact on real-world queries is less significant than claimed.

3. The paper proposes a modality-aware loss (LImw) for training, which encourages the model to identify the single most relevant modality. However, at inference, a different, holistic scoring function (LIcontext) that aggregates scores across all modalities is used. This disconnect between the training objective and the inference procedure weakens the central claim of "dynamic modality selection." It's unclear whether the model is truly learning to select modalities or if the performance gain is simply a byproduct of a more complex training objective that acts as a form of regularization.

4.The late-interaction approach introduces significant computational and memory overhead during the offline indexing phase compared to standard dual-encoder models. Given that the performance improvement on MSR-VTT is not as dramatic as on the synthetic dataset, the practical utility of CLAMR is questionable for applications where efficiency is a key concern.

### Questions
1. How can you ensure that the substantial performance gains on MULTIVENT 2.0++ are not primarily due to the model learning the specific artifacts of the Gemma-3-27b-it query generator? Have you considered a cross-validation experiment where you train on queries generated by one LLM and test on queries generated by a completely different LLM?


2. Could you provide a clearer justification for the discrepancy between the training objective and the inference objective? If the goal is to teach the model modality selection, why not use a scoring function at inference that also reflects this selection process? Does this design choice not suggest that the model isn't actually performing explicit modality selection at test time?

3. The paper's baselines for multimodal fusion include simple averaging and a "hard" router. A potentially stronger and more relevant baseline would be a "soft-routing" or attention-based mechanism that learns to dynamically weight the similarity scores from different modalities based on the query. How would you expect CLAMR to perform against such a baseline?

4. Given the high indexing cost of late-interaction and the more moderate performance gains on MSR-VTT, what is the compelling practical argument for deploying CLAMR over a simpler, highly optimized dual-encoder model that is fine-tuned on the same comprehensive data?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a contextualized late-interaction retriever for multimodal video content retrieval, designed to address limitations of conventional systems that treat modalities as independent or use naive fusion. The proposed model jointly encodes four modalities—video frames, transcribed speech, on-screen text, and metadata—via a unified vision-language backbone (or omni-model for raw audio) to enhance contextualization. Experiments show it outperforms single/multimodal baselines.

### Strengths
1. Unlike baselines that encode modalities separately, the proposed model uses a unified backbone for cross-modal contextualization.

2. The proposed MULTIVENT 2.0++ fills the gap of modality-specific training data, supporting effective modality selection learning.

3. The proposed model consistently outperforms baselines across MULTIVENT 2.0++, MSR-VTT.

### Weaknesses
1. The performance drops noticeably when vision is the sole relied-on modality, showing weaker handling of visual-only signals. Why is this? Why is "vision the least informative," as stated in Line 430? Could this be unfriendly to most scenarios (given that vision is the most common and readily available modality)?

2. Primary evaluations focus on MULTIVENT 2.0++ and MSRVTT; tests on other multimodal benchmarks (e.g., MSVD, DiDeMo, ActivityNet) are limited, reducing generalizability evidence.

3. Why was the ablation experiment performed on MULTIVENT 2.0 instead of MULTIVENT 2.0++? Furthermore, ablation experiments should also be performed on other datasets.

### Questions
Please refer to the weakness.

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
This paper presents CLAMR, a contextualized late-interaction retriever for multimodal video content retrieval, and makes three core contributions. Proposing a unified vision-language backbone that jointly encodes four modalities to enhance cross-modal contextualization, addressing the limitations of independent modality encoding in conventional methods. Introducing MULTIVENT 2.0++, a large-scale synthetic dataset with 371k modality-targeted queries, solving the scarcity of fine-grained modality-specific training data for multimodal retrieval.

### Strengths
The joint encoding of multiple modalities and modality-wise late-interaction balance fine-grained token-level matching and computational efficiency, filling the gap of late-interaction’s underutilization in multimodal video retrieval.
Evaluates on multiple benchmarks and conducts extensive ablations, ensuring the reliability of conclusions.

### Weaknesses
Focuses only on four modalities and does not explore other critical modalities in video content.
 Evaluations are primarily based on event-centric and general video datasets.
While the joint encoding backbone aims to align modalities, the paper lacks analysis of alignment failures in temporally or semantically misaligned video content. 
The model’s performance heavily relies on high-quality ASR transcripts and OCR text, making it vulnerable to low-resource scenarios where these modalities are noisy or unavailable. 
The paper does not provide interpretability into how the modality-aware mechanism selects the “most relevant” modality for a given query.

### Questions
How does the modality-wise late-interaction mechanism (LIₘᵥ) specifically resolve conflicts when multiple modalities contain relevant information? The paper emphasizes single-modality targeting but lacks analysis of cross-modal corroboration scenarios.
What are the root causes of the lower performance on vision-targeted queries? 
For videos longer than 60 minutes, how does CLAMR’s frame sampling and retrieval efficiency degrade? Is there a strategy to optimize long-sequence processing?

### Soundness
3

### Presentation
3

### Contribution
3
