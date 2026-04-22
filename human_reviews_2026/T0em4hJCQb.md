# Memory-Augmented Personalized Retrieval for Long-Context Egocentric Video

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Recent advances in AI and wearable devices, such as augmented-reality glasses, have made it possible to augment human memory by retrieving personal experiences in response to natural language queries.  However, existing egocentric video datasets fall short in supporting the personalization and long-context reasoning required for episodic memory retrieval. To address these limitations, we introduce EgoMemory, a benchmark derived from Ego4D, enriched with 165,795 user-specific object annotations over 245 videos from 45 participants, yielding 639 distinct, human-curated, and evaluated queries for rich and individualized episodic memory retrieval. Leveraging this resource, we present EgoRetriever, a novel, training-free retrieval framework that combines Multimodal Large Language Models with reflective Chain-of-Thought prompting. Our approach enables interpretive inference of user intent and generates detailed target video descriptions by leveraging contextualized personal memory for video retrieval. Extensive experiments on EgoMemory, EgoCVR, and EgoLifeQA benchmarks demonstrate that EgoRetriever consistently and substantially outperforms state-of-the-art baselines, highlighting its strong generalizability and practical potential for personalized, long-context egocentric video retrieval.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a new benchmark based on the Ego4D dataset, and proposes a new framework called EgoRetriever for long-context retrieval of egocentric videos. The proposed benchmark, called EgoMemory, enhances the Natural Language Queries (NLQ) benchmark from the Ego4D benchmark suite by grouping the videos from the same participant together and only keep the queries that refers to personal objects requires retrieval across multiple videos. The proposed EgoRetriever is a training-free framework that utilizes MLLMs (with chain-of-thought) and a personal memory bank constructed from the videos to process the user query and reference frame and generate a desired target video description. Experiments on EgoMemory, EgoCVR, and EgoLifeQA show that EgoRetriever outperforms a number of baselines on the egocentric video retrieval task.

### Strengths
- The paper is overall well-written and mostly easy to understand. The proposed framework is described in great detail.
- The paper includes a comprehensive set of ablations studies to validate the effectiveness of each component and design decision in EgoRetriever.
- In addition to the quantitative comparisons, the paper also includes a number of qualitative examples and analysis of Common Failure Cases.

### Weaknesses
- The motivation for the new benchmark is not very clear. The authors fail to provide sufficient evidence why existing benchmarks do not provide a good evaluation for the long-context personalized retrieval task. See "Questions" below for details.
- The candidate set for retrieval in the proposed benchmark does not seem to be very large. According to the statistics around line 302, the average context length of each participant is 33 x 103.82s ≈ 1 hr. Meanwhile, there are many videos that are quite long (≈ 30 min) in the Ego4D dataset for the original NLQ benchmark, not to mention that the context length in EgoLife is a lot longer. As a result, I found it questionable to claim that EgoMemory is a "long context" retrieval benchmark.
- It might be an overclaim that the compared baselines are "state-of-the-art" (as in the abstract). It seems that the baselines in table 1 are completely copied from the EgoCVR paper, and fail to include some more recent works (e.g. GazeNLQ for the Ego4D NLQ Challenge).
- The proposed framework lacks much novelty. All the modules in the proposed EgoRetriever framework have already been validated in a number of recent works (e.g. VideoAgent).
- The presentation of the paper could be improved. The meanings of different colors in Figure 2 is a bit unclear. Both Figure 1 and Figure 2 are a bit busy and it is unclear for the reader where to start or focus on. There is a cycle in the pipeline of Figure 1 which makes it even more confusing (if I understand correctly the pipeline starts with the user query but this is very unclear).

### Questions
- Regarding the motivation for the new benchmark: is the performance of MLLMs a lot lower on these filtered queries than others in the Ego4D NLQ benchmark? Are the ranking / relative performance of MLLMs different on these filtered queries than others in the Ego4D NLQ benchmark?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a method applicable for personalized/composed video retrieval in egocentric videos. Firstly, the paper introduces EgoMemory, a personalized retrieval benchmark curated from Ego4D. Here, the task is to retrieve a specific clip based on a user's history. For this, the paper proposes "EgoRetriever", which is loosely based on prior works on training-free composed visual retrieval (which involve captioning, reasoning, and retrieval in a training-free manner). Results of the proposed method on EgoMemory, and related benchmarks (EgoCVR,  EgoLifeQA) indicate that the proposed method outperforms related baselines/prior works for these tasks.

### Strengths
The task of personalization and managing memory in videos is quite important and has been relatively underexplored, and to that extent the paper is quite interesting. 

The paper is also extremely easy to follow, and provides good ablations that clarify the design choices in the method clearly.

### Weaknesses
Separating Captioning from Reasoning: Older work on Composed Image/Video Retrieval had to separate captioning and the LLM reasoning since the older LLMs did not support direct image/video input. However, with recent open and closed-source (Qwen2.5, GPT-4o, Gemini) models supporting images/videos directly as input, it seems odd to not take advantage of this and separate the visual input from the LLM since there is bound to be loss of information in this process. 

Previous work on VLM personalization: There has been some prior work on personalizing VLMs (to specific objects, users etc.) [a,b], and it would be a good idea for the paper to acknowledge it. 

Limited Amount of Personalization: Owing to the training-free method and separating the visual captioning from the reasoning, I also believe that there's a limited amount of personalization that this framework supports. Unlike [a,b] which are able to learn specific instances (i.e individual people/pets/objects etc. here the representation of these items would be limited to plain descriptions. As an example, if there are two similar dogs in the memory, one would always need a precise description in the user query (which would then have to be converted in LLM generated query) to retrieve the correct video. Previous work that explicitly aims to learn visual tokens for personalized objects would at least in theory be better equipped to deal with this challenge and allow the user to have a special token to directly retrieve them. To that end, while EgoRetriever is definitely a sensible baseline to have for this task that one must compare to, it does not appear to be an especially optimized architecture for this task on first glance. 


[a] Alaluf et al. "MyVLM: Personalizing VLMs for User-Specific Queries", ECCV 2024
[b] Cohen et al. ""This is my unicorn, Fluffy": Personalizing frozen vision-language representations", ECCV 2022

### Questions
[Major]
While I like the overall direction of the paper, I do find a few shortcomings in the specific approach, however, I'd like to clarify these details before making my final decision. Specifically, a) Is the proposed EgoRetriever an ideal architecture that one could design for this task? While incorporating CoT reasoning is a major improvement over prior work, the use of a visual captioner + LLM reasoning + CLIP retrieval appears to be a fairly inefficient pipeline (with obvious shortcomings) that's been carried over for legacy reasons. 
b) If EgoRetrieval is mostly a baseline for this task, is EgoMemory a sufficiently meaningful benchmark on its own for long-video personalization? i.e would it have sufficiently challenging scenarios to benchmark multimodal LLMs specifically fine-tuned for personalization?

[Minor]
While the paper focuses on egocentric tasks in all the experiments, is there anything in the method that's specifically designed for egocentric videos beyond using EgoVLP as the captioner?

### Soundness
3

### Presentation
4

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
This paper introduces EgoMemory, a benchmark for personalized, long-context egocentric video retrieval. The benchmark is derived from the Ego4D dataset but is enriched with new MLLM-generated annotations designed to capture user-specific object details. The authors also propose EgoRetriever, a training-free retrieval framework that reasons on captions of videos using reflective CoT.

### Strengths
- The focus on personalization and long-context reasoning is a timely contribution. Retrieving personal experiences from long-form egocentric video in response to natural language queries is a critical application. 
- The proposed EgoRetriever framework is simple and training-free
- The paper includes a comprehensive ablation on different design choices and caption sources.

### Weaknesses
- The contribution over Ego4D NLQ is unclear. This paper doesn't introduce new video data, but just with new MLLM-generated annotations. As the paper argues, the main weakness of NLQ is that "videos are not grouped by user at retrieval time, ownership or user-linkage cues are not modeled, and queries need not require long-horizon evidence" (l267-268). However, ownership is still missing in the proposed benchmark. The authors are still using the queries in NLQ (but after filtering), which is designed for retrieval within an individual video clip. It's unclear how the proposed benchmark improves NLQ, other than grouping videos of the same user.
- The paper's reliance on a text-based, attribute-driven memory bank is a major weakness. First, rich visual information from video is compressed into discrete text captions or annotations, a process that inherently discards visual nuance. This lossy text is then filtered into a predefined, rigid schema of 12 attributes. This fixed structure is inflexible; if the true distinguishing feature of a personal object is not one of those 12 attributes, that vital information is permanently lost during memory construction. For example, if a user owns two "black bicycles," their text-based attribute entries in the memory bank could be identical, thus system would then be unable to differentiate between them. This ambiguity undermines the very goal of "personalized" retrieval.

### Questions
- The paper criticizes Ego4D NLQ for not modeling "ownership or user-linkage cues". However, the proposed annotations are also unable to have ownership information. How does your benchmark define and evaluate "user-linkage" in a way that is superior to the implicit personalization already present in Ego4D, beyond just object frequency?
- How does the proposed method perform on Ego4D NLQ test set?
- How's the cost (API expenses or GPU time if using open-sourced model) for constructing a memory bank for the tested three datasets? It's nice to see that the proposed method outperforms baselines in EgoVCR and EgoLifeQA, but isn't it much more expensive due to memory bank construction?

### Soundness
2

### Presentation
3

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
The paper targets long-horizon, personalized egocentric retrieval: it formalizes how to build a per-user memory bank from first-person videos and proposes a training-free pipeline where an LLM produces brief reflective descriptions that are matched against video/text embeddings for retrieval; the authors release a benchmark and report results across multiple egocentric datasets, highlighting modularity and easy reproducibility. While the framing is clear and the pipeline simple, the study likely couples the same type of LLM for both labeling and inference (risking preference leakage), relies on relatively small personalized data (limiting generality), offers only high-level privacy claims without a concrete threat model or field-level redaction/DP/on-device alternatives, lacks head-to-head comparisons with general long-video systems (e.g., VideoRAG, LVAgent/VideoAgent), and evaluates an offline-built memory bank without testing the streaming, online updates that real egocentric use cases require.

### Strengths
1. The paper explicitly operationalizes “personalization” for long-horizon egocentric retrieval and builds the EgoMemory benchmark (639 curated queries; 45 participants; 245 videos; ≈91.6% labeled “personal”). The multi-stage pipeline (GPT-4o CoT pre-screen → long-context filtering → manual verification) is clearly described.

2. EgoRetriever cleanly decouples reflective CoT description generation from video–text embedding retrieval, making the framework easy to swap components and reproduce. 

3. Results appear on EgoMemory, EgoCVR, and EgoLifeQA; the appendix claims scripts/configs/prompts and latency breakdowns.

### Weaknesses
1. Although curated, the effective scale (45 users; 245 videos) is small relative to Ego4D’s breadth, risking over-fitting to frequent objects/locales and under-sampling long-tail personal artifacts. Require distributional disclosure (per-user hours, per-class frequency, by-scene/location) and sensitivity curves vs. memory size (≤300 / 1k / 5k+ entries) and candidate-set sizes. Compare against Ego4D EM/NLQ task formulations to show genuine added difficulty. 

2. The paper doesn’t compare against strong, generic long-video systems like VideoRAG (RAG over extremely long videos) and agent-style solvers (e.g., LVAgent, VideoAgent). Without these, it’s unclear whether the proposed pipeline is truly better or just specialized.

3. The paper builds the user’s memory bank offline and queries it later, but egocentric use-cases are inherently streaming: video is coming in continuously, and the system must update memory online while answering in near real time. Current streaming literature shows that maintaining long-horizon context with low latency and on-the-fly memory building is key in first-person settings (e.g., online/streaming QA, online action detection/segmentation, and active/online egocentric memory). Your setup doesn’t test any of this.

### Questions
1. How to handle streaming settings?

2. What is the main difference between egoretriever and general video RAG framework?

### Soundness
3

### Presentation
3

### Contribution
3
