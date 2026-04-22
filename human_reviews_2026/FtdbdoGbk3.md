# Memento: Toward an All-Day Proactive Assistant for Ultra-Long Streaming Video

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Multimodal large language models have demonstrated impressive capabilities in visual-language understanding, particularly in offline video tasks. More recently, the emergence of online video modeling has introduced early forms of active interaction. However, existing models, typically limited to tens of minutes, are not yet capable of all-day proactive understanding over ultra-long video streams. They struggle to maintain long-term context online, as they suffer from token accumulation and lack scalable memory mechanisms. These limitations hinder critical tasks such as reminding users that medication was taken hours earlier—an ability that exemplifies the shift from reactive to memory-oriented assistants with long-term reasoning. To bridge this gap, we present Memento, the first proactive vision-language framework for ultra-long streaming video. To avoid token growth and support scalable long-duration understanding, we introduce Dynamic Memory and Query-related Memory Selection, enabling sparse memory retention and efficient retrieval. To address the training challenges of memory-based modeling, we propose Step-Aware Memory Attention, which aligns memory access with temporal steps for stable supervision. To support both training and evaluation of active, long-term behavior, we construct Memento-54K and MementoBench, a dataset-benchmark suite covering diverse tasks on text, object, and action across video streams up to 7 hours. Experiments demonstrate that Memento achieves superior performance, paving the way toward reliable all-day proactive video assistants.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Memento-54K and MementoBench, a dataset and benchmark for proactive long egocentric video understanding. The paper also presents Memento, a training-based framework for streaming proactive video understanding with multimodal large language models. Memento enables ultra long video understanding by devising a dynamic memory mechanism and an efficient memory retrieval method. Qualitative and quantitative evaluations show that Memento greatly outperforms existing offline or online multimodal large language models in terms of proactive video understanding tasks.

### Strengths
- The paper is mostly well written, clear and easy to understand. The problem is well-motivated with clear motivating examples in the beginning demonstrating the failure modes of existing methods.
- The paper presents a large amount of work, including a framework, a dataset and a benchmark, which are all good contributions to the field.
- The paper presents a training-based framework. Despite being more expensive, training-based methods have more potential than training-free methods, and are more technically challenging to get to work. The paper presents many details about their model training strategy such as the step-aware memory attention (SAMA) that could be used in other streaming understanding applications.
- The paper includes detailed ablation studies that provides useful insights for configuration the model and training.

### Weaknesses
- The paper only presents a very limited number of baselines in the experiments, mostly just VideoLLM-online, which has been shown to be a fairly weak model even when compared to others in the online setting. I would encourage the authors to add more simple baselines with existing offline video models. More specifically, if would be helpful to show quantitative results for the "long video LLMs" row in figure 1 and further demonstrate the inefficacy of existing models regardless of prompting strategies.
- Although the paper devises a dynamic memory mechanism to prevent undesirable accumulation of visual tokens over time, it still requires repeated filtering of the memory for each user query (line 182). Hence it is a valid concern that the model might not scale very well with a much bigger number of queries (which is completely conceivable in real-world applications). I hope the authors to provide more discussion on this limitation.

### Questions
- Are the quality of the LLM-generated content for the dataset and benchmark verified by humans? Given that LLM-generated content can still be unreliable, it is important and common practice to manually inspect and verify the QA-pairs or detected objects/text. It would also be helpful to provide some examples in the appendix and discuss some failure cases, if any.

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
3

### Summary
This paper introduces Memento, a framework that aims to endow video-language models with persistent long-term memory for streaming video understanding. Instead of treating each clip independently, Memento incrementally updates a memory bank of “key events” or “summaries,” enabling the model to maintain temporal continuity and answer long-term queries across extended video streams.

### Strengths
* The paper tackles an important and timely problem, i.e., enabling continuous, long-duration video understanding rather than static or episodic reasoning.
* The conceptual framing (“all-day memory”) is inspiring and forward-looking.
* The hierarchical structure separating short-term vs. long-term memory mirrors cognitive and neuroscience-inspired models, offering conceptual richness.
* Memento appears flexible enough to plug into existing Video-LLMs, potentially generalizing beyond the specific tested backbone (LLaMA 3.1).
* The paper presents detailed ablation studies.

### Weaknesses
* It’s not fully evident why such a complex hierarchical design is necessary. 
* The overall design reads as over-engineered for the stated goal. Similar effects might be achievable with simpler keyframe/event summarization or query-conditioned caching.
* Experiments are restricted mainly to OVO, with no results on other benchmarks (e.g., StreamingBench, EgoSchema, LongVideoBench, etc.), making the generality claims weak.
* Comparisons are missing with more baselines (for instance non-streaming methods).
* No offline results or cross-dataset evaluations are provided. It’s unclear if Memento benefits standard (non-streaming) settings.

### Questions
* Why are results shown only on VideoLLM-Online? Have you tested on other long-video or streaming benchmarks such as StreamingBench, LVBench, or EgoSchema?
* Does Memento improve performance in non-streaming (offline) scenarios as well, or is its advantage restricted to streaming conditions?
* The paper emphasizes long-term efficiency — could you provide actual latency, memory usage, or computational cost metrics to substantiate this claim?
* How does Memento compare to other recent memory-based systems under similar settings?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a vision-language framework that can handle proactive-understanding in a long (multi-hour) streaming video setting. It addressed the memory exploision issues in prior video-LLMs and related work videoLLM-online. 

The method mainly provides three modules as the technical contributions: 
* It used a dynamic memory module to handle redundant frames, using the heuristic attention based mechnism to compare the similarity of new frames to history frames. 
* It introduce a query-related memory selection mechanism to retrieve key frame memory related to current query. 
* To handle the frame alignment with the presence of two modules above, it introduces a step-aware memory attention mask to prevent access to expired frames during training. 

To evaluate, the paper presents a Memento-54K, a benchmark dataset built on top of existing ego4D dataset, and they generate QA pairs categorized into action, object and text modalities. The ablations confirm the effect of the method. Compared to prior seminar work videoLLM-online (trained similarly), the proposed method can effectively handle multi-hour long inference, while videoLLM-online will be run into OOM after roughly 20 mins in the same setting.

### Strengths
Overall I found this paper is well-motivated, technically sound, and validated well with the experiments. 
* The proposed three-modules handles the redundant frames, query-memory attention, while still being frame-aligned as videoLLM-online. The three modules compound to each other well. I believe such mechanism can also be extended to other long-video understanding problems. 
* The performance on long-video understanding is validated well, on the new proposed dataset, compared in an ablation setting and with prior work. The method in particular highlights its strong recall, which is an impressive improvement.

### Weaknesses
* It is not entirely clear how the method will perform in a non-templated QA scenario (if not categorized in the action, object, text query). I am particular interested in that how will the query-related memory selection module generalized in non-templated scenario. It may or may not generalize depends on the complex of the query, which may provide some clues whether current simple cross-attention heuristics is sufficient to handle a more complex setting. 

* A small writing issue, the paper use big "M" and small "m" to describe the accumulated historical attentions frames and individual frames if I understand correctly. It will be more easily understood if can be clarified in the writing.

### Questions
Mostly the first question I left in the weakness. I think even some simple experiments to evaluate the current model with existing trained checkpoints on a few small cross-domain datasets will be good to give some clues. I would expect it may degenerate, but I am interested to see how reliable it is to handle things if query is getting out of distributions.

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
3

### Summary
The paper introduces Memento, a vision-language framework for proactive assistance on ultra-long streaming videos. It proposes a Dynamic Memory (DM) mechanism to fuse redundant frames and a Query-related Memory Selection (QMS) module for efficient retrieval. This paper also introduces Memento-54k and MementoBench, a new benchmark derived from Ego4D for training and evaluating long-term proactive tasks.

### Strengths
- The paper addresses a highly significant and practical problem on long-term online video understanding, which has impact on real-life applications.
- The paper is well-written with high-quality figures, making it easy to follow.
- The new benchmark can benefit research in ultra-long streaming videos

### Weaknesses
- The only baseline compared is VideoLLM-online. Figure5 shows this model is fundamentally unequipped for the task, hitting OOM at 25 minutes. To strengthen the experiments, the model needs to be compared with other sota models, such as VideoLLM-MoD and LION-FS. To provide a comprehensive evaluation, it would be helpful to validate the proposed method on the Ego4D Narration task that is the same as used in VideoLLM-online.
- Some evaluation metrics are ambiguous. (i) Score is a numeric value from 1 to 10 assigned by gpt3.5. It's unclear if it is a calibrated metric and how good gpt3.5's judegement is. (ii) Redundancy is "the proportion of model responses outside the time window in TimRecall". It's unclear if it's a delay of responses or hallucinations.
- The Dynamic Memory mechanism appears very sensitive to hyperparameters. For example, Figure 6 and Table 4 show that changing $\epsilon$ from 0.7 to 0.8 causes the memory bank to grow 10x larger for only a marginal performance gain. This suggests that the model needs to be carefully tuned when applying to new domains.

### Questions
- To solidify SAMA as a key contribution, did the authors attempt to train the DM module without SAMA (i.e., using standard causal attention)? What was the result?
- In fugure6, why does timerecall drop when increasing r_qms from 50% to above 90%? Does this  suggest the LLM gets "distracted" by too much context?
- How to choose the best model, considering the tradeoff between timerecall, score and redundancy?
- How is u in eq3 chosen?

### Soundness
2

### Presentation
3

### Contribution
3
