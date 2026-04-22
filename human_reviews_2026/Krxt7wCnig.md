# Long-range Modeling and Processing of Multimodal Event Sequences

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
Temporal point processes (TPPs) have emerged as powerful tools for modeling asynchronous event sequences. While recent advances have extended TPPs to handle textual information, existing approaches are limited in their ability to generate rich, multimodal content and reason about event dynamics. A key challenge is that incorporating multimodal data dramatically increases sequence length, hindering the ability of attention-based models to generate coherent, long-form textual descriptions that require long-range understanding. In this paper, we propose a novel framework that extends LLM-based TPPs to the visual modality, positioning text generation as a core capability alongside time and type prediction. Our approach addresses the long-context problem through an adaptive sequence compression mechanism based on temporal similarity, which reduces sequence length while preserving essential patterns. 
We employ a two-stage paradigm of pre-training on compressed sequences followed by supervised fine-tuning for downstream tasks. Extensive experiments, including on the challenging DanmakuTPP-QA benchmark, demonstrate that our method outperforms state-of-the-art baselines in both predictive accuracy and the quality of its generated textual analyses.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes a unified encoding paradigm that integrates time, event type, text, and image to achieve multimodal TPP modeling. It introduces an adaptive temporal compression mechanism, where adjacent events with time differences below a defined threshold are replaced by a special token—representing temporally similar events), enabling contextual compression and enhancing the model’s long-range sequence modeling capability.

### Strengths
1. The target task has high practical value and represents an important direction in the field of long-term reasoning and multimodal generation.

2. The proposed method in this paper is concise and effective. The proposed time-similarity-based compression mechanism can significantly reduce sequence length.

3. The experimental design of the paper is very comprehensive, with particularly thorough ablation studies.

### Weaknesses
The temporal compression approach proposed in the paper may have certain limitations in some scenarios, as it only addresses the compression of the number of events but not the compression within individual events.

### Questions
1. When events are compressed, will the compression be applied consecutively? For example, if there are three events A, B, and C, after compressing ⟨A, B⟩, will it further compress to ⟨A, B, C⟩? If so, this should be clarified in the paper.

2. When a particular moment is controversial — for instance, when some comments (danmaku) are positive while others are negative — does the current method only consider the earliest event? This could indicate a potential limitation.

3. I suggest adding a comparative experiment that randomly drops some events, representing the most naive form of compression, and comparing it against the proposed adaptive compression method. This would make the results more convincing.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a unified framework, multimodal temporal point processes (MM-TPP), that extends Language-TPP to predict event time and type while generating rich text conditioned on visual, textual, and temporal inputs. The authors further introduce a compression mechanism to reduce sequence length. Finally, they introduce a benchmark TAXI_PRO to evaluate their proposed method.

### Strengths
1. The paper is well organized.  

2. The topic of using LLMs for multimodal event sequences is worth exploring in the research community. This paper targets this important problem.

3. The paper provides the code to help reviewers better understand of the proposed method.

### Weaknesses
There are some concerns and questions about this paper:

1.	In Section 4.3, the authors mention that temporal similarity between events can be calculated to reduce sequence length. However, how is this similarity calculated? The authors do not seem to mention the calculation method.

2.	In video understanding, I learned that only 64 frames are needed to help the model understand what is happening in the video. So, for the video input mentioned in the paper, how many video frames are actually input into the model?

3.	To reduce sequence length, did the authors try keeping the input sequence length constant while using token merging or pruning methods? How does this method perform compared to the methods mentioned in the paper?

4.	Among the comparison methods listed in Table 1, only Language-TPP was published in 2025, while the others were published in 2022 or earlier. Could the authors update the comparison methods or include more recent methods published in 2024 or 2025 for a fairer comparison?

**Please note that I am not very familiar with this research area. I would suggest that the ACs place greater weight on the comments from the other reviewers.**

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes MM-TPP, a multi-modal, generative framework for temporal point processes that models multi-modal event data and predicts the next event’s time/type while generating text. To cope with very long sequences induced by multimodal tokenization, the authors introduce an adaptive temporal-similarity compression that replaces runs of events with similar inter-arrival times by a special `<|similar_event|>` token, enabling longer effective context within a fixed window. The system is built on Qwen2.5-VL-3B with a two-stage training recipe including continued pre-training and SFT. Experiments on DanmakuTPP and a newly proposed TAXI-PRO dataset show improved performance over strong TPP baselines and Language-TPP.

### Strengths
1. **First multi-modal TPP dataset.** This article proposed TAXI-PRO, which may be a useful testbed for future work.
2. **Simplicity and effectiveness.** The `<|similar_event|>` token is easy to implement and effectively increases the total event counts in a single context window.

### Weaknesses
1. **Evaluation breadth.** Only DanmakuTPP is chosen as existing benchmarks to be evaluated, which could be biased.
2. **Lack to efficient MLLM baselines.** The author should include token pruning baselines as baseline methods (such as [1]), as they are closely related to the addressed topic.

[1] An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models (ECCV 2024)

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper extends TPPs to the visual domain for modeling event sequences, avoiding traditional issues in modeling multimodal data via TPPs such as long sequence length. The authors implement this extension by introducing an adaptive sequence compression mechanism that retains key patterns, and pretrains on compressed sequences and then fine-tunes for specific tasks. The authors evaluate their method on the DanmakuTPPQA benchmark.

### Strengths
- The extensive documentation of details of the experiment setup are appreciated.

- The experiments are extensive. The authors compare against a range of TPP approaches, and select fair experiment details for an appropriate comparison. Selected metrics seem appropriate.

- Standard deviations for experiment results are presented in Table 1 alongside key metrics.

- The paper is well-written and very detailed. I do not have many questions after reading through the experiments section.

- The ablation study included in Sec. 5.5 is very helpful for understanding details of the proposed method and how it relates to other approaches.

### Weaknesses
- Given that only two benchmarks were selected for evaluation, it would be helpful to add a sentence or two explaining why these datasets were selected and why more are not needed for a fair evaluation.

- Figure 1 is difficult to read due to the font choices and text size.

- It would be interesting to consider other backbone architectures beyond Qwen-2.5 and its different sizes.

### Questions
See weaknesses. Generally speaking, the paper is very clear.

### Soundness
4

### Presentation
4

### Contribution
3
