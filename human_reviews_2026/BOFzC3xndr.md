# Video-in-the-Loop: Span-Grounded Long Video QA with Interleaved Reasoning

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
We present $\textit{Video-in-the-Loop}$ (ViTL), a two-stage long-video QA framework that preserves a fixed token budget by first $\textit{localizing}$ question-relevant interval(s) with a low-fps skim and then $\textit{answering}$ via span-aware reallocation of visual tokens at higher effective frame rate, emitting an interleaved output with both spans and the final option for direct attribution. We also introduce $\textit{VGrounding-QA}$, which converts description based event graphs into $\textit{span-grounded}$ multiple-choice QA by pairing each question with $\textit{ground-truth}$ time span(s) and related reasoning. ViTL is trained end-to-end with an interleaved group-relative objective that couples temporal IoU for localization with answer correctness, allowing credit to flow from answers back to spans without increasing compute. Under fixed token budgets, ViTL attains up to 8.6\% with 50\% less frame input on long-video QA and temporal grounding (e.g., Charades-STA, ActivityNet-Captions) and ablations show that span-aware token reallocation consistently surpasses uniform sampling. Together, $\textit{VGrounding-QA}$ and ViTL provide an interpretable, compute-efficient recipe for scalable long-video QA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ViTL, a two-stage framework designed to improve the efficiency and accuracy of long VideoQA under fixed computational budgets compared to uniform sampling ViTL tackles this with a "skim-then-zoom" approach, which includes a grounding stage (full video at low frame rate) and answering stage (zooming at high fps). To train this model, a synthetic annotation pipeline is used to create a span-grounded multi-choice QA dataset, where the distractors are picked from other events in the same video. The model is trained with GRPO, a RL technique that couples temporal localization quality and answer correctness. ViTL achieves significant improvements in temporal grounding and some gains in long video QA benchmarks at fixed token budget.

### Strengths
- Intuitive and interpretable framework with dedicated training strategy involving data generation and RL 
- Strong temporal grounding results

### Weaknesses
- There is a big discrepancy in the gains in between tasks. The big gains in moment retrieval could also be due to adaptation to the output moment distribution. It would be good to see temporal grounding results on debiased subsets as in https://arxiv.org/abs/2111.04321. It would also be good to see the performance on QVHighlights for moment retrieval.
- The framework largely relies on multi-choice MCQA, it is unclear if it would be effective on open-ended QA (e.g. EgoTempo).

### Questions
- How does the model perform when retrieving a single span vs multi span? (does it actually benefit from multiple spans?)
- How does the model perform with GT spans or random spans, so to have upper and lower bounds of the framework?
- Note: non maximal numbers are bolded e.g. in tab 5 for R@0.3 
- Have non parametric frame selection been considered as baseline e.g. https://arxiv.org/abs/2301.11507 ?
- How much of the performance comes from two stage answering? E.g. it would be interesting to see a baseline that tries to answer directly in a first stage, and in the second stage critiques the answer and refines it if needed (without span localization).

### Soundness
2

### Presentation
3

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
This paper proposes ViTL (Video-In-The-Loop) a two stage framework for long video question answering, that preserves a fixed token budget while reallocating tokens where it matters (useful evidence to answer the question). ViTL works in two stages: Stage-1 performs a low frame rate skim over the entire video and predicts one or more temporal spans that are important or relevant for the question. Stage-2: trims the video using the spans predicted in Stage 1, zooms into those spans at a higher frame rate and use that to give an answer. The training process is done using GRPO, which uses a compose reward that couples temporal localization and answer correctness. The authors also introduce VGrounding-QA which is a training set where each QA item is paired with time spans (which are necessary/useful to answer the questions). ViTL improves accuracy on LongVideoBench and LVBench an shows strong zero-shot temporal video grounding on Charades-STA and ActivityNet-Captions.

### Strengths
Paper Strengths:
1. The paper proposes framing long-video QA using token reallocation to specific evidence with an interleaved span+answer output.
2. ViTL makes use of a GRPO objective that jointly rewards temporal IoU and answer correctness, which is useful to teach the network what parts of the video are useful.
3. The creation of VGrounding-QA dataset is a useful contribution, which couples temporal spans with answer supervision, which is missed in existing long-video datasets according to Table 2.

### Weaknesses
Paper Weaknesses:
1. Personally I think the paper lacks examples to visually see what happens at each stage, I believe that Figure 1 should be redesigned with more details, or create a different image where the proposal is clearly stated. By only seeing Figure 1, which is the main Figure of the paper, is difficult to grasp the proposed idea, the left part I think is not really useful and creates a lot of confusion, and the right part lacks some labels and separation in some way of stage 1 and 2 (I think that the figure is also lacking some details about the entire procedure). Additionally referring a bit more to this figure when the explanation is done in the paper it would help the reader to get the ideas in a better way, this lack of clarity of the main Figure made it really confusing sometimes to really understand some parts in my opinion. For example the paper describes that stage 1 uses a grounding query distilled form the question, however this is not shown in the main figure. This one also contains boxes with different colors and the lack of labels makes it difficult to understand.
2. Table 3 shows the model's performance on LongVideoQA benchmarks, specifically LongVideoBench, LVBench and MLVU M-Avg, however for the latter the performance is not given for the proposed model ViTL. Please report this performance. On the other hand, when the authors compare in LVBench ViTL with open-sourced baselines is done with only two model families VideoLLaMa and Qwen2.5 (from 7 in the table) which I think is not sufficient to claim a superiority of the proposed model.
3. Table 8 is showing performances of models of different sizes without any sorting in the table, without any separation of close and open source models, the authors use Bold to indicate their model's results but this mix of models makes it complicated to compare the proposed model to other baselines. Additionally, it lacks some references.
4. VGrounding-QA is a good dataset, but because it was created using different models, a human audit of timestamps faithfulness is necessary and this would increase the dataset's reliability.

### Questions
Please refer to weaknesses for my questions and doubts. Overall, I think the paper brings good contributions for long video QA, however I believe that the paper still needs work before publishing, mainly in the figures and tables as described above, which will help to improve the overall understanding of the proposed ideas. However I want to see the clarifications to my questions and concerns and sorry if I misunderstood something, I look forward to see the author's responses. Currently, I'm giving borderline reject, after rebuttal I will revise my decision.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces VGrounding-QA, a framework that constructs a training dataset based on an event-centric knowledge graph. Long videos are divided into uniformly sized short segments, each paired with corresponding descriptions. These segment-description pairs are then used to train multimodal large language models (MLLMs) for temporal grounding within videos. The trained ViTL model outperforms existing multimodal large language models (MLLMs) on both long video question answering and temporal grounding benchmarks, demonstrating its effectiveness in handling extended temporal contexts.

### Strengths
- The paper proposes a novel training dataset constructed from an event-based knowledge graph, providing structured and semantically rich supervision for long video temporal grounding.
- It introduces a two-stage GRPO framework that leverages grounded spans, improving the alignment between language queries and temporally localized video segments.
- The proposed method achieves notable performance improvements over existing MLLMs on both long video question answering and temporal grounding benchmarks.

### Weaknesses
- It is unclear which dataset was used for training. 
- The set of baselines appears incomplete. For instance, it is unclear whether models like LLaVA-Video or InternVL3 utilize a greater number of video frames, which could impact performance. The comparison table should also include computational metrics—such as the number of input frames, inference time, and memory usage—for each method to ensure a fair and comprehensive evaluation.
- How does the proposed data construction pipeline differ from existing approaches in weakly supervised video learning or synthetic data generation? A more explicit comparison would help clarify the novelty and advantages of the method.
- Since the dataset is specifically designed for temporal grounding and question answering, the improved performance on these tasks is expected. However, the generalization ability of the proposed method to other video understanding tasks remains unclear and is not thoroughly evaluated.

### Questions
- How does the proposed method compare in terms of computational efficiency? It would be helpful to report training and inference time, as well as memory usage, to better understand the method’s practicality and scalability.
- Compared to existing approaches, which component or design choice of the proposed method contributes most significantly to the observed performance improvements? A detailed analysis or ablation would help clarify this.

### Soundness
3

### Presentation
2

### Contribution
2
