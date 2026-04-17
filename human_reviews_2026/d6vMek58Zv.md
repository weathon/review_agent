# Measure Twice, Cut Once: A Semantic-Oriented Approach to Video Temporal Localization with Video LLMs

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Temporally localizing user-queried events through natural language is a crucial capability for video models. Recent methods predominantly adapt video LLMs to generate event boundary timestamps for temporal localization tasks, which struggle to leverage LLMs' pre-trained semantic understanding capabilities due to the uninformative nature of timestamp outputs. In this work, we explore a timestamp-free, semantic-oriented framework that fine-tunes video LLMs using two generative learning tasks and one discriminative learning task. We first introduce a structural token generation task that enables the video LLM to recognize the temporal structure of input videos based on the input query. Through this task, the video LLM generates a sequence of special tokens, called structural tokens, which partition the video into consecutive segments and categorize them as either target events or background transitions. To enhance precise recognition of event segments, we further propose a query-focused captioning task that enables the video LLM to extract fine-grained event semantics that can be effectively utilized by the structural tokens. Finally, we introduce a structural token grounding module driven by contrastive learning to associate each structural token with its corresponding video segment, achieving holistic temporal segmentation of the input video and readily yielding the target event segments for localization. Extensive experiments across diverse temporal localization tasks demonstrate that our proposed framework, MeCo, consistently outperforms methods relying on boundary timestamp generation, highlighting the potential of a semantic-driven approach for temporal localization with video LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes MeCo, a semantic-oriented approach to temporal localization with Video LLMs. Instead of directly predicting boundary timestamps, MeCo first produces structural tokens, which contain an event token (<ent>) and a transition token (<tst>), to segment a video. Before each <ent>, the model inserts an event caption to enrich semantics. During training, each structural token is grounded to its corresponding temporal segment via a contrastive objective. Experiments across grounding and dense captioning tasks show consistent gains over timestamp-centric baselines.

### Strengths
1. **Effective grounding objective.** The contrastive grounding loss (Eq. 4) is well-aligned with temporal localization and empirically appears to improve boundary precision.
2. **Strong zero-shot generalization.** Trained on E.T.-Instruct data, MeCo reports competitive or superior zero-shot performance across grounding and dense captioning, indicating good transfer without task-specific fine-tuning.
3. **Ablations support design choices.** The ablation study isolates the contributions of structural tokens and event captions, showing both components are necessary for the full effect.

### Weaknesses
1. **Missing implementation details.** The paper presents an inference approach but does not fully specify how timestamps are derived from selected frames or how invalid sequences are handled (e.g., if the index chosen for the first <ent> exceeds that of the second <ent>, whether re-ordering or smoothing is applied).
2. **Limited comparisons to recent methods.** The paper primarily compares with earlier methods (e.g., TimeChat, E.T.Chat). ** More recent temporal grounding approaches, such as DisTime, LLaVA-MR, UniVTG, or Mr.BLIP, are not included. On Charades-STA in particular, several of these report strong results. The paper should either compare these approaches or discuss why MeCo lags behind them.
3. **Title–method mismatch.** The metaphorical title “Measure Twice, Cut Once” does not clearly convey the core technical contribution (semantic structural tokens + contrastive grounding), which could be confusing for readers.

### Questions
Please see the weakness part. Particularly, what were the reasons for omitting recent baselines (e.g., DisTime, LLaVA-MR, UniVTG, Mr.BLIP)?

### Soundness
3

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
The paper introduces MeCo, a framework for temporal localization that replaces timestamp generation with structural tokens: event <ent> and transition <tst>. It augments these tokens with query-focused captioning (QFC), generating detailed captions before each <ent> to refine event semantics. A contrastive structural token grounding module aligns tokens with corresponding video frames to achieve holistic temporal segmentation. Trained on E.T.Instruct along with QFC data, MeCo has strong zero-shot performance across grounding, dense captioning, and complex reasoning tasks.

### Strengths
I think the paper has proposed a new shift from timestamp prediction to semantic segmentation with the help of structural tokens (<ent>, <tst>) plus query-focused captioning (QFC). 

Also, the author present some results under self conducted experiments with fair comparisons to demonstrate the effectiveness of the proposed method.

### Weaknesses
However, I have some severe concern about the paper:

1. Unfair comparison between the proposed method and previous methods. Since the author utilized extra model (MiniCPM-V-2.6) to build the training set, dense captioning will bring more labels to the training set. Thus compared to those timestamp prediction models, it is naturally that the proposed method will have better results. Thus, I think the author should compare previous models with their best results, rather than some complex settings. 

2. Critical comparison are missing including TimeChat and TimeChat-T series model. Also, for Charades-STA and QVHighlights, What if the proposed method also fine-tuned on target dataset, rather than the E.T. Instruct dataset? I think the performance gap will be minor or none. 

3. the method shows smaller gains for action-focused queries, and the paper acknowledges this limitation without a concrete explanation.

### Questions
Please mainly see the weaknesses section above. In general I will give a borderline reject rating.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces MeCo (Measure Twice, Cut Once), a semantic-oriented framework for video temporal localization using Large Language Models (LLMs). It proposes a conceptual shift from previous methods that rely on generating uninformative boundary timestamps, which often fail to fully leverage the LLM's pre-trained semantic understanding capabilities.MeCo fine-tunes a video LLM using a combination of two generative and one discriminative learning task:Structural Token Generation: A generative task where the LLM partitions the input video based on the user query by outputting a sequence of new, special structural tokens.Query-Focused Captioning (QFC): A second generative task that requires the LLM to produce a detailed caption for each queried event segment immediately. Structural Token Grounding: A discriminative task, implemented using a contrastive learning objective, that maps the semantic information encoded in the structural tokens to the corresponding video segments. Extensive experiments demonstrate that MeCo consistently outperforms timestamp-centric approaches across diverse temporal localization tasks.

### Strengths
1.The paper is written with good clarity.
2.The motivation is clear
3. The framework is well-presented

### Weaknesses
1.The structural token generation relies on Supervised Fine-Tuning (SFT) labels that require a contiguous, non-overlapping segmentation of the entire video, achieved by augmenting Ground Truth (GT) event boundaries with neighboring transition segments ($\langle tst \rangle$ tokens) to cover the full duration. This presupposes the existence of exhaustive, high-quality, segment-level annotations for the whole video.
2.  The current asymmetric Structural Token Grounding loss introduces a significant imbalance: the number of negative frame samples for a structural token $s_i$ is $T$ (total frames), whereas the number of negative structural token samples for a frame $h_t$ is only $M+K$ (total segments)28. This massive disparity in negative sample cardinality could lead to the loss function being overly dominated by the frame-level discrimination task.

### Questions
na

### Soundness
2

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
This paper presents MeCo, a semantic-oriented framework that adapts video LLMs for temporal localization by replacing direct timestamp prediction with structured generation. MeCo first emits structural tokens—<ent> (events) and <tst> (transitions)—to outline video structure, augments each event with query-focused captions to enrich semantics, and then grounds tokens to segments using a contrastive objective. Evaluated on E.T. Bench and standard temporal localization benchmarks, MeCo delivers strong and consistent performance.

### Strengths
1) **Motivation & insight.**
   The paper makes a compelling case that asking LLMs to emit *uninformative numeric timestamps* underutilizes their semantic reasoning. This observation is clearly articulated and convincingly motivates a *semantic-oriented* alternative.

2) **Strong, broad improvements.**
   MeCo delivers substantial gains across diverse tasks—notably **59.1% vs. 38.6% F1 on TVG (E.T. Bench)**—with **consistent improvements across nine tasks**. It is especially effective for **multi-segment** scenarios (e.g., **26.2 mAP vs. 1.5** for E.T.Chat on QVHighlights).

3) **Comprehensive evaluation & fair comparisons.**
   The experiments span multiple domains (grounding, dense captioning, complex reasoning) and benchmark suites, with thorough comparisons against numerous baselines under matched, transparent settings.

### Weaknesses
1) **Incremental novelty via integration.**
   The triad—**structural tokens**, **query-focused captions (QFC)**, and **contrastive grounding**—is well motivated but each component echoes established ideas (tokenized structure, CoT-style captioning, CLIP-like contrast). The contribution lies primarily in a **reformulation and clean integration for localization**, rather than a single, fundamentally new algorithmic primitive.

2) **Temporal granularity concerns.**
   Real videos exhibit **richer, nested structures** (actions ↔ sub-actions, scene changes, shot boundaries). A binary **event/transition** scheme may be too coarse: how are **gradual transitions**, **overlapping events**, or **multi-threaded activities** represented?

3) **Caption overhead vs. payoff.**
   Generating and then attending to QFC introduces **non-trivial compute/latency**. How much of the gain is attributable to **QFC** versus **structural tokens** alone? While Table 5 indicates both matter, the **interaction and marginal utility** (A → A+QFC, A → A+QFC+contrast) aren’t dissected in depth.

4) **Failure-mode analysis is thin.**
   Beyond noting “prominently off” cases (Fig. 4), there’s no **systematic breakdown** of when/why the method fails (e.g., heavy occlusion, rapid camera motion, fine-grained action boundaries, dialogue-driven cues). A short taxonomy with **representative examples** would be valuable.

5) **Limited sensitivity studies.**
   Only **β** and **λ** receive analysis (Fig. 4). Important **design knobs**—**NMS threshold** (0.7), **temperature τ**, **projector architecture/width**, **negative sampling strategy**, **frame sampling rate**—are not explored. A small **hyper-sweep and robustness table** would strengthen claims of generality.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
