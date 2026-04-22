# MECAT: A Multi-Experts Constructed  Benchmark for Fine-Grained Audio Understanding Tasks

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 4, 6, 4, 2

## Abstract
While large audio-language models have advanced open-ended audio understanding, they still fall short of nuanced human-level comprehension. This gap persists largely because current benchmarks, limited by data annotations and evaluation metrics, fail to reliably distinguish between generic and highly detailed model outputs. To this end, this work introduces MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. Generated via a pipeline that integrates analysis from specialized expert models with Chain-of-Thought large language model reasoning, MECAT provides multi-perspective, fine-grained captions and open-set question-answering pairs. The benchmark is complemented by a novel metric: DATE (Discriminative-Enhanced AudioText Evaluation). This metric penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability. A comprehensive evaluation of state-of-the-art audio models is also presented, providing new insights into their current capabilities and limitations. The data and code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This submission proposes a novel benchmark, MECAT, for finegrained open-ended audio understanding evaluation. MECAT is constructed with a complex pipeline with many AI experts in the loop, including annotation, aggregation, filtering, and more. The paper claims MECAT is more diverse than previous benchmarks and has higher quality. In addition, the paper proposes a new method for open-ended captioning evaluation.

### Strengths
The proposed benchmark is very useful to the community as it reveals many detailed aspects of the quality of audio language models. Especially, there are sub-metrics with focus on different domains, lengths and cognitive categories, which could shed light on current audio language models and guides the community to develop better models accordingly. The AI expert annotations are also well-designed such that the community can easily reproduce such samples with an automated pipeline.

### Weaknesses
For the data construction, since all steps are based on AI models and there is no human-in-the-loop, it is questionable whether the outputs are high-quality enough or have some hidden bias. Since none of the AI models are super reliable to my knowledge, I tend to believe such biases or inaccurate test samples exist. Therefore, this work should at least verify the quality of the testset via human inspection or tests. Furthermore, there is no solid evidence the proposed benchmark is siginificantly more useful than MMAU/MMAR, despite it being open-ended. 

For the metric design part, while I agree that current captioning-based metrics (CIDEr, SPICE, FENSE, etc) are not good enough, I do not think the proposed DATE addresses the issues. First, the TF-IDF based $v_T$ computation does not capture the temporal information because it is a sum $\sum_{t\in T}$. It is therefore more similar to a bag-of-word sementic embedding, which is undesirable in captioning evaluation. The Setence-BERT embedding also does not consider the complex meaning of particular words given the context, as the specific use of some word can be very different from its "averaged" meaning as represented by $E(\cdot)$. Moreover, the cross-sample discriminability considers the whole audio testset pool, making the metric not per-sample static and dependent on the entire testset. However, we expect the metric to be fixed for a given sample, regardless what the other samples are.

For the experimental design part, only very few baselines are evaluated. This is not acceptable for a benchmark-focused paper -- which should check all existing open models as a bottom line (see how many baselines MMAR reported). There is not much analysis on different audio language models given the scores. What do the scores imply besides simply ranking the models? Do the scores reveal any potential bias of existing models? Do the scores show noticeable hints that prior benchmarks (MMAU, MMAR) did not show? Are there PEFT or training-free methods to improve baseline models on the benchmark? These are some examples that can show the usefulness of the proposed benchmark.

### Questions
Please refer to the weakness section.

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
5

### Summary
The paper proposes MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. The benchmark comes with a novel evaluation metric: DATE (Discriminative-Enhanced Audio Text Evaluation). The metric penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability. Audios are collected from ACAV100M and the benchmark consists of 20,000 Creative Commons-licensed audio clips, each with a maximum duration of 10 seconds.

### Strengths
- I appreciate the use of open models for benchmark curation. 
- The results and discussion section is nice and appreciated. Provides useful insights.
- DATE is novel and well motivated.

### Weaknesses
- The maximum duration is only 10 seconds. Short audio snippets can only be as complex.
- The comprehensive of models evaluated is not great. Many more models have been released, including Audio Flamingo 3, etc
- The tasks for coverage are very foundational and have been explored earlier. So I find a slight lack of novelty here.

### Questions
- Why were general sounds not included? Any specific reason?
- What is the impact of the CoT reasoning?

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
5

### Summary
The paper introduces MECAT, a benchmark for fine-grained audio understanding with two tasks, captioning and open-set QA. Data are built by a multi-expert pipeline that runs domain-specific audio models, then uses an LLM with chain-of-thought to synthesize rich captions and QA pairs. The benchmark spans pure and mixed domains across speech, music, sound events, silence, and their combinations. The paper also proposes DATE, a metric that blends weighted semantic similarity and cross-sample discriminability to reward specific, distinctive descriptions.

### Strengths
1. The paper is easy to read. Task definitions, pipeline diagrams, and scoring formulas are explicit. Caption scoring lists categories and weights, and QA aggregation is uniformly specified. This clarity lowers the barrier for reimplementation and for fair comparison.

2. Specialized audio models first detect speech content, speakers, musical attributes, events, and acoustic conditions. A language model then synthesizes these signals into captions and QA with step by step reasoning. This staged design helps surface details that single labelers often miss and supports mixed clips that blend speech, music, and environmental sounds.

3. DATE combines a tf-idf weighted semantic similarity term with a cross-sample discriminability term, joined by a harmonic mean. This discourages generic captions and favors precise descriptions that fit the target clip better than others.

### Weaknesses
1. The construction stack depends on ASR, diarization, emotion, and event taggers. Mistakes at these stages can leak into captions and QA despite later filtering. The paper notes filtering, yet residual noise is likely in complex audio. This critiques robustness, not the clarity of the written specification.

2. Audio Flamingo 2 appears in analysis during construction and also in evaluation. This can advantage that model family. A clearer separation between builder models and evaluated systems would reduce this risk.

3. Caption weights are fixed, and the QC pipeline uses an empirical GLAP threshold. The formulas are clear, yet the motivation for these constants is brief, which can affect reproducibility across domains.

### Questions
1. What design principle determined the caption category and subcategory weights, and are these weights intended to reflect perceived task importance or observed reliability of references. Please clarify whether weights are constant across domains and why.

2. Please explain why DATE combines a tf–idf weighted semantic similarity term with a cross-sample discriminability term, and why you fuse them with a harmonic mean. Which failure modes in audio captioning does this design aim to fix, and why prefer these components over alternatives such as pure embedding similarity or learned weighting.

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
Compared to human beings, audio language models fall short of human-level audio understanding. The author points out that one of the reasons for this is that current benchmarks developed for audio LMs have significant limitations on annotations and metrics. They constructed a benchmark, called MECAT, and they also proposed a new metric. Then, the author evaluated SOTA audio LMs on this new benchmark to demonstrate their capabilities and limitations.

### Strengths
The introduction and explanation of machine hearing and the limitations of the current metric are clear.

The less fine-grained labeling problems existing in current captioning and question-answer benchmarks are explained in detail.

I also like the idea of the distinct audio domains designed. The domain experts are explained in detail.

### Weaknesses
**1.** The structure of some content can be improved. At the end of line 44, the authors state that the benchmark is often a neglected bottleneck. But right after it, they started to discuss the limitations of the current metric. I would suggest putting the following paragraph, the one started on line 53, right after it to ensure consistency and better presentation.

**2.** Figure 1 has too many written descriptions. I would suggest a sub-figure that contains all the explanations for clarity.

**3.** For audio understanding tasks, one of the problems in current audio LMs is that the length of audio they can process (before a significant performance drop) is very short, usually less than 30 seconds. Since this work focuses on understanding more nuanced, more descriptive content for each audio clip, I think the 10-second audio length in MECAT is very limited. This is one major drawback from the perspective of the contribution of the work.

**4**. The discussion on the limitations of the work is too superficial and shows limited insight.

### Questions
**1.** In Appendix D, the multiple choice options under section 3.1.1, how are these selected?

**2.** So, the overall objective of 'synthesize' is for labeling or some other purposes?

**3.** What will DATE represent, or mean, on a dataset level?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. It is generated through a pipeline that integrates analysis from specialized expert models with COT large language model reasoning. It provides multi-perspective,
fine-grained captions and open-set question-answering pairs. The paper also introduces a novel metric: DATE (Discriminative-Enhanced Audio Text Evaluation) which penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability.

### Strengths
- The paper introduced tf-idf–weighted embeddings + cross-sample rank to penalize generic captions
- Novel use of specialized audio-related experts models, including content-specific models and content unrelated models followed by Chain-of-Thought enhanced LLM reasoning for caption generation
- MECAT provides 18 reference captions per clip and reports a vocabulary of 22,595 unique words, much larger than other literature
- Public release of data and code

### Weaknesses
- Potential overlap of test set with the training data of model since the test set is made from ACAV100M
- Audio Flamingo 2 hallucinates a lot and is an overall poor model for any real world task, using that to generate data is not good. The paper uses it not just for sound but also for music.
- Limited baselines - there are a lot more LALMs (open source and proprietary) models available in the literature
- Since a lot of pre-trained models are used to generate the benchmark a lot of biases can creep into the dataset

### Questions
- What is the logic or explanation behind the equations/coeffs in section 3.2?
- is there a correlation between DATE and human task preference?
- how does DATE correlate qualitatively with the model's performance on a captioning/QA task
- why no proprietary models?

### Soundness
2

### Presentation
3

### Contribution
2
