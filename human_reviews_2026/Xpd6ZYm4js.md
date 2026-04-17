# How Do LLMs Use Their Depth?

- Decision: Reject
- Scores: 6, 8, 2, 2

## Abstract
Growing evidence suggests that large language models do not use their depth uniformly, yet we still lack a fine-grained understanding of their layer-wise prediction dynamics. In this paper, we trace the intermediate representations of several open-weight models during inference and reveal a structured and nuanced use of depth. Specifically, we propose a "Guess-then-Refine" framework that explains how LLMs internally structure their computations to make predictions. We first show that the top-ranked predictions in early LLM layers are composed primarily of high-frequency tokens, which act as statistical guesses proposed by the model early on due to the lack of appropriate contextual information. As contextual information develops deeper into the model, these initial guesses get refined into contextually appropriate tokens. Even high-frequency token predictions from early layers get refined >70\% of the time, indicating that correct token prediction is not ``one-and-done''. We then go beyond frequency-based prediction to examine the dynamic usage of layer depth across three case studies. (i) Part-of-speech analysis shows that function words are, on average, the earliest to be predicted correctly. (ii) Fact recall task analysis shows that, in a multi-token answer, the first token requires more computational depth than the rest. (iii) Multiple-choice task analysis shows that the model identifies the format of the response within the first half of the layers, but finalizes its response only toward the end. Together, our results provide a detailed view of depth usage in LLMs, shedding light on the layer-by-layer computations that underlie successful predictions and providing insights for future works to improve computational efficiency in transformer-based models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work provides a variety of empirical studies demonstrating that different tasks, and even different token generations within a task (based on part-of-speech), require different model-depths and that LLMs are able to dynamically assign depth. Results show that as data propagates through the LLM, the LLM's prediction goes from a training-frequency-motivated prediction to a more context-specific one.

### Strengths
The paper is written in a clear and concise way. Motivations, rationales, and design decisions are always outlined in advance. Analyses of results are always clearly given. Figures are easy to read. The findings suggest an intuitive and useful interpretation for some underlying mechanisms of the LLM, which may prove especially useful for early exiting work.

### Weaknesses
1. At times, "refinement" is used when really only "change" is demonstrated. For example, Figure 3 demonstrates that early predictions are often changed. This is described as demonstrative of refinement in the paper, but if early predictions are changed incorrectly or not correctly (i.e. are originally correct and flipped to incorrect, or are originally incorrect and changed to a different incorrect token), this is not refinement but simply an artifact of low-confidence in early-layer predictions. Figure 3 also shows a similar trend for all three token bins, making this the only real drawable conclusion. Showing specifically "correct" flips might be more informative.
2. Some valuable and apparent continuations are missing (e.g. layer-wise behavior analysis for correct vs. incorrect final predictions, identifying a priori internal indicators of finality of prediction/high confidence-ness). 
3. Some experimental design is questionable: it is never explained why only correctly answered questions are analyzed in Case Study 2. 
4. In general, the notion that the LLM intelligently assigns depth is invoked. This is somewhat problematic because the LLM continues to process these tokens through the full depth of the model. While most of the time a mid-layer high-confidence prediction may not be flipped, it on occasion will be and these flips can contribute to small changes overall performance metrics.

### Questions
1. The work claims and aims to demonstrate that the LLM has an innate sense of depth-requirement for predictions. If the LLM knows this a priori as is implied by the language in the paper, could predictor signals in the layer-wise hidden states of the model be found? This would be greatly beneficial from a cost-saving perspective, as we could early-exit the LLM, avoiding at times the majority of the forward-pass computation. 
2. Why do you only show results for correctly-answered problems in Case Study 2
3. Does the rank-threshold behavior shown apply similarly to correctly and incorrectly answered questions? Could we potentially identify problematic LLM answers in advance by noting, for example, unusually late threshold-passing?
4. How frequently do predictions enter the top-k threshold and then leave it later? Are these cases more typically correct or incorrect predictions?
5. Could the part-of-speech distinction in layer-wise confidence progression be a simple matter of some parts of speech having much fewer options than others? For example, there are only a handful of punctuation marks, but thousands of verbs. Presumably, the LLM has a strong sense of grammar and understands the necessary part of speech at each token prediction well in advance of identifying a high-confidence prediction. Showing how often a token's predicted part of speech flips may be interesting.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
How LLMs use their depth for various tasks has been a key mystery. This paper does a systematic analysis across 3 categories of tasks to try and understand how LLM predictions vary as we go deeper into the stack. Observations highlight a “guess-and-refine” mechanism, wherein the model guesses high frequency tokens early on. In fact, another interesting (and intuitive) observation is that harder tasks need more depth. The paper is well structured, neatly written and the claims are well grounded.

### Strengths
- The three categories of tasks analysed not only show the breadth of the observations and the study, but rather also help clarify the observations from other categories.
- The claims made are very clear and well supported. No extravagant claims are present. 
- The task difficulty vs layer prediction analysis is super nice.

### Weaknesses
- Any analysis on reasoning/CoT tasks would further significantly improve the quality and scope of manuscript.
- The manuscript would benefit from a more detailed discussion on the results, what implications they might have for practitioners or any suggestions or algorithms to improve the performance (say prediction depth) on hard tasks, based on the observations in the manuscript.

### Questions
See the weakness section

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates how LLMs use depth at inference and advances a “guess-then-refine” account: early layers produce frequency-biased provisional guesses, while later layers integrate context to finalize tokens. Using TunedLens to decode intermediate states, it measures onset and flip-rate metrics across multiple models and tasks (POS, fact recall, multiple-choice), yielding an empirical picture of layerwise dynamics with implications for early exit and routing.

### Strengths
* The paper propose a simple, direct “guess-then-refine” view of depth with lightweight metrics. It provides analysis of TuneLens method results.
* Experiments cover multiple model families and tasks with transparent data.
* The findings might have a practical usage beyond interpretability (e.g. early-exit and routing in LLM systems).

### Weaknesses
* The probe by tunelens is trained to mimic the final distribution, so agreement with the final layer is not independent evidence and may imprint the reported pattern. 
    - If the layer embeddings are matched toward that of the last, naturally, it would generate related tokens and patterns, but that is from the affine mapping, but not the model. 
    - I think this is the most critical issue.
    - The authors can try add a probe that does not target the final distribution (or potentially combine with LogitLens) and compare with activation patching to test causality.
* No causal validation - decoded ≠ computed: All claims come from decoding intermediate states rather than causal tests. A probe can read out patterns that are present but not functionally responsible for the final prediction. So the curves may show “guess-then-refine” because it’s easy to decode, not because those layer states actually cause the model’s choice. I understand this is a general limitation in interpretability work, but the lack of causal evidence—and with possible probe-driven bias—makes the findings not convincing.
* Fact-recall analyses keep only correct cases, which induces survivorship bias and makes the depth story look cleaner than it is. Include failure-conditioned analyses and compare onset and flip rates when the model is wrong.
* The first-crossing onset metric is brittle under non-monotonic ranks and can mark spurious early onsets. Report stability-based onsets that require persistence in top-k and include threshold sensitivity analyses.
* Frequency buckets are built on English Wikipedia with model-specific tokenizers, so Frequency-Conditioned Onset may be driven by domain or tokenizer artifacts like whitespace tokens or common symbols. Recompute buckets on other domains and languages, show Top-k composition, and rerun the curves.

### Questions
- For TunedLens, which probes are trained from scratch vs. reused from prior work, and on what data?
- In the multiple-choice setups, what exactly are the option strings the model sees (A/B/C/D characters vs. full words), and are the few-shot exemplars fixed across runs?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper applies TunedLens to a number of open weights LLMs and documents several observations about patterns. For example, the authors note that early layers promote frequent words while middle and later layers promote less frequent words. Later, the authors note that, for multiword tokens, the first token in the phrase doesn't emerge until a later layer, but subsequent words emerge at earlier layers. There are several other related analyses, which extend these observations to specific domains, such as fact recall and multiple choice QA.

I appreciate the author's interest in interpretability work, and I like the question they ask ("how are LLMs using their layers?") but I don't feel there is any novel contribution in this work. All of the observations the authors make boil down to some version of a well-know principle: the more context required for the prediction, the more layers required. I think this is an observation that is almost true by fiat, but which has also been documented fairly regularly over the past few years of studies that involve logit lens, tuned lens, patchscopes, and similar methods.

### Strengths
* The authors ask an interesting question: how do LLMs use their many layers?

### Weaknesses
There is no novelty or contribution here. The authors simply report a few observations that stem from a single existing method (tunedlens). In my experience working with tuned lens and similar methods, the observations described here are taken for granted, and are probably documented at least implicitly in every paper that uses these tools. Moreover, I think the basic claims made by the authors are also previously reported elsewhere, with additional mechanistic detail and insight. For example:

https://aclanthology.org/2022.emnlp-main.3/ -- documents the concept of "saturation" which is the same idea as what the present paper discusses as models "complexity aware depth use"

https://aclanthology.org/2024.naacl-long.281/ -- reports that early layers generate frequent tokens and then later refines (specific to fact recall, but does describe the mechanism for the refinement)

https://aclanthology.org/2025.naacl-long.155/ -- reports the idea that models refine as they incorporate more context in higher layers

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1
