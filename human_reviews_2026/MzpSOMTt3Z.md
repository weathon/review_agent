# Audio Turing Test: Benchmarking the Human-likeness of Large Language Model-based Text-to-Speech Systems in Chinese

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Recent advances in large language models (LLMs) have significantly improved text-to-speech (TTS) systems, enhancing control over speech style, naturalness, and emotional expression, which brings TTS Systems closer to human-level performance.
Yet evaluation still relies largely on the Mean Opinion Score (MOS), whose subjectivity, environmental variability, and limited interpretability prevent it from faithfully capturing how human-like the synthesized audio is.
Existing evaluation datasets also lack a multi-dimensional design, often neglecting factors such as speaking styles, context diversity, and trap utterances, which is particularly evident in Chinese TTS evaluation.
To address these challenges, we introduce the **A**udio **T**uring **T**est (ATT), a multi-dimensional Chinese corpus dataset ATT-Cropus paired with a simple, Turing-Test-inspired evaluation protocol. Instead of relying on complex MOS scales or direct model comparisons, ATT asks evaluators to judge whether a voice sounds human. This simplification reduces rating bias and improves evaluation robustness.
To further support rapid model development, we also finetune Qwen2-Audio-Instruct with human judgment data as Auto-ATT for automatic evaluation. 
Experimental results show that ATT effectively differentiates models across specific capability dimensions using its multi-dimensional design. 
Auto-ATT also demonstrates strong alignment with human evaluations, confirming its value as a fast and reliable assessment tool.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the Audio Turing Test (ATT), a novel evaluation framework for assessing the human-likeness of LLM-based Chinese Text-to-Speech (TTS) systems. ATT integrates a multi-dimensional corpus (ATT-Corpus) covering five key linguistic dimensions (e.g., Chinese-English code-switching, polyphonic characters), a Turing Test-inspired human evaluation protocol, and an automatic evaluation tool (Auto-ATT) fine-tuned on Qwen2.5-Omni-7B. Unlike traditional Mean Opinion Score (MOS) methods, ATT uses ternary judgments ([Human], [Unclear], [Machine]) to reduce bias and improve discriminative power. Experiments with 857 native Chinese listeners and 5 state-of-the-art TTS models show that ATT effectively distinguishes model performance, with top-performing Seed-TTS achieving a Human-likeness Score (HLS) of only 0.4—revealing gaps between synthetic and human speech. Auto-ATT demonstrates strong alignment with human judgments and outperforms conventional MOS predictors on trap items.

### Strengths
Targeted Solution to Critical Gaps: Addresses MOS’s limitations (subjectivity, low interpretability) and the lack of multi-dimensional, Chinese-specific TTS evaluation datasets, filling a key niche in LLM-driven TTS assessment.
Comprehensive Framework Design: Combines a well-constructed corpus (semi-automated generation + expert validation), rigorous human evaluation (trap items, consistency checks), and an efficient automatic tool, enabling both qualitative and quantitative analysis.
Robust Experimental Validation: Large-scale human evaluations (857 participants) and statistical tests (GLMM) confirm ATT’s reliability, while Auto-ATT’s superior performance over UTMOSv2 and DNSMOS Pro highlights its practical value for rapid model iteration.
Actionable Insights: Identifies specific weaknesses of current TTS systems (e.g., prosodic unnaturalness, flat emotional expression) and provides fine-grained comparisons across models, voices, and linguistic dimensions.

### Weaknesses
Language and Scenario Limitation: The framework is exclusively designed for Chinese, limiting generalizability to other languages with distinct linguistic features (e.g., tonal vs. non-tonal languages).
Narrow Trap Item Diversity: While trap items monitor attention, the paper only mentions "deliberately flawed synthetic clips" and "genuine human recordings"—more diverse trap types (e.g., edge-case linguistic structures) could strengthen robustness.
Auto-ATT Training Data Opacity: The paper references "additional private evaluation data" for Auto-ATT training without detailing its size, distribution, or how it complements public ATT-Corpus, raising questions about reproducibility.
Lack of Longitudinal or Real-World Testing: Evaluations focus on controlled audio clips; performance in real-world scenarios (e.g., background noise, dialogue context) is not explored, limiting insights into practical applicability.

### Questions
Given ATT’s Chinese-specific design, what key adaptations would be required to extend the framework to non-tonal languages (e.g., English) or languages with unique prosodic features (e.g., Japanese)?
How does Auto-ATT’s performance degrade when evaluating TTS models not included in its training data (e.g., newly developed models with novel architectures), and what strategies could mitigate this?

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
3

### Summary
This paper proposes Audio Turing Test (ATT), a human-likeness evaluation framework for Chinese LLM-TTS that pairs (i) a multi-dimensional corpus (ATT-Corpus) spanning numerals/special characters, code-switching, paralinguistics & emotions, classical prose/poetry, and polyphonic characters, with (ii) a ternary, Turing-style human protocol that labels each clip as Human / Unclear / Machine and derives a Human-Likeness Score (HLS) (1.0, 0.5, 0.0). The authors also fine-tune Qwen2.5-Omni-7B with human judgments to create Auto-ATT, a model-as-a-judge that predicts HLS and reportedly aligns strongly with human ratings. Benchmarks across five model families show clear separation and notably low absolute human-likeness (best model ≈0.4), in contrast with high MOS reported elsewhere.

### Strengths
* Multidimensional corpus targets common Chinese difficulty factors (polyphony, poetry syntax, code-switching).

* Ternary human protocol + rationales is a simple but meaningful shift away from MOS.

* The implementation of trap items as a good quality control.

* Auto-ATT is a useful direction; training a speech-judge model is under-explored, and the demonstrated correlation to humans is promising.

* The benchmark highlights meaningful gaps between SOTA models and human speech.

### Weaknesses
1. (Interpretability of Human-Likeness) HLS collapses three distinct cases (Human mistaken as Machine, Machine mistaken as Human, and Unclear) into a single linear score. Without reporting how often each category is chosen, it is unclear whether high HLS reflects genuine human-likeness or annotator uncertainty. Excessive “Unclear” selections may artificially inflate scores.

2. (Filtering Bias From Manual Spot Checks) The authors state that samples failing “synthesis success” or “synthesis consistency” are verified. Eliminating weak samples before evaluation can bias results toward best-case outputs. The paper does not clarify what was done with samples that failed this spot check.

3. (Sampling Policy Ambiguity) It is unclear whether annotators see one sample from each system or a random subset. If some participants repeatedly select “Unclear,” this may distort HLS. Details on randomization, system coverage, and balancing are not reported.

4. (Annotator and Expert Clarity) The paper inconsistently reports annotator counts (437 vs 857). Expert selection criteria are unclear, and post-hoc alignment of participant justifications to labels is subjective. The number of experts per sample and conflict resolution process are not specified.

5. Missing Citations -
* Praveen S V, Sherry Thomas, Sai Teja M S, Suvrat Bhooshan, Mitesh M. Khapra "The State Of TTS: A Case Study with Human Fooling Rates." Proc. Interspeech 2025
* Nguyen, Binh, and Thai Le. "TURING’S ECHO: Investigating Linguistic Sensitivity of Deepfake Voice Detection via Gamification." Proc. Interspeech 2025

6. (No Comparison With Established Listening Tests) The benchmark is not compared against MUSHRA or CMOS for ranking fidelity. While ATT may separate systems well, this has not been demonstrated relative to standard perceptual tests.

### Questions
1. Regarding manual spot checks: “We examine synthesis success and consistency.” What is the removal policy for failed samples? How many samples were discarded per system? Could this bias the evaluation toward cherry-picked successes?

2. How are samples assigned to annotators? Does each participant hear all systems so that within-subject comparison is possible?

3. How often do annotators choose Human / Machine / Unclear? Does widespread “Unclear” bias the HLS scale?

4. How were expert reviewers selected, and how many participated? In cases of disagreement regarding the presence or absence of an artifact, what adjudication procedure was applied? 

5. The annotator count is inconsistently reported as 437 and 857. Which value is correct?

### Soundness
2

### Presentation
3

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
The authors propose the AudioTuringTest Benchmark, a new evaluation framework for Chinese TTS models. The paper attempts to address the reproducibility and saturation issues of MOS-like metrics, the de-facto evaluation protocol for TTS models. To do this, the ATT benchmark attempts to simplify the rating criteria into a Turing Test-like metric, whether or not a speech sample is from a human. ATT is created with human judgements of synthetic and real data. The authors also create Auto-ATT a model-as-a-judge version of ATT. Results suggest that ATT is able to capture key differences between TTS systems along several axes and Auto-ATT correlates well with human judgement.

### Strengths
- ATT attempts to address the critical limitations of MOS / pseudo-MOS by disentagling speech characteristics at the data level and simplifying the evaluation scheme 
- ATT evaluates along several axes, such as numerals, code-switching, paralinguistics, and poetry.
- ATT can clearly distinguish the strengths and weakness of different model along each axes, allowing fine-grained insights of TTS performance 
- Auto-ATT is a novel model-as-judge that can be used to automate the application of ATT at scale

### Weaknesses
- The ATT corpus is developed using the TTS models the authors intend on evaluating. It is unclear how it and AutoATT generalize to unseen systems, which does not address the claimed robustness issue of pseudo-MOS.
- ATT cannot distinguish speaker-level characteristics, which makes evaluation using speaker similiarity MOS or neural embeddings still required

### Questions
- Will the annotator ratings be released with the dataset?

### Soundness
3

### Presentation
3

### Contribution
3
