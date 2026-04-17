# AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Emotion understanding is essential for building socially intelligent agents. Although recent multimodal large language models (MLLMs) have shown strong performance on this task, two key challenges remain: (i) spurious associations between emotions and irrelevant audiovisual cues and (ii) hallucination of audiovisual cues driven by text priors in the language model backbone. To quantify and understand these issues, we introduce **EmoReAlM**, a benchmark designed to evaluate MLLMs for cue–emotion associations, hallucinations and modality agreement. We then propose **AVEm-DPO**, a preference optimization technique that aligns model responses with both audiovisual inputs and emotion-centric queries. Specifically, we construct preferences over (i) responses exhibiting spurious associations or hallucinations and (ii) audiovisual input pairs guided by textual prompts. We also include a regularization term that penalizes reliance on text priors, thereby mitigating modality-specific cue hallucinations. Experimental results on DFEW, RAVDESS and EMER demonstrate that our method significantly improves the performance of the reference baseline models (6-19\% of relative performance) in zero-shot settings. By providing both a rigorous benchmark and a robust optimization framework, this work enables principled evaluation and improvement of MLLMs for emotion understanding and social AI.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on emotion reasoning, addressing two key challenges: (1) Spurious associations between emotions and cues (reasoning errors), and (2) Hallucinations (perception errors). The authors first construct EmoReAIM, a benchmark designed to evaluate MLLMs on cue-emotion associations, hallucinations, and modality agreement. They then propose a framework, AVEm-DPO, which leverages DPO and text-prior debiasing techniques to enhance performance in emotion reasoning. Experimental results on three datasets demonstrate the effectiveness of their approach.

### Strengths
1.	This paper tackles a cutting-edge emotion reasoning task by proposing a novel benchmark and solution.

2.	The authors classify emotion reasoning errors into perception errors (hallucinations) and reasoning errors (spurious associations). This classification effectively covers most errors in emotion reasoning.

3.	The paper is well-written and well-motivated, with experimental results validating the effectiveness of the proposed method. It provides new insights into emotion reasoning.

### Weaknesses
1.	From my perspective, "Spurious AV cue associations" and "AV cue hallucinations" are essentially reasoning errors and perception errors, respectively, aligning with the errors in MLLMs. Thus, I suggest classifying errors primarily as reasoning and perception errors in the main text, followed by additional explanations from the "Spurious AV cue associations" and "AV cue hallucinations" perspectives. This would improve readability and attract broader interest from the MLLM research community.

2.	The paper provides a clear definition of "hallucinated cues" (i.e., descriptions containing clues that do not exist in the video). However, I have some concerns about "emotion-irrelevant cues". For example, in Figure 2 ("Emotion reasoning - basic"), the statement "The presence of light music in the background suggests a relaxed state" does relate to the person’s emotion state and should not be considered "emotion-irrelevant." Similarly, in Figure 1, "The dark green color of the background supports the negativity" could also be seen as a valid cue for inferring emotion. Overall, from my perspective, background music and color can be viewed as a useful clue for emotion reasoning and should not be dismissed as "emotion-irrelevant cues."

3.	The paper introduces two types of preference data, which is an interesting contribution. "Emotion-based Response Preference" aligns with my understanding of DPO. However, "Prompt-based Modality Preference" is less clear. Does it involve feeding different audiovisual inputs while expecting the same response? More examples would help clarify this type of DPO data.

### Questions
See weakness

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
The authors noticed that current models often misinterpret emotions by relying on irrelevant cues or hallucinated details, so they propose EmoReAlM, a benchmark for assessing audiovisual emotion reasoning and hallucination robustness in multimodal large language models (MLLMs), which contains 4,000 human-verified MCQA samples to evaluate emotion reasoning, modality alignment, and hallucination stress tests. Additionally, they also introduce AVEm-DPO, a Direct Preference Optimization approach that introduces prompt-based audiovisual preferences and text-prior debiasing to reduce hallucinations. Experiments show that AVEm-DPO significantly improves MLLM performance and robustness on emotion reasoning tasks compared with existing methods.

### Strengths
1. The paper is well-organized and easy to follow, with clear and informative tables and figures that effectively support the presentation.
2. To reduce hallucinations, the authors propose using Direct Preference Optimization (DPO). The method incorporates fine-grained, modality-level preferences based on the input text and reasoning about whether a response is hallucinatory or relevant to emotion prediction. Additionally, a text-prior debiasing strategy is introduced to mitigate hallucination effects. The overall approach is both reasonable and methodologically sound.
3. The experiments and ablation studies are comprehensive, and the presentation of results is clear and easy to follow.

### Weaknesses
1. What is the motivation to use LLMs for visual and audio emotion prediction? It is challenging for LLMs to accurately infer emotions based solely on captions, even for advanced models such as GPT-4o. Moreover, even when an LLM’s prediction matches the ground truth, it does not necessarily imply that the emotional trigger or the reasoning process behind the prediction is correct.

2. Is there any analysis on the individual roles of the visual and audio modalities? For example, which modality provides the essential cues for the final prediction, and do we need both modalities for effective emotion reasoning?

3. In data creation, captions play a crucial role in translating visual information into text. Was any sampling strategy applied during this process? How do the authors ensure that the essential frames or emotional triggers are preserved during caption generation?

### Questions
None

### Soundness
4

### Presentation
4

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
This paper introduce a benchmark designed to evaluate MLLMs for cue–emotion associations, hallucinations and modality agreement and propose a preference optimization technique that aligns model responses with both audiovisual inputs and emotion-centric queries.

### Strengths
- A comprehensive suite of 4,000 human-verified multiple-choice questions (MCQs) across 2,649 unique videos, designed to evaluate three critical aspects of emotion reasoning.
- A multimodal direct preference optimization (DPO) method to align MLLMs with both audiovisual inputs and emotion-centric queries.
- Demonstrates that AVEm-DPO outperforms baselines by 6–19% in zero-shot settings across existing benchmarks and EmoReAlM, with qualitative and user studies confirming reduced hallucinations and spurious associations.

### Weaknesses
- EmoReAlM is derived exclusively from the DFEW dataset, which may limit generalizability to videos with different cultural contexts, demographics, or emotion types
- AVEm-DPO’s training data is generated automatically via Gemini 2.5 (without human verification). While the authors report performance gains, unvalidated preference pairs may introduce hidden biases

### Questions
See weakness

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
4

### Summary
This paper investigates audiovisual emotion reasoning in multimodal large language models (MLLMs). It identifies two key problems: spurious associations between emotion labels and irrelevant audiovisual cues, and hallucination of cues (models inventing non-present audio/visual evidence). To address evaluation gaps, the authors introduce EmoReAlM, a large benchmark with tasks for emotion reasoning (basic and stress-test) and modality agreement, including human verification and statistics drawn from sources such as DFEW. For optimization they propose AVEm-DPO, a preference-optimization approach that combines explicit preferences (prompt-based modality preference and emotion-based response preference) with a text-prior debiasing (TPD) regularizer to reduce reliance on textual priors. Experiments across many baselines and ablations show substantial gains: AVEm-DPO improves zero-shot performance and reduces hallucination and spurious associations (reported improvements of ~6–19% in several settings, and large gains in user-evaluation metrics). The paper provides extensive benchmark details, human verification procedures, and ablative analyses demonstrating which components contribute most to the improvements.

### Strengths
- EmoReAlM benchmark: A comprehensive, human-verified benchmark for audiovisual emotion understanding that tests (a) cue-emotion associations, (b) modality agreement, and © robust stress tests designed to reveal spurious associations and hallucinations. The benchmark includes balanced tasks, adversarial cases, and metrics for spurious associations, modality agreement, and hallucination.
- AVEm-DPO optimization framework: A novel preference optimization method tailored to audiovisual emotion reasoning. It integrates prompt-based modality preference (PMP) and emotion-based response preference (ERP) with text-prior debiasing (TPD) to steer MLLMs toward using actual audiovisual cues and away from text priors or hallucinated cues. The method is evaluated against several baselines and ablations.
- Extensive evaluation and analysis: Thorough quantitative and qualitative experiments (including human evaluation) that show large improvements in emotion attribution, reduced hallucination, and better modality alignment. The paper includes ablation studies, sensitivity analyses, class-wise results, and adversarial modality tests that clarify which components are most effective.

### Weaknesses
- Reliance on proprietary or large LLM tooling for some steps: The paper mentions using GPT-5 to polish text and using LLMs for annotation/evaluation. This reliance can raise reproducibility concerns if those tools or their prompts are not fully disclosed; it may also bias dataset construction and evaluation unless careful controls are provided.
- Potential dataset and evaluation biases: Although the benchmark is human-verified, the document suggests many generated QA items and uses subtitled non-English videos with English subtitles. This raises questions about cultural, linguistic, or subtitle-transcription biases influencing emotion labels and whether the benchmark fully generalizes across languages, contexts, and production styles.
- Limited discussion of failure modes and external validity: While many stress tests and ablations are provided, the paper could better characterize remaining failure cases (e.g., what kinds of emotions or multimodal interactions still confuse the model), and whether AVEm-DPO overfits to particular datasets or annotation styles. Also, practical deployment implications (latency, compute cost, or safety concerns when misclassifying emotions) are not deeply discussed.

### Questions
SEE WEAKNESS

### Soundness
3

### Presentation
3

### Contribution
3
