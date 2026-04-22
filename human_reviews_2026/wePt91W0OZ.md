# PodEval: A Multimodal Evaluation Framework for Podcast Audio Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 2, 6, 4

## Abstract
Recently, an increasing number of multimodal (text and audio) benchmarks have emerged, primarily focusing on evaluating models' understanding capability. However, exploration into assessing generative capabilities remains limited, especially for open-ended long-form content generation. Significant challenges lie in no reference standard answer, no unified evaluation metrics and uncontrollable human judgments. In this work, we take podcast-like audio generation as a starting point and propose PodEval, a comprehensive and well-designed open-source evaluation framework. In this framework: 1) We construct a real-world podcast dataset spanning diverse topics, serving as a reference for human-level creative quality. 2) We introduce a multimodal evaluation strategy and decompose the complex task into three dimensions: text, speech and audio, with different evaluation emphasis on "Content" and "Format". 3) For each modality, we design corresponding evaluation methods, involving both objective metrics and subjective listening test. We leverage representative podcast generation systems (including open-source, close-source, and human-made) in our experiments. The results offer in-depth analysis and insights into podcast generation, demonstrating the effectiveness of PodEval in evaluating open-ended long-form audio. This project is open-source to facilitate public use: https://anonymous.4open.science/r/PodEval-iclr.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes PodEval, a comprehensive, open-source evaluation framework designed for multimodal, long-form podcast audio generation. The work addresses the significant challenge of evaluating open-ended generative tasks, where there is no single ground-truth reference. The framework decomposes the evaluation into three core dimensions—text (conversation content), speech (spoken dialogue), and audio (overall sound including music, sound effects, and their integration)—with a clear distinction between "Content" and "Format" quality. To support this, the authors construct a real-world podcast dataset ("Real-Pod") spanning diverse topics, serving as a reference for human-level creative quality. For each dimension, PodEval provides a suite of evaluation methods: a combination of quantitative metrics and LLM-as-a-judge for text; objective metrics (WER, SIM, DNSMOS) and a subjective Dialogue Naturalness test for speech; and objective metrics (Loudness, SMR, CASP) along with a novel Questionnaire-based MOS test for audio. The framework is validated through experiments on a range of podcast generation systems, and the entire project is released as open-source.

### Strengths
Highly Practical and Timely Contribution: The paper addresses a critical and under-served problem in the AIGC community: the evaluation of open-ended, long-form generative content. With the rise of AI-generated podcasts and similar content, a standardized evaluation framework is desperately needed. PodEval provides a well-structured, practical solution that researchers can immediately adopt, making it a valuable community resource.

Comprehensive and Well-Designed Evaluation Methodology: The decomposition of the task into text, speech, and audio dimensions is clear and logical. The framework's strength lies in its combination of multiple evaluation strategies. The use of LLM-as-a-judge for text content, the adaptation of the MUSHRA framework for a Dialogue Naturalness test with attention checks, and the novel approach of concatenating beginning/middle/end segments for long-form audio MOS testing are all well-considered and effective design choices that address the core challenges of reliability and scalability.

Creation of a Valuable Benchmark Dataset: The construction of the "Real-Pod" dataset is a significant contribution in itself. By curating a diverse collection of real-world podcasts across 17 categories, the authors provide a high-quality reference point for human-level quality. This dataset serves as a crucial benchmark for future work in podcast generation and evaluation, enhancing the long-term value of this paper.

### Weaknesses
While the framework is novel and well-integrated, many of the individual evaluation techniques are adaptations of existing methods. For instance, using LLMs as judges (e.g., MT-Bench), DNSMOS for speech quality, and FAD for audio quality are well-established practices. The paper's primary contribution is the curation and combination of these methods rather than the invention of fundamentally new evaluation algorithms.

### Questions
The paper describes a manual review process for curating the Real-Pod dataset but fails to specify the number of human experts involved or their qualifications, leaving the potential for bias and subjectivity in the benchmark's creation unaddressed.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces PodEval, a comprehensive evaluation framework for the open-ended task of podcast audio generation. The authors identify key challenges in this domain, such as the lack of reference standards and the difficulty of evaluating long-form, multi-modal content. To address this, they propose a structured approach that decomposes the evaluation into three dimensions: text, speech, and audio. For each dimension, a suite of objective metrics and subjective listening tests are designed. A key contribution is the creation of the Real-Pod dataset, a curated collection of real-world podcasts to serve as a human-level quality reference. The framework is demonstrated by evaluating several podcast generation systems, including open-source, closed-source, and human-created examples.

### Strengths
1. Problem Significance: The paper tackles the highly relevant and difficult problem of evaluating open-ended, long-form audio generation, a critical bottleneck for progress in this area.
2. Comprehensive Structure: The decomposition of the evaluation into text, speech, and audio dimensions, with a further distinction between "Content" and "Format," provides a logical and structured way to approach this complex task.
3. Valuable Dataset: The Real-Pod dataset is a well-curated and useful resource. The process for its creation is transparent and methodologically sound, providing a much-needed benchmark for human-level quality.
4. Rigorous Subjective Test Design: The subjective evaluation methodologies, particularly the Dialogue Naturalness test based on MUSHRA with high/low-quality anchors and the detailed questionnaire-based MOS test, are well-designed and incorporate best practices like spammer detection to ensure data validity.

### Weaknesses
1. Contradiction Between Objective and Subjective Results: This is the most significant weakness of the paper. The PodAgent system, which appears to be a focal point of the evaluation, consistently scores well on many of the proposed objective metrics (e.g., high SPTD in Fig 3, perfect SMR_SCORE and good CASP in Table 9 and Fig 5), yet it performs very poorly in the subjective tests for dialogue naturalness (Fig 4) and overall listening experience (Fig 6). An evaluation framework is only effective if its objective metrics are meaningful proxies for human perception. This stark discrepancy suggests that the proposed objective metrics are either insufficient or are measuring the wrong signals, thus failing to capture what actually constitutes a high-quality podcast.
2. Potential for Evaluation Bias: The paper heavily features PodAgent. The objective metrics seem to align particularly well with the architectural strengths of PodAgent (e.g., distinct speaker timbres leading to high SPTD, clear separation of speech and music leading to good SMR/CASP). However, the system's core weakness—its reliance on a single-sentence TTS model, leading to unnatural dialogue flow—is only revealed by the subjective tests. This raises concerns that the framework may be unintentionally tailored to a specific type of system, diminishing its generalizability and fairness as a benchmark. The framework's value is diminished if it rewards systems for being objectively "correct" on certain technical measures while failing on the ultimate goal of perceptual quality.
3. Lack of Correlation Analysis: The paper presents objective and subjective results separately but provides no quantitative analysis of the correlation between them. For an evaluation paper, demonstrating a strong correlation between proposed automatic metrics and human judgments is paramount to validating the metrics. Without this analysis, the utility of the objective part of PodEval remains unproven.

### Questions
1. The core issue is the discrepancy between objective and subjective scores for systems like PodAgent. How should a user of PodEval interpret a scenario where a system excels on objective metrics but fails catastrophically in human evaluation? Does this not invalidate the utility of the objective metrics for predicting real-world performance?
2. Could the authors provide a correlation analysis (e.g., Pearson or Spearman correlation) between the scores from the objective metrics (especially the novel ones like SPTD, and audio-related ones like SMR and CASP) and the subjective scores (Dialogue Naturalness, Human Likelihood, etc.) across all tested systems? This would be the most direct way to validate the claim that the framework is effective.
3. Given that PodAgent's poor subjective performance is attributed to its single-sentence TTS backend, it seems dialogue-level prosody and coherence are the most critical factors for naturalness. None of the proposed objective metrics appear to directly measure this. Do the authors believe a new objective metric is needed to capture this crucial aspect, and if so, why was it not included in the PodEval framework?
4. What steps were taken during the design of the objective metrics to ensure they were general and not biased towards the specific outputs of the PodAgent system?

### Soundness
2

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
4

### Summary
This paper introduces PodEval, a multimodal evaluation framework for podcast-like long-form audio generation. The authors argue that existing multimodal benchmarks mostly focus on understanding tasks, whereas open-ended generative evaluation, especially for long-form audio, is underexplored. PodEval decomposes the evaluation into three dimensions: text, speech, and audio, each assessed through both objective metrics (e.g., BLEU, DNSMOS, CASP) and subjective listening tests. They also curate a real-world dataset called Real-Pod, spanning 17 podcast categories and 51 topics, to serve as a human-quality reference. The framework includes both quantitative
metrics (Distinct-N, Info-Density, SPTD, etc.) and LLM-as-a-judge evaluations. Experimental results compare various open-source, closed-source, and human-created podcast systems (PodAgent, MoonCast, NotebookLM, etc.) and demonstrate the reliability of PodEval as a standardized evaluation pipeline.

### Strengths
• Tackles an underexplored but important problem: evaluating open-ended long-form multimodal generation.

• Provides a clear multimodal decomposition (text/speech/audio) with both objective and subjective evaluation methods.

• Introduces the SPTD metric for speaker timbre diversity and validates it empirically.

• Comprehensive dataset (Real-Pod) and open-source toolkit increase reproducibility and potential community impact.

• Thoughtful subjective test design with attention checks and justification mechanisms.

### Weaknesses
• The work’s novelty is primarily integrative rather than conceptual; it reuses many established metrics and evaluation settings.

• The proposed SPTD metric lacks deeper theoretical analysis or correlation studies with human perception.

• Subjective tests use a relatively small participant pool (n=20), which may limit generalizability.

• Some sections (especially methodology details) are overly descriptive and could be summarized more succinctly.

• The domain focus on podcasts might limit transferability to other long-form generative modalities (e.g., audiobooks or radio).

### Questions
• How robust is the SPTD metric across different TTS models and languages? Does it correlate well with human-perceived speaker distinction?

• Could the authors generalize PodEval to other long-form generative audio types (e.g., storytelling or news broadcast)?

• Are there any findings on inter-rater reliability in the subjective listening tests?

• How might LLM-as-a-judge results change if newer audio-language models are used instead of GPT-4?

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
PodEVAL proposes a multimodal evaluation framework specifically for systems that generate long-form dialogue in a podcast-like format. It evaluates along three dimensions: text-based, speech-based, and audio-based, and introduces additional metrics beyond the objective metrics traditionally used in speech synthesis and LLM evaluation, allowing dialogue generation models to be assessed using multiple objective metrics. For subjective evaluation, it also proposes a human evaluation procedure with filtering methods and a structured questionnaire to improve the reliability of evaluating long-form dialogue. Using the proposed evaluation framework, the paper evaluates several dialogue generation models, including PodAgent, GPT4, NotebookLM, and Mooncast, as well as real human dialogue, and presents interpretations based on the results of this evaluation.

### Strengths
- The research goal is well motivated. Establishing a unified evaluation framework for long-form dialogue generation seems important.
- Presents a reliable subjective evaluation procedure and analyzes its relationship to each model’s performance.
- Provides consistent objective metrics across multiple dialogue generation models, enabling results to be analyzed in a consistent way.
- Proposes a method for subjective evaluation of long-form dialogue generation by stitching together 1-minute FIRST / MIDDLE / FINAL segments, allowing assessment without length bias.
- Open-sourced evaluation framework

### Weaknesses
- Text-based and speech-based evaluations are already well established and not particularly original. Because podcasts are fundamentally about spoken communication, at this stage the use of audio-based evaluation for dialogue generation feels less motivated to me. What seems more important is proposing evaluation methods that capture how well the generated podcast actually communicates, including how natural, engaging, and listenable it is for a human listener.
- For long-form dialogue generation, it would have been better to see more reliable and strongly correlated objective metrics at the speech level that can actually reflect what we care about. The current speech-related objective metrics are largely based on existing, standard approaches.
- It is unclear whether other dialogue generation models will actually adopt this framework to evaluate themselves. The text-based evaluation is basically an evaluation of the LLM that produced the base script, so for models that generate dialogue conditioned on a given LLM script, the speech-based evaluation feels more important. However, the speech-related objective metrics mostly feel like pre-existing methods. If the authors want this evaluation framework to be followed, it would help to present a stronger motivation for why this specific setup should be used.

### Questions
- “Regarding proposed SPTD”: SPTD mostly just reflects that the speakers don’t sound identical, and as long as voices aren’t confusingly similar, pushing that gap larger doesn’t automatically mean the dialogue is clearer. So using it as direct evidence of “better clarity” feels overstated. In addition, SPTD does not capture within-speaker behavior. Measuring within-speaker segment-level similarity over time could help evaluate whether a dialogue system can generate expressive but identity-consistent voices across different emotional or conversational states, which is important for evaluating naturalness in dialogue generation.
- When evaluating a dialogue generation model, shouldn’t we also evaluate what the model actually produces, for example by running ASR on the generated audio and letting an LLM judge those transcripts, instead of only evaluating the base script?

### Soundness
3

### Presentation
3

### Contribution
2
