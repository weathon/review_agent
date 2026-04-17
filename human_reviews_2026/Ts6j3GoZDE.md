# STAR-Bench: Probing Deep Spatio-Temporal Reasoning as Audio 4D Intelligence

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6, 6

## Abstract
Despite rapid progress in Multi-modal Large Language Models and Large Audio-Language Models, existing audio benchmarks largely test semantics that can be recovered from text captions, masking deficits in fine-grained perceptual reasoning.
We formalize audio 4D intelligence that is defined as reasoning over sound dynamics in time and 3D space, and introduce STAR-Bench to measure it. STAR-Bench combines a Foundational Acoustic Perception setting (six attributes under absolute and relative  regimes) with a Holistic Spatio-Temporal Reasoning setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories. Our data curation pipeline uses two methods to ensure high-quality samples. For foundational tasks, we use procedurally synthesized and physics-simulated audio. For holistic data, we follow a four-stage process that includes human annotation and final selection based on human performance. Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5\% temporal, -35.2\% spatial), evidencing its focus on linguistically hard-to-describe cues. Evaluating 19 models reveals substantial gaps compared with humans and a capability hierarchy: closed-source models are bottlenecked by fine-grained perception, while open-source models lag across perception, knowledge, and reasoning. Our STAR-Bench provides critical insights and a clear path forward for developing future models with a more robust understanding of the physical world.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces STAR-Bench, a benchmark designed to evaluate “4D Audio Intelligence”, defined as reasoning over sound dynamics in both time (1D) and 3D space. The authors argue that existing audio benchmarks (e.g., MMAU, MMAR) largely assess coarse, text-representable semantics rather than fine-grained perceptual reasoning. STAR-Bench aims to fill this gap through two complementary levels: a) Foundational Acoustic Perception — quantitative evaluation of six acoustic attributes (pitch, loudness, duration, azimuth, elevation, distance) under absolute and relative regimes.b) Holistic Spatio-Temporal Reasoning — tests requiring temporal segment reordering (continuous and discrete processes) and spatial reasoning (localization, multi-source relations, dynamic trajectories). The dataset combines synthetic and real-world audio, curated through a four-stage pipeline involving procedural synthesis, AI-assisted filtering, human annotation, and expert validation. The benchmark comprises 2,353 questions across these tasks.

### Strengths
- Establishes a clear and rigorous formalization of “audio 4D intelligence.”
- I agree on the problem formulation.
- The benchmark is novel and challenging. At the same time, it is useful.
- Benchmark design is systematic, combining physical simulation, human validation, and multiple subtask layers.
- Offers a quantitative diagnostic structure — separating perceptual, temporal, and spatial reasoning.
- Provides valuable model insights (e.g., Gemini’s bottleneck in perception vs. open-source deficits in reasoning and grounding).

### Weaknesses
- (Minor) The benchmark’s scale (2.3k samples) is relatively small compared to typical multimodal datasets.
- Evaluation focuses mainly on multiple-choice QA
- I don’t have many questions. The work is impressive. Some similar works have popped up lately but can be considered as parallel.
- Some points about Table 1 I am not sure how the ticks and crosses were made. What does Deep Reasoning meaning and why Spatial DR and Temporal DR are crossed for MMAU-Pro? I think spatial is also covered in MMAU-Pro? Can the authors provide more clarity?
- I have no questions. This looks like a well-rounded paper, and I vote for acceptance.

### Questions
I don't have questions.

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
4

### Summary
This paper introduces STAR-BENCH, a new benchmark designed to evaluate what the authors term "audio 4D intelligence", the ability to reason about sound dynamics in 3D space and time. The authors argue that existing audio benchmarks primarily test for semantic content that is easily described by text captions, a claim they support with a "caption-only" evaluation experiment where model performance drops only slightly. STAR-BENCH is proposed to fill this gap by focusing on "linguistically hard-to-describe" cues.

### Strengths
1. The paper clearly identifies a weakness in existing audio benchmarks. The "caption-only" experiment (Figure 1) provides empirical evidence that current benchmarks often test text-level semantics rather than fine-grained audio perception.
2. The paper's most significant contribution is identifying and proving the insufficiency of standard audio preprocessing in LALMs. The "additive inverse" experiment in Figure 3 is a simple and effective demonstration that a model's spatial reasoning capabilities are non-existent if it just averages channels to mono.
3. The analysis provides clear takeaways for the community, such as the need for models that natively process multi-channel audio and the severe limitations of open-source models in multi-audio grounding (as shown in Figure 9 ablation).

### Weaknesses
1. My primary concern is with the "channel-wise input" proposed for spatial reasoning. While this is a clever workaround for the "mono-averaging" problem, it fundamentally changes the task. Instead of evaluating native spatial audio perception, this method tests a model's ability to reason about two separate mono audio streams guided by an explicit textual prompt (e.g., "Audio 1 is the left-ear channel and Audio 2 is the right-ear channel"). This confounds auditory intelligence with text-based reasoning and an understanding of the experimental setup. It does not truly measure the model's ability to process and fuse interaural cues (ITD, ILD) from a single binaural stream. The paper acknowledges models are not trained on multi-audio inputs, but I think this workaround is a significant compromise.
2. The temporal reordering task relies on "strong sequential uniqueness". This is robust for tasks governed by physics (e.g., Doppler effect , fluid dynamics). However, I am less convinced about the "Daily Scene Scripts" and "Event-Triggered Consequences"  subcategories. For example, in the teeth-brushing case study (Figure 12) , the "correct" order is given as brushing -> spit foam -> turn on tap/spit. However, an alternative like GPT-4o Audio's turn on water -> brush -> rinse/spit is just as logical and common. This suggests "logical universality"  is not as high as claimed.
3. The foundational perception tasks use sine waves for non-spatial attributes. While this provides control, its relevance to perceiving pitch or loudness in complex, real-world sounds (the paper's motivation) is unclear. More importantly, the spatial perception tasks are simulated in rooms with simple source sounds ("alarm," "applause," "telephones"). A critical aspect of real-world spatial hearing is localization and tracking in the presence of diffuse background noise and significant reverberation. It is not clear from the paper or appendix if the benchmark tests performance under varying SNRs or reverberation times (RTs), which are standard parameters in auditory scene analysis.
4. Missing ablation on spatial realism: how do models perform if reverberation or stereo cues are removed?
5. Section 5 implies that STAR-Bench measures general “audio reasoning ability,” but all examples rely on synthetic or controlled sounds. Thus, the benchmark may not generalize to naturalistic auditory scenes (e.g., urban soundscapes, conversational dynamics). At least testing transfer performance would strengthen external validity.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces STAR-Bench, a benchmark to evaluate 4D Audio Intelligence - reasoning over time (1D) and 3D space—via two levels: (a) Foundational Acoustic Perception (six attributes: pitch, loudness, duration, azimuth, elevation, distance; tested in absolute and relative regimes) and (b) Holistic Spatio-Temporal Reasoning (temporal segment re-ordering for continuous/discrete processes; spatial reasoning over static localization, multi-source relations, and dynamic trajectories). The dataset has 2,353 questions and is built through a four-stage pipeline (AI-assisted filtering, human annotation, expert validation) drawing on synthetic audio and real-world corpora (e.g., FSD50K, Clotho, STARSS23). Evaluation is multiple-choice with “robust” perturbations (AA/ACR). Results show humans outperform all models (e.g., ~88% temporal), while the best model (Gemini-2.5 Pro) reaches ~49.6% MA, and a larger drop occurs when answering from captions only, supporting the claim that STAR-Bench targets linguistically hard cues.

### Strengths
- Clear problem framing ('audio 4D intelligence) and a structured task design that disaggregates perception, temporal reasoning, and spatial reasoning.
- Useful diagnostic split: absolute vs. relative perceptual tests; temporal re-ordering; spatial subtasks (single-source localization, multi-source relations, dynamic trajectories).
- Strong curation pipeline with AI filtering, human annotation, and expert validation; explicit use of public datasets + simulated audio for coverage and control. 
- Robustness intent (AA vs. ACR; circular option shuffling; segment order perturbations) is a step forward relative to one-shot MCQ protocols. 
- Empirical evidence that 'caption-only' shortcuts collapse on STAR-Bench more than on MMAU/MMAR, supporting the target of non-linguistic cues. (Claim supported in text/figures.)

### Weaknesses
- Everything is framed as multiple-choice with string-match grading. No open-ended QAs, while they are more real world.
- A large portion of audio comes from widely used corpora (FSD50K, Clotho, STARSS23); many models likely pretrained on them. The authors argue the task formulation is novel (re-ordering, spatial relations), but clip-level memorization of events/timbres is still possible.
- The paper doesn't quantify inter-rater agreement, item rejection rates, or ambiguity sources.
- In table 1, the authors claim that previous benchmarks like MMAR and MMAU-Pro, do not have spatial reasoning questions, which is wrong. MMAR also has multi-audio questions.
- MMAU, MMAU-Pro and MMAR also contains Temporal Reasoning questions, what does "Deep" signify in the task name in table 1?
- With the recent advancements in Large Audio Language Models where the models are capable of reasoning over all three modalities of audio - sound, speech and music, this benchmark only focuses on sounds which is limiting. Although the authors explicitly frame STAR-Bench as a move away from traditional ASR/captioning style evaluations toward 4D (spatio-temporal) acoustic reasonin - another signal that speech/music tasks per se are out of scope, I strongly feel there can be multiple tasks framed around speech in spatio-temporal setting.

### Questions
See the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces STAR-Bench which measures Audio 4D Intelligence (reasoning over sound dynamics in time and 3D space). They combine a Foundational Acoustic Perception setting with a Holistic Spatio-Temporal Reasoning setting that includes segment reordering for continuous and discrete processes and spatial tasks spanning static localization, multi-source relations, and dynamic trajectories. Unlike prior benchmarks where caption-only answering reduces accuracy slightly, STAR-Bench induces far larger drops (-31.5% temporal, -35.2% spatial), evidencing its focus on linguistically hard-to-describe cues. They also provide a comprehensive evaluation of 19 LALMs/OLMs.

### Strengths
- Introduces and benchmarks multi-audio segment reordering and stereo spacial reasoning which has been ignored by the previous benchmarks
- Proper coverage of non-spatial attributes (Loudness, Pitch, Duration) and spatial attributes (Azimuth, Elevation, Distance)
- Detailed error analysis on why models dont do well on the proposed benchmark
- Thorough reporting of AA and ACR metrics to test the model's reliability
- Really like the finding "a fundamental inability to effectively compare, ground, and integrate information from multiple audio inputs"

### Weaknesses
- missing details of AI-Assisted Automated Filtering. What exactly are we filtering using gemini 2.5 pro
- Spatial data is synthetic and does not represent real world use cases.
- Fig 8 not readable
- Not sure if the baselines support spatial audio. Analysis on those models would not provide beneficial information
- Support for spatial audio in these models can lead to huge jumps in model performance on the benchmark which questions the difficulty of the benchmark

### Questions
- Do you think gemini 2.5 pro captioning stage filters the data points to be more biased towards the current capabilities of the models? 
- How easy/difficult is it to hill climb on this benchmark for audio models?
- Do the baselines actually support spatial audio?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
A very topical benchmark for large audio language models, which adds basic audio perception modalities, and explores categories where linguistic descriptions are hard.

### Strengths
A clear way of generating spatial audio and temporal reasoning benchmarks is described and implemented. The results show clear lack of perception of current LALMs on tasks requiring these skills. This is a valuable and distinct addition to the plethora of benchmarks coming out

### Weaknesses
The audio scenes were generated binaurally, but this restricts the range of applications of the benchmark to perhaps humanoid listeners. Also, the HRTF used was a Kemar one, which is somewhat limited; and the scene rendering was also a bit limited. Nonetheless a good beginning. The models are of course set up to fail in this testing.

The spatial audio tasks are relatively hopeless, and analysis of the models was not too much possible To an extent, I would have liked more detailed analysis of the temporal reasoning tasks. An ablation I would like to see is if the models perform better if the  with just single channel presentation of these events to the models.

### Questions
How would you generalize the presentation of spatial audio to models, beyond Kemar? How would you accommodate ambisonics? It would have been good to provide some example sounds.

### Soundness
3

### Presentation
3

### Contribution
3
