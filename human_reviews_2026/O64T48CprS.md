# SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 6

## Abstract
The rich and multifaceted nature of human social interaction, encompassing multimodal cues, unobservable relations and mental states, and dynamical behavior, presents a formidable challenge for artificial intelligence. To advance research in this area, we introduce SIV-Bench, a novel video benchmark for rigorously evaluating the capabilities of Multimodal Large Language Models (MLLMs) across Social Scene Understanding (SSU), Social State Reasoning (SSR), and Social Dynamics Prediction (SDP). SIV-Bench features 2,792 video clips and 8,792 meticulously generated question-answer pairs derived from a human-LLM collaborative pipeline. It is originally collected from TikTok and YouTube, covering a wide range of video genres, presentation styles, and linguistic and cultural backgrounds. It also includes a dedicated setup for analyzing the impact of different textual cues—original on-screen text, added dialogue, or no text. Our comprehensive experiments on leading MLLMs reveal that while models adeptly handle SSU, they significantly struggle with SSR and SDP, where Relation Inference (RI) is an acute bottleneck, as further examined in our analysis. Our study also confirms the critical role of transcribed dialogue in aiding comprehension of complex social interactions. By systematically identifying current MLLMs' strengths and limitations, SIV-Bench offers crucial insights to steer the development of more socially intelligent AI.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SIV-Bench, a video benchmark for evaluating MLLMs on social interaction understanding across three dimensions: Social Scene Understanding (SSU), Social State Reasoning (SSR), and Social Dynamics Prediction (SDP). The benchmark contains approx 2800 videos from TikTok/YouTube representing 14 relationship types, with approx 8800 QA pairs generated via human-LLM collaboration. Experiments on 10+  models reveal that while models handle SSU adequately, they struggle with SSR and SDP.

### Strengths
* Important problem: Social interaction understanding is relatively underexplored in video benchmarks
* Scale and diversity: 2,792 videos across multiple genres, styles and languages
* Comprehensive evaluation: 10+ models tested (commercial + open-source) with detailed results
* Thoughtful dataset construction: Human-LLM collaborative pipeline with consensus validation, multiple subtitle conditions, relationship-centric 
* Reproducibility efforts: Detailed prompts, annotation guidelines in appendices

### Weaknesses
1. Consensus based selection and only gold labels provided: social reasoning inherently has multiple views/biases, i.e., different interpretations are often valid and should be included in the dataset; further, butchering the dataset to contain only unanimous decisions makes it less valuable (bcs you only focus on the simple unambiguous cases). This might explain also that SSU→SSR→SDP does not perform increasingly much worse (as expected).
2. Absence of statistical testing - you confidence interval should be around .9% so differences below that should be treated with care - even if you assume independence among questions and you use 8.8K significance is around .3% again be careful when making claims.
3. No human baseline so we are not able to contextualize model scores (is 76% good or poor?) - for these social tasks humans still serve as the benchmark
4. Unvalidated task hierarchy - lacks some factor analysis for the SSU→SSR→SDP claimed chain or regression testing or correlation between subtasks. I would also like to see the microskill decomposition of the tasks/subtasks.
5. Some mLLMs are used for both generating the data and evaluating performance. This is ok - it just needs to be clarified and carefully disclaimed.
6. Important competing datasets are downplayed or never mentioned, e.g., Social Genome - need to be clear on what is novel here and do a detailed comparison on why this dataset is novel - claims have to be downplayed a bit in the text.

### Questions
1. What is human performance on SIV-Bench?
2. Can you provide statistical significance tests and confidence intervals for Table 2 results?
3. Why does Chain-of-Thought show minimal improvement (Table 11)? 
4. What proportion of QAs were discarded due to disagreement?
5. Can you provide soft labels or multiple valid answers where annotators disagreed?

### Soundness
3

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
4

### Summary
The paper introduces SIV-Bench, a new video QA benchmark designed to evaluate the ability of Multimodal Large Language Models (MLLMs) to understand and reason about complex human social interactions. The dataset is diverse, covering various genres, presentation styles, and cultural backgrounds. The authors conducted comprehensive experiments on leading MLLMs and found that while models perform well on the foundational SSU (Social Scene Understanding) tasks, they significantly struggle with the more complex SSR (Social State Reasoning) and SDP (Social Dynamics Prediction) tasks.

### Strengths
- The dataset's scale and diversity are significant strengths. The curation across 14 distinct social relationship types, multiple languages, and varied cultural contexts provides a robust foundation for social reasoning evaluation.
- The paper provides a thorough evaluation of a wide range of MLLMs. The inclusion of a comparative analysis of how audio and subtitles affect performance is also interesting and valuable.

### Weaknesses
- The reliance on an automated pipeline (LLM-generated QAs and distractors) raises concerns about the benchmark's true difficulty and potential for shortcut learning. A large portion of the questions (3,273) are chosen from those that are answered correctly by Gemini-2.0-Flash, Gemini-2.0-Pro, and GPT-4o-mini. However, this also indicates that the questions are easier to answer. The paper also mentions limited adversarial filtering (e.g., removing questions solvable without video), which does not account for potential biases from the distractor generation model that cause questions to be easily answerable with superficial visual cues. Finally, while the authors used GPT to normalize option style, this falls short of more rigorous adversarial debiasing modules to mitigate superficial vision-text shortcuts. 
- The above concern about dataset difficulty is supported by the high accuracy of SOTA models, suggesting the benchmark may already be approaching saturation. For example, Gemini-2.5-Pro achieves over 90% accuracy on the SSU category and 70-75% on the SSR and SDP tasks. 
- In light of these high accuracies, the authors did not provide a human baseline, making the numbers hard to interpret. A 75% accuracy on could be near human-level performance (indicating limited room for improvement) or it could be well below it (indicating the benchmark is valid and challenging).

### Questions
- While I appreciate the dataset's scale and diversity, I am concerned about the quality and saturation of the dataset that could be due to the automated generation pipeline. Could the authors provide a human baseline as reference to ensure that the benchmark is not saturating?
- The 14 distinct social relationship types are an important component of the benchmark's design. Do the authors have an analysis on model performance broken down by these 14 relationship types?

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
3

### Summary
SIV-Bench is a video benchmark for social interaction understanding which is organized along three capabilities: Social Scene Understanding (SSU), Social State Reasoning (SSR), and Social Dynamics Prediction (SDP). It contains 2,792 short videos collected from TikTok or YouTube, and 8,728 multiple-choice QAs built through a human–LLM collaborative pipeline. The benchmark is relation-centric, grounded in Fiske’s Relational Models Theory (4 model families instantiated as 14 relation types), and ships three subtitle conditions—origin, +sub (transcribed & translated dialogue), and –sub (on-screen text removed)—to study the role of linguistic cues. Experiments cover open/closed-source MLLMs; top systems excel at SSU but struggle on SSR (especially Relation Inference) and, to a lesser extent, SDP; subtitles and audio help, with Gemini-2.5-Pro peaking at 76.50% overall with +sub.

### Strengths
1. Relation-centric design grounded in Fiske’s theory with fourteen relation types; aligns mental state, perception, and dynamics to the social context.
2. Diverse, real-world videos with multilingual presence; three subtitle conditions enable language-cue studies, which are controlled.
3. The work is also equipped with video-dependence check, model-consensus filtering, and difficulty curation, which contribute a lot.
4. Last but not least, standardized prompting or parsing, broad model set, and analyses (subtitle or audio ablations; fine-grained task radar; failure patterns).

### Weaknesses
1. Video-capable models get raw videos; image-only models receive 16 frames, risking budget-driven gaps; per-model frame/FLOP parity wasn’t normalized.
2. Human verification is described, but inter-annotator agreement ( for instance, κ) isn’t reported; this matters for subtle social labels.
3. Heavy use of LLMs for QA creation, filtering, and option standardization could induce stylistic priors; leakage/over-templating risks need auditing.
4. Accuracy seems to be the only metric; also, no per-relation calibration curves or human baselines for comparison on SSR/SDP are provided. In addition, failure analysis is qualitative.

### Questions
1. Can you provide an ablation with fixed frames/FLOPs/tokens across all models (including closed-source) to isolate modeling effects from budget differences?

### Soundness
3

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
The paper presents SIV-Bench, a video benchmark for social interaction understanding that evaluates MLLMs along three dimensions: Social Scene Understanding (SSU), Social State Reasoning (SSR), and Social Dynamics Prediction (SDP), further decomposed into 10 fine-grained tasks (e.g., action/expression recognition, relation/intent/emotion inference, factual/counterfactual prediction). The dataset contains 2792 TikTok/YouTube clips and 8792 MCQs created via a human–LLM collaborative pipeline, with built-in controls to analyze the role of language through three subtitle settings: original, +sub (transcribed/translated dialogue added), and −sub (on-screen text removed). Experiments using VLMEvalKit show that models are relatively strong on SSU but struggle on SSR/SDP with Relation Inference the most consistent bottleneck.

### Strengths
- While many video benchmarks are action- or object-centric (e.g., classic action recognition and general video QA suites), human social interaction—relations, implicit mental states, and interaction dynamics—remains underrepresented. SIV-Bench explicitly fills this gap by defining a people-centered evaluation space (SSU/SSR/SDP), curating original, relation-typed videos (14 relation types), and providing subtitle/audio signals to probe multimodal social reasoning; this dataset definition and collection are, in themselves, a substantive contribution.

- Clear empirical takeaways. Models are relatively strong on SSU but struggle on SSR/SDP, with Relation Inference the most consistent bottleneck; transcribed subtitles measurably help.

### Weaknesses
- While the multiple-choice QA format simplifies evaluation and boosts reliability, it likely underrepresents the open-ended, interactive social reasoning required by agents. Expanding the benchmark with generative or dialog-based tasks (e.g., free-form rationales reasoning, intent explanations, multi-turn interactions) would provide a deeper probe of interactive social understanding.

### Questions
- Several SSR failures involve decoding why people react as they do. The SMILE dataset [1] introduces Video Laugh Reasoning and text explanations for why people laugh in social videos. Could SIV-Bench incorporate this kind of reasoning task to probe this facet of social affect reasoning?


[1] https://arxiv.org/pdf/2312.09818

### Soundness
3

### Presentation
3

### Contribution
3
