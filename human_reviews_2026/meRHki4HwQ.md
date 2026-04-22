# MDAR: A Multi-scene Dynamic Audio Reasoning Benchmark

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 2

## Abstract
The ability to reason from audio, including speech, paralinguistic cues, environmental sounds, and music, is essential for AI agents to interact effectively in real-world scenarios. Existing benchmarks mainly focus on static or single-scene settings and do not fully capture scenarios where multiple speakers, unfolding events, and heterogeneous audio sources interact. To address these challenges, we introduce MDAR, a benchmark for evaluating models on complex, multi-scene, and dynamically evolving audio reasoning tasks. MDAR comprises 3,000 carefully curated question–answer pairs linked to diverse audio clips, covering five categories of complex reasoning and spanning three question types. We benchmark 26 state-of-the-art audio language models on MDAR and observe that they exhibit limitations in complex reasoning tasks. On single-choice questions, Qwen2.5-Omni (open-source) achieves 76.67% accuracy, whereas GPT-4o Audio (closed-source) reaches 68.47%; however, GPT-4o Audio substantially outperforms Qwen2.5-Omni on the more challenging multiple-choice and open-ended tasks. Across all three question types, no model achieves 80% performance. These findings underscore the unique challenges posed by MDAR and its value as a benchmark for advancing audio reasoning research. Code and benchmark can be found at https://anonymous.4open.science/r/MDAR-8981.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a benchmark, MDAR, to evaluate audio reasoning capabilities of LALMs. 
MDAR emphasizes complex tasks with multi-scene and potentially dynamically evolving contexts. 
MDAR contains 3K curated samples for 5 categories and 3 question types. 
The paper evaluates MDAR on 26 SOTA LALMs and find these models have limitations on these tasks.

### Strengths
- Overall, the paper aims at studying a very important problem in the audio understanding domain. It remains a challenge to evaluate and enhance the complex reasoning abilities of current LALMs, and this paper proposes a solid benchmakr for more holistic evaluation of this area. 

- The test data construction pipeline is very clear, transparent, and reproduce-able. The taxonomy of different types of test samples (different categories and question types) could benefit the community to conduct more fine-grained analysis and evaluation of their models.  These also complement existing benchmarks (e.g. MMAU and MMAR) that have different taxonomies (e.g. mainly on audio domains). 

- The baseline evaluation is quite extensive. The paper evaluates many open and closed models and the results are valuable. In addition to just overall accuracies, the paper further conducts detail analysis such as error types in Fig 6. The results are very valuable as it tells us that most mistakes are related to mis-reasoning and some are mis-perception. These could benefit further RL design choices of LALM post training.

### Weaknesses
- As stated in L215, test samples are all from Chinese films and therefore the distribution can be biased. This could limit the versatility of the proposed benchmark, and may also be unfair to LALMs that are only trained on English corpus / data. 

- While Fig 3 shows the distribution and other strengths of the proposed MDAR, it is hard to quantitively prove how much better MDAR is compared to existing benchmarks like MMAU and MMAR. To me it sounds more like a complement to those benchmarks. I suggest the authors to conduct a rigorous human study on how MDAR is better than prior benchmarks, despite that MDAR covers more skills and contains longer audio. Similarly, the paper claims the samples are complex, multi-scene, and the context can be dynamic, but these require rigorous proof. 

- The evaluation of open-ended samples is based on LLM-as-a-judge. This can be unstable and unreliable in terms of rigorous comparison. 

- The experimental analysis is not extensive at all. While there are main accuracies and error analysis, there is no in-depth analysis on different types of errors. For instance, why do the models have these errors? What are the causes of these errors -- capacity issues, hallucination, randomness when responding, or others? It worths studying these deeper information so that the community could further improve their models. 

- Following the above point, the paper does not indicate readers how to build better LALMs. Should we use better SFT or better RL? Should we scale the base LLM or not? What types of data to curate to fill the gap? Does Fig 6 indicate we should blend more text-specific data? How to design efficient methods for reliable understanding of longer audio? I would expect the paper to show some route for building better LALMs -- while the ultimate goal is always to have more and higher-quality data in the end -- the benchmark analysis should guide us to reach the goal more efficiently and effectively with principled recipes and experiences.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces MDAR, a new benchmark designed to evaluate multi-audio, multi-scene reasoning abilities in LALMs. Unlike prior benchmarks that focus on single-audio or localized perception tasks, MDAR tests whether models can reason compositionally across multiple audio scenes involving speech, environmental sounds, and music. The benchmark aims to push beyond simple perception toward holistic scene understanding, including temporal order, causal interactions, and semantic relationships between events. The final dataset contains 8,524 QA pairs across five task types: causal reasoning, temporal reasoning, multi-speaker dialogues, cross-scene summarization, and auditory anomaly detection.

### Strengths
- To the best of my knowledge, this is the first benchmark to target multi-scene dynamic reasoning, a previously unaddressed challenge for LALMs. MMAU-Pro released recently addresses this, but I understand it may be parallel work.
- Well-structured taxonomy of reasoning skills.
- Balanced synthetic-real composition pipeline ensuring diversity and controlled difficulty.
- The dataset also has an open and MCQ part, which is exciting and important.
- Comprehensive evaluation across open- and closed-source systems.

### Weaknesses
- Heavy reliance on synthetic compositions may limit real-world generalization.
- I do not have many weaknesses to point out.
- Is the dataset in Mandarin? I see the benchmark and looks in Supplementary and looks like its all Mandarin?

### Questions
- I am curious how multi-audio evaluation was done on models like AF2 and AF3 and maybe Qwen-3-omni, I did not find a straightforward way to do this.

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
This paper introduces MDAR (Multi-scene Dynamic Audio Reasoning), a benchmark targeting complex, real-world audio reasoning across three complementary test suites: MDAR-main (1,500 single-choice QAs across five categories), MDAR-open (500 open-ended QAs scored by an LLM judge), and MDAR-multi (825 multi-audio, multi-answer QAs emphasizing cross-clip reasoning). Data are drawn from ~20–40s segments of Chinese movies with speaker diarization and global clustering; FunASR supplies transcripts; Gemini-2.5-Pro and Qwen-2.5-VL generate multimodal descriptions and help author questions/distractors under an expert-designed taxonomy and multi-step QA screening. MDAR reports diverse metrics (regex-based accuracy for single-choice, LLM-judge 0–10 for open-ended, and EM/JI/Precision/Recall for multi-audio). Baselines show the task is hard: best non-cascaded model in MDAR-main (Qwen2.5-Omni) reaches ~76.7%; MDAR-open tops out near 7.46/10; ablations (e.g., replacing audio with Gaussian noise) substantially degrade accuracy, underscoring audio dependence.

### Strengths
- The three suites (single-choice, open-ended, multi-audio) cover perception --> understanding --> cross-clip reasoning, with five high-level categories (scene, social, event, temporal, safety).
- Well-defined metrics: Regex-based accuracy for MDAR-main, an LLM-judge rubric for MDAR-open, and EM/JI/Precision/Recall for MDAR-multi are specified.
- The data generation pipeline is strong and well thought out.

### Weaknesses
- One of my biggest concerns is that all the QA are in Chinese in the supplementary material, but the examples show in the paper are in English. If the questions are fed to models like Audio Flamingo 3, which are not trained on QA pairs in the specific language, that might not be a fair evaluation. Also, the authors should explicitly mention that the QAs are in Chinese.
- The evaluated methods score significantly higher than they do on the benchmarks already released, which questions the difficulty of the benchmark.
- Since, most of the benchmarks is derived from movie clips, it would benefit from having questions around music (background/foreground).
- No correlation is shown between the Human & LLM as a judge. What is the correlation (e.g., Spearman/Pearson, Krippendorff’s α) between the LLM-judge scores and a human panel across categories?
- Mixing accuracy (closed-set), judge scores (open-ended), and set metrics (multi-audio) complicates aggregate interpretation; there’s no unified “overall” score or calibration across metrics.

### Questions
See weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces MDAR which is a benchmark for multi-scene, dynamically evolving audio reasoning. It contains 3000 question-ans pairs spanning five different categories (Scene Understanding, Social Relationships, Event Reasoning, Temporal Reasoning, Anomaly and  Safety) and three formats: single choice on a single audio (also called MDAR-main), open-ended responses (MDAR-open), and (iii) a new multi-audio, multiple-choice setting (MDAR-multi). The data prep pipeline takes 500 Chinese films, samples 20 - 40s clips, applies speaker diarization and global segment clustering, then uses ASR plus LLMs (Gemini/Qwen) with expert screening to finalize questions and distractors.

### Strengths
- Dynamic, multi-scene coverage across five reasoning axes and three formats
- Longer and information dense clips (average 25.11s) compared to the prior work, raising temporal reasoning difficulty in the benchmakr
- Comprehensive evaluation of 26 models
- Systematic evaluation of both perceptual and high-level reasoning abilities
- Highlights key areas for improvement in next-generation audio reasoning agents

### Weaknesses
- Chinese movies as the metadata source biases the benchmark towards a particular accent/way of speaking. The benchmark should be diverse in the data to avoid biased results
- The SOTA performance on the benchmark closing in on 80%
- (General comment) The paper has a low excitement factor and raises a concern of lack of novelty with it being yet another benchmark. Even the paper/analysis/diagrams format exactly follows MMAU
- MMAU pro has not been included in the comparison with existing benchmarks
- error analysis has been done by Gemini-2.5-flash - ideally the errors should be identified and categorised by a human to get an accurate analysis. what prompt was used for this analysis? what are the examples of gemini-2.5-flash failing to do the required job?

### Questions
1. How do you define complex reasoning and how is it different from reasoning? The paper mentions the word complex multiple times but it needs to define what it means for audio
2. How difficult is it to improve performance on the benchmark?
3. fig 3b total questions don't sum up to 3000. can you confirm?
4. the paper selects best prompt for accuracy, how sensitive are rankings to different prompts? 
5. how much would the human eval score be?

### Soundness
3

### Presentation
3

### Contribution
2
