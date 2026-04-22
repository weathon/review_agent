# RIVER: A Real-Time Interaction Benchmark for Video LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Multimodal large language models (MLLMs) have demonstrated impressive capabilities, yet nearly all operate in an offline paradigm, hindering their real-time interactivity. To address this gap, we introduce the Real-tIme intERaction Benchmark for Video LLMs (RIVER Bench), designed for evaluating their real-time interaction ability with humans through perceiving the streaming videos. RIVER Bench introduces a novel evaluation framework comprising Retrospective Memory, Live-Perception, and Proactive Response tasks, closely mimicking interactive dialogues with humans rather than understanding the entire videos at once. We conduct detailed annotations using videos from diverse sources and varying lengths, and precisely defined the real-time interactive format. Evaluations across various model categories reveal that while offline models perform well in single question-answering tasks, they struggle with real-time processing. Addressing the limitations of existing models in online interaction paradigm, especially their deficiencies in long-term memory and future perception, we proposed a general improvement method that enhances models’ flexibility in real-time interaction. We believe this
work will significantly advance the development of real-time interactive video understanding models and inspire future research in this emerging field. The code and data will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
RIVER Bench proposes a benchmark for real-time video interaction, shifting evaluation from offline QA to online, temporally grounded tasks. It formalizes three competencies—Retrospective Memory, Live-Perception, and Proactive Anticipation—with precise query, cue, and response timings. Built from curated long-video sources, it spans 1,067 videos and 4,278 questions, enabling forgetting-curve and latency–accuracy analyses. The authors also adapt offline MLLMs via sliding windows plus a long–short memory module and release training data that improves proactive responses. Experiments across closed- and open-source models expose gaps in online processing and show gains from RIVER-based fine-tuning.

### Strengths
**1. Precise task formalization and timing.** Clear definitions of Retro-Memory, Live-Perception, and Pro-Anticipation with explicit query/cue/answer timestamps yield faithful temporal grounding and enable memory-decay and response-latency analyses. 

**2. Broad coverage with actionable baselines.** 1,067 videos and 4,278 questions assembled from multiple long-video sources; results reported across closed/open models with consistent prompts and a streaming setting support fairer comparisons. 

**3. Practical online adaptation.** A sliding-window plus long–short memory design and a targeted training set improve proactive response metrics and repurpose offline MLLMs for streaming use.

### Weaknesses
**1. Novelty of method.** The “make-offline-models-online” recipe (sliding windows + long–short memory with nearest-neighbor averaging) feels incremental amid prior memory-cache approaches for streaming video.

**2. Comprehensiveness of the benchmark.** Despite 1,067/4,278 scale, sources cluster around a few public datasets (e.g., LVBench, LongVideoBench, Ego4D, QVHighlights), leaving underexplored domains like AR navigation, robotics, multi-party meetings, or broadcast sports—potentially limiting ecological validity. 

**3. Audio omission.** The current release excludes audio, a core signal for live interaction (speech, alarms, ambience).

**4. Evaluator reliance and metric scope.** Open-ended answers are judged by a single LLM (Qwen2.5-72B), and Pro-Anticipation timing is reduced to response-within-window (“Res Acc”). Broader, judge-agnostic metrics (timing error, hesitation penalties, continuous narration quality) could reduce bias. 

**5. Closed-source comparability.** GPT-4o/Gemini are limited to 50 frames while “online-ized” open models run at 1–4 fps; heterogeneous context budgets and sampling policies risk conflating interface limits with modeling capacity. 

**6. Data generation and leakage risk.** LLM-synthesized anticipatory questions/distractors, though filtered, may import language priors; publishing templates and rejection sets would help audit artifacts and overfitting. 

**7. Robustness to timestamp noise.** The benchmark assumes precise cue/query/answer times; stress-tests under timestamp jitter, dropped frames, or (future) ASR lag would better mirror deployment. 

**8. Scalability and latency costs.** The long-term memory keeps 16 slots mirroring short-term token size. Growth behavior, pruning policy, and on-device latency/compute trade-offs remain under-explored.

### Questions
**1.** How would adding synchronized audio (speech and environmental sounds) alter forgetting curves and response timing? Which fusion (early vs. late; streaming ASR alignment) best supports Pro-Anticipation? 

**2.** Can the fixed 16-slot long-term memory evolve into content-adaptive eviction/merge policies or learned key-value compression? What are the latency/recall impacts over 1-hour horizons? 

**3.** Do memory and anticipation degrade differently across egocentric vs. third-person sources, scene types, or demographics? Could per-subset breakdowns and bias diagnostics guide responsible deployment? 

**4.** What is agreement between different LLM judges and humans for open-ended scoring, and could pairwise preference tests or reference-free metrics capture timing quality without a single judge model? 

**5.** Beyond coarse time bins, can item-response curves and psychometric difficulty by cue type (fine-grained, causal, background) support curriculum design and targeted fine-tuning? 

**6.** How do results transfer to embodied/assistive agents where interruptions, safety stops, or missed cues have costs? Can RIVER simulate intervention thresholds and recovery from misfires? 

**7.** Will future releases broaden domains (AR, robotics, meetings, sports) and ship licenses, prompt templates, and rejection sets so the community can audit and reliably reproduce results?

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
3

### Summary
This paper presents RIVER Bench (Real-tIme Video intERaction Bench), a new benchmark designed to evaluate online and real-time video understanding in multimodal large language models (MLLMs). Unlike traditional offline settings, RIVER Bench defines three interactive task types — Retrospective Memory, Live Perception, and Proactive Anticipation — to simulate dynamic, conversational interactions with ongoing video streams. Comprehensive annotations and evaluations reveal that while existing offline MLLMs perform well on static QA tasks, they struggle with real-time comprehension and long-term consistency. To address this, the authors propose a general enhancement method that improves model adaptability and responsiveness in real-time scenarios, setting a foundation for future interactive video models.

### Strengths
1. The paper fills a clear gap by introducing RIVER Bench for real-time video interaction, moving beyond the traditional offline paradigm. Its design with Retrospective Memory, Live Perception, and Proactive Anticipation tasks realistically mimics dynamic, interactive scenarios.

2. The dataset is diverse and well-annotated, combining multiple video sources with fine-grained temporal labeling and strong quality control, ensuring high reliability.

3. The experiments are comprehensive, covering various model types and introducing well-designed metrics (e.g., response localization, memory decay) that align with real-time understanding needs.

4. The method is simple yet effective, using a sliding-window and long-short memory mechanism to adapt offline models for online use, showing clear performance gains (+11.28% on RIVER).

### Weaknesses
1. RIVER Bench only supports video-text interaction, not including audio, which is crucial for real-time tasks like voice navigation or human-robot interaction. While this is mentioned as a limitation, it would be helpful to test and report ASR performance, which would increase the benchmark’s practical value.

2. The data primarily comes from Ego4D-Narration, focusing on simple, static tasks like desk operations and furniture organization, and lacks more complex dynamic scenarios such as traffic flow prediction or industrial fault detection. This could limit the benchmark’s generalizability to real-world applications.

3. Many experimental settings lack ablation studies, such as the impact of memory compression strategies or the number of memory slots. Additionally, inference latency (e.g., per-frame processing time, response delay) is not reported, which is crucial for real-time interaction. Key details like frame sampling rates and the computational cost of memory modules are missing, making it hard to assess the feasibility of this method in actual deployments.

### Questions
1. How is the response time window for the Proactive Anticipation task defined (e.g., fixed ±1s, or dynamically adjusted based on event duration)? Will the authors consider adding an uncertainty prediction metric (e.g., model confidence in its predictions) to assess decision-making in ambiguous scenarios?

2. What is the proportion of samples in the Retrospective Memory task for the "extremely long duration" (>3600s) category?

3. The experiments show that all models perform poorly on causal clue (CC) questions. What specific technical directions do the authors suggest for improving the integration of visual perception and event attribution in such tasks?

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
3

### Summary
This paper proposes RIVER Bench, a benchmark for online video interaction evaluating multimodal LLMs (MLLMs) on three temporally grounded competencies: live-perception, retrospective memory, and proactive anticipation. The authors formalize an online video-text-to-text task with explicit timestamps for query, clue, and response. They offer task taxonomies, dataset statistics, and metrics that jointly assess accuracy, timeliness, and latency–accuracy tradeoffs. RIVER aggregates and restructures items from Vript-RR, LongVideoBench, LVBench, Ego4D, and QVHighlights into precisely timed QA pairs and streaming tasks. The evaluations make a comparison between public and closed-sourced MLLMs, and demonstrate sizable gaps between offline QA success and real-time performance.

### Strengths
1. The paper creates an accurate online task formalization with retro/live/pro-anticipation split and timing semantics. In addition, the windowed formulation ties the accuracy to when the answer is generated.
2. Curated and broad construction across long videos, equipped with explicit filtering to mitigate language-only priors and ambiguous items.
3. Operational online protocol that makes many offline models evaluable in real time, which enables informative cross-family comparisons.
4. Comparatively useful metrics and analyses, which include duration-stratified forgetting curves, cue-type breakdowns and proactive Res Acc.

### Weaknesses
1. OE scoring depends on Qwen2.5-72B; prompts, thresholds, and sensitivity analyses are not deeply reported. The Res Acc window width and tolerance are not exhaustively justified; in addition, user-centric latency–utility tradeoffs are not evaluated as well.
2. From my perspective, even though the pipeline filters items that are language-answerable, deep LLM participation risks unintended stylistic mimicry and inherent bias.. More transparent human IAA and QA metrics would be quite helpful.
3. The non-public models use the determined 50 frames the meanwhile, others utilize 16 frames or 1 frame per second streams; the differences in budgets are able to confound absolute rankings. More apples-to-apples ablations would strengthen the claims to a great extent.

### Questions
1. Can you add a study equating tokens/frames/FLOPs across models (e.g., standardizing to a fixed #frames or fps) to isolate modeling differences from budget differences?

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
2

### Summary
The paper targets online video interaction for MLLMs and proposes RIVER Bench, a benchmark and protocol to evaluate three competencies under streaming conditions: Retrospective Memory (recall past events; forgetting curves), Live-Perception (time-sensitive understanding with latency–accuracy trade-offs), and Proactive Anticipation (detect/forecast future states with correct timing). RIVER reformulates and precisely timestamps items from multiple sources (Vript-RR, LVBench, LongVideoBench, Ego4D, QVHighlights), yielding: Retro-Memory (≈1.5k MCQs across four recall intervals), Live-Perception (≈0.4k items), Pro-Anticipation (≈1.2k “stream” continuous narration + ≈1.4k “instant” trigger items). Quality control uses LLM + human filtering and distinctive time-anchored events. Experiments cover commercial models, “native online” models, offline models adapted via a sliding window + long/short-term memory wrapper (16 slots with NN-averaged compression), and a fine-tuned online model (SiGLIP encoder + LLaMA3-8B with LoRA; LM + streaming losses). Metrics include MC exact match, open-ended judging via Qwen2.5-72B, and a response accuracy (Res Acc) that credits answers emitted within a ground-truth time window. Findings: offline models that excel at single-turn QA struggle online; the proposed online adaptation narrows the gap and improves pro-anticipation (+11.28% over baseline); memory modules flatten forgetting by ~12%; causal-cue retro-memory is hardest.

### Strengths
•	Clear, timely problem: shifts evaluation from offline video QA to interactive, streaming settings with explicit timing of query/cue/response.

•	Three-facet design: jointly measures recall, live perception, and anticipation, and ties performance to the temporal gap \Delta (forgetting/anticipation curves).

•	Protocol precision: items carry exact timestamps; “instant” vs “stream” anticipation reflects real interaction patterns (trigger vs continuous narration).

•	Methodological baselines: shows how to wrap offline models for online inference (sliding window + memory), a practical recipe many will try.

•	Analyses: memory-duration breakdowns; cue-type breakdown (fine-grained / causal / background); evidence that more frames/time help under this protocol (unlike some offline MCQ benches).

### Weaknesses
•	Data novelty & provenance: benchmark reuses existing datasets; novelty is primarily the reconstruction into an online protocol. That’s valuable, but the paper should be transparent about how much is new annotation vs relabeling/retiming, and quantify human effort & agreement.

•	Mixed evaluation formats: retro-memory remains MCQ, while other parts use open-ended judged by Qwen2.5-72B. This mixture complicates cross-task comparability and may inherit LLM-judge biases (version drift, style sensitivity).

•	Res Acc metric under-specified: scoring “within a window” ignores latency error magnitude, early/late asymmetry, and false positives (spurious triggers). For stream narration, quality/coverage metrics (recall@IoU, redundancy, hallucination) are not detailed.

•	Sampling & budget fairness: per-model frame budgets differ (e.g., 50 frames for GPT-4o vs 16 for several open-source; 4 fps for “native online”), and the adapted wrapper uses 1 fps windows + 16 memory slots. Without token/pixel budget normalization or policy ablations (uniform vs shot/motion-aware), comparisons may be confounded.

•	Memory module specificity: “nearest-neighbor averaging” over 16 slots is only lightly described. How are keys built, how is compression performed, what is the retention/refresh policy, and what’s the context token cost? Gains could stem from more tokens, not better memory.

•	Quality control dependence on LLMs: LLMs filter “answerable-without-video” items and help synthesize anticipation questions/distractors, which can inject generator artifacts and style priors. Human audit/IAA and leakage checks are not quantified.

•	Forgetting-curve confounds: longer recall intervals may correlate with different content types/difficulties. Matching controls (same target type at different lags) are not described.

### Questions
* Please detail the compression, slot update, and NN averaging mechanics, and report ablations on #slots, compression ratio, prompt format, and context tokens to separate memory design from context size.

* How stable are results under alternative judges (e.g., Claude, Gemini) and prompts? For Pro-Anticipation, can you report latency error distributions (mean absolute time error), early/late penalties, and false-trigger rates in addition to Res Acc?

### Soundness
3

### Presentation
3

### Contribution
3
