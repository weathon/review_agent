# IF-VidCap: Can Video Caption Models Follow Instructions?

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 8, 4

## Abstract
Although Multimodal Large Language Models (MLLMs) have demonstrated proficiency in video captioning, practical applications require captions that follow specific user instructions rather than generating exhaustive, unconstrained descriptions. 
Current benchmarks, however, primarily assess descriptive comprehensiveness while largely overlook instruction-following capabilities.
To address this gap, we introduce IF-VidCap, a new benchmark for evaluating controllable video captioning, which contains 1,400 high-quality samples.
Distinct from existing video captioning or general instruction-following benchmarks, IF-VidCap incorporates a systematic framework that assesses captions on two dimensions: format correctness and content correctness.
Our comprehensive evaluation of 26 prominent models reveals a nuanced landscape: despite the continued dominance of proprietary models, the performance gap is closing, with top-tier open-source solutions now achieving near-parity.
Furthermore, we find that models specialized for dense captioning underperform general-purpose MLLMs on complex instructions, indicating that future work should simultaneously advance both descriptive richness and instruction-following fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes IF-VidCap, the first benchmark explicitly targeting controllable, instruction-following video captioning. IF-VidCap consists of 1,400 high-complexity samples—each composed of a video, a detailed, compositional set of user instructions (averaging six constraints across 27 types), and an evaluation checklist grounded in both rule-based and open-ended question assessments. The benchmark enables systematic study of MLLM video captioning models on both content accuracy and their ability to adhere to fine-grained, user-defined instructions. Detailed experimental results and analyses highlight shortcomings in current models and demonstrate the strength of instruction-specific tuning.

### Strengths
**1. Novelty and Relevance:**
The paper identifies a clear gap in current video captioning evaluation: lack of systematic assessment for instruction-following. : Shifts video captioning evaluation from “describe everything” to controllable generation with compositional constraints, a real need for editing, generation, and content ops. The benchmark enumerates 27 constraint categories with clear coverage (format, stylistic, content; Fig. 3d), which is more granular than prior video captioning benchmarks.

**2. Evaluation Metrics Design:**
The rule+LLM hybrid judging (LLM extracts; scripts verify) for format, complemented with retrieval-based QA for content, is sensible and scalable. Inclusion of both CSR/ISR and their rule/open-ended breakdowns is helpful for diagnosing failure modes.

### Weaknesses
**1. Insufficient Baseline:**
While Table 2 compares many models, some critical vision MLLM baselines from the missing related works (particularly methods leveraging procedural or hierarchical modeling and dynamic storyline composition) are not included or discussed, limiting the ability to guarantee current models are being fairly and comprehensively evaluated, such as LLaVA-Video, VideoChat, InternVideo, KiMi-VL, Keye-VL, MiMo-VL, GLM-4.1V and so on.

**2. Unclear Difficulty Calibration:**
Best ISR ≈ 27.8% (Gemini-2.5-Pro), with open-ended CSR ≈ 35%–36% across leaders. The paper interprets this as “difficult,” but doesn’t calibrate human or near-oracle ceilings. 

**3. Narrow Analysis of IF-Captioner-Qwen:**
Gains are shown only on IF-VidCap. Need other video captioning benchmark (e.g., VidCapBench, Dream-1K and so on) to demonstrate transfer and rule out overfitting to this benchmark’s constraint taxonomy.

**4. Figures Need Deeper Interpretation:**
Several figures are highly informative but under-discussed in the text. For example, Figure 5’s constraint-type heatmap is comprehensive, but the main text lacks deeper discussion of which constraint categories most differentiate current MLLMs, and how these signal bottlenecks for the field.

### Questions
1. Could you add part of human-written captions under constraints to measure Human-ISR/CSR and a scripted case for format-only ceilings; otherwise, hardness is anecdotal.

2. Could you quantify template overlap with IF-VidCap constraints (n-gram/Jaccard/TF-IDF) and provide cross-benchmark results to show that IF-Captioner-Qwen’s gains persist on external IF-video evaluations?

Other concerns please see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces IF-VidCap, a new benchmark designed to evaluate whether multimodal large language models can generate video captions that adhere to specific user instructions. It proposes a structured evaluation framework that measures both format correctness and content correctness, combining LLM-based and rule-based checking. The benchmark includes 1,400 samples and an associated fine-tuning dataset to enhance instruction-following abilities. Through evaluation of over 20 models, the authors provide a comprehensive analysis of current MLLMs’ capabilities and limitations in controlled video captioning.

### Strengths
1.  Valuable and timely evaluation benchmark — the proposed dataset fills a significant gap in assessing instruction-following behavior for video captioning models. 
2. Covers a wide range of different settings, including multiple constraint types, compositional tasks, and diverse video sources. 
3.  Includes a fine-tuning dataset, enabling reproducibility and extension for future research.
4. Two-format setting (rule-based vs. open-ended checking) is well-designed and helps assess both structural and semantic capabilities. 
5. Strong methodological contribution: The combination of LLM-based and rule-based evaluation is novel and well-justified. 
6. Reliability verification — the authors confirm the stability and consistency of their evaluation metrics. 
7. Comprehensive analysis of model capabilities — Figure 5 and related results provide meaningful insights into the strengths and weaknesses of current models.
8. Clear structure and thorough experiments: The benchmark is well-documented, with strong empirical validation across multiple models and metrics.

### Weaknesses
1. Lack of detail on video selection and preprocessing: It’s unclear how the 350 base videos were chosen and filtered beyond general quality criteria. The authors should provide a full list or dataset summary for reproducibility. 
2. Limited discussion on annotation consistency: Although human refinement is mentioned, inter-annotator agreement or quality control statistics are not detailed. 
3. Benchmark scope limitation: The dataset focuses primarily on short or medium-length videos (2–60 seconds), leaving longer temporal reasoning largely unexplored. 
4. Complexity vs. accessibility trade-off: The multi-step evaluation protocol may limit broader adoption unless supported by easily usable code or interfaces.

### Questions
see above

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces IF-VidCap, a new benchmark for instruction-following video captioning that evaluates a model's ability to adhere to diverse, multi-constraint user instructions. Comprising 1,400 video-instruction pairs with complex constraints, the benchmark reveals through extensive evaluation that even top models like GPT-4o achieve only modest instruction fidelity, with specialized captioning models struggling significantly. The work also provides a 46K-pair training dataset and shows that fine-tuning can improve performance, positioning IF-VidCap as a key driver for future research in controllable video description.

### Strengths
This work demonstrates significant strengths through its creation of IF-VidCap, the first benchmark systematically evaluating instruction-following in video captioning with complex, real-world constraints. The benchmark is built on high-quality, carefully curated data and features a comprehensive, human-validated evaluation protocol. Its extensive experiments across ~20 diverse models yield clear insights into scaling effects and model capabilities, while the accompanying training dataset proves practically useful for model improvement. The analysis is both thorough and accessible, supported by effective visualizations, making a strong case for the benchmark's value in advancing controllable video captioning research.

### Weaknesses
The benchmark has several limitations, including its focus on short videos which excludes long-form content and constrained summarization tasks. Its evaluation, while efficient, relies on automated LLM judgments that may miss nuanced errors and depends on proprietary models, raising reproducibility concerns. Although fine-tuning demonstrates improvement, the absolute performance gains remain modest, and the analysis lacks a deeper investigation into the underlying reasons. Furthermore, the paper provides limited implementation details for key techniques like "thinking mode" and offers few qualitative examples to illustrate model failures, which could hinder comprehensive understanding and diagnosis of current shortcomings.

### Questions
1. Will the authors release the IF-VidCap dataset, annotation tools, prompts, and evaluation scripts? Making these public is crucial for adoption.

2. Can the authors clarify how the “thinking” mode is applied across models? Is it literally GPT-style CoT prompting, or something else? Providing the exact prompts or settings would help reproducibility.

3. How do models perform on each of the 27 constraint types? Are there systematic failures on particular formats (e.g. JSON vs Markdown) or content categories (e.g. spatial vs temporal constraints)? Understanding this could guide future improvements.

4. The training instructions were generated from captions using DeepSeek-V3.1. How natural and diverse are these instructions? Do they cover the same linguistic structures as the test instructions? Has any human evaluation been done on the quality of these synthetic instructions?

5. For context, how do these models perform on standard video captioning metrics (BLEU, CIDEr, CLIPScore) on the same videos? It would be informative to see the drop in performance when moving from free captioning to constrained captioning.

6. The paper relies on automated scores for the main results. Was any human evaluation of model outputs on IF-VidCap done (even a small sample) to check alignment with CSR/ISR? If so, what were the results?

7. Have the authors considered how real-world users might specify instructions beyond the benchmark’s templates? For example, could models handle more open-form instructions (“Describe the video as a news report”) that go beyond fixed constraints?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces IF-VIDCAP, the first benchmark to systematically evaluate instruction-following in video captioning models, shifting focus from traditional descriptive accuracy to fine-grained, compositional constraint adherence (e.g., format, content, style, reasoning). It comprises 1,400 high-quality video-instruction-checklist triplets (27 constraint types, averaging 6 constraints per instruction) and proposes a hybrid evaluation protocol (rule-based checks + LLM-as-Judge) for format correctness and semantic fidelity. Experiments on 20+ models show that while proprietary models lead, top open-source variants are catching up, and general-purpose MLLMs surpass specialized captioners on complex instructions. The authors further release a training dataset and their fine-tuned IF-Captioner-Qwen, which demonstrates substantial gains in instruction-following.

### Strengths
1. IF-VidCap is the first benchmark to explicitly evaluate instruction-following in video captioning (27 constraint types), addressing a critical gap beyond traditional accuracy/fluency metrics.
2. 1,400 video-instruction-checklist triplets via a two-stage pipeline (auto-generation + human refinement), with 83.6% modification rate and consensus-based validation.
3. Combines rule-based checks (deterministic) + LLM-as-Judge QA (semantic), achieving 96.33% human-agreement for reliable assessment.

### Weaknesses
1. 1,400 samples is relatively small compared to text-only instruction-following benchmarks (e.g., IFEval, CFBench). And videos average 20.5s and max out at 60s — does not test long-form temporal reasoning or multi-scene narratives.
2. Evaluation focuses on compliance, not quality. Does not assess fluency, coherence, or creativity of generated captions.
3. Training data distribution gap: Uses a "caption-to-instruction" generation method, which may not reflect real user instruction distributions.

### Questions
1. Do you plan to extend IF-VIDCAP to longer videos (e.g., >1 min) or multi-scene narratives that require temporal summarization or causal reasoning?
2. Are there plans to support multi-turn instruction-following, where users refine their requests iteratively?

### Soundness
3

### Presentation
3

### Contribution
3
