# OOTSM: A Decoupled Linguistic Framework for Effective Scene Graph Anticipation

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
A scene graph is a structured representation of objects and their spatio-temporal relationships in dynamic scenes. Scene Graph Anticipation (SGA) involves predicting future scene graphs from video clips, enabling applications in intelligent surveillance and human-machine collaboration. While recent SGA approaches excel at leveraging visual evidence, long-horizon forecasting fundamentally depends on semantic priors and commonsense temporal regularities that are challenging to extract purely from visual features.
We therefore propose Linguistic Scene Graph Anticipation (LSGA) as an independent forecasting task in the language domain, treating visual detection as a pluggable front-end while focusing on how temporal relational dynamics should be modeled and evaluated. Concretely, we introduce Object -Oriented Two -Staged Method (OOTSM), an object-oriented two-stage framework that enhances LSGA through dynamic object prediction and relationship forecasting with temporal consistency constraints, accompanied by a dedicated LSGA benchmark derived from Action Genome annotations. Extensive experiments demonstrate that compact open-source LLMs fine-tuned for LSGA surpass strong zero-shot APIs (i.e., GPT-4o, GPT-4o-mini, DeepSeek-V3) in most evaluation metrics under equivalent input conditions and context-window constraints. In particular, integration with frozen scene-graph detectors enables these LSGA advancements to yield superior video-based SGA performance, especially in long-horizon prediction scenarios (+21.9% mR@50), highlighting the substantial complementarity for the SGA task. Code is available at https://github.com/only4anonymous/OOTSM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces OOTSM, a two-stage framework for Linguistic Scene Graph Anticipation (LSGA), a novel variant of the Scene Graph Anticipation (SGA) problem. The key idea is to decouple visual perception from semantic temporal reasoning, which is reformulated as a text-based forecasting task handled by a fine-tuned LLM. The proposed method consists of two modules: Global Object Anticipation (GOA), which predicts future object sets; and Object-Oriented Relationship Anticipation (OORA), which forecasts object-specific relationships with temporal consistency.
Experiments on the Action Genome dataset show that fine-tuned compact open-source LLMs (e.g., Llama 3B) outperform strong zero-shot models such as GPT-4o in long-horizon prediction and that integrating OOTSM with visual detectors improves standard video-based SGA metrics.

### Strengths
1. The problem formulation of Linguistic Scene Graph Anticipation (LSGA) is novel, interesting and solid. LSGA is a clean and modular reformulation of SGA that disentangles vision from reasoning. This conceptual clarity is valuable and could inspire new research directions bridging symbolic reasoning and video understanding.

2. The two-stage OOTSM design is well-motivated, addressing both context window limits in small LLMs and temporal inconsistency in relation prediction.

3. GOA provides a novel solution to the continuous-object limitation in SGA.

4. The ablation studies are thorough and insightful. The authors isolate the effects of cosine-weighted losses and transition regularization, and provide sensitivity analyses over observation window size and weighting parameter β.

### Weaknesses
1. The novelty in modeling architecture is limited. 

1.1 While the problem formulation is novel, the core technical contributions (weighted CE loss, KL-based transition regularizer) are relatively straightforward extensions rather than fundamentally new learning paradigms.

1.2 As a measure of novelty, OOTSM's performance on video based SGA is not significantly better than SceneSayer. It is not surprising that OOTSM outperforms SceneSayer on mR@20/50 because OOTSM introduces LLMs which naturally address the long-tail distribution issue in SGA. It is concerning why OOTSM cannot outperform SceneSayer on R@10, especially for longer anticipation windows (F=0.3, 0.5), despite the usage of LLMs. 

1.3 This also raises concerns on the necessity of using LLMs. Which part of the performance gain is actually contributed by the LLMs? What are OOTSM's training / inference efficiency and computational resource cost compared to SceneSayer?

2. The experiments are not enough sufficient. 

2.1 The evaluation on video based SGA is not complete. The authors only evaluates on the easiest setting GAGS, where ground truth objects and bounding boxes are available. OOTSM's performance under more challenging settings PGAGS (only GT bounding box available) and AGS (no GT available) is not evaluated, limiting the impact of this research. 

2.2 The performance of GOA should be explicitly evaluated since it is one of the core contributions of this paper. How well does GOA detect appearing / disappearing objects?

2.3 The results under the No Constraint SGA setting are not provided. 

3. The writing clarity of Section 3.3 and 3.4 could be improved by explaining Figure 3 and Figure 4 in more detail in the main text. While the figures themselves are informative, the captions are not, and there is no further explanation of the workflow of GOA and OORA in the main text. 

4. Qualitative analysis is very limited, and the paper could benefit from showcasing success / failure in temporal reasoning.

### Questions
See Weaknesses 1.2, 1.3 and 2.2.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper reframes Scene Graph Anticipation (SGA) as a linguistic forecasting problem (LSGA) and proposes OOTSM, a two-stage, text-only anticipation framework. The framework includes GOA, a global object-set forecaster trained with cosine-weighted CE over future tokens to down-weight distant horizons; and OORA, an object-oriented relation forecaster with a temporal transition regularizer, a symmetrized KL over transition histograms, and an auxiliary BCE head. The authors construct an LSGA benchmark by converting Action Genome to temporally ordered SG sequences, and they report that small open LLMs outperform zero-shot API models on text-only LSGA, and that plugging OOTSM behind a frozen SGG tool yields strong video-based SGA results, especially for long horizons.

### Strengths
This two-stage architecture leverages GOA to overcome context limitations and employs OORA to model per-object multi-label relationships with a principled transition prior, which is technically motivated.

Noise-injection studies distinguish between drop noise and modify noise, revealing that short-term predictions are sensitive to later-frame errors. In contrast, long-term performance remains remarkably stable (only a 0.68% drop at 25% frame errors), highlighting the true source of the model’s signal.

### Weaknesses
All core claims are tied to Action Genome. While AG is a standard, results could overfit the dataset priors, limiting generality claims.

The paper compares fine-tuned 1.5–3B models to zero-shot closed-model APIs, which is not an apples-to-apples comparison. The author claims that compact LLMs surpass zero-shot
APIs is plausible, but methodologically conflated with tuning.

GOA weighting and OORA regularization give limited gains. The ablation shows +0.7 on R@20 and +2.1 on mR@20 from the two losses combined, which is useful but small, without deeper analysis across $\beta, \lambda$.

### Questions
It seems that in Table 1, the R@50 has the same performance as R@20, and w/ GOA is not noticeably better than w/o GOA. The author should explain why.

Figure 5(b) and Section 4.5 shows that including $\beta$ in the GOA loss has a negligible effect on overall performance. This finding is corroborated by the marginal improvement between w/o GOA and w/ GOA in Table 1.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a novel task termed Linguistic Scene Graph Anticipation (LSGA) and proposes an Object-Oriented Two-Staged Method (OOTSM) to address it. The core idea is to decouple the Scene Graph Anticipation (SGA) problem: a visual scene graph detection model first converts a video clip into a sequence of historical scene graphs, and then a purely text-based model, operating on this sequence, predicts future scene graphs.

### Strengths
1. Comprehensive Evaluation:The experimental design is thorough. The paper provides convincing evidence on two fronts: first, by rigorously evaluating the LSGA task itself against powerful baselines, and second, by demonstrating that advancements in LSGA directly translate to superior performance on the established video SGA task. The substantial improvement in long-horizon prediction (e.g., +21.9% mR@50) is a strong result that highlights the practical value of this approach.

### Weaknesses
1. Task Delineation Requires Emphasis:While LSGA is novel, the paper does not sufficiently articulate the fundamental distinctions between LSGA and existing tasks like Dynamic Scene Graph Generation (SGG) or Video SGGin the main text (particularly in the Introduction and Related Work sections). Existing video SGG tasks primarily focus on describingthe present or past by detecting and linking objects across frames. In contrast, LSGA is fundamentally about anticipationor forecastingfuture states. This shift from "what is/was" to "what will be" is crucial and introduces unique challenges (e.g., heavy reliance on commonsense and causal reasoning) and applications. This distinction should be a central point of the paper's narrative.

2. Dependence on Front-end Performance:The performance of the LSGA module is inherently upper-bounded by the quality of the initial scene graph detection. Errors from the front-end model will inevitably propagate and could be amplified in the forecasting stage. The paper should discuss this limitation more explicitly.

3. Benchmark Details:More detailed information about the constructed LSGA benchmark is needed. Key statistics (e.g., dataset size, train/val/test splits, temporal horizons for prediction, class distributions) and the precise construction methodology are critical for reproducibility and future benchmarking.

### Questions
1. As highlighted in the weaknesses, the core differentiator of LSGA is the "anticipation of future states" versus the "description of current/past states" in Video SGG.Could you please elaborate on this distinction more explicitly in the paper? Specifically, what are the unique advantages and challenges of formulating this forecasting problem in the linguistic modality? This is essential for readers to fully grasp the contribution.

2. How does the performance of OOTSM degrade with errors in the input scene graph sequence from the front-end detector? Is there any analysis or inherent robustness in your method to such noisy inputs?

### Soundness
2

### Presentation
3

### Contribution
2
