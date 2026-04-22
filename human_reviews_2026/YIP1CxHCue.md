# K-frames: Scene-Driven Any-k Keyframe Selection for Long Video Understanding

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Multimodal Large Language Models (MLLMs) have demonstrated significant capabilities in image understanding, but long-video are constrained by context windows and computational cost. Uniform frame sampling often leads to substantial information loss. Meanwhile existing keyframe selection methods such as text-frame retrieval or RL-based frame optimization typically yield sparse and temporally disjointed frames, overlooking scene continuity and lacking flexibility for multi-scale frame selection.
To address these limitations, we introduce K-frames, a novel paradigm for scene-driven keyframe selection that preserves temporal continuity. Instead of selecting individual frames, K-frames predicts semantically coherent, query-relevant clips, which enables any-k keyframes selection to meet diverse user budgets. To achieve this approach, we first introduce PeakClips, a dataset of 200K video highlights conditioned by query. Building on this dataset, K-frames learns clip2frame selection using a three-stage progressive curriculum. It involves two Supervised Fine-Tuning stages for temporal grounding and key-clip perception, followed by a Reinforcement Learning stage that directly optimizes the scene-driven prediction policy for downstream task without further annotations. Extensive experiments on major long-video understanding benchmarks demonstrate that K-frames provides an effective, interpretable, and plug-and-play solution for keyframe selection at various scales. Our dataset and model will be available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces K-frames, a scene-driven frame sampling framework for long video understanding that prioritizes semantically relevant segments based on user queries. The authors construct PeakClips, a large-scale dataset with hierarchical (video/chapter/scene-level) captions and LLM-annotated query-relevance scores (P1/P2 priority). Key contributions include: (1) a three-stage pipeline—scene segmentation, hierarchical captioning, and LLM-guided relevance scoring; (2) two sampling strategies—Focused Sampling (from high-relevance clips only) and Hybrid Sampling (combining relevant clips and background); and (3) extensive experiments showing consistent gains over uniform sampling across multiple benchmarks (e.g., MLVU, VideoMME) and video-LLMs (e.g., Qwen2.5-VL, LLaVA-Video), especially under low frame budgets (e.g., +25.6% on MLVU with 8 frames). The method effectively bridges semantic scene structure with efficient visual token selection for query-conditioned video understanding.

### Strengths
The paper demonstrates strong originality by reframing keyframe selection as a scene-driven, query-aware process, shifting from uniform or heuristic sampling to a semantically grounded, relevance-guided paradigm. Its technical quality is high: the proposed K-frames method is well-motivated, accompanied by a carefully constructed dataset (PeakClips) with hierarchical annotations and LLM-derived relevance scores, and validated through rigorous experiments across multiple benchmarks and model scales. The clarity of presentation is excellent, with intuitive figures, clear algorithmic descriptions, and transparent reporting of design choices (e.g., P1/P2 prioritization, frame allocation). Most importantly, the work holds significant practical impact: it enables more efficient and accurate long video understanding under tight frame budgets, a common real-world constraint, while offering a model-agnostic front-end that can enhance any video-LLM without architectural changes. By aligning frame selection with semantic scene structure and query relevance, the paper advances both the methodology and infrastructure for scalable video reasoning.

### Weaknesses
- The article is a bit confusing in its use of symbols, font, and formulas. This is not conducive to readers' understanding of the article.

- The paper used some hyperparameters 4.3, 4.9, $w_1$, $w_2$, $\alpha$, and $\beta$ when constructing the PeakClips dataset, but lacked further explanation of the hyperparameter selection.

- Are the claims about model size in Table 1 fair? K-frames based on Qwen2.5-VL will introduce additional 3b parameters during inference and evaluation.

- The reward function design for the RL stage is not clear enough

- Lack of discussion of limitations.

- Minor Weaknesses
  - Line 97: `visual content.;` extra period
  - Line 155: $\{ clip_{i}^{v}\}=[frame\_start_{i}^{v}, frame\_end_{i}^{v}]$ can it be more simple? Moreover, the specific meaning of the superscript $i$ does not seem to be introduced.
  - Line 193: `SIGLIP similarity`, missing necessary citations
  - Line 258: `Mao et al. (2023)`, inconsistent citation format
  - Line 377: `k keyframes` -> $k$ keyframes

### Questions
- The proposed K-frames is coupled with MLLM (Qwen2.5-VL-7B) during the training of Stage-2(SFT) and Stage-3 (RL). Although Qwen2.5-VL-7B is frozen during training, will the performance fluctuate if other models are used?

- In Table 2, for the same 8 frames, the performance of Qwen2.5-VL-72B is lower than that of Qwen2.5-VL-7B on Needle-QA, which seems a bit counterintuitive.

- Can the authors provide a comparison of K-frames with other frame selection methods?

- The paper lacks additional computational overhead and latency associated with K-frames.

I look forward to discussions with the author during the rebuttal phase and would be happy to improve my score.

### Soundness
3

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
4

### Summary
This paper proposes K-Frames, a _scene-driven_ and _query-aware_ framework for keyframe selection in long-video understanding.
The method first constructs a _scene similarity graph_ from frame-level embeddings, connecting visually and temporally adjacent nodes. Node salience is then scored using a combination of _query relevance_ and _diversity penalties_. Top-K keyframes are sampled from the most salient scene nodes. Additionally, the authors introduce a new dataset, **PeakClips** (200K video-query pairs), annotated with scene boundaries and hierarchical captions, and train their model through a three-stage curriculum: two supervised fine-tuning stages followed by a reinforcement-learning (RL) stage that aligns clip selection with downstream QA accuracy. Experiments on VideoMME, MLVU, and LVBench demonstrate consistent improvements over uniform sampling and recent baselines (e.g., ViaRL).

### Strengths
- **Graph-based formulation.** The idea of modeling temporal relations among scenes as a graph is conceptually appealing and promotes global consistency in keyframe selection.

- **Balanced relevance–diversity scoring.** The integration of query relevance with diversity control is reasonable and supported by formal design.

- **Extensive experiments.** Evaluation covers multiple benchmarks (MLVU, VideoMME, LVBench) and both open- and closed-source MLLMs (QwenVL, Gemini, GPT-4o), demonstrating cross-model generality.

- **Dataset contribution.** The PeakClips dataset provides a large-scale resource for scene-level grounding and may be useful for future works on temporal reasoning.

### Weaknesses
- **Limited conceptual novelty.** The combination of scene segmentation, query-aware relevance scoring, and diversity optimization has been widely explored in recent works. Beyond earlier methods such as Frame-Voyager (Yu et al., 2024), AKS (Tang et al., 2025), and EFS (Sun et al., 2025), several newer approaches—T* (Ye et al., 2025), Logic-in-Frames (Guo et al., 2025), mDP3 (Sun et al., 2025), VSI (He et al., 2025), and Nar-KFC (Fang et al., 2025)—already integrate scene- or event-level temporal structuring with query-conditioned frame retrieval and diversity-aware selection. Compared to these, K-Frames largely reformulates the same paradigm through a graph abstraction and an added SFT→RL curriculum, without introducing fundamentally new principles of temporal reasoning or representation learning.
Overall, the contribution appears incremental rather than conceptual, as many of the above methods already address temporal coherence, multi-scale selection, and reasoning alignment more systematically.

- **Questionable contribution of the graph structure.** The paper does not convincingly show why graph-based modeling is superior to simpler alternatives such as clustering-based grouping or temporal-adjacency filtering. There is no ablation isolating the graph connectivity or node-edge design—so it remains unclear whether the graph actually improves selection quality beyond standard grouping.

- **Heuristic-heavy and weakly learned pipeline.** Despite being presented as a learning framework, most components rely on fixed heuristics:
Scene boundaries from histogram differences, Relevance computed via frozen Gemini 2.5 Pro + SIGLIP, Hand-tuned thresholds for P1/P2 classification. The RL stage fine-tunes only a small policy head and uses rule-based rewards. Consequently, the system remains largely training-free and non-adaptive to new domains or tasks.

- **Dataset quality and scalability concerns.** The PeakClips dataset is automatically labeled via LLM prompts, which raises reproducibility and reliability concerns. No human verification or inter-annotator analysis is reported. Moreover, since PeakClips depends heavily on proprietary Gemini outputs, its release or reproducibility for academic use is questionable.

- **Marginal improvement over strong baselines.** Although Table 1 shows large gains over uniform sampling, this baseline is weak.
When compared to recent training-based methods (e.g., ViaRL, EFS, BOLT), the gains are modest (≈ 2–4 points). It is unclear whether the added complexity of dataset construction and RL fine-tuning is justified by these limited improvements.

- **Terminology mismatch.** The term “Scene Graph” is misused: the proposed structure lacks entities, relations, or semantic predicates typically associated with scene graphs in computer vision. A more accurate term would be “temporal-visual similarity graph.”

- **Evaluation fairness and transparency.** The reported results use proprietary models (Gemini 2.5 Pro, GPT-4o) for scoring and evaluation.
It is unclear how the model’s outputs were filtered or whether any cherry-picking occurred. Furthermore, no statistical significance or variance across runs is provided.

- **Computational efficiency not demonstrated.** The paper claims “any-K sampling and improved efficiency,” but no profiling or runtime analysis is given. In practice, the three-stage pipeline (scene segmentation → SFT → RL) may be substantially more expensive than simple similarity-based retrieval.

### Questions
1. **Role and stability of RL fine-tuning.** How exactly does the RL stage affect selection quality compared with SFT-only training (Table 3)? Are the observed improvements statistically significant and consistent across random seeds?
2. **Graph formulation justification.** Could the authors demonstrate that the proposed graph structure yields measurable improvement over simpler temporal clustering or adjacency-based grouping? What concrete advantage does the “scene graph” abstraction provide beyond naming?
3.  **Generalization and compositional reasoning.** How robust is the system when handling compositional or cross-scene queries beyond single-event localization, and does the model generalize to non-QA tasks such as captioning or temporal reasoning?
4. **Dataset reliability and openness.** Since PeakClips is auto-labeled using Gemini 2.5 Pro, can the dataset be publicly released under license, and how was its annotation quality or bias verified?
5. **Interpretability and rationale faithfulness.** The paper highlights interpretable rationale outputs—can the authors quantify or evaluate their faithfulness, e.g., via human evaluation or alignment with visual evidence?
6. **Efficiency and computational overhead.** Please report the end-to-end runtime and computational cost (scene segmentation + SFT + RL) relative to uniform or retrieval-based baselines, especially when scaling to long videos.

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
4

### Summary
This paper proposes K-frames, a scene-driven approach for keyframe selection in long video understanding. Instead of selecting individual frames, it reframes the task as "clip-to-frame" prediction: first identifying query-relevant video segments, then sampling keyframes from these coherent clips. This addresses issues like temporal discontinuity and inflexibility in prior methods.

To support this, the authors construct PeakClips, a large-scale dataset with 200K query-conditioned annotations, and train the model using a three-stage curriculum (two SFT stages + RL). Experiments show K-frames works as a plug-and-play front-end that boosts various MLLMs' performance on long-video benchmarks.

### Strengths
Clip-level selection: Choosing coherent segments first ensures temporal continuity.

Valuable dataset: PeakClips provides rich annotations useful for broader research.

Model-agnostic: Consistently improves various models (Qwen, GPT-4o, Gemini) without modification.

### Weaknesses
Missing key comparisons: Lacks direct comparison with advanced temporal search methods (e.g., VideoAgent, T*, VideoTree).

Indirect validation: Relies solely on downstream QA accuracy for evaluation, without directly measuring clip selection quality (e.g., overlap with ground-truth segments).

Unclear overhead: Fails to report the additional time cost of the "clip-finding" step, leaving practical efficiency in question.

### Questions
See Weaknesses

### Soundness
2

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
4

### Summary
This paper proposes K-frames, a scene-driven keyframe selection paradigm that selects coherent query-relevant clip rather than isolated frames, enabling flexible “any-k” frame budgets. The method is trained with a progressive 3-stage curriculum (two SFT stages + RL) using PeakClips, a newly constructed 200K query-conditioned highlight dataset. Experiments on long-video benchmarks show that K-frames yields more effective, interpretable, and plug-and-play keyframe selection across scales.

### Strengths
This paper proposes a novel key-frame sampling method. It explicitly considers scene structure in videos rather than treating them as simple independent frames — this is a very good starting motivation. In addition, the dataset construction is comprehensive and well designed.

### Weaknesses
1.What is the advantage of scene-based splitting compared to directly segmenting the video into equal-length intervals?

2.Unlike traditional frame-level semantics, “K-FRAMES” selects key frames based on complete temporal structure. This advantage should manifest in tasks such as action recognition and temporal reasoning tasks that depend on continuous dynamics rather than static appearance. Can the authors provide results on such tasks?

3.The authors train an additional module (Qwen-2.5-VL 3B) for key-frame sampling. I am curious how its performance compares to off-the-shelf multimodal retrievers such as LanguageBind.

4.In Section 4.3, the temporal localization experiments are not very convincing because 'Needle QA' is too easy. Consider adding stronger evidence (more challenging temporal localization proof).

5.The “Priority Tag” idea is interesting, but the authors do not discuss it in depth. Could the authors demonstrate whether this design is necessary with ablation experiments?

### Questions
If the authors can answer the questions raised in the Weaknesses section, I will consider raising my score.

### Soundness
3

### Presentation
3

### Contribution
2
