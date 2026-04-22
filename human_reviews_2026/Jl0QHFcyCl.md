# AnomSeer: Reinforcing Multimodal LLMs to Reason for Time-Series Anomaly Detection

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Time-series anomaly detection (TSAD) with multimodal large language models (MLLMs) is an emerging area, yet a persistent challenge remains: MLLMs rely on coarse time-series heuristics but struggle with multi-dimensional, detailed reasoning, which is vital for understanding complex time-series data. We present AnomSeer to address this by reinforcing the model to ground its reasoning in precise, structural details of time series, unifying anomaly classification, localization, and explanation. At its core, an expert chain-of-thought trace  is generated to provide a verifiable, fine-grained reasoning from classical analyses (e.g., statistical measures, frequency transforms). Building on this, we propose a novel time-series grounded policy optimization (TimerPO) that incorporates two additional components beyond standard reinforcement learning: a time-series grounded advantage based on optimal transport and an orthogonal projection to ensure this auxiliary granular signal does not interfere with the primary detection objective. Across diverse anomaly scenarios, AnomSeer, with Qwen2.5-VL-3B/7B-Instruct, outperforms larger commercial baselines (e.g., GPT-4o) in classification and localization accuracy, particularly on point- and frequency-driven exceptions. Moreover, it produces plausible time-series reasoning traces that support its conclusions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ExpCoT and TimerPO to enhance fine-grained reasoning of multimodal large language models for time-series anomaly detection, showing clear motivation, solid methodology, and notable performance improvements.

### Strengths
Clear and meaningful motivation addressing the lack of fine-grained reasoning in MLLMs for time-series tasks.
Well-designed framework combining ExpCoT (expert chain-of-thought supervision) and TimerPO (reinforcement optimization).
Strong experimental results demonstrating improved interpretability and reasoning quality.

### Weaknesses
My concerns are as follows:
1. The generation of ExpCoT requires traditional statistical analyses (FFT, residual detection, Matrix Profile, etc.), and each anomaly type needs specific parameters and templates. When transferring to new domains, the “expert reasoning templates” need to be redefined; and it is difficult to automatically scale to large heterogeneous datasets.
2. Since each anomaly type is defined by fixed parameters and templates, if such reasoning chains are already effective, I am curious why the task does not simply adopt traditional detection methods instead of fine-tuning a large model to learn them.
3. Regarding the choice of OT for semantic alignment, there is a lack of comparison and analysis. It should be explained why OT is preferred over other alternatives. In my understanding, OT is computationally expensive and highly sensitive to hyperparameters.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates a notable limitation of MLLMs in TSAD: their reliance on superficial visual cues rather than quantitative reasoning. The authors propose ANOMSEER, a post-training framework aimed at improving MLLMs’ reasoning ability for anomaly detection, localization, and explanation.

The method integrates two ideas. First, the ExpCoT uses classical TSAD tools such as statistical indicators and frequency transforms to generate verifiable reasoning traces. Second, a reinforcement learning component, “TimerPO,” introduces a time-series reasoning advantage based on Optimal Transport distance, serving as an auxiliary signal orthogonal to the main task objective.

The model is trained only on synthetic data and evaluated on both synthetic and real-world benchmarks. The reported results show clear improvements in detection accuracy and reasoning coherence.

### Strengths
The idea of incorporating expert-generated reasoning traces from classical TSAD methods is conceptually sound. It provides a structured and verifiable way to include numerical priors into the model. The TimerPO algorithm is also technically interesting, especially its use of Optimal Transport to measure reasoning similarity.

The framework demonstrates decent generalization, with stable performance across datasets despite being trained only on synthetic data. This suggests a certain degree of robustness, though the gains appear moderate.

### Weaknesses
- Several issues limit the strength of the paper’s claims. First, the evaluation on the AnomLLM dataset may be affected by potential information leakage, since ExpCoT traces include ground-truth anomaly intervals.
- Second, the paper lacks ablation studies isolating the effects of ExpCoT and GRPO, which makes it difficult to understand their individual contributions.
- Third, the manuscript does not provide clear definitions or implementation details for the reported Affinity-Precision, Affinity-Recall, and Affinity-F1 metrics. Given that these are central to the model’s evaluation, a formal definition or clear explanation is essential for reproducibility and comparability.
- In addition, the choice of baselines seems incomplete. The comparisons mainly focus on LLM-based methods, but omit classical or linear time-series anomaly detectors (such as ARIMA-type or statistical thresholding models), which are often strong baselines in TSAD literature. Including or at least justifying the exclusion of such baselines would make the evaluation more convincing.
- Finally, the framework currently supports only univariate series, limiting its applicability to real-world multivariate scenarios. The model also appears less effective at detecting short-term or boundary anomalies, indicating limited temporal sensitivity.

### Questions
1. Does TimerPO risk encouraging stylistic imitation of expert reasoning rather than genuine reasoning improvement? 
2. Could the authors include ablation experiments (ExpCoT-only, GRPO-only) to clarify the role of each module?
3. Since the evaluation metrics (especially Affinity-based ones) are not fully defined, will the authors provide formal descriptions and implementation details in the revision?
4. Have the authors considered adding representative classical or linear baselines to support a fairer comparison and contextualize the reported improvements?

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
This paper tackles time series anomaly detection (TSAD) with multimodal LLMs. The authors argue that existing MLLMs rely on coarse “eyeballing” of plotted series and fail on subtle anomalies (e.g., small frequency shifts, weak trends). They propose:

**ExpCoT**: a structured, *algorithm-grounded* reasoning trace—Observation → Reasoning & Validation → Conclusion—whose content is produced by classic TSAD tools (FFT, Matrix Profile, gradient statistics, etc.) and optionally refined by humans. These traces act as interpretable, verifiable evidence.

**TimerPO**: a GRPO/PPO-style RL objective that augments the main outcome reward with a time-series reasoning reward. The latter turns an optimal transport distance between the model’s generated reasoning tokens and the ExpCoT tokens into a scalar reward: The final advantage is a *projected* combination, i.e., auxiliary advantage is orthogonally projected to avoid conflicting with the main objective.

On benchmarks (synthetic AnomLLM, mixed VisualTimeAnomaly, and real-world TSB-UAD), AnomSeer (Qwen2.5-VL-3B/7B backbones) outperforms zero-shot commercial MLLMs and a strong RL baseline (TimeMaster), with the largest gains on hard frequency/trend anomalies. Ablations indicate both ExpCoT and the orthogonal projection matter.

### Strengths
- **Originality (compositional):** A thoughtful pairing of *process-evidence alignment* via OT with *orthogonal advantage composition* inside GRPO, targeted at TSAD. The ExpCoT design grounds CoT in verifiable TSAD signals rather than generic text heuristics.
- **Quality:** Solid performance improvements over zero-shot MLLMs and a strong RL baseline, with the biggest wins on the hard frequency/trend categories that motivated the paper. Ablations show each component matters; sensitivity to (\alpha) suggests robustness.
- **Clarity:** Method is clearly derived; equations are consistent; qualitative analyses (token distributions, embedding shifts, case studies) help interpret why it works.
- **Significance:**  Practical implications for domains where subtle anomalies matter.

### Weaknesses
- **Related Work coverage:** Missing discussion of **OT in RL/alignment** and **multi-objective/gradient-projection** literature (e.g., PCGrad). As a result, novelty may be under-justified as more than a careful composition.
- **Baselines:** Ablations remove components, but comparisons lack *alternative* multi-objective schemes:
   (i) simple weighted-sum (no projection),
   (ii) PCGrad-style gradient orthogonalization,
   (iii) replacing OT with cosine/CLIP-style similarity.
   These are crucial to attribute gains to OT geometry and projection at the advantage level.
- **Data scaling & convergence:** Training uses ~3.2k synthetic instances; no learning curves or scaling study (1k/2k/3.2k/5k) to show whether performance is saturated or still improving.
- **Comparisons with classic TSAD:** Since ExpCoT is built from FFT/Matrix Profile/gradients, including **direct classical TSAD baselines** (and hybrids) would calibrate absolute difficulty and show when AnomSeer offers more than well-tuned classical detectors.
- **Reproducibility details:** Clarify how marginals (u,v) are formed for OT, training split comparability with TimeMaster, compute cost and wall-clock/GPU hours, and the scope of “human refinement” on ExpCoT (portion, protocol, QC).

### Questions
1. **OT marginals & implementation:** How exactly are (u) and (v) constructed—uniform over positions or probability-weighted by token posteriors? Is OT solved with entropic Sinkhorn; what (\varepsilon) and iteration cap?
2. **Where does OT backpropagate (if at all)?** Is (W) used only to *shape rewards* (no gradient to embeddings/policy), or is any part made differentiable?
3. **Why orthogonalization at the advantage level vs. gradient level?** Please compare to PCGrad-style gradient surgery and a naïve weighted-sum baseline. Does advantage-space orthogonalization bring stability/variance benefits?
4. **Cosine vs. OT:** Replace OT with cosine similarity in the process reward. How much of the gain is geometry-specific to OT?
5. **Training splits & statistics:** Are AnomSeer and TimeMaster trained on exactly the same 3.2k split and initialization? Provide confidence intervals or significance tests for the main table.
6. **Learning curves & scaling:** Show performance vs. RL steps and vs. data size (1k/2k/3.2k/5k). Does performance plateau?/
7. **Classical TSAD baselines:** Include Matrix Profile / FFT-based detectors (and optionally simple ensembles) to contextualize absolute difficulty and where MLLM reasoning adds value.
8. **ExpCoT refinement scope:** What percentage of traces received human edits? What was the QC protocol?

### Soundness
3

### Presentation
3

### Contribution
3
