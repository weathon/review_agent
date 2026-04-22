# Complementing Self-Consistency with Cross-Model Disagreement for Uncertainty Quantification

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Large language models (LLMs) often produce confident yet incorrect responses, and uncertainty quantification is one potential solution to more robust usage. Recent works routinely rely on self-consistency to estimate aleatoric uncertainty (AU), yet this proxy collapses when models are overconfident and produce the same incorrect answer across samples. We analyze this regime and show that cross-model semantic disagreement is higher on incorrect answers precisely when AU is low. Motivated by this, we introduce an epistemic uncertainty (EU) term that operates in the black-box access setting: EU uses only generated text from a small, scale-matched ensemble and is computed as the gap between inter-model and intra-model sequence-semantic similarity. We then define total uncertainty (TU) as the sum of AU and EU. In a comprehensive study across five 7-9B instruction-tuned models and ten long-form tasks, TU improves ranking calibration and selective abstention relative to AU, and EU reliably flags confident failures where AU is low. We further characterize when EU is most useful via agreement and complementarity diagnostics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a key failure mode of existing LLM uncertainty metrics like self-consistency, which primarily estimate Aleatoric Uncertainty (AU). These methods fail when models are "confidently wrong"—producing the same incorrect answer with high consistency. To overcome this, the authors introduce a complementary Epistemic Uncertainty (EU) term, which operates in a black-box setting by measuring the "cross-model semantic disagreement" within a small ensemble of different, scale-matched models. The study finds that this EU term reliably flags incorrect answers precisely when AU is low, effectively identifying these confident failures. By defining Total Uncertainty asTU = AU + EU, a comprehensive study—spanning five 7-9B models, GPT-4o, and Claude, across ten long-form tasks—demonstrates that TU significantly outperforms AU alone in distinguishing correct from incorrect responses.s

### Strengths
1. The EU methodology is black-box and highly practical, as it relies solely on the model's generated text outputs rather than requiring access to internal logits or weights.
2. The experimental design is rigorous. The study spans five different 7-9B open-source models and two top-tier API models (GPT-4o, Claude), evaluated on ten diverse long-form tasks using two standard metrics: AUROC (for ranking calibration) and AURC (for selective abstention).
3. The method is effective, with the proposed Total Uncertainty (TU) metric demonstrating significant and consistent improvements over the Aleatoric Uncertainty (AU) baseline across nearly all tasks and models.

### Weaknesses
1. The method's performance is task-dependent. It is acknowledged to perform poorly on tasks that permit multiple semantically distinct yet correct answers (e.g., open-ended generation or summarization), where disagreement does not imply incorrectness.
2. The approach is highly reliant on the diversity of the model ensemble. If all models in the ensemble share similar biases, they may fail homogeneously by confidently producing the same incorrect answer, rendering the EU metric ineffective.
3. The use of an LLM-as-a-judge for evaluation lacks robustness. Any misjudgments or inherent biases from the judge model directly compromise the precision of the paper's core conclusion that TU outperforms AU.
4. The method incurs substantial practical overhead due to the need to load and run multiple models. While the authors suggest amortization via 5x hardware parallelism, this solution itself represents a much higher hardware and deployment cost than the single-model baseline.

### Questions
1. Why is $TU = AU + EU$ the optimal combination? Was a simple additive sum found to be superior to a weighted sum?
2. The paper motivates "scale-matched" ensembles , but ablations (Fig. 6, App. A.5)  suggest *stronger* auxiliary models yield better TU. Please clarify this discrepancy. Is scale-matching or ensemble strength the key factor?
3. How does TU handle 'homogeneous failures,' where all ensemble models are confidently and *identically* wrong? In this case, EU would be zero, and TU would fail.
4. Why was the evaluation for WMT16-de-en and XSum based *only* on the LLM-Judge, rather than also including standard metrics like BLEU or ROUGE?

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
This paper proposes a unified framework for estimating total uncertainty in large language models by decomposing it into aleatoric uncertainty and epistemic uncertainty. Instead of relying on output probabilities, the authors measure uncertainty through semantic similarity between generated responses, defining TU as the expected dissimilarity between samples from the same or different models.

### Strengths
1. Relevant and timely topic.
The paper tackles an important and underexplored problem. This is a meaningful direction for improving reliability and selective prediction in modern LLMs.
2. Extensive and systematic experiments.
The authors evaluate the proposed method across diverse datasets, model scales, and uncertainty estimation tasks. The experimental coverage is broad and the reported metrics (AUROC, AURC, selective prediction) show consistent gains over standard AU-based baselines.
3. Well-articulated discussion of limitations.
The paper clearly acknowledges the limitations of EU estimation (e.g., its reduced discriminative power in open-ended or multi-answer tasks) and provides thoughtful reasoning for these observations.

### Weaknesses
1. Limited methodological novelty.
While the proposed TU = AU + EU framework is conceptually clear and practically useful, the mathematical formulation largely builds upon existing uncertainty decomposition ideas and replaces probability-based divergences with semantic similarity measures. The innovation is therefore more in engineering design than in theoretical advance.

2. Lack of computational cost analysis.
The paper should provide an analysis of runtime and compute overhead. In particular, the multi-model sampling required for TU/EU estimation can be significantly more expensive than single-model AU estimation. Reporting inference latency, token usage, and compute normalization (e.g., equal-time or equal-token comparison) would make the evaluation more rigorous.

3. Missing ablation on judge-model sensitivity.
Since semantic similarity plays a central role, the conclusions may depend on the specific LLM used as the ``judger''. An additional experiment using alternative LLMs  as similarity evaluators would clarify how robust the TU/EU metrics are to the choice of judge model.

### Questions
1. A weighted version of TU/EU, reflecting the posterior quality of each auxiliary model, could further strengthen the analysis.

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
This paper addresses unreliable uncertainty estimates in large language models by **complementing self-consistency (aleatoric uncertainty, AU)** with a black-box estimate of **epistemic uncertainty (EU)** derived from **cross-model semantic disagreement**. AU is measured as *intra-model* response similarity across samples; EU is defined as the gap between *inter-model* similarity (a small, scale-matched ensemble of open-weight models) and *intra-model* similarity, and **Total Uncertainty (TU) = AU + EU**. Using only generated text (no logits), this paper evaluates TU across five 7–9B instruction-tuned models on ten long-form tasks (QA, translation, summarization, math), and also on API models (GPT-4o, Claude). TU consistently improves **correctness calibration (AUROC)** and **selective abstention** over AU, and EU is particularly informative on **confident failures** where AU is low. Diagnostics relate EU’s usefulness to dataset redundancy vs. complementarity, clarifying when cross-model disagreement best predicts error.

### Strengths
- **Originality.** This paper proposes a **text-only, black-box EU** estimator via cross-model semantic disagreement and shows how it complements AU; the additive **TU** is simple and broadly applicable.  
- **Quality.** Clear formulation (AU/EU/TU as intra-/inter-model similarities), principled ensemble construction (scale-matched, cross-family), and comprehensive experiments (10 tasks, multiple models, risk–coverage, AUROC, ablations on ensemble/sampling).  
- **Clarity.** Well-motivated failure mode (overconfident, wrong, low-AU generations), intuitive figures, and diagnostics that explain **when** EU helps (unique-answer tasks; high redundancy).  
- **Significance.** TU improves calibration and abstention across diverse models and tasks, including API models, without requiring logits or retraining—practical for black-box deployments.

### Weaknesses
- **Multiple-valid-answers regimes.** On open-ended tasks (e.g., abstractive summarization), semantic diversity among valid outputs can inflate EU and weaken AUROC; this paper acknowledges but does not fully resolve this.  
- **Ensemble design & weighting.** Uniformly weighted, scale-matched ensembles are pragmatic but may bias EU; robustness to **ensemble composition**, size, and weighting (e.g., by validation risk) needs deeper study.  
- **Labeling & metrics.** Reliance on LM-as-a-judge for correctness and a single sentence-embedding similarity can introduce bias (length/style effects); sensitivity analyses are limited.  
- **Compute budget.** Although sample budgets are matched to AU, running **m models × n samples** adds latency/cost; end-to-end throughput/memory profiling is not fully reported.  
- **Theory.** Additive TU is intuitive, but formal guarantees (e.g., bounds linking TU to error) are not provided.

### Questions
1. **Similarity backbone.** How sensitive are AU/EU/TU to the sentence encoder and to length normalization? Please report alternatives and length-controlled evaluations.  
2. **Ensemble weighting.** Can EU improve by weighting auxiliary models by **posterior credibility/validation risk** rather than uniformly?  
3. **Budget–performance trade-offs.** Provide wall-clock, memory, and FLOP profiles vs. *(m, n)* and show Pareto curves (accuracy/calibration vs. cost).  
4. **Triggering EU.** Explore **adaptive triggering** (invoke cross-model sampling only under low AU or specific heuristics) to cut cost while keeping gains.  
5. **Open-ended tasks.** For many-valid-answer settings, consider **reference-guided similarity**, entailment scoring, or diversity-aware TU that discounts benign variation.  
6. **Judging robustness.** Verify results with multiple judges and human audits; report inter-judge agreement to strengthen calibration claims.  
7. **Generality.** Test beyond 7–9B models and add domains like code or multi-modal tasks; study TU under retrieval-augmented generation and distribution shift.

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
This paper addresses a critical limitation of existing uncertainty quantification (UQ) methods for large language models (LLMs): the failure of self-consistency-based aleatoric uncertainty (AU) to detect "confident yet incorrect" responses (where models repeatedly produce the same wrong answer, leading to low AU). To solve this, the authors propose three core contributions: 1) They identify that cross-model semantic disagreement is disproportionately high for incorrect answers when AU is low, a key empirical insight. 2) They introduce an epistemic uncertainty (EU) metric designed for black-box LLM access—EU is computed as the gap between inter-model and intra-model sequence-semantic similarity, using only outputs from a small, scale-matched ensemble (5×7–9B instruction-tuned models). 3) They define Total Uncertainty (TU) as the sum of AU and EU, and validate it across 10 long-form tasks (QA, summarization, translation, math reasoning) plus API models (GPT-4o, Claude 3.7 Sonnet). Experiments show TU outperforms AU in ranking calibration (AUROC) and selective abstention, while EU reliably flags confident failures missed by AU. The work also characterizes EU’s utility via agreement (Jaccard redundancy) and complementarity (oracle coverage gain) diagnostics, highlighting its effectiveness for tasks with unique correct answers (e.g., factual QA, translation).

### Strengths
1. Proposes an EU estimation method for black-box scenarios, leveraging the ecological advantages of open-weight LLMs (scale-matched ensembles). This avoids the strong task/architecture dependencies of existing methods, demonstrating prominent originality.
2. The method design is theoretically grounded (kernel divergence bounds), with rigorous experimental control of variables (sample budget, judge validation), ensuring high reproducibility of results.
3. On tasks such as HotpotQA (multi-hop reasoning) and WMT16-de-en (translation), TU achieves an AUROC improvement of over 0.1, significantly enhancing LLM reliability and providing practical guidance for deployment in high-stakes scenarios.

### Weaknesses
1. The paper fixedly uses an ensemble of 5 7–9B models but fails to explore how ensemble size (e.g., 2–3 models) or diversity (same-family vs. cross-family) impacts TU. For example, is a small-scale ensemble still effective in resource-constrained scenarios? And what effects would a large parameter gap between ensemble models have? The lack of such analyses limits the practical application scope of TU.
2. Although the paper notes that TU performs poorly on tasks with multiple valid answers, it does not propose solutions. For instance, could EU weights be dynamically adjusted via an "answer uniqueness score"? Or could TU calculation be optimized by incorporating task characteristics (e.g., summary relevance vs. QA factuality)? Current analysis only "identifies the problem" without advancing to "solving the problem."
3.  It only compares with traditional baselines (semantic entropy, perplexity) and fails to comprehensively benchmark against methods in the LLM UQ field (e.g., SelfCheckGPT), making it impossible to clearly demonstrate TU’s relative advantages.

### Questions
1. How do the performances of EU and TU change when adjusting the number of auxiliary models (e.g., 2 vs. 5) or their diversity (same-family vs. cross-family)? For instance, can an ensemble of 2 cross-family models achieve performance close to that of an ensemble of 5 models? This is crucial for the deployment of TU in resource-constrained scenarios .
2. For tasks with multiple valid answers (such as XSum), can an improved version of TU be designed? For example, weighting EU by an "answer uniqueness score" (e.g., the number of correct answers for a single input) to avoid misclassifying differences between valid answers as uncertainty .
3. Please supplement the comparison of AUROC and AURC between TU and methods such as Kernel Language Entropy (Nikitin et al., 2024) and Self-CheckGPT (Manakul et al., 2023) on the same tasks, so as to clarify the position of TU in the field .
4. What specific types of problems do the "confident failures" flagged by EU belong to? Are they knowledge gaps (e.g., incorrect facts) or reasoning biases (e.g., flawed multi-hop logic)? A qualitative analysis can help readers understand the boundary of EU’s applicable scenarios .
5. The paper verifies the effectiveness of TU when using 7–9B reference models. If the scale of the reference model is expanded (e.g., 70B+), will TU still be effective? For example, large-scale models may be closer to the ideal model ω*, so will the gain from EU decrease in this case ?


If you address my concerns, I will consider adjusting your score upward accordingly.

### Soundness
3

### Presentation
2

### Contribution
2
