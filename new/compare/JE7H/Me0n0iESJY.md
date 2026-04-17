---
job_id: 8a76f70c-7635-4f54-a171-62d0b7c649fc
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Me0n0iESJY.pdf
paper: OptMerge: Unifying Multimodal LLM Capabilities and Modalities Via Model Merging
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is on data-free model merging for multimodal LLMs, including a benchmark, theory, and an optimization method; this is squarely within ICLR’s scope (representation learning, multimodal LMs, optimization, benchmarks).

## Minimum Quality
Pass ✅.  
The paper is complete (Abstract, Introduction, Related Work, Methodology, Experiments, Results/Analysis, Conclusion), written in clear English, provides substantial experiments with baselines and ablations, and includes nontrivial theoretical analysis. I do not see fatal methodological, theoretical, or evaluation flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no evidence of hidden prompts, manipulative instructions to reviewers, or steganographic content.

---

# Expected Review Outcome:

## Summary

The paper introduces a benchmark for data-free model merging in multimodal large language models (MLLMs), covering five capability categories (VQA, Geometry, Chart, OCR, Grounding) and both full-finetuning and LoRA settings, as well as a modality-merging setup (vision, audio, video).  
Building on a theoretical analysis (Theorem 3.1) that links fine-tuning dynamics to merging error, the authors propose OptMerge, which denoises task vectors via SVD-based low-rank approximations and stabilizes the data-free optimization of merged task vectors.  
Extensive experiments on InternVL2.5, Qwen2-VL, Vicuna-based multimodal models, and real Hugging Face checkpoints show that model merging can match or surpass mixture training and that OptMerge yields the best average performance among ten merging baselines.

## Strengths

1. **Timely, well-scoped benchmark for MLLM merging.**  
   The benchmark is carefully constructed to isolate *capabilities* (VQA, Geometry, Chart, OCR, Grounding) with ≥100k training samples per capability (Table 1, Page 7), and to evaluate them through specialized downstream datasets rather than generic all-in-one benchmarks. This separation of fine-tuning data and evaluation, plus the coverage of two popular backbones (InternVL2.5-1B-Instruct and Qwen2-VL-7B-Base), provides a valuable testbed for future work on multimodal model merging.

2. **Strong and broad experimental section, including community models.**  
   - Tables 2 and 3 (Page 8) present a comprehensive comparison of ten merging algorithms plus mixture training across nine evaluation datasets per backbone. The authors correctly separate the “Individual” experts, base models, mixture training, and various static merging strategies.  
   - Table 5 (Page 9) evaluates modality merging (vision–audio–video) and compares static merging with online composition methods (NaiveMC, DAMC) and individual modalities, showing that model merging can indeed exploit cross-modal complementarity.  
   - Table 6 (Page 9) uses actual heterogeneous Hugging Face checkpoints (math RL, Pokemon, PDF OCR, Vietnamese OCR/VQA) sharing a Qwen2-VL backbone; the fact that OptMerge delivers clear gains over all individual models on almost all tasks is a strong signal that the proposed technique is robust in real-world open-source ecosystems.  
   - Additional experiments on scale (Table 9, Qwen2.5-VL-32B) and general multimodal QA benchmarks (Table 10) further support claims about scalability and emergent integrated capabilities.

3. **Nontrivial theoretical analysis linking fine-tuning dynamics and merge quality.**  
   Theorem 3.1 (Page 4–5, formalized as Theorem A.10/A.11 in Appendix A) derives an upper bound  
   \[
   \mathcal{L}_i(\Theta+\tau_m) \le C_i + \mathcal{O}(\gamma^T) + \mathcal{O}(\delta \eta T) + \mathcal{O}(\eta^2 T^2)
   \]  
   under PL and smoothness assumptions, with \(\gamma=1-\eta\mu\). This gives a principled explanation of the empirically observed phenomenon that “over-finetuned” experts can hurt merging performance: as \(T\) grows, cross-task interference \(\mathcal{O}(\delta \eta T)\) and curvature \(\mathcal{O}(\eta^2 T^2)\) can dominate the residual convergence term \(\mathcal{O}(\gamma^T)\). The link between parameter drift, learning rate, and merging quality is well articulated and empirically corroborated by Figure 6 (Page 22), which shows merging performance peaking at intermediate fine-tuning steps.

4. **Insightful empirical analysis of task vectors and LoRA vs full finetuning.**  
   Figure 2 (Page 5) is a nice diagnostic:  
   - Panels (a–b) show magnitude distributions of task-vector parameters for InternVL2.5 (full finetune) and Qwen2-VL (LoRA). The right-skew vs multi-modal distributions concretely illustrate the structural differences between full-rank and low-rank finetuning.  
   - Panels (c–d) depict normalized Frobenius norms across layers and tasks, indicating substantial variation in layer-wise impact across tasks, which motivates careful per-layer low-rank treatment in Eq. (3).  
   This level of analysis is rarely seen in merging papers and adds real understanding instead of just reporting scores.

5. **OptMerge’s design is grounded in issues observable in the baseline optimizer.**  
   - Equation (1) defines the WUDI loss, and Figure 3 (Page 6) shows geometrically how unconstrained optimization of \(\tau_m\) tends to increase its norm to satisfy orthogonality-based interference minimization.  
   - Figure 4 (Page 6) then empirically plots the evolution of the Frobenius norm of the merged vector over iterations; WUDI’s norm drifts up steadily, while OptMerge keeps the norm almost constant while still reducing loss.  
   - The proposed changes (SVD centering and truncation for full-finetuned models via Eq. (3); SGD + low-rank truncation + mean initialization for LoRA) are carefully justified by both geometry and empirical behavior. Table 4 ablations (Page 8) clearly isolate the impact of each component on Qwen2-VL and modality merging.

6. **Clear empirical message: merging can rival or beat mixture training.**  
   - In Table 2, OptMerge and WUDI both reach or exceed the “Mixture Training” row on several metrics for InternVL2.5, and notably outperform the base InternVL2.5-Instruct average by a wide margin.  
   - In Table 3, OptMerge on Qwen2-VL achieves an average of 78.24, beating the strong Qwen2-VL-Instruct baseline (75.63) and clearly outperforming all individual tasks.  
   - Table 10 shows large gains (e.g., MMMU 39.33 vs best single 38.00, ScienceQA 91.89 vs best single 76.54) on general QA benchmarks, supporting the claim that properly merged “capability experts” can exhibit emergent multi-skill behavior not present in any single expert.

7. **Computational efficiency and practical relevance.**  
   Table 7 (Page 9) provides wall-clock and memory comparisons: the proposed 300-iteration data-free optimization uses ~0.22h/2.6GB for InternVL2.5 and ~3.8h/22GB for Qwen2-VL, versus ~25h/240–256GB for mixture training. These numbers make a convincing case that this merging regime is appealing for practitioners with limited compute.

8. **Overall presentation quality.**  
   The paper is generally well written, with equations and assumptions (Assumptions A.1–A.5, Lemmas A.6–A.9) made explicit. The methodology section is relatively clear, and the benchmark pipeline, datasets, and evaluation tooling (VLMEvalKit, LMMs-Eval, GPT-4o-mini answer extraction) are described in sufficient detail for reproduction.

## Weaknesses

1. **Theoretical assumptions are quite strong and not well matched to realistic MLLM finetuning, which limits the practical force of Theorem 3.1.**  
   - The analysis in Appendix A assumes deterministic GD with fixed step size, PL condition (Assumption A.2), and L-smoothness for each task loss. Real MLLM finetuning uses stochastic optimizers (Adam, possibly with weight decay, warmup, etc.), non-stationary learning-rate schedules, and highly nonconvex losses that are unlikely to satisfy PL globally.  
   - Directional similarity (Assumption A.3: \(\cos(-\nabla \mathcal{L}_i,\tau_i)\ge\kappa\)) and approximate orthogonality between task vectors (Assumption A.4) are plausible but unverified; the “near-orthogonal task vectors” claim is cited from prior work, yet the paper does not empirically show cosines or leakage statistics \(\delta\) on their own MLLM benchmark.  
   - As a result, the bound in Theorem 3.1 largely serves as a qualitative justification of the “over-training hurts merging” story, which is also already observed in earlier works. It would be more compelling if the authors empirically sanity-checked some of these assumptions on their actual models, e.g., measuring \(\cos(\tau_i,\tau_j)\) across capabilities or layering the analysis with SGD noise.  
   This is not a fatal flaw, but the theoretical section oversells the practical precision of the bound relative to its assumptions.

2. **Definition and implementation details of OptMerge are under-specified in critical places.**  
   - Equation (3) replaces \((\tau_{i,l})^\top\) by \(\Sigma_{1:k}V_{1:k}^\top\), but it is not fully clear how this interacts with the dimensionality of linear layers in transformers. Explicit shapes for \(U\in\mathbb{R}^{m\times r}, \Sigma\in\mathbb{R}^{r\times r}, V\in\mathbb{R}^{n\times r}\) are given, but “m” and “n” are not linked to model dimensions (input vs output). This is minor but makes it harder to mentally simulate the optimization.  
   - In Sec. 4.2, the authors argue that gradients are nonzero only in the directions of nonzero singular values for LoRA task vectors; this is true for the WUDI-style loss, but the statement as written is easy to overinterpret as a property of general optimization, which is not accurate. A more explicit link to the loss structure (Eq. (1)) and its effective subspace would clarify the point.  
   - The optimization is said to be applied only to linear layers (Page 7), yet most modern MLLMs use attention projections, feed-forward layers, and sometimes non-linear adapters. It is unclear which exact weight matrices are optimized (Q/K/V/O, MLP in, MLP out, Q-Former, connector MLPs?) and how this was chosen. The choices may materially influence whether merging affects mainly language or multimodal components.

3. **Hyperparameter selection uses evaluation data for λ, which weakens “data-free” claims.**  
   - Page 7 specifies that the scalar merging coefficient \(\lambda\) is tuned by grid search over [0.1, 0.3, 0.5, 0.7, 1.0, 1.5] “for all model merging methods,” but it is not clearly stated whether the tuning is done on held-out validation sets or directly on the benchmarks used for reporting (e.g., VizWiz/GQA/TextVQA).  
   - If test sets are used for \(\lambda\) tuning, this is a methodological issue that could inflate reported performance and is at odds with the “data-free static merging” framing. Even if small, this should be clarified, and ideally a validation split would be used (especially given that many of the evaluation datasets already have train/val/test splits).  
   - The same concern applies to the rank size \(k\) heuristic (“rank of each task vector divided by the number of tasks”) and learning rates; while Table 8 explores different \(k\) ratios, the main experiments fix the rank without fully justifying why this heuristic is optimal or at least robust.

4. **In places, the paper overstates novelty relative to prior multimodal merging and omni-benchmarks.**  
   - The related work section cites UnIVAL, VL-merging, VisionFuse, DAMC, AdaMMS, and UQ-Merge, which is good. However, there is no discussion of more recent benchmarks and MLLM unification efforts (see “Potentially Missing Related Work” below), some of which also attempt to assess unified multimodal capability or model composition across modalities.  
   - The paper claims “we introduce the first model merging benchmark that provides a fine-grained categorization of MLLM capabilities” (Page 2). This might be true for *data-free weight-space merging* per se, but it would be more accurate if the claim were less absolute and acknowledged neighboring efforts on unified multimodal evaluation.

5. **Effect size of OptMerge over strong baselines is modest and sometimes inconsistent.**  
   - On InternVL2.5 (Table 2), WUDI achieves an average of 74.48 vs OptMerge’s 73.94. The text states “average improvement of 0.44% over WUDI” (Page 9), but this appears to be referring to other settings; in this particular table, WUDI is actually slightly higher. The authors should double-check and reconcile this statement.  
   - On Qwen2-VL (Table 3), OptMerge’s 78.24 is indeed better than TSV (77.76), TIES (77.77), and Task Arithmetic (75.85). The relative gain over the best baseline is about +0.5 points absolute, which is meaningful but not large given the noise typical in MLLM evaluation. Confidence intervals or repeated runs would strengthen the claim that OptMerge’s advantage is statistically robust.  
   - For modality merging (Table 5), TSV Merging actually attains the best average performance (67.34) among static methods, slightly surpassing OptMerge (67.00). For Hugging Face checkpoints (Table 6), improvements of OptMerge over TIES+DARE and TSV are minor (≤0.4 on the average column). Overall, OptMerge is a strong and stable method, but “achieving the best results” is not consistently supported across all scenarios.

6. **Benchmark design choices are partially ad hoc and could bias the comparison.**  
   - The fine-tuned experts are *constructed by the authors* using specific hyperparameters (Sec. C, Page 23). While this is unavoidable to some extent, small choices such as learning rate (e.g., 4e-5 full finetune vs 1e-5 LoRA) and one-epoch training, plus deciding to drop Chinese data for Qwen2-VL, could significantly affect task-vector norms (Figure 2) and thus the relative difficulty of merging.  
   - There is no comparison to *off-the-shelf* expert MLLMs beyond the Hugging Face experiment (Table 6), and even there, the base Qwen2-VL-Instruct is treated as a separate line rather than the central baseline for the merging of those four experts. It would be informative to see, for example, “Mixture training” of those four heterogeneous HF datasets on a common base to better contextualize merging performance.

7. **Some important implementation details and reproducibility aspects are pushed to the appendix or left implicit.**  
   - Evaluation uses GPT-4o-mini to extract numeric/choice answers for MathVista and MATH-Vision (Page 23–24). While the prompt is included, there is no analysis of extraction errors or sensitivity to the judge model. A simple sanity check (e.g., measuring extraction accuracy on a subset with ground-truth outputs) would be reassuring.  
   - The choice to apply OptMerge “exclusively to the linear layer in the model” (Page 7) needs clearer articulation: which exact layers, and do they include LoRA-only parameters or also backbone weights? This matters particularly in modality merging where the connectors and encoders stay fixed.

8. **Mathematical exposition could be crisper in key derivations.**  
   - In Sec. 4.1, Eq. (2) gives the SVD of \(\tau_{i,l}-\bar\tau_l\), but the footnote “noise present in the top and lower singular vectors” is conceptually confusing; usually one associates noise with low singular values. Later, the text correctly says they take \(U_{1:k},\Sigma_{1:k},V_{1:k}^\top\), so the “top singular vectors” should represent signal, not noise. This inconsistency should be fixed.  
   - In Appendix D.1, Equations (4–6) attempt to interpret task vectors as weighted sums of input vectors. However, \(\sum_t \sum_n (\partial \mathcal{L}/\partial(\theta x)) x^\top\) is labeled as “coefficient · x^\top”, but the coefficient term still depends on \(n\); the derivation could benefit from explicitly indexing or averaging over samples to avoid confusion. Clarifying these manipulations would make the intuition about task vectors as data surrogates more convincing.

Overall, the paper is strong, but cleaning up some of these issues would significantly improve its rigor and clarity.

## Potentially Missing Related Work

1. **Xie et al., “MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models,” 2026.**  
   - Relevance: Proposes a benchmark for unified multimodal understanding and generation, which is closely aligned with the paper’s goal of assessing unified multimodal capabilities, albeit via model merging.  
   - Suggested integration: Discuss in Section 2 (Related Work) alongside MMBench, Seed-Bench, etc., to better position OptMerge’s benchmark within the landscape of unified multimodal evaluation.

2. **Smirnov & Carruba, “Evaluating Multimodal Commercial and Open-Source Large Language Models for Dynamical Astronomy,” 2026.**  
   - Relevance: Provides a systematic evaluation of multimodal LLMs on a specialized domain; relevant as another example of capability-specific multimodal benchmarking and could inform how domain-specific experts might be merged.  
   - Suggested integration: Briefly mention in Section 2 when discussing domain-specific multimodal benchmarks and the need for capability-focused evaluation.

3. **Wu et al., “Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging,” 2025.**  
   - Relevance: Directly uses model merging to improve reasoning efficiency, conceptually similar in treating task vectors as composable units.  
   - Suggested integration: Compare and contrast in Section 2 under “Model merging” and in Section 5.2 when discussing the potential of merging to enhance reasoning and general capabilities.

4. **Luo et al., “ImageScope: Unifying Language-Guided Image Retrieval via Large Multimodal Model Collective Reasoning,” 2025.**  
   - Relevance: Deals with unifying language-guided image retrieval through multimodal model collectives, conceptually related to modality and capability composition.  
   - Suggested integration: Add to “Model merging in MLLMs” discussion, clarifying how OptMerge differs by operating in weight space instead of collective reasoning or routing.

5. **Ma et al., “Unifying Multimodal Retrieval via Document Screenshot Embedding,” 2024.**  
   - Relevance: Another approach to unifying multimodal tasks, here via representation-level unification rather than merging weights, useful as a contrast to weight-space merging.  
   - Suggested integration: Brief mention in Section 2 as an alternative unification paradigm, reinforcing that OptMerge is one of several possible paths toward omni-models.

6. **Hemker et al., “Multimodal Lego: Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine,” 2024.**  
   - Relevance: Explores model merging and fine-tuning across modalities in a specific domain (biomedicine), directly related to the paper’s modality-merging focus.  
   - Suggested integration: Discuss alongside DAMC, UnIVAL, and VL-merging in Section 2, emphasizing differences in domain, architecture, and whether merging is data-free or data-based.

## Questions

1. **Data usage for hyperparameter tuning.**  
   - How exactly is the merging coefficient \(\lambda\) selected? Is it tuned on the same datasets used for final evaluation (e.g., VizWiz, GQA, TextVQA), or is there a separate validation split or held-out subset? Please clarify the protocol and, if currently tuned on test data, consider re-running key results with a proper validation setup.  
   - Similarly, for the rank-size heuristic \(k = \text{rank}(\tau_{i,l})/5\), did you try alternative definitions (e.g., cumulative energy thresholds)? How sensitive are the main results in Tables 2–3 and 5–6 to this choice?

2. **Empirical validation of directional similarity and orthogonality assumptions.**  
   - Could you provide empirical measurements of \(\cos(\tau_i,\tau_j)\) and the cross-task cosine leakage \(\delta\) on your InternVL and Qwen2-VL benchmarks? Even a histogram or summary statistics would help ground Assumptions A.3–A.4 and Lemma A.6.  
   - Have you examined \(\cos(-\nabla \mathcal{L}_i,\tau_i)\) during finetuning to confirm the directional similarity assumption?

3. **Layer selection for optimization and impact on modalities.**  
   - Which specific linear layers are optimized by OptMerge? For example, do you apply it to attention Q/K/V/O projections and MLP layers inside the language model only, or also to visual/audio/video connectors and Q-Former layers?  
   - In the modality merging setup (Table 5), where the encoders and connectors are kept fixed, is OptMerge applied only to the Vicuna layers with LoRA parameters? Sharing more detail here would help others reproduce and extend your setup.

4. **Statistical significance and variance.**  
   - Are the reported results in Tables 2, 3, 5, 6, and 9 averages over multiple runs, or single runs per method? If single runs, do you have any indication of run-to-run variance, particularly for OptMerge vs TSV / TIES on Qwen2-VL and modality merging?  
   - If feasible, please provide standard deviations for at least a subset of settings, or at least clarify that variance is small enough that the observed improvements (e.g., ~0.5–1 point) are meaningful.

5. **Robustness of GPT-4o-mini-based answer extraction.**  
   - For MathVista and MATH-Vision, have you quantified how often the GPT-4o-mini “answer extractor” fails or disagrees with a simple regex/heuristic on a subset? Could extraction errors systematically favor or hurt particular models (e.g., those that produce verbose vs concise answers)?  
   - If not, consider adding a small validation showing that extraction is accurate enough not to dominate the benchmark results.

6. **Potential for dynamic or test-time merging.**  
   - You explicitly focus on data-free static merging, but the theoretical insights about \(\eta T\) and \(\delta\) seem relevant for test-time adaptation as well. Do you foresee a way to adapt OptMerge’s low-rank denoising to incremental or task-conditioned merging (e.g., dynamic selection of subspaces based on input)?

Clarifications on these points would increase my confidence in both the methodology and the experimental conclusions.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methodology is well motivated, the optimization formulation (Eq. (3)) is reasonable, and experiments are extensive and largely convincing, though theory relies on strong assumptions and some hyperparameter-tuning details need clarification.

## Presentation Rating

3: good.  
The paper is generally clear, with sensible structure, helpful figures (e.g., Figures 2–4, 5–6), and detailed experimental tables. A few mathematical derivations and implementation details could be sharpened.

## Contribution Rating

3: good.  
The combination of a capability-structured MLLM merging benchmark, a theoretically grounded analysis of fine-tuning vs merging, and a practically effective optimization-based merging method is a solid and useful contribution, even if improvements over the strongest baselines are sometimes modest.

## Overall Rating

8: Accept, good paper (poster).  
Despite some oversimplified theoretical assumptions and minor methodological questions, the paper provides a valuable benchmark, a well-motivated optimization variant (OptMerge), and strong empirical evidence that data-free model merging can rival or surpass mixture training for multimodal LLMs and modality fusion. This is a timely and practically relevant contribution that should be of interest to the ICLR community.

## Reviewer Confidence

4: confident.  
I am familiar with model merging, multimodal LLMs, and optimization-based data-free techniques and have read the math and experiments carefully. Some assumptions regarding PL and gradient behavior in large models are difficult to fully verify, but they do not critically undermine my overall assessment.