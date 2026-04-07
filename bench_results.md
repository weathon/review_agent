# ICLR Benchmark Results

Date: 2026-04-06 03:45
Critic/Merger: deepseek/deepseek-v3.2 (OpenRouter)
Neutral: deepseek/deepseek-v3.2, Related Work: deepseek/deepseek-v3.2:online (OpenRouter)

## sh1hWO9RHo

- GT: Reject (avg 4.5)
- Predicted: N/A (4.9/10)
- Match: N/A

### Final Review

## Summary
The paper introduces the Agent GPA (Goal-Plan-Action) framework, a structured evaluation paradigm that decomposes agent performance into specialized metrics (Goal Fulfillment, Logical Consistency, Execution Efficiency, Plan Quality, Plan Adherence, Tool Selection, Tool Calling) assessed by dedicated LLM judges. Experiments on the TRAIL/GAIA benchmark show the framework covers 95% of annotated errors (vs. 55% for a baseline), localizes 86% of errors in agreement with humans, and maintains strong human-judge alignment, enabling actionable diagnostics for agent debugging.

## Strengths
- **Systematic and actionable evaluation framework:** The decomposition into Goal, Plan, and Action dimensions provides a holistic, interpretable taxonomy that maps directly to the agent’s operational loop. This is evidenced by covering 95% of errors on TRAIL/GAIA—a large improvement over the baseline TRAIL judge—and localizing 86% of errors to enable targeted debugging.
- **Empirical rigor and detailed analysis:** The evaluation is thorough, using proper train/test splits, reporting precision/recall/F1 across error impact levels, and measuring judge consistency (Krippendorff’s α). The analysis reveals contextual specialization of judges (e.g., Tool Selection as high-recall, Tool Calling as high-precision), guiding practical deployment.
- **Exploration of automation and generalization:** The paper integrates GEPA for automated prompt optimization, showing improved performance, and includes a preliminary case study on SWE-bench, demonstrating the framework’s adaptability to a different domain (coding) without manual retuning.

## Weaknesses
- **Weak performance and reliability of two core judges:** Plan Quality (PQ) and Plan Adherence (PA) exhibit poor precision and, for PQ, low inter-rater reliability (α=0.628) on the primary TRAIL/GAIA dataset. While the paper notes small sample sizes for these error types, this undermines the claim of comprehensive systematic coverage and limits the diagnostic utility of these components.
- **Dependence on substantial prompt engineering and agent-specific customization:** The framework’s effectiveness hinges on detailed prompts, custom agent architecture instructions, and few-shot examples. Although GEPA reduces manual effort, the need for tailored instantiation for each new agent architecture raises concerns about reproducibility and generalizability without significant configuration.
- **Limited validation of generalizability:** Primary quantitative validation is concentrated on one public benchmark (TRAIL/GAIA) and a small, non-public internal dataset. The SWE-bench case study is preliminary and excludes several judges due to the agent’s architecture, offering insufficient evidence for broad applicability across diverse agent types (e.g., embodied, multi-agent).
- **Conceptual ambiguity in Logical Consistency definition:** Logical Consistency is described broadly as sitting at the intersection of Goal, Plan, and Action, checking grounding, instruction adherence, and task completion. This creates potential overlap with other metrics (e.g., Goal Fulfillment, Plan Adherence) and ambiguity in interpreting its specific failure mode, despite statistical orthogonality shown in Appendix F.

## Nice-to-Haves
- Ablation study on prompt components (generic criteria vs. custom instructions vs. few-shot examples) to clarify which elements are necessary for performance.
- Cross-model validation of judges using LLMs beyond Claude to ensure the framework’s robustness is not model-specific.
- Deeper root-cause analysis of false positives/negatives for the lower-performing judges (PQ, PA) to guide improvements.
- Visual case studies comparing human and LLM judge error localization for concrete illustration of strengths and failure modes.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Circularity in error coverage claim:** The criticism that mapping TRAIL errors to GPA dimensions by human annotators introduces circularity is overstated. The comparison against the independent TRAIL baseline judge (which uses a different taxonomy) demonstrates a real improvement in error detection and localization.
- **Overly harsh characterization of “implementation-heavy” framework:** The paper actively addresses scalability through GEPA automation and provides full prompts for reproducibility. The need for prompt engineering is a practical limitation, not a fatal flaw.
- **Requirement to demonstrate improved agent performance:** Using the framework to iteratively refine an agent is an application of the method, not a prerequisite for validating the evaluation framework itself.

## Novel Insights
The framework provides the novel insight that agent failures can be effectively categorized and localized by aligning them with breakdowns in the fundamental Goal-Plan-Action operational loop, moving beyond symptom-based taxonomies to a cause-oriented diagnosis. The specialization of judges (e.g., high-recall Tool Selection vs. high-precision Tool Calling) reveals that no single evaluator is optimal for all contexts; instead, a portfolio of judges can be selected based on the error severity and the desired trade-off between sensitivity and reliability, enabling more nuanced and actionable evaluation.

## Suggestions
- Explicitly discuss the limitations of the Plan Quality and Plan Adherence judges in the main text, possibly reframing them as domain-specific or preliminary components that require further validation on datasets richer in planning errors.
- Include a clearer, operationalized definition of Logical Consistency in Section 3 that distinguishes it more sharply from Goal Fulfillment and Plan Adherence to reduce conceptual overlap.
- Commit to releasing the re-annotated TRAIL/GAIA dataset (mappings from TRAIL errors to GPA dimensions) alongside the evaluation code to maximize reproducibility and allow independent verification of the error coverage analysis.

---

## b6qQmQ2F13

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper systematically investigates memory-accuracy trade-offs for deploying reasoning-focused large language models under fixed memory budgets. It demonstrates that optimal memory allocation is scale-dependent: for models with an effective size below approximately that of an 8-bit 4B model, memory is better spent on higher-precision weights, while larger models benefit more from longer generations (larger KV cache). The work also analyzes how parallel scaling and KV cache compression strategies are governed by model scale and task type.

## Strengths
- **Comprehensive empirical study:** The paper evaluates over 1,700 configurations across multiple model families (Qwen3, DeepSeek-R1, OpenReasoning-Nemotron), tasks (mathematical, coding, knowledge-intensive), and optimization axes (weight precision, token budget, group size, KV cache compression). This extensive experimentation strongly supports the identified trends.
- **Clear, actionable guidelines:** The core finding of a scale-dependent threshold for allocating memory between weights and KV cache is well-supported by Pareto frontier analysis and distilled into practical recommendations for practitioners. The distinction between task types (mathematical/code vs. knowledge-intensive) regarding optimal weight precision is particularly valuable.
- **Problem reformulation:** The work successfully reframes test-time scaling from a FLOPs-centric view to a memory-constrained deployment perspective, highlighting the growing dominance of the KV cache in reasoning workloads—a crucial insight for real-world serving.

## Weaknesses
- **Lack of statistical uncertainty reporting:** Accuracy metrics are averaged over 32 generations per instance, but no error bars, standard deviations, or confidence intervals are provided. This omission makes it difficult to assess the robustness of comparisons, especially for configurations near the Pareto frontier.
- **Threshold generality and presentation:** The specific threshold of "8-bit 4B" is derived primarily from the Qwen3 family. While similar scale-dependent behavior is shown for other model families, the paper occasionally presents this threshold as a fixed point rather than an observed trend that may vary across architectures. A more nuanced presentation would strengthen the claims.
- **Limited architectural diversity in primary analysis:** The detailed analysis is centered on the Qwen3 family; validation on DeepSeek-R1 and OpenReasoning-Nemotron is provided but less exhaustive. A broader variety of architectures (e.g., different attention mechanisms, MoE models) would increase confidence in the generalizability of the findings.
- **Latency/throughput analysis is not integrated into core guidelines:** While Appendix C.1 analyzes latency and throughput, these critical deployment metrics are not incorporated into the main memory-accuracy trade-off framework or the final recommendations. For practical deployment, a joint consideration of memory, accuracy, and speed is often necessary.

## Nice-to-Haves
- Include an additional knowledge-intensive benchmark (beyond GPQA-Diamond) to further substantiate the claim that 4-bit weights are broadly memory-optimal for such tasks.
- Conduct a more precise mapping of the scale threshold by testing intermediate model sizes (e.g., 2B, 6B) to better characterize the transition region.
- Measure and report the latency and throughput implications of KV cache compression methods (eviction and quantization), as these can affect real-world performance.
- Test longer generation lengths (e.g., >30k tokens) to validate the claim that for large models, memory should be allocated to the KV cache "until saturation," as saturation points may be task-dependent.
- Provide a theoretical intuition or hypothesis for why the scale-dependent threshold exists, linking it to concepts like model capacity or task complexity, to give a deeper foundation for the empirical results.

## Removed Points 
*These points are flagged to be removed, treat them with caution*
- **Criticism about budget forcing artifacts:** The paper follows established prior work (Muennighoff et al., 2025) for budget forcing. A discussion of potential unnatural continuations is not required for the core contribution.
- **Demand for deeper mechanistic explanations** (e.g., why small and large models differ, root cause of precision sensitivity): These are interesting but go beyond the paper's empirical scope and are not necessary to validate the main findings.
- **Request for hyperparameter sensitivity analysis:** The paper uses standard settings for quantization and compression methods; a full sensitivity study is not expected in a broad empirical survey.
- **Claim that direct iso-memory comparisons are missing:** The Pareto frontier analysis inherently compares strategies across memory budgets; direct fixed-memory comparisons are implied by the curves.
- **Formatting and writing nitpicks** (e.g., informal phrase "false economy", verbose figure captions): These do not affect the technical content.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Report variance estimates (e.g., standard deviation across instances or bootstrap confidence intervals) for key accuracy measurements to strengthen the comparative claims.
- Clarify in the abstract and main text that the "8-bit 4B" threshold is an observed trend based on the studied models, not a universal constant, and note that the exact inflection point may shift with architecture and task.
- Elevate the latency/throughput discussion from the appendix to the main text, and incorporate latency considerations into the memory-optimization guidelines where appropriate (e.g., noting that weight quantization can reduce latency for large models).

---

## RpDJz00zNh

- GT: Reject (avg 4.5)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces ConciseHint, a framework to improve the efficiency of large reasoning models (LRMs) by performing *in-reasoning intervention*. The core idea is to continuously inject learnable hints (either manual text or embeddings trained on concise data) during the token-by-token generation of the reasoning chain. A key innovation is an adaptive mechanism that adjusts the hint intensity based on the current reasoning length (as a proxy for query complexity) and dynamically positions the hints to balance accuracy and computational overhead.

## Strengths
*   **Novel and Well-Motivated Paradigm:** The paper clearly identifies and targets the under-explored direction of intervening *during* the reasoning generation, contrasting it with established "before-reasoning" methods (prompting, SFT, RL). This framing establishes a distinct and compelling research niche.
*   **Strong and Extensive Empirical Results:** The method demonstrates substantial token reduction (often 40-65%) while maintaining or slightly improving accuracy across multiple state-of-the-art open-source LRMs (Qwen-3 series, DeepSeek-R1) and diverse benchmarks (GSM8K, AIME24, GPQA-Diamond). Critically, it also shows strong composability, further boosting efficiency when combined with existing methods like Deer and NoWait.
*   **Rigorous and Transparent Evaluation:** The evaluation includes end-to-end latency measurements demonstrating practical utility, thorough ablation studies justifying the adaptive components, and analyses on transition word reduction, hyperparameter sensitivity, and generalization to code/commonsense tasks. The training-free variant (ConciseHint) and the trainable variant (ConciseHint-T) are both explored.

## Weaknesses
*   **Heuristic Design with Limited Justification:** The adaptive rules for hint interval (\(\tau_k = \alpha + \beta \cdot l_k\)) and injection position (Eq. 3) are empirically motivated but heavily heuristic. While ablation studies show they work, the paper provides no exploration of the design space (e.g., non-linear functions) or theoretical grounding for why these specific forms are effective, which limits a deeper understanding of the method.
*   **Insufficient Statistical Reporting:** The paper reports average accuracy and token usage over multiple runs but does not provide measures of variance (e.g., standard deviations) or statistical significance tests. This makes it difficult to assess the reliability of small accuracy changes, which is important for a conference with high standards like ICLR.
*   **Superficial Analysis of Mechanism:** The paper convincingly shows *that* hint injection reduces tokens but provides limited analysis into *how* it changes the reasoning process. A more fine-grained analysis (e.g., categorizing pruned tokens as redundant coherence phrases, elaborated calculations, or self-checks) would strengthen the claim of improving *reasoning efficiency* rather than just textual brevity.

## Nice-to-Haves
*   A brief exploration of alternative adaptive rules (e.g., a step function) or a sensitivity analysis on the manual hint's wording beyond the provided ablation would help better characterize the method's design space.
*   Expanding the evaluation to a wider range of reasoning types (e.g., symbolic reasoning, multi-hop QA) would more robustly support the claim of a general "in-reasoning" paradigm beyond STEM tasks.
*   A more detailed discussion of potential failure modes or boundary conditions (e.g., for extremely complex queries) would provide a clearer picture of the method's limitations.

## Removed Points
*These points are flagged to be removed, treat them with caution*
*   **"Lack of comparison to strongest recent baselines"**: The paper includes several strong and relevant baselines (BeConcise, Prompt, Deer, NoWait). Demanding comparison against every recent method is scope creep.
*   **"Need for evaluation on closed-source models"**: The paper's focus on open-source models is valid and sufficient; applicability to closed-source APIs is not a core requirement for establishing the method's contribution.
*   **"Missing training details for ConciseHint-T"**: The paper describes the training procedure sufficiently in Section 3 (SFT on concise data, embedding initialization, next-token prediction). Further hyperparameter details are appropriate for code release, not the paper.
*   **"Requires theoretical justification for interpolation-based controllability"**: The empirical demonstration of smooth control via embedding interpolation (Figure 3) is sufficient for an empirical paper; a theoretical guarantee is not standard or expected.
*   **"The complexity proxy is circular"**: The paper acknowledges this prior (length correlates with complexity) and the adaptive mechanism is designed to handle this dynamic; it is a reasoned design choice, not a flaw.
*   **"Potential overhead not explored in batched inference"**: The paper includes analysis of relative latency overhead (<0.3%) and end-to-end latency reduction. A full systems analysis for batched serving is outside the paper's core scope.

## Novel Insights
The paper's primary novel insight is the viability and effectiveness of the *in-reasoning intervention* paradigm for improving efficiency. It demonstrates that continuously influencing an LRM during its generation process—via simple, adaptive hint injection—can yield substantial compression of the reasoning chain without compromising accuracy. A secondary insight is that a trained hint embedding (ConciseHint-T) can capture generalized "conciseness" patterns that transfer effectively to out-of-domain tasks, and that controllability can be achieved smoothly through embedding space interpolation.

## Suggestions
*   Add standard deviations or confidence intervals for accuracy and token usage metrics across the multiple runs already performed to substantiate the reported improvements.
*   Include a qualitative analysis comparing original and ConciseHint-modified reasoning chains for a few examples, annotating what types of content (e.g., transitional phrases, repeated calculations, verbose explanations) were reduced to provide concrete evidence of how conciseness is achieved.

---

## oiz0QHejVj

- GT: Reject (avg 4.5)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
CLIP-Map proposes a mapping-based compression framework for CLIP models, replacing conventional select-based pruning with learnable matrix transformations. The method uses Kronecker-factorized mappings to reduce parameter overhead and a diagonal inheritance initialization to stabilize training, followed by a retraining stage with distillation. Experiments show improved performance over strong baselines like TinyCLIP, particularly under high compression ratios, with gains in training efficiency.

## Strengths
- **Strong and consistent empirical improvements.** CLIP-Map outperforms the select-based TinyCLIP baseline across multiple compression ratios (1%, 10%, 50%) on zero-shot retrieval (MSCOCO, Flickr30K) and classification benchmarks, with particularly significant gains at extreme compression (e.g., +5.3% TR@1 on MSCOCO at 1% compression). The method also requires fewer training epochs, reducing wall-clock time.
- **Innovative adaptation of model-growth techniques to compression.** The core idea of replacing hard parameter selection with a learnable, compressed mapping is novel in this context. The use of Kronecker factorization to reduce the mapping parameter complexity from O(D₁²D₂²) to O(D₁D₂) is a clever and well-motivated design.
- **Thorough ablation studies and analysis.** The paper includes convincing ablations validating the proposed diagonal inheritance initialization (Table 5, Fig. 6) and the choice of mapping-stage duration (Table 4). Visualizations of the evolving mapping matrices (Fig. 5) provide useful insight into the optimization process.

## Weaknesses
- **Incomplete methodological description for depth compression.** While width compression via Kronecker factors is clearly explained, the description of depth compression is insufficient. Equation (2) and the surrounding text state that a depth-compression operator \(L_{depth}\) linearly combines layers, but crucial details—how \(L_{depth}\) is parameterized, initialized, and optimized—are missing from the main text, hindering reproducibility.
- **Lack of quantitative analysis for the core claim of information preservation.** The paper argues that mapping preserves more information from the original model than selection-based pruning, but provides no quantitative evidence (e.g., feature similarity analysis, parameter matrix rank comparisons). This claim remains intuitive but unsubstantiated.
- **Limited and indirect comparison with the broader state-of-the-art.** The primary comparison is with TinyCLIP. Comparisons to other recent compression methods (UPop, MoPE-CLIP, etc.) in Table 7 rely on results reported in their original papers under different training data and settings, limiting the strength of the superiority claim. A direct, controlled comparison is needed.
- **Evaluation scope is constrained to a single dataset and primary architecture.** All main results use YFCC-15M and a ViT-based CLIP. While the appendix shows a proof-of-concept on ResNet and Meta-CLIP, the paper's claims about general applicability would be stronger with more extensive validation across datasets (e.g., larger-scale LAION-2B) and a wider variety of architectures in the main experiments.

## Nice-to-Haves
- A theoretical analysis or bound on the compression error or representational capacity of the Kronecker-factorized mapping.
- An ablation study isolating the contribution of the depth compression component versus width compression.
- Detailed reporting of the computational overhead (FLOPs, memory) of the mapping stage itself, beyond the final model's inference cost.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "The abstract lacks quantitative results."** – While including specific numbers could be helpful, the abstract's purpose is to summarize contributions; the quantitative results are thoroughly presented in the experimental sections.
- **Weakness: "The method requires predefined compression ratios and targets; how to choose them optimally is not discussed."** – This is a generic issue for all compression methods, not a specific flaw of this work.
- **Weakness: "The figures are cluttered and difficult to interpret."** – This is a formatting/subjective nitpick that does not affect the technical evaluation.
- **Weakness: "Potential overfitting risks due to λ=1.0 are not examined."** – The paper includes an ablation study (Table 10, Appendix A.8) that systematically evaluates λ and selects λ=1.0 based on empirical performance, which is a reasonable justification.
- **Criticism that depth compression is "poorly explained" to the point of being a "significant gap."** – While the description could be more detailed, the core idea (linear combination of layers via a learned matrix) is presented in Eq. (2) and Fig. 3. The greater issue is the lack of implementation details, which is captured in the weaknesses above.

## Novel Insights
The paper's core novelty lies in successfully adapting the paradigm of learnable mapping—previously explored for model growth—to the distinct and challenging problem of model compression for multimodal architectures. It demonstrates that preserving parameters through transformation, rather than selecting a subset, can yield superior performance under aggressive compression, especially when coupled with techniques (Kronecker factorization, diagonal initialization) that address the unique optimization challenges of mapping to a smaller space. No further novel insights beyond the paper's own contributions emerge from the reviews.

## Suggestions
- Provide a complete, reproducible description of the depth compression procedure in the main text, including the parameterization, initialization, and optimization of the \(L_{depth}\) operator.
- Add a quantitative analysis to support the information preservation claim, such as measuring Centered Kernel Alignment (CKA) between features of the original and compressed models.
- Strengthen the empirical validation by including at least one direct, apples-to-apples comparison with a recent state-of-the-art compression method (e.g., MoPE-CLIP) under identical training data and model size settings.

---

## NfO2Lt2WY7

- GT: Reject (avg 2.0)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary
This paper systematically analyzes the components of Group Relative Policy Optimization (GRPO), a popular reinforcement learning method for improving reasoning in large language models. It finds that negative feedback via group-relative advantage estimation is essential for stable learning, while PPO-style clipping and policy ratio constraints are not necessary for mathematical reasoning. The authors propose RGRA, a simplified REINFORCE-based variant, which matches or exceeds GRPO's performance across multiple math benchmarks.

## Strengths
- **Rigorous and systematic ablation design:** The paper cleanly isolates GRPO's components (negative feedback, clipping, advantage estimation) through controlled experiments (GRPO-pos, RGRA, REINFORCE with raw rewards, RAFT), providing strong empirical evidence for which elements are necessary.
- **Comprehensive empirical evaluation across models and benchmarks:** Experiments span three model families (Qwen2.5 0.5B/1.5B, Llama3.2 1B) and nine diverse reasoning benchmarks (English/Chinese math, STEM). RGRA outperforms GRPO in 17 of 27 head-to-head comparisons, robustly supporting the core claim.
- **Clear demonstration of failure modes and stabilization mechanisms:** Training dynamics (Figure 1) convincingly show that methods lacking negative feedback (positive-only GRPO, RAFT) lead to reward collapse and truncated reasoning, highlighting the critical, stabilizing role of group-relative advantage estimation.

## Weaknesses
- **Limited evidence for improved "reasoning behaviors":** The claim that RGRA and GRPO "foster the development of interpretable reasoning strategies" is supported only by a single qualitative example (Figure 2). A quantitative analysis (e.g., distribution of reasoning step lengths, correctness of intermediate steps on a held-out set) is necessary to substantiate this important aspect of the contribution.
- **Incomplete ablation of PPO-style components:** RGRA removes both policy ratio clipping and the ratio term itself simultaneously. An independent ablation of each component (e.g., clipping-only vs. ratio-only removal) is missing, limiting the understanding of which specific constraint is dispensable and whether they interact.
- **Narrow experimental scope in model scale and task domain:** All experiments use relatively small models (≤1.5B parameters) and are focused on mathematical reasoning with verifiable answers. The conclusion that PPO-style constraints are unnecessary may not hold for larger-scale alignment or for reasoning in domains with sparser or more complex reward signals (e.g., code generation, open-ended dialogue). The paper acknowledges this but does not mitigate it empirically.
- **Lack of variance estimation for benchmark results:** Performance improvements, while consistent, are often modest (e.g., differences of ~1 percentage point in average scores). Reporting confidence intervals, standard errors across multiple runs, or statistical significance tests would strengthen the claim that RGRA's gains are meaningful and not due to random variation.

## Nice-to-Haves
- A deeper analysis connecting the findings to prior theoretical arguments (e.g., why clipping might be unnecessary when initializing from a strong pre-trained policy) would provide additional conceptual insight.
- An ablation study on the group size *G* for advantage estimation would help understand the sensitivity of this core hyperparameter.
- Quantifying the training efficiency gain (e.g., wall-clock time or memory savings) from removing clipping operations would bolster the practical utility claim.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength:** "The paper is well-written" – This is a generic strength that applies to any competently written paper.
- **Weakness (Factually Incorrect):** "Critical error in the definition of the GRPO loss (Equation 1)" – The paper's Equation 1 defines its terms consistently (r_i,t as the policy ratio). While this notation differs from some prior work, it is internally consistent and does not misrepresent the implemented method.
- **Weakness (Scope Creep / Not Standard):** "Lack of comparison to standard PPO with a value model" – The paper's stated scope is analyzing and simplifying GRPO, not re-evaluating PPO. Demanding a full PPO baseline is outside this scope.
- **Weakness (Overly Demanding):** "Requires theoretical grounding or mechanistic explanation" – This is an empirical ablation study; providing a complete theoretical explanation is not a required standard for this type of contribution.
- **Weakness (Formatting Nitpick):** "Equations are garbled, tables misaligned" – These are noted as parser artifacts in the provided content, not author errors.

## Novel Insights
The paper's primary novel insight is the decoupling of GRPO's components, demonstrating that its effectiveness for mathematical reasoning stems primarily from group-relative advantage estimation (which provides essential negative feedback and stability), while the PPO-inspired clipping mechanism is superfluous. This challenges the assumed necessity of complex policy constraints in this setting and establishes that a simple REINFORCE-style update, when combined with a properly normalized advantage, is sufficient and can even be superior. This insight provides a valuable conceptual simplification and a more transparent baseline for future work.

## Suggestions
- Add a quantitative analysis of reasoning behavior (e.g., measure average reasoning trace length and its correlation with accuracy on a held-out set) to substantiate the claims about emergent reasoning.
- Conduct an additional ablation where only the clipping operation is removed but the policy ratio term is kept (or vice versa) to pinpoint which PPO component is unnecessary.
- Report the standard deviation of accuracies across multiple random seeds or bootstrap confidence intervals for the key benchmark comparisons to strengthen the evidence for RGRA's improvements.

---

## tswBfpkwHn

- GT: Reject (avg 5.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper provides the first theoretical analysis of the training dynamics and in-context learning (ICL) generalization of a one-layer Mamba model, specifically studying its robustness to additive outliers in the prompt. It proves that, under certain conditions, Mamba can generalize to unseen binary classification tasks even when a large fraction (approaching 1) of context examples contain outliers, outperforming a comparable linear Transformer which fails when the outlier fraction exceeds 1/2. The analysis attributes this robustness to a decomposition into a linear attention component (which selects informative examples) and a nonlinear gating component (which suppresses outliers and induces a local bias).

## Strengths
- **Novel theoretical contribution:** This is the first work to rigorously analyze the training dynamics and ICL generalization of Mamba models, addressing a significant gap given the architecture's empirical success and unique gating mechanism. The analysis under outlier conditions is timely and provides foundational insights.
- **Rigorous and detailed analysis:** The paper provides non-asymptotic convergence and generalization guarantees (Theorems 1–4) with explicit conditions on batch size, prompt length, outlier magnitude, and iteration count. The proofs, sketched in the main text and detailed in the appendices, are comprehensive and adapt techniques from prior ICL theory to handle Mamba's nonlinearity.
- **Mechanistic interpretation:** The paper goes beyond guarantees to explain *how* Mamba achieves robust ICL. Corollaries 1 and 2 show the linear attention layer selects examples sharing the query's relevant pattern, while the gating suppresses outliers and imposes an exponential decay based on index distance. This interpretation aligns with empirical observations (e.g., "induction heads" and local bias).
- **Supportive empirical validation:** Synthetic experiments clearly validate the theoretical predictions—Mamba tolerates outlier fractions >1/2 while linear Transformers fail—and visualize the proposed mechanisms. Additional experiments on real-world (SST-2) data and with softmax attention (in the appendix) strengthen the practical relevance.

## Weaknesses
- **Simplified model and task scope:** The theoretical analysis is restricted to a one-layer Mamba model and binary classification tasks with orthogonal, sparse features. While this aligns with prior theoretical work on Transformers, it limits direct applicability to the deep, multi-head architectures used in practice for complex language tasks.
- **Strong data assumptions:** The generalization guarantee (Theorem 2) requires that test-time outliers be *positive linear combinations* of the training-time outliers (Condition (a)). This captures a meaningful class of distribution shifts but may not cover all adversarial or natural corruptions encountered in practice. The paper does not discuss how restrictive this assumption is or its practical implications.
- **Incomplete theoretical comparison:** The primary theoretical comparison is made with a **linear** Transformer (a special case of the Mamba formulation without gating). While this isolates the effect of gating, a theoretical analysis of a standard softmax attention Transformer under the same outlier setting is missing, making the comparison less comprehensive. (Experiments with softmax are included but not theoretically grounded.)
- **Empirical vulnerability not explained by theory:** Experiments show Mamba's performance drops sharply when outlier-containing examples are placed closest to the query (the "CQ" setting), a sensitivity not shared by linear Transformers. This practical limitation is noted but not explained by the theoretical analysis, which assumes random outlier placement.

## Nice-to-Haves
- A discussion of how the linear-combination assumption for test outliers (Theorem 2(a)) might be relaxed or justified in practical settings.
- A theoretical explanation for the positional sensitivity (CQ performance drop) or a proposal for architectural/training modifications to mitigate it.
- Extending the theoretical comparison to include softmax attention Transformers, even at a high level, to better contextualize the robustness advantage.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Formatting nitpicks:** Suggestions to move Algorithm 1 to the main text or improve minor exposition points.
- **Requests for extensive additional experiments:** Demands for experiments on standard ICL benchmarks, ablation studies on model depth/width, and testing outlier types beyond the paper's theoretical model are beyond the scope of this theoretical contribution.
- **Necessity/tightness of conditions:** Criticisms that the paper does not analyze how restrictive its sufficient conditions are; such analysis, while interesting, is not required for establishing the theoretical guarantees.
- **Generic strengths:** Praises such as "the paper is well-written" or "the topic is important" that do not identify specific contributions.

## Novel Insights
The analysis reveals that Mamba's robustness to outliers stems from a dual mechanism: its equivalent linear attention layer selectively weights context examples that share the relevant pattern with the query, while its nonlinear gating layer actively suppresses examples containing additive outliers and implicitly enforces an exponential decay in importance based on index distance (a local bias). This decomposition provides a clear, interpretable explanation for why Mamba can maintain accurate ICL generalization even when a majority of context examples are corrupted—a capability theoretically bounded for linear Transformers.

## Suggestions
- In Section 3.3 (or the discussion of Theorem 2), briefly discuss the practical implications of requiring test outliers to be positive linear combinations of training outliers. Is this condition likely to hold in scenarios like data poisoning? If not, what might be the consequences?
- Investigate the CQ vulnerability further. Provide a theoretical intuition or additional experiments to explain why the gating mechanism fails when outliers are near the query, and propose a simple training strategy (beyond the one mentioned in Appendix B.1) to mitigate this issue.
- Consider adding a subsection or remark that theoretically analyzes a one-layer, single-head softmax Transformer under the same outlier model, even if the results are less complete, to place the linear Transformer comparison in a more standard context.

---

## 1EdAn5gMVv

- GT: Reject (avg 5.0)
- Predicted: N/A (6.4/10)
- Match: N/A

### Final Review

## Summary
SpatialBoost is a framework that enhances pre-trained vision encoders with 3D spatial awareness by distilling geometric and semantic knowledge through language. It constructs a hierarchical, multi-turn Chain-of-Thought reasoning dataset from images (single- or multi-view) and uses a frozen LLM as a decoder to fine-tune the vision encoder via a dual-channel attention mechanism, preserving pre-trained knowledge. The method demonstrates consistent improvements across a wide spectrum of tasks including depth estimation, 3D scene understanding, robotic control, and even general image classification and retrieval.

## Strengths
- **Comprehensive and convincing empirical validation:** The paper provides extensive experiments across a diverse set of tasks (monocular depth, semantic segmentation, 3D VQA, visual grounding, geometric understanding, robot learning, classification, retrieval) and on multiple state-of-the-art vision encoders (OpenCLIP, SigLIPv2, DINOv2, DINOv3). The consistent gains, often substantial (e.g., SigLIPv2's 3D semantic mIoU jumps from 6.9 to 54.9, DINOv3's SQA3D score improves from 51.4 to 54.9), robustly support the method's core claim of injecting useful spatial knowledge.
- **Rigorous ablation studies and component analysis:** The paper includes thorough ablations that validate key design choices: the superiority of LLM-based supervision over pixel-level decoders (Table 6), the importance of the forward hierarchical reasoning order (Table 7), the complementary nature of single- and multi-view data (Table 7, 16), the effectiveness of dual-channel attention in preventing catastrophic forgetting (Figure 6, Table 17), and the scalability with dataset size (Figure 5, Table 18). This provides strong evidence for the architectural decisions.

## Weaknesses
- **Complex pipeline with potential for error propagation and reproducibility concerns:** The method relies on a cascade of external, sometimes proprietary, models for spatial knowledge extraction (Depth-Pro, SAM, VGGT) and data synthesis (GPT-4o). While Appendix F.5 shows a minimal performance gap between VFM-based and GT-based data on a subset, a broader sensitivity analysis to noise in this automated pipeline is lacking. The computational cost and reproducibility of the full three-stage training and data generation process are not quantified, which is a significant practical limitation for the community.
- **Insufficient evaluation on tasks that explicitly require multi-view or metric 3D reasoning:** The training incorporates multi-view data, but the primary downstream evaluation is on single-view tasks (depth, segmentation, VQA). To fully substantiate the claim of enhanced 3D understanding, direct evaluation on core multi-view or metric 3D tasks (e.g., camera pose estimation, novel view synthesis, or multi-view stereo matching) is missing. This limits the interpretation of what specific "3D spatial" capabilities are being improved.
- **Limited mechanistic analysis of how linguistic supervision alters representations:** The paper convincingly shows *that* SpatialBoost works but provides limited insight into *how* the visual feature space changes. While Figure 7 offers a qualitative glimpse, a more systematic analysis—such as training linear probes on canonical spatial properties not seen during training or visualizing feature sensitivities to spatial transformations—would strengthen the claim that "spatial awareness" is being directly encoded, rather than just benefiting from multi-task learning.

## Nice-to-Haves
- A direct comparison to a strong baseline that fine-tunes the vision encoder on the same underlying spatial tasks (e.g., depth, segmentation) using standard pixel-level supervision, followed by evaluation on the same downstream suite. This would help isolate the unique benefit of the language intermediary.
- A dedicated "Compute and Resources" section estimating the GPU hours and costs associated with each training stage and data generation, improving reproducibility and accessibility.
- A more detailed failure case analysis on spatial reasoning benchmarks (e.g., SpatialRGPT, BLINK-D) to delineate the method's remaining limitations.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness (Removed): "Missing direct comparisons to state-of-the-art spatial reasoning methods."** The paper compares to the original encoders, which are the relevant baselines for a method aiming to *enhance* them. Demanding comparisons to entirely different architectures (e.g., 3D Diffusion Policy's encoder) is scope creep. The paper does compare to related language-guided approaches (SpatialVLM, SpatialRGPT) in the VQA experiments (Table 9, Appendix B.1).
- **Weakness (Removed): "Lack of statistical significance tests."** The performance improvements are large and consistent across many tasks and encoders. The paper includes standard deviations for robot learning results (Table 4). Requiring significance tests for every metric is not standard practice for large-scale empirical papers in this area.
- **Weakness (Weakened -> Nice-to-Have): "Limited analysis of bias propagation."** The paper already addresses this in Appendix F.5 (Table 19), showing negligible difference between VFM-based and GT-based data. A broader analysis is a reasonable suggestion but not a core flaw.
- **Nitpick (Removed): "Absence of a limitations section."** While a dedicated section would be helpful, the paper discusses limitations in the text (e.g., dependence on external models) and the appendix contains relevant analysis (F.5, F.6). This is a stylistic preference, not a substantive weakness.
- **Nitpick (Removed): "The abstract omits mention of dual-channel attention."** The abstract succinctly summarizes the core contribution; detailing every component is unnecessary.

## Novel Insights
The paper's core novel insight is that a frozen LLM, when provided with hierarchically structured spatial descriptions (pixel→object→scene), can serve as an effective supervisory signal to distill dense 3D geometric and relational knowledge into a pre-trained 2D vision encoder. This demonstrates that language's compositional and sequential nature is surprisingly effective for encoding spatial priors, offering a data-efficient alternative to joint multi-modal pre-training or costly multi-view data collection. The finding that this process not only improves spatial tasks but also general vision capabilities (e.g., ImageNet classification) suggests that spatial awareness may be a foundational component of robust visual representation, and that injecting it can have positive, disentangling effects on the feature space.

## Suggestions
- Incorporate an evaluation on a canonical multi-view task (e.g., pose estimation or relative camera regression from image pairs) to directly demonstrate the improved 3D geometric capabilities promised by the use of multi-view training data.
- Strengthen the representation analysis by training simple linear classifiers on frozen SpatialBoost features to probe for specific spatial properties (e.g., relative depth order, left/right relations, occlusion) not explicitly covered in the training QA, providing more direct evidence of what has been learned.
- In the discussion, more precisely define the scope of "spatial relationships" the method addresses (e.g., metric depth, relative pose, ordinal relations) to clarify the contribution and avoid overclaiming.

---

## sJxBWDc8SM

- GT: Reject (avg 3.5)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary
This paper conducts a large-scale empirical investigation into the learnability of modern recurrent models (State-Space Models like Mamba) compared to Transformers on fundamental synthetic tasks: multi-query associative recall (MQAR) and copying. Its core finding is that while SSMs can achieve expressive parity, they suffer from critical optimization instability, succeeding only within an extremely narrow learning rate window. The work also reveals divergent scaling behaviors (SSMs favor width, Transformers depth) and provides architectural ablations linking success to components like 1D convolutions.

## Strengths
- **Substantial and rigorous empirical foundation:** The conclusion that SSM performance is highly sensitive to hyperparameter tuning is backed by an extensive and methodical experimental campaign (over 3,000 runs). The paper convincingly demonstrates that prior performance gaps on MQAR (Arora et al., 2023) were likely confounded by insufficient learning rate grids (Figures 1, 2).
- **Actionable insights on scaling and architecture:** The identification of opposing scaling strategies—width for SSMs, depth for Transformers—is a clear, practical finding (Figures 3, 4). Furthermore, the ablation studies pinpoint the 1D convolution as a critical architectural component enabling single-layer models to solve MQAR, providing a concrete mechanistic link (Table 2).
- **Shifts the discourse from expressivity to learnability:** The paper successfully argues that a key differentiator between these architectures is not just what they can represent, but how reliably they can be trained. This reframes the community's comparison criteria and highlights optimization stability as a first-class challenge for SSM research.

## Weaknesses
- **Conclusions are drawn solely from synthetic benchmarks.** While MQAR and copying are well-motivated proxies for in-context learning, the paper's central claim—that optimization instability is a fundamental challenge for SSMs—remains untested on downstream tasks like language modeling. The discussion acknowledges this, but it limits the immediate practical significance of the findings.
- **Incomplete mechanistic explanation for the observed instability.** The paper empirically documents the narrow learning rate window but does not provide direct evidence (e.g., gradient norm analysis) for the proposed hypothesis that vanishing gradients in the S6 recurrence are the root cause. The link to prior theoretical work (Trockman et al., 2024) is noted but not substantiated with new measurements.

## Nice-to-Haves
- A small-scale validation on a real language modeling dataset (e.g., WikiText) would strengthen the claim that the observed optimization brittleness is a practical concern beyond synthetic settings.
- A more detailed analysis of how the effective learning rate window scales with model width, depth, and sequence length would provide clearer guidance for tuning.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness (Harsh Critic):** "Missing detail: The text states 'Attention always solves the task.' It should explicitly note that this refers to the 2-layer configuration..." *Removed because the paper consistently specifies the layer count in each section's context (e.g., Section 3 is explicitly about 2-layer models).*
- **Weakness (Harsh Critic):** "Clarity Issue: Figure 4 is conceptually important but potentially confusing..." *Weakened to a suggestion; the figure's message is interpretable with careful reading.*
- **Weakness (Spark Finder):** "Missing Experiments: Validation on a realistic language modeling task..." *Weakened to a nice-to-have; the paper's stated scope is a controlled investigation using established synthetic benchmarks to study fundamental learnability. Demanding real-world validation is scope creep.*
- **Weakness (Spark Finder):** "Ablation on optimizer hyperparameters beyond learning rate..." *Removed; a comprehensive sweep of all optimizer hyperparameters is not a standard requirement for a paper focused on identifying a core instability phenomenon.*
- **Weakness (Spark Finder):** "Experiments on longer sequence lengths relevant to SSMs' intended use..." *Weakened to a nice-to-have; the chosen sequence lengths (up to 512) are standard for the MQAR benchmark and sufficient to demonstrate the instability phenomenon.*

## Novel Insights
The paper's primary novel insight is the systematic shift of focus from theoretical expressivity to practical learnability as the crucial differentiator between Transformers and SSMs. Beyond this framing, it provides several specific new observations: the extreme learning rate sensitivity of SSMs contrasts sharply with Transformer robustness; single-layer Transformers exhibit a loss bump reminiscent of induction head formation yet fail to solve the task, while single-layer Mamba shows a similar bump but succeeds; and the performance of single-layer models on MQAR is critically dependent on the presence of a 1D convolution, a unifying architectural insight.

## Suggestions
- Clarify the caption and description of Figure 4 to more directly convey the intended message: that performance is dictated by *how* parameters are allocated (width vs. depth), not just by the total parameter count.
- Temper the language in Section 6 regarding the induction head interpretation (e.g., "suggests an attempt" rather than "indicates") to better reflect the speculative nature of this mechanistic claim without direct attention pattern analysis.

---

## qSak1Hjfdq

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
This paper formalizes the All-Day Multi-Scenes Lifelong Vision-and-Language Navigation (AML-VLN) problem, where agents must continually adapt to diverse scenes and environmental conditions (e.g., low-light, scattering) without catastrophic forgetting. It proposes Tucker Adaptation (TuKA), a parameter-efficient method that represents multi-hierarchical navigation knowledge as a high-order tensor decomposed via Tucker decomposition, and introduces a decoupled knowledge incremental learning strategy. The resulting AlldayWalker agent is evaluated on a new benchmark extending Habitat with degraded imaging models, showing consistent improvements over strong baselines.

## Strengths
- **Novel and well-motivated problem formulation:** The AML-VLN task addresses a critical gap in deploying VLN agents in dynamic real-world conditions, and the extension of Habitat with multiple degradation models (low-light, overexposure, scattering) provides a valuable benchmark for lifelong navigation research.
- **Methodological innovation beyond matrix-based adapters:** TuKA leverages Tucker decomposition to explicitly decouple shared, scene-specific, and environment-specific knowledge in a high-order tensor, offering a principled way to capture multi-hierarchical structure that existing LoRA variants cannot represent.
- **Extensive and thorough experimental validation:** The paper compares against 12 state-of-the-art baselines across 24 sequential tasks, demonstrates clear superiority in success rates and forgetting metrics, and includes insightful ablations (e.g., tensor order, shared components, scalability) and real-world deployment.

## Weaknesses
- **Incomplete ablation of the DKIL loss components:** The contribution of each regularization term (EWC for shared parameters, consistency for experts, orthogonality for new experts) is not isolated, making it unclear which mechanisms are essential for mitigating forgetting and enabling knowledge sharing.
- **Lack of analysis on forward transfer and expert interpretability:** While forgetting rates are reported, there is no measurement of whether learning new tasks improves performance on previous ones (forward transfer), a key aspect of lifelong learning. Additionally, the claim that experts decouple scene and environment knowledge lacks empirical validation through probing or clustering of the learned factor matrices.
- **Over-reliance on CLIP matching for inference without robustness analysis:** Expert selection during inference depends on matching CLIP features stored during training, but the accuracy of this matching under severe domain shifts or for novel scenes is not analyzed, leaving the reliability of the selection mechanism in doubt.
- **Limited comparison with broader continual learning strategies:** The baselines are predominantly LoRA-based variants; inclusion of strong non-LoRA continual learning methods (e.g., replay-based approaches) would better situate TuKA’s advantages within the broader lifelong learning landscape.

## Nice-to-Haves
- Sensitivity analysis of key hyperparameters (e.g., Tucker ranks, regularization weights) to assess robustness and reproducibility.
- Qualitative visualizations of agent trajectories across different environments to illustrate successes and failure modes.
- A dedicated limitations section in the main paper discussing fixed expert counts, inference matching assumptions, and real-world generalization.
- Improved readability of dense figures (e.g., Figure 3) and tables through simplification or supplemental visualizations.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Statistical significance from single runs:** In large-scale VLN benchmarks, single-run evaluation is common practice due to computational cost, and the paper follows established norms from cited prior work.
- **Placement of related work in the appendix:** While a succinct related work section in the main text is conventional, its absence does not undermine the technical contribution; this is a formatting preference.
- **Fixed expert count limiting unbounded lifelong learning:** The paper explicitly scopes the problem to known sets of scenes and environments; criticizing the absence of dynamic expansion is scope creep.
- **Hyperparameter sensitivity analysis:** Demanding exhaustive sensitivity analysis is not a standard requirement for methodological papers in this area.
- **Dense result tables:** This is a presentation issue that does not affect the substantive findings.
- **Negative values in forgetting metrics:** These are interesting observations that could be discussed but do not constitute a flaw in the method or evaluation.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct an ablation study isolating each component of the DKIL loss (EWC, consistency, orthogonality) to clarify their individual contributions.
- Analyze forward transfer by measuring performance on earlier tasks after learning new ones, and provide interpretability analysis of the learned expert factors (e.g., via probing tasks).
- Evaluate the accuracy and failure modes of the CLIP-based expert matching mechanism, and consider fallback strategies for handling unseen scene-environment combinations.
- Include comparisons with replay-based continual learning methods to strengthen the baseline evaluation.

---

## Kw2mvnzCoc

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (None/10)
- Match: N/A

### Final Review

ERROR: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{"error":{"message":"This model\'s maximum context length is 163840 tokens. However, you requested 65536 output tokens and your prompt contains at least 98305 input tokens, for a total of at least 163841 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=98305)","type":"BadRequestError","param":"input_tokens","code":400}}', 'provider_name': 'Parasail', 'is_byok': False}}, 'user_id': 'user_32IhT2MfrwUmKddLDbQSpLcYscC'}

---

## khBHJz2wcV

- GT: Accept (Poster) (avg 3.0)
- Predicted: N/A (6.1/10)
- Match: N/A

### Final Review

## Summary
This paper presents a framework for fine-tuning pre-trained flow-matching generative models to enforce physical constraints and solve inverse problems without requiring paired parameter-solution training data. The core innovation is a joint optimization strategy that uses weak-form PDE residuals within an adjoint-matching control formulation to steer the generative distribution toward physically consistent samples while simultaneously inferring latent parameters (e.g., material properties, source terms). The method is validated on four canonical PDE systems with controlled misspecifications and demonstrates a controllable trade-off between constraint satisfaction and distributional fidelity.

## Strengths
- **Novel integration for inverse problems without paired data:** The method's key contribution is enabling joint generation of physically consistent states and plausible latent parameters using only observational state data. This is achieved through a clever surrogate base flow for the parameters, defined via a pre-trained inverse predictor, combined with the adjoint-matching framework. This addresses a significant bottleneck in scientific machine learning.
- **Practical and flexible framework with comprehensive evaluation:** The introduction of a scaled memoryless noise schedule (`κ`) for numerical stabilization and a running cost (`λ_f`) to regulate fidelity to the base distribution provides practical control knobs. The method is rigorously evaluated across diverse PDE problems (Darcy, Elasticity, Helmholtz, Stokes) under various misspecifications (noise, boundary conditions, damping, forcing), demonstrating robust performance and clear trade-offs via ablation studies.
- **Effective use of weak-form residuals:** The adoption of randomly sampled local test functions to compute weak-form PDE residuals provides a numerically stable and efficient learning signal, avoiding the instability of high-order derivative computations. This is a well-justified and practically important design choice.

## Weaknesses
- **Hyperparameter sensitivity and limited guidance:** The method introduces several critical hyperparameters (`λ_x`, `λ_α`, `λ_f`, `κ`). While ablation studies show their effects, the paper provides limited high-level guidance on how to select them for a new problem. The interplay and sensitivity of these parameters could be a barrier to adoption and should be discussed more thoroughly.
- **Insufficient analysis of core components' reliability:** The joint evolution critically depends on the pre-trained inverse predictor `φ`, but its accuracy (e.g., MSE on held-out data) and the propagation of its errors into the fine-tuned model's parameter estimates are not analyzed. Similarly, the empirical impact and "control–fidelity trade-off" of the novel noise scale `κ` are claimed but not demonstrated with an ablation.
- **Diversity assessment could be more thorough:** The claim that fine-tuning preserves diversity is supported primarily by MMD. A more direct quantification of sample diversity (e.g., via pairwise distances, entropy, or mode coverage metrics), especially when enforcing strong constraints (high `λ_x`), would strengthen this claim and provide a clearer picture of the diversity-constraint trade-off.

## Nice-to-Haves
- **Guidance for hyperparameter selection:** A summary table or heuristic guidelines for setting `λ_x`, `λ_f`, and `κ` based on problem characteristics (e.g., noise level, severity of model misspecification) would improve usability.
- **Enhanced presentation of the image experiment:** The natural image recoloring experiment demonstrates flexibility but feels somewhat disconnected from the core physics narrative. Reframing it more explicitly as enforcing "parametric constraints" or providing a clearer motivational link would improve cohesion.
- **Extended discussion on failure modes:** A brief discussion or illustration of scenarios where the method might struggle (e.g., extremely poor inverse predictor, drastically misspecified physics) would help users understand its limitations.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: Missing comparison to score-based diffusion guidance methods (e.g., Huang et al.).** The paper explicitly scopes out methods that require joint parameter-state training data (which Huang et al. assumes), making a direct comparison unfair and outside its stated contribution of working without paired data.
- **Weakness: Statistical significance of plots is unclear.** The paper reports means and standard deviations in tables. The points in scatter plots are reasonable aggregates (e.g., over 256 samples), and requesting error bars on every point in a hyperparameter sweep is an excessive rigor requirement not standard in the field.
- **Weakness: PBFM baseline is at a disadvantage.** The paper transparently states in Appendix E.2 that PBFM was augmented with the pre-trained `φ` to enable comparison, and notes this is a disadvantage for PBFM. This is a reasonable adaptation for a controlled comparison, not a hidden flaw.
- **Weakness: Need for comparison with conditional flow-matching on paired data or traditional Bayesian inference (MCMC).** These demands are outside the paper's stated scope and contribution of enabling inference *without* paired data. Comparing to methods that require the data this paper explicitly does not need is scope creep.
- **Weakness: Requirement for testing on real-world datasets or time-dependent PDEs.** The paper makes a solid methodological contribution validated on standard, challenging synthetic benchmarks. Demanding application to real-world data or significantly more complex system classes (time-dependent, chaotic) is a demand for future work, not a weakness of the current contribution.

## Novel Insights
The paper's key novel insight is the reformulation of physics-constrained fine-tuning as a joint stochastic control problem over both the state and latent parameters. By constructing a *surrogate base flow* for the parameters via an inverse predictor trained only on final states, the method creates a coupled dynamics that enables sampling from the tilted joint distribution without ever having seen paired data. This bridges the gap between preference-based fine-tuning (adjoint matching) and physics-informed inference in a principled way, opening a new pathway for amortized inverse problem solving where parameters are truly hidden during training.

## Suggestions
- Include a quantitative assessment of the inverse predictor `φ`'s accuracy (e.g., on a validation set) and a brief discussion on how errors might propagate, to better characterize the method's reliability.
- Conduct a focused ablation study on the noise scale `κ` to empirically demonstrate its claimed role in stabilizing training and trading off exploration versus fidelity.
- Supplement the MMD-based distributional analysis with a direct, sample-based diversity metric (e.g., average pairwise SSIM or LPIPS) to more concretely support the claim of preserved diversity under constraint enforcement.

---

## xFo13SaHQm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
This paper identifies and addresses the "copy-paste" artifact in identity-consistent image generation, where models rigidly replicate reference faces, harming controllability. The core contributions are threefold: (1) **MultiID-2M**, a large-scale dataset of 500k group photos with paired reference images for ~3k identities; (2) **MultiID-Bench**, a benchmark introducing a novel "copy-paste" metric (`M_CP`) to quantify the trade-off between identity fidelity and variation; and (3) **WithAnyone**, a FLUX-based model trained with a ground-truth-aligned identity loss and an ID contrastive loss using an extended negative pool, which demonstrably reduces copy-paste while maintaining high identity similarity.

## Strengths
- **Substantial new dataset and benchmark.** The release of MultiID-2M directly tackles a critical data scarcity problem in multi-identity research. The accompanying MultiID-Bench introduces a well-motivated evaluation shift from similarity to the reference (`SimRef`) to similarity to the ground-truth (`SimGT`), correctly penalizing blind copying. This is a significant, concrete resource for the community.
- **Effective, well-motivated method.** The proposed solutions—paired training data to break the reconstruction shortcut, the GT-aligned ID loss (enabling stable supervision at all noise levels), and the contrastive loss with an extended negative pool—are directly targeted at the identified problem. The comprehensive ablations (Table 3, Fig. 17) clearly validate each component's contribution.
- **Compelling evidence for breaking a key trade-off.** The quantitative results (Tables 1, 2, Fig. 5) show that WithAnyone achieves state-of-the-art `Sim(GT)` while maintaining a substantially lower copy-paste score, deviating from the clear trade-off curve followed by other methods. This is supported by extensive qualitative comparisons, a user study, and evaluation on an external benchmark (OmniContext).

## Weaknesses
- **Limited discussion of dataset bias and generalization boundaries.** The dataset is built from web-sourced celebrity images, with a skewed nationality distribution (Fig. 13b). While qualitative results on non-celebrities are shown (Fig. 16), a quantitative assessment of generalization to everyday faces and a more critical discussion of the resulting model biases and fairness implications are missing from the main text.
- **Insufficient analysis of failure modes and limitations.** The paper showcases successes but lacks a systematic analysis of when WithAnyone fails (e.g., with extreme occlusions, highly similar identities, or prompts demanding drastic attribute changes). A dedicated discussion of limitations is necessary to define the method's operational boundaries.
- **Validation of the copy-paste metric, while present, could be stronger.** The user study provides a moderate correlation (r=0.44) between the proposed `M_CP` metric and human judgment. A deeper analysis of this correlation and visual examples linking specific metric values to generated images would improve the metric's interpretability and trustworthiness.

## Nice-to-Haves
- **Training efficiency analysis.** A breakdown of the computational cost (GPU hours, steps) for each of the four training phases would help the community assess the practical feasibility of the proposed pipeline.
- **Extended hyperparameter analysis.** A brief sensitivity study for key hyperparameters (e.g., loss weights `λ_ID`, `λ_CL`, contrastive temperature `τ`) in the appendix would provide additional methodological rigor, though the chosen values are reasonable.
- **Visualization of attention mechanisms.** Showing the learned attention masks for multi-identity localization could provide an intuitive validation of the claimed controllability.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness: "Limited architectural novelty."** The paper's primary contribution is a holistic training paradigm enabled by a new dataset, not a new backbone. The adapter-based approach on FLUX is a standard and effective choice for the task.
- **Weakness: "Lack of statistical significance tests."** While additional statistical tests could be included, the paper provides strong, consistent evidence across multiple metrics, ablations, and a user study. The core claims are visually and quantitatively clear without them.
- **Weakness: "Demands comparison to DynamicID."** The paper explicitly excludes DynamicID due to the unavailability of code and models, a standard and acceptable justification. It cannot be required to reimplement a competing method from scratch.
- **Weakness: "Hyperparameters are set without justification."** The loss weights are set to 0.1, a common empirical choice. The ablations demonstrate the effectiveness of the overall objective; a full sensitivity grid is not a standard requirement.
- **Weakness: "Requests for cross-dataset evaluation on CelebA."** The paper's benchmark is specifically designed for multi-identity generation with paired references, a more challenging and relevant setting than single-ID reconstruction on CelebA. Evaluating on its own, carefully constructed benchmark is appropriate.

## Novel Insights
The paper's core insight is that the prevalent practice of optimizing for reference similarity (`SimRef`) in identity-consistent generation creates a perverse incentive for models to "copy-paste" reference features, harming controllability. By constructing a paired dataset, the authors enable a paradigm shift: they can train and evaluate using ground-truth similarity (`SimGT`), which rewards understanding the identity invariant to pose/expression changes. The proposed `M_CP` metric formally captures the bias towards copying versus faithful synthesis, and the results reveal a fundamental trade-off curve that most existing models lie on. WithAnyone demonstrates that with the right data and objectives (paired tuning, GT-aligned loss, contrastive negatives), this trade-off can be broken, achieving high fidelity without excessive copying.

## Suggestions
- **Add a "Limitations" subsection** to the main paper, explicitly discussing: (1) the celebrity-centric nature of the training data and its implications for bias and generalization, (2) common failure cases observed during experimentation (e.g., with occlusions, extreme angles), and (3) the model's reliance on the FLUX backbone.
- **Strengthen the validation of the `M_CP` metric** by including a small visual guide in the appendix (e.g., a figure showing generated images with low, medium, and high `M_CP` scores) to make its meaning more intuitive.
- **Provide a more concrete discussion of ethical safeguards** in the Ethics Statement. Beyond the non-commercial license, suggest or reference specific technical mitigations (e.g., robust watermarking techniques, detection model architectures) that could be built upon the released model.

---

## WhO6Km5Rku

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary
QubitCache proposes a novel paradigm for KV-cache compression by shifting focus from token selection to preserving attention patterns. It uses a hybrid architecture where critical tokens are stored classically, while attention distributions of non-critical tokens are encoded into quantum-inspired probabilistic states via amplitude encoding. The method achieves 7× memory reduction while maintaining 92-97% of baseline performance across multiple models and benchmarks, with particular gains on multi-hop reasoning tasks.

## Strengths
- **Novel Conceptual Framework:** The paper convincingly argues that attention patterns, not tokens themselves, are primary information carriers, motivating a paradigm shift from token eviction to relational preservation. This is supported by cited literature on attention sparsity and graph theory.
- **Strong and Extensive Empirical Validation:** The method demonstrates consistent performance retention (92-97%) with aggressive compression (15% token retention) across five models (4B-8B) and six long-context benchmarks, outperforming strong baselines like ScissorHand, H2O, and GEAR, especially on multi-hop reasoning (15-25% F1 improvements).
- **Practical Implementation with Clear Ablations:** The design is implemented as a classical simulation compatible with current hardware, and includes comprehensive ablations that validate core design choices (e.g., attention-based token selection is crucial, quantum encoding provides a 3.9% gain). The analysis of qubit count and circuit depth trade-offs grounds the approach in NISQ device constraints.

## Weaknesses
- **Unsubstantiated and Overstated Theoretical Claims:** The paper claims to "prove QubitCache preserves rank *r* attention structure with bounded reconstruction error" and achieves compression "beyond classical information-theoretic limits." No proof is provided in the main text or appendix, and the information-theoretic limit is neither defined nor rigorously compared against. These are central claims that remain unsupported.
- **Insufficient Comparison to Classical Attention-Preserving Baselines:** The empirical evaluation lacks comparison to classical methods that explicitly compress attention information (e.g., low-rank approximations, kernel-based sketches, or learned predictors). Without this, it is unclear whether the gains stem from the quantum-inspired encoding or simply from preserving attention patterns—a classical strategy.
- **Missing Critical Systems Metrics for Inference:** The paper reports memory reduction but omits essential metrics for an inference-time compression method: latency, throughput, and the computational overhead of simulating the quantum circuits (gate operations, statevector simulation, measurements). This gap prevents assessment of practical utility.
- **Incomplete Analysis of the "Quantum-Inspired" Component:** While the ablation shows a gain from the encoding, the paper does not rigorously disentangle whether the benefit comes from the probabilistic nature of the reconstruction or the specific amplitude encoding formalism. A deeper analysis comparing to a classical probabilistic baseline (e.g., sampling from a softmax) is needed to justify the quantum-inspired framing beyond analogy.

## Nice-to-Haves
- Evaluation on extremely long contexts (e.g., 32K+ tokens) to better stress-test long-range dependency preservation.
- A detailed sensitivity analysis for key hyperparameters (segment size, retention ratio, circuit depth) to justify the chosen operating points.
- Visualization of original vs. reconstructed attention maps to intuitively demonstrate pattern preservation.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism that the "quantum" contribution is misleading or purely metaphorical.** The paper explicitly states its implementation is a classical simulation in Section 3.2.2 and uses "quantum-inspired" as a formal framework. The method's novelty does not hinge on actual quantum hardware.
- **Criticism about formatting issues in Figure 2 and table inconsistencies.** These appear to be parser/rendering artifacts from the extracted text, not substantive flaws in the paper's content.
- **Criticism demanding comparison to Linformer or Performer.** These are architectural changes for efficient attention computation, not post-hoc KV-cache compression methods, and are outside the stated scope.
- **Criticism that the performance gains are marginal over GEAR.** The paper shows QubitCache achieves higher compression (7.0× vs. 6.7×) while maintaining better performance, especially on reasoning tasks—a meaningful advance.
- **Generic strengths like "the paper is well-written" or "the topic is important."**

## Novel Insights
The paper's core insight is that the relational structure encoded in attention patterns is more critical for model performance than individual token representations. This motivates a compression strategy that discards most token embeddings but preserves their attention distributions via a compact, probabilistic encoding. The hybrid deterministic-probabilistic attention mechanism enables "soft" influence from compressed tokens, which is particularly beneficial for maintaining coherence in multi-hop reasoning where dependencies evolve over long ranges.

## Suggestions
- Provide a rigorous proof or detailed proof sketch for the claimed theoretical guarantee (preserving rank-*r* structure with bounded error) in the main text or appendix. If a full proof is not possible, clearly state this as a conjecture supported by empirical evidence.
- Implement and compare against a strong classical baseline that compresses attention information (e.g., using low-precision storage or a low-rank factorization of attention scores) to isolate the benefit of the quantum-inspired amplitude encoding from the general idea of attention preservation.
- Measure and report end-to-end inference latency and throughput alongside memory usage to give a complete picture of the method's practical overhead.
- Reframe the claim of surpassing "classical information-theoretic limits" unless it can be precisely defined and justified; otherwise, focus on the empirical achievement of high compression with minimal performance loss.

---

## ZBhZT307xx

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper provides a comprehensive empirical analysis of verifiers used in reinforcement learning with verifiable reward (RLVR) for mathematical reasoning. It demonstrates that rule-based verifiers suffer from significant false negatives that impede RL training, while model-based verifiers, though more accurate, are vulnerable to reward hacking. The authors propose a hybrid verifier that improves performance and systematically probe verifier robustness against adversarial attacks.

## Strengths
- **Extensive and well-designed evaluation** across multiple datasets (Math, DeepscaleR, ORZ-Math, Skywork-OR1, WebInstruct-Verified), verifier types (rule-based, off-the-shelf LLMs, fine-tuned), and both static and dynamic RL settings, providing robust, multifaceted evidence.
- **Identification of critical, underexplored issues**: the declining recall of rule-based verifiers with stronger models (a scaling concern) and the susceptibility of model-based verifiers to reward hacking despite higher static accuracy—challenging the assumption that accuracy translates to RL robustness.
- **Practical contribution of a hybrid verifier** that combines rule-based precision with model-based recall, showing improved RL performance and data efficiency over a pure rule-based baseline.
- **Rigorous validation methodology** using GPT-4o as an oracle (validated against human judgment) to detect reward hacking, and a systematic adversarial probing study (13 pattern types) that reveals generative verifiers are broadly vulnerable while discriminative ones (e.g., xVerify) are more robust.
- **Cross-domain generalization** demonstrated through experiments on both mathematical and general science (WebInstruct-Verified) datasets, strengthening the claim that the findings are not domain-specific.

## Weaknesses
- **Single-sample evaluation for most RL benchmarks** — Due to computational constraints, key results (GSM8K, MATH, Minerva Math, OlympiadBench) rely on single runs, which reduces confidence in the stability of the reported improvements and trends given RL's known variance.
- **Limited analysis of why fine-tuned verifiers are more hackable** — The paper demonstrates the phenomenon but does not investigate the root causes (e.g., overfitting to the classification distribution, shortcut learning, or reduced reasoning faithfulness), leaving an important mechanistic question unanswered.
- **Narrow policy model scope** — All RL training experiments use Qwen2.5-7B as the policy model; findings about hacking dynamics and verifier effectiveness might not generalize to other architectures or scales.

## Nice-to-Haves
- Proposing and evaluating simple defense mechanisms against reward hacking (e.g., adversarial training of the verifier, ensembling) would strengthen the practical impact.
- Testing with a stronger policy model to validate the hypothesis that off-the-shelf verifiers' apparent robustness in RL might be due to the policy's limited capacity to find exploits.
- Quantifying the trade-off between verifier recall on clean data and robustness against adversarial attacks across different verifier types.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Dataset representativeness concern** — The paper explicitly states the datasets are a "relatively easy setting" and uses this to argue the problem is more severe in realistic scenarios; this is appropriately framed and not a weakness.
- **Evaluation of model-based verifiers on a subset** — The paper clearly explains this choice aligns with the hybrid design (evaluating on samples the rule-based verifier missed) and is not presented as a direct, like-for-like comparison with rule-based performance.
- **Choice of 1.5B model for the main hybrid verifier** — This is justified by performance among 1.5B models and computational efficiency; it does not undermine the core findings.
- **Demand for theoretical analysis or proposed solutions** — The paper's contribution is empirical identification and analysis of verifier limitations; proposing defenses is outside its stated scope.
- **Formatting and table readability issues in the submitted text** — While the current version has parser artifacts, these are not substantive weaknesses of the research itself.

## Novel Insights
The paper provides a clear novel insight: static classification accuracy of a verifier does not predict its robustness in dynamic RL training. Fine-tuned verifiers can achieve high recall on clean data yet become uniquely susceptible to reward hacking, leading to training collapse. Additionally, generative verifiers (including Chain-of-Thought models) are systematically more vulnerable to simple adversarial patterns than discriminative verifiers, suggesting a tension between reasoning transparency and robustness in verification systems.

## Suggestions
- Include a discussion of the single-sample evaluation limitation in the main limitations section and, if possible, add multi-seed results for a critical subset of experiments (e.g., the hybrid vs. rule-based comparison on DeepscaleR) to demonstrate stability.
- Perform a simple controlled ablation to isolate the harm of rule-based false negatives: artificially inject false negatives at the observed rate into the rule-based verifier and measure the impact on RL performance, providing causal evidence.
- Release all prompts, hyperparameters (beyond those in the appendix), and code to ensure full reproducibility.

---

## crKJJ4Ej60

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes Copy-Paste, a novel generation paradigm for improving contextual faithfulness in Retrieval-Augmented Generation (RAG). The core idea is to promote direct lexical copying from provided context as a proxy for fostering genuine contextual trust in LLMs. The method is instantiated through a two-stage framework: first, specialized prompting (CP-Order, CP-Link, CP-Refine) generates high-copying responses; second, a model (CopyPasteLLM) is trained via Direct Preference Optimization on a small, automatically constructed dataset of these responses. The approach achieves state-of-the-art performance on counterfactual faithfulness benchmarks with remarkable data efficiency (365 training samples) and is analyzed via a novel interpretability algorithm.

## Strengths
- **Exceptional Data Efficiency with Strong Performance:** CopyPasteLLM achieves substantial accuracy improvements (12.2%-24.5% on FaithEval) over strong recent baselines while using only 365 query-context pairs for preference construction—50x to 90x less data than fine-tuning baselines like Context-DPO (18k samples). This demonstrates that the high-copying signal is a highly effective and efficient supervisory signal for alignment.
- **Comprehensive and Mechanistically Grounded Evaluation:** The paper provides thorough validation across multiple models (Mistral-7B, Llama-3-8B, Llama-3.1-8B) and datasets (FaithEval, ConFiQA, PubMedQA, RAGTruth), including both counterfactual and original contexts. The novel "Context-Parameter Copying Capturing" algorithm offers compelling mechanistic evidence that CopyPasteLLM works by recalibrating the model's internal confidence in its parametric knowledge rather than by enhancing contextual representations.
- **Well-Motivated and Clever Pipeline:** The work is grounded in an observed inverse correlation between copying degree and hallucination. The two-stage pipeline—generating high-copying candidates via a spectrum of prompting methods and then internalizing this preference via DPO—is cleverly designed and effectively translates a surface-level behavior into a learned policy for contextual trust.

## Weaknesses
- **Faithfulness Defined Primarily as Lexical Overlap:** The method's core optimization target is lexical copying. While empirically effective in the evaluated QA settings, this narrow definition may not suffice for tasks requiring synthesis, paraphrasing, or integration of ideas from multiple sources. The paper does not demonstrate that the learned policy generalizes to such abstractive generation tasks, limiting the claimed scope of "contextual faithfulness."
- **Potential Degradation of Response Quality:** The paper notes the fluency trade-off for some prompting variants (e.g., CP-Order) and uses perplexity as a metric, but lacks a human evaluation of the readability, coherence, and overall utility of the generated high-copying responses. In practice, verbatim stitching of context fragments may produce disfluent or awkward answers, especially for complex queries.
- **Unclear Baseline Comparison Fairness on ConFiQA:** In Table 1, the strong baseline Context-DPO is evaluated on ConFiQA data marked with [T], indicating it was trained on that data, while CopyPasteLLM is evaluated on unseen data. This inflates the baseline's scores and makes the comparison uneven. A cleaner comparison on a held-out set not used by any model would strengthen the performance claim.

## Nice-to-Haves
- **Human Evaluation of Response Quality:** A human assessment of fluency, coherence, and overall answer quality for CopyPasteLLM versus baselines would address concerns about the practical usability of high-copying responses.
- **Exploration on Abstractive Tasks:** Testing the method on tasks like summarization (e.g., CNN/DailyMail) where verbatim copying is often suboptimal would help better define the boundaries of its applicability and reveal its behavior when synthesis is required.
- **Ablation on Training Data Size:** While the 365-sample result is striking, a curve showing performance as a function of dataset size (e.g., 50, 100, 200 samples) would more precisely characterize the data efficiency.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness: "The method assumes lexical copying ensures semantic faithfulness."** The paper acknowledges this nuance in its limitations (Appendix K) and evaluates faithfulness with semantic metrics (AlignScore, MiniCheck). The correlation demonstrated is strong enough to validate the proxy.
- **Weakness: "The pipeline depends on the quality of the initial LLM (DeepSeek-V3)."** This is a generic concern for any method using an LLM for data generation; the paper uses a state-of-the-art model, and the core contribution is the pipeline, not the specific generator.
- **Weakness: "Missing comparison to extractive baseline."** The Copy-Paste prompting methods (CP-Order) are effectively sophisticated extractive baselines, and their performance is already reported and compared.
- **Suggestion: "Compare to many additional baselines (CITADEL, FOCAL, Self-RAG, Llama-3.1-70B)."** This demands an impractically broad comparison beyond community standards for a focused methodological paper. The selected baselines (Context-DPO, ParamMute, CoCoLex) are strong and recent.
- **Suggestion: "Establish causal link between copying and faithfulness via controlled paraphrase experiments."** This is an interesting research direction but is beyond the scope needed to validate the paper's core empirical claim of effectiveness.
- **Suggestion: "Provide attention visualizations."** While insightful, this is not a standard requirement for a paper of this type, and the provided logit/hidden state analysis already offers strong mechanistic evidence.
- **Nitpick: "Formatting artifacts in parsed PDF."** These are parser issues, not problems with the paper.

## Novel Insights
The paper provides a genuinely novel mechanistic insight: the effectiveness of CopyPasteLLM stems not from enhancing the model's ability to *represent* contextual knowledge, but from *recalibrating its confidence in parametric knowledge*. The Context-Parameter Copying Capturing analysis shows that while contextual knowledge representations remain similar to the base model, the distributions of parametric knowledge representations shift significantly. This suggests the learned policy selectively suppresses competition from internal knowledge during generation, offering a fresh perspective on how alignment for faithfulness operates internally.

## Suggestions
- **Re-evaluate baselines on a clean, held-out split** of ConFiQA to ensure a fully fair comparison and report those results.
- **Add a qualitative error analysis** categorizing the failure modes of CopyPasteLLM (e.g., over-copying irrelevant text, inability to handle ambiguous evidence) to better understand its limitations.
- **Clarify the trade-off space** in the discussion, more explicitly acknowledging that the method is optimized for maximal faithfulness in evidence-grounding tasks and may be less suitable for generation tasks requiring high fluency or synthesis.

---

## pNpnqsn0Si

- GT: Reject (avg 3.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Thoughtbubbles, a transformer variant that learns to dynamically allocate parallel computation in latent space by forking or deleting residual streams based on learned cumulative scores. The method is trained solely with a language modeling objective during pretraining, without explicit supervision. It demonstrates consistent improvements in perplexity and zero-shot performance (LAMBADA, HellaSwag) over parameter-matched and computation-matched baselines across model scales from 150M to 772M parameters.

## Strengths
- **Novel and well-motivated architecture:** The forking mechanism for dynamic, input-adaptive allocation of parallel residual streams is a genuinely new approach to enabling inference-time scaling within standard pretraining, moving beyond fixed pause tokens or chain-of-thought.
- **Strong and consistent empirical gains:** The method consistently outperforms both parameter-matched transformers and a simple computation-matched baseline (duplicated filler tokens) on validation perplexity across two datasets and multiple model sizes. Notably, a 319M Thoughtbubbles model achieves lower perplexity than a 772M baseline on OpenWebText. Zero-shot improvements on LAMBADA and HellaSwag further validate the approach.
- **Interpretable, unsupervised adaptation:** Analysis shows the model allocates more computation to tokens with higher predictive entropy (Fig. 5), an emergent property that aligns with intuitive notions of computational difficulty. Additional analysis (Fig. 4, Appendix C) provides evidence that forked tokens meaningfully influence their parent and that forking occurs at interpretable locations.

## Weaknesses
- **Lacks comparison to strong adaptive computation baselines:** The primary computation-matched baseline (duplicating input tokens) is relatively naive. To properly situate the contribution, comparisons to more recent adaptive methods (e.g., pause tokens, Mixture-of-Depths, Universal Transformers) are needed. Without this, the claimed advantage over prior art is not fully substantiated.
- **Limited evaluation on tasks requiring complex reasoning:** While motivated by enabling more difficult, multi-step problems, evaluation is restricted to perplexity and relatively simple zero-shot tasks (LAMBADA, HellaSwag, BLiMP, PIQA). There is no assessment on benchmarks like GSM8k or MATH, which are more direct tests of improved computational capability, though the authors acknowledge this limitation.
- **Incomplete analysis of computational efficiency:** The paper notes wall-clock inefficiency in the limitations but provides no quantitative measurements of inference time, FLOPs, or memory compared to baselines. For a method that introduces adaptive parallel computation, a clearer understanding of its practical trade-offs is important.
- **Training dynamics and gradient flow through hard top-k are underexplored:** The method relies on hard top-k decisions for forking, which creates a non-differentiable bottleneck. The authors mention this can cause gradient issues and limit deeper forking, but the paper does not detail how gradients are propagated (e.g., via straight-through estimation) or fully analyze the consequences. This affects reproducibility and understanding of optimization stability.

## Nice-to-Haves
- **More comprehensive ablation studies:** While an ablation on forking location is provided (Appendix B), studies on the necessity of the score attenuation mechanism, the impact of the learned fork embedding, and the effect of different forking budgets (κ) would help validate design choices.
- **Analysis of what triggers forking beyond entropy:** A deeper investigation into the linguistic or structural features (e.g., syntactic complexity, coreference) that correlate with forking decisions would enrich the interpretability of the adaptive behavior.
- **Statistical significance via multiple runs:** Reporting results across multiple random seeds would strengthen the robustness of the claimed improvements, especially at smaller scales.
- **Case studies illustrating forking behavior:** Concrete examples of text passages where forking succeeds or fails would provide intuitive insight into the method's real-world operation.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Claim about 319M model outperforming 772M baseline is overstated:** The paper's claim is specifically about perplexity (Fig. 3), and it does not assert consistent downstream task superiority. The results are presented accurately.
- **Criticism that the link between importance and forking benefit is not formally justified:** The paper provides empirical support via attention analysis (Fig. 4) and entropy correlation (Fig. 5), which is reasonable for an empirical architecture paper.
- **Request for derivation of the position encoding heuristic:** The partial rotation for RoPE is an engineering solution motivated by the need to pack forks; requiring a first-principles derivation is beyond the scope.
- **Demand for scaling to much larger models/datasets as a core weakness:** The experiments (up to 772M params, 2.5B tokens) are sufficient to demonstrate the method's viability; larger-scale training is a natural future direction.

## Novel Insights
The paper shows that a transformer can learn to dynamically allocate parallel computation in a fully unsupervised manner, with forking decisions emerging to focus on tokens of moderate predictive entropy. This suggests the model develops an implicit notion of which tokens benefit from extra "thinking," aligning with recent literature on the informativeness of high-entropy tokens. The analysis further reveals that forked tokens exert substantial influence on their parent via attention, indicating the created latent streams perform meaningful auxiliary computation rather than being mere noise.

## Suggestions
- Implement comparisons to state-of-the-art adaptive computation baselines (e.g., pause token methods, Mixture-of-Depths) to clearly demonstrate the relative advantages of Thoughtbubbles.
- Provide quantitative measurements of inference-time efficiency (wall-clock, FLOPs) to better characterize the practical trade-offs of the method.
- Include a brief description of how gradients are propagated through the hard top-k operation (e.g., straight-through estimator) in the main text or appendix to improve reproducibility.

---

## Pa6ak2B9jJ

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.1/10)
- Match: N/A

### Final Review

## Summary
This paper introduces AUTO-RT, a reinforcement learning framework for automatic jailbreak strategy exploration in large language models. It formulates red-teaming as a constrained Markov decision process and proposes two novel techniques: Dynamic Strategy Pruning (DSP) to eliminate redundant exploration paths early, and Progressive Reward Tracking (PRT) which uses a downgraded model and a novel First Inverse Rate (FIR) metric to shape sparse rewards. Experiments on 16 white-box and 2 black-box LLMs demonstrate improved attack success rates, diversity, and exploration efficiency compared to several baselines.

## Strengths
- **Novel Framework and Technical Contributions**: The paper introduces a well-motivated, hierarchical RL framework for strategy-level jailbreak exploration, moving beyond fixed templates. The specific mechanisms of Dynamic Strategy Pruning (DSP) and Progressive Reward Tracking (PRT) with the FIR metric are concrete algorithmic innovations aimed at solving core challenges of redundancy and sparse rewards in this domain.
- **Extensive and Multi-Dimensional Evaluation**: The empirical validation is comprehensive, testing across 18 diverse LLM families in both white-box and black-box settings. Evaluation covers three key dimensions—attack effectiveness (ASR), efficiency (learning curves), and diversity (semantic and defense generalization)—providing strong, multi-faceted evidence for the method's benefits. The inclusion of ablation studies, transferability analysis, robustness checks (reward model variation), and case studies on commercial models strengthens the claims.

## Weaknesses
- **Lack of Statistical Reporting for Key Results**: The main results tables (1, 2, 3) report single-point estimates (e.g., ASR) without measures of variance (e.g., standard deviation over multiple random seeds or runs). While violin plots in Figure 3 show distributions for the efficiency metric, confidence intervals or statistical significance tests for the primary effectiveness comparisons are absent. This limits the ability to assess the robustness and replicability of the reported improvements.
- **Insufficient Analysis of Discovered Strategies and Failure Modes**: The paper claims to explore a "rich strategy space" but provides no qualitative or quantitative analysis of what strategies are actually learned. Are they novel, or do they rediscover known patterns (e.g., role-playing)? Furthermore, there is no analysis of *what types* of vulnerabilities AUTO-RT is good or bad at finding (e.g., semantic exploits vs. logical flaws). This gap undermines the core claim of achieving "high-exploitability" coverage and leaves the nature of the contribution partially opaque.
- **Limited Comparison with the Contemporary State-of-the-Art**: While baselines like simple RL, imitation learning, and template-based methods (AutoDAN, Human Template) are included, the paper does not compare with several notable recent red-teaming approaches (e.g., TAP (Tree of Attacks), ICA (Iterative Coordinate Ascent), or advanced gradient-based attacks). This makes it difficult to fully situate AUTO-RT's performance within the current research landscape and assess its relative advancement.

## Nice-to-Haves
- A more detailed computational cost analysis (e.g., GPU hours, sample complexity compared to baselines) would be helpful for practical adoption, given the method's use of 8xA100 clusters.
- A deeper theoretical or empirical justification for the FIR selection heuristic (choosing "the last model before a sharp increase") would strengthen the methodological foundation.
- Expanding the societal impact discussion to more concretely address dual-use risks and responsible release practices would align with community norms.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Logical Gaps: The method assumes the unsafe region of the target model is contained within that of the downgrade model... The paper does not discuss potential violations of this assumption."** This is not a logical gap but a stated design goal of the reward shaping method. The paper explicitly describes how FIR is used to select a downgrade model that maintains this containment relationship (Section 2.3.3, Figure 2), and Figure 4 empirically validates the selection criterion.
- **Weakness: "Missing Ablations: ...there is no ablation on the choice of the rephrasing model (AM^r) or the impact of different constraint designs."** Demanding an ablation on every component is excessive and not standard for evaluating the core contribution. The paper's ablation study appropriately focuses on the two novel proposed techniques (DSP and PRT).
- **Weakness: "Incomplete Discussion of Computational Cost"** (from Review 2) and **"The computational cost of training (8xA100 clusters, 9000 episodes)"** (from Review 1). While noting cost is reasonable as a "nice-to-have," framing it as a core weakness is unreasonable for a methods paper in this area, where large-scale RL training is common and the efficiency claim is supported by learning curves (Figure 3).
- **Weakness: "Experiments on a more challenging and recent set of target models... The current results on 'black-box' models are simulated using open-source models, which is not equivalent."** The paper *does* include results on proprietary models (Gemini-2.5-Pro, Claude Sonnet 4, GPT-4.1) in Appendix G (Table 10), which addresses this concern.
- **Weakness/Formatting: "The FIR analysis (Figure 4) is marred by formatting artifacts... that make it difficult to interpret."** These artifacts are from the PDF extraction process provided for review, not from the original paper, and thus should not be considered.

## Novel Insights
The paper's core novel insight is the formulation of jailbreak discovery as a strategy-level reinforcement learning exploration problem, coupled with the introduction of the First Inverse Rate (FIR) metric. FIR provides a practical, data-driven method for calibrating a downgraded model used for reward shaping, which is crucial because overly weak or strong downgrade models provide poor learning signals. The insight that a "sharp increase" in FIR indicates a break in the monotonic progression of model failure modes, and thus a point beyond which the downgrade model becomes misaligned for guiding exploration, is a specific and useful contribution to reward design in adversarial RL settings.

## Suggestions
- Incorporate statistical measures (e.g., standard deviations over multiple runs, confidence intervals) for the primary success rate metrics in Tables 1, 2, and 3 to substantiate the robustness of the improvements.
- Add a qualitative or quantitative analysis section examining the learned strategies. This could include categorizing successful strategies, providing concrete examples, and analyzing whether they represent novel attack vectors or known patterns, thereby directly validating the "strategy-level exploration" claim.
- Strengthen the comparative analysis by including results against one or two strong, recent baseline methods (e.g., TAP or ICA) to more clearly demonstrate the state-of-the-art positioning of AUTO-RT.

---

## GMP1S4R6Ke

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
LoRA-Mixer introduces a modular mixture-of-experts (MoE) framework that routes task-specific LoRA adapters into the linear projection layers of attention and state-space models, rather than replacing entire blocks or adding parallel branches. It proposes a novel Routing Specialization Loss (RSL) that balances global load balancing with input-aware specialization via entropy regularization. The method demonstrates strong performance across 15 benchmarks using significantly fewer parameters than prior LoRA-MoE approaches and supports flexible usage regimes, including plug-and-play composition of pre-trained LoRAs.

## Strengths
- **Novel architectural contribution**: The decision to apply MoE routing specifically to the core projection layers (Q, K, V, O) of attention/SSM modules is a distinct and well-motivated departure from prior work that focuses on FFN layers or parallel branches. This design enables fine-grained token-level specialization while maintaining drop-in compatibility with both Transformers and SSMs, as evidenced by consistent gains across LLaMA, Mistral, and Falcon-Mamba.
- **Effective and theoretically grounded routing loss**: The proposed RSL loss, which incorporates entropy regularization to trade off load balancing and input-aware specialization, is supported by convergence analysis and generalization bounds (Appendix A.1-A.2). Empirically, RSL enables robust routing with minimal data (e.g., 2k samples) and outperforms alternative routing losses (Table 8).

## Weaknesses
- **Missing critical baseline**: The paper does not compare against a straightforward baseline of training a single LoRA adapter on the combined multi-task data. This omission makes it difficult to attribute performance gains to the MoE routing mechanism rather than simply training on more diverse data.
- **Insufficient ablation for architectural claim**: The core claim that routing at projection layers is superior to routing at FFN layers is not directly validated. A controlled ablation applying the identical MoE mechanism to FFN layers (as in MixLoRA) within the same framework is necessary to isolate the architectural contribution.
- **Lack of statistical significance reporting**: Although experiments were run three times, the paper reports only average performance without standard deviations or confidence intervals. This undermines the reliability of the claimed improvements (often 1–4%), which is particularly important for high-variance tasks like code generation (HumanEval).
- **Incomplete reproducibility details**: The architecture of the router network (e.g., the form of α(x)) is not specified, and key evaluation protocols (prompts for few-shot tasks, exact data splits for routing training, details for HumanEval pass@1) are missing, hindering replication.
- **Ambiguous comparison with optimized loss baselines**: For Table 8, it is unclear whether the baselines (GMoE, DsMoE, AESL) are integrated into the same LoRA-Mixer architecture or evaluated in their native settings. If the latter, the comparison may be confounded by architectural differences rather than solely the loss function.

## Nice-to-Haves
- A quantitative analysis of routing alignment (e.g., measuring the correlation between router top-choice and task labels) would strengthen the claim of “input-aware specialization.”
- Investigating layer-wise variation in routing patterns could provide insights into whether uniform application across all layers is optimal, as noted in the conclusion.
- Scaling experiments with an increasing number of experts (beyond six) would demonstrate the method’s robustness for large-scale modular composition.
- A more detailed analysis of inference computational cost (e.g., FLOPs per token) compared to a single LoRA adapter would clarify the trade-off between parameter efficiency and runtime overhead.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Missing visualizations**: The complaint about missing Figures 3 and 4 is due to parser artifacts in the provided text; the actual paper contains these figures.
- **Incomplete GLUE benchmark coverage**: The paper’s choice of GLUE subsets is justified by consistency with prior work (e.g., LoRA-LEGO, MixLoRA) and is not a major flaw.
- **Hyperparameter sensitivity**: While RSL introduces hyperparameters, the paper includes a grid search ablation (Table 15) and discusses tuning strategies, adequately addressing this concern.
- **Complexity of hard routing**: The description of hard routing for joint training (using domain labels) is clear and represents a specific, valid training regime.

## Novel Insights
The paper’s key insight is that the linear projection layers within attention/SSM modules are a highly effective and previously under-explored location for inserting modular, task-specific adaptations via MoE routing. This allows the model to leverage the inherent attention mechanism for fine-grained token-level specialization without architectural disruption. Furthermore, the theoretical analysis reveals that entropy regularization in RSL provides strong convexity and stability in routing optimization, leading to improved data efficiency and generalization—a principled advance over standard auxiliary losses that tend to over-average.

## Suggestions
- Add a comparison to a single LoRA trained on the combined multi-task data to validate the necessity of the MoE routing mechanism.
- Conduct an ablation study where the MoE routing is applied to FFN layers instead of projection layers, keeping all other factors constant, to directly demonstrate the architectural advantage.
- Report standard deviations or confidence intervals for all experimental results to allow assessment of statistical significance.
- Clearly specify the router architecture (e.g., a linear layer or small MLP) and provide full evaluation details (prompts, few-shot settings, exact data splits for routing training) in the appendix.
- Clarify the experimental setup for Table 8: indicate whether the baseline losses were implemented within the LoRA-Mixer framework or taken from their original papers.

---

## XX5EZoe4ec

- GT: Reject (avg 2.0)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
RetrievalFormer is a dual-encoder transformer architecture for sequential recommendation that addresses inference scalability and item cold-start. It uses a transformer-based user tower to encode interaction sequences and a feature-based item tower with attention fusion, enabling efficient Approximate Nearest Neighbor (ANN) retrieval and zero-shot recommendation of unseen items. The paper demonstrates competitive accuracy on public benchmarks, massive latency reductions via ANN, and introduces a rigorous Leave-One-Out Cold (LOOC) evaluation protocol.

## Strengths
- **Solves two critical practical problems**: The architecture directly addresses the inference bottleneck of transformer softmax over large catalogs and the cold-start problem by enabling ANN retrieval and feature-based generalization to unseen items. Evidence: Motivation in Introduction, efficiency gains in RQ4, and cold-start evaluation in RQ3.
- **Substantial efficiency gains with sub-linear scaling**: Using ANN (IVF-PQ), RetrievalFormer achieves up to 288× lower latency at 10M items compared to exhaustive scoring, with latency growing sub-linearly. Evidence: Figure 2 and analysis in Section 4.5.
- **Rigorous cold-start evaluation protocol**: The proposed LOOC protocol ensures zero item leakage between training and evaluation, providing a realistic assessment of cold-start capability. Evidence: Section 4.4 and Table 2, which honestly reports performance drops.
- **Comprehensive ablation studies**: Ablations validate key design choices, showing that attention fusion, shared embeddings, and uniformity loss contribute to performance. Evidence: Table 3 in Appendix E and discussion in RQ2.

## Weaknesses
- **No analysis of ANN retrieval approximation error**: The paper does not report metrics like the recall of the ANN index (e.g., percentage of true top-K items retrieved compared to exhaustive search). This omission undermines confidence that the efficiency gains do not come at the cost of missing relevant items during retrieval. Evidence: End-to-end metrics are reported with ANN, but no retrieval-quality analysis is provided.
- **Insufficient comparison to retrieval-oriented baselines**: Baselines are limited to ID-softmax transformers (e.g., SASRec, BERT4Rec, AttrFormer). There is no comparison to other dual-encoder or two-tower sequential models, making it difficult to isolate the contribution of the proposed architecture versus the dual-encoder paradigm itself. Evidence: Section 4.2 compares only to transformer baselines, not to retrieval-focused models.
- **Significant cold-start performance drop**: Under the LOOC protocol, Recall@20 drops by 25–35% compared to standard evaluation, indicating limited effectiveness for completely unseen items despite the feature-based design. This highlights a key limitation for real-world deployment. Evidence: Table 2 shows drops from 0.1208 to 0.0804 on Amazon Beauty.
- **Accuracy trade-off relative to strongest baselines**: While competitive with some transformers (e.g., 96.8% of SASRec on MovieLens-1M), RetrievalFormer falls short of AttrFormer’s reported Recall@20 (0.337 vs. 0.4128). The paper’s claim of "competitive accuracy" is nuanced, as it compares to an "established baseline cluster" rather than the state-of-the-art, which may overstate performance. Evidence: Discussion in Section 4.2 and Table 1.

## Nice-to-Haves
- Ablation study on the necessity of the transformer user tower (e.g., versus a simpler MLP encoder) to validate the role of sequential modeling.
- Sensitivity analysis of ANN index parameters (e.g., `nprobe`, PQ dimensions) on retrieval recall and latency.
- Deeper analysis of which feature types or richness correlate with cold-start performance under LOOC.
- Visualizations (e.g., t-SNE of embeddings or attention heatmaps) to interpret the learned representations.

## Removed Points
These points are flagged to be removed, treat them with caution.
- **Criticism about Mixed Negative Sampling (MNS) details being too brief in the main text**: The paper adequately covers MNS in Section 3.5 and Appendix C, which is standard for methodological details.
- **Criticism about feature fusion clarity in the main text**: The core mechanism is described in Section 3.2, with variable-length handling detailed in appendices, which is acceptable.
- **Criticism about broader impact not discussed**: This is not a standard requirement for a technical paper of this nature.
- **Suggestion to compare end-to-end latency to optimized transformer serving pipelines (e.g., with sampled softmax)**: The paper’s comparison to exhaustive scoring is standard for demonstrating ANN benefits; optimized serving is beyond the scope.

## Novel Insights
The paper’s novel insight is the integration of transformer-based sequential modeling with a dual-encoder retrieval framework, enabling accurate sequence understanding while achieving scalable serving via ANN. The attention fusion mechanism for heterogeneous features and shared embeddings across towers enhance representation alignment and cold-start generalization. The LOOC protocol provides a rigorous evaluation framework for cold-start scenarios, moving beyond standard splits that leak item information.

## Suggestions
- Include analysis of ANN retrieval recall (e.g., recall@K of the ANN index versus exhaustive top-K) to validate that efficiency gains do not compromise retrieval quality.
- Add comparison to a simple two-tower baseline (e.g., with the same features but a GRU user encoder) to isolate the contribution of the transformer user tower and attention fusion.
- Provide deeper analysis of the cold-start performance drop, such as examining how feature coverage or types affect LOOC results, to guide improvements.

---

## 0cbUKCyBsH

- GT: Reject (avg 3.5)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper identifies the "self-stimulation" assumption—predicting time series using only historical values—as a fundamental barrier to forecasting progress. Through a control-theoretic analysis, it formally proves this imposes an irreducible error bound and introduces Influence-Aware Time Series Forecasting (IATSF), a paradigm that incorporates external influences. IATSF is operationalized with a new leak-free benchmark and FIATS, a lightweight model featuring channel-aware mechanisms, which demonstrates substantial improvements across synthetic, physical, and market datasets.

## Strengths
- **Foundational theoretical contribution:** Propositions 2.1 and 3.1 rigorously derive an error bound for self-stimulated forecasting and prove that incorporating influences reduces it, providing a principled explanation for the field's performance plateau.
- **Valuable community resource:** The IATSF benchmark is carefully designed with leak-free, temporally-synced textual influences across diverse datasets, addressing critical gaps in existing multimodal time series resources.
- **Principled and effective model design:** FIATS embodies the theoretical insights through novel channel-aware mechanisms (CASM and CAPS), enabling interpretable influence modeling without relying on large language models, and achieves state-of-the-art performance against heavy baselines.

## Weaknesses
- **Idealized theoretical assumptions:** The analysis assumes influences are independent of historical states and full observability, which often do not hold in practice. The paper does not fully discuss how violations affect the error bounds or practical applicability, limiting the theory's generality.
- **Insufficient statistical validation:** While mean performance is reported, measures of variance, confidence intervals, or statistical significance tests are missing for most comparisons (except limited data in Appendix M). This undermines confidence in the claimed superiority of FIATS over baselines.
- **Lack of causal evidence for text-time series links:** The core premise that textual influences causally drive the time series is assumed but not empirically validated (e.g., via Granger causality tests). Improvements could stem from spurious correlations rather than genuine influence modeling, challenging the paradigm's foundation.
- **Architectural contribution not isolated:** The paper lacks an ablation against a simple baseline that concatenates text embeddings with time series input (e.g., into PatchTST). Without this, it is unclear how much gain comes from the novel CASM/CAPS mechanisms versus merely having access to textual data.

## Nice-to-Haves
- More extensive robustness studies under realistic influence conditions (e.g., missing influences, imperfect forecasts, correlated influences).
- Deeper error decomposition analysis to pinpoint when influence-aware modeling helps most (e.g., for trends vs. periodic components).
- Exploration of extending FIATS to non-textual influence modalities to demonstrate broader generality of the IATSF framework.
- Providing parameter counts and FLOPs for FIATS relative to baselines to substantiate the "lightweight" claim.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- Criticism about dense architectural descriptions (writing style nitpick).
- Concern over LLM usage for dataset preprocessing, as the paper clarifies this is only for data augmentation and provides raw samples (Appendix O.4.4).
- Demand for comparison on the flawed Time-MMD dataset as a primary benchmark, since the paper critiques its issues in Appendix N and focuses on its own rigorously constructed data.
- Request for "obvious next steps" like uncertainty modeling, which are already noted in the limitations and future work.

## Novel Insights
The paper's most novel insight is the control-theoretic formalization of the "self-stimulation" barrier, proving that traditional forecasting models converge to predicting conditional expectations with an error floor determined by the system's sensitivity to unobserved influences. This theoretical foundation not only explains the persistent performance plateau but also directly motivates the design of influence-aware models. The channel-aware mechanisms in FIATS operationalize this insight by learning sensitivity to influences per channel, offering a new perspective that shifts the field's focus from architectural complexity to the inclusion of external context.

## Suggestions
- Conduct statistical significance tests or report confidence intervals for key experimental results to strengthen empirical claims.
- Perform causality tests (e.g., Granger causality) on the benchmark datasets to validate the influence relationship between text and time series.
- Add an ablation experiment comparing FIATS to a baseline that simply concatenates text embeddings with time series patches to isolate the contribution of CASM/CAPS.
- Include a dedicated discussion in the main text on the limitations of the independence and full observability assumptions, and their practical implications.

---

## WwDNiisZQm

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (6.6/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Content-Aware Mamba (CAM), a novel state-space model designed to overcome two key limitations of standard Mamba in learned image compression: rigid, content-agnostic scanning and strict causality. CAM employs Content-Adaptive Token Permutation to group similar tokens and Global-Prior Prompting to inject global context, enabling more effective redundancy removal while preserving linear complexity. The resulting model, CMIC, achieves state-of-the-art rate-distortion performance across multiple datasets.

## Strengths
- **Novel and well-motivated architectural contributions.** The paper clearly identifies underexplored problems with applying Mamba to image compression and proposes two cohesive, task-specific solutions: a codebook-based clustering mechanism for token permutation and a clustering-derived global prompting strategy to relax causality. The design is directly tailored to the core need of capturing long-range, content-dependent redundancy.
- **Extensive and compelling empirical validation.** CMIC demonstrates significant BD-rate improvements (e.g., -21.34% vs. VTM-21.0 on Tecnick) and consistently outperforms a wide range of recent CNN-, Transformer-, and Mamba-based learned methods across Kodak, Tecnick, and CLIC datasets for both PSNR and MS-SSIM. The gains are substantial and well-documented.
- **Excellent ablation studies and analysis.** The paper provides thorough ablations showing the additive benefits of each component. The Effective Receptive Field (ERF) visualizations are particularly compelling, demonstrating quantitatively and visually that CAM achieves larger, more content-adaptive, and less causal receptive fields compared to prior methods. Analyses of cluster activation and throughput confirm the efficiency and adaptivity of the approach.

## Weaknesses
- **Limited analysis of computational overhead from clustering.** While the paper states clustering adds only ~5% training time overhead and minimal inference latency, a more detailed profiling breakdown (e.g., time spent on distance computation vs. sorting) is missing. For very high-resolution images, the O(NK) assignment step could become a bottleneck; a brief discussion of this scaling would strengthen the efficiency claims.
- **Insufficient integration of comparison with prior adaptive methods.** A detailed comparison with another clustering-based LIC method (Zhang et al., 2024b) is relegated to the appendix. The core methodological distinctions (fine-grained, permutation-equivariant clustering vs. grid-anchored pooling) are important for novelty and should be highlighted in the main text or experiments.
- **Missing standard "Limitations" section.** For ICLR, a discussion of limitations is expected. The paper should address potential failure modes (e.g., images with extremely fine-grained, non-stationary textures that may not cluster cleanly), the sensitivity of performance to the cluster count K beyond the provided ablation, and the inherent trade-off of potentially disrupting local spatial coherence when reordering tokens.

## Nice-to-Haves
- A direct comparison with a Mamba variant employing multi-directional scans (e.g., Vision Mamba) in a controlled, similar-capacity setting would further solidify the claim that CAM's content-adaptive approach is superior to simply adding more fixed scans.
- Quantitative metrics for clustering quality (e.g., alignment with semantic segments, cluster consistency across training) would complement the strong visualizations and provide a more objective link between clustering efficacy and performance gains.
- A case study showing images where CMIC performs relatively poorly or where clustering results are suboptimal would provide a more balanced view and help identify boundaries of the method's effectiveness.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **"Section 3.2 is completely empty" (Harsh Critic).** The paper's structure is clear: Section 3.1 is Preliminaries, and Section 3.3 begins the method description. Figure 2 provides the architectural overview. There is no missing section that breaks the narrative.
- **"Lack of theoretical grounding" (Neutral Reviewer).** This is an empirical systems paper presenting novel architectures and achieving SOTA results. Demanding theoretical guarantees for convergence or redundancy removal imposes an arbitrary rigor requirement not standard in this subfield.
- **"Superficial comparison...relegated to the appendix" (Neutral Reviewer) – Weakened and moved.** This is a valid point about presentation, but the comparison is detailed in Appendix A.2. The weakness is rephrased to focus on the need for better integration into the main narrative.
- **"Unfair comparison with MambaIC due to model scale" (Harsh Critic).** The comparison is fair as it evaluates end-to-end models. CMIC's superior performance with significantly lower parameters, FLOPs, and memory is a core strength, not a flaw. The paper also compares favorably against many other models of similar or larger scale.
- **Requests for out-of-domain dataset evaluation, application to other tasks, or differentiable clustering (Spark Finder).** These are interesting research directions but are outside the stated scope of the paper, which focuses on advancing learned image compression.

## Novel Insights
The paper provides a novel synthesis of ideas from vector quantization and state-space modeling specifically for compression. The key insight is that Mamba's efficiency for long sequences can be harnessed more effectively for images by dynamically reordering the sequence based on content similarity and conditioning the SSM on a global prior derived from that same clustering. This breaks the fundamental mismatch between Mamba's 1D causal design and the 2D, non-causal redundancy structure of images. The ERF visualizations offer a novel, intuitive demonstration that the model's receptive field becomes both global and semantically aligned, which is a direct result of the proposed mechanisms.

## Suggestions
- Integrate the core methodological comparison with Zhang et al. (2024b) from Appendix A.2 into the main related work or experiment sections to better contextualize the novelty of the fine-grained, non-Euclidean clustering.
- Add a "Limitations" subsection before the conclusion to discuss the points raised in the weaknesses section (computational scaling, potential failure modes).
- In the computational overhead analysis, consider adding a brief note or experiment profiling the clustering time separately on a very high-resolution image (e.g., 4K) to preemptively address scaling concerns.

---

## USyGD0eUod

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper demonstrates that common automated evaluation metrics for sparse autoencoders (SAEs), particularly aggregate auto-interpretability scores, fail to reliably distinguish between SAEs trained on standard transformer language models and those trained on transformers with randomly initialized weights. Through extensive experiments on the Pythia model suite (70M to 6.9B parameters) and multiple randomization schemes, the authors show these metrics can produce similar values in both settings, indicating they are insufficient proxies for identifying learned, computationally relevant features. The paper advocates for routine randomized baselines in mechanistic interpretability evaluation and suggests token distribution entropy as a preliminary metric for feature "abstractness."

## Strengths
- **Critical Sanity Check with Strong Empirical Support**: The paper performs a necessary and often-overlooked validation by applying a strong null model (randomly initialized networks) to a popular interpretability method. The core finding—that aggregate auto-interpretability scores are similar—is robustly supported across model scales (70M to 6.9B), multiple randomization variants (Step-0, re-randomized with/without embeddings, Gaussian control), and is shown to be stable across SAE hyperparameters and training data sizes (Appendices C, F, G).
- **Identifies a Meaningful Distinction with a Simple Metric**: While aggregate scores fail, the paper successfully identifies a qualitative difference: SAEs from trained models learn more "abstract" features, especially in later layers. It proposes and demonstrates token distribution entropy as a simple, quantitative proof-of-concept metric that captures this distinction, showing trained-model features have higher entropy (activations spread across many tokens) compared to random-model features (Section 3, Figure 20, Appendix H).

## Weaknesses
- **Lack of Statistical Confidence for Large-Scale Results**: The paper shows uncertainty estimates (5 random seeds) only for Pythia-70m (Appendix E). For the larger, more expensive models where the main effect is most pronounced (e.g., Pythia-6.9B), the number of independent seeds is not stated, leaving the statistical robustness of these key results unclear. While trends appear consistent, formal statistical comparisons (e.g., of score distributions between Trained and Randomized variants) would strengthen the claim.
- **Underdeveloped Mechanistic Explanation**: Section 4 presents toy models to hypothesize why random networks might preserve or amplify superposition, but this analysis is preliminary and inconclusive. It does not rigorously establish whether the observed phenomenon in transformers is due to data structure, architectural bias, or an interaction. The connection between the toy models and the main results remains speculative, leaving the core mechanism underexplored.
- **Reliance on a Single Auto-Interpretability Pipeline**: The evaluation depends entirely on one automated pipeline (using Llama-3.1-70B for explanation generation and scoring). While this is standard practice, the potential for this LLM to generate superficially plausible explanations for any pattern—a bias that could inflate scores for random-model features—is not discussed. This is a minor but relevant methodological limitation that should be acknowledged.

## Nice-to-Haves
- A more detailed analysis comparing the *distributions* of per-latent scores (not just aggregates) could reveal if, for example, the highest-scoring features from trained models are qualitatively better than those from random models.
- Extending the core experiment to at least one other popular model family (e.g., GPT-2 or Gemma) would help assess the generality of the finding beyond the Pythia suite.
- Exploring a more direct test of "computational relevance," such as comparing the efficacy of features from trained vs. random SAEs for a downstream task like model steering, could provide stronger evidence for the paper's central argument about feature quality.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Missing human evaluation of feature quality."** The paper's claim is about *automated metrics* failing; it does not claim human evaluation would also fail. Proving the metrics are misleading does not require proving the features are meaningless to humans—showing the metrics give similar scores for clearly different underlying models is sufficient.
- **Weakness: "Needs testing on other SAE variants (e.g., Gated SAEs)."** The paper explicitly uses a standard TopK SAE architecture and shows robustness to hyperparameters. Demanding validation across all SAE variants is scope creep; the paper’s contribution is to demonstrate a problem with *common metrics*, not to audit every SAE method.
- **Weakness: "Requires ablation with permuted embeddings or synthetic data."** The paper already includes a strong control variant with Gaussian i.i.d. embeddings and a thorough toy model section. These suggested experiments are more appropriate for a follow-up mechanistic study.
- **Weakness: "The 'So What?' test: compare best features."** The paper's central point is that *aggregate* metrics used for evaluation are flawed. Analyzing top features is a different, more exploratory question.
- **Weakness: "Needs correlation analysis between metrics."** This is an interesting analysis but not required to substantiate the paper's primary claim.

## Novel Insights
The paper provides a crucial, field-relevant insight: the standard practice of reporting high aggregate auto-interpretability scores as evidence of successful feature discovery is fundamentally insufficient. It shows that these scores can be high even when the underlying model has learned nothing from data, due to structure inherent in the architecture or input. This forces a re-evaluation of what constitutes validation for interpretability methods. Furthermore, it demonstrates that this failure becomes more pronounced with model scale, a critical detail for a field increasingly focused on large models. The proposed token entropy metric, while simple, successfully isolates a dimension (feature abstractness) that the standard aggregate metrics miss, providing a concrete starting point for better evaluation.

## Suggestions
- State the number of random seeds used for each model size in the main experiments, particularly for the large models (1B, 6.9B). If only one seed was used for cost reasons, acknowledge this as a limitation in the main text.
- In the Limitations section (Section 5), briefly acknowledge the potential for bias in the LLM-based auto-interpretability pipeline itself (e.g., its capability to generate plausible-sounding text) as a factor that could contribute to inflated scores for random-model features.
- Strengthen the connection between the toy model section and the main results. Even a brief discussion speculating on which mechanism (preservation vs. amplification of sparsity) seems more consistent with the observed entropy trends in transformers would make Section 4 feel less detached.

---

## ZNAY3ivd62

- GT: Reject (avg 4.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
GUI-Spotlight introduces a sample-efficient method for GUI visual grounding by training a multimodal LLM to iteratively invoke specialized tools (crop, extract, find color) to narrow focus on target screen regions. Key contributions include a modified reinforcement learning objective that stabilizes multi-tool coordination and comprehensive empirical insights into algorithm and reward design. The model achieves state-of-the-art performance for 7B models on benchmarks like ScreenSpot-Pro (52.8% accuracy) using only 18.5K training samples.

## Strengths
- **High data efficiency and strong performance**: GUI-Spotlight outperforms comparable 7B models trained on orders of magnitude more data (e.g., V2P-7B with 9.6M samples) on challenging benchmarks like ScreenSpot-Pro, UI-Vision, and OSWorld-G, demonstrating effective sample utilization.
- **Stabilized RL with valuable empirical insights**: The introduction of an auxiliary cross-entropy loss within a modified GSPO objective prevents training collapse in multi-tool scenarios, as empirically validated in Section 4.1. The paper transparently documents algorithm selections and reward design ablations, including negative results, providing practical guidance for the community.

## Weaknesses
- **Unabated tool design justification** — The choice of the three specific tools (extract, crop, find color) is presented without ablation studies to justify their necessity or optimality. This leaves open whether the toolset is efficient or if alternative tools could improve performance, affecting the method's design credibility.
- **Missing computational cost analysis** — The iterative inference process requires multiple LLM forward passes and tool executions per query, but the paper omits analysis of inference latency, token usage, or trade-offs between accuracy and computational cost. This gap hinders assessment of practical deployment feasibility.
- **Insufficient failure mode analysis** — The paper lacks a breakdown of error types (e.g., tool selection errors vs. coordinate regression errors) or qualitative examples of failures. Without this, the limitations and robustness of the method are unclear, limiting understanding of where and why it fails.
- **Incomplete ablation to isolate iterative tool use** — While the paper compares against training-free iterative baselines (Section 5.4), it does not compare to a strong baseline trained with the same RL procedure but without tool invocation (e.g., direct coordinate prediction). This makes it difficult to disentangle the contribution of iterative tool coordination from improved RL training alone.

## Nice-to-Haves
- Sensitivity analysis of reward function weights to demonstrate robustness to hyperparameter choices.
- Visualization of learned policy trajectories or attention maps to interpret the reasoning process behind tool selection.
- Learning curve showing performance versus training data scale to further support the data efficiency claim.

## Removed Points
These points are flagged to be removed, treat them with caution.
- Criticisms about presentation artifacts in equations and tables (e.g., "find ~~c~~ olor"), as these are likely due to PDF parsing issues and not inherent to the paper's clarity.
- Demand for benchmarking against other iterative refinement methods not included in the standard benchmarks (e.g., UniVGR), as the paper already evaluates on established GUI grounding benchmarks and such comparisons may constitute scope creep.

## Novel Insights
The paper offers novel insights into reinforcement learning for GUI visual grounding, demonstrating that a simple auxiliary cross-entropy loss on format-correct samples can prevent training collapse in multi-tool scenarios, and that sparse final rewards yield better accuracy than dense, center-shaped rewards in this iterative setting. These findings, derived from systematic experimentation, provide actionable guidance for stabilizing RL in agentic visual reasoning tasks.

## Suggestions
- Conduct an ablation study to evaluate the contribution of each tool in the suite, e.g., by training variants with subsets of tools.
- Include metrics on average inference steps per query and discuss the accuracy-computational cost trade-off to address practical concerns.
- Perform a qualitative analysis of failure cases, categorizing error types and providing examples to inform future improvements.

---

## bH5M0ts8Y6

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes VINCIE, a framework for learning in-context image editing (editing conditioned on a sequence of past images and text) directly from unlabeled video data. The core idea is to treat sparsely sampled video frames as a natural sequence of "edits," automatically annotate visual transitions using vision-language models, and train a diffusion transformer with three proxy tasks: next-image prediction, current segmentation prediction, and next-segmentation prediction. The authors also introduce a new benchmark, MSE-Bench, for evaluating multi-turn editing. Experiments demonstrate the model's strong performance, scalability with data, and emergent capabilities in composition and story generation.

## Strengths
- **Novel and well-motivated approach to a fundamental bottleneck.** The paper convincingly argues that video provides a scalable, naturally coherent source of sequential visual dynamics, bypassing the need for costly curation of paired image-editing data. The core research question—can in-context editing be learned solely from videos?—is compelling and significant.
- **Strong empirical validation of the core thesis.** The scalability curve (Fig. 5) shows clear, near log-linear improvement in multi-turn success rates with more training data, directly supporting the promise of video-driven scaling. Ablation studies (Tabs. 3, 4, 5) effectively demonstrate the benefits of segmentation tasks, context, and video sequence data over pairwise data.
- **Competitive and state-of-the-art performance.** The model achieves strong results on the established MagicBrush benchmark and outperforms prior academic methods on the challenging multi-turn MSE-Bench, demonstrating the practical viability of the approach.

## Weaknesses
- **Reproducibility is hampered by reliance on unspecified proprietary components.** The method depends critically on an "in-house" vision-language model for annotation and an "in-house MM-DiT" video foundation model for initialization. Descriptions of these components are insufficient for replication, and their specific architectures, training data, and release plans are not detailed. This is a significant barrier for the community.
- **Reliability of the proposed MSE-Bench evaluation is not fully established.** The benchmark uses GPT-4o for both prompt generation ("imagination") and scoring. While a correlation with human judgment is shown (Appendix D.2), this does not fully validate that GPT-4o is a reliable judge for this specific, complex task. The potential for evaluation circularity and bias remains a concern.
- **Insufficient analysis of limitations inherent to the video data prior.** The paper acknowledges but does not deeply analyze how learning from natural video dynamics biases the model. A systematic breakdown of performance by edit type (e.g., common object motion vs. rare scene/style changes) or a quantitative analysis of failure modes is missing. This leaves the boundaries of the "video prior" unclear.

## Nice-to-Haves
- A quantitative evaluation of the claimed emergent capabilities (e.g., multi-concept composition, story generation) on established relevant benchmarks would strengthen these claims beyond qualitative showcases.
- A more detailed analysis of the model's "chain-of-thought" during chain-of-editing (e.g., visualizing predicted segmentation masks) could provide mechanistic insight into how the planning tasks aid control.
- A clearer explanation of the block-wise causal attention variant and its trade-offs compared to full attention would aid architectural understanding.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "The description of 'block-wise causal attention' is very brief."** The appendix (Fig. 11, Sec. C.4) provides a clear diagram and explanation.
- **Weakness: "Equation 2 is confusing..."** The text following the equation and Fig. 12 clarify that dropout is applied to contextual elements, not the target.
- **Weakness: "The statement 'Position collisions are avoided...' is not fully convincing."** This is a technical implementation detail adequately resolved by the use of separate modality weights and bias terms, as stated.
- **Nitpick: "The hybrid frame sampling strategy... rationale... is not provided."** The strategy is standard for capturing varied dynamics; a detailed ablation is not required.
- **Nitpick: "The context dropout rates... are given without ablation or justification."** These are standard hyperparameters; a full ablation is not expected.
- **Suggestion: "A controlled experiment isolating video-only training (no SFT)."** The paper's core claim is the feasibility of learning from video, not that video-only training surpasses all methods. The primary results (Tabs. 1, 2) already show strong performance before SFT, and the scalability study (Fig. 5) uses the video-only model. Demanding SOTA performance without any task-specific tuning is outside the paper's scope.

## Novel Insights
The paper's central insight is that the temporal coherence and inherent visual transitions in videos—such as objects entering/exiting, poses changing, or camera movements—provide a powerful and scalable supervisory signal for learning the operations fundamental to in-context image editing (addition, removal, modification). By framing video frames as an interleaved multimodal sequence and training with proxy segmentation tasks, the model learns to disentangle and control these dynamics, unlocking capabilities like grounded editing and multi-turn consistency without ever seeing curated edit pairs. This demonstrates a promising alternative data paradigm that leverages the web's vast video corpus.

## Suggestions
- Provide significantly more detail on the "in-house" components (the VLM and MM-DiT foundation model) in an appendix or via open-source release to ensure reproducibility. At minimum, specify model architectures, training data sources, and capabilities.
- To bolster confidence in MSE-Bench, release the full benchmark with the human judgments used for the correlation study (Appendix D.2) and consider incorporating human evaluation as a primary metric for future benchmarking.
- Include a dedicated analysis section or figure that categorizes and visually showcases common failure cases, particularly those linked to the video data prior (e.g., edits requiring non-physical transformations or drastic scene changes not reflected in natural dynamics).

---

## kMfVTka2WB

- GT: Reject (avg 2.0)
- Predicted: N/A (2.3/10)
- Match: N/A

### Final Review

## Summary
This paper argues that the principles of Support Vector Machines (max-margin, KKT conditions) are inherently Euclidean and thus not directly applicable in the original "non-Euclidean" statistical space of the data, where distance is governed by covariance. It proposes a Covariance-Adjusted SVM (CSVM) that uses class-specific Cholesky decomposition to whiten data, solves the SVM in the transformed Euclidean space, and derives an iterative algorithm (SM) to estimate population covariance from samples. Empirical results on five binary datasets show CSVM often outperforms standard SVM kernels and global whitening methods.

## Strengths
- **Clear, intuitive motivation rooted in geometry:** The paper effectively connects the Mahalanobis distance, data whitening, and vector space concepts to argue for class-specific preprocessing, providing a clear and accessible rationale for its approach.
- **Practical iterative algorithm for a real problem:** The proposed SM Algorithm is a concrete and novel heuristic to address the practical challenge of applying class-specific whitening without test labels, framing it as an iterative label estimation and covariance update problem.
- **Thorough empirical comparison on diverse data:** The method is validated on five datasets from different domains, showing consistent improvements over several standard SVM kernels and two common whitening techniques (PCA, ZCA) across multiple metrics (accuracy, F1, AUC).

## Weaknesses
- **Overstated and unsubstantiated theoretical claims:** The core lemmas (2.1, 2.3) make strong, sweeping claims (e.g., KKT conditions are "invalid" in non-Euclidean spaces) that are not rigorously proven. The derivation shows the margin formula changes under a transformation, but this does not invalidate the optimization framework; it merely redefines the geometry. This overclaim undermines the paper's theoretical contribution.
- **Missing comparison with the most relevant prior work:** The paper dismisses prior covariance-incorporating SVMs (e.g., Minimum Class Variance SVM, Mahalanobis-distance-based SVMs) for alleged "gaps" and "dimensional inconsistencies" without a detailed explanation or a direct empirical or mathematical comparison. This omission makes it impossible to assess whether CSVM represents a substantive advance over existing techniques.
- **Confusing and inconsistent theoretical narrative:** Lemma 2.2 claims an N-class problem yields N distinct classifiers in the input space, which is non-standard and poorly explained. This claim is not reconciled with the final proposed algorithm, which outputs a single classifier after iterative adjustment, creating internal inconsistency and confusion for the reader.
- **Incomplete algorithmic analysis and ablation:** The SM Algorithm is presented heuristically without analysis of its convergence, sensitivity to initialization, or computational complexity. Furthermore, no ablation study isolates the contribution of the *iterative* algorithm from the simpler (and likely major) benefit of *class-wise whitening*, leaving the source of performance gains ambiguous.

## Nice-to-Haves
- An ablation study comparing: (a) global whitening + SVM, (b) class-wise whitening + SVM (non-iterative), and (c) the full iterative SM Algorithm.
- A synthetic 2D experiment to visually demonstrate and validate the core geometric claim that the margin splits according to class covariance.
- Reporting statistical significance tests (e.g., over multiple data splits) to bolster the empirical claims, given that some performance differences in the tables are small.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength:** "The paper is well-written" (Too generic).
- **Weakness:** "The derivation of the margin in Eq. 9 is inconsistent because it uses the Euclidean formula in a non-Euclidean space." (The paper's derivation is consistent: it computes the margin in the Euclidean space and then uses the transformation to express it in the input space. The critic misunderstands the pullback operation.)
- **Weakness:** "The algorithm's step (e) is an arbitrary post-processing step." (While the adjustment heuristic is not derived from a new objective function, it is a direct consequence of the derived margin ratio (Eq. 14), making it a reasoned design choice, not purely arbitrary.)
- **Weakness:** "Missing details like dataset sizes and hyperparameter tuning harm reproducibility." (While more details are always helpful, the absence of these specifics is a common minor shortcoming, not a core flaw that invalidates the results. It is moved to a suggestion.)
- **Weakness:** "The paper does not discuss broader impact." (This is not a standard expectation for a methodological paper of this type.)

## Novel Insights
The paper provides a coherent vector-space interpretation for why data whitening (e.g., via PCA/ZCA) improves model performance: it frames whitening as a transformation from a non-Euclidean statistical space (where distance is Mahalanobis) to a Euclidean space, where the geometric foundations of algorithms like SVM are naturally valid. This perspective cleanly unifies preprocessing and model geometry. Furthermore, the iterative SM Algorithm presents a novel, practical strategy for performing class-conditional whitening in the absence of test labels, a common real-world constraint.

## Suggestions
- Reframe the theoretical claims to be more precise and modest. Focus on deriving how the optimal classifier under a Mahalanobis geometry leads to the proposed optimization adjustments, rather than claiming the entire SVM framework is "invalid."
- Add a direct experimental comparison with at least one key prior method (e.g., Minimum Class Variance SVM) to substantiate the claim of addressing gaps in prior work.
- Clarify the narrative around Lemma 2.2. Either provide a clear explanation of how multiple classifiers are reconciled into a final decision rule or reformulate the lemma to avoid confusion with the single-classifier algorithm.
- Formalize the SM Algorithm with pseudo-code, specify the convergence criterion, and include a basic analysis (even empirical) of its convergence behavior and sensitivity.

---

## d2pUyiXwcm

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (6.6/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Simulation-Calibrated Scientific Machine Learning (SCaSML), a framework that systematically improves pre-trained surrogate models (e.g., PINNs, Gaussian Processes) for high-dimensional semi-linear parabolic PDEs at inference time without retraining. The core innovation is the "Structural-preserving Law of Defect," a new PDE that exactly describes the surrogate's error and retains the original problem's structure, enabling efficient correction via stochastic simulation (Multilevel Picard methods). Theoretically, the final error is bounded by the product of the surrogate error and simulation error, yielding an accelerated convergence rate. Empirically, SCaSML reduces errors by 20–80% across PDEs up to 160 dimensions with high statistical significance.

## Strengths
- **Novel inference-time scaling paradigm for SciML:** The paper successfully adapts the inference-time compute idea from large language models to scientific machine learning, proposing a principled hybrid that combines the speed of surrogates with the rigor of Monte Carlo simulation. This enables "elastic compute," where additional inference-time resources target refinement of a fixed surrogate.
- **Theoretical foundation with multiplicative error bound:** The derivation of the Structural-preserving Law of Defect is exact and preserves the semi-linear structure, allowing the use of efficient stochastic solvers. Theorem 2.5 proves the final error is bounded by the product of the surrogate error and simulation error, leading to a provably faster convergence rate (Corollary 2.6).
- **Comprehensive and rigorous empirical validation:** Experiments cover four challenging high-dimensional PDE families (linear convection-diffusion, viscous Burgers, Hamilton–Jacobi–Bellman, diffusion-reaction) up to 160 dimensions, using two distinct surrogate types (PINNs and Gaussian Processes). SCaSML consistently reduces errors across all norms (\(L^2, L^\infty, L^1\)) with high statistical significance (\(p \ll 0.001\)). The appendix includes detailed statistical tests, fixed-budget efficiency comparisons, and empirical verification of the improved scaling law.

## Weaknesses
- **Strong regularity assumptions for theoretical guarantees:** The core theorems (e.g., Theorem 2.5) rely on Assumption 2.4, which requires the surrogate error to be bounded in \(L^\infty\) and \(W^{1,\infty}\) norms. While these are standard in PDE analysis to obtain explicit rates, neural network surrogates do not inherently guarantee such smoothness, and the theory does not address how violations might affect performance.
- **Scope limited to semi-linear parabolic PDEs:** The method is developed and validated exclusively for semi-linear parabolic equations. Its applicability to other important PDE classes (e.g., hyperbolic, elliptic, or problems with discontinuous solutions) remains an open question and is not discussed, which may limit immediate broader impact.
- **Non-negligible inference-time overhead per query:** While SCaSML improves accuracy, the Multilevel Picard correction step adds substantial computational cost per evaluation point. For applications requiring a full-field solution at many points, this overhead could become prohibitive, and the paper does not thoroughly analyze the trade-off between query count and total computational budget.

## Nice-to-Haves
- A more detailed complexity analysis comparing wall-clock time versus accuracy for SCaSML against alternatives (e.g., training a larger surrogate or pure simulation) across multiple query points.
- Testing the framework on a broader class of PDEs (e.g., non-parabolic or with non-smooth solutions) to probe its generality and limitations.
- Investigating whether iterative application of the defect correction (using the corrected solution as a new surrogate) yields further gains or stability issues.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Formatting/style nitpicks:** Parsing artifacts in the provided text (e.g., garbled table entries) are not paper problems.
- **Insufficient related work discussion:** The paper adequately contrasts its approach with classical defect correction and iterative methods in Section 2.2 and the introduction.
- **Demand for control variate baseline comparison:** The paper explicitly notes the connection to control variates (Conclusion), and a direct empirical comparison is not required to establish the core contribution.
- **Requirement for variance reduction quantification:** While insightful, quantitative variance analysis is not essential to validate the main claims, given the comprehensive error metrics provided.
- **Failure mode analysis:** Exploring failure cases is a valuable research direction but not a mandatory component for this paper.

## Novel Insights
The key novel insight is the exact formulation of the defect correction that preserves the semi-linear structure of the original PDE, enabling the use of efficient stochastic solvers (like Multilevel Picard) for the error itself. This transforms the correction step into a problem that is both well-posed and computationally tractable in high dimensions. Furthermore, the paper introduces the inference-time scaling paradigm to scientific machine learning, showing that allocating additional compute to targeted simulation-based refinement can yield better returns than simply training a larger surrogate, as evidenced by the multiplicative error bound and fixed-budget experiments.

## Suggestions
- Add a brief discussion of limitations in the main text, explicitly noting the smoothness assumptions, per-query computational cost, and current restriction to semi-linear parabolic PDEs.
- Include an ablation study that systematically varies surrogate accuracy (e.g., by training duration or network size) and plots the resulting SCaSML error to empirically verify the multiplicative error relationship claimed in Theorem 2.5.
- Provide practical guidance on choosing MLP parameters (number of levels, samples) and discuss their impact on performance and runtime, perhaps via a sensitivity analysis in the appendix.

---

## GiaF5cFIpI

- GT: Reject (avg 3.5)
- Predicted: N/A (4.4/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a real-time framework for adaptive stimulation and response modeling of latent neural dynamics. It integrates streaming latent space construction (including a novel streaming jPCA method), a nonparametric stimulation-response map, and a constrained optimization that designs high-dimensional stimuli to drive low-dimensional dynamics along desired directions under experimental sparsity and non-negativity constraints.

## Strengths
- **Novel streaming jPCA (sjPCA):** Proposes a new streaming variant of jPCA for real-time identification of rotational latent dynamics, demonstrated to converge to offline fits (Section 2.1, Fig. 1a).
- **Adaptive, nonparametric stimulation-response modeling:** Uses kernel regression to map stimulations to latent effects, accounting for state-dependence and temporal non-stationarity; shown to robustly adapt to drifts and discontinuities (Section 2.3, Eq. 7, Fig. 2e).
- **Practical optimization with experimental constraints:** Formulates an optimization problem for high-dimensional stimulus design with non-negativity and sparsity penalties, directly addressing limitations of tools like holographic optogenetics (Section 2.4, Eq. 8, Fig. 4).
- **Real-time performance on diverse data:** Validates the integrated framework on simulated data and two real neural datasets (calcium imaging, electrophysiology), with end-to-end runtimes under 100 ms, enabling future *in vivo* applications (Sections 3, 4, Appendix H).

## Weaknesses
- **No validation with real neural perturbations:** Experiments on real data use simulated stimulation effects (autoregressive model), not actual optogenetic or electrical stimulation responses. This leaves the core claim of causally driving latent dynamics unproven for real biological systems (Sections 4.1, 4.2).
- **Insufficient comparison to state-of-the-art baselines:** The optimization is compared only to random strategies and a naive model; it lacks comparison to established stimulation design methods like Bayesian optimization or active learning cited in related work, limiting assessment of its advancement (Fig. 4a, no comparison to methods such as Minai et al. 2024).
- **Approximate sparsity constraint handling:** The optimization uses an L1 penalty to approximate L0 sparsity, but the paper does not analyze how closely this enforces exact neuron counts or the trade-offs involved, which is critical for experiments with hard target limits (Eq. 8, no analysis of achieved sparsity vs. constraint).
- **Incomplete evaluation of adaptive latent space selection:** The framework runs multiple latent spaces in parallel, but the benefit and mechanism of adaptively selecting the best representation for stimulation design are not thoroughly quantified or explained (Fig. 1c, Appendix A.4).
- **Lack of statistical rigor:** Key quantitative results (e.g., alignment angles in Fig. 4, prediction errors in Figs. 2e, 3c) are presented without measures of variance or statistical significance across multiple runs, reducing confidence in performance claims.

## Nice-to-Haves
- Ablation study on the components of the kernel regression model (e.g., state, stimulus, and time kernels) to justify its complexity.
- Visualization of optimized high-dimensional stimulus patterns to illustrate what the method designs.
- Analysis of optimization failure modes, such as for infeasible directions, to delineate method limits.
- Quantification of adaptation speed to non-stationary stimulus-response mappings.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Insufficient methodological details for sjPCA and kernel regression:** The paper provides algorithmic descriptions and promises code release; missing derivations do not invalidate the contributions.
- **Demand for exploration of nonlinear manifolds:** The paper explicitly scopes to affine latent spaces; nonlinear methods are outside its stated contributions.
- **Formatting or writing style nitpicks:** None were substantive enough to include.

## Novel Insights
The paper's primary novel insight is the integration of streaming latent space estimation, adaptive stimulation-response modeling, and constrained high-dimensional optimization into a single real-time framework for causal neural dynamics interrogation. Beyond this, the reviews highlight that the method's efficacy on real perturbations and comparisons to existing approaches are critical gaps that, if addressed, would significantly strengthen its impact.

## Suggestions
- Apply the framework to a publicly available dataset with recorded neural responses to real optogenetic or electrical stimulations (e.g., from Daie et al. 2021 or Draelos et al. 2025) to validate the stimulation-response learning and optimization.
- Benchmark the optimization against state-of-the-art baselines like Bayesian optimization or active learning for stimulation design, at least in simulation.
- Include error bars or confidence intervals in key figures to provide statistical context for performance metrics.
- Elaborate on the adaptive latent space selection mechanism and evaluate its benefit for stimulation design in the experiments.

---

## 7yvz93kBw9

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (4.7/10)
- Match: N/A

### Final Review

## Summary
This paper proposes D²GS, a method to improve 3D Gaussian Splatting (3DGS) for sparse-view novel view synthesis. It identifies and addresses two key failure modes: overfitting (excessive Gaussian density) in near-field regions and underfitting in far-field regions. The solution combines a Depth-and-Density Guided Dropout (DD-Drop) module and a Distance-Aware Fidelity Enhancement (DAFE) module. The paper also introduces a new distribution-based metric, Inter-Model Robustness (IMR), to evaluate the stability of the learned 3D Gaussian representation.

## Strengths
- **Clear, data-driven problem analysis and well-motivated solution.** The paper effectively diagnoses distinct spatial failure modes (Figure 1) and designs two complementary components (DD-Drop for near-field, DAFE for far-field) that directly address them.
- **Novel and thoughtful evaluation metric.** The proposed IMR metric, based on Wasserstein distance between Gaussian mixture distributions, moves beyond standard 2D image metrics to directly assess the robustness and stability of the 3D representation itself, offering a new lens for analysis in the field.
- **Extensive experimental validation.** The method is evaluated on multiple standard datasets (LLFF, Mip-NeRF360, DTU) under various sparse settings, consistently showing improved performance over strong baselines. Ablation studies are thorough and validate the contribution of each component.

## Weaknesses
- **Dependence on external depth priors.** Both core modules (DD-Drop's depth layering and DAFE's supervision mask) rely on monocular depth estimates. While an ablation shows consistent gains across different estimators (Table 6), the method's performance is inherently tied to the quality and generalizability of this external prior, which may fail in challenging scenes (e.g., textureless or reflective surfaces).
- **The utility and validation of the IMR metric are under-explored.** While novel, the paper does not establish the practical significance of IMR for end-users. For instance, it does not show a correlation between IMR and perceptual quality or training instability, nor does it compare IMR scores for key baselines (e.g., 3DGS, DropGaussian) to substantiate the claim that D²GS yields "more robust" distributions.
- **Incomplete ablation baseline.** The primary ablation study (Table 4) uses vanilla 3DGS as the baseline. Since the method is built upon and directly improves DropGaussian's dropout strategy, a direct ablation comparing guided dropout (DD-Drop) against uniform random dropout (DropGaussian) is necessary to cleanly isolate the benefit of the proposed guidance mechanism.

## Nice-to-Haves
- A more detailed breakdown of the computational overhead introduced by each new component (density computation, depth estimation, IMR calculation) to better assess the practicality of the added cost.
- An exploration of simple adaptive schemes for the hand-crafted thresholds (e.g., depth tertiles, DAFE masking ratio τ) to improve generalization across diverse scenes.
- A brief discussion correlating IMR scores with more intuitive measures of instability (e.g., variance in PSNR across runs) to help the community interpret the metric.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Insufficient methodological details for reproducibility (density computation, IMR sampling)."** The paper provides key details: density is computed via k-NN (k=6 stated in Appendix B), IMR uses depth-stratified importance sampling. Further implementation specifics are standard for the community.
- **Weakness: "Lack of statistical validation for quantitative improvements (no standard deviations)."** Single-run evaluation for PSNR/SSIM is the standard in this field for large-scale benchmarks. The stability claim is separately addressed by the IMR metric over multiple runs.
- **Weakness: "Missing comparison with recent feed-forward methods (PixelSplat, MVSplat)."** The paper's scope is improving optimization-based sparse-view 3DGS. Feed-forward methods represent a different paradigm (test-time feed-forward prediction) and are not standard baselines for this line of work.
- **Weakness: "No validation of the Bures distance approximation."** The approximation is derived to improve numerical stability and efficiency. Demanding an error analysis for a derived approximation used in a novel metric is an arbitrary rigor requirement beyond community standards for an empirical paper.
- **Weakness: "Omitted discussion of broader impact."** While good practice, its absence is not a technical weakness of the contribution.
- **Strength: "The paper is well-written."** This is generic and applies to any competently written paper.

## Novel Insights
The paper provides a systematic analysis revealing that sparse-view 3DGS fails in spatially distinct ways: it overfits by placing too many Gaussians in texture-rich near-field regions and underfits by placing too few in far-field regions. This insight directly motivates a unified solution with two spatially complementary components. Furthermore, it introduces a novel distribution-based robustness metric (IMR) that shifts evaluation from 2D image space to the stability of the 3D representation itself, a conceptual advance for assessing 3D reconstruction methods.

## Suggestions
- Compute and report the IMR metric for key baseline methods (e.g., 3DGS, DropGaussian) to provide direct, quantitative evidence supporting the claim of improved robustness.
- Include DropGaussian (uniform random dropout) as an ablation baseline in Table 4 to directly demonstrate the advantage of the proposed depth-and-density guidance over a naive dropout strategy.
- Add a brief analysis or discussion on how the method might perform when the monocular depth prior is particularly noisy or unreliable, to better characterize its limitations.

---

## CTEXdHB1BB

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.2/10)
- Match: N/A

### Final Review

## Summary
The paper introduces Conditional Advantage Estimation (CANON), a novel advantage estimation method for reinforcement learning with verifiable rewards (RLVR) in large reasoning models. CANON incorporates human priors on training metrics (e.g., entropy, response length) without assuming a directional preference by regrouping responses into two groups based on the metric and computing inter-group and intra-group advantages. Experiments across three LLMs and multiple reasoning benchmarks demonstrate improved accuracy and token efficiency, and the method achieves a superior Pareto frontier in performance-cost trade-offs.

## Strengths
- **Novel and principled methodology**: The core idea of conditional regrouping to avoid hardcoded directional priors is innovative and addresses a clear limitation in prior reward/advantage shaping techniques. The method is well-motivated and theoretically grounded with two theorems showing amplification properties.
- **Extensive and rigorous evaluation**: The paper evaluates three different LLMs on six math reasoning benchmarks and three high-complexity logic reasoning tasks, comparing against a wide array of strong baselines (including ReMax, RLOO, GRPO, DR.GRPO, and entropy/length-specific methods). The results consistently show improvements in accuracy (e.g., +1.9 points on math tasks, up to +5.2 points on hard logic tasks) and efficiency (33.8% token reduction).
- **Flexibility and practical benefits**: CANON supports dynamic scheduling (via µ) to balance exploitation and exploration across tasks, and weighting (via α) for efficient reasoning, achieving a new Pareto frontier. The framework is general and can be applied to different metrics.

## Weaknesses
- **Lack of statistical significance reporting**: The paper reports single accuracy numbers without confidence intervals or standard errors, which is a standard expectation for empirical results at ICLR. This makes it difficult to assess the reliability of the claimed improvements, especially for marginal gains.
- **Hyperparameter tuning for dynamic scheduling**: While the core CANON method with fixed µ already shows gains, the dynamic scheduling introduces additional hyperparameters (µ scheduling strategies) that require task- or model-specific tuning. The paper selects the best schedule per model, which could overstate the benefits of scheduling without proper ablation (e.g., comparing to a fixed µ baseline other than 0.5).
- **Limited analysis of the method's mechanisms**: The paper provides training dynamics and metric trends, but lacks qualitative analysis of the final models' reasoning behaviors (e.g., case studies comparing reasoning chains) and a deeper investigation of how inter- and intra-group advantages affect gradient updates. This would strengthen the understanding of why CANON works.

## Nice-to-Haves
- **Exploration of additional metrics and domains**: Testing CANON on other metrics (e.g., confidence, diversity) and non-reasoning tasks (e.g., code generation) would further demonstrate its generality.
- **Sensitivity analysis of theoretical assumptions**: Empirical investigation of how deviations from equal group sizes or condition independence affect performance would bolster the theoretical claims.
- **Computational overhead discussion**: Explicitly stating the negligible overhead of sorting and group mean calculations would address practical deployment concerns.

## Removed Points 
These points are flagged to be removed, treat them with caution:
- **Missing comparison with sophisticated baselines**: The paper does compare with Entropy Adv and Clip-Cov for entropy, and with length penalty methods, as shown in Table 1 and Table 3.
- **Ablation on grouping operation**: The paper includes an experiment with random regrouping (Table 12) showing no improvement, which addresses the necessity of meaningful grouping.
- **Formatting issues in tables**: These are likely parser artifacts from extraction and not inherent to the paper.

## Novel Insights
The paper's main novel insight is that by regrouping responses based on a metric and comparing across and within groups, one can amplify the metric's influence without imposing a prior on its direction, thereby enabling adaptive exploitation of beneficial trends (e.g., low entropy for math, high entropy for complex logic) and efficient reasoning. The theoretical analysis further shows that this amplification is selective to the chosen metric.

## Suggestions
- **Report statistical significance**: For key benchmarks, provide confidence intervals (e.g., via bootstrapping) or standard errors over multiple runs to substantiate the improvements.
- **Include qualitative case studies**: Show concrete examples of reasoning chains where CANON-Inter or CANON-Intra leads to correct solutions that baselines miss, and analyze failure cases.
- **Ablation on fixed vs. dynamic scheduling**: Compare the best dynamic schedule against a few fixed µ values (e.g., 0.2, 0.8) to better isolate the benefit of scheduling.

---

## DZUehXNiBn

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
VISTA is a modular framework for large-scale causal structure learning that decomposes the global problem into local Markov Blanket subgraphs, aggregates them via a novel weighted voting mechanism that down-weights low-support edges, and enforces acyclicity with a Feedback Arc Set heuristic. The method is model-agnostic, comes with finite-sample error bounds and asymptotic consistency guarantees, and demonstrates improved accuracy and efficiency across a variety of base learners in synthetic experiments.

## Strengths
- **Genuinely model-agnostic and modular design.** The framework imposes no assumptions on the base learner, Markov Blanket estimator, or data distribution, acting as a plug-and-play wrapper. This is evidenced by consistent improvements across diverse base learners (NOTEARS, GOLEM, DAG-GNN, etc.) in Tables 1 and 2.
- **Rigorous theoretical grounding.** The paper provides finite-sample error bounds for the weighted voting scheme (Theorems 3.2, 3.4) and proves asymptotic consistency under mild conditions (Theorem 3.5), a significant contribution over heuristic merging methods.
- **Comprehensive and convincing empirical evaluation.** Experiments cover multiple graph families (ER, scale-free), sizes (up to 300 nodes), and data-generating models (linear, nonlinear), consistently showing reductions in False Discovery Rate (FDR) and Structural Hamming Distance (SHD) while often improving F1 score (Tables 1, 2, 9-14). Runtime improvements are substantial (Table 3).

## Weaknesses
- **Limited validation on large real-world datasets.** The only real-data experiment is on the small Sachs network (11 nodes). While synthetic scalability to 300 nodes is shown, demonstrating performance on larger, real-world benchmarks with hundreds/thousands of variables is needed to fully substantiate the practical scalability claim.
- **Performance trade-offs are not thoroughly discussed.** In some cases (e.g., NOTEARS+VISTA-WV on ER5 in Table 1), the weighted voting improves precision but reduces True Positive Rate (TPR) compared to the baseline. The paper would benefit from a clearer discussion of when and why this recall trade-off occurs and how it relates to the theoretical precision-recall trade-off controlled by λ.
- **Theoretical bounds rely on an idealized independence assumption.** The analysis assumes votes from different subgraphs are independent, which is acknowledged as an idealization since subgraphs overlap and share data. The paper does not quantify how violations of this assumption affect the practical validity of the bounds, leaving a gap between theory and practice.
- **Lacks concrete guidance for hyperparameter selection.** While Theorem 3.4 provides a feasible range for λ and Figure 4 shows sensitivity, the paper uses fixed values (λ=0.5, t=0.7) without a data-driven procedure for choosing them when ground truth is unavailable. A practical tuning strategy would strengthen usability.

## Nice-to-Haves
- **Runtime breakdown and parallel scaling analysis.** Reporting the time spent on Markov Blanket identification, local learning, and aggregation separately would clarify the source of speedups. Demonstrating strong scaling with more cores would better support the parallelization claim.
- **Comparison with a broader set of modular baselines.** The comparison with DCILP is valuable; including other recent divide-and-conquer methods (e.g., Shah et al. 2024) would further contextualize the contribution.
- **Inclusion of constraint-based base learners.** Testing with algorithms like PC or FCI would further validate the model-agnostic claim across fundamentally different learner families.
- **Visual case studies.** Side-by-side visualizations of true vs. recovered graphs for representative cases could intuitively show the types of errors VISTA corrects.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism that the GreedyFAS acyclicity enforcement is "heuristic."** Using an efficient approximation for the NP-hard Feedback Arc Set problem is standard practice; the paper justifies its ordering (FAS before thresholding) and the method provides a valid DAG.
- **Demand for statistical significance testing.** Reporting mean and standard deviation over multiple runs is standard in the field; formal significance testing is not a universal requirement.
- **Criticism that hyperparameters are fixed "without justification."** The paper provides a theoretical admissible range (Theorem 3.4) and empirically explores the precision-recall trade-off (Figure 4), which constitutes reasonable justification for the chosen operating point.
- **Suggestion that the framework "sometimes harms recall" is an overstatement.** The overall results show consistent improvements in F1 and SHD; the occasional TPR drop is a recognized trade-off for substantially improved precision, which is discussed in the context of the weighted voting mechanism.

## Novel Insights
The paper’s core novel insight is the design of a weighted voting aggregation rule with an exponential decay term (1−e^{-λm}) that acts as a data-dependent pseudo-count, dynamically regularizing edges based on their frequency of appearance across subgraphs. This provides a principled, tunable mechanism to suppress low-support noise while preserving high-confidence signals, moving beyond simple majority voting. The accompanying theoretical analysis explicitly links the hyperparameter λ to a feasible operating range and to the precision-recall trade-off, offering a formal understanding of how the aggregation calibrates confidence.

## Suggestions
- **Supplement the real-data evaluation** with at least one larger-scale benchmark where a consensus causal structure or interventional data can serve as a proxy ground truth (e.g., a gene regulatory network dataset).
- **Add a brief discussion or empirical analysis** on how correlated votes (due to overlapping subgraphs) might affect the concentration bounds in practice, perhaps by estimating vote correlations in the synthetic experiments.
- **Provide a practical recommendation** for selecting λ and t in the absence of ground truth, such as using a score function (e.g., BIC) on a held-out validation set or proposing a default value based on graph sparsity.

---

## ZMzha5gbnF

- GT: Accept (Poster) (avg 7.0)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
This paper identifies a "priming vulnerability" specific to Masked Diffusion Language Models (MDLMs), where injecting affirmative tokens at intermediate steps of the iterative denoising process can steer safety-aligned models to generate harmful responses. The authors propose a novel defense, Recovery Alignment (RA), which trains models to recover and produce safe outputs even when starting from such adversarially contaminated states. Experiments show RA significantly mitigates this vulnerability and improves robustness against conventional jailbreak attacks while largely preserving general task performance.

## Strengths
- **Identifies a novel, DLM-specific safety weakness:** The work provides clear, quantitative evidence (e.g., Attack Success Rate jumps from 2% to 21% with a single token intervention via the "anchoring attack") of a critical vulnerability inherent to the parallel, iterative denoising process of MDLMs, differentiating it from prior work on autoregressive models.
- **Effective and tailored mitigation:** The proposed Recovery Alignment (RA) is well-motivated and directly addresses the core issue by training models to generate safe responses from intentionally contaminated intermediate states. Empirical results are strong, showing RA drastically reduces ASR across multiple priming-based attacks and outperforms baseline alignment methods (SFT, DPO, MOSA) across three MDLMs.
- **Comprehensive and rigorous evaluation:** The paper validates its claims through extensive experiments on two datasets (JBB-Behaviors, AdvBench) using three safety evaluators (GPT-4o, guardrail model, keyword matching), multiple attack families (including proposed First-Step GCG), and ablation studies on scheduling and generation length. General capability is preserved across 11 diverse benchmarks.

## Weaknesses
- **Performance degrades under very strong attacks:** As shown in Table 2, for very late intervention steps (e.g., `t_inter=32`), where many harmful tokens are anchored, RA's Attack Success Rate remains high (50–79%). The paper notes generating a safe response from many fixed anchors is "practically impossible," but a deeper analysis of these failure modes (e.g., does the model output gibberish, partial harm, or a different unsafe response?) is missing.
- **Theoretical assumption's scope is not fully characterized:** Theorem 4.1, which enables the efficient First-Step GCG attack, relies on a monotonicity assumption. While empirically validated in Appendix C.2 for the studied models and attack states, a more formal discussion of the conditions under which this assumption may fail (e.g., for highly unnatural sequences) would strengthen the theoretical contribution.
- **Computational cost of alignment:** RA, as an RLHF-style method, incurs higher training cost (~16 hours on 4 H100 GPUs) compared to supervised baselines like SFT or DPO (Appendix C.4). While reasonable for the study, this may impact scalability and practicality for very large models.

## Nice-to-Haves
- A qualitative analysis of denoising trajectories for RA versus baseline models to illustrate the hypothesized "recovery" mechanism in action.
- A preliminary exploration of a supervised (e.g., DPO-style) variant of RA to assess if similar robustness can be achieved with lower training cost, as suggested in the Limitations section.
- Reporting the inference-time latency/throughput of RA-aligned models compared to originals to assess any deployment overhead.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness about reward model justification:** The paper uses DeBERTaV3 as a reward model, citing its use in prior work (Köpf et al., 2023). This is a reasonable, established choice, and an ablation is not required for the core claim.
- **Weakness demanding comparison against adapted ARM jailbreak methods:** The paper's scope is DLM-specific vulnerabilities and defenses. Requiring adaptation and benchmarking of all state-of-the-art ARM attacks is tangential and not a core evaluation flaw.
- **Weakness about over-reliance on LLM-as-judge:** The paper employs three distinct evaluation metrics (GPT-4o, guardrail model, keyword matching), which is a robust and standard practice in the field. Consistency across metrics is discussed in the appendix.
- **Weakness about potential safety degradation from training on harmful data:** The paper evaluates general capability and shows no substantial degradation (Table 4). A specific "poisoning" evaluation is beyond the standard scope and is not required to validate the method's efficacy.
- **Weakness about testing RA against intervention steps > 96:** The paper systematically evaluates up to `t_inter=32` (25% of total steps) and shows a clear trend. Testing even later steps, while interesting, is not necessary to establish the core contribution—identifying the vulnerability and providing a significant mitigation.

## Novel Insights
The paper provides a novel and important insight: the iterative, parallel denoising process of MDLMs introduces a unique safety vulnerability where early affirmative tokens can irrevocably bias the generation trajectory toward harmful content, a phenomenon distinct from attacks on autoregressive models. Furthermore, it demonstrates that standard alignment, which only trains models from a fully masked state, is fundamentally insufficient to defend against this, necessitating alignment that explicitly conditions on and recovers from contaminated intermediate states—a principle that also generalizes to improve robustness against conventional jailbreak attacks.

## Suggestions
- Conduct a qualitative analysis of model outputs in high-ASR failure cases (e.g., for `t_inter=32`) to better characterize the failure mode and inform potential complementary defenses.
- Provide a more formal discussion or empirical bounds on the monotonicity assumption in Theorem 4.1, clarifying the types of sequences or model states where it may not hold.

---

## MwuSvrthXq

- GT: Reject (avg 4.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper presents WeCAN, an end-to-end reinforcement learning framework for heterogeneous DAG scheduling with task-resource compatibility constraints. Its key contributions are a weighted cross-attention (WeCA) network that adaptively embeds variable-sized environments using compatibility coefficients, and a theoretically-grounded method to enable "skip" actions within a single-pass inference scheme to address an optimality gap inherent in list scheduling.

## Strengths
- **Innovative and adaptable architecture.** The weighted cross-attention layer uses compatibility coefficients as attention biases outside the softmax, allowing the model to fully utilize detailed environmental information without being constrained to a fixed number of task types or resource pools. This enables strong generalization to unseen environment sizes and types, as evidenced by experiments across eight distinct fluctuating settings (Section 5.2, Figure 2, Appendix F.2).
- **Theoretical analysis and mitigation of a fundamental limitation.** The paper provides a formal analysis of the solution space, proving why standard list-scheduling generation maps cannot guarantee optimal solutions (Section 4, Appendix A.3). It then designs a skip-action mechanism that, within the single-pass framework, theoretically closes this gap (Theorem 1). Empirical results on "heavy-task" cases validate that this mechanism yields significant improvements where the gap is most pronounced (Figure 3, Table 8).
- **Strong and efficient empirical performance.** WeCAN demonstrates consistent and significant improvements in makespan over strong heuristic (e.g., HEFT, Tetris) and neural baselines (PPO-BiHyb, One-Shot) on both real-world-derived (TPC-H) and synthetic (Computation Graphs) benchmarks, with gains up to 18.1% and 13.4%, respectively (Tables 1, 2). Its single-pass design ensures inference time (in greedy mode) is comparable to fast heuristics (Table 20).

## Weaknesses
- **Lack of open-source code.** Reproducing the complex system—including the network architecture, training loop, and GPU-accelerated generation map—is severely hampered without publicly available code. This is a significant barrier to verification and adoption, falling short of ICLR's reproducibility standards.
- **Incomplete empirical justification for the skip action in standard settings.** The theoretical necessity of the skip action is well-argued, and its benefit is demonstrated on a specialized "heavy-task" benchmark. However, the paper does not provide a clear ablation quantifying its contribution on the primary TPC-H and Computation Graphs benchmarks (where it was disabled per Appendix H.3). This leaves a disconnect between the theory (skip fixes a general gap) and the main empirical results (where gains are attributed to other components).
- **Dense theoretical and methodological presentation.** The descriptions of the reduced/original spaces, projection maps (Section 4), and the decoder's skip-score mechanism (Section 3.2) are highly technical and lack intuitive grounding in the main text. While the appendix contains necessary details, the flow impedes accessibility for a broad audience.

## Nice-to-Haves
- **Visualization or analysis of learned attention patterns.** Showing that the WeCA layer indeed attends more to compatible pools would provide direct evidence that the architecture works as intended.
- **Further characterization of the skip action's impact.** A deeper analysis of the graph/resource conditions (beyond "heavy tasks") where skip is most beneficial or detrimental would provide more predictive insight for practitioners.
- **Extended discussion of baseline adaptations.** A more detailed comparison of the fairness and limitations of simplifying adaptations made for neural baselines (e.g., using average features in One-Shot) would further contextualize WeCAN's advantages.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Lack of statistical significance tests for greedy evaluations."** The paper reports standard deviations for sampling methods. Running multiple seeds for greedy evaluation is not a standard requirement in the field for reporting point estimates, and the performance gaps are large and consistent.
- **Weakness: "The skip action's functional form lacks motivation."** The paper explains the form `ua*(1-2k/n)^ub + uc` is designed to prevent endless idling by making the score dynamic yet computable in a single pass. While more intuition could be helpful, the design is justified by the constraint of single-pass efficiency.
- **Weakness: "Need for systematic generalization tests beyond the eight fluctuations shown."** The paper already provides extensive generalization tests across pool number, type, task type, and capacity (Figure 2, Appendix F.2). Demanding tests on orders-of-magnitude larger graphs or entirely novel configurations is beyond the paper's stated scope of adaptability within heterogeneous environments.
- **Weakness: "Requirement for a controlled ablation replacing WeCA with a simple MLP encoder."** The ablation study (Table 3) already includes variants like "WeCA-inside" and "WeCA-final-only," which demonstrate the importance of the specific design. A direct comparison to a simple adaptable baseline, while interesting, is not required to validate the core architectural contribution.
- **Weakness: "Runtime comparison should be on identical hardware."** The paper provides a detailed runtime breakdown (Table 20) and explains the use of GPU/CPU appropriately for different scenarios (training vs. greedy inference). The key claim—that single-pass inference keeps total time close to the heuristic-dominated generation map—is supported.

## Novel Insights
The paper provides a novel synthesis of architectural and theoretical insights. The weighted cross-attention mechanism offers a principled way to embed pairwise compatibility matrices into a scalable, size-agnostic neural architecture, moving beyond fixed-dimensional embeddings or averaging. More significantly, the formal framing of the scheduling process via original/reduced spaces and generation maps yields a general criterion (surjectivity of `TS`) for when a method can guarantee optimal solutions. This analysis concretely identifies the optimality gap in list scheduling and motivates the skip-action design, which is shown to close this gap while maintaining single-pass efficiency—a non-trivial integration of theoretical correction into a practical, efficient system.

## Suggestions
- **Commit to releasing full source code**, including training scripts, baseline implementations, and dataset processing code, to meet conference reproducibility standards.
- **Add an ablation experiment** on the primary benchmarks (TPC-H, Computation Graphs) comparing WeCAN with and without the skip action, clearly reporting the performance delta to empirically ground its contribution in the standard evaluation setting.
- **Improve the exposition in Sections 3.2 and 4** by adding a concise, intuitive explanation of the skip-score dynamics and a simple, concrete example in the main text to illustrate the optimality gap and how the skip action addresses it.

---

## khHNHzRjMy

- GT: Reject (avg 3.0)
- Predicted: N/A (4.1/10)
- Match: N/A

### Final Review

## Summary
This paper introduces EmoSign, the first multimodal dataset dedicated to emotion recognition in American Sign Language (ASL). It provides 200 video clips annotated by Deaf native signers with sentiment scores, fine-grained emotion labels, and descriptions of visual emotion cues. The authors benchmark several multimodal LLMs, revealing that current models fail to effectively use visual information for affective reasoning and exhibit a bias towards positive/neutral predictions.

## Strengths
- **Novel, High-Value Dataset**: EmoSign is the first ASL dataset with detailed emotion and sentiment labels annotated by Deaf native signers, addressing a critical gap in sign language and affective computing research. The inclusion of open-ended descriptions of visual emotion cues (e.g., facial expressions, signing speed) provides unique qualitative insights.
- **Rigorous and Ethical Annotation**: The dataset construction is methodologically sound, involving three Deaf ASL signers with professional interpretation expertise. The reported inter-annotator agreement (average Krippendorff’s alpha = 0.593) is solid and favorably compares to established emotion datasets like MELD and IEMOCAP.
- **Insightful Benchmark Analysis**: The ablation studies (caption-only, video-only, video+caption) clearly demonstrate a key failure mode of current multimodal LLMs: severe underperformance and systematic bias in the video-only condition, coupled with an over-reliance on textual captions for emotion reasoning, which is particularly problematic for sign language.

## Weaknesses
- **Potential Selection Bias from Text-Based Filtering**: The dataset was constructed by filtering an existing corpus (ASLLRP) using VADER sentiment scores on the English captions to select the "most positive" and "most negative" clips. The paper notes in the limitations that VADER results often differed from the annotators' judgments, suggesting visual emotional content can diverge from textual sentiment. This filtering method may have introduced a bias in the sample, potentially excluding clips where strong emotion is expressed visually but not in the text. This critical methodological choice and its implications should be discussed more thoroughly in the main methodology section (Section 3.1).
- **Unreliable Per-Class Evaluation for Emotion Classification**: For the single-label emotion classification task, the paper reports per-class accuracy on highly imbalanced classes (some with as few as 5-10 instances, as seen in Figure 24). These per-class numbers are statistically unreliable and can be misleading. The analysis should primarily rely on the reported weighted metrics (wAcc, wF1) and include a caution against over-interpreting the per-class accuracies, or use a more robust metric like macro-averaged F1.
- **Lack of Demonstration for Model Training Utility**: The benchmark is limited to zero-shot evaluation of pre-trained models. The paper does not include any fine-tuning experiments, which limits the demonstration of EmoSign's practical utility for improving model capabilities on this task. A fine-tuning experiment, even as a proof-of-concept, would significantly strengthen the claim that the dataset can "inspire new architectures."
- **Purely Qualitative Grounding Analysis**: The "emotion cue grounding" analysis (Section 5.3) is conducted solely via manual inspection of model reasoning outputs. While illustrative, the lack of any quantitative evaluation (e.g., measuring alignment between model-generated cues and annotator descriptions) weakens the claim about models' lack of visual grounding. A simple quantitative measure would add necessary rigor.

## Nice-to-Haves
- Benchmarking specialized sign language recognition/translation models (e.g., pose-based models, SignLLM) to differentiate general MLLM failures from a specific lack of ASL visual understanding.
- Conducting a multi-label emotion classification evaluation to fully utilize the annotation scheme and present a more realistic task.
- Providing visualizations (e.g., saliency maps) to contrast model-attended regions with the emotion cues described by Deaf annotators for key failure cases.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Lack of train/validation/test split."** This is a benchmark evaluation of zero-shot model performance; using the entire annotated set for evaluation is standard and acceptable. The paper should explicitly state this, but it is not a flaw.
- **Weakness: "Demanding statistical significance tests for benchmark metrics."** In large-scale model benchmarking, single-run evaluation is common practice. Requesting confidence intervals or significance tests imposes a rigor requirement not standard in this subfield.
- **Weakness: "The related work section is somewhat brief."** This is a generic criticism. The related work adequately covers multimodal emotion recognition and sign language ML research, and includes a specific comparison to the closest work (FePh).
- **Strength: "The paper is well-written."** This is a generic strength that applies to many papers and does not highlight what this specific paper does well.

## Novel Insights
The benchmark results provide a concrete, evidence-based demonstration of a broader issue in multimodal AI: a severe imbalance in modality reliance. For a task where the primary signal (emotional expression in sign language) is visual, state-of-the-art models perform nearly as well with text captions alone as they do with full video+text input, and fail catastrophically with video-only input. This reveals that current "multimodal" systems are often language-dominant classifiers that lack genuine, fine-grained visual understanding, a critical shortcoming for applications in non-verbal communication.

## Suggestions
- In Section 3.1, include a direct analysis (e.g., a correlation or confusion matrix) between the initial VADER sentiment scores used for filtering and the final annotator-provided sentiment labels to quantify and discuss the divergence.
- Replace or heavily caveat the per-class accuracy reporting in Table 4 and the associated analysis. Emphasize the weighted metrics and consider reporting macro-F1 for the single-label task.
- Add a proof-of-concept fine-tuning experiment using one of the open-source models (e.g., Qwen2.5-VL) on EmoSign to demonstrate that the dataset can be used to improve model performance, particularly in the video-only condition.
- Propose and implement a simple quantitative measure for the grounding task, such as calculating the overlap (e.g., Jaccard similarity) between key visual cue terms extracted from model reasoning and from annotator descriptions for a subset of videos.

---

## CQ0U1wZYoy

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
PRISM introduces a prompted conditional diffusion framework for the compound and controllable restoration of scientific images. It combines weighted contrastive disentanglement with compound-aware supervision to enable simultaneous removal of overlapping degradations and selective, prompt-guided correction. The method outperforms state-of-the-art baselines on complex mixtures and demonstrates that controllability improves downstream scientific accuracy across domains like microscopy, remote sensing, and ecology.

## Strengths
- **Targets a critical, underexplored problem:** The paper convincingly argues that scientific images suffer from compound degradations requiring simultaneous correction and expert control, moving beyond sequential or aesthetics-focused restoration.
- **Novel technical design:** The weighted contrastive disentanglement objective, coupled with compound-aware supervision, creates a structured latent space that supports both joint restoration and precise, prompt-driven intervention—a step beyond existing all-in-one or diffusion-based methods.
- **Comprehensive and domain-relevant evaluation:** Introduces a new Mixed Degradations Benchmark and evaluates downstream scientific utility (e.g., classification, segmentation) across multiple real-world domains, showing that selective restoration meaningfully improves task performance.

## Weaknesses
- **Lack of direct validation for compositional latent space:** The central claim of a "structured, compositional latent space" is supported only indirectly through downstream results; no quantitative metrics (e.g., disentanglement scores) or visualizations (e.g., t-SNE plots) are provided to confirm that degradation primitives correspond to separable directions or that mixtures interpolate compositionally.
- **Controllability evaluation is not quantitatively rigorous:** While downstream tasks show benefits from selective restoration, there is no measure of "prompt faithfulness"—how accurately the model removes only the specified distortions without affecting others. This limits assessment of the precision of control.
- **Downstream utility assessment may conflate factors:** Using off-the-shelf pretrained models for downstream tasks does not isolate the effect of restoration from those models' domain adaptation capabilities; a more controlled experiment (e.g., training from scratch on restored data) would strengthen the claim.
- **Insufficient analysis of failure modes and limitations:** The discussion of limitations is brief; deeper analysis is needed on scenarios where PRISM might fail, such as under extreme distortion intensities, non-linear interactions not captured by synthetic augmentations, or when real-world distortions deviate significantly from the training primitives.
- **Key implementation details are relegated to the appendix:** Critical components like the quality regularizer \(\hat{p}(c|e_{\text{clean}})\) implementation and the performance of the automatic distortion-prediction MLP are only briefly mentioned in the main text, reducing clarity and making it harder to assess the method's robustness.

## Nice-to-Haves
- Including key ablation results (e.g., the individual contributions of the contrastive loss and quality regularizer) in the main text would improve readability and justification.
- A more detailed computational cost comparison against non-diffusion baselines in the main text would help practitioners evaluate deployment trade-offs.
- Extending the sensitivity analysis for prompt phrasing variations beyond the appendix would strengthen the claim of a robust natural-language interface.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Criticism about unfair comparison due to different training data:** The paper explicitly states that all baselines are trained on the same fixed set of primitive distortions (Sec. 3.2), making this point factually incorrect.
- **Demand for evaluation on entirely new primitive distortions:** The paper's scope is compositional generalization to unseen mixtures of known primitives, not handling novel distortion types; criticizing the absence of the latter is scope creep.
- **Nitpicks about writing style or formatting:** No substantive formatting issues were identified in the reviews that warrant inclusion.
- **Claim that the method does not address real-world gaps:** The paper acknowledges reliance on synthetic data and discusses generalization limits, so this is already reflected in the weaknesses; the reviewer's version was overly harsh and partially addressed.

## Novel Insights
The paper's core novel insight is that in scientific imaging, controllable restoration—enabling experts to selectively remove specific distortions—is not merely a convenience but a necessity for preserving task-relevant signals. This is empirically demonstrated through downstream evaluations where selective restoration outperforms full restoration in three of four domains, highlighting that indiscriminate correction can erase faint but meaningful features or introduce artifacts detrimental to scientific analysis.

## Suggestions
- Provide direct validation of the compositional latent space, such as through t-SNE visualizations colored by degradation type or quantitative disentanglement metrics, to substantiate the claimed structure.
- Develop and report a quantitative metric for prompt faithfulness (e.g., measuring the removal of targeted distortions while preserving others) to rigorously evaluate controllability beyond downstream tasks.
- Conduct a controlled downstream experiment by training a simple model from scratch on restored versus clean images, isolating the effect of restoration from domain adaptation of off-the-shelf models.

---

## ey7CXUBn1g

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes AdaSVD, a method for compressing Large Language Models via Singular Value Decomposition. It introduces two core components: **adaComp**, an iterative procedure that compensates for SVD truncation error by alternately updating the singular matrices using a stabilized Moore-Penrose pseudoinverse solution, and **adaCR**, a heuristic that assigns layer-specific compression ratios based on a simple input-output similarity metric. The method demonstrates consistent performance improvements over existing SVD-based baselines across multiple model families and high compression ratios.

## Strengths
- **Comprehensive and convincing empirical evaluation.** The paper validates its method across multiple LLM families (LLaMA, OPT, Mistral, Vicuna), compression ratios (40%-80%), and task types (perplexity, QA, VLM captioning). The gains are particularly pronounced at higher compression ratios (e.g., 60%), which is a critical target for deployment.
- **Effective ablation studies and practical engineering.** The ablation studies in Tables 3a-3d cleanly isolate the contributions of each component, showing that adaComp is crucial at high ratios and adaCR provides consistent gains. The "stack-of-batch" strategy for calibration data is a clever, practical solution to a real memory constraint.
- **Orthogonality to other compression techniques.** The paper demonstrates that AdaSVD can be effectively combined with weight quantization (GPTQ), outperforming the quantized version of the strongest baseline (SVD-LLM+GPTQ), which enhances its practical utility.

## Weaknesses
- **Lack of computational cost analysis.** The paper claims efficiency but provides no quantification of the computational overhead of the iterative adaComp procedure or the layer importance estimation. The wall-clock time and memory cost of the compression process itself (not inference) compared to baselines like SVD-LLM are critical for assessing practical deployment trade-offs and are absent.
- **Weak justification and analysis for the adaptive compression ratio (adaCR) heuristic.** The core importance metric—cosine similarity between a layer's input and output—is intuitive but simplistic. The paper does not justify why this is a good proxy for a layer's importance to the *final model performance* under SVD compression, nor does it analyze its sensitivity or compare it to other possible metrics (e.g., gradient-based). This leaves the foundation of adaCR somewhat under-supported.
- **Severely hindered clarity due to parsing artifacts.** While not a flaw of the scientific content, the extracted text contains garbled tables, broken equations, and misplaced text/figure references (e.g., Table 1, Section 3.1 derivations). This significantly obstructs a detailed assessment of the methodology and results. For a conference submission, the authors must ensure a clean, readable PDF.

## Nice-to-Haves
- A more thorough discussion relating the adaCR importance metric to prior work on layer importance in pruning and compression.
- A sensitivity analysis of the adaComp procedure to the amount and quality of calibration data.
- Exploration on even larger-scale models (e.g., 70B parameters) to further stress-test the method's scalability.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "Evaluate on larger-scale LLMs (e.g., 13B, 70B parameters)."** The paper already evaluates models up to 13B parameters (LLaMA-13B). Demanding evaluation on a 70B model is a "larger model" generic request and is not required to validate the core contribution.
- **Weakness: "Benchmark on diverse downstream tasks beyond perplexity and QA" and "Compare with non-SVD compression baselines (e.g., pruning)."** These are requests for scope creep. The paper's contribution is an improvement within the SVD-based compression paradigm, evaluated against SVD baselines on standard LM/QA benchmarks. Requiring comparisons to fundamentally different compression families (pruning) or a broader suite of generation tasks is not standard for establishing a SOTA advance in this specific area.
- **Weakness: "Convergence and stability analysis for adaComp."** While interesting, demanding a theoretical convergence analysis for an empirical, post-training compression method is imposing a rigor requirement not standard in this applied field. The empirical ablation (Table 3c) shows the procedure works effectively.
- **Weakness: "Report inference speed or latency measurements."** The paper's focus is on reducing memory footprint via compression, a valid goal independent of immediate inference speed measurements. Furthermore, SVD compression inherently changes the matmul structure, making direct latency comparison complex and hardware-dependent; its absence is not a core flaw.
- **Strength/Weakness about generic writing or topic importance.** Removed as per instructions.

## Novel Insights
The paper's primary novel insight is the integration of a stabilized, alternating update mechanism (adaComp) to directly minimize the *task-relevant* SVD truncation error (\(||U_k^\sigma V_k^{\sigma\top}X - WX||_F^2\)), moving beyond the standard Frobenius norm on weights. The use of the Moore-Penrose pseudoinverse within a least-squares formulation provides a numerically stable solution, and the "stack-of-batch" calibration strategy is a simple but effective tactic to maximize data utility under memory constraints. The combination of this compensation with a layer-adaptive compression scheme (adaCR) within a single, training-free SVD framework is also a new synthesis.

## Suggestions
- **Fix presentation completely.** Provide a clean, properly formatted PDF with legible tables, correctly rendered equations, and clear figure/text alignment for the final submission.
- **Add a computational cost analysis.** Include a table or subsection reporting the wall-clock time and peak GPU memory consumption required to compress a standard model (e.g., LLaMA2-7B) at a few key compression ratios, comparing AdaSVD directly to SVD-LLM.
- **Strengthen the discussion of the adaCR importance metric.** Justify the choice of input-output similarity more deeply, discuss its potential limitations, and optionally provide a brief comparison against another simple baseline (e.g., uniform or random allocation) to better isolate its contribution.

---

## NFB4QGGS65

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (6.6/10)
- Match: N/A

### Final Review

## Summary
This paper establishes a rigorous equivalence between the GPTQ post-training quantization algorithm and Babai’s nearest plane algorithm for the closest vector problem (CVP) on a lattice defined by the layer’s Hessian. This geometric interpretation yields a tight, layer-wise error bound for the no-clipping regime. Leveraging this theory, the authors propose new quantization methods (HPTQ, SSQR) and an efficient GPU inference kernel that outperform standard GPTQ, especially at aggressive bitwidths.

## Strengths
- **Foundational theoretical contribution:** The paper proves that GPTQ executed back-to-front is mathematically identical to Babai’s algorithm without basis reduction. This equivalence provides the first geometric interpretation and theoretical grounding for GPTQ’s empirical success, explaining why its greedy updates work well globally.
- **Derived practical benefits:** From the theory, the authors design two novel quantization methods: HPTQ (Huffman-encoded) and SSQR (scale-adjusted sparse), which avoid weight clipping and consistently outperform original GPTQ across bitwidths. A min-pivot order heuristic is also derived from the error bound analysis.
- **Comprehensive empirical validation:** The methods are evaluated on multiple model families (Qwen3, Llama), sizes (0.6B–14B), bitwidths, and benchmarks (perplexity, zero-shot tasks), showing robust gains. An optimized CUDA kernel for SSQR demonstrates ~2× end-to-end speedups in low-batch decoding.

## Weaknesses
- **Limited applicability of the core theoretical guarantee:** The error bound (Theorem 5) and the exact equivalence hold only in the no-clipping setting (`Z† = Z`). Since standard low-bit quantization (e.g., INT4) relies on clipping to a finite grid, the theoretical results do not directly cover the most common practical scenario. The paper acknowledges this but defers analysis of the clipped case to future work.
- **Incomplete empirical assessment of theoretical components:** While the proposed methods are evaluated, key aspects derived from the theory are not fully validated empirically. For instance, the impact of the min-pivot order on final accuracy is only discussed anecdotally (Section 4.5), and the tightness of the error bound is not measured against actual quantization errors across layers.
- **Scalability to very large models not demonstrated:** Experiments are limited to models up to 14B parameters, whereas the paper claims relevance for “billion-parameter models.” Demonstrating effectiveness on a standard large-scale model (e.g., 70B) would strengthen the claim of broad applicability.

## Nice-to-Haves
- A more comprehensive comparison with state-of-the-art quantization methods (e.g., QuIP#, AQLM, QTIP) across all evaluated models and bitwidths, beyond the Llama-2-7B results in Table 16.
- Ablation studies to isolate the contribution of individual components, such as the min-pivot order versus the new quantization schemes.
- Visualization of the lattice and Babai steps for a concrete low-dimensional example using real data, to make the geometric intuition more accessible.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Heavy reliance on appendix for proofs:** The main algebraic proofs are deferred to the appendix, but this is standard practice for complex derivations and does not constitute a substantive weakness.
- **Modest practical gains over strong baselines:** This criticism is nuanced; the paper shows clear improvements over GPTQ and competitiveness with SOTA methods, which is sufficient given its primary theoretical contribution. Overemphasizing incremental gains detracts from the novelty.
- **Lack of exploration of basis reduction (e.g., LLL):** The paper explicitly mentions this as future work (Section 6), and it is outside the current scope.
- **Experiments could be more diverse:** The evaluation across two model families and multiple sizes is already comprehensive for the claims made.

## Novel Insights
The paper’s core insight is that GPTQ, a widely-used empirical quantization method, is equivalent to a classical lattice algorithm (Babai’s nearest plane). This connection not only provides a geometric interpretation of GPTQ’s error propagation but also imports established error bounds from lattice theory, opening a new avenue for designing quantization algorithms via insights from computational geometry. The derivation of a tight layer-wise error bound for no-clipping quantization is a novel analytical contribution.

## Suggestions
- Include an empirical evaluation of the error bound’s tightness by measuring actual layer-wise quantization errors and comparing them to the theoretical bound across different layers and models.
- Conduct an ablation study to quantify the accuracy improvement attributable to the min-pivot order versus the act-order baseline.
- Extend experiments to a very large model (e.g., Llama-2 70B) to demonstrate scalability and robustness across the full range of claimed model sizes.

---

## 4Ha2srdhPN

- GT: Reject (avg 4.5)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper introduces GRAID, a framework for generating high-quality spatial reasoning Visual Question Answering (VQA) data using only 2D bounding box geometry from object detectors. It avoids the cascading errors of single-view 3D reconstruction and the hallucinations of caption-based generative methods. The authors generate over 8.5 million VQA pairs from three driving datasets, report a 91.16% human-validated accuracy for non-depth questions, and demonstrate that fine-tuning on GRAID data improves model performance on held-out question types and external benchmarks.

## Strengths
- **High-Quality, Large-Scale Dataset Generation**: The paper provides strong empirical evidence for GRAID's data quality through a human evaluation showing 91.16% validity on non-depth questions, a significant improvement over the 57.6% reported for a dataset from a prior method (SpatialVLM). The generation of over 8.5 million VQA pairs from three established datasets represents a substantial and valuable resource.
- **Efficient and Practical Framework Design**: The SPARQ (Sieve Predicates And Realize Questions) component is a clever engineering contribution. By using lightweight predicate checks for early rejection, it achieves significant speedups (up to 1400× for some templates), making large-scale generation feasible. The framework's reliance on standard object detector outputs enhances practicality.
- **Compelling Evidence for Transferable Concept Learning**: The experimental results robustly support the paper's claims. Fine-tuning on GRAID data leads to performance gains on held-out question types within the same dataset and generalizes to entirely different datasets. Crucially, it also improves performance on diverse, non-driving external benchmarks (e.g., BLINK, A-OKVQA), demonstrating transfer of learned spatial concepts beyond the training domain and templates.

## Weaknesses
- **Quality Contingent on Detector Accuracy, Analysis Lacking**: The quality of generated VQA pairs is inherently tied to the precision of the input bounding boxes and class labels. While the paper uses ground-truth annotations for its primary evaluation, it does not analyze how errors from a standard pretrained object detector (e.g., false positives/negatives, mislabeling, inaccurate boxes) propagate to question validity. This is a significant limitation for real-world application where ground truth is unavailable.
- **Incomplete Human Evaluation for the Full Dataset**: The impressive 91.16% human-validated accuracy is reported only for the *non-depth* variant of the BDD dataset. The validity of the depth-based questions (Closer, Farther, DepthRanking), which rely on monocular depth estimation and SAM masks, is not assessed. Since the paper releases and uses datasets with depth questions, this omission leaves the quality of a portion of the contributed data unverified.
- **Limited Direct Evidence for Domain Agnosticism**: The authors correctly state the framework is domain-agnostic, but all instantiated datasets and the primary evaluations are from the autonomous driving domain (BDD, NuImages, Waymo). While improved performance on general VQA benchmarks is encouraging indirect evidence, a direct application and quality evaluation on a distinctly different domain (e.g., indoor scenes) would be necessary to substantiate this claim fully.
- **Heuristics for Ambiguity Lack Specification and Analysis**: Several question realizers (e.g., `LeftOf`, `RightOf`) use conditions like "lie on similar planes" or check for non-overlapping boxes to avoid ambiguity. The specific thresholds or heuristics for these checks are not detailed, potentially affecting reproducibility. Furthermore, there is no analysis of whether these heuristics correctly filter ambiguous cases or inadvertently discard valid questions.

## Nice-to-Haves
- A more detailed quantitative comparison with other data generation methods (e.g., SpaRE) on a common set of images would strengthen the quality claim.
- An analysis categorizing the failure modes of the ~9% of human-invalidated GRAID samples could guide future refinements to the template logic.
- Investigating the correlation between performance on specific GRAID question types and improvements on specific external benchmark sub-tasks (e.g., which templates help most with BLINK's "Relative Depth") would provide deeper insight into concept transfer.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Inconsistent Use of 3D Information / Contradiction"** – The paper's core claim is avoiding single-view 3D reconstruction for its primary method. The depth-based questions are presented as an optional extension to demonstrate the framework's extensibility, not as part of the core contribution. The paper is clear that these questions use depth estimation, which has its own error profile, and configurable margins are used to manage ambiguity. This is a design choice, not a logical inconsistency.
- **Weakness: "Missing Limitations Section"** – While a dedicated section is conventional, the paper discusses key limitations implicitly (e.g., dependence on detector/annotation quality, domain of instantiated datasets). The absence of a formal heading is a stylistic nitpick, not a substantive flaw.
- **Weakness: "Statistical Significance Not Reported"** – For the large-scale benchmark evaluations presented (with performance differences often exceeding 10-20 points), reporting confidence intervals is not a standard practice in the field. The point estimates are sufficient to demonstrate a clear trend.
- **Weakness: "Training Setup for Baseline Not Matched"** – The paper clearly states the training setup for its main experiments (Sec 5, Appendix A.3). Demanding an exactly matched setup for a baseline that uses a different dataset of unspecified size is an unreasonable burden; the comparison serves to show that GRAID data is more effective, not to conduct a hyperparameter ablation.

## Novel Insights
The paper provides a key insight that high-fidelity training data for spatial reasoning can be generated reliably from 2D geometric primitives alone, bypassing the need for error-prone 3D reconstruction. More importantly, it demonstrates that training on *qualitative* spatial relationships (e.g., left/right, relative size) enables models to perform better on *quantitative* and complex spatial reasoning tasks (e.g., metric depth, multi-view reasoning) in held-out evaluations and external benchmarks. This suggests that VLMs can acquire a foundational, transferable understanding of spatial concepts from logically grounded, 2D-derived data, which generalizes beyond the specific relationships and domain presented during training.

## Suggestions
- Conduct a sensitivity analysis to quantify how GRAID's output quality degrades with controlled noise injected into input bounding boxes (e.g., coordinate shifts, dropped detections). This would clearly establish the framework's robustness when used with imperfect detectors.
- To substantiate the domain-agnostic claim, apply GRAID to a non-driving dataset (e.g., a subset of COCO with object annotations) to generate a small VQA set and perform a human evaluation. This would provide direct, powerful evidence for the framework's generality.
- Specify the heuristic thresholds (e.g., for "similar planes," non-overlapping IoU) used in key question realizers in the appendix to ensure full reproducibility.

---

## iIEEgI6WsF

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (7.2/10)
- Match: N/A

### Final Review

## Summary
This paper revisits the Parameter Server (PS) architecture to address a critical inefficiency in modern Fully Sharded Data Parallel (FSDP) training. The authors identify that FSDP's per-layer collective communications create synchronization barriers that lead to severe device under-utilization under the imbalanced workloads prevalent in LLM post-training due to variable sequence lengths. They propose **On-Demand Communication (ODC)**, which replaces collective `all-gather` and `reduce-scatter` with asynchronous, point-to-point primitives (`gather`/`scatter-accumulate`), effectively reframing FSDP as a decentralized PS. This relaxes synchronization to the minibatch level, decouples device progress, and enables more effective load balancing.

## Strengths
- **Novel and well-motivated synthesis**: The paper presents a clever and practical integration of the straggler tolerance from classic PS architectures into the memory-efficient, sharded FSDP paradigm. The core idea of decomposing collectives into on-demand point-to-point operations is well-explained and represents a fresh perspective on a widespread problem in modern LLM training.
- **Comprehensive and rigorous evaluation**: The evaluation is extensive, covering multiple model scales (1.5B to 32B), key post-training tasks (SFT and RL on diverse datasets), and system configurations. The parametric study cleanly isolates factors affecting ODC's benefit, and the analysis of "bubble rate" (idle time) directly links performance gains to the core problem. Throughput improvements of up to 36% are demonstrated.
- **Strong implementation and open-source contribution**: The implementation leverages low-level RDMA capabilities (CUDA IPC, NVSHMEM via Triton-Distributed) for non-blocking communication, a non-trivial engineering feat. The code is open-sourced, and correctness is verified through convergence checks.

## Weaknesses
- **Significant inter-node communication overhead**: ODC's point-to-point communication pattern forgoes the hierarchical optimizations in libraries like NCCL, leading to significantly lower inter-node bandwidth (Figure 11). While overlapping computation and hybrid sharding are proposed as mitigations, this remains a core limitation that can curtail ODC's performance and scalability in multi-node settings, especially for workloads with shorter sequences. The evaluation's focus on long-sequence tasks (which hide communication) partially sidesteps this issue.
- **Conflated benefits of communication change and improved load balancing**: The paper convincingly shows that ODC enables a simpler, minibatch-level load balancing strategy (LB-Mini). However, the experiments do not fully disentangle how much of the speedup stems from the new communication paradigm versus the improved packing algorithm. An ablation study separating these two components would strengthen the causal claim.
- **Gains are limited by existing framework constraints in RL**: The speedups for Reinforcement Learning (up to ~10%) are notably smaller than for SFT. The authors correctly attribute this partly to implementation constraints in the verl framework, which prevents ODC from using its more flexible load balancer (LB-Mini). This indicates that real-world integration hurdles and rigid framework assumptions can curtail ODC's benefits in some settings.

## Nice-to-Haves
- A more detailed breakdown of the end-to-end time trade-off, quantifying the reduction in idle time ("bubble") against the added point-to-point communication latency, especially in multi-node scenarios.
- A discussion or simple experiment comparing ODC's approach to other classic techniques for mitigating synchronization overhead under imbalance (e.g., bounded-staleness asynchronous SGD), to better contextualize its novelty within the broader design space.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: Lack of comparison to a modern Parameter Server baseline** (from Spark Finder). This is scope creep. The paper's contribution is specifically *adapting* PS principles into the FSDP framework, not building a standalone PS. The comparison against the dominant paradigm (collective FSDP) is appropriate.
- **Weakness: Need for ablation on RDMA dependency** (from Spark Finder). The paper's contribution includes the specific implementation using RDMA for non-intrusive communication. Demanding an ablation with a different transport mechanism is not required to evaluate the core concept.
- **Weakness: Reproducibility/portability concerns due to RDMA/Triton** (from Harsh Critic). The implementation is detailed in Appendix B and open-sourced. The use of advanced interconnects is common in high-performance ML systems research. The paper discusses hybrid sharding as a mitigation for environments without optimal RDMA.
- **Weakness: Absence of formal proof for gradient consistency** (from Harsh Critic). For a systems paper introducing a semantically equivalent communication scheme, the empirical verification of convergence (Appendix F) is a reasonable and standard demonstration of correctness.
- **Weakness: Writing and formatting nitpicks** (from multiple reviewers). These are minor and do not affect the assessment of technical contribution.

## Novel Insights
The paper's core novel insight is that the classic Parameter Server architecture, often considered outdated for homogeneous clusters, is inherently better suited than collective communication for the imbalanced workloads that are fundamental to LLM post-training. By reframing FSDP's sharded state as a decentralized PS and replacing synchronous collectives with on-demand point-to-point transfers, the work successfully transplants the PS's straggler tolerance into the modern, memory-efficient sharding paradigm. This challenges the prevailing assumption that collectives are unconditionally superior and provides a principled path to recover significant computational efficiency lost to synchronization barriers.

## Suggestions
- To strengthen the claim that ODC's benefits are primarily due to the communication scheme, consider adding an ablation where both ODC and the collective baseline use the *same* advanced packing algorithm (LB-Micro), isolating the performance difference attributable solely to the replacement of collectives with point-to-point operations.

---

## D5PJX02Jki

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.7/10)
- Match: N/A

### Final Review

## Summary
This paper proposes RoPE++, an extension of Rotary Position Embeddings (RoPE) that reincorporates the imaginary component of the complex-valued attention computation, which is standardly discarded. The method introduces two configurations: RoPE++EH (equal attention heads, halved KV cache) and RoPE++EC (equal cache size, doubled attention heads). Theoretical analysis suggests the imaginary component better captures long-range dependencies, and empirical results across 376M, 776M, and 1.5B parameter models show consistent improvements on short- and long-context benchmarks over standard RoPE and other position encoding baselines.

## Strengths
- **Novel and Well-Motivated Core Idea**: The paper identifies a previously overlooked aspect of RoPE—the discarded imaginary component—and provides a principled, theoretically grounded argument for its utility via analysis of characteristic curves (sine vs. cosine integrals). The insight that this corresponds to a simple rotation of queries is elegant.
- **Comprehensive Empirical Evaluation**: The authors conduct extensive pre-training experiments at three model scales (up to 1.5B parameters) and evaluate on a wide suite of standard short- and long-context benchmarks. Results consistently show RoPE++ variants outperform RoPE and other PE methods, with gains more pronounced in long-context settings.
- **Practical Efficiency Benefits**: The RoPE++EH configuration offers a tangible efficiency gain, achieving comparable or superior performance to vanilla RoPE while halving the KV cache size and QKV parameters, leading to measurable reductions in memory cost and improvements in decoding throughput, as validated in Figure 4 and Table 11.

## Weaknesses
- **Limited Mechanistic Analysis of "Why It Works"**: The theoretical analysis focuses on expected behavior, and the empirical support (noise perturbation experiment, example attention patterns) is a good start. However, a deeper, quantitative dissection of how imaginary attention heads function *in practice* within trained models is missing. For instance, a statistical analysis of average attention distances per head type across layers and tasks would solidify the claim about long-context capture.
- **Model Scale for Definitive Scaling Claims**: While experiments at 376M-1.5B are valuable and the authors acknowledge resource limits, the current LLM research landscape often expects validation at larger scales (e.g., 7B+) to make robust, generalizable claims about architectural improvements. The positive trends are promising but not fully conclusive for state-of-the-art model sizes.

## Nice-to-Haves
- Include a more detailed quantitative analysis of attention distance distributions for real vs. imaginary heads to statistically validate the claimed functional difference.
- Provide a wall-clock time comparison during long-context inference for RoPE++EC vs. RoPE to better characterize the compute vs. memory trade-off.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: Statistical significance not reported** - Single-run evaluation is standard for large-scale LLM benchmarks; demanding confidence intervals imposes an arbitrary rigor requirement not typical in the field.
- **Weakness: Lack of realistic long-context task evaluation** - The paper uses standard synthetic long-context benchmarks (RULER, BABILong). Requesting additional task types (e.g., summarization) is scope creep for an architectural contribution focused on position encoding.
- **Weakness: Theoretical assumptions about i.i.d. queries/keys are unrealistic** - The analysis using expectations over random vectors is a standard and accepted theoretical tool for analyzing RoPE's properties; its purpose is to provide intuition, not to model a trained transformer exactly.
- **Weakness: Parameter sharing necessity is not fully justified** - The paper clearly states (Section 3.3) that allocating separate parameters would collapse to standard RoPE because rotating the query by π/2 in imaginary attention yields real attention, making independent heads redundant under the shared rotation framework. This is a reasonable architectural constraint.
- **Weakness: Extrapolation claims are overstated** - The paper does not overstate; it explicitly notes in Section 3.4 and shows in Figure 6 that RoPE++ does not extend the stable context window but slows the perplexity rise afterward, and discusses this as a limitation in Appendix D.
- **Weakness: Inconsistent 1.5B results are not discussed** - The paper does highlight the best results in Table 6 and discusses scaling in Appendix C.1. The performance is not "mixed" in a way that undermines the core claim; RoPE++ variants achieve the best average scores on key metrics.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct a deeper mechanistic analysis, such as computing the average attention distance per head type (real vs. imaginary) across all layers and heads on held-out long sequences to quantitatively validate the claim about long-range focus.
- If computationally feasible, include a pre-training experiment at a larger scale (e.g., 7B parameters) to strengthen the scaling claim and impact, even if limited to a smaller number of tokens.

---

## XIAta0WOJ6

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces F²SA-p, a class of fully first-order methods for stochastic bilevel optimization that generalizes prior work by using p-th order finite differences to approximate the hyper-gradient. For problems with p-th order smoothness in the lower-level variable, it achieves an improved SFO complexity of Õ(pϵ^{-4-2/p}), and establishes an Ω(ϵ⁻⁴) lower bound, showing near-optimality when p is large.

## Strengths
- **Novel algorithmic insight**: The paper provides a fresh interpretation of existing first-order bilevel methods as forward-difference approximations, naturally motivating extensions to higher-order finite differences. This elegant perspective leads to a generalizable family of algorithms (F²SA-p) and is clearly presented in Section 3.1 and Lemma 3.1.
- **Strong theoretical contributions**: The authors prove improved upper bounds that beat prior Õ(ϵ⁻⁶) complexity for first-order smooth problems, and complement this with an Ω(ϵ⁻⁴) lower bound via a reduction to single-level optimization, demonstrating near-optimality in ϵ for large p. These results are formally stated in Theorem 3.1 and Theorem 4.1.

## Weaknesses
- **High-order smoothness assumption limits applicability**: The improved rates require Assumption 2.5 (p-th order smoothness in the lower-level variable y), which may not hold in many practical bilevel problems, e.g., those involving non-smooth activations like ReLU. While justified with examples like logistic regression, this restricts the direct practical relevance of the theoretical acceleration.
- **Insufficient empirical validation of theoretical scaling**: Experiments are conducted on a smooth logistic regression problem but lack systematic verification of how complexity scales with p or direct comparison to Hessian-vector-product methods under the same smoothness assumptions. Without this, the core claim of faster rates for higher p is not fully empirically substantiated (Figure 1 shows performance but no ablation on p or oracle counts).
- **Loose condition number dependence**: The upper bound scales as κ^{9+2/p}, which is large and may hinder efficiency for ill-conditioned problems. Although concurrent works improving this are cited, the paper’s own dependence remains loose, and the gap is acknowledged but not resolved (Table 1 and open problems).

## Nice-to-Haves
- More extensive experiments on synthetic problems to directly verify the O(ϵ^{-4-2/p}) scaling with p and the effect of the finite-difference parameter ν.
- Discussion on the per-iteration cost of F²SA-p (which requires solving p lower-level subproblems) compared to F²SA in wall-clock time.
- Inclusion of error bars or multiple runs in experiments to account for stochastic variability.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- Formatting nitpicks about reference placeholders (e.g., mention of "ICLR" in citations).
- Demands for experiments on non-smooth problems (e.g., ReLU networks) or non-convex lower-level problems, which are outside the paper's stated scope of high-order smooth bilevel optimization.
- Requests for deep derivation of variance and bias of the hyper-gradient estimator, which goes beyond standard complexity analysis in this literature.
- Criticisms about missing broader impact statements, as these are not required for technical evaluation at ICLR.

## Novel Insights
The paper provides a novel perspective by linking bilevel optimization to finite difference approximations, which not only unifies prior first-order methods but also naturally leads to accelerated algorithms for higher-order smooth problems. The insight that smoothness only in the lower-level variable y (not jointly in x and y) suffices for acceleration is non-trivial and contrasts with existing joint smoothness assumptions in the literature.

## Suggestions
- In the experiments, include a controlled synthetic study to plot gradient norm vs. SFO calls for different p values, directly validating the theoretical complexity improvement.
- Clarify in the main text or caption for Appendix F that the MLP experiments are exploratory and outside the theoretical assumptions, to avoid misleading readers about the method's applicability to non-smooth settings.

---

## Mz98kwANpF

- GT: Reject (avg 4.5)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper challenges the prevailing multi-component paradigm in multi-task LoRA, which isolates task-specific knowledge via complex architectures like multi-adapters or multi-heads with routers. It first shows that a simplified multi-head LoRA (M-LoRA) with high head similarity outperforms these complex variants, and that simply increasing the rank of a standard single-adapter LoRA achieves competitive performance. This leads to a new hypothesis: learning task-shared representations is a highly effective path for multi-task adaptation. To validate this, the authors propose Align-LoRA, which adds an explicit alignment loss (KL divergence or MMD) to the shared down-projection space, achieving superior performance with zero inference overhead.

## Strengths
- **Effectively challenges a dominant paradigm with clear evidence.** The paper provides compelling empirical results (Tables 1-3) showing that architectural complexity for task isolation is not necessary for strong multi-task performance. The findings that M-LoRA (high similarity) wins and that high-rank single LoRA matches multi-component designs directly question a major research trend.
- **Comprehensive and practical experimental evaluation.** Experiments span multiple model families (Qwen2.5, LLaMA), scales (3B to 14B), and benchmarks (in-domain tasks, BBH for generalization). The proposed Align-LoRA consistently outperforms strong LoRA variants (Tables 4,5) while using fewer parameters and enabling weight merging for zero inference latency—a key practical advantage.
- **Well-motivated method with a clear hypothesis.** The alignment loss is a direct, principled instantiation of the task-shared representation hypothesis, applied to the down-projection matrix where prior work suggests task-general features reside. The inclusion of both KL and MMD variants demonstrates the generality of the alignment principle.

## Weaknesses
- **The core design choice—aligning the down-projection matrix A—is not sufficiently justified or ablated.** The paper motivates this by citing prior observations that A tends to be task-general, but provides no direct evidence (e.g., probing tasks, similarity analysis) that A in their method indeed captures shared features or that aligning A is optimal compared to other layers. An ablation comparing alignment applied to A vs. B or other representations is missing.
- **Limited analysis of the method's limits under high task heterogeneity or potential negative transfer.** While Appendix H.2 shows results on a diverse 10-task set, there is no systematic study varying task similarity or investigating scenarios where forcing alignment might degrade performance due to conflicting tasks. The risk of "over-alignment" is mentioned but not empirically explored.
- **Theoretical contribution is not tightly integrated with the method.** The derived generalization bound (Section 5.3) is a standard MTL bound with an added alignment term. It provides post-hoc justification but does not guide design (e.g., how λ relates to the bound) and its assumptions (e.g., Gaussian distributions) are not validated. The connection between minimizing the practical alignment loss and tightening the bound is not explicitly shown.

## Nice-to-Haves
- Comparison with other parameter-efficient MTL approaches (e.g., multi-task prompt tuning, adapter-based MTL) to better situate Align-LoRA within the broader landscape.
- Deeper analysis of training dynamics (e.g., gradient conflicts, representation convergence over time) to explain why M-LoRA's high similarity leads to better performance.
- Exploration of a principled hybrid approach that explicitly balances shared and task-specific knowledge, building on the brief experiment in Appendix I.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness: "Marginal performance gains in some settings."** The gains, while sometimes modest in absolute points (e.g., ~0.5-1.0 on BBH), are consistent across multiple models and benchmarks. In the context of established baselines and the field's standards, these improvements are meaningful and the paper does not overclaim.
- **Weakness: "Lack of statistical significance testing / multiple runs."** For large-scale LLM fine-tuning benchmarks, single-run evaluation is common practice due to computational cost. The paper follows the standard in its comparisons (e.g., same as HydraLoRA, R-LoRA). Demanding multiple runs imposes a rigor requirement not typically expected in this area.
- **Weakness: "Limited novelty of core alignment idea."** While representation alignment is known in general MTL/domain adaptation, its novel application within the LoRA/PEFT framework for LLMs, specifically targeting the low-rank down-projection space, constitutes a clear and timely contribution.
- **Weakness: "Missing comparison to stronger/more recent baselines (e.g., MTLLoRA, MALoRA)."** The paper compares against prominent and representative methods (HydraLoRA, R-LoRA, LoRA MoE). It is not required to include every recent variant; the selected baselines adequately represent the multi-adapter and multi-head paradigms under critique.
- **Weakness: "Should explore scaling multi-component variants for a fairer comparison."** The paper's core claim is that architectural complexity *itself* may be unnecessary. Fair comparison is achieved by matching total parameter budgets (e.g., high-rank LoRA vs. multi-head). Asking to scale multi-component variants changes their parameter count and moves the goalposts.

## Novel Insights
The paper's most significant insight is a conceptual shift in multi-task PEFT: away from architecturally isolating task-specific components (the prevailing trend) and towards explicitly learning task-shared representations within a unified, simple adapter. This is substantiated by the paradoxical finding that high head similarity correlates with better multi-task performance, and that a single high-rank adapter suffices. The proposed Align-LoRA operationalizes this insight via a direct alignment loss, demonstrating that a simpler, mergeable method can outperform complex, latency-inducing alternatives. This challenges the field to reconsider the necessity of structural complexity for multi-task adaptation.

## Suggestions
- Conduct an ablation study to justify aligning the down-projection matrix **A**. Compare the effects of applying the alignment loss to **A** vs. **B** or other intermediate representations, and provide quantitative evidence (e.g., probing task performance) that **A** indeed becomes more task-general.
- Add a controlled analysis on task similarity. Systematically vary the relatedness of tasks in the training mixture (e.g., using curated task clusters) and measure Align-LoRA's performance relative to baselines, providing clearer boundaries for when the alignment hypothesis holds strongest.
- Tighten the connection between the method and theory. Empirically estimate the distribution discrepancy term ∆ (e.g., using the KL/MMD metric) during training with and without alignment, showing concretely how Align-LoRA reduces it and whether this correlates with improved generalization.

---

## PFhrOUJZ5o

- GT: Reject (avg 5.0)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces LAION-Comp, a large-scale dataset of 540K+ aesthetic images annotated with structured scene graphs (objects, attributes, relations) via GPT-4o and partial human verification. To advance compositional image generation, the authors propose CompSGen Bench, a new benchmark, and develop baseline models by augmenting diffusion and flow-matching backbones (SDXL, SD3.5, FLUX) with a GNN-based scene graph encoder. Experiments show models trained on LAION-Comp outperform prompt-only and existing scene-graph-to-image models on compositional metrics. The structured conditioning also enables a novel, training-free framework for fine-grained image editing.

## Strengths
- **High-quality, large-scale structured dataset:** LAION-Comp provides a foundational resource of 540K+ scene graph annotations with verified high accuracy (objects: 98.8%, attributes: 97.5%, relations: 95.7%). It contains richer semantics and a greater proportion of non-spatial relations compared to prior datasets like Visual Genome, directly addressing a core data bottleneck for compositional generation.
- **Comprehensive empirical validation:** The paper establishes a dedicated benchmark (CompSGen Bench) and conducts extensive experiments across multiple backbones and datasets. Results consistently show that models trained on LAION-Comp achieve superior performance on compositional accuracy metrics (SG-IoU, Entity-IoU, Relation-IoU) while maintaining image quality (FID), convincingly validating the dataset's utility.
- **Effective extension to fine-grained editing:** The structured annotations naturally enable a training-free, scene-graph-based image editing framework using RF inversion. Quantitative and qualitative results demonstrate significant improvements over strong text-based and SG-based editing baselines, showcasing the practical controllability unlocked by the dataset.

## Weaknesses
- **Potential evaluation bias:** Core evaluation metrics (SG-IoU, etc.) are computed by extracting scene graphs from generated images using GPT-4o, the same model used for dataset annotation. This creates a potential circularity where the evaluator may favor its own annotation style, threatening the objectivity of the reported gains. The paper does not validate these metrics with an independent extraction method or human evaluation.
- **Lack of explicit spatial control:** Scene graphs capture semantic relations but do not encode precise spatial information (e.g., bounding boxes, relative positions). This limits fine-grained control over object layout, which is often critical for complex scene generation. The paper acknowledges this but does not integrate spatial conditioning, leaving a gap for precise compositional control.
- **Incomplete ablation on the GNN encoder's role:** The gains are attributed to the structured data, but the paper introduces a new GNN-based scene graph encoder. A critical baseline is missing: feeding a linearized version of the scene graph (as plain text) into the model's original text encoder. Without this ablation, it is unclear how much of the improvement stems from the novel encoder architecture versus the structured annotation format itself.

## Nice-to-Haves
- **Broader comparison with state-of-the-art compositional T2I methods:** While the paper compares to several SG2IM methods, inclusion of recent advanced T2I models specifically designed for composition (e.g., MIGC, 3DIS) in the main experiments would better contextualize the absolute performance level.
- **Analysis of error propagation from annotations:** A deeper analysis correlating the identified annotation error types (e.g., color attributes, spatial relations) with specific failure modes in the generated images would strengthen the understanding of the dataset's limitations and their impact on model training.
- **Human evaluation for compositional correctness:** For a paper centering on compositional generation, supplementing automated metrics with a large-scale human evaluation on complex scenes would provide stronger, bias-free evidence for the claimed superiority.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Statistical Significance & Reporting" (Harsh Critic):** The reviewer requests confidence intervals for metrics on a 20K+ sample benchmark. Single-run evaluation on large-scale benchmarks is standard in the field; demanding statistical tests is an arbitrary rigor requirement not commonly expected.
- **Weakness: "Limited vocabulary diversity" (Neutral Reviewer):** While the vocabulary is smaller than LAION-Aesthetics, the paper explicitly scopes its contribution to structured annotations for compositional semantics, not open-vocabulary generation. Criticizing the absence of long-tail concepts is scope creep.
- **Weakness: "Comparison with the strongest text-only models (e.g., DALL-E 3)" (Spark Finder):** The paper's claim is about the value of structured data within a reproducible research framework. Requiring comparison with closed, proprietary models like DALL-E 3 or Imagen 3 is outside the paper's scope and feasibility for a research contribution.
- **Weakness: "Evaluation of the editing framework against strong baselines (e.g., Prompt-to-Prompt)" (Spark Finder):** The editing evaluation reasonably compares against relevant, recent SG-based (SGEdit) and instruction-based (InstructPix2Pix) editing methods. Demanding comparisons against a large zoo of generic editing methods is not required to establish the utility of the proposed SG-based approach.

## Novel Insights
The paper's core novel insight is the empirical demonstration that scaling up high-quality, structured annotations is a critical and previously under-addressed factor for advancing compositional image generation. It shows that existing model architectures, when provided with a large-scale scene graph dataset, can achieve significant improvements in rendering complex object relations, moving beyond the limitations of unstructured text prompts. Furthermore, it reveals that such structural conditioning inherently provides a powerful interface for intuitive, object-level image editing, bridging generation and editing within a unified structured representation.

## Suggestions
- To mitigate potential evaluation bias, run a subset of evaluations using an alternative, independently trained scene graph generator (or human annotators) to compute the IoU metrics and verify the robustness of the conclusions.
- In future work, extend the scene graph representation to optionally include spatial primitives (e.g., bounding boxes) or integrate with a layout predictor to bridge the gap between semantic and spatial control.
- Include an ablation experiment where the scene graph is linearized into a descriptive sentence and fed into the base model's text encoder, to disentangle the contribution of the data structure from the specialized GNN encoder.

---

## j3htU5i01r

- GT: Reject (avg 4.0)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a framework for compositional meta-learning by learning a probabilistic generative model of tasks. The model separates within-module dynamics (via module RNNs) from between-module sequencing (via a gating RNN). After training, new tasks are solved through probabilistic inference (particle filtering) without parameter updates, enabling rapid one-shot learning and handling of sparse feedback. The approach is validated on synthetic rule-learning and motor-skill tasks where it recovers ground-truth components and generalizes to longer sequences.

## Strengths
- **Novel integration of compositional structure with probabilistic inference:** The paper's core contribution is a principled formulation that replaces gradient-based adaptation on new tasks with inference over a learned generative model. This combines the expressivity of RNNs with the data efficiency of Bayesian inference, enabling parameter-free task solving—a distinct advance over standard meta-learning.
- **Clear and controlled empirical validation:** The experiments on synthetic rule and motor tasks provide direct evidence that the model can recover ground-truth modules and transition statistics (Figures 2, 4). Ablations (Figure 3) convincingly demonstrate the necessity of both the gating network and the inference procedure, particularly for sparse-feedback generalization.

## Weaknesses
- **Training instability and sensitivity:** The paper notes that training is prone to instability and local minima, requiring careful weight initialization (small `winit`, Appendix A.1). This sensitivity is a practical limitation that is not thoroughly analyzed; robustness to hyperparameters (e.g., learning rate, particle count) is not quantified.
- **Fixed, predefined module count:** The number of modules is set a priori and cannot be inferred from data. While mismatch experiments (Fig. A1) show some robustness, the inability to dynamically grow or prune the module library limits flexibility for open-ended task distributions.
- **Computational cost of particle filtering is unexamined:** Inference requires running a particle filter with many particles (250 used), each evaluating the RNNs. The computational and memory costs, as well as trade-offs with particle count, are not discussed, leaving practicality unclear.
- **Lack of direct comparison to the closest prior works:** The discussion cites related methods (Alet et al., 2019; Hummos et al., 2024) but does not provide quantitative comparisons on the same tasks. This omission makes it difficult to assess the specific improvement offered by the proposed inference-based sequencing over, e.g., search-based or embedding-optimization approaches.

## Nice-to-Haves
- A more systematic analysis of sparse-feedback performance (e.g., varying sparsity levels) would strengthen the claimed robustness.
- An exploration of more complex transition structures (beyond simple duration rules) could better demonstrate the gating RNN's capacity to learn "grammars."
- Reporting inference time and scaling with sequence length or modules would help assess practical utility.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Demand for theoretical grounding (identifiability proofs):** This is an empirical systems paper; theoretical guarantees are not standard or expected in this context.
- **Criticism that the abstract overclaims generality:** The abstract appropriately states the contribution, and the Discussion explicitly notes the "proof-of-principle" nature and synthetic tasks.
- **Request for comparisons to in-context learning (transformers) or standard meta-learning benchmarks (Mini-ImageNet, Meta-World):** The paper's contribution is a novel framework demonstrated on controlled synthetic tasks; requiring immediate validation on complex benchmarks is scope creep for a proof-of-concept.
- **Nitpicks about writing clarity (e.g., particle filter description being dense):** The explanation is sufficiently detailed, and the appendix provides further implementation notes.
- **Suggestion that error bars are missing in some figures:** The paper shows multiple seeds where appropriate (e.g., Fig. 2a); other figures show representative examples, which is acceptable for illustrative plots.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add a quantitative comparison to the most closely related methods (Alet et al., 2019 and Hummos et al., 2024) on the same synthetic tasks to clearly demonstrate the advantages of probabilistic inference over learned transition statistics.
- Include a brief analysis of computational cost (e.g., inference time vs. particle count, memory usage) to address scalability concerns.
- Expand the discussion of failure modes, using the data-model mismatch analysis (Fig. A1e) as a starting point, to better characterize the method's limitations.

---

## FlcMckO6x5

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper establishes foundational theoretical results for separable neural networks (SepNNs), which factorize multivariate functions into linear combinations of univariate networks. It proves a universal approximation theorem for CP, TT, and Tucker SepNNs, derives their neural tangent kernel (NTK) regimes, and proposes an efficient separable preconditioned gradient descent (SepPGD) method that reduces preconditioning complexity to O(nD) for n^D grid samples. Experiments on kernel ridge regression, implicit neural representations, and physics-informed neural networks validate the theoretical insights and demonstrate SepPGD's effectiveness in accelerating convergence.

## Strengths
- **First universal approximation theorem for multivariate SepNNs:** The paper rigorously proves that CP, TT, and Tucker SepNNs can approximate any continuous multivariate function, extending prior bivariate results and providing a solid theoretical foundation for these architectures.
- **Novel NTK characterization under infinite and finite rank:** The analysis shows that the NTK of a CP SepNN converges to a deterministic kernel under infinite width and rank, but to a random kernel under fixed rank, offering new insights into the training dynamics and spectral bias of SepNNs.
- **Efficient separable preconditioning algorithm:** SepPGD leverages the separable structure to precondition gradients with O(nD) complexity per iteration for grid data, a significant improvement over the O(n^D) cost of standard NTK preconditioning. The equivalence to classical NTK preconditioning is formally established for the bivariate case (Lemma 2).

## Weaknesses
- **Lack of approximation rates:** The universal approximation theorem is existential and does not provide explicit error bounds in terms of rank or width, which limits practical guidance for architecture selection.
- **NTK analysis is restricted in scope:** The derived NTK results are explicitly for CP SepNNs with two-layer factor MLPs; extensions to TT/Tucker architectures and deeper networks are only briefly mentioned (Remark 1) without detailed derivations, leaving the full generality of the claims unsubstantiated.
- **Incomplete equivalence proof for SepPGD:** Lemma 2 proves the equivalence between SepPGD and classical NTK preconditioning only for the bivariate case (D=2). While the paper claims the result extends to D>2, no proof or rigorous sketch is provided, creating a gap in the theoretical justification.
- **Limited experimental comparisons with state-of-the-art:** Due to memory constraints, comparisons with the modified spectrum kernel (MSK) method are run in mini-batch mode for larger-scale tasks, while SepPGD operates in full-batch mode. This discrepancy may bias wall-clock time comparisons and leaves the efficiency advantage relative to the most relevant baseline incompletely validated.
- **No generalization guarantees for practical regimes:** The NTK analysis focuses on asymptotic regimes (infinite width/rank), but the paper does not provide generalization bounds or convergence guarantees for the fixed-rank, finite-width settings commonly used in practice.

## Nice-to-Haves
- Deriving explicit approximation error rates for SepNNs in terms of rank and width to inform architecture selection.
- Extending the NTK analysis to TT and Tucker SepNNs with detailed derivations in the appendix.
- Including wall-clock time comparisons between SepPGD and mini-batch versions of other preconditioners to better substantiate efficiency claims.
- Visualizing the evolution of the NTK eigenvalue distribution with and without SepPGD to directly illustrate spectral bias alleviation.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness about the matrix product in Eq. (8) being not rigorously justified:** The paper provides a complexity analysis in Remark 4 that reasonably supports the efficiency claim; the criticism is overly nitpicky.
- **Weakness about SepPGD not being applied to PDE residual loss in PINNs:** The paper explicitly states this as a practical compromise and suggests it as future work; it is not a core flaw of the current contribution.
- **Weakness about narrow applicability to grid data:** The paper explicitly focuses on grid-structured applications (INRs, PINNs) where the efficiency gain is most relevant; this is a scope choice, not a weakness.
- **Strength about the paper being well-written and the topic important:** These are generic and apply to many papers; we list only specific strengths.

## Novel Insights
Beyond the paper's own contributions, the synthesis of reviews highlights that the random NTK regime under fixed rank is a particularly novel observation with implications for understanding the training dynamics of low-rank SepNNs. Additionally, the connection between SepPGD and Kronecker product preconditioning provides a new perspective on how to exploit separability for efficient optimization. However, no fundamentally novel insight beyond the paper's stated contributions emerges from the reviews.

## Suggestions
- Provide a proof sketch or formal extension of Lemma 2 to the multivariate case (D>2) to solidify the equivalence between SepPGD and NTK preconditioning.
- Conduct a more equitable comparison with mini-batch preconditioning baselines by either implementing a mini-batch version of SepPGD or clearly discussing the limitations of the current comparison in the main text.
- Include a brief theoretical or empirical analysis quantifying how SepPGD improves the condition number of the NTK matrix, strengthening the claim of spectral bias alleviation.
- Add a discussion on the practical selection of rank and width based on the approximation theorem and NTK analysis, even if exact rates are not available.

---

## wSbVv6xaRr

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces FedMPDD, a federated learning algorithm that compresses client gradients via multi-projected directional derivatives along random vectors. It reduces uplink communication from O(d) to O(m) (m ≪ d) while providing empirical protection against gradient inversion attacks through the rank deficiency of the projection mechanism. The method achieves a convergence rate of O(1/√K), matching FedSGD, with m growing logarithmically in dimension d.

## Strengths
- **Novel joint design for efficiency and privacy.** The core mechanism of encoding gradients via multiple directional derivatives is a fresh approach that simultaneously tackles communication overhead (transmitting only m scalars) and provides privacy via the projection's nullspace, distinct from separate compression or additive-noise methods.
- **Rigorous theoretical analysis.** The paper provides a complete convergence analysis for both the single-projection (FedPDD) and multi-projection (FedMPDD) variants, establishing an O(1/√K) rate for FedMPDD under standard assumptions and a Johnson-Lindenstrauss condition on m. Privacy is analytically characterized via gradient reconstruction error (Lemma 1) and a lower bound on data recovery (Lemma 2).
- **Comprehensive empirical validation.** Experiments span multiple datasets (MNIST, FMNIST, CIFAR-10), models, and data distributions (IID/non-IID). Evaluation under both fixed communication budgets and fixed accuracy targets convincingly demonstrates FedMPDD's advantages in communication savings and empirical resistance to gradient inversion attacks (using SSIM and visual reconstructions) compared to baselines like FedSGD, QSGD, Top-k, and LDP.

## Weaknesses
- **Incorrect convergence rate stated in the abstract.** The abstract claims a convergence rate of O(1/K), but Theorem 2 correctly states O(1/√K). This is a significant error that misrepresents the theoretical guarantee and must be corrected.
- **Computational overhead on clients is not empirically quantified.** While Remark 1 discusses the O(dm) encoding cost and mentions projected-forward methods (Jacobian-vector products) as a potential optimization, the paper does not measure the actual client-side runtime or energy consumption compared to baselines. This is important for assessing practical suitability in resource-constrained settings, as the increased computation could offset communication savings.
- **Privacy claims are heuristic and lack formal comparison to state-of-the-art private FL.** The privacy analysis (Lemmas 1 & 2) establishes a reconstruction error bound, providing a computational security argument against gradient inversion attacks. However, the language ("inherent privacy," "privacy guarantee") risks overstatement, as the method does not provide a formal, composable guarantee like differential privacy. Furthermore, the comparison is primarily against simple LDP and non-private compression; benchmarking against strong private FL baselines (e.g., DP-FedAvg, DP with secure aggregation) would better contextualize the privacy-utility trade-off.
- **Experimental results lack statistical robustness.** The paper mentions using five random seeds but reports results as single numbers (e.g., test accuracy). Reporting means and standard deviations (or confidence intervals) across seeds is essential for ICLR to assess the significance of the reported advantages, especially in the main tables (1, 2, and Appendix tables).

## Nice-to-Haves
- Include wall-clock time measurements (computation + communication) to provide a holistic efficiency assessment.
- Extend evaluation to larger-scale models (e.g., ResNet) to better validate the logarithmic scaling claim for m.
- Provide a more detailed discussion of the multi-round privacy bound (Remark 2) in the main text, emphasizing that privacy degrades with more observations and discussing how evolving gradients affect this in practice.

## Removed Points 
*These points are flagged to be removed, treat them with caution*
- **Weakness about missing FedAvg comparison.** The paper's scope is a gradient compression and privacy mechanism built on the FedSGD framework. While comparing to FedAvg could be interesting, it is not a core requirement for evaluating the proposed encoding scheme.
- **Weakness about requiring formal DP guarantees.** The paper explicitly positions its privacy mechanism as an alternative to DP, analyzing protection against gradient inversion via reconstruction error. Demanding a formal DP proof is scope creep for this contribution.
- **Weakness about the Lipschitz constant in Lemma 2 weakening the bound.** The bound's dependence on the model-specific Lipschitz constant is a standard feature of such reconstruction error analyses; it does not invalidate the derived relationship between m and reconstruction error.
- **Nitpicks about formatting artifacts in the extracted PDF.** These are parser issues, not paper problems.
- **Suggestion to evaluate against membership inference attacks.** The paper focuses on gradient inversion attacks, which is appropriate for its stated privacy threat model.

## Novel Insights
The paper's key novel insight is that averaging multiple independent, low-rank (rank-1) gradient projections can overcome the dimension-dependent convergence penalty of a single projection while preserving the inherent privacy afforded by each projection's nullspace. This multi-projection mechanism creates a tunable trade-off: the number of projections m controls the bias-variance trade-off of the gradient estimator (affecting convergence speed and communication cost) and simultaneously governs the dimensionality of the remaining nullspace (affecting privacy against reconstruction). This unified perspective on the communication-privacy-accuracy trade-off via a single parameter m is a distinct and valuable contribution.

## Suggestions
- Correct the convergence rate in the abstract from O(1/K) to O(1/√K) to align with Theorem 2.
- Augment key experimental results (e.g., Tables 1, 2) with means and standard deviations across multiple runs to establish statistical significance.
- In the main text, more precisely frame the privacy property (e.g., "empirical protection against gradient inversion via obfuscation" or "computational privacy based on reconstruction hardness") to avoid potential misinterpretation as a formal DP guarantee. Expand the limitations section to explicitly note the heuristic nature of the guarantee and the multi-round erosion bound.
- Include a simple empirical evaluation of client-side encoding time (even for the naive O(dm) method) versus baseline gradient computation to ground the discussion of computational overhead.

---

## rBj2iVyrhh

- GT: Reject (avg 2.0)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
This paper identifies that alternating training methods for multimodal learning fail to prevent classifier bias toward faster-converging modalities, perpetuating modality imbalance. To address this, the authors propose Classifier-Constrained Alternating Training (CCAT), a two-stage framework that pre-trains an unbiased shared classifier with contribution-aware regularization, freezes it as a stable decision anchor, and employs modality-specific LoRA adapters during alternating training. A sample-level re-optimization mechanism further targets severely imbalanced instances.

## Strengths
- **Well-motivated and targeted problem analysis**: The paper clearly identifies a critical, underexplored flaw in existing alternating training methods—persistent classifier bias—supported by empirical tracking of modality contributions (Figure 1). The connection drawn to class imbalance provides a coherent conceptual lens for the approach.
- **Comprehensive and convincing empirical validation**: The method achieves consistent and substantial accuracy gains across three diverse benchmarks (e.g., +6.76% on Kinetic-Sound). Ablation studies (Table 2) clearly demonstrate the value of each component (classifier freezing, alternating training, secondary updates, LoRA), and feature visualization (Figure 5) provides supporting evidence for improved discriminability.
- **Practical and reproducible design**: The integration of a frozen classifier with lightweight LoRA adapters is a clever and implementable solution to the distribution mismatch problem. The training pipeline is clearly detailed (Algorithm 1), and hyperparameter searches (Table 3, Figure 4) are documented, aiding reproducibility.

## Weaknesses
- **Incomplete comparison with relevant state-of-the-art baselines**: The main results table (Table 1) omits direct comparisons with key recent methods explicitly mentioned in the text (MLA, MMPareto, LFM) and other sample-level imbalance methods (e.g., SMSL). This omission makes it difficult to conclusively assess the claimed superiority and situate the contribution within the current landscape.
- **Theoretical section is informal and overstated**: Section 3.1 presents a valuable intuitive analogy between class and modality imbalance but frames it as a "unified theoretical framework" and "proof." The gradient analysis is heuristic, introducing fusion coefficients (γ) that are not part of the standard gradient derivation, and it lacks the rigor (e.g., formal assumptions, bounds) expected for a theoretical contribution. This section should be reframed to avoid overclaiming.
- **Limited analysis of secondary update mechanism and unimodal trade-offs**: The paper does not analyze how many samples are selected for secondary updates, how their contribution scores evolve, or whether this step genuinely rectifies imbalance on those samples. Furthermore, ablation results (Table 2, Kinetic-Sound) show that the full CCAT can sometimes yield lower unimodal accuracy for a modality than an ablated variant, a trade-off that is not discussed.

## Nice-to-Haves
- **Computational cost analysis**: A discussion of the training time and parameter overhead introduced by the two-stage process and LoRA modules compared to standard end-to-end or alternating training baselines would be informative.
- **Experiments on larger-scale or trimodal datasets**: While the three benchmarks are appropriate, testing on a larger dataset (e.g., AudioSet) or a trimodal task would strengthen claims about generalizability and scalability, as noted in the future work.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "The mutual information estimator (Eq. 5) is unconventional... justification is lacking."** The paper cites prior work (Zhou et al., 2025b) for this metric. While its properties could be discussed further, its use is not a fundamental flaw.
- **Weakness: "Inference procedure... not specified."** The paper states: "These are fused at the decision level for final output." Common practice is averaging, and the exact method does not impact the core contribution.
- **Weakness: "Missing statistical significance tests."** While reporting variance is good practice, the paper follows the common norm in the field by reporting average accuracy over three seeds for these benchmarks. This is not a required standard for rejection.
- **Weakness: "Evaluated only on classification tasks."** The paper's scope is clearly modality imbalance for multimodal classification. Demanding evaluation on other tasks is scope creep.
- **Weakness/Strength: Generic statements about writing quality or topic importance.** These have been filtered out.

## Novel Insights
The core novel insight is the identification and mitigation of *classifier entrenchment bias* as a fundamental failure mode of modality-alternating training. While alternating updates decouple encoders, the classifier can become structurally biased toward early-dominant modalities, suppressing later learning from weaker ones. The paper's key innovation is treating this as analogous to decision boundary bias in class-imbalanced learning and applying a remedy—freezing a pre-regularized classifier as a stable anchor—within the multimodal setting. The integration of LoRA adapters to handle the unimodal/fused feature distribution mismatch while preserving this anchor is a clever and practical implementation of this insight.

## Suggestions
- **Include missing baselines in the main results table**: Add rows for MLA, MMPareto, LFM, and a recent sample-level method (e.g., SMSL or its variant) to Table 1 to provide a complete and fair comparison.
- **Reframe Section 3.1 as an analogy/motivation**: Revise the section title and text to present the gradient dynamics discussion as an insightful motivating analogy rather than a formal theoretical proof, to avoid overstatement.
- **Add analysis for the sample-level re-optimization**: Include a brief analysis tracking, for a subset of epochs, the number of samples selected for secondary updates and the change in their weak-modality contribution scores or accuracy, to validate the mechanism's operation.

---

## C6WWMryELL

- GT: Reject (avg 5.5)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper identifies and addresses the critical problem of output length volatility in long-form LLM generation. It introduces VOLTBench, a comprehensive benchmark for quantifying this instability; probes internal attention patterns linked to failure; and proposes SELB, a lightweight decoding strategy that significantly improves length accuracy and stability without retraining.

## Strengths
- **Systematic Benchmarking**: VOLTBench is a substantial, well-designed contribution. It systematically quantifies length volatility across diverse dimensions (structured/unstructured tasks, two languages, multiple complexities, and extreme scales up to 500 sections/100k words) using appropriate, clearly defined metrics (LSD, LVC, MLA, FAD, SCA, UCA). The use of fine-grained constraints for automated quality evaluation is particularly innovative.
- **Effective, Lightweight Mitigation**: SELB is a simple, training-free decoding intervention that demonstrates impressive empirical results. On the 100-section benchmark task, it improves the base model's mean output length by 148% and reduces length volatility (LVC) by 69% while maintaining high generation quality (e.g., 100% SCA, 86.7% UCA), convincingly outperforming specialized models like LongWriter.

## Weaknesses
- **Insufficient Validation of SELB's Generality**: The core mitigation results are demonstrated almost exclusively on Qwen2.5-7B. The paper's claim that SELB is a general, lightweight decoding strategy is significantly undermined by the lack of evidence showing consistent improvements across different model families (e.g., Llama, Mamba, or other architectures in the benchmark). This is a major limitation for a method proposed as a general solution.
- **Preliminary and Correlational Analysis**: The attention trace analysis, while insightful, is presented as a root-cause investigation but remains preliminary. It correlates attention patterns ("Collapse," "Instability") with failure but does not establish causality (e.g., via intervention experiments). Furthermore, the analysis is limited to two models on a single task, making the claim of identifying "common internal patterns" premature.
- **Incomplete Methodological Details and Fair Comparisons**: Critical implementation details for reproducibility are missing or relegated to the appendix. Most importantly, the method for identifying the set of title tokens \(V_{title}^{(p+1)}\) for structural enforcement is not specified. The comparison against other decoding baselines (Repetition Penalty, etc.) is potentially unfair, as their hyperparameters and tuning process are not described, weakening the claim that existing decoding methods are inadequate.

## Nice-to-Haves
- An analysis of the computational overhead of SELB (e.g., inference slowdown) would be valuable for assessing its practical utility as a "lightweight" method.
- A discussion or sensitivity analysis of SELB's key hyperparameters (e.g., the boosting strength \(\beta\), the banned token set) would provide practical guidance for adaptation.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Statistical Significance & Reporting (N=5 samples is insufficient)"** – Removed. The paper explicitly sets N=5 for volatility measurement, which is a practical and reasonable choice for a large-scale benchmark covering many models and tasks. Demanding confidence intervals or statistical tests for every metric across this scale is not standard practice for such empirical evaluations.
- **Weakness: "Overstated Novelty Claims (not the 'first' to study volatility)"** – Weakened to scope clarification. While related work like LIFEBench studies length adherence, this paper's integrated focus on *stability across multiple generations* (volatility) and its combination with mechanistic probing is a distinct and novel contribution. The claim is reasonable within its defined scope.
- **Weakness: "Lack of a Dedicated Limitations Section"** – Removed. The absence of a formal subsection labeled "Limitations" is a stylistic choice, not a substantive flaw. The review itself identifies key limitations, which are now included in the Weaknesses section above.
- **Weakness: "Formatting/Style Nitpicks and Appendix Reliance"** – Removed. Comments about OCR artifacts, dense appendices, or narrative flow are matters of presentation, not scientific merit. The core content is accessible and the appendix provides necessary, referenced detail.

## Novel Insights
The paper's primary novel insight is the identification of a specific, measurable failure mode in LLMs—multi-run length volatility—and the empirical demonstration that it is widespread and severe, even in models specialized for long-form generation. The proposed link between this external volatility and internal attention dynamics (collapse, instability) provides a plausible, data-grounded hypothesis for the phenomenon, moving beyond mere observation. The SELB method operationalizes this insight by using structural enforcement to counteract attention decay.

## Suggestions
- **Demonstrate SELB on Multiple Models**: To support the claim of generality, add a key experiment applying SELB (with appropriate hyperparameter tuning) to at least 2-3 other base models from the benchmark (e.g., Llama3.1-8B, Deepseek-V3) and report the results.
- **Integrate and Detail Free-Form Generalization**: The SELB-Hybrid method for free-form tasks is a significant extension but is currently buried in Appendix I. Move a concise description of its adaptation logic and key equations into the main Method section (6.4) to improve clarity and emphasize its importance.
- **Clarify Critical Implementation Details**: In the Method section, explicitly describe how the set of title tokens \(V_{title}^{(p+1)}\) is defined (e.g., is it prompted, hardcoded, or extracted from the instruction?) and provide an example of the banned token set \(V_{banned}\). This is essential for reproducibility.

---

## cZFgsLq8Gs

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
DeepScientist is an autonomous, LLM-based AI scientist system designed for goal-oriented, long-horizon scientific discovery. It formalizes discovery as a Bayesian Optimization problem with a persistent Findings Memory to balance exploration and exploitation. In a large-scale evaluation consuming ~20,000 GPU hours, the system generated novel methods that surpassed human state-of-the-art (SOTA) baselines on three frontier AI tasks: agent failure attribution, LLM inference acceleration, and AI text detection.

## Strengths
- **Architectural Innovation and Coherent Framework:** The three-stage iterative loop (hypothesize, implement, analyze) integrated with a persistent Findings Memory and a Bayesian acquisition function (UCB) provides a principled and concrete advance over prior "one-shot" AI scientist pipelines. It explicitly addresses the challenge of balancing focused exploration with knowledge reuse over long timelines.
- **Substantial Empirical Demonstration at Scale:** The system was deployed on three distinct, recent, and challenging SOTA tasks (ICML/ACL/ICLR 2024-25) over a month-long period, generating ~5,000 ideas and validating ~1,100, culminating in performance improvements (e.g., +183.7%, +1.9%, +7.9% AUROC). This scale of autonomous, goal-directed exploration on modern AI research problems is unprecedented.
- **Valuable Meta-Analysis and Honest Bottleneck Identification:** The analysis of the discovery trajectory (e.g., Fig. 4, 5) provides genuine insights for the field, quantifying the low success rate (~0.4% of ideas to papers), highlighting implementation errors as a major bottleneck (~60% of failures), and suggesting a near-linear scaling relationship between compute and discoveries (Fig. 6). This diagnostic work is a contribution in itself.

## Weaknesses
- **Lack of Statistical Rigor and Precision in Reporting Key Results:** The reported performance improvements lack measures of variance, confidence intervals, or statistical significance tests. For example, the 1.9% gain in inference speed (190.25 to 193.90 tokens/s) and the AUROC improvements are presented as single numbers without an assessment of noise or practical significance. Furthermore, the abstract's highlight of a 183.7% improvement (for the Algorithm-Generated setting) risks overshadowing the more modest 142.8% improvement on the Hand-Crafted set without clear immediate contextualization.
- **Insufficient Clarity on Autonomy vs. Human Supervision:** While the paper claims "fully autonomous scientific discovery," Section 4 and Appendix A reveal that "three human experts supervise the process to verify outputs and filter out hallucinations." The nature and extent of this intervention are not quantified (e.g., how many ideas were filtered/corrected, whether key insights originated from humans). This ambiguity significantly undermines the core claim of autonomy and makes it difficult to attribute the discoveries conclusively to the AI system.
- **Critical Ablation Studies and Component Analysis are Missing:** The paper does not demonstrate the necessity of its novel components. There is no ablation showing performance without the Findings Memory, without the Bayesian optimization loop (e.g., vs. random search or greedy selection based on the surrogate alone), or with different LLM backbones. Without these, the reported gains could be attributed primarily to massive parallel trial-and-error rather than the intelligent guidance of the proposed architecture.
- **Reproducibility is Severely Limited:** Essential implementation details are relegated to the appendix and remain high-level. The exact prompts for the surrogate and agent models, the schema of Findings Memory records, the retrieval model specifics, and the configuration of the MCP tools are not provided. The associated high cost (~$100k) and lack of a fully reproducible environment (e.g., containerized with all prompts) prevent the community from verifying or building upon this work.
- **Scaling Analysis is Preliminary and Decoupled from Core Claim:** The scaling experiment (Fig. 6) uses a simplified setup where limitations are pre-identified and assigned to parallel paths, which differs from the integrated, end-to-end discovery process evaluated in the main experiments. Its relevance to the primary claim of autonomous, progressive discovery is therefore indirect and requires further validation.

## Nice-to-Haves
- A direct, controlled efficiency comparison (e.g., ideas or discoveries per GPU hour) between DeepScientist and other contemporary AI scientist systems (e.g., AI Scientist-V2, Zochi) on a common task would help isolate the contribution of the proposed architecture from pure scale.
- A more detailed cost-benefit discussion or estimate comparing the system's resource use to a human research team's effort on a similar problem would contextualize the "compressed timeline" claim.
- Testing the system's core discovery mechanics on a broader set of tasks, including non-AI domains or standard algorithmic discovery benchmarks, would better indicate its generality, even if full-scale runs are prohibitively expensive.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism about misreported improvement percentages:** The paper does distinguish between the Handcraft (142.8%) and Algorithm-Generated (183.7%) datasets in Table 1 and Figure 3. The abstract could be more precise, but the data is present.
- **Demand for comparison against "more recent" AI text detection SOTAs:** The selected baseline (Binoculars, ICML 2024) is a strong, established zero-shot SOTA. The paper's claim is about surpassing a recent human SOTA, not necessarily the absolute latest.
- **Request for submission of generated papers to real conferences:** This is an unreasonable expectation for a research paper about the system itself. The simulated peer-review process is a reasonable evaluation proxy.
- **Suggestion to test on "biology or chemistry tasks" for generality:** The paper explicitly scopes its contribution to "modern, resource-intensive AI research problems." Demanding evaluation on wet-lab sciences is scope creep.
- **Criticism that discovered methods are merely "recombinations":** The generated papers (provided in appendices) describe non-trivial conceptual advances (e.g., causal scaffolding with A2P, wavelet-based non-stationarity analysis with TDT). The claim of "redesigning core methodologies" is supported, though the process is one of innovative recombination and deep optimization.

## Novel Insights
The paper provides a concrete, large-scale demonstration that the primary bottleneck in autonomous AI science is not hypothesis generation but efficient search and validation. The system's logs reveal an immense exploratory funnel (~5k ideas) with a very low success rate (~21 progress findings), dominated by implementation failures. This empirically validates the critical need for systems that intelligently select what to test (via surrogates and acquisition functions) and robustly execute experiments. The proposed Bayesian optimization framework with a Findings Memory is a direct architectural response to this insight, aiming to improve search efficiency over brute-force parallel trial-and-error. The observation of near-linear scaling with parallel resources further suggests that the knowledge-sharing mechanism (Findings Memory) can effectively amortize exploration costs across concurrent efforts.

## Suggestions
- Conduct and report critical ablation studies within the main paper: (1) Remove the Findings Memory to show its impact on avoiding redundant exploration, (2) Replace the Bayesian acquisition strategy with random selection to quantify the intelligence of the search, and (3) Test the system with a less capable/expensive LLM backbone to assess the sensitivity to foundation model quality.
- Add a dedicated subsection in the method or experiment section that transparently details the human supervision protocol. Quantify the interventions (e.g., "experts filtered X% of ideas for hallucination, corrected Y implementation errors") and provide clear examples to delineate the boundary between system autonomy and human oversight.
- Report key results with basic measures of variance (e.g., standard deviation over multiple independent evaluation runs of the final discovered method) or confidence intervals to establish statistical reliability, especially for the modest performance gains.

---

## XKLPlnfZzM

- GT: Reject (avg 3.5)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper introduces the Temporal Deaggregation Diffusion Model (TDDM), a hierarchical framework for generating large-scale human trajectories. TDDM factors generation into spatial occupancy priors (marginal distributions over geographic cells) and temporal dynamics, using coordinate canonicalization to enable a single model to generalize across regions. It establishes a multi-city benchmark and shows improved fidelity and coverage over strong baselines, with compelling zero-shot generalization to unseen city areas and entirely new cities.

## Strengths
- **Novel factorization**: Separating spatial priors from temporal dynamics is an elegant and well-motivated solution to limitations of prior work (e.g., sample-specific conditioning), enabling controllability and reducing memorization risk.
- **Comprehensive evaluation**: Rigorous benchmarking across three diverse cities (Beijing, Porto, San Francisco) with a suite of 10 metrics covering fidelity, coverage, proportionality, usefulness, and generalization. TDDM consistently outperforms strong GAN, VAE, and diffusion baselines.
- **Impressive generalization**: Demonstrates compelling zero-shot generalization, both intra-city (trained on 25% of a city) and city-to-city (trained on one city and applied to another), validating the benefits of spatial-temporal factorization and canonicalization.
- **Strong reproducibility**: Provides detailed pseudocode, architecture diagrams, hyperparameters, preprocessing steps, and commits to releasing runnable code, facilitating replication and future work.

## Weaknesses
- **Mathematical clarity**: The mathematical formulation in Section 3 (Equations 1–5) is garbled and must be rewritten with correct notation to clearly convey the mixture model and the definition of the spatial prior \(H\).
- **Inconsistent descriptions**: Canonicalization is described as mapping to \([-1,1]^D\) in the text but to \([0,1]^D\) in Algorithm 1 and Appendix C.3. This inconsistency must be resolved.
- **Architectural ambiguity**: The tokenization of the spatial prior \(H\) (a 64×64 grid) into 64 tokens is not clearly explained in the main text. The exact method (e.g., patching) should be specified for reproducibility.
- **Algorithmic error**: Algorithm 2, line 4 contains garbled text for calculating \(N_{rc}\) and must be corrected.
- **Generalization requirements**: Zero-shot transfer requires access to aggregate target data to compute the spatial prior \(H\). The paper should explicitly discuss this requirement and its implications (e.g., what if only a coarse prior is available?).
- **Missing baseline comparisons**: The claim of state‑of‑the‑art would be stronger with direct comparisons to modern conditional baselines like ControlTraj and COLA, even if reimplementation is needed.
- **Lack of memorization analysis**: The claim that spatial priors reduce memorization risk is not substantiated by experiments (e.g., nearest‑neighbor distance or membership inference tests).
- **Incomplete ablation**: The contribution of canonicalization is not isolated; an ablation training without canonicalization (using absolute coordinates) is needed to validate its role in generalization.
- **Limited failure analysis**: The increased Length error in city‑to‑city transfer indicates a weakness. A deeper analysis of what temporal dynamics fail to transfer (e.g., speed profiles, turn patterns) is missing.
- **Sample‑level realism**: The evaluation relies on aggregate metrics; an analysis of physical plausibility (e.g., acceleration constraints, adherence to road networks) at the trajectory level would strengthen the fidelity claim.

## Nice-to-Haves
- Reporting uncertainty estimates (e.g., standard deviations over multiple runs) for key distributional metrics.
- A deeper investigation into why Porto serves as a particularly good source city for generalization.
- Sensitivity analysis on the robustness of the method to noisy or sparse spatial priors.
- A simple two‑stage baseline to disentangle the contribution of the spatial prior from the diffusion model.
- Computational cost comparison with baseline methods.
- Visualizations of failure cases and side‑by‑side comparisons in held‑out regions for generalization.
- A case study demonstrating controllability by editing the spatial prior \(H\).

## Removed Points
*These points are flagged to be removed, treat them with caution.*  
- None. All points raised by the reviewers are substantive, though some have been weakened or moved to nice‑to‑haves.

## Novel Insights
The paper’s core insight—that trajectory generation can be factorized into spatial occupancy priors and temporal dynamics, and that canonicalization enables a single model to generalize across cities—is novel and impactful. The finding that training on Porto generalizes well to other cities suggests that certain datasets may act as “universal sources” for trajectory generation, which is an interesting observation for future research.

## Suggestions
- Revise Section 3 to correct the mathematical notation, resolve the canonicalization inconsistency, and clearly explain the tokenization of \(H\).
- Correct Algorithm 2, line 4.
- Add a discussion on the requirements for zero‑shot transfer (aggregate target data for \(H\)).
- Include an ablation study on canonicalization and a memorization test.
- Analyze the failure modes for Length error in city‑to‑city transfer.
- Consider adding a comparison with ControlTraj and COLA, even if via reimplementation, to solidify the state‑of‑the‑art claim.

---

## iaoAKDRAJQ

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.8/10)
- Match: N/A

### Final Review

## Summary
This paper provides a unified theoretical comparison between adaptive optimizers (e.g., Adam, Shampoo) and normalized steepest descent (NSD) methods. It extends the notion of adaptive smoothness to nonconvex settings, showing it governs the convergence of adaptive optimizers, and demonstrates that this stronger smoothness enables acceleration with Nesterov momentum in convex settings. Additionally, it introduces adaptive gradient variance and proves that NSD with momentum achieves dimension-free rates under this assumption, while dimension dependence is unavoidable under standard variance.

## Strengths
- **Unified nonconvex analysis**: Theorems 3.1 and 3.2 establish convergence rates for adaptive optimizers on nonconvex functions, governed by adaptive smoothness and matching the optimal \(\widetilde{O}(T^{-1/4})\) rate, extending prior convex results.
- **Acceleration under adaptive smoothness**: Theorem 4.3 shows that adaptive optimizers with Nesterov momentum achieve an accelerated \(\widetilde{O}(T^{-2})\) rate under adaptive smoothness, while standard \(\ell_\infty\) smoothness cannot exceed \(\Omega(T^{-1})\) (Guzmán & Nemirovski, 2015), confirming a concrete benefit of the stronger assumption.
- **Dimension-free rates via adaptive variance**: Theorem 4.5 proves that NSD with momentum attains a dimension-free rate under the introduced adaptive variance, complemented by a lower bound (Theorem 4.7) showing dimension dependence under standard variance, establishing a clear separation.
- **Key technical innovation**: Lemma 3.3 provides a novel matrix inequality that handles noncommutativity in general preconditioner sets, enabling the extension from diagonal to non-diagonal cases and serving as a central tool for the nonconvex analysis.

## Weaknesses
- **Practical relevance of stronger assumptions**: The paper does not thoroughly discuss when adaptive smoothness is significantly larger than standard smoothness (or when they are comparable) in practice, which affects the interpretation of the acceleration result. Similarly, adaptive variance is a stronger noise assumption; its plausibility in typical machine learning problems (e.g., deep neural networks) is not examined.
- **Limited scope of lower bound**: The lower bound for NSD under standard variance (Theorem 4.7) is established only for the \(\ell_\infty\) norm (i.e., SignGD). A more general lower bound for arbitrary norms would strengthen the claim that dimension-free rates require adaptive variance.
- **Unclear tightness of logarithmic factors**: Convergence bounds for general well-structured preconditioner sets include logarithmic factors in dimension (e.g., \(\log d\) in Theorem 3.2) that are absent in diagonal cases. The paper does not discuss whether these factors are tight or could be removed, leaving a gap in understanding the cost of noncommutativity.

## Nice-to-Haves
- Empirical illustrations on synthetic functions to demonstrate the predicted convergence differences (e.g., adaptive vs. NSD under varying smoothness conditions).
- A more detailed comparison with concurrent work (e.g., Kovalev & Borodich, 2025) to clarify the novelty of the adaptive variance assumption and resulting rates.
- Improved exposition with high-level overviews of proof techniques to enhance accessibility for a broader audience.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Missing experiments**: While experiments could strengthen the paper, the theoretical contributions are substantial on their own for a theory-focused venue. This point is moved to Nice-to-Haves.
- **Dense technical exposition**: This is subjective and not a substantive flaw; the paper is necessarily technical given its content. However, a suggestion for improved clarity is included in Nice-to-Haves.
- **Boundedness assumption for acceleration**: The paper addresses the need for a known diameter \(D\) via a projected variant (Algorithm 8 and Remark 4.4), so it is not an unaddressed weakness.

## Novel Insights
The paper reveals that adaptive optimizers and NSD exploit non-Euclidean geometry through distinct smoothness notions: adaptive smoothness (stronger) versus standard smoothness. This difference is not merely technical; it leads to concrete algorithmic benefits: adaptive smoothness enables acceleration for adaptive methods, and analogously, adaptive variance enables dimension-free rates for NSD. The work thus provides a unified geometric perspective that explains the separation between the two algorithm families and deepens the theoretical understanding of adaptivity in optimization.

## Suggestions
- Include a discussion on typical scenarios where adaptive smoothness might be close to standard smoothness (or where the gap is large), perhaps by analyzing simple function classes or citing empirical studies on neural network loss landscapes.
- Consider extending the lower bound (Theorem 4.7) to other norms beyond \(\ell_\infty\), or at least provide a discussion on the challenges of such an extension.
- Add a remark on the tightness of the logarithmic dimension factors in the non-diagonal case, possibly conjecturing whether they are necessary.

---

## v05SW2X3IC

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper bridges classical information theory with modern representation learning by proposing a learnable three-channel codec based on the Gray-Wyner network. It theoretically bounds lossy common information, derives an optimization objective for the transmit-receive rate tradeoff, and validates the approach on synthetic data and vision benchmarks, demonstrating reduced redundancy compared to independent coding.

## Strengths
- **Theoretical grounding and novel extension:** The paper provides a principled, information-theoretic foundation by deriving new bounds for lossy common information (Theorem 1) and an optimizable objective for the transmit-receive tradeoff (Theorem 2). This is a significant extension of classical Gray-Wyner theory to the learned representation setting.
- **Comprehensive empirical validation:** The method is rigorously evaluated, first on a synthetic dataset to validate control over the tradeoff and architectural ablations, then on controlled edge cases (colored MNIST), and finally on two challenging real-world vision task pairs (Cityscapes and COCO). The results consistently show the method's ability to reduce redundancy and navigate the tradeoff.

## Weaknesses
- **Empirical gap to theoretical bounds:** The empirical rates achieved on the synthetic dataset are notably higher than the theoretical rate-distortion limits (Figures 3, 9, Tables 2-5). While the paper acknowledges this common issue in learned compression, a deeper analysis of the sources of this gap (e.g., entropy model suboptimality, quantization, architecture capacity) is missing and would strengthen the work.
- **Limited quantitative analysis of disentanglement:** For the real-world vision experiments, the paper lacks a quantitative analysis of what information is captured in the common versus private channels (e.g., via mutual information estimates or auxiliary task probes). While qualitative MNIST reconstructions (Fig. 10) are provided, quantitative evidence for the claimed disentanglement on complex tasks would solidify the core contribution.

## Nice-to-Haves
- **Extension to more than two tasks:** The conclusion mentions the exponential scaling of channels for more tasks as a limitation. A preliminary experiment or concrete architectural sketch for a three-task scenario would help readers assess the framework's scalability.
- **Hyperparameter sensitivity analysis:** A more systematic ablation of the tradeoff parameter β and the auxiliary loss weight γ, especially on the vision benchmarks, would provide clearer guidance for practitioners.
- **Comparison to broader multi-task learning baselines:** While the paper appropriately compares to its own derived baselines (Joint, Independent), a comparison to a modern multi-task learning method that learns shared representations (without explicit rate constraints) could better contextualize the practical value of the rate-distortion efficiency gained.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Criticism of restrictive Markov assumptions:** The paper initially states Markov conditions (Eq. 1) but explicitly removes this requirement in Section 3.3, stating the architecture provides access to both sources for all branches. Therefore, this is not a valid weakness.
- **Demand for experiments on task conflicts or negative transfer:** The paper's scope is tasks with some common information. Testing on antagonistic tasks is outside its stated focus.
- **Request for theoretical proofs of representation compatibility:** The paper's core contribution is empirical and algorithmic; Appendix C provides a theoretical discussion, but demanding formal proofs for this analysis imposes an arbitrary rigor requirement not standard for this type of work.
- **Criticism of formatting/style nitpicks:** Minor typographical errors (e.g., in the rate-distortion function definition) do not constitute a substantive weakness.

## Novel Insights
The paper's key novel insight is the operationalization of the Gray-Wyner rate region's transmit-receive tradeoff via a tunable parameter (β) in a neural network optimization objective. This provides a direct, learnable mechanism to navigate the fundamental information-theoretic tradeoff between total bitrate and the bitrate required when tasks are decoded separately. The synthetic experiments (Fig. 3a) clearly demonstrate this control, showing the common channel rate moving from above to below the empirical mutual information as β shifts from 1 to 2.

## Suggestions
- **Analyze the rate gap:** In the discussion or appendix, provide a focused analysis hypothesizing why the empirical rates diverge from theoretical bounds (e.g., limitations of the entropy model, quantization, or function family capacity). This would turn a noted limitation into a constructive direction.
- **Quantify channel information for vision tasks:** Perform an additional analysis, perhaps using a simple proxy, to estimate the task-relevant information contained in each channel for the Cityscapes/COCO experiments. This would provide concrete evidence for the learned disentanglement.

---

## W42oLSwI9p

- GT: Reject (avg 5.0)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces three one-step, end-to-end diffusion solvers (CMILP, SCMILP, MFILP) for integer linear programming, directly addressing the slow inference of prior diffusion-based methods and their limitation to binary variables. Key contributions include a novel Iterative Integer Projection (IIP) layer to handle non-binary integers without problem binarization and a momentum-enhanced objective-guided sampling scheme. Experiments demonstrate order-of-magnitude speedups over prior neural solvers while maintaining high feasibility, and the ability to solve non-binary problems directly.

## Strengths
- **Significant inference speedup for neural ILP solvers:** By adapting one-step generative models (consistency, shortcut, meanflow), the proposed methods achieve solving times in seconds versus hours/minutes for prior diffusion-based approaches (e.g., IP Guided DDPM/DDIM), as shown in Tables 1-3, while maintaining high dataset feasibility.
- **First neural solver for general non-binary ILP via a novel projection layer:** The proposed Iterative Integer Projection (IIP) layer provides a differentiable mechanism for integer variables across the real domain, enabling direct handling of bounded integers without the exponential variable increase from binarization. Results in Tables 2 and 4 confirm its effectiveness and the computational advantage over binarized versions.
- **Effective enhancement of sampling via momentum:** Reframing objective guidance as a non-convex optimization problem leads to a momentum-based gradient descent update that consistently improves solution feasibility and optimality gap, as validated in the ablation study (Table 5).

## Weaknesses
- **Substantial optimality gap compared to traditional solvers:** While much faster than prior neural methods, the proposed solvers often produce solutions with significantly higher objective values (e.g., gaps of 70-90% on binary problems in Table 1, 10-20% on many non-binary problems) compared to exact solvers like Gurobi (0% gap). The paper does not sufficiently analyze the practical utility of this speed-quality trade-off, merely noting it as a limitation.
- **Insufficient analysis of the core IIP component:** The Iterative Integer Projection layer is central to the non-binary extension but lacks theoretical or empirical characterization of its convergence properties, approximation error, and gradient behavior. The strategy of using fewer iterations during training than testing is pragmatic but introduces a train-test discrepancy that is not analyzed for potential negative effects.
- **Evaluation on limited problem scales and types:** Experiments are conducted on generated datasets and classic benchmarks of moderate size (up to 2000 variables). Claims of "strong scalability" are not fully supported by tests on large-scale, industrial-grade MILP instances (e.g., from MIPLIB), which are necessary to assess practical utility for hard NP-hard problems.

## Nice-to-Haves
- Provide pseudo-code or a clearer algorithmic description in the main text to improve reproducibility.
- Include a sensitivity analysis for key hyperparameters like the number of IIP iterations (K) and the penalty coefficient (λ_penalty).
- Report variance measures (e.g., standard error) for gap and feasibility metrics across test instances.
- Conduct an ablation study directly comparing IIP against alternative differentiable rounding techniques (e.g., straight-through estimator).

## Removed Points
These points are flagged to be removed, treat them with caution.
- **Weakness:** "The train-test mismatch for the IIP layer is a non-standard practice that could lead to distribution shift." *Reason: The paper explicitly explains this design choice (fewer iterations for training efficiency, more for test accuracy), which is a reasonable engineering trade-off, not a methodological flaw.*
- **Weakness:** "The descriptions of SCMILP and MFILP are relegated to the appendix, making the methodology incomplete." *Reason: The core concepts are summarized in the main text, with details in the appendix, which is acceptable for a conference paper. The main contribution is the framework, not the re-derivation of these models.*
- **Weakness:** "The paper lacks a comparison of momentum guidance against other advanced sampling techniques like Langevin dynamics." *Reason: This is a request for an extensive new comparison outside the paper's scope. The paper's contribution is the novel application and enhancement of guidance, validated against the relevant baseline (single-step GD).*
- **Weakness:** "No statistical significance or variance is reported." *Reason: While variance reporting is good practice, single-run evaluation on benchmark sets is common in the field. The weakness is moved to Nice-to-Haves as a suggestion for improvement, not a core flaw.*
- **Strength:** "The paper is well-written and the topic is important." *Reason: This is a generic strength that applies to many papers and does not highlight what this specific paper does exceptionally well.*

## Novel Insights
The paper's primary novel insight is the integration of fast, one-step generative modeling paradigms (consistency, shortcut, meanflow) into the domain of constrained combinatorial optimization, demonstrating that high-speed, single-pass inference is possible while maintaining constraint feasibility. A secondary, valuable insight is that a simple, differentiable iterative projection (IIP) can effectively relax general integer constraints for neural solvers, avoiding the costly and size-exploding step of binarization that has limited prior work. Finally, reframing diffusion guidance as a gradient descent process reveals that adding momentum—a standard optimization technique—can tangibly improve the sampling search, a simple but effective innovation.

## Suggestions
- Strengthen the discussion of the speed-optimality trade-off: explicitly position the work as providing a fast, high-feasibility heuristic for scenarios where a "good enough" solution is needed quickly, and discuss how the optimality gap might be mitigated (e.g., by using the output to warm-start a traditional solver).
- Provide a brief empirical analysis of the IIP layer: plot convergence to integer values and the effect of the number of iterations (K) on final solution feasibility and integrality error for a representative set of values.

---

## oh9ChF7Pv0

- GT: Accept (Poster) (avg 4.7)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
EGG-SR introduces a unified framework that integrates symbolic equivalence into symbolic regression via equality graphs (e-graphs). It accelerates learning in Monte Carlo Tree Search (MCTS), deep reinforcement learning (DRL), and large language models (LLMs) by pruning redundant exploration, aggregating rewards across equivalent expressions, and enriching feedback prompts. Theoretically, it offers a tighter regret bound for MCTS and reduces gradient variance for DRL; empirically, it improves accuracy across several benchmarks.

## Strengths
- **Unified integration across multiple paradigms**: The paper systematically adapts e-graphs to enhance MCTS (via equivalence-aware backpropagation), DRL (via reward aggregation), and LLMs (via prompt enrichment), demonstrating a cohesive framework for leveraging symbolic equivalence in diverse SR algorithms.
- **Theoretical grounding**: Theorems 3.1 and 3.2 provide formal justifications—showing a tighter regret bound under optimistic planning assumptions and proving variance reduction for the policy gradient estimator—with detailed proofs in the appendix.
- **Empirical improvements and efficiency**: Tables 1 and 2 show consistent reductions in normalized MSE across trigonometric and scientific benchmarks when EGG is added. Figures 4 and 5 confirm substantial memory savings and minimal runtime overhead, making the approach practical.

## Weaknesses
- **Theory-practice gap for MCTS**: Theorem 3.1 assumes an optimistic planning (OPD) framework, but the implemented MCTS uses the UCT heuristic. The paper does not empirically validate that the theoretical acceleration translates to standard UCT-based MCTS, leaving the practical benefit unclear.
- **Lack of statistical rigor**: Results are reported as point estimates (e.g., median NMSE) without confidence intervals, standard deviations, or multiple independent runs. This makes it difficult to assess the robustness and significance of the improvements.
- **Narrow benchmark scope for MCTS/DRL**: Experiments for MCTS and DRL are primarily on trigonometric functions; broader, standard benchmarks like SRBench are not included, limiting claims about generalizability to diverse symbolic regression problems.
- **Absence of comparison with GP baselines**: Prior work (e.g., de França & Kronberger) has integrated e-graphs into genetic programming for SR. A direct comparison with such methods would better demonstrate the unified framework's advantage over existing equivalence-aware approaches.
- **Unverified gradient variance reduction**: Theorem 3.2 claims variance reduction for EGG-DRL, but no direct measurement of gradient variance is provided—Figure 3(right) only shows variance of the objective estimate, not the gradient estimator itself.
- **No discussion of failure modes**: The paper does not analyze when EGG might fail or underperform, such as when no applicable rewrite rules exist, when domain restrictions cause numerical errors, or when e-graph construction overhead outweighs benefits.

## Nice-to-Haves
- Ablation study on the impact of different rewrite rule sets and the number of sampled equivalents (K) on performance and efficiency.
- Evaluation on comprehensive, widely-adopted benchmarks like SRBench to strengthen claims of broad applicability.
- Deeper analysis of computational overhead for MCTS and LLM integrations, beyond the DRL-focused Figure 5.
- Discussion of practical strategies to handle domain restrictions for rewrite rules (e.g., filtering invalid expressions during sampling).

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **"Simplistic reward function"**: The reward function 1/(1+NMSE) is standard in symbolic regression; criticizing it as simplistic does not undermine the core contribution.
- **"Marginal LLM improvements"**: While improvements in Table 2 are sometimes small, they are consistent across models and datasets; minor gains can still be meaningful in this context.
- **"Lack of a dedicated limitations section"**: The paper discusses open problems and constraints in Sections 3.3 and B.2; a separate section is not mandatory.
- **"Formatting/style nitpicks"**: Artifacts from PDF extraction (e.g., broken figure references) do not affect the technical content.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct multiple independent runs with different random seeds to report mean and standard deviation (or confidence intervals) for NMSE metrics, enhancing statistical reliability.
- Include a comparison with a state-of-the-art genetic programming method that uses e-graphs (e.g., de França & Kronberger, 2025) to better position the unified framework's novelty.
- Directly measure and report the variance of the gradient estimator in DRL experiments to empirically validate Theorem 3.2.
- Add a brief case study or discussion illustrating scenarios where EGG does not improve performance, helping to define the method's boundaries.

---

## c7OsKOOZo8

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (4.1/10)
- Match: N/A

### Final Review

## Summary
This paper proposes an end-to-end framework for multi-view diabetic retinopathy (DR) grading that reduces dependency on external annotations. It introduces a Grade-Activated Lesion Proposal (GALP) module to self-generate lesion cues from grade-conditioned evidence maps and a Cross-View Lesion Expert Guided Regional Fusion (LGRF) module for selective, context-aware feature integration. The method achieves competitive or state-of-the-art performance on two multi-view DR datasets without requiring lesion or vessel annotations.

## Strengths
- **Practical Impact**: The framework directly addresses a key deployment bottleneck by eliminating the need for costly, expert-provided annotations (e.g., lesion masks) while maintaining end-to-end training and competitive accuracy, as demonstrated by outperforming all end-to-end baselines and matching several externally-informed methods.
- **Strong Empirical Results**: Comprehensive experiments on two standard benchmarks (MFIDDR and DRTiD) show that the lesion-free variant surpasses all end-to-end methods and rivals annotation-dependent approaches, with the lesion-enhanced version achieving state-of-the-art results on multiple metrics (e.g., accuracy, kappa).
- **Novel Integration**: The combination of self-supervised lesion proposal generation via auxiliary classification and a dynamically gated mixture-of-experts fusion mechanism represents a technically sound and innovative approach for capturing subtle, cross-view lesion evidence.

## Weaknesses
- **Quantitative Proposal Validation Missing**: The core claim that self-derived proposals act as effective lesion surrogates is not directly validated; no metrics (e.g., Dice score, IoU) compare proposals to available ground-truth lesion masks (e.g., on MFIDDR), leaving uncertainty about their localization accuracy.
- **No Computational Efficiency Analysis**: The overhead introduced by stage-wise auxiliary classifiers, an expert pool, and cross-view attention is not quantified (e.g., FLOPs, inference time), which is critical for assessing practical deployment trade-offs.
- **Performance Gap in Severe Cases**: Without external annotations, the model underperforms on Grade 4 (proliferative DR) compared to some externally-informed methods (Table 2), suggesting limitations in handling complex pathologies where fine-grained lesion cues may be crucial.
- **Lack of Limitations Section**: The paper omits a dedicated discussion of limitations, such as the assumption that grade-discriminative regions perfectly align with lesions, sensitivity to hyperparameters (e.g., patch size \(q\)), and the residual performance gap when annotations are absent.
- **Hyperparameter Study Lacks Statistical Rigor**: Figure 3 reports performance trends for key hyperparameters (retention ratio \(\alpha\), number of experts \(K_2\)) without error bars, confidence intervals, or multiple runs, reducing confidence in the robustness of these design choices.

## Nice-to-Haves
- Extending evaluation to other multi-view medical imaging tasks (e.g., glaucoma assessment) to demonstrate broader applicability of the GALP and LGRF modules.
- Analyzing the specialization of different experts in the MoE framework to provide insights into what visual patterns or lesion types they capture.
- Conducting an ablation study that isolates the contribution of the auxiliary classification loss from the proposal selection mechanism (e.g., by comparing with random region selection).

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Missing comparison to LFMVDR on DRTiD**: This criticism assumes LFMVDR is a standard baseline for DRTiD, but the paper’s Table 3 includes methods previously evaluated on that dataset, and LFMVDR may not have been benchmarked there.
- **Minor typos and formatting nitpicks** (e.g., "whcih"): These do not affect the technical content or readability.
- **Speculative training instability** due to auxiliary loss interplay: No evidence is provided, and the paper uses standard loss weighting.
- **Demand for backbone variation experiments**: The contribution is modular and not claimed to be backbone-agnostic; using a strong pretrained backbone (Swin-B) is common practice.
- **Reproducibility concerns about routing network architecture**: The paper describes the Router as a linear projection (Eq. 9), which is sufficient for replication.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add a quantitative evaluation of lesion proposal quality using the available segmentation masks on MFIDDR (e.g., compute Dice score between top-K proposal regions and ground-truth lesions).
- Include a computational analysis comparing model size, FLOPs, and inference time with key baselines (e.g., MVCINN, CVSA) to contextualize efficiency trade-offs.
- Incorporate a limitations section addressing the Grade 4 performance gap, assumptions in proposal generation, hyperparameter sensitivities, and potential failure modes.

---

## ppXAVexrAM

- GT: Reject (avg 4.0)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces ARSS, the first decoder-only autoregressive transformer for novel view synthesis from a single image with explicit camera trajectory control. It employs a video tokenizer for temporal consistency, a camera autoencoder for 3D positional guidance, and a spatial permutation strategy to adapt causal modeling to visual data. The method achieves competitive performance against diffusion-based baselines, particularly in maintaining quality over long camera trajectories.

## Strengths
- **Novel paradigm**: ARSS is the first to rigorously adapt a GPT-style, causal autoregressive model to camera-controlled novel view synthesis, opening a new direction for sequential 3D-aware generation.
- **Comprehensive evaluation**: Experiments on RealEstate10K, ACID, and zero-shot DL3DV benchmarks are thorough, with an insightful error accumulation analysis (Fig. 6) demonstrating robust long-horizon performance.
- **Technically sound design**: The integration of a video tokenizer (VidTok) to preserve temporal coherence, a camera autoencoder with explicit geometric constraints (Eq. 5), and a spatial permutation strategy effectively address key challenges in autoregressive visual generation for 3D tasks.

## Weaknesses
- **Performance trade-offs**: While ARSS excels in pixel-level and perceptual metrics (PSNR, LPIPS), it shows slightly lower geometric consistency (SSIM, FID) compared to the best diffusion-based baseline (SEVA) on some datasets (Table 1). The comparison is partially confounded by SEVA's training on larger-scale data, which the paper acknowledges but does not fully equalize.
- **Incomplete ablation study**: The paper lacks an ablation on the camera autoencoder's contribution (e.g., removing camera tokens or using a simpler pose encoding). This is necessary to substantiate the claim that learned camera tokens provide essential 3D guidance.
- **Limited evidence of 3D awareness**: The method is claimed to have "3D spatial awareness," but no direct analysis (e.g., depth accuracy, geometric consistency measures beyond image metrics) is provided to verify how well it models underlying geometry.
- **Methodological clarity gaps**: The conditioning mechanism of camera tokens—how they precisely guide visual token prediction in the interleaved sequence (Eq. 6, 8)—and the handling of causal attention masks after spatial permutation could be more explicitly detailed for full reproducibility.

## Nice-to-Haves
- Comparison to autoregressive video-generation baselines (e.g., adapted from Pang et al. 2025) to isolate the benefits of the proposed architecture over generic AR video models.
- Testing on longer trajectories beyond the trained sequence length (17 frames) to further validate scalability claims for "large environments."
- Analysis of failure modes (e.g., under large view changes or textureless regions) and visualization of full sequences as videos to better assess temporal consistency.
- Discussion of computational efficiency (training/inference cost) relative to diffusion baselines.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Formatting nitpicks**: Concerns about Equation 7 formatting are parser artifacts, not paper errors.
- **Scope creep**: Demanding comparison to autoregressive video-generation baselines is outside the paper's focused contribution to novel view synthesis with camera control.
- **Overly specific demands**: Requests for attention maps or synthetic 3D dataset evaluation are insightful but not standard requirements for this paper's community; they are moved to nice-to-haves.

## Novel Insights
None beyond the paper's own contributions. The paper's core insight is demonstrating that a decoder-only autoregressive model, when augmented with video tokenization, camera conditioning, and spatial permutation, can effectively perform 3D-aware novel view synthesis with causal trajectory generation—a novel and promising direction.

## Suggestions
- Conduct an ablation study to evaluate the necessity of the camera autoencoder, e.g., by comparing against a baseline that uses raw Plücker coordinates or simple embeddings for camera conditioning.
- Provide a more detailed, step-by-step explanation of how camera tokens condition visual token prediction during training and inference, including the attention masking scheme after permutation.
- Incorporate a direct measure of 3D consistency (e.g., depth estimation accuracy or novel-view metrics on synthetic data) to strengthen claims about geometric awareness.

---

## Vit5M0G5Gb

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.1/10)
- Match: N/A

### Final Review

## Summary
This paper presents a unifying theoretical framework that explains stage-like learning dynamics (saddle-to-saddle) and a corresponding simplicity bias across neural network architectures. The core mechanism is built on three pillars: (1) embedded fixed points, where solutions of narrower networks become saddles in wider ones; (2) invariant manifolds connecting these saddles, along which the network behaves like a simpler one; and (3) timescale separation that guides dynamics along these manifolds. The authors distinguish between data-induced separation (leading to low-rank weights) and initialization-induced separation (leading to sparse weights), providing a common explanation for phenomena observed in linear, ReLU, convolutional, and attention-based networks.

## Strengths
- **Unifying Theoretical Framework**: The paper successfully generalizes prior architecture-specific results into a single, cohesive theory. Theorems 1 and 3 establish the existence of embedded fixed points and invariant manifolds for a broad class of architectures (fully-connected, convolutional, self-attention) defined by a general layer equation, providing a foundational structural explanation for simplicity bias.
- **Mechanistic Insight into Timescale Separation**: The analysis cleanly disentangles two distinct origins of saddle-to-saddle dynamics: timescale separation between directions driven by data statistics (Theorem 4, leading to low-rank weights) and timescale separation between units driven by initialization (Proposition 5, leading to sparse weights). This clarifies why different architectures exhibit different dynamical signatures.
- **Predictive Power and Empirical Validation**: The theory generates concrete, testable predictions about how network width, data distribution (e.g., power-law singular values), and initialization affect plateau durations and learning stages. These predictions are convincingly validated through systematic simulations across the six featured architectures (Figure 1, 2) and on MNIST (Figure 3).

## Weaknesses
- **Limited Rigor in Nonlinear Dynamics Analysis**: While the existence results (Theorems 1, 3) are general and the linear case analysis (Theorem 4) is rigorous, the treatment of dynamics for nonlinear cases (e.g., quadratic networks in Section 5.2) relies on heuristic arguments and approximations. Proposition 5 provides intuition but lacks the formal probabilistic proof suggested by its "almost surely" claim. The Taylor expansion argument for general nonlinear activations is also heuristic.
- **Dynamics Analysis is Restricted to Two-Layer Networks**: The detailed dynamical analysis in Section 5 focuses on two-layer networks with homogeneous polynomial activations. Although fixed points and invariant manifolds apply to deep networks (via Corollary 2), the analysis of when and how saddle-to-saddle dynamics occurs in deep networks is relegated to conjectures and preliminary simulations (Section 7, Figure 5). A theoretical treatment of deep network dynamics remains an open challenge.
- **Experimental Validation Primarily Uses Synthetic/Toy Data**: The empirical support, while illustrative and aligned with predictions, is largely based on low-dimensional synthetic data or binary MNIST classification. Demonstrating that the predicted effects (e.g., on width and initialization) scale and persist in larger, state-of-the-art architectures on complex datasets would strengthen the paper's practical relevance.

## Nice-to-Haves
- A more rigorous treatment of the dynamics near saddles for quadratic and higher-order polynomial networks, potentially using singular perturbation theory, could solidify the heuristic arguments in Section 5.2.
- Extending the experimental validation to deeper architectures (e.g., multi-layer transformers or CNNs) on standard benchmarks (e.g., CIFAR, simple language tasks) would help demonstrate the broader applicability of the theoretical predictions.
- A quantitative analysis tracking the distance of the training trajectory from the predicted invariant manifolds during plateaus versus transitions would provide direct evidence for the proposed mechanism.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength**: "The paper is well-written" and "the topic is important" – these are generic and removed.
- **Weakness**: "The broader impact statement is missing" – this is a formatting/ submission requirement issue, not a scientific weakness of the paper's content.
- **Weakness**: "The Taylor expansion argument for general nonlinear activations is heuristic and not backed by theorem" – the paper explicitly frames this as an intuition and supports it with examples (Figure 4); demanding a theorem here is outside the stated scope of the analysis.
- **Weakness**: Demands for "exhaustive empirical validation" on large-scale SOTA architectures – the paper's goal is to establish a theoretical framework and test its predictions in controlled settings; this criticism is scope creep.
- **Weakness**: Criticisms about the non-standard notation for self-attention (Equation 2) – the paper clearly states its purpose is to show the general form incorporates it, which is sufficient for the theoretical argument.

## Novel Insights
The paper's primary novel insight is the unification of saddle-to-saddle dynamics across diverse architectures under a single framework, revealing that the operative notion of "simplicity" is the minimal number of effective units (neurons, kernels, heads) required to express the current solution. A further key insight is the distinction between data-induced timescale separation (favoring low-rank weight structures) and initialization-induced separation (favoring sparse weight structures), which clarifies previously observed but seemingly disparate dynamical behaviors in linear versus quadratic-like networks.

## Suggestions
- Strengthen the discussion in Section 5.2 (Quadratic Case) by providing a more formal analysis of the timescale separation, potentially using differential inequalities or a singular perturbation approach to bound the growth of units relative to each other.
- In the experiments, include a direct, quantitative measure of "effective width" (e.g., rank of weight matrices, number of active heads) and correlate its step-wise increases with the observed loss plateaus for a broader set of architectures.

---

## KsWRLyIAKP

- GT: Withdrawn (treated as Reject) (avg 3.2)
- Predicted: N/A (4.1/10)
- Match: N/A

### Final Review

## Summary
This paper reformulates lead-lag detection in financial markets as a temporal link prediction task on dynamic graphs. It introduces a custom dataset of 37 assets, adapts and evaluates eight deep learning models (including TGNNs and an LSTM baseline), and systematically assesses two relationship scenarios. The key finding is that the simple GraphMixer model outperforms more complex temporal graph neural networks.

## Strengths
- **Novel and well-motivated problem formulation.** The paper clearly reframes lead-lag detection as a temporal link prediction problem, leveraging graph structure to capture multi-asset interdependencies beyond pairwise statistical methods (Sections 1, 3.1).
- **Comprehensive empirical evaluation.** The study rigorously compares eight models across two dataset variants using multiple ranking metrics, reports results over five runs with standard deviations, and validates statistical significance via Friedman and Conover tests (Tables 1–2, Figure 2, Appendix F).
- **Insightful ablation study.** The analysis of feature impact (Table 3) reveals that static node embeddings often suffice and adding temporal features can degrade performance, prompting important questions about the necessary complexity for this task.

## Weaknesses
- **Lack of comparison to established non-ML baselines.** The paper acknowledges that direct comparison to traditional financial methods (e.g., Granger causality, threshold-based networks) is complex but scopes it out (Section 3.1). Without such baselines, it is impossible to assess whether the proposed TGNN framework offers a practical advance over existing techniques, undermining its contribution to the finance domain.
- **No sensitivity analysis for critical graph construction parameters.** The lead-lag definition relies on fixed thresholds (ε=5%, τ=1 day) justified by prior work but without ablation on these values (Sections 3.1–3.2). The performance and graph dynamics are highly sensitive to these choices, leaving robustness concerns unaddressed.
- **Unclear temporal generalization evaluation.** The train/validation/test split procedure is not explicitly described as temporal (Section 4.2). For a time-series task, a random split risks data leakage and overoptimistic performance; a strict temporal split is necessary to assess forecasting ability.
- **Poorly motivated and confusing model variant (GM-TNF).** The description of GraphMixer-Temporal Node Features is brief and its comparison to standard GraphMixer is conflated with feature choices (Section 3.4, Figure 5). This muddles the analysis of whether temporal node features are beneficial.

## Nice-to-Haves
- Efficiency comparison of models in terms of training/inference time, especially given the mention of APAN’s focus on speed.
- Deeper analysis of why GraphMixer excels, such as examining learned representations or graph connectivity patterns to hypothesize about task complexity.
- Visualization of top predicted lead-lag pairs to assess economic plausibility and case studies aligning predictions with market events.

## Removed Points 
*These points are flagged to be removed, treat them with caution.*
- **Criticism about dataset size being too small:** The dataset serves as a proof-of-concept benchmark for a novel task, and its scale (37 nodes, 1257 time steps) is sufficient for the methodological claims without harming core conclusions.
- **Demand for a profitability backtest:** While relevant for financial applications, trading simulations are beyond the scope of a machine learning methodology paper focused on model evaluation and benchmarking.
- **Hyperparameter tuning inconsistencies for all models:** The paper follows established practices from prior work (TGL framework, Cong et al. 2023 setup), and minor tuning variations are unlikely to invalidate the relative performance trends shown.

## Novel Insights
The paper’s finding that a simple MLP-based model (GraphMixer) consistently outperforms sophisticated TGNNs with attention or memory mechanisms echoes recent “less is more” trends in graph learning. This suggests that lead-lag patterns in this financial setting may be captured by local, short-term dependencies rather than complex temporal memories, offering a valuable counterpoint to default assumptions about necessary model complexity for dynamic graphs.

## Suggestions
- Add at least one simple non-ML baseline (e.g., a rule-based method using the same threshold logic or a linear model on historical returns) to establish a performance floor and contextualize the gains from TGNNs.
- Conduct sensitivity analysis on ε and τ to demonstrate the robustness of the framework and model rankings to these critical parameters.
- Clarify the data split strategy in the methodology, ensuring it is strictly temporal (e.g., train on early data, validate on middle, test on latest) to properly evaluate forecasting ability and avoid data leakage.
- Include a limitations section discussing the dataset’s heuristic selection, parameter sensitivity, scope of comparison to traditional methods, and the broader impact of financial prediction models.

---

## 3icvqeC1sA

- GT: Reject (avg 4.5)
- Predicted: N/A (6.4/10)
- Match: N/A

### Final Review

## Summary
This paper introduces ChaosNexus, a foundation model for universal chaotic system forecasting. Its core contribution is the ScaleFormer, a U-Net-inspired Transformer architecture designed to explicitly capture the multi-scale temporal structure of chaotic dynamics, augmented with Mixture-of-Experts (MoE) layers and a wavelet-based frequency fingerprint. The model demonstrates strong zero-shot generalization on a large testbed of synthetic chaotic systems and achieves state-of-the-art, data-efficient performance on real-world 5-day global weather forecasting.

## Strengths
- **Novel and well-motivated multi-scale architecture**: The ScaleFormer's hierarchical patch merging/expansion with skip connections is a principled design to capture the intrinsic multi-scale nature of chaotic dynamics. Ablation studies (Table 1) confirm that removing these components causes significant degradation in both point-wise accuracy and long-term attractor fidelity, validating the core architectural innovation.
- **Comprehensive and compelling empirical results**: The model is evaluated on a massive, diverse test set (>9,000 synthetic systems) using a holistic suite of metrics (point-wise sMAPE, correlation dimension error, KL divergence between attractors, etc.). ChaosNexus achieves competitive point-wise accuracy and superior long-term statistical fidelity compared to strong baselines. Its real-world impact is demonstrated by exceptional zero-shot performance (MAE <1°C) on 5-day weather forecasting, surpassing models fine-tuned on significantly more data.
- **Valuable scaling analysis and practical insight**: The scaling experiments (Figure 4) provide a concrete, evidence-based principle for building scientific foundation models: generalization improves more with the diversity of training systems than with the volume of data per system. This insight is substantiated by the model's remarkable few-shot data efficiency.

## Weaknesses
- **Lack of a dedicated limitations section**: The paper does not critically discuss the boundaries of its approach. Key limitations include: the model is pretrained solely on synthetic ODEs; its performance on highly stochastic real-world systems or spatiotemporal PDEs beyond the preliminary appendix result is untested; and the computational cost/benefit trade-off of the multi-scale MoE architecture is not formally analyzed. Acknowledging these scope conditions is essential for a complete scientific presentation.
- **Theoretical motivation could be more deeply integrated**: While the architectural choices (multi-scale, MoE, wavelet fingerprint) are intuitively justified and linked to chaotic dynamics in Appendix G, the main text lacks a concise, formal connection to established theory (e.g., timescale separation in Lyapunov spectra, invariant measures). Strengthening this link would solidify the methodological foundations.
- **Insufficient detail on few-shot experimental protocol**: The methodology for creating the 0.1% and 0.5% few-shot subsets for weather forecasting is not specified in the main text or appendices (e.g., random sampling vs. structured selection). This omission makes it difficult to assess potential data leakage or to precisely reproduce the few-shot learning claims.

## Nice-to-Haves
- **Deeper analysis of learned representations**: A quantitative analysis correlating MoE expert activation patterns with system invariants (e.g., Lyapunov exponents) would provide stronger evidence for the claimed specialization mechanism. Similarly, evaluating the discriminative power of the frequency fingerprint across systems with similar temporal but different spectral properties would strengthen its justification.
- **Extended real-world validation**: Testing zero-shot performance on a broader class of real-world chaotic time series (e.g., from finance, biology, or engineering) would further substantiate the claim of "universal" forecasting and help delineate the model's effective domain of application.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "Incomplete baseline comparison"**: The paper compares against a extensive set of contemporary foundation and task-specific models (Panda, DynaMix, Parrot, TimesFM, Chronos, etc.). While no paper can include every baseline, the selected set is comprehensive and appropriate for the stated claims. Demanding inclusion of other specific models is scope creep.
- **Weakness: "Overstated claims about general-purpose models"**: The paper's claim is supported by evidence: general-purpose models (Chronos, TimesFM) perform poorly, and fine-tuning one (Chronos-S-SFT) on the chaotic corpus improves performance. This validly demonstrates the need for domain-specific design *and* data, not an overstatement.
- **Weakness: "Lack of statistical significance tests"**: The paper reports results with 95% confidence intervals across thousands of test systems, which is standard and sufficient for the field. Demanding formal hypothesis tests is imposing a methodological practice not routinely expected in large-scale empirical ML papers.
- **Nitpicks on grammar and typos**: (e.g., "the model can generalizes"). These are minor copy-editing issues that do not affect the scientific content.

## Novel Insights
The primary novel insight, supported by rigorous scaling experiments, is that effective generalization for chaotic system forecasting is driven more by the diversity of systems in the pretraining corpus than by the volume of trajectories per individual system. This provides a concrete, counter-intuitive design principle for building scientific foundation models, moving beyond the default assumption of simply "more data." The architectural synthesis—a U-Net-like multi-scale Transformer with MoE and a fixed wavelet fingerprint—is itself a novel and insightful approach to embedding the structural priors of chaotic dynamics into a foundation model.

## Suggestions
- **Add a "Limitations" section** before the conclusion, explicitly discussing the scope of the pretraining data (synthetic ODEs), potential challenges with stochasticity or high-dimensional spatiotemporal systems, and the computational trade-offs of the architecture.
- **Integrate key theoretical motivations from Appendix G into the main methodology section** (Section 3) to more firmly root the design choices in the language of chaotic dynamics (e.g., multi-scale processing for timescale separation, MoE as a basis for local vector fields).
- **Clarify the sampling procedure** for the few-shot weather data subsets (0.1%, 0.5%) in Appendix F to ensure full reproducibility.

---

## Ksvv8x00eo

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (5.9/10)
- Match: N/A

### Final Review

## Summary
This paper introduces CaTS-Bench, a large-scale, multimodal benchmark for context-aware time series captioning and reasoning. It aggregates 11 real-world datasets, providing time series segments, metadata, line plots, and captions generated via a scalable LLM-based pipeline, complemented by a human-revisited subset. The benchmark includes both a captioning task and a suite of diagnostic multiple-choice Q&A tasks. Comprehensive evaluation of leading vision-language models reveals significant weaknesses in numeric fidelity and, notably, a failure to effectively leverage visual inputs despite their availability.

## Strengths
- **Comprehensive and novel benchmark construction:** CaTS-Bench integrates numeric series, rich metadata, visual plots, and validated captions at a scale (~570k timesteps, 20k samples) and modality breadth that surpasses existing time series captioning resources (Table 1). The inclusion of diagnostic Q&A tasks for fine-grained reasoning is a valuable addition.
- **Rigorous validation of the semi-synthetic data pipeline:** The authors provide extensive quality checks for the LLM-generated reference captions, including manual factual verification (98.6% accuracy), human indistinguishability studies (41.1% detection accuracy), and diversity analyses (Sections 3.2, H.1-H.4). This systematic validation mitigates core concerns about using synthetic references.
- **Insightful and thorough empirical analysis:** The benchmarking of a wide range of models in zero-shot, finetuned, and program-aided settings, with tailored numeric fidelity metrics, provides a clear snapshot of the field. The analysis revealing that visual input provides negligible or negative benefit for most models (Section 4.3, Figure 4) is a critical and well-supported finding that highlights a fundamental gap in current multimodal integration.

## Weaknesses
- **Inherent reliance on a single LLM's stylistic bias for evaluation:** Despite rigorous validation, the majority of reference captions are generated by one model (Gemini 2.0 Flash). The paraphrasing experiment (H.3) suggests robustness, but the benchmark's evaluation could still subtly favor models whose output distribution aligns with this oracle. This is a fundamental trade-off in the scalable pipeline design.
- **Under-explored necessity of the visual modality in the task design:** A key finding is that models ignore the provided plots. While this correctly indicts current VLMs, it also prompts the question: does the task, as formulated (with full numeric series and metadata provided textually), genuinely *require* visual reasoning? A deeper discussion on whether the benchmark successfully tests multimodal integration or inadvertently allows a text-only solution is needed.
- **Incomplete analysis of what finetuning teaches models:** The paper notes that finetuned models can become overconfident, reporting statistics even when inaccurate (Appendix K.2). A more systematic analysis is needed to distinguish whether improved numeric scores stem from learning approximate computation or from pattern-matching the stylistic tendency of the ground-truth captions to include such statistics.

## Nice-to-Haves
- **Expanded human-authored references:** Increasing the size and domain coverage of the human-revisited subset would strengthen the benchmark as a gold standard for evaluating nuanced, human-like captioning.
- **Stronger textual baseline:** Including a powerful text-only LLM (e.g., GPT-4) provided with only the numeric series and metadata would better isolate and quantify the marginal value, or lack thereof, of the visual input in the current evaluation setting.
- **Failure mode analysis for Q&A:** Categorizing the reasons models fail specific Q&A tasks (e.g., value misreading, temporal confusion) would provide more actionable diagnostic guidance beyond accuracy scores.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "Abstract discrepancy in timestamps vs. samples."** The paper clearly states 570k total timestamps are used to create 20k samples; this is not a discrepancy but a difference in reported units.
- **Weakness: "Lack of statistical significance tests."** Reporting such tests is not a standard requirement for large-scale benchmark evaluations where single-run results are stable (variance is shown to be minimal in H.5) and the performance gaps are large and consistent across multiple metrics and models.
- **Weakness/Suggestion: "Need for domain-shift generalization test."** This is a potential future extension of the benchmark, not a core flaw in its construction or evaluation. The paper's temporal train/test split is a standard and valid approach to prevent leakage.
- **Weakness/Suggestion: "Requirement for human evaluation of model outputs."** While insightful, this is a substantial additional study beyond the paper's primary contribution of establishing the benchmark and baseline evaluations. The proposed automated metrics, especially the novel numeric fidelity scores, are well-justified proxies.
- **Suggestion: "Ablation on metadata influence."** The paper explicitly defines the benchmark as "context-aware" and provides metadata; testing its necessity is an interesting experiment but falls outside the scope of validating the benchmark itself.

## Novel Insights
Beyond creating the needed benchmark, the paper yields a crucial, counter-intuitive insight: current vision-language models, when presented with a time series captioning task, largely fail to utilize the visual plot modality, even when it is provided. Performance does not meaningfully drop—and sometimes improves—when the plot is removed (Figure 4). This finding, coupled with attention analysis showing focus on textual chart elements rather than trend lines, strongly suggests a fundamental misalignment in current multimodal architectures for this domain. The models default to textual and numeric priors, indicating that nominal multimodality does not equate to genuine integrated reasoning, thereby establishing a clear and specific target for future research.

## Suggestions
- In the discussion of the visual modality results (Section 4.3), more explicitly address the question of whether the benchmark's design minimizes the necessity of visual reasoning and what this implies for future benchmark iterations or model architectures.
- Enhance the discussion of the finetuning trade-off (Section 4.1, K.2) to explicitly warn that improved metric scores may come with overconfidence and to encourage future work to investigate learning mechanisms beyond stylistic mimicry.
- Ensure the released code includes straightforward scripts for replicating the main experimental results and for using the data generation pipeline, maximizing the benchmark's utility and adoption.

---

## aiM6bRd6bG

- GT: Reject (avg 4.0)
- Predicted: N/A (4.1/10)
- Match: N/A

### Final Review

## Summary
This paper formulates the problem of PPI candidate ranking to prioritize novel protein-protein interactions for experimental testing. It proposes a two-stage framework that first uses interpretability-guided retrieval based on active embedding regions from known interactions, then re-ranks top candidates with multiple biological signals including interaction scores, structural plausibility, and semantic features. Evaluation on a large-scale prospective dataset from STRING v11 to v12 shows substantial improvements in ranking metrics over sequence-based PPI prediction models.

## Strengths
- **Novel task formulation with practical motivation:** The paper introduces PPI candidate ranking and sets up a prospective evaluation using successive STRING releases, mimicking real-world discovery scenarios where novel interactions from a later database version serve as the test set. (Evidence: Sections 1, 4, 5.1)
- **Creative methodological contribution:** The interpretability-guided retrieval mechanism leverages active residue regions from contact maps of known interactions to compute embedding similarities, moving beyond raw interaction scores for ranking. (Evidence: Section 4.1, Figure 1)
- **Comprehensive integration of diverse evidence:** The re-ranking module incorporates multiple complementary signals—interaction score, structural (pDockQ), functional annotations, and LLM-based semantics—providing a holistic approach to candidate prioritization. (Evidence: Section 4.2)
- **Large-scale and thorough evaluation:** Experiments on the STRING v11→v12 transition cover a wide range of ranking metrics (Recall@k, MAP, nDCG, etc.) and demonstrate significant improvements, e.g., for D-SCRIPT, Recall@10 increases from <2% to >25% and MRR by 4-6×. (Evidence: Table 1, Section 5.3)
- **Insightful analysis of complementary signals:** Pairwise comparison of re-ranking strategies reveals which evidence types (e.g., PubMedBERT, lightweight semantic overlaps) most consistently improve rankings, offering valuable guidance for future work. (Evidence: Table 2)

## Weaknesses
- **Exaggerated improvement claims:** The paper states "improvements by two orders of magnitude" in the Introduction and Conclusion, but the actual metrics show substantial gains that are closer to one order of magnitude (e.g., Recall@10 improvement of ~12.5×). This overstatement misrepresents the results. (Why it matters: Accuracy in reporting is essential for credibility and proper assessment of contributions.)
- **Risk of data leakage in evaluation:** The use of STRING v12 as "novel" interactions is not explicitly validated to ensure no overlap with v11 evidence beyond the described filtering, potentially inflating prospective performance. (Why it matters: The core claim of prospective evaluation hinges on strict temporal separation between training and test data.)
- **Incomplete baseline comparisons:** The paper only compares against raw interaction scores of PPI prediction models (D-SCRIPT, Topsy-Turvy, xCAPT5), missing key ranking-specific baselines such as collaborative-filtering (e.g., ranking by average interaction score with known partners) or network-based methods. (Why it matters: Without stronger and more relevant baselines, the superiority of the proposed framework is not fully established.)
- **Lack of statistical validation:** Results are presented as point estimates (averages) without measures of variance, confidence intervals, or statistical significance testing across target proteins. (Why it matters: For high-stakes decisions in ML, robustness and reliability of improvements must be demonstrated.)
- **Missing ablation studies:** The individual contributions of interpretability-guided retrieval versus the re-ranking module are not isolated, and the impact of each re-ranking signal is only shown pairwise, not in an integrated manner. (Why it matters: Understanding which components drive performance is critical for methodological insight and future improvements.)
- **Ambiguity in region selection method:** The algorithm for identifying the "most active region" from contact maps is described textually but lacks precise steps (e.g., thresholding, contiguity enforcement), affecting reproducibility. (Why it matters: Reproducibility is a cornerstone of scientific validation, especially for novel methods.)
- **Generalizability concerns:** Primary evaluation is on a single dataset (STRING); appendix results on the PiNUI dataset show much lower rediscovery ratios (0.38 vs. 0.97), and the core assumption that novel interactions resemble known ones may fail for proteins with few interactors, which is not quantified. (Why it matters: Practical applicability depends on performance across diverse datasets and conditions, including cold-start scenarios.)
- **Potential data leakage in LLM fine-tuning:** The PubMedBERT model is fine-tuned on STRING v11 annotations but pre-trained on biomedical literature that may include information about v12 interactions, risking indirect leakage. (Why it matters: This could artificially boost re-ranking performance, compromising the validity of the semantic signal analysis.)
- **High computational cost:** Retrieval requires hundreds of hours (Figure 2), and structural re-ranking with SpeedPPI is prohibitively slow (~13 minutes per pair), limiting scalability for genome-wide screening. (Why it matters: Efficiency is critical for real-world adoption in large-scale biological discovery.)
- **Limited interpretability:** While interpretability is used as a structural tool to extract active regions, the final rankings do not provide biological explanations for why candidates are ranked high, as acknowledged by the authors. (Why it matters: Explainability enhances trust and can offer biological insights beyond mere prioritization.)

## Nice-to-Haves
- Visualizations or case studies illustrating successful and failed rankings to enhance intuitive understanding.
- Exploration of integrated re-ranking models that combine multiple signals via learned weights or ensembles.
- Cross-organism evaluation (e.g., on yeast or mouse data) to demonstrate robustness beyond human STRING.
- More detailed algorithmic description or pseudo-code for the active region selection step to improve reproducibility.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **"Two orders of magnitude" claim is hyperbolic:** This is kept as a weakness because it is factually based on the paper's own metrics, but note that the improvement is still substantial.
- **Criticisms about formatting or style nitpicks:** None were present in the reviews.
- **Demands for theoretical proofs or user studies:** Not applicable as this is an empirical systems paper.
- **Suggestions that the paper should have implemented obvious next steps (e.g., end-to-end ranker):** These are moved to Nice-to-Haves as they are improvements beyond the current scope.

## Novel Insights
The paper's key novel insight is the use of model interpretability not for explanation but as a methodological device to guide retrieval: by focusing embedding similarities on active residue regions identified from contact maps of known interactions, it effectively prioritizes novel candidates that share functional or structural patterns with established partners. This approach transforms internal model representations into a ranking mechanism. Additionally, the analysis reveals that semantic signals from LLMs and functional annotations consistently complement sequence-based methods, underscoring the importance of multi-evidence integration for prospective PPI discovery where no single signal suffices.

## Suggestions
- Address data leakage concerns by explicitly verifying no overlap between STRING v11 and the novel v12 interactions used in evaluation, perhaps through evidence timestamps or curation details.
- Include additional baselines such as collaborative-filtering (e.g., ranking candidates by average interaction score with known partners) or network propagation methods to strengthen the ranking comparison.
- Conduct ablation studies to quantify the individual contribution of interpretability-guided retrieval versus re-ranking, and test combined re-ranking strategies beyond pairwise comparisons.
- Perform statistical significance testing (e.g., paired tests across target proteins) on ranking improvements to demonstrate robustness.
- Quantify performance degradation for proteins with few known interactors (cold-start problem) to assess the limits of the core assumption.
- Implement controls for LLM data leakage, e.g., by using LLMs with pre-training cut-offs before v12 data or carefully curating training corpora.
- Provide a more precise algorithm or pseudo-code for the active region selection step in interpretability-guided retrieval to ensure reproducibility.

---

## zKQSyT7a7n

- GT: Reject (avg 6.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces VT-WM, a multi-task visuo-tactile world model for robot manipulation that integrates fingertip tactile sensing with vision. The core contribution is demonstrating that grounding world model imagination in contact physics via touch leads to significantly improved physical fidelity in autoregressive rollouts (33% better object permanence, 29% better compliance with motion laws) and, consequently, to substantially more reliable zero-shot planning on a real robot for contact-rich tasks, with success rate improvements of up to 35%.

## Strengths
- **Compelling Real-Robot Validation:** The paper provides strong, quantifiable evidence that the model's improved imagination translates to tangible robotic capability. In zero-shot real-robot experiments, VT-WM achieves significantly higher success rates than a vision-only baseline on contact-rich tasks like pushing, wiping, and stacking, demonstrating a clear path from model improvement to application impact.
- **Rigorous and Creative Evaluation of Imagination:** The evaluation of world model quality via object permanence and causal compliance is thorough. Using CoTracker and Fréchet distance provides a concrete, quantitative measure of physical coherence in rollouts, backed by statistical significance tests. The visualization of predicted tactile signatures alongside vision offers compelling qualitative evidence.
- **Sensible and Modern Architectural Design:** The model effectively leverages established, pre-trained encoders (Cosmos for vision, Sparsh-X for touch) within a transformer-based, action-conditioned dynamics model. The use of factorized spatio-temporal attention and a combined teacher-forcing/sampling loss is a standard and appropriate approach for this problem.

## Weaknesses
- **Unverified Mechanism for Planning Improvement:** The paper claims tactile grounding indirectly improves planning by providing better initial context for disambiguation. However, there is no ablation to confirm this mechanism—for instance, by running the VT-WM planner with zeroed-out or noisy tactile context to see if performance degrades. Without this, it remains unclear whether the tactile signal is actively used during planning or if the gains stem from other differences in training.
- **Incomplete Baseline for Data Efficiency Claim:** The data efficiency experiment compares VT-WM fine-tuning against a *single-task* Behavior Cloning (BC) policy. A more rigorous and convincing comparison would be against a **multi-task BC policy** trained on the same pre-training dataset, which would better isolate whether the efficiency gain comes from the world model framework or from multi-task pre-training.
- **Limited Analysis of Computational Cost:** The paper acknowledges that CEM planning with autoregressive rollouts is computationally expensive but provides no quantification (e.g., planning time per decision, scaling with horizon/particles). For a method aimed at real-world robotics, this is a significant practical limitation that should be analyzed to understand its feasibility.

## Nice-to-Haves
- Testing generalization to objects with novel physical properties (e.g., shape, texture) would strengthen claims about learning general contact dynamics.
- Implementing a closed-loop model-predictive control (MPC) scheme, rather than open-loop chunk execution, would enhance practical relevance.
- A deeper qualitative analysis of failure cases, categorizing why plans fail, would provide clearer directions for future work.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength (from Review 2):** "Clear Problem and Solution" — This is a generic strength applicable to many well-written papers and does not identify something specific this paper does exceptionally well.
- **Weakness (from Harsh Critic):** "Lack of ablation study isolating multi-task training from tactile modality" — The paper states both V-WM and VT-WM are trained on the same multi-task dataset (Section 4, Appendix A.0.1), making this criticism factually incorrect. The reported gains are explicitly relative to this multi-task V-WM baseline.
- **Weakness (from Harsh Critic):** "Action Space Ambiguity" — The action dimensionality (7) and formation are specified in Algorithm 1 and Section 3.2.3, so this criticism is addressed in the paper.
- **Weakness (from Spark Finder):** "Comparison to a late-fusion or tactile-only baseline" — This demands methodological exploration beyond the paper's stated scope and contribution of introducing and validating a joint visuo-tactile model. The proposed baselines are not standard in the field for this type of work.
- **Weakness (from Spark Finder):** "Statistical significance... for all key metrics" — The paper reports p-values for key task comparisons in Figs. 4 & 6. Demanding full statistical reporting for all tasks is an arbitrary rigor requirement beyond the norm for this community.
- **Weakness (from Harsh Critic/Spark Finder):** "Missing comparison to SOTA visual world model" — The V-WM baseline is a multi-task world model trained on the authors' dataset. Demanding comparison to a different model trained on different data is scope creep and introduces an unfair variable (dataset).

## Novel Insights
The paper provides a clear, evidence-backed insight: integrating tactile sensing into a multi-task world model specifically mitigates failure modes inherent to vision-alone models—namely, hallucinating object interactions under occlusion or contact ambiguity. This grounding directly translates to more physically plausible imagination and, crucially, to more reliable robot plans for tasks where maintaining contact is essential. The work convincingly shows that touch is not just an auxiliary signal but a core component for endowing world models with basic physical commonsense for manipulation.

## Suggestions
- Conduct a simple but critical ablation: run the VT-WM planner on the real-robot tasks while providing zeroed-out or random initial tactile latents. This will directly test whether the tactile context is functionally necessary for the observed planning improvements.
- Strengthen the data efficiency claim by comparing against a multi-task BC policy baseline (trained on the original dataset and fine-tuned on the 20 new demos) in Section 4.3.
- In the limitations/discussion, quantify the computational cost of CEM planning (e.g., time per planning step, latency) to provide a realistic assessment of the method's current deployability.

---

## Me0n0iESJY

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces OptMerge, a data-free method for merging Multimodal Large Language Models (MLLMs) that applies low-rank approximation to task vectors and optimizes the merged vector via a tailored loss. It also presents a comprehensive benchmark for MLLM merging, categorizing capabilities into VQA, Geometry, Chart, OCR, and Grounding, and explores merging across vision, audio, and video modalities.

## Strengths
- **Well-constructed benchmark**: The benchmark provides a fine-grained categorization of MLLM capabilities, includes both full fine-tuning and LoRA settings for two model families (InternVL2.5 and Qwen2-VL), and releases checkpoints and code, facilitating future research.
- **Effective method with supporting analysis**: OptMerge addresses noise and interference in task vectors through SVD-based denoising and robust optimization, motivated by an analysis of task vector properties (Fig. 2) and a theoretical upper bound linking fine-tuning dynamics to merging performance (Theorem 3.1).
- **Extensive empirical validation**: Experiments cover capability merging, modality merging, real-world checkpoints from Hugging Face, ablation studies, and scaling to 32B models, demonstrating that merging can match or surpass mixture training and integrate multiple modalities.

## Weaknesses
- **Incremental methodological novelty**: OptMerge primarily combines existing techniques—SVD denoising and the WUDI Merging optimization framework—with heuristic adaptations for full vs. LoRA fine-tuning, without a groundbreaking algorithmic advance.
- **Heuristic hyperparameter choices**: Key parameters, such as the rank size \(k\) (set to rank(task_vector)/number of tasks) and the optimization settings (e.g., Adam for InternVL, SGD for Qwen2-VL), are justified empirically but lack principled derivation, affecting reproducibility and generalizability.
- **Overstated claims about outperforming experts**: The paper claims the merged model "can even outperform expert MLLMs in their respective capabilities," but results (Tables 2 and 3) show the merged model typically performs between the base and expert models on individual tasks, not exceeding the best expert on its specialty. This misrepresents the more accurate strength: strong multi-task performance.
- **Missing comparison with MLLM-specific merging methods**: While the paper cites AdaMMS and UQ-Merge, it does not experimentally compare OptMerge against these MLLM-focused methods, weakening the claim of advancement in MLLM merging.
- **Limited modality merging evaluation**: Modality merging is evaluated only on audio-visual QA datasets (Table 5), without assessing whether the merged model retains unimodal capabilities (e.g., vision-only VQA), leaving robustness to catastrophic forgetting unverified.

## Nice-to-Haves
- Sensitivity analysis for hyperparameters like rank \(k\) and optimization iterations to guide users.
- Comparison of all merging methods on integrated benchmarks (e.g., MMMU, ScienceQA) to better demonstrate emergent multi-capability performance.
- Visualization of task vector interference (e.g., cosine similarity matrices) to directly illustrate OptMerge's denoising effect.
- Expanded discussion on limitations, such as sensitivity to fine-tuning regimes and the assumption of a common base model architecture.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Statistical significance reporting**: The paper reports single scores, but in model merging literature, single-run evaluation is common for large-scale benchmarks; demanding variance estimates is not standard practice here.
- **Demand for more scales or modalities**: The paper includes experiments up to 32B models and three modalities, which is sufficient for a benchmark; requesting more is scope creep.
- **Criticism of the Qwen2-VL mixture training baseline**: The paper acknowledges using Qwen2-VL-Instruct as a proxy due to practical constraints, and this does not invalidate the core findings.
- **Ablation study design**: The sequential ablation in Table 4, while showing interactions, is sufficient to demonstrate component contributions; a full factorial design is not required for the paper's claims.

## Novel Insights
The benchmark provides a structured framework for evaluating MLLM merging across distinct capabilities and modalities, highlighting the complementarity of modal information. The theoretical analysis (Theorem 3.1) offers a novel explanation for how fine-tuning extent (learning rate and iterations) influences merging performance, linking parameter drift to cross-task interference. However, these insights are largely within the paper's own contributions; no fundamentally new observations beyond the paper emerge from the reviews.

## Suggestions
- Add an experimental comparison with MLLM-specific merging methods like AdaMMS and UQ-Merge to solidify claims of advancement.
- Provide failure analysis or qualitative examples to illustrate cases where merging degrades performance, helping to identify limitations.
- Justify hyperparameter choices more rigorously, for instance by linking rank selection to singular value energy thresholds or validating coefficients on a held-out set.

---

## 7L7kmHHfgf

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes PIRN, a prototype-based reconstruction framework for few-shot multimodal anomaly detection (MAD). To address data scarcity, it introduces three components: Balanced Prototype Assignment (BPA) using optimal transport to prevent codebook collapse, Adaptive Prototype Refinement (APR) to update prototypes at test time, and Multimodal Normality Communication (MNC) for cross-modal knowledge exchange. PIRN achieves state-of-the-art performance on MVTec 3D-AD, Eyecandies, and Real-IAD benchmarks in few-shot settings while being computationally efficient.

## Strengths
- **Strong and consistent empirical performance in the target setting:** PIRN demonstrates significant and consistent improvements over strong baselines across three challenging benchmarks (MVTec 3D-AD, Eyecandies, Real-IAD) under various few-shot regimes (5, 10, 50 shots). The gains (e.g., +3.9 AUROC_I on MVTec 5-shot) are substantial and validate the core contributions.
- **Computational efficiency as a practical advantage:** The model is not only more accurate but also more efficient than recent SOTA methods. As shown in Table 4, PIRN outperforms FIND with 85% fewer FLOPs and 4.35x lower latency, making it highly suitable for real-world deployment.
- **Well-motivated and comprehensive design:** The three proposed components (BPA, APR, MNC) directly address identified weaknesses of existing methods (alignment and memory-based) in few-shot scenarios. The ablation studies (Tables 2, 3, 5, 6, 7) and visualizations (Figs. 1, 3, 4) provide thorough validation of each design choice and the method's interpretability.

## Weaknesses
- **Insufficient analysis of APR's robustness to anomalous contamination:** The claim that APR's OT-based context selection and GRU gating prevent prototype corruption by anomalies is not rigorously validated. While Figure 6 shows a qualitative example, a quantitative analysis measuring prototype drift or performance degradation when processing samples with increasingly large or pervasive anomalies is missing. This is a substantive concern for a method that adapts prototypes during inference.
- **Performance gap on Real-IAD image-level detection:** On the challenging Real-IAD D3 benchmark, PIRN achieves the best localization (AUROC_P) but is second-best in image-level detection (AUROC_I) to the tri-modal D3M method. The paper attributes this to D3M's extra modality but does not analyze *why* the prototype-reconstruction approach might be less sensitive to certain global anomalies that affect image-level scores. This indicates a potential limitation in the method's sensitivity for holistic anomaly detection.
- **Lack of granular ablation for the MNC module:** The ablation study removes the entire MNC module. A more detailed analysis isolating the contribution of its two stages (prototype alignment vs. cross-attention injection) and comparing it to simpler fusion baselines (e.g., feature concatenation) would better justify the architectural complexity and clarify the source of gains from cross-modal communication.

## Nice-to-Haves
- Exploring adaptive or learned prototype counts (K) per category, rather than a fixed heuristic choice, could improve generalization across objects with varying pattern complexity.
- A brief discussion on the societal impact of industrial anomaly detection systems, noting potential positive applications and considerations regarding bias in defining "normal" data.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Statistical significance not reported (confidence intervals)."** – The paper shows consistent gains across multiple benchmarks and few-shot settings. Reporting confidence intervals for large-scale benchmark evaluations is not a standard requirement in this field; the presented results are convincing evidence.
- **Weakness: "Training loss formulation not specified."** – The paper cites the specific method (Luo et al., 2025) for the "soft mining loss," which is sufficient for reproducibility given the detailed methodology provided elsewhere.
- **Weakness: "Need for cross-category few-shot evaluation."** – The paper's scope and experiments are focused on the standard and challenging class-specific few-shot learning paradigm. Evaluating on cross-category splits is a different, more extreme problem setting not required to validate the paper's claims.
- **Weakness: "Missing comparison with FIND in main table."** – FIND is compared directly in the efficiency analysis (Table 4), and its surface normal generation procedure is adopted. A full comparison in the main table would be beneficial but its absence does not invalidate the other extensive comparisons presented.
- **Weakness: "Ambiguity in GRU initialization for APR."** – The description in Sec. 3.3 and Appendix B.1 is clear: the prototype vectors themselves (learned during training) serve as the initial hidden states for the GRU at the first decoder layer during inference.

## Novel Insights
The paper's core novelty is the integration of a balanced optimal transport formulation for prototype learning with a gated, adaptive refinement mechanism and cross-modal communication at the prototype level, specifically tailored for data-scarce multimodal anomaly detection. This combination effectively addresses the codebook collapse problem in few-shot learning, allows the model to cautiously generalize to unseen normal variations at test time, and leverages cross-modal complementary information without relying on unreliable dense feature alignments. The demonstration that a compact, adaptive prototype codebook can outperform large static memory banks or complex alignment modules in few-shot settings is a significant insight.

## Suggestions
- Conduct a controlled experiment to quantitatively analyze APR's safety: measure the distance prototypes move when processing normal samples versus samples with anomalies of varying size/severity, and correlate this with performance.
- Perform an additional ablation for the MNC module, separating the effects of the prototype graph alignment stage and the cross-attention injection stage, and include a simple feature fusion baseline for comparison.
- Add a short discussion in the limitations section explicitly addressing the observed performance characteristic on Real-IAD D3 (superior localization but slightly lower image-level detection than the best method) and potential reasons.

---

## dCtkwjkK0E

- GT: Reject (avg 2.0)
- Predicted: N/A (3.7/10)
- Match: N/A

### Final Review

## Summary
This paper introduces an active learning framework for flow matching models, specifically targeting applications with expensive continuous labels (e.g., shape design). It proposes a theoretical analysis using piecewise-linear neural networks to explain how data affects model diversity and accuracy. Based on this, the authors derive two novel, competing query strategies: one to maximize diversity (QD) and one to maximize accuracy (QA). A weighted hybrid strategy balances this trade-off. Experiments on synthetic and real-world aerodynamic shape datasets demonstrate the strategies' effectiveness over active learning methods designed for discriminative models.

## Strengths
- **Novel Problem Formulation:** The paper rigorously addresses the underexplored problem of "active learning *for* generative models" (specifically flow matching), moving beyond the common paradigm of using generative models *for* active learning. This identifies and fills a clear gap.
- **Theoretically-Motivated Strategies:** The core query strategies (QD, QA) are directly derived from a formal, analytical framework that connects dataset composition to the diversity-accuracy trade-off in generation. This provides a principled foundation rare in active learning work.
- **Substantial and Relevant Empirical Validation:** Experiments are conducted on multiple non-trivial, real-world shape design tasks (airfoil, flying wing, starship) where label acquisition via numerical simulation is costly. Results consistently show the proposed strategies outperform adapted discriminative baselines on their respective targets, and the hybrid strategy provides tunable control.

## Weaknesses
- **Strong and Unverified Core Assumption:** The entire theoretical framework rests on the assumption that the flow matching model's neural network behaves as a piecewise-linear function. While motivated by citations on network condensation, this assumption is not empirically validated for the trained models in the paper. The generality of the theoretical claims and their applicability to standard flow matching architectures is therefore uncertain.
- **Limited Domain Demonstration and Baselines:** All real-world experiments are confined to the specific domain of aerodynamic shape design with low-dimensional continuous labels (1D to 4D). The paper's claims are framed generally, but efficacy on other domains (e.g., image generation) or with higher-dimensional conditions remains unshown. Furthermore, a simple but critical baseline—ongoing random sampling across active learning rounds—is omitted, making it harder to gauge the absolute improvement offered by the proposed strategies.
- **Insufficient Detail for Reproducing QD:** The diversity strategy QD (Eq. 4) combines three terms with weighting coefficients (α, β, γ) and uses a ∆_entropy_ term based on label clustering. The paper does not specify how these coefficients are set, how clusters are formed, or what distance thresholds are used. This lack of detail hinders reproducibility.

## Nice-to-Haves
- A discussion or simple experiment on the computational complexity and scalability of the distance calculations in data and label space for large unlabeled pools.
- Exploration of how the accuracy of the RBF network used for label prediction impacts the query strategies' performance.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** "The proof of Lemma 1 is notationally dense and somewhat difficult to follow." *(This is a subjective comment on presentation, not a substantive flaw in the paper's contribution.)*
- **Weakness:** "Training for 4,000,000 steps is computationally intensive... not discussed." *(The computational cost of training the final generative model is orthogonal to the active learning contribution, which focuses on reducing labeling cost. The query strategies themselves are model-agnostic and efficient.)*
- **Weakness:** "The paper does not compare against recent, sophisticated active learning methods..." *(The reviewer does not cite specific existing methods for active learning *for* generative models, making this an unverifiable demand. The paper clearly compares to relevant, adapted discriminative baselines.)*
- **Suggestion:** "Extend the theoretical analysis to more general network architectures..." *(This demands work beyond the paper's stated scope and contribution. The paper's analysis is explicitly built on the piecewise-linear framework.)*
- **Criticism:** "The description of the entropy term in QD is insufficiently detailed..." *(This point is valid and has been incorporated into the "Weaknesses" section as a reproducibility issue.)*

## Novel Insights
The paper's core novel insight is the data-centric explanation of the diversity-accuracy trade-off in conditional flow matching models. Through the piecewise-linear analysis, it demonstrates that data points sharing the same label primarily contribute to the *diversity* of generated samples for that condition, while data points with distinct labels improve the model's *accuracy* by reducing interpolation error across the condition space. This insight directly motivates two fundamentally conflicting query objectives (QD and QA), providing a principled perspective on a well-known challenge in generative modeling.

## Suggestions
- Provide an empirical validation of the piecewise-linear assumption, for instance by visualizing whether generated samples for intermediate conditions (not in the training set) approximate linear interpolations of nearby training data in the synthetic task.
- Include an ongoing random sampling baseline in the experiments to clearly establish the added value of the proposed query strategies.
- In the experiment section or an appendix, specify the values or tuning procedure for the coefficients (α, β, γ) in QD and provide details on the label clustering process (e.g., distance threshold) to ensure reproducibility. Releasing code would strongly support this.

---

## s7oURFZTQD

- GT: Reject (avg 3.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Multi-Grade Deep Learning (MGDL), a training framework that decomposes deep network optimization into sequential stages, each training a shallow network on residuals from previous grades. It provides theoretical convergence guarantees for gradient descent, shows that for single-layer ReLU grades the problem reduces to convex subproblems, and offers extensive empirical demonstrations of improved stability and performance across image reconstruction, CIFAR classification, and transformer-based time series regression.

## Strengths
- **Theoretical contributions**: Theorems 1 and 2 establish convergence of gradient descent for both SGDL and MGDL under smooth activations, and Theorem 3 proves that for single-layer ReLU grades, MGDL decomposes into a sequence of convex programs, extending convexification results to deep architectures.
- **Comprehensive empirical validation**: MGDL consistently outperforms SGDL in image regression, denoising, deblurring (PSNR gains of 0.42–4.23 dB), and CIFAR-100 classification (lower training loss), with evidence from fully connected networks, CNNs, and transformers (Tables 1-3, Figures 10-19).
- **Insightful mechanistic analysis**: Eigenvalue analysis of the GD iteration matrix shows that MGDL keeps eigenvalues within (-1,1), leading to stable convergence, while SGDL eigenvalues often exit this range, causing oscillations (Section 7, Figures 4-6).
- **Demonstrated robustness**: MGDL is empirically more robust to learning rate choices in synthetic and image regression tasks, maintaining performance over a wider range than SGDL (Section 6, Figure 20).
- **Novel extension to transformers**: Application to Multi-Grade Transformers (MGT) shows improved generalization on synthetic and financial time series regression, with test error reductions of 84% and 80%, respectively (Section 8, Tables 4-5).

## Weaknesses
- **Theory-experiment mismatch** — Convergence and eigenvalue analyses (Theorems 1, 2, 4) assume twice or thrice continuously differentiable activations, but all experiments use ReLU, which is non-smooth. This undermines the theoretical relevance to the empirical results and leaves the guarantees inapplicable to the presented settings.
- **Limited scope of convexity result** — Theorem 3 only applies to single-layer, bias-free ReLU grades, yet experiments use multi-layer grades with biases (e.g., in image tasks and transformers). The practical relevance of this theoretical insight to deep MGDL is unclear and unsubstantiated.
- **Unclear architectural parity** — The paper does not ensure that SGDL and MGDL models have comparable total parameters or depth (e.g., in image regression, SGDL: (2,1,128,8) vs. MGDL: (2,1,128,2,4)), raising concerns that improvements may stem from capacity differences rather than the training strategy.
- **Non-standard loss for classification** — CIFAR-100 experiments use mean squared error (MSE) instead of the standard cross-entropy loss without justification, which may disadvantage SGDL unfairly and limits the interpretability of classification performance.

## Nice-to-Haves
- Ablation studies on the number of grades and depth per grade to understand sensitivity and design choices.
- Extended learning rate robustness analysis to classification tasks (beyond regression).
- Computational efficiency comparison including wall-clock time and FLOPs to substantiate scalability claims.
- Comparison to related progressive training methods (e.g., greedy layer-wise training) to clarify novelty, though the paper focuses on MGDL vs. SGDL.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Statistical significance tests or multiple runs**: While valid, single-run evaluations are common in the field for the tasks presented, and the consistency across experiments mitigates this concern.
- **Demand for large-scale benchmarks (e.g., ImageNet)**: The paper claims broad empirical improvements on established tasks; large-scale benchmarks are a future direction rather than a core flaw.
- **Insufficient hyperparameter tuning details**: The use of Adam with described architectures is typical, and the learning rate study partially addresses robustness.
- **Requests for visualizations like loss landscapes or residual targets**: These would enhance the paper but are not required for the core claims.

## Novel Insights
The paper provides a novel convexification of deep ReLU networks through multi-grade decomposition for single-layer grades, extending prior work on shallow networks to a sequential setting. The eigenvalue analysis offers a mechanistic explanation for MGDL's stability by linking spectral properties of the GD iteration matrix to optimization dynamics, showing that MGDL confines eigenvalues within (-1,1) while SGDL does not.

## Suggestions
- Address the theory-experiment mismatch by either adapting the theory for non-smooth activations (e.g., via subgradients) or using smooth approximations in experiments to align with assumptions.
- Ensure fair comparisons by matching model capacities (e.g., reporting parameter counts and depths) or justifying architectural choices to isolate the effect of the training strategy.
- Report test accuracy for CIFAR classification tasks and consider using cross-entropy loss for standard benchmarking, or justify the use of MSE.
- Clarify in the discussion how the convexity result for single-layer grades relates to practical multi-grade networks, acknowledging limitations and potential extensions.

---

## 2EQPpEZtEK

- GT: Reject (avg 3.3)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
DISTAR introduces a zero-shot text-to-speech framework that couples an autoregressive language model with a masked diffusion model, operating entirely in a discrete residual vector quantization (RVQ) token space. The hybrid design aims to achieve blockwise parallelism, mitigate autoregressive exposure bias, and provide explicit controllability through features like RVQ layer pruning. The paper demonstrates state-of-the-art or competitive results on standard zero-shot TTS benchmarks.

## Strengths
- **Strong empirical performance:** DISTAR achieves leading or competitive scores on LibriSpeech-PC and Seed-TTS in key metrics (WER, SIM, UTMOS) and subjective evaluations (CMOS, SMOS), validating its effectiveness.
- **Practical and well-motivated design:** The method includes several impactful features: stochastic RVQ layer truncation during training enables test-time bitrate and compute control without retraining; the fully discrete pipeline eliminates the need for a separate duration predictor or forced alignment.
- **Comprehensive ablation studies:** The paper provides ablations on patch size, classifier-free guidance strategies, and decoding methods, offering clear justification for design choices.

## Weaknesses
- **Missing critical ablation to validate the hybrid approach:** The paper lacks a direct comparison to a strong, pure-autoregressive (AR) baseline trained on the same RVQ tokens and dataset. This is essential to substantiate the core claim that coupling AR with masked diffusion provides benefits over pure AR modeling for RVQ-based TTS.
- **Insufficient evidence for computational efficiency claims:** While the paper states DISTAR maintains "inference cost close to its continuous counterpart," it provides no direct comparison of inference latency, throughput, or FLOPs against key baselines (e.g., DiTAR, F5TTS). The main quality comparison uses DISTAR with NFE=24 versus DiTAR with NFE=10, making the efficiency claim unsupported.
- **Technical ambiguity in the training formulation:** The use of overlapping context windows (stride S < patch size P) is not fully reconciled with the likelihood factorization in Equation (1). It is unclear how tokens within overlapping conditioning contexts are handled during training, which could affect reproducibility and the theoretical clarity of the training objective.
- **Reliance on heuristic decoding strategies:** The proposed layer-wise and position-wise temperature shaping, along with a hybrid sampling schedule, are presented as empirical fixes for a "tail-first bias." While they improve results, their necessity suggests a potential modeling bias or optimization issue, and they are not derived from or justified by the underlying masked diffusion framework.
- **Statistical significance of subjective improvements is unclear:** The reported CMOS improvement (0.22 ± 0.13 for DISTAR vs. 0.01 ± 0.12 for F5TTS) has overlapping confidence intervals. Without statistical significance testing, the claimed superiority in naturalness is not fully substantiated.

## Nice-to-Haves
- More thorough analysis of hyperparameter sensitivity (e.g., stride S, number of diffusion steps) beyond the patch size ablation.
- Evaluation on challenging, out-of-distribution or very long-form prompts to better validate the claimed robustness.
- Deeper qualitative or quantitative analysis of the division of labor between the AR drafter and the diffusion refiner (e.g., error rates by token position/layer).
- Clarification in Figure 2 that the x-axis "RVQ Layers" refers to the total number of layers used during inference (i.e., 9 - ℓ pruned layers).

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Criticism that the iterative masked diffusion decoding is "heuristic" and lacks theoretical linkage:** The method follows established masked diffusion frameworks (e.g., LLaDA, D3PM) and the described confidence-based decoding is a standard practical approach for such models.
- **Criticism about data dependency and reproducibility due to use of the proprietary Emilia dataset:** While a limitation, the use of large-scale, curated datasets is common in contemporary TTS research and does not constitute a methodological flaw specific to DISTAR.
- **Criticism demanding a user study or theoretical proofs:** These are not standard requirements for an algorithmic/engineering contribution in this domain.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add an experiment comparing DISTAR to a strong pure-autoregressive RVQ baseline (e.g., a VALL-E 2 style LM trained on the same tokens) to isolate the benefit of the hybrid AR-diffusion design.
- Include standardized inference latency/throughput benchmarks (e.g., real-time factor, tokens/sec) comparing DISTAR to key baselines under controlled hardware settings.
- Clarify the training objective and how overlapping context windows are handled in the likelihood computation (Section 3.1.1).
- Perform statistical significance testing (e.g., t-test) on the subjective evaluation scores to confirm the reported improvements.

---

## QryPmx2MNh

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a novel task of automatically discovering learning-friendly orders (permutations) for the decoder's target sequence to improve Transformer training on arithmetic reasoning. The core method uses early training dynamics ("loss profiling") to identify permutations where loss drops quickly and employs a two-stage hierarchical search to navigate the factorial search space. Experiments on synthetic, order-sensitive arithmetic tasks demonstrate the method can recover optimal orders from billions of candidates and rediscover a known beneficial order for multiplication.

## Strengths
- **Novel Problem Formulation and Insight**: The paper clearly defines the underexplored problem of optimizing the order of decoder tokens (an "unraveled" chain of thought) for learning efficiency. The key insight—that orders which are easier to learn exhibit faster loss drop in early training—is a fresh and clever application of easy-to-hard learning dynamics.
- **Effective and Efficient Method**: The proposed hierarchical search (global block ordering + local intra-block refinement) is a pragmatic and necessary solution to the combinatorial challenge. It scales effectively, demonstrated by finding a single solution from ~6 billion permutations for sequences of length 13 and, with a structured initialization, up to length 40.
- **Strong Empirical Validation with Well-Designed Tasks**: The paper constructs convincing, non-injective arithmetic tasks (RELU, SQUARE-19, INDEX) where the forward order is trivially learnable, providing a clean testbed. The method consistently recovers high-performing orders on these tasks and successfully rediscovers the known reverse-digit order for multiplication, providing robust proof-of-concept.

## Weaknesses
- **Narrow Scope and Uncertain Generality**: All experiments are on synthetic, deterministic arithmetic tasks with fixed-length outputs. While the paper's claims are appropriately scoped to arithmetic, the framing ("chain of thought") invites broader implications. The method's applicability to more complex, real-world reasoning tasks (e.g., natural language, symbolic logic) remains entirely unvalidated and is a significant limitation for the perceived impact.
- **Incomplete Analysis of Method's Core Mechanism**: The paper relies on the empirical correlation between early loss drop and final performance but does not provide a deeper analysis of *why* this correlation holds or under what conditions it might break. For instance, on the hardest INDEX task, even the top-ranked orders yield near-zero success rates (Sec. 5.4), suggesting the signal can be weak. A quantitative analysis (e.g., correlation coefficients) of this relationship across tasks would solidify the method's foundation.
- **Limited and Underdeveloped Baseline Comparison**: The main text lacks a rigorous comparison to alternative search strategies. While an Evolutionary Strategy (ES) baseline is presented in Appendix C, its results are not quantitatively compared to the proposed method in terms of search efficiency (e.g., number of model trainings or wall-clock time to find a good order). This makes it difficult to assess the true advantage of the hierarchical loss-profiling approach.

## Nice-to-Haves
- A pilot experiment on a non-arithmetic, multi-step reasoning task (even a simple symbolic one) to suggest broader applicability.
- A more detailed analysis of the properties of the discovered orders (e.g., their alignment with the task's causal graph) beyond final accuracy.
- A sensitivity analysis for key hyperparameters like the number of profiling epochs or block size in the hierarchical search.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness: "Lack of theoretical proof for the early-loss assumption."** This is an empirical paper; demanding theoretical proofs for a heuristic based on training dynamics is not a standard requirement.
- **Weakness: "Results are from single runs without statistical significance."** Single-run evaluation is common for large-scale benchmarks in this area; the paper's results are consistent and clear across tasks.
- **Weakness: "The description of the hierarchical search is hard to follow."** While the text is dense, Figure 4 and the step-by-step description in Section 4 provide sufficient clarity for an expert reader.
- **Weakness: "The method fails for longer sequences without structured initialization."** This is correctly presented in the paper (Sec. 5.5) as a limitation and scaling strategy, not a hidden flaw.
- **Nitpick: "Inconsistent notation between π(Y) and YP."** This is a minor presentation issue that does not affect understanding.

## Novel Insights
The paper's most novel insight is the operationalization of "easy-to-hard" learning dynamics to solve a combinatorial search problem. By training on a mixture of permuted sequences and using the *speed* of early loss reduction as a proxy for permutation quality, it turns an intractable search into a manageable filtering process. This provides a fresh perspective on how training dynamics can be repurposed for meta-optimization of sequence structure itself.

## Suggestions
- Add a quantitative analysis (e.g., a scatter plot and correlation score) showing the relationship between the loss after a few profiling epochs and the final task success rate for a large set of permutations. This would directly validate the core heuristic.
- Strengthen the baseline comparison by tuning the ES baseline more thoroughly and reporting a direct efficiency comparison (e.g., number of model forward/backward passes or GPU hours) against the proposed method to reach a target performance threshold.

---

## mDuTDAK6KU

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (3.3/10)
- Match: N/A

### Final Review

## Summary
KOALA introduces a novel adversarial detector that flags inputs as attacked when predictions from two complementary similarity metrics—KL divergence (sensitive to dense, low-amplitude perturbations) and a custom L0-based score (sensitive to sparse, high-impact changes)—disagree. The method requires only lightweight fine-tuning on clean images to align embeddings and is accompanied by a formal detection guarantee under a set of assumptions and a sufficient condition on prototype separation.

## Strengths
- **Novel detection principle**: The core idea of forcing detection via disagreement between two geometrically motivated, complementary metrics (KL and L₀) is conceptually fresh and well-motivated by an analysis of perturbation types. This offers a new perspective beyond single-metric or semantics-driven detectors.
- **Theoretical grounding**: The paper provides a formal theorem (Theorem 1) with a detailed proof (Appendix B) that guarantees detection when a sufficient "coordinate gap" exists between class prototypes. This effort to provide rigorous guarantees is rare and valuable in the empirical landscape of adversarial detection.
- **Practical and lightweight design**: KOALA operates as a plug-in detector without adversarial training, architectural changes, or semantic priors. The required fine-tuning uses only clean images and a composite loss, making it easily deployable.

## Weaknesses
- **Theoretical assumptions limit practical guarantees**: The theorem’s guarantees rely on strong assumptions (A1–A3), particularly A3 (coordinate-wise perturbation bound |δ_i| ≤ (3/2)|p_i*|) and the implicit reliance on a Lipschitz constant to link pixel-space and feature-space bounds (A2). These are not empirically validated and may not hold broadly, making the theoretical guarantee conditional and less applicable to real-world deployments.
- **Unclear operationalization of “theorem-compliant” criterion**: Experiment 1 splits data based on whether “the sufficient inter-class prototype separation” holds, but the paper does not specify the threshold Γ_i(ϵ) or the exact procedure for determining compliance. This lack of reproducibility undermines the empirical validation of the core theorem.
- **Missing comparison to state-of-the-art detectors**: The evaluation compares KOALA only to ablations of itself (different metric combinations). Without benchmarking against established detection methods (e.g., LID, Mahalanobis, feature squeezing), it is impossible to assess its relative performance and contribution to the field.
- **Limited attack evaluation**: Experiments are confined to ℓ∞-bounded attacks (PGD, CW, AutoAttack). The detector’s efficacy under other threat models (e.g., ℓ₂, ℓ₁) or against adaptive attacks specifically designed to evade the two-metric disagreement is unexplored, leaving its general robustness in question.

## Nice-to-Haves
- Sensitivity analysis for hyperparameters τ (L₀ threshold) and ϕ (smoothing parameter), with guidance on setting them.
- Extension to additional architectures (e.g., Vision Transformers) and larger-scale datasets (e.g., full ImageNet) to further support the “plug-and-play” claim.
- Visualization of feature-space perturbations or prototype/embedding structures to illustrate the dense vs. sparse attack patterns and the effect of fine-tuning.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Strength**: “The paper is well-written” – generic.
- **Weakness**: “The L₀ metric definition is unconventional” – the paper clearly defines it as a sparse-change detector; the design choice is justified.
- **Weakness**: “The proof is extremely long and dense” – not a substantive critique; detailed proofs are appropriate for an appendix.
- **Weakness**: “Missing a dedicated limitations section” – while a limitations discussion would strengthen the paper, its absence is not a core flaw; the assumptions and non-compliant results implicitly highlight limitations.
- **Weakness**: “The loss weights ω_L₀=0.9, ω_KL=0.1 are not justified” – the paper notes L₀ is harder to optimize; an ablation would be nice but is not essential.

## Novel Insights
The paper’s key novel insight is that adversarial perturbations under an energy budget tend to manifest as either dense, low-amplitude shifts or sparse, high-impact changes, and that these two types can be captured by two complementary metrics (KL divergence and an L₀-based score). By forcing detection when predictions from these metrics disagree, the method creates a mutually exclusive condition that can be formally guaranteed under certain geometric separations in the embedding space. This geometric perspective on detection is a distinct contribution beyond purely empirical or semantics-driven approaches.

## Suggestions
- Clearly define the operational criterion (e.g., the threshold Γ_i(ϵ)) used to split “theorem-compliant” and “non-compliant” samples in Experiment 1, ensuring reproducibility.
- Add a comparative evaluation against state-of-the-art adversarial detectors (e.g., LID, Mahalanobis, feature squeezing) on the same benchmarks to establish KOALA’s relative performance.
- Evaluate the detector under additional threat models (ℓ₂, ℓ₁ norms) and consider testing against an adaptive attacker aware of the two-metric disagreement mechanism.

---

## U6ROetm5nW

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper proposes new data structures for fast Kernel Density Estimation (KDE) using asymmetric Locality-Sensitive Hashing (LSH). The main contribution is achieving the first explicit time-space tradeoffs for KDE, yielding significantly improved query time exponents (e.g., ≈1/µ^0.051) compared to prior symmetric LSH approaches, at the cost of higher space complexity (e.g., ≈1/µ^4.15). For the linear-space regime, it also improves the prior data-independent query exponent from 0.25 to 0.1865.

## Strengths
- **Novel Application of Asymmetric LSH:** The paper creatively and rigorously adapts advances in asymmetric LSH to the KDE problem, moving beyond the symmetric LSH framework that limited prior tradeoffs. This constitutes a meaningful extension of the technical toolkit.
- **First Time-Space Tradeoffs for KDE:** The paper provides the first known parameterized tradeoff (Theorem 2) between query time (exponent ξ(δ)) and space (exponent 1+δ) for Gaussian KDE. This is a clear theoretical contribution.
- **Improved Theoretical Bounds:** The work delivers concrete improvements: a dramatically lower query exponent (down to ~0.05) in high-space regimes, and a non-trivial improvement (0.1865) in the linear-space, data-independent setting over the previous best of 0.25 (Charikar et al. 2020).

## Weaknesses
- **Prohibitively High Space for Best Query Time:** The most impressive query time exponent (~0.05) requires space with exponent ~4.15. For typical small µ (e.g., 10^-4), this implies a space factor of ~10^-16.6, which is astronomically large and renders this point on the tradeoff curve impractical, significantly limiting the real-world impact of the extreme performance claim.
- **Heavy Reliance on Numerical Optimization without Closed Forms:** The core results, including the key exponents (0.051, 0.1865) and the tradeoff function ξ(δ), are obtained via numerical optimization of complex expressions (Eq. 10, Sec. 5). While acceptable in theory, the lack of closed-form bounds or an analytical characterization reduces elegance and makes it difficult to gain intuitive insight into the precise dependencies or verify the results independently beyond running the provided script.
- **Dense Presentation Obscures Intuition:** The technical sections (Sec. 4, App. C) are very dense, with a proliferation of variables and minimal intuitive scaffolding. The connection between the high-level idea (using asymmetric LSH) and the formal optimization is difficult to follow, hindering understanding and verification. A table of notation and more explanatory commentary would greatly help.

## Nice-to-Haves
- A brief discussion contextualizing the astronomical space costs (e.g., what absolute memory would be required for a typical n and µ) to help readers assess the practical relevance of different points on the tradeoff curve.
- A more intuitive explanation or a simple running example illustrating why asymmetric LSH unlocks this tradeoff and why the optimization leads to the observed query time plateau around exponent 0.05.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Demand for Empirical Experiments/Comparisons:** The request for empirical validation on real datasets or comparisons to tree-based methods is a scope creep for this purely theoretical, asymptotic contribution. ICLR accepts theoretical papers without experiments.
- **Criticism of "Nice Range" Assumption:** The paper explicitly handles boundary scales (j < c0J, j > (1-c1)J) by using the prior data structure (Lemma 27) and argues their contribution is negligible (Sec. 3, Thm. 16). The criticism that this lacks justification is addressed in the paper's framework.
- **Complaint about Numerical Results in Abstract:** The abstract correctly presents the results as informal theorems; the numerical nature is clarified in the main text (Sec. 5). This is not a substantive weakness.
- **Formatting Nitpicks about Figure 1:** While the caption in the provided text has artifacts, critiquing figure formatting is a minor presentation issue, not a core weakness.

## Novel Insights
The paper's core novel insight is that asymmetric LSH constructions, which allow independent tuning of query and space exponents for approximate near neighbor search, can be leveraged within the established KDE framework to create the first explicit time-space tradeoffs. A secondary insightful observation is the identification of a plateau in the tradeoff curve: even with arbitrarily large polynomial space, the query time exponent cannot be driven to zero (constant query time) with this approach, which the paper informally argues is a barrier inherent in current ANN technology.

## Suggestions
- **Improve Figure and Caption Clarity:** Ensure Figure 1 is presented clearly in the final version, with properly labeled axes and a legend that explains how to interpret the tradeoff curves for different δ values.
- **Add Intuition for the Optimization:** Include a high-level, more intuitive walkthrough of the optimization problem in Section 4 (perhaps with a simplified example) to help readers understand where the bottleneck arises and why the maximum query time occurs at an internal distance scale.

---

## ngOOlatCK6

- GT: Reject (avg 5.3)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces the conditional causal bandit problem, where arms are single-node conditional interventions in a known causal graph aimed at maximizing a target variable Y. Its core contribution is a graphical characterization of the minimal set of nodes (mGISS) guaranteed to contain the optimal intervention node, and a linear-time algorithm (C4) to compute this set. Empirical results demonstrate significant search space reduction and accelerated convergence when integrated into a bandit algorithm.

## Strengths
- **Novel and well-motivated problem formulation.** The paper clearly argues for the importance of conditional over hard interventions in real-world decision-making (e.g., medical treatment) and rigorously defines the novel setting of single-node conditional causal bandits.
- **Strong theoretical foundation.** The graphical characterization via Λ-structures and the equivalence between conditional and deterministic atomic intervention superiority (Proposition 4) are elegant and provide the basis for the minimal search space result (Theorem 13). Proofs are provided and appear rigorous.
- **Efficient and correct algorithm.** The proposed C4 algorithm runs in O(|V|+|E|) time, is simple to implement using the connector concept, and is proven correct (Theorem 16), making the theoretical contribution practically usable.

## Weaknesses
- **Strong assumption of no latent confounders.** The entire analysis assumes causal sufficiency. This is a significant limitation for real-world applicability, as latent confounding is common. While acknowledged as future work, its absence curtails the current contribution's direct utility.
- **Lack of theoretical bounds on pruning effectiveness.** The paper provides no theoretical guarantees on the size of mGISS relative to the set of all ancestors of Y. Without such bounds, it is difficult to predict the utility of the method for an arbitrary graph class.
- **Empirical evaluation could be broader and more comparative.** The synthetic graph analysis uses only the Erdős-Rényi model; evaluation on other generative models (e.g., scale-free) would strengthen generalizability claims. The bandit experiments show improvement over a brute-force search but do not compare against natural baselines (e.g., intervening only on the parents of Y), making the added value of the full characterization less clear.
- **Experimental details in the main text are sparse.** While code is provided, key details for reproducibility—such as the exact specification of the synthetic structural equations and reward functions for the bnlearn datasets, and the hyperparameters for the CondIntUCB algorithm—are insufficiently detailed in the paper itself.

## Nice-to-Haves
- Integration of the mGISS pruning step with more advanced causal or contextual bandit algorithms to demonstrate its utility beyond a simple UCB adaptation.
- Runtime measurements on very large graphs to empirically confirm the linear-time scalability in practice.
- A sensitivity analysis investigating how the choice of the conditioning sets **Z_X** (within the assumed constraints) impacts the bandit performance.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** The abstract does not explicitly mention the assumption of no latent confounders. *This limitation is clearly stated in the introduction.*
- **Weakness:** The introduction provides insufficient intuition for why single-node interventions are "more challenging." *The paper explains that with multi-node interventions one can intervene on all parents of Y, which is not possible in the single-node case, justifying the complexity.*
- **Weakness:** The paper does not discuss the complexity of the policy space after node pruning. *The paper's contribution is precisely to prune the node space; the subsequent policy selection for a given node is a separate problem explicitly left to standard bandit algorithms.*
- **Nitpick:** Concerns about notation clarity (e.g., E_n f̄_Y^{do(X=g(Z_X))}(n)). *The notation is standard in the causality literature and is further clarified in the appendix.*

## Novel Insights
The equivalence established between conditional intervention superiority and deterministic atomic intervention superiority (Proposition 4) is a key insight that simplifies the problem and enables the subsequent graphical analysis. Furthermore, the characterization of the minimal search space via Λ-structures (Theorem 12) provides an intuitive and computationally tractable graphical criterion that directly leads to the efficient C4 algorithm.

## Suggestions
- Provide a theoretical bound or a worst-case family of graphs illustrating the potential size of mGISS relative to An(Y)\{Y\}.
- Enhance the bandit experiments by including a comparison to a baseline that only considers intervening on the parents of Y, to better isolate the value of the full mGISS characterization.
- Broaden the synthetic graph analysis to include other common graph models (e.g., preferential attachment networks) to strengthen claims about performance on realistic structures.

---

## VgVeQpagf7

- GT: Reject (avg 4.7)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces SPS and SPS+, algorithms for generating differentially private synthetic datasets by distilling summary statistics from a sensitive dataset using a public pre-trained model. SPS+ achieves higher accuracy than DP-SGD on CIFAR-10 and CIFAR-100 at strict privacy budgets (ε=1), marking the first time a generation-based method surpasses gradient-based approaches in image classification, while enabling flexible downstream use like ensembling and federated learning without additional privacy cost.

## Strengths
- **State-of-the-art accuracy on standard benchmarks**: SPS+ outperforms DP-SGD on CIFAR-10/100 across multiple ε values, e.g., 96.2% vs. 94.8% on CIFAR-10 at ε=1 (Table 1), validating its core claim.
- **Demonstrated practical flexibility**: The synthetic dataset can be reused without extra privacy loss, enabling ensembling, federated learning (asynchronous, no synchronization rounds), and continual learning—capabilities often infeasible with DP-SGD due to composition constraints, as shown in Figures 2, 5, and Table 1.
- **Novel algorithmic contributions**: Key innovations include adapting D3S to DP by removing reliance on a privately trained model, privatizing only summary statistics, and introducing multistage clipping and grouped pseudo-classes that significantly boost performance in high-privacy regimes (Table 8).

## Weaknesses
- **Dependence on public pre-trained models**: Performance hinges on the availability of a relevant public model; while tested on domain-shifted data (CAMELYON17), extreme mismatches or lack of public data could limit applicability, and the method does not provide a fallback, affecting real-world deployment.
- **Incomplete justification for grouped pseudo-classes (GPC)**: The claim that GPC helps only through "optimization dynamics" (Section 4.2) is not substantiated with analysis or experiments, leaving a key component poorly understood and raising questions about its robustness.
- **Hyperparameter tuning under DP constraints**: Critical parameters like projection dimensions \(D_G, D_C\) are chosen arbitrarily for different settings (Table 10), and the paper does not discuss how to select them in a privacy-preserving way, which is essential for practical use where tuning consumes privacy budget.
- **Federated learning with overlapping client data**: The federated experiment assumes disjoint data partitions (Section 5.5); if client data overlaps, privacy guarantees would require composition across clients, an unaddressed scenario that impacts real-world applicability.

## Nice-to-Haves
- More detailed efficiency comparison with DP-SGD, including wall-clock time and memory usage on comparable hardware, to better contextualize trade-offs.
- Extension to additional image benchmarks like downsampled ImageNet to further validate scalability beyond CIFAR.
- Analysis of synthetic data quality beyond FID, e.g., per-class accuracy or diversity metrics, to characterize limitations more thoroughly.

## Removed Points
These points are flagged to be removed, treat them with caution:
- Formatting issues in figures (e.g., "Col1" in captions) are parser artifacts, not paper flaws.
- Criticism about missing experiments on non-image modalities is outside the paper's stated scope on image classification.
- Demand for deeper privacy-utility trade-off analysis using distributional metrics (e.g., Wasserstein distance) is not standard in DP image classification literature where accuracy is the primary metric.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Provide an intuitive explanation or simple experiment to illustrate why grouped pseudo-classes improve optimization but not direct mean estimation, enhancing methodological clarity.
- Discuss strategies for hyperparameter selection under DP, such as using public validation data or allocating a small privacy budget for tuning, to guide practitioners.
- Clarify the privacy implications for federated learning when client data may overlap, and suggest how SPS could be adapted (e.g., via centralized generation with composed guarantees).

---

## 41JeFWdVFa

- GT: Reject (avg 4.7)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper proposes LDP, a lightweight denoising autoencoder plug-in designed to improve the generalization of single-image super-resolution models to unseen degradations. LDP models the degradation process by conditioning on LR high-frequency components and enforces LR cyclic consistency. It operates in two modes: as an auxiliary training loss and as an inference-time post-processing module for diffusion models via posterior sampling. Experiments across GAN-, transformer-, Mamba-, and diffusion-based SR models show consistent improvements on synthetic and real-world benchmarks.

## Strengths
- **Comprehensive and convincing evaluation across diverse architectures and degradation types.** The paper demonstrates performance gains for four distinct SR model families (FeMaSR, SwinIR, MambaIR, StableSR) across five synthetic degradation categories and three real-world datasets, using both reference and non-reference metrics. Evidence: Tables 3, 4, and 5 show consistent improvements (e.g., StableSR gains up to +2.16 PSNR on hybrid degradations).
- **Effective and non-trivial degradation modeling.** The method is shown not to collapse into simple downsampling, a common failure mode. Evidence: Table 2 shows LDP’s predicted LR images have significantly lower similarity to downsampled SR images than strong baselines like DRN, while Table 1 shows it achieves strong LR prediction metrics across diverse degradations.
- **Practical, flexible, and lightweight design.** With only ~642K parameters, LDP functions as a plug-in training loss and an inference-time correction module, making it widely applicable. Evidence: Sections 3.3 and 4 demonstrate successful application in both fine-tuning and posterior sampling settings, and Table 14 confirms its low memory overhead compared to alternatives.

## Weaknesses
- **Missing direct performance comparison with the most relevant contemporary baseline (Lway).** The paper compares LDP’s training cost to Lway but does not provide a direct comparison of super-resolution performance on the same benchmarks. Since Lway is a directly competing degradation-modeling method for SR generalization, its absence from Tables 3 and 4 undermines the claim that LDP is a superior plug-in. Evidence: Lway is discussed in related work and compared only for training efficiency in Table 14.
- **Insufficient discussion of the computational trade-off for inference-time posterior sampling.** While LDP improves consistency, its use in posterior sampling can incur a substantial inference time penalty (e.g., ~9x slowdown for StableSR in Table 13). The paper buries this analysis in the appendix and does not adequately discuss whether the modest quantitative gains (e.g., often <0.01 in CLIPIQA for UPSR in Table 5) justify this cost in the main limitations. This is a practical concern for deployment.
- **Conceptual reliance on prior work without full ablation of core components.** The core idea of using noise alignment to bridge HR and LR features builds directly on DR2 (Wang et al., 2023b). While the conditional DAE formulation is a useful instantiation, the ablation study (Table 6) does not isolate the contribution of the proposed patch-dependent noise or the necessity of the Degradation Prediction Module versus a simpler conditioning mechanism. This makes it harder to attribute gains specifically to the novel design choices.

## Nice-to-Haves
- **Evaluation on a broader suite of real-world degradation benchmarks** (e.g., NTIRE challenges, RealBlur) to further substantiate claims of robust generalization.
- **Deeper failure mode analysis** beyond the noted texture-artifact trade-off for FeMaSR, such as per-image performance analysis or identifying degradation types where LDP underperforms.
- **Ablation comparing the conditioning signal *yhf* to other potential signals** (e.g., learned embeddings) to more rigorously justify its design.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism that the diffusion alignment property is not explained.** The paper provides a citation to Wang et al. (2023b) and an intuitive statement in Section 3.1; a full derivation is not required for an empirical paper.
- **Criticism about vague training details.** Hyperparameters for fine-tuning each model are provided in Appendix D, which is sufficient for reproducibility.
- **Criticism about formatting artifacts in equations.** These are parser issues, not paper problems.
- **Suggestion to integrate training and inference modes for a single model.** This is an interesting next step but not required to validate the current contribution.
- **Demand for comparisons to DANv2, DDNM, GRL, or MAN.** The paper’s scope is a plug-in for existing models, not a new SOTA SR method; comparisons to degradation-modeling plug-ins (like Lway) are more relevant than full SR methods.

## Novel Insights
The paper’s primary novel insight is the formulation of a lightweight, conditional denoising autoencoder as a dual-mode plug-in for SR generalization. By using LR high-frequency components as a condition and integrating patch-dependent noise, it provides a practical and efficient way to enforce cycle consistency across diverse SR architectures. While the underlying principles of degradation modeling and consistency are known, the specific instantiation as a tiny, trainable plug-in applicable during both training and inference is a useful engineering insight with demonstrated empirical benefits.

## Suggestions
- Add a direct super-resolution performance comparison between LDP and Lway (the most relevant baseline) on the same synthetic and real-world benchmarks used in Tables 3 and 4 to firmly establish superiority.
- Move the discussion of computational cost for posterior sampling (from Appendix F, Table 13) into the main limitations section, with a clear analysis of the fidelity-versus-speed trade-off.
- Consider a supplementary ablation experiment that trains an LDP variant without patch-dependent noise (i.e., using a global timestep) to quantify the importance of this design choice.

---

## X2yzXtH4wp

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Ambig-SWE, an interactive benchmark for evaluating LLM agents on underspecified software engineering tasks. By creating underspecified variants of SWE-Bench Verified and simulating a user with full information, the authors decompose agent performance into three key capacities: detecting missing information, asking targeted clarification questions, and integrating acquired information to solve tasks. Their empirical study across proprietary and open-weight models reveals that while interaction significantly improves task success, most models default to non-interactive behavior and are poor at autonomously detecting underspecification.

## Strengths
- **Structured, diagnostic evaluation framework:** The paper cleanly isolates and measures three distinct agent capabilities (detection, questioning, integration) across controlled settings (Full, Hidden, Interaction). This multi-stage breakdown provides a valuable blueprint for the community to pinpoint weaknesses in interactive agent design.
- **Actionable, model-specific behavioral insights:** The analysis moves beyond aggregate scores to reveal concrete strategies and failure modes. For instance, it identifies that Claude Sonnet models employ an efficient "explore-first, ask-later" strategy, while Qwen 3 Coder exhibits rigid protocol-following despite high information extraction, and most models heavily rely on user-provided navigational cues (Table 1).
- **Rigorous experimental design and reproducibility:** The methodology is clearly described using established frameworks (OpenHands, SWE-Bench), includes appropriate statistical tests (Wilcoxon signed-rank), compares a diverse set of models, and commits to releasing code and data, aligning with conference standards.

## Weaknesses
- **Uneven experimental conditions confound efficiency comparisons:** Claude Sonnet 4 and Qwen 3 Coder were allowed up to 100 interaction turns, while other models were limited to 30, justified by their "greater reasoning and planning capacity." This differential treatment introduces a confounding variable when comparing efficiency (e.g., steps per task) and may inflate performance gains for these models, undermining fair comparison.
- **Limited mechanistic analysis of core failure modes:** The paper compellingly documents *that* models fail (e.g., Qwen 3 Coder's 100% false negative rate in detection), but provides insufficient investigation into *why*. A deeper error analysis categorizing whether failures stem from poor task understanding, misaligned training objectives, or architectural limitations would transform the findings from descriptive to diagnostic.
- **Simplified user interaction model limits ecological validity:** The simulated user proxy (GPT-4o) is a perfectly cooperative oracle that only provides information explicitly in the full issue. Real-world users may be uncooperative, provide incorrect or partial information, or be uncertain themselves. This simplification may overestimate the robustness of current agents in real deployments.

## Nice-to-Haves
- A discussion of the cost-efficiency trade-off: interaction improves effectiveness but not step efficiency; analyzing whether performance gains justify increased time/user burden is relevant for practical deployment.
- A deeper causal analysis linking question types (beyond navigational/informational) to specific task failures to better understand which information gaps are most critical.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** "The claim of 'significant improvements in performance, up to 74% over the non-interactive settings' is slightly ambiguous." — The paper clearly states this is relative improvement over the non-interactive (Hidden) setting (Section 3, Figure 3), not a percentage point increase.
- **Weakness:** "The mention of 'data leakage' as a possible reason for better Hidden performance is vague." — The paper presents this as a hypothesis (Section 3.2: "likely due to their superior programming acumen, or data leakage") and does not treat it as a finding.
- **Weakness:** "Missing analyses: The paper would benefit from reporting confidence intervals or variance estimates." — Demanding statistical practices not standard in large-scale SWE-Bench evaluations is scope creep; the paper uses appropriate significance tests.
- **Weakness/Suggestion:** "Evaluate on naturally underspecified SWE-Bench issues." — The paper explicitly justifies using synthetic issues because naturally underspecified examples lack the paired ground-truth specifications necessary for causal measurement (Section 2.1).
- **Weakness:** "Test with a simpler, rule-based user proxy." — Using an LLM as a simulated user is a standard practice in related work (e.g., Xu et al., 2024; Zhou et al., 2024b, cited). The proxy's conservative design (only providing explicit information) is a strength for isolation.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Equalize interaction turn limits across all models in future experiments, or rigorously justify and account for differential allowances (e.g., by reporting normalized efficiency metrics).
- Conduct a deeper error analysis on detection failures (RQ2) to categorize root causes (e.g., failure to comprehend what's missing vs. failure to initiate dialogue) and on integration failures (RQ3) to understand why high information gain doesn't always translate to task success.

---

## 1j0ormf8uI

- GT: Accept (Poster) (avg 5.2)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a method for constructing Lower Prediction Bounds (LPBs) for counterfactual survival times under different treatments in the presence of general right-censored data. By combining conformal prediction with a potential outcomes framework, it introduces a reweighting scheme that transforms the problem into a weighted conformal inference task. The method provides a marginal coverage guarantee (up to weight estimation error) and possesses a doubly robust property. Empirical results on synthetic and real-world clinical data demonstrate valid coverage and informative prediction bounds.

## Strengths
- **Novel and important problem formulation**: The paper tackles the crucial challenge of uncertainty quantification for counterfactual survival outcomes with general right-censoring, a significant gap in high-stakes domains like personalized medicine.
- **Strong theoretical foundation**: The core methodological contribution—a reweighting scheme to handle the covariate shift induced by censoring—is theoretically justified. Theorem 4.1 provides a non-asymptotic, distribution-free coverage lower bound, and Theorem 4.2 establishes doubly robust asymptotic coverage, improving upon prior PAC-type guarantees.
- **Comprehensive and convincing empirical evaluation on synthetic data**: Experiments across six diverse synthetic settings show the method consistently achieves coverage close to the nominal level and produces less conservative (more informative) LPBs than other coverage-guaranteeing conformal baselines (e.g., "Naive", "Focus"). Robustness to outliers is also demonstrated.

## Weaknesses
- **Theoretical guarantee for optimized τ is not proven**: The method optimizes the quantile level τ per test point to maximize the LPB. While Figure 11 suggests this does not harm coverage empirically, the theoretical guarantees (Theorems 4.1 & 4.2) are stated for a fixed τ. The lack of a formal proof for the data-dependent τ* leaves a gap between theory and practice.
- **Empirical validation of the doubly robust property is absent**: Theorem 4.2 is a key theoretical advantage, but the paper does not include experiments that intentionally misspecify either the quantile model or the weight model to demonstrate that coverage is maintained when the other is correct. This limits the empirical support for a major claim.
- **Real-world coverage claim is not directly verifiable**: On the real clinical dataset, true survival times for censored patients are unknown, making it impossible to directly evaluate the empirical coverage guarantee—the method's central promise. While the results are clinically plausible, the claim of "validity" in real data is indirect and relies on synthetic experiments.
- **Sensitivity to violations of core assumptions is unexplored**: The method relies on strong ignorability (including independence between potential outcomes and censoring time) and SUTVA. The paper does not investigate, even via simulation, how violations of these untestable assumptions affect the coverage guarantee, which is important for practical reliability.

## Nice-to-Haves
- Including a sensitivity analysis for the core causal assumptions (e.g., introducing simulated unmeasured confounding) would help users understand the method's robustness in real-world settings.
- Providing confidence intervals or statistical tests for the reported empirical coverage rates (e.g., via binomial tests) would strengthen the evaluation, as is standard in conformal prediction work.
- Discussing computational cost and scalability more explicitly, given the need to train both a quantile regressor and a weight model, and to optimize τ for each test point, would be helpful for practitioners.

## Removed Points
*These points are flagged to be removed, treat them with caution*

**Strength or Weaknesses that are removed:**
- **(Weakness - Overstated "exact" guarantee)**: The paper's use of "exact" coverage is slightly overstated, as Theorem 4.1 includes an error term dependent on weight estimation. However, this is a standard presentation in conformal prediction literature (e.g., weighted conformal prediction provides "exact" coverage conditional on the weight estimate), and the theorem clearly quantifies the error. This is more a matter of terminology than a substantive flaw.
- **(Weakness - Requires data splitting)**: The need for a separate calibration set is a inherent limitation of split-conformal methods, not a specific weakness of this paper's contribution. The paper follows standard practice in the field.
- **(Weakness - Limited comparison to non-conformal baselines)**: While broadening comparisons could be interesting, the paper's primary contribution is relative to other conformal methods for survival analysis. The chosen baselines ("Uncab", "Naive", "Focus", "Fused") are the most directly relevant state-of-the-art.
- **(Weakness from Spark Finder - Missing validation on realistic synthetic data derived from real covariates)**: This is a specific experimental suggestion that, while valuable, is not a core flaw. The paper's synthetic data is already designed to mimic realistic clinical trial scenarios (see Appendix C.1 and Table 3).

## Novel Insights
The paper's key novel insight is recognizing that, under standard causal assumptions, the problem of providing a marginal coverage guarantee for counterfactual survival times with right-censoring can be transformed into a covariate shift problem between the distribution of all covariates (P_X) and the distribution of covariates for uncensored, treated individuals (P_{X|W=w,e=1}). This shift can be corrected via reweighting, allowing the application of weighted conformal prediction to achieve a strong, non-asymptotic coverage bound. This insight elegably bridges causal survival analysis with the conformal prediction toolkit.

## Suggestions
- Provide a theoretical justification or proof sketch for why the post-hoc optimization of τ (to maximize the LPB for each test point) does not violate the marginal coverage guarantee, or at least discuss this point explicitly in the theory section.
- Add an experiment demonstrating the doubly robust property: for example, on a synthetic dataset, show that coverage remains valid when the quantile regression model is severely misspecified but the weight model is correct, and vice-versa.
- In the real-data experiment section, more clearly state that the coverage rate cannot be directly computed and that the "validity" claim is extrapolated from synthetic results and the plausibility of the derived LPBs. Consider adding a semi-synthetic experiment using the real covariates to bolster this claim.

---

## wUzBBsrdB1

- GT: Reject (avg 5.0)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
This paper demonstrates that the L0 hyperparameter (average active features per token) in Sparse Autoencoders is not a neutral trade-off but is critical for learning correct, monosemantic features. Through toy models with ground truth, the authors show that if L0 is set too low or too high, SAEs "hedge" by mixing correlated or anti-correlated features, degrading disentanglement. They further show that standard sparsity-reconstruction tradeoff plots are misleading, as they can favor these incorrect SAEs. The paper proposes a simple unsupervised metric—decoder pairwise cosine similarity (c_dec)—which minimizes near the correct L0 in toy models and correlates with peak sparse probing performance in LLMs, suggesting many existing SAEs use a suboptimally low L0.

## Strengths
- **Identifies and mechanistically explains a fundamental SAE failure mode:** Using controlled toy models, the paper provides clear visual and theoretical evidence (including a proof in Appendix A.5) that an incorrect L0 forces SAEs to mix features, with MSE loss actively incentivizing this mixing when L0 is too low. This directly challenges the prevailing view of L0 as a free parameter.
- **Delivers a significant critique of standard evaluation practice:** The demonstration that a ground-truth SAE can score worse on sparsity-reconstruction plots than an incorrect, polysemantic SAE (Section 3.4, Figures 4-5) is a powerful and actionable insight that undermines a common evaluation method in the field.
- **Provides a practical, empirically validated diagnostic tool:** The proposed c_dec metric is simple, unsupervised, and shown to be minimized at the true L0 in toy models. In LLMs, its characteristic "elbow" correlates strongly with peak performance on an external sparse probing benchmark, offering practitioners a concrete guide to avoid severely suboptimal L0 settings.

## Weaknesses
- **Correlation does not establish direct causation for feature quality:** The paper validates c_dec against sparse probing performance, which is a reasonable proxy. However, it does not provide direct evidence (e.g., automated interpretability scoring or case studies) that SAEs trained at the c_dec-identified L0 yield more monosemantic features than those trained at common low-L0 settings. This leaves a gap between the metric's signal and the ultimate goal of improved feature disentanglement.
- **The metric's application requires heuristic interpretation:** As shown in Figures 8 and 9, the c_dec vs. L0 curve can have long flat regions or shallow minima, making the precise "correct" L0 ambiguous. The authors' suggestion to use the "elbow" before the low-L0 jump is a qualitative guideline rather than a rigorous, quantitative criterion, which may limit its reliability for automated use or cross-model comparison.
- **The proposed method for automatic L0 tuning is preliminary and costly:** Finding the optimal L0 currently requires training a sweep of SAEs, which is computationally expensive. The automatic tuning method sketched in Appendix A.11 is acknowledged to be unstable and requires significant hyperparameter tuning, falling short of a practical, ready-to-use solution.

## Nice-to-Haves
- **Broader empirical validation across models and layers:** Extending the analysis to more layers within the tested models and to other model families would strengthen the claim that the findings are general and not specific to a few layers.
- **Direct comparison to alternative L0 selection methods:** An empirical comparison with approaches like MDL-SAE or AFA-SAE would help better situate the contribution and clarify the relative advantages of the proposed metric.
- **More detailed analysis of the connection between c_dec and feature polysemanticity in LLMs:** While the toy models and projection histograms are suggestive, a deeper analysis (e.g., examining activation patterns for known concept pairs) could more directly link high c_dec to the mixing of interpretable features in real SAEs.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness about theoretical grounding of c_dec:** The paper provides a formal proof in Appendix A.6 showing that feature mixing increases c_dec. While the uniqueness of the global minimum might depend on data structure, the empirical evidence across toy models and LLMs is strong.
- **Weakness about the assumption of linear features being a major flaw:** The paper explicitly scopes its investigation to the Linear Representation Hypothesis, which is the foundational assumption for most SAE work. Criticizing it for not addressing non-linear features is scope creep.
- **Weakness about limited LLM experiments being insufficient for a top conference:** The paper studies two modern LLMs across multiple layers. Demanding exhaustive evaluation across all models and layers is an unreasonable burden for a single paper.
- **Suggestion that an ablation on SAE width is required:** The paper's focus is on L0. While width interacts with L0, isolating its effect is a separate research question outside this paper's stated contributions.

## Novel Insights
The paper synthesizes several novel and impactful observations: 1) Incorrect L0 does not merely trade reconstruction for sparsity but causes systematic corruption of the learned dictionary through feature mixing, a manifestation of "feature hedging." 2) Consequently, the ubiquitous sparsity-reconstruction tradeoff plot is an unsound evaluation tool, as it can actively prefer incorrect, polysemantic SAEs over correct ones. 3) This corruption leaves a measurable signature in the SAE decoder weights (increased pairwise cosine similarity), yielding a simple, unsupervised metric that can guide hyperparameter selection. Together, these insights necessitate a reevaluation of standard SAE training and evaluation practices.

## Suggestions
- **Clarify the practical guideline for using c_dec:** To address the ambiguity in flat regions, propose a more concrete, quantitative criterion (e.g., "choose the lowest L0 where c_dec is within X% of its minimum over the sweep").
- **Include a direct interpretability comparison:** To strengthen the causal claim, add a small case study comparing the highest-activating tokens for a sample of latents from a low-L0 SAE versus the c_dec-optimal SAE, demonstrating visibly improved monosemanticity.

---

## OuMNJoKJBQ

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper argues that current LLM safety alignment is superficial, relying on shallow refusal heuristics rather than deep reasoning, which leaves models vulnerable to jailbreaks. To address this, the authors propose two main contributions: 1) a publicly released Chain-of-Thought (CoT) fine-tuning dataset that pairs safety and utility prompts with step-by-step rationales, and 2) Alignment-Weighted DPO (AW-DPO), a method that applies separate preference weights to the reasoning and final-answer segments of a response for finer-grained optimization. Experiments across multiple models and jailbreak benchmarks show improved safety robustness while maintaining utility.

## Strengths
- **Novel and well-motivated method:** AW-DPO is a principled extension of DPO, directly motivated by a qualitative error analysis of CoT fine-tuning that revealed mismatches between reasoning safety and answer safety. The idea of applying component-specific weights based on harmfulness scores is innovative and addresses a clear gap in standard preference optimization.
- **Comprehensive and rigorous evaluation:** The paper evaluates across multiple model families (LLaMA, Mistral), sizes (3B-8B), and a diverse set of 20 jailbreak strategies from SorryBench. Comparisons include strong recent baselines (SAFECHAIN, STAIR, Representation Rerouting), and the authors conduct valuable auxiliary analyses (transferability of DPO datasets, performance on prefix attacks, and application to pre-aligned models).
- **Valuable resource contribution:** The construction and promised release of a novel CoT safety fine-tuning dataset that combines safety-critical and general-purpose utility examples is a significant contribution that will facilitate reproducibility and future work in reasoning-aware alignment.

## Weaknesses
- **Ambiguous formulation and missing implementation details for AW-DPO:** The mathematical description of AW-DPO (Equations 3-4) is ambiguous. It introduces a token-level reward sum with binary masks `w_{s_t} ∈ {0,1}`, then states it computes separate DPO losses for reasoning and response segments. The derivation from the standard sequence-level DPO objective to this per-segment version is not provided, leaving the theoretical grounding unclear. Furthermore, the weight calculation `w_reasoning = d_respond/(d_reasoning + d_respond)` can produce undefined or negative values if the harmfulness score differences (`d_reasoning`, `d_respond`) are negative or sum to zero. The paper does not specify how these cases are handled (e.g., absolute values, clipping, epsilon), which is critical for reproducibility.
- **Overstated claim from causal intervention experiment:** The preliminary experiment in Section 3 uses linear probing on *prompt classification* (safe vs. unsafe) to conclude that alignment is "superficial" and that "refusals do not rely on reasoning ability." However, high accuracy in classifying input prompts does not directly demonstrate that the model's *generative refusal behavior* is non-reasoning-based. The experiment shows that input discrimination persists after ablating reasoning neurons, but it does not establish a causal link between those neurons and the safety of generated refusals. This weakens the foundational evidence for the "superficial alignment" hypothesis.
- **Dependence on a proprietary LLM judge and lack of robustness analysis:** The AW-DPO pipeline relies on GPT-4o to assign harmfulness scores to reasoning and response segments for preference construction. This introduces a dependency on a proprietary model, its associated biases, and cost. While Appendix J.3 provides a basic robustness check via prompt paraphrasing, the correlations for reasoning-only scores are only moderate (0.5761). A more thorough analysis comparing against alternative judges (e.g., Llama Guard) or human evaluation on a subset is needed to validate the consistency and reliability of this critical component.

## Nice-to-Haves
- **Ablation on the contribution of CoT utility data:** An experiment training AW-DPO starting from a model fine-tuned *only* on safety CoT data (without the general-purpose utility examples) would help disentangle whether improvements stem from reasoning-augmented safety data specifically or from a larger, more diverse fine-tuning mixture.
- **Direct comparison against a stronger DPO baseline:** A more rigorous baseline would apply standard DPO using the *same* candidate generation and preference pairing procedure as AW-DPO (i.e., using the judge model and threshold γ on full-answer harmfulness). This would better isolate the benefit of the per-segment weighting from gains due to better candidate filtering.
- **Qualitative examples of error correction:** Including 2-3 annotated case studies showing a jailbreak prompt where the CoT-SFT model fails (illustrating one of the two identified error types) and how the AW-DPO model successfully handles it would make the "fine-grained correction" more concrete and compelling.

## Removed Points 
*These points are flagged to be removed, treat them with caution*
- **"Major conceptual flaw" in preliminary experiments (Harsh Critic):** While the link between prompt classification and generative behavior is not directly proven, the experiment is presented as *supporting evidence* for the hypothesis, not definitive proof. The authors also provide benchmark results after neuron ablation (Appendix D, Table 6) showing safety rates remain high while reasoning accuracy drops, which is consistent with their argument. The criticism is overstated but the core concern about the strength of this evidence is retained as a weakened point above.
- **Lack of statistical significance testing (Harsh Critic):** The paper reports standard deviations across attack categories (the "Std↓" column in tables). While confidence intervals or statistical tests would be a nice addition, the consistent, often substantial performance gaps across multiple models and the comprehensive head-to-head comparisons provide strong empirical evidence. Demanding formal statistical tests is not a standard requirement for large-scale benchmark comparisons in this field.
- **Missing "Limitations" section (Harsh Critic):** The paper includes an Ethics Statement and Reproducibility Statement. While a dedicated Limitations section is good practice, its absence does not constitute a substantive weakness of the technical contribution. The relevant limitations (dependence on GPT-4o, computational cost) are captured in the weaknesses above.
- **Request for evaluation on additional jailbreak benchmarks (Spark Finder):** The paper uses SorryBench, a comprehensive and recent benchmark with 20 diverse attack strategies across 44 harm categories. This is a standard and sufficient evaluation for the claimed contribution. Demanding evaluation on every possible benchmark (HarmBench, AdvBench) is scope creep.
- **Request for "theoretical foundation" for AW-DPO weighting (Positive-Leaning):** The weighting scheme is heuristic but empirically motivated by the observed error modes. While a theoretical justification would be nice, the method is presented as an empirical improvement to DPO, and its effectiveness is demonstrated through extensive experiments. This is a common and acceptable approach in the field.

## Novel Insights
The paper provides a novel synthesis of causal analysis, dataset creation, and algorithmic innovation to address jailbreak robustness. The key novel insight is the identification and quantification of specific failure modes in reasoning-augmented alignment (correct reasoning/unsafe answer and incorrect reasoning/safe answer), which directly motivates the design of AW-DPO. This fine-grained error analysis moves beyond treating responses as monolithic and enables a targeted optimization strategy that standard DPO cannot achieve. Furthermore, the finding that strong general reasoning models (e.g., Phi-4-Reasoning) are not inherently safer underscores that alignment requires reasoning capabilities tailored specifically to safety contexts, not just improved reasoning in general.

## Suggestions
- **Clarify the AW-DPO formulation and implementation:** In the method section or appendix, provide a clearer, step-by-step explanation of how the separate DPO losses for reasoning and response are computed from the segmented rewards. Explicitly state how the weight calculation handles cases where `d_reasoning + d_respond` is zero, negative, or very small (e.g., using absolute values, adding a small epsilon, or clipping).
- **Strengthen the analysis of the judge model's role:** Expand Appendix J.3 to include a comparison of AW-DPO performance when using a different, open-source safety judge (e.g., Llama Guard) for scoring, or report inter-annotator agreement between GPT-4o and human evaluators on a sample to better substantiate the reliability of the scoring mechanism.
- **Temper the claim from the causal intervention experiment:** Reword the conclusion of Section 3 to more accurately reflect what the experiment shows: that the *ability to discriminate harmful from safe prompts* persists after ablating reasoning-critical neurons, which is *consistent with* the hypothesis that generative safety alignment may operate via shallow heuristics. Avoid the direct causal claim about refusal generation.
- **Add a brief discussion of computational cost:** Acknowledge in the text that AW-DPO incurs additional cost during dataset construction due to candidate generation and LLM judging, and briefly discuss the trade-off with the achieved performance gains, perhaps referencing the transferability result as a way to amortize this cost.

---

## 32mrjmaeMP

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (6.2/10)
- Match: N/A

### Final Review

## Summary
This paper proposes TAK (Task Arithmetic with KFAC regularization), a method to improve weight disentanglement in task arithmetic without requiring access to other tasks' data. By linking representation drift regularization to the Generalized Gauss-Newton matrix and approximating it via a Kronecker-Factored Approximate Curvature (KFAC), the authors derive a dataless regularizer. A key innovation is a heuristic to merge per-task curvature factors into a single surrogate, achieving constant memory and computational complexity in the number of tasks. The method demonstrates state-of-the-art performance on task addition and negation benchmarks in vision and language, offers robustness to task vector rescaling, and promotes clear task localization.

## Strengths
- **Dataless regularization matching data-dependent performance:** In the linearized fine-tuning regime, TAK achieves performance on par with or superior to the data-dependent τJp method on the 8 Vision benchmark (Tables 1, 2) and strong results on task negation, a significant advantage for privacy and modularity.
- **Scalable design with constant complexity:** The proposed aggregation of per-task KFAC factors into a single surrogate (Eq. 8) reduces storage and computation from O(T) to O(1) in the number of tasks, validated empirically with minimal performance drop (Table 3, Fig. 6).
- **Empirically validated robustness and task localization:** The method exhibits strong robustness to the task vector scaling coefficient α (Figs. 4, 11), often eliminating the need for held-out tuning. It also induces clear task localization, as shown by the separation between in-task and out-of-task Jacobian-projected outputs (Fig. 5, 13, 14).

## Weaknesses
- **Suboptimal performance in the language domain:** On T5-base, TAK is consistently outperformed by the data-dependent τJp method (Table 1, Fig. 3). This suggests the curvature approximation or the linearization assumption may be less effective for textual tasks, limiting the method's universality.
- **High memory footprint for the KFAC factors:** Storing the full KFAC matrices requires quadratic memory in layer dimensions, which can be prohibitive for very large models (Appendix B). While compression is explored (Fig. 7b), it involves a non-trivial accuracy-storage trade-off.
- **Theoretical gap for the non-linear regime:** The method is derived and theoretically justified for linearized fine-tuning. Its application to the non-linear regime (via pairing with Attention-Only Fine-Tuning) is empirically motivated but lacks a theoretical grounding, making its effectiveness in standard non-linear fine-tuning less certain.

## Nice-to-Haves
- A quantitative metric (e.g., AUC) to summarize the task localization separation shown in Figures 5, 13, and 14, enabling more direct comparison across methods.
- Exploration of combining TAK with parameter-efficient fine-tuning (PEFT) techniques like LoRA, which could further reduce memory demands and broaden applicability.
- A more detailed analysis of why the performance gap with τJp exists in the language domain, potentially guiding improvements for textual tasks.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Statistical reporting lacks confidence intervals."** While reporting standard deviations is good practice, the paper's core claims are supported by extensive multi-seed ablations (Table 5) and consistent trends across models and tasks. The absence of error bars in main tables does not invalidate the results.
- **Weakness: "Need to compare against random matrix approximations or simple norm penalties."** The paper already includes a strong and relevant baseline: diagonal GGN approximation (Porrello et al., 2025). Adding more simplistic regularizers does not strengthen the evaluation of the specific curvature-based contribution.
- **Missing Experiment: "Scale to 20+ tasks to prove constant complexity."** The claim of constant complexity is algorithmic (O(1) storage). The empirical validation on 8 tasks (Table 3) and the analysis of the merging heuristic (Appendix C) are sufficient; demanding an arbitrary larger scale is scope creep.
- **Missing Experiment: "Evaluate on standard full-parameter non-linear FT."** The paper explicitly scopes its non-linear application to settings that induce approximately linear behavior (Attention-Only FT). Demanding evaluation on standard non-linear FT asks the method to operate outside its designed and justified regime.
- **Weakness: "Heuristic merging lacks theoretical justification."** Appendix C provides a formal bound on the approximation error of the merging heuristic. This is a reasonable theoretical contribution for a primarily empirical paper.

## Novel Insights
The paper provides a novel connection between representation drift regularization for weight disentanglement and second-order optimization techniques. It demonstrates that a well-known curvature approximation (KFAC) can be repurposed as a dataless regularizer that effectively prevents cross-task interference. Furthermore, the insight that per-task curvature factors can be aggregated into a single surrogate with constant complexity without significant performance loss is a key contribution for scalable multi-task model editing.

## Suggestions
- In the main text, include a brief intuitive explanation or summary of the merging error bound from Appendix C to make the heuristic's justification more accessible.
- Commit to releasing the pre-computed KFAC factors for the models and tasks used in the paper alongside the code, as this aligns perfectly with the vision of sharing dataless "assets" for downstream applications.

---

## eETr3lrOQB

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper introduces VQ-Transplant, a framework that enables efficient plug-and-play replacement of the Vector Quantization (VQ) module within a frozen, pre-trained visual tokenizer (e.g., VAR). To address decoder-quantizer mismatch, it proposes a lightweight decoder adaptation stage. A secondary contribution is MMD-VQ, a novel VQ method using Maximum Mean Discrepancy for distributional alignment. The method achieves near state-of-the-art reconstruction fidelity while drastically reducing training cost.

## Strengths
- **Solves a Clear Practical Problem**: The paper effectively targets the significant computational bottleneck of training modern adversarial tokenizers from scratch, which severely hinders research into novel VQ algorithms. The proposed framework directly lowers this barrier to innovation.
- **Extensive and Convincing Empirical Validation**: The evaluation is comprehensive, testing multiple VQ algorithms (Vanilla, EMA, Online, Wasserstein, MMD) in both multi-scale and fixed-scale configurations on ImageNet-1k. Results show VQ-Transplant with MMD-VQ can surpass the original VAR's reconstruction fidelity (r-FID 0.81 vs. 0.92) with massive speedups. Strong cross-dataset results (FFHQ, CelebA-HQ, LSUN-Churches) demonstrate impressive generalization.
- **Well-Designed and Analyzed Methodology**: The two-stage process (substitution + adaptation) is clearly motivated and explained. The paper provides useful analyses, such as tracking r-FID over adaptation epochs and comparing decoder-only versus joint optimization (in the appendix), which clarify design choices and trade-offs.

## Weaknesses
- **Ambiguous and Potentially Unfair Efficiency Comparison**: The core efficiency claim (21.8x speedup, 95% cost reduction) in Table 1 compares training on different datasets (VAR trained on OpenImages vs. VQ-Transplant adaptation on ImageNet-1k) and uses different GPU counts. This confounds the comparison and undermines the precise quantification of computational savings. A normalized comparison on the same dataset is needed to solidly support the claim.
- **Insufficient Analysis of VQ Compatibility and Framework Generality**: The results consistently show distribution-alignment VQs (Wasserstein, MMD) work best, but the paper provides only a surface-level explanation (lower quantization error, higher utilization). A deeper analysis of what properties make a VQ method "compatible" for transplantation is missing. Furthermore, the framework's performance drops significantly when applied to the LDM tokenizer (Appendix D). While hypotheses are offered (capacity, continuous vs. discrete pretraining), this important limitation regarding the framework's dependency on the base tokenizer's properties is under-explored and buried in the appendix.
- **Lacks Ablations on Framework Components**: The paper does not ablate the necessity of its core design choices. For example, what is the impact of the uniqueness loss in Stage I? What happens if decoder adaptation is skipped? How critical is freezing the encoder versus jointly fine-tuning it? These ablations would solidify the understanding of which components are essential for the framework's success.

## Nice-to-Haves
- **Compute-Aware Performance Trade-off Analysis**: A plot showing the Pareto frontier of reconstruction fidelity (r-FID) versus total training compute (GPU-hours) for VQ-Transplant versus full training from scratch would more convincingly demonstrate the efficiency claim.
- **Extended Analysis of Adaptation Dynamics and Stability**: While 5-epoch adaptation is emphasized, Table 5 shows gains with more epochs. A clearer discussion of the performance-compute trade-off for adaptation epochs, along with training stability curves (e.g., discriminator/generator loss), would be helpful for practitioners.
- **Visualization of Latent Space Alignment**: t-SNE visualizations comparing the encoder feature distribution and the codebook distribution before and after transplantation for different VQ methods could provide intuitive evidence for the distribution alignment claim of MMD-VQ.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "Limited Novelty of Core Concept"**: The application of a module replacement and selective fine-tuning paradigm to the specific, underexplored problem of visual tokenizer VQ modules is novel and constitutes a clear contribution.
- **Weakness: "Underwhelming Baseline Comparison for From-Scratch Training"**: The paper explicitly states that tokenizers require hundreds of epochs to converge from scratch, making a short 5-7 epoch comparison intentionally illustrative of the infeasibility of this approach. The provided comparison is valid for its stated purpose.
- **Weakness: "Redundant and Somewhat Disorganized Presentation"**: While minor formatting artifacts exist (e.g., Table 1 appears twice in the extracted content), the paper's core narrative, method description, and results are logically structured and clear.
- **Weakness: "Missing baseline: fine-tuning the entire tokenizer"**: The paper includes this exact comparison in Appendix C (Joint Optimization), discussing it as an alternative that offers slightly better performance at increased cost. The choice of decoder-only adaptation is justified by the core efficiency goal.
- **Suggestion: "Downstream task evaluation"**: The paper's scope and contribution are squarely on enabling efficient VQ module replacement while preserving reconstruction fidelity. Evaluating downstream generation or classification is an orthogonal extension, not a requirement for validating the core claim.
- **Suggestion: "Standalone evaluation of MMD-VQ"**: The paper introduces MMD-VQ specifically as a method designed for compatibility within the VQ-Transplant framework. Evaluating it as a standalone VQ method in a full from-scratch training pipeline is a different research question outside the paper's stated contributions.
- **Criticism: "The decoder adaptation still uses adversarial training, which can be unstable"**: The paper explicitly follows the established, stable training recipe from VAR (using DINO-S discriminator, DiffAug, consistency regularization, LeCAM), as noted in Section 4.1. It does not claim to solve adversarial training instability but leverages a known stable setup.

## Novel Insights
The paper's core novel insight is the decoupling of VQ algorithm development from the prohibitive cost of training entire tokenizers from scratch. By identifying and addressing the decoder-quantization mismatch through lightweight adaptation, it demonstrates that state-of-the-art reconstruction fidelity can be maintained or even improved while freezing most of a pre-trained model. This enables a new, efficient research paradigm where novel VQ techniques can be iteratively tested by "transplanting" them into powerful, frozen backbones.

## Suggestions
- **Conduct a Fair, Normalized Efficiency Comparison**: Re-run the efficiency comparison (Table 1) using normalized GPU-hours on the same dataset (e.g., ImageNet-1k) to provide an unambiguous, apples-to-apples validation of the computational savings.
- **Move and Expand the Limitation Discussion**: Elevate the discussion on LDM tokenizer compatibility from Appendix D to a main "Limitations" section. Expand the analysis to more concretely diagnose the cause of the performance gap (e.g., analyze feature distribution shifts) and discuss the framework's dependency on base model properties.
- **Add Key Ablation Studies**: Include ablations on the necessity of the Stage I uniqueness loss, the impact of skipping decoder adaptation, and the effect of unfreezing the encoder during adaptation. This will rigorously validate the framework's design.

---

## opU91paIvZ

- GT: Withdrawn (treated as Reject) (avg 3.3)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper addresses the problem that Chain-of-Thought (CoT) reasoning is often unfaithful or verbose, limiting its use for interpretability and safety monitoring. The authors formalize CoT monitorability as a constrained optimization problem, identify why standard reinforcement learning fails due to sparse gradients, and propose a prior-guided distillation pipeline. This pipeline uses a stronger instruction-tuned model to transform base model CoTs into faithful and concise versions, creating a dense supervision dataset for fine-tuning. Experiments show improved faithfulness (relative gain ~10%) and significantly shorter CoTs (up to 60% reduction) while largely preserving task accuracy on MMLU-Pro, GSM8K, and MATH500.

## Strengths
- **Insightful analysis of a fundamental learning obstacle:** The paper provides a clear gradient analysis (Section 3) demonstrating why naive policy optimization fails to improve monitorability due to vanishing gradients when the desired behavior is sparse under the initial policy. This theoretical motivation is solid and well-presented.
- **Creative and effective solution to sparse feedback:** The core method of using a capable prior model (`π_s`) to transform flawed reasoning traces into monitorable ones is novel and elegant. It successfully converts an intractable sparse-reward RL problem into a manageable supervised learning task, as evidenced by the empirical results.
- **Clear empirical demonstration across two axes:** The paper provides comprehensive validation across the two defined dimensions of monitorability (faithfulness and conciseness) on established benchmarks. The results consistently show the method's ability to improve these properties while maintaining task performance.

## Weaknesses
- **Evaluation of faithfulness relies heavily on an LLM judge:** The key metric for faithfulness uses an LLM (Qwen 14B) to judge whether a hint is verbalized. While practical, this inherits potential biases and subjectivity from the judge model. The paper acknowledges this limitation but does not supplement it with human validation or more robust automated checks, which weakens confidence in the central claim of improved faithfulness.
- **Insufficient comparison to relevant baselines:** For conciseness, the method is not compared to simple prompting strategies (e.g., instructing the base or prior model to "be concise") or other recent techniques for generating concise reasoning. For faithfulness, comparisons to methods like contrastive training or direct supervision for hint verbalization are absent. This makes it difficult to gauge the specific contribution of the proposed pipeline versus the inherent capability of the instruction-tuned prior.
- **Lack of analysis on the quality of shortened reasoning:** The paper reports large reductions in CoT length but does not analyze what semantic content or logical steps are lost. It is critical to verify that "conciseness" does not come at the cost of omitting necessary reasoning, which would violate the faithfulness objective. A qualitative analysis or an automated check for missing steps is missing.
- **Method's success is contingent on a strong, external prior:** The proposed pipeline depends on a prior model (Qwen 2.5-7B Instruct) that is significantly larger and more capable than the base model being trained. The paper does not ablate the prior's strength or explore the method's performance with weaker priors, limiting the understanding of its generalizability and practical requirements.

## Nice-to-Haves
- **Justification for conciseness thresholds:** The specific token-length constraints (e.g., 125 for GSM8K, 950 for MATH500) are presented without motivation. Explaining how these targets were chosen (e.g., based on data distributions or monitoring needs) would improve clarity.
- **Extended failure mode analysis:** A discussion of cases where the method fails (e.g., accuracy drops, constraints are not met) or where the prior produces incorrect transformed traces would provide a clearer picture of its limitations.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism that the filtering condition in Algorithm 1 is "ambiguous" when the original answer is incorrect:** The paper's logic (line 13: `R(x, yi) = R(x, y)`) is clear—it filters for candidates that preserve the reward of the *original* trace, whatever that reward was. This is a valid design choice, not an ambiguity.
- **Criticism about "missing statistical robustness" (error bars, significance tests):** For the scale of the benchmarks used (e.g., MATH500 with 500 samples) and within the common practice in this subfield, reporting single-run results is standard. Demanding statistical tests imposes a rigor requirement beyond current norms.
- **Criticism about the "narrow operationalization" of faithfulness:** The paper explicitly focuses on hint verbalization as a concrete, measurable proxy for faithfulness, which is a reasonable and common simplification for a methodological paper. Critiquing this as too narrow is scope creep.
- **Suggestion to add "attention or saliency maps for faithfulness":** This demands a specific mechanistic interpretability analysis that is orthogonal to the paper's core contribution of a training methodology. It is not a standard expected evaluation for this type of work.
- **Suggestion for "integration of the constraint into a single training loop" as an "obvious next step":** While an interesting research direction, the two-stage pipeline is a valid and effective solution. Framing this as a missing component of the current work is unfair.

## Novel Insights
The paper's key novel insight is the formal identification of the sparse gradient problem when optimizing for monitorability properties under a standard policy. This leads to the creative solution of using a capable external model as a "monitorability prior" to densely populate the region of the reasoning space where the desired property holds, thereby transforming the learning problem. This insight—that monitorable traces are often *compatible* with high reward but *rarely sampled*—and the subsequent method to overcome this sparsity via guided transformation and distillation, constitutes a meaningful advance for improving reasoning transparency.

## Suggestions
- **Strengthen the faithfulness evaluation:** Conduct a small-scale human evaluation on a subset of data to validate the LLM-as-a-judge scores for hint verbalization. Alternatively, implement and report agreement scores between multiple LLM judges to estimate reliability.
- **Add simple but critical baselines:** For conciseness, report the performance of the base model and the prior model when given a direct instruction to "provide a concise reasoning trace" in a zero-shot or few-shot setting. This will help isolate the contribution of the fine-tuning pipeline.
- **Include a qualitative analysis of shortened traces:** Provide a dedicated subsection analyzing several examples of shortened CoTs, commenting on whether essential logical steps are preserved or omitted. This is crucial for validating that conciseness does not compromise faithfulness.

---

## hQZQVLJrH9

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper establishes a first-order equivalence between activation steering and influence functions, introducing Influence-Aligned Steering (IAS) as a constructive mapping. It provides theoretical guarantees on alignment, optimality, and generalization, supported by experiments on language and vision models.

## Strengths
- **Novel theoretical unification**: The paper derives closed-form mappings between steering vectors and influence weightings, proving their first-order equivalence (Theorems 4.2, 5.2) and introducing alignment bounds via a scalar measure ω(x) (Theorem 5.1). This formally connects two previously disparate research areas.
- **Practical diagnostic and optimization tools**: The alignment metric ω(x) offers a feasible pre-check for steering success, and the spectral method for optimal steering directions (Theorem 5.3) provides a principled alternative to handcrafted vectors, both relying on efficient Jacobian-vector products.
- **Empirical validation of core theoretical claims**: Experiments confirm high cosine similarity (0.978) between predicted and actual first-order logit shifts (Figure 1), show alignment improves with layer depth (Figure 2), and demonstrate statistical significance of spectral directions on ResNet-50 (Figure 3).

## Weaknesses
- **Limited empirical scope undermines scalability claims**: Experiments are confined to GPT-2 Medium (355M parameters) and ResNet-50, with no validation on billion-parameter models as suggested by the paper's motivation. This leaves the claimed applicability to large-scale models unsubstantiated.
- **Insufficient demonstration of practical advantage**: In the detoxification task, IAS underperforms the Contrastive Activation Addition (CAA) baseline in both toxicity reduction and perplexity (Table 1), failing to show clear empirical benefit over existing steering methods.
- **Missing validation of key workflow component**: The paper does not empirically demonstrate the promised mapping from steering vectors to causal training examples (via ϱ_s), which is central to the contribution of data attribution and debugging. Without this, the practical utility of the equivalence remains unproven.
- **Unexplored boundaries of the first-order regime**: While the theory assumes small edits, there is no empirical analysis quantifying how large steering magnitudes can be before the first-order approximation breaks down, leaving practical applicability uncertain.

## Nice-to-Haves
- Extend experiments to larger models (e.g., Llama, GPT-J) and diverse tasks (e.g., factual editing, bias mitigation) to better support scalability and generality claims.
- Provide a quantitative analysis of error growth with steering magnitude to delineate the valid regime of the first-order approximation.
- Include more detailed implementation specifics (e.g., damping parameter choices, layer selection heuristics) to enhance reproducibility.

## Removed Points
*These points are flagged to be removed, treat them with caution:*
- "Abstract omits critical condition (ω)" – The abstract summarizes contributions, and ω is part of the theoretical development, not an omission required in the abstract.
- "Connection to convex analysis not elaborated" – This is an optional depth for insight, not a flaw in the paper's core contributions.
- "Proofs are deferred" – Common practice for conference papers; not a substantive weakness.
- "Systematic scaling discrepancy with slope 1.50 indicates a problem" – The paper notes the slope in Figure 1 and states it is consistent with the linear regime; without further evidence, this is not a clear error.
- "Experiments lack computational cost discussion" – The paper outlines computational primitives (Jacobian-vector products, small SVDs) and acknowledges scalability challenges, so this criticism is partially addressed.

## Novel Insights
The paper introduces the novel insight that activation steering and influence functions are dual projections of the same sensitivity tensor, with the alignment metric ω(x) quantifying when steering can perfectly replicate data influence. This unification provides a geometric framework for diagnosing feasibility and offers a principled bridge between model intervention and data attribution.

## Suggestions
- Conduct an experiment that applies the IAS mapping to identify training examples for a specific model behavior (e.g., a bias or hallucination), validating the causal attribution claim with qualitative analysis.
- Compare IAS to state-of-the-art steering methods (e.g., SAKE, representation engineering) on standard benchmarks to better assess its practical value relative to existing approaches.
- Discuss concrete computational strategies for approximating pseudoinverses and Hessian inverses in large-scale settings to address scalability concerns more transparently.

---

## Rt9SeEAMWv

- GT: Reject (avg 4.8)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a new framework for deriving worst-case generalization bounds for stochastic optimization algorithms. It proposes the concept of *random set stability*, tailored for data-dependent random sets like optimization trajectories, and combines it with empirically relevant topological complexity measures. The key contribution is deriving bounds that avoid the intractable mutual information terms common in prior work, while still capturing the interplay between algorithmic stability and geometric complexity.

## Strengths
- **Novel Stability Framework:** The introduction of random set stability (Assumption 3.1) is a principled and technically sound extension of classical algorithmic stability to data-dependent random sets, explicitly incorporating algorithmic randomness. Lemma 3.2 and Corollary 3.3 effectively connect it to established stability notions and show it holds for practical algorithms like projected SGD under standard assumptions.
- **Theoretical Unification and Improvement:** Lemma 3.4 provides a core bound linking expected worst-case error to Rademacher complexity and stability, gracefully recovering classical stability and uniform convergence bounds as special cases (Corollaries 3.5, 3.6). Theorems 4.3 and 4.4 successfully derive mutual-information-free versions of recent topological generalization bounds (based on box-counting dimension, weighted lifetime sums, and positive magnitude), addressing a significant limitation in the literature.
- **Empirical Validation with Non-Trivial Models:** The experimental validation on Vision Transformers and GraphSAGE demonstrates that the proposed bounds are non-vacuous and adapt meaningfully to hyperparameter changes (Table 1). The investigation of the coupling between stability and topological complexity across sample sizes provides empirical support for the theoretical product structure in the bounds (Figures 2, 3).

## Weaknesses
- **Expectation Bounds Only:** All theoretical bounds are stated in expectation, not with high probability. This limits direct comparability with PAC-style generalization guarantees and is a notable restriction for a learning theory contribution, as acknowledged in the paper's limitations.
- **Heuristic and Optimistic Stability Estimation:** The stability parameter \(\beta_n\) is central to the bounds but is estimated via a heuristic (Algorithm 1) that approximates a supremum over the data space with a finite held-out set. This very likely yields an optimistic (underestimated) value, making the empirical bound evaluation less rigorous and the claimed tightness less credible.
- **Strong and Unverified Local Lipschitz Assumption:** Assumption 4.1 requires a local Lipschitz constant \(L_{S,U}\) for the loss on the random set \(W_{S,U}\), uniform over the data space. This is a strong condition whose validity is not examined empirically; the constant's role in the bounds is not assessed, leaving open how it might affect their magnitude in practice.
- **Limited Empirical Scope:** The experimental validation, while solid, is confined to two model architectures (ViT, GraphSAGE) on two datasets. The framework's applicability to a broader range of optimizers (beyond the analyzed SGD), loss functions, and architectures remains undemonstrated, which affects the generalizability of the empirical claims.

## Nice-to-Haves
- A more extensive empirical study involving different optimizers (e.g., Adam), architectures, and datasets would strengthen confidence in the framework's generality.
- A sensitivity analysis of the bound's dependence on free parameters (e.g., \(J\) in Lemma 3.4, \(\lambda\) in Theorem 4.4) would provide insight into the robustness of the chosen values.
- Discussing potential pathways to derive high-probability bounds, even if not implemented here, would better position the work within the generalization theory literature.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Weakness about the requirement that \(\beta_n^{-2/3}\) divides \(n\):** This is a minor technical condition for simplifying theorem statements and does not affect the core theoretical contribution or its interpretation.
- **Weakness about "missing comparison to baseline bounds":** The paper's primary contribution is a new class of bounds that remove intractable terms; a direct comparison to the very bounds it aims to improve (which rely on intractable mutual information) is not feasible by design.
- **Weakness about "not analyzing \(\beta_n\) for Adam":** The paper provides a general stability framework and proves it holds for SGD as an illustrative example. Deriving explicit \(\beta_n\) for every optimizer is outside the paper's scope; the framework is applicable if the stability assumption holds.
- **Strength about "the paper is well-written":** While true, this is a generic strength that applies to many papers and does not highlight a specific contribution of this work.

## Novel Insights
The paper successfully bridges the concepts of algorithmic stability and data-dependent topological complexity for the first time, providing a unified framework that recovers classical results and yields new, fully computable generalization bounds. The key insight is that the stability parameter \(\beta_n\) can replace intractable mutual information terms in topological bounds, making them empirically relevant. The theoretical product structure (\(\beta_n^{1/3}\) multiplied by a complexity measure) and its empirical corroboration reveal a genuine coupling between how sensitive an algorithm's trajectory is to data changes and the geometric complexity of that trajectory.

## Suggestions
- To address the heuristic estimation of \(\beta_n\), the authors could perform a sensitivity analysis by increasing the size of the held-out set used in Algorithm 1 and reporting how the estimate changes, thereby quantifying the potential underestimation.
- The local Lipschitz constant \(L_{S,U}\) appears in the bounds. The authors could discuss, even informally, how one might approximate or bound this quantity in practice (e.g., via gradient norms along the trajectory) to make the bounds more concrete.

---

## ahpO7S1Ppi

- GT: Reject (avg 3.5)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
Pctx introduces a personalized context-aware tokenizer for generative recommendation (GR), conditioning semantic IDs on a user’s interaction history to capture diverse interpretations of the same item. The method addresses sparsity via adaptive clustering and redundant ID merging, and demonstrates significant improvements over non-personalized GR baselines on three Amazon datasets.

## Strengths
- **Novel contribution**: This is the first work to personalize tokenization in GR, directly addressing the limitation that static tokenization enforces a universal similarity standard. The core idea is well-motivated and clearly differentiated from prior context-aware tokenizers (e.g., ActionPiece) that only consider local context.
- **Comprehensive empirical validation**: Experiments on three datasets show statistically significant gains (up to 8.9% NDCG@10) over a wide range of strong baselines. Ablation studies rigorously validate each component, and additional analyses (model ensemble, hyperparameter sensitivity, explainability) substantiate the claims.
- **Reproducibility**: The paper provides extensive implementation details, hyperparameter settings, and publicly released code, aligning with ICLR’s reproducibility standards.

## Weaknesses
- **Inference aggregation method is unspecified**: Section 2.3 states that probabilities from multiple semantic IDs for the same item are aggregated, but the exact operation (sum, max, etc.) is not given, hindering exact replication.
- **Computational efficiency and scalability are unaddressed**: The pipeline involves pre-training an auxiliary model (DuoRec), clustering, and merging steps. The overhead relative to static tokenization and scalability to very large datasets are not discussed, which is a practical concern for a paradigm that often emphasizes efficiency.
- **Limited comparison with multi‑identifier baselines**: While the paper discusses MTGRec (which assigns multiple static IDs per item) in Section 2.4, it does not include an experimental comparison. This leaves open whether the gains stem from personalization or merely from having multiple IDs per item.
- **Missing controlled baseline for token diversity**: The ablation with random target (γ=1) shows that arbitrary swapping hurts, but a stronger baseline that assigns multiple non‑personalized IDs per item (e.g., via clustering item features alone) is absent. Without it, the isolated effect of personalization versus token diversity is unclear.
- **Suboptimal quantization choice**: Main experiments use RQ‑VAE, but Appendix G.1 shows RK‑Means yields better performance. This suggests the reported gains may be conservative and raises questions about the primary quantization method selection.

## Nice-to-Haves
- Analysis of how context length (e.g., last 5 vs. all interactions) affects personalization, to clarify whether long‑term history is necessary.
- Performance breakdown by user/item frequency (head vs. tail) to see if improvements are uniform or concentrated.
- Visualization of the semantic ID space (e.g., t‑SNE of fused representations) to visually confirm that different IDs for the same item correspond to distinct context‑driven clusters.
- Quantification of how often beam search produces different semantic IDs for the same candidate item, verifying the claimed “multi‑facet” generation behavior.

## Removed Points 
*These points are flagged to be removed, treat them with caution*
- “Abstract does not specify baseline”: The abstract states improvement “over non‑personalized action tokenization baselines,” and Table 2 identifies ActionPiece as the strongest baseline.
- “Auxiliary model choice is insufficiently justified”: The paper includes an ablation comparing DuoRec and SASRec (Table 3) and explains DuoRec’s contrastive learning yields more distinguishable representations.
- “Hyperparameter sensitivity of the clustering scheme is a barrier”: Appendix B shows the optimal configurations are robust across datasets and provides detailed tuning ranges.
- “Explainability experiment relies on an external LLM”: This is a supplementary analysis using GPT‑4o, acknowledged as such, and does not affect the core results.
- “Missing explicit limitations section”: While not a dedicated section, limitations (e.g., dependency on an auxiliary model, need for item features) are discussed in the text and future work.

## Novel Insights
The paper’s key insight is that static tokenization in GR implicitly enforces a universal item‑similarity standard, which can be broken by conditioning semantic IDs on the user’s interaction history. This allows the same item to be interpreted differently across users, capturing personalized facets. The work demonstrates that meaningful personalization can be achieved through context‑aware tokenization, and the gains are not merely an artifact of increased token diversity (as shown by the random‑target ablation). This opens a new direction for personalization in generative recommendation.

## Suggestions
- Specify the probability aggregation method used during inference (e.g., sum or max over semantic IDs of the same item) in Section 2.3.
- Add a baseline that assigns multiple static semantic IDs per item via clustering of item features alone, to isolate the effect of personalization.
- Include an experimental comparison with MTGRec or other recent multi‑identifier GR methods to strengthen the claim that personalization is the key driver.
- Discuss computational overhead (time/memory) of the tokenization pipeline relative to static tokenization, either in the main text or appendix.
- Consider using RK‑Means as the primary quantization method, or justify the choice of RQ‑VAE more prominently, given the appendix results.

---

## bm3rbtEMFj

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (6.2/10)
- Match: N/A

### Final Review

## Summary
ELMUR is a transformer architecture augmented with layer-local external memory tracks, bidirectional token-memory cross-attention, and an LRU-based update rule. It demonstrates exceptional long-term retention, solving a synthetic T-Maze task with corridors up to one million steps while using a context window of only 10, and significantly improves performance on visual robotic manipulation and diverse memory-intensive benchmarks.

## Strengths
- **Demonstrates extreme long-horizon retention.** ELMUR achieves 100% success on the T-Maze task with inference corridors up to 1 million steps, extending effective memory horizons by 100,000x beyond its attention window (Figure 3). This directly validates the core claim of scalable long-term memory.
- **Strong, broad empirical gains.** On the MIKASA-Robo benchmark of sparse-reward visual manipulation tasks, ELMUR nearly doubles the aggregate success rate of the strongest prior method (RATE) and ranks first on 21 of 23 tasks with non-zero success (Table 1, Appendix Table 8). It also achieves the top aggregate score on the diverse POPGym-48 suite.
- **Rigorous mechanistic analysis.** The paper provides a theoretical analysis of memory retention (exponential forgetting, half-life) and stability (Proposition 1, 2). Extensive appendix analyses—including memory probing, PCA visualizations, update patterns, and attention maps (Figures 9-15)—convincingly show that performance gains stem from functional use of the external memory, not simply increased capacity.

## Weaknesses
- **Missing comparisons to key architectural baselines.** While compared to RATE and DT, the paper does not benchmark against other prominent long-context or memory-augmented transformers (e.g., Transformer-XL, Compressive Transformer, Memorizing Transformer) or state-space models like Mamba. This omission makes it difficult to fully situate ELMUR's novelty and performance within the architectural landscape.
- **Limited analysis of failure cases and task-type sensitivity.** ELMUR does not win on 24 of the 48 POPGym tasks. The paper does not analyze these cases—whether they are reactive tasks where memory is less needed, or if ELMUR has specific weaknesses—which would provide a more nuanced understanding of its capabilities and limitations.
- **Detached memory limits gradient-based credit assignment for writing.** Training uses detached memory (`sg(m^{i-1})`) between segments (Algorithm 1), preventing gradients from flowing through memory across segment boundaries. This design choice stabilizes training but may restrict the model's ability to learn *how* to write to memory based on very long-term credit assignment. The implications of this are not sufficiently discussed.
- **Exclusive focus on offline imitation learning.** All experiments are conducted in an offline IL setting. The method's efficacy and sample efficiency in online reinforcement learning—where exploration and long-horizon credit assignment are fundamental—remain untested, limiting assessment of its broader impact for RL.

## Nice-to-Haves
- A more detailed computational complexity analysis comparing FLOPs, memory footprint, and latency against baselines as context length and memory size scale, to better substantiate claims of efficiency.
- Exploration of adaptive or learned memory update policies (e.g., a learned blending factor λ) instead of the fixed LRU rule, which could be a natural extension.
- Testing on environments with multiple, interleaving long-term dependencies (beyond remembering a single cue) to more rigorously stress-test memory capacity and management.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **"The paper does not analyze the actual content and usage of memory slots."** → Extensive memory probing, PCA, and attention analyses in Appendix A.9-A.11 directly address this.
- **"The paper should evaluate on established long-horizon benchmarks like D4RL."** → Appendix A.5 (Table 4) includes results on D4RL MuJoCo tasks.
- **"The theoretical analysis is not a major advance."** → While straightforward, it provides necessary formal grounding for the empirical claims and is kept as a strength.
- **"The use of MoE-FFN is not justified."** → The ablation (Table 3) shows it is not essential for the memory mechanism; this is noted but not a core flaw.
- **Formatting nitpicks and generic strengths (e.g., "well-written")** are removed per the instructions.

## Novel Insights
The paper's most novel insight is that a simple, structured, layer-local external memory with an LRU-based update rule can enable transformers to retain information over horizons up to 100,000 times longer than their native attention window, solving extreme long-horizon POMDPs that defeat standard architectures. The combination of bidirectional cross-attention (mem2tok/tok2mem) and temporal grounding via relative bias creates a coherent read-write interface that, coupled with the LRU policy, yields bounded yet persistent storage. The extensive appendix analyses further reveal that the model learns to perform sparse, one-shot writes of task-critical variables into dedicated memory slots and preserves them with high fidelity until retrieval.

## Suggestions
- Add comparisons to Transformer-XL and/or other contemporary long-context architectures on the same benchmarks to clarify ELMUR's relative advantages.
- Include a brief analysis of the POPGym tasks where ELMUR does not achieve top performance, discussing whether the shortfall relates to task type (e.g., reactive vs. memory-intensive).
- In the limitations section, expand the discussion of the implications of using detached memory during training and the method's current restriction to offline IL.

---

## IdJakw2jta

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Long-Form Spatio-Temporal Video Grounding (LF-STVG), a new task aiming to localize targets in videos of 1-5 minutes, addressing a gap between current research on short clips and real-world applications. It proposes ART-STVG, an autoregressive transformer framework that processes frames sequentially, equipped with selective spatial and temporal memory banks and a cascaded spatio-temporal decoder. Experiments on newly extended datasets show significant improvements over state-of-the-art methods on long videos, while maintaining competitive performance on short-form benchmarks.

## Strengths
- **Novel and well-motivated problem formulation**: The paper is the first to explicitly define and tackle LF-STVG, clearly articulating the limitations of existing methods that process all frames at once and the need for scalable solutions. Evidence: The abstract and introduction state this, and related work distinguishes it from other long-term video tasks.
- **Coherent and effective architecture**: ART-STVG integrates autoregressive processing, memory banks with simple yet effective selection strategies, and a cascaded decoder where spatial localization informs temporal localization. Ablations validate each component: selective memories improve performance (Tables 2-3), and the cascaded design outperforms a parallel one (Table 4).
- **Strong empirical results**: On five extended long-form benchmarks (1-5 minutes), ART-STVG substantially outperforms existing STVG methods, with gains generally increasing for longer videos (Table 1, Figure 2). It also remains competitive on the standard short-form HCSTVG-v2 benchmark (Table 7).
- **Practical efficiency advantage**: While inference is slower due to sequential processing, ART-STVG uses significantly less GPU memory (7.9G vs. ~25G for others), making it more scalable for long videos—a key claim of the work (Supplementary Table 8).

## Weaknesses
- **Underspecified memory management**: The memory bank update is described as appending queries without removing existing memories, which could lead to unbounded growth in streaming scenarios. This matters because it affects the practicality for truly long videos and reproducibility, yet no capacity management or forgetting mechanism is discussed.
- **Lack of quantitative validation for memory selection**: The spatial (top-N similarity) and temporal (TextTiling-inspired) selection strategies are motivated heuristically, but no quantitative metrics (e.g., relevance scores or boundary accuracy) are provided to verify their effectiveness. This matters because it leaves the selection mechanisms as black boxes, reducing interpretability and confidence in the design.
- **Training-inference mismatch**: Training uses a fixed frame length (N_f=64), but inference is claimed to handle streaming, arbitrary-length videos. The paper does not clarify how this transition is managed (e.g., via truncation or sliding windows), which matters for assessing true streaming capability and generalization to very long sequences.
- **Limited ablation across video lengths**: Ablation studies (e.g., on memory selection and cascaded design) are conducted only on the 3-minute dataset, not across all five lengths. This matters because the core claim is effectiveness for long-form videos of varying durations, and the impact of components might differ with length.
- **Dataset constraints and reproducibility**: The long-form evaluation relies solely on an extended validation set of HCSTVG-v2, which may not capture full diversity, and the extended datasets are not released. This matters because it limits validation of generalizability and hinders community benchmarking for this new problem.

## Nice-to-Haves
- Comparison to adapted baselines, such as applying state-of-the-art STVG methods to video chunks with overlap, to better isolate the benefits of the autoregressive and memory design.
- More detailed failure analysis with quantitative categorization of error types (e.g., spatial vs. temporal, due to ambiguous boundaries or distractions) to guide future improvements.
- Visualization of memory selection over time, showing which frames or memories are selected during grounding, to enhance interpretability.

## Removed Points
- **Claim of being "first" overstated**: The paper correctly positions itself as the first for LF-STVG specifically, not for long-video understanding in general, as clarified in related work.
- **Demand for statistical significance**: While error bars would strengthen claims, single-run evaluation is common in this field, and the improvements are substantial and consistent.
- **Formatting issues with figures**: The garbled tables/figures are parser artifacts, not paper problems.
- **Criticism about insufficient comparison to long-form methods from other tasks**: The paper's scope is STVG; demanding comparisons to action detection or VQA methods is scope creep.

## Novel Insights
The paper identifies that current STVG methods fail on long videos due to computational bottlenecks and irrelevant information, and proposes an autoregressive approach with selective memory to address this. The cascaded decoder design, where spatial localization informs temporal localization, is a novel insight for leveraging fine-grained cues in complex long sequences. Beyond the paper's own contributions, the reviews suggest that the autoregressive framework itself might be a major factor in the gains, hinting at future work to explore simpler autoregressive baselines.

## Suggestions
- Clarify the memory bank update mechanism, including any capacity limits or forgetting strategies for streaming inference, in the main paper or supplement.
- Provide quantitative metrics for memory selection, such as calculating similarity scores for spatial memories or evaluating temporal boundary detection accuracy.
- Explicitly describe how inference handles videos longer than the training frame length, e.g., through sequential processing of segments or truncation.
- Conduct ablations on multiple long-form datasets (e.g., 1min, 5min) to show consistency of component contributions across video lengths.
- Release the extended validation sets or provide detailed documentation on the extension process (e.g., source video handling, annotation propagation) to facilitate reproducibility and benchmarking.

---

## 31CznLfRIS

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
VideoJudge introduces a bootstrapping framework to train specialized, small MLLM judges (3B/7B) for evaluating video understanding model outputs. The method uses an iterative generator-evaluator pipeline to synthesize a large-scale training dataset with aligned quality ratings, avoiding costly human annotation. The trained judges match or outperform much larger (e.g., 32B/72B) general-purpose MLLMs on several meta-evaluation benchmarks and can generate instance-specific evaluation rubrics.

## Strengths
- **Novel and scalable bootstrapping framework:** The paper presents a clearly described, iterative generator-evaluator loop (Algorithm 1) that creates over 100k rating-aligned training examples without human annotation. This addresses a critical data scarcity problem for video evaluation.
- **Strong empirical performance with small models:** VideoJudge-7B consistently matches or surpasses significantly larger models (Qwen2.5-VL-32B/72B) on multiple benchmarks (e.g., VideoJudgeLLaVA, VideoJudgeVCG, LongVideoBench). The rubric-trained VideoJudgeR-3B demonstrates that specialized fine-tuning can close the performance gap to models 10x its size.
- **Comprehensive analysis and released resources:** The paper includes valuable ablations (frames, decoding temperature) and a detailed error analysis. The release of code, models, bootstrapped datasets, and meta-evaluation benchmarks promotes reproducibility and provides essential resources for the community.

## Weaknesses
- **Substantial "closed-loop" evaluation concern:** The training data and two primary pointwise meta-evaluation benchmarks (VideoJudgeLLaVA, VideoJudgeVCG) are constructed using the same bootstrapping pipeline. While results on independent benchmarks (VATEX, VideoAutoArena, LongVideoBench) are positive, the strongest performances are on the bootstrapped benchmarks, leaving uncertainty about true generalization to human judgment.
- **Severe calibration issues and overestimation bias:** The error analysis (§6.2) reveals a consistent and critical flaw: judge models are poorly calibrated and exhibit a strong overestimation bias. For example, 46.6% of rating-3 responses are incorrectly scored as 5, and 81.3% of rating-4 responses are inflated to 5. This undermines the reliability of the judges for precise evaluation.
- **Missing methodological details affecting reproducibility:** Key parameters of the bootstrapping algorithm are not specified in the main text, such as the acceptance threshold α (used in Algorithm 1) and the exact identity of the generator (`G`) and evaluator (`E`) models during data synthesis. While some information is in the appendix, these omissions hinder precise replication.

## Nice-to-Haves
- **Ablation on bootstrapping components:** Studying the contribution of the iterative feedback loop versus a single generation pass would better characterize the framework's necessity and robustness.
- **Deeper diagnostic analysis of overestimation bias:** Investigating the root cause (e.g., data imbalance, loss function, rubric design) would move beyond observation to inform fixes.
- **Explicit cost-benefit analysis:** Quantifying the computational cost (GPU hours/API cost) of the bootstrapping phase versus the benefit of small, efficient judges would provide a more complete picture of the method's practicality.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strengths:** "The paper is well-written" (generic).
- **Weaknesses:** Demands for comparison to proprietary models (GPT-4V, Gemini) – the paper's scope is open-source models and the comparison is fair within that context.
- **Weaknesses:** Criticism that the generator/evaluator identity is unspecified – the models are identified in Section 4.1 and Appendix A.2.
- **Weaknesses:** Requests for societal impact analysis (bias amplification, environmental cost) – while valid considerations, they are not standard core requirements for a technical methodology paper at ICLR.
- **Weaknesses:** Claim that human evaluation of rubrics is unfair because baseline models were not fine-tuned – the comparison demonstrates the value of the full training recipe, which is a valid contribution.

## Novel Insights
The paper demonstrates that a carefully bootstrapped, self-consistent pipeline can generate high-quality supervision for a complex multimodal task, enabling small models to specialize and rival the evaluation capability of much larger general-purpose models. A key finding is that providing visual input is crucial for video evaluation (MLLMs outperform text-only LLMs), and extended chain-of-thought reasoning does not compensate for its absence. Furthermore, the framework shows that models can be trained to generate instance-specific evaluation rubrics, a step towards interpretable and context-grounded automated assessment.

## Suggestions
- To address the closed-loop concern, prioritize and expand analysis on all available *independent*, human-annotated benchmarks (e.g., seek additional pointwise datasets beyond VATEX) and present those results as primary evidence of generalization.
- Actively tackle the calibration weakness in revision. Propose and test a concrete mitigation, such as incorporating a balanced loss, temperature scaling, or augmenting training data with synthetically generated "hard negatives" near the top of the rating scale.
- In the main methodology section, explicitly state the acceptance threshold α and the specific models used as the generator (`G`) and evaluator (`E`) during bootstrapping to ensure full reproducibility.

---

## piylyBPSau

- GT: Reject (avg 4.0)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes GenCoGS, a method for few-shot novel view synthesis that enhances 3D Gaussian Splatting (3DGS) with two generative completion strategies. A point cloud completion strategy initializes Gaussians with a more complete scene structure, while a pseudo-view completion strategy, using an image-to-video diffusion model, optimizes Gaussians in unobserved regions. The method demonstrates state-of-the-art performance across LLFF, DTU, and Shiny benchmarks under 3, 6, and 9-view settings.

## Strengths
- **Strong and consistent empirical performance:** The method achieves significant quantitative gains over a wide range of strong baselines (NeRF-based, 3DGS-based, and diffusion-based). For example, on the challenging Shiny dataset with 3 views, it improves PSNR by +1.47 dB over the best prior method (Table 3).
- **Well-motivated and novel integration:** The paper identifies a clear limitation in prior 3DGS-based few-shot methods—their inability to reason about unobserved regions—and proposes a principled, two-pronged solution using generative completion. The unified design of complementary strategies for initialization (GCGI) and optimization (GCGO) is a novel and effective contribution to the field.
- **Extensive and thorough ablation studies:** The paper provides comprehensive experiments validating each component (GCGI and GCGO) and key design choices (e.g., perturbed trajectory, filtering modules, loss terms). Hyperparameter sensitivity is explored in the appendix, demonstrating robust design (Tables 4-6, 9, Figures 9-12).

## Weaknesses
- **Increased computational cost and system complexity:** While 3DGS is valued for efficiency, GenCoGS adds significant overhead from its generative components (diffusion model, point completion network). Training time (40 min) and memory (4 GB) are notably higher than other 3DGS baselines (Table 10). The reliance on large pre-trained models (I2V, CLIP) also increases system complexity, which is a practical limitation for deployment.
- **Partial reliance on heuristics without deeper justification:** Several core design choices, while effective, are presented as heuristics. These include the sinusoidal camera perturbation, the adaptive thresholding parameters (δ₂, δ₃) for the confidence mask, and the filtering threshold (δ₁). Although ablation studies show these values work well, a more principled discussion of why these specific forms and values are optimal is lacking.
- **Indirect evaluation of geometric improvement:** The claim that GCGI provides better geometric initialization is supported by a chamfer distance comparison to the *final optimized* Gaussians (Table 8), creating a somewhat circular evaluation. A more direct analysis against ground-truth geometry or a breakdown of where added points are correct/incorrect would strengthen this claim.

## Nice-to-Haves
- **Evaluation under more extreme sparsity:** Testing with 1 or 2 input views would further stress-test the scene completion capability and better define the method's limits.
- **More detailed analysis of failure modes:** While the paper discusses a trade-off ("see-saw effect") and includes a limitations section, a dedicated qualitative analysis of challenging cases (e.g., complex textures, transparency) would provide a clearer understanding of the method's boundaries.
- **Direct metrics for hallucination reduction:** Quantifying view consistency or semantic drift in the completed pseudo-views could provide additional, direct evidence for the claimed reduction in generative hallucination.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Architectural details are sparse"** — The method description cites established architectures (DGCNN, Transformer, FoldingNet) and provides the necessary operational detail (e.g., k-NN in Transformer blocks). Further low-level implementation details are appropriate for code, not the paper.
- **Weakness: "Cloning Gaussian attributes assumes representativeness"** — This is a standard and reasonable initialization strategy in 3DGS. The paper's approach (cloning from the nearest point in the high-confidence reference cloud) is a sensible heuristic.
- **Weakness: "Should ablate the choice of I2V diffusion model"** — The use of ViewCrafter is a reasoned choice given its focus on multi-view consistency. Requiring comparison against every alternative diffusion prior is scope creep.
- **Weakness: "Non-standard AVGE metric"** — While AVGE is less common, the paper clearly defines it and, more importantly, reports all standard metrics (PSNR, SSIM, LPIPS) prominently. The additional aggregate metric does not hinder comparison.

## Novel Insights
The paper's core novel insight is the two-pronged application of generative models to address different facets of the few-shot 3D reconstruction problem. Rather than using generation only for optimization or only for initialization, the method shows that these are complementary: point cloud completion provides crucial structural priors for initial placement, while temporally consistent pseudo-view completion provides photometric guidance for optimization in unobserved areas. Furthermore, the paper introduces simple but effective mechanisms (kd-tree filtering, adaptive confidence masking) to mitigate the hallucination inherent in these generative processes, which is a key technical insight for reliable integration.

## Suggestions
- **Provide a detailed breakdown of computational overhead:** In addition to total time/memory, analyze the cost attributed to the point completion network and the diffusion model denoising steps. This would help readers understand the trade-offs and guide future efficiency improvements.
- **Strengthen the geometric evaluation:** Consider providing a per-scene visual comparison of the initial SfM point cloud, the completed cloud **P_f**, and the final Gaussian centers to qualitatively validate the structural improvements from GCGI.
- **Include a brief discussion on the heuristic choices:** Add a short paragraph in the method or discussion section explaining the rationale behind the forms of the heuristics (e.g., why a sinusoidal perturbation) and why the chosen parameter values are sensible from a first-principles perspective (e.g., δ₁ scales with point cloud density).

---

## 1E4Bltg6Xb

- GT: Accept (Poster) (avg 4.7)
- Predicted: N/A (3.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes the Dynamics Feature Representation (DFR) framework for Reinforcement Learning (RL) in dynamic path planning. DFR hierarchically refines high-dimensional global traffic dynamics into a compact state representation using a pre-computed distance-based policy attention to sparsify the graph and an n-hop neighborhood method to localize features. The goal is to balance information completeness and computational efficiency for RL agents.

## Strengths
- **Clear and Well-Motivated Framework:** The paper clearly identifies the core trade-off between global and local dynamics in RL state representation for path planning and proposes a logically structured, three-level hierarchical refinement (global → task-related → node-related) to address it.
- **Thorough and Relevant Empirical Evaluation:** Experiments are conducted on multiple real-world urban road networks using several core RL algorithms (DQN, PPO, GCN+DQN). The evaluation includes key metrics like planning optimality gap, success rate, feature compactness, and planning time, providing a holistic view of the framework's benefits.
- **Informative Ablation Study:** A detailed analysis of the hyperparameters *k* (policy attention breadth) and *n* (neighborhood hops) provides practical insights into their effects on performance and offers sensible deployment recommendations (e.g., moderate *k*, smaller *n*).

## Weaknesses
- **Superficial Theoretical Grounding:** The connection to Predictive State Representations (PSR) is mentioned as a theoretical basis but is not developed rigorously. The claim that the refined state preserves policy optimality (Eq. 8) is asserted rather than proven or formally analyzed, missing an opportunity to strengthen the paper's theoretical contribution.
- **Strong and Unvalidated Assumption in Policy Attention:** The core "policy attention" mechanism relies on a pre-trained static shortest-path policy. This assumes that paths optimal under a static distance metric are a good proxy for the relevant subgraph under dynamic travel-time conditions. The paper does not analyze the consequences when this assumption breaks down (e.g., under severe, non-uniform congestion), which is a significant limitation of the proposed approach.
- **Incomplete Empirical Validation of Core Claims:** The paper claims DFR helps achieve a "Markovian state representation," but provides no empirical validation (e.g., by comparing performance using history versus the current DFR state). Furthermore, results are presented as averages without reporting variance measures (e.g., standard error over multiple runs), which is expected for rigorous evaluation at a venue like ICLR.

## Nice-to-Haves
- **Exploration of Adaptive Parameters:** As noted in the conclusion, a method to automatically adapt *k* and *n* based on graph properties or learned context would enhance the framework's practicality and scalability.
- **Extended Baseline Context:** While the paper's focus is internal to the RL paradigm, a comparison with a simple random subgraph sparsification baseline of comparable size could help isolate the benefit of "task-relevance" from mere dimensionality reduction.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **"Triangle visualization is unconventional and difficult to parse."** – This is a presentation/style nitpick; the underlying metrics (GAP, SR, CR) are clearly defined and reported.
- **"The term 'attention' is misleading for a static, pre-computed filter."** – While semantically debatable, the paper defines its mechanism clearly, so this criticism does not identify a factual error or substantive flaw.
- **"Lack of comparison to traditional dynamic planning algorithms (e.g., D* Lite)."** – The paper explicitly scopes its contribution to improving state representation *within* RL-based approaches, as stated in Section 5.1. Demanding comparisons outside this scope is unreasonable.
- **"Need for user studies or theoretical proofs."** – These are not standard expectations for an empirical systems paper in this domain.

## Novel Insights
The paper's core novel insight is the specific hierarchical refinement pipeline for state representation in graph-based RL: using a task-specific, pre-computed structural prior (distance-based policy attention) for coarse, global sparsification, followed by agent-centric localization (n-hop neighborhoods) for fine-grained feature extraction. This provides a practical blueprint for constructing compact, decision-relevant states in large-scale dynamic environments, balancing the often-conflicting goals of information sufficiency and computational efficiency.

## Suggestions
- **Strengthen the Theoretical Discussion:** Provide a more formal analysis or proof sketch under what conditions the DFR-compressed state preserves sufficient information for near-optimal decision-making, properly leveraging PSR concepts.
- **Add Variance Reporting:** Include standard deviations or confidence intervals for key results (e.g., Mean GAP, Planning Time) across multiple training runs to demonstrate statistical robustness.
- **Conduct a Sensitivity Analysis:** Add an experiment analyzing how the performance of DFR degrades as the optimal dynamic path deviates from the top-k static shortest paths, to better characterize the limitations of the policy attention assumption.

---

## rI2Fa13fUL

- GT: Reject (avg 5.0)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Generative Trajectory Policies (GTP), a new policy class for offline reinforcement learning that learns the solution map of a continuous-time generative ODE. To make this practical, the authors propose two adaptations: a score approximation for efficient training and a variational advantage-weighted objective for policy improvement. Empirical results show state-of-the-art performance on D4RL Gym and AntMaze benchmarks, including perfect scores on several AntMaze tasks.

## Strengths
- **Unifying Framework**: The paper elegantly frames diffusion, flow matching, and consistency models as instances of a continuous-time ODE trajectory, providing a principled foundation for designing expressive generative policies.
- **Strong Empirical Performance**: GTP significantly outperforms prior generative and offline RL methods across D4RL Gym and AntMaze suites, achieving perfect or near-perfect scores on challenging sparse-reward tasks.
- **Practical Adaptations**: The score approximation (Theorem 1) and advantage-weighted guidance (Theorem 2) are theoretically grounded and effectively address computational cost, training stability, and policy improvement, as validated through ablations.

## Weaknesses
- **Incomplete Benchmark Evaluation**: The paper claims "state-of-the-art performance on D4RL benchmarks" but only reports results on Gym and AntMaze domains, omitting Adroit and Kitchen. This gaps undermines the breadth of the claim.
- **Limited Ablation and Robustness Analysis**: Key ablations (e.g., score approximation, value guidance) are conducted only on a single task (hopper-medium-expert), leaving their general importance across tasks unverified. Sensitivity to hyperparameters like advantage temperature η and sampling horizon T is also examined on just one task.
- **Insufficient Efficiency Trade-off Substantiation**: While GTP aims to balance expressiveness and efficiency, training time comparisons with baselines are absent, and inference efficiency gains over consistency models are modest (e.g., GTP with T=2 is slower than consistency models with T=2 in Table 6). The analysis of performance versus inference steps is brief and not systematic.
- **Unanalyzed Design Choices**: The value-guidance scheme clips negative advantages (max(0, A)), which may bias the policy by ignoring suboptimal actions, but no analysis is provided on how often this occurs or its impact. Similarly, the choice of score approximation is not justified against alternatives.

## Nice-to-Haves
- Quantitative measures of multi-modal capture (e.g., mode coverage, MMD) on tasks with known multi-modality, such as AntMaze, to bolster expressiveness claims.
- Visualizations of generated trajectories in AntMaze environments to illustrate planning capabilities and failure cases.
- Exploration of alternative advantage-weighting schemes (e.g., non-clipped, adaptive temperature) to potentially improve performance and robustness.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Novelty of Unification**: Criticisms that the ODE framework is merely a synthesis of existing ideas are weakened, as the paper clearly cites prior work (CTMs, Shortcut Models) and focuses on applying the framework to RL.
- **Theoretical Contributions Are Modest**: While Theorems 1 and 2 are incremental (justifying common heuristics), they are correctly applied and support the method; thus, this point is kept but phrased as a weakness rather than invalid.
- **Missing Non-Generative Baselines**: Requests to include methods like SfBC are outside the paper’s scope on generative policies and are removed.
- **Statistical Significance Tests**: The paper reports standard deviations; demanding formal tests is a generic rigor requirement not standard in this field.
- **Network Architecture Details**: These are likely in the appendix, and their absence from the main text does not constitute a core flaw.

## Novel Insights
None beyond the paper’s own contributions.

## Suggestions
- Include results on Adroit and Kitchen domains to validate broader applicability across D4RL.
- Conduct ablation studies across multiple tasks (not just hopper-medium-expert) to confirm the importance of each component.
- Compare training times with diffusion and consistency policy baselines, and systematically analyze the performance versus inference steps trade-off across tasks.

---

## Vgm77U4ojX

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.3/10)
- Match: N/A

### Final Review

## Summary
SIGMADOCK introduces a novel fragment-based SE(3) Riemannian diffusion model for rigid-receptor molecular docking. It decomposes ligands into rigid-body fragments via a new fragmentation scheme (FR3D), defines a diffusion process over fragment poses in SE(3)^m, and uses a geometrically grounded architecture with soft triangulation constraints. The method achieves state-of-the-art performance on the PoseBusters benchmark (79.9% Top-1 success with RMSD<2Å and PB-validity), surpassing classical physics-based docking and prior deep learning methods under a fair train-test split, while generating chemically plausible poses without requiring energy minimization or a separate confidence model.

## Strengths
- **Novel and theoretically grounded framework:** The shift from torsional to fragment-based SE(3) diffusion is well-motivated by a rigorous analysis (Theorem 1) showing that disjoint rigid fragments yield a factorized product measure, avoiding the geometric entanglement inherent in torsional parameterizations. The invariance properties (Theorem 2) and the use of a Newton-Euler prediction head provide a principled foundation.
- **Strong empirical performance with careful controls:** SIGMADOCK achieves impressive results on standard re-docking benchmarks (PoseBusters, Astex) while trained only on PDBBind(v2020), isolating methodological gains from data scale. Ablations confirm the importance of key components (triangulation, protein-ligand interactions, FR3D), and analysis shows consistent generalization across sequence similarity splits and robustness to pocket size variation.
- **Practical efficiency and simplicity:** The model generates high-quality, chemically plausible poses without post-hoc energy minimization or a separately trained confidence model, relying on a cheap heuristic for sample ranking. Sampling is computationally efficient (0.57s/seed), making it suitable for high-throughput settings.

## Weaknesses
- **Potential circularity in evaluation metric:** The inference heuristic ranks samples using a score that incorporates "average PoseBusters validity checks" (Appendix F.2). If these are the *exact same* checks used to compute the final reported "PB-Valid" success rate, this introduces a circularity that could artificially inflate the PB-validity metric. The authors must clarify whether the heuristic uses a distinct, possibly stricter, set of stereochemical checks.
- **Limited evaluation to re-docking only:** The paper deliberately focuses on the rigid-receptor, holo-conformation re-docking task for benchmarking. While this is a standard and practically relevant setting, it does not demonstrate performance on more challenging scenarios like cross-docking, apo-structure docking, or flexible receptor docking, limiting claims about general applicability.
- **Fragmentation algorithm description could be clearer:** The FR3D algorithm (Appendix D.4) and the handling of chiral centers during fragmentation are described at a high level. The `ValidState` function and how it detects retained torsions after merging (`detect_torsions`) are not fully detailed, making the method somewhat difficult to reproduce or assess fully.
- **Missing broader impact statement:** The paper does not include a discussion of societal impact, which is a standard requirement for ICLR submissions. This should address potential benefits (accelerated therapeutic discovery) and risks (e.g., misuse for toxin design).

## Nice-to-Haves
- **Additional failure analysis:** A deeper breakdown of failure modes (e.g., incorrect global placement vs. local fragment assembly errors) and visualization of representative failures, especially for complexes with co-factors or high symmetry, would strengthen the understanding of limitations.
- **Assessment of sampling quality beyond RMSD:** Metrics like interaction fidelity (recovery of key hydrogen bonds) or ensemble diversity for multi-pose generation could provide a more holistic view of sample quality.
- **Direct comparison with classical docking under identical conditions:** Running a baseline like AutoDock Vina on the exact same test set with the same pocket definition used for SIGMADOCK would further solidify the claim of superiority over classical methods.
- **Architectural schematic:** A figure illustrating the construction of the input graph \(G_{\text{input}}\) with its virtual nodes and various edge types would improve clarity.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness about Figure 4 presentation:** The formatting artifacts (strikethroughs) in Figure 4 are likely parser issues and do not obscure the key numbers. This is a minor presentation nitpick.
- **Weakness demanding implementation of a torsional diffusion baseline:** The paper provides theoretical and empirical comparison to prior torsional methods (e.g., DiffDock) using reported numbers. Implementing a new torsional baseline from scratch is outside the paper's scope.
- **Weakness demanding evaluation on cross-docking or blind docking benchmarks:** The paper explicitly scopes its contribution to the standard re-docking setting for fair comparison. Expanding to other tasks is a natural future direction, not a required evaluation for this work.
- **Weakness demanding ablation of the backbone architecture vs. vanilla EquiformerV2:** The architectural modifications (smooth cutoff, bias removal) are justified, and the paper's focus is on the fragment-based framework, not an architecture bake-off.
- **Weakness about indirect comparisons with AF3/Uni-Mol:** The paper carefully discusses the differences in training data and potential test-train leakage when comparing to these models (Appendix J.2), providing a fair and contextualized comparison.

## Novel Insights
The paper's core novel insight is that representing a ligand as a set of rigid fragments and performing diffusion directly in the product space SE(3)^m simplifies the learning problem compared to torsional diffusion. This is supported by Theorem 1, which shows that torsional updates induce a non-product, entangled measure in Cartesian space due to non-local geometric couplings, while disjoint rigid fragments maintain a factorized product measure. This theoretical advantage is realized practically through the FR3D fragmentation scheme, which reduces degrees of freedom while preserving flexibility via soft triangulation constraints, and an architecture that enforces invariance to fragment local coordinate choices. The result is a model that reliably generates chemically plausible poses and generalizes well, demonstrating that carefully designed geometric inductive biases can enable deep learning docking to surpass classical methods without massive scale.

## Suggestions
- Clarify in the main text or appendix whether the PoseBusters checks used in the inference heuristic (Appendix F.2) are identical to those used for the final "PB-Valid" metric. If they are the same, modify the heuristic to use a different, independent set of stereochemical checks to avoid circular evaluation.
- Add a brief "Broader Impact" section discussing potential positive applications in drug discovery and any ethical considerations.
- In the method section (2.2.3), add a sentence or two elaborating on how FR3D's `ValidState` attempts to preserve chiral centers or handle stereochemistry, linking to the limitation mentioned in Appendix J.1.

---

## JEN4nsDgh9

- GT: Reject (avg 3.5)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a novel benchmark for evaluating text-to-image (T2I) models on the task of generating images for taxonomy concepts, specifically WordNet synsets. It proposes a comprehensive suite of nine evaluation metrics, including taxonomy-aware similarity scores grounded in information theory, pairwise preference evaluation using human judges and GPT-4, and standard image quality metrics. The benchmark evaluates 12 open-source T2I models across multiple datasets, finding that model rankings for this structured semantic task differ significantly from those on general T2I benchmarks, with Playground-v2 and FLUX emerging as top performers. The work highlights the potential of T2I models for automating the visual enrichment of taxonomic resources.

## Strengths
- **Theoretically grounded, taxonomy-specific metrics.** The paper introduces novel metrics (Lemma, Hypernym, Cohyponym Similarity, and Specificity) derived from principles of KL divergence and mutual information (Appendix D), providing a principled framework for evaluating how well generated images align with a concept's position in a taxonomic hierarchy. This moves beyond generic image-text alignment.
- **Extensive and multifaceted empirical evaluation.** The study evaluates 12 diverse, publicly available T2I models and a retrieval baseline across three carefully constructed datasets (common-sense, random WordNet split, LLM-predicted concepts) using nine metrics. The scale (thousands of concepts and pairwise comparisons) and the integration of human preferences, GPT-4-as-a-judge, and automated metrics provide a robust and holistic performance analysis.
- **Actionable findings and resources.** The paper identifies clear performance leaders and reveals that optimal models for this task are not the same as for standard T2I generation, underscoring the task's unique demands. The commitment to releasing generated images, preference data, and code enhances reproducibility and provides a valuable resource for extending visual coverage to under-represented taxonomic concepts.

## Weaknesses
- **Core metrics are intrinsically tied to CLIP's biases and knowledge.** The proposed taxonomy similarity metrics rely entirely on CLIP embeddings to approximate semantic probabilities. While the authors report high correlation with human rankings, the metrics ultimately evaluate alignment with CLIP's embedding space, which may not perfectly capture WordNet's fine-grained distinctions or be equally reliable for all concept types. This fundamental dependency is acknowledged but remains a limitation.
- **Significant, unmitigated bias in the GPT-4 evaluator.** The GPT-4 pairwise evaluation exhibits a strong positional bias (preference for the first option, as shown in Figure 5 and the confusion matrix in Figure 12). Although the Bradley-Terry model can compensate for consistent bias to produce a final ranking, the paper does not implement standard mitigation techniques (e.g., position swapping, blind model names, majority voting). This undermines confidence in the GPT-4 preference scores as a standalone reliable metric, even if the final model rankings correlate with human judgments.
- **Benchmark scope is limited to English WordNet.** The evaluation is conducted exclusively on WordNet. While this provides a clear and structured foundation, it leaves open the question of how well the findings and metrics generalize to other taxonomies (e.g., multilingual, domain-specific like medical ontologies, or differently structured resources like Wikidata). This limits the claimed generality of the benchmark.

## Nice-to-Haves
- Including a stronger, modern retrieval baseline (e.g., using a large-scale multimodal retriever like CLIP on a massive image corpus) would have provided a more rigorous comparison to establish the advantage of generation over retrieval for covering long-tail concepts.
- A more systematic quantitative analysis of failure modes (beyond the qualitative appendix), such as categorizing errors by concept type (abstract, rare, leaf vs. internal nodes) and reporting their frequency per model, would provide clearer guidelines on the limitations of current models for this task.
- Testing the proposed metrics with alternative vision-language models (beyond CLIP) would help assess their sensitivity to the choice of the underlying embedding model.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "The rationale for the skewed dataset sampling probabilities is not fully justified."** The paper states the probabilities are set based on the needs of the TaxoLLaMA training methodology (Section 2.2). This is a design choice for creating a specific test set, not a methodological flaw.
- **Weakness: "No ablation on the prompt template."** The paper explicitly analyzes the effect of adding definitions (Appendix C, Fig 6), which is the most relevant prompt variable for this task. Demanding ablation on prompt phrasing is outside the paper's stated scope.
- **Weakness: "Insufficient analysis of the LLM Predictions subset."** The paper reports performance on this subset across all metrics (e.g., Table 2, "P-" columns). A deeper comparative analysis is a potential extension but not required to support the core benchmark contribution.
- **Weakness: "No comparison to proprietary T2I models (DALL-E 3, Midjourney)."** The authors explicitly limit their evaluation to open-source models, citing practical and reproducibility reasons (Appendix A, Limitations). This is a valid scoping decision.
- **Weakness: "No ablation on the core components of the novel metrics."** The paper's contribution is the proposal and application of these metrics within a benchmark. Requiring an internal ablation study to "justify their novelty" is scope creep; their utility is demonstrated by their use in the comprehensive evaluation and their correlation with human judgment.
- **Weakness: "Weak retrieval baseline."** The retrieval baseline uses a common source (Wikimedia Commons) and serves to illustrate the practical difficulty of finding images for many concepts. The paper's goal is to evaluate generation models, not to benchmark state-of-the-art retrieval.
- **Suggestion: "Include visualization of performance trends across the taxonomy graph."** While interesting, this is a supplemental analysis, not a core requirement for presenting the benchmark and its results.

## Novel Insights
The benchmark reveals a key, non-obvious insight: the ranking of T2I models on this taxonomy-focused task diverges significantly from rankings on general T2I benchmarks. Specifically, models that excel at broad aesthetic quality and instruction following (like Playground and FLUX) top the preference-based rankings, while models optimized for text-image alignment in the CLIP space (like SDXL-turbo) lead on similarity metrics. This divergence indicates that successfully visualizing taxonomic concepts requires a balance of semantic precision, adherence to hierarchical relations, and human-judged quality that is not captured by standard T2I evaluations. Furthermore, the clear inferiority of even a simple retrieval approach for covering the long tail of concepts demonstrates a concrete practical utility for T2I generation in automating the enrichment of structured knowledge bases.

## Suggestions
- For a camera-ready version, implement and report results with standard techniques to mitigate GPT-4's positional bias in pairwise evaluation (e.g., swapping image positions in the prompt and averaging, or using anonymous model labels) to strengthen the reliability of this metric.
- Expand the error analysis in Appendix I into a more systematic, quantitative summary (e.g., a table categorizing failure modes and their prevalence across top and bottom-performing models) to provide clearer guidance on the current limitations of T2I models for taxonomy depiction.

---

## cEXEmyW77N

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper investigates whether LLM-generated bibliographies can be distinguished from human ones by analyzing their induced citation graphs. Using a dataset of 10,000 focal papers, the authors compare structural graph features and semantic embeddings from titles/abstracts, finding that structural signals alone yield near-chance discrimination (~0.60 accuracy), while semantic embeddings enable strong detection (~0.83 accuracy with RF, up to 0.93 with GNNs). The results are robust across multiple LLMs (GPT-4o, Claude) and embedding models, leading to the conclusion that LLM references mimic human citation topology but retain detectable semantic fingerprints.

## Strengths
- **Large-scale, well-controlled empirical study**: The work leverages 10,000 focal papers with carefully constructed random baselines (field-matched, subfield-matched, temporally constrained), providing strong causal isolation and statistical power.
- **Clear, incremental methodology**: The progressive analysis—from interpretable structural features to aggregated embeddings to GNNs—cleanly disentangles the contributions of topology versus semantics, making the core finding highly convincing.
- **Comprehensive robustness checks**: Findings are validated across two LLM families (GPT-4o, Claude Sonnet 4.5), two embedding backbones (OpenAI, SPECTER), and via cross-generator generalization experiments, demonstrating the consistency of the semantic fingerprint.

## Weaknesses
- **Statistical significance of structural separation not established**: The reported RF accuracy of ~0.60 for ground truth vs. GPT is marginally above chance, but the paper does not provide statistical tests (e.g., p-values or confidence intervals) to substantiate the claim that structural features "do not separate at statistically significant levels." This weakens the argument that topology alone is indistinguishable.
- **Unconventional GNN node feature construction for structural analysis**: In the GNN experiments using structural features, each node is assigned a 5D vector that includes graph-level statistics (e.g., total edge count) repeated for all nodes. This approach is non-standard and may not effectively leverage local graph structure; the justification is lacking, and it risks circular reasoning since the features are derived from the graph being classified.
- **Lack of justification or ablation for embedding aggregation**: The semantic signal is derived by summing node embeddings to a graph-level vector, but no justification is provided for this choice, and no ablation compares it to other pooling methods (e.g., mean, attention). This leaves uncertainty about whether the aggregation method optimally captures discriminative information.
- **Limited interpretation of the semantic fingerprint**: While embeddings enable high detection accuracy, the paper does not analyze which semantic dimensions (e.g., recency, prestige, topical focus) drive separability. The "semantic fingerprint" remains a black box, limiting insights for debiasing or deeper understanding.
- **Directional citation signals ignored**: Converting directed citation edges to undirected graphs simplifies topology analysis but discards potentially informative directional cues (e.g., temporal flow, citation recency). A discussion or experiment on directionality is missing, which could affect detection performance in real-world settings.

## Nice-to-Haves
- Incorporate directed graph analysis to assess whether direction-aware features improve discrimination.
- Perform feature importance analysis (e.g., via SHAP or embedding projections) to identify interpretable semantic biases driving separability.
- Conduct ablation studies comparing GNNs to MLPs on node embeddings to quantify the added value of graph structure via message passing.
- Stratify results by graph size (number of references) to ensure detection robustness across bibliography lengths.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Fuzzy-matching details missing**: The paper references prior work (Algaba et al., 2025) and includes prompts in the appendix, so reproducibility concerns are addressed.
- **Selection bias from graph removal**: The removal of graphs without generated references is a minor methodological choice that does not undermine the core findings.
- **Demand for more structural metrics**: The paper focuses on interpretable, standard graph features; requesting additional metrics is outside its stated scope.
- **Ethical implications omitted**: While relevant, this is not a core technical contribution of the paper.
- **Related work could be more critical**: The related work section adequately positions the paper; brevity is not a substantive flaw.
- **Full-text embedding comparison**: The paper explicitly scopes its analysis to title/abstract text, acknowledging this as a limitation; demanding full-text is outside its contributions.
- **Comparison to non-random baselines**: The paper's goal is to distinguish human from LLM-generated references, not to benchmark against other generators; such comparisons are beyond its scope.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add statistical significance tests (e.g., permutation tests or confidence intervals) for the structural classification accuracy to clarify whether the ~0.60 result is meaningfully above chance.
- Revise the GNN structural-feature experiments to use node-level attributes without graph-level repeats, or provide a clear justification for the current design to address concerns about circular reasoning.
- Include a brief ablation comparing different embedding aggregation methods (e.g., sum vs. mean) to demonstrate that the semantic signal is robust to pooling choice.
- Expand the discussion to analyze what semantic attributes might underlie the detection signal, perhaps by correlating embedding dimensions with bibliometric features like publication year or venue prestige.

---

## L2rfd2Czbj

- GT: Accept (Poster) (avg 6.7)
- Predicted: N/A (6.4/10)
- Match: N/A

### Final Review

## Summary
The paper proposes wd1, a novel reinforcement learning method for diffusion-based large language models (dLLMs) that reformulates policy optimization as a weighted log-likelihood objective. This eliminates the need for explicit policy ratio estimation, thereby reducing computational overhead and mitigating bias/variance from likelihood approximation errors. The method is theoretically interpreted as energy-guided diffusion training combined with negative sample unlearning. Experiments on reasoning benchmarks show significant accuracy improvements and reduced training cost compared to baselines, with an extension (wd1++) achieving state-of-the-art results on MATH500 and GSM8K.

## Strengths
- **Directly addresses a core technical challenge**: The paper clearly identifies and tackles the problem of intractable likelihoods in dLLMs, which leads to error amplification and high computational cost in existing ratio-based RL methods like GRPO. The proposed ratio-free weighted objective is a well-motivated and effective solution.
- **Strong and comprehensive empirical validation**: wd1 achieves dramatic improvements on planning-intensive tasks (e.g., +58.8% on Sudoku, +16% on Countdown) without supervised fine-tuning, while also matching or exceeding baselines on standard math reasoning. The extended wd1++ method sets new SOTA results (44.2% on MATH500, 84.5% on GSM8K) with notably fewer training steps.
- **Novel theoretical grounding**: The paper provides a novel and sound theoretical interpretation, showing that the positive component of the objective is equivalent to training an energy-guided discrete diffusion model, while the negative component relates to data unlearning. This elevates the work beyond a purely empirical contribution.
- **Rigorous ablation studies**: Ablations convincingly demonstrate the necessity of both the positive and negative weighting terms and validate the balanced combination, providing clear empirical justification for the design choices.
- **Clarity and reproducibility**: The method is clearly described with full derivations, the algorithm is presented, code is released, and experimental details (hyperparameters, rewards, datasets) are thoroughly documented in the appendix.

## Weaknesses
- **Efficiency claim for wd1++ requires clarification**: The reported "10× fewer rollouts" compares the number of *final* generated completions. However, wd1++ utilizes all intermediate denoising-step completions, increasing the total number of training samples per rollout. A fairer efficiency comparison should account for the total number of samples or forward passes used for training to accurately assess computational trade-offs.
- **Evaluation limited to a single model family**: All experiments are conducted on the LLaDA-8B model architecture. While results are compelling, demonstrating effectiveness on at least one other dLLM (e.g., Dream-7B or a SEDD-based model) would strengthen claims of general applicability across the dLLM paradigm.
- **Heuristic element in the full objective**: The negative weight term (w⁻) is introduced primarily based on empirical motivation (to fully utilize samples and actively penalize low-advantage completions). Although later connected to unlearning theory and validated via ablation, its integration into the core theoretical derivation (from reverse-KL optimization) is less direct than the positive term. A more explicit discussion of its role within the theoretical framework would strengthen the presentation.

## Nice-to-Haves
- A sensitivity analysis for the hyperparameter ψ (which controls the sharpness of the exponential weights) across different tasks would provide practical guidance for users.
- Visualizing the distribution of weights (w⁺ and w⁻) as a function of advantage during training could offer intuitive insights into the method's balancing mechanism.
- A breakdown of the computational cost (time/FLOPs) into sampling, likelihood approximation, and gradient computation components would further pinpoint the source of efficiency gains.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Ambiguity in Equation 3**: The formulation of the diffusion-GRPO objective using `min` and the clipping term is mathematically correct and aligns with standard implementations; it does not cause confusion.
- **Theoretical complexity is a weakness**: The theoretical interpretation, while dense, is appropriate for the venue and is clearly presented.
- **Reproducibility gaps**: The paper provides code, detailed hyperparameters, and dataset descriptions, meeting standard reproducibility expectations.
- **Requirement to compare to on-policy/other off-policy baselines**: The paper's comparisons to the established baseline (d1) and several strong concurrent methods (MDPO, SDPO) are sufficient to demonstrate its contribution within the current research landscape for dLLM RL.
- **Demand for analysis of approximation bias vs. more accurate estimators**: The paper's core contribution is the ratio-free objective; the choice of likelihood approximator (d1-based) is an implementation detail shared with the baseline. Analyzing other approximators is an interesting extension but not required to validate the main claim.

## Novel Insights
The paper provides a genuinely novel theoretical insight by formally connecting the weighted log-likelihood objective (derived from reverse-KL regularized policy optimization) to energy-guided discrete diffusion training. Specifically, it proves that maximizing the advantage-weighted likelihood is equivalent to minimizing an Advantage-Weighted Denoising Concrete Score Matching (AW-D-CSM) loss, which steers the diffusion model's generation toward high-advantage regions. This interpretation provides a fresh, principled perspective on RL for diffusion models beyond the standard policy gradient framework.

## Suggestions
- Clarify the efficiency metrics for wd1++ in the main text or caption of Table 3 (right), explicitly stating that "rollouts" refer to final completions and discussing the trade-off of using intermediate samples.
- Expand the discussion in the limitations section (Appendix D) to include practical mitigation strategies for the identified failure mode (e.g., when all completions in a batch receive identical rewards), such as reward shaping or adjusting the group size.

---

## GRufFX1gAy

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces InnoGym, a principled framework and benchmark for evaluating the innovation potential of AI agents. It moves beyond correctness-only evaluation by defining innovation via two complementary metrics: *Performance Gain* (improvement over best-known solutions) and *Novelty* (methodological dissimilarity). The work includes iBench, a curated set of 18 "improvable" tasks from real-world competitions, and iGym, a unified execution environment. Experiments reveal a key gap: current agents can produce novel methods but lack the robustness to translate novelty into meaningful performance gains.

## Strengths
- **Novel and Principled Evaluation Framework:** The paper clearly formalizes innovation as a combination of performance gain (*G*) and novelty (*N*), grounded in a task quadruple *(P, S, V, D)*. This dual-axis evaluation is a significant conceptual advance over standard correctness-only benchmarks, directly addressing a recognized gap in the literature.
- **High-Quality, Reproducible Benchmark Construction:** iBench is built through a rigorous, two-stage filtering process (resource availability, evaluator quality) resulting in 18 standardized tasks from credible, real-world sources. The detailed steps for normalization (absolute scoring, containerization, validation) ensure fairness and reproducibility, which is critical for community adoption.
- **Valuable and Well-Supported Core Finding:** The experiments convincingly demonstrate that existing general-purpose agent frameworks fail to surpass human state-of-the-art and, crucially, that high novelty often does not correlate with high performance due to a lack of robustness. This insight—"the primacy of robustness over novelty"—is an important corrective for the field and is well-supported by the success rate analysis and main results.

## Weaknesses
- **Insufficient Validation of the Core Novelty Metric:** The novelty score *N(s)* is instantiated via an "Agent-as-judge" procedure using proprietary models (Codex, GPT-5). While Appendix F provides initial validation on small, hand-picked triplets (8 from EquiBench, 3 from other domains), this is inadequate to establish the metric's reliability, robustness, and freedom from bias for a benchmark intended for community-wide use. The correlation with human judgment needs to be demonstrated across a larger, more representative set of solutions from the benchmark's own tasks.
- **Limited Experimental Breadth Undermines Benchmark Claims:** The main evaluation covers only 10 of the 18 curated tasks, justified by computational constraints. While practical, this significantly weakens the claim of providing a comprehensive evaluation of the benchmark. Furthermore, the primary agent comparison uses a single backbone LLM (DeepSeek-v3.1) for most tasks, limiting the generalizability of conclusions about the agent frameworks themselves.
- **Superficial Analysis of Agent Failures:** The paper notes that agents frequently fail to produce valid submissions on complex tasks (e.g., CDML, PTTALC) but offers only high-level explanations (e.g., "intricate data formats"). A deeper, qualitative analysis of specific failure modes (e.g., planning breakdowns, tool misuse, syntax errors) would greatly enhance the benchmark's diagnostic value for guiding future research.

## Nice-to-Haves
- Increase the number of independent runs per configuration and report measures of variance (e.g., confidence intervals) to strengthen the statistical robustness of the agent comparisons.
- Expand the discussion on how the *G/N* framework would apply to the other task categories in the taxonomy (Solved and Exploratory problems), perhaps with a brief pilot study, to clarify the framework's generalizability.
- Include a more direct comparison of iGym's features (e.g., recovery, concurrency) against existing agent SDKs to substantiate claims about its added value.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "The paper lacks a detailed discussion of the diversity of the 18 tasks."** -> The paper provides Table 3 with domain, source, and a calculated diversity score (`Div`). A deeper taxonomy is not required for the core contribution.
- **Weakness: "Running only three trials and reporting the best score leads to optimistic estimates."** -> The paper explicitly states this follows the protocol of MLE-Bench, a standard in the field. The request for more runs is a generic rigor requirement not universally expected for large-scale agent benchmarks.
- **Weakness: "The choice of rubric dimensions for novelty scoring seems arbitrary."** -> The rubric dimensions are detailed in Appendix H.2. While their design could be discussed, their absence is not a critical flaw.
- **Suggestion: "Evaluate on all 18 curated tasks."** -> This is impractical due to the stated computational constraints and does not invalidate the findings on the evaluated subset.
- **Suggestion: "Compare against state-of-the-art innovation agents like AlphaEvolve."** -> The paper's scope is evaluating *general-purpose* agent frameworks on innovation potential. Including specialized systems like AlphaEvolve is a different comparison and is not required.
- **Suggestion: "Include non-LLM baselines (e.g., random search)."** -> The paper's focus is on AI *agents*, which are inherently LLM-based. Comparing to non-agent baselines is outside its stated scope.

## Novel Insights
The paper provides a novel synthesis by rigorously defining and measuring innovation as a two-dimensional construct. The key empirical insight—that current agents can generate methodologically novel solutions but are fundamentally bottlenecked by an inability to produce robust, correct implementations—is significant. This clearly identifies a research challenge distinct from pure performance optimization or idea generation: true innovation requires coupling novelty with executional reliability. The complex-plane visualization for solution trajectories also offers a genuinely new way to interpret the joint evolution of performance and novelty.

## Suggestions
- Conduct a more extensive validation of the novelty metric *N(s)*. Execute a human evaluation study where domain experts rate the novelty of a stratified sample of agent-generated and baseline solutions across multiple iBench tasks. Report correlation coefficients (Pearson/Spearman) between these human ratings and the automated *N(s)* scores to properly establish metric validity.
- Deepen the failure mode analysis. For a subset of tasks where agents consistently fail (e.g., CDML, PTTALC), categorize and provide concrete examples of the most common error types (e.g., invalid tool calls, incorrect file handling, logical flaws in generated code). This would transform the benchmark from a report card into a diagnostic tool.
- To address the limited model breadth, include a supplemental experiment where one agent framework (e.g., the best-performing MLAB) is evaluated across all 10 main tasks using 2-3 different, strong backbone LLMs (including open-source options). This would help decouple framework capabilities from model capabilities.

---

## vGkXf8nvt9

- GT: Reject (avg 4.7)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Forget-to-Focus (F2F), a two-stage protocol that first applies targeted unlearning on a “forget” set of general-domain data to remove irrelevant pretraining knowledge, then fine-tunes on a domain-specific dataset. The method consistently improves performance over standard fine-tuning across coding, medical, and mathematical tasks for models ranging from 0.6B to 72B parameters, accompanied by analysis showing representational shifts and better calibration.

## Strengths
- **Novel and impactful repurposing of unlearning**: The work reframes machine unlearning from a privacy tool to a deliberate intervention for domain specialization, demonstrating clear empirical gains across diverse domains (e.g., +32.5% pass@1 on HumanEval for Qwen-0.6B) and model scales.
- **Rigorous and multi-faceted evaluation**: Experiments span multiple model families, sizes, and domains, with in-depth mechanistic analysis via centered kernel alignment, SVCCA, Fisher information, PCA shifts, and calibration studies, providing convincing evidence that unlearning reshapes representations and reduces overconfidence.

## Weaknesses
- **Heuristic and under-specified forget set construction**: The forget sets (BC-Select, BC-Cosine) rely on manual curation or cosine similarity without clear criteria for domain-irrelevance, making the method difficult to reproduce and generalizing uncertainly to new domains.
- **Limited calibration evidence**: Improved calibration is shown only on medical QA (MedMCQA); without results from coding and mathematical tasks, the claim that F2F enhances reliability broadly is not fully supported.
- **Oversimplified theoretical analysis**: The convex linear model analysis (Proposition and Corollary) is a severe simplification of non-convex LLM optimization and does not meaningfully justify why the gradient ascent/descent procedure works in practice.
- **Inconsistent experimental setups**: Comparisons across model scales are confounded by varying fine-tuning strategies (e.g., Qwen-72B uses 50% of the dataset and QLoRA, while smaller models use full data and SFT), undermining fair assessment of F2F’s scalability.
- **Representation analysis lacks performance linkage**: While CKA and SVCCA show representational shifts after unlearning, the paper does not correlate these changes with accuracy gains, leaving it unclear which geometric alterations are beneficial for specialization.

## Nice-to-Haves
- Control experiments matching total optimization steps between F2F and standard fine-tuning to isolate the effect of unlearning from additional training.
- Broader evaluation on general-knowledge benchmarks (e.g., MMLU) to comprehensively verify that core capabilities are preserved.
- A more principled approach to constructing forget sets, such as using gradient-based influence scores to identify harmful knowledge automatically.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Formatting artifacts from PDF parsing**: These are extraction issues, not problems with the paper.
- **Demand for exhaustive baseline comparisons**: The paper includes standard fine-tuning, DAPT, LoRA, and CurlLoRA; requiring all possible parameter-efficient methods is scope creep.
- **Nitpicks about abstract omissions**: The abstract summarizes key contributions appropriately; missing caveats is not a substantive flaw.
- **Claim that theoretical section is an “afterthought”**: Subjective and not factually incorrect; the section is provided as intuitive motivation.

## Novel Insights
None beyond the paper’s own contributions.

## Suggestions
- Detail the forget set construction process, including the specific Transformer encoder used for cosine similarity, similarity thresholds, and manual curation criteria, to ensure reproducibility.
- Extend calibration analysis (reliability diagrams, ECE) to coding and mathematical benchmarks to substantiate claims about improved reliability across domains.
- Acknowledge the limitations of the convex theoretical analysis and supplement it with empirical observations on optimization dynamics (e.g., loss landscape or gradient conflict) to better ground the method’s intuition.

---

## iDki7djO2K

- GT: Reject (avg 4.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a novel, general definition of forgetting in machine learning as a violation of self-consistency in a learner's predictive distribution over future experiences. This yields a measure, the propensity to forget, which the authors empirically validate across regression, classification, generative modeling, continual learning, and reinforcement learning, demonstrating that forgetting is pervasive and influences learning efficiency.

## Strengths
- **Novel and unifying theoretical framework:** The paper provides the first algorithm- and task-agnostic definition of forgetting based on predictive self-consistency, moving beyond paradigm-specific definitions and offering a principled foundation for analyzing information retention. The formalism is carefully constructed (Sections 3–4) and motivated by insightful thought experiments (Appendix C).
- **Comprehensive empirical validation:** The paper supports its theory with experiments across diverse learning settings (supervised, generative, RL, CL), consistently showing non-zero forgetting and revealing dynamics such as trade-offs with efficiency and spikes at task boundaries. This breadth strongly supports the claim that forgetting is a fundamental property of learning.
- **Actionable insights and measure:** The derived propensity to forget measure varies meaningfully with hyperparameters (e.g., momentum, batch size, architecture) and correlates with learning dynamics, providing a new tool for analysis. The empirical demonstration that optimal training efficiency often occurs at non-zero forgetting is a particularly intriguing finding.

## Weaknesses
- **Limited scalability and practical utility of the measure:** The propensity to forget requires computationally expensive particle-based rollouts (e.g., 1000 particles, k=40 steps) as described in Algorithm 1 and Appendix D.1. The paper does not address the feasibility of applying this measure to large-scale models (e.g., modern LLMs or vision transformers) or discuss approximations that would make it practical for real-time analysis.
- **Insufficient empirical scale to fully support claims:** While the experiments cover multiple paradigms, they are largely on simple tasks (sinusoid regression, two-moons classification, cartpole RL). The single CIFAR-10 experiment (Figure 11) is a step toward larger-scale validation but is not enough to substantiate the claim that "forgetting is everywhere" in deep learning. More challenging benchmarks are needed to generalize the findings.
- **Lack of comparison with existing forgetting metrics:** The paper does not quantitatively compare its measure against standard forgetting metrics (e.g., average forgetting, backward transfer) from continual learning literature. Such a comparison is necessary to demonstrate that the proposed measure better disentangles forgetting from backward transfer or provides unique insights beyond established metrics.
- **Theoretical scope and interpretation in non-stationary environments:** The definition treats any violation of self-consistency as forgetting, which may include rational adaptation in non-stationary or model-misspecified settings. While thought experiments (Appendix C) address some edge cases, the interpretation under realistic distribution shifts or model misspecification remains unclear, and the hybrid distribution \(q_e\) is not fully specified for complex environments.

## Nice-to-Haves
- Ablation study on the sensitivity of the measure to the number of particles and horizon \(k\) to ensure robustness.
- Visualizations of predictive distribution evolution for a simple regression task to concretely illustrate what specific capabilities are forgotten.
- Discussion of how the measure could be approximated more efficiently (e.g., via fewer particles, shorter horizons) for practical use as a diagnostic tool or regularizer.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Overstatement of novelty:** The claim that "no unified definition has emerged" is strong but supported by the paper's review of existing definitions and the genuinely new perspective offered.
- **Complexity of the two-update formalism:** The distinction between learning-mode \(u\) and inference-mode \(u'\) updates is justified to separate belief updates from auxiliary state evolution and is necessary for the theory.
- **Missing broader impact discussion:** While a discussion of societal implications is often valuable, this foundational paper focuses on theoretical and empirical contributions; however, a brief consideration of potential impacts would be beneficial.
- **Demands for user studies or extensive theoretical proofs beyond scope:** The paper appropriately combines theoretical formalism with empirical validation; requiring additional theoretical derivations or user studies is not standard for this type of contribution.

## Novel Insights
The paper's core insight is that forgetting can be characterized as a lack of self-consistency in a learner's predictive distribution, which naturally yields a general measure. This reframes forgetting from a failure mode limited to specific settings to a fundamental property of learning dynamics. The empirical finding that optimal training efficiency often occurs at non-zero forgetting suggests that forgetting is not merely a negative phenomenon but a regulated aspect of learning that can be beneficial, providing a new perspective for analyzing and designing learning algorithms.

## Suggestions
1. Conduct experiments comparing the propensity to forget with existing forgetting metrics on standard continual learning benchmarks (e.g., Split MNIST/CIFAR) to validate its advantages in disentangling forgetting from backward transfer.
2. Demonstrate the measure on at least one larger-scale dataset and architecture (e.g., ResNet on CIFAR-100) to show scalability and provide more convincing evidence for the pervasiveness of forgetting in deep learning.
3. Include a discussion on how to approximate the measure more efficiently (e.g., via variance reduction techniques) and its potential use as a diagnostic tool or regularizer in algorithm design.
4. Clarify the interpretation of the definition under model misspecification and non-stationary environments, possibly via additional thought experiments or analysis in Section 4.2.

---

## x6bG2Hoqdf

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
CALM introduces a co-evolution framework for automatic heuristic design that jointly optimizes prompt generation (verbal guidance) and the underlying LLM via reinforcement learning (numerical guidance). It incorporates novel evolutionary operators, a collapse mechanism, and a tailored reward function for efficient fine-tuning. Experiments demonstrate that CALM outperforms state-of-the-art baselines across multiple optimization tasks while running on a single 24GB GPU with a compact 7B model.

## Strengths
- **Novel co-evolution paradigm**: The integration of RL-based LLM fine-tuning directly into the evolutionary heuristic search loop addresses a clear gap in fixed-model AHD methods, enabling the model to adapt and improve over time (Sections 1, 4).
- **Strong empirical performance**: CALM consistently outperforms SOTA baselines (including API-based methods) on four challenging tasks (OBP, TSP, CVRP, OP) in both in-domain and out-of-domain settings, as shown in Tables 1–3, with significant p-values in Appendix I.6.
- **Resource efficiency and practicality**: The method runs locally on a single 24GB GPU using a 7B INT4-quantized model, fine-tuning only 1.15% of weights, making it accessible without costly API dependencies (Section 5, Appendix I.1).
- **Thorough ablation studies**: Ablations in Table 4 and Section 5.2 validate the contribution of key components (RL fine-tuning, collapse mechanism, reward design, operators), providing evidence for design choices.

## Weaknesses
- **Hyperparameter sensitivity not fully characterized**: While ablations exist for some settings (e.g., reward parameters in Appendix I.7), a systematic sensitivity analysis across all tasks and hyperparameters (e.g., collapse parameters, operator weights) is lacking, affecting reproducibility and robustness.
- **Lack of quantitative diversity metrics**: The paper claims diversity-aware operators aid exploration, but no measure of heuristic diversity (e.g., code edit distance, idea overlap) over time is provided, making it difficult to verify this claim and understand search dynamics.
- **Incomplete ablation coverage**: Ablation studies in Table 4 focus primarily on OBP and OP; similar analyses for TSP and CVRP are missing, weakening the generalizability of the findings about component contributions.
- **Reward distribution unreported**: The distribution of rewards during training (e.g., frequency of infeasible, duplicate, or improving heuristics) is not analyzed, obscuring the quality of the learning signal and model adaptation.
- **Limited statistical runs**: Main results average over three runs, which, despite p-values for some tasks, reduces statistical confidence; more runs would strengthen the empirical claims.

## Nice-to-Haves
- Comparison with a supervised fine-tuning baseline using curated heuristics to isolate the effect of co-evolution versus fine-tuning on static data.
- Ablation on the number of responses per prompt (G) to understand its impact on RL training and advantage estimation.
- Visualization of heuristic diversity and performance over time to illustrate search dynamics and collapse effects.
- Case studies comparing discovered heuristics with seeds to explain performance improvements qualitatively.
- Extension of sensitivity analyses to all hyperparameters across all tasks.

## Removed Points
These points are flagged to be removed, treat them with caution:
- Request for explicit bulleted contributions in the introduction (formatting nitpick).
- Criticism about generality to non-code heuristics (outside the paper's stated scope of code-based AHD).
- Concern over EvoTune re-implementation (justified in paper for fair comparison under resource constraints).
- Demand for theoretical convergence analysis (not standard for an empirical systems paper).
- Suggestion to compare computational cost with API-based baselines (time breakdown is provided, and API costs are variable).

## Novel Insights
None beyond the paper's own contributions. The paper successfully introduces and validates the co-evolution paradigm, but the reviews do not surface additional novel insights beyond what is presented.

## Suggestions
- Incorporate quantitative diversity metrics (e.g., code edit distance or idea token overlap) to validate the effectiveness of diversity-aware operators and collapse mechanism.
- Extend ablation studies to all tasks, particularly TSP and CVRP, to ensure component contributions are consistently beneficial across problems.
- Report the distribution of rewards during training to provide insight into the learning process and signal quality.
- Consider increasing the number of runs for main experiments to enhance statistical robustness, or include confidence intervals where feasible.

---

## M14YpuTejd

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper provides a critical analysis of the emerging two-stage protocol for online map-based motion prediction in autonomous driving. It identifies three core misconceptions: a train-validation gap from inappropriate dataset splits, a mismatch between the limited perception range of map models and the broader needs of motion prediction, and non-discriminative evaluation metrics. To address these, the authors propose OMMP-Bench, a new benchmark featuring a spatially-disjoint data partition, refined evaluation on non-ego moving agents, and a novel baseline that uses image features via deformable attention to supplement missing map information for distant agents.

## Strengths
- **Compelling Problem Diagnosis**: The paper provides clear, evidence-backed identification of foundational flaws in the existing protocol. It demonstrates a severe train-val performance gap (Table 1) due to map accuracy distribution shift and convincingly argues that evaluating only the ego vehicle (which is often within the map's limited range) misrepresents the task's true challenge.
- **Well-Motivated and Actionable Solutions**: Each identified issue is met with a direct, practical solution. The new three-way data split is spatially disjoint to eliminate overlap and the train-val gap. The refined evaluation focuses on all moving non-ego agents and separately reports performance for agents within and outside the map's range, providing a more honest and informative assessment.
- **Effective and Insightful Baseline**: The proposed image-feature baseline is a simple yet effective application of deformable attention to provide environmental context for agents beyond the online map's coverage. It consistently improves performance for distant agents across multiple model combinations (Table 7), directly addressing a key limitation of prior work.

## Weaknesses
- **Analysis Limited to a Single Dataset**: All experiments and the proposed benchmark are built exclusively on nuScenes, as it is currently the only public dataset with aligned sensor, map, and trajectory data. While necessary, this limits the demonstrated generality of the findings. The paper should discuss whether the identified issues (e.g., range mismatch) are inherent to the problem or might vary with dataset characteristics (e.g., sensor suite, urban layout).
- **Incomplete Mechanistic Analysis of the Baseline**: While the image-feature baseline shows strong empirical results, the paper does not deeply analyze *how* it helps. A qualitative analysis or visualization demonstrating that the retrieved image features correspond to meaningful environmental context (e.g., road edges) for faraway agents would solidify the claim that the mechanism supplements missing map information, rather than just providing a generic feature boost.
- **Validation of the New Split's Generality**: The performance improvement from the new split is demonstrated with two motion prediction models (HiVT, DenseTNT) and two map models (MapTR variants). While sufficient, testing the split with a wider variety of modern architectures (e.g., StreamMapNet, MTR) would more robustly prove that it universally mitigates the train-val gap and is not sensitive to specific model choices.

## Nice-to-Haves
- A brief discussion on the computational or latency overhead of the image-feature baseline would be useful for practitioners considering its deployment.
- Providing a visualization of the geographic boundaries of the proposed map-train, motion-train, and motion-val splits would offer immediate, intuitive proof of their spatial disjointness.
- A simple ablation comparing the image-feature baseline against a naive alternative (e.g., using a learned token or no map for out-of-range agents) would more cleanly establish the value added by the image features specifically.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "Incremental nature of technical contributions"**: The core contribution is the holistic critique and reformalization of the protocol, not solely the image-feature module. The paper's significance lies in its systematic analysis and benchmark creation.
- **Weakness: "Lack of comparison with end-to-end paradigms"**: The paper explicitly scopes itself to the two-stage protocol to isolate the impact of map errors, stating that end-to-end approaches entangle prediction with detection/tracking errors. Criticizing the absence of this comparison is scope creep.
- **Suggestion: "Controlled experiment with a low-accuracy, large-range map model"**: This experiment is effectively provided in the paper. Table 2 shows map accuracy plummets with extended range, and Table 3 shows motion prediction with ground-truth maps of different ranges, implying that a poor-quality large-range map would not help.
- **Suggestion: "Direct metric comparison on the same models (old vs. new metrics)"**: This comparison is inherent in the paper's results. Table 6 and the surrounding text explicitly show that performance is near-perfect for static agents and much worse for moving non-ego agents, demonstrating why the old ego-only metric is non-discriminative. A side-by-side table of the same model under both metrics is redundant.
- **Criticism: "Reproducibility concerns due to lack of split details"**: The paper states the split is manually checked to be spatially disjoint and commits to releasing code and checkpoints. The methodology is acceptable for a benchmark paper; exact scene indices can be released with the code.

## Novel Insights
The paper's primary novel insight is the holistic synthesis of several subtle but critical flaws that collectively mislead evaluation in online map-based motion prediction. While issues like spatial overlaps in nuScenes have been noted elsewhere, the paper uniquely connects them to create a coherent critique: the standard protocol (1) uses a data split that creates a large train-val gap in map quality for the motion predictor, (2) evaluates only the ego vehicle, which often resides within the map's limited range, thereby hiding the severe problem of missing map context for most relevant agents, and (3) uses metrics inflated by easy-to-predict static agents. The proposed OMMP-Bench framework systematically resolves these interconnected issues, establishing a more rigorous foundation for future research.

## Suggestions
- Add a subsection or paragraph discussing the generalizability of the identified protocol flaws. Comment on whether the range mismatch, for instance, is a fundamental challenge of camera-based online mapping or if it might be less severe with different sensor modalities (e.g., LiDAR) or in other datasets.
- Strengthen the analysis of the image-feature baseline. Include a qualitative case study or attention visualization for a few faraway agents to show that the model is attending to relevant image regions (e.g., road lanes, intersections), providing evidence for the proposed mechanism beyond quantitative metrics.
- In the final version, ensure all table placeholders (e.g., "Col2", "Col3" in Table 7) are replaced with descriptive headers and that all figure/table references in the text are correct and present.

---


# Summary

Papers: 97 | Accuracy: N/A
