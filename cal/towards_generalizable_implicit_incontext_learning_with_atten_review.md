=== CALIBRATION EXAMPLE 55 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution: a method for generalizable implicit ICL via attention routing. The abstract clearly states the problem (limited generalizability of vector-based implicit ICL), the proposed solution (In-Context Routing), and the key outcome (strong performance and OOD generalization). The claim that ICR is the first "train-once-and-reuse" method for zero-shot inference across diverse new tasks is bold and needs to be substantiated in the experiments.

### Introduction & Motivation
The introduction effectively sets up the limitations of explicit ICL (cost, brittleness) and vector-based implicit ICL (poor generalization, post-hoc steering). The empirical probe using multi-task ICL (Figure 1) is a compelling motivator, showing that latent cross-task patterns exist but are noisy in explicit prompting. The core research question—can we internalize ICL for seamless generalization?—is well-posed.

**Major Concern: Novelty and "Internalization" Claims**
The paper argues that prior work "falls short of utilizing the structural mechanisms" and that ICR is the first to "internalize" ICL. This is an overstatement. Work like Function Vectors (Todd et al.) explicitly identifies and manipulates causal attention heads. The distinction needs refinement: ICR's novelty lies in modulating attention *logits* via *low-rank, query-conditioned compositions* of *PCA-derived directions* from pooled multi-task data. This is a specific operationalization of "attention routing." The introduction should more precisely differentiate ICR's mechanism from prior attention-based interventions to avoid appearing to unfairly dismiss related work.

### Method / Approach (Sec. 2 & 3)
The overall pipeline (Fig. 3) is clear. The idea of extracting Principal ICL Directions (PIDs) via PCA on pooled Q/K projections and using a router to apply low-rank biases is novel and well-described.

**Theoretical Foundation (Sec. 2.3 & Appendix A): Significant Weakness**
The theoretical justification is the paper's weakest point and must be strengthened for ICLR.
1.  **Spiked Covariance Assumption:** Equation 4 presents a spiked covariance model decomposition (shared `S`, domain-specific `B`) as a given, without derivation or justification. Why should the Q/K covariance from ICL prompts follow this exact form? This is a central, unverified assumption.
2.  **Hand-wavy Averaging Argument:** The claim that domain-specific directions `B` "average out toward isotropy" if "sufficiently diverse and lack consistent alignment" (p.3) is speculative and crucial to the argument that PCA recovers shared structure. This needs empirical validation or a more rigorous statistical argument.
3.  **Appendix A Analysis:** The Davis-Kahan bound application is standard but doesn't provide deep insight. The theory feels like a post-hoc rationalization rather than a guiding principle. **Suggestion:** Provide empirical evidence that the top eigenvectors from the pooled covariance are indeed more stable (e.g., have higher cosine similarity across different data subsamples) or lead to better transfer than those from single domains. This would substantiate the core claim.

**Router Design & Training:**
The query-conditioned router using a frozen external encoder (MiniLM) is sensible. However, it raises a question: Why not use the LLM's own representations (e.g., the query's last token hidden state) to condition the router? This would better maintain consistency with the LLM's feature space. An ablation or justification is needed.
The training losses are well-designed. The minor impact of `L_conf` (Table 4) suggests it may not be essential.

### Experiments & Results
The experimental setup is extensive, covering 3 models, 12 datasets, and clear ID/OOD splits.

**Critical Baseline Omission:**
The most significant experimental gap is the **lack of comparison with a strong, direct attention manipulation baseline**. The paper contrasts ICR with methods that add vectors to residual streams. To truly validate the "attention routing" paradigm, you must compare against a method that directly learns to bias attention logits *without* the proposed PID structure. For example:
- **Learned Attention Bias:** A baseline that learns a full (or low-rank) bias matrix `B` to add to the attention logits of the last few layers, conditioned on the same query encoder. This would test whether the PID extraction and composition mechanism is key, or if any learned attention modulation suffices.
- **Task-Specific Direct Steering:** Could a method compute a `ΔA` directly from a few retrieved demonstrations (e.g., via cross-attention between query and demo tokens) and apply it? While perhaps less efficient, it would be a strong *capability* baseline.
Without these comparisons, it's unclear how much gain comes from the sophisticated PID/routing design versus simply allowing the model to learn to tweak attention.

**Other Baseline Concerns:**
- The application of training-free methods (TV, FV, ICV) for ID tasks needs clarification: were they applied in their standard, task-specific way (using demonstrations from the target dataset)? If so, it's an unfair comparison to ICR's multi-task-trained router. The comparison should be framed carefully.
- For OOD, the multi-task few-shot baseline ("each ID dataset provides 3-shot ICDs") needs more detail: total ICD count, concatenation order, shuffling procedure. This affects reproducibility and interpretation.

**Main Results (Table 1):**
Results are strong. ICR consistently outperforms implicit ICL baselines (I2CL, LIVE, M²IV) on both ID and OOD, and often beats multi-task few-shot on OOD. The "Collapse" metric is useful. The gains on far-OOD (e.g., +6.5% on Qwen2.5-7B) are impressive and support the generalization claim.

**Ablation Study (Sec. 4.3 & Appendix G):**
Generally thorough and supports design choices.
- The rank ablation (r=4,8,12) shows a clear trade-off: lower rank helps ID but hurts far-OOD, indicating a diversity-coverage tradeoff in the subspace.
- The finding that balanced sampling beats similarity-based sampling for PID extraction (Table 12) strongly supports the hypothesis that diverse exemplars are needed to capture general patterns.
- The late-layer intervention result (Table 13) is important and well-explained: early-layer attention routing disrupts low-level syntax.

**Analysis (Sec. 5):**
- **5.1 "ICLness" Tokens:** The analysis is intriguing but somewhat superficial. Listing tokens like "illustrated" and "constitution" as evidence of reasoning shifts is suggestive but not conclusive. A deeper analysis linking these token biases to actual changes in attention maps for specific examples would be more compelling.
- **5.2 Domain Distributions:** This is an excellent experiment. Table 5 clearly shows that matched and diverse extraction/training data (MATCHED-5) is crucial for OOD generalization, empirically supporting the theoretical motivation.
- **5.3 Hierarchical Analysis:** The layer, head, and PID importance visualizations are insightful. They show ICR identifies shared "hub" layers/heads while allowing task-specific adaptation in others, demonstrating structured and interpretable behavior.

### Writing & Clarity
The paper is very well-written, logically structured, and easy to follow. Figures are clear.
**Minor Issue:** Notation slightly conflates head-level (`A^{l,h}`) and layer-level (`ΔA^l`) operations. The transition in Eq. 3 and Eq. 10 could be more explicit.

### Limitations & Broader Impact
**Major Omission:** There is no dedicated limitations section. This is a critical shortcoming for an ICLR submission. Key limitations that must be acknowledged include:
1.  **Theoretical Gaps:** The spiked covariance model justification is heuristic and needs stronger empirical or theoretical grounding.
2.  **Dependence on Multi-Task Data Collection:** ICR requires a pre-collected set of labeled examples from multiple source domains for PID extraction. This "train-once" cost is non-trivial and limits applicability compared to true zero-shot or single-task methods.
3.  **Router Generalization Boundaries:** The router is trained on a specific mix of five ID domains. Its performance on *truly* far-OOD tasks (beyond the seven tested) is unknown and likely to degrade.
4.  **Scalability:** While intervening only in the last 1/3 of layers mitigates overhead, storing and applying `U_q, U_k` for all layers in very large models (e.g., 70B+) adds non-negligible memory/compute.
5.  **Incomplete Baseline Comparison:** As noted, the lack of a direct attention bias baseline leaves the core novelty claim partially unverified.
6.  **Comparison to PEFT:** The LoRA comparison (Appendix E.2) is good, but a broader comparison with other few-shot PEFT methods (prompt tuning, (IA)³) would better contextualize ICR's efficiency/performance trade-off.

Broader impact is not discussed, which is acceptable for this technical work.

### Overall Assessment
This paper presents a novel and promising method for implicit in-context learning. The core idea—routing attention logits via query-conditioned composition of PCA-extracted directions—is interesting and well-executed. Empirically, the results are strong, demonstrating consistent improvements over existing implicit ICL methods and impressive out-of-domain generalization. The ablation studies and analyses are thorough and support the design.

However, for acceptance at ICLR, the paper must address significant concerns:
1.  **Sharpen the novelty claim** relative to prior attention-steering work.
2.  **Substantially strengthen the theoretical justification** for PID extraction (Sec. 2.3), moving beyond hand-wavy averaging arguments.
3.  **Include a critical baseline** that tests direct attention manipulation without the PID/routing mechanism.
4.  **Add a comprehensive limitations section** acknowledging the method's dependencies and boundaries.

If the authors can convincingly address these points—particularly by providing a more rigorous foundation for the PID extraction and adding the missing baseline—the paper would make a strong contribution suitable for ICLR. In its current form, the contribution is promising but requires these revisions to meet the conference's high standard.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **In-Context Routing (ICR)**, a novel implicit in-context learning method that modulates attention logits instead of injecting residual shift vectors. ICR extracts reusable "Principal ICL Directions" (PIDs) via PCA from multi-task ICL demonstrations and employs a small, trainable query-conditioned router to adaptively bias attention computations. The method aims to capture generalizable cross-task ICL patterns, enabling a "train-once-and-reuse" framework that improves zero-shot inference without task-specific retraining or retrieval.

### Strengths
1.  **Strong Empirical Results & Comprehensive Evaluation:** The paper provides extensive experiments across 12 diverse datasets (5 ID, 7 OOD) and three open-source LLMs (Llama2-7B, Qwen2.5-7B, Llama3.1-8B). ICR consistently outperforms prior implicit ICL baselines (e.g., I2CL, LIVE, M²IV) and often matches or surpasses few-shot prompting, particularly on OOD tasks where baselines struggle. The robustness (zero "collapse" cases) is noteworthy.
2.  **Effective OOD Generalization:** A core claim is that ICR internalizes generalizable ICL patterns. The results substantiate this, showing significant gains on far-OOD tasks (e.g., CB, COPA, CREAK) where multi-task few-shot and vector-based methods frequently degrade. The analysis linking OOD stability to pooled PCA and the Davis-Kahan theorem (Appendix A.4) provides theoretical support.
3.  **Novel Paradigm and Theoretical Analysis:** The shift from post-hoc residual steering to "attention routing" is a conceptually novel direction for implicit ICL. The theoretical motivation using a spiked covariance model to justify that PIDs capture shared, domain-stable patterns (Sec. 2.3, Appendix A) adds depth beyond purely empirical work.
4.  **Practical Efficiency:** ICR maintains the inference efficiency benefits of implicit ICL (no demonstration tokens). The analysis shows faster inference than few-shot prompting, especially for longer contexts, and a manageable parameter cache (2*r*d*L). The "train-once-and-reuse" capability enhances practical utility.

### Weaknesses
1.  **Unexplained Performance Gaps on Specific Tasks:** On some tasks, especially CSQA with Llama2-7B, ICR's absolute performance remains very low (24.8%) despite relative gains. The paper does not adequately diagnose why the method fails to lift performance closer to the task's ceiling, leaving doubts about its effectiveness for certain reasoning types.
2.  **Theoretical-empirical Gap:** While the spiked covariance model offers intuition, the connection between this population-level analysis and the actual PCA performed on a finite, constructed set of ICL prompts is not rigorously bridged. The theory suggests benefits from more domains, but the empirical ablation (Table 5) shows mixed results, and the claim that domain-specific directions "average out" is more of a plausible hypothesis than a proven guarantee.
3.  **Dependence on External Encoder and Design Choices:** ICR's router is conditioned on representations from a frozen MiniLM encoder. The impact of this external component's quality and domain bias is not fully ablated (only one alternative encoder is tested). The method also involves several non-trivial design choices (e.g., applying routing only to the last third of layers, using last-token Q/K for PID extraction, specific loss weights) whose optimality is not thoroughly justified.
4.  **Limited Analysis of Attention Mechanism Interaction:** The paper states ICR works by "internalizing ICL dynamics" but provides limited causal evidence. While the layer/head/PID importance analyses are insightful, they are correlative. A more mechanistic analysis (e.g., how routing alters specific attention heads known to be crucial for ICL like induction heads) would strengthen the claim of fundamentally steering attention mechanisms.

### Novelty & Significance
**Novelty:** The concept of "attention routing" is a novel and meaningful contribution to the implicit ICL landscape. Moving the intervention point from residual streams to attention logits and using a low-rank, PCA-derived subspace for reusable patterns differentiates ICR clearly from prior vector-based methods. The query-conditioned router is also a novel adaptive element.
**Significance:** The work is significant for both practical and research reasons. Practically, it demonstrates a train-once-and-reuse method with strong OOD generalization, pushing toward more deployable implicit ICL systems. From a research perspective, it successfully argues that more generalizable ICL signals exist in the attention geometry and provides a viable framework for exploiting them, opening a new direction for improving model adaptability.

### Suggestions for Improvement
1.  **Deepen Analysis on Task-Specific Failures:** Investigate and discuss the poor absolute performance on tasks like CSQA with Llama2-7B. Is it due to the PIDs lacking relevant reasoning directions, the router's failure, or the base model's inherent weakness? Error analysis and comparison of attention patterns with successful explicit ICL could be revealing.
2.  **Strengthen the Theoretical Connection:** Provide empirical validation for the core theoretical claim—that PCA on multi-domain ICL bases indeed isolates a shared, low-rank subspace while dispersing domain-specific variance. This could involve measuring the alignment between PIDs from different domain subsets or analyzing the eigenvalue spectrum of the pooled covariance.
3.  **Ablate Key Dependencies:** Conduct a more systematic ablation on the role of the frozen text encoder (e.g., try using the LLM's own embeddings, test more encoders, or explore removing it). Justify the choice of last-token Q/K extraction more convincingly by comparing the information content versus alternative pooling strategies mentioned in Appendix G.4.
4.  **Clarify Comparison to Fine-tuning Baselines:** The comparison to LoRA (Appendix E.2) is brief. A more detailed discussion on the trade-offs (parameter efficiency, OOD generalization, training data needs) between ICR and parameter-efficient fine-tuning methods would better position ICR within the broader adaptation landscape.
5.  **Improve Readability of Figures:** Some figures (e.g., Fig 1) suffer from parsing artifacts that make them hard to interpret. In the final version, ensure all figures are clearly legible and their captions fully explain the observed phenomena.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Lack of baseline comparisons with non-vector-based implicit ICL methods.** The paper only compares to vector-based methods (e.g., I2CL, LIVE) and few-shot prompting. To properly establish ICR's advancement, it should be compared to other attention-modifying approaches (e.g., adapters, prompt tuning) and recent generalizable implicit ICL methods. Without this, the claim that ICR is a superior paradigm is not fully substantiated.
2. **Limited evaluation on model scale and architecture.** Experiments are conducted only on 7B/8B decoder-only models (Llama2, Qwen2.5, Llama3.1). To support claims of generalizability, the method must be tested on significantly larger models (e.g., 70B) and different architectures (e.g., encoder-decoder models). Failure to do so leaves the scalability and architectural robustness in doubt.
3. **No ablation on the frozen text encoder.** The router relies on MiniLM embeddings; the paper does not test how sensitive performance is to this choice. Using a different encoder (e.g., MPNet) or removing the encoder entirely (using the LLM's own representations) would reveal whether the routing mechanism is robust or brittle to representation quality.
4. **Absence of evaluation on truly novel, unrelated tasks.** The "out-of-domain" tasks are still NLP classification/QA tasks. To test the core claim of generalizable implicit ICL, the method should be evaluated on completely different task types (e.g., code generation, mathematical reasoning, or dialogue) that were not represented in the training domains.

### Deeper Analysis Needed (top 3-5 only)
1. **Qualitative interpretation of Principal ICL Directions (PIDs).** The paper claims PIDs capture general ICL patterns but provides no analysis of what these directions represent. Without probing the PIDs (e.g., via token activation or projection onto known linguistic features), it is unclear whether they encode meaningful, reusable structures or just noise.
2. **Empirical validation of the spiked covariance model assumption.** The theoretical justification relies on the assumption that shared ICL patterns dominate domain-specific variations. The authors should empirically verify this by analyzing the alignment of top eigenvectors from individual domain covariances with the pooled PIDs. Without this, the theoretical foundation is not convincing.
3. **Analysis of how ICR actually changes attention patterns.** The claim that ICR "routes attention" is not backed by any visualization or quantitative comparison of attention maps. Showing how attention distributions shift for the same input with and without ICR (and comparing to few-shot ICL attention) is necessary to confirm the proposed mechanism.
4. **Interpretability of the router's outputs.** The routing vectors \(\alpha(x)\) and gates \(\gamma(x)\) are learned but not analyzed. Are they consistent for inputs from the same task? Do they vary meaningfully across tasks? Analyzing their patterns would help validate that the router is making sensible, input-adaptive decisions.

### Visualizations & Case Studies
1. **Visualization of attention map changes.** For a set of representative examples (both ID and OOD), plot the attention maps of key layers/heads for zero-shot, few-shot ICL, and ICR. This would visually demonstrate whether ICR successfully mimics the attention patterns of explicit ICL, directly supporting the "attention routing" claim.
2. **Case studies of failure modes.** The paper reports no collapse below zero-shot, but there are tasks where ICR improves only marginally or underperforms few-shot. Examining specific instances where ICR fails would expose its limitations (e.g., when task-specific content in demonstrations is irreplaceable) and guide improvements.
3. **Analysis of "ICLness" tokens in context.** Table 16 lists tokens consistently upweighted by ICR, but it is not shown how these tokens influence predictions. For a few examples, show the token distributions and how the highlighted tokens (e.g., "capture", "connections") appear in the model's reasoning or output.

### Obvious Next Steps
1. **Ablation on the diversity and number of source domains for PID extraction.** The paper uses five fixed domains. Systematically varying the number and type of source domains during PID extraction would reveal how domain diversity affects OOD generalization, which is central to the method's claimed benefit.
2. **Ablation on token selection for PID extraction.** PIDs are extracted from the last token's Q/K representations. The paper should justify this choice by comparing alternatives (e.g., pooling over all demonstration tokens, or using the query token) and show the impact on performance.
3. **Cross-model transferability of PIDs.** A strong test of generalizability would be to extract PIDs from one model and use them to route attention in a different model (of similar or different scale). This would demonstrate whether the captured patterns are model-agnostic, significantly amplifying the contribution.
4. **Analysis of computational cost and scalability.** The paper briefly mentions efficiency but does not detail the cost of PID extraction (data collection, PCA) as model size and domain count increase. Providing scaling laws would help assess the practical feasibility of ICR for larger deployments.

# Final Consolidated Review
## Summary
This paper introduces In-Context Routing (ICR), a novel implicit in-context learning method that modulates attention logits via query-conditioned composition of low-rank directions extracted from multi-task demonstrations. ICR aims to capture generalizable ICL patterns, enabling a train-once-and-reuse framework that improves zero-shot inference without task-specific retraining or retrieval.

## Strengths
- **Strong and robust empirical performance**: ICR consistently outperforms prior implicit ICL baselines across 12 diverse datasets (5 in-domain, 7 out-of-domain) and three LLMs (Llama2-7B, Qwen2.5-7B, Llama3.1-8B), showing particular gains on out-of-domain tasks where other methods collapse. It achieves zero collapses below zero-shot and often matches or surpasses few-shot prompting.
- **Novel paradigm of attention routing**: The shift from post-hoc residual vector injection to directly steering attention logits via PCA-extracted Principal ICL Directions (PIDs) and a learnable router represents a meaningful advance in implicit ICL, supported by thorough ablations (e.g., PID rank, layer intervention, domain diversity).
- **Practical efficiency and generalization**: ICR maintains the inference speed benefits of implicit ICL (faster than few-shot for long contexts) and demonstrates a train-once-and-reuse capability with strong out-of-domain transfer, validated by analysis linking OOD stability to pooled PCA and domain alignment.

## Weaknesses
- **Theoretical justification requires stronger empirical validation**: The core assumption—that PCA on multi-domain ICL bases isolates a shared low-rank subspace while domain-specific directions "average out"—is motivated via a spiked covariance model but lacks direct empirical proof. The theory provides intuition but does not rigorously bridge to the finite-data PCA actually performed.
- **Missing baseline to isolate the contribution of PID structure**: The paper compares ICR to vector-based implicit ICL methods but omits a direct attention manipulation baseline (e.g., learning a full or low-rank attention bias without PIDs). This gap makes it unclear how much gain stems from the PID extraction and routing mechanism versus simply allowing learned attention modulation.
- **Unexplained performance gaps on specific tasks**: Despite overall gains, absolute performance remains low on some tasks (e.g., CSQA with Llama2-7B at 24.8%). The paper does not diagnose whether this is due to missing reasoning directions in PIDs, router limitations, or base model weaknesses, leaving uncertainty about method effectiveness for certain reasoning types.
- **Limited causal analysis of attention changes**: While layer/head/PID importance analyses are insightful, they are correlative. There is no direct visualization or quantitative comparison of how ICR alters attention maps relative to zero-shot or few-shot ICL, weakening the claim of "internalizing" ICL dynamics mechanistically.

## Nice-to-Haves
- Evaluation on non-NLP tasks (e.g., code generation, mathematical reasoning) to test generalizability beyond the text classification/QA domains used in training.
- Cross-model transferability analysis: extracting PIDs from one model and using them to route attention in a different model to assess model-agnostic patterns.
- More detailed scaling laws for computational cost (PID extraction, training) as model size and domain count increase.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **"No evaluation on larger models"**: Addressed in Appendix E.3 with results on Qwen3-32B and Llama3.1-70B.
- **"No ablation on the frozen text encoder"**: Addressed in Appendix G.5 with an ablation using mpnet encoder.
- **"No ablation on token selection for PID extraction"**: Addressed in Appendix G.4 comparing last-token, pooling, and attention-rollout strategies.
- **Formatting and style nitpicks** (e.g., notation conflation, figure readability issues) are minor and do not affect scientific contribution.
- **"Missing comparison with non-vector-based implicit ICL methods"**: While interesting, the paper's scope is implicit ICL, and it adequately covers prominent vector-based baselines; demanding unrelated methods is scope creep.

## Novel Insights
The paper demonstrates that generalizable in-context learning patterns can be captured in low-rank subspaces of attention logits, extracted via PCA on pooled multi-task query/key representations. By conditioning a lightweight router on input queries to compose these directions, ICR enables zero-shot inference to mimic the attention dynamics of explicit ICL without demonstration tokens, showing robust out-of-domain transfer. This positions attention routing as a viable paradigm for internalizing ICL mechanisms beyond post-hoc residual steering.

## Suggestions
- Add a direct attention manipulation baseline (e.g., a learned low-rank bias matrix conditioned on the query encoder) to isolate the contribution of PID extraction and routing versus generic attention modulation.
- Provide empirical validation for the spiked covariance assumption, such as analyzing the eigenvalue spectrum of pooled versus per-domain covariances or measuring subspace stability across domain subsets.
- Include a limitations section acknowledging dependencies on multi-task data collection for PID extraction, potential degradation on very far-out-of-distribution tasks, and the theoretical gaps noted above.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0, 6.0]
Average score: 5.0
Binary outcome: Reject
