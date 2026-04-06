=== CALIBRATION EXAMPLE 61 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title:** Appropriately reflects the core contribution (attention routing for implicit ICL).
- **Abstract:** Clearly states the problem with existing implicit ICL (limited generalizability) and claims a novel method (ICR) that outperforms prior work and generalizes robustly. Claims are bold but appear supported by the extensive experiments. The abstract is well-written and sets clear expectations.

### Introduction & Motivation
- **Strengths:** Clearly articulates the limitations of explicit ICL and vector-based implicit ICL. The research question (“Can we design an implicit ICL method that truly internalizes ICL?”) is well-motivated. The use of multi-task ICL as an empirical probe (Figure 1) effectively motivates looking beyond residual vectors to attention patterns.
- **Concerns:** None major. The introduction flows logically and contributions are stated clearly.

### Method / Approach (Sections 2 and 3)
- **Section 2 (Attention Routing):**
  - **Clarity:** The transition from vector-based steering to attention routing is logical, but the initial formulation in Sec 2.2 is incomplete. Equation 3 introduces a routing vector α^l but does not specify how it is obtained; this is only clarified in Sec 3.2 with the router. This could confuse readers.
  - **Theoretical Grounding:** The spiked covariance model (Sec 2.3) and perturbation analysis (Appendix A.3-A.4) provide intuitive motivation for why pooled PCA might recover shared patterns. However, the analysis is heuristic and not a rigorous proof. While acceptable for a ML paper, claims should be tempered accordingly (e.g., “suggests” rather than “proves”).
  - **Design Choice:** In Eq. 3, the same bias ∆A^l is added to every head’s logits. The later introduction of head gates (γ) in Sec 3.2 mitigates this, but the initial description should note that head-specific modulation will be added.

- **Section 3 (ICR):**
  - **PID Extraction:** Requires offline collection of Q/K representations from explicit ICL across multiple domains using labeled demonstrations. This step is costly and assumes access to labeled data from diverse domains. The paper does not discuss how sensitive performance is to the number and choice of domains (beyond Table 5). An ablation on the number of domains would strengthen the analysis.
  - **Router Design:** The use of a separate frozen text encoder (MiniLM) is not justified. Why not use the LLM’s own representations? This choice should be explained or ablated.
  - **Training Objective:** The combination of cross-entropy, confidence alignment, and sparsity losses is well-motivated. The layer-increasing sparsity weight is heuristic but reasonable.
  - **Reproducibility:** The method is described in detail with pseudocode (Appendix C). Hyperparameters are provided in Sec 4.1 and Appendix D.2, but a consolidated table would improve reproducibility.

### Experiments & Results (Section 4)
- **Experimental Setup:** Comprehensive. Models (Llama2, Qwen2.5, Llama3.1) and datasets (5 ID, 7 OOD) are diverse. Baselines include a wide range of implicit ICL methods. Evaluation protocol (500 test instances, 3 seeds) is sound.
- **Main Results (Table 1):** ICR consistently outperforms implicit ICL baselines on both ID and OOD tasks, often matching or exceeding few-shot prompting. The “Collapse” metric (zero-shot underperformance) shows ICR is robust while baselines sometimes fail. These results strongly support the paper’s claims.
  - **Key Question:** The paper claims ICR is “train-once-and-reuse.” However, PID extraction is tied to the specific domains used. If applying ICR to a completely new set of tasks (e.g., code generation), must PIDs be re-extracted? The OOD generalization shown is promising, but the limits of this generalization are not explored. This is an important limitation.
- **Ablation Studies (Sec 4.3 & Appendix G):** Thorough. Ablations on PID rank, loss components, ICD sampling, routing layers, and pooling strategies provide strong evidence for design choices. However, missing an ablation on the number of domains used for PID extraction.
- **Efficiency Analysis (Appendix F):** Shows ICR is more parameter-efficient than few-shot and has faster inference than explicit ICL. The offline cost (GPU hours) is comparable to per-task baselines, but amortizes over multiple tasks.

### Analyses (Section 5)
- **Interpretable Effects (5.1):** The “ICLness” token analysis (Appendix H) is interesting but somewhat subjective. The method for identifying tokens is sound, but the connection to “reasoning-oriented structures” is qualitative.
- **Domain Distributions (5.2):** Table 5 shows that aligned and diverse domains improve OOD generalization, supporting the theoretical motivation.
- **Hierarchical Internalization (5.3):** Layer, head, and PID importance analyses are insightful and demonstrate that ICR captures structured, task-adaptive patterns.

### Writing & Clarity
- Overall well-written and logically structured. Some minor clarifications needed:
  - The notation for Q/K (head vs. layer level) requires careful reading.
  - The transition from Sec 2.2 to Sec 3.2 could be smoother.
- The appendix is extensive and provides necessary details.

### Limitations & Broader Impact
- **Limitations:**
  1. **Offline PID Extraction:** Requires labeled demonstrations from multiple domains. This limits applicability in settings where such data is unavailable.
  2. **Domain Dependence:** While OOD generalization is shown, the method’s performance on tasks radically different from the extraction domains (e.g., code, math) is untested.
  3. **Access to Model Internals:** Requires Q/K projections, so not applicable to black-box API models.
  4. **Theoretical Gaps:** The analysis in Sec 2.3 is heuristic, not rigorous.
- **Broader Impact:** Not discussed. The paper should include a statement (even if brief) on potential societal impacts, e.g., efficiency benefits vs. transparency concerns.

### Overall Assessment
This paper presents a novel and well-executed approach to implicit in-context learning. The core idea—modulating attention logits via extracted principal directions—is innovative and appears to yield substantial improvements in generalization over existing methods. The experiments are comprehensive, ablations are thorough, and analyses provide convincing evidence that ICR internalizes meaningful ICL patterns. While there are limitations (offline extraction, domain dependence, theoretical heuristics), the empirical results are strong and the contribution is significant for the ICL community. The paper meets ICLR’s standards for novelty, rigor, and clarity. With minor revisions to address the noted concerns (especially clarifying limitations and the method’s domain dependence), it would be a strong candidate for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes **In-Context Routing (ICR)**, a novel implicit in-context learning (ICL) method that modulates attention logits via extracted Principal ICL Directions (PIDs) and a query-conditioned router. Unlike prior vector-based implicit ICL methods that inject shift vectors into residual streams, ICR aims to internalize reusable, cross-task ICL patterns in the attention space, enabling a train-once-and-reuse framework. Experiments across 12 datasets and multiple LLMs show consistent improvements over implicit ICL baselines and robust out-of-domain generalization.

### Strengths
1. **Novel and well-motivated approach**: The shift from additive residual vectors to attention logit modulation via low-rank PIDs is a principled advance. The paper clearly identifies limitations of existing implicit ICL (limited generalizability, post-hoc steering) and proposes a well-reasoned alternative grounded in attention mechanisms.
2. **Comprehensive empirical validation**: Extensive experiments on 12 datasets (5 in-domain, 7 out-of-domain) and multiple LLMs (Llama2-7B, Qwen2.5-7B, Llama3.1-8B) demonstrate consistent gains over strong baselines. ICR avoids performance collapse on OOD tasks—a key weakness of prior methods—and often matches or exceeds few-shot prompting.
3. **Theoretical and analytical depth**: The paper provides a theoretical foundation using spiked covariance models and Davis–Kahan perturbation theory to justify why PIDs capture general ICL patterns. Ablations (PIDs rank, routing layers, loss components) and analyses (layer/head/PID importance, interpretable token shifts) rigorously validate design choices and offer insights into the method’s workings.

### Weaknesses
1. **Limited comparison to broader ICL literature**: While comparisons to implicit ICL baselines are thorough, the paper does not situate ICR against other relevant approaches for improving ICL generalization (e.g., prompt tuning, meta-ICL, or retrieval-augmented methods). A broader comparison would better contextualize its contributions.
2. **Computational overhead of PIDs extraction**: The offline step of collecting ICL bases and computing PCA across multiple domains requires running thousands of ICL prompts, which may be costly. Although inference is efficient, the upfront cost and dependency on multi-domain data could limit practicality in resource-constrained settings.
3. **External encoder dependency**: The router relies on a frozen text encoder (MiniLM) for query conditioning. While an ablation shows performance improves with a stronger encoder, the impact of this external component—and alternatives such as using the LLM’s own representations—is not deeply explored, adding potential complexity.

### Novelty & Significance
The paper introduces a novel paradigm—attention routing—for implicit ICL, moving beyond additive residual vectors to structural modulation of attention logits via extracted cross-task directions. The idea of reusing generalizable ICL patterns from multi-domain attention statistics is innovative and well-supported by theory and experiments. The demonstrated out-of-domain generalization addresses a key limitation of existing implicit ICL methods, pushing the boundary toward more practical, efficient ICL. The work is significant for the ICL community and aligns well with ICLR’s emphasis on novel, impactful methods with solid empirical and theoretical grounding.

### Suggestions for Improvement
1. **Broaden baseline comparisons**: Include comparisons to other lines of work that aim to improve ICL generalization, such as prompt tuning (e.g., soft prompts), meta-ICL, or adapter-based fine-tuning. This would help clarify ICR’s relative advantages and limitations within the broader landscape.
2. **Quantify and discuss upfront costs**: Provide more explicit details on the computational cost (GPU hours, memory) of PIDs extraction for different model sizes and domain counts. Discuss how the method scales and its sensitivity to the number and diversity of domains used.
3. **Explore router conditioning alternatives**: Investigate conditioning the router on the LLM’s own internal representations (e.g., pooling over query hidden states) to eliminate the external encoder dependency. An ablation or discussion of this direction would strengthen the method’s self-contained nature.
4. **Clarify limitations and failure modes**: Explicitly discuss scenarios where ICR may underperform (e.g., tasks requiring novel reasoning patterns absent from PIDs, or when domain diversity is insufficient). This would provide a more balanced view and guide future work.
5. **Improve accessibility of theoretical sections**: While the theoretical analysis is a strength, some parts (e.g., Section 2.3, Appendix A) are dense. Adding more intuitive explanations or illustrative examples could make these insights more accessible to a broader audience.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Compare ICR to standard soft prompt tuning methods (e.g., Prefix Tuning, Prompt Tuning).** The router generates low-rank attention biases conditioned on the input, which is conceptually similar to learning input-conditioned prompts. Without this comparison, it's unclear if ICR's gains stem from a genuinely new mechanism or are simply a re-implementation of established conditioning techniques.
2.  **Ablate the role of the frozen text encoder (MiniLM).** The router is conditioned on an external encoder's representation of the query. A critical experiment is to replace this with a simple projection from the LLM's own [CLS] or mean-pooled token embeddings. The current design conflates the contribution of the routing mechanism with that of an external, potentially powerful, semantic encoder.
3.  **Evaluate on a true "no labeled data" out-of-domain (OOD) benchmark.** The OOD evaluation uses datasets with their own test labels. A more compelling test of generalization is to evaluate on entirely new task *formats* not seen during PID extraction or training (e.g., mathematical reasoning, code generation). This would directly test the claim of "seamless generalization across diverse ICL scenarios."
4.  **Test on longer-context and more complex tasks.** The paper uses standard classification/QA benchmarks. ICR's core claim is about structural attention routing; this should be validated on tasks where attention patterns are more critical, such as multi-hop reasoning (HotpotQA) or long-document understanding (NarrativeQA), to show it doesn't just work on short, simple inputs.

### Deeper Analysis Needed (top 3-5 only)
1.  **Quantify *how much* ICL pattern is captured vs. ignored.** The analysis in Sec. 5.1 lists tokens ICR upweights, but this is indirect. Perform a direct probing study: train a linear classifier on attention maps from explicit ICL to predict the task, then see if the attention maps produced by ICR are similarly classifiable. This would prove the "internalization" claim.
2.  **Analyze the failure modes of PIDs.** The ablation shows random orthogonal bases hurt OOD performance, but why? Analyze the correlation between PCA eigenvalues of the ICL bases and per-PID importance weights (α). If only a few high-variance directions are ever used, it suggests the method is mostly noise filtering, not leveraging a rich subspace.
3.  **Disentangle the contribution of the confidence loss (`L_conf`).** The ablation shows dropping `L_conf` has mixed effects. This loss is meant to prevent entropy inflation, but its empirical benefit is unclear. A deeper analysis should show if it primarily prevents catastrophic failure on a subset of samples or provides a consistent, small boost.
4.  **Provide a more mechanistic explanation for layer importance results.** Figure 4 shows "hub layers," but this is a post-hoc observation. The authors should hypothesize *why* these specific layers (e.g., 23, 26 in Llama2) are hubs—do they correspond to known "induction" or "reasoning" layers from prior work? Link the empirical finding to known transformer circuitry.

### Visualizations & Case Studies
1.  **Visualize attention maps before and after ICR routing for specific queries.** For a few OOD examples, show the zero-shot attention map and the ICR-modulated map. Highlight if ICR successfully redirects attention to semantically relevant parts of the query (e.g., from a distractor to a key premise), providing concrete evidence of "routing."
2.  **Case studies of where ICR fails vs. few-shot ICL.** The paper notes ICR sometimes underperforms few-shot on ID tasks. Pick 2-3 such instances and analyze them: is it because the query requires *content* from demonstrations that PIDs cannot encode? This would clarify the method's fundamental limitations.
3.  **Trace the influence of individual PIDs.** For a given input, visualize how the final prediction logit changes as each PID's weight (`α`) is ablated. This would show if PIDs specialize (e.g., one handles negation, another handles causal links) or act as a monolithic block.

### Obvious Next Steps
1.  **Apply ICR to decoder-only generation tasks, not just classification/QA.** The evaluation is limited to next-token prediction for multiple-choice. The method should be tested on open-ended generation (e.g., summarization, translation) to see if the attention routing generalizes beyond manipulating logits over a fixed answer set.
2.  **Explore merging ICR with retrieval-augmented methods.** The paper positions ICR against methods needing retrieval. A logical next step is to combine them: use a retriever to find a few relevant examples, extract their PIDs on-the-fly, and let the router integrate them. This hybrid approach could be stronger than either alone.
3.  **Conduct a sensitivity analysis on the number and choice of source domains for PID extraction.** The results in Table 5 show mismatched domains hurt. A systematic study is needed: how does performance scale with the number of source domains? Is there a point of diminishing returns? Are some domain combinations universally better?
4.  **Test the transferability of the trained router across different LLM families.** The router is trained on one model (e.g., Llama2-7B). An obvious check is to freeze the router and PIDs extracted from Model A, and apply them to Model B (e.g., Qwen2.5). If the method captures general patterns, it should transfer, at least partially.

# Final Consolidated Review
## Summary
This paper proposes In-Context Routing (ICR), a novel implicit in-context learning method that modulates attention logits via Principal ICL Directions extracted from multi-domain demonstrations. ICR aims to internalize reusable ICL patterns, enabling robust generalization without task-specific retrieval or retraining.

## Strengths
- Introduces a new paradigm of attention routing for implicit ICL, structurally modulating attention logits via low-rank biases derived from cross-task attention statistics, moving beyond additive residual vectors. Evidence: Method in Sec 2-3, empirical gains in Table 1.
- Demonstrates strong out-of-domain generalization without performance collapse, addressing a key limitation of prior implicit ICL methods. Evidence: Table 1 shows ICR has zero collapses and consistently outperforms baselines on OOD tasks.
- Provides extensive empirical validation across multiple models and datasets, supported by thorough ablations and analyses that offer insights into the method's workings and interpretability. Evidence: Sec 4.2, 4.3, and Section 5.

## Weaknesses
- The PID extraction process requires offline access to labeled demonstrations from multiple domains, limiting applicability where such data is unavailable and incurring upfront computational cost. The paper does not ablate the sensitivity to the number of source domains, leaving the scalability unclear.
- The router conditions on representations from an external frozen text encoder (MiniLM) without justification for why the LLM's own representations are not used or exploration of alternatives, adding unnecessary complexity and potential dependency.
- The theoretical analysis in Sec 2.3, while intuitive, is heuristic and lacks rigorous proofs, which may overstate the claims about why PIDs capture general patterns.
- The evaluation of OOD generalization, though comprehensive, is confined to classification and QA tasks similar to training domains. A test on entirely novel task types (e.g., code or math) would better validate the claim of "seamless generalization."

## Nice-to-Haves
- Compare ICR to other conditioning methods like soft prompt tuning to better contextualize its contributions.
- Explore router conditioning using the LLM's own internal representations to eliminate the external encoder dependency.
- Conduct a sensitivity analysis on the number and diversity of source domains used for PID extraction.
- Test ICR on longer-context or more complex reasoning tasks to validate its attention routing mechanism.

## Novel Insights
The paper introduces the concept of attention routing and shows that generalizable ICL patterns can be extracted as low-rank directions in the attention space. Through hierarchical analyses, it demonstrates that ICR adaptively routes attention via shared hub layers and heads, internalizing ICL dynamics in a task-aware manner. These insights advance the understanding of how ICL can be structuralized beyond post-hoc steering.

## Suggestions
- Clarify the transition from Eq. 3 to the router in Sec 3.2 to improve readability.
- Add a dedicated limitations section discussing the domain dependence of PID extraction and the external encoder choice.
- Justify the use of the external encoder or provide an ablation with LLM-based conditioning.
- Include an ablation on the number of source domains for PID extraction to strengthen the scalability analysis.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0, 6.0]
Average score: 5.0
Binary outcome: Reject
