=== CALIBRATION EXAMPLE 35 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is clear and poses a direct, intriguing research question relevant to the AI-for-Science and reasoning communities. The abstract succinctly outlines the problem (accuracy-based evaluation ignoring reasoning dependence), the method (systematic deletion framework), the domain (physics), key findings (accuracy under 40-60% deletion, "cramming," inconsistent recovery), and the implication (need for faithfulness assessment). The claims are specific and seem supported by the described experiments.

### Introduction & Motivation
The introduction effectively establishes the "faithfulness gap" in CoT reasoning and motivates its investigation in the context of physics and AI-for-Science. The need to move beyond mere accuracy evaluation is well-argued. The three stated contributions are clear and map directly onto the paper's structure. A minor point: the transition from the general faithfulness problem to the physics domain could be slightly more detailed, explaining why physics is a *uniquely* stringent testbed beyond its structured nature (e.g., the consequences of unfaithful reasoning in scientific applications).

### Method / Approach
This is the core of the paper and contains both strengths and significant concerns.
- **Systematic Deletion Framework:** The core idea is simple, novel, and powerful for probing reasoning dependence. The three deletion strategies (end, random, physics-aware) are well-chosen to test different hypotheses.
- **Reproducibility Gaps:** A major weakness is the lack of precise technical details on *how* the CoT is intercepted and deleted "mid-generation." Do the authors pause generation, edit the prompt+partial-generation context, and then continue? This is a critical implementation detail that must be clarified for reproducibility. The description "before decoding" is insufficient.
- **Evaluation Metrics:** The choice of metrics is problematic and threatens the validity of the conclusions.
    1.  **LLM-as-Judge (Claude-4):** Using a proprietary LLM to score correctness on a 0-1 scale for physics problems introduces an uncalibrated black-box component. For a study critiquing the faithfulness of reasoning, relying on another model's potentially unfaithful or biased judgment is a serious methodological flaw. The authors should provide evidence of the judge's reliability (e.g., correlation with human scores, detailed rubric) or use more objective metrics (e.g., exact match of final numeric answer, unit correctness).
    2.  **Information Overlap Metrics:** Jaccard similarity and Manhattan distance on Bag-of-Words are extremely shallow for quantifying the recovery of *reasoning*. They measure token overlap but cannot assess the logical coherence, correctness, or ordering of recovered steps. Two texts sharing the same equations but in a different, incorrect order could have high lexical overlap but represent flawed reasoning. The analysis based on these metrics is therefore weak evidence for "reconstruction" or "recovery."
- **Physics-Aware Deletion:** Using Claude-4 to annotate physics-specific tokens creates a circular dependency and potential bias. A rule-based or keyword-based approach would be more transparent and controllable.
- **Calibration:** The sample size calibration (Fig. 8) is appropriate, but the description is confusing ("5 prompts" – likely meaning 5 independent runs per problem?).

### Experiments & Results
- **Baseline (Prompting Styles):** Figure 2 confirms the expected result that more explicit reasoning leads to higher scores, establishing a sensible baseline.
- **Deletion Sweeps:** The primary results (Figs. 3, 4, 5, 6, 9-14) are compelling and visually clear. The consistent trends—accuracy plateaus then drops, while answer length increases ("cramming")—are robust across models, datasets, and deletion strategies. This is strong empirical evidence that models are not strictly dependent on the full CoT trace.
- **Analysis of "Cramming":** The observation is interesting, but the analysis remains at the output level. The claim that models "reconstruct" lost reasoning is supported only by the weak overlap metrics and length increase. The authors do not distinguish between useful reconstruction (correctly reinstating necessary steps) and verbose, potentially erroneous justification. A case study analyzing specific examples of "crammed" answers versus original CoT would be far more convincing than aggregate overlap scores.
- **Missing Ablations/Controls:** A critical control experiment is missing: **What is the performance with *no* CoT (direct answer generation) compared to the various deletion levels?** This would anchor the "cramming" behavior and show whether a heavily deleted CoT still provides any benefit over a zero-CoT baseline. Figure 2 compares High/Medium/Low reasoning but not a "No CoT" condition.

### Analysis and Discussion
- **Faithfulness Implications:** The discussion correctly highlights that CoT appears "informative and redundant." The suggestion that CoT should not be treated as a transparent explanation is an important, well-supported takeaway.
- **Superficial Analysis:** The discussion of "information overlap" (Section 4.2) is limited by the shallow metrics used. The interpretation of Fig. 7 is therefore overreaching; increased token overlap does not equate to faithful recovery of reasoning structure.
- **Practical Implications:** The suggestions about early stopping for efficiency are reasonable extrapolations from the data.
- **Limitations:** The listed limitations are appropriate but incomplete. They should explicitly mention: (1) the reliance on an LLM-as-Judge for scoring, (2) the simplistic nature of the overlap metrics for evaluating reasoning recovery, (3) the lack of a "No CoT" baseline, and (4) the unclear implementation of the "interception" mechanism.

### Writing & Clarity
The paper is generally well-written and logically structured. Some figure references are broken (e.g., "Figure C", "Figure 11 in §B"), which is likely a parser artifact but should be fixed. The description of the calibration study (Section 3.1) is slightly confusing.

### Overall Assessment
This paper presents a clever and conceptually important methodological framework (systematic deletion) to probe the often-assumed but rarely tested dependence of LLMs on their own Chain-of-Thought. The core empirical finding—that models maintain accuracy under substantial CoT deletion and exhibit compensatory "cramming"—is valuable and challenges common assumptions. However, for ICLR, where methodological rigor is paramount, the paper is undermined by significant weaknesses in its evaluation methodology: the use of an unverified LLM-as-Judge and superficial lexical metrics to assess reasoning recovery. The contribution stands as a compelling demonstration of a problem (CoT non-faithfulness) and a promising approach (deletion probing), but the analysis supporting the specific nature of "cramming" and "recovery" is not yet at the required level of rigor. Major revisions addressing the metric and analysis limitations are necessary for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates the faithfulness of chain-of-thought (CoT) reasoning in large language models applied to physics problem-solving. The authors introduce a systematic deletion framework where portions of the CoT scratchpad are removed mid-generation, and they measure the downstream impact on answer accuracy, length, and information recovery. Applied to three open-source models across three physics benchmarks, the study finds that models remain surprisingly robust to moderate deletions (40-60%) and exhibit "cramming" behavior—compensating by generating longer final answers that often reconstruct deleted content. The analysis reveals that while CoT traces boost performance, models do not faithfully depend on them, highlighting a gap between accuracy and reasoning fidelity.

### Strengths
1. **Novel and Well-Motivated Methodology:** The systematic deletion framework (end, random, and physics-aware) is a creative and straightforward probe for assessing reasoning dependence. It directly operationalizes the question of faithfulness in a way that is rarely explored in prior work, especially within the structured domain of physics.
2. **Rigorous and Comprehensive Empirical Evaluation:** The paper conducts extensive experiments across three diverse models (Phi-4, Qwen-A3B, Magistral) and three physics benchmarks of varying difficulty (UG Physics, PhysReason, PhyBench). The use of multiple deletion strategies and careful calibration (e.g., sample size analysis in Figure 8) strengthens the validity of the findings.
3. **Meaningful Analysis and Insightful Findings:** The identification of "cramming" behavior—where models increase final answer length to compensate for deleted reasoning—is a compelling empirical result. The information overlap analysis (Jaccard, Manhattan) provides quantitative evidence that models opportunistically reconstruct content rather than faithfully relying on the original CoT sequence.

### Weaknesses
1. **Limited Scope of Models and Domain:** While the choice of open-source models is justified for control, the study excludes closed-source state-of-the-art models (e.g., GPT-4o, Claude 3.5) which may exhibit different reasoning behaviors. Furthermore, the conclusions are drawn solely from physics; generalizability to other scientific or mathematical reasoning domains is asserted but not demonstrated.
2. **Reliance on an LLM-as-Judge for Scoring:** The core accuracy metric (Score) is evaluated using Claude-4 Sonnet. While this is a common practice, it introduces a potential confounder: the judge's own biases and errors. The paper does not include human evaluation or ground-truth verification to validate the judge's scores, which is particularly important for a claim about evaluation reliability.
3. **Lack of Mechanistic Explanation:** The study is correlational and descriptive. It identifies "cramming" and overlap patterns but does not probe *why* they occur (e.g., via attention head analysis, latent space interventions, or causal tracing). The discussion remains at the behavioral level, leaving the underlying mechanisms (e.g., template recall vs. adaptive computation) as speculation for future work.

### Novelty & Significance
**Novelty:** The core idea of deleting parts of a CoT trace to test dependence is intuitive but not trivial. While prior work has studied CoT faithfulness via perturbation (e.g., Turpin et al., 2023), applying a systematic deletion sweep in the structured context of physics problem-solving is a novel contribution. The "cramming" characterization and the domain-aware deletion strategy are also fresh insights.

**Significance:** The paper successfully argues that accuracy-alone evaluation is insufficient for AI-for-Science, where reasoning fidelity is paramount. The findings challenge the assumed role of CoT as a faithful scratchpad and have direct implications for how we benchmark, prompt, and potentially design reasoning models for scientific applications. This aligns well with ICLR's focus on rigorous evaluation and understanding model capabilities.

### Suggestions for Improvement
1. **Broaden the Empirical Base:** To strengthen claims about generality, include at least one closed-source API model (if feasible under review anonymity) and extend deletion experiments to a core mathematical reasoning benchmark (e.g., MATH or GSM8K). This would help disentangle domain-specific effects from general reasoning behaviors.
2. **Validate Evaluation Metrics:** Augment the LLM-as-judge scoring with human evaluation on a subset of responses, or at least report the inter-annotator agreement between the LLM judge and human experts for a sample. Alternatively, use exact match or programmatic verification for problems with deterministic answers to ground the "Score" metric.
3. **Deepen the Analysis:** A follow-up experiment or analysis section could more directly test the "cramming" hypothesis. For example, analyze whether the increase in answer length is primarily filler text or genuinely contains the critical deleted equations/facts. Additionally, a simple ablation varying the *position* of deletions (beginning vs. middle vs. end) could yield further insights into what parts of CoT are most "essential."
4. **Clarify Limitations and Future Work:** The limitations section should more explicitly discuss the potential impact of using an LLM judge. The future work section could be expanded to propose specific model architectures or training objectives (e.g., latent reasoning supervision) that might enforce greater CoT faithfulness, moving beyond diagnostic critique to constructive research directions.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No-CoT Baseline**: The paper lacks a critical baseline where models are prompted to answer *without* generating any CoT. Without this, claims about CoT being "bypassable" or "redundant" are not fully supported, as the true performance delta and necessity of CoT are not quantified.
2. **Content-Preserving Control**: The deletion experiments should be compared against a control where CoT tokens are *shuffled* or replaced with semantically neutral placeholders, rather than deleted. This would isolate whether performance drops are due to loss of *information* or merely disruption of *structure/fluency*.
3. **Comparison to State-of-the-Art Closed Models**: The study is limited to three open-source models. Including a top-performing closed-source model (e.g., GPT-4o, Claude 3.5) is essential to assess whether the observed "cramming" and faithfulness issues are general properties of reasoning LLMs or specific to the chosen models.
4. **Physics-Specific Answer Correctness Metric**: Relying solely on a general LLM (Claude) as a judge for "score" is insufficient for a scientific domain. The paper needs a rigorous, automated metric that checks unit consistency, equation correctness, and numerical accuracy against ground truth.
5. **Deletion at Different CoT Stages**: The deletion sweeps treat the CoT monolithically. An ablation deleting only *early* vs. *late* reasoning steps (e.g., concept setup vs. final calculation) is needed to test if models depend more on certain parts of the scratchpad.

### Deeper Analysis Needed (top 3-5 only)
1. **Causal Link Between Overlap and Score**: The paper reports overlap metrics and score drops separately but does not analyze if higher overlap *causes* better score recovery. A per-instance correlation analysis is required to substantiate the claim that "cramming" (longer answers with overlap) mitigates performance loss.
2. **Characterization of "Cramming" Content**: The analysis does not distinguish whether regenerated content in answers is *faithful* (correctly reconstructs deleted logic) or just *plausible* but erroneous physics. A manual error categorization (e.g., correct equation recovery vs. hallucinated substitutions) is needed to interpret "cramming."
3. **Model Consistency Analysis**: The results show variability across models and datasets. The paper needs an analysis (e.g., based on model size, training data, or reasoning fine-tuning) to hypothesize *why* some models cram more or show different deletion thresholds, rather than just presenting the variation.
4. **Internal Mechanism Probes**: The claims about internal processes are speculative. Simple analyses—like checking if attention to deleted tokens shifts to later layers or if answer token logits change—could provide more direct evidence of compensatory mechanisms beyond output length.

### Visualizations & Case Studies
1. **Side-by-Side Examples of Success/Failure**: Provide concrete examples comparing original CoT, the deleted version, and the final "crammed" answer for cases where score was maintained vs. dropped. This would visually demonstrate what faithful vs. unfaithful reconstruction looks like.
2. **Error Case Studies for Physics-Aware Deletion**: Show specific problems where deleting key equations or constants leads to score collapse despite long answers, illustrating the limits of "cramming" and the non-recoverability of critical facts.
3. **Overlap Heatmaps**: For a sample of problems, visualize token-level overlap between the deleted CoT span and the final answer to show if recovery is concentrated in specific sections (e.g., equations) or is diffuse.

### Obvious Next Steps
1. **Include a No-CoT Baseline**: This is a standard and necessary control for any study questioning CoT necessity and should have been in the main experiments.
2. **Perform the Correlation Analysis**: The relationship between answer length increase, information overlap, and score recovery should be quantitatively established, not just implied.
3. **Manual Audit of "Cramming"**: A small-scale qualitative analysis (50-100 instances) categorizing the correctness and faithfulness of regenerated content is essential to ground the interpretation of the overlap metrics.
4. **Test a Simpler Hypothesis**: Before concluding about general "faithfulness," test if the performance drop under deletion is simply proportional to the *amount of task-relevant information* removed, using a more precise measure than token count.

# Final Consolidated Review
## Summary
This paper introduces a systematic deletion framework to probe the faithfulness of chain-of-thought (CoT) reasoning in large language models applied to physics problem-solving. By intercepting and deleting portions of CoT traces mid-generation across three open-source models and three physics benchmarks, the authors find models maintain accuracy under substantial deletions (40-60%) and exhibit "cramming"—compensating with longer final answers that often superficially reconstruct deleted content. The work argues that accuracy-based evaluation is insufficient for scientific domains and highlights a need for methods that assess reasoning fidelity.

## Strengths
- **Novel and effective methodology:** The systematic deletion framework (end, random, physics-aware) is a simple, creative, and directly operationalized probe for testing reasoning dependence, a core but often assumed property in CoT research.
- **Compelling and robust empirical findings:** The consistent observation of accuracy plateaus followed by drops, coupled with increasing answer length ("cramming") across three models, three benchmarks, and multiple deletion strategies, provides strong evidence that CoT traces are informative yet partially bypassable.
- **Relevant domain and clear implications:** Using physics—a structured, high-stakes domain with equations and units—provides a stringent testbed. The paper successfully argues that evaluating reasoning faithfulness, not just final answer accuracy, is critical for advancing AI-for-Science.

## Weaknesses
- **Superficial quantification of reasoning recovery:** The analysis relies on lexical overlap metrics (Jaccard, Manhattan distance) which measure token co-occurrence but cannot assess the logical coherence, correctness, or ordering of recovered steps. This limits the strength of the claim that models "reconstruct" reasoning, as high overlap could indicate surface-level repetition of equations within an incorrect derivation. A more nuanced analysis (e.g., correctness of recovered elements) is needed to substantiate the nature of "cramming."
- **Potential confounder in the primary evaluation metric:** The core "Score" metric is assigned by a proprietary LLM (Claude-4 Sonnet) without validation against human evaluation or objective ground-truth checking (e.g., unit consistency, exact numeric match). For a paper critiquing evaluation reliability, this introduces an uncalibrated black-box component. While LLM-as-judge is common, some verification of its alignment with domain-correctness is warranted.
- **Limited exploration of why cramming occurs:** The study is descriptive and behavioral; it identifies the cramming pattern but does not investigate its mechanistic causes (e.g., is it driven by memorized templates, latent redundancy, or adaptive decoding strategies?). This limits the depth of the contribution regarding *how* models bypass CoT.

## Nice-to-Haves
- **Include a direct "No CoT" baseline comparison:** While the paper compares High/Medium/Low reasoning prompts, a condition with explicit instruction to answer directly without any scratchpad would more cleanly anchor the necessity and benefit of CoT versus its deleted versions.
- **Deepen the analysis of cramming content:** A qualitative analysis or finer-grained categorization (e.g., separating correct equation recovery from hallucinated substitutions) on a subset of instances would ground the interpretation of the overlap metrics and better characterize "faithful" versus "opportunistic" reconstruction.
- **Expand model and domain scope:** Testing one closed-source state-of-the-art model (if feasible) and extending deletion probes to a core mathematical reasoning benchmark (e.g., MATH) would help assess the generality of the findings beyond the chosen open-source models and physics domain.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness: "Lack of a 'No CoT' baseline"** – The paper includes a "Low Reasoning" prompt condition which explicitly asks the model to "minimize reasoning" and "provide a quick answer with only minimal or implicit thought steps." This serves as a functional no-CoT baseline.
- **Weakness: "Unclear implementation of CoT interception"** – The paper states it intercepts the CoT scratchpad "prior to decoding," which is a standard and reproducible concept for open-source models (editing the context before continuing generation). Demanding further low-level implementation details is not required for reproducibility at this level.
- **Weakness: "Physics-aware deletion uses an LLM, creating bias"** – Using an LLM for annotation is a pragmatic choice; a rule-based approach may be less adaptable. This does not invalidate the strategy as a meaningful probe.
- **Strength: "The paper is well-written" / "The experiments are extensive"** – These are generic and apply to many papers; strengths should be specific to this work's contributions.
- **Suggestion: "Perform internal mechanism probes (e.g., attention analysis)"** – This demands methodological practices (e.g., mechanistic interpretability) that are outside the stated scope of this empirical evaluation paper and are not standard for this type of contribution.

## Novel Insights
The paper provides a novel operationalization of the CoT faithfulness question through systematic deletion sweeps, revealing the counterintuitive "cramming" behavior where models compensate for lost reasoning by elongating final answers. This demonstrates that CoT traces function as both informative scaffolding and redundant exposition, challenging the assumption that generated step-by-step reasoning is a faithful account of the model's internal computation process, especially in structured scientific domains.

## Suggestions
- Enhance the analysis of information recovery by moving beyond bag-of-words metrics. For a subset of problems, manually categorize whether regenerated content in the final answer correctly and logically reinstates the deleted steps or merely produces plausible but erroneous physics.
- In the limitations section, explicitly discuss the potential impact of using an LLM-as-judge without validation and the shallow nature of the overlap metrics for assessing reasoning fidelity.
- Consider a simple correlation analysis to test if the degree of answer length increase or lexical overlap is associated with score preservation under deletion, which would strengthen the causal link in the "cramming" narrative.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
