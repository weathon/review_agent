---
job_id: c253cdbc-5805-4a67-b220-6d97d4d8d179
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: sh1hWO9RHo.pdf
paper: What Is Your Agent’s GPA? A Framework for Evaluating Agent Goal-Plan-Action Alignment
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.  
The submission is a full-length paper with abstract, main sections, figures, tables, and appendices.

## Topic Compatibility
Pass ✅.  
The work is about evaluation frameworks and benchmarks for LLM-based agents, fitting well within ICLR topics on evaluation of LLMs, agentic systems, and datasets/benchmarks.

## Minimum Quality
Pass ✅.  
All required scientific sections are present (Abstract, Introduction, Related Work, Method/Framework, Experiments, Results/Analysis, Conclusion). The work is written in English and appears technically competent with substantial experiments on public and internal benchmarks. No obvious fatal methodological errors or test leakage are evident from the description.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The paper contains many prompts for LLM judges in the appendix, but they are clearly part of the methodology, not attempts to influence the review. I did not detect hidden instructions targeting reviewers or other manipulative content.

---

# Expected Review Outcome:

## Summary

The paper proposes the Agent GPA (Goal‑Plan‑Action) framework, which decomposes agent evaluation into six dimensions: Goal Fulfillment (GF), Logical Consistency (LC), Execution Efficiency (EE), Plan Quality (PQ), Plan Adherence (PA), Tool Selection (TS), and Tool Calling (TC). Each dimension is instantiated as a dedicated LLM-as-a-judge with carefully crafted prompts (and optionally GEPA-optimized prompts) that operate reference‑free on execution traces.

The authors evaluate these judges on TRAIL/GAIA (with TRAIL annotations as ground truth), a small internal dataset from a production data agent, and a preliminary TRAIL/SWE‑bench case study. They show high coverage and localization of TRAIL-annotated errors, substantially outperforming the TRAIL baseline judge, strong agreement between human raters and LLM judges, and reasonable inter‑run stability measured by Krippendorff’s α and a “Semantic Consistency Index”.

## Strengths

1. **Conceptual decomposition of agent evaluation is clear and intuitive.**  
   The paper argues that many existing agent evaluations conflate different failure modes and proposes a decomposition aligned with the agent operational loop (Goal → Plan → Act). Figure 1’s Venn diagram and metric list (Goal, Plan, Action with overlapping regions labeled 1–5 and the stacked list of metrics) provides a concise mental model for how LC, EE, PA, PQ, TS, TC, and GF relate to each other. This is useful both for research and for practitioners who want interpretable diagnostics rather than a single scalar score.

2. **Strong empirical improvement over a competitive baseline on TRAIL/GAIA.**  
   Using TRAIL’s own LLM judge as a baseline, the GPA judges capture 95% of annotated errors on the GAIA test set versus ~54% for the baseline (Table 2). Similarly, localization coverage improves from 49% to 86% (Table 5). These are sizeable and practically meaningful improvements, especially given the difficulty of TRAIL tasks. Table 3 and Table 6 further show that several judges (notably EE and TC) have balanced precision–recall profiles with F1-scores around 0.84–0.92 on the test set, suggesting they are not merely gaining recall by flagging everything.

3. **Fine‑grained error localization and specialization of judges.**  
   Beyond detecting that an error occurred, the judges attempt to localize it to specific spans and dimensions, enabling more targeted debugging. The paper does a good job interpreting per-judge behavior across impact levels: e.g., TS acts as a high‑recall, low‑precision “liberal” detector of tool selection errors, whereas TC is a high‑precision, lower‑recall “conservative” detector (Section 4.1.3, Tables 3, 6, 13–18). This specialization analysis is a real contribution for anyone designing agent evaluation pipelines.

4. **Careful analysis of judge reliability across runs.**  
   Section 4.1.4 and Table 7, together with Figure 5 and Figure 6, provide a rare and welcome look at the stability of LLM-as-judge scores. For instance, Krippendorff’s α is ≥ 0.8 for EE, PA, TS, and TC, and average per-trace standard deviation across 5 runs is small (e.g., 0.053 for EE, 0.059 for TS). The Semantic Consistency Index histograms in Figure 2 (panels (a)–(c)) show that rationales for EE are more semantically stable than for PQ and LC, which matches the intuition that synthesis-heavy metrics are noisier. This reliability analysis goes beyond most LLM-judge papers, which usually report single-run numbers.

5. **Human–LLM agreement is quantified reasonably well.**  
   Table 4 reports off-by-one accuracy, bucketed 3-point accuracy, and correlation between GPA judges and human scores across dimensions, with many metrics achieving high agreement (e.g., PA correlation 0.917, TS correlation 0.895 on the test set; EE off‑by‑one accuracy 0.949). Appendices E and F further provide Krippendorff’s α and Cohen’s κ between humans and LLMs, and cross‑metric orthogonality analysis. This gives some confidence that the judges are not merely idiosyncratic heuristics.

6. **Cross‑metric orthogonality analysis is insightful.**  
   The authors binarize metric activations and compute Krippendorff’s α, Cohen’s κ, Jaccard similarity, and phi correlations across metrics (Tables 22–25). The relatively low agreement across LC, EE, PA, PQ, TS, and TC supports the claim that these judges capture complementary failure modes rather than redundantly firing on the same traces. The strong co‑occurrence between EE and TC is a nice empirical confirmation that execution inefficiency is often associated with tool-calling problems.

7. **Preliminary generalization and automation via GEPA.**  
   The GEPA experiments on TRAIL/GAIA and TRAIL/SWE‑bench (Section 4.1.5, Tables 8 and 9) show that reflective prompt optimization can further improve recall, especially for LC (e.g., from 80.7% to 87.9% recall on GAIA test, and from 28.8% to 75.3% on SWE‑bench). This suggests the framework can be bootstrapped to new domains without extensive manual prompt engineering, which is important for scalability.

8. **Application to an internal production data agent.**  
   The internal ANON‑Data‑Agent study, although small (17 traces), shows that LC and EE judges can be applied to real-world SQL and retrieval workflows. Table 10 and Table 19 report ~82% agreement with humans on a 3-point scale and reasonable NMAE, and the authors describe that this enabled “targeted improvements” in the system architecture. This is useful evidence that the framework is not just benchmark‑driven.

## Weaknesses

1. **Despite the name, Goal Fulfillment is not actually evaluated.**  
   While GF is described as a primary metric in Section 3 (and appears first in Figure 1 and the metric list), the experimental sections (4.1, 4.2) and all core tables (Tables 1–8, 11–21) report results only for LC, EE, PA, PQ, TS, and TC. GF is absent from the quantitative evaluation and from the reliability and agreement analyses. This is a substantial gap since the framework is marketed as “Goal‑Plan‑Action,” yet the “Goal” metric is not empirically instantiated or validated. At a minimum, the paper should either (i) include GF in all key experiments, with error coverage and human‑LLM agreement analogous to the other metrics, or (ii) clearly mark GF as future work and tone down claims about a full GPA framework.

2. **The framework is largely prompt‑defined and informal rather than methodologically precise.**  
   The “method” for each metric is encoded mainly as long, English prompts (Appendix B and C), sometimes several pages per judge. There is no more abstract formalization of what counts as a LC/EE/PA/etc. error beyond descriptive prose. For instance, Section 4.1.2 says scores are on a 0–3 scale, but there is no precise algorithmic mapping from trace features to that score, only heuristic instructions to the LLM. This makes reproducibility across models and implementations fragile: small changes in LLM architectures, pretraining, or sampling could produce qualitatively different behavior. A more rigorous design would define, say, LC as checking explicit logical predicates over the trace or quantify error types with a formal taxonomy, with prompts treated as one implementation, not the definition.

3. **Heavy dependence on a single proprietary LLM as judge and meta‑judge.**  
   Most experiments (TRAIL/GAIA and SWE‑bench, GEPA, consistency studies) use Claude‑4‑Sonnet or Claude‑Sonnet‑4.5 as the only judge. The meta‑judge used to evaluate GEPA‑optimized prompts is also an LLM that is assumed to be “strongly aligned” but not empirically validated beyond a brief comparison in Appendix G. This raises concerns about (i) model‑specific biases, and (ii) potential circularity when the same family of models is both generating and judging traces (especially for the internal agent). Table 19 partially addresses this by comparing different models on the internal dataset, but that experiment is tiny and limited to LC/EE, not the full GPA suite. A more convincing evaluation would systematically compare multiple families of judges (e.g., Claude vs GPT vs open‑source) on TRAIL/GAIA with human ground truth, and analyze failure modes where they disagree.

4. **Mapping TRAIL’s error taxonomy to GPA dimensions is subjective and under‑specified.**  
   Section 4.1.2 states that two human annotators mapped each of TRAIL/GAIA’s 570 errors to one or more GPA dimensions, with a third annotator “cross‑checking” the mappings. However, the paper does not report inter‑annotator agreement, criteria for mapping one TRAIL error type into multiple GPA labels, or any examples that show borderline cases. Since the central quantitative claim in Section 4.1.3 is that “GPA captures all 570 errors,” but that coverage is defined with respect to this mapping, the mapping procedure is crucial. Without a transparent and reproducible mapping protocol (possibly with a small illustrative table of examples), the reported coverage in Table 1 can be questioned as partially definitional.

5. **Some metrics are empirically weak or unstable yet still treated as first‑class components.**  
   The Plan Quality (PQ) judge, and to a lesser extent Plan Adherence (PA), consistently underperform and are noisy. For example, Table 3 shows PQ’s F1 on the GAIA test set is only 0.49 with recall 0.71 but precision 0.37, and Table 6 shows F1 0.43 for localization. Table 7 reports Krippendorff’s α of only 0.628 and an average per‑trace standard deviation of 0.171 for PQ, clearly worse than other metrics. Figure 6 highlights PQ as having the largest per‑trace variance and widest confidence intervals. Appendix A (Table 11 and 12) shows especially poor coverage and localization for low‑impact errors. The paper acknowledges some of this but still markets the full six‑judge suite as a coherent system. For an ICLR‑level contribution, I would expect deeper analysis of why PQ is unstable (e.g., failure cases, prompt ablations, dependency on explicit [PLAN] segments) and stronger justification for including it, or else a more focused core set of reliable metrics.

6. **GEPA and meta‑judge evaluations introduce another layer of circularity and lack transparency.**  
   In Section 4.1.5, GEPA-optimized prompts are evaluated using a “meta‑judge” that grades the outputs of the GPA judges against TRAIL errors. This meta‑judge is itself an LLM, not a human or deterministic metric, and its alignment with humans is only briefly justified in Appendix G via a single number (159/198 errors “caught”). However, the main comparisons in Table 8 and Table 9 use this meta‑judge rather than human verification, yet these are then used to argue that GEPA “matches or outperforms manually engineered prompts”. Since both the prompts and the evaluator are learned from and evaluated on related data and models, it is unclear how robust these results are. At minimum, some spot‑checked human evaluation on the GEPA-optimized prompts and error types would strengthen the claim.

7. **Generalization claims are based on small or limited datasets.**  
   The SWE‑bench experiments (Table 9) use only 16 test traces and consider just LC, EE, and TC (no PQ, PA, or TS because the agent does not plan explicitly). The internal ANON‑Data‑Agent dataset has only 17 traces. While it is reasonable to start small, the paper’s broader claim that “the GPA framework generalizes effectively to unseen agentic tasks (e.g., coding)” (Section 4.1.5) is not strongly supported by such small‑scale studies. Confidence intervals or per‑task breakdowns would help here, and at least one more independent agent/domain beyond web research, SWE‑bench, and the proprietary data agent would make the generalization story more convincing.

8. **Mathematical and procedural details of scoring are sometimes inconsistent or underspecified.**  
   A few examples:
   - Section 4.1.4 states: “For each trace and metric, we collect scores in \([0,1]\) across 5 independent runs...” but earlier in Section 4.1.2 the judges are described as producing integer scores from 0 to 3. It is unclear whether scores were normalized as \(s' = s/3 \in [0,1]\), or whether some other mapping was applied. This matters for interpreting Krippendorff’s α and standard deviations in Table 7 and Figure 6.
   - In Appendix F, metric scores are binarized for cross‑metric agreement analysis, but the thresholding rule is not explicitly defined (e.g., is any score <3 treated as 0? or >0 treated as 1?). The paper mentions focusing on whether a metric “fires” but does not specify the mapping \(f: \{0,1,2,3\} \to \{0,1\}\). This ambiguity affects the φ correlations and Jaccard values in Tables 22–25.
   - The procedure for detecting and localizing an error from a 0–3 score is unclear: is any non‑maximal score considered an “error,” or is there a separate binary label predicted by the judge? The per‑judge F1 in Table 3 treats error detection as a classification problem, but the predictor is never formalized.

   These are not fatal mathematical errors, but they do make the methodology harder to reproduce and to interpret precisely.

9. **Positioning in the fast‑growing agent‑evaluation literature is incomplete.**  
   While the paper cites many important recent works (TRAIL, AgentRewardBench, multi‑agent failure taxonomies, industrial evaluators, planning benchmarks), several highly relevant and recent works on goal-directedness and agent evaluation frameworks are not discussed (see “Potentially Missing Related Work” below). This weakens the claim that the proposed framework is meaningfully differentiated from other emerging taxonomies of agent behavior and evaluation.

10. **Limited discussion of limitations in LLM‑as‑a‑judge evaluation longer or multi‑agent traces.**  
    The paper acknowledges in Related Work that LLM judges can overestimate success on long traces, but the main experiments are on preprocessed TRAIL traces (Section 4.1.2), where messages are filtered heavily to fit context windows. It would be helpful to see a more explicit analysis of how context length and trace preprocessing choices affect detection/localization performance, and whether GPA judges degrade on fully realistic long‑horizon traces.

Overall, the work is promising and empirically solid in some dimensions (LC, EE, TS, TC on GAIA) but currently feels somewhat over‑claimed relative to the actual evaluated metrics (no GF) and the robustness of weaker judges (PQ, PA), and it relies heavily on a single proprietary model and an LLM meta‑judge.

## Potentially Missing Related Work

These works are not cited in the paper but are directly relevant and should be discussed and positioned against the proposed GPA framework:

1. **Arghal, R., Chen, F., Dalton, N. (2026). “A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents.”**  
   This paper explicitly evaluates goal‑directedness, which is closely related to the proposed Goal Fulfillment and Logical Consistency dimensions. It should be discussed in Section 2 (Related Work) as a complementary or alternative framework for assessing whether agents are genuinely pursuing goals versus superficially aligned behavior. A comparison of what GPA’s GF/LC capture versus their behavioral tests would sharpen the notion of “goal alignment.”

2. **Baby, A. (2026). “Agent Evaluation Frameworks.”**  
   This work surveys or proposes generic frameworks for evaluating agents on correctness, safety, efficiency, and robustness. Many of these axes overlap with GPA’s LC, EE, and TS/TC. It would be appropriate to cite and compare in Section 2 and possibly in the conclusion when claiming interpretability and coverage of diverse failure modes.

3. **Evalvista (2026). “Agent Evaluation Framework: 5 Approaches Compared.”**  
   This paper compares five agent evaluation approaches, including trajectory‑based and outcome‑based methods. GPA’s multi‑judge architecture could be contrasted with these approaches to clarify how it differs in terms of reliance on references, localization capability, and dimensionality of feedback. Section 2 and the discussion of TRAIL/GAIA results (Section 4.1.3) would be natural places to add this.

4. **Stanford CS329T (2025). “Trustworthy Machine Learning: Building and Evaluating Agentic Systems.”**  
   Course materials specifically discuss evaluating agentic systems and even mention “agent GPA” as an evaluation concept. This is directly relevant background and would help contextualize how the presented framework builds on or elaborates earlier notions used in education or practice. It should be cited in Section 1 or 2 and discussed when motivating the need for dimension‑wise evaluation.

5. **IdeaPlan (2025). “Specifying AI Agent Behaviors.”**  
   This guide focuses on specifying agent goals and behaviors, especially around planning and execution, which is conceptually close to GPA’s Plan Quality and Plan Adherence. Bringing this into Section 2 would help situate GPA relative to broader work on explicit behavior specification and might inspire a clearer formalization of what constitutes a “good” plan or “adherence” beyond LLM prompts.

## Questions

1. **Goal Fulfillment metric: implementation and evaluation.**  
   - How exactly is GF implemented as a judge in your system (prompt structure, rubric)?  
   - Why is GF absent from all reported quantitative results (Tables 1–8, 11–21)? Is this because GAIA tasks have clear reference answers, making GF redundant with TRAIL’s annotations, or was the GF judge not ready?  
   - Could you provide GF error coverage, localization, and human‑LLM agreement analogous to LC/EE on GAIA and, if possible, on the internal data agent?

2. **Score normalization and binarization details.**  
   - In Section 4.1.4, what exact mapping did you apply from 0–3 scores to “scores in \([0,1]\)” for computing Krippendorff’s α and per‑trace standard deviation? Was it simply \(s' = s/3\) or something else?  
   - For Appendix F, how do you map multi‑level scores into binary activations? Is any non‑maximum value treated as an “error,” or do you use a threshold such as \(s \leq 1\)?  
   - How does this choice affect the cross‑metric agreement numbers in Tables 22–25?

3. **Error detection vs severity scoring: what is the decision rule?**  
   - For Tables 3 and 6 (precision/recall/F1 for caught/localized errors), what outputs from a judge constitute a positive detection for a specific error? Is it any non‑zero score for the whole trace, or do judges also explicitly output span‑level labels?  
   - If a judge assigns a lower but nonzero global score without localizing an error span, is this counted as detection but not localization, or as a miss? Clarifying the mapping from free‑form rationales to binary detection/localization would help.

4. **Human–LLM mapping of TRAIL errors to GPA dimensions.**  
   - Could you describe the annotation protocol in more detail? Did annotators have a written rubric linking TRAIL’s taxonomy (e.g., hallucination, tool misuse) to specific GPA dimensions?  
   - What was inter‑annotator agreement on these mappings before reconciliation?  
   - How often were errors mapped to multiple dimensions (e.g., both LC and TS), and how does that affect coverage statistics like Table 1?

5. **Plan Quality failure modes.**  
   - PQ appears to be the least reliable judge (low F1, lowest α, largest standard deviation in Figure 6). Can you provide qualitative examples where PQ disagrees with humans?  
   - Did you try simpler rubrics for PQ (e.g., binary “clearly invalid plan vs reasonable plan”) to see if reliability improves?  
   - Given current unreliability, why not present PQ as exploratory and remove it from the “core” GPA metrics in the main claims?

6. **Meta‑judge calibration and validity.**  
   - How is the meta‑judge prompt structured, and how does it decide whether a given GPA judge output “caught” a TRAIL error?  
   - Besides the 159/198 vs 177/198 comparison in Appendix G, did you perform any systematic human evaluation of the meta‑judge’s correctness?  
   - Could you provide per‑error‑type analysis (e.g., hallucination vs tool misuse) of where the meta‑judge deviates most from human assessments?

7. **Impact of trace preprocessing.**  
   - In Section 4.1.2 you mention stripping duplicated messages and only retaining certain spans to fit context limits. How much does this reduce the average trace length, and could some TRAIL‑annotated errors be “hidden” by preprocessing?  
   - Have you tested GPA judges on raw or less aggressively preprocessed traces to quantify performance degradation?

Clarifications and additional experiments along these lines, especially around GF, scoring/binarization, and PQ/PA reliability, could significantly strengthen the paper.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The empirical work on TRAIL/GAIA is careful and shows strong improvements over a baseline, with reasonable statistical analysis (precision/recall/F1, α, κ, etc.). However, key parts of the framework (Goal Fulfillment, Plan Quality) are not robustly evaluated, scoring procedures are sometimes underspecified or inconsistent, and there is heavy reliance on a single proprietary model and an LLM meta‑judge, which limits the methodological solidity.

## Presentation Rating

3: good.  
The paper is generally well written and organized, with helpful figures (e.g., Figure 1 for the metric structure, Figures 5–6 for inter‑run variability) and comprehensive tables. Some methodological details (score normalization, binary thresholds, error‑dimension mapping) are under‑explained, and the sheer length of prompt text in the appendix can obscure the core method, but overall clarity is above average.

## Contribution Rating

2: fair.  
The idea of decomposing agent evaluation into goal/plan/action-aligned judges and the empirical demonstration of improved error coverage and localization over TRAIL are valuable. However, the conceptual novelty is moderate given concurrent work on agent evaluation taxonomies, and the most ambitious parts of the framework (full GPA including GF, reliable PQ) are not fully realized in the experiments. The contribution is interesting and practically useful, but falls somewhat short of a strong ICLR contribution in its current form.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper offers a practically useful, empirically supported decomposition of agent evaluation into multiple specialized LLM judges and demonstrates clear gains over a strong baseline on a public benchmark. However, the absence of quantitative results for the Goal Fulfillment component, the empirical weakness of some judges (especially Plan Quality), the reliance on subjective mappings and an LLM meta‑judge, and several under‑specified methodological details prevent me from recommending acceptance at ICLR’s main track this year. With a more complete and rigorously specified implementation of the full GPA framework, plus broader and more diverse evaluation, this line of work could become a strong contribution.

## Reviewer Confidence

4: confident.  
I am familiar with LLM‑as‑a‑judge evaluation, agent benchmarks such as GAIA, SWE‑bench, and TRAIL, and have carefully read the paper, tables, and appendices. Some implementation details (e.g., exact prompts, meta‑judge behavior) cannot be fully verified from the text alone, but the main methodological and empirical claims are clear enough for a confident assessment.