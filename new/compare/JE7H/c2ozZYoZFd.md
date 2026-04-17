---
job_id: ca0ce356-7700-4193-a5cd-675e5a8eaf24
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: c2ozZYoZFd.pdf
paper: A Min-P Blueprint for More Rigorous Science in Empirical Machine Learning Research
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅. The paper addresses methodological rigor, evaluation practices, hyperparameter tuning, and reproducibility in empirical ML and LLM evaluation, which clearly falls within ICLR’s scope (evaluation, benchmarks, reproducibility, and empirical methodology).

## Minimum Quality
Pass ✅. The paper has a clear abstract, introduction, several substantial analysis sections (human evals, benchmarks, LLM-as-a-judge, community adoption), and a discussion / limitations section. The work is written in English, is technically coherent, and provides non‑trivial analyses and results. While framed as a case study / blueprint rather than a new algorithm, it offers concrete empirical and methodological content.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅. I did not detect any hidden prompts, manipulative instructions targeting automated reviewers, or other integrity violations in the main paper content.

---

# Expected Review Outcome:

## Summary

The paper presents an in‑depth re‑analysis of the ICLR 2025 Oral paper “Turning Up the Heat: Min‑P Sampling for Creative and Coherent LLM Outputs” (Nguyen et al., 2024).  
Using the original authors’ released data and code, the paper re‑examines four main lines of evidence for min‑p sampling (human evaluation, NLP benchmarks, LLM‑as‑a‑judge, and community adoption) and finds that under more careful analysis min‑p does not consistently outperform standard samplers such as basic, top‑k, or top‑p.  
From this case study, the authors articulate a set of methodological lessons aimed at improving rigor in empirical ML research, especially around fair hyperparameter comparison, statistical testing, data transparency, and reporting practices.

## Strengths

1. **Careful, data‑driven re‑analysis of a high‑profile result.**  
   The paper goes beyond opinion and reproduces and extends the original experiments using Nguyen et al.’s data and code. The re‑analysis of human evaluation scores in **Figure 1** and hypothesis tests in **Table 1** is systematic: the authors include the previously omitted “basic” sampling condition, compute paired one‑sided t‑tests for each metric/temperature/sampler pair, and apply Bonferroni correction and an intersection‑union test. This is a strong example of how to interrogate published claims using standard statistical machinery rather than rhetoric.

2. **Use of uncertainty visualization and re‑interpretation of human evals.**  
   **Figure 1** re‑plots human scores with 95% confidence intervals, immediately showing that differences between samplers are small and largely overlapping across temperatures and metrics. **Figure 3** similarly visualizes the follow‑up human study, explicitly mapping quality–diversity trade‑offs; joint scatter plots make it visually clear that min‑p does not occupy a Pareto‑superior region. This concrete visualization work directly supports the argument that original “consistent superiority” claims were overstated.

3. **Re‑evaluation of qualitative feedback.**  
   The manual annotation and aggregation of free‑form worker comments in **Figure 2** is a nice touch: converting qualitative feedback into a simple count of preferred samplers exposes that more participants explicitly preferred basic sampling than min‑p, which contradicts the original narrative that human subjects “frequently” favored min‑p. The paper supports this with verbatim quotes in Appendix B, avoiding cherry‑picked excerpts from only favorable responses.

4. **Large‑scale, controlled hyperparameter sweeps on GSM8K.**  
   The GSM8K Chain‑of‑Thought sweeps are substantial in scope (∼6000 A100‑hours) and cover 9 models, two training stages (base/instruct), 4 samplers, 31 temperatures, 6 sampler‑specific hyperparameters, and 3 sampling seeds. The “Best‑of‑N” methodology is well explained:  
   - For each sampler, subsample N hyperparameters and take the best exact‑match score.  
   - Compare curves across samplers as N grows.  
   **Figure 4** then shows per‑model curves where min‑p’s best‑of‑N accuracy is extremely similar to top‑p/top‑k/basic once hyperparameter search volume is controlled, and **Figure 5** plots min‑p’s margin over the best non‑min‑p sampler as N increases, revealing it is usually near zero or negative. This is a concrete and reusable methodological contribution for fair comparison of heavily tuned methods.

5. **Critical assessment of LLM‑as‑a‑judge methodology and hyperparameter fairness.**  
   The analysis in Section 4 identifies multiple under‑specified aspects in the original LLM‑as‑a‑judge evaluations and highlights the non‑transitivity confound introduced by comparing everything only to basic(τ=1). **Figure 6 (left)** quantifies that min‑p received roughly 2× the hyperparameter configurations of top‑p and 10× that of basic, whereas **Figure 6 (right)** shows win rates with 95% confidence intervals where min‑p usually fails to outperform alternatives. This is a concrete and important point about how uneven hyperparameter tuning can create an illusion of superiority.

6. **Fact‑checking of community adoption claims and explicit corrections.**  
   The authors scrutinize the original claims of “54k GitHub repositories” and “1.1M total stars”, attempt to replicate them through an analysis of major LM repositories, find clear inconsistencies, and document that the numbers were later retracted. This is an important illustration of how unvetted “community adoption” metrics can mislead reviewers and ACs and why such claims must be reproducible and verifiable.

7. **Clear articulation of methodological lessons and reviewer guidance.**  
   Section 6 distills concrete guidelines: controlling for hyperparameter volume, correcting for multiple comparisons, insisting on full data transparency, checking qualitative summaries against raw responses, requiring methodological clarity sufficient for reproduction, and avoiding selective reporting. These points are directly grounded in the case study and offer reviewers a practical checklist rather than purely abstract principles.

## Weaknesses

1. **The “blueprint” remains high‑level and under‑developed relative to the case study.**  
   Despite the title promising “A min‑p blueprint for more rigorous science,” almost all of the technical depth is focused on the specific re‑analysis of Nguyen et al. (2024). Section 6’s lessons are important but brief and largely rephrase well‑known concerns (hyperparameter fairness, multiple‑comparison correction, data transparency, selective reporting). There is no structured framework, taxonomy, or formal protocol that could genuinely guide future empirical ML studies beyond this one example. For instance, the Best‑of‑N method from Section 3.1 is not generalized into a clear procedural recommendation (e.g., how many N’s should be tested, how to report sensitivity plots like **Figure 4**/**5** in future work, or how to standardize hyperparameter budgets across papers).

2. **Limited breadth and generalization beyond a single case.**  
   The argument for broader significance is that the errors found in Nguyen et al. (2024) are “common in empirical machine learning research,” but the paper does not substantiate this claim with systematic evidence. No survey of a sample of recent NLP or LLM papers is provided, no statistics on how many use uncorrected multiple comparisons or uneven hyperparameter budgets, and no meta‑analysis of evaluation practices. As a result, the work risks being read as a very detailed comment on one paper rather than a broadly applicable blueprint. Extending the analysis even to a small random sample of contemporary LLM sampling or evaluation papers would dramatically strengthen the claim that the issues identified are systemic.

3. **Some analyses rely on external artifacts without fully self‑contained exposition.**  
   Several critical claims hinge on data or communications that are not presented in detail in the main paper. For example:  
   - Section 2.1 mentions that one‑third of human eval scores (the basic sampler) were omitted and that the authors “publicly confirmed” this, but the main paper does not show a concise summary table of the original vs. extended setup.  
   - Section 4.3 refers to a Telegram link where differing win rates for min‑p and top‑p are visible and states that the higher min‑p score and lower top‑p score were selectively reported, but the actual numerical table or clear side‑by‑side comparison is not included in the paper.  
   - Section 5’s GitHub star analysis is described only at a high level; there is no explicit methodology (e.g., exact search queries, date of collection, criteria for counting “using min‑p”), nor a table similar to **Table 1** demonstrating mismatches.  
   These omissions weaken the reproducibility and persuasiveness of the critique, especially since the paper is advocating for transparency.

4. **Statistical treatment is focused but somewhat narrow and could be more rigorous itself.**  
   In Section 2.2, the authors correctly perform 12 one‑sided paired t‑tests and apply a Bonferroni correction. However, the paper does not discuss statistical power, effect sizes, or alternative multiple‑comparison procedures that might be more appropriate (e.g., Holm–Bonferroni, Benjamini–Hochberg FDR). The decision to frame the central claim as requiring an “Intersection‑Union Test” across all 12 conditions is reasonable given the wording “consistently across all settings”, but the paper does not fully formalize this IUT or specify assumptions (e.g., independence of test statistics) beyond saying “largest p‑value is 0.378”. A brief explanation of the IUT test statistic and why simply checking the maximum p‑value suffices here would align better with the paper’s own rigor standards. Similarly, there is no discussion of the assumptions behind paired t‑tests on Likert‑scale ratings (approximate normality, interval‑scale interpretation), which is a common contentious point in human evaluation.

5. **The treatment of the second human study (Appendix C.2 of Nguyen et al.) is somewhat cursory.**  
   While **Figure 3** compellingly visualizes that min‑p does not dominate in quality–diversity space, the analysis here stops at visual inspection and one suspected data error (the 7.80 vs. 5.80 discrepancy in Table 15). There are no formal statistical tests analogous to **Table 1** for the new study, no consideration of whether different implementations (temperature before/after truncation, new participant pool, new story prompts) might reasonably change the evaluation question, and no discussion of possible improvements (e.g., pre‑registered analysis plans, power calculation given the larger number of conditions). The paper criticizes the original for under‑specified statistical practice, but the re‑analysis of the new experiment does not itself set a stronger bar beyond plotting means and CIs.

6. **Benchmark scope is limited to GSM8K CoT and excludes other tasks from the original paper.**  
   Section 3.1 focuses entirely on GSM8K CoT, noting compute constraints and mentioning GPQA only in passing. While the sweeps on GSM8K are quite extensive, the original paper’s claim was “superior performance across benchmarks and temperatures.” The critique would be more convincing if at least a subset of GPQA settings, or another representative benchmark with different characteristics, were included in the Best‑of‑N analysis. As it stands, the conclusion “min‑p does not outperform other samplers when controlling for hyperparameter volume” is strictly valid only for GSM8K, though the text occasionally generalizes beyond that.

7. **The blueprint does not materially engage with or extend the broader reproducibility / rigor literature.**  
   The paper situates itself among several meta‑science references (e.g., Pineau et al. 2017; Sinha et al. 2023; Ivanova et al. 2025; Schaeffer et al. 2025b), but it omits or only tangentially engages with foundational works on reproducibility and evaluation design in ML (see “Potentially Missing Related Work” below). There is little explicit discussion of how the proposed practices relate to, differ from, or improve upon, for instance, existing reproducibility checklists or frameworks for quantifying replicability. This lack of deeper positioning undermines the claim that the paper offers something substantially new to that subfield rather than a well executed single‑case critique.

8. **Tone and framing risk drifting into a targeted post‑hoc rebuttal rather than a general scientific contribution.**  
   The paper focuses intensely on a single contemporary paper, repeatedly naming Nguyen et al. (2024) and describing private or semi‑private interactions (“we publicly confirmed with the authors”, “after we showed these results to the authors”). While the analysis is factual and generally respectful, the narrative sometimes reads closer to a blog‑style post‑mortem than a neutral scientific study. For example, Section 5 notes that “3 of 4 ICLR 2025 reviewers and the Area Chair identified these retracted community adoption numbers as the main justification for their strong endorsement”; this ventures into commentary on the review process without more systematic evidence. A more depersonalized framing (e.g., anonymizing the case during analysis, or supplementing it with additional anonymized examples) would strengthen the paper’s standing as a general blueprint rather than a public dissection of a single group’s mistakes.

9. **Limited mathematical or formal content restricts technical depth.**  
   Apart from t‑test formulas and Bonferroni/IUT concepts (described verbally) and the Best‑of‑N hyperparameter sampling procedure, there is little explicit mathematical development. For instance, the Best‑of‑N analysis could be formalized as an estimator of the maximal performance under a given hyperparameter budget, with consideration of variance across random subsets and seeds, or even asymptotic behavior as N grows. Right now, the method is clear at an intuitive/algorithmic level but lacks a precise statistical characterization (e.g., formal notation such as letting each hyperparameter configuration h define a performance random variable \(X_h\), defining \(Z_N = \max_{h \in S_N} X_h\) for random subset \(S_N\), and discussing properties of \(\mathbb{E}[Z_N]\)). For a venue like ICLR that values methodological sharpness, the paper would benefit from at least some such formalization.

10. **Some figures are under‑annotated for readers not already familiar with the original paper.**  
    For example, **Figures 4–5, 7–8** show grids of 16 subplots with curves for various models, but axes and legends are described only briefly: the y‑axis is EM score (or margin difference), and the x‑axis is “Number of Hyperparameters Swept”. Readers unfamiliar with the specific models (e.g., Qwen2.5 0.5B Base vs. Instruct) may find the 4×4 grid dense; small annotations highlighting key cases where min‑p is better/worse, or a summary table synthesizing across models, would make the central message easier to digest. Similarly, **Figure 6 (right)** could benefit from explicit labeling of which points correspond to which sampler–temperature pairs, since much of the argument depends on win rates overlapping around 50%. The current text explains the interpretation, but more explicit figure captions or annotations would aid comprehension.

## Potentially Missing Related Work

Below are related works that appear directly relevant to the paper’s focus on rigor, evaluation, and reproducibility in empirical ML and LLM research, but are not cited:

1. **Pineau et al., “Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)”, 2020.**  
   This report proposes concrete reproducibility checklists and workflow changes at a major ML conference. It is directly relevant to Section 6’s blueprint; it should be discussed in the context of community‑level policies for methodological rigor, perhaps in the Introduction and Discussion, and could help position this work as a complementary case‑study‑driven contribution.

2. **Raff, “A Step Toward Quantifying Independently Reproducible Machine Learning Research”, 2019.**  
   This work attempts to empirically quantify reproducibility rates in ML. It would fit naturally alongside the cited reproducibility‑oriented works (e.g., Pineau et al., 2017; Sinha et al., 2023) and could be referenced when arguing that issues identified in the min‑p case are symptomatic of broader trends.

3. **Doshi‑Velez & Kim, “Towards A Rigorous Science of Interpretable Machine Learning”, 2017.**  
   Although focused on interpretability rather than evaluation per se, this paper develops a framework for rigorous empirical study of interpretability claims. Its methodological stance aligns with the blueprint’s call for carefully designed human evaluations and might be fruitfully cited in Sections 2 and 6 as an example of principled experimental design in ML.

4. **Pouchard et al., “A rigorous uncertainty-aware quantification framework is essential for reproducible and replicable machine learning workflows”, 2023.**  
   This work emphasizes uncertainty quantification and rigorous evaluation as prerequisites for reproducible ML. It directly connects to the paper’s emphasis on confidence intervals and transparent statistical testing (e.g., in **Figure 1**, **Figure 6 right**), and would be appropriate to discuss in Section 2.2 and the broader Discussion.

5. **Patterson et al., “Empirical Design in Reinforcement Learning”, 2024.**  
   While targeted at RL, this paper presents a meta‑scientific analysis of empirical design choices (hyperparameters, baselines, reporting). The hyperparameter‑volume concerns and Best‑of‑N analysis in Section 3.1 parallel issues raised in RL. Citing and contrasting with this work in Section 3 or 6 could strengthen the generalizability argument.

6. **Vanschoren et al., “OpenML: Networked Science in Machine Learning”, 2014.**  
   OpenML provides infrastructure for sharing datasets, code, and experiment results, which ties directly into the call for data and code transparency in Sections 2.1, 3.1, and 6. Incorporating this reference could anchor the blueprint in existing tools that already support the advocated practices.

7. **Michaud et al., “Precision Machine Learning”, 2023.**  
   This paper argues for highly controlled, precise empirical methodology in ML. Its philosophy overlaps significantly with this work’s objective and could be referenced in the Introduction and Discussion to position the min‑p case study as a concrete instantiation of “precision” principles in the LLM evaluation context.

8. **Pessach & Shmueli, “A review on fairness in machine learning”, 2022.**  
   While centered on fairness, this review devotes substantial attention to evaluation methodology and reproducibility in fairness studies. It could be briefly referenced in Section 1 or 6 as an example of domain‑specific rigor efforts that the proposed blueprint complements.

9. **Karpatne et al., “Knowledge Guided Machine Learning: Accelerating Discovery using Scientific Knowledge and Data”, 2023.**  
   This work argues for integrating scientific knowledge and formal reasoning into ML workflows. While not directly about evaluation, it aligns with the paper’s view that empirical ML must be scientifically grounded. A short discussion in the Introduction or Discussion could help position this paper among broader efforts to treat ML as a scientific discipline rather than only an engineering field.

10. **Benigni et al., “Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)”, 2026.**  
    This is a later article examining reproducibility issues and improvements in ML. If available at the time of writing, it would complement the existing reproducibility references and should be cited in the Discussion when arguing that community‑level structures (e.g., reproducibility tracks, checklists) are important for preventing cases like the min‑p story.

## Questions

1. **Scope and generalization of the blueprint.**  
   Could the authors clarify how they envision the proposed practices being adopted in typical empirical ML projects beyond LLM sampling? For example, would they recommend that all benchmark comparisons with tunable methods present Best‑of‑N curves like **Figures 4–5**, and if so, what N ranges and reporting standards do they suggest?

2. **Statistical methodology choices.**  
   In Section 2.2, why did the authors choose Bonferroni correction rather than, say, Holm–Bonferroni or Benjamini–Hochberg FDR, given the modest number of comparisons (12)? Do the conclusions about lack of “consistent superiority” change under alternative corrections? Also, can the authors formalize the Intersection‑Union Test they invoke and explicitly show how the maximum p‑value threshold corresponds to the chosen α?

3. **Assumptions about Likert‑scale data and t‑tests.**  
   Human evaluation scores are typically ordinal (e.g., 1–10 scales). Did the authors check normality assumptions or robustness of the conclusions using non‑parametric alternatives like Wilcoxon signed‑rank tests? Providing such checks (even in an appendix) would further buttress the critique of the original statistical handling.

4. **Details of the GSM8K sweep protocol.**  
   For the GSM8K CoT experiments, how were the 6 hyperparameter values for each sampler chosen in detail, especially when “lightly edited to make them more evenly distributed”? Were the original Nguyen et al. values always included among the 6, or did some get dropped? Also, since basic sampling has no extra hyperparameter besides temperature, was any attempt made to compensate for its smaller search space (e.g., more temperature points) or to explicitly show that this disadvantage is minor?

5. **Re‑analysis of the second human study.**  
   The new study introduced in Nguyen et al.’s Appendix C.2 changes multiple dimensions simultaneously (implementation, prompts, participant pool, rubric, hyperparameters). Have the authors considered running analogous t‑tests and multiple‑comparison corrections on that data as in **Table 1**? If so, do those tests reach similar conclusions? If not, could they elaborate why qualitative inspection of **Figure 3** alone suffices?

6. **Quantifying the extent of selective reporting in LLM‑as‑a‑judge results.**  
   Regarding Section 4.3, could the authors provide a small table in the paper summarizing the two candidate win rates for each sampler (e.g., min‑p at p=0.01 vs. 0.05, top‑p at p=0.9 vs. 0.98) and explicitly show how the published Table 3(b) picks the more favorable min‑p value and the less favorable top‑p value? This would make the critique much more transparent and self‑contained.

7. **Meta‑level: from this case to systemic evidence.**  
   Do the authors have any preliminary evidence from examining other recent LLM sampling or LLM‑as‑a‑judge papers that similar patterns (unequal hyperparameter tuning, uncorrected multiple comparisons, under‑specified methods) occur frequently? Even a brief pilot survey or replication of 5–10 randomly chosen papers would significantly strengthen the argument that this blueprint addresses a field‑wide problem rather than an isolated incident.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The empirical re‑analysis appears careful and technically sound, and the conclusions drawn from **Table 1**, **Figures 1–6**, and the GSM8K sweeps are largely supported by the data. Some aspects (e.g., formalization of the IUT, power analysis, handling of ordinal human ratings, and limited benchmark scope) could be tightened, but there are no obvious fatal methodological errors.

## Presentation Rating

3: good.  
The paper is clearly written, with well‑structured sections, informative figures (particularly **Figures 1–3** for human evals and **Figures 4–5** for GSM8K), and a coherent narrative. That said, some key external artifacts are only described informally, and a few dense figure grids could benefit from additional annotation or synthesis.

## Contribution Rating

2: fair.  
The work provides a valuable and painstaking critique of a prominent paper and introduces a useful hyperparameter‑volume‑controlled Best‑of‑N evaluation procedure, but the claimed “blueprint” is relatively high‑level and does not significantly extend existing reproducibility / rigor frameworks. The impact is thus more in the case study and concrete demonstration than in introducing fundamentally new methodological tools.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper delivers a thorough and well‑documented re‑analysis of a high‑visibility ICLR paper and surfaces genuinely important issues in human evaluation, hyperparameter tuning fairness, LLM‑as‑a‑judge usage, and community‑adoption claims. The GSM8K Best‑of‑N methodology and the detailed breakdown of human eval misinterpretations are particularly strong. However, the work remains heavily centered on a single case, the “blueprint” component is comparatively shallow and not well situated within the broader reproducibility literature, and several analyses lean on externally described artifacts without fully self‑contained evidence. As a result, I see this as a valuable and provocative piece that is slightly below the bar for the main ICLR track as currently framed, though I would not object if other reviewers judge the meta‑scientific value to warrant acceptance.

## Reviewer Confidence

4: confident.  
I am familiar with LLM evaluation, human‑in‑the‑loop assessment, and reproducibility discussions, and I carefully checked the main statistical and experimental arguments (e.g., **Table 1**, **Figures 1–6**). Some external artifacts (e.g., Telegram logs, full GitHub scraping scripts) are not fully available in the main text, so there is some residual uncertainty, but overall my assessment is unlikely to change dramatically.