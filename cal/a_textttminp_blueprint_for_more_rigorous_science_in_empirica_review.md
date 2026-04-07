=== CALIBRATION EXAMPLE 37 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title clearly reflects the paper's dual contribution: a case study of the min-p paper and a derived blueprint for rigor. The abstract accurately summarizes the key findings from each line of evidence re-analyzed and states the derived lessons. The claims are specific and supported by the subsequent detailed analysis.

**Introduction & Motivation:** The introduction effectively frames the broader crisis of rigor in ML research, citing relevant literature. It clearly identifies the target case study (Nguyen et al., 2024) and its high visibility, providing strong motivation for a detailed scrutiny. The contributions (re-analysis across four lines of evidence and deriving general lessons) are stated clearly.

**Method / Approach (Sections 2, 3, 4, 5):** The paper's core methodology is a forensic re-analysis of existing data and code, supplemented by new, extensive experiments.
*   **Section 2 (Human Evaluations):** The re-analysis is methodical. Identifying the omission of 1/3 of the data is a critical finding. The statistical re-evaluation (correcting for multiple comparisons, proposing IUT) is appropriate and convincingly undermines the original claim. The manual annotation of qualitative feedback provides concrete counter-evidence. The analysis of the authors' new human study is appropriately critical of its changed parameters and notes a potential data entry error. The methodological approach here is sound and reproducible.
*   **Section 3 (NLP Benchmarks):** This section introduces a novel and highly valuable methodological contribution: the "Best-of-N" analysis to control for hyperparameter tuning volume. The experimental design is thorough (9 models, 4 samplers, extensive hyperparameter grids). The dual-analysis approach (best performance per sampler and min-p's advantage) is complementary and convincing. Acknowledging and re-running experiments after a formatting issue was raised demonstrates rigor. The conclusion that min-p's advantage vanishes with equal tuning is strongly supported by the data in Figures 4, 5, 7, and 8.
*   **Section 4 (LLM-as-a-Judge):** The criticism here is more qualitative but valid. Points about underspecified methodology, indirect comparison design (prone to non-transitivity issues), and the apparent imbalance in hyperparameter tuning are well-taken. The allegation of selective reporting (higher score for min-p, lower for top-p) is serious and backed by a cited source. This section is somewhat less conclusive than others due to reliance on external data, but the identified methodological flaws are significant.
*   **Section 5 (Community Adoption):** The investigation is straightforward and damning. The fact that unsubstantiated claims were retracted after inquiry, and that these claims significantly influenced the original paper's reviewers, is a powerful illustration of a broader problem in the field.

**Experiments & Results:** The new experiments in Section 3 are a major strength. The compute budget (~6000 A100-hours) is substantial and lends significant credibility. The sweep is comprehensive, and the results are visualized clearly across many models and settings. The central finding—that min-p does not consistently outperform when hyperparameter search space is controlled—is robust. The human evaluation re-analysis is also experimental in nature, re-processing the original study's data with correct statistical methods. The results (overlapping CIs, few significant tests after correction) directly contradict the original conclusions.

**Writing & Clarity:** The paper is exceptionally well-written and structured. Complex analyses are explained clearly. Figures are informative and central to the argument. Links to data and code are provided, supporting transparency and reproducibility. The narrative builds logically from one line of evidence to the next.

**Limitations & Broader Impact:** The paper explicitly states its key limitation: conclusions are based on the available evidence, and new evidence could change them. This is appropriate. The broader impact is positive and central to the paper's thesis: promoting rigor, transparency, and robust methodology in empirical ML. The "General Lessons" (Section 6) are a concrete and valuable output for the community. The societal impact is the potential reduction of misleading claims in published research.

### Overall Assessment
This paper presents a meticulous, well-evidenced, and compelling critical analysis of a high-profile publication. It uncovers serious flaws in the original work across multiple dimensions (data omission, statistical errors, unbalanced hyperparameter tuning, unsubstantiated claims) through a combination of careful data re-analysis and substantial new experimentation. The derived "blueprint" for rigorous research is a valuable synthesis. The work meets and exceeds ICLR's standards for rigor, reproducibility, and contribution to the community's methodological health. The most important concern is the inherently adversarial nature of the work, but it is conducted with appropriate evidence and tone. The contribution—both as a specific corrective and a general guide—stands strong.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents a detailed critical re-examination of a high-visibility ICLR 2025 Oral paper that introduced min-p sampling. Through a comprehensive re-analysis of the original work's four lines of evidence (human evaluations, NLP benchmarks, LLM-as-a-Judge evaluations, and community adoption), the authors conclude that the original claims of min-p's superiority are not supported by the data. From this case study, the paper derives a blueprint of six concrete lessons for conducting more rigorous empirical machine learning research, emphasizing fair hyperparameter comparisons, correct statistical testing, data transparency, and scrutiny of qualitative claims.

### Strengths
1.  **Extensive and Rigorous Re-analysis:** The authors provide a meticulous, multi-faceted audit. They re-analyze the original human evaluation data with proper statistical tests (including multiple comparison corrections), conduct an extensive new hyperparameter sweep on GSM8K (∼6000 A100-hours), scrutinize methodological ambiguity in LLM-as-a-Judge evaluations, and fact-check community adoption claims. The evidence is compelling and well-documented with links to original data and code.
2.  **Concrete Methodological Contribution:** The "Best-of-N" analysis framework for comparing methods while controlling for the volume of hyperparameter tuning is a novel and practical contribution. It provides a quantifiable way to detect potential cherry-picking and ensures fair comparisons, addressing a common flaw in empirical research.
3.  **Actionable Blueprint for the Community:** The six derived lessons are specific, actionable, and highly relevant to current crises in ML research regarding reproducibility and rigor. They translate the specific case study findings into general principles that can guide both authors and reviewers.
4.  **Transparency and Engagement:** The work models the transparency it advocates. The authors document their engagement with the original paper's authors (via GitHub, OpenReview), publicly post their annotations and re-analysis code, and clearly state the limitations of their own conclusions.

### Weaknesses
1.  **Limited Generalizability of the Blueprint:** While the case study is deeply analyzed, the paper's primary evidence for its broader "blueprint" is a single paper (Nguyen et al., 2024). A stronger argument for the generality and urgency of the proposed principles could be made by briefly linking the specific flaws found to a wider body of literature on common evaluation pitfalls.
2.  **Confrontational Tone:** The paper's tone is occasionally forceful (e.g., "conclusions are invalidated by its own data"). While the critique is evidence-based, a more neutral and constructive phrasing would better align with ICLR's collegial standards and strengthen the persuasive power of the argument.
3.  **Niche Impact of the Core Finding:** The central finding that min-p does not outperform established baselines, while methodologically sound, is a negative result specific to one sampling technique. The paper’s primary significance to the broader ICLR audience lies in its methodological lessons, not in disproving min-p. The connection between the case study and the general blueprint could be more explicitly foregrounded.
4.  **Incomplete Discussion of Re-analysis Limitations:** The authors note their conclusions are limited to the available evidence but do not deeply discuss potential limitations of their own "Best-of-N" methodology (e.g., the choice of hyperparameter ranges for each sampler, the assumption that performance improves monotonically with more hyperparameters tried) or their interpretation of qualitative feedback.

### Novelty & Significance
**Novelty:** The paper's novelty is twofold: (1) It provides a novel, in-depth "forensic" analysis template for scrutinizing empirical claims, going beyond typical reproducibility checks to include statistical re-analysis, hyperparameter fairness, and data transparency audits. (2) It introduces a practical methodological tool (the "Best-of-N" analysis under controlled hyperparameter volume) for fair method comparison.

**Significance:** The significance is high for the ICLR community. The paper directly tackles pressing issues of evaluation rigor, reproducibility, and potential selective reporting. The proposed blueprint offers concrete, actionable steps that, if adopted, could meaningfully improve the validity of empirical claims published at ICLR and similar venues. It serves as a valuable educational resource for researchers and reviewers alike.

### Suggestions for Improvement
1.  **Reframe to Emphasize the Blueprint:** To increase broad impact, restructure the narrative to foreground the general lessons for rigorous science early on, using the min-p case as a detailed, illustrative example rather than the central focus. The title and abstract could more directly highlight the proposed methodological framework.
2.  **Moderate the Tone:** Revise language to be more neutral and constructive. Focus on identifying methodological flaws and suggesting corrections rather than framing the work as "invalidating" the original paper. This will make the critique more persuasive and professionally aligned with conference norms.
3.  **Strengthen Generalizability:** Add a concise discussion or references linking each identified flaw (e.g., omitted data, lack of multiple comparison correction, unequal hyperparameter tuning) to prior work documenting these as common issues in ML research. This would solidify the claim that the min-p case is symptomatic of wider problems.
4.  **Expand on Methodological Limitations:** Briefly discuss the limitations and assumptions of the proposed "Best-of-N" fairness analysis. Acknowledge choices made in defining hyperparameter ranges and how they might affect the conclusions. This would enhance the methodological rigor of the blueprint itself.
5.  **Improve Visualizations:** Some figures (e.g., Figure 1) have formatting artifacts or are difficult to interpret (e.g., the table in Section 4.3). Ensure all figures are clearly legible and their captions fully explain what is being shown and why it matters to the argument.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Run hyperparameter sweeps on GPQA.** The original min-p paper used GPQA; extending the "Best-of-N" analysis to this benchmark is necessary to substantiate the claim that min-p's superiority vanishes across NLP benchmarks.
2. **Re-run LLM-as-a-Judge evaluations with a direct, properly-specified comparison.** The critique identifies methodological flaws but does not provide corrected results. Without this, the claim that min-p does not outperform remains speculative for this line of evidence.
3. **Compare against a broader set of sampling baselines** (e.g., typical sampling, mirostat). To robustly claim min-p offers no advantage, it must be shown that it does not outperform other modern samplers under equal tuning.
4. **Conduct a human evaluation with equal hyperparameter tuning budgets.** A new study where each sampler is given the same tuning effort would directly test the core claim about fairness, rather than only re-analyzing existing imbalanced data.

### Deeper Analysis Needed (top 3-5 only)
1. **Perform statistical tests on the new human evaluation data** (with multiple comparisons correction). The critique visualizes the data but does not provide statistical significance analysis, leaving the claim that "min-p offers no apparent advantage" unquantified.
2. **Provide uncertainty estimates (e.g., confidence intervals) for the "Best-of-N" curves.** The averages plotted in Figures 4-5 lack measures of variability, making it impossible to judge if observed differences are meaningful.
3. **Quantitatively analyze the quality-diversity trade-off.** The claim that min-p does not improve this trade-off requires a direct metric (e.g., plotting Pareto frontiers) rather than only separate scores for quality and diversity.
4. **Analyze the sensitivity of conclusions to hyperparameter ranges.** The critique uses fixed sets; it should test whether the outcome changes if baselines are given more favorable or different hyperparameter ranges.

### Visualizations & Case Studies
1. **Show side-by-side visualizations of the original human evaluation data (with omitted data included) and the new study.** This would directly illustrate how inclusion of omitted data and methodological changes affect the results.
2. **Provide example outputs from each sampler for human inspection.** Case studies of generated text would ground the claims about lack of superior creativity/coherence and allow readers to judge for themselves.
3. **Plot LLM-as-a-Judge win rates with confidence intervals for all pairwise comparisons.** The current visualization (Fig. 6) is limited; a full matrix of direct comparisons with uncertainty would clarify performance relationships.

### Obvious Next Steps
1. **Release all analysis code and data from the critique.** To fully embody the principle of data transparency and allow independent verification, the authors must share their re-analysis code and processed data.
2. **Formalize the "blueprint" into a concrete checklist or guidelines.** The paper motivates general lessons but does not provide an actionable tool for researchers and reviewers to implement them.
3. **Extend the hyperparameter volume analysis to creative writing tasks** (e.g., using AlpacaEval). The critique focuses on GSM8K; testing on the tasks where min-p was originally promoted would strengthen the conclusion.
4. **Discuss the implications for the review process.** Given the critique's focus on a high-profile conference paper, specific recommendations for how reviewers can detect such flaws would increase impact.

# Final Consolidated Review
## Summary
This paper conducts a detailed forensic re-analysis of a high-visibility publication on min-p sampling, systematically auditing its four lines of evidence. The re-analysis reveals substantial methodological flaws—including omitted data, incorrect statistics, unfair hyperparameter comparisons, and unsubstantiated claims—that invalidate the original paper's conclusions. From this case study, the authors derive a concrete blueprint with six principles for conducting more rigorous empirical machine learning research.

## Strengths
- **Meticulous and transparent re-analysis:** The paper provides a comprehensive, multi-faceted audit of the target paper's evidence. Critical findings—such as the omission of one-third of human evaluation data, the application of incorrect statistical tests without multiple comparison corrections, and the retraction of unsubstantiated adoption metrics—are thoroughly documented with links to original data and code, embodying the transparency it advocates.
- **Novel methodological contribution for fair comparison:** The paper introduces a "Best-of-N" analysis framework to compare methods while controlling for the volume of hyperparameter tuning. This is a practical and novel tool to detect potential cherry-picking and ensure equitable comparisons, demonstrated through an extensive sweep on GSM8K (~6000 A100-hours) that shows min-p's claimed advantage disappears when tuning effort is equalized.
- **Actionable synthesis for the community:** The derived blueprint translates specific flaws into six general, actionable lessons for researchers and reviewers (e.g., controlling hyperparameter volume, applying statistical tests rigorously, demanding full data transparency). This synthesis provides concrete guidance to address widespread crises of rigor in empirical ML.

## Weaknesses
- **Blueprint primarily grounded in a single case:** The proposed principles are compellingly illustrated through the min-p case, but the argument for their broad applicability and urgency rests heavily on this one example. A concise discussion linking each identified flaw to prior literature documenting these as common issues in the field would strengthen the claim that the case is symptomatic, not isolated.
- **Limited analysis of the proposed methodology's own assumptions:** The "Best-of-N" fairness analysis involves choices (e.g., the hyperparameter ranges for each sampler, the assumption that performance improves with more hyperparameters sampled) that could affect conclusions. A brief acknowledgment and discussion of these limitations would enhance the methodological rigor of the blueprint itself.

## Nice-to-Haves
- Extending the hyperparameter sweep analysis to the GPQA benchmark (used in the original paper) would further solidify the claim that min-p's advantage vanishes across NLP benchmarks.
- Providing confidence intervals or other measures of variability for the "Best-of-N" curves would help readers assess the robustness of the small observed differences.
- Formalizing the blueprint into a concrete checklist or set of guidelines would make it even more actionable for authors and reviewers.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "Confrontational Tone"** – While the paper's language is direct, it is evidence-based and professionally focused on methodological flaws. The tone is appropriate for a rigorous critique and does not violate collegial standards.
- **Weakness: "Niche Impact of Core Finding"** – The paper's primary contribution is the methodological blueprint and forensic analysis template, not merely the negative result about min-p. This is clearly foregrounded in the abstract and structure.
- **Suggestion: "Run new human evaluations with equal tuning budgets"** – This demands a new, large-scale study beyond the paper's scope as a re-analysis and critique. The paper's analysis of existing (imbalanced) data sufficiently demonstrates the fairness problem.
- **Suggestion: "Re-run LLM-as-a-Judge evaluations with direct comparisons"** – While ideal, conducting new large-scale LLM evaluations is a major undertaking outside the paper's scope. The critique successfully identifies critical underspecification and apparent selective reporting in the original work.
- **Suggestion: "Compare against a broader set of sampling baselines"** – The paper's focus is on auditing the original claims, which compared against basic, top-p, and top-k. Expanding the model zoo is not required for its core contribution.

## Novel Insights
The paper provides two key novel insights beyond its own contributions. First, it demonstrates a comprehensive "forensic" analysis template for scrutinizing empirical claims, moving beyond basic reproducibility checks to audit statistical practice, hyperparameter fairness, data transparency, and qualitative reporting simultaneously. Second, it introduces a practical, quantifiable methodology (the "Best-of-N" analysis under controlled hyperparameter volume) to objectively assess whether a method's claimed superiority is robust or an artifact of unequal tuning effort—a common but rarely formalized issue in empirical comparisons.

## Suggestions
- To strengthen the argument for the blueprint's generalizability, add a short paragraph or references in the discussion linking each type of flaw uncovered (e.g., omitted data, lack of multiple comparison correction) to existing literature that identifies these as pervasive problems in ML evaluation.
- Ensure all figures are fully legible in the final version (some contain parser artifacts in the provided content), and consider adding a brief caption note to Figure 1 clarifying the visualization method for the confidence intervals.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0]
Average score: 2.7
Binary outcome: Reject
