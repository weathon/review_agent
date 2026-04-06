=== CALIBRATION EXAMPLE 39 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly signals the paper’s dual purpose: a critical case study of min-p sampling and a blueprint for improved rigor. The abstract succinctly summarizes the re-analysis findings and the derived lessons. The claims are specific and appear supported by the paper’s evidence. However, the abstract could more explicitly highlight the novelty of the proposed methodology (e.g., “Best-of-N” analysis for hyperparameter tuning control) and its broader significance for the ICLR community.

### Introduction & Motivation
The introduction effectively frames the crisis of rigor in ML research and motivates the case-study approach. The target paper (Nguyen et al., 2024) is a high-visibility ICLR Oral, making it a pertinent example. The paper’s goal—to derive a blueprint for more meticulous science—is clear. However, the introduction could more crisply enumerate the specific contributions of this work (e.g., a novel method for fair hyperparameter comparison, a template for rigorous re-analysis) rather than leaving them implicit.

### Re-analysis of Human Evaluations (Section 2)
The re-analysis is thorough and methodologically sound. The authors identify critical issues: omitted data (1/3 of human scores), incorrect statistical testing (pooling across conditions without multiple-comparison correction), and mischaracterized qualitative feedback. The use of Bonferroni correction and Intersection-Union Test is appropriate. The visualization (Fig. 1) and Table 1 clearly show that min-p does not consistently outperform baselines.

**Concerns:**
- The decision to focus solely on the “high diversity” setting is justified but may appear selective. A brief analysis of the “low diversity” setting (even if flawed) would strengthen the claim that the original conclusions are invalid across all settings.
- In Section 2.4, the authors note a potential numerical discrepancy (7.80 vs. 5.80) in the new human study but do not fully explain how this was verified or its impact. This should be clarified.
- The qualitative analysis (Fig. 2) is compelling but relies on manual annotation; inter-annotator agreement or a more systematic coding scheme would bolster reliability.

### Extending NLP Benchmark Evaluations (Section 3)
This section presents the paper’s most novel methodological contribution: a “Best-of-N” analysis to control for hyperparameter tuning volume. The extensive sweep (9 models, 4 samplers, 31 temperatures, ~6000 A100-hours) is impressive and credible. Figures 4 and 5 convincingly demonstrate that min-p’s perceived advantage vanishes when hyperparameter search space is equalized.

**Concerns:**
- The analysis is limited to GSM8K CoT. While compute constraints are understandable, the conclusion that min-p does not outperform “across benchmarks” (as claimed by the original paper) is weakened by not evaluating GPQA. This limitation should be explicitly discussed.
- The choice of hyperparameter values (e.g., 6 per sampler) is reasonable but somewhat arbitrary; sensitivity to this choice could be briefly addressed.
- In Figure 5, a few models (e.g., Qwen 2.5 0.5B Instruct) show a slight edge for min-p. The authors should acknowledge these exceptions to avoid overgeneralizing.

### Investigating LLM-as-a-Judge Evaluations (Section 4)
The critique highlights important methodological flaws: under-specified procedures, indirect comparisons (against a common baseline), and non-transitivity of LLM judgments. The observation that min-p received more hyperparameter tuning (Fig. 6) is a valuable insight.

**Concerns:**
- The data for Figure 6 is sourced from “ongoing work” and a public GitHub repository; the provenance should be more clearly explained to ensure reproducibility.
- The evidence of selective reporting (reporting higher scores for min-p, lower for top-p) is damning but presented anecdotally. A more systematic tabulation of all scores would strengthen this claim.

### Community Adoption Claims (Section 5)
This section is straightforward: the authors demonstrate that the original claims about GitHub repositories and stars were unsubstantiated and retracted. This serves as a cautionary tale about verifying such metrics. The link to reviewer influence is noteworthy.

### Discussion and Limitations
The discussion effectively synthesizes the findings into six general lessons for rigorous research. The lessons are concrete and well-supported by the case study. The stated limitation—that conclusions are based on available evidence—is appropriate. However, the authors should also acknowledge the limitations of their own study (e.g., single benchmark for sweeps, focus on high diversity setting only) and discuss the broader impact of their critique (e.g., correcting the scientific record, potential community backlash).

### Writing & Clarity
The paper is generally well-organized and clear. Some figures (e.g., Fig. 1, 2) could be more thoroughly described in the text. The use of bold for emphasis is helpful but occasionally overdone. The paper meets ICLR standards for clarity.

### Overall Assessment
This paper provides a rigorous, well-evidenced critique of a high-visibility ICLR paper and extracts valuable methodological lessons for the community. The novel “Best-of-N” analysis for hyperparameter tuning control is a significant contribution that could influence evaluation practices. The re-analysis of human evaluations and the exposure of unsubstantiated claims are also impactful. While the study has limitations (primarily scope of benchmarks and some selective focus), the overall argument is compelling and the blueprint for rigor is timely and important. For ICLR, this work represents a meaningful contribution to the meta-discourse on research standards, aligning with the conference’s emphasis on sound science. I recommend **acceptance with minor revisions**, provided the authors address the specific concerns noted above, particularly regarding limitations and methodological clarifications.

# Neutral Reviewer
## Balanced Review

### Summary
This paper conducts a critical re-examination of "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs" (Nguyen et al., 2024), an ICLR 2025 Oral paper. The authors meticulously re-analyze the original paper's four lines of evidence (human evaluations, NLP benchmarks, LLM-as-a-Judge evaluations, and community adoption) and conclude that its central claims of superiority are not supported by its own data. From this case study, they derive several general lessons for promoting more rigorous science in empirical machine learning research.

### Strengths
1.  **Exemplary Rigor and Thoroughness:** The paper provides a model of rigorous, post-publication scrutiny. The re-analysis is comprehensive, covering all original evidence lines, and is backed by extensive new experiments (e.g., the hyperparameter sweep on GSM8K requiring ~6000 A100-hours). The work demonstrates a deep commitment to reproducibility by using the original authors' code and data, and by publicly sharing their own analyses and findings.
2.  **Concrete Identification of Methodological Flaws:** The paper provides specific, evidence-backed critiques. Key findings include: the omission of one-third of human evaluation data, incorrect application of statistical tests (lack of multiple comparisons correction), selective reporting in LLM-as-a-Judge results (reporting the higher score for min-p but the lower for a baseline), and unsubstantiated claims about community adoption that were later retracted. Each flaw is documented with clear references to data, code, or public communications.
3.  **Novel and Practical Methodological Contribution:** The "Best-of-N" analysis for controlling hyperparameter tuning volume (Figures 4, 5, 7, 8) is a novel and highly practical methodological contribution. It provides a concrete, quantitative framework for fair comparison of methods that require extensive hyperparameter searches, addressing a common source of bias in empirical ML research.

### Weaknesses
1.  **Underdeveloped "Blueprint" Framework:** While the paper's title and abstract promise a "blueprint for more rigorous science," the derived lessons (Section 6) are somewhat generic and presented as a list rather than a structured framework or actionable checklist. The paper excels as a detailed case study and critique but falls short of fully delivering on the promised higher-level organizational blueprint for the field.
2.  **Narrative Complexity from Public Discourse:** The paper heavily integrates public discourse (GitHub issues, OpenReview notes, Telegram messages) with the original authors. While this transparency is commendable, it occasionally makes the narrative harder to follow for a reader not immersed in the controversy and sometimes shifts the focus from pure methodological critique to a record of dispute.
3.  **Overly Dense Visualizations:** Some figures (e.g., Figures 1, 9) are extremely dense with data, making them difficult to parse and interpret quickly. While the data is valuable, the visual presentation could be improved for clarity, perhaps by separating information across multiple sub-figures or using more selective highlighting.

### Novelty & Significance
**Novelty:** The primary novelty lies in the comprehensive, forensic-grade re-evaluation of a high-profile publication, serving as a case study in how to audit empirical claims. The novel "Best-of-N" hyperparameter comparison methodology is a significant secondary contribution. The act of publishing such a detailed critique as a standalone paper, rather than a blog post or comment, is also relatively novel in the ML conference context.

**Significance:** The paper is highly significant for the ICLR community and the broader ML field. It directly addresses the ongoing crisis of rigor and reproducibility. By deconstructing the evaluation of a popular method (min-p), it provides a cautionary tale and a set of concrete, evidence-based lessons for authors, reviewers, and readers. It reinforces ICLR's expected standards and provides tools (like the hyperparameter analysis) to enforce them. The work has the potential to positively influence evaluation practices.

### Suggestions for Improvement
1.  **Structure the "Blueprint" More Formally:** To fulfill the promise of the title, the lessons in Section 6 should be organized into a more formal framework. This could include a checklist for authors, a set of questions for reviewers, or a structured workflow diagram for conducting fair comparisons. This would elevate the paper from a superb case study to a seminal guide.
2.  **Streamline the Narrative Around the Public Debate:** Consider consolidating the references to GitHub issues and public replies into a single timeline or appendix. The core methodological arguments (omitted data, incorrect stats, unfair hyperparameter comparison) are strong enough to stand on their own; the blow-by-blow account of the interaction, while transparent, can be distracting.
3.  **Improve Figure Readability:** Redesign the most complex figures (especially 1 and 9) for improved clarity. For Figure 1, consider using simpler aggregated visualizations (e.g., bean plots) with confidence intervals. For Figure 9, the small multiples are useful but could be grouped more logically or summarized with a key finding before showing the full grid.
4.  **Expand on the Implications for the Review Process:** The paper notes that unsubstantiated community adoption claims swayed reviewers. A deeper discussion on how reviewers can be equipped to detect such issues and apply the paper's lessons (e.g., checking for hyperparameter tuning parity, demanding full data) would be valuable, aligning with ICLR's focus on improving the peer-review mechanism itself.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Extend the hyperparameter sweep to the GPQA benchmark.** The original paper claimed superiority on both GSM8K and GPQA. By only analyzing GSM8K, your critique cannot fully invalidate the claim of "superior performance across benchmarks." A similar sweep on GPQA is required.
2. **Perform a controlled, direct LLM-as-a-Judge comparison.** Your analysis highlights methodological flaws but does not provide a corrected, head-to-head evaluation (e.g., min-p vs. top-p vs. basic on AlpacaEval with matched hyperparameter tuning). Without this experiment, your argument against that line of evidence remains speculative.
3. **Conduct an ablation study on hyperparameter tuning volume.** Your "Best-of-N" analysis is compelling, but you should demonstrate that the performance gap closes specifically when tuning *budget* (e.g., number of trials) is equalized, not just when the hyperparameter *set size* is matched. This would strengthen your central fairness argument.
4. **Test on tasks that explicitly measure "creativity" or "diversity."** The original paper's core claim involves a quality-diversity trade-off. Your analysis relies heavily on a mathematical reasoning benchmark (GSM8K). Evaluating on a creative writing or open-ended generation task is necessary to directly challenge that claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Provide a power analysis for the human evaluations.** Your re-analysis correctly applies multiple comparison corrections, but you should calculate the statistical power of those tests. If the study was underpowered, the lack of significant differences is not strong evidence of equivalence, undermining your conclusion.
2. **Quantify the impact of the temperature scaling implementation change.** You note the authors changed from applying temperature after truncation to before truncation in their new human study. This is a major methodological shift. An analysis quantifying how this change alone affects output distributions (e.g., entropy, top-k repetition) is needed to interpret the new study's null result.
3. **Analyze the variance and stability across sampling seeds.** Your NLP benchmark analysis averages over three seeds. Reporting confidence intervals or the range of best-of-N scores across different seed sets would show if min-p's performance is more or less stable than baselines, which is relevant for practitioners.
4. **Systematically analyze the omitted basic sampling data.** You show it was omitted, but a deeper dive into *why* it might have been omitted (e.g., was it competitive or superior in preliminary results?) would strengthen the case for questionable research practices beyond a simple oversight.

### Visualizations & Case Studies
1. **Show side-by-side example outputs from all samplers.** To support your claim that samplers are qualitatively similar, provide concrete examples of generated text (e.g., from AlpacaEval or the human study prompts) for min-p, top-p, and basic sampling under the same conditions. This would make the qualitative argument tangible.
2. **Create a detailed error case study for GSM8K.** Visualize specific math problems where min-p fails but a baseline succeeds (and vice-versa). This would help diagnose if min-p has a systematic weakness (e.g., in multi-step reasoning) not captured by aggregate scores.
3. **Plot the Pareto frontier of quality vs. diversity.** The original claim is about a superior trade-off. For the human evaluation data, plot quality score against diversity score for each (sampler, hyperparameter) point to visually show whether min-p defines a better frontier.

### Obvious Next Steps
1. **Re-run the LLM-as-a-Judge evaluation with a sound, direct protocol.** This is the most obvious missing piece. You must replace your methodological criticism with a corrected experiment that uses direct pairwise comparisons, consistent hyperparameter tuning budgets, and reports uncertainties.
2. **Test on a broader set of models, especially larger ones.** Your sweep uses models up to 9B parameters. The original paper may have used larger models (not fully specified). Testing on models at the 70B scale or larger is necessary to ensure your conclusion generalizes, as sampling behavior can change with model size.
3. **Formalize and release your "Best-of-N" fairness assessment as a tool.** The paper positions itself as a blueprint. The logical next step is to package your hyperparameter volume control methodology into a reusable library or benchmark suite, which would significantly increase the paper's impact.
4. **Perform a cost-benefit analysis for practitioners.** Given your conclusion that methods are equal with fair tuning, an analysis of the compute cost (time, energy) to find good hyperparameters for each sampler would provide concrete, actionable guidance beyond a scientific critique.

# Final Consolidated Review
## Summary
This paper conducts a rigorous critical re-examination of "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs" (Nguyen et al., 2024), a high-visibility ICLR oral paper. Through a detailed case study, it demonstrates that the original paper's claims of min-p's superiority are invalidated by its own data, uncovering flaws such as omitted human evaluations, incorrect statistical testing, and unfair hyperparameter comparisons. From this analysis, the authors derive general lessons to promote more meticulous science in empirical machine learning research.

## Strengths
- The re-analysis is exceptionally thorough and methodologically sound, identifying concrete, evidence-backed flaws in the original work. For example, it reveals that one-third of human evaluation data was omitted without justification, statistical tests were applied incorrectly (e.g., lacking multiple-comparison corrections), and hyperparameter comparisons were unbalanced—all verified using publicly released data and code.
- It introduces a novel and practical "Best-of-N" methodology for controlling hyperparameter tuning volume, demonstrated through extensive sweeps on GSM8K (~6000 A100-hours). This provides a quantitative framework for fair method comparison, addressing a common source of bias in empirical ML research.

## Weaknesses
- The hyperparameter sweep is limited to the GSM8K benchmark, despite the original paper claiming superiority "across benchmarks" including GPQA. This omission weakens the paper's conclusion that min-p does not outperform others broadly, as the evidence remains incomplete for the full claim.
- The promised "blueprint" for more rigorous science is underdeveloped; the lessons are presented as a list rather than a structured framework or actionable checklist. This reduces the practical utility of the blueprint, which the title and abstract emphasize as a key contribution.

## Nice-to-Haves
- A direct, corrected LLM-as-a-Judge comparison with matched hyperparameter tuning would strengthen the critique of that evidence line beyond methodological criticism.
- Statistical power analysis for the human evaluations could provide deeper insight into whether the null results indicate equivalence or insufficient sensitivity.
- Testing on larger models or tasks explicitly measuring creativity/diversity might enhance the generality of the findings, though the human evaluations already partially address this.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- To fulfill the blueprint promise, consider organizing the derived lessons into a formal checklist or workflow diagram for authors and reviewers, enhancing its actionable guidance.
- Explicitly discuss the limitation of not evaluating GPQA in the context of the original claim, and suggest this as a direction for future work to complete the critique.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0]
Average score: 2.7
Binary outcome: Reject
