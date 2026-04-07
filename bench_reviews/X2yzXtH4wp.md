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