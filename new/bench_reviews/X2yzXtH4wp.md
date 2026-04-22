Now I have all the information needed. Let me write the final consolidated review.

## Summary

The paper introduces Ambig-SWE, an underspecified variant of SWE-Bench Verified, and decomposes the problem of handling underspecification in agentic code generation into three steps: detection, clarification, and leveraging interaction. It evaluates six LLMs across these steps, finding that interaction can substantially recover performance lost to underspecification, but models default to non-interactive behavior and struggle with detection.

## Strengths

- **Three-step decomposition (detection → clarification → leveraging)** provides a genuinely useful analytical framework for diagnosing where models fail in handling underspecification. This is an advance over prior work like Chen et al. (2025) and Kim et al. (2024), which study single missing details rather than multiple interdependent gaps in multi-step workflows. The decomposition enables targeted improvement recommendations (§3, §4, §5).

- **Finding that information extraction volume does not predict task success** (§5.2) is a nuanced and important insight. Claude Sonnet 3.5 and Haiku extract nearly identical information (0.136 vs 0.135 cosine distance) but differ by ~13 percentage points in resolve rate (39.6% vs 26.8%). This reveals that *how* models integrate information matters as much as how much they extract—a finding not established by prior work.

- **Qualitative analysis of question-asking strategies** (§5.3) identifies concrete, actionable behavioral patterns: Claude models explore the codebase first then ask only what cannot be independently discovered (3.80–4.03 questions), while Deepseek asks overly specific implementation questions exceeding realistic user knowledge, and Qwen asks immediately and excessively (6.02 questions). This goes beyond benchmark-and-report to provide design guidance.

- **Navigational vs. informational decomposition** (§3.3, Table 1) is a valuable analysis that reveals model-specific dependency patterns, such as Qwen 3 Coder's performance *worsening* with navigational info (55.4% → 52.4%) due to rigid protocol-following, and Deepseek's strong dependence on file locations.

- **Demonstration that models default to non-interactive behavior** and that prompt engineering alone is insufficient for reliable detection (Table 2) is an important and concerning finding, with Qwen 3 Coder showing 100% FNR across all prompt levels.

## Weaknesses

### Fatal
None.

### Major

- **Oracle file locations in the user proxy inflate interaction benefits.** The paper states in §2.3 that "the proxy has access to file locations that need modification and can provide them when queried." These file locations are SWE-Bench gold-standard annotations derived from the patch—not information a real issue reporter would reliably possess. This contradicts the proxy's stated design principle of responding "only using information explicitly present in [the full issue]" (§2.2). Table 1 shows that navigational information substantially boosts several models (e.g., Claude Sonnet 3.5: 37.94% → 59.52%; Deepseek-v2: 4.62% → 13.19%), meaning a significant portion of the headline "interaction benefit" comes from access to oracle metadata rather than genuine user-agent dialogue. While the paper deserves credit for providing the navigational/informational breakdown in Table 1, the headline claims (abstract's "up to 74%") conflate both effects, and the paper does not acknowledge this as a limitation in §7. A control experiment rerunning the Interaction setting without file locations in the proxy's knowledge would cleanly isolate the value of interaction.

- **Unequal turn limits confound cross-model comparisons.** Claude Sonnet 4 and Qwen 3 Coder receive up to 100 interaction turns, while all other models receive 30 (§3.1). The stated justification ("to account for their greater reasoning and planning capacity") is circular—it gives more resource to already stronger models. This makes it impossible to determine whether performance differences stem from interaction capability or simply from having 3.3× more steps. The paper does not acknowledge this as a limitation, and no ablation equalizing turn limits is provided.

### Minor

- **The "up to 74%" headline claim is a relative improvement that can mislead.** This figure represents the maximum relative improvement ((Interaction−Hidden)/Hidden) across models, which inflates small absolute improvements on weaker models. The abstract presents this as the headline result without contextualizing that it is a relative metric. The absolute improvements across models (and the more modest relative gains for stronger models like Claude Sonnet 4) are the more informative numbers.

- **RQ2 conflate detection with interaction propensity.** The detection experiment (§4) measures whether models *choose to interact*, not whether they can *identify* underspecification. A model that correctly recognizes missing information but rationally decides it can resolve independently would be counted as a false negative. The paper calls this "detection" which somewhat overstates what is measured. This is partially acknowledged in the paper's discussion but the framing could be more precise.

- **Synthetic underspecification primarily removes information rather than reflecting real missing-intent patterns.** The generated underspecified issues strip code snippets and error messages from complete issues, whereas real underspecification often involves intent or context that was never articulated. The paper acknowledges distributional differences (§2.1) but dismisses them too quickly, and the guarantee that the underspecification is always resolvable (the answer exists in the original issue) may not hold for real-world cases.

- **Data leakage concern for Qwen 3 Coder is raised but not investigated.** The paper notes in §3.2 that "Qwen 3 Coder relies on its internal knowledge for key insights about missing information" and that "correct assumptions potentially inflate its performance." This is a serious concern that could distort model comparisons but is only mentioned in passing without probing (e.g., via temporally held-out issues).

### Trivial
None.

## Nice-to-Haves

- Trajectory-level case studies showing exactly what information was obtained at each interaction turn and how it changed the agent's subsequent actions, making the "information integration" claim more concrete.
- A forced-choice classification task (separate from the interaction decision) to cleanly measure underspecification *recognition* vs. interaction *propensity*.
- Investigating the data leakage concern for Qwen 3 Coder through temporal holdout or analyzing whether its assumptions align with training data.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Critic: "The proxy receives the full issue and responds only using information explicitly present in it, preserving the original knowledge boundaries of the issue reporter" is broken by file locations.** — This is a valid observation that I've incorporated into the Major weakness above. However, the critic's framing that this "compromises the paper's central claim that interaction recovers performance lost to underspecificity" is overly strong; the paper does provide Table 1 data allowing readers to separate effects, and the informational-only interaction still shows benefits for most models.

- **Critic: Missing related works.** — Removed per hard rules; cannot verify existence of suggested references.

- **Critic: Missing appendix/proofs.** — Removed per hard rules; parser strips appendices that exist in original submission.

- **Critic: Reproducibility concerns about hyperparameters.** — Removed per hard rules as trivial reproducibility nitpick.

- **Strength Finder: "Demonstration of substantial performance recovery through interaction (up to 74%)" as a core strength.** — This is partially undercut by the oracle file location issue. The interaction recovery is real but the magnitude is inflated by oracle metadata. Downgraded; the qualitative/structural finding that interaction helps is still a strength, but the quantitative "up to 74%" claim is weakened.

- **Strength Finder: "Methodologically sound dataset construction with paired ground-truth specifications."** — Partially valid but the synthetic underspecification limitation (information removal vs. missing intent) tempers this. Kept as a supporting strength but not elevated to core.

- **Human finder weaknesses about LLM-simulated environments producing artificially easy benchmarks.** — The paper explicitly acknowledges that the proxy may be more cooperative than real users (§7), and this is a known design tradeoff, not a novel criticism.

## Novel Insights

The most novel insight emerging from the reviews is the disconnect between information extraction volume and task success: models that extract similar amounts of information (measured by cosine distance or LLM-as-judge) can differ dramatically in resolve rates, revealing that *information integration*—how models incorporate acquired details into their reasoning and planning—is the bottleneck, not information acquisition. This has direct implications for training: current approaches that optimize for question quality or quantity may miss the more critical capability of adaptively revising plans based on user input.

## Suggestions

- Rerun the Interaction setting with the user proxy stripped of file location knowledge. This single control experiment would cleanly separate the value of genuine user-agent dialogue from oracle navigational hints, and would substantially strengthen (or appropriately temper) the headline claims.

- Equalize turn limits across models (e.g., 30 turns for all, or 100 for all) and report results, at minimum as an ablation. This would allow fair cross-model comparison of interaction effectiveness.

- In the abstract and §3, report both relative and absolute improvement figures, and clearly specify which model produces the "up to 74%" figure, so readers can assess practical significance.

## Score and Decision

**Calibration comparison:**

| Anchor Paper | Avg Score | Comparison |
|---|---|---|
| Active Task Disambiguation (JAMxRSXLFz) | 7.33 | Much stronger formalization (Bayesian Experimental Design) and cleaner methodology; this paper lacks equivalent theoretical grounding |
| ToolEmu (GEcwtMk1uA) | 7.33 | Stronger validation (human evaluation of 68.8% failure identification rate); this paper's user proxy validation is weaker |
| Proactive Agent (sRIU6k2TcU) | 5.50 | Similar LLM-simulated user concerns; this paper has more thorough decomposition and analysis but also more serious methodological confounds (oracle file locations, unequal turns) |
| Agent-as-Judge (DeVm3YUnpj) | 5.67 | Similar benchmark + evaluation framework contribution; this paper has more novelty in decomposition but similar concerns about evaluation validity |
| LLF-Bench (H0UcwHgwEO) | 4.75 | Rejected for insufficient experiments/analysis; this paper has significantly better analysis and qualitative insights |
| Planning Benchmark (koza5fePTs) | 2.00 | Weak novelty, reproduces known trends; this paper has genuinely novel findings |
| Omnipotent Adversarial Training (QgMaqtB444) | 4.33 | Oracle information confound leads to rejection; this paper has a similar but less severe confound, partially addressed by Table 1 |

The paper sits above the rejected benchmark papers (4-5 range) due to its genuinely novel decomposition, insightful qualitative analysis, and actionable findings. It sits below the spotlight papers (7+ range) due to the oracle file location confound and unequal turn limits, which partially undermine the core quantitative claims. The closest comparison is Proactive Agent (5.50), which also uses LLM-simulated users and faces overclaiming concerns. This paper has stronger analytical contributions but also more serious methodological issues. I place it at 5.5, reflecting real contributions tempered by significant but partially addressable methodological concerns.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>