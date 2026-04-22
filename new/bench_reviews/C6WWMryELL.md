Now I have sufficient information to write the final consolidated review. Let me compile everything.

## Summary

This paper addresses the overlooked problem of output volatility in long-form LLM generation through a three-stage approach: (1) VOLTBench, a novel benchmark that systematically quantifies length volatility across structured/unstructured tasks, languages, and complexity levels up to 100k words; (2) an attention trace analysis identifying "Attention Collapse" and "Attention Instability" as internal patterns driving volatility; and (3) SELB, a training-free decoding strategy that suppresses EOS tokens, forces section-header generation, and bans conversational filler to mitigate volatility. SELB achieves LVC of 14.02% (69% reduction vs. LongWriter-8B), MLA of 78.25%, and SCA of 100% on 100-section structured tasks.

## Strengths

- **First systematic quantification of multi-sample output volatility**: VOLTBench introduces length volatility (LSD, LVC) as a first-class metric and is the first benchmark to include both multiple sampling and stability evaluation (Table 1), filling a genuine gap. The empirical finding that even LongWriter-8B shows LVC of 45.4% on 100-section tasks (Table 2) is striking and quantifies a real operational problem.

- **Comprehensive benchmark design**: VOLTBench covers structured (code) and unstructured (story) tasks, two languages, three complexity levels, and scales from 5 to 500 chapters (~100k words). The fine-grained constraint framework (Section 4.2) with character-level, keyword, and theme constraints provides diagnostic depth beyond aggregate quality metrics.

- **Strong empirical effectiveness on structured tasks**: On the 100-section task (Section 6.3), SELB achieves SCA 100% vs. LongWriter-8B's 32.6%, MLA 78.25% vs. 31.6%, and reduces LVC from 45.4% to 14.02%. The training-free nature of SELB (Equations 2–3) makes it immediately applicable to any model without fine-tuning.

- **Interesting finding on structured vs. unstructured volatility**: Figure 3 shows structured tasks yield more stable output, with a plausible explanation that format constraints provide generation guidance. This is actionable insight for prompt design.

## Weaknesses

### Fatal
None.

### Major

- **Disconnect between probing and mitigation stages**: The paper frames a three-stage pipeline (benchmarking → probing → mitigation) and claims to go "beyond mere phenomenological observation" (Introduction, Section 1), but SELB's three operations—EOS suppression, section-header boosting, filler banning—are surface-level token interventions that address the *symptoms* identified (premature termination, section skipping) without modifying or engaging the *internal attention dynamics* that Section 5 identifies as root causes. The paper states "Targeting the identified internal patterns, we propose...SELB" (Section 1), yet the interventions could have been devised from simply observing that models stop early and skip sections, without any attention trace analysis. The probing stage provides narrative motivation but no functional contribution to the method design. This structural gap between what is analyzed (attention collapse/instability) and what is intervened on (logit modifications for known tokens) is significant because it undermines the paper's central claim of a principled, insight-driven pipeline.

- **Evaluation scope favors SELB's mechanism**: SELB requires knowing section headers in advance (Equation 2 requires V^{(p+1)}_{title}), and the primary evaluation (Section 6.3) is on a 100-section structured task—the exact format SELB is hardwired for. The comparison with LongWriter-8B is informative but inherently favors SELB, which enforces the correct number of chapters by construction. The more challenging free-form generalization (SELB-Hybrid, Section 6.4) is relegated to the appendix with limited methodological detail (the "Hybrid Keep-Alive" mechanism is only sketched). This asymmetry means the headline results come from the setting where SELB has maximum structural advantage, while generalization is under-examined.

### Minor

- **Limited statistical rigor for volatility metrics**: All volatility metrics (LSD, LVC, FAD) are computed from N=5 generations per prompt (Section 3.2). With N=5, the standard error of the sample standard deviation is approximately 0.37σ, meaning the 69% volatility reduction claim has non-trivial uncertainty. No confidence intervals or bootstrap estimates are reported. While N=5 is common practice in large-scale LLM evaluation, the paper's core contribution is precisely the measurement of volatility, making the absence of uncertainty quantification more consequential than it would be in a standard generation-quality study.

- **The "148% improvement" claim is unclear**: The abstract states SELB "improves the mean output length of the base model by 148%," but on the 100-section task (Table 2), the base model Qwen2.5-7B produces 445 words while SELB produces 15,651 words—a 3,417% increase. The 148% figure appears to compare against LongWriter-8B (6,320 words) rather than the "base model" as stated, which is misleading framing.

- **Quality evaluation under forced generation needs deeper scrutiny**: When a model is prevented from stopping (EOS suppression) and forced to continue, text quality may degrade through repetition, incoherence, or semantic drift. While SCA=100% validates code task quality and Appendix G analyzes n-gram repetition and TTR, the UCA metric (LLM-as-Judge for unstructured tasks, scoring 86.7%) may not reliably detect subtle semantic degeneration in very long outputs. This is partially addressed but deserves more prominence given the forced-generation mechanism.

### Trivial
None.

## Nice-to-Haves

- **Ablation of SELB components**: Isolating the individual contributions of EOS suppression, section-header boosting, and filler banning would clarify what drives the improvement. As presented, all three are composed without individual effect sizes.

- **Quantitative link between attention patterns and failures**: Reporting what fraction of early terminations are preceded by attention collapse (α(t) → 0), with a detection threshold and rate, would strengthen the claim that attention dynamics causally drive volatility rather than merely correlate with it.

- **Evaluation on external benchmarks**: Testing SELB on established long-form generation benchmarks (e.g., LIFEBench, LongGenBench) would demonstrate generalization beyond VOLTBench.

## Removed Points

- **"Length Constraint baseline performs poorly without explanation"**: This is not a weakness of the paper—a naive length constraint that only suppresses EOS without structural enforcement may indeed produce worse results due to uncontrolled rambling. The poor performance is informative rather than suspicious.

- **"Attention analysis based on 1-2 model examples"**: The paper analyzes Qwen2.5-3B and Qwen2.5-7B, which is reasonable for a mechanistic analysis. While more models would strengthen the generality, this is a nice-to-have rather than a core flaw.

- **"Baselines not properly tuned"**: The paper evaluates several standard decoding strategies (Repetition Penalty, Entropy-Stopping, Length Constraint, Lookahead Decoding). Without specific evidence of misconfiguration, this is speculative.

- **"N-gram repetition and TTR in supplementary section rather than core results"**: The paper does address text quality degradation in Appendices G and H (representational stability analysis). Whether this belongs in the main text vs. appendix is a presentation choice, not a methodological flaw.

- **"Missing confidence intervals on volatility metrics"**: While I kept a minor version of this point, the harsh critic's framing that this is a "serious gap" overstates the case—N=5 is standard for large-scale LLM generation tasks, and demanding bootstrap CIs on every metric is above community norms.

## Novel Insights

The most interesting empirical finding is that structured generation tasks yield significantly more stable outputs than unstructured tasks (Figure 3), suggesting that explicit structural constraints in prompts can serve as scaffolding to reduce volatility. This has immediate practical implications for prompt engineering, independent of SELB itself. The attention trace analysis, while not functionally informing SELB, does provide a useful diagnostic—the identification of "Attention Collapse" (premature attention decay) and "Attention Instability" (abnormal attention spikes) as precursors to generation failure could inform future mechanistic interventions.

## Suggestions

- Revise the paper's framing to accurately describe the relationship between probing and mitigation: SELB addresses the observed failure *patterns* (premature termination, section skipping) rather than the identified *mechanisms* (attention collapse, attention instability). This is an honest and still-valuable contribution without overclaiming mechanistic intervention.

- Include a component ablation (EOS suppression alone, section-header boosting alone, filler banning alone) to quantify each component's contribution.

- Promote the SELB-Hybrid free-form evaluation from the appendix to the main paper, with full methodological detail, to address concerns about evaluation scope.

- Clarify the "148%" figure: specify exactly what comparison this refers to and whether it is an aggregate across all task sizes or a specific task comparison.

## Score and Decision

**Calibration anchors:**

- **High anchors (avg >7):** syThiTmWWm (avg 7.75, Oral) — LLM benchmark gameability with clean execution and clear problem framing; mtSSFiqW6y (avg 8.0, Oral) — decoding strategy with strong results; xoXn62FzD0 (avg 8.0, Oral) — SMC for constrained generation with principled approach. These papers have cleaner methodological pipelines and stronger connections between analysis and contribution.

- **Medium anchors (avg 4-6):** gfFVATffPd (avg 6.0, Accept poster) — attention probing + SAT Probe method for factual error prediction. Similar structure to our paper (analysis → method), with comparable practicality concerns. Scored 6.0. MbtA7no8Ys (avg 5.0, Reject) — mechanistic analysis + selective fine-tuning, with concerns about the connection between analysis and method and about evaluation scope.

- **Low anchors (avg <3):** MGceYYNvXp (avg 1.50, Reject) — ad-hoc benchmark with disconnected analysis; NlY3XppPt3 (avg 2.00, Withdrawn) — misleading framing with only case studies. Our paper is far above these in rigor and contribution.

**Assessment:** This paper has a genuine benchmark contribution (VOLTBench) and produces strong empirical results, but suffers from a structural disconnect between its probing and mitigation stages, and the evaluation scope is tilted toward SELB's strengths. Compared to gfFVATffPd (6.0, similar analysis→method structure with practicality concerns), our paper has a more significant benchmark contribution but a weaker analysis→method pipeline. Compared to MbtA7no8Ys (5.0, mechanistic analysis + method with disconnect concerns), our paper has stronger empirical results but a more fundamental gap between what is analyzed and what is intervened upon. I place this paper between these anchors.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>