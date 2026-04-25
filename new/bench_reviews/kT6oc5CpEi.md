Now let me do calibration searches to properly score this paper.Now I have sufficient information to write the final review. Let me compile everything.

---

## Summary

BlackDAN introduces a black-box multi-objective jailbreaking framework for LLMs that applies NSGA-II (a Pareto-dominance genetic algorithm) to jointly optimize two fitness functions: unsafe token probability (via LlamaGuard2) and semantic consistency (via all-MiniLM-L6-v2 cosine similarity). The paper demonstrates that jointly optimizing for harmfulness and semantic relevance outperforms single-objective baselines across nine LLMs and two multimodal models. It also introduces a "Rank Boundary Hypothesis" analyzing geometric separability of Pareto-ranked prompt embeddings.

---

## Strengths

- **Strong empirical results on the core claim (Table 2):** BlackDAN achieves 95.4% ASR and 93.8 GPT4-Metric on Llama2-7b and 97.5% / 96.0 on Vicuna-7b—dramatically outperforming DeepInception (77.5% / 31.2 on Llama2-7b), especially on the GPT4-Metric which assesses contextual harmfulness, not just keyword-avoidance.

- **Valid internal ablation of multi-objective vs. single-objective (Figure 3):** The heatmap's bottom row ("Multi-objective") directly compares NSGA-II with two objectives against the same NSGA-II with one objective (diagonal self-attack entries), showing consistent gains across all nine models—confirming it is the multi-objective formulation, not just the genetic search, responsible for improved performance.

- **Well-motivated problem framing (Figure 1):** The 2×2 grid demonstrating that single-objective optimization yields either incoherent harmful outputs or coherent refusals concisely captures a genuine gap in prior work and provides clear motivation for the joint optimization.

- **Broad model coverage:** Nine LLMs plus two multimodal models is one of the widest evaluation sweeps in comparable jailbreak papers, and Figure 4 shows cross-modal MO > SO consistently across three scenario types.

- **Time efficiency advantage (Table 1):** ~2 min/sample vs. ~15 min (GCG) and ~12 min (AutoDAN) for Llama2-7b-chat is a genuine practical advantage at comparable or superior ASR.

---

## Weaknesses

### Fatal
*None* — the core claim (multi-objective NSGA-II improves jailbreak effectiveness and contextual relevance) is empirically supported.

### Major

- **Stealthiness is prominently claimed but never implemented or measured.** The abstract states BlackDAN aims at "minimizing detectability"; the conclusion states "The inclusion of multiple objectives—specifically ASR, *stealthiness*, and semantic consistency—sets a new benchmark." Yet Section 3.1 defines *only two* fitness functions (f₁ = unsafe token probability, f₂ = semantic consistency). Stealthiness is mentioned in the contributions section only as a *possible* user-definable extension ("Users can customize and prioritize different factors… such as harmfulness, stealthiness, or relevance"). No stealthiness metric is ever formalized, computed, or evaluated. This is a material gap between the abstract's claims and the actual implementation. A paper that names stealthiness in the title ("contextual jailbreaking… minimizing detectability") must either implement and measure it or remove the claim.

- **GPT-2-XL is included as a jailbreak target (Figure 3, Figure 4), invalidating the corresponding rows/columns.** GPT-2-XL (Radford et al., 2019) is a pre-training-only base model with no safety alignment, RLHF, or refusal behavior. By construction, any prompt will produce high-ASR outputs—there is no safety mechanism to bypass. Including it as a row and column in the transfer matrix inflates apparent generalizability and may contaminate aggregate conclusions.

### Minor

- **PAIR outperforms BlackDAN on GPT4-Metric for GPT-4 (Table 2), but the paper claims otherwise.** PAIR achieves 30.0 GPT4-Metric on GPT-4 while BlackDAN achieves 28.0. The text states "BlackDAN still significantly surpasses other methods like DeepInception and PAIR on GPT-4," which is factually inaccurate for the GPT4-Metric. This is the most practically important target model in Table 2 (the highest-alignment commercial model), and the result reversal is never acknowledged or explained.

- **Multimodal extension (Figure 4) is methodologically underspecified.** The paper claims applicability to multimodal LLMs but does not explain how image inputs are incorporated into the optimization loop, how f₁ and f₂ are applied to image-text pairs, or what "SD," "SD+Typo," and "Typo" scenarios entail methodologically. This is a meaningful claimed contribution that needs methodological grounding.

- **The Rank Boundary Hypothesis (Figure 5) is partially circular.** Prompts are ranked by Pareto dominance on two objectives computed from proxy embedding models. Showing that best-rank and worst-rank prompts are separable in embedding space is expected by construction—prompts with high f₁ and f₂ (which are embedding-derived) will differ in embedding space from prompts with low f₁ and f₂. The SVM visualization lacks statistical evaluation (classification accuracy, margin) and does not illuminate anything that wasn't already assumed by the objective design.

- **LlamaGuard2 log-probability access requires clarification.** f₁ computes `log P(unsafe token | R)` from LlamaGuard2. This requires local model access or a probability-returning API. If LlamaGuard2 is run locally, the "black-box" claim applies only to the *target* model but not the judge. The paper should clarify this distinction since it affects the reproducibility of the framework on fully API-gated targets.

### Trivial

- Internlm2-chat-7b appears twice in the model list (Section 5.1), apparently as a duplication.

---

## Nice-to-Haves

- A direct qualitative comparison of responses from BlackDAN vs. PAIR vs. DeepInception for the same harmful query, scored by human annotators for both harmfulness and contextual relevance, would concretize the GPT4-Metric gap and validate the semantic consistency claim beyond cosine similarity.
- A query-count budget comparison alongside time-per-sample would more fairly situate BlackDAN vs. iterative LLM-based methods (PAIR, TAP), since NSGA-II's fitness evaluations are cumulative queries.
- If stealthiness is a stated future objective, running generated prompts through Perspective API or OpenAI's moderation endpoint and reporting bypass rates would be a meaningful experiment even as a preliminary result.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

1. **[Harsh Critic] "AutoDAN is mislabeled as gray-box."** The paper cites *two* AutoDAN papers: Zhu et al. (2023) in Table 1 (labeled gray-box) and Liu et al. (2023b) in related work (labeled black-box). The Zhu et al. variant does use gradient/logit-level information from target models as part of its genetic selection, making it legitimately gray-box in the standard taxonomy. The critic conflated the two AutoDAN papers; the label is defensible. **Removed.**

2. **[Harsh Critic] "The multi-objective vs. single-objective ablation does not isolate the contribution of multi-objective optimization."** Figure 3's bottom row ("Multi-objective") vs. the diagonal (single-objective NSGA-II self-attacks on the same framework) is exactly the within-framework ablation the critic demands. The claim that this comparison is absent is incorrect. **Removed.**

3. **[Harsh Critic] "WordNet synonym mutation will degrade coherence."** While this is a reasonable concern, it is a standard design choice in genetic algorithms and the paper shows empirically high ASR despite this mutation operator. Without an ablation showing degradation, this remains speculative. **Weakened to Nice-to-Have** (ablation of mutation operator).

4. **[Strength Finder] "Rank Boundary Hypothesis and embedding space analysis."** Partially retained as evidence of interpretability contribution, but the circular nature is flagged as a minor weakness above rather than a strength.

5. **[Strength Finder] "Consistent superiority across all models in Figure 3."** Retained as a core strength.

---

## Novel Insights

The most genuine novel insight in this paper—one the reviewers do not fully articulate—is that optimizing jointly for unsafe token probability *and* semantic consistency using Pareto dominance produces dramatically higher GPT4-Metric scores (a measure of *contextual harmfulness*) even when the ASR gap is small. The 50+ percentage-point improvement in GPT4-Metric vs. DeepInception on Llama2-7b (93.8 vs. 31.2) while ASR improves only ~18 points (95.4 vs. 77.5) suggests that semantic consistency optimization preferentially filters *incoherent* successful bypasses—a useful diagnostic for understanding why existing keyword-ASR metrics systematically overestimate practical jailbreak risk.

---

## Suggestions

1. **Remove stealthiness from the abstract and conclusion, or implement it.** Replace "minimizing detectability" with "improving contextual relevance" in the abstract/title framing, and clarify in the introduction that stealthiness is a *supported but unimplemented extension* of the framework. Alternatively, implement f₃ = Perspective API toxicity score and report detection bypass rates—this would turn a major weakness into a genuine strength.

2. **Remove or clearly caveat GPT-2-XL results.** Either exclude GPT-2-XL from Figure 3 and Figure 4 or add a footnote explaining that it has no safety alignment and its inclusion is for completeness only (not as a safety-relevant target).

3. **Correct the claim about GPT-4 results.** Acknowledge that PAIR outperforms on GPT4-Metric for GPT-4 and discuss why (e.g., PAIR's iterative LLM refinement may be better suited to GPT-4's alignment regime).

4. **Expand multimodal methodology.** Add a dedicated subsection explaining how image inputs are concatenated with jailbreak templates, how f₁ and f₂ are adapted for image-text responses, and what the SD/Typo scenarios involve.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| AutoDAN (genetic algorithm, stealthy jailbreak) | `7Jwpw4qKkb.md` | 7.0, Accept (poster) | Similar method (hierarchical genetic algorithm for jailbreak), but actually implements perplexity-based stealthiness and has cleaner methodology. BlackDAN's stealthiness claim without implementation puts it below this. |
| DAG-Jailbreak (black-box jailbreak framework) | `xQIJ5fjc7q.md` | 5.5, Reject | Similar scope: comprehensive black-box jailbreak framework with empirical gaps and some methodological underspecification. BlackDAN is comparable in overall quality—solid empirical results but key claims (stealthiness) unsupported. |
| SoC attacks (MAB-based jailbreak) | `jCDF7G3LpF.md` | 6.25, Accept (poster) | Comparable ASR results but with theoretical bounds; accepted partly for the theoretical contribution. BlackDAN is more empirically driven and lacks theoretical backing. |
| TIARA (transferable red-team attacks) | `4GcZSTqlkr.md` | 4.5, Withdrawn | Similar red-teaming scope, rejected for missing baselines and limited novelty. BlackDAN has stronger empirical results but similar overclaiming issues. |
| MRCJ (multi-round jailbreak) | `KyKTjRtyNG.md` | 3.0, Reject | Minimal novelty, rejected. BlackDAN is clearly above this level—the two-objective formulation with NSGA-II is a real contribution. |
| KDA (distilled jailbreak attacker) | `UWuTZYPSxJ.md` | 2.5, Reject | Weak contribution, poor experiments. BlackDAN is substantially stronger. |

**Positioning:** BlackDAN sits between the DAG-Jailbreak (5.5, rejected) and AutoDAN (7.0, accepted). The core empirical contribution is real and the results are strong, but the most prominent framing claim (stealthiness) is not implemented, and experimental design has notable issues (GPT-2-XL). This is closer to the DAG-Jailbreak tier. I place it at **4.5**—the overclaiming is too prominent to overlook, and the methodological gaps (stealthiness, multimodal underspecification, GPT-2-XL inclusion) collectively push this below the acceptance line.

**Evaluation by axis:**
- *Originality:* Moderate — applying NSGA-II to multi-objective jailbreak template selection is novel, but the genetic algorithm jailbreak paradigm was established by prior work.
- *Importance:* High — semantic consistency in jailbreaks is a genuine and underexplored concern.
- *Claims supported:* Partially — the semantic consistency + ASR claim is well-supported; the stealthiness claim is not.
- *Soundness of experiments:* Fair — Figure 3 provides a valid internal comparison; GPT-2-XL inclusion and uncorrected analysis of GPT-4 results weaken soundness.
- *Clarity:* Fair — core methodology is clear; multimodal extension and stealthiness are underspecified.
- *Value to community:* Moderate — the GPT4-Metric results are genuinely informative; the stealthiness overclaim risks confusing the field.

**Final Score: 4.5 / Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>