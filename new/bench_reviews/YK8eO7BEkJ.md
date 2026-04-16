## Summary
This paper studies normalization in Mamba blocks along three axes: normalization type, placement relative to the SSM module, and pairwise combinations before/after SSM. The main empirical takeaways are that normalization is crucial in the tested setups, post-SSM normalization often outperforms pre-SSM normalization, and some heterogeneous pairs can beat homogeneous choices. The topic is practical and timely, but the evidence does not fully support the paper’s broad recommendation-style framing.

## Strengths
- **Addresses a genuinely useful and underexplored design question.** The paper focuses on a practical issue that is currently handled inconsistently across Mamba variants: what normalization to use, where to place it, and whether to combine different ones.
- **The empirical decomposition is sensible and easy to follow.** Structuring the study into normalization **type**, **position**, and **combination** is a coherent way to map the design space.
- **Some empirical signals are real and potentially useful.** In the reported experiments, normalization dramatically improves over no-normalization, and post-SSM normalization is often better than pre-SSM, especially for GN/LN-like choices (Tables 2–3).
- **The related-work taxonomy is helpful.** Section 2’s grouping into no normalization / before SSM / after SSM / combined normalization is a useful organizational lens for the literature.
- **The paper at least attempts to go beyond raw tables.** The L2-norm analysis in Section 4.6 is not conclusive, but it is a reasonable attempt to provide intuition rather than only presenting leaderboard-style comparisons.

## Weaknesses

###: Fatal

### Major:
- **The central claims are broader than the evidence supports.** The paper repeatedly promises “practical recommendations” for designing Mamba architectures in general, but the main evidence comes from one primary sequence dataset (Breakfast) and one primary vision dataset (ImageNet-100), plus a very thin validation table on ListOps and ImageNet-1k. The observed “best” choices also vary by task: GN is strongest on Breakfast, LN/GN are strongest on ImageNet-100, IN→LN is best on ListOps, and RMSN→BN is best on ImageNet-1k. That supports the narrower claim that normalization design matters and post-SSM often helps, but not strong architecture-level guidance for Mamba broadly.
- **The comparative study is under-specified for the level of conclusion the paper draws.** Since the contribution is primarily empirical comparison, missing protocol detail materially weakens confidence in the rankings: the paper text provided does not specify model size, training schedule, optimizer details, number of runs, or uncertainty estimates. This matters because several headline differences are tiny (e.g., 86.5 vs 86.7 vs 86.8 in Table 3; 87.1 vs 87.3 in Table 4), yet the paper makes winner-style claims without any variance analysis. For a benchmarking paper, this is a substantive evidential gap, not a minor reproducibility nit.
- **The mechanistic explanation is suggestive, not validated.** The paper itself says Section 4.6 is an “intuitive inference” and “not intended as an essential explanation,” which is appropriate. However, elsewhere the writing leans too heavily on this explanation, e.g., the Introduction claims that post-SSM normalization “helped the model maintain consistency in the scale of the L2-norms across different layers during training and even making the weight updates more stable, thereby improving training stability.” The actual evidence for this mechanism is limited to a small number of configurations on a 4-layer ListOps model (Figures 4–5), and it does not establish a causal or generally validated explanation for the broader empirical findings.
- **Validation is too thin to substantiate transferability.** Section 4.5 evaluates only one selected “ours” configuration against one “original” configuration on two additional datasets. That is far too limited to validate the broader recommendations or to show that the trends transfer robustly across settings.

### Minor
- **There is a concrete ambiguity/inconsistency around the validation setup.** In Section 4.5, Table 5 reports the sequence “Ours” configuration as **IN→SSM→LN**, but the paragraph below says “For sequence tasks, RMSN→SSM→RMSN represents the original Mamba’s normalization configuration, while **IN→SSM→IN** represents our proposed normalization configuration.” This is not a formatting nit; it creates uncertainty about what was actually validated.
- **Some recommendation language overstates stability of the findings.** For example, Section 4.4 says “LN emerges as a versatile and consistently strong performer across tasks,” but the tables show more task dependence than this phrasing suggests. Similarly, “applying normalization after the SSM module is generally more beneficial” is directionally supported, but the strength of the conclusion should be softened given the absence of uncertainty estimates and the presence of exceptions (e.g., RMSN on ImageNet-100 is slightly worse after SSM than before).
- **The analysis section uses stronger interpretive language than the evidence warrants.** Terms such as “pathological curvature landscapes” and “harmonic structure” are not really established by the presented plots. As qualitative intuition this is acceptable, but as explanatory support it is overstated.
- **There appears to be at least one suspicious result entry that should be checked.** In Table 4, “GN→SSM→RMSN” is reported as **68.1** for both sequence and image accuracy. That may be correct, but it is unusual enough that it should have been verified or discussed.

### Trivial

## Nice-to-Haves
- Evaluate on additional standard Mamba settings and larger/deeper models to test whether the observed normalization trends persist with scale.
- Report multi-seed mean/std for the main tables, especially where differences are below 1%.
- Extend the L2-norm analysis to the actual top-performing combinations (e.g., IN→LN, RMSN→BN), not just illustrative BN-based cases.
- Provide training-curve evidence if the paper wants to claim improved **training stability**, rather than only improved final accuracy.
- Clarify exactly how BN/IN/GN/LN/RMSN are instantiated for sequence vs image inputs, since these choices can materially affect outcomes.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“No experiments on Mamba2.”** The paper explicitly scopes itself to Mamba blocks and mentions Mamba2 as future work in the conclusion. While evaluating Mamba2 would strengthen the paper, its absence is not a fair core weakness for a paper framed as an empirical study of normalization in Mamba.
- **“Unfair baseline comparison because the validation uses VMamba without FFN.”** The paper explicitly states this was done “for fair comparison.” Since removing FFN handicaps the baseline rather than the authors’ method, this asymmetry actually makes the comparison harder to overstate in the authors’ favor under the paper’s own normalization-focused setup. I therefore do not keep this as a main criticism, though the validation remains too thin overall.
- **Pure complaints about lack of larger language-model benchmarks or production-scale models.** These are scope-expanding demands. The real issue is not that the paper failed to test trillion-scale or LLM-scale settings, but that its claims are too broad relative to the scale it *did* test.
- **Formatting/parser issues.** The extracted text contains some inconsistencies and awkward phrasing, but parser artifacts should not be treated as paper weaknesses.

## Novel Insights
The paper’s strongest credible contribution is narrower than its framing: it does not establish a universal normalization recipe for Mamba, but it does provide evidence that **where** normalization is placed may matter at least as much as **which** normalization is chosen. In particular, the most robust pattern across the reported results is not a single best norm type, but the asymmetry between pre-SSM and post-SSM placement. That suggests future work on Mamba stabilization may benefit more from understanding *interface conditioning around the SSM block* than from searching for a universally superior normalization family.

## Suggestions
- Narrow the claims substantially: present the paper as an empirical study that identifies **promising trends** rather than general design rules for Mamba normalization.
- Fix the Section 4.5 inconsistency about whether the validated sequence configuration is **IN→LN** or **IN→IN**.
- Add multi-seed statistics and uncertainty estimates for all core tables; otherwise, avoid ranking claims based on sub-1% differences.
- Expand validation beyond a single pairwise comparison on two extra datasets; at minimum, test whether the post-SSM trend persists across more than one sequence and one vision setup.
- Recast Section 4.6 explicitly as qualitative intuition and avoid mechanistic language not directly supported by the evidence.
- If space permits, include training dynamics and/or deeper-model analysis to support the paper’s repeated claims about training stability.

## Score and Decision
**Assessment by axis:**  
- **Originality:** Moderate. The contribution is primarily a systematic empirical sweep rather than a new method.  
- **Importance of the question:** Good. Normalization choices in Mamba are practically relevant and worth studying.  
- **Claims vs support:** Weak-to-moderate. Some useful empirical trends are shown, but the recommendation-level claims are overextended.  
- **Experimental soundness:** Moderate at best. The sweep is broad over normalization choices, but narrow over datasets/settings and missing uncertainty reporting.  
- **Clarity:** Mixed. The overall organization is understandable, but some recommendation language and explanatory claims are overstated, and Section 4.5 contains a real inconsistency.  
- **Community value:** Moderate. Even a limited mapping of normalization behavior in Mamba could be useful, but the paper needs tighter claims and stronger validation to be archival-quality.

**Calibration against human-reviewed anchors:**  
I compared this paper against:
- **Mix-LN** (`/home/wg25r/review_agent/human_reviews/BChpQU64RG.md`, scores 6/6/8/5/6, Accept): that paper also studies normalization, but appears stronger because it introduces a clearer method contribution and was judged sufficient for acceptance despite some limits. The current submission is weaker due to narrower evidence and thinner validation.
- **Stable-Transformer** (`/home/wg25r/review_agent/human_reviews/lkRjnNW0gb.md`, scores 6/6/5, Reject): similar in theme of training stability and normalization; this submission is in a comparable range, with useful ideas but insufficiently convincing support.
- **Beyond Standardization** (`/home/wg25r/review_agent/human_reviews/9ut3QBscB0.md`, scores 3/5/6/8/8, Reject): this provides a lower anchor showing that normalization papers can still be rejected when evidence or positioning is contentious. The current paper is somewhat more practically grounded than the low-end anchor, but still not as well-supported as accepted work.
- **Mamba State-Space Models Are Lyapunov-Stable Learners** (`/home/wg25r/review_agent/human_reviews/i9RTCC6whL.md`, scores 3/6/5, Reject): another Mamba/stability paper rejected despite interesting analysis, which is a useful calibration for the current paper’s “interesting but not fully substantiated” profile.
- **GroupMamba** (`/home/wg25r/review_agent/human_reviews/RmmrHEH6Nx.md`, scores 3/3/3, Reject): a low-end Mamba anchor; the current paper is clearly better than this, since it asks a legitimate empirical question and provides some meaningful evidence.

Relative to these anchors, this paper lands in the **borderline-reject** zone: stronger than clearly weak rejected Mamba papers, but weaker than accepted normalization work and not sufficiently substantiated for acceptance in its current form.

**Final score: 4.5 / 10 — Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>