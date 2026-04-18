Looking at the paper content provided in the prompt, let me verify the key claims:

**On the "circular" definition of sparsity (Critic's Point #1):**

The paper says in Section 4.1:
- "we define the noise terms to be the largest set of terms whose sum is less than or equal to zero. The indices of the remaining terms constitute S_ij. Terms denoted by S_ij are strictly positive, and are the largest positive terms."

This is indeed a heuristic. But is it circular? The definition selects terms that are positive and collectively explain the score. The key question is: would any random decomposition produce "sparse" results under this heuristic? The paper shows in Figure 3 that |S_ij| is typically 2-4 for IOI data. But the critic's point is valid that they don't compare to alternative bases.

However, the paper does show in Figure 3(b) that the sparsity also holds on non-IOI data, and Figure 2 shows consistency across prompts - which provides SOME evidence this isn't purely an artifact. The concern about alternative bases is legitimate though.

**On the contribution measure (Critic's Point #2):**

Equation 7 assigns contributions as c_{i,j}^{ℓa,ℓb} = Σ_{k∈S} √σ_k u_k^T o_i^b. The critic argues this isn't a principled causal measure. This is a fair point - the √σ_k splitting is heuristic, and it doesn't account for non-linear transformations. However, the paper does validate these via interventions in Section 5.4.

**On circular validation (Critic's Point #3):**

The critic argues that since interventions are along SVD-identified directions and these directions matter by construction, the validation is circular. But this isn't quite right - the paper also includes "random" interventions (Δ_random) using randomly chosen singular vectors, showing they have much less effect. This is shown in Figure 6 (orange/blue bars). This addresses the concern partially.

**On lack of baselines:**

This is a fair concern. The paper compares "all slices" vs "minimal set of slices" in Figure 4, but doesn't compare to other circuit tracing methods like activation patching.

Now let me calibrate the score. Looking at comparable papers:

- Sparse Feature Circuits (Marks et al.) - Oral, scores 8,8,8,8 - much more complete evaluation, broader scope
- Circuit Component Reuse - Spotlight, scores 8,6,6,6 - good empirical work with clear comparison
- Efficient Automated Circuit Discovery (CD-T) - Poster, scores 8,6,5 - stronger baselines comparison
- Is This the Subspace You Are Looking For - Poster, scores 8,3,8 - identified important limitation, well-validated
- Automatically Identifying Sparse Circuits - Rejected, scores 5,3,1,5,3 - similarly lacked baselines and had circular validation concerns

This paper is between the rejected Hierarchical Tracing paper (which had significant issues with faithfulness and lack of baselines) and the accepted CD-T paper (which had stronger baselines and comparison). The current paper has an interesting and novel idea but with significant methodological concerns.

Let me now write the final review.</think>## Summary

The paper introduces "singular vector tracing," a method for tracing circuits in transformers by leveraging the observation that attention scores are sparsely decomposable in the SVD basis of the combined QK matrix Ω. By projecting residual stream contributions onto low-dimensional "signal" subspaces identified by this sparse decomposition, the method traces which upstream attention heads causally contribute to downstream head computations—all in a single forward pass, without requiring counterfactual datasets. The approach is demonstrated on GPT-2 small performing the IOI task, where it recovers known circuits and identifies novel communication pathways and redundant structures.

## Strengths

- **Novel and mathematically clean formulation.** The reformulation of attention score computation as a bilinear form via Ω (Eqs. 2–3) and the subsequent SVD decomposition into orthogonal slices is elegant and provides a principled mathematical basis for identifying signal subspaces. This is genuinely distinct from prior SVD-based interpretability work on OV matrices or attention matrix low-rank approximation.

- **Compelling qualitative evidence for the sparse decomposition phenomenon.** Figure 1 directly shows that a small number of orthogonal slices dominate attention scores. Figure 3 shows that |S_ij| is typically 2–4 on IOI data and remains low on non-IOI data (The Pile), suggesting the phenomenon is not an artifact of the specific task. Figure 2 shows consistent usage of specific slice sets across prompts, which is consistent with the hypothesis that singular vectors encode functionally relevant features.

- **Clear demonstration of the denoising/filtering effect.** Figure 4 is one of the paper's strongest results: using full residuals produces a noisy, uninformative contribution map, while projecting onto the signal subspaces cleanly recovers known functionally important heads (e.g., (8,6), (7,3), (7,9)). This directly motivates the utility of the approach over naive residual inspection.

- **Causal validation via interventions.** The paper goes beyond qualitative circuit recovery by performing local and global ablation/boosting interventions on identified edges (Figure 6) and structural interventions on parallel paths (Figure 7). The inclusion of random-subspace interventions as a control (showing much weaker effects) provides some evidence that the identified directions are specifically meaningful, not just any directions.

- **Single forward pass, no counterfactuals.** A practical advantage: the method avoids well-known pitfalls of patching approaches (self-repair, counterfactual dataset construction, computational cost).

## Weaknesses

### Fatal
None.

### Major

- **The "sparse decomposition" criterion (S_ij definition) is heuristic and lacks comparison to alternative bases.** The paper defines S_ij as "the largest set of terms whose sum is ≤ 0" and keeps the remaining positive terms as "signal." While this is a reasonable heuristic, it is not tested against alternatives. The central claim that "attention scores are typically sparsely decomposable given the right basis" would be substantially strengthened by demonstrating that alternative bases (e.g., random orthonormal bases, PCA of activations, top-k singular vectors by magnitude without the noise-filtering heuristic) produce less sparse or less interpretable decompositions. As it stands, the sparsity in Figure 3 could be a property of how positive terms distribute in any orthogonal decomposition of a bounded bilinear form, not uniquely of the SVD basis. The paper would need a null model comparison (e.g., comparing |S_ij| under the SVD basis vs. random rotations) to firmly establish that the sparsity is a distinctive property of the SVD basis rather than an artifact of the selection rule. The evidence in Figure 3 showing sparsity on non-IOI data partially addresses this by showing it isn't task-specific, but does not address the basis-choice concern.

- **No comparison to other circuit tracing methods.** The paper claims advantages over patching-based methods (no counterfactuals, avoids self-repair, faster) but never runs head-to-head comparisons measuring precision/recall versus activation patching, ACDC, EAP, or the method of Ferrando & Voita (2024) on the same IOI benchmark. The only quantitative agreement reported is precision 0.52 and recall 0.69 against Wang et al.'s circuit—not a strong result by itself, and no baseline comparison is provided (e.g., what precision/recall would a naive method like "all heads in layers 7–10" achieve?). Without such comparison, it is unclear whether the method recovers circuits more accurately or just differently.

- **The contribution measure c_{i,j}^{ℓa,ℓb} (Eq. 7) is a heuristic without a derivation from first principles.** The paper states this "conceptually" estimates how much an upstream head changes a downstream score, but does not derive it as a first-order Taylor approximation or show equivalence to gradient-based influence measures. The √σ_k splitting between source and destination is arbitrary (why not σ_k on one side and 1 on the other?). The paper acknowledges that intermediate processing (layernorm, MLPs) is not captured, but treats c as a causal proxy regardless. Since edges in the traced graph G are selected based on these contributions (70% threshold), the method's output depends on an unvalidated proxy for causal influence.

### Minor

- **Limited scope: only GPT-2 small on IOI.** The paper positions its findings with general claims about "attention heads" and "language models," but all detailed tracing is confined to one model and one task. The IOI circuit is well-studied and relatively simple. Generalization to other tasks, larger models, or different architectures is entirely unknown. This is acknowledged in the limitations but still tempers the strength of general claims in the abstract and introduction.

- **The interpretability of identified signals is underdeveloped.** The abstract claims the method identifies "features used to communicate between attention heads," but the main text provides only one brief example (head (9,9) separating names from non-names, details deferred to the appendix). For a paper centered on "identifying signals," this central claim is thinly supported.

- **Arbitrary thresholds without sensitivity analysis.** The 70% upstream contribution threshold and 50% firing threshold are presented heuristically. No sensitivity analysis shows how the traced circuit changes as these vary.

- **MLPs are excluded from the analysis.** The method traces only attention-head-to-attention-head communication. MLPs play important roles in known IOI circuits, so the trace is incomplete. This is acknowledged as future work, but it means important causal pathways could be missed or misattributed.

### Trivial
- None.

## Nice-to-Haves

- Comparison of identified SVD subspaces to SAE features on corresponding heads, to ground the claim that these subspaces correspond to "features" in the standard mechanistic interpretability sense.
- Evaluation on at least one additional task (e.g., greater-than or docstring circuits) to demonstrate generality of the sparse decomposition phenomenon.
- Formal sparsity metric and comparison to null models (random rotations, PCA) to rigorously establish that the SVD basis is uniquely beneficial.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Critic's claim that validation is "circular" because interventions use SVD-identified directions**: This is overstated. The paper includes random-subspace controls (Δ_random) in Figure 6, showing substantially weaker effects than SVD-identified directions. This is a meaningful (though not complete) control—intervening along *some* directions working better than random directions is evidence beyond mere circularity. However, the lack of comparison to other principled non-random direction choices (e.g., gradient-based, top-k by magnitude) remains a valid concern, listed above.

- **Critic's claim that the sparsity definition is purely "self-confirming" and any vector of real numbers can be so decomposed**: While the selection rule does by construction guarantee that the positive terms approximate the score, this doesn't mean *any decomposition* would yield sparsity of 2–4 out of 64 terms. The empirical finding that |S_ij| is consistently small IS informative—it's just not as informative without comparison to alternative bases. The concern is legitimate but overstated as "definitional circularity."

- **Human Finder's point about "interpretability illusions from subspace interventions"**: The cited paper (Ebt7JgMHv1) is about a different phenomenon (dormant parallel pathways in subspace patching) and the connection is speculative. The current paper does provide local vs. global intervention comparisons that partially address this concern.

- **Critic's extensive Section-by-Section Notes**: Many of these are valid observations but are already captured in the major points above. Several are formatting/presentation preferences that fall under "remove pure formatting/style nitpicks."

- **Neutral Reviewer's point about computational cost comparison**: The paper doesn't claim dramatic speedups over patching (it claims "single forward pass" as a conceptual/architectural advantage). Wall-clock comparison is a nice-to-have, not a weakness.

- **Human Finder's point about "moderate precision of 0.52"**: While 0.52 precision is indeed modest, the paper also achieves recall of 0.69 and identifies edges *not* in the canonical Wang et al. circuit. The contribution is not solely measured by agreement with prior work; new edges that validate causally are also important. The concern about lack of baseline comparison is already captured above.

## Novel Insights

The observation that attention scores in GPT-2 are sparsely constructed in the SVD basis of Ω—with typically only 2–4 orthogonal slices needed to explain most of the score—is genuinely interesting and, if validated as a property specific to the SVD basis (rather than any orthogonal decomposition), could have significant implications for how we think about inter-head communication in transformers. The finding that local interventions sometimes have greater impact than global ones (suggesting downstream modification or redundancy) and that the traced graph reveals a "distinct lattice structure" at layers 7–9 (consistent with but more detailed than prior work on backup paths) provide novel structural insights into the IOI circuit.

## Suggestions

- **Add a null model comparison for sparsity**: Compare |S_ij| under the SVD basis vs. random orthonormal bases or vs. PCA of activations. Even a simple random rotation experiment would significantly strengthen the claim that the SVD basis is uniquely beneficial.

- **Compare against one standard circuit tracing method**: Run ACDC or activation patching on the same IOI task and report precision/recall against Wang et al.'s reference circuit alongside your own method's metrics.

- **Provide sensitivity analysis**: Show how the traced graph and precision/recall change as the 70% contribution threshold and 50% firing threshold vary.

- **Quantify and expand the interpretability analysis**: Go beyond the single name/non-name example. Show what multiple heads' signal subspaces encode (even via projection onto vocabulary space or simple clustering).

## Score and Decision

**Calibration**: Compared to papers with similar profiles:
- Sparse Feature Circuits (Oral, 8/8/8/8): Much more complete evaluation, broader scope, strong baselines. This paper is below that.
- Circuit Component Reuse (Spotlight, 8/6/6/6): Clearer empirical contribution with intervention experiments, but somewhat narrow IOI scope. This paper has comparable empirical scope but weaker baseline comparisons.
- Efficient Automated Circuit Discovery / CD-T (Poster, 8/6/5): Strong baseline comparisons, multiple tasks. This paper is below that in terms of evaluation rigor.
- Automatically Identifying Sparse Circuits (Rejected, 5/3/1/5/3): Had circular validation concerns and lacked baselines—this paper has similar concerns but a more novel core insight.
- Subspace Illusions paper (Poster, 8/3/8): Identified an important limitation with clear demonstration. This paper doesn't match that level of incisiveness.

The paper has a genuinely novel and mathematically clean core idea, compelling qualitative evidence (Figures 1–4), and meaningful causal validation (including random-subspace controls). However, the major weaknesses—the lack of comparison to alternative bases for the sparsity claim, no comparison to other circuit tracing methods, and the heuristic nature of the contribution measure—are significant enough to substantially limit confidence in the generality and superiority of the method. The paper makes a promising contribution to the mechanistic interpretability toolbox, but the evaluation is not yet at the level needed to establish its claimed advantages conclusively.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>