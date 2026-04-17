---
job_id: fea2e7e1-7531-426e-8a4e-c197b788ccd1
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: wUzBBsrdB1.pdf
paper: Sparse But Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is about sparse autoencoders for LLM representations, sparsity hyperparameters, and interpretability metrics, which fits squarely within representation learning and interpretability topics at ICLR.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Background/Methodology, Experiments, Results/Analysis, Related Work, Discussion, Reproducibility statement) are present. The work is technically nontrivial, clearly written, and supported by both toy and LLM experiments; I do not see fatal methodological, theoretical, or empirical flaws at the level that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no signs of prompt injection, hidden instructions to reviewers, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper investigates how the activation sparsity hyperparameter \(L_0\) in sparse autoencoders (SAEs) affects whether the learned latents correspond to underlying “true” features under the Linear Representation Hypothesis. Using controlled toy models with known ground-truth features and experiments on Gemma-2-2B and Llama-3.2-1B, the authors show that if \(L_0\) is below or above the “true” average number of active features, the SAE systematically mixes correlated and anti‑correlated features into single latents, undermining monosemanticity. They further propose decoder-based metrics, primarily the decoder pairwise cosine similarity \(c_{\mathrm{dec}}\), as a proxy to locate a reasonable \(L_0\); they show that in toy models this metric is minimized at the true \(L_0\), and in LLMs its “elbow” coincides with peak sparse probing performance.

## Strengths

1. **Clear and focused conceptual contribution around a widely used hyperparameter.**  
   The central claim, that \(L_0\) is not a benign sparsity knob but must roughly match the data’s “true” feature sparsity for SAEs to recover meaningful features, is clearly articulated and repeatedly supported. This is particularly valuable given that many recent interpretability efforts treat “sparsity–reconstruction tradeoffs” as the primary evaluation tool.

2. **Carefully constructed toy models with ground-truth structure.**  
   The toy models in Section 3 are well specified: features \(f_i\) are orthogonal, have tunable firing probabilities \(p_i\), magnitudes \(\mu_i, \sigma_i\), and a controlled Bernoulli correlation matrix \(C\). The authors can hence construct a “ground-truth SAE” with \(\mathbf{W}_{\text{enc}} = \mathbf{F}^T, \mathbf{W}_{\text{dec}} = \mathbf{F}\), and directly inspect when trained SAEs deviate from this solution.

   - **Figure 2** and **Figure 3** are especially compelling: for the 5‑feature toy model, they show correlation matrices (left) and heatmaps of decoder–true‑feature cosine similarities (middle/right) for \(L_0 = 2\) (correct) vs \(L_0 = 1.8\) (too low). When \(L_0\) is correct, the heatmap is essentially an identity matrix; when \(L_0\) is too low, the latents for \(f_1\)–\(f_4\) acquire a consistent positive/negative component of \(f_0\), visually demonstrating feature mixing aligned with the sign of correlation.

3. **Sharp critique of sparsity–reconstruction plots using ground-truth counterexample.**  
   The analysis in Section 3.4 and **Figure 4** is important. By plotting variance explained vs \(L_0\) for (i) the trained SAEs and (ii) the ground-truth SAE with fixed dictionary but varying \(L_0\), they show that for \(L_0\) below the true sparsity, the *incorrect* trained SAE with mixed features has *better* reconstruction than the perfectly disentangled ground-truth SAE. **Figure 5** then shows the cosine similarity matrices at \(L_0 = 1\) and \(L_0=5\) for the trained SAEs, which look dramatically more polysemantic than the ground-truth SAE, despite outperforming it on variance explained. This is a strong, concrete argument against using reconstruction curves as an interpretability proxy.

4. **Simple and cheap decoder-based metrics with some theoretical backing.**  
   The decoder pairwise cosine similarity metric \(c_{\mathrm{dec}}\) (Equation (4), Page 6) is conceptually straightforward and cheap to compute after training. The authors provide toy‑model evidence that it is minimized near the true \(L_0\) (see **Figure 6**) and they give a clean theoretical justification (Theorem 2, Appendix A.6), where they explicitly model decoder vectors as mixtures \(\sqrt{1-\gamma_i^2}\, \mathbf{f}_i + \gamma_i \mathbf{g}\) and prove that any shared mixed component \(\mathbf{g}\) across latents increases average \(|\cos|\). The alternative metric \(s_n^{\text{dec}}\) (Appendix A.9, Equation (58)–(59), **Figure 17–18**) is also well motivated and is backed by Theorem 3 (Appendix A.10).

5. **Nontrivial LLM‑scale experiments linking metrics to sparse probing performance.**  
   Section 4 runs 32k‑latent BatchTopK SAEs on Gemma‑2‑2B and Llama‑3.2‑1B at various \(L_0\) values using SAELens, and evaluates both \(c_{\mathrm{dec}}\) and K‑sparse probing F1 (Kantamneni et al., 2025).  
   - In **Figure 8**, for Gemma layer 5 and Llama layer 7, the “elbow” in \(c_{\mathrm{dec}}\) just before a sharp rise at low \(L_0\) aligns with peak sparse probing F1.  
   - **Figure 9 (left)** for Gemma layer 12 shows this alignment for both BatchTopK and JumpReLU SAEs. These figures make the case that decoder metrics are not just a toy‑world curiosity but correlate with a task‑based interpretability proxy on real models.

6. **Nuanced comparison of JumpReLU vs BatchTopK SAE behavior.**  
   Section 3.6 and Section 4.1, plus Appendix A.16, provide a fairly nuanced analysis:  
   - **Figure 7** shows that \(L_0\) in JumpReLU SAEs is relatively insensitive to \(\lambda_s\) and tends to “stick” near the true sparsity even when \(\lambda_s\) varies; decoder similarity curves are minimized around the true toy \(L_0\).  
   - **Figure 9**, **Figure 26**, and **Figure 27** illustrate that JumpReLU achieves lower pairwise cosine similarity at high \(L_0\) and better sparse probing performance compared to BatchTopK, and that per‑latent thresholds adapt more flexibly. This is practically useful guidance for SAE practitioners.

7. **Useful practical recommendations and empirical survey of common practice.**  
   The discussion in Section 6, combined with the histogram of open‑source SAEs’ \(L_0\) values in **Figure 22**, makes a strong case that a large share of current SAEs use \(L_0\) that is likely too low (often <100 for large models), which the earlier analyses suggest leads to systematic feature mixing. This is actionable for practitioners.

## Weaknesses

1. **“Correct \(L_0\)” is conceptually fragile and not well characterized in real LLM settings.**  
   The paper treats the true \(L_0\) as “how many features fire on average”, which is well defined in the synthetic toy models but much murkier in real LLMs where:
   - Features may not be strictly linear (see Engels et al., 2025, which the authors cite) and can be hierarchical or compositional.
   - The concept of “feature” itself depends on the chosen SAE width, architecture, and even SAE training objective.  
   In Section 4, the authors fall back to “the elbow of \(c_{\mathrm{dec}}\) that aligns with peak sparse probing performance” as an operational definition. This risks circularity: the metric is validated against probing, which itself is an imperfect and task‑specific proxy for “true” features. The paper would benefit from an explicit acknowledgment and formal definition of what “correct \(L_0\)” means for non‑toy data and a clearer discussion of how much slack is acceptable (e.g., within 2×? 10%?).

2. **Limited experimental breadth on real LLMs and absence of quantitative tables.**  
   All LLM experiments focus on only two models (Gemma‑2‑2B, Llama‑3.2‑1B) and only a handful of layers (Gemma layers 5, 12, 20 and Llama layer 7). It is unclear whether the observed elbow behavior of \(c_{\mathrm{dec}}\) and its alignment with sparse probing generalizes:
   - Across deeper transformer layers or MLP vs attention sublayers;
   - Across significantly larger models (e.g. >7B) or different architectures (Mixture‑of‑Experts, decoder‑only vs encoder‑decoder).  
   Moreover, there are no results tables quantifying, for example, exact sparse probing F1 at each \(L_0\) alongside \(c_{\mathrm{dec}}\). Everything is in curves. A simple table (e.g., rows = \(L_0\), columns = mean F1, std, \(c_{\mathrm{dec}}\)) for at least one representative layer would make the empirical link between the metric and task performance much more concrete and allow easier comparison to future work.

3. **Theoretical analysis is compelling in a toy regime but narrow and assumes quite special structure.**  
   Theorem 1 (Appendix A.5) proves that with two orthonormal features \(\mathbf{f}_1, \mathbf{f}_2\) and a tied SAE with \(L_0=1\), the MSE‑optimal solution mixes the two features when co‑occurrence is frequent enough. This is pedagogically nice, but:
   - It assumes orthonormality, a tied encoder/decoder, exactly two features, no biases, and Top‑1 selection driven solely by inner product magnitude.  
   - Real SAEs use overcomplete dictionaries (\(h \gg d\)), non‑tied encoders, biases, and complex learned thresholds; LLM activations have heavy‑tailed and structured distributions.  
   There is no extension of the theorem to, say, three or more correlated features or to overcomplete regimes, and the paper gives no conditions under which feature mixing is *inevitable* versus just empirically common. A partial generalization (even in linearized form) would greatly strengthen the central claim that low \(L_0\) inherently incentivizes mixing, not just in an idealized corner case.

   Concretely, the derivation around Equations (25)–(31) chooses specific probabilities \(P_1, P_{12}\) and magnitudes \(m_1, m_2\) to show an example where \(\mathbb{E}[\mathcal{L}(\alpha=0.6)] < \mathbb{E}[\mathcal{L}(\alpha=1)]\). However, the paper never characterizes the full region in \((P_1, P_2, P_{12})\)-space where mixing is preferable, nor does it bound how much mixing (\(\alpha^*\)) is optimal as a function of these parameters. This makes it hard to judge how robust the phenomenon is beyond the illustrated example.

4. **Decoder metrics are largely proxy signals, not predictive tools, and their limitations are under‑explored.**  
   The paper is quite honest in Section 6 that \(c_{\mathrm{dec}}\) is not a perfect guide and can be flat over wide ranges of \(L_0\). However, there are several underexplored issues:
   - **Shape variability:** **Figure 8 (left vs right)** and **Figure 9 (left)** show notably different shapes for \(c_{\mathrm{dec}}\) at high \(L_0\) across layers and architectures. Sometimes the global minimum is in a shallow region, sometimes it is more pronounced. The paper does not analyze why this happens or what it implies about the metric’s robustness.
   - **Sensitivity to dictionary normalization and width:** The metrics assume normalized decoders (Appendix A.7) and fixed width \(h = 32768\). It is unclear how \(c_{\mathrm{dec}}\) behaves when widths differ or when normalization is imperfect, especially given superposition noise. Since \(c_{\mathrm{dec}}\) essentially averages \(|\cos|\) over \(\binom{h}{2}\) pairs, the scale and distribution over dictionary norms matter.  
   - **Over‑reliance on sparse probing as validation:** All validation is via K‑sparse probing F1. No human interpretability assessments or other downstream tasks are used to confirm that the “good” \(L_0\) suggested by the metric actually yields more interpretable concepts.

5. **Metric‑based automatic \(L_0\) tuning is not convincingly demonstrated.**  
   Appendix A.11 describes a meta‑optimization scheme that uses \(s_n^{\text{dec}}\) to adjust \(L_0\) during training, including gradient estimation, biasing, and an Adam optimizer. The authors themselves admit it “requires a lot of hyper‑parameter tuning” and “works very well in toy models, but … limited utility” in real LLMs. There is no main‑paper experiment showing that such a procedure can reliably converge to a reasonable \(L_0\) on LLMs. As a result, the proposed metrics are mainly *post hoc diagnostics*; they do not yet solve what practitioners actually want (an automatic way to choose \(L_0\) without sweeping). The title and abstract could be clearer that the paper offers guidance and diagnostics, not a truly automated method.

6. **Connection to broader SAE literature on robustness and geometry is under‑developed.**  
   The paper’s Related Work section focuses on a handful of interpretability‑oriented SAE papers, feature hedging/absorption, and MDL‑SAEs/AFA. However, there is now a small but growing body of work on robustness/illusions of SAE interpretability, geometric structure of SAE features, and general SAE methodology and theory. These are not cited or discussed, which weakens the positioning of this work (details in “Potentially Missing Related Work”).

7. **Some mathematical and conceptual details are under‑specified or confusing.**  
   A few concrete points where clarity should be improved:

   - **Definition of \(L_0\) for BatchTopK vs JumpReLU.** In Section 2, \(L_0\) is defined as average number of active latents per input, but for BatchTopK SAEs, sparsity is per-batch and can vary per example. The paper later reports fractional \(L_0\) values (e.g., 1.8). It would be helpful to specify precisely how \(L_0\) is measured during and after training for each architecture (e.g., average non‑zeros in \(\mathbf{a}\) over a validation set, per‑token vs per‑sequence).  
   - **Equation (1) and decoder bias.** The encoder subtracts \(\mathbf{b}_{\text{dec}}\) from \(\mathbf{x}\) before applying \(\mathbf{W}_{\text{enc}}\). This is a common centering trick, but the motivation is not clearly explained, and it interacts with the projection‑based metrics that also subtract \(\mathbf{b}_{\text{dec}}\). Some discussion of how sensitive \(c_{\mathrm{dec}}\) and \(s_n^{\text{dec}}\) are to this bias choice would be useful.  
   - **Choice of \(n\) in \(s_n^{\text{dec}}\).** In Appendix A.9 the authors recommend \(n<h/2\) and empirically find \(n \approx h/2\) works best (Appendix A.15, **Figures 23–25**). However, the exact tradeoff between \(n\), batch size, and dimensionality remains quite heuristic. A brief rule‑of‑thumb (e.g., “for 32k latents we found \(n\in[8k, 20k]\) robust across layers/models”) in the main text would make this metric more practically usable.

8. **No qualitative feature examples to connect metric behavior to interpretability.**  
   All the evidence of “mixing” in LLMs is indirect through decoder metrics and sparse probing scores. There are no qualitative visualizations of individual SAEs’ features (e.g., top‑activating tokens, neuronpedia‑style feature names) at low vs near‑optimal vs high \(L_0\) to show that:
   - Low \(L_0\) features indeed have bizarre mixtures of anti‑correlated concepts, in analogy with Figure 2/3 toy examples.
   - Near‑optimal \(L_0\) features look more coherent or monosemantic.  
   Even one or two side‑by‑side qualitative comparisons would make the “sparse but wrong” story more tangible.

9. **Lack of systematic analysis of the “too high \(L_0\)” regime.**  
   The paper asserts that \(L_0\) can also be “too high”, producing degenerate solutions that mix features (e.g., large toy model in Section 3.2 and **Figure 1**, right panel). There is some interesting analysis in Section 4.2 and **Figure 9 (right)** about decoder projection histograms at L0=10, 200, 750, 2000, suggesting that at intermediate high \(L_0\) some latents are too active and others too inactive. However:
   - There is no analog of Theorem 1 for the high‑\(L_0\) case showing that, even without sparsity pressure, reconstruction plus auxiliary losses naturally induce mixing.  
   - The boundary where “too high” begins is not characterized, and the practical consequences (for probing, steering, or human interpretability) are mostly anecdotal. This makes the “too high” half of the story less developed than the “too low” half.

## Potentially Missing Related Work

1. **Li et al., “Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations” (2025).**  
   This paper analyzes failures of SAE‑based interpretability under adversarial perturbations, which is highly relevant to the paper’s theme that SAEs can be “sparse but wrong”. It should be cited in the Related Work section and discussed as complementary evidence that apparent monosemantic features can be misleading even when reconstruction looks good.

2. **O’Neill et al., “Disentangling Dense Embeddings with Sparse Autoencoders” (2024).**  
   This work evaluates how well SAEs disentangle semantic concepts from dense text embeddings. It seems directly related to the authors’ claim that mis‑set \(L_0\) harms feature disentanglement. It would be appropriate to reference it in Section 2 or 5 and compare whether their results might already implicitly reflect low‑\(L_0\) problems.

3. **Korznikov et al., “The Geometry of Concepts: Sparse Autoencoder Feature Structure” (2025).**  
   This study explores the geometric organization of SAE features. It aligns closely with the present paper’s geometric interpretation of feature mixing and the use of cosine similarities in decoder space (e.g., Theorem 2). It should be mentioned in Related Work and linked to the proposed \(c_{\mathrm{dec}}\) metric.

4. **Makhzani & Frey, “k-Sparse Autoencoders” (2013).**  
   A classic paper that introduces \(k\)-sparse autoencoders and discusses the relationship between sparsity constraints and learned representations. Given that the present work criticizes naive usage of “sparsity–reconstruction tradeoffs”, citing this foundational work in Section 2 would better situate the analysis within the broader history of sparsity‑constrained autoencoders.

5. **Chung et al., “Sparse Autoencoder Methodology” (2025).**  
   This paper discusses implementation and methodological choices for SAEs, including sparsity mechanisms. It should be included in Related Work to give a broader methodological context and to contrast their hyperparameter recommendations (if any) with the current work’s arguments about \(L_0\).

6. **Gille et al., “Sparse Autoencoder Models” (2025).**  
   Presents different SAE model variants and applications. Including it in the Related Work section will provide a more complete picture of the space of SAE model designs within which the current results about \(L_0\) sit.

7. **Stevens et al., “Sparse Autoencoders: Theory and Practice” (2025).**  
   This paper offers an overview of SAE theory and practice, including hyperparameter tuning. It would be useful to cite and contrast with the current work’s emphasis that \(L_0\) is not a “free” hyperparameter but has a correctness notion.

8. **Chanin & Garriga‑Alonso, “L0 is not a neutral hyperparameter” (2025).**  
   A closely related article by (some of) the same authors, apparently elaborating the same theme. It should be referenced in Related Work or the Discussion to acknowledge overlapping ideas and to clarify what is new here (e.g., more thorough toy experiments, formalized metrics \(c_{\mathrm{dec}}\) and \(s_n^{\text{dec}}\), LLM‑scale validation).

## Questions

1. **Operational definition and tolerance of “correct \(L_0\)” in LLMs.**  
   Can the authors more precisely define what they consider the “correct \(L_0\)” for an LLM layer, beyond “roughly where \(c_{\mathrm{dec}}\) has an elbow and sparse probing F1 peaks”? In particular:
   - How sensitive are their conclusions if one chooses an \(L_0\) that is, say, 0.5× or 2× the elbow point?  
   - Could they provide a small ablation (perhaps as a table) showing sparse probing and \(c_{\mathrm{dec}}\) within a ±50% window of the chosen \(L_0\) to quantify how sharp or forgiving the optimum is?

2. **Extent of feature mixing in high‑\(L_0\) regimes.**  
   For the LLM experiments (e.g., Gemma layer 12 in **Figure 9**), do the authors observe qualitatively different kinds of feature mixing at \(L_0\) ≫ optimal compared to \(L_0\) ≪ optimal? For example, can they show concrete examples of features at L0=750 and L0=2000 where top‑activating tokens reveal systematic “multi‑concept” mixtures?

3. **Generalization of Theorem 1.**  
   Are the authors able to extend Theorem 1 beyond the 2‑feature, tied‑decoder, \(L_0=1\) case? For instance:
   - What happens with three orthogonal features \(\mathbf{f}_1,\mathbf{f}_2,\mathbf{f}_3\) where \(\mathbf{f}_1\) is positively correlated with both \(\mathbf{f}_2\) and \(\mathbf{f}_3\)?  
   - Can they characterize, even qualitatively, when the MSE optimum will mix *all three* vs mixing only the frequent “hub” feature?  
   A sketch of such an extension would substantially increase my confidence that the phenomenon is robust.

4. **Alternative evaluation beyond sparse probing.**  
   Have the authors attempted any human‑in‑the‑loop interpretability evaluations (e.g., Neuronpedia‑style labeling) across different \(L_0\) settings, or other downstream tasks (e.g., steering, concept editing) to assess whether the suggested \(L_0\) indeed yields better behavior? Even a small‑scale pilot might help decouple “good for sparse probing” from “good for interpretability”.

5. **Automatic \(L_0\) selection in practice.**  
   Given the difficulties described in Appendix A.11, do the authors have a practical recipe they would recommend today (e.g., sweep over a coarse grid of \(L_0\), choose the smallest \(L_0\) beyond the elbow in \(c_{\mathrm{dec}}\), then maybe refine)? It would be helpful if Section 6 explicitly summarized a minimal recommended protocol for practitioners.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The toy‑model setup and analyses are solid and carefully checked; the decoder metrics are theoretically motivated with clean proofs in simplified regimes. The main limitations are the narrow theoretical assumptions and limited breadth of real‑world validation, not obvious logical or methodological errors.

## Presentation Rating

3: good.  
The paper is generally well written, with clear figures (especially Figures 2–6, 8–9) and detailed appendices. A few definitions (e.g., \(L_0\) measurement, choice of \(n\) in \(s_n^{\text{dec}}\)) could be more explicit in the main text, and the Related Work could be broadened.

## Contribution Rating

3: good.  
The work delivers a clear conceptual message (low \(L_0\) can make SAEs “sparse but wrong”), a well‑motivated critique of standard evaluation practice, and practically useful decoder‑space diagnostics with some theoretical support. It is not a complete solution to automatic hyperparameter selection or a broad new algorithm, but it meaningfully advances understanding of SAEs.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper makes a substantive and timely contribution to the SAE interpretability literature by clearly demonstrating, both theoretically (in simplified settings) and empirically, that mis‑setting \(L_0\) leads to systematic feature mixing and that standard sparsity–reconstruction plots can actively favor incorrect dictionaries. The decoder‑based metrics are simple, grounded in geometry, and show encouraging correlation with sparse probing performance on real LLMs. The main weaknesses are the narrow theoretical scope, limited experimental breadth on LLMs, and the fact that the metrics are diagnostic rather than a truly automated solution. On balance, the strengths and practical relevance justify a positive but not glowing recommendation.

## Reviewer Confidence

4: confident.  
I am familiar with SAE‑based interpretability and sparse coding, carefully followed the math in the appendices, and see no major gaps in my understanding, though I have not independently reimplemented the experiments.