---
job_id: 3198a660-2170-4931-9dd3-0ff6dfea425f
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: abgW71sKIt.pdf
paper: Rethinking Output Alignment for 1-Bit Post-Training Quantization of Large Language Models
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper focuses on extreme low-bit post-training quantization of LLMs, output alignment objectives, and attention-preserving techniques, which fits squarely within ICLR’s core areas of representation learning, optimization, and efficient large-scale models.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments/Results, Conclusion) are present and reasonably detailed. The work is technically nontrivial, clearly about ML methodology, and supported by quantitative experiments, although there are significant weaknesses that affect its strength rather than its basic eligibility.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, instructions to reviewers, or suspicious formatting aiming to manipulate an automated review process.

---

# Expected Review Outcome:

## Summary

The paper studies why layer-wise output alignment objectives have performed poorly for 1‑bit post‑training quantization (PTQ) of large language models, despite being conceptually closer to the goal of matching full‑precision outputs than standard weight-alignment objectives. 

The authors analyze error accumulation and attention degradation under ARB‑X-style output alignment, then propose a modified output-error objective that mixes full‑precision inputs \(X\) and quantized inputs \(\hat X\), a row/column-scaled binary parameterization with closed-form updates, a gradient-sign masking scheme called Attention Matrix Preservation (AMP), and a selective strategy that applies output alignment only to the final fully-connected layer in each transformer block.

Experiments on OPT and LLaMA-2/3 models show modest perplexity and QA improvements over prior 1‑bit PTQ methods such as PB-LLM, BiLLM, ARB‑RC, and ARB‑X, with ablations for AMP and the choice of loss.

## Strengths

1. **Careful empirical diagnosis of output alignment failure**  
   The paper provides a reasonably thorough empirical analysis of why naive layer-wise output matching (Activation-conditioned Error) can hurt block-level or network-level behavior in 1‑bit PTQ.  
   - **Figure 1** (Page 4) is quite informative: by plotting which layers of LLaMA‑2‑7B yield lower *block-level* loss when quantized with ARB (weight alignment) versus ARB-X (layer-wise output alignment), it visually shows many layers where ARB-X wins at the *layer* loss but loses at the *block* loss. This nicely supports the claim that purely local layer-wise output matching can be misaligned with block-level performance.  
   - **Figure 2 (top)** shows that across blocks, ARB‑X keeps the cosine similarity high between \(\hat X W\) and \(\hat X \hat W\) (its internal target), yet MSE relative to the true full-precision outputs \(XW\) grows, illustrating error accumulation and target drift.

2. **Methodical incorporation of activation accumulation into the objective**  
   The proposed loss  
   \[
   \mathcal{L}(X,l)=\|XW-\hat X\hat W\|_F^2
   \]
   in **Equation (3)** replaces the ARB‑X objective \(\|\hat X W-\hat X\hat W\|_F^2\) and is analytically developed into closed-form update rules for \(\alpha_r,\alpha_c,B\) under the diag-binary parameterization (Equations (5)–(8), (19)). Even though the derivations mirror ARB‑RC/ARB‑X, the explicit use of cross Gram matrices \(S=\hat X^\top X\) is a meaningful tweak that more directly accounts for mismatch between full‑precision and quantized activations.

3. **Attention-aware perspective and token-similarity analysis**  
   The paper connects output alignment to attention degradation by measuring token similarity matrices and their drift:  
   - **Figure 2 (bottom)** plots block-wise MSE between token similarity matrices of quantized and full-precision models and shows clear divergence with depth, supporting the claim that naive output alignment can distort token interactions.  
   - Section 3.3 further formalizes “token-similarity error” and uses it as a proxy for attention pattern preservation. While somewhat heuristic, this is a nontrivial attempt to quantify a usually implicit phenomenon.

4. **Attention Matrix Preservation (AMP) as a targeted safeguard**  
   The AMP mechanism in Section 4.1 uses the gradient sign of an attention-similarity objective \(\mathcal{L}_{AMP}\) (Equation (9)) to define masks \(M^r, M^c, M^B\) that gate closed-form updates of \(\alpha_r,\alpha_c,B\) (Equation (11)).  
   - The ablation in **Table 3** (Page 9) convincingly shows that AMP has a strong effect on LLaMA‑2‑7B: enabling AMP reduces C4 perplexity from 29.12 to 19.25 and WikiText2 PPL from 26.24 to 15.42, which is a large and practically important difference. This is consistent with the paper’s argument that architectures with RMSNorm are more sensitive to directional distortions and thus benefit more from an attention-preserving mechanism.

5. **Empirical improvements over strong 1-bit PTQ baselines on several models**  
   On OPT and LLaMA models, the proposed method generally outperforms ARB‑RC and ARB‑X across several datasets:  
   - In **Table 1**, for OPT‑6.7B on C4, perplexity improves from 21.46 (ARB‑RC) and 22.54 (ARB‑X) to 19.90; on WikiText2 from 19.84 (ARB‑RC) and 20.07 (ARB‑X) to 18.25. The average zero-shot QA accuracy for OPT‑13B improves slightly from 55.01% (ARB‑RC) to 55.06% (ours) and for OPT‑30B from 57.11% (ARB‑RC) to 57.70%.  
   - In **Table 2**, LLaMA‑2‑7B PPL on C4 drops from 20.4 (ARB‑RC) and 28.02 (ARB‑X) to 19.25, and on WikiText2 from 16.25 (ARB‑RC) to 15.42. Similar, if modest, gains hold on LLaMA‑3‑8B.  
   These improvements, while often incremental, are consistent, and they show that carefully designed output alignment can indeed compete with and sometimes surpass weight-centric 1-bit PTQ methods.

6. **Selective layer strategy backed by ablation**  
   The paper does not simply assert that “final FC only” is better, it provides **Table 5** (Appendix, Page 16) showing PPL results when applying the proposed output alignment to each of the Q, K, V, attention-out, and final FC layers. For both LLaMA‑2‑7B and OPT‑6.7B, “Final FC” is clearly best, legitimizing the design choice in Section 4.2 to restrict output alignment to that layer.

7. **Closed-form optimization and overhead analysis**  
   The use of closed-form or least-squares solutions for \(\alpha_r, \alpha_c\) and row-wise binary updates for \(B\) keeps PTQ computationally efficient. **Table 6** in the appendix shows quantization times are comparable to ARB‑RC (slightly higher, but much lower than ARB‑X), and the text clarifies there is no additional inference-time or storage overhead. This makes the method practically attractive when compared to heavier 1‑bit schemes such as PB‑LLM or approaches that introduce extra parameters.

## Weaknesses

1. **Novelty over ARB‑RC / ARB‑X is incremental and somewhat narrow**  
   Conceptually, the main differences from ARB‑X/ARB‑RC are:  
   - Using \(XW - \hat X \hat W\) in the loss (Equation (3)) instead of \(\hat X W - \hat X \hat W\).  
   - Gating closed-form updates with an AMP mask defined via an attention-similarity objective.  
   - Restricting output alignment to the final FC layer.  
   All of these are meaningful tweaks, but they build very directly on ARB‑RC’s binary parameterization and ARB‑X’s data-aware loss and do not change the overall framework. The paper does not fully articulate why these modifications could not be straightforward extensions of ARB‑X, nor does it present a significantly new theoretical framing. Given the strong emphasis on “rethinking output alignment,” one would expect a deeper re-formulation beyond adjusting which activations appear in the loss and adding heuristic masks.

2. **Mathematical clarity and rigor issues in the core objective and derivations**  
   While the derivations in Appendix B are detailed, there are several inconsistencies and points of confusion that matter for understanding and potentially reproducing the method:

   - **Ambiguity around \(S\) and \(\hat S\)**: In Equation (2) on Page 5, the loss is written as  
     \[
     \mathcal{L}(X,l)=\|\hat X W - \hat X\hat W\|_F^2 = \operatorname{Tr}[(W-\hat W)^\top S(W-\hat W)]
     \]
     but the text defines \(\hat S = \hat X^\top \hat X\). Equation (2) uses \(S\), not \(\hat S\), which is confusing; the parameter definitions later use \(S = \hat X^\top X\). There is a lack of consistent indexing and naming between \(S, \hat S\), and cross Gram matrices. This is compounded in Equations (5)–(8), where both \(S\) and \(\hat S\) appear, but their precise definitions are relegated to brief mentions and are inconsistent with earlier notation. This makes it hard to verify the algebra, and it is easy to mis-implement.

   - **Equation (4) mixes per-layer and block-level notation**: It starts with \(\mathcal{L}(X,L) = \|f_Q(X)-f_{FP}(X)\|_F^2\) and then reduces to \(\|XW - \hat X\hat W\|_F^2\). It is unclear whether \(L\) denotes the network depth, the block index, or still the layer \(l\). The mapping between layer-wise and block-wise objectives is never made precise, which weakens the claim that the method is “block-level output matching”.

   - **Assumptions behind closed-form updates**: The derivation for \(\alpha_r\) leads to a linear system  
     \[
     (\hat S\odot C)\alpha_r = \mathrm{Diag}(S W \operatorname{diag}(\alpha_c)B^\top)
     \]
     and the text proposes to solve this via a pseudoinverse or least-squares solver. Yet, the conditioning of \(\hat S\odot C\), its dimensionality, and whether it is well-posed in typical LLM settings (where Gram matrices can be highly collinear) are not analyzed. There is no discussion of regularization or how degeneracy impacts quantization quality. This is important, because the method depends critically on these closed-form updates.

   - **AMP gradient masks lack formal derivation**: Equations (9)–(11) define AMP by taking the sign of the gradient of \(\mathcal{L}_{AMP}\) w.r.t. \(\alpha_r,\alpha_c,B\), but the gradients themselves are not derived, and it is not rigorously argued why a sign mask (rather than e.g. a real-valued weight or trust-region style constraint) is appropriate for preserving attention. This makes AMP feel heuristic rather than a principled optimization component.

   Overall, the math contains no obvious contradiction, but the notational sloppiness and missing rigor reduce confidence in the theoretical underpinnings and reproducibility.

3. **Theoretical framing of “block-level loss reduction” remains qualitative**  
   The paper’s motivation in Section 3.1 and Section 4 is that layer-wise output matching does not guarantee block-level loss reduction, and that their selective strategy addresses this. However, there is no formal argument that the proposed objective or layer selection *does* reduce block-level loss, even in an approximate sense.  
   - **Figure 1** nicely shows empirically that ARB‑X can increase block loss, but there is no counterpart figure demonstrating that the proposed method consistently *improves* block loss when applied only to final FC layers, compared to ARB‑RC or ARB‑X.  
   - The selective strategy is mostly justified by **Table 5** (which is about final PPL, not intermediate block losses), so the “block-level” language in Section 4 slightly oversells what is empirically substantiated.

4. **Attention/similarity analysis and AMP objective are only loosely tied to actual self-attention**  
   The work uses token similarity matrices \(\hat X \hat W \hat W^\top \hat X^\top\) and \(X W W^\top X^\top\) as proxies for “attention masks” and defines AMP via the trace of their elementwise product (Equation (9)). There are several conceptual issues:

   - Real self-attention depends on Q,K,V projections, softmax, and subsequent transformations. The similarity matrices in Section 3.3 are purely feature-space cosine similarities after one linear layer, not the actual attention scores or probabilities. **Figure 2 (bottom)** shows divergence in these proxy matrices, but it is unclear how strongly this correlates with degradation in attention distributions.

   - AMP’s objective \(\mathcal{L}_{AMP}\) is maximized via gating but never incorporated as an explicit regularizer in the main optimization objective. There is no ablation that compares “Output Error + AMP penalty” vs. simply using AMP masks. As a result, it is hard to tell whether the big PPL improvements in Table 3 are due to a fundamentally sound “attention preservation” idea or just an ad hoc masking that prevents some harmful closed-form updates.

   - **Figure 3** in the appendix (Page 15) shows block-wise MSE between quantized and full-precision attention scores “with and without AMP”, but the blue (AMP) curve is actually slightly *higher* than “No AMP” over most layers except the last, which looks contradictory to the narrative that AMP reduces attention degradation. The text claims AMP reduces degradation, yet the plotted MSE curve for AMP is generally above or comparable to No AMP over depth, and only slightly below at the last layer. This discrepancy is not discussed.

   These gaps undermine the conceptual story around AMP and its claimed connection to attention.

5. **Experimental scope and baselines, while standard, miss several directly related recent works**  
   The experiments compare only against PB‑LLM, BiLLM, and ARB‑LLM variants ARB‑RC/ARB‑X. Given the fast-moving literature in extreme low-bit PTQ for LLMs, key recent methods are missing both from the Related Work and from the experimental comparison. In particular, the paper does not mention or compare to:

   - NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models (Chong et al., 2026)  
   - PT-BitNet: Scaling up the 1-Bit Large Language Model with Post-Training Quantization (Guo et al., 2025)  
   - PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models (Zhao et al., 2025)  
   - VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models (Liu et al., 2024)  

   These works are directly in the same niche (extreme/1‑bit PTQ for LLMs), often with attention to error accumulation and activation-aware objectives. Not positioning the proposed method relative to them makes it hard to judge its significance and could understate prior art.

6. **Perplexity results on PTB for LLaMA models are extremely high and inadequately addressed**  
   In **Table 2**, PTB perplexities for LLaMA‑2‑7B are:  
   - Full precision: 37.91  
   - ARB‑RC: 763.19  
   - ARB‑X: 681.24  
   - Ours: 3166  

   Even for LLaMA‑2‑13B and LLaMA‑3‑8B, PPLs are in the hundreds. The paper states that “the large perplexity indicates that the metric cannot provide a meaningful evaluation,” but this is not convincing: the fact that all PTQ methods catastrophically fail on PTB suggests that (a) the quantization procedure, calibration data, or evaluation pipeline is heavily mismatched to PTB or (b) there is a serious robustness issue for this dataset. Simply dismissing these results without deeper investigation, alternative metrics, or a sanity check (e.g., unquantized model with the same pre-processing pipeline) is unsatisfying and leaves a question mark on the generality of the method.

7. **Improvements over ARB‑RC are often small relative to variance and may not justify complexity**  
   While the paper highlights reductions like “up to 4.85 PPL” in some challenging settings, many gains are relatively minor:  
   - On OPT‑13B C4, PPL improves from 15.07 (ARB‑RC) to 14.71 (ours), and QA accuracy from 55.01 to 55.06 (**Table 1**).  
   - On OPT‑30B C4, 13.34 to 13.15; WikiText2 11.19 to 10.94.  
   - On LLaMA‑2‑7B C4, 20.4 to 19.25; on WikiText2, 16.25 to 15.42 (**Table 2**).  
   No confidence intervals, multiple runs, or seeds are reported, so it is impossible to know if these improvements exceed run-to-run variation. Given that the method introduces extra steps (AMP masks, solving systems for \(\alpha_r\), etc.), the practical cost-benefit trade-off is not fully quantified.

8. **Design choices are partly heuristic and not fully explored**  
   Several design decisions feel ad hoc or underexplored:

   - The decision to apply output alignment only to the final FC layer is motivated by ablations in **Table 5**, but that table only considers *one layer at a time*, not combinations. It could be that targeting some subset of Q/K/V plus final FC is even better.  
   - The hyperparameter \(k\) controlling when to update \(\alpha_c\) and \(B\) is only briefly ablated in **Table 8**, and performance can be non-monotonic (e.g., for LLaMA‑2‑7B, \(k=10\) worsens PPL compared to \(k=5\)). The method seems sensitive, but there is no discussion of how to select \(k\) or general guidelines.  
   - AMP masks are hard-thresholded sign functions with no annealing or magnitude dependence, which may be brittle. Alternative designs (e.g., soft masks or trust-region constraints) are not considered.

9. **Clarity and presentation issues**  
   While the writing is generally understandable, there are several clarity problems:

   - Multiple typos and duplicated references (e.g., Bisk et al. 2020a/2020b) and minor copy-paste issues (Algorithm 1 line 3–4 uses the same symbol \(S\) twice for different Gram matrices).  
   - The method section sometimes oscillates between per-layer and whole-network notation without clear separation, which makes it difficult for readers to reconstruct the algorithm from the equations alone, despite the presence of Algorithms 1–2.  
   - Some key implementation details are hidden in the appendix or not specified in the main text (e.g., calibration dataset size, whether the same C4 subset is reused across all experiments, exact normalization for token similarity).

Altogether, these issues do not make the paper invalid, but they significantly weaken its impact and clarity.

## Potentially Missing Related Work

The following directly related works are neither cited nor discussed and should be incorporated:

1. **Chong, H., Kim, D., Kim, C. (2026), “NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models”**  
   - Relevance: Proposes PTQ schemes for binary and sub-1‑bit LLMs, directly targeting the same problem setting.  
   - Recommendation: Should be cited and compared in Section 2 (“1-Bit Quantization for Language Models”) and mentioned in the experiments as a baseline or at least discussed as a contemporary alternative that also addresses extreme quantization.

2. **Guo, Y., Hao, Z., Shao, J. (2025), “PT-BitNet: Scaling up the 1-Bit Large Language Model with Post-Training Quantization”**  
   - Relevance: Post-training 1‑bit quantization aimed at scaling LLMs, conceptually very close to this paper’s goal and likely to have overlapping technical components.  
   - Recommendation: Add to Section 2 and discuss differences in objectives (e.g., weight vs output alignment), calibration, and whether PT-BitNet addresses error accumulation.

3. **Zhao, J., Zhang, M., Wang, M. (2025), “PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models”**  
   - Relevance: Focuses on extremely low-bit PTQ (including near-1‑bit regimes) and may discuss similar issues like accumulated error and activation-aware schemes.  
   - Recommendation: Include in the Related Works on LLM quantization and, ideally, provide empirical comparison or at least a reason why direct comparison is not feasible.

4. **Liu, Y., Wen, J., Wang, Y. (2024), “VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models”**  
   - Relevance: Uses vector quantization for extremely low-bit settings and is part of the same ecosystem of techniques that push PTQ beyond 4–8 bits.  
   - Recommendation: Discuss VPTQ in Section 2 as an alternative line of work (vector PTQ vs binary scalar PTQ) and situate the proposed method within this broader landscape.

Incorporating and contrasting with these works would significantly strengthen the paper’s positioning.

## Questions

1. **On the correlation between token similarity proxies and actual attention**  
   Can you provide quantitative evidence that your token similarity matrices (Section 3.3) correlate strongly with changes in actual attention maps (QKᵀ-softmax) and/or downstream task performance? For example, could you compute the cosine similarity between full-precision and quantized attention matrices and compare that correlation with token-similarity error, to justify AMP’s design more directly?

2. **Clarification of the Gram matrices and cross-term definitions**  
   Please clarify and standardize the definitions of \(S\), \(\hat S\), and any cross Gram matrices used in Equations (2), (5)–(8). In particular:  
   - Is \(S = X^\top X\), \(\hat S = \hat X^\top \hat X\), and is there also a cross-activation matrix \(C = \hat X^\top X\)?  
   - Could you re-derive Equations (5)–(8) in the main text with explicit shapes and definitions, so that readers can verify the algebra without toggling between inconsistent notation?

3. **Behavior on PTB and robustness to dataset shift**  
   The PTB results for LLaMA models are extremely poor (e.g., PPL > 3000). Can you diagnose the cause more concretely (e.g., is there a mismatch in tokenization, sequence length, or calibration data distribution)? Would using PTB-like text as calibration data improve things, and if so, by how much? Alternatively, could you report a different metric (e.g., negative log-likelihood after filtering out OOV tokens) or sanity-check that the full-precision PPL is indeed ~38/51 with your pipeline?

4. **Statistical significance of improvements over ARB‑RC**  
   Have you run multiple seeds or calibrations to estimate the variability of PPL and QA accuracy? Many of your gains over ARB‑RC are in the 0.2–1.0 PPL range and <1% QA. Reporting standard deviations or confidence intervals would help determine whether these differences are robust.

5. **AMP design choices**  
   What motivated using the sign of the AMP gradient as a hard mask, instead of, say, scaling updates by a continuous factor based on gradient magnitude or introducing a combined loss \(\mathcal{L}=\mathcal{L}_{out} - \lambda\mathcal{L}_{AMP}\)? Have you experimented with softer variants, and if so, how do they compare?

6. **Applicability beyond transformer LLMs and RMSNorm architectures**  
   Your attention analysis and AMP are framed in the context of LLaMA-like architectures with RMSNorm. Do you expect similar benefits for GPT-style models with LayerNorm, or for non-language transformer architectures (e.g., vision transformers)? If you have any preliminary evidence (even small-scale), it would help clarify the generality of the method.

7. **Layer selection strategy beyond “Final FC only”**  
   Table 5 tests one layer at a time. Did you experiment with applying output alignment on subsets of layers (e.g., Q+K, Q+V+final FC) within each block? If so, can you report those results or explain why they were discarded? This would strengthen the argument that final FC is indeed the optimal location for output matching.

Author responses that convincingly address these points, particularly clarifying the math, explaining PTB behavior, and situating the method against the missing related work, would significantly increase my confidence in the contribution.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating
2: fair.  
The method is plausible and empirically effective in several settings, but there are notable gaps in mathematical clarity, limited analysis of numerical stability for the closed-form updates, and unresolved anomalies in the PTB experiments.

## Presentation Rating
2: fair.  
The core ideas are explainable and the figures/tables are informative, but notation is inconsistent, some derivations are hard to follow, several important implementation details and design justifications are under-specified, and related work coverage is incomplete.

## Contribution Rating
2: fair.  
The work offers a useful but incremental improvement over ARB‑RC/ARB‑X with some insightful diagnostics of output alignment, yet novelty is modest and comparison to contemporary 1‑bit PTQ methods is incomplete, limiting its overall impact.

## Overall Rating
4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper presents a thoughtful empirical analysis of 1‑bit output alignment, a reasonable modification to the objective that accounts for activation mismatch, and a practical attention-aware masking mechanism that yields consistent but mostly moderate gains over prior ARB-based methods. However, novelty is limited, the mathematical exposition has clarity issues, key recent baselines are missing, and some experimental results (especially on PTB) raise unresolved concerns. With stronger positioning, clearer math, and a more comprehensive and robust experimental evaluation, this work could reach ICLR standards.

## Reviewer Confidence
4: confident.  
I am familiar with LLM quantization literature (including ARB, GPTQ, BiLLM, PB‑LLM) and have gone through the equations and experimental sections carefully, though the notational inconsistencies limit complete verification of every derivation.