---
job_id: 8a0dceb3-7d8e-4479-befb-666185efe199
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: pNpnqsn0Si.pdf
paper: Thoughtbubbles: An Unsupervised Method for Parallel Thinking in Latent Space
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new transformer variant for adaptive parallel computation in language modeling, evaluated on standard LM benchmarks, which fits squarely within ICLR’s core topics of representation learning, large-scale language models, and architectural optimization.

## Minimum Quality
Pass ✅.  
The paper is in English and includes all required scientific sections: Abstract, Introduction, Method (Sections 2.x), Experiments / Results (Sections 3–5, Table 1, Figures 3–7), Related Work (Section 6), Conclusion (Section 7), plus limitations and appendices. The methodology is technically detailed, experiments are non-trivial on two corpora across three scales with several baselines, and I do not see any single fatal theoretical or empirical flaw that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not observe any hidden text, meta-instructions, or attempts to manipulate automated reviewing systems within the main paper content.

---

# Expected Review Outcome:

## Summary

The paper introduces **Thoughtbubbles**, a decoder-only transformer architecture that performs *latent* adaptive computation by dynamically forking and deleting residual streams during forward passes. Each residual stream carries a cumulative scalar score that is used (i) in a top‑k forking/deletion mechanism subject to a budget \(\kappa\), (ii) to attenuate attention and residual updates, and (iii) to weight multiple decoded distributions for the same token at the output. 

Models are pretrained from scratch on OpenWebText and peS2o at 150M, 319M, and 772M parameter scales and compared to a standard GPT‑2‑style baseline and to non-adaptive “copy‑k” baselines with duplicated filler tokens. Thoughtbubbles achieves consistently lower validation perplexity and better zero‑shot performance on LAMBADA and HellaSwag, and analysis shows that extra computation is concentrated on higher entropy tokens in an interpretable way.

---

## Strengths

1. **Clear and well-specified architectural mechanism for latent adaptive computation.**  
   The core machinery (Sections 2.3–2.5) is technically well described. The forking decision function \(f_\theta^{(k)}: \mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^2\) (Eq. (1)), cumulative scores \(\hat{p}_{\text{fork}}, \hat{p}_{\text{keep}}\) (Eqs. (2–3)), the forced-keep rightmost score (Eq. (4)), and the construction of the new residual set via top‑k over the concatenated list \(P\) (Eqs. (5–6)) together provide a precise algorithm for creating and deleting residual streams under a budget. The attenuation in attention and MLP (Eqs. (8–10)) is also explicit and ties the scoring mechanism tightly to the network’s computation.

2. **Non-trivial empirical evidence that the architecture improves language modeling efficiency.**  
   Table 1 is quite comprehensive: across OpenWebText and peS2o, and for 150M, 319M, and 772M model sizes, Thoughtbubbles with \(\kappa = 2L\) or \(4L\) consistently attains lower perplexity than the parameter‑matched baseline and the computation‑matched Copy‑3/Copy‑5 models. For instance, on OpenWebText at 772M, the baseline perplexity is 21.22, Copy‑5 achieves 20.90, whereas Thoughtbubbles reaches 19.74 with \(\kappa = 4L\). The 319M Thoughtbubbles model even beats the 772M baseline on OpenWebText perplexity, which is a compelling efficiency claim.

3. **Parallel, latent rather than explicit chain-of-thought style computation.**  
   A key conceptual contribution is that adaptive computation is implemented *inside* the residual stream rather than via explicit additional tokens in the input or output. Unlike pause tokens or CoT prompting, the forking behavior is learned during standard LM pretraining from scratch, without auxiliary supervision or hand-crafted thinking schedules. This is meaningfully different in spirit from most existing “pause token” or CoT-based test‑time scaling approaches.

4. **Evidence that extra compute is allocated to semantically meaningful and higher-uncertainty regions.**  
   Section 5, Figure 5, and Appendix C (Figure 7) give interpretability-style analyses. In Figure 5, the heatmaps show the normalized number of forks per 4-token window as a function of mean entropy, both under the forking model and under a separate baseline LM. The concave relationship, where computation increases with entropy but dips at extreme entropy, suggests that the architecture is not just forking everywhere but is targeted to moderately uncertain, decision-relevant regions. Figure 7 further visualizes fork counts over layers for OpenWebText and CLUTRR; on the synthetic CLUTRR task, forks cluster near coreferent entities and query delimiters, qualitatively supporting the claim that the system has learned meaningful dynamic allocation.

5. **Thoughtful treatment of positional encoding in the presence of forks.**  
   Appendix D’s partial rotation scheme (Eq. (13)) is not just a hack; it is a coherent extension of RoPE to multiple residuals per token. By using \((k - p/q)\theta\) as the effective position angle for the \(p\)-th fork among \(q\), the method keeps forks of the same token close in positional space while preserving their ordering, which is important to avoid pathological behavior when large numbers of forks appear.

6. **Careful discussion of autoregressive deployment and distribution shift.**  
   Section 5.1 and Figure 6 analyze the discrepancy between blockwise evaluation and autoregressive sampling. The experiments show that naive fixed-budget autoregression hurts perplexity (23.10 vs. 20.97 blockwise), but a dynamic budget scaling \(\kappa' = r L'\) largely closes the gap (21.18) and still improves over the baseline autoregressive perplexity (22.15). This is a useful and honest characterization of an important engineering issue for adaptive compute architectures.

7. **Figures are generally helpful and not just schematic decoration.**  
   Figure 1 gives an intuitive picture of how bubbles of latent computation form and then collapse back, which supports understanding the more formal mechanism. Figure 2 is especially helpful: it walks through the forking process and visualizes which residuals are kept or forked, how top‑k over \(\hat{p}_{\text{fork}}\) and \(\hat{p}_{\text{keep}}\) operates, and how the subsequent transformer block is “score‑attenuated”. This makes the relatively complex mechanism much easier to grasp.

---

## Weaknesses

1. **Comparative baselines are limited and somewhat favorable to the proposed method.**  
   The only computation‑matched comparison is “Copy‑3/Copy‑5,” i.e., naive duplication of input residuals with no adaptivity. This is a very weak competitor relative to existing adaptive computation or depth‑scaling methods. The Related Work section cites several closely related adaptive compute architectures (Mixture-of-Depths, pushdown layers, skip‑layer attention, latent token methods, thinking tokens, etc.), but none are instantiated as baselines. It remains unclear whether Thoughtbubbles is truly competitive against, say, a depth‑adaptive transformer (e.g., Mixture-of-Depths) tuned to the same FLOPs or a recent latent-token-pausing architecture, as opposed to only beating a deliberately naive “copy” baseline that seems unlikely to be strong in practice.

2. **Ambiguity in FLOPs accounting and fairness of computation matching.**  
   The paper states that \(\kappa = 4L\) is “roughly FLOPs-matched against copy-5 baseline” (caption of Table 1) but does not explicitly derive or tabulate FLOPs per forward pass. Table 3 lists “Expanded Size” but does not translate that into FLOPs; the copy baselines always have expanded size 3L/5L for *every* layer, whereas Thoughtbubbles fork only at layers 3, 7, 11, so their average sequence length per layer is lower. This likely makes the comparison more favorable to Thoughtbubbles than the caption implies. A per‑configuration FLOPs table, or at least a principled calculation, is needed to substantiate the “computation‑matched” claim, and the current phrasing risks overstating the efficiency advantage.

3. **Top‑k non-differentiability and gradient flow are under-analyzed.**  
   The core forking decision relies on a hard top‑k over the concatenated list \(P\) of \(\hat{p}_{\text{fork}}\) and \(\hat{p}_{\text{keep}}\) (Eq. (4) and subsequent paragraph). As the authors note in Appendix B under “Top‑K Gradient Bottleneck,” this creates a serious gradient-routing issue: early layers can assign high cumulative scores to a residual that is then dropped by a later top‑k, yielding zero gradient to the earlier forking parameters. However, the main text only briefly mentions this and does not systematically study its impact. There is no comparison to softer alternatives (e.g., Gumbel‑top‑k, continuous relaxations, straight‑through estimators) or even ablations where the top‑k threshold is varied. This is a central design choice and deserves more rigorous treatment.

4. **Some mathematical details are underspecified or slightly inconsistent.**  
   - In Eq. (7), \(P^{(k)} = [p_{\text{cum},1}^{(k)}, \dots, p_{\text{cum},\kappa}^{(k)}]\) implies a length‑\(\kappa\) vector, but earlier the number of residual streams \(N\) after forking is bounded by \(\kappa\) yet not necessarily equal. It is unclear whether the unused slots in \(P^{(k)}\) are zero‑padded, masked, or simply not present. This matters, because Eq. (8) uses \(\log P^{(k)}\) added to attention logits; if any entry is zero, \(\log 0\) is undefined. The text notes that scores are kept in log‑space in practice, but the actual shapes and masking semantics during attention are not formally spelled out.
   - In Eq. (8), the added bias \(\mathbbm{1} \log(P^{(k)})^\top\) is applied to the attention logits. It is not explicitly stated whether this is per‑head or shared across heads, and whether it is broadcast across the query dimension correctly. The equation suggests a matrix of shape (seq_len × seq_len), but with dynamic seq_len per layer it would be helpful to specify indexing more concretely.
   - The output averaging formula Eq. (11) is slightly confusing in notation: it uses \(x_i^{(k)}\) on the left, but the indexing of layers vs. tokens vs. forks is ambiguous, and the variable name is reused for the averaged *distribution* rather than a residual vector. It would be clearer to denote the output distribution as \(p(y_i \mid x_{i,\cdot}^{(f)})\) or similar.

5. **Limited scope and depth of downstream task evaluation.**  
   While Table 1 covers several widely used zero‑shot tasks (LAMBADA, HellaSwag, BLiMP, PIQA), the evaluation remains relatively shallow and sometimes mixed. Gains on HellaSwag and LAMBADA are consistent and substantial, which is good. However, on BLiMP the adaptive model sometimes underperforms Copy‑3/Copy‑5 (e.g., OpenWebText 319M, \(\kappa=2L\), BLiMP 78.3 vs Copy‑3 80.5) and on PIQA the margins are tiny. The paper acknowledges some of this but does not dig into *why* BLiMP, a syntax-focused benchmark, behaves differently, nor does it include reasoning-specific benchmarks (GSM8k, MMLU subsets, etc.) even at the smallest feasible scale. For a paper whose main motivation is enabling harder reasoning through adaptive compute, a more targeted reasoning evaluation, even on small synthetic or toy tasks beyond CLUTRR, would strengthen the case.

6. **Interpretability results are promising but somewhat anecdotal.**  
   Figure 4 presents box plots of attention scores between the main token and its children/siblings/others, showing that the “og” token attends strongly to its forked children. While this does support the idea that forks meaningfully contribute to computation, the analysis is based on a single model and limited samples, without quantitative metrics like fraction of attention mass, KL divergence between fork and parent representations, or causal interventions (e.g., zeroing out fork residuals). Similarly, Figure 5’s entropy vs. forks relationship is intriguing but is averaged over a relatively small context window (4 tokens) and the statistical significance is not assessed. Stronger quantification or ablations would add weight to these claims.

7. **Autoregressive behavior is only tested in a limited setting.**  
   Section 5.1 and Figure 6 analyze autoregression on a “smaller subset” of the OpenWebText dev set, but details are sparse: it is unclear how many tokens, whether the same subset is used for all methods, whether teacher forcing is employed, and whether the dynamic‑budget recipe generalizes to other tasks or sequence lengths. Since most LM usage is autoregressive, the paper would benefit from a more exhaustive AR evaluation, potentially revisiting the zero‑shot tasks using AR scoring and comparing stability under varying prompt lengths.

8. **Related-work coverage on latent reasoning and latent-space computation is incomplete.**  
   The Related Work section is mostly focused on chain-of-thought, adaptive depth, and “pause token” style methods for language models. However, there is a burgeoning literature on *latent* iterative reasoning and latent space manipulation that is directly relevant and is not cited (see section below). This absence weakens the positioning of the paper as a contribution to “latent parallel thinking”, because it does not clearly articulate how the proposed mechanism compares to or generalizes earlier latent reasoning architectures outside the immediate LM test-time-scaling niche.

9. **Minor clarity and consistency issues.**  
   - In Table 1, for OpenWebText 150M, there are two rows labeled “Ours (\(\kappa = 2L\))” with different numbers, which appears to be a typo (one is presumably \(\kappa = 4L\)). This must be corrected, as it affects interpretation of scaling behavior (and Figure 3 might rely on those numbers).
   - Notation occasionally shifts (e.g., \(\kappa_{\text{sample}}\) vs. \(\kappa_{\text{inference}}\)), and layer indices in Sec. 2.4 / 2.5 are not always aligned, which forces the reader to infer intent.
   - The placement of forking layers only at layers 3, 7, 11 for *all* model depths is discussed in Appendix B, but some brief intuition and an explicit pointer to that ablation in the main text of Section 3.1 would help.

---

## Potentially Missing Related Work

1. **Altabaa, Chen, Lafferty, “Unlocking Out-of-Distribution Generalization in Transformers via Latent Space Reasoning” (2025).**  
   This work explicitly uses latent space reasoning mechanisms in transformers to improve OOD generalization. It is directly relevant to the idea of parallel reasoning in latent space, even though the application focus is somewhat different. It should be discussed in Section 6 alongside latent computation and adaptive reasoning methods, clarifying how Thoughtbubbles’ dynamic forking differs from or complements their latent reasoning strategies.

2. **Voynov & Babenko, “Unsupervised Discovery of Interpretable Directions in the GAN Latent Space” (2020).**  
   Although in the GAN domain, this paper is a canonical example of unsupervised discovery of structured computation in latent spaces. Since Thoughtbubbles is framed around unsupervised parallel thinking in latent space, it would be appropriate to cite this work in the Introduction or in Section 6 to situate the contribution within the broader context of unsupervised latent-space structure learning.

3. **Saunshi, Zhu, Arora, “Latent Iterative Reasoning” (2025).**  
   This paper proposes looped transformers and latent diffusion mechanisms to implement iterative reasoning in latent space, with a conceptual goal quite close to that of Thoughtbubbles (iterative processing without explicit natural language traces). It deserves explicit comparison in Section 6 and possibly a short discussion in the Introduction, outlining how the proposed parallel forking differs from their looped / iterative mechanism and whether they could be combined.

4. **Kang, Lee, Kim, “Computation-aware Transformer-based Encoding for Efficient Latent Spatial Neural Architecture Search” (2026).**  
   This work introduces computation-aware transformer encodings for latent search, which are relevant to the notion of budget-aware latent computation. It should be cited in the Related Work discussion of adaptive compute and budgeted transformers, with a brief note about the differences between encoding-level compute allocation for NAS and token-level latent computation allocation in Thoughtbubbles.

---

## Questions

1. **FLOPs and compute-matching.**  
   Can you provide explicit FLOPs-per-token or FLOPs-per-sequence estimates for each configuration in Table 1 (Baseline, Copy‑3/5, \(\kappa=2L,4L\)) and confirm for which settings Thoughtbubbles is indeed computation‑matched to Copy‑5? It would help to see a small table expressing FLOPs as a multiple of the baseline.

2. **Effect of hard top‑k vs. softer relaxations.**  
   Have you tried any differentiable or approximate relaxations of the top‑k selection (e.g., Gumbel‑softmax/top‑k, sparsemax‑style gating, or even a straight‑through estimator) and, if so, how do they compare in terms of perplexity and forking patterns? If not, could you comment on whether you expect such relaxations to mitigate the gradient bottleneck discussed in Appendix B?

3. **Behavior under fully autoregressive evaluation on downstream tasks.**  
   For LAMBADA, HellaSwag, BLiMP, and PIQA, are your reported scores based on blockwise scoring or on true autoregressive scoring with dynamic budget? If it is the former, can you provide a small study comparing blockwise vs. AR for at least one dataset (say, LAMBADA) to show that the performance gains persist under AR usage?

4. **Robustness of entropy–fork correlation.**  
   In Figure 5, the concave relationship between entropy and forks looks plausible but somewhat noisy. How stable is this pattern across different random seeds, checkpoints, or datasets? Could you report quantitative metrics (e.g., Spearman’s correlation between entropy and fork count) to back up the visual impression?

5. **Why BLiMP lags behind computation-matched baselines in some settings.**  
   Do you have hypotheses or diagnostic analyses explaining why the BLiMP scores sometimes fall below the Copy‑3/Copy‑5 models, despite improved perplexity? Is this an artifact of training duration, or does the pruning mechanism specifically harm fine-grained syntactic distinctions?

6. **Potential benefits of deeper or later forking.**  
   Appendix B shows a small ablation where more extensive forking slightly worsens perplexity at 25k steps. Is this trend stable at full training length (75k steps)? Could alternative initialization or noise injection schemes help preserve gradients for late-layer forks?

---

## Flag For Ethics Review

No ethics review needed.

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

3: good.  
The method is technically sound at the architectural level, the equations largely make sense, and the empirical evidence supports the main claims, although compute matching and gradient-flow issues are not fully explored, and baselines are limited.

---

## Presentation Rating

3: good.  
The paper is overall clear and well written, with helpful figures (especially Figures 1–3 and 7) and detailed appendices. Some notation is slightly sloppy in Sections 2.4–2.5 and Table 1 contains at least one typo, but these are fixable.

---

## Contribution Rating

3: good.  
The work introduces a clearly new architecture for latent adaptive computation that is trainable from scratch with standard LM loss and demonstrates meaningful perplexity and zero-shot improvements. The lack of stronger baselines and more diverse reasoning evaluations keeps it from being outstanding, but it is a solid and interesting contribution.

---

## Overall Rating

8: Accept, good paper (poster).  
The paper presents a well-executed and conceptually interesting adaptive computation architecture that improves LM efficiency without supervised CoT or explicit thinking tokens, and backs it up with non-trivial experiments and analyses. The main limitations are the weakness of comparative baselines and incomplete FLOPs / gradient analyses, but these do not undermine the central result that Thoughtbubbles can successfully learn and use dynamic latent computation. With some additional clarity and stronger positioning against related adaptive compute and latent reasoning work, this would make a valuable contribution to ICLR.

---

## Reviewer Confidence

4: confident.  
I am familiar with transformer architectures, adaptive computation methods, and LM evaluation, and I have carefully read the equations and experimental setup. Some implementation details (e.g., exact FLOPs, training-time stability across seeds) are necessarily inferred, but the overall assessment is unlikely to change drastically.