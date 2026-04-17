---
job_id: bc0bc241-bdaa-420b-a622-8d43fb5ffcdb
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: rzGEfYr2ZC.pdf
paper: Don’t Be Greedy, Just Relax! Pruning LLMs via Frank-Wolfe
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a pruning method for large language models using Frank–Wolfe optimization, clearly within core ICLR topics (representation learning, optimization for deep nets, efficient LLM inference).

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Theory, Conclusion) are present and in English. The method is technically nontrivial, math is mostly sound, experiments cover several modern LLMs with quantitative results (e.g., Table 1, Table 2, Figures 2–5), and there is no obvious fatal error or data leakage.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No hidden prompts or manipulative instructions toward automated reviewers are present in the main content.

---

# Expected Review Outcome:

## Summary

The paper addresses layerwise pruning of large language models without retraining. It formulates the per-layer mask selection problem as a convex relaxation over the convex hull of binary masks, then solves it using the Frank–Wolfe (FW) algorithm with an efficient top‑k Linear Minimization Oracle, and finally thresholds the relaxed mask. The resulting method, SparseFW, is evaluated on several modern GPT-style models under unstructured and semi‑structured sparsity, and the authors provide data‑dependent approximation guarantees relating the relaxed solution to the original combinatorial problem.

## Strengths

1. **Clear convex formulation and use of Frank–Wolfe.**  
   The paper gives a clean convex relaxation of the binary mask constraints (Eq. (10), Eq. (11)) and exploits the fact that the objective is a convex quadratic in the mask. The construction of the LMO over \(\mathcal{C}_k\) (Eq. (12)) is straightforward yet well justified, and the precomputation trick \(G=XX^\top\), \(H=WG\) in Section 2.3 is an important practical detail that makes the approach independent of sequence length \(L\) and calibration batch size.

2. **Careful analysis of existing greedy methods.**  
   Section 2.1 giving a unified view of SparseGPT, Wanda, and RIA as greedy single‑weight pruning procedures is insightful. In particular, the derivation that Wanda’s saliency score arises from optimizing Eq. (4)–(5) for one weight is neat and clarifies exactly what objective Wanda is implicitly solving. The reinterpretation of RIA as Wanda applied to a renormalized weight matrix (Eq. (6)–(7)) is also helpful for positioning.

3. **Strong empirical results across multiple LLMs.**  
   Table 1 shows SparseFW improving either perplexity or zero‑shot accuracy (often both) over Wanda/RIA on Gemma‑2‑9B, Yi‑1.5‑9B, DeepSeek‑7B, Qwen2.5‑7B/14B, and LLaMA‑3.1‑8B, for 50%, 60% unstructured, and 2:4 sparsity. The gains are most convincing in the higher sparsity regimes and on accuracy metrics, where SparseFW is consistently better (e.g., for 60% sparsity, Gemma‑2‑9B accuracy improves from 63.19% to 65.35% with SparseFW(RIA)). This breadth of models is a plus compared to many pruning papers evaluated on 1–2 architectures.

4. **Substantial reduction in local pruning error with insightful analysis.**  
   Figure 2 shows per‑layer relative pruning error reduction versus Wanda on LLaMA‑3.1‑8B at 60% unstructured sparsity, with typical reductions between 20–40% and peaks around 80%. Figure 4 further disentangles continuous vs. thresholded masks: the continuous iterate improves monotonically with iterations, while the thresholded mask initially worsens before catching up, matching the theoretical optimization/rounding error discussion. This sort of diagnostic analysis is much deeper than in most pruning work.

5. **Nontrivial theoretical guarantee that directly matches the algorithm.**  
   Lemma 2 in Appendix E (formal version of Lemma 1 in Section 4) gives a clear bound on \(f(\hat m)-f(m^{\text{int}})\) which splits into optimization error \(\varepsilon\) and a rounding term scaling with \(\lambda_{\max}(Q)(k + \sqrt{d_{\text{in}}k})\). The proof is detailed, uses explicit inequalities on \(\|v\|_1,\|v\|_2\), and connects well to the empirical curves in Figure 4. It is rare for pruning papers to provide explicit guarantees all the way through rounding.

6. **Computational design and ablations are thoughtful.**  
   The design choice to fix a fraction \(\alpha\) of high‑saliency weights and only run FW on the remaining ones is empirically and conceptually motivated. Table 2 systematically explores \(\alpha\) from 0 to 1 across six models and two sparsity regimes, showing that \(\alpha\approx 0.9\) is often best and that \(\alpha=0\) (pure FW) underperforms Wanda, which the authors openly acknowledge. Figure 3 ablates both iterations and calibration samples, highlighting that SparseFW actually benefits from more calibration data, unlike Wanda.

7. **Clarity and writing.**  
   Overall exposition is strong. The algorithm is clearly stated (Algorithm 1 and Algorithm 2), notation is consistent, and the link between the formalism and implementation aspects (precomputing \(G, H\), LMO variants for 2:4 in Appendix D) is well laid out. Figures are generally informative and directly support the narrative.

## Weaknesses

1. **Practical method depends heavily on heuristic warm‑starting and fixing, which undermines the “clean” optimization story.**  
   Although Section 2.2–2.3 presents a principled convex relaxation solved by FW, the empirically successful variant (Algorithm 2, Appendix B; Section 2.3 main text) *requires* fixing a large fraction (\(\alpha \approx 0.9\)) of the highest Wanda-saliency weights and only optimizing over the remaining 10%. The paper explicitly notes that \(\alpha=0.0\) (pure FW) “consistently yields worse results than the baselines” and that SparseFW “tends to prune weights crucial for overall performance” (Conclusion, Page 9–10). This means the actual practical method is closer to “Wanda + small FW refinement” than to a standalone convex optimization‑based pruning scheme. This tension between the elegant theory (which is for full optimization over \(\mathcal{C}_k\)) and the practice (which optimizes over a strongly restricted, nonconvex subset) is underplayed; the theory does not cover the fixed‑weight variant.

2. **Global performance gains are modest and sometimes mixed, especially at 50% sparsity.**  
   While some improvements in Table 1 are meaningful, others are tiny or even negative. For example, at 50% unstructured sparsity on LLaMA‑3.1‑8B, Wanda has perplexity 10.09 and SparseFW(Wanda) is worse at 10.21; SparseFW(RIA) at 9.95 is better than Wanda, but all differences are small. For DeepSeek‑7B 50% unstructured, SparseFW(Wanda) and SparseFW(RIA) are both slightly worse in perplexity than Wanda (7.79 vs. 7.89/7.93), and some 2:4 entries for Qwen2.5‑14B are also worse. The paper claims “drastically reduces per-layer pruning error” and “outperforms strong baselines,” which is accurate locally but somewhat overstated for global language modeling metrics, where improvements are often incremental.

3. **Lack of comparison to more diverse or recent LLM pruning baselines.**  
   The experimental section compares only to Wanda and RIA, arguing they “also aim to find a better pruning mask by solving (MASK SELECTION)” and thus omitting SparseGPT and other reconstruction‑based or adaptive schemes. However, from an application standpoint, a user cares about final perplexity/accuracy at a given sparsity and a given compute budget, not about whether the method is mask‑only or includes reconstruction. Excluding SparseGPT (especially given its popularity) or newer LLM pruning methods like adaptive / learning‑based ones makes it hard to judge practical competitiveness. At minimum, a discussion of how SparseFW compares qualitatively in performance‑vs‑cost tradeoffs is needed.

4. **Theory–algorithm mismatch: the guarantees do not cover the full pipeline used in experiments.**  
   The theoretical result in Section 4 and Appendix E focuses on row‑wise optimization over \(\mathcal{C}_k\) with exact FW and simple top‑k rounding, and the bound depends on \(T\) and \(\lambda_{\max}(Q)\). In the actual method:
   - FW is warm‑started from a specific mask (implicitly a vertex derived from Wanda/RIA), not from arbitrary \(M_0\).
   - A large part of the mask is frozen via \(\overline M\) (Algorithm 2, Line 3 and 7), so the feasible set is effectively a face of \(\mathcal{C}_k\) with fixed entries.
   - The step size in Algorithm 1 is written as \(\eta_t=\pi/(t+2)\), which is inconsistent with the text earlier (“we stick to \(\eta_t=2/(t+2)\)”) and with the theory.
   None of these design choices are reflected in the stated lemmas. While not fatal, it weakens the interpretability of the theorem as a guarantee for the *implemented* algorithm and should be made explicit.

5. **Some mathematical derivations gloss over key details and could be clarified.**  
   Examples:
   - In Eq. (5), the algebra leading to \(w_q^2(XX^\top)_{qq}\) is plausible but not spelled out; given the role of this derivation in re‑interpreting Wanda, a compact but precise derivation (e.g., explicitly showing that all cross terms cancel) would be appropriate.
   - In Section 2.3, the loss is written as \(\mathcal{L}(M_t)=\text{Tr}(W(1-M_t)XX^\top(1-M_t)^\top W^\top)\). This expression implicitly treats \(1-M_t\) as being broadcast like a mask; for clarity, the authors could explicitly note that the Hadamard structure leads to \(W\odot(1-M_t)\) and that their matrix product representation is equivalent.  
   - The Hessian \(Q\) in Appendix E is defined as \(\mathrm{Diag}(w)XX^\top\mathrm{Diag}(w)\). However, in the main text, Section 4 ambiguously refers to \(Q\) as “the Hessian of the objective function” without being explicit about row‑wise vs. matrix‑wise formulation, which can confuse readers checking dimensions.

6. **Limited discussion of compute cost and wall‑clock tradeoffs relative to baselines.**  
   The paper acknowledges SparseFW is “clearly more compute‑intensive than Wanda and RIA” but gives no quantitative numbers (e.g., per‑layer or total pruning time on LLaMA‑3.1‑8B; FLOPs or GPU hours). Figure 3 shows how perplexity improves with iterations and samples, but there is no axis or table for runtime. Given that pruning is often used in resource‑constrained settings, a more explicit analysis (for example, time to prune a 7B model to 60% sparsity with each method) would strengthen the claim that the extra cost is worthwhile.

7. **Generalization beyond language modeling is not demonstrated.**  
   All experiments are on autoregressive LLMs with language modeling objectives and EleutherAI zero‑shot tasks. Although the method is generic for linear layers and applicable to vision or other modalities, no evidence is provided that the method remains robust off GPT‑style architectures. Considering that the introduction claims broad applicability of convex optimization techniques to pruning, even a small experiment on a vision transformer or encoder‑only LLM would make this argument less speculative.

8. **Missing directly related work using Frank–Wolfe in similar contexts.**  
   The related‑work section cites FW in training and sparsity contexts, but omits very closely related recent work that uses Frank–Wolfe for model pruning or LLM model operations (see “Potentially Missing Related Work”). This undercuts the positioning of SparseFW as the first or primary FW‑based approach in the LLM setting.

## Potentially Missing Related Work

1. **H. M. Chen, S. X. Hu, W. Luk, “FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization”, 2025.**  
   Uses Frank–Wolfe for large language model merging with tight memory constraints. It demonstrates FW’s applicability to large transformer‑scale models and analyzes memory vs. performance tradeoffs, which is directly relevant to SparseFW’s emphasis on FW and memory efficiency. It should be mentioned in the FW paragraph of Related Work (Page 3), and discussed as another example of FW enabling scalable LLM operations.

2. **H. E. Shili, N. Patnaik, I. Ruble, “Projection-Free CNN Pruning via Frank-Wolfe with Momentum: Sparser Models with Less Pretraining”, 2025.**  
   This paper applies FW (with momentum) specifically to pruning CNNs. It is highly relevant because it already explores FW‑based pruning, including practical algorithmic variants and convergence behavior. It should be discussed in Section 2’s preliminaries or Related Work as another FW‑pruning approach, with a comparison emphasizing differences between CNN vs. transformer/LLM settings and between momentum variants vs. the basic FW procedure used here.

3. **M. Mozaffari, Y. Hourri, “LEAP: Learnable End-to-End Adaptive Pruning of LLMs”, 2025.**  
   LEAP proposes a learnable adaptive pruning method targeting LLMs, focusing on unstructured sparsity and maintaining accuracy. This is a directly comparable LLM pruning method, and should be mentioned in the pruning‑for‑LLMs part of Related Work (Page 3) and ideally used as an additional baseline and/or qualitative reference, since it competes in the same application niche.

## Questions

1. **Effect of \(\alpha\) on theory and optimization guarantees.**  
   The theoretical results assume optimization over \(\mathcal{C}_k\) with top‑k rounding. When a fraction \(\alpha\) of entries is fixed using \(\overline M\) (Algorithm 2), what can be said theoretically about the resulting constrained problem? Can the authors characterize conditions under which the fixed‑entry FW still approximates the best *integral* mask consistent with the fixed set?

2. **Scaling and runtime numbers.**  
   Can the authors provide pruning wall‑clock times or FLOPs for at least one representative model (e.g., LLaMA‑3.1‑8B at 60% unstructured and 2:4 sparsity) comparing Wanda, RIA, and SparseFW with their best \(\alpha\), 2000 iterations, and 256 samples? This would help practitioners judge whether the perplexity and accuracy improvements justify the extra cost.

3. **Sensitivity to initialization and warm‑start choice.**  
   SparseFW is evaluated with both Wanda and RIA warm‑starts, and most gains seem similar between them. Have the authors tried other initializations (e.g., magnitude‑based, random masks at the same sparsity)? How sensitive is the final performance and convergence curve (e.g., in Figure 4) to the warm‑start’s quality?

4. **Behavior with lower sparsity (e.g., 30–40%) and very high sparsity (e.g., 80–90%).**  
   The experiments focus on 50%, 60%, and 2:4. Does SparseFW still offer per‑layer error reductions at lower sparsities, and do they translate to global metrics? Similarly, does the method remain stable at extreme sparsity, or does the thresholding error in Figure 4 become dominant?

5. **Alternative step‑size schedules and potential momentum variants.**  
   Given the observed behavior where the thresholded curve in Figure 4 initially worsens, have the authors experimented with different step sizes (e.g., smaller or line search) or with FW variants (away‑steps, momentum) that might track vertices more closely and reduce thresholding error? A brief empirical comparison would be insightful, especially since FW with momentum has been explored in pruning contexts.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The convex relaxation, FW optimization, and LMO are technically correct and well explained, and the theoretical lemma is sound for the idealized setting. However, the strongest empirical variant deviates from the theoretical setting by fixing large parts of the mask, and some baselines and compute tradeoffs are underexplored.

## Presentation Rating

3: good.  
The paper is clearly written with solid organization, informative figures (e.g., Figure 2, Figure 3, Figure 4), and explicit algorithms. A few inconsistencies (e.g., step-size typo, slightly compressed math derivations) could be polished but do not impede understanding.

## Contribution Rating

3: good.  
The contribution of recasting mask selection as a convex relaxation solved with FW, plus the unifying analysis of greedy methods and the nontrivial empirical evaluation on multiple LLMs, is meaningful and relevant for the community, though the practical method’s reliance on Wanda‑style heuristics tempers its conceptual impact.

## Overall Rating

8: Accept, good paper (poster).  
Despite some gaps in theory–practice alignment and limited baseline diversity, the combination of a principled FW‑based formulation, clear per-layer error improvements, competitive global metrics across several strong LLMs, and nontrivial theoretical guarantees makes this work a solid and timely contribution that should be of broad interest.

## Reviewer Confidence

4: confident.  
I am familiar with FW methods, pruning for LLMs, and the cited baselines; I carefully checked the main derivations (especially Section 2, Eq. (10)–(12), and Appendix E). Some empirical design choices (e.g., \(\alpha\) tuning, exact implementation details) could still hide subtleties, but they are unlikely to change the main assessment.