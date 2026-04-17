---
job_id: 79a39edb-dfaa-4bb2-86f7-cba25e67a7a7
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ey7CXUBn1g.pdf
paper: ADASVD: Adaptive Singular Value Decomposition for Large Language Models
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper addresses SVD-based compression for large language models, clearly fitting within ICLR’s core areas of representation learning, optimization, and efficient large-scale models.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present. The work is technically coherent, experiments are non-trivial and on standard LLMs/VLMs, and the paper is written in clear English.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden instructions, prompt injections, or attempts to manipulate automated reviewing systems within the main paper content.

---

# Expected Review Outcome:

## Summary

The paper proposes **AdaSVD**, an SVD-based post-training compression method for large language models. AdaSVD has two main components: (1) **adaComp**, which adaptively compensates truncation error by alternately updating the truncated SVD factors \( \mathcal{U}_k^\sigma \) and \( \mathcal{V}_k^{\sigma\top} \) using Moore–Penrose pseudoinverses and a stack-of-batch calibration strategy; and (2) **adaCR**, which assigns layer-wise compression ratios based on a cosine-similarity–based notion of layer importance. Experiments on several 7B-scale LLMs, a VLM, and multiple NLP benchmarks show improved perplexity and accuracy over prior SVD-based methods, especially at high compression ratios.

## Strengths

1. **Clear conceptual extension of SVD-LLM with explicit optimization of data-dependent error**

   The paper starts from the data-dependent compression loss \(\mathcal{L}_{\mathrm{SVD}} = \|\widehat{\mathcal{W}}\mathcal{X} - \mathcal{W}\mathcal{X}\|_F^2\) in Eq. (4) and derives closed-form updates for \( \mathcal{U}_k^\sigma \) and \( \mathcal{V}_k^{\sigma\top} \) (Eqs. (6)–(7)), then reformulates these as least-squares problems solved via Moore–Penrose pseudoinverses (Eqs. (8)–(13)). This gives a principled, data-aware refinement of the truncated SVD factors instead of just keeping the top singular vectors. Within the landscape of SVD-based compression, this is a reasonably clean and self-contained improvement.

2. **Good use of figures to illustrate optimization behavior and stability**

   - **Figure 3(a)** clearly contrasts the “naive update” (NU) formulas from Eqs. (6)–(7) with the Moore–Penrose pseudoinverse update (MPPU). The NU curves oscillate wildly, whereas MPPU converges smoothly, visually validating the numerical-stability motivation behind the LSE + pseudoinverse design.
   - **Figure 3(b)** shows that adding the stack-of-batch calibration strategy (SobC) further lowers the compression error compared with naive calibration, providing evidence that the bucketed averaging in Eqs. (14)–(15) is not just a memory hack but actually improves the optimization landscape.
   - **Figure 3(c)** visualizes the distribution overlap before and after several alternating updates of \( \mathcal{U}_k^\sigma \) and \( \mathcal{V}_k^{\sigma\top} \), showing increasing overlap between compressed and original outputs; this directly supports the core claim of adaComp narrowing the functional gap.

3. **Systematic ablations that isolate each proposed component**

   **Table 3** is well-organized into four subparts:
   - **Table 3a** (adaComp on/off) shows that for LLaMA2‑7B, turning on adaComp consistently improves perplexity over both SVD-LLM and AdaSVD without compensation across 40–60% targets (e.g., at 60%: 89.90 → 78.82 → 50.33 on WikiText-2).
   - **Table 3b** (adaptive vs constant CR) demonstrates that even with constant CR, AdaSVD already slightly improves over SVD-LLM, and adaCR further improves (40% WikiText-2 from 15.38 → 14.76 and C4 from 60.43 → 56.98).
   - **Table 3c** explores the number of adaComp iterations, showing that 1 iteration is often best at moderate compression (suggesting overfitting with limited calibration data), while higher iterations help at more aggressive compression, which is a nuanced and useful insight.
   - **Table 3d** studies minimum retention ratio \(mrr\) in Eq. (19), indicating relative robustness at lower compression and more sensitivity at 60%, which is practically relevant for tuning.

4. **Layer-importance analysis is intuitive and backed by visualization**

   The layer-importance metric \(\mathcal{I}(\mathcal{W}) = \text{similarity}(\mathcal{X}, \mathcal{W}\mathcal{X})\) in Eq. (17), normalized as in Eq. (18), leads to layer-specific compression ratios via Eq. (19). **Figure 4** plots the normalized importance across layers for several models and shows a consistent pattern: the first layer is highly important and, for LLaMA-family models, importance follows a “bowl”-shaped curve. This visualization supports the central design choice in adaCR and gives practitioners intuition about which layers to protect.

5. **Empirical improvements across models, datasets, and settings**

   - **Table 1** demonstrates that on LLaMA2-7B, AdaSVD yields better perplexity than SVD-LLM on WikiText-2/PTB/C4 under 40–60% compression (e.g., at 60%: WikiText-2 89.90 → 50.33; C4 561.0 → 239.18), and modest but consistent gains on average accuracy across five reasoning benchmarks.
   - **Table 2** shows that at 60% compression on WikiText-2, AdaSVD improves perplexity over SVD-LLM across four different LLM families (e.g., Vicuna‑7B: 64.06 → 56.97).
   - **Table 4** indicates that when combined with GPTQ INT4 quantization, AdaSVD still yields lower perplexity than SVD-LLM + GPTQ across most compression ratios, supporting the claim that AdaSVD is orthogonal and complementary to quantization.
   - **Figure 1** provides an overall comparison of perplexity-vs-compression curves for LLaMA2-7B, and AdaSVD clearly sits below SVD-LLM and far below vanilla SVD/FWSVD/ASVD in log-scale perplexity, summarizing the quantitative benefits.

6. **Applicability to VLMs with qualitative evidence**

   **Figure 5** applies SVD, SVD-LLM, and AdaSVD to LLaVA-7B for image captioning. The AdaSVD captions are notably more coherent and relevant to the images than SVD-LLM (which sometimes produces degenerate or nonsensical text). This suggests the method can transfer beyond pure text LLMs and helps substantiate the claim of broad applicability.

7. **Implementation and evaluation choices are mostly reasonable**

   Using 256 WikiText-2 calibration samples and the LM-Evaluation-Harness for zero-shot evaluation follows common practice. Running all methods on a single A100‑80GB GPU with reproduced baselines (FWSVD, ASVD, SVD-LLM) supports basic reproducibility and fairness.

## Weaknesses

1. **Conceptual novelty is incremental relative to SVD-LLM and broader low-rank correction literature**

   The core of adaComp is to take the truncated SVD factors from SVD-LLM (already using whitening) and post-hoc refine them to better fit \(\mathcal{W}\mathcal{X}\), essentially solving a data-weighted low-rank approximation with fixed rank via alternating least squares. This is a standard idea in matrix factorization, and the specific use of Moore–Penrose pseudoinverses for each factor update is textbook linear algebra. Similarly, adaCR’s layer-wise CR allocation is based on layer-wise cosine similarity of \(\mathcal{X}\) and \(\mathcal{W}\mathcal{X}\), which is essentially a simple sensitivity heuristic; related notions of importance- or redundancy-aware layer-wise compression appear in existing pruning/slicing/compression works that the paper itself cites (e.g., LaCo, ShortGPT, and dynamic slicing). The paper does not convincingly articulate what is fundamentally new beyond: “do SVD-LLM, then alternate LSE-based updates with pseudoinverses, and vary rank per layer using a heuristic similarity measure”, which is a relatively modest extension.

2. **Mathematical derivation glosses over critical steps and assumptions**

   - Eqs. (6)–(7) are claimed as solutions obtained by “computing partial derivatives” of \(\mathcal{L}_{\mathrm{SVD}}\), but the derivations are omitted to the supplement. Given that this is the starting point for the improved updates, at least a sketch should be included, especially because \(\mathcal{X}\) appears in Eq. (6) but not in Eq. (7), and the objective in Eq. (5) is not symmetric in \( \mathcal{U}_k^\sigma \) and \( \mathcal{V}_k^{\sigma\top} \). As written, Eq. (7) simplifies to \(\mathcal{V}_k^{\sigma\top} = ((\mathcal{U}_k^\sigma)^\top \mathcal{U}_k^\sigma)^{-1}(\mathcal{U}_k^\sigma)^\top \mathcal{W}\), which corresponds to minimizing \(\|\mathcal{U}_k^\sigma \mathcal{V}_k^{\sigma\top} - \mathcal{W}\|_F^2\), not \(\|\mathcal{U}_k^\sigma \mathcal{V}_k^{\sigma\top} \mathcal{X} - \mathcal{W}\mathcal{X}\|_F^2\). This mismatch is not discussed.
   - The LSE formulations in Eqs. (8)–(10) assume we can safely compute the SVD of \(\mathcal{A} = \mathcal{X}^\top \mathcal{V}_k^\sigma\), but no complexity or conditioning analysis is given. In high dimensions and for large ranks, performing SVD on \(\mathcal{A}\) for every layer and every iteration could be expensive, and the paper does not bound or characterize this overhead.
   - In Eq. (13), the update \(\mathcal{V}_k^{\sigma\top} = ((\mathcal{U}_k^\sigma)^+)^\top \mathcal{W}\) is presented as a solution to minimizing \(\|\mathcal{U}_k^\sigma \mathcal{V}_k^{\sigma\top} \mathcal{X} - \mathcal{W}\mathcal{X}\|_F^2\), but this again ignores \(\mathcal{X}\). If the intent is instead to solve \(\arg\min_{\mathcal{V}_k^{\sigma\top}}\|\mathcal{U}_k^\sigma \mathcal{V}_k^{\sigma\top} - \mathcal{W}\|_F^2\), then the loss in Eq. (5) is inconsistent with the actual update. This conceptual slippage between data-weighted and weight-only losses should be clarified, because it affects the claimed optimality of adaComp.
   - The alternating scheme in Eq. (16) is stated to run “until convergence”, but there is no convergence criterion, nor a guarantee that the sequence \((\mathcal{U}_k^\sigma, \mathcal{V}_k^{\sigma\top})\) converges with pseudoinverse-based updates. In Table 3c, increasing iterations sometimes degrades performance, which suggests non-trivial behavior that contradicts the “minimizing error” narrative in Section 3.1.

3. **Insufficient analysis of computational cost and scalability**

   The method performs for each layer: (i) an SVD for whitening (inherited from SVD-LLM), (ii) SVD of \(\mathcal{A}\) in Eq. (9), (iii) Moore–Penrose pseudoinverses of \(\Sigma_A\) and \(\mathcal{U}_k^\sigma\), and (iv) multiple alternating iterations over all layers. There is no complexity analysis or wall-clock / memory comparison against SVD-LLM. The text claims stability and efficiency, but:
   - **Figure 2** and **Algorithm 1** show additional modules (SOB, ADA\_UPDATE) that are clearly more complex than SVD-LLM, yet no quantitative runtime or GPU memory statistics are reported.
   - The stack-of-batch strategy (Eqs. (14)–(15)) trades sample count for batched averaging; it is plausible that this reduces effective information content in calibration data. The paper does not explore the tradeoff between larger \(M\) vs. information loss, or compare to simply using fewer sequences without averaging.
   This is particularly important because one of the main motivations is efficient deployment on resource-constrained devices, but we only see model-size compression and perplexity, not end-to-end runtime or energy/latency benefits.

4. **Layer-importance metric and adaCR lack deeper justification and analysis**

   The importance measure \(\mathcal{I}(\mathcal{W}) = \text{cos\_sim}(\mathcal{X}, \mathcal{W}\mathcal{X})\) in Eq. (17) is heuristic. There are several open questions:
   - Cosine similarity between inputs and outputs can be high simply because of mean components or norm differences; there is no argument why this correlates with actual downstream sensitivity or gradient-based importance.
   - Eq. (19) linearly maps normalized importance to per-layer CR between \(mrr\) and \(trr\), but this implicitly assumes a linear relationship between this similarity and “needed rank”, which is not justified.
   - **Figure 4** shows the first layer being very important and later layers varying by architecture, but there is no experiment that directly compares adaCR against, for example, a simple rule like “keep more rank at first and last K layers” or against alternative importance metrics (e.g., Hessian-based, Fisher-based, or gradient norms). In **Table 3b**, the gains of adaCR over constant CR are modest (e.g., 16.11 → 14.76 perplexity at 40%); it is unclear whether such a small relative gain justifies the additional machinery.

5. **Limited experimental scope regarding tasks and calibration data**

   - All quantitative evaluations focus on perplexity and a standard set of QA benchmarks. There is no examination of instruction-following, chat quality, or downstream fine-tuning tasks, despite the fact that changes in low-rank structure could disproportionately affect different capabilities.
   - The method uses a single calibration dataset (256 samples from WikiText-2) for all models and tasks. While this matches prior work, the proposed adaComp is explicitly data-dependent; one would expect its effectiveness to vary with the calibration distribution. The paper does not explore sensitivity to calibration source/domain or sample size (other than noting GPU memory constraints).
   - For VLMs, **Figure 5** shows a few qualitative captions but no quantitative metrics (e.g., CIDEr, BLEU, SPICE) or a systematic comparison across images. As a result, the evidence for VLM applicability remains anecdotal.

6. **Some confusion around reported compression ratios and parameter accounting**

   Eq. (20) defines the compression ratio \(\mathcal{CR}(\mathcal{W}_i) = \frac{\#\text{params of }\mathcal{U}_k^\sigma + \#\text{params of }\mathcal{V}_k^{\sigma\top}}{\#\text{params of }\mathcal{W}_i}\). However:
   - The paper often speaks of “40%, 50%, 60% compression ratio” but does not explicitly clarify whether this is a retention ratio (i.e., 40% of original parameters) or compression level (i.e., 60% removed). The tables (e.g., Table 1) label “RATIO” but not whether this refers to \(\mathcal{CR}\) or \(1-\mathcal{CR}\). The abstract and intro mention “target retention ratio” \(trr\), but the main tables only list “RATIO”, which can be confusing.
   - When adaCR assigns different per-layer \(\mathcal{CR}(\mathcal{W}_i)\), it is only stated that “with fixed target compression ratio” performance improves. How the global constraint \(\sum_i \#\text{params}_i \cdot \mathcal{CR}(\mathcal{W}_i)\) is enforced to match a target model-level ratio is not described. Without this, reproducibility is hindered.

7. **Lack of robustness and failure-mode analysis**

   - **Table 3c** shows non-monotonic behavior with more adaComp iterations (sometimes performance gets worse). This suggests overfitting to calibration data or numerical issues, but the paper treats this as a tuning detail rather than analyzing when and why the optimization behaves poorly.
   - There is no investigation of sensitivity to the choice of bucket size \(M\) in the stack-of-batch strategy, or to the random shuffling in Eq. (14). Re-running with different random seeds and reporting variance would be informative.
   - The method is evaluated only at 7B-scale models; it is unclear whether the numerical stability and computational overhead scale to 13B, 34B, 70B models, where pseudoinverse computations become heavier.

8. **Presentation and exposition issues**

   - Some notation is inconsistent or sloppy: \(\nu^\top\) appears once instead of \(\mathcal{V}^\top\); sometimes \(\mathcal{V}_k^{\sigma^\top}\) uses a double superscript; and matrix dimensions are never explicitly specified. This makes it harder to verify that operations like \(\mathcal{X}^\top \mathcal{V}_k^\sigma\) in Eq. (8) are dimensionally consistent.
   - **Algorithm 1** uses abstract routines (e.g., `WHITENING`, `LAYER_CR`, `ADA_UPDATE`) without specifying their exact computations or arguments, and the pseudocode does not mention the use of adaCR explicitly. A reader cannot implement the method from Algorithm 1 alone.
   - Several figures are compressed into composite panels (**Figure 2**, **Figure 4**) with small text and multiple subplots, which makes it hard to read exact values from the importance histograms or ablation curves. While this may be acceptable for camera-ready, the current version feels visually crowded.

9. **Missing discussion of very closely related low-rank correction / SVD variants**

   Beyond SVD-LLM, ASVD, and FWSVD, there is a growing body of work that combines SVD-compressed layers with learnable corrective modules or alternative low-rank optimization schemes, some of which aim at exactly the “residual correction” problem attacked by adaComp. These are not discussed, which weakens the positioning of the contribution (details in “Potentially Missing Related Work” below).

## Potentially Missing Related Work

1. **Chong & Qu, “Singular Value Decomposition on Kronecker Adaptation for Large Language Model”, 2025**  
   Combines Kronecker-product tensor factorization with SVD-based initialization for efficient adaptation of LLMs. It is directly relevant because it leverages SVD factorization to reduce parameter cost during fine-tuning. It should be cited and compared in Section 2.2 as another SVD-based method for adapting/compressing LLMs, and briefly discussed in relation to adaComp (which corrects truncation error) vs. SoKA (which alters the parameterization).

2. **Kautsar et al., “CALR: Corrective Adaptive Low-Rank Decomposition for Efficient Large Language Model Layer Compression”, 2025**  
   CALR explicitly adds a learnable low-rank corrective module on top of SVD-compressed layers to recover functional residuals. This is very close conceptually to adaComp’s goal of compensating SVD truncation error. It should be discussed in Section 2.2 and in Section 3.1 as an alternative residual-correction mechanism that uses learned modules instead of closed-form pseudoinverse updates; ideally, an empirical comparison on at least one model would strengthen the claims.

3. **Modoranu et al., “SVD-Free Low-Rank Adaptive Gradient Optimization for Large Language Models”, 2025**  
   Proposes an SVD-free low-rank optimization approach achieving comparable performance at lower computational cost. Given that AdaSVD adds additional SVD and pseudoinverse computations, this paper is relevant for contrasting computational efficiency and could be discussed in Section 2.2 and in a short runtime discussion in Section 4.2.

4. **Zhang et al., “GF-SVD: Global knowledge-infused singular value decomposition of large language models”, 2026**  
   GF-SVD integrates global information to guide SVD-based compression, addressing inter-layer coherence and cross-domain generalization. Since adaCR is also about coordinating layer-wise compression via a global importance profile, GF-SVD should be mentioned in Section 2.2 and compared conceptually to highlight differences in how “global knowledge/importance” is modeled.

5. **Chen et al., “MGIE-SVD: Multidimensional Gaussian Information Entropy-driven SVD compression method for transformer architectures”, 2026**  
   MGIE-SVD focuses on non-uniform redundancy and information entropy across layers and heads, which is closely related to the paper’s concern about non-uniform layer importance. It should be cited in Section 2.2 and possibly referenced near Eq. (19), emphasizing why cosine similarity is preferred over entropy-based importance and whether similar non-uniformity observations are found.

## Questions

1. **Clarify the optimization objective and derivations in Section 3.1**  
   - Are Eqs. (6)–(7) derived from minimizing \(\|\mathcal{U}_k^\sigma \mathcal{V}_k^{\sigma\top}\mathcal{X} - \mathcal{W}\mathcal{X}\|_F^2\) or \(\|\mathcal{U}_k^\sigma \mathcal{V}_k^{\sigma\top} - \mathcal{W}\|_F^2\)?  
   - Please provide the main derivation steps in the rebuttal (and ideally in the final paper) and explain why \(\mathcal{X}\) disappears from Eq. (7) and Eq. (13). If there is a typo or if a different loss is actually optimized, clarifying this would significantly increase my confidence.

2. **Complexity and runtime overhead**  
   Could you provide wall-clock runtime and memory comparisons between SVD-LLM and AdaSVD for compressing LLaMA2-7B at 40%, 60%, and 80% ratios on the A100-80GB, including the cost of the whitening step, SVD of \(\mathcal{A}\), and alternating pseudoinverse updates? Even approximate scaling trends (e.g., how cost grows with rank or number of iterations) would help justify practicality.

3. **Global compression ratio enforcement with adaCR**  
   How do you ensure that the layer-wise compression ratios from Eq. (19) satisfy a desired *global* target compression ratio? Is there normalization or rescaling of \(\{\mathcal{CR}(\mathcal{W}_i)\}_i\) to match a global constraint? Please describe the exact procedure (including how \(mrr\) and \(trr\) are chosen) and how sensitive final performance is to this choice.

4. **Alternative importance metrics and baselines for adaCR**  
   Have you tried simpler or alternative schemes, such as (a) allocating higher rank to the first and last K layers only, (b) using activation variance or gradient norms as \(\mathcal{I}(\mathcal{W})\), or (c) random per-layer CRs with the same global average? A small ablation could help establish whether the cosine similarity metric is genuinely effective or just a weak heuristic.

5. **Calibration data dependence and robustness**  
   How does AdaSVD perform if calibration data comes from a different distribution (e.g., C4, instruction-tuning data) or with fewer/more than 256 samples? Given that adaComp heavily relies on \(\mathcal{X}\), understanding robustness to calibration domain and sample size is important.

6. **Behavior at larger scales and more iterations**  
   Have you attempted applying AdaSVD to larger models (e.g., LLaMA2-13B or 70B)? If so, did the pseudoinverse computations remain numerically stable and computationally manageable? Also, can you comment on the non-monotonic behavior in Table 3c (more iterations sometimes hurt), and whether a stopping criterion based on calibration loss is feasible?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The method is empirically validated and mostly sensible, but the derivation of the update rules is incomplete and somewhat inconsistent with the stated loss, convergence behavior is not well understood, and computational aspects are not analyzed, which reduces confidence in the overall soundness.

## Presentation Rating

2: fair.  
The paper is readable and the main ideas come through, but mathematical exposition is incomplete in key places, Algorithm 1 is high-level, some notation is inconsistent, and several figures are visually dense.

## Contribution Rating

2: fair.  
The work offers a useful, data-aware refinement to SVD-LLM with solid empirical gains, but conceptually it is an incremental extension of existing SVD-based and low-rank correction methods, with limited theoretical depth and missing discussion of closely related recent work.

## Overall Rating

4: marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper demonstrates clear empirical improvements over prior SVD-based compression on several 7B LLMs and introduces reasonably principled refinements (alternating pseudoinverse compensation and layer-wise CR allocation). However, the conceptual novelty is modest, the derivations and objectives are not fully clean or justified, computational overhead is not analyzed, and related work on corrective low-rank decompositions and advanced SVD variants is incomplete. With stronger theoretical clarity, runtime analysis, and better positioning against closely related work, this could become a solid contribution.

## Reviewer Confidence

4: confident.  
I am familiar with SVD-based compression and low-rank methods for LLMs, have carefully examined the equations and experiments, and I am reasonably confident in the assessment, though some derivation details hidden in the supplement leave a bit of uncertainty.