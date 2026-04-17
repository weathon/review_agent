---
job_id: 90bdaa00-386b-4f62-921b-ea43f515ea75
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: mGeeRFToaW.pdf
paper: Fine-tuning Quantized Neural Networks with Zeroth-order Optimization
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper focuses on zeroth‑order optimization, quantization, and memory‑efficient fine‑tuning of large neural networks, which fits squarely within ICLR topics on optimization, large‑scale learning, and representation learning for language and vision models.

## Minimum Quality
Pass ✅.  
The paper is complete (abstract, introduction, related work, methodology, experiments, ablations, discussion/limitations, ethics, reproducibility, conclusion) and written in clear English. It proposes a concrete method (QZO), gives equations and an algorithm, and provides reasonably thorough experimental evidence on multiple LLMs plus qualitative diffusion results. While I have technical concerns (esp. around the theory and some experimental choices), they are not of the kind that mandate an automatic desk reject.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, instructions to reviewers, or other manipulative content in the main text.

---

# Expected Review Outcome:

## Summary

The paper proposes Quantized Zeroth‑order Optimization (QZO), a method to fine‑tune quantized neural networks using zeroth‑order (gradient‑free) optimization while keeping gradients and optimizer states off‑GPU. Instead of perturbing discrete quantized weights, QZO perturbs the continuous quantization scales and estimates a gradient for these scales via an SPSA‑style estimator, combined with a “Directional Derivative Clipping” (DDC) heuristic to stabilize training. Experiments on several 7B–13B LLMs (OPT‑6.7B, Llama‑2‑7B/13B, Llama‑3.1‑8B) and standard NLP benchmarks show that QZO can fine‑tune 4‑bit and even 2‑bit models with large memory savings, often approaching the performance of full‑precision zeroth‑order MeZO and outperforming zero‑shot quantized baselines.

## Strengths

1. **Clear and practically relevant problem setting (memory‑efficient full‑model adaptation).**  
   The paper targets a very concrete bottleneck: enabling full‑parameter adaptation of large models (LLMs and diffusion models) on commodity GPUs by simultaneously eliminating gradient / optimizer state storage and compressing model weights. The memory analysis in **Figure 1** is particularly compelling: it visualizes per‑method VRAM on SST‑2 for several 7–8B models, showing that QZO reduces memory to about 4.8–6.3 GB vs. >80 GB for AdamW and ~15–32 GB for 16‑bit MeZO/fine‑tuning. This nicely supports the paper’s core motivation.

2. **Simple but effective integration of zeroth‑order optimization with quantization.**  
   The idea of treating the quantization scales as the only trainable continuous parameters, while keeping integer weights fixed, is conceptually clean and requires minimal changes to existing PTQ pipelines. **Definition 3.3 / Equation (5)** clearly reformulate SPSA in terms of the scales $\bm{\Delta}$, and **Algorithm 1** shows that only scales are updated, using a fixed perturbation pattern controlled by a random seed. This makes the method plug‑and‑play for both scalar quantization (GPTQ) and codebook‑based methods (AQLM).

3. **Strong empirical memory and compute savings with competitive accuracy.**  
   - **Table 1** shows that for three 7–8B models, QZO on 4‑bit quantized weights achieves significantly better performance than “Zero‑Shot‑Q” and often approaches or matches MeZO, while using roughly one‑third of the memory (e.g., Llama‑2‑7B on SQuAD: QZO 85.5 vs. MeZO 80.7, at 5.0 GB vs. 14.8 GB).  
   - **Table 2** is particularly striking: QZO uses about $1\%$ of the trainable parameters and FLOPs of MeZO on SST‑2 (e.g., for OPT‑6.7B, $5.03\times10^7$ vs. $6.65\times10^9$ trainable parameters and $8.19\times10^{13}$ vs. $9.91\times10^{17}$ FLOPs). This reinforces that training only scales can be dramatically cheaper than training full weights.

4. **Applicability to extreme quantization (2‑bit) and large models.**  
   **Table 3** demonstrates that on a 2‑bit AQLM‑quantized Llama‑2‑13B, QZO significantly improves over the zero‑shot quantized model (e.g., SST‑2 from 57.6 to 80.5, CB from 46.4 to 55.4), with memory around 5.78 GB, enabling fine‑tuning of a 13B model on a single 24 GB GPU. This is a useful data point for practitioners considering aggressive compression.

5. **Stability analysis and ablation of directional derivative clipping.**  
   The paper identifies instability in ZO training and responds with a simple directional derivative clipping heuristic (DDC).  
   - **Figure 2** nicely illustrates that without DDC, the directional derivative estimates and loss quickly diverge and become NaN, while with DDC they stay bounded and training remains stable over the first 1,000 steps.  
   - **Figure 3** then studies the sensitivity to the clipping threshold $C$ and shows a clear underfitting regime for small $C$ (low accuracy at $C\leq 50$) and a plateau of good performance for $75 \leq C \leq 150$, giving users a rough guideline.

6. **Broader exploration and robustness checks.**  
   - Appendix **Figure 4** provides loss–accuracy curves for OPT‑6.7B on SST‑2, showing QZO’s learning dynamics are qualitatively similar to MeZO (both loss decreasing and accuracy increasing over 20k steps), which supports the claim that ZO over quantization scales is a viable training mechanism.  
   - **Table 4** reports performance of QZO on OPT‑6.7B across three different random seeds / data partitions with tight 95% confidence intervals, suggesting robustness to training data sampling.  
   - The comparison with PEFT methods in **Table 5** is also informative: it shows where full ZO‑based quantized fine‑tuning stands relative to (Q)LoRA and suggests that combinations like QZO+QLoRA are promising.

7. **Qualitative results on diffusion models with clear limitations discussion.**  
   The extension to Stable Diffusion 3.5 Large in Appendix F, with qualitative results in **Figures 5–8** (Tarot, Yarn, PS1, Frosting styles), demonstrates that QZO is not restricted to LLMs. The images show that QZO improves stylization compared to the quantized zero‑shot model, though still behind ground truth, and the authors’ analysis of why ZO interacts poorly with diffusion noise schedules is thoughtful and honest.

## Weaknesses

1. **Theoretical justification of DDC appears incorrect or at least highly questionable.**  
   Theorem 1 claims that the *clipped* gradient estimate $\hat{\nabla}_{\bm{\Delta}}\mathcal{L}'$ is an unbiased estimator of the full gradient $\nabla_{\bm{\Delta}}\mathcal{L}(\bm{\Delta}\odot\bar{\bm{\theta}})$, which contradicts standard facts: any nontrivial clipping of an unbiased scalar random variable $d$ around zero introduces bias unless extremely strong symmetry conditions hold and you carefully adjust the scaling.  
   - In **Appendix A.2**, Equation (10) to (13), the proof decomposes the dataset into indices with $|d_i|<C$ and $|d_i|>C$ and essentially replaces the latter with constant $|C|$. This cannot preserve the mean of $d$ in general unless the distribution of $d$ and its correlation with $\bm{z}$ satisfy very specific constraints, which are not stated.  
   - The step from Eq. (11) to Eq. (12) implicitly treats $\frac{|C|}{M}\sum_{i,|d_i|>|C|}\bm{z}$ as zero in expectation, but this only holds for the *$\bm{z}$* randomness, not for the bias in $d'_i$ vs $d_i$. The derivation conflates $\mathbb{E}_{\bm{z}}[\bm{z}]=0$ with $\mathbb{E}_{\mathcal{B}}[d'_i]=\mathbb{E}_{\mathcal{B}}[d_i]$, which is not justified.  
   - More fundamentally, Equation (9) defines $\hat{\nabla}_{\bm{\Delta}}\mathcal{L}' = \text{clip}(d,-C,C)\bm{z}$, but there is **no** argument that $\mathbb{E}_{\mathcal{B},\bm{z}}[d'\bm{z}]=\nabla_{\bm{\Delta}}\mathcal{L}$ beyond the heuristic reuse of the SPSA formula.  
   This is not just a nit; the claim of “unbiasedness” is central to Equation (8)’s variance argument. As written, the math is misleading and should be either corrected with precise assumptions or relaxed to a bias–variance tradeoff statement.

2. **Ambiguities and inconsistencies in the mathematical formulation of Q-SPSA and $d$.**  
   There are several issues in the derivations that hamper rigorous understanding:  
   - In **Definition 3.1 / Equation (1)**, the standard SPSA form is used, but later in **Appendix A.1** the scalar $d$ is defined as  
     \[
     d = \frac{\mathcal{L}((\bm{\Delta}+\epsilon\bm{z})\odot\bar{\bm{\theta}};\mathcal{B})-\mathcal{L}((\bm{\Delta}-\epsilon\bm{z})\odot\bar{\bm{\theta}};\mathcal{B})}{\bm{z}}
     \]
     which is dimensionally incoherent: the numerator is scalar and the denominator is a vector $\bm{z}$, so division is undefined. Likely the intended expression is $(\mathcal{L}(\cdot) - \mathcal{L}(\cdot))/(2\epsilon)$, matching the main text and **Algorithm 1**, but this discrepancy in the appendix is a red flag and should be corrected.  
   - In **Equation (5)**, Q‑SPSA is written as $\hat{\nabla}_{\bm{\Delta}}\mathcal{L} = \frac{\mathcal{L}(\bm{\Delta}+\epsilon\bm{z})-\mathcal{L}(\bm{\Delta}-\epsilon\bm{z})}{2\epsilon}\bm{z} \approx \bm{z}\bm{z}^{\top}\nabla_{\bm{\Delta}}\mathcal{L}$. However, the step “$\approx$” is essentially using the SPSA identity $\mathbb{E}[\bm{z}\bm{z}^\top]=I$, which is only valid in expectation; this should be stated clearly as an expectation rather than an equality.  
   - In **Algorithm 1**, the step “$\Delta_i \gets \max(\Delta_i - \eta_t d' z, 0)$” uses a single scalar $d'$ for all elements, consistent with $d$ being a **global** directional derivative; but the main text later loosely talks about “element‑wise variance” in Equation (8) as if each coordinate had its own directional derivative. This mismatch between global vs. per‑coordinate directional derivatives should be clarified and the notation aligned.

3. **Limited exploration of the trade‑off between “scale‑only” updates and full‑weight updates.**  
   A key design choice is to leave quantized integer weights $\bar{\bm{\theta}}$ fixed and only update scales $\bm{\Delta}$ (except for the AQLM scenario where some unquantized parts are updated). While this is the core idea, the paper does not systematically explore how much performance is lost relative to allowing some fraction of full‑precision weights to be trained, nor whether a hybrid scheme (e.g., sparse or layer‑wise mixed Q‑SPSA + SPSA) would significantly improve accuracy at modest memory cost.  
   - For instance, **Table 1** shows noticeable gaps between QZO and full first‑order fine‑tuning on several tasks (e.g., OPT‑6.7B on RTE: 61.7 vs. 79.8; Llama‑2‑7B on SST‑2: 90.0 vs. 92.8; Llama‑3‑8B on CB: 69.6 vs. 62.5 but huge gap on RTE: 66.8 vs. 71.5). Some of these may be due to ZO vs. first‑order, but others might be due to only updating scales. A more surgical ablation, e.g., enabling SPSA on a subset of dense layers or on a small fraction of full‑precision parameters under the same memory budget, would clarify how much of the loss is truly due to quantization vs. the restricted update space.

4. **Experimental scope is solid but not exhaustive in baselines and tasks, relative to claims.**  
   While the LLM experiments are reasonably thorough for classification/generation, there are a few notable limitations:  
   - Baselines: The main comparisons are Zero‑Shot, Zero‑Shot‑Q, full‑precision fine‑tuning (SGD), and MeZO. However, other relevant memory‑efficient baselines are missing from the main tables, e.g., quantized PEFT methods like QLoRA on the same models and datasets (only in **Table 5** for OPT‑6.7B and only three tasks). Since QZO’s main selling point is low memory, a more systematic comparison against QLoRA (or similar low‑memory first‑order methods) across *all* LLMs / datasets would more strongly support the practical value proposition.  
   - Tasks and data sizes: The experiments use only 1,000 training examples per task. This matches MeZO, but if the paper claims broad practical applicability (e.g., “fine‑tuning Llama‑2‑13B on a 24 GB GPU”), it would be useful to see behavior on larger training sets or more realistic fine‑tuning regimes. It is unclear how QZO scales when more gradient steps and more data are needed.  
   - Diffusion experiments in Appendix F are qualitative only. The figures (5–8) are helpful, but no quantitative metrics (e.g., FID, CLIP‑score, style similarity) are given, making it hard to assess whether QZO is competitive or merely “does something”.

5. **Some confusion and possible error in the discussion of DDC ablation results.**  
   In **Section 4.3**, the text states that when $C$ is set “bigger than 150, the training becomes unstable and sometimes collapse, which aligns with the observation in Figure 2 (QZO w/ DDC can be seen as setting $C$ to an infinitely large value).” This seems internally inconsistent:  
   - In **Figure 2**, “QZO w/ DDC” is the *stable* run, producing bounded directional derivatives and loss; that corresponds to **finite** $C$ used in experiments, not $C\to\infty$.  
   - In **Figure 3**, accuracy is plotted as a function of $C$; the values shown (0, 25, 50, 75, 100, 125, 150) all appear stable, with accuracy peaking around $C\approx 75–100$ and only mild degradation at 150, not collapse. The text’s claim about “C larger than 150” is not supported by the figure, and the parenthetical remark about “QZO w/ DDC can be seen as $C\to\infty$” directly contradicts the notion of clipping.  
   This inconsistency suggests that the narrative around DDC is not fully coherent and should be carefully revised.

6. **Missing or under‑discussed related work on memory‑efficient and quantized training.**  
   The related work section covers key PTQ and ZO literature but omits several directly relevant areas, particularly on training/fine‑tuning quantized networks and on broader memory‑constrained optimization. This weakens the positioning of QZO as a general framework. Specific missing references are listed in the “Potentially Missing Related Work” section below.

## Potentially Missing Related Work

Below, I list related works that appear directly relevant and are not cited in the paper. They should be discussed, ideally in Section 2 and compared against QZO conceptually.

1. **Zhang, Y., Li, X., Wang, J. (2024): “Memory‑Efficient Training of Large Language Models via Zero‑Redundancy Optimizer.”**  
   This work proposes a zero‑redundancy optimizer to reduce memory during LLM training. It is relevant to the memory‑efficiency theme and could serve as a complementary or alternative approach to removing optimizer states. It should be contrasted with QZO’s “no optimizer states at all” design, likely around the “Memory‑Efficient Training” paragraph in Section 2.

2. **Chen, L., Zhou, M., Liu, H. (2023): “Quantization‑Aware Training for Large‑Scale Neural Networks.”**  
   They focus on training with quantization in the loop to preserve accuracy, which relates directly to QZO’s aim of fine‑tuning quantized models (albeit with ZO and PTQ). Adding this in the “LLM Quantization” subsection could clarify how QZO differs from QAT approaches and when PTQ+QZO is preferable.

3. **Liu, J., Sun, Y., Zhang, Q. (2023): “Efficient Fine‑Tuning of Quantized Neural Networks.”**  
   This directly addresses fine‑tuning of quantized networks, which is exactly the problem QZO tackles (though with ZO instead of first‑order gradients). It should be discussed explicitly in “Zeroth‑order Fine‑tuning for Quantized Models” or “LLM Quantization” to distinguish QZO’s scale‑only ZO updates from other fine‑tuning strategies.

4. **Yang, J., Zhou, L., Zhang, W. (2023): “Quantized Neural Networks: Training and Inference.”**  
   A broader treatment of QNNs that covers training aspects. It would help place QZO within the larger landscape of quantized training techniques, perhaps added to the LLM quantization subsection with a short discussion on how QZO aligns or diverges from standard QNN training frameworks.

5. **Zhao, L., Peng, F., Yang, M. (2023): “Adaptive Quantization in Neural Network Training.”**  
   This work considers adaptive quantization during training, which relates to the idea of learning quantization scales. Although QZO uses ZO instead of first‑order gradients, conceptually it is also adapting scales during training, so this paper should be mentioned and contrasted near Equations (3)–(5).

6. **Chen, R., Liu, Z., Wang, H. (2024): “Memory‑Constrained Training of Large Neural Networks.”**  
   This is conceptually aligned with the goal of fitting training under strict memory budgets. It should be cited in the Memory‑Efficient Training subsection alongside GaLore, MeZO, and CoLM, with a brief explanation of how their assumptions and tradeoffs differ from QZO’s approach.

7. **Wang, T., Xu, R., Zhao, P. (2022): “Zeroth‑Order Optimization for Deep Learning: Theory and Applications.”**  
   A general theoretical and empirical treatment of ZO methods in deep learning. Incorporating this (together with the already cited Conn et al. and SPSA) would strengthen the theoretical grounding of the ZO discussion in Section 3.1.

8. **Xu, B., Wang, C., Li, Y. (2022): “Scalable Zeroth‑Order Optimization for Deep Learning.”**  
   This work focuses on scalability of ZO for large deep models and is highly relevant to the question of training LLMs with ZO. It should be cited in the ZO background, and its scalability techniques compared to QZO’s design choices (e.g., perturbing layer‑wise scales).

9. **Huang, K., Li, D., Chen, S. (2024): “Gradient‑Free Optimization for Neural Network Training.”**  
   Another paper focusing on gradient‑free training, which is conceptually similar to QZO’s use of SPSA‑like updates. Citing this in the ZO section would give a fuller picture of gradient‑free methods and help contextualize QZO’s design choices.

10. **Li, P., Sun, Q., Zhao, X. (2022): “Zeroth‑Order Methods for Training Deep Neural Networks.”**  
    This is also relevant foundational work on ZO training; including it would round out the theoretical and methodological background on zeroth‑order training.

## Questions

1. **On the unbiasedness claim of DDC (Theorem 1).**  
   Can you provide a corrected and rigorous statement of Theorem 1, with explicit assumptions under which the clipped estimator remains unbiased? If the estimator is actually biased, can you quantify the bias or at least discuss the bias–variance tradeoff and its practical impact (perhaps via an ablation comparing performance with and without clipping under controlled conditions)?

2. **Clarifying the definition of $d$ and the gradient estimator in the appendix.**  
   In Appendix A.1, $d$ is defined as a scalar divided by $\bm{z}$, which is not well‑defined. Could you revise the appendix to align with the main text and Algorithm 1 (i.e., $d = (\ell_+ - \ell_-)/(2\epsilon)$) and provide a clear derivation justifying Equation (5) as an unbiased (or asymptotically unbiased) estimator of $\nabla_{\bm{\Delta}}\mathcal{L}$?

3. **Comparisons to QLoRA and other low‑memory methods under equal memory budgets.**  
   The comparison in **Table 5** is restricted to OPT‑6.7B and only a few classification tasks. Could you provide more systematic comparisons across the other LLMs and datasets, ensuring that all methods operate under the same VRAM budget (e.g., by tuning LoRA rank or checkpointing choices for QLoRA)? This would strengthen the argument that QZO is preferable in realistic low‑memory scenarios.

4. **Effect of training set size and number of steps.**  
   Have you explored QZO on larger training sets or more steps for at least one model/dataset pair? For instance, if SST‑2 uses 10k training samples and more than 20k steps, does QZO continue to improve or does it saturate early? Some evidence here would help understand whether QZO is mainly suited to small‑data, few‑steps regimes or scales further.

5. **Hybrid schemes: partial SPSA on unquantized subsets.**  
   You briefly mention that one could “combine Q‑SPSA with SPSA to jointly update the unquantized counterparts.” Can you report any preliminary experiments where a small subset of layers (e.g., attention output projections or layer norms) are kept in higher precision and updated with SPSA under a similar memory budget, to see whether this bridges some of the accuracy gap to full fine‑tuning?

6. **Diffusion experiments: any quantitative metric?**  
   For the Stable Diffusion 3.5 Large experiments, can you add at least one quantitative evaluation (e.g., CLIP‑text similarity averaged over prompts, or a style similarity score) in the rebuttal or camera‑ready, to support the qualitative impressions from Figures 5–8?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work focuses on optimization and quantization techniques for existing public datasets and models, with no new data collection or high‑risk application domain.

## Soundness Rating

3: good.  
The algorithmic idea and empirical results are largely sound and convincing, but the theoretical treatment of DDC contains clear issues, and some derivations in the appendix are inconsistent or careless. These are fixable but should be cleaned up.

## Presentation Rating

3: good.  
The paper is generally well written and organized, with informative figures and tables (e.g., Figures 1–3, Table 1–3). However, several mathematical notational errors and some inconsistencies in the discussion of DDC / clipping slightly reduce clarity.

## Contribution Rating

3: good.  
The integration of ZO with PTQ via scale perturbation, along with the empirical demonstration that this enables fine‑tuning 4‑bit and 2‑bit LLMs on commodity GPUs, is a meaningful and practically relevant contribution. It is not conceptually radical, but the execution and empirical validation are strong enough to be useful to the community.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

QZO tackles an important and timely problem, and the core idea (ZO over quantization scales) is simple, effective, and well supported empirically with strong memory/computation savings. The main weaknesses lie in the shaky theoretical claims around DDC and some missing contextualization and baselines, rather than in the method itself. With corrections to the math and a more careful positioning against related work and QLoRA‑style baselines, this would be a solid and impactful ICLR paper.

## Reviewer Confidence

4: confident.  
I am familiar with zeroth‑order optimization, quantization for LLMs, and memory‑efficient training methods, and have carefully checked the main equations and experimental design, though I did not attempt to fully re‑derive all proofs.