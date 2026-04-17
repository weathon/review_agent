---
job_id: 8c32afa6-6155-43ae-8a9a-d1ddf94ea665
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 0nvQ5kHXf4.pdf
paper: Efficient Resource-Constrained Training of Transformers via Subspace Optimization
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a low‑rank subspace optimization method (WASI) for efficient training of transformer models under resource constraints, clearly within ICLR’s core topics of representation learning, optimization, and efficient large‑scale training / on‑device learning.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present. The method is technically non‑trivial, the math is mostly coherent, and the experiments are substantial and reproducible enough for a serious review. No obvious fatal methodological flaws or data leakage are apparent.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any attempts at prompt injection, hidden reviewer instructions, or similar manipulation in the paper text.

---

# Expected Review Outcome:

## Summary

The paper proposes WASI (Weight‑Activation Subspace Iteration), a framework for training transformer models under tight memory and compute constraints by jointly compressing both weights and activations into low‑rank subspaces during fine‑tuning. The method uses one initial SVD per layer to determine a rank that meets a target explained‑variance threshold, then replaces repeated SVD with warm‑started subspace iteration for both weights (WSI) and activations (ASI), integrating these into forward and backward passes. Experiments on ViT, Swin Transformer, and TinyLlama across several vision datasets and one QA dataset, including on‑device evaluation on Raspberry Pi and Jetson boards, show substantial reductions in memory (up to 62×) and FLOPs (up to 2× in main vision experiments, more on TinyLlama) with small or negligible accuracy loss compared to vanilla training and prior compression baselines.

## Strengths

1. **Joint treatment of weights and activations in a low‑rank training scheme.**  
   Most prior work on on‑device / resource‑constrained training focuses either on weights (LoRA, SVD‑LLM, ASVD, etc.) or on activations (AMC, ASI). WASI provides a coherent formulation that simultaneously compresses both, which is conceptually clean and practically important. Equations (8)–(11) clearly show how the low‑rank factors \(L_i, R_i\) enter both forward and backward passes, and Appendix A.1 spells out the low‑rank gradient computation for 3D and 4D activations.

2. **Well‑executed complexity analysis and its visualization.**  
   Section 3.4, together with Appendix A.3, derives closed‑form expressions for memory and FLOPs in vanilla vs WASI (e.g., \(F_{\text{vanilla}}\) vs \(F_{\text{WASI}}\) in Eqs. (33) and (35), and \(C_{\text{training}}, S_{\text{training}}\) in Eqs. (39)–(46)). **Figure 2** uses these formulas to provide a parametric view of how compression and speedup behave as rank \(K_i\) varies, which helps the reader understand when WASI is likely to be beneficial and when it degenerates to vanilla.

3. **Good empirical validation of the “stable subspace” intuition for weights and activations.**  
   The central hypothesis is that both weights and activations lie in relatively stable low‑dimensional subspaces during fine‑tuning. **Figure 3(a)** nicely visualizes the near‑constancy of singular values of a ViT layer’s weights across epochs, and **Figure 3(b)** shows that reusing the subspace with WSI achieves comparable or better accuracy at substantially reduced FLOPs relative to recomputing truncated SVD every iteration. **Figure 4**’s 3D bar plots of explained variance of activation singular values across layers and modes give convincing evidence that energy is indeed concentrated in a few leading components, supporting the ASI part of WASI.

4. **Strong empirical results on resource metrics with competitive accuracy.**  
   The main ViT–CIFAR‑10 results in **Figure 5** and **Table 1** demonstrate substantial training‑time memory reduction (up to ~60× when all linear layers are compressed, Table 1) and ~3× reduction in training FLOPs at moderate \(\varepsilon\), while keeping accuracy within 1–3% of vanilla. In **Figure 6**, for SwinT across multiple datasets, WASI yields up to 62× memory reduction and 1.5× FLOPs reduction at \(\varepsilon = 0.9\) with essentially no accuracy drop, even surpassing vanilla on CUB. **Figure 7** is particularly impressive: when fine‑tuning TinyLlama on BoolQ, WASI attains up to ~954× activation memory and ~30× weight memory savings, along with >10× FLOPs savings, at no apparent accuracy loss.

5. **Real on‑device evaluations rather than only simulation.**  
   Section 4.4 and **Figure 8** plus **Tables 2–3** report actual training and inference latencies on Raspberry Pi 5, Jetson Orin, Jetson Nano, and Raspberry Pi 4. WASI consistently reduces per‑iteration runtime vs vanilla (e.g., ViT on Raspberry Pi 5: ~1.4× speedup in training and inference even at \(\varepsilon = 0.9\)). This is important validation that the theoretical and simulated FLOP/memory savings translate to wall‑clock benefits under realistic constraints.

6. **Clear exposition of the low‑rank backpropagation for higher‑order activations.**  
   Appendix A.1 carefully unrolls \(f_{\text{LR}}\) for both 3D and 4D activation maps, going from Eq. (13)–(18) (3D) to Eq. (20)–(26) (4D). This demonstrates non‑trivial implementation work to extend ASI‑style decompositions beyond the 3D case, which is essential to handle SwinT’s 4D activations and is more technically involved than just “doing SVD per mode”.

7. **Reasonable robustness across seeds and datasets.**  
   **Figure 9** reports mean ± std over three seeds for ViT on Pets across \(\varepsilon\), showing low variance in both accuracy and memory, which is consistent with the algorithm being mostly deterministic once ranks are fixed. Additional experiments in **Figure 10** and **Figure 11** extend trends to more datasets and show qualitatively consistent trade‑offs.

## Weaknesses

1. **Limited theoretical justification for “subspace stability” and error control.**  
   The central motivation in Section 3.3 is that “the intrinsic subspace remains relatively stable” and that truncation error is controlled by an explained‑variance threshold \(\varepsilon\). However, beyond citing prior work and providing single‑dataset visualizations (**Figure 3(a)**, **Figure 4**), there is no formal argument linking:
   - stability of singular vectors across iterations,  
   - stability of ranks \(K_i\), and  
   - the effect of truncation error (\(\varepsilon\)) on training loss or final accuracy.  
   For example, the variance‑explained criterion in Eqs. (5)–(7) bounds the Frobenius reconstruction error of \(\mathcal{W}_i\), but the paper does not quantify how this error propagates through gradients in Eq. (3) and the low‑rank gradients in Eq. (10)/(18). Even a simple bound comparing \(\|\widetilde{\frac{\partial \mathcal{L}}{\partial \mathcal{A}_i}} - \frac{\partial \mathcal{L}}{\partial \mathcal{A}_i}\|_F\) to \((1-\varepsilon)\) would strengthen the claims about “controlled information loss”. As is, the choice of \(\varepsilon\) remains largely empirical.

2. **Rank‑selection and perplexity‑based ASI are underspecified in the main paper and somewhat opaque.**  
   The paper claims in Section 3.3 that a “dynamic‑programming strategy” determines activation ranks \(\mathbf{r}_i\) by minimizing memory under a target pre‑tuning perplexity, reducing search from exponential to linear. However, the main text gives almost no concrete description, deferring to Appendix A.2. There, the perplexity \(\mathcal{P}_{i,j}\) (Eq. (28)) is defined as the Frobenius norm difference between exact and compressed weight gradients, and rank selection is formulated in Eqs. (29)–(32).  
   Problems:
   - Eq. (30) defines a memory‑bounded optimization (\(\sum_i M_i \le \mathcal{B}\)) for ASI, whereas Eq. (32) for WASI removes the memory constraint and just minimizes total memory. It is not clearly explained how this aligns with the stated “target perplexity” objective.  
   - The claim that the DP search is linear in the number of candidate thresholds \(E\) needs more concrete complexity analysis; there is still a dependence on the number of layers and modes, and it is not obvious that all relevant rank configurations are even considered.  
   - There is no empirical ablation isolating the effect of this rank‑selection scheme versus, say, fixed‑rank or heuristic ranks; we only see performance vs. \(\varepsilon\) but not against alternative rank strategies.

3. **Some mathematical and algorithmic details are confusing or appear inconsistent.**  
   A few examples that matter for reproducibility and correctness:
   - In **Algorithm 1 (WSI)**, line 6 sets \(R_i^T_t = W_i^T_t \cdot L_i(t-1)\), but \(R_i(t)\) is never explicitly orthogonalized or normalized. In Eq. (6)–(7), \(R_i = V_{i,(K_i)}^{T}\), which is orthonormal, but Algorithm 1 only orthogonalizes the product \(W_i(t) R_i^T_t\) to obtain \(L_i(t)\). The resulting factorization \(L_i(t) R_i(t)\) is not guaranteed to match the explained‑variance condition \(\varepsilon\) over time. Some discussion of how numerical drift affects the approximation quality (or whether re‑SVD is occasionally required) is missing.  
   - In Appendix A.1, Eq. (24) for 4D activations has a likely indexing error: \(\mathcal{Z}^{(3)}_{r_1,h,r_3,o} = \sum_{r_3=1}^{\mathbf{r}_3} \mathcal{Z}^{(1)}_{b,h,w,o}\tilde{U}^{(3)}_{w,r_3}\) mixes indices \((b,w)\) inconsistently with the summation and output indices. Similar issues appear in Eq. (25), where the summation index over \(r_4\) is not explicitly shown. While the broad idea is clear, these mis‑indexed equations make it harder to verify the correctness of \(f_{\text{LR}}\).  
   - In Section 3.4, the assumption that the “same optimal rank is applied to both \(\mathcal{A}_i\) and \(\mathcal{W}_i\)” is strong and not obviously satisfied by the actual implementation, which uses separate schemes for weight ranks \(K_i\) (based on \(\varepsilon\) for weights) and activation ranks \(\mathbf{r}_i\) (based on perplexity). This approximation is fine for an illustrative figure, but the paper should state this explicitly in the caption of **Figure 2** to avoid confusion.

4. **Baselines and positioning relative to closely related work are incomplete.**  
   The paper compares against ASI, SVD‑LLM, and vanilla, which is a good start, but given the focus on resource‑efficient training of transformers via low‑rank / subspace compression, some very relevant recent works are missing:
   - Methods that *simultaneously* compress gradients and activations into low‑rank subspaces for training efficiency, e.g., INSTANT: Compressing Gradients and Activations for Resource‑Efficient Training (Doan et al., 2026), which conceptually resembles WASI’s focus on both activations and parameter updates and should at minimum be discussed in Section 2 and compared in the experimental discussion.  
   - Transformer‑specific tensor‑compressed optimization for on‑device / FPGA training such as “Ultra Memory‑Efficient On‑FPGA Training of Transformers via Tensor‑Compressed Optimization” (Tian et al., 2025), which is directly in the same problem space. Even if hardware is different, their tensor compression and training‑time strategies should be contrasted with WASI in Related Work.  
   Moreover, in the main experiments SVD‑LLM is adapted from an LLM‑oriented method to ViT without much detail on how its truncation‑aware whitening is implemented for 3D activations here. **Figure 5** shows that SVD‑LLM can even consume *more* memory than vanilla at low compression; this should be explained more carefully to ensure the baseline is configured optimally and fairly.

5. **Experimental scope is somewhat narrow relative to the claimed generality.**  
   While the vision experiments are fairly extensive, several limitations reduce how broadly we can generalize the claims:
   - All main end‑to‑end tasks are image classification on relatively small datasets (CIFAR‑10/100, Pets, CUB, Flowers). There are no experiments on harder long‑sequence or dense prediction tasks, where transformer subspace structure and rank behavior might differ (e.g., object detection, segmentation, or long‑document LMs).  
   - For TinyLlama, only the last 1–5 decoder layers are fine‑tuned at a single \(\varepsilon = 0.1\); **Figure 7** is compelling in resource terms, but without more ablations (different \(\varepsilon\), more layers, or a stronger LLM benchmark) it remains more of a proof‑of‑concept than strong evidence that WASI works well for large language models.  
   - There is limited analysis of per‑layer rank distributions and where accuracy starts to break (e.g., what happens at \(\varepsilon = 0.2\) or 0.3 on SwinT; which layers are most sensitive). **Table 1** provides a nice sweep of \(\varepsilon\) for ViT, but similar tables for SwinT or TinyLlama would clarify robustness.

6. **On‑device evaluations lack some key baselines and metrics.**  
   **Figure 8** and **Tables 2–3** show WASI’s latency improvements over vanilla and over ASI for ViT on CIFAR‑10. However:
   - There is no on‑device comparison against SVD‑LLM (even in the ViT case), which would give a more complete view of the training‑side vs inference‑side trade‑offs between weight‑only and joint weight‑activation compression.  
   - **Table 4** reports energy consumption for WASI on Jetson Orin only, without any vanilla or ASI/SVD‑LLM energy numbers. Since energy is a key claimed benefit, showing relative improvements (e.g., J per iteration vs vanilla) would significantly strengthen the argument.  
   - All timing measurements are for a single batch size and model configuration. Some sensitivity analysis w.r.t. batch size or number of fine‑tuned layers (especially for TinyLlama) would help practitioners understand where WASI is most effective.

7. **Minor clarity and editorial issues.**  
   There are various smaller problems that reduce polish: duplicated or mis‑attributed references (e.g., Clark et al. and Dhar et al. entries in the references list are garbled), some typos (“forecasts the speedup ratios”, inconsistent use of bold/italic for variables), and a few awkward phrasings. **Figure 1** is conceptually useful as a high‑level pipeline diagram, but some of its notation (e.g., arrows labeled with \(\tilde{\mathcal{A}}_i\), \(\frac{\partial \mathcal{L}}{\partial \mathcal{W}_i}\) in both original and subspace) is hard to parse without a legend stating which paths correspond to compressed vs full tensors.

Overall, none of these issues appear fatal, but they collectively reduce clarity and somewhat weaken the strength of the claims.

## Potentially Missing Related Work

1. **Doan, T., Tran, T., Tartaglione, E. (2026), “INSTANT: Compressing Gradients and Activations for Resource‑Efficient Training”**  
   - *Why related*: INSTANT explicitly compresses both gradients and activations into low‑rank representations to save memory and bandwidth during backpropagation, conceptually similar to WASI’s joint treatment of weights and activations for resource‑efficient training.  
   - *Where to add*: This work should be discussed in Section 2 (perhaps a small subsection “Low‑rank compression for training dynamics”) and referenced when introducing WASI in Section 3.3 as an alternative approach that focuses on gradients versus WASI’s focus on weights and activations.

2. **Tian, J., Lu, J., Li, H. (2025), “Ultra Memory‑Efficient On‑FPGA Training of Transformers via Tensor‑Compressed Optimization”**  
   - *Why related*: This paper addresses the same high‑level goal of training transformers under tight resource constraints using tensor compression, particularly on constrained hardware. Although targeting FPGA rather than general edge devices, it provides a meaningful point of comparison for the design choices in WASI (e.g., which tensors to compress, how to maintain convergence).  
   - *Where to add*: It should be cited in the Introduction (Section 1) when motivating on‑device training of transformers and in Section 2 among methods for low‑rank / tensor decomposition in transformer training, with a brief comparison highlighting similarities and differences in compression targets and hardware assumptions.

## Questions

1. **Stability and re‑SVD schedule.**  
   How often, if ever, do you need to recompute a full SVD on the weights during training to refresh the subspace? In all experiments, do you only compute SVD once at \(t=0\) and then run WSI indefinitely, or is there a periodic re‑initialization schedule? An experiment where you vary the frequency of full SVDs (e.g., every epoch vs every 10 epochs vs never) and measure accuracy / FLOPs would clarify robustness of the “stable subspace” assumption.

2. **Layer‑wise rank profiles and sensitivity.**  
   Could you provide more detailed plots or tables showing the per‑layer ranks \(K_i\) and activation ranks \(\mathbf{r}_i\) for ViT and SwinT under a given \(\varepsilon\)? This would help understand which layers are most compressible and how much variation there is across layers. Also, do some layers particularly benefit from higher ranks (e.g., last layers, attention projections) while others can be compressed more aggressively without loss?

3. **Comparison to INSTANT‑style gradient compression.**  
   Conceptually, how does WASI compare to compressing only gradients and activations (as in INSTANT) versus compressing weights and activations? For example, could WASI be combined with gradient compression for additional gains, or do the approximations interfere? Even a small ablation where you only apply WSI or only ASI or only gradient compression (if feasible) on ViT‑CIFAR‑10 would be very informative.

4. **Energy measurements relative to baselines.**  
   In **Table 4**, you report absolute energy for WASI at different \(\varepsilon\), but not for vanilla or ASI. Could you provide these comparisons and perhaps an energy vs accuracy plot similar to **Figure 5**? This would better substantiate claims about carbon / energy savings in on‑device training.

5. **Applicability to larger LLMs and sequence tasks.**  
   Given the promising TinyLlama results in **Figure 7**, what are the main obstacles to scaling WASI up to large LLMs (e.g., 7B parameters) on GPU clusters, aside from pure hardware availability? Are there algorithmic or numerical issues (e.g., subspace instability in deeper layers, mismatch between weight and activation ranks) that you anticipate? Any preliminary evidence, even on simulated smaller models, would help assess generality.

Author responses that clarify these points, especially around theoretical error control, subspace refresh policies, and additional baselines/energy comparisons, would increase confidence in the method and might justify an even stronger recommendation.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is technically sensible and consistent with known properties of low‑rank approximations and subspace iteration; derivations are mostly correct, and the experimental evidence is solid, though some theoretical claims (error control, stability) remain heuristic and some mathematical details could be clearer.

## Presentation Rating

3: good.  
The paper is generally well structured and readable, with informative figures such as **Figures 2–8** and detailed appendices, though some notation is inconsistent and certain algorithmic details (rank selection, 4D backprop expressions) need clarification and minor corrections.

## Contribution Rating

3: good.  
WASI provides a useful and nontrivial combination of weight and activation low‑rank training tailored to transformer on‑device learning, with strong empirical results and clear practical impact, even if the conceptual novelty over prior activation/weight compression work is incremental rather than transformative.

## Overall Rating

8: Accept, good paper (poster).  
The paper presents a well‑executed, practically important method for resource‑constrained training of transformers, with convincing empirical gains and thoughtful complexity analysis. While the theoretical underpinnings and comparisons to some closely related work could be strengthened, the contribution is substantial and will likely be valuable to the ICLR community studying efficient training and on‑device learning.

## Reviewer Confidence

4: confident.  
I am familiar with low‑rank methods, activation compression, and efficient transformer training, and I have carefully checked the main derivations and experiments, though I did not fully re‑derive every formula in the appendices.