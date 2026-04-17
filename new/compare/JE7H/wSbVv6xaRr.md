---
job_id: 0230f6c8-a8e0-4bc0-b990-f3218f2f7c19
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: wSbVv6xaRr.pdf
paper: Communication-Efficient and Private Federated Learning via Projected Directional Derivative
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is clearly about optimization, communication-efficient FL, and privacy against gradient inversion, which are all within ICLR’s core scope.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion) are present. The work is technically nontrivial, in English, with a substantial theoretical and empirical component; while there are issues, they do not rise to the level of automatic desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no indications of hidden prompts, attempts to steer LLM-based reviewing, or other manipulative content in the provided manuscript.

---

# Expected Review Outcome:

## Summary

The paper proposes FedMPDD, a federated learning algorithm that replaces full gradient uploads with “multi-projected directional derivatives.” Each client samples \(m\) random Rademacher directions, sends only the scalar directional derivatives plus a seed, and the server reconstructs an unbiased gradient estimator by re-generating the directions. The authors analyze convergence, derive dimension- and \(m\)-dependent reconstruction/error bounds that they interpret as privacy guarantees against gradient inversion attacks, and empirically compare FedMPDD to FedSGD and several compression and LDP methods on MNIST, FMNIST, and CIFAR-10 under various gradient inversion attacks.

## Strengths

1. **Conceptual idea and technical formulation are clear and reasonably original.**  
   Using projected directional derivatives in FL, with client-side transmission restricted to scalar inner products and server-side reconstruction via seeds, is a neat twist on random projections / sketching. The formal definition in **Definition 1** and the unbiasedness argument following it are straightforward and technically sound. The extension from single projection (FedPDD) to multi-projection (FedMPDD) and the explicit mapping \(\hat g_i(x_k) = \frac{1}{m} U_{k,i} U_{k,i}^\top g_i(x_k)\) on **Page 4–5** is well explained.

2. **Theoretical analysis of convergence is fairly complete and transparent.**  
   The paper gives a clear negative result for the single-direction FedPDD in **Theorem 1** and **Remark 3**, showing an \(O(d/\sqrt{K})\) rate driven by \(\sqrt d\) variance amplification (equation right after (3)). This sets up the multi-direction solution FedMPDD and **Theorem 2**, which leverages the Johnson–Lindenstrauss lemma (**Lemma 6**) to show an \(O(1/\sqrt{K})\) rate when \(m = O(\log(d/\delta)/\varepsilon^2)\). The derivations in **Appendix E** are detailed enough to follow, and the decomposition from (21) through (30)/(34) is standard but correctly executed.

3. **Privacy analysis is mathematically explicit (even if the notion of privacy is weaker than DP).**  
   Unlike many FL compression papers that hand-wave privacy, this work actually computes reconstruction errors. **Lemma 1** derives \(\mathbb{E}\|\hat g_i - g_i\|^2 / \|g_i\|^2 = (d-1)/m\), and **Lemma 2** translates that into a lower bound on input reconstruction error under a Lipschitz assumption. The multi-round bound in **Appendix D** (Theorem 2 there, “worst-case multi-round privacy bound”) quantifies that unique gradient recovery is impossible if \(T m < d\). While this is not a DP-style guarantee, it is a precise linear-algebraic statement about ambiguity that is nice to see.

4. **Empirical evaluation of privacy vs. communication is fairly systematic.**  
   The paper goes beyond accuracy-only metrics and evaluates privacy with SSIM under two gradient inversion attacks (Yu et al. 2025, Zhu et al. 2019). **Figure 2** is particularly informative: the left panel shows SSIM evolution over iterations, and the right panel qualitatively compares reconstructed CIFAR-10 images across FedSGD(+noise), QSGD, and FedMPDD. FedMPDD images remain essentially unrecognizable while several baselines leak clear structure.  
   Quantitatively, **Tables 1 and 2** provide a compact summary under fixed total byte budgets and fixed target accuracy. FedMPDD consistently achieves far better accuracy at the same budget, and dramatically lower SSIM (e.g., Table 2: SSIM ~0.14–0.22 vs 0.74–0.96 for compressive baselines).

5. **Communication analysis and experiments are concrete and multi-faceted.**  
   The communication accounting is consistent: they compare total uplink bytes required to reach a target accuracy or under a fixed budget, not only per-round costs. **Figure 3** (LeNet on MNIST, IID) and the numerous plots in **Figures A.1–A.9** clearly separate “accuracy vs. rounds” and “accuracy vs. bits transmitted,” and show that FedMPDD curves are shifted significantly left in the bits axis relative to FedSGD and QSGD. Table A.9 (effect of varying \(m\)) further illustrates the dependence of accuracy on the number of directions, matching Theorem 2’s claim that too-small \(m\) hurts convergence.

6. **Some attention to computational cost on client side.**  
   The authors do not ignore the cost of computing \(m\) inner products. **Remark 1** and **Appendix F** discuss using Jacobian–vector products (projected-forward approaches) and compare complexity (Table F.1). **Table A.10** gives empirical per-client latency per round for various \(m\), showing sub-millisecond overheads for LeNet, which is plausible and reassuring.

7. **Figures are generally well-aligned with the narrative.**  
   - **Figure 1** (SSIM vs epochs on MNIST/LeNet with \(m=600\)) supports the claim that privacy level is time-invariant; the SSIM values remain around ~0.02–0.03 for 100 epochs.  
   - **Figure 3** and its siblings in Appendix A (A.1–A.9) clearly show that, plotted against bits, FedMPDD often reaches a given accuracy in significantly fewer bits than quantization and sparsification baselines.  
   - **Figure A.16** further confirms the “constant privacy” story across epochs using another dataset/architecture.

## Weaknesses

1. **Privacy notion is ad hoc and not grounded in standard formal frameworks.**  
   The central “privacy guarantees” are based on gradient reconstruction error and linear-algebraic underdetermined systems, but there is no formal privacy definition (e.g., differential privacy, mutual-information bounds, or any adversary-knowledge–based indistinguishability). **Lemma 1**’s \((d-1)/m\) error and **Lemma 2**’s lower bound rely on a fixed gradient and Lipschitzness in the input; they do not consider adaptive attackers or realistic priors over data. The honest-but-curious threat model in **Definition 2** is extremely broad, yet the guarantees are proved only for a *specific* inversion loss and a specific optimization attacker. For example, Lemma 2 assumes the adversary minimizes exactly \(\mathcal L(\hat v)\) involving the FedMPDD projection; a different attack objective could, in principle, exploit more side information. The paper repeatedly phrases these as “formal defense” and “fundamental privacy guarantee,” which feels overstated relative to what is actually proven.

2. **Multi-round privacy composition is fragile and under-discussed.**  
   The worst-case composition result in **Appendix D** requires \(T m < d\) to avoid unique recovery when the gradient is static. In realistic settings with moderately sized \(m\) and long-lived clients, this bound can be binding. For instance, with a 300k-parameter CNN and \(m=2000\) (used in **Table 2**, CIFAR-10), the bound allows up to ~150 rounds before the worst-case guarantee evaporates. Many FL deployments run for far more rounds or epochs. The main text’s **Remark 2** acknowledges this but then downplays it by saying gradient evolution “provides stronger practical protection” without any supporting analysis, even though this undermines the claim of “consistent privacy” over training. This is a significant limitation that should be front-and-center in the discussion, especially for recurrent attacks that aggregate information over time.

3. **Comparison to Local Differential Privacy and DP mechanisms is incomplete and somewhat unfair.**  
   The LDP comparison is mostly limited to additive Laplace/Gaussian noise on full gradients, with no end-to-end privacy accounting and no competing scheme that also compresses. In **Table 2**, for example, “FedSGD + Laplace(var=10)” is treated as a valid privacy baseline, but the chosen variance has no calibration to any \((\varepsilon,\delta)\)-DP budget; it is simply tuned until reconstructions fail. This is apples-to-oranges relative to FedMPDD, whose “privacy level” is not translated to any standard privacy parameter either. **Remark 5** formalizes a “relative reconstruction error” metric for LDP and criticizes dependence on \(\|g\|\), but the same metric is not applied in a rigorous way to FedMPDD under a comparable adversarial model. So while the empirical SSIM curves (**Figures 1,2,A.10–A.16**) are compelling qualitatively, the theoretical “LDP is fluctuating and ours is consistent” message feels too strong relative to the evidence.

4. **Theoretical bounds for privacy and for convergence are only loosely coupled to the practical parameter choices.**  
   - **Theorem 2** requires \(m = O(\log(d/\delta)/\varepsilon^2)\) with some distortion \(\varepsilon\), but the experiments set \(m\) heuristically (e.g., 0.2%–4% of \(d\)) without reporting the implied \(\varepsilon,\delta\), nor checking whether the JL norm-preservation bound actually holds numerically. **Table A.9** explores some \(m\) values, but this is more of an empirical sweep than an explicit validation of the theoretical scaling.  
   - On the privacy side, Lemma 1’s \((d-1)/m\) relative error decays linearly in \(1/m\), yet some experimental choices use relatively large \(m\) (e.g., \(m=2000\) on CIFAR-10), where the relative error is no longer huge. Table 2 still shows low SSIM for those \(m\), but there is no analysis that bridges the \((d-1)/m\) gradient error to SSIM scores, nor any study of how close one is to the “privacy cliff” where increasing \(m\) starts making inversion effective. The privacy-utility trade-off discussion on **Page 7** is qualitatively correct but remains high level.

5. **Experiments are limited to small-scale vision models and synthetic client settings.**  
   All experiments are on MNIST, FMNIST, or CIFAR-10 with LeNet or small CNNs (Tables H.1–H.3). There are no results on more realistic FL workloads such as ResNet-scale models, language models, or cross-device heterogeneity beyond simple IID/non-IID partitions into 2 classes per client. Given that the central selling point is “communication-efficient and private FL for large-scale models,” it is disappointing that the largest model is ~300k parameters, and that the privacy/communication stories are not stress-tested in truly high-dimensional regimes or with more realistic client dynamics (dropout, varying participation, etc.). This raises questions about how the \(T m < d\) privacy bound and the JL-based convergence claims behave in more realistic settings.

6. **Some aspects of the communication and baseline setup could be more transparent.**  
   - The fixed byte budgets in **Tables 1 and 2** are helpful, but they obscure per-round behavior: FedSGD is marked as exceeding the budget “in the first iteration” (★) without clarifying how many clients, epochs, and local steps are assumed, and whether budget allocation is normalized across methods in a fully fair way (e.g., same number of global updates, same client participation schedule).  
   - Several baselines (Top-k, lp-proj, SA-FedLora) are used only in certain tables and figures (e.g., Table 2, Figure 2, Figure 3), not across the full experimental suite. Hyperparameter tuning for baselines is summarized in Tables H.4–H.7, but there is no discussion of whether additional tuning (e.g., sparsity level \(k\), sketch dimension, LoRA rank) was performed to match the communication constraints optimally, whereas \(m\) for FedMPDD is chosen essentially to satisfy theory. This could bias the reported “FedMPDD dominates across metrics” narrative.

7. **Mathematical and algorithmic presentation has some inconsistencies and minor sloppiness.**  
   - In **Algorithm 2**, line 13 sets \(\Delta_{\mathrm{sim}} \gets \bar{\mathbf{u}}_d\) (a symbol not defined elsewhere), and line 20 uses \(\hat{\mathbf{g}}(x_k) = \frac{1}{\beta N}\Delta_{\mathrm{sim}}\), whereas the accumulation in lines 17 and 19 is done in \(\Delta_{\mathrm{sum}}\). This appears to be a typo and would confuse an implementer; presumably \(\Delta_{\mathrm{sum}}\) is intended everywhere.  
   - Some notation around \(\varepsilon\) in **Theorem 2** and **Equation (5)** is inconsistent: the main text uses \(\epsilon\) interchangeably with \(\varepsilon\), and the JL lemma in **Lemma 6** uses \(\varepsilon\) while the dependence in FedMPDD is written as “distortion parameter \(\epsilon\)” without being explicit whether this is the same quantity.  
   - In **Equation (10)** (Remark 5, LDP reconstruction error), the denominator is written as \(\|\mathbf g_i(x_k)\|^2\) but the text immediately after speaks of “relative reconstruction error” without clarifying whether they are taking expectations over gradient noise, data, or both. Similar issues appear in a few other places (e.g., in the proof of **Lemma 2** where \(\mathcal L(\hat v^\star)=0\) is assumed in one inequality and then expectations are taken).

8. **Privacy interpretations over-claim relative to what is proved.**  
   The text repeatedly uses strong language: “fundamental advantages over LDP,” “consistent privacy without harming utility,” “uniform privacy protection regardless of the magnitude of the clients’ gradients” (end of **Page 6**, **Remark 5**), and “constant privacy guarantee throughout the entire training process” (caption of **Figure A.16**). All of these rely on the single-round \((d-1)/m\) reconstruction error and on ignoring multi-round composition beyond the crude \(Tm<d\) condition. As soon as gradients or directions are correlated across rounds, or the attacker is allowed to jointly optimize across rounds, the effective ambiguity can shrink substantially before \(Tm\) reaches \(d\). This does not mean FedMPDD is insecure, but the claims as phrased are too absolute and will mislead readers who might conflate this with something like formal DP.

9. **Missing and under-discussed related work on joint communication and privacy in FL.**  
   The related-work section focuses heavily on sketching, structured updates, and DP-compression hybrids (e.g., Amiri et al. 2021, Lyu 2021), but misses several recent lines of work that also explicitly aim at *joint communication efficiency and privacy*, including distillation-based and DP-based methods (see next section). This weakens the positioning: some of these works also provide tunable trade-offs and could be used as baselines in Tables 1–2.

10. **Figures do not fully separate the different axes of comparison (privacy vs. utility vs. bits) for all methods.**  
    The figures are visually clear but sometimes conflate aspects. For example, in **Figure 2**, the left panel plots SSIM vs. iterations across methods, but it is hard to see for which methods the total bit budget is comparable to FedMPDD, and whether the comparison is at equal-accuracy or equal-bits. **Figure 3** nicely separates “versus rounds” and “versus bits” for the LeNet-MNIST case, but other figures in **Appendix A** (A.1–A.9) crowd many methods with overlapping confidence bands, making it tricky to see, for instance, where exactly FedMPDD overtakes Top-k or lp-proj in bits-to-accuracy trade-off. Some additional highlighting or summarized break-even points in the main text would help.

## Potentially Missing Related Work

(All of the following works are directly related to the claimed joint goals of communication efficiency and privacy in federated learning and are not cited in the paper.)

1. **Wu, C., Wu, F., Lyu, L. (2022). “Communication-Efficient Federated Learning via Knowledge Distillation.”**  
   - Relevance: Proposes FedKD, which reduces communication via knowledge distillation while maintaining accuracy. It is conceptually similar in that it trades gradient communication for compressed statistics while aiming to preserve performance, and can provide some privacy by not transmitting raw gradients.  
   - Where to cite: Add to the “Communication Cost Management” and “Related Work” sections (Section 1 and Section “Related Work”) as another major class of communication-efficient FL beyond gradient compression and sketching. It would also be natural to mention when discussing SA-FedLora and other parameter-efficient personalization methods.

2. **Li, Y., Du, W., Han, L. (2023). “A Communication-Efficient, Privacy-Preserving Federated Learning Algorithm Based on Two-Stage Gradient Pruning and Differentiated Differential Privacy.”**  
   - Relevance: Explicitly tackles both communication reduction and privacy via structured gradient pruning and differential privacy. The problem statement is extremely close to this paper’s stated goal of “jointly improved communication efficiency and privacy.”  
   - Where to cite: In the “Statement of Contribution” on **Page 2–3** and in the “Related Work” section, as a directly comparable joint comm+privacy method. It should also be discussed qualitatively in the privacy experiments (Tables 1–2) as a potentially relevant baseline design (even if not implemented).

3. **Guo, S., Yang, J., Long, S. (2024). “Federated Learning with Differential Privacy via Fast Fourier Transform for Tighter-Efficient Combining.”**  
   - Relevance: Uses FFT-based techniques to combine DP-noisy information efficiently, focusing on privacy–utility–communication trade-offs. This is very much in the same design space as FedMPDD, which also uses a structured transform (random directions) to reduce individual gradient exposure.  
   - Where to cite: In the privacy-related part of Section 1 and the comparison vs. LDP in **Remark 5**, as another approach that tries to circumvent naive additive noise inefficiencies.

4. **Shao, J., Wu, F., Zhang, J. (2024). “Selective Knowledge Sharing for Privacy-Preserving Federated Distillation Without a Good Teacher.”**  
   - Relevance: Targets privacy and communication via selective sharing of distilled knowledge instead of gradients, with an explicit focus on privacy-preserving aspects.  
   - Where to cite: In the “Related Work” section as part of “privacy-preserving, communication-efficient FL via distillation and selective sharing,” and compared conceptually in the numerical section as an alternative way to get privacy without directly sending raw or perturbed gradients.

5. **Pan, Y., Chao, Z., He, W. (2024). “FedSHE: Privacy Preserving and Efficient Federated Learning with Adaptive Segmented CKKS Homomorphic Encryption.”**  
   - Relevance: Combines privacy-preserving homomorphic encryption with communication-efficiency techniques, again squarely targeting the same trade-offs as FedMPDD but via cryptographic rather than geometric mechanisms.  
   - Where to cite: In the “Privacy Preservation Measures” part of the introduction and in the “Related Work” section as an example of cryptographic approaches that achieve privacy+efficiency with different assumptions (trusted key management, heavier computation).

## Questions

1. **Clarification of the formal privacy notion.**  
   How do the authors propose that practitioners interpret the \((d-1)/m\) gradient reconstruction error and Lemma 2’s input reconstruction lower bound, in terms of a *formal* privacy guarantee? Is there any way to map these results to a more standard (even if weaker than DP) notion, such as a bound on mutual information between transmitted messages and client data, or an \((\varepsilon,\delta)\)-style indistinguishability metric?

2. **Multi-round attacks beyond the \(Tm<d\) linear system argument.**  
   Can the authors provide empirical evidence (e.g., with Yu et al. 2025 or DLG) of attacks that aggregate information across multiple rounds against FedMPDD, particularly in a setting where \(Tm\) is significantly larger than \(d\)? For example, run an attack that observes the same client across many rounds with shared directions or correlated directions, and see whether SSIM degrades significantly compared to the single-round attack scenario.

3. **Choice and tuning of \(m\) relative to theory.**  
   For the main experimental configurations (e.g., LeNet on MNIST, CNN on CIFAR-10), what are the actual values of \(d\), \(m\), \(\varepsilon\), and \(\delta\) consistent with Theorem 2’s JL requirement? Could the authors provide a small table or figure that shows how the empirical norm-distortion \(\|\hat g_i\|/\|g_i\|\) behaves as a function of \(m\), and how it compares to the theoretical \((1\pm\varepsilon)\) band?

4. **Fairness and tuning of baselines under fixed budgets.**  
   In **Tables 1 and 2**, how were the hyperparameters of QSGD, Top-k, lp-proj, and SA-FedLora chosen to satisfy the communication budget? Were they tuned to minimize communication for a given target accuracy, or were off-the-shelf configurations used? It would help to know whether each baseline had its own “\(m\)/rank/k/bitwidth” parameter tuned in a comparable way to FedMPDD’s \(m\).

5. **Scalability to larger models and longer training.**  
   Do the authors have any preliminary results or insights for models with tens of millions of parameters (e.g., ResNet-18, as used in the motivating example on **Page 2**)? Specifically:  
   - How would they choose \(m\) in that setting to satisfy both Theorem 2 and a reasonable \(Tm<d\) privacy bound?  
   - Are there any numerical stability or memory issues with generating and applying large Rademacher matrices \(U_{k,i}\) on the server side for many clients?

6. **Alternative attacks and threat models.**  
   The privacy lemmas assume a particular optimization-based gradient inversion objective. Have the authors tried stronger or structurally different attacks (e.g., multi-step model inversion that uses auxiliary data, or attacks that adaptively choose dummy inputs across rounds)? If yes, how does FedMPDD perform; if not, could they comment on whether the current proofs can be extended to those scenarios?

7. **Clarification of Algorithm 2 variables.**  
   Can the authors confirm and correct the apparent typos in **Algorithm 2**, particularly around \(\Delta_{\mathrm{sim}}\) vs. \(\Delta_{\mathrm{sum}}\)? This is minor but important for reproducibility.

Answers and clarifications on these points, especially 1–3 and 5, could significantly increase my confidence in the practical relevance and robustness of the method.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The convergence analysis and the gradient reconstruction bounds are technically sound within their assumptions, but the jump from those results to strong, time-uniform privacy guarantees is overstated, and the multi-round privacy analysis is quite weak relative to the claims.

## Presentation Rating

3: good.  
The paper is generally well written, with clear structure and helpful figures/tables, but has some notational inconsistencies, minor algorithmic typos, and over-enthusiastic wording around privacy claims that slightly weaken clarity.

## Contribution Rating

2: fair.  
The idea of reusing multi-projected directional derivatives in FL is reasonably interesting, but the incremental step from existing sketching/projection approaches is moderate, the privacy notion is weaker and less standard than claimed, and experiments are limited to relatively small models and datasets.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper presents a technically solid and clearly described method with a nice convergence story and extensive empirical evidence that directional-derivative–based compression can offer strong empirical resistance to known gradient inversion attacks at much lower communication cost than several baselines. However, the privacy analysis is informal relative to modern standards, the multi-round story is not convincing, and the experimental scope is limited. With a more careful and modest treatment of privacy, stronger baselines from the joint comm+privacy literature, and at least one large-scale experiment, this could become a solid ICLR paper; in its current form it falls slightly short.

## Reviewer Confidence

4: confident.  
I carefully checked the main mathematical derivations, convergence arguments, and reconstruction lemmas, and I am familiar with the FL compression and privacy literature. Some empirical and threat-model nuances could still change my opinion slightly after rebuttal, but major misunderstandings are unlikely.