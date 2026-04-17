---
job_id: f1ae2f17-db69-4731-803c-0429e613cc20
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Y9TgNFsNyP.pdf
paper: FF-Erase: Machine Unlearning and Verification for Forward-Forward Models
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is clearly about machine learning, specifically FF training, machine unlearning, and membership inference, which are squarely within ICLR’s scope (representation learning, optimization, privacy).

## Minimum Quality
Pass ✅.  
The paper has all required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results / Analysis, Conclusion). Experiments are non-trivial and results are reported. I do not see fatal methodological flaws such as test-set leakage.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The only meta-text is an explicit disclosure of LLM use in writing; I see no instructions aimed at manipulating reviewers or LLM systems.

---

# Expected Review Outcome:

## Summary

The paper studies machine unlearning for Forward-Forward (FF) models, arguing that standard gradient-ascent–style unlearning methods designed for backprop-based networks are unstable and can cause “model collapse” when applied to FF training.  

It proposes FF-Erase, which uses a separate “guidance model” trained without the forget set and performs layer-wise KL-based goodness matching to steer the original model’s goodness distributions away from the forgetting data while periodically “recovering” performance on the remaining data.  

The paper also introduces G-MIA, a membership inference attack that uses the layer-wise goodness vectors as features and is proposed as a black-box verification tool for unlearning in FF models; experiments on MNIST/Fashion-MNIST/CIFAR-10/100 and several FF CNN architectures compare FF-Erase to retraining and naïve gradient ascent, and evaluate G-MIA against several MIA baselines.

---

## Strengths

1. **Timely problem: first systematic look at FF unlearning.**  
   The paper tackles an under-explored but important problem: how to perform machine unlearning for FF models, which are increasingly studied as backprop-free alternatives. The discussion in §1 and §2 is useful in articulating why FF layer-wise training and “goodness” make existing BP-oriented unlearning algorithms brittle, and the qualitative illustration in **Figure 1(a,b)** helps communicate the intuition that layer updates can diverge and cause collapse.

2. **Concrete algorithm tailored to FF structure.**  
   FF-Erase is structurally aligned with how FF models are trained. The method operates on layer-wise goodness vectors and uses KL divergence to match the original model’s goodness distribution to that of a guidance model that has never seen the forgetting data (Equations (5)–(6), **Algorithm 1**, **Figure 2(b)**). This is a natural and fairly clean way to adapt distillation-style unlearning to the FF training paradigm instead of only touching final logits.

3. **Goodness-based membership inference as verification for FF.**  
   The idea of using multi-layer goodness vectors as features for a membership classifier (G-MIA) is natural for FF and gives a verification tool beyond pure accuracy. **Figure 3(a–d)** shows that on several models and datasets G-MIA clearly outperforms standard black-box MIAs that only see final outputs (FL) and in some deeper model settings even matches or exceeds the considered white-box attacks, suggesting that goodness vectors capture richer membership signals in FF architectures.

4. **Empirical evidence that naïve gradient ascent is dangerous in FF.**  
   The experiments in **Figure 4** and **Figure 5** support the claim that straightforward GA, even with tuning of the trade-off parameter λ, either collapses the model or fails to forget. For example, in **Figure 5(a,b)**, GA with large λ destroys test accuracy, and with small λ maintains high forget-set accuracy (~84%) and high G-MIA scores (**Figure 5(c)**), indicating weak unlearning. This is valuable empirical evidence for the community that “just run GA” is not safe in FF models.

5. **Systematic efficiency–effectiveness tradeoff study for guidance models.**  
   **Table 1** is a strong point: it quantifies how different guidance strategies (mini-retraining vs fast-distillation) and their hyperparameters (α₁, α₂) affect total unlearning time, G-MIA ACC/AUC, and accuracy on both forgetting and test sets. The table convincingly shows that (a) a randomly initialized guidance model (R.G.M) severely harms utility (test accuracy drops to ~55%), and (b) using reasonably small α₁, α₂ (e.g., R-(0.5,0.2) or D-(0.3,0.2)) achieves forget/test performance close to full retraining while using ~40% of the retrain time.

6. **Layer-wise analysis of residual knowledge.**  
   The CKA analysis in **Table 3** (Appendix C.2) is insightful: it shows that retraining changes middle layers most, and that FF-Erase consistently reduces representational similarity to the original model across layers while avoiding the extreme representation drift seen in GA(λ=10) or FYE (which essentially fully collapse). This layer-wise view ties nicely back to the FF-specific argument that unlearning must handle layer-wise residual information.

7. **Breadth of experimental coverage (variants & ablations).**  
   While limited to small/medium-scale vision benchmarks, the empirical section is reasonably thorough in terms of axes explored: several datasets (MNIST, FMNIST, CIFAR-10/100), multiple FF backbones (TinyCNN, AlexNet, VGG-variants; see **Figure 8** and **Figure 9**), exploration of GA hyperparameters, multiple unlearning baselines in the appendix (FATS, FYE, SURE, Bad Teacher in **Figure 10**), and the impact of guidance training schedules (**Figure 11**).

---

## Weaknesses

1. **Core FF methodology is under-specified and somewhat inconsistent mathematically.**  
   - In **Equation (1)**, \(g^l = \|\boldsymbol{h}^l\|_1\) is defined as an L1 norm of the activation vector, which is a scalar, yet immediately after the text states that \(\boldsymbol{g}^l = [g_1^l,\dots,g_J^l]\) is a vector of class-wise goodness scores. It is unclear how the per-class structure arises from a single L1 norm; presumably there are J channels or label-embedding concatenations, but that is not formalized.  
   - This shape inconsistency propagates into **Algorithm 1**, where `Norm(h^l)` is used to compute \(g^l\) and \(\boldsymbol{g}_*^l\), and then a KL divergence between `[g^l]` and `[g_*^l]` is computed. If \(g^l\) is scalar, the KL is degenerate; if it is a J-dimensional vector, Equation (1) is incorrect.  
   - Similarly, in the normalization \(z^l = (h^l - g^l)/(\sqrt{\sigma^2} + \epsilon)\), subtracting a scalar \(g^l\) from a vector \(h^l\) is not explained: is this per-channel offset or per-class goodness broadcast? The current notation makes it hard to reconstruct the exact FF architecture, which is critical for reproducibility and for understanding how goodness flows during unlearning.

   These inconsistencies are not just cosmetic; they obscure whether the KL-based loss in **Equation (5)** is properly defined as a divergence between probability vectors over classes, and how layer-wise gradients relate to the high-level objective in **Equation (4)**.

2. **No formal connection between the FF-Erase algorithm and the stated unlearning objective.**  
   The formal objective in **Equation (4)** is a generic “forget vs retain” loss over the entire model, whereas FF-Erase’s actual update rules, **Equations (5)–(6)**, minimize KL\((\bm{g}^l,\bm{g}_*^l)\) for forgetting data and perform standard FF loss minimization for remaining data. There is no derivation showing that these updates correspond to a descent-ascent procedure on (4), or that they approximate the effect of retraining on \(\mathbb{D}_{\mathrm{remain}}\) in any quantifiable sense.  
   For example, one could ask whether, in the limit of sufficiently trained guidance models (e.g., \(\theta_g = \theta_r\)), the procedure converges to \(\theta_r\), or at least whether the unlearned model’s predictions on \(\mathbb{D}_{\mathrm{forget}}\) and \(\mathbb{D}_{\mathrm{remain}}\) are bounded relative to retraining. Currently the method is entirely heuristic, and the paper does not attempt even a simple analysis of convergence, fixed points, or stability beyond empirical plots.

3. **Theoretical and notational error in the CKA formula.**  
   In **Equation (12)**, the CKA similarity is written as  
   \[
   \mathrm{CKA}(X_i^a, X_i^u)
     = \frac{\mathrm{HSIC}(X_i^a, X_i^u)}
     {\sqrt{\mathrm{HSIC}(X_i^a, X_i^a)\,\mathrm{HSIC}(X_i^a, X_i^u)}},
   \]
   where the second term in the denominator is again \(\mathrm{HSIC}(X_i^a, X_i^u)\) instead of \(\mathrm{HSIC}(X_i^u, X_i^u)\). This is inconsistent with the standard CKA definition (e.g., Kornblith et al. 2019) and mathematically unsound, since it would simplify to \(\sqrt{\mathrm{HSIC}(X_i^a,X_i^u)/\mathrm{HSIC}(X_i^a,X_i^a)}\) rather than a proper normalized similarity.  
   While the experiments might have used the correct formula in code, the text as written is incorrect and should be fixed. Given that **Table 3**’s CKA scores are used to argue fine-grained layer-wise behavior, the authors must ensure that the implementation and formula are consistent and that conclusions do not rest on a mis-specified measure.

4. **G-MIA’s “black-box” assumption is much stronger than acknowledged and not well grounded in realistic threat models.**  
   The paper claims G-MIA is a black-box attack, but it assumes the attacker can query *all layer-wise goodness vectors* of the target FF model. In typical API-style deployments, exposure is limited to final predictions or at best logits; access to internal per-layer activations is normally considered semi- or full white-box.  
   The comparison in **Figure 3** also reflects this asymmetry: G-MIA sees an entire stack of per-layer goodness vectors across L layers, while the FL baseline sees only final-layer outputs. White-box baselines (GR, GAP, ST) use gradients or full activations, but from an access-control perspective, G-MIA is closer to these than to FL. Calling it “strict black-box” is therefore misleading. This matters because G-MIA is then used as the *main* verification metric for unlearning effectiveness. A more honest positioning (e.g., “intermediate-access attack” vs “logit-only black box”) and discussion of whether FF deployments would expose goodness vectors is needed.

5. **Verification story is narrow relative to the fast-evolving unlearning verification literature.**  
   The paper largely positions G-MIA against standard MIAs (Shokri et al., Nasr et al., etc.), but does not engage with the recent dedicated verification work that directly targets unlearning robustness and residual knowledge. For example:
   - UMA / Unlearning Mapping Attack (Xuan & Li, 2025)  
   - IndirectVerify using influential sample pairs (Xu et al., 2024)  
   - SMS for self-supervised model seeding (Wang et al., 2025)  
   - “Verification of Machine Unlearning is Fragile” (Zhang et al., 2024)  
   - “Mirror Mirror on the Wall…” (Brimhall et al., 2025)  
   - Explainable-AI-based verification (Pujol Vidal et al., 2024)  

   None of these are cited or discussed, even at a conceptual level. This omission weakens the claim that G-MIA constitutes a “reliable tool for unlearning verification” (§5), and it also leaves unaddressed known issues such as verification brittleness and dependence on evaluation protocols (see also Tu et al., 2024, which is cited but not deeply connected to the G-MIA design).

6. **Experimental setup and statistical robustness are limited.**  
   - The main text focuses heavily on a single configuration (VGG13, CIFAR-10) in **Figure 4** and **Figure 5**, with other results relegated to the appendix.  
   - Reported gains in G-MIA ACC/AUC are often quite small (e.g., RE vs FF-Erase(D) vs FF-Erase(R) in **Figure 4(c)** differ at the third decimal place: 0.532 vs 0.5245 vs 0.5260), yet no standard deviations or multiple runs are reported. Without error bars, it is unclear whether these differences are statistically meaningful.  
   - Similarly, accuracy drops of 1–3% on \(\mathbb{D}_{\mathrm{test}}\) and \(\mathbb{D}_{\mathrm{forget}}\) could easily be within training noise; the paper claims “only a minor 1.6–3.3% degradation” but does not show distributions across seeds.  
   - G-MIA itself is trained with a specific MLP architecture; there is no sensitivity study on architecture choice, shadow data quality, or the attacker’s synthetic data assumptions in §5. Since G-MIA is proposed as an evaluation tool, its robustness to such choices is important.

7. **Scope and scale of experiments are modest relative to the claims.**  
   All experiments are on small to medium vision datasets (MNIST, Fashion-MNIST, CIFAR-10/100) and CNN-like FF architectures (TinyCNN, AlexNet, VGG). However, §2 highlights FF variants for convolutional, recurrent, and graph models in more challenging domains. There is no demonstration that FF-Erase scales to larger datasets (e.g., ImageNet, long-sequence tasks) or to other FF architectures (e.g., FF-LSTM, ForwardGNN), nor any profiling of memory/runtime on larger models.  
   Given that a central selling point of FF is better scaling in resource-constrained or distributed settings, the efficiency claims (“1.9–3.1× faster than retraining”) might not hold under more realistic large-scale workloads where retraining or guidance model training is more costly.

8. **Efficiency analysis is mostly heuristic and limited to one regime.**  
   The time analysis in **Equation (9)** and §4.3 uses α₁, α₂, β, and K to claim that FF-Erase typically incurs 25–35% of retrain cost, but this is supported only by the CIFAR-scale experiments. **Table 1** indeed shows \(t_{\mathrm{unl}}\) around 0.39–0.53 \(t_{\mathrm{ret}}\) for the tested settings, but FF-Erase relies on training an entire guidance model, which is \(O(t_{\mathrm{ret}})\) in big-O terms. There is no exploration of scaling behavior as model depth L, dataset size, or β (forget proportion) vary beyond β=0.2.  
   In particular, when β is small but the remaining dataset is huge, training a guidance model on α₁·α₂·\(\mathbb{D}_{\mathrm{remain}}\) may dominate the overall cost compared to simply retraining incrementally or using specialized exact methods.

9. **Threat model and synthetic data assumptions for G-MIA are not fully justified.**  
   §5 assumes that an attacker can synthesize data with similar distribution to the training set via model inversion, and then train multiple shadow models. For complex domains and privacy-sensitive tasks, this is not trivial and is a strong assumption. The paper treats this as standard but does not discuss failure modes if synthetic data distribution mismatches, nor whether G-MIA remains reliable when real data distributions are substantially more complex than CIFAR-100. Since G-MIA’s ACC/AUC are used as the *quantitative* unlearning metric (e.g., **Figure 4(c)**, **Figure 5(c)**, **Table 1**), these assumptions should be examined more critically.

10. **Some reliance on appendices for core empirical comparisons.**  
    While space is constrained, a number of important results are only in the appendix, notably the comparisons with FATS, FYE, SURE, and Bad Teacher in **Figure 10**. These baselines are crucial to support the claim that “classical unlearning methods are infeasible for FF models,” yet in the main text only the weakest baseline, GA, is shown in detail. For an ICLR main-track paper, a more balanced main-text presentation of baselines (even if partially summarized) would be preferable.

---

## Potentially Missing Related Work

The following directly related works on *verification of machine unlearning* are not cited or discussed and should be integrated into the related work and verification discussion (Sections 2 and 5):

1. **Xuan & Li, “Verifying Robust Unlearning: Probing Residual Knowledge in Unlearned Models”, 2025.**  
   Introduces Unlearning Mapping Attacks (UMA) to probe residual knowledge in unlearned models. This is directly relevant to the paper’s goal of verifying FF unlearning effectiveness and should be discussed alongside G-MIA in §5, especially regarding robustness to residual knowledge and protocol design.

2. **Hu, Lou & Liu, “ERASER: Machine Unlearning in MLaaS via an Inference Serving-Aware Approach”, 2023.**  
   Proposes an inference-serving-aware framework for unlearning in hosted systems. This is pertinent to the deployment scenario where only limited model interfaces are observable. It would be useful to contrast ERASER’s assumptions with the “black-box” assumptions made for G-MIA in §5.

3. **Xu, Zhu & Zhang, “Really Unlearned? Verifying Machine Unlearning via Influential Sample Pairs”, 2024.**  
   Presents IndirectVerify, which uses influential sample pairs to assess unlearning. This is a complementary verification strategy to MIAs and should be cited in §2 and connected to the discussion of G-MIA as one of several possible verification methods.

4. **Wang, Zhang & Tian, “SMS: Self-supervised Model Seeding for Verification of Machine Unlearning”, 2025.**  
   Proposes SMS, a self-supervised seeding approach for verifying unlearning. The SMS methodology is relevant to the shadow-model design in G-MIA; it should be contrasted and cited around §5.

5. **Zhang, Chen & Shen, “Verification of Machine Unlearning is Fragile”, 2024.**  
   Analyzes fragility and shortcomings in existing unlearning verification protocols. This is highly relevant to the reliability of MIA-based verification and should be explicitly discussed when motivating G-MIA and interpreting ACC/AUC scores (e.g., in §5 and §6.2).

6. **Brimhall, Mathew & Fendley, “Mirror Mirror on the Wall, Have I Forgotten it All? A New Framework for Evaluating Machine Unlearning”, 2025.**  
   Proposes a new framework for evaluating unlearning, with emphasis on robustness of evaluation. This should be cited in §2 and §5, and the authors should clarify where G-MIA sits relative to these broader evaluation frameworks.

7. **Pujol Vidal, Johansen & Jahromi, “Verifying Machine Unlearning with Explainable AI”, 2024.**  
   Explores using explainable AI techniques for unlearning verification. Given that the paper already uses layer-wise goodness vectors, this line of work is directly relevant and could be discussed as a complementary or alternative way of interpreting residual information.

---

## Questions

1. **Clarification of the goodness definition and dimensionality.**  
   - Please clarify the exact shape of \(h^l\) and \(g^l\). Is \(h^l\) a J-channel tensor with one channel per class, and is \(\boldsymbol{g}^l\) computed as a per-class L1 norm over spatial dimensions / neurons?  
   - If so, could you rewrite **Equation (1)** to reflect that \(g_j^l = \|h_j^l\|_1\) for each class j, and explain how \(z^l\) is computed (per-channel normalization vs scalar)? A corrected and unambiguous notation would improve clarity and ensure Equation (5)’s KL divergence is well-defined.

2. **Implementation of CKA in experiments vs. Equation (12).**  
   - Did you implement CKA using the standard formula with \(\sqrt{\mathrm{HSIC}(X^a,X^a)\,\mathrm{HSIC}(X^u,X^u)}\) in the denominator? If yes, please correct **Equation (12)** and confirm that the numbers in **Table 3** do not rely on the mis-specified variant.  
   - If you actually used the given formula, could you re-run the CKA analysis with the standard definition and report whether this changes the qualitative conclusions about layer-wise behavior?

3. **Access assumptions for G-MIA and deployment realism.**  
   - In what realistic deployment scenarios would an attacker have access to the full set of per-layer goodness vectors while lacking access to parameters/gradients? Are you envisioning FF models being deployed such that goodness is exposed as part of an API?  
   - Could you add experiments evaluating a stricter black-box variant that only sees final-layer goodness or logits, to clarify how much of G-MIA’s advantage stems from deeper layer access?

4. **Statistical robustness of reported gains.**  
   - How many random seeds were used for each experiment (training FF models, unlearning procedures, G-MIA training)? Can you provide mean ± standard deviation for key metrics in **Figure 3**, **Figure 4**, **Figure 5**, and **Table 1**?  
   - In particular, are differences like 0.5245 vs 0.532 in G-MIA ACC statistically significant? If not, please temper claims about G-MIA “matching” or “surpassing” white-box attacks.

5. **Scaling behavior with dataset size and forget proportion.**  
   - Have you evaluated FF-Erase for smaller forget fractions, e.g., β=0.01 or 0.001, which are common in unlearning requests? In such regimes, does the fixed cost of obtaining a guidance model still make FF-Erase more efficient than retraining?  
   - Could you provide at least a synthetic scaling experiment (e.g., increasing dataset size while fixing model architecture) to empirically validate the asymptotic cost model in **Equation (9)**?

6. **Comparison to more recent verification frameworks.**  
   - How do you view G-MIA relative to methods like UMA, IndirectVerify, or SMS? Do you see G-MIA as complementary, or could any of these be adapted straightforwardly to FF goodness vectors?  
   - It would help if you could position G-MIA in this broader landscape and possibly discuss whether combining these ideas might yield more robust FF unlearning verification.

Author responses that clarify the mathematical inconsistencies, strengthen the threat model and statistical analysis, and better position G-MIA relative to modern verification work would significantly improve my confidence.

---

## Flag For Ethics Review

- No ethics review needed.

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

2: **fair**.  
The empirical methodology is mostly reasonable, and the main qualitative claims (GA instability for FF, usefulness of guidance models) are supported by multiple experiments and figures (e.g., **Figures 4, 5, 8, 9, 10**, **Table 1**, **Table 3**). However, mathematical inconsistencies in the FF/goodness definition and the CKA formula, lack of formal connection between the algorithm and the unlearning objective, and limited statistical analysis lower the soundness.

---

## Presentation Rating

2: **fair**.  
The paper is generally readable and figures are informative (especially **Figure 2** and the time–accuracy plots in **Figures 4–5**), but there are important notation inconsistencies (e.g., goodness definition), an incorrect CKA equation, and some over-claims about “black-box” access and verification reliability. The related-work positioning on verification is incomplete.

---

## Contribution Rating

2: **fair**.  
The problem (FF unlearning) is worth studying, and the paper offers both an FF-adapted unlearning algorithm (FF-Erase) and a goodness-based MIA (G-MIA). These are useful starting points, but they are largely heuristic adaptations of existing paradigms (distillation, MIAs) with limited theoretical grounding and modest empirical scale. The contribution is interesting but not yet at a level that clearly warrants ICLR acceptance.

---

## Overall Rating

4: **marginally below the acceptance threshold. But would not mind if paper is accepted.**  
The work raises an important and underexplored question (machine unlearning for FF models) and offers a reasonably well-engineered solution plus a verification tool. The experiments are broad enough to suggest that naïve GA is indeed problematic for FF and that guidance-based FF-Erase is a promising direction. However, key mathematical definitions are inconsistent, the claimed “black-box” verification setting is overstated, statistical rigor is limited, and the verification story is not well tied into the latest unlearning evaluation literature. With these issues, I see this as a solid but not yet fully polished contribution; a revision addressing the above concerns could make it much stronger.

---

## Reviewer Confidence

4: **confident.**  
I am familiar with machine unlearning, MIAs, and FF-style training, and I carefully examined the equations, algorithms, and key tables/figures. While some experimental details are not fully specified, my overall assessment is unlikely to change dramatically, though clarifications and additional results could shift the recommendation slightly.