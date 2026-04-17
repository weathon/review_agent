---
job_id: edf96990-9a0b-4157-ac0b-bf4390f85402
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Xpj0yeMhpz.pdf
paper: Decoupling the Class Label and the Target Concept in Machine Unlearning
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.  
The submission is a full-length paper with abstract, main sections, and extensive appendix.

## Topic Compatibility
Pass ✅.  
The paper is on machine unlearning, representation dynamics, and optimization for deep classifiers, directly within ICLR’s scope (representation learning, safety/privacy, learning theory, optimization, trustworthy ML).

## Minimum Quality
Pass ✅.  
The paper is in English and contains all required components: Abstract, Introduction, related-work discussion (in Preliminaries + Appendix A), Methodology (Sections 3.1–3.3), Experiments (Section 4 and Appendix F), quantitative and qualitative Results (Tables 2–5, 12, 18, 21–36; Figures 2–7, 8–10, 14–17, 19–20), and Conclusion. Method and experiments are non-trivial and reasonably well justified, with no obvious fatal theoretical or experimental flaws.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts, instructions to reviewers, or manipulative content in the main paper.

---

# Expected Review Outcome:

## Summary

The paper questions the standard assumption in class-wise machine unlearning that the “forget” target coincides with a training class label. It formalizes three new mismatch scenarios between the label domain of the forgetting data, the model outputs, and the target concept (target mismatch, model mismatch, data mismatch) in addition to the usual “all matched” case. 

The authors analyze forgetting dynamics at the representation level, proving a bound (Theorem 3.2) that links gradient-ascent forgetting on one subset to loss changes on nearby subsets in feature space (“representation gravity”), and use this to motivate TARF, a three-phase unlearning framework combining annealed gradient ascent on the known forget set with target-aware gradient descent on selected “hard-to-affect” remaining data. Experiments across CIFAR-10/100, TinyImageNet, ImageNet-1k, Stable Diffusion, and an LLM unlearning benchmark show that TARF generally matches or improves upon prior approximate-unlearning baselines, especially in the new mismatch settings.

## Strengths

1. **Clear conceptual expansion of the unlearning problem.**  
   The decoupling of “forget classes” from “target concept” via three label domains \(\mathcal{L}_D, \mathcal{L}_M, \mathcal{L}_T\) is crisp and useful. Figure 1 and Table 10 / Table 11 nicely systematize all combinations (two-layer and three-layer hierarchies), and the four main scenarios (all matched, target mismatch, model mismatch, data mismatch) are instantiated concretely on CIFAR-10/100 with classes vs superclasses (Table 13, Figure 13). This alone is valuable as it exposes realistic request patterns that prior works implicitly ignore.

2. **Insightful representation-level analysis and use of “gravity”.**  
   Theorem 3.2 (Eq. (2), Page 4–5) ties the loss difference after a gradient-ascent step on subset \(s_1\) to the expected representation distance \(d_h(x_1,x_2)\) between \(s_1\) and \(s_2\), via the Jacobian spectral norm and Lipschitz constant. While the result is not extremely deep, it gives a precise handle on when forgetting spills over to related data. The associated Remarks 3.2 and 3.3 apply this to explain *insufficient representation* (when \(\mathcal{L}_D\prec\mathcal{L}_T\)) and *decomposition lacking* (when \(\mathcal{L}_T\prec\mathcal{L}_M\)). Figure 3’s t-SNE plots and loss curves directly visualize this: for superclass-trained models (left of **Figure 3**), forgetting “boy/girl” drags down “man/woman/baby”; for class-trained models (right), forgetting a subset only weakly affects the rest of the target concept. This theoretical–empirical alignment is a strong point.

3. **General and interpretable algorithmic framework (TARF).**  
   TARF’s loss (Eq. (3)) is a simple combination of
   \[
   L_{\text{TARF}} = k(t)\,L_f + L_u(\tau),
   \]
   with \(k(t)\) an annealing schedule (Eq. (5)) for gradient ascent on the forget set \(\mathcal{D}_f\), and \(\tau(x,y,t)\in\{0,1\}\) an indicator selecting which remaining points to train on with descent based on a representation-gravity proxy \(I_{\text{con}}\). **Figure 4** clearly illustrates the three phases: Phase I uses GA only to expose which remaining samples/classes exhibit large loss/accuracy changes; Phase II jointly applies GA and GD to separate target vs retain representation; Phase III turns off GA (\(k(t)\to 0\)) and finishes with pure retraining on identified retaining data, pulling the model back toward retraining behavior (as also illustrated empirically in **Figure 5(b)**). The interpretation of each phase is coherent and re-used across all mismatch scenarios.

4. **Extensive and multi-angle experimental evaluation.**  
   - **Main quantitative results:** **Table 3** is comprehensive: for CIFAR-10/100, across four scenarios, TARF has the smallest or second-smallest “Gap” to retraining in almost all cases. In the challenging *target mismatch* and *data mismatch* settings, existing methods leave UA around 40–60% (indicating large residual knowledge of the target concept), while TARF drives UA down near 0% with competitive RA/TA and MIA = 100%. Similarly, for *model mismatch*, TARF’s Gap is significantly lower than L1-sparse and FT, and competitive with SCRUB, while reducing UA (forget accuracy) closer to the retrained reference.
   - **Fine-grained analysis:** **Table 2** decomposes UA into “forget part” and “background part” within a superclass on CIFAR-10/100, showing TARF achieves better separation between foreground (to forget) and background (to retain) than FT/L1-sparse/SCRUB in terms of Gap, even if some baselines have slightly better UA on one split.  
   - **Scaling to large datasets:** **Table 4** (ImageNet-1k) and **Table 23** (TinyImageNet) show TARF maintains small gaps to retraining with reasonable time overhead, outperforming or matching FT, L1-sparse, and SCRUB across all four mismatch scenarios. The TIME column demonstrates TARF is far cheaper than retraining and comparable to other SGD-based approximate methods.
   - **Case studies on generative and language models:** **Figure 6** and Tables 19–20 show that, in a *data mismatch* regime for Stable Diffusion, TARF erases visual concepts like “tench” and “English springer” more cleanly than CL while preserving other semantics. **Table 5** shows that wrapping TARF around GA or SWT improves LLM forgetting on TOFU, reducing QA success on forget paths while keeping QA on retain paths significantly higher than GA alone, particularly under mismatch setups. These applications support the claim that the ideas are not tied to small classification models.

5. **Thorough ablation and robustness analysis.**  
   The paper does a good job stress-testing its own design choices:
   - **Figure 7 (and Figure 17)** vary the initial forgetting weight \(k\), the schedule \(k(t)\) (constant / annealed / increasing), and timing parameters \(t_1, t_0\). The plots show a broad stable regime where TARF’s Gap remains low, with only very aggressive \(k\) or pathological scheduling hurting RA and TA, and empirically justify the three-phase design (annealed GA, early target identification, and non-trivial Phase III).
   - The right panel of **Figure 7** compares using gradient ascent vs “gradient cleaning” (zero gradient) on the discovered false-retain set \(\mathcal{D}_{\text{fr}}\). Gradient cleaning preserves RA better with little UA difference, explaining the choice in the final algorithm.  
   - The middle-right panel of **Figure 7** and **Table 26** compare architectures (ResNet-18, VGG-19, WRN-50). TARF consistently has smaller Gap than FT across structures, and the analysis notes that smaller-capacity models are harder to disentangle, which is a useful observation.  
   - **Table 17** and **Table 35** probe sensitivity to the amount and representativeness of the initial forget set and to the quantile used for \(\beta\). The authors frankly show degradation when the forget set poorly represents the target concept but still outperform raw GA in those adverse regimes.

6. **Clarity, structure, and reproducibility.**  
   The exposition is generally well structured. Section 3 flows from problem formulation (3.1) to theoretical analysis (3.2) to algorithm (3.3). Preliminaries and notation (Table 1, Tables 7–8, 13–15) are detailed. Most equations are well defined, and pseudo-code in Algorithms 1–2 plus hyperparameter descriptions in Appendix F.1/E.1 make re-implementation very feasible. Figures 2–5 and 14–17 give an unusually rich picture of representation geometry and forgetting trajectories, which helps readers build intuition instead of treating TARF as a black box.

## Weaknesses

1. **Theory is modest and somewhat loose relative to the algorithmic claims.**  
   The main theoretical contribution, Theorem 3.2 (Eq. (2), Page 4–5, restated as Theorem C.2), is essentially a Taylor expansion–based bound on the change in loss difference between two subsets after a gradient-ascent step. It requires Lipschitz-smooth \(\ell_h\) and controls the term \((\nabla L_{s_1}-\nabla L_{s_2})^\top\Delta\theta\) via \(\lambda_{\max}(J_\theta) C_\ell \mathbb{E}[d_h(x_1,x_2)]\). This is fine as a qualitative justification, but:
   - There is no analysis of stochastic mini-batch SGD, which is what TARF uses. The dynamics and noise from mini-batching could significantly change the effective “gravity” behavior.  
   - Assumption 3.1 / C.1 about Lipschitz-smoothness of \(\ell_h\) with respect to hidden representation is standard but unchecked; more concerning is that the theorem is stated per-epoch, but the bound is only derived for a single small update step using learning rate \(\eta\), with an \(\mathcal{O}(\eta^2)\) term swept under the rug. There is no stability/accumulation analysis over multiple epochs.  
   - The key qualitative conclusion “smaller representation distance implies stronger co-movement under forgetting” is persuasive, but the theorem’s inequality itself is never used quantitatively to set \(k(t)\), choose \(\beta\), or reason about convergence of \(L_{\text{TARF}}\to L_{\text{retrain}}\) (Eq. (4)), so the bridge from Eq. (2) to the three-phase scheme is still somewhat heuristic.  

   In short, the math is helpful intuition but does not rise to a rigorous characterization of when TARF provably approximates retraining, which slightly weakens the “representation gravity” narrative.

2. **Strong assumptions about label-domain information and target size.**  
   Several parts of the framework rely on knowledge that may be unrealistic in many real cases:
   - In target/data mismatch scenarios (Section 2, “Dataset partition” and Appendix D.4), the method assumes that the number of classes within the remaining set that belong to the target concept is known, in order to set \(\beta\) via class-wise accuracy change; this is non-trivial in practice when “concepts” are amorphous or cross multiple labels.  
   - The main experiments on CIFAR-100 use the official superclass structure (Table 14) and an author-defined grouping for CIFAR-10 (Table 15). As discussed in Appendix F.2 and Figure 16, some superclasses (e.g., “aquatic mammals” vs. “fish”) are not cleanly separable in representation, and TARF struggles there, but the main results emphasize “nice” superclasses like “people”.  
   - Appendix E.4/E.5 hint at relaxations (pseudo-labels, approximate taxonomies), but the empirical evidence is limited and mainly for semi-supervised-like scenarios where class predictions are already quite good (Table 22).  
   Overall, the evaluation is somewhat biased toward clean hierarchical taxonomies, whereas the most interesting applications (fairness, safety, spurious correlations) often lack such structure.

3. **Complexity and sensitivity of the hyperparameter schedule.**  
   TARF introduces several additional knobs beyond standard unlearning methods: the initial GA weight \(k\), the annealing schedule \(k(t)\) with \(t_0\), the identification start time \(t_1\), and the quantile threshold \(\beta\) for \(\tau(x,y,t)\). While Appendix E.1 provides a qualitative tuning guide and Figures 7 and 17 show some robustness, there are still some concerns:
   - The choice of \(k\) is dataset- and scenario-dependent (e.g., \(k=0.04\) / \(0.02\) on CIFAR-10 vs \(0.5\) / \(0.05\) on CIFAR-100, Appendix F.1). It is unclear how one would systematically pick \(k\) without access to a retrained reference, especially when the target concept is subtle.  
   - The quantile selection for \(\beta\) is ad hoc (e.g., “top-5%” or “top-10%” of loss/accuracy-drop classes). Table 17 shows that when the false-retain set becomes large, TARF’s UA remains quite high (23% or more), and the Gap grows, yet there is no principled method to detect this failure regime.  
   - Phase boundaries \(t_1, t_0\) are often fixed to 1 and 2 epochs respectively, but Figure 17 shows some non-trivial sensitivity; an inattentive practitioner could easily pick suboptimal timings and either under-identify targets or over-destroy representations.  

   Compared to simpler baselines like FT or GA, this hyperparameter surface is more complex, and the paper does not fully quantify tuning overhead.

4. **Evaluation metric “Gap” is a bit coarse and conflates heterogeneous desiderata.**  
   The main scalar metric used for ranking methods is “Gap” defined as the average absolute difference in UA, RA, TA, and MIA (Section 4.1). However, these four metrics are not on equal conceptual footing:
   - In some scenarios, the retrained reference itself does not have UA=0, e.g., *model mismatch* (Table 3, row “Retrained (Ref.)”), because UA is evaluated using superclass labels. This is explained in Section 4.2, but leads to slightly unintuitive interpretations of UA and the Gap.  
   - MIA is either 100 or near 0 for most methods in most tasks, so averaging it into Gap can dominate or wash out differences that are actually small in UA/RA/TA. E.g., in Table 3 (all matched, CIFAR-100), SCRUB and FT have similar Gap despite very different UA and TA, largely because MIA is already saturated.  
   - Different applications might prioritize low UA over high RA, or vice versa; a single unweighted average obscures these tradeoffs. Some more nuanced multi-objective Pareto analysis or at least per-metric highlight would strengthen the empirical story.

5. **Some missing or under-discussed related work on “what makes unlearning hard” and general methods.**  
   The related-work discussion in Appendix A is long but still misses a few directly-relevant recent works:
   - Work that explicitly analyzes *difficulty* and structures of unlearning beyond simple class-wise settings (e.g., papers studying why certain forget sets or concepts are hard to unlearn, or the geometry of forget vs retain sets). These are close in spirit to the representation-gravity and “entanglement vs under-entanglement” framing and could sharpen the positioning of Section 3.2.  
   - Recent general-purpose class-centric unlearning methods based on decoupled distillation or teacher–student structures, which arguably are direct algorithmic competitors to TARF in handling complicated unlearning targets beyond a single class.  
   While the paper does cover SCRUB, LAU, SFR-on, and SG in Appendix B/Table 6, it would benefit from explicitly engaging with work that frames unlearning difficulty at the conceptual/representation level, since that is precisely what the authors claim as their main insight.

6. **Limitations in more challenging or ambiguous concept regimes.**  
   The authors briefly acknowledge in the Conclusion that when target concepts are “inherently ambiguous, weakly clustered, or attribute-entangled”, the gravity signal weakens and ranking becomes noisier. However, the main experiments still live in relatively clean, closed-world settings (vision classification with balanced label hierarchies). There is no systematic stress test on:
   - Multi-attribute overlaps (e.g., forgetting a sensitive attribute that spans many classes),  
   - Long-tailed or open-world settings where the forget set is tiny relative to the full support, or  
   - Multi-modal cues where text semantics might be misaligned with visual clusters.  
   The TOFU and SD cases hint at such difficulties but remain small-scale and somewhat cherry-picked. This undermines the strength of the claim that TARF is “general” to practical unlearning demands.

7. **Some notation and table inconsistencies.**  
   There are a few minor but real issues that make the paper harder to parse:
   - Notation for datasets is overloaded: \(\mathcal{D}_l, \mathcal{D}_t, \mathcal{D}_f, \mathcal{D}_r, \mathcal{D}_{ar}, \mathcal{D}_{fr}, \mathcal{D}_{un}\) are all introduced (Table 1, Page 3), and later \(\mathcal{D}_I, \mathcal{D}_{as}, \mathcal{D}_m\) appear in Algorithms 1–2, not always consistently. A consolidated diagram or table would help.  
   - Some equations use slightly inconsistent subscripts (e.g., \(L_{\text{renan}}\) in Eq. (1) appears to be a typo for \(L_{\text{retrain}}\)), and Eq. (17)/(18)/(19) in Appendix E use slightly different notation from Eq. (3)/(5) in the main text.  
   - A few table labels and references appear off (e.g., “L2-sparse” in Table 24 vs “L1-sparse” elsewhere; “RS” instead of “BS”), which suggests some copy-paste editing issues.  

   These are not fatal, but in a dense paper like this they increase cognitive load.

## Potentially Missing Related Work

1. **K. Zhao, M. Kurmanji, G.-O. Barbulescu, “What Makes Unlearning Hard and What to Do About It,” 2024.**  
   This work explicitly studies the structural and geometric factors that make certain unlearning tasks difficult and proposes strategies to handle them. It is closely related to Section 3.2’s discussion of representation entanglement and insufficient representation. It should be cited in the discussion of forgetting dynamics (around Theorem 3.2 and Figure 3) and in Appendix D.6 when describing scenario commonalities and challenges.

2. **Y. Zhou, D. Zheng, Q. Mo, “Decoupled Distillation to Erase: A General Unlearning Method for Any Class-Centric Tasks,” 2025.**  
   Proposes a general teacher–student distillation framework for class-centric unlearning, which directly competes with TARF as a “general” method in class-wise and beyond-class scenarios. It should be discussed alongside SCRUB, LAU, SFR-on, and SG in Appendix B.1/B.2 and compared in terms of handling mismatched label domains.

3. **Y. Cao, B. Yang, Y. Rong, “Towards Federated Unlearning: A Survey on Machine Unlearning in Federated Learning,” 2022.**  
   While primarily about federated settings, this survey covers general unlearning methodologies and problem formulations that intersect with this paper’s broader motivation (data regulations, user requests, fairness/safety). It would be appropriate to mention in Appendix A.1 as part of the contextualization of unlearning beyond centralized supervised training.

4. **S. Schelter, A. Grafberger, J.-H. Böse, “HedgeCut: Maintaining Data Privacy in Machine Learning via Adaptive Data Sharding,” 2021.**  
   This work uses data sharding to enable efficient removal of user data and thus contributes an alternative system-level approach to achieving unlearning-like guarantees. It would fit into the discussion of approximate vs exact unlearning in Appendix A.1, especially when contrasting algorithmic modification (like TARF) with data-management strategies.

(Graves et al. 2017 on automated curriculum learning is only tangentially related and does not need to be cited.)

## Questions

1. **Quantitative effect of mini-batch SGD vs full-batch dynamics.**  
   Theorem 3.2 is derived under a deterministic gradient-ascent step. Do the authors have any empirical or theoretical insight into how mini-batch noise affects the representation-gravity signal used by \(I_{\mathrm{con}}\)? For example, have you compared class-wise accuracy-drop rankings with very small vs very large batch sizes in Phase I to see if the identification of \(\mathcal{D}_{fr}\) is stable?

2. **Guiding hyperparameter selection without retraining reference.**  
   In practice, users will not have the retrained model to compute Gap. Can you propose a practical heuristic or validation strategy for choosing \(k\), \(t_1\), and \(t_0\) that does *not* rely on access to a retrained model, perhaps using only hold-out validation data or trends in UA/RA estimated on labeled subsets?

3. **Behavior under severely under-representative forget sets.**  
   Table 17 and Table 35 hint at degradation when the forget set is small or biased. Could you characterize more precisely when TARF fails? For instance, is there an empirical ratio of \(|\mathcal{D}_f|/|\mathcal{D}_t|\) or a minimum mutual information between forget set and target concept below which representation gravity becomes unreliable and TARF behaves no better than GA?

4. **Comparison with decoupled or teacher–student unlearning methods on mismatch tasks.**  
   You already compare against SCRUB, LAU, SFR-on, and SG in Table 6 for CIFAR-100. For at least one mismatch scenario (say, target mismatch), could you provide a more detailed breakdown (analogous to Table 2) showing how these methods fare on UA/RA/Fine-grained UA, and discuss whether TARF’s advantage is due more to better target identification or cleaner separation in Phase II?

5. **Applicability to non-hierarchical concepts.**  
   Many real unlearning requests will involve non-hierarchical or attribute-based concepts (e.g., “remove all NSFW content” or “forget all images with a watermark”) rather than superclass relationships. Can you clarify how TARF would operate when the target concept cuts across many classes and no clear superclass structure exists? For instance, could you replace class-wise accuracy-drop with cluster-wise or embedding-based grouping in Phase I, and do you have any preliminary experiments in that direction?

6. **Choice of design for \(L_{\text{TARF}}\).**  
   In Eq. (3), the retaining loss is computed over \(\mathcal{D}_{un}\) with binary \(\tau\). Have you considered a *soft* weighting \(\tau(x,y,t)\in[0,1]\) based on the magnitude of \(I_{\text{con}}\) (rather than a hard threshold via \(\beta\)) so that highly affected retain points are down-weighted rather than excluded? If so, how did it compare in terms of stability and Gap?

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A. The work uses standard public datasets (CIFAR, TinyImageNet, ImageNet, Stable Diffusion, LLama-based models, TOFU) and focuses on improving the *removal* of data influence, which is generally positive from a privacy and safety standpoint. There is no evident misuse, unfairness, or data-licensing concern beyond what is typical for these benchmarks.

## Soundness Rating

3: good.  
The overall methodology is sound and empirically well validated. The main theoretical result is more of an intuition formalization than a deep guarantee, and some assumptions (label-domain knowledge, target size) are strong, but there are no obvious fatal flaws in derivations or experiments.

## Presentation Rating

3: good.  
The paper is well written and organized, with many helpful figures (e.g., Figures 1–5, 7) and detailed appendices. Some notation/tables have minor inconsistencies, and the heavy notation for dataset partitions could be streamlined.

## Contribution Rating

3: good.  
The work makes a meaningful conceptual contribution by formalizing label-domain mismatches in class-wise unlearning, provides a reasonably original and general framework (TARF) grounded in representation dynamics, and backs it with extensive experiments and case studies. The theoretical depth is moderate rather than strong, and some assumptions limit immediate generality, but overall the contribution is solid and relevant for ICLR.

## Overall Rating

8: Accept, good paper (poster).  
The paper significantly broadens the scope of class-wise unlearning, offers a clear representation-based analysis, and proposes an effective algorithmic framework that consistently matches or outperforms strong baselines across a wide variety of settings, including realistic misuse scenarios on generative and language models. Despite some theoretical looseness and reliance on hierarchical label information, the work is carefully executed, empirically thorough, and likely to influence future research on practical unlearning and concept erasure.

## Reviewer Confidence

4: confident.  
I am familiar with machine unlearning and representation learning, carefully checked the core math (Eq. (2), Eq. (3)–(5)) and experimental design, and compared with known baselines. Some peripheral application details (e.g., specific LLM prompts) were not exhaustively verified, but they do not affect the main technical assessment.