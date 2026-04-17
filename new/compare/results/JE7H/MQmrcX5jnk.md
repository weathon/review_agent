---
job_id: f365e272-39f2-4bd8-9637-db4938e25689
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: MQmrcX5jnk.pdf
paper: Learning Boltzmann Generators via Constrained Mass Transport
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper develops a variational mass-transport framework with KL and entropy constraints, instantiated with normalizing flows for molecular Boltzmann generators. This is squarely within probabilistic methods, generative models, and applications to physical sciences, all core ICLR topics.

## Minimum Quality
Pass ✅.  
All required sections are present (Abstract, Introduction, Method, Experiments, Results, Related Work, Conclusion). The work is technically substantial, the math is mostly sound with detailed proofs, and the experiments on multiple peptide systems with strong baselines are thorough. I see no fatal theoretical or experimental flaw warranting desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The manuscript contains no instructions aimed at automated reviewers, no suspicious hidden text, and no signs of manipulation of the review process.

---

# Expected Review Outcome:

## Summary

The paper introduces Constrained Mass Transport (CMT), a variational framework that constructs an annealing path of intermediate distributions between a base distribution and an unnormalized target by solving a sequence of KL-minimization problems subject to constraints on (i) KL divergence to the previous iterate (trust region) and (ii) entropy decay.  

Closed-form expressions for the optimal intermediate densities are derived for three cases: trust-region only, entropy-only, and combined constraints, and these are shown to induce geometric, tempered, and geometric–tempered annealing paths, respectively. CMT is instantiated with normalizing flows to learn molecular Boltzmann generators purely from energy evaluations and evaluated on several peptide systems, including a new ELIL tetrapeptide benchmark, where it outperforms state-of-the-art variational methods in EUBO, ESS, and Ramachandran-plot metrics.

## Strengths

1. **Clear variational formulation and analytical intermediate densities.**  
   The paper sets up the core constrained problems in Equations (2), (7), and (9), and derives closed-form parametrizations of the optimal intermediate densities via Lagrangian duality (Propositions 2.1–2.3). For example, Equation (10) gives
   \[
   q_{i+1}(x,\lambda,\eta)\propto q_i(x)^{\frac{\lambda}{1+\lambda+\eta}}\tilde p(x)^{\frac{1}{1+\lambda+\eta}},
   \]
   which is elegant and makes the dependence on both constraints completely explicit. The derivations in Appendix A are mostly clean and check out under the stated assumptions (convexity, Slater, etc.).

2. **Connection between constrained optimization and annealing paths is insightful.**  
   Theorem 2.4 shows that iterating these constrained updates produces geometric (G), tempered (T), and geometric–tempered (GT) paths, all of the form \(q_i \propto q_0^{1-\beta_i} (\tilde p^{\alpha_i})^{\beta_i}\) with \(\beta_i\uparrow1\). This ties together disparate annealing strategies into a single variational perspective and clarifies how the KL and entropy multipliers map to “temperature” and mixing with the prior. Figure 1 is particularly helpful: it visually contrasts a naive geometric schedule, the trust-region schedule from (2), the entropy-only path from (7), and the combined GT path from (9). You can directly see mass teleportation appearing in the standard geometric AP and being suppressed once the entropy constraint is active.

3. **Algorithmic instantiation with normalizing flows is carefully thought through.**  
   Section 3 explains how to approximate each analytic \(q_i\) with a flow \( \hat q_i \in \mathcal{Q}_{\mathrm{NF}}\) via an importance-weighted forward KL (Equation (15)), reusing samples from \(q_i\). The choice of forward KL encourages mode coverage, counteracting the reverse KL’s mode-seeking bias. The trust-region constraint is then used not just conceptually but practically to bound the variance of the importance weights and hence stabilize training, with a quantitative ESS lower bound derived in Appendix C.3 (Equation (21)). This is a technically interesting bridge between path construction and sample-efficiency / variance control.

4. **Strong and carefully controlled experimental evaluation.**  
   The experimental section is unusually thorough for this domain. Table 1 presents the main quantitative comparison across four peptide systems (alanine dipeptide, tetra-, hexa-peptides, and ELIL tetrapeptide) and shows consistent improvements over FAB and TA-BG in EUBO and ESS, often with the same or fewer target evaluations. For instance, on alanine hexapeptide, CMT improves ESS from 18.22% (TA-BG) and 14.55% (FAB) to 29.63%, while slightly improving EUBO. On the most challenging ELIL tetrapeptide, ESS roughly doubles compared to TA-BG (26.06% vs 13.75%) and far exceeds FAB and reverse KL. The extended metrics in Table 2 (Ram KL, Ram TV, TICA Wasserstein, etc.) further support the claim that CMT reduces mode collapse relative to reverse KL and improves over other annealing-based methods on most forward-style metrics.

5. **Good qualitative analysis and use of figures.**  
   Figures 2 and 3 are particularly valuable. In Figure 2a–b, you see how different constraint configurations affect the entropy trajectory and the ESS between successive intermediates; the “no constraint” and trust-region-only variants rapidly collapse entropy, while entropy-only becomes unstable at higher dimension. Figure 2c–d then links these behaviors to final EUBO and target ESS. Figure 3’s Ramachandran plots make mode collapse painfully clear: only the tempered (7) and GT (9) variants preserve the high-energy metastable regions. Figure 4 and Figure 9 extend this qualitative check across systems and via TICA projections, and they visually justify many of the quantitative claims in Tables 1 and 2.

6. **Non-trivial new benchmark and fair comparison protocol.**  
   The introduction of the ELIL tetrapeptide (d=219) as a large, complex system trained purely from energy evaluations, without MD samples, is a meaningful contribution to the Boltzmann generator literature. The authors carefully match architectures across methods and tightly control the target-evaluation budget. The detailed hyperparameter tables (Tables 10–16) and computational cost summary (Table 7) significantly help reproducibility and honest comparison.

7. **Theoretical link to importance-weight variance / ESS.**  
   The argument in Appendix C.3 that the KL trust-region bound \(\varepsilon_{\mathrm{tr}}\) implies an approximate ESS lower bound,
   \[
   \mathrm{ESS}(q_i,q_{i+1}) \gtrapprox \frac{1}{1 + 2\varepsilon_{\mathrm{tr}}},
   \]
   via the \(\chi^2\)–KL relation is conceptually neat and aligns with empirical ESS traces in Figure 6. This bridges theory and practice: Figure 6 shows how smaller \(\varepsilon_{\mathrm{tr}}\) yields a dimension-independent ESS floor that matches the derived curve quite closely across system sizes.

## Weaknesses

1. **Conceptual novelty is more “unification and adaptation” than fundamentally new.**  
   The core ingredients, viewed abstractly, are (i) KL trust-region constraints on probability measures plus (ii) entropy constraints, both of which have extensive history in RL and variational inference. The paper’s main conceptual move is to port these constraints to sampling / annealing and then show that their combination recovers and generalizes geometric and tempered paths. This is useful and well-executed, but the line between “new framework” and “clever combination and repackaging” is thin. For instance, Theorem 2.4 expresses the resulting paths as yet another quasi-geometric family, which resembles existing annealing-path discussions (e.g. geometric, deformed-log, etc.). The paper would benefit from a more candid positioning relative to earlier “variational annealing” views and adaptive AIS/SMC schedule optimization, clearly delineating what is genuinely new and what is reinterpreted.

2. **Some mathematical pieces and notation are sloppy or confusing in critical spots.**  
   While the main derivations in Appendix A are mostly sound, there are places where notation or apparent typos could mislead an attentive reader:
   - Equation (16) seems incorrect: the exponent uses \(1+3+\eta\) in several places instead of \(1+\lambda+\eta\), which is what Equation (10) suggests and what the dual in Equation (11) depends on. This is not just cosmetic, since it enters the Monte Carlo estimator used for dual optimization; at minimum, a corrected expression and a brief derivation should appear in the main text.
   - In Section C.3, the expression for importance weights for the trust-region-only case has \((\bar p(x)/q_i(x))^{1/(1+\lambda_i)}\), but earlier the notation \(\tilde p\) is used. It would be cleaner to maintain a consistent notation for unnormalized vs normalized densities throughout, especially since the ESS bound in Equation (21) leans on approximations where normalization may matter.
   - Some steps in the proof of Theorem 2.4 (especially the definition and interpretation of \(\alpha_i\)) are quite algebraically dense. There is a risk that readers infer stronger properties about the annealing path (e.g., monotonicity of “temperature” in a thermodynamic sense) than actually proven. It would help to separate what is rigorously shown (monotonic \(\beta_i\)) from heuristic interpretations (e.g., “smooth temperature evolution”).

3. **Limited evidence beyond molecular Boltzmann generators and narrow baseline family.**  
   All experiments are on peptide systems in internal coordinates with essentially identical normalizing-flow architectures across methods. This is an appropriate and important application domain, but it leaves open how broadly CMT’s benefits extend. In particular:
   - There is no non-molecular benchmark (e.g., a high-dimensional Gaussian mixture, rugged Bayesian posterior, or spin-glass-like system) where MD tools and Ramachandran plots are not available.  
   - Baselines are primarily closely related flow-based annealing schemes (FAB, TA-BG) plus forward/reverse KL. For a “general framework”, one would like to see at least one comparison outside the specific BG literature, such as to a strong AIS/SMC scheme with optimized geometric path or to recent non-equilibrium transport samplers, on a problem where those techniques are standard.
   This does not invalidate the peptide results but does limit the claim that CMT is generally superior as a mass-transport framework rather than as “a very good BG training regime”.

4. **No direct quantitative metric for “mass teleportation”.**  
   A central motivation is avoiding “mass teleportation”. Figure 1 nicely visualizes how density mass appears without overlap in geometric APs vs the constrained paths, and Figures 2–3 & 4 qualitatively show better coverage of high-energy modes. However, the paper never defines a concrete quantitative teleportation (or overlap) metric, such as a bound on \(\int \min(q_i, q_{i+1})\), Wasserstein distance between consecutive steps, or a KL-based asymmetry that specifically tracks “new modes appearing where \(q_i\approx 0\)”. Instead, the argument is indirect: trust-region constraints ensure \(D_{\mathrm{KL}}(q_{i+1}\|q_i)\le \varepsilon_{\mathrm{tr}}\), and empirically ESS remains high (Figure 6). Since teleportation is used as a key selling point, not just a side remark, it would be stronger to either (i) define and plot a teleportation diagnostic across methods, or (ii) tone down this narrative and emphasize overlap/ESS more explicitly.

5. **Entropy-constraint tuning and sensitivity are somewhat hand-wavy.**  
   The entropy constraint is conceptually appealing, but in practice it introduces a non-trivial hyperparameter \(\varepsilon_{\mathrm{ent}}\) that is tuned per system. While Appendix C.4 provides guidance and Figure 10 shows a sweep on alanine hexapeptide, several issues remain:
   - The recommended heuristic “let the constraint become inactive halfway through training” is informal and may not transfer across targets or architectures.
   - Figure 5 shows that entropy-only constraints can be violated in high dimension (alanine hexapeptide) if \(\varepsilon_{\mathrm{ent}}\) and the number of steps are not calibrated correctly. This suggests non-trivial sensitivity that deserves more systematic exploration.
   - The experiments report that \(\varepsilon_{\mathrm{ent}}\) is in \([0.8,1.8]\) across systems, but this is a fairly wide range. A more explicit analysis of performance vs \(\varepsilon_{\mathrm{ent}}\) on at least one smaller molecule, including failure modes, would help practitioners understand how to set it without manual trial-and-error on expensive targets.

6. **Interpretation and reliability of some secondary metrics are muddled.**  
   The paper itself points out that several metrics (ELBO, some Wasserstein-2 distances with only \(10^4\) samples) are unreliable, yet they are still reported extensively in Table 2 and Appendix tables. On the one hand, this is honest; on the other, it introduces some confusion:
   - ELBO is strongly impacted by rare extreme log-weights and is heavily clamped. Marking it with “★” and explaining the problem (Appendix D.3) is helpful, but then ELBO should not be used qualitatively in the text as supporting evidence. Right now, the reader has to mentally filter out an entire block of Table 2.
   - The Wasserstein-2 distances on TICA projections (Table 4 and Figure 9) are emphasized as not very sensitive with \(10^4\) samples. If these metrics are not reliable, it might be better to demote them to a short supplementary mention and focus the narrative on EUBO, ESS, Ram KL/TV, and energy histogram metrics, which are already comprehensive.
   This is not a fatal issue, but the story would be cleaner if the main text were more selective.

7. **Minor but non-trivial inconsistencies in some tables.**  
   There are a few places where the numbers are confusing or potentially mis-typed:
   - In Table 3, for alanine dipeptide, Ram TV values are written as \((9.13 \pm 0.05)\times 10^{-2}\) etc., whereas Table 1 reports Ram TV on the order of \(10^{-2}\) as well but with much smaller standard errors. It is not obvious why a variant “without any constraints” yields Ram TV of approximately \(10^{-1}\) (if the exponent is indeed \(-2\)), which would be dramatically worse than all main methods in Table 1. Clarifying exponents and units, or explicitly flagging that Table 3 is using a different normalization, would help.
   - Similarly, some reported ELBOs in Table 2 (e.g., \(-17658\) or \(-417884\)) are so extreme for FAB that they are practically meaningless. While the text later points this out, in the table itself these values visually dominate the scale and can be misinterpreted.

8. **Limited discussion of approximation error from \(q_i\) to \(\hat{q}_i\).**  
   The theoretical discussion is about exact intermediate densities \(q_i\), but in implementation each \(q_i\) is approximated by a flow \(\hat q_i\). This introduces two sources of mismatch:
   - The dual optimization (Equation (11), Section 3) uses Monte Carlo estimates under \(q_i\), but sampling in practice is done from \(\hat q_i\).
   - The ESS bound in Equation (21) and the annealing path characterizations in Theorem 2.4 technically apply to the ideal sequence \((q_i)\), not to \((\hat q_i)\). Figure 6 suggests that ESS still behaves nicely, but there is no analysis of how approximation error accumulates across many annealing steps, especially for large \(I\).  
   A brief discussion quantifying or at least bounding \(\mathrm{KL}(q_i \|\hat q_i)\) and its impact on the ESS bound would make the theoretical claims more robust.

## Potentially Missing Related Work

1. **Wu et al., “Variational Annealing: A Variational Approach to Bayesian Inference” (2020).**  
   This work explicitly studies variational annealing, i.e., constructing sequences of variational distributions connecting a prior and a posterior via annealing-type procedures. It seems very close in spirit to the proposed constrained mass-transport: both view annealing through a variational optimization lens, though Wu et al. operate in a Bayesian-inference setting. This paper should be discussed in Section 4 (Improved annealing paths / constrained optimization) as a nearby approach that also formulates annealing as a sequence of variational problems, and the authors should clarify how CMT’s trust-region and entropy constraints differ from or extend the variational annealing formulations there (e.g., in how the path is parameterized and how constraints vs regularizers are used).

(Other papers from the provided list such as VAEs, NUTS, neural ODEs, FFJORD, and generic gradient estimators are broadly relevant to probabilistic modeling or flows but not directly about constrained annealing paths or Boltzmann generators; I do not see them as “directly related” in the sense required for explicit comparison.)

## Questions

1. **Clarification on Equation (16).**  
   Can you confirm whether Equation (16) should read
   \[
   \mathcal{Z}_{i+1}(\lambda,\eta) = \mathbb{E}_{x\sim q_i}\left[\left(\frac{\tilde p(x)}{q_i(x)^{1+\eta}}\right)^{\frac{1}{1+\lambda+\eta}}\right]
   \]
   instead of the current \(1+3+\eta\) in the exponent and denominator? If so, please correct this and indicate whether the implementation uses the corrected formula.

2. **Quantifying mass teleportation / overlap.**  
   Do you have measurements of an explicit overlap metric between successive intermediates, such as \(\int \min(q_i, q_{i+1})\) or a Wasserstein distance, comparing geometric, tempered, and GT paths? Even if approximated via samples from \(\hat q_i\), such a plot across \(i\) for, say, alanine hexapeptide would give a much sharper quantitative picture than the current qualitative Figure 1.

3. **Behavior if only one constraint is used but tuned differently.**  
   For the entropy-only variant, Figure 5 indicates violations of the linear decay target on alanine hexapeptide. Is this mainly due to too aggressive \(\varepsilon_{\mathrm{ent}}\) given a fixed number of steps, or do you observe such violations even if you choose a smaller bound so that the constraint theoretically remains active longer? Some clearer guidance on the stability region (pairs of \(\varepsilon_{\mathrm{tr}},\varepsilon_{\mathrm{ent}}\)) would be useful.

4. **Generality beyond molecular systems.**  
   Have you tried CMT on any “standard” multimodal targets such as mixtures of Gaussians in higher dimension, funnel posteriors, or rugged Bayesian logistic regression? If so, how does its performance compare to well-tuned AIS/SMC with optimized geometric paths? Even a brief mention in the rebuttal or supplementary experiments would help clarify whether CMT’s advantages are somewhat specialized to BG-style energy landscapes.

5. **Approximation error propagation along the path.**  
   Can you provide either an empirical or theoretical discussion of how the mismatch between \(q_i\) and \(\hat q_i\) affects subsequent dual optimization and the ESS lower bound? For example, do you observe any drift in \(\mathrm{D}_{\mathrm{KL}}(\hat q_i \| q_i)\) as \(i\) increases, and does this correlate with deviations from the predicted ESS floor in Figure 6?

6. **Complexity and scalability of dual optimization.**  
   In practice, you note that dual optimization accounts for only about 0.01% of training time on alanine dipeptide (Appendix D.4). Can you comment on how this scales with dimension and the number of annealing steps? Is there any scenario where the dual optimization becomes a bottleneck, or can one expect it to remain negligible for, say, d ~ 10^3 with similar architectures?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The theoretical framing and derivations are solid overall, and the experimental methodology on molecular systems is careful and convincing. There are a few notational/typo issues (notably Eq. (16)) and some gaps in analyzing approximation error and teleportation quantitatively, but they do not fundamentally undermine the main claims.

## Presentation Rating

3: good.  
The paper is generally well written, with clear structure, useful figures (especially Figures 1–3, 5–6), and very detailed appendices. Some notation inconsistencies, overloaded symbols, and a crowded metric landscape (Table 2, Wasserstein/TICA metrics) detract slightly from clarity.

## Contribution Rating

3: good.  
The contribution is meaningful for the Boltzmann generator community: a principled constrained optimization view on annealing paths, a concrete and effective algorithm, and strong empirical gains on challenging peptide systems including a new benchmark. Conceptual novelty is moderate rather than dramatic, but the execution and empirical impact are strong.

## Overall Rating

8: Accept, good paper (poster).  
The work offers a coherent constrained variational perspective on annealing paths, a well-engineered instantiation with normalizing flows, and compelling improvements over strong baselines on important molecular benchmarks. While some aspects are incremental relative to broader constrained-optimization and annealing literatures and some details (entropy tuning, teleportation metrics) could be sharpened, the combination of theory, algorithm, and thorough empirical validation clearly meets and exceeds ICLR’s standards for a solid accepted paper.

## Reviewer Confidence

4: confident.  
I am familiar with variational inference, annealing/AIS/SMC, and Boltzmann generators, and I read the math and experimental sections carefully. I did not re-derive every proof line-by-line, but I checked the core derivations and scrutinized key equations and tables.