---
job_id: 1a4193e3-8a68-44d1-956b-99ce42e2d67e
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: Vit5M0G5Gb.pdf
paper: Saddle-to-Saddle Dynamics Explains a Simplicity Bias Across Neural Network Architectures
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper develops a theoretical framework for learning dynamics and simplicity bias in neural networks (linear, convolutional, ReLU, attention), squarely within learning theory, representation learning, and optimization topics central to ICLR.

## Minimum Quality
Pass ✅.  
All key sections are present (Abstract, Introduction, Related work, Methodology/Theory, Experiments/Simulations, Implications, Discussion). The math is nontrivial but mostly coherent, experiments are adequate for a theory paper (though limited), and there are no obvious fatal methodological or statistical flaws.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden instructions or attempts to manipulate automated reviewing systems within the paper content.

---

# Expected Review Outcome:

## Summary

The paper studies gradient-flow learning dynamics of a broad class of neural networks and proposes a unifying explanation for “simplicity bias” via saddle‑to‑saddle dynamics.  

For architectures that fit their general layer template (Eq. (1)) – including fully‑connected linear and ReLU networks, convolutional networks, quadratic networks, and linear self‑attention – the authors:  
1) characterize a hierarchy of embedded fixed points where solutions of narrower networks appear as saddles of wider networks (Theorem 1, Corollary 2);  
2) identify invariant manifolds on which wider networks behave like effectively narrower ones (Theorem 3);  
3) show that timescale separation (between directions in linear models, Theorem 4, and between units in quadratic models, Proposition 5) causes training trajectories to move near these invariant manifolds, producing stage‑like saddle‑to‑saddle dynamics and incremental increases in “effective width”.  

They support the theory with simulations across multiple architectures (Figures 1–5) and use it to predict how width, data spectrum, and initialization affect the presence and length of plateaus (Figure 2, Table 1).

## Strengths

1. **Conceptually unifying framework across architectures.**  
   The main conceptual contribution is to connect three ingredients – embedded fixed points (Theorem 1 / Corollary 2), invariant manifolds (Theorem 3), and timescale separation (Theorem 4, Proposition 5) – into a single story explaining stage‑like dynamics and “simplicity bias” as incremental recruitment of effective units. This applies not only to classical deep linear networks, but also to convolutional layers and self‑attention (via the generic layer form in Eq. (1)), which is a nontrivial generalization of prior, mostly architecture‑specific analyses.

2. **Clear structural theorems about the loss landscape.**  
   Theorems 1 and 3 are clean structural results:  
   - Theorem 1 systematically enumerates classes of embedded fixed points in wider networks, extending and organizing earlier constructions (Fukumizu & Amari) beyond the basic duplication/zero‑unit cases to homogeneous and linear activations (Eqs. (6),(7)).  
   - Theorem 3 shows that various algebraic relations between units (equality, proportionality, linear dependence, or being zero) define invariant manifolds. Section F.3 clarifies that these manifolds correspond to effective narrowing of the network. This gives a rigorous basis for talking about “effective width” and simplicity as “minimal units required”.

3. **Mechanistic explanation of saddle‑to‑saddle dynamics.**  
   The analysis of two concrete families – linear networks (Section 5.1) and quadratic networks / linear self‑attention (Section 5.2) – is insightful:  
   - In the linear case, Theorem 4 shows that small‑init gradient flow behaves like a linear system whose eigenmodes are singular vectors of Σ_yz, leading to exponential amplification along top singular directions and hence approximately low‑rank weights. This directly links the spectrum of Σ_yz to plateau durations and explains why early stages look rank‑1, then rank‑2, etc.  
   - In the quadratic case, Proposition 5 shows that distinct initial magnitudes across units induce a rich‑get‑richer effect, so one unit’s weights become O(1) while others stay O(ε) for long times. Combined with Theorem 3(ii), this explains why self‑attention and quadratic nets exhibit sparsity‑like effective width 1, then 2, etc.  
   These mechanisms are conceptually different (data‑induced vs initialization‑induced), which is nicely disentangled.

4. **Figures and simulations that tightly reflect the theory.**  
   - **Figure 1** is particularly effective: panel A’s cartoon visually explains saddle‑to‑saddle transitions as motion from one invariant manifold (cyan, 1 effective unit) to a higher one (yellow, 2 effective units), and panels B–G show loss curves with clear plateaus and abrupt drops, alongside first‑layer weight scatter plots illustrating rank‑1 / rank‑2 (linear), kink counts (ReLU), kernel/feature duplication (conv), and active heads (attention/quadratic). This makes the notion of “effective width” tangible.  
   - **Figure 2** then uses carefully controlled synthetic setups to test predictions: width scaling (Fig. 2A), spectrum exponent κ (Fig. 2B), initialization structure (Fig. 2C), and initialization scale (Fig. 2D). The qualitative match between theory and loss curves is convincing, e.g., in Fig. 2B where κ → 0 eliminates intermediate plateaus in linear nets but not in linear self‑attention.  
   - **Figure 3** and **Table 1** provide a nice bridge to “real” data: for MNIST binary tasks, plateaus and singular‑value growth are visible, and Table 1 lists singular values that roughly govern plateau lengths for different digit pairs, supporting the link between SVD gaps and plateau duration.  
   - **Figure 5** extends the story to deep networks, showing that three‑layer and transformer‑like architectures also exhibit saddle‑to‑saddle dynamics and that the visited saddles match the fixed‑point types in Theorem 1.

5. **Nontrivial predictions and explanatory power.**  
   Section 6 uses the framework to generate concrete, testable predictions:  
   - In linear networks, increasing width H does little once H is “enough”, whereas in quadratic/self‑attention models, increasing H shortens plateaus (Fig. 2A).  
   - Flattening the data spectrum (κ ↓ 0) collapses multiple linear‑net stages into a single jump from rank‑0 to rank‑D (Fig. 2B), while self‑attention still exhibits multiple stages because its timescale separation is driven by initialization, not spectrum.  
   - Initializing near but not at saddles (Fig. 2C) yields saddle‑to‑saddle dynamics without an initial plateau, an interesting regime that people rarely study.  
   - Varying initialization scale (Fig. 2D) smoothly interpolates between strong stage‑like behavior and more “lazy” exponential decay.  
   These are not trivial consequences of earlier linear‑network analyses.

6. **Thoughtful discussion and connections to broader literature.**  
   The Discussion (Section 7) and Appendix A/C situate the work relative to prior analyses of spectral bias, NTK dynamics, feature vs lazy learning, etc., and articulate limitations and open questions (e.g., exhaustiveness of invariant manifolds, deep transformers, Markovianity of saddle transitions). The connections to permutation symmetry breaking and to recent “symmetry‑to‑symmetry” hypotheses are well argued.

## Weaknesses

1. **Rigor gap between the approximate dynamical systems and the full nonlinear dynamics.**  
   The key dynamical results (Theorem 4 for linear nets and Proposition 5 for quadratic nets) are stated for simplified systems (Eqs. (10) and (14)), derived as leading‑order approximations near small initialization. However, the paper often interprets them as if they fully explain the behavior of the original gradient flow (Eqs. (9) and (44)).  
   - In the linear case, the approximation Σ_yz − W Σ_zz ≈ Σ_yz is valid only as long as W remains small. Yet saddle‑to‑saddle transitions involve excursions where weights grow to O(1). While Appendix G.3 describes the linearization near higher‑rank saddles (Eqs. (41),(42)), there is no rigorous control of the error terms or proof that trajectories indeed stay in a regime where the low‑rank approximation and the induced timescale separation remain valid up to the next plateau.  
   - In the quadratic case, Proposition 5 analyzes Eq. (14), which drops the Σ_ZZ‑dependent interaction terms in Eq. (44). The argument (based on the scalar analogy \(\dot v_i = v_i^2\)) is intuitive but heuristic; it does not bound the contribution of the Σ_ZZ‑dependent cross‑terms or prove that they cannot destroy the rich‑get‑richer behavior before O(1) growth.  
   As a result, the link “full gradient flow ⇒ repeated saddle‑to‑saddle transitions following invariant manifolds” remains at the level of a well‑motivated heuristic plus simulations, not a theorem. This matters because the paper sometimes phrases these dynamics as if they were proved, rather than as conjectural extrapolations from the approximations.

2. **Strong and somewhat restrictive assumptions on the setting.**  
   The theoretical results rely on full‑batch gradient flow, squared loss, and mostly small isotropic Gaussian initialization, with no explicit noise or mini‑batch SGD.  
   - The embedded fixed‑point and invariant‑manifold constructions (Theorems 1 and 3) are fairly general, but the dynamical analysis relies heavily on small‑init asymptotics and, in practice, full‑batch updates (Appendix I uses GD, not SGD). It is not clear to what extent saddle‑to‑saddle dynamics and “effective width increments” survive under typical large‑scale training regimes where mini‑batch noise and adaptive optimizers are present.  
   - The focus on activations that are linear or homogeneous in the weight parameters u (or polynomial after a Taylor expansion) means important practical cases like tanh or sigmoid are not fully captured. Section 5 and Figure 4 discuss some nonlinear activations, but the theory explicitly notes that for tanh (Fig. 4D) rank‑one weights do not correspond to invariant manifolds, so the same mechanism does not apply. This makes the claimed “universality” of the mechanism weaker than the narrative in the introduction suggests.

3. **Novelty relative to prior work on singularities and embedded solutions is partly incremental.**  
   The paper’s landscape results build on a long line of work on singularities / embedding of narrower networks into wider ones (Fukumizu & Amari 2000; Amari et al.; Wei et al.; Zhang et al. 2021; Simsek et al. 2021; Fukumizu et al. 2019).  
   - Equation (4) and (5) essentially recapitulate classic constructions (duplication and zero units) from Fukumizu & Amari. The main new pieces are the homogeneous extension (Eq. (6)) and the fully linear combination case (Eq. (7)), along with the explicit invariant‑manifold statements (Theorem 3). While these are useful generalizations, they are not conceptually far from what one could derive once the embedding idea and permutation symmetry are understood.  
   - The timescale‑separation in linear networks is closely related to, and in some cases more general than, “silent alignment” (Atanasov et al. 2022) and early‑phase linearization (Hu et al. 2020). Theorem 4 extends silent alignment to vector‑output settings, but again this feels like an elaboration of known linear‑network phenomena rather than a qualitatively new behavior.  
   The paper would benefit from a crisper, more honest positioning of what is fundamentally new versus what is a synthesis and extension of existing elements.

4. **Limited empirical validation beyond toy settings.**  
   While this is primarily a theory paper, the empirical side is mostly synthetic or low‑dimensional, with the only “real” dataset being binary MNIST (Figure 3, Table 1).  
   - Figures 1–2, 4–5 mostly consider 2D inputs and small networks, or linear self‑attention on synthetic in‑context regression. These are idealized and carefully controlled, which is good for illustrating the math, but they do not demonstrate that the proposed mechanism is the dominant explanation for stage‑like training dynamics in modern large‑scale models (e.g., transformers on language tasks).  
   - For transformers, the only experiments are linear or very shallow models (Fig. 1F, Fig. 4A, Fig. 5E). It remains unclear whether non‑linear attention with residuals and MLPs, trained with typical initializations and optimizers, actually follows saddle‑to‑saddle paths aligned with the invariant manifolds described.  
   - The predictions in Section 6 (effects of κ, H, initialization scale) are only tested on narrow families (linear FC nets, linear self‑attention). It would strengthen the contribution to see at least one moderately realistic setting (e.g., a mid‑scale CNN or transformer) where plateau durations quantitatively track singular value gaps as the theory suggests.

5. **Some mathematical arguments are compressed or rely on slightly opaque steps.**  
   A few specific points where the exposition or rigor could be improved:  
   - In the proof of Theorem 1(iii) and Theorem 3(iii), Eq. (25) uses Euler’s homogeneous function theorem to assert that \(\frac{\partial \phi(z;u)}{\partial u_n}\big|_{u=\gamma u^*} = \frac{\partial \phi(z;u)}{\partial u_n}\big|_{u=u^*}\) for degree‑1 homogeneous φ. This equality is not obvious in general: degree‑1 homogeneity implies certain scaling relations (e.g., φ(αu) = αφ(u)) and relations between φ and its gradient, but it does not generally make the gradient invariant under scaling. The authors appear to rely on additional smoothness and specific structural properties; spelling out the precise conditions under which Eq. (25) holds (and perhaps giving a short derivation) would clarify the validity of this step.  
   - Similarly, Proposition 5 assumes Σ_yZ is symmetric with positive and negative eigenvalues and then reduces the dynamics to Eq. (55), but the asymptotic argument bounding t_final and concluding that only the largest‑initialized unit reaches O(1) requires several approximations (Eqs. (59)–(63)). The final step that typical spreads in initial conditions imply O(1) spreads in t_∞, hence “one unit gets big, the rest stay O(ε)” would benefit from a more formal probabilistic statement (e.g., high‑probability bounds over the Gaussian initialization) rather than order‑of‑magnitude reasoning.

6. **Scope of “simplicity bias” notion is somewhat narrow and architecture‑specific.**  
   The paper defines simplicity in terms of “effective width” (number of effective units, kernels, or heads). While this is natural within their framework, it does not directly connect to other popular notions of simplicity, such as low‑frequency preference, smoothness, or MDL/description‑length based measures.  
   - For example, a rank‑2 linear map is “more complex” than rank‑1 in the sense of effective units, but it may still be simple in other complexity measures.  
   - The discussion section briefly touches on connections to stationary simplicity bias and Occam/MDL work, but the theoretical bridge is not fully developed. This limits how broadly one can interpret the results as an explanation of “simplicity bias” writ large, as opposed to a particular architectural bias toward low effective width in the training dynamics.

7. **Presentation is dense and occasionally hard to follow for non‑experts.**  
   The paper packs a lot of content: general layer formalism (Eq. (1)), multiple theorems, two detailed dynamical analyses, and long discussions and appendices. While the writing is generally careful, it demands significant effort from the reader.  
   - Some central definitions (e.g., “effective width” and its formal link to invariant manifolds) are scattered between main text (Sections 4–5) and Appendix F.3; consolidating them in the main body might improve accessibility.  
   - The interplay between Theorem 1 and Theorem 3 could be emphasized more clearly and earlier: right now the story “embedded fixed points ⇔ invariant manifolds ⇔ saddle‑to‑saddle” is clear only after careful reading.  
   - The repeated discussion of linear vs quadratic vs general activations (Section 5, Figure 4, Appendix C) sometimes feels redundant and could be streamlined to highlight the essential mechanisms.

8. **Empirical negative cases and boundary regimes are underexplored.**  
   Section 7 briefly mentions that tanh networks likely do not have saddle‑to‑saddle dynamics because homogeneity is missing and rank‑1 weights do not lie on invariant manifolds, and Figure 4D supports this with one toy example. However, there is no systematic exploration of when saddle‑to‑saddle fails across architectures, datasets, and initialization regimes.  
   - For example, the paper argues that large isotropic initialization moves one away from invariant manifolds (Fig. 2D), but it does not quantify how large is “too large” or map out phase diagrams of dynamics (e.g., in terms of initialization scale and κ, analogous to rich vs lazy regimes in linear nets).  
   - Similarly, the negative example of tanh is isolated; more systematic variation across activations (Fig. 4 shows several, but with limited analysis beyond descriptive comments) would give a clearer picture of the limits of the mechanism.

## Potentially Missing Related Work

The paper’s related‑work coverage on simplicity bias is extensive but omits several recent, directly relevant works:

1. **Chang, X., Wang, T., Sun, C. (2026). “A Modern Look at Simplicity Bias in Image Classification Tasks.”**  
   This paper empirically analyzes simplicity bias in modern image classifiers (including CLIP) and how it affects generalization and robustness. It is directly relevant to the broader context of simplicity bias discussed in Sections 1 and 7, and should be cited and contrasted in the Related Work (Appendix A.3) and possibly in the Discussion, to connect the proposed dynamical mechanism with empirical observations in realistic vision models.

2. **Marty, T., Elmoznino, E., Gagnon, L. (2026). “A Compression Perspective on Simplicity Bias.”**  
   This work frames simplicity bias explicitly in MDL / compression terms. Given that Section A.3 and the Discussion reference MDL, Occam’s razor, and Kolmogorov‑complexity‑based explanations, this paper should be included in that discussion. It would help clarify how the authors’ “effective width” notion might relate to compression‑based measures.

3. **Du, Z. (2023). “Hierarchical Simplicity Bias of Neural Networks.”**  
   This paper studies simplicity bias at multiple hierarchical levels in deep networks, which is conceptually close to the idea of “effective width per layer” and the nested hierarchy of saddles (Corollary 2). It should be discussed in the main Related Work section on simplicity bias and in Section 7 (“Deep networks”) when the authors speculate about which layers recruit new effective units.

4. **Gatmiry, K., Li, Z., Reddi, S. (2024). “Simplicity Bias via Global Convergence of Sharpness Minimization.”**  
   This paper links sharpness minimization objectives to simplicity bias. Since the current work ties simplicity to the structure of the loss landscape (embedded saddles and invariant manifolds) rather than explicit sharpness regularization, a short comparison in Appendix A.3 or Section 7 would be valuable, highlighting how the two perspectives (dynamics of plain GD vs explicit sharpness minimization) may or may not be compatible.

## Questions

1. **Scope of the approximate dynamics.**  
   For the linear case, can the authors clarify more formally under what conditions (on initialization scale and training time) the approximation Σ_yz − W Σ_zz ≈ Σ_yz in Eq. (9) is valid enough that Theorem 4 accurately predicts the first and subsequent saddle escapes? In particular, is there a quantifiable bound on \(\|W\Sigma_{zz}\|\) relative to \(\|\Sigma_{yz}\|\) at the point where the trajectory is near a rank‑1 or rank‑r saddle?

2. **Role of mini‑batch noise.**  
   Have the authors checked whether similar saddle‑to‑saddle dynamics and effective‑width increments occur under mini‑batch SGD with realistic batch sizes (e.g., on the MNIST binary tasks of Figure 3 or the in‑context regression in Figure 1F)? If so, do the plateaus and singular‑value growth persist, or does noise smear out the timescale separation?

3. **Conditions for Eq. (25).**  
   Could the authors provide a brief derivation or additional assumptions under which Eq. (25) holds for degree‑1 homogeneous φ? In its current form, it is not obvious that the gradient with respect to u is invariant to scaling u → γu; a short justification would significantly increase confidence in Theorem 1(iii) and Theorem 3(iii).

4. **Quantitative tests of plateau duration predictions.**  
   In Figure 3 and Table 1, there is a qualitative match between singular values and plateau lengths. Could the authors provide a more quantitative comparison, for instance plotting plateau duration vs 1/(s_k − s_{k+1}) or similar, to show that the timescale separation predicted by Theorem 4 holds numerically? This would strengthen the empirical support for the claimed mechanism.

5. **Beyond effective width: connection to other simplicity notions.**  
   Do the authors see a way to link their “effective width over time” notion to MDL‑style complexity or to spectral bias (low‑frequency first)? For example, in the linear setting, does increasing rank correspond to increasing some compression or frequency‑based complexity measure, and can this be made precise?

6. **Negative cases and phase diagrams.**  
   The Discussion touches on tanh networks and large initialization as cases where saddle‑to‑saddle is unlikely. Could the authors systematically map out, for at least one architecture family (say, two‑layer ReLU or quadratic nets), a phase diagram over initialization scale and data spectrum κ indicating where clear plateaus appear, where they disappear, and how strongly the trajectory adheres to invariant manifolds? That would help delineate the real scope of the theory.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A. The work is purely theoretical/algorithmic with standard synthetic and MNIST data and does not raise evident ethics, privacy, or safety concerns.

## Soundness Rating

3: good.  
The structural results (Theorems 1 and 3) are sound and clearly proved; the dynamical analyses (Theorem 4, Proposition 5) for simplified systems are also carefully derived. However, the extension of these approximations to full gradient flow is heuristic, and some steps (e.g., Eq. (25), details of Proposition 5) could be more rigorous.

## Presentation Rating

3: good.  
The paper is generally clearly written and well organized, with helpful figures (especially Figures 1–3, 5 and Table 1). It is, however, quite dense and technically heavy, and a few key arguments are somewhat compressed, which may limit accessibility.

## Contribution Rating

3: good.  
The work offers a meaningful and broadly relevant conceptual framework for understanding stage‑like training dynamics and simplicity bias across multiple architectures. While parts of the theory build on and extend existing ideas, the synthesis, cross‑architecture perspective, and implications are valuable to the community.

## Overall Rating

8: Accept, good paper (poster).  
The paper provides a well‑articulated and largely sound theoretical framework that ties together embedded saddles, invariant manifolds, and timescale separation to explain simplicity bias and saddle‑to‑saddle dynamics across several architectures. While some aspects of the dynamics are heuristic and empirical validation is limited to relatively simple settings, the conceptual contribution, breadth of applicability, and quality of analysis make this a strong and worthwhile addition to the ICLR program.

## Reviewer Confidence

4: confident.  
I am familiar with the literature on learning dynamics, deep linear networks, and simplicity bias, and I carefully checked the main derivations and figures. Some technical details in the appendices and finer points of the quadratic dynamics could still conceal subtleties, but it is unlikely that these would overturn the overall assessment.