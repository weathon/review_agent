---
job_id: 683a14d8-9e8e-4748-8388-99558a8e9747
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: TgLW2DiRDG.pdf
paper: Characterizing the Discrete Geometry of ReLU Networks
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies the geometry and combinatorics of ReLU networks, including bounds on linear regions and connectivity graphs, which lies squarely within learning theory / representation learning and interpretation of neural networks, all core ICLR topics.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work in Appendix A, Methodology/Theoretical Approach in Sections 2–3, Algorithm in Section 4, Experiments/Results in Section 5, Discussion/Conclusion in Section 6) are present. The work is technically nontrivial, the mathematics is carefully stated with detailed proofs in the appendix, and the experiments are coherent (though limited by enumeration costs). I do not see fatal methodological, theoretical, or evaluation flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no instructions targeting automated review systems, no hidden prompts, and no suspicious formatting or content indicative of manipulation.

---

# Expected Review Outcome:

## Summary

The paper studies the discrete geometry of fully connected ReLU networks by analyzing the polyhedral complex of linear regions and its connectivity graph (regions as nodes, shared faces as edges). It proves that, under standard genericity/supertransversality assumptions and for almost all weights, (i) the average degree of the connectivity graph is at most \(2d\) and in fact approaches \(2d\) as the network grows, (ii) every region has at least \(\min(n_1, d)\) neighbors, and (iii) the graph diameter is bounded by \(O((m+1)^\ell)\), independent of input dimension. The authors also give an LP-based algorithm (Algorithm 1) to enumerate regions and empirically verify the bounds on synthetic and real data, while exploring how connectivity interacts with where training data lies.

## Strengths

1. **Clear, non-obvious geometric results with tight bounds.**  
   The main theorems are conceptually clean and nontrivial: Theorem 3.1 / 3.4 show that the *average* number of faces of a \(k\)-cell is at most \(2k\), and in particular the average degree of the region adjacency graph is at most \(2d\), independent of depth and width. The monotonicity and asymptotic tightness results (Theorems 3.6–3.7) are technically neat, especially the hyperplane-arrangement argument using Buck’s formula. The diameter bound in Theorem 3.8, \(O(m^\ell)\) independent of \(d\), is also quite insightful given the exponential region growth in \(d\).

2. **Technically careful use of sign-sequence combinatorics and BH decomposition.**  
   The extension of hyperplane arrangement results (Fukuda et al., 1991) to bent-hyperplane (BH) complexes is nontrivial. The categorization of cells into three types relative to a BH (Lemma 3.2) and the counting recursion in Lemma 3.3 are well thought out. The inductive proof of Theorem 3.4 in Appendix B, combining induction in both dimension \(d\) and number of BHs \(n\), is carefully spelled out and seems internally consistent.

3. **Good integration with the sign-sequence and topological framework.**  
   The paper builds directly on Masden (2025) and Grigsby & Lindsey (2022), using the bijection between cells and sign sequences and the properties of ReLU complexes under restriction. Section 2, with Figure 2 and the surrounding explanation, gives a very intuitive geometric picture of BHs, how they bend, and how they induce the complex; this significantly improves accessibility of the subsequent proofs.

4. **Concrete algorithmic instantiation and exhaustive enumeration.**  
   Even though the main contributions are theoretical, Section 4 gives a clear LP-based algorithm (Algorithm 1) to construct the connectivity graph from sign sequences, including the explicit layerwise formulas for the half-spaces (Equations (2)–(3) visually annotated in **Figure 9 / img-8.jpeg**). This is valuable for anyone wanting to reproduce or extend the experiments and for practitioners attempting to inspect actual complexes.

5. **Empirical characterization supports and enriches the theory.**  
   The experiments go beyond a toy check. On synthetic Gaussian-cluster data, **Figure 4 (img-9.jpeg)** and **Table 3 / Table 1** show that (i) the average number of neighbors in every complex is below \(2d\), (ii) the mean approaches \(2d\) as depth/width grow, and (iii) the distributions are unimodal and right-skewed, which is a stronger qualitative statement than the theorem. **Figure 5 (img-11.jpeg)** compares empirical diameters with the theoretical upper bound and shows diameters grouping mainly by depth and width, essentially independent of input dimension, which nicely backs Theorem 3.8.

6. **Interesting and somewhat surprising data-related observations.**  
   Section 5.2 shows that regions containing data tend to have higher degrees than average (e.g., **Figure 6 (img-14–16.jpeg)**), and that in classification tasks these regions are disproportionately unbounded compared to overall, while in the regression task they are more often bounded (**Figure 7 (img-17–19.jpeg)**). These are not just sanity checks; they suggest a structural bias of training toward more highly connected regions which could matter for generalization and robustness.

7. **Clarity and pedagogical quality of figures.**  
   **Figure 1 (img-0–3.jpeg)** efficiently introduces the notion of regions, complexes, and the connectivity graph, with region A’s neighbors visually highlighted and a degree histogram in panel (d). **Figure 3 (img-5–7.jpeg)** is particularly helpful: it maps the abstract category decomposition from Lemma 3.2 into color-coded cells and nodes, making the proof strategy much easier to digest.

## Weaknesses

1. **Assumption set is strong and not deeply discussed in terms of limitations.**  
   The main theorems rely on genericity and supertransversality assumptions (Definition B.3 and the two conditions following it), essentially requiring that at most \(d\) BHs intersect at a point and that all BH arrangements in all layers are “nice”. While this follows Masden (2025) and holds almost surely under continuous random initializations, trained networks often develop structured weight correlations (e.g., low-rank or near-parallel filters). The paper does not analyze how sensitive the bounds are to mild violations of these assumptions or provide explicit counterexamples; nor does it discuss whether the bounds might still hold in some approximate sense for non-generic weights. A short discussion quantifying or at least illustrating failure modes would strengthen the applicability claims.

2. **Scalability of Algorithm 1 is inherently poor and not quantified.**  
   Constructing the connectivity graph via BFS with a linear program per neuron per region is clearly exponential in \(d, n\). While this is expected, the paper does not provide any formal complexity analysis of Algorithm 1, beyond citing that enumeration is intractable. For instance, in Section 4, the SOLVELP subroutine is only described qualitatively, and there is no discussion of worst-case complexity in terms of number of regions or dimension. This matters because a key empirical claim is based on “complete enumeration” or 8M-vertex BFS truncations for some nets; readers would benefit from explicit runtime and memory scaling curves, or at least a concise complexity statement.

3. **Experimental regime is necessarily low-dimensional and somewhat narrow.**  
   All complexes are computed for classifier parts whose effective input dimension \(d\) is at most 10 (MNIST, CIFAR10 latent spaces, and 8-D California Housing), and even then the enumeration is truncated at 8M regions in CIFAR10 and housing. This is understandable, but it means the empirical evidence for bounds like Theorem 3.8 is based on relatively small diameters (tens to low hundreds) and architectures with widths up to 16 or 128 in the last layers. The paper does not attempt any extrapolation via partial sampling (e.g., random walks or local region sampling) to speak more convincingly about very wide/deep modern networks. At minimum, a discussion of how much we can trust the observed trends for large-scale models would be appropriate.

4. **Some parts of the mathematics, while correct-looking, are opaque and could be clarified.**  
   - In Theorem 3.4 (Appendix B, Pages 17–19), the double induction in both \(d\) and \(n\) is dense. In particular, the step where Eq. (5) is applied to \(h_i\), treating it as a \((d-1)\)-dimensional ReLU complex with \(n-1\) neurons, is crucial; only a brief sentence justifies this. It would be helpful to explicitly state that the restriction of genericity/supertransversality to \(h_i\) implies that its own \((d-1)\)-dimensional complex satisfies the induction hypothesis, and to verify carefully that \(h_i\) indeed has \(n-1\) BHs, not \(n\) (since neuron \(i\) itself does not induce a BH inside \(h_i\)).  
   - In Section 3.6 (Theorem 3.6’s proof, Page 19–20), the final step that “since after the \(d\)-th term the sequence is monotonically increasing and bounded below and above by \(d\) and \(2d\)” is slightly abrupt; the argument about the asymptotic limit could be more explicit or moved to Theorem 3.7 to avoid confusion.  

   These are not correctness issues but hurt accessibility, especially for readers less familiar with BH complexes.

5. **Algorithm 1 pseudo-code has notation and indexing that are confusing.**  
   In Algorithm 1 (Page 7), line 5 uses `for i ∈ {0,..., n}` but \(n\) is the number of hidden neurons; indices elsewhere (e.g., in Section 2 and 4) start from 1. Similarly, the use of \(-\Phi_{s_i}\) and \(\beta_s + e_i\) on line 6 is not fully specified in the main text, only in Appendix D, leading to potential confusion on exactly which row is being tested. For a paper that aims to be a reference for geometry enumeration, a fully consistent and explicit description of indices and shapes of matrices \(\Phi_s, \beta_s\) would be valuable.

6. **Connections to prior/complementary graph-theoretic work could be sharpened.**  
   The introduction briefly mentions Dhayalkar (2025) and Liu et al. (2023b), but the relationship to those graph-based frameworks is only sketched. For example, Dhayalkar’s ReLU Transition Graph encodes edges when two regions differ by a single neuron’s activation, similar to the connectivity graph here, but with more emphasis on functional transitions. It would help to explicitly contrast the connectivity graph degree/diameter results with any existing results on transition graphs (e.g., VC-dimension bounds) and clarify whether the graphs are isomorphic or if the connectivity graph is a subgraph or augmentation.

7. **Empirical insights on data-containing regions are intriguing but rather speculative.**  
   In Section 5.2, the authors comment that in classification tasks, training data tends to lie in more unbounded, highly connected regions, while in regression they lie more often in bounded regions. While **Figure 6** and **Figure 7** do show clear histograms, the discussion is qualitative and somewhat hand-wavy (“for classification, the network may have to focus its complexity on the spaces between classes…”). This is fine as a hypothesis, but framing should be more cautious, or supported with additional controlled experiments (e.g., varying class separability, margin, or regularization).

8. **Missing discussion of some very recent related work.**  
   While the references are broad and include Masden (2025), Huchette et al. (2023), and Fan et al. (2024), a few highly relevant recent works on the geometry/topology of ReLU networks and their parameter space (see “Potentially Missing Related Work”) are not discussed. This does not invalidate results, but it weakens the positioning in a fast-moving literature.

## Potentially Missing Related Work

1. **Nurisso, Leroy, Petri, “Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Singularities,” 2026.**  
   This work studies the connectivity and singularities of the *parameter* space of ReLU networks. While the present paper focuses on the *input* space complex, both works analyze global geometric constraints imposed by ReLU architectures. It would be appropriate to mention it in Appendix A (Detailed Related Work) and briefly contrast their parameter-space connectivity results with the input-space connectivity graph properties established here.

2. **Ghafoor and Akutsu, “ReLU Networks for Exact Generation of Similar Graphs,” 2026.**  
   This paper uses ReLU networks for generating graphs within a bounded edit distance. Since the current submission heavily relies on graph representations of region adjacency, it would be relevant to cite this work, perhaps in Appendix A when discussing applications of ReLU geometry to graph-structured data or generative settings, clarifying that the focus here is on the geometry of region graphs, not on graph generation.

*(Dhayalkar (2025) is already cited and briefly discussed in the introduction, so I do not list it as missing.)*

## Questions

1. **Robustness to non-generic weights.**  
   Do the authors have concrete examples (even small 2D networks) where genericity/supertransversality fail and the average degree exceeds \(2d\) or the diameter bound fails, or do the inequalities still hold empirically in such degenerate regimes? A small synthetic experiment deliberately enforcing parallel or coincident first-layer hyperplanes would clarify how fragile the bounds are.

2. **Approximate enumeration for higher dimensions.**  
   Have the authors considered using local random walks or MCMC on the connectivity graph to estimate degree distributions and diameters without full BFS + LP enumeration? If so, any preliminary evidence on whether the observed approach-to-\(2d\) persists for, say, \(d=20\) or larger MLPs would greatly increase the practical impact.

3. **Relation to ReLU Transition Graphs and other graph abstractions.**  
   Could the authors clarify if their connectivity graph is isomorphic to the ReLU Transition Graph of Dhayalkar (2025) when restricted to forward transitions (or vice versa), or if there are structural differences (e.g., directionality, edge multiplicity)? Some formal mapping would help integrate the literatures.

4. **Tightness of the diameter upper bound.**  
   Theorem 3.8 gives an \(O((m+1)^\ell)\) bound, but empirically in **Figure 5** and **Figure 11** the diameters seem to grow much more slowly (almost logarithmic in the bound). Do the authors have any conjecture or partial result on an improved upper bound that better matches observed scaling, perhaps under additional assumptions on weight distributions?

5. **Algorithmic reproducibility details.**  
   For Algorithm 1 and Appendix D: what precise relaxation magnitude is used in \(\beta_s + e_i\), and how sensitive are results to this choice? Are there observed numerical issues (e.g., near-degenerate facets, duplicate regions) and how are they handled?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

4: excellent.  
The theoretical arguments are nontrivial but well structured; the main combinatorial identities and asymptotic arguments appear correct, and the empirical results are consistent with the theory and clearly described, despite some limitations in scale.

## Presentation Rating

4: excellent.  
The paper is well organized, figures are carefully designed (especially Figures 1–3, 4, 5, 6–7, 10–12), notation is mostly consistent, and both the main text and appendices provide a coherent story. Minor notation/indexing issues in Algorithm 1 and a few dense proof segments could be polished.

## Contribution Rating

4: excellent.  
The work provides new, architecture-level bounds and asymptotic characterizations of the connectivity of ReLU region complexes, filling a clear gap between “number of regions” analyses and purely local geometric studies. These results are likely to be reused in future theoretical and empirical work on deep network geometry.

## Overall Rating

8: Accept, good paper (poster).  
The submission offers solid, technically sound advances in understanding the discrete geometry of ReLU networks, with clear theorems, proofs grounded in contemporary topological/combinatorial machinery, and supportive experiments that illuminate both average degree and diameter behavior and the role of training data. Some assumptions are strong, the enumeration-based experiments are necessarily limited in scale, and a few connections to very recent related work could be deepened, but overall the contribution is substantial and well argued.

## Reviewer Confidence

4: confident.  
I am comfortable with the geometric/combinatorial arguments and checked the key theorems and algorithmic details; while I did not rederive every step in the appendices, I am confident in the overall assessment and believe any remaining issues are minor.