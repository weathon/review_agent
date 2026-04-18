Let me re-read the key technical sections from the paper content provided in the user message, focusing on the LRF canonicalization (Eqs. 5-12) and the theoretical claims.

The paper states in Eqs. 9-11 that:
- ē₃ = [0, 0, 1]^T (fixed, world gravity direction)
- ē₁ is constructed from ū by subtracting its ē₃ component
- ē₂ is constructed from v̄ by subtracting its ē₁ and ē₃ components

This is a Gram-Schmidt process where ē₃ is fixed as gravity. The harsh critic raises a concern that setting ē₃ = [0, 0, 1]^T "breaks full E_g(3)-equivariance" because it hardcodes a fixed orientation in world coordinates. However, let me think about this carefully.

Under the group E_g(3), the permitted transformations are: O ∈ O(3) such that Og⃗ = g⃗ (i.e., O preserves the gravity direction), and translations along g⃗. Since O satisfies Og⃗ = g⃗, and if g⃗ = [0, 0, 1]^T, then Ō[0, 0, 1]^T = [0, 0, 1]^T. So setting ē₃ = g⃗ (which equals [0, 0, 1] in the world frame) is *not* a hardcoded assumption — it's using the gravity vector which transforms correctly under E_g(3). The harsh critic's specific concern that "ē₃ is not derived from equivariant features" is actually wrong: ē₃ IS the gravity vector g⃗, which is part of the subequivariant graph and transforms correctly under E_g(3).

Wait, but the paper says ē₃ = [0, 0, 1]^T. If g⃗ = [0, 0, 1]^T, then this is just g⃗, so it transforms correctly. Under any O ∈ O_g(3) (where Og⃗ = g⃗), we have Oē₃ = O[0,0,1]^T = Og⃗ = g⃗ = [0, 0, 1]^T = ē₃. So ē₃ is actually invariant under O_g(3), not "breaking" anything. The harsh critic was wrong about this.

However, there IS a subtler valid concern: the Gram-Schmidt procedure can produce sign ambiguities or reflection ambiguities. The paper does not discuss what happens when ū has a very small component perpendicular to ē₃ (i.e., is nearly parallel to gravity), which could make the frame ill-conditioned. And there could be a sign ambiguity in ē₁ since if ū flips sign, so does ē₁.

The harsh critic's concern about "reflections that flip lateral directions" is partially valid but actually addressed by the E_g(3) formulation itself — O_g(3) includes rotations AND reflections that preserve g⃗. These reflections can flip the lateral direction, but the LRF construction from two predicted vectors ū and v̄ that are equivariant under O_g(3) should handle this: if the environment is reflected, ū and v̄ are reflected correspondingly, and the LRF transforms accordingly.

Let me also verify the "confounded empirical comparison" claim. The baselines are described as:
- EquiX: subequivariant but no morphology transform
- EvoX: morphology transform but no subequivariance  
- X: neither
- EquiEvoX: both

The concern is about architectural parity. Let me check what architectures are used for each.

The paper says: "We use conventional neural network φ^b_θ (such as MLP) to instantiate the behavior control sub-policy" and for EquiX, they would use the same network but on LRF-canonicalized inputs. The key question is whether non-equivariant baselines receive the same structured input or use flattened scalars.

Looking at section 3.2: "For Evo+DN: a variant using hand-crafted normalization where z⃗ are treated as scalars and the goal direction is used to construct the LRF" and "Evo: a non-equivariant variant that treats z⃗ as scalars without any LRF construction."

This confirms that non-equivariant baselines treat vector states as scalars (losing directional information), while EquiEvo preserves them as vectors. This IS a confound — the non-equivariant baselines have a fundamentally different (and weaker) state representation.

However, the Evo+HN and Evo+DN baselines DO use LRF normalization with vector states, just with hand-crafted frames rather than learned ones. These comparisons are fairer but still mix representation and architecture.

The ablations in Figure 8 (EquiActorCritic, EquiActor, EquiCritic, NoEqui) also mix these factors since NoEqui presumably also uses flattened scalars.

Now let me think about the overall assessment. The paper has:

**Real strengths:**
1. Novel and well-motivated problem formulation (subequivariance in co-evolution)
2. Good empirical results showing consistent improvements
3. Insightful analysis of how reward design affects evolved morphologies
4. Clear conceptual framework

**Real weaknesses:**
1. The empirical comparisons do NOT properly isolate subequivariance from representation differences (non-equivariant baselines lose vector state information)
2. The theoretical claim about LRF invariance is not fully verified in the main text (though the appendix claims proofs exist)
3. Limited task diversity and no generalization tests (e.g., testing on unseen rotations)
4. The LRF construction may have numerical instability when predicted vectors are near-parallel to gravity

**Weaknesses from harsh critic that I should weaken or remove:**
- The claim that ē₃ = [0, 0, 1]^T breaks E_g(3) equivariance is WRONG. ē₃ = g⃗ which IS equivariant/invariant under E_g(3).
- The claim that the paper doesn't justify morphological invariance at all is overstated — Section 3.1 does provide an argument, even if informal.

Let me now write the final review.</think>Now I have a thorough understanding of the paper and relevant calibration anchors. Let me write the final consolidated review.

---

## Summary

The paper introduces 3DS-MB, a benchmark for morphology-behavior co-evolution in 3D environments requiring rich directional reasoning, and EquiEvo, a framework that leverages $E_g(3)$-subequivariant graph neural networks and learned Local Reference Frames (LRFs) to inject geometric symmetry into the co-evolution process. EquiEvo projects agent states into dynamically computed invariant reference frames, enabling consistent value estimation across geometrically equivalent configurations and reducing the search space. Experiments on navigation and sumo tasks show EquiEvo consistently outperforms non-equivariant and morphology-only baselines.

## Strengths

- **Well-motivated problem formulation:** The identification that prior co-evolution benchmarks (Transform2Act, CompetEvo) rely on fixed-direction tasks that don't require rich spatial reasoning, and the proposed extension to variable-direction navigation and adversarial sumo tasks, addresses a genuine gap. The argument that geometric symmetry is underexploited in co-evolution is compelling.

- **Clear empirical improvements with insightful analysis:** EquiEvo consistently outperforms baselines across all three tasks (Ant Navigation, Humanoid Navigation, Ants Sumo). The ablations (EquiActor vs. EquiCritic, hand-crafted normalization variants) are informative, and the morphology-task mapping analysis (Figure 10) showing how reward design shapes evolved morphology symmetry is a particularly insightful contribution.

- **Principled architectural design:** The combination of subequivariant GNNs for LRF prediction with invariant policy/value networks is conceptually sound and well-aligned with recent geometric learning literature. The insight that consistent morphology value estimation requires equivariant behavior networks (Section 3.1) is a genuine and important observation for the co-evolution setting.

## Weaknesses

### Fatal
None.

### Major

- **Empirical comparisons do not isolate subequivariance from representation differences.** The non-equivariant baselines (X, EvoX, described as treating $\vec{z}$ as scalars) receive a fundamentally different and weaker state representation than EquiEvo, which preserves vector structure. Even the hand-crafted normalization ablations (Evo+DN, Evo+HN) still "treat $\vec{z}$ as scalars" and reconstruct an LRF from scalar inputs, rather than processing vector features through a standard GNN. This means the performance gap between EquiEvo and baselines conflates (a) the benefit of subequivariance as a structural constraint with (b) the benefit of richer vector-valued state representations processed by graph-structured architectures. A fairer comparison would use a non-equivariant GNN on the same vector-valued graph inputs (with arbitrary rotations of input coordinates) to isolate the symmetry constraint from the representational gain. Without this, the claim that "subequivariance is key to success" (Sec. 4.3) overstates what the experiments demonstrate. — This matters because the central scientific contribution is that symmetry exploitation, not just better engineering, drives the improvements.

- **No direct verification of symmetry exploitation claims.** The paper claims that subequivariance enables generalization to "diverse task spatial structures" and provides "consistent value for morphology value estimation," yet there is no experiment testing this directly. For example: (i) training with arbitrary orientations and evaluating on held-out orientations to test generalization; (ii) measuring whether learned morphology values are actually invariant across rotated initializations. The morphology-task mapping analysis (Figures 9-10) shows that reward design affects evolved morphologies, but this is about task structure rather than symmetry exploitation per se. Without at least one experiment verifying that symmetry generalization actually occurs (e.g., zero-shot transfer to rotated goals), the core theoretical claim remains an assumption rather than a demonstrated property. — This matters because the paper's main narrative is about exploiting geometric symmetry, but the evidence shows performance gains without confirming the mechanism.

### Minor

- **Structural transformation is omitted in two of three tasks.** The paper frames itself as addressing morphology-behavior co-evolution, but the "most challenging" Ant Navigation task is the only one that includes structural transformation (adding/removing limbs). Humanoid Navigation skips structural transformation to "maintain humanoid structure" and Ants Sumo omits it following prior work. This limits the generality of claims about co-evolution specifically, since only one task exercises the full pipeline. — This matters for accurately scoping the contribution, though does not invalidate the subequivariance results.

- **The LRF construction may have numerical instability issues.** The orthonormalization procedure (Eqs. 9-11) constructs ē₁ by subtracting the gravity-aligned component from ū. If ū is nearly parallel to gravity, the remaining perpendicular component could be very small, leading to numerical instability. The paper does not discuss this failure mode or any mitigation strategy (e.g., thresholding, regularization). — This is a practical concern worth acknowledging but unlikely to invalidate the core method.

- **The "dimensionality reduction" claim in the introduction is informal and partially misleading.** The statement that "under rotational symmetry, states and actions in any direction can be treated as equivalent, effectively reducing a 2D/3D problem to a simpler 1D/2D one" is conceptually appealing but informal. $E_g(3)$ does not include full rotational symmetry — it preserves gravity direction, so rotations are only around the gravity axis. This reduces the relevant symmetry from 3D to essentially 2D (horizontal plane), not 3D to 1D. — This is a minor presentation issue that could confuse readers about the actual symmetry group being used.

### Trivial
- The notation $\mathcal{G}_m$ (morphology graph) vs. $\tilde{\mathcal{G}}_g$ (subequivariant state graph) vs. $\mathcal{G}_g$ (invariant state graph) is used consistently but could be more clearly distinguished in a summary table for readability.

## Nice-to-Haves

- Comparison with data augmentation (random rotation of agent orientation during training) as an alternative symmetry exploitation strategy, to establish that architectural subequivariance provides benefits beyond what simpler augmentation can achieve.
- Wall-clock time / parameter count comparison between EquiEvo and baselines, to assess computational overhead introduced by subequivariant GNN layers.
- An experiment testing zero-shot generalization to held-out orientations (e.g., training on orientations in [0°, 180°] and evaluating on [180°, 360°]).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The LRF construction breaks E_g(3)-equivariance because ē₃ = [0,0,1]^T is hardcoded."** — This is factually wrong. Since $\vec{g} = [0,0,1]^T$ in the world frame, $\bar{e}_3 = \vec{g}$ is not "hardcoded" — it is the gravity vector, which is explicitly part of the subequivariant graph $\tilde{\mathcal{G}}_{\bar{g}}$ and transforms correctly under $E_g(3)$. For any $O \in O_g(3)$, $O[0,0,1]^T = [0,0,1]^T$ by definition of $O_g(3)$. The harsh critic incorrectly claimed this breaks equivariance.

- **"Missing comparison with other equivariant RL methods"** — This is asking for baselines outside the paper's scope. The paper already provides a 2×2 factorial comparison (equivariance × morphology transform) plus hand-crafted normalization ablations, which is a reasonable evaluation strategy for introducing subequivariance into co-evolution.

- **"Sim-to-real gap"** — This is a scope-creep criticism. The paper makes no claims about real-world deployment; sim-only evaluation is standard in morphology co-evolution research.

- **"Missing related work citations (BodyGen, etc.)"** — Per the rules, I should not flag missing related works as I cannot confirm their existence or relevance.

- **"Concerns about reproducibility / undisclosed hyperparameters"** — Per the rules, reproducibility concerns about hyperparameters are removed. The paper provides a hyperparameter table (Table 2 mentioned in Section 4.2).

- **"The formal proof of LRF invariance is only in the appendix"** — While true, deferring proofs to appendices is standard practice in ML venue papers. The main text provides the mathematical construction, and the appendix provides verification. This is not a fatal flaw, though the main text could benefit from a theorem sketch.

- **"Limited task diversity"** — Three tasks (two navigation variants + sumo) is a reasonable scope for a methods paper. Requesting more tasks is a generic "add more experiments" demand.

- **"Only 3 seeds, no confidence intervals"** — Single-run evaluation with 3 seeds is standard in RL/morphology co-evolution papers. Requesting statistical significance tests is a nice-to-have, not a weakness.

## Novel Insights

The morphology-task mapping analysis (Figures 9-10) reveals a striking finding: altering the reward structure (adding a forward reward) changes the evolved morphology from radially symmetric to laterally symmetric, demonstrating that the co-evolutionary process naturally discovers symmetry patterns dictated by task demands rather than developer-imposed constraints. This provides empirical evidence that geometric symmetry in morphology emerges from environment interaction — supporting the paper's central thesis that subequivariance helps the optimization process discover these task-appropriate symmetries more efficiently, rather than imposing them a priori.

## Suggestions

- **Add a non-equivariant vector-GNN baseline** that processes the same vector-valued graph representation as EquiEvo but without equivariance constraints (e.g., standard GNN on un-canonicalized coordinates). This would cleanly isolate the benefit of subequivariance from the benefit of structured representations.
- **Include at least one symmetry verification experiment**, such as evaluating a trained EquiEvo policy on rotated goal configurations without further training, or measuring value function consistency across geometrically equivalent states. This would directly validate the claimed mechanism.
- **Clarify the dimensionality reduction claim** in the introduction: replace "reducing a 2D/3D problem to a simpler 1D/2D one" with a more precise statement about horizontal-plane equivariance under $E_g(3)$.
- **Discuss the potential numerical instability** of the LRF Gram-Schmidt procedure when predicted vectors are nearly parallel to gravity, including any mitigations used in practice.

## Evaluation on Key Axes

- **Originality:** Moderate-high. The integration of subequivariant GNNs with LRF canonicalization into morphology-behavior co-evolution is novel, though the individual components (subequivariant GNNs, LRF canonicalization, Transform2Act framework) are from prior work.

- **Importance of research question:** High. Leveraging geometric symmetry in co-evolution of morphology and behavior is an important and underexplored direction.

- **Claim support:** Moderate. The central empirical claim that subequivariance drives improvement is confounded by representation differences in baselines. The theoretical claim of LRF invariance is asserted but only verified in the appendix, and the symmetry exploitation mechanism is not directly tested experimentally.

- **Soundness of experiments:** Moderate. The ablation grid (2×2 design) is conceptually clean, but the lack of architectural parity between equivariant and non-equivariant baselines limits causal inference. The three tasks are well-chosen but could benefit from a direct symmetry generalization test.

- **Clarity:** Moderate-good. The paper is generally well-written and organized, though some technical claims are informally stated and the dimensionality reduction claim is imprecise.

- **Value to community:** Moderate. Provides a useful benchmark extension and demonstrates that geometric symmetry matters for co-evolution, though the evidence for the specific mechanism (subequivariance via learned LRFs vs. richer representations) could be stronger.

## Score and Decision

**Calibration anchors:**
- BodyGen (morphology co-design, scores 6-8): Accepted spotlight. Strong empirical results with clear improvements, but some concerns about baselines and interpretability.
- EDiSon (design+control, scores 3-5): Rejected. Limited baselines, incremental novelty.
- EqNIO (subequivariant canonicalization, scores 5-8): Accepted poster. Clean theoretical contribution with good empirical results, but some questions about generalization.
- "Can Euclidean Symmetry Help RL" (equivariant RL, scores 3-6): Rejected. Overclaiming about symmetry benefits without sufficient experimental isolation.

This paper shares weaknesses with EDiSon (limited baselines) and "Can Euclidean Symmetry Help RL" (confounded empirical comparison for symmetry claims). However, it has stronger contributions than EDiSon (novel task formulation, principled architecture) and is methodologically more sound than "Can Euclidean Symmetry Help RL" (better ablation design). It is weaker than EqNIO (which has cleaner theoretical grounding) and much weaker than BodyGen (which has stronger baselines and cleaner experimental isolation).

The confounded empirical comparison is the main weakness that prevents acceptance as-is: the paper cannot cleanly attribute its improvements to subequivariance rather than representational advantages. However, the theoretical framework is sound (the LRF construction does appear to be E_g(3)-equivariant, contrary to the harsh critic's claim about ē₃), the problem is well-motivated, and the results are promising. This is a solid paper that needs better-controlled experiments to validate its central claim.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>