Now I have enough calibration material. Let me synthesize the final review.

The key papers I'll calibrate against:
- DDRL (scores 3-5, withdrawn/reject): RL for constrained CO, limited baselines, small problem sizes, scope concerns
- FneYHZU19U (scores 3-6, reject): Constrained graph clustering with Cheeger inequality, loose bounds, limited baselines
- VeFmnRmoaW/MetroGNN (scores 3-6, reject): RL+GNN for traffic, missing statistical measures, underspecified MDP
- oqdcThIQjA (scores all 3, withdrawn): Fast graph clustering/normcut, limited experiments, small datasets
- Coreset Spectral Clustering (scores 3-10, accept): Strong theory + experiments for normalized cut

Now I'll write the consolidated review:</think>## Summary

The paper introduces Wedge and Ring Transformers (WRT), a reinforcement learning approach using PPO with Transformer architectures for minimizing the Normalized Cut (NC) on weighted planar graphs under shape constraints—specifically partitions shaped as rings and wedges around a predefined center. The key technical idea is to transform graph nodes into polar-coordinate-based sequential representations (ring/wedge) that make the partition problem amenable to Transformer processing, converting a continuous action space into discrete "split point" actions. The paper also provides Cheeger-type bounds for ring/wedge partitions on unweighted spider-web graphs and demonstrates empirical advantages over METIS, spectral clustering, ClusterNet, and NeuroCUT on synthetic and city traffic graphs.

## Strengths

- **Novel and well-motivated problem formulation:** The paper identifies a genuine gap—classical and existing RL-based partitioning methods cannot enforce shape constraints on partitions. The traffic simulation motivation is concrete and the constrained action space is a meaningful contribution to the partitioning literature.

- **Elegant structural insight:** The ring and wedge transformations (Section 5.2) that project graph nodes onto 1D sequences (a line for rings, a circle for wedges) based on polar coordinates are intuitive and effective. The conversion from continuous to discrete action space (choosing split points between consecutive nodes) is a clean simplification that reduces learning difficulty.

- **Thoughtful training design:** The two-stage training strategy (wedge-first, then ring with frozen wedge parameters) addresses a real RL optimization challenge where joint training causes the components to interfere. This is a practical and potentially transferable insight for multi-stage structured RL problems.

- **Comprehensive ablation variants:** The paper includes WRT_c2e, WRT_sr, WRT_nfw, and WRT_npr variants to validate design choices, which strengthens the methodological contribution.

- **Competitive empirical results in-domain:** Within the ring/wedge-structured setting, WRT achieves the best NC scores across all configurations in Table 1 and shows transfer capability in Table 2.

## Weaknesses

### Major

- **Evaluation is heavily skewed toward data matching the method's inductive bias:** The synthetic "Predefined-weight" graphs are explicitly constructed with lower weights on inter-ring/inter-wedge edges (Section 6.1), making them tailor-made for ring/wedge partitioning. The "Random-weight" graphs still use spider-web topology. The real-world evaluation uses subgraphs from a single city's traffic map, which naturally have radial structure. There is no evaluation on planar graphs that lack radial symmetry (e.g., grids, random geometric graphs, mesh graphs) where WRT's constraint would be misaligned with the data. This means the headline empirical advantage over unconstrained baselines (METIS, spectral clustering)在很大程度上是预设的: a method hard-wired for ring/wedge structures excelling on ring/wedge-structured data does not demonstrate generally superior NC optimization. The paper's claims (e.g., Section 6.3.1: "METIS and Spectral Clustering... cannot reach better performance because it is hard to find best results in such huge action space") overstate what the experiments show.

- **Unfair comparison structure favors WRT by design:** General-purpose baselines (METIS, spectral clustering, NeuroCUT, ClusterNet) optimize over all possible partitions, while WRT restricts to ring/wedge shapes—a strictly smaller search space well-aligned with the data. The paper does not include baselines that incorporate shape constraints (e.g., METIS post-processed to produce ring/wedge partitions, or dynamic programming on the transformed 1D sequence). Additionally, Section 2.2 states that NeuroCUT and ClusterNet "do not handle weighted graphs," yet they appear as baselines in Section 6.2 without explaining how edge weights are handled. The "Bruteforce" baseline that searches over ring/wedge partitions would be the fairest comparison, but its implementation is not clearly described and it performs surprisingly poorly—this warrants discussion. The paper also does not clarify whether NeuroCUT and ClusterNet are retrained with the same NC objective on the same data, or applied off-the-shelf.

- **Theory-practice gap in Cheeger bounds:** Proposition 1 provides Cheeger-type inequalities only for *unweighted* spider-web graphs $G_{n,r}$, while all experiments use *weighted* graphs that are far from this toy family. No extension to weighted or general planar graphs is sketched, and more critically, there is no quantitative connection between the bounds and WRT's performance—no evaluation of NC vs. the bound, no analysis showing WRT approximates the minimizers $\phi_{n,r}(k)$ or $\psi_{n,r}(k)$. The theoretical contribution is disconnected from the practical method and contributes little beyond confirming that ring/wedge partitions satisfy *some* Cheeger-type bound on a very special graph class.

### Minor

- **No error bars or statistical confidence:** All tables report single mean values across 100 test graphs. PPO is a stochastic method; single-run results are insufficient for reliable comparison, especially given that WRT uses multiple random samples at test time (Section 5.5.2) while it is unclear whether baselines receive similar multi-sampling.

- **Incomplete specification:** Key details are deferred to the appendix: definitions of Ringness/Wedgeness metrics, the post-refinement algorithm, the dynamic programming approach for ring partition, and ablation results. The observation space $S = \{G, k_r, k_w, P\}$ is described conceptually but the graph encoding is underspecified. The Partition-Aware MHA's attention mask generation ("element-wise transformation on V") is vague. These gaps limit reproducibility.

- **Center point $o$ is assumed given:** The method requires a predefined center for polar coordinate projection, but no analysis of sensitivity to center selection is provided. In real applications, choosing the center is non-trivial and could significantly affect partition quality.

- **Scalability claims are unsubstantiated:** The paper claims WRT "scales to graphs with different sizes effectively" based on transfer experiments from 100 to 200 nodes per ring (~600-1200 total nodes), but no runtime comparison or experiments on larger graphs are provided. Real road networks can have orders of magnitude more nodes.

- **Overclaimed scope in framing:** The title says "Solving Normalized Cut Problem with Constrained Action Space" and the abstract claims WRT "is the first method to explicitly constrain the shape of NC" and opens up "a principled approach for fine-grained shape-controlled generation." The method only handles rings and wedges, with no mechanism for other shapes, and the problem framing implicitly assumes radial graph structure. The claims about "fine-grained shape control" and general applicability to "weighted planar graphs" are overstated relative to what is delivered.

## Nice-to-Haves

- Evaluation on non-radially-structured planar graphs to honestly delineate the method's scope
- A constrained baseline (e.g., DP on the transformed 1D sequence) to isolate the contribution of RL/Transformer from the contribution of the structural prior
- Extension of Proposition 1 to weighted graphs, or at least a discussion of how the bounds relate to the empirical results
- Error bars across multiple training seeds and multiple test-time sampling runs
- Runtime comparison with classical methods

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"No evaluation on multiple real-world datasets"** (from Spark): While evaluating on more cities would strengthen the paper, the single-city evaluation is a valid start for a focused application paper. This is moved to nice-to-have.

- **"NeuroCUT is initialized by METIS and fails—likely misconfigured"** (from Harsh Critic): Figure 1 caption states this as an observed result, not a criticism of the implementation. The paper provides the explanation that NeuroCUT's initialization from METIS causes it to get stuck, which is an inherent limitation of that method for constrained problems. This is not a paper flaw.

- **"Standard deviations missing"** (from Human Finder): While important for RL papers, this is noted under Minor weaknesses above. It does not invalidate the results on its own given the 100-graph test set.

- **"The method cannot handle non-planar graphs"** (from multiple reviewers): The paper explicitly scopes to planar graphs and acknowledges this in the conclusion. Criticizing scope creep beyond this is inappropriate.

- **"Not first to constrain partition shape"** (from Spark): This claim is reasonable in context; no prior method constrains to rings/wedges specifically.

- **"NC definition uses max instead of sum"** (from Spark): This is a valid design choice, not a flaw. The paper defines it clearly in Eq. 1-2.

## Novel Insights

The paper's most interesting insight is that converting graph partitioning into a 1D sequential decision problem via polar-coordinate projections enables Transformer-based RL to directly produce structurally constrained partitions without needing initialization from classical methods. This is distinct from prior GNN+RL approaches that refine existing partitions. However, the insight is tightly coupled to the ring/wedge geometric assumption, and the empirical evaluation does not separate how much of WRT's advantage comes from this structural prior versus from the RL optimization itself.

## Suggestions

- Add at least one experiment on planar graphs without radial structure (e.g., grids, Delaunay triangulations) to delineate the method's applicability boundaries honestly.
- Include a simple constrained baseline—e.g., dynamic programming on the 1D transformed sequence—to demonstrate that the RL/Transformer component adds value beyond the structural reduction.
- Narrow the title and claims to reflect that this is about ring-and-wedge constrained NC on radially-structured planar graphs, not "Solving Normalized Cut Problem" in general.

## Score and Decision

**Calibration**: Compared to DDRL (RL for constrained CO, scores 3-5, withdrawn/reject), this paper shares concerns about limited applicability and baseline fairness, but has a more coherent application motivation and a cleaner structural insight. Compared to FneYHZU19U (constrained graph clustering with Cheeger-type inequality, scores 3-6, reject), this paper has a similarly narrow theoretical contribution but stronger application grounding. Compared to MetroGNN (RL+GNN for traffic, scores 3-6, reject), this paper has similar RL methodology concerns (underspecified MDP, no error bars) but offers a more novel structural insight. Compared to Coreset Spectral Clustering (normalized cut, scores 3-10, accept poster), this paper has far weaker theory and narrower empirical scope.

The core idea of constraining partitions to ring/wedge shapes and using polar-coordinate transforms to enable Transformer-based RL is novel and well-motivated for traffic simulation. However, the evaluation is substantially biased toward the method's inductive bias, the comparative setup favors WRT by design, the theoretical contribution is disconnected from practice, and the claims are overstated relative to what is demonstrated. These are not minor issues—they undermine the paper's central empirical claim that WRT is a strong NC optimizer. The paper would need either (a) honest scoping to its applicable domain plus fair constrained baselines, or (b) evidence of broader applicability, to be convincing.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>