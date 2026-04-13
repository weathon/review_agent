# ICLR Benchmark Results

Date: 2026-04-12 22:21
Critic/Merger: claude:claude-sonnet-4-6 (OpenRouter)
Neutral: qwen/qwen3.5-plus-02-15, Related Work: qwen/qwen3.5-flash-02-23:online (OpenRouter)

## 0GC81gpjOo

- GT: Reject (avg 4.5)
- Predicted: Reject (4.8/10)
- Match: YES

### Final Review

## Summary
This paper investigates the relationship between Theory of Mind (ToM) capabilities and cooperative trends in LLM-based multi-agent systems, finding that higher-level ToM agents (k=2) do not always exhibit better cooperation than lower-level ToM agents (k=1). The authors propose a stable coalition matching mechanism that forms teams based on belief-action alignment and specialized abilities, demonstrating improved cooperation and task performance across programming, debate, and reasoning tasks.

## Strengths
- **Counter-intuitive empirical finding:** The observation that lower ToM agents can exhibit better cooperative trends than higher ToM agents (Table 1) is genuinely interesting and challenges the assumption that more cognitive sophistication always improves coordination. This finding is consistent across multiple models (GPT-3.5-turbo, GLM-4, Llama-3-70b, Gemini, Claude) and tasks (HUMANEval, MBPP).

- **Creative integration of stable matching with cognitive modeling:** The paper bridges game-theoretic coalition formation with ToM-based belief alignment in a novel way. Defining coalition preferences based on belief-action alignment scores (Equation 2) provides a principled mechanism for team formation that goes beyond simple performance-based selection.

- **Multi-task, multi-model empirical coverage:** Experiments span iterative programming (HUMANEval, MBPP), debate, logical reasoning (AQUA-RAT), and general reasoning (MMLU), with five different LLM backbones. Results show consistent improvements with the proposed matching mechanism over baselines (MetaGPT, ChatEval, DyLAN).

## Weaknesses
- **Self-evaluation circularity in the core metric:** The belief-action alignment score φ is computed by prompting the LLM to evaluate its own predictions (Section 4.2, footnote 1). This creates a circular dependency where agents judge the quality of their own beliefs without external validation. LLMs may systematically bias these scores toward optimism or may hallucinate alignment. A simple cross-validation using a separate evaluator model or ground-truth annotation would substantially strengthen confidence in the FTM metric.

- **No statistical significance testing:** The paper reports single-run results without error bars, confidence intervals, or p-values. Given the stochastic nature of LLM outputs and the relatively small effect sizes in some cases (e.g., GLM-4 HUMANEval R=1: 65.5 vs 65.2), this omission undermines confidence in whether observed differences are meaningful rather than noise.

- **Specialized ability adaptation is unevaluated:** Section 5.2 introduces a modified preference score incorporating specialized ability α_j with weight λ=1, but no experiments isolate this contribution. The paper claims this as part of the contribution (contribution 2 in Section 1), yet readers cannot assess whether this adaptation provides any benefit over belief alignment alone.

- **Algorithm 1 has ambiguous re-matching logic:** The re-matching trigger (Lines 12-14) is confusing: initialization sets `rematching_required = -1`, incrementing goes from -1→0, yet Line 6 requires `rematching_required = 1`. This logical gap makes the algorithm difficult to implement correctly. Additionally, Line 12's condition "for j ∈ μ(i)" is ambiguous about whether mismatch triggers re-matching for any single partner or requires all partners to exceed tolerance.

- **Missing baseline comparisons:** Table 3 compares against MetaGPT but not against simpler alternatives like random team selection or top-k selection by historical performance. Without these controls, it's unclear whether gains come from the stable matching mechanism specifically or simply from any team reconfiguration strategy.

## Nice-to-Haves
- Ablation study isolating the matching mechanism from ToM augmentation: comparing "Random matching + k=2 ToM" against the proposed mechanism would clarify whether the gains come from matching sophistication or cognitive capability.

- Verification that k=1 and k=2 prompting actually induces different ToM reasoning depths (e.g., using established ToM benchmarks like ToMi or Sally-Anne tasks).

- Analysis of computational overhead: k-level ToM reasoning requires recursive prompt calls; token costs and latency measurements would help assess practical viability.

## Removed Points
These points are flagged to be removed, treat them with caution:
- *Claim that Table 4 is missing or misnumbered*: The paper correctly references Table 4 in Section 6.4 for debate win rates; this is a reading error.
- *Criticism that psychological grounding is "thin"*: The Ridinger & McBride (2017) citation provides reasonable motivation; demanding stronger justification from a primarily computational paper is scope creep.
- *Demand for k=0 or k≥3 comparisons*: The paper reasonably focuses on k=1 and k=2, which aligns with literature on practical ToM depth (Section 3). Requesting additional levels is a nice-to-have, not a flaw.
- *Formatting nitpicks about notation inconsistencies*: Minor symbol variations (hat vs tilde) do not impede understanding.
- *Criticisms about the coalition size condition being "strange"*: While notation could be clearer, the intended meaning (minimum coalition size) is adequately explained for implementation.
- *Demand for comparison to cooperative game theory solution concepts*: The paper defines its own stability condition; requiring connection to the core or Nash stability is reasonable for theoretical depth but not essential for empirical contribution.

## Novel Insights
The finding that higher ToM can impair cooperation—attributed to "overthinking" and conflict anticipation—aligns with psychological literature on human cooperation (Ridinger & McBride, 2017) and extends it to LLM agents. The mechanism by which stable matching reverses this effect is insightful: matching provides a structured environment where higher-order belief reasoning becomes advantageous rather than anxiety-inducing. This suggests that cognitive sophistication in artificial agents, as in humans, requires appropriate institutional scaffolding to translate into cooperative behavior.

## Suggestions
- Replace self-evaluated alignment scores with cross-model evaluation or a separate judge LLM, and report agreement rates between self- and external evaluation.
- Run each experimental condition with at least 5 random seeds and report mean ± standard deviation; perform statistical significance tests for key comparisons.
- Provide a focused ablation in the main paper showing: (a) random matching vs. proposed matching, (b) matching without the specialized ability term, (c) matching with k=1 vs k=2 agents separately.
- Clarify Algorithm 1's re-matching trigger logic with corrected pseudocode and explicit threshold handling for individual vs. coalition-wide misalignment.
- Add computational cost analysis: report average token usage and latency per cooperation round for k=1 vs k=2 agents across task types.

---

## ERv8ptegFi

- GT: Accept (Poster) (avg 6.2)
- Predicted: Accept (6.3/10)
- Match: YES

### Final Review

## Summary

GPUDrive is a GPU-accelerated, multi-agent driving simulator built on the Madrona Game Engine that achieves over one million simulation steps per second by parallelizing hundreds of environments with hundreds of agents each. The simulator integrates real-world data from the Waymo Open Motion Dataset, supports multiple sensor modalities (including LiDAR and human-like vision cones), provides Gymnasium-compatible Python interfaces, and includes pre-trained reinforcement learning agents that achieve 95% goal-reaching rates on 1000 training scenarios.

## Strengths

- **Exceptional throughput with strong empirical validation.** The paper demonstrates peak throughput of 2.3 million Agent Steps Per Second (ASPS) and introduces the Controlled ASPS metric to account for variable agent counts. Figure 3 shows credible scaling curves comparing against Nocturne (CPU-based) and includes comparisons across consumer (RTX 4080) and datacenter (A100) GPUs. The 200-300× speedup over CPU-based Nocturne is well-supported by the wall-clock training time comparisons in Figure 5.

- **Thoughtful systems engineering for real-world data at scale.** The Bounding Volume Hierarchy for collision detection, Visvalingam-Whyatt polyline decimation achieving 10-15× reduction in road points, and memory allocation proportional to actual rather than maximum agent counts (Section 3.1) demonstrate careful optimization for the unique challenges of driving simulation.

- **Transparent acknowledgment of limitations.** Section 3.2 explicitly documents simulator sharp edges: absence of a lane graph, convex-only collision objects, ~2% unsolvable goals due to dataset labeling errors, and initialization modes that filter out stationary agents. This candor is commendable and helps users understand appropriate use cases.

- **Strong reproducibility and accessibility.** The paper provides Dockerfiles, pre-trained agents, Gymnasium environments for both PyTorch and JAX, and training loops via PufferLib. The claim that experiments can be reproduced on a single A100 in 16 hours is specific and testable.

## Weaknesses

- **No evaluation on held-out test scenarios.** All reported results (95% goal-reaching, Figure 5 training curves, Figure 6 amortized costs) are on training scenarios. The Waymo Open Motion Dataset has official train/test splits, but the paper does not report performance on held-out data. For a simulator intended to enable research on learned driving policies, this leaves generalization entirely unvalidated. The 98% ceiling attributed to mislabeled road edges is based on informal analysis, not empirical measurement.

- **Collision rates and safety metrics are not reported.** Section 3.2 explicitly states "collision penalties... are not used in the experiments reported in this work," and no collision rates are provided. For a simulator explicitly positioned for autonomous driving and safety-critical settings, this omission makes it impossible to assess whether the trained agents are learning safe behaviors or simply reaching goals through aggressive driving. The qualitative description of agents as "extremely aggressive about reaching their goals" (Section 3.2) underscores this concern.

- **The Waymax comparison lacks hardware specification and may be unfair.** Section 4.1 states "we could not run more than 16 environments in parallel due to Out of Memory (OOM) issues" for Waymax, which becomes a key pillar of the scalability comparison. However, the paper does not specify the GPU memory configuration used for Waymax, whether the Waymax authors' recommended settings were followed, or whether memory optimization was attempted. Without this information, readers cannot determine whether the OOM reflects a fundamental limitation of Waymax or a configuration issue.

- **The headline "1 Million FPS" claim relies on ASPS rather than CASPS.** While the paper clearly defines both metrics, the abstract and title emphasize the million-step figure, which counts all agents including parked cars. The more practically relevant metric (CASPS) peaks at ~200,000 for typical WOMD scenarios. This is still impressive, but the 5× gap between headline and learning-relevant throughput should be explicit in the abstract.

- **Multi-agent coordination capabilities are claimed but not demonstrated.** The introduction frames GPUDrive as enabling "multi-agent learning" and "self-play" research, but all experiments use Independent PPO with sparse goal-reaching rewards. There is no experiment testing emergent coordination, negotiation, or game-theoretic properties. The paper would benefit from at least one experiment that demonstrates multi-agent interaction benefits over single-agent baselines or scripted traffic.

## Nice-to-Haves

- **Lane compliance or rule-following metrics.** Although the simulator lacks a formal lane graph, proxy metrics for driving quality (e.g., distance from road centerline) would contextualize the 95% goal-reaching rate against actual driving behavior.

- **Ablation of observation types on learning performance.** Figure 4 shows LiDAR is faster than radial filter, but the paper does not analyze whether learned policy quality differs between observation modalities—critical for researchers choosing configurations.

- **Analysis of failure modes.** Qualitative examination of the 5% of scenarios where agents fail (collision vs. stuck vs. unreachable) would provide actionable insights for algorithm development.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **CARLA multi-agent table entry.** The claim that CARLA should have a multi-agent checkmark is a minor table formatting dispute. CARLA does support multi-agent scenarios via TrafficManager, but the distinction between full multi-agent RL support and scenario scripting is nuanced. This is not a substantive issue.

- **Equation units complaint.** The criticism that Equation (1) has inconsistent units misunderstands the simplified bicycle model formulation—the term in parentheses gives a distance, and the steering command s has units of curvature (rad/m), making the units consistent. This is standard in vehicle dynamics literature.

- **LiDAR vs. radial filter implementation asymmetry as a flaw.** The observation that LiDAR is faster because it's GPU-accelerated while radial filter uses linear scan is accurate but reflects design tradeoffs, not a paper defect. Both observation types are available to users.

- **"Simulator" baseline in Figure 5 label.** The figure could be clearer about what "Simulator" refers to, but this is a minor clarity issue, not a fundamental methodological concern.

## Novel Insights

The amortized per-scene training cost decreasing as the scenario dataset grows (Figure 6) is a genuinely interesting finding: at 1024 scenarios, solving an additional scene costs ~15 seconds compared to minutes per scene when training on fewer scenarios. This suggests strong positive transfer across scenarios and implies that larger training sets may be more efficient overall—counter to the intuition that more scenarios require proportionally more computation. This finding has implications for how researchers should structure large-scale RL training for driving.

## Suggestions

- Add a test-set evaluation using WOMD's official validation split to demonstrate that learned policies generalize beyond training scenarios.
- Report collision rates alongside goal-reaching rates, even for agents trained without collision penalties—this is essential for assessing whether GPUDrive-trained agents are suitable for downstream safety research.
- Specify the GPU memory and configuration used for all baselines (especially Waymax) and, if possible, attempt memory optimization or report the maximum batch size achieved before OOM.

---

## 2c7pfOqu9k

- GT: Accept (Spotlight) (avg 7.5)
- Predicted: Accept (6.2/10)
- Match: YES

### Final Review

## Summary
DEFT proposes a hardware-efficient attention algorithm for tree-structured LLM inference (e.g., speculative decoding, tree-of-thoughts). The key contributions are KV-Guided Grouping—loading shared prefix KV cache once for all queries that need it—and Flattened Tree KV Splitting—ensuring load-balanced partitions across GPU streaming multiprocessors. The method is implemented in Triton and evaluated on A100 GPUs across few-shot prompting, multi-step reasoning, and speculative decoding tasks, achieving up to 2.23× decoding and 3.59× attention speedup over baselines.

## Strengths
- **Clear problem formulation:** The paper correctly identifies that existing tree-attention implementations optimize computation and storage but overlook memory access (IO) patterns for shared prefixes—a genuine bottleneck in memory-bound LLM inference. The framing into C1 (prefix-awareness) and C2 (load balancing) is precise.
- **Strong empirical results in the speculative decoding regime:** Table 5 shows consistent speedups of 1.29–2.23× decoding latency over Radix Attention for speculative decoding with token trees of size 32–256. The attention speedup (3.59× at t=256) directly addresses the stated IO bottleneck.
- **Thorough ablation on design choices:** Table 6 compares DEFT-Node, DEFT-Node-Chunk, and DEFT-Flatten, demonstrating that balanced partitioning is essential—DEFT-Node without flattening is actually slower than the baseline in some settings, validating the combined approach.
- **Practical implementation:** The open-source Triton implementation and support for both paged and unpaged memory management (Table 3) facilitate reproducibility and potential integration with existing systems.

## Weaknesses
- **Modest gains for multi-step reasoning:** The decoding speedup for multi-step reasoning tasks ranges from 1.03× to 1.10× (Table 5), which is marginal. The paper claims DEFT is "versatile for various tree-structured tasks," but the evidence for multi-step reasoning is weak. This should be acknowledged more prominently.
- **Theoretical analysis deferred to appendix:** The IO complexity analysis—the primary theoretical justification for DEFT's superiority—is entirely in Appendix A.5. For a paper whose core contribution is IO reduction, the asymptotic IO expressions should appear in the main text.
- **QKV Preparation Phase overhead unquantified:** The paper describes a two-phase approach (preparation + calculation) but never measures the latency of Phase 1 (metadata processing, grouping, bitmask generation). For small trees, this planning overhead could negate attention savings.
- **Workload reconstruction limits ecological validity:** Table 4 indicates multi-step reasoning trees are "reconstructed from interaction records with GPT-3.5" and speculative decoding trees are recorded from Medusa runs, then replayed. This ensures controlled comparison but may not reflect actual Llama3-8B token distributions or dynamic branching behavior.
- **No statistical significance reported:** All latency numbers are point estimates without error bars, confidence intervals, or run counts. GPU kernel execution has inherent variability that should be quantified.
- **Bit Causal Mask (BCM) cost claimed but not measured:** Remark 3.1 states BCM overhead is "negligible" compared to dense causal masks, but provides no empirical measurement. For large or highly branched trees (e.g., t=256), BCM generation cost could be non-trivial.
- **DEFT-Node alone is counterproductive:** Table 6 shows DEFT-Node is slower than Radix Attention for few-shot (10.59s vs 5.99s) and multi-step reasoning tasks. This reveals that KV-Guided Grouping without load balancing actually hurts performance—a finding that deserves clearer emphasis in the main narrative.

## Nice-to-Haves
- **Comparison with concurrent works:** The paper discusses concurrent IO-optimized methods (Ye et al., 2024; Juravsky et al., 2024) in Section 2 but does not benchmark against them. Empirical comparison would strengthen SOTA claims.
- **Evaluation on deeper trees:** Most experiments use 2-level trees (few-shot) or moderate-depth trees from reconstructed traces. Testing on deep search trees (e.g., beam search depth 20+) would validate the "arbitrary tree depth" claim.
- **Multi-tree batching:** All experiments process a single tree per forward pass. How DEFT composes with batched multi-tree inference is unexplored.
- **Accuracy results in main text:** Table 15 (Appendix) validates inference accuracy but receives no discussion in the main paper. For an algorithmic modification to attention, confirming numerical equivalence is important.

## Removed Points
*These points are flagged to be removed, treat them with caution*

- **"Table 1 doesn't demonstrate prefix KV sharing":** While Table 1 shows equal IO-KV (12.40 TB) for Tree Attention and DEFT, this is one teaser example. The main results in Tables 16-17 (Appendix) do show KV IO reduction. The paper's claim is supported elsewhere.
- **"Abstract overclaims by not caveating that largest speedups are in speculative decoding":** The abstract states "up to" speedups and lists three workloads, which is accurate representation. The magnitude variation across tasks is visible in the results.
- **"Missing experiments on H100 or multi-GPU":** This is scope creep—the paper targets single-GPU A100 optimization. Architectural generalization is future work, not a core flaw.
- **"Full system end-to-end latency with tree search scheduler":** The paper explicitly excludes framework overheads (10-15%) to focus on attention optimization, stating these are consistent across baselines.
- **"Requests for latest vLLM/SGLang versions":** The paper compares against Radix Attention (SGLang's attention) which is the relevant baseline for tree-structured inference.

## Novel Insights
The key insight from combining these reviews is that DEFT's contribution is not simply "KV-Guided Grouping" but specifically the *combination* of grouping with flattened, load-balanced splitting. Table 6 reveals this clearly: DEFT-Node alone underperforms the baseline in narrow-tree settings, while DEFT-Flatten succeeds. This matters because it shows that prefix-aware IO optimization is insufficient without addressing GPU utilization—the two problems (C1 and C2 in the paper) are coupled, and solving one without the other can backfire. Additionally, the regime sensitivity is underappreciated: DEFT shines when the KV cache is large relative to queries (speculative decoding, long prompts), but offers limited gains when attention is a smaller fraction of decoding latency (narrow multi-step reasoning trees). Users considering DEFT should assess whether their workload falls in the high-KV, high-parallelism regime.

## Suggestions
- Add error bars or confidence intervals to latency measurements across multiple runs.
- Move a condensed version of IO complexity analysis (at minimum, asymptotic expressions) from Appendix A.5 to Section 3.3.
- Report QKV Preparation Phase latency separately in Figure 4's breakdown to quantify the planning overhead.
- Measure and report BCM generation time for the largest tree sizes (t=256) to validate the "negligible" claim empirically.
- Acknowledge the limited gains for multi-step reasoning prominently in the conclusion, characterizing which workloads benefit most.

---

## 0VP3LuzZ8K

- GT: Reject (avg 6.2)
- Predicted: Accept (7.1/10)
- Match: NO

### Final Review

## Summary
This paper establishes time-independent information-theoretic generalization bounds for Stochastic Gradient Langevin Dynamics (SGLD) in non-convex settings. The authors prove that under dissipativity assumptions, KL and Rényi divergences between SGLD iterates on different datasets remain bounded uniformly in time, resolving an open question from Vempala & Wibisono (2019) on uniform Log-Sobolev inequalities for discrete Langevin iterates. A secondary result relaxes dissipativity using ergodicity arguments, yielding polynomial dimension dependence at the cost of introducing ergodicity-related error terms.

## Strengths
- **Resolution of a stated open problem:** Theorem 12 establishes that all SGLD iterates satisfy a uniform Log-Sobolev Inequality under dissipativity, extending prior results that required strong convexity. This directly addresses the open question from Vempala & Wibisono (2019) about whether such uniform bounds exist beyond convex settings.
- **Novel analysis template:** The expansion-contraction decomposition (splitting Gaussian noise into two halves, analyzing gradient and noise steps separately) provides a clean modular framework that unifies the analysis of generalization and differential privacy through Rényi stability.
- **Relaxation of strong convexity:** The work successfully moves beyond strong convexity—known to be unrealistic for neural networks—to dissipativity and isoperimetric assumptions, while maintaining finite bounds as iteration count increases. The two-track structure (dissipativity yielding Rényi bounds; ergodicity without dissipativity yielding KL bounds) provides flexibility in assumptions.

## Weaknesses
- **Unusual stepsize constraint:** Theorem 12 requires $\frac{31}{32m} < \eta \leq \frac{m}{2L^2}$. The lower bound on $\eta$ is highly nonstandard—typical SGLD analyses require only an upper bound. For this interval to be nonempty, we need $L \lesssim m$, meaning the smoothness constant cannot be much larger than the dissipativity constant. This effectively restricts the theory to well-conditioned losses. The authors note that "constant factors are loose" but do not discuss whether this qualitative restriction ($L \approx m$) is fundamental to their approach or an artifact of the proof technique.
- **Implicit $n$-dependence in final bounds:** The corollaries (14.1, 15.1, 20.1) bound $D_{KL}(X_k|X'_k)$ but do not explicitly display the scaling with dataset size $n$. For generalization guarantees via Lemma 2, readers need to see how the numerator scales with $n$ (typically $O(1/n)$ for adjacent datasets via gradient sensitivity terms). This should be made explicit.
- **Key terms relegated to appendix:** Theorem 18, one of two main technical results, is stated with placeholder terms "erg(...)" and "ProbConst" that are only defined in equations 8-9 in the appendix. Given the importance of this result for the non-dissipative setting, the main text should provide at least qualitative descriptions of these terms (e.g., whether they involve KL divergences to stationary distributions, their scaling behavior).
- **Initialization dependence in Corollary 20.1:** The bound includes $D_{KL}(X_0|\pi)$ terms, where $\pi \propto e^{-\beta F_n}$ is the Gibbs distribution. Since $X_0$ is typically random initialization far from $\pi$, this term could be large or even infinite. The paper specifies initialization in Section 5 but the treatment in Section 6 is less clear, creating a practical gap in the second main result.
- **Exponential dimension dependence under dissipativity:** The LSI constant bound in Theorem 12 scales as $\exp(O(d))$, which the authors acknowledge. While this matches the LSI constant of the target distribution $e^{-\beta F_n}$ and may be unavoidable, it severely limits practical applicability to over-parameterized models. The contrast with Section 6's polynomial dependence could be highlighted more clearly.

## Nice-to-Haves
- A brief discussion of how plausible uniform dissipativity (Assumption 13) is for neural network losses would help readers assess practical relevance. Weight decay can encourage dissipativity-like properties, but the assumption requires dissipativity for every individual data point's loss function.
- A comparison table summarizing the assumptions, dimension dependence, and time-dependence of this work versus prior results (Farghly & Rebeschini 2021; Futami & Fujisawa 2024; Zhu et al. 2024) would aid comparison, since Appendix A is mentioned but not visible in the main submission.

## Removed Points
These points are flagged to be removed, treat them with caution.
- **Requests for empirical validation:** The spark finder requested experiments on CIFAR-10/ImageNet, empirical verification of dissipativity assumptions, and plots comparing bound tightness. This is a theoretical paper making novel mathematical contributions; empirical validation is not a standard requirement and is outside the stated scope.
- **Extension to standard SGD/Adam:** The spark finder requested extending analysis to standard SGD without explicit noise or to adaptive methods. The paper explicitly studies SGLD; asking for extensions to other algorithms is scope creep. The explicit noise assumption is clearly stated as part of the algorithm.
- **Generic formatting/style comments:** Any nitpicks about notation consistency fall into this category and have been removed.

## Novel Insights
The expansion-contraction analysis template reveals a fundamental asymmetry in how gradient steps and noise steps affect stability: gradient steps can only *expand* the divergence between parallel chains (Theorem 5), while noise steps *contract* it (Theorem 6). This decomposition cleanly separates the source of overfitting (gradient sensitivity to data) from the source of regularization (noise injection). The insight that uniform LSI can be established under dissipativity—even without strong convexity—hinges on the fact that dissipative gradient maps are "approximately contractive" (Lemma 11), keeping iterates in a region where sub-Gaussianity can be upgraded to LSI via Chen et al. (2021). The Gaussian convolution argument in Section 6 further shows that explicit dissipativity can be replaced by the regularizing effect of noise itself, as long as one accepts ergodicity-related error terms.

## Suggestions
- In the discussion following Theorem 12, explicitly discuss whether the stepsize lower bound ($\eta > 31/(32m)$) and the implied restriction $L \lesssim m$ are fundamental or improvable, and what classes of losses satisfy both conditions.
- Make the $n$-dependence explicit in Corollaries 14.1 and 15.1: when $\mathcal{D}$ and $\mathcal{D}'$ are adjacent datasets, $S_\infty = O(1/n^2)$ due to minibatch averaging, leading to generalization gap $O(1/\sqrt{n})$ via Lemma 2. State this scaling clearly.
- Add 2-3 sentences in the main text describing the form of the "ergodicity error term" and "ProbConst" terms in Theorem 18—readers should not need to consult the appendix for qualitative understanding of these key quantities.

---

## 5btqauRdz0

- GT: Reject (avg 5.5)
- Predicted: Accept (6.9/10)
- Match: NO

### Final Review

## Summary

This paper introduces STAGE, a method for zero-shot generalization of GNNs to graphs with entirely different node attribute domains. The key insight is to transform raw node features (which are domain-specific) into representations of statistical dependencies between features (which can transfer across domains), specifically by constructing "STAGE-edge-graphs" from empirical conditional probability matrices. The approach is grounded in the theory of maximal invariants from statistical testing, and experiments demonstrate substantial improvements over baselines in link prediction (40–103% relative improvement in Hits@1) and node classification (~10% improvement in accuracy) when transferring to unseen attribute domains.

## Strengths

- **Novel problem formulation with principled solution:** The paper identifies a genuine gap—existing GNN foundation models cannot handle graphs with entirely different attribute spaces—and proposes a conceptually elegant solution: rather than learning feature values, learn their statistical dependencies, which can have analogous structure across domains. This is a meaningful shift from prior approaches that either ignore features, use LLM textification, or assume shared feature semantics.

- **Strong theoretical grounding:** The connection to maximal invariants (Bell 1964; Berk & Bickel 1968) provides principled justification for why rank-based dependency representations enable domain transfer. Theorem 3.4 establishes that STAGE is provably invariant to the specified class of domain transformations (COGGs), giving formal grounding for the approach.

- **Substantial and consistent empirical gains:** The improvements over baselines are large (40–103% in link prediction, ~10% in node classification) and consistent across six test domains. The robustness to the extreme H&M domain shift (a completely different data provider with different products, customers, and features) is particularly compelling evidence for the method's transfer capability.

- **Handles mixed feature types naturally:** The conditional probability definitions (Equation 2) explicitly accommodate both continuous and categorical features, addressing a practical challenge that standard normalization or embedding approaches struggle with.

## Weaknesses

- **Theory applies only to fixed-dimensional feature spaces, but experiments use variable dimensions.** Section 3 explicitly restricts theoretical results to "domains with a fixed number of features to simplify the proofs." Yet the core empirical contribution involves transfer across domains with different feature dimensions (e.g., smartphones with RAM, display vs. clothes with size, color). The gap between what is proved and what is demonstrated is significant and should be addressed directly—the paper should clarify whether the GNN's ability to handle variable-sized inputs provides the necessary extension, or whether the theoretical guarantees simply do not apply to the main use case.

- **Theoretical guarantees rely on "most-expressive" GNNs not used in practice.** Theorems 3.2 and 3.3 condition on maximally expressive GNN encoders, but practical implementations use standard message-passing networks (NBFNet, GINE) which have known expressivity limitations under the Weisfeiler-Leman hierarchy. The drop of feature-ID labels to achieve COGG-invariance (Section 3.2) is explicitly acknowledged to "sacrifice maximal expressivity." A more honest framing would clarify that the theorems establish what is *representable* rather than what will be *learned* in practice.

- **Computational cost is under-analyzed in the main text.** For each edge in the original graph, STAGE constructs a STAGE-edge-graph with 2d nodes and O(d²) edges. For a graph with |E| edges, this creates |E| subgraphs. For moderate d and large |E|, this is non-trivial. Appendix F reportedly contains complexity analysis, but scalability claims for large graphs (e.g., Friendster with millions of edges) need prominent discussion in the main text, including any approximations or efficiency considerations.

- **No ablation isolating the contribution of conditional dependencies.** The core claim is that modeling *statistical dependencies* (off-diagonal elements of S^{uv}) enables transfer. Yet the paper does not compare against a simpler baseline that uses only marginal probabilities (diagonal of S^{uv})—essentially rank-normalizing each feature independently without capturing cross-feature dependencies. Such an ablation would isolate whether the method succeeds because dependencies transfer, or simply because rank-based normalization removes domain-specific value scales.

- **Node classification evaluation is thin.** The node classification experiments use only one train–test pair (Friendster→Pokec) and one task (gender prediction). The age regression task is mentioned but results are apparently uninformative. With only one domain pair, it is unclear whether the 10% improvement generalizes beyond this specific setting. Additional node classification benchmarks would substantially strengthen this portion.

- **Missing connection to copula theory.** The empirical conditional CDF representations (p(x_i^u | x_j^v)) are mathematically related to empirical copulas, a well-established framework for modeling dependence structures independently of marginal distributions. The paper's theoretical framing would be strengthened by acknowledging this connection and situating the work relative to copula-based methods in statistics and machine learning.

## Nice-to-Haves

- **Explicit probability estimation protocol:** The paper should clarify how empirical conditional probabilities are computed for continuous features—is this done via empirical CDF, binning, or kernel density? This detail affects reproducibility and behavior on sparse data.

- **More seeds for statistical confidence:** While 3 seeds is not unusual, the variance in some baselines (e.g., std=0.025 for GINE-gaussian in Table 2) suggests that additional seeds would provide more robust conclusions, particularly for the node classification results.

- **Discussion of failure modes:** The method assumes analogous statistical dependencies exist across domains. If the dependency structure fundamentally differs (e.g., income correlates with price in train but not in test), how does the method degrade? Some analysis of this failure mode would be valuable.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Claim that NBFNet-raw achieving 0.0000 is an implementation bug:** This is not a bug—it correctly reflects that raw feature dimensions mismatch between train and test domains, so a model trained on one feature space cannot process a different feature space. This is the fundamental problem STAGE is designed to solve.

- **Criticisms about the supervised H&M baseline being weak:** The structural-supervised baseline serves its intended purpose: showing that zero-shot STAGE can compete with a supervised model that only uses graph structure. A stronger supervised model using H&M's own features would be an upper bound comparison, but the current baseline adequately contextualizes STAGE's capability.

- **Concern about 3 seeds being insufficient for statistical power:** While more seeds would strengthen the paper, 3 seeds with the magnitude of improvements shown (e.g., 0.47 vs 0.23 MRR) and the consistent variance patterns provides reasonable evidence for the main claims. This is within ICLR norms.

- **Demand for user studies or confidence intervals on large benchmarks:** The reviewer requested more rigorous statistical testing, but the experiments follow standard practices for the benchmarks and task types used.

- **Formatting and style nitpicks:** These were correctly filtered as not substantive.

- **Bipartite graph handling as a flaw:** The approach of adding edges between nodes of the same type is a reasonable engineering solution for bipartite graphs and does not undermine the method's contribution.

## Novel Insights

The key insight from synthesizing the reviews is that STAGE's fundamental contribution can be understood through the lens of *representation alignment without semantic correspondence*. Unlike domain adaptation methods that assume shared label spaces or feature alignment techniques that require corresponding features, STAGE transfers by identifying that certain *statistical patterns* (correlations, dependencies) recur across domains even when the underlying features have nothing in common. This is conceptually similar to how meta-learning finds "learning algorithms" that transfer, but applied to dependency discovery. The empirical finding that performance *improves* with more training domains (Figure 4)—unique to STAGE among all methods—suggests it is genuinely learning transferrable dependency patterns rather than overfitting to specific feature statistics. This positions STAGE as learning a "dependency discovery algorithm" that can be applied to any domain, which is a distinct conceptual contribution beyond the specific architecture.

## Suggestions

- **Add an ablation using only marginal probabilities:** Include a "STAGE-marginal" variant that uses only diagonal elements of S^{uv} (marginal CDFs without conditional dependencies). This would directly test whether capturing dependencies drives the improvements versus simple rank-based normalization.

- **Prominently discuss computational complexity in main text:** Move key complexity analysis from Appendix F to the main paper, including runtime/memory measurements on the experimental datasets. Discuss any approximations for large graphs.

- **Extend theory section to address the fixed-d limitation:** Either extend the proofs to variable dimensions (if straightforward), or add a discussion of how the theoretical guarantees might degrade or hold under dimension mismatch. Clarify whether the GNN's ability to handle variable-sized inputs provides a practical bridge.

- **Add at least one more node classification benchmark:** A second train–test domain pair would substantially strengthen confidence in the node classification results, which currently rest on a single dataset pair.

---

## 52Idqv2FNY

- GT: Reject (avg 4.8)
- Predicted: Reject (4.0/10)
- Match: YES

### Final Review

## Summary
This paper investigates the relationship between NLP benchmark scores and human evaluations for chat language models. Using four Llama 2 Chat models (7B, 13B, 34B, 70B), the authors compute correlations between 160 NLP benchmark scores and 55 human evaluation categories from a custom taxonomy, finding generally high correlations except for safety and adversarial categories. They also explore predicting human evaluation scores from benchmark scores using overparameterized linear regression with leave-one-out cross-validation.

## Strengths
- **Comprehensive evaluation coverage:** The study evaluates models on 160 NLP benchmarks and constructs a detailed human evaluation taxonomy spanning 9 areas with nested categories. The human evaluation dataset is substantial: 11,291 single-turn and 2,081 multi-turn samples annotated by 2,104 unique annotators (Section 3).

- **Identification of specific benchmark-human misalignments:** The finding that Safety, Adversarial Dishonesty, and Adversarial Harmfulness categories are anti-correlated with most NLP benchmarks (Section 4, Figure 4) is significant. The paper correctly identifies a gap: "these adversarial and safety-focused categories are more easily transgressed by more capable LMs" or alternatively that "safety benchmarks simply are not especially good" — either interpretation has practical implications for practitioners relying on benchmarks.

- **Well-motivated research question:** The tension between expensive/noisy human evaluations and cheap/precise automated benchmarks is genuine. Understanding which benchmarks predict human preference for chat LMs has practical value for model development.

## Weaknesses
- **Severe statistical limitation (N=4):** All correlation and regression analyses rest on exactly 4 data points — the four Llama 2 model variants. Pearson correlations computed over N=4 have 95% confidence intervals spanning roughly ±0.95; a correlation of r=0.8 is not statistically distinguishable from r=-0.3 at conventional significance levels (p<0.05 requires |r|>0.95 for N=4). The paper presents 160×55 = 8,800 correlation coefficients in heatmaps and violin plots (Figures 3, 4, 5) with no confidence intervals, significance thresholds, or multiple testing corrections. Claims that "benchmarks are broadly highly correlated with human evaluations" cannot be supported at meaningful confidence levels with this sample size.

- **Limited model diversity undermines generalizability:** All four models belong to the Llama 2 family, differing primarily in parameter count. Since NLP benchmark performance and human preference both tend to improve monotonically with scale, observed correlations may simply reflect that both metrics track model size rather than any intrinsic relationship between benchmark validity and chat quality. The paper acknowledges using Llama 2 for "consistency" but does not address that this consistency trades off against the ability to draw general conclusions about benchmark-human relationships. Including even a few models from different families (e.g., Mistral, Gemma, older GPT variants) would substantially strengthen external validity.

- **Overparameterized regression with inadequate data for validation:** The prediction task fits ~150 benchmark features to predict human scores from N=4 models. Leave-one-out cross-validation means each training fold has only 3 samples. While the paper cites benign overfitting theory, this theory does not validate generalization in the extreme N<<p regime. The tight clustering around the identity line in Figure 7 is consistent with interpolation artifacts from minimum-norm solutions, not meaningful prediction. Critically, no baseline comparison is provided — would predicting from model scale alone perform equally well? Without this comparison, it is unclear whether benchmarks add any signal beyond the obvious correlation that larger models score higher on both benchmarks and human preference.

- **Missing inter-annotator agreement metrics:** The paper reports using at least 3 annotators per comparison and averaging scores, but provides no inter-annotator agreement metrics (e.g., Cohen's κ, Krippendorff's α). For a paper whose central analysis depends entirely on human evaluation scores, this omission makes it impossible to assess the reliability of the ground truth.

- **Human evaluations are relative, benchmarks are absolute:** All human evaluation scores measure pairwise preference over GPT-3.5-0301, while NLP benchmarks measure absolute task performance. This asymmetry means the study tests whether benchmarks predict relative preference against a specific baseline model, not whether benchmarks predict absolute chat quality — a narrower claim than the framing suggests. The paper does not discuss this interpretive constraint.

## Nice-to-Haves
- Baseline comparison showing whether model scale alone predicts human preference as well as the full benchmark suite
- Inter-annotator agreement metrics for the human evaluation data
- Bootstrap confidence intervals for correlation coefficients to quantify uncertainty
- Deeper analysis of the safety anti-correlation finding: is this because safer models appear less helpful (more refusals), or because safety benchmarks measure something orthogonal to human notions of safety?

## Removed Points
These points are flagged to be removed, treat them with caution:
- **"Large-scale study" criticism:** The paper accurately describes the human evaluation data collection as large-scale (11k+ prompts, 2k+ annotators). While the number of models is small (N=4), this is clearly disclosed in the methods section. Criticizing the "large-scale" framing is semantic rather than substantive.
- **Benchmark contamination speculation:** While contamination is a valid concern for machine learning evaluation broadly, the paper does not make claims about contamination, and raising this as a critique requires external evidence not provided by any reviewer.
- **Goodhart's Law implications:** This is a valid conceptual point about future work but not a weakness of the current paper's methodology or claims.

## Novel Insights
The anti-correlation between safety/adversarial benchmarks and human preference deserves attention. The paper proposes two interpretations: (1) more capable models are more easily "tripped up" on adversarial tasks, or (2) current safety benchmarks are fundamentally misaligned with human notions of safety. With N=4, distinguishing these hypotheses is impossible, but the result has practical implications: standard capability benchmarks may mislead practitioners about safety-critical properties. Practitioners optimizing for benchmark performance might inadvertently be selecting against human safety preferences, or conversely, safety benchmarks may need fundamental redesign.

## Suggestions
- Explicitly acknowledge that N=4 fundamentally limits statistical reliability and frame conclusions as preliminary observations requiring validation with more diverse model sets. Phrases like "benchmarks are broadly highly correlated" should be qualified.
- Report inter-annotator agreement to establish the reliability of human evaluation scores.
- If feasible, expand to include models from different families (architectures, training procedures) to test whether observed correlations generalize beyond the Llama 2 scaling trajectory.
- Include a scale-only baseline in the prediction analysis to isolate whether benchmarks provide signal beyond model size.

---

## Ij9ilPh36h

- GT: Accept (Poster) (avg 6.2)
- Predicted: Accept (6.3/10)
- Match: YES

### Final Review

## Summary

This paper introduces "hyperfitting"—fine-tuning pre-trained LLMs on very small datasets (2000 samples) until near-zero training loss—and demonstrates that this counter-intuitive procedure improves greedy decoding quality for open-ended text generation. Across multiple model sizes (TinyLlama 1.1B to Llama 3.1 70B) and modalities (text and ImageGPT), hyperfitted models produce less repetitive text and achieve higher human preference scores despite significantly worse validation perplexity, distinguishing this phenomenon from grokking and double descent.

## Strengths

- **Novel counter-intuitive finding with extensive empirical validation:** The discovery that severe overfitting improves generation quality challenges conventional early-stopping wisdom. The paper provides evidence across 5 model families, 3 text datasets, and image generation, with consistent improvements in TTR and human preference (Table 1, Table 4, Figure 6).

- **Rigorous memorization analysis:** The citation-blocking experiments and overlap analysis (Table 2, Figure 3) directly address the obvious concern that improved outputs come from memorization. The finding that performance persists even when training sequences are blocked strengthens the claim of generalizable improvement.

- **Substantial human evaluation:** Over 20,000 annotations comparing model outputs to original human-written text provide a robust quality signal, and the paper appropriately notes the breakdown of perplexity as a quality metric in this setting.

- **Data ordering experiments:** Section 6.1 demonstrates that identical data in different order yields ~30% different top-1 predictions, ruling out deterministic memorization and providing insight into the stochastic nature of the process.

- **Clear distinction from related phenomena:** Section 7.2 thoughtfully differentiates hyperfitting from grokking and double descent across five dimensions, acknowledging limitations honestly.

## Weaknesses

- **Main experiments use the worst-performing hyperfitting dataset:** Table 4 shows Fiction hyperfitting yields 40.73% average preference, while News yields 66.37%—a dramatic difference. Yet all main experiments in Section 4 use Fiction. This choice is never explained, and presenting results from the poorest configuration undermines confidence in the representative findings.

- **No mechanistic explanation for the core phenomenon:** The "top-rank encouragement" hypothesis in Section 7.3 restates the observation (low training loss correlates with desirable top-rank tokens on OOD data) without providing causal insight. The paper acknowledges entropy drops (Table 3) but does not analyze *which* representations change, *where* in the network, or *why* sharpened distributions generalize. The hypothesis remains speculative.

- **Missing critical baseline:** No comparison to standard fine-tuning with early stopping on the same data. Without this, it is unclear whether the improvement comes from fine-tuning generally or specifically from near-zero training loss—undermining the core claim that overfitting itself is beneficial.

- **Human evaluation lacks inter-annotator agreement:** With 3 annotators per comparison and a 3-way choice (A preferred, B preferred, equal), reporting Cohen's kappa or similar is essential for interpreting preference percentages. The paper provides no agreement statistics despite acknowledging the subjective nature of the task.

- **Image generation results are purely qualitative:** Section 7.1 presents only visual inspection (Figure 6) with no quantitative metrics (FID, IS, classification accuracy). This significantly weakens the multimodality claim.

- **Overstated claim about parameter efficiency:** The introduction claims hyperfitted models "outperform models with 10x the number of parameters." Table 1 shows TinyLlama (1.1B) hyperfitted at 34.3% vs Llama 3.1 70B original at 34.4%—essentially tied, not outperforming, and the ratio is ~64×, not 10×.

- **No learning rate ablation:** The paper fixes lr=1e-6 without testing sensitivity. This is critical because larger LR might catastrophically forget pre-trained knowledge, while smaller LR might not achieve near-zero loss. The robustness of the phenomenon to this key hyperparameter is unknown.

## Nice-to-Haves

- Evaluate hyperfitted models on standard capability benchmarks (MMLU, GSM8K) to assess whether other abilities degrade during hyperfitting—a practical concern for deployment.

- Compare hyperfitted+Top-P sampling against Original+Top-P to establish whether hyperfitting complements or substitutes for sampling strategies.

- Test instruction-tuned models to assess real-world applicability, since most practical deployments use chat variants.

- Report statistical significance (confidence intervals, p-values) for human preference comparisons given modest sample sizes.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Criticizing the 32-token context length as "too short" for practical use* — the paper explicitly studies open-ended continuation from 32-token contexts, which is a valid experimental design choice for measuring generation quality, even if not covering all use cases.

- *Demanding safety/jailbreaking analysis* — while important for deployment, this is outside the paper's stated scope of discovering and characterizing the hyperfitting phenomenon.

- *Requiring comparison to DPO or contrastive decoding* — these methods serve different purposes and the paper's contribution is orthogonal; comparing to every alternative method is scope creep.

- *Criticizing lack of theoretical proof* — the paper makes empirical claims supported by experiments; demanding formal theory is not aligned with ICLR standards for empirical contributions.

- *Claims that the hyperfitting dataset choice invalidates the phenomenon* — while using Fiction for main experiments is suboptimal, Table 4 shows all hyperfitting datasets improve over baselines, so the phenomenon is real regardless of dataset choice.

## Novel Insights

The finding that near-zero training loss yields sharply peaked distributions (entropy dropping from ~3.5 to ~1.3 nats) which somehow generalize to produce *better* top-rank predictions on OOD data—despite worse perplexity—suggests a fundamental decoupling between next-token prediction loss and generative quality. The data ordering experiments (30% different top-1 predictions from shuffled data) reveal that the specific tokens emerging from hyperfitting are stochastic, suggesting the process collapses the model's uncertainty rather than memorizing specific continuations. This points toward a hypothesis: pre-trained models maintain a "corpus-average" multi-modal prediction space, and extreme training on small data prunes alternative modes while preserving semantic coherence learned during pre-training.

## Suggestions

- Reorganize Section 4 to present News hyperfitting results (the best-performing configuration) as the primary result, with Fiction and Wiki as ablations—this would present the phenomenon in its strongest light while preserving all empirical content.

- Add inter-annotator agreement statistics for human evaluation; without this, preference percentages are difficult to interpret.

- Include one early-stopped fine-tuning baseline to isolate the effect of near-zero loss specifically, establishing the novelty claim more rigorously.

- Report quantitative metrics for image generation (even if imperfect) to strengthen the multimodality claim beyond visual inspection.

---

