=== CALIBRATION EXAMPLE 32 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately signals the main idea: learning for dynamic combinatorial optimization without external training data. That said, “without training data” could be read as “no learning data at all,” whereas the paper actually does instance-specific optimization on each snapshot and uses prior snapshots as a warm-start signal.
- The abstract clearly states the problem setting, the proposed DyCO-GNN approach, and the benchmark tasks (dynamic MaxCut, MIS, TSP). It also gives a headline performance claim (“3–60x faster”).
- The abstract’s strongest claim is somewhat broader than what the paper substantiates. The “3–60x faster” speedup is not uniformly supported across the reported tables; it appears to refer to comparisons against certain static baselines under selected budgets, but the abstract does not qualify the dependence on dataset, task, or whether this is against converged static PI-GNN or wall-clock normalized comparisons.

### Introduction & Motivation
- The problem is well motivated: dynamic CO is practically important, and the paper makes a plausible case that re-solving each snapshot from scratch is wasteful when snapshots share structure.
- The gap in prior work is identified clearly enough: existing learning-based CO methods are mostly static, and PI-GNN-style methods are instance-specific but not dynamic. However, the paper’s claim that it is “the first to apply machine learning to DCO problems” is too broad. There is at least prior learning-based work on dynamic TSP cited in the references (Zhang et al., 2021), so the novelty claim should be narrowed to the specific PI-GNN-style instance-specific unsupervised setting.
- The contributions are stated clearly, but one needs more precision about what is actually novel: the method is essentially a warm-start heuristic plus SP perturbation adapted from Ash & Adams (2020), rather than a new learning paradigm from first principles. ICLR reviewers will likely expect a sharper articulation of why this adaptation is scientifically meaningful beyond “it works better than naive warm start.”

### Method / Approach
- The overall method is understandable: optimize PI-GNN on the first snapshot, then initialize subsequent snapshots by either direct warm start or shrink-and-perturb (SP) of the previous parameters.
- Reproducibility is partially undermined by several missing details:
  - The exact PI-GNN architecture/objective details are not fully specified for all tasks, especially for TSP, where the decoding differs by problem size.
  - The algorithmic description in Algorithms 1 and 2 is high-level and does not clearly define how many optimization steps are taken per snapshot beyond epoch budgets.
  - It is not fully clear how snapshot-to-snapshot optimization interacts with checkpoint selection, especially for TSP where “best checkpoint” uses total decoding time across checkpoints.
- The core assumption behind the method is that neighboring snapshots are similar enough that previous parameters are useful. That is plausible, but the paper does not quantify similarity or characterize when the assumption fails. This matters because the performance varies substantially across datasets and tasks.
- A major concern is the explanation for why naive warm start fails. The paper attributes this to “overconfidence” and local minima, but this is presented more as intuition than as an empirically validated mechanism. There is no diagnostic evidence (e.g., gradient norms, entropy, parameter distance, basin analysis) showing that this is the actual failure mode in PI-GNN.
- The theoretical support is weak relative to the method claim. Theorem 1 is about perturbing the SDP solution in the Goemans–Williamson MaxCut algorithm, not DyCO-GNN. This is at best an analogy, not theoretical justification for the learning method. The theorem also has limited relevance to the main empirical claims, since DyCO-GNN is a GNN-based unsupervised optimizer, not GW SDP rounding.
- The proof of Theorem 1 appears incomplete and somewhat problematic as written:
  - The claim “there exists a λ > 0 such that” the optimal-cut probability increases is asserted via a positive-measure argument, but the proof does not rigorously connect the perturbation distribution and projection to the required probability increase.
  - The final step claims \(P(R(X_0)=c^*)=0\) from \(X_0\notin C_{opt}\), which follows from the definition of \(C_{opt}\), but the crucial strict inequality for some perturbed λ is not fully established.
  - The corollary relies on “local continuity” of the SDP solver, which is a very strong assumption and not formalized enough to support the result.
- Edge cases/failure modes are not addressed. For example, if the previous snapshot is substantially different, SP may be harmful; the paper only shows favorable regimes.

### Experiments & Results
- The experiments do test the stated claims to some extent: dynamic MaxCut, MIS, and TSP under multiple time budgets, with ablations over SP placement (“emb,” “GNN,” “full”), plus sensitivity over SP hyperparameters.
- However, the evaluation is narrower than the paper’s opening implies. The paper claims broad applicability to DCO, but only three tasks are tested, and all are graph-structured. That is fine, but the generality claim should be modest.
- The baseline set is not fully adequate for ICLR-level claims:
  - The main comparisons are only against static PI-GNN and warm-started PI-GNN, i.e., a very restricted baseline family.
  - Although non-neural baselines are included in Appendix D.3, they are not central in the main comparison and are not systematically integrated into the core claims.
  - The exclusion of generalizable supervised/RL methods is defended, but it leaves open whether DyCO-GNN is competitive with state-of-the-art learning-based solvers in dynamic settings, especially since the paper claims practical effectiveness.
- The missing ablations are important:
  - There is no direct ablation isolating whether the gains come from SP specifically versus simply adding noise, rescaling, or using a different restart heuristic.
  - There is no comparison to resetting only the last layer or partially reusing embeddings beyond the reported layer-group variants.
  - There is no ablation on the similarity of snapshots, despite this being central to the method.
- Statistical reporting is insufficient for ICLR standards:
  - The paper says experiments were repeated five or ten times, but the main tables report only mean ApR with no standard deviations, confidence intervals, or significance tests.
  - Given that many results are close, especially on MIS and TSP, uncertainty matters a lot.
- Some results support the claims strongly, especially where DyCO-GNN beats both static and warm-start PI-GNN under moderate budgets. But the conclusions are less clean than the narrative suggests:
  - On dynamic MIS, warm-start PI-GNN is sometimes very strong and even better than DyCO-GNN at small budgets in some settings.
  - On TSP, the “full” variant can underperform the embedding-only variant at short budgets, indicating the method is sensitive to where SP is applied.
  - The paper sometimes highlights the best-performing checkpoint for TSP but final checkpoints for MaxCut/MIS, which makes comparison protocol non-uniform and potentially optimistic for TSP.
- Dataset choice is reasonable, but the dynamic construction is synthetic in an important sense:
  - For MaxCut/MIS, snapshots are generated by deterministic edge accumulation/deletion from timestamped datasets.
  - For TSP, a single node is moved along a line. This is a controlled benchmark, but not necessarily representative of richer DCO dynamics.
- The reported “speedup” claims should be interpreted carefully. In several tables, DyCO-GNN reaches a better ApR earlier than static PI-GNN, but the comparison uses different epoch budgets and sometimes different wall-clock times. The paper should more carefully define what counts as a speedup and against which baseline.

### Writing & Clarity
- The paper is generally readable and the overall narrative is coherent.
- The main clarity issue is that the method is described with intuition, but the empirical protocol is not always transparent enough to reproduce or interpret all claims:
  - The meaning of “final checkpoint” versus “best-performing checkpoint” is not consistently handled across tasks.
  - The exact wall-clock measurement protocol, especially for TSP decoding and checkpoint selection, needs more precision.
- Figures and tables convey the broad story, but several are hard to interpret from the text alone:
  - The snapshot-level plots are useful, but the main paper does not clearly summarize their quantitative takeaway.
  - The TSP results are more difficult to compare because of differing evaluation conventions and decoder choices.
- The main conceptual distinction between “instance-specific unsupervised optimization” and “training-free dynamic adaptation” should be explained more crisply, because this is the paper’s central contribution and could otherwise be misunderstood.

### Limitations & Broader Impact
- The paper acknowledges some limitations in Section 6, mainly that SP placement matters and could be made adaptive. That is a useful start, but it is not sufficient.
- Important limitations are missing:
  - The method is still snapshot-by-snapshot optimization and may not handle abrupt changes well.
  - It assumes access to a meaningful previous solution and temporal continuity across snapshots.
  - It does not address scalability to much larger or denser dynamic graphs beyond the tested instances.
  - The method is evaluated only on graph problems with QUBO formulations; broader DCO classes are not addressed.
- Broader societal impact is not discussed. Given the application domains are generic optimization problems, this is not a major omission, but the paper should still acknowledge that improved dynamic optimization can be used in surveillance, resource allocation, or other high-stakes settings.

### Overall Assessment
This paper addresses a real and interesting gap: how to adapt instance-specific unsupervised GNN optimization to dynamic combinatorial problems without offline training data. The empirical idea of combining warm starts with shrink-and-perturb is plausible and the reported gains over naive warm start are often convincing. However, for ICLR, the contribution is somewhat narrower than the framing suggests, the novelty claims are too broad, and the evidence is not yet strong enough to fully support the generality and theoretical narrative. The main weakness is that the paper compares mostly within a PI-GNN family, provides limited statistical analysis, and offers theory that is only loosely connected to the actual method. I would view this as a promising and practical incremental step, but not yet a fully compelling ICLR-level advance without stronger baselines, clearer experimental rigor, and a more directly relevant analysis of why DyCO-GNN works.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes DyCO-GNN, an unsupervised, instance-specific graph neural network approach for dynamic combinatorial optimization (DCO) that warm-starts from previous snapshots but adds shrink-and-perturb noise to improve escape from poor local minima. The paper evaluates the method on dynamic MaxCut, MIS, and TSP benchmarks and reports improved approximation ratios over static PI-GNN and naive warm-starting, often under tight runtime budgets.

### Strengths
1. **Addresses an important and underexplored problem setting.**  
   The paper targets dynamic combinatorial optimization, which is practically relevant when graph instances evolve over time. For ICLR, this is a meaningful direction because it combines learning with an online/adaptive optimization setting rather than only static benchmarks.

2. **Clear practical motivation for warm-starting and adaptation.**  
   The paper identifies a reasonable failure mode of naive warm-starting: it accelerates early progress but can converge to worse solutions than retraining from scratch at longer budgets. This is supported empirically in Tables 1–3, where warm-start often does well at small budgets but is overtaken by static PI-GNN at larger budgets.

3. **Simple method with low conceptual overhead.**  
   DyCO-GNN’s main mechanism—shrink-and-perturb initialization—is easy to understand and implement. The approach is also broadly applicable across the considered tasks because it is built on the PI-GNN framework and QUBO formulations.

4. **Empirical evaluation covers multiple problem families.**  
   The paper evaluates on three distinct CO tasks: MaxCut, MIS, and TSP, with several datasets and multiple time budgets. This breadth is valuable because it suggests the approach is not narrowly specialized to a single problem.

5. **Some ablation/sensitivity analysis is included.**  
   The paper includes variants applying shrink-and-perturb to different layers and a parameter sensitivity study in Appendix D.4, which helps justify the chosen hyperparameters and indicates the authors considered robustness.

### Weaknesses
1. **Novelty is limited; the core idea is a direct adaptation of known warm-start perturbation.**  
   The main technical contribution is to apply shrink-and-perturb, previously studied in supervised learning warm-starting, to PI-GNN-style instance-specific optimization. This is a reasonable adaptation, but the paper’s novelty appears incremental rather than fundamentally new, especially given the lack of a substantially new algorithmic principle or theoretical insight specific to DCO.

2. **The theoretical result is weakly connected to the actual method.**  
   Theorem 1 is about perturbing the GW SDP solution for MaxCut, not DyCO-GNN itself, and the assumptions are quite strong and somewhat detached from the learned GNN optimization setting. As written, it does not provide a convincing theory for why DyCO-GNN should work in practice on MaxCut, MIS, or TSP.

3. **Evaluation is narrow relative to ICLR expectations.**  
   The baselines are mostly confined to the PI-GNN family, with only limited comparison to non-neural heuristics in the appendix. For an ICLR paper, reviewers will likely expect stronger comparisons to dynamic optimization baselines, classical online/dynamic algorithms, and more modern learning-based alternatives where applicable.

4. **Experimental protocol raises some reproducibility and fairness questions.**  
   The paper uses different operators and different decoding strategies across tasks, and TSP especially uses best-performing checkpoints with decoding time included in a way that complicates comparisons. The need for GCNConv on some tasks and SAGEConv on TSP, plus the different snapshot handling across datasets, makes it harder to isolate what exactly drives gains.

5. **The dynamic problem setup is somewhat synthetic and limited.**  
   The dynamic MaxCut/MIS settings are created by converting timestamped graphs into edge-addition or edge-deletion snapshots, and the TSP setting moves one extra node along a straight line. These are plausible, but they do not fully demonstrate robustness to richer dynamic changes, concept drift, or adversarial/non-monotone dynamics.

6. **The claim “first to apply machine learning to DCO” is likely overstated.**  
   The paper cites prior learning-based work on dynamic TSP (e.g., Zhang et al., 2021), so the claim needs to be qualified. Overclaiming novelty is a concern at ICLR, where precise positioning relative to existing work matters.

7. **Clarity of the exposition is uneven.**  
   The core method is understandable, but the paper sometimes mixes conceptual claims with implementation details in a way that makes it hard to see what is essential. In particular, the role of the “full/emb/GNN” SP variants and the choice of checkpointing criteria for TSP could be explained more cleanly.

### Novelty & Significance
**Novelty: moderate to low.** The paper’s main idea is a sensible adaptation of shrink-and-perturb warm-starting to dynamic instance-specific GNN-based CO, but it does not introduce a fundamentally new learning paradigm or a deep theoretical advance. The significance is higher on the application side: dynamic combinatorial optimization is practically important and relatively underexplored in learning-based CO.

**Clarity: moderate.** The problem setting and motivation are clear, but the method description, theory, and evaluation protocol could be organized more cleanly. Some claims are stronger than the evidence warrants.

**Reproducibility: moderate.** The paper gives useful implementation details, datasets, and parameter values, which helps. However, because performance depends on task-specific decoding, checkpoint selection, time-budget accounting, and problem-specific preprocessing, reproducing the exact reported numbers would still require careful implementation alignment.

**Significance: moderate.** If validated more broadly, the idea of adapting instance-specific CO solvers to dynamic settings without training data is interesting. At present, though, the empirical gains are shown on a limited benchmark suite and the method seems more like a useful heuristic than a high-impact methodological advance.

Overall, under ICLR’s usual acceptance bar, this would likely be seen as a promising application-oriented paper with solid empirical evidence, but not yet a clearly strong acceptance unless the novelty and breadth of validation were improved.

### Suggestions for Improvement
1. **Strengthen the benchmark suite and baselines.**  
   Add comparisons against dynamic or online combinatorial optimization methods, classical reoptimization heuristics, and stronger non-neural baselines beyond the appendix. ICLR reviewers will want to know whether gains persist against the best available dynamic solvers, not just PI-GNN variants.

2. **Clarify what is genuinely new in the algorithm.**  
   Make explicit whether the contribution is primarily the dynamic adaptation framework, the shrink-and-perturb initialization, or the empirical observation that it works well for PI-GNN. If the method is mostly an adaptation, frame it as such and avoid overclaiming.

3. **Provide a more task-aligned theoretical argument.**  
   If theory is included, it should more directly analyze the actual warm-start-plus-perturb optimization used by DyCO-GNN, ideally in a simplified model closer to GNN training dynamics rather than only via GW/SDP analogies.

4. **Standardize and justify evaluation choices.**  
   Explain checkpoint selection, decoding, and timing uniformly across tasks. For TSP, report both final and best checkpoints for all methods or justify why the comparison is fair. Also clarify how wall-clock timing was measured and whether decoding is included consistently.

5. **Broaden dynamic scenarios.**  
   Evaluate on richer forms of temporal change: both node and edge churn, non-monotone graph evolution, abrupt changes, and varying degrees of temporal correlation. This would make the claims about “dynamic” optimization more compelling.

6. **Add stronger ablations on the SP mechanism.**  
   Explore perturbation magnitude, shrink factor, where to inject noise, and whether perturbing parameters vs. embeddings vs. optimizer state matters. A more detailed ablation would help determine why the method works.

7. **Tone down or refine the novelty claim.**  
   Replace “first to apply machine learning to DCO problems” with a more precise statement acknowledging prior learning-based dynamic TSP and related work, while emphasizing the specific PI-GNN-based unsupervised dynamic framework introduced here.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons to stronger dynamic optimization baselines, not just PI-GNN variants. The current evidence does not rule out that simple dynamic heuristics or reoptimization methods from the DCO literature outperform or match DyCO-GNN; this directly weakens the claim of being the first and best learning-based DCO method.

2. Add a true “recompute from scratch every snapshot” baseline using the same architecture and budget protocol. Without this, the speedup claims are ambiguous because it is unclear whether gains come from the proposed dynamic strategy or just from giving the model more effective optimization steps per snapshot.

3. Add ablations that isolate warm start, shrink-only, perturb-only, and different layer-wise SP placements on every task. The paper’s central claim is that SP fixes warm-start failure; without these ablations, it is not convincing that both shrink and noise are necessary or that the reported gains are not a fragile implementation detail.

4. Add more dynamic settings with larger and smaller temporal drift, plus at least one setting with edge removals/additions, node arrivals/departures, or more than one moving node for TSP. The current benchmarks are narrow and mostly monotone graph growth/deletion, so the claim that the method broadly solves DCO is under-supported.

5. Add scaling experiments on larger graphs and longer snapshot sequences. ICLR reviewers will expect evidence that the method remains effective when the number of snapshots, graph size, or update frequency increases; the current datasets are too small to justify broad “practical effectiveness” claims.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify when warm start fails and why SP helps, using optimization diagnostics across snapshots. The paper currently asserts “overconfidence” and “local minima” without showing gradient norms, loss landscapes, entropy/confidence, or parameter drift, so the mechanism remains speculative.

2. Report variance across random seeds and significance tests for all main results. Many reported gains are modest at later budgets, and without variance/error bars it is unclear whether the improvements are robust or just noise from stochastic optimization.

3. Analyze the runtime-quality tradeoff more carefully with wall-clock matched comparisons, not epoch-matched only. ICLR reviewers will question whether the claimed 3–60x speedups are meaningful if per-epoch costs differ by model variant, solver overhead, or decoding/postprocessing.

4. Clarify how the “ground truth” and approximation ratios are computed for dynamic TSP and whether the QUBO/decoder pipeline introduces bias. The current evaluation mixes relaxed optimization, greedy/beam decoding, and exact or bounded references, so the meaning of ApR is not fully trustworthy across tasks.

5. Check whether DyCO-GNN actually transfers anything across snapshots beyond initialization. A simple analysis of solution overlap, parameter change, and snapshot-to-snapshot improvement is needed to show the method exploits temporal structure rather than just acting like a restart heuristic.

### Visualizations & Case Studies
1. Show per-snapshot trajectories of objective value, gradient norm, and confidence for static PI-GNN, warm start, and DyCO-GNN. This would directly reveal whether SP helps escape premature convergence or merely adds noise without changing optimization behavior.

2. Include a failure-case visualization where warm start beats DyCO-GNN early but loses later, or vice versa. That would expose whether the method’s gains are stable across snapshots and time budgets, which is essential for trusting the dynamic claim.

3. Visualize how solution changes track graph changes, e.g., edge additions/deletions versus node assignment flips in MaxCut/MIS and route edits in TSP. This would test whether the model is adapting sensibly to temporal perturbations or producing essentially unrelated solutions each time.

4. Show ablation plots for SP parameters and layer choices on all tasks, not just tables. The paper’s central method choice is architectural and optimization-driven; visualizing sensitivity would reveal whether the proposed defaults are genuinely robust.

### Obvious Next Steps
1. Extend the method to benchmark suites designed specifically for dynamic optimization, with explicit update operations and stronger baselines. This is the most direct way to support the claim that DyCO-GNN is a general DCO method rather than a tailored demonstration on three datasets.

2. Add online/update-aware evaluation where the model must process each snapshot under a fixed per-update latency. The current setting is close to batched re-optimization; an online protocol is needed for the paper’s “rapidly evolving, resource-constrained settings” claim to be convincing.

3. Replace the ad hoc dynamic TSP construction with multiple dynamic route-change scenarios. A single moving node is too narrow to establish generality for dynamic routing.

4. Investigate adaptive or learned SP schedules instead of fixed shrink/perturb constants. Since the paper itself notes layer-wise and parameter sensitivity, a robust adaptive strategy is the natural next step and should be part of the core contribution if the method is to be practically useful.

# Final Consolidated Review
## Summary
This paper proposes DyCO-GNN, an unsupervised, instance-specific GNN-based framework for dynamic combinatorial optimization that reuses parameters across graph snapshots and applies shrink-and-perturb to improve adaptation. The paper evaluates the method on dynamic MaxCut, MIS, and TSP, and reports better approximation ratios than static PI-GNN and naive warm-starting under many time budgets.

## Strengths
- The paper tackles a genuinely relevant and underexplored setting: dynamic combinatorial optimization with no offline training data, which is practically important and a natural extension of the PI-GNN line of work.
- The core mechanism is simple and plausible. Warm-starting plus shrink-and-perturb is easy to implement, and the empirical results do show that naive warm start can become suboptimal at longer budgets while DyCO-GNN often recovers better solutions across snapshots.

## Weaknesses
- The novelty is modest and somewhat overstated. The method is largely an adaptation of existing warm-start perturbation ideas to PI-GNN-style instance-specific optimization, not a fundamentally new learning principle for DCO.
- The paper’s strongest theory is weakly connected to the actual method. The theorem is about perturbing the Goemans–Williamson SDP for MaxCut, not DyCO-GNN itself, so it functions more as an analogy than as real justification for the proposed GNN optimizer.
- The evaluation is too narrow for the paper’s broad claims. Most comparisons are only against static PI-GNN and warm-start PI-GNN, with stronger dynamic/reoptimization baselines largely relegated to the appendix. This leaves open whether DyCO-GNN is actually competitive beyond a restricted PI-GNN family.
- The experimental protocol is not fully clean or uniform. TSP uses best checkpoints while MaxCut/MIS use final checkpoints, and decoding/time-accounting details differ by task. That makes the reported speedup and quality comparisons harder to interpret fairly.
- The dynamic benchmarks are limited and somewhat synthetic. The graph snapshots are generated by monotone edge accumulation/deletion, and TSP is simulated by moving one node along a line. This is not enough to support broad claims about general dynamic optimization.
- The paper does not convincingly explain why warm start fails and SP helps. The “overconfidence/local minima” story is plausible, but it is not supported by diagnostic evidence such as gradient norms, entropy, or snapshot-to-snapshot optimization traces.

## Nice-to-Haves
- Report variance across seeds, confidence intervals, or significance tests for the main tables.
- Add an ablation separating shrink-only, perturb-only, and the full SP scheme on all tasks.
- Include a baseline that recomputes from scratch each snapshot with the same architecture and budget protocol, to make the speedup claim cleaner.
- Add richer dynamic settings with larger temporal drift, non-monotone changes, or multiple moving entities.

## Novel Insights
The most interesting aspect of the paper is not the shrink-and-perturb trick itself, but the observation that instance-specific unsupervised optimization behaves differently in a dynamic setting than in the static setting PI-GNN was designed for: a warm-start can be beneficial early yet harmful later, and a small amount of parameter destabilization can restore progress. That is a useful empirical insight for dynamic CO, but the paper currently oversells it as a major methodological advance rather than a practical adaptation of a known restart idea.

## Potentially Missed Related Work
- Zhang et al., 2021 — prior learning-based dynamic TSP work, relevant because the paper’s “first to apply machine learning to DCO” claim is too broad.
- Yang et al., 2012 — dynamic combinatorial optimization heuristics, relevant as a classical DCO baseline perspective.
- Wasim & King, 2020; Assadi et al., 2018; Ausiello et al., 2009 — dynamic MaxCut/MIS/TSP theory and algorithms, relevant for stronger non-neural comparisons.
- Wang & Li, 2023; Li et al., 2024 — instance-specific or meta-learning CO methods, relevant to positioning DyCO-GNN against the broader unsupervised CO literature.

## Suggestions
- Reframe the contribution more precisely as a dynamic adaptation of PI-GNN with shrink-and-perturb, rather than as the first learning-based DCO framework.
- Add direct comparisons to stronger dynamic and reoptimization baselines, and include a same-architecture from-scratch baseline under identical budget accounting.
- Provide diagnostics showing when warm start fails and whether SP actually changes optimization behavior beyond adding noise.
- Standardize evaluation protocol across tasks, especially checkpoint selection and runtime accounting, so the speedup claims are easier to trust.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 2.0]
Average score: 2.7
Binary outcome: Reject
