Now I have all the information needed to write the final review. Let me synthesize all verified findings.

## Summary

SPARC (Simple Parameter Isolation in a Restricted Capacity) proposes a rehearsal-free, surrogate-free continual learning method that decomposes depth-wise separable convolutions (DSCs) into task-specific depth-wise filters ("working memories") and shared point-wise filters updated via EMA ("semantic memory"), plus a weight re-normalization technique to counteract task-specific bias in the classification layer. The method achieves strong Class-IL performance on harder benchmarks (Seq-CIFAR100, Seq-TinyImageNet, Seq-ImageNet100) while using significantly fewer parameters than surrogate-based baselines.

## Strengths

- **Strong empirical results on harder benchmarks where rehearsal methods struggle**: On Seq-CIFAR100 (5 tasks), SPARC achieves 49.03% Class-IL accuracy, outperforming all compared methods including CLS-ER (43.80%), OCDNet (44.29%), and TAMIL (41.43%) (Table 1). On Seq-TinyImageNet (10 tasks), SPARC achieves 32.29% vs. CLS-ER's 23.47%—a 9-point advantage. These are the settings where buffer-to-class ratios are unfavorable for rehearsal, directly validating the paper's core motivation.

- **Clean and well-motivated system architecture**: The decomposition of DSCs into task-specific depth-wise filters and shared point-wise filters with EMA consolidation (Eq. 3-4, Section 3.1-3.2) is a principled mapping of parameter isolation into an efficient architecture. The ablation in Table 5 demonstrates this quantitatively—semantic consolidation (1.04M) nearly matches full separation (1.65M, 59% more parameters) with only ~5% relative performance gap.

- **Identification and mitigation of task-specific bias**: The observation that parameter isolation amplifies output magnitude disparity across tasks (Section 3.3) is a real practical problem. The IQR-based re-normalization technique (Eq. 5) addresses it without additional validation sets or model components, and the method appears effective in the results.

- **Favorable stability-plasticity tradeoff**: Figure 4 (left) shows SPARC achieves the highest stability (~50%) and best tradeoff (~48%) on Seq-CIFAR100 compared to ER, DER++, and LIDER, supporting the claim that isolation plus semantic consolidation effectively balances stability and plasticity.

- **Scalability advantage over parameter isolation peers**: Figure 2 and Table 4 show SPARC achieves 88.18% Task-IL on 20-task Seq-CIFAR100 with only 3.62M parameters, versus PNNs' 2645.05M—a clear advantage over the strongest parameter isolation baseline (CPG at 80.89%).

- **ImageNet-scale performance**: Table 3 shows SPARC achieves 50.90% on Seq-ImageNet100 (10 tasks) with 1.9M parameters and no dataset-specific tuning, outperforming LUCIR (41.4%) and all other compared methods.

## Weaknesses

### Fatal
None.

### Major

- **The headline "6% of parameters" efficiency claim conflates architecture choice with CL method contribution**: SPARC uses DSC-based convolutions (inherently parameter-efficient) while baselines use standard ResNet-18 convolutions. The dramatic parameter reduction is primarily a property of DSCs—an established architectural technique—rather than an innovation of the CL method. The paper is transparent about this architectural difference (Section 4, line 123: "While most baselines use ResNet-18 as the backbone, SPARC utilizes a ResNet-18-like architecture with DSC layers") and mentions a same-architecture comparison in Appendix D.2, but relegates this critical control to supplementary material. Without it in the main paper, the reader cannot disentangle how much of SPARC's efficiency is architectural versus algorithmic. This does not mean SPARC lacks efficiency—it clearly does—but the "6%" framing is misleading as a measure of the CL method's contribution.

- **Class-IL inference cost scales linearly with tasks and remains unreported**: Section 3.4 explicitly states that Class-IL inference requires "each image [to be] independently processed through all sub-networks," meaning k forward passes for k tasks. Table 1 reports "1F, 1B" which appears to be per-task training cost. For a paper whose central framing is efficiency and scalability, the absence of any inference cost analysis (wall-clock time or FLOPs) for the primary evaluation setting (Class-IL) is a significant gap. This linear inference scaling directly affects SPARC's practical relevance in the "memory-constrained environments" it targets.

### Minor

- **The "scalable" framing in the abstract is overbroad**: The abstract calls SPARC "a scalable CL approach," but Table 4 shows clear linear growth (1.04M→1.90M→3.62M). While this is vastly better than PNNs' quadratic growth, linear growth is not scalable in the unlimited-task setting the CL community typically considers. The Limitations section (Section 5) honestly acknowledges that "SPARC grows way beyond other rehearsal-based and weight regularization counterparts" in longer sequences, but this honesty is undermined by the abstract's unqualified "scalable" claim. The paper demonstrates *more scalable than some parameter isolation methods*, not *scalable* in the general sense.

- **"Matches rehearsal-based methods" in abstract is imprecisely scoped**: On Seq-CIFAR10, SPARC (61.22%) substantially trails OCDNet (73.38%), TAMIL (68.84%), and CLS-ER (66.19%). While the paper body is honest about this (Section 4.1: "In simpler scenarios like Seq-CIFAR10, SPARC's performance is competitive but lags behind"), the abstract's "matches rehearsal-based methods on various CL benchmarks" is overly broad. A more precise claim—"matches or exceeds rehearsal-based methods on benchmarks where buffer-to-class ratios are unfavorable"—would be more accurate and arguably stronger.

- **κ=5 re-normalization constant lacks sensitivity analysis**: The weight re-normalization constant (Eq. 5) is set to 5 without justification beyond experimental results. A simple ablation would strengthen this design choice.

- **Task boundary assumption restricts applicability**: The paper requires task boundary knowledge for sub-network allocation. While acknowledged in Limitations, this restricts SPARC to task-incremental settings—increasingly considered a narrow subset of the CL problem.

### Trivial
None.

## Nice-to-Haves

- Same-architecture comparison (baselines with DSC backbone or SPARC with standard convolutions) as a main-paper table, since this would directly address the most significant concern about the efficiency claim.
- Inference cost analysis for Class-IL (FLOPs or wall-clock time) across varying numbers of tasks.
- Sensitivity analysis for κ.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic claim: "SPARC lacks any fast-learning temporary buffer analogous to the hippocampus"**: This misunderstands the CLS analogy. SPARC maps depth-wise filters to working memory (fast, task-specific) and shared point-wise filters to semantic memory (slow consolidation). The mapping is to working/semantic memory, not hippocampus/neocortex directly. The biological analogy can be critiqued as overstated, but it isn't structurally missing a component.

- **Harsh Critic claim: "The EMA update with α close to 1 means semantic memory isn't contributing; the method is essentially hard parameter isolation"**: Table 5 directly contradicts this: shared point-wise + separate depth-wise (0.43M, 42.77%) vs. semantic consolidation (1.04M, 49.13%) shows a clear performance gain. The semantic memory does contribute. The harsh critic's alternative reading of Figure 4 (α=1.0 being best) suggests slow or no aggregation works best, but this does not mean shared features contribute nothing—they are still trained through gradient descent during the first task and carried forward.

- **Strength Finder claim: "Rehearsal-free and full-surrogate-free design with competitive performance" as a unique advantage**: While true, this is somewhat generic—several parameter isolation methods are also rehearsal-free. The specific contribution is the efficient architectural decomposition, not merely being rehearsal-free.

- **Harsh Critic claim about "false dichotomy" between SPARC and surrogate methods**: The paper does not create a false dichotomy—it correctly notes that surrogate-based methods introduce overhead and proposes an alternative. There is no logical error in this framing.

- **Harsh Critic concern about Figure 4 using buffer size 500 while Table 1 uses 200**: The stability-plasticity analysis is a complementary metric, not a direct comparison with Table 1 results. Using a different buffer size for this specific analysis is a design choice, not an inconsistency that invalidates anything.

## Novel Insights

The most novel insight in this paper is that depth-wise separable convolutions naturally decompose into components that align with complementary learning systems: spatial (depth-wise) filters are inherently task-specific and should be isolated, while channel-projection (point-wise) filters can be productively shared across tasks via slow EMA consolidation. This is not just using DSCs for efficiency—it leverages the structural factoring of the convolution operation itself as a design principle for CL. The weight re-normalization for task-specific bias, while simple, identifies a concrete and underappreciated practical problem in parameter isolation methods that deserves broader attention.

## Suggestions

- Move the same-architecture comparison from Appendix D.2 into the main paper as a primary table. This single addition would resolve the most significant concern about the efficiency claims and would likely require only a small additional experiment.
- Report Class-IL inference FLOPs or wall-clock time in Table 1 (or a supplementary table), broken down by number of tasks. This need not be exhaustive—even 5/10/20-task numbers would contextualize the practical tradeoff.

## Score and Decision

**Calibration anchors:**

| Paper | Score | Comparison |
|-------|-------|-----------|
| SD-LoRA (5U1rlpX68A) | 7.50 (Oral) | Stronger: has theoretical analysis, fair parameter comparison, rehearsal-free with LoRA decoupling — SPARC lacks theoretical depth and has the apples-to-oranges issue |
| Budgeted CL (dOAkHmsjRX) | 7.50 (Spotlight) | Stronger: uses FLOPs/bytes as fair budget metrics, addresses the exact inference cost concern SPARC omits |
| SEED (sSyytcewxe) | 7.00 (Poster) | Comparable: both are parameter isolation with linear growth concerns, but SEED's comparison is fairer while SPARC has stronger empirical results |
| OVOR (FbuyDzZTPt) | 6.00 (Poster) | Comparable: rehearsal-free prompt-based CIL, also claims efficiency but has missing baselines — SPARC has a similar apples-to-oranges concern but stronger results |
| HyperGKL (9lvyCHhQix) | 4.00 (Reject) | Weaker: unclear method, computational concerns, reviewer confusion — SPARC is clearly above this |
| OWA (HCCkCjClO0) | 3.00 (Reject) | Much weaker: unfair comparison, weak baselines, overclaimed efficiency without real results — SPARC has genuine strong results but shares overclaimed efficiency framing |

SPARC presents a genuine, well-designed contribution with strong results on harder CL benchmarks. However, the apples-to-oranges parameter comparison and unreported Class-IL inference cost are significant concerns that undermine the central efficiency framing. Compared to accepted CL papers at ICLR (SD-LoRA at 7.5, Budgeted CL at 7.5, SEED at 7.0), SPARC lacks theoretical depth, fair parameter comparison, and inference cost analysis. Compared to borderline/low papers (OVOR at 6.0, HyperGKL at 4.0), SPARC has stronger empirical results and a cleaner design. The paper is above the reject threshold—the method works and the design is sound—but the efficiency claims need honest rescoping. A score of 5.5 reflects a paper with real contributions that are partially undermined by framing choices.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>