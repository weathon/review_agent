# Distilling Expressive GNNs into MPNNs

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
We investigate the distillation of expressive graph neural networks (GNNs) into simple message passing graph neural networks (MPNNs) for the case of molecular graph data. Under the standard training protocols used in prior work, expressive GNNs substantially outperform simple MPNNs on popular molecular benchmarks. We demonstrate that knowledge distillation closes 50 to 100% of this gap. Importantly, the distilled MPNNs are 2 to 33 times faster than their teachers. Our results suggest that for these molecular tasks, the performance gap is largely due to optimization challenges rather than fundamental expressivity limitations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper examines whether the performance gap between “expressive” graph neural networks and plain message-passing neural networks truly stems from greater expressivity or instead from optimization and data issues, and it shows that distilling expressive teachers into a simple Graph Isomorphism Network student consistently narrows or closes much of this gap while substantially accelerating inference. Using three complementary strategies—soft-label distillation for classification, layer-wise embedding alignment for regression, and teacher-labeled data augmentation—the study transfers supervision from five diverse teacher families to a standard GIN and evaluates the result on molecular benchmarks under a unified training protocol. The authors propose a systematic distillation framework from expressive GNNs to MPNNs and, through comprehensive experiments, demonstrate that distilled GINs recover 50–100% of the accuracy gap, mitigate overfitting and oversmoothing, and deliver 2×–33× faster inference than the teachers, with especially strong gains on ZINC and MolHIV. Overall, the results indicate that the apparent “expressivity advantage” on popular molecular datasets can be matched by optimization-oriented supervision without increasing the student’s theoretical power, suggesting that expressivity is neither necessary nor sufficient for top performance on these tasks.

### Strengths
1. The work isolates whether the gap between “expressive” GNNs and plain MPNNs is due to expressivity or to optimization/data, and it deliberately centers on molecular benchmarks where this gap is prominent (e.g., ZINC, MolHIV), avoiding confounds from domains where expressive models rarely help.


2. The methodology is easy to adopt and extend: soft-label distillation for classification, layer-wise embedding alignment for regression, and a pragmatic teacher-labeled augmentation scheme. Each component is specified and compatible with standard GIN students, lowering the barrier to replication.


3. Distilled GINs recover roughly 50–100% of the accuracy gap in many cases, particularly on classification tasks; they also mitigate overfitting and oversmoothing, enabling deeper MPNNs to remain stable and competitive.


4. Students deliver substantial inference speedups (about 2×–33×) relative to expressive teachers while maintaining comparable accuracy, offering a practical path to favorable accuracy–latency/throughput trade-offs without heavy architectures.


5. By matching much of the “expressivity advantage” through optimization-oriented supervision without increasing theoretical power, the study suggests expressivity is neither necessary nor sufficient for top performance on these benchmarks, guiding practitioners toward data and training improvements and redirecting research toward better optimization and supervision signals.

### Weaknesses
1. The paper lacks a formal guarantee that distillation endows the student with the teacher’s expressive power: knowledge distillation aligns to teacher signals on a given data distribution but does not expand the student’s hypothesis class (e.g., a GIN/MPNN), so parity with the teacher’s decision boundary or isomorphism-discriminating capability is not theoretically ensured in general or worst-case regimes. Evidence of effectiveness is primarily empirical, making the claims contingent on the reported benchmarks rather than principled. The work provides no realizability assumptions (whether the teacher function lies within the student class), no explicit bounds separating approximation/estimation/optimization errors, and no sample-complexity or robustness guarantees under distribution shift; nor does it offer provable effects of key hyperparameters (temperature, alignment weight, augmentation policy) on generalization and stability. As a result, “transfer of expressivity” remains an empirical observation rather than a theoretically substantiated property.

2. The paper’s evidence for distillation largely comes from molecular benchmarks, whose graphs are often “tree-like” (sparse, low treewidth, small rings, long chains). On pure trees, the classical result is that 1-WL—and thus MPNNs with 1-WL expressivity—can distinguish any pair of non-isomorphic trees ([1]Corollary 1.8.2). This implies that, on such distributions, MPNNs already have complete isomorphism-discrimination power. Therefore, observing that a distilled student approaches the teacher on predominantly tree-like molecular data does not establish that the same holds on broader graph families with high symmetry, rich cyclic structure, large treewidth, or heterogeneous/temporal relations. In other words, the current evidence mainly demonstrates optimization gains in regimes where 1-WL is already sufficient or nearly sufficient, without truly stress-testing distillation on settings that demand higher expressivity (e.g., known 1-WL counterexamples, structure-counting/global properties, strongly regular/high-automorphism graphs, or node/edge-level tasks with long-range dependencies). To strengthen generality, the study should include 1-WL-hard benchmarks (e.g., strongly regular or circular-skip-link–type datasets), report stratified results on 1-WL-distinguishable vs. indistinguishable subsets, expand beyond molecular graphs to heterogeneous, temporal, and large-scale graphs, and verify whether distillation still closes the teacher–student gap on node/edge-level tasks.

3. The paper infers that, because graphs in many popular molecular benchmarks are distinguishable, “MPNNs should be able to match the performance of more expressive GNNs given enough parameters.” This extrapolation is not warranted. Many molecular graphs are effectively tree-like, and for trees, 1-WL—and hence MPNNs with 1-WL expressivity—already distinguishes any pair of non-isomorphic trees. Thus, success on predominantly tree-like molecular datasets reflects the structural simplicity of the domain rather than a general gain in expressivity. Crucially, capacity ≠ expressivity: increasing depth/width only improves approximation within the 1-WL hypothesis class; it does not break 1-WL’s inherent limits. As a result, plain MPNNs still struggle on families that are hard for 1-WL (e.g., highly symmetric or strongly regular graphs), on tasks requiring global/subgraph counting or long-range dependencies, and on graphs with large treewidth or complex cyclic structure. Therefore, using performance on molecular benchmarks to claim that MPNNs can, in general, match more expressive GNNs is unconvincing. To support broader claims, the study should evaluate on 1-WL-hard benchmarks (including synthetic counterexamples), tasks that demand explicit subgraph counting or global properties, heterogeneous/temporal/large-scale graphs, and node/edge-level settings, and then verify whether distilled MPNNs still close the gap.

4. A  fundamental limitation of the approach is that distillation occurs only during training, while inference remains a lightweight MPNN whose expressivity is bounded by 1-WL. Consequently, for any pair of nodes (or graphs) that are 1-WL–equivalent, the distilled student must still produce identical representations and predictions at test time: distillation adjusts parameters but does not expand the hypothesis class or alter the model’s 1-WL aggregation/update mechanism, so it cannot break symmetries that 1-WL cannot distinguish. At best, distillation improves optimization and calibration—reducing empirical risk within the existing 1-WL function class—rather than upgrading the student’s theoretical capacity to separate structures that require higher-order reasoning (e.g., strongly regular graphs, global counting properties, or long-range dependencies). Unless the teacher’s stronger structural signals are injected as inference-time features (e.g., subgraph counts, k-WL/spectral/RW positional encodings, virtual nodes/line graphs) or the architecture itself is made more expressive, training-only knowledge transfer will not raise the student’s expressivity at inference. In short, the reported gains reflect better supervision and optimization, not an intrinsic increase in expressive power; on 1-WL–equivalent cases, a distilled MPNN will still output the same predictions, leaving the expressivity bottleneck intact.

5. Although the distilled student runs faster at inference, the life-cycle cost may still be high once factoring in teacher pretraining/finetuning, large-scale teacher labeling (forward passes, storage), student distillation and hyperparameter search, plus data-augmentation generation and filtering. The paper does not provide a unified accounting of wall-clock time, GPU-hours, energy (e.g., kWh/CO₂e), or cost-per-gain trade-off curves, nor amortization analyses under different deployment horizons and traffic levels. As a result, it is unclear whether distillation is more economical than alternatives such as deploying a stronger small baseline or using pruning/quantization.

6. Insufficient robustness and reproducibility: Sensitivity to key knobs—temperature, alignment weight, and augmentation ratio/policy—is only lightly explored; robustness under alternative splits (official/random/stratified), distribution shift (OOD), larger graph scales (more nodes/edges, deeper dependencies), and cross-seed/hardware stability is under-evaluated. This leaves the results’ dependence on implementation details and training stochasticity unclear, raising replication and deployment risk.

7. The empirical support comes almost entirely from graph-level molecular benchmarks—typically small, sparse, and tree-like with graph-level targets—while omitting systematic evaluation on node-/edge-level tasks (node classification, link prediction, edge property prediction), large-scale non-molecular graphs (social, e-commerce, finance, transportation), and heterogeneous/temporal graphs (multi-type nodes/relations, time-evolving interactions). As a result, claims that distilled MPNNs remain effective more broadly lack a solid basis for extrapolation. These settings differ in separability, long-range dependencies, automorphism/symmetry, noise, and distribution shift (OOD); success on molecular data does not establish robustness on high-symmetry or high–treewidth graphs, richly cyclic structures, cross-domain transfer, or large-scale real-time inference.

[1] Neil Immerman and Eric Lander. Describing graphs: A first-order approach to graph canonization. In Complexity theory retrospective, pages 59–81. Springer,1990.

### Questions
Regarding the seven weaknesses of the paper, I raise the following questions;

1. Can authors provide a formal guarantee that distillation endows the student with the teacher’s expressive power?

2. (1)Given that many molecular graphs are tree-like (where 1-WL is sufficient), how do authors' results generalize to graphs with high symmetry, rich cycles, or large treewidth?（2）Can authors include 1-WL-hard benchmarks (e.g., strongly regular or circular skip-link graphs) and report stratified results on 1-WL-distinguishable vs. indistinguishable subsets? (3)Beyond molecules, will authors evaluate on heterogeneous/temporal/large-scale graphs and node/edge-level tasks to test whether distillation still closes the teacher–student gap?

3. (1) On what basis do authors claim that “an MPNN can match more expressive GNNs given enough parameters”? How do authors separate capacity from expressivity? (2) Can authors evaluate on synthetic 1-WL counterexamples and high-expressivity tasks, and report whether the teacher–student gap closes as parameters scale?

4. (1)Under training-only distillation with a 1-WL-bounded MPNN at inference, isn’t the student inherently unable to distinguish any pair of 1-WL-equivalent structures?(2)How do authors validate that current gains reflect optimization/calibration rather than expressivity?

5. Can authors report end-to-end costs covering teacher training/labeling, student distillation and hyperparameter search, and data augmentation (wall-clock, GPU-hours, energy/CO₂e, cost–gain curves)?

6. (1)How sensitive are the results to temperature, alignment weight, and augmentation ratio/policy? Do you report systematic sensitivity curves and confidence intervals?(2)Are results stable under alternative splits, OOD, larger graph scales, and across seeds/hardware?

7. (1)Beyond graph-level molecular benchmarks, will you evaluate node-/edge-level tasks (node classification, link prediction, edge properties)? (2)Will you include large-scale non-molecular graphs (social, e-commerce, finance, transportation) and heterogeneous/temporal graphs to test performance under high symmetry, large treewidth, and long-range dependencies?

This work exhibits strong potential;If these issues are addressed, I will raise my score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work, the authors transfer knowledge from a highly expressive GNN as a teacher model to a MPNNs with lower expressiveness but faster inference as a student model, thereby improving the performance of the student model.

### Strengths
This work proposes a universal distillation framework that compresses multiple high expressive teachers into MPNN students with linear complexity and improves the performance on multiple molecular benchmarks.

### Weaknesses
The authors select the strongest teacher in multiple test sets, weakening the reliability of the core conclusion that expressive power is not the main cause. As the experiment mainly focuses on molecular benchmarks, the generalizability is not clear.

### Questions
1.Need more experimental or theoretical support to determine that 'expressive power is not the main cause'.

2.Experiments on other datasets beyond molecular benchmarks are necessary. It can provide a clearer explanation of generalizability.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the use of knowledge distillation (KD) to transfer the performance of expressive, but computationally expensive, Graph Neural Networks (GNNs) to simpler, faster Message-Passing Neural Networks (MPNNs). The authors conduct extensive benchmarking on molecular datasets, demonstrating a significant performance gap between expressive GNNs and MPNNs. They then show that through KD (using soft labels, layer alignment, and data augmentation), the MPNN student can close 50-100% of this performance gap while achieving 2x to 33x inference speedups. The central claim is that this success challenges the common assumption that the performance gap is primarily due to the limited expressivity of MPNNs, suggesting instead that optimization difficulties are a major factor.

The core idea is compelling, the experimental setup is rigorous, and the results are significant. The work successfully makes a strong empirical case that KD can effectively bridge the performance-efficiency gap and that expressivity alone does not explain the superiority of modern GNNs. However, the paper's central claim regarding the "death of expressivity" is overstated and not fully supported by the evidence, leading to some conceptual overreach. The practical contributions are clear, but the theoretical implications require more careful framing.

### Strengths
**Rigorous and Extensive Benchmarking:** The paper provides a comprehensive and fair comparison across a diverse set of expressive GNN architectures (GSN, CWN, DSS, L2GNN, GRIT) and a strong MPNN baseline (GIN) on multiple molecular benchmarks. The use of statistical significance testing and hyperparameter sensitivity analysis adds considerable weight to the findings.

**Clear Practical Utility:** The demonstration that KD can make MPNNs competitive with state-of-the-art expressive models while being drastically faster is a valuable contribution. The speedup figures of 2x to 33x are impressive and have immediate practical implications for deploying GNNs in resource-constrained environments.

**Effective and Simple KD Methods:** The paper shows that relatively simple KD techniques (soft labels, layer alignment) are highly effective. This is a strength, as it avoids the complexity of more intricate distillation methods and makes the approach accessible.

**Insightful Analysis:** The analysis of hyperparameter sensitivity (showing expressive GNNs are often less sensitive) and the exploration of data augmentation scaling are insightful and provide useful guidance for the community.

### Weaknesses
**1. Overstated Claims on Expressivity:** The paper's most significant weakness is the leap from "KD works well" to "the performance gap is not caused by expressivity." This conclusion is not fully justified.

   - Distillation Transfers Inductive Bias, Not Just Labels: A more nuanced interpretation is that KD successfully transfers the inductive bias of the expressive teacher to the MPNN student. The teacher's architecture, by being more expressive, is better at learning a useful function from the data. The student, through KD, is guided to approximate this function without having to discover it from scratch, thus circumventing its own optimization challenges. This does not mean expressivity is irrelevant; it means the teacher's expressivity was crucial in finding the solution that the student then imitates.

   - The WL Test is a Worst-Case Measure:** The theoretical expressivity of MPNNs is bounded by the WL test, which is a worst-case, graph-level discriminative measure. The tasks in the paper are not about distinguishing all non-isomorphic graphs but about learning specific predictive functions on a given data distribution. It is entirely possible that the ideal function for these specific tasks lies within the function class representable by an MPNN, but that finding it via ERM is difficult. KD acts as a powerful regularizer and optimization guide. The paper confuses the representational capacity of the model class with the learnability of a specific function within that class via standard training.

   - Suggestion: The authors should temper their claims. They should frame their results as demonstrating that optimization and inductive bias are critical factors that have been overlooked in the pursuit of expressivity, rather than claiming expressivity is not a cause of the performance gap.

**2. Limited Exploration of "Why KD Works":** The paper thoroughly shows that KD works but provides limited analysis into why. For instance, the hypothesis that KD acts as a regularizer (Figure 4) is mentioned but not deeply investigated. A more detailed analysis of the loss landscape or the representations learned by the distilled MPNN vs. the baseline MPNN could provide stronger mechanistic evidence for their claims.

**3. Dataset Selection and Generalizability:**

   - Molecular Focus: The exclusive focus on molecular datasets, while justified by the performance of expressive GNNs there, limits the generalizability of the conclusions. The claim that "the performance gap is not caused by expressivity" would be stronger if tested on a non-molecular dataset where graph structure is known to be critical and MPNNs theoretically struggle (if such a dataset can be identified). The dismissal of non-molecular datasets, while understandable, also conveniently avoids a scenario that might contradict the core thesis.


**4. Clarity of the "Layer Alignment" Method:** The description of layer alignment for models that compute embeddings for non-node structures (e.g., CWN's edges and cycles) is vague. Stating "we compare the mean embeddings at every layer, where the mean is computed over all embedded objects" is ambiguous. How are embeddings of different dimensions and from fundamentally different objects (nodes, edges, cycles) made comparable? This needs a precise mathematical description.

### Questions
- How do you reconcile the success of distillation with the theoretical expressivity limits of MPNNs? Could it be that the teacher's expressivity is crucial for learning a good function from the data, which the student then approximates, rather than the student's expressivity being sufficient for discovering it independently?

- It is better if the authors can discuss some other related works:

   - Polarized message-passing in graph neural networks
   - Graph Spiking Attention Network: Sparsity, Efficiency and Robustness

- Given that you select teachers based on test set performance, how can you ensure that your reported results for GIN+KD are not optimistically biased due to information leakage from the test set?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper distills a variety of expressive teacher GNNs into GIN MPNN students, showing 50–100\% gap closure with 2–33× inference speedups, especially on molecular benchmarks where WL-level expressivity often suffices. For classification tasks, soft-label distillation consistently lifts GIN, while for regression, layer alignment plus teacher-labeled augmentation helps on ZINC and alchemy but provides limited or no gains on some datasets (e.g., QM9). Findings suggest optimization and inductive biases, rather than inherent expressivity limits, are primary drivers of observed performance gaps in many molecular settings, making KD a practical path to near-teacher accuracy with MPNN efficiency.

### Strengths
1. Substantial gains with simple KD and strong runtime benefits across multiple datasets.
2. Nuanced reframing of the expressivity-performance link with empirical evidence on molecules.
3. Diverse teachers and a clear GIN baseline improve comparability and insight.

### Weaknesses
1. Claims appear broad relative to molecular scope; tone down “invalidates” beyond covered domains.
2. Need more details on budgets, fairness, and variance to support generality.
3. Limited guidance on when KD fails due to true expressivity gaps, especially off molecules.

### Questions
1. Report calibration, robustness, and OOD performance of distilled models.
2. Sensitivity to student size, augmentation, and KD loss; consider cross-validation.
3. Share checkpoints and pipelines to support reproducibility.

### Soundness
3

### Presentation
3

### Contribution
3
