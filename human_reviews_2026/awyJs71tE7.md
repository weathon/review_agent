# Navigating the Accuracy-Size Trade-Off with Flexible Model Merging

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Model merging has emerged as an efficient method to combine multiple single-task fine-tuned models. The merged model can enjoy multi-task capabilities without expensive training. While promising, merging into a single model often suffers from an accuracy gap with respect to individual fine-tuned models. On the other hand, deploying all individual fine-tuned models incurs high storage costs. We propose FlexMerge, a novel data-free model merging framework that: (a) flexibly generates merged models of varying sizes, spanning the full spectrum from a single merged model to retaining all individual fine-tuned models; and (b) supports multiple merging algorithms in a unified framework. Using FlexMerge, we systematically characterize the accuracy–size trade-off of different  algorithms. Our study reveals two key findings: first, even modestly larger merged models can yield steep accuracy gains (up to 13.5% when just doubling the size); second, algorithm rankings are not consistent as size increases, with some methods overtaking others beyond the one-model regime. These results uncover a new design dimension for model merging: developing and comparing algorithms across the full spectrum of sizes rather than only at the single-model limit. Extensive experiments on vision and NLP benchmarks, with up to 30 tasks, confirm the generality and practicality of FlexMerge.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a data-free model fusion framework called FlexMerge, aimed at addressing the trade-off between accuracy and model size in multi-task model fusion. FlexMerge supports generating fusion models of any size (including non-integer multiples) and can integrate various data-free fusion algorithms within a unified framework. Through experiments, the authors pointed out: moderately increasing the model size can bring significant accuracy improvements, and the performance rankings of different fusion algorithms are inconsistent at different sizes.

### Strengths
S1: The author systematically studied the global trade-off between accuracy and size in model fusion for the first time, breaking through the limitations of traditional 'single-model fusion'.
S2: The paper proposes a data-free framework supporting the fusion of model with non-integer sizes.
S3: The author reveals that the phenomenon of "algorithm performance re-ranking under different model sizes.

### Weaknesses
W1: Lack of insight. Although the experiments are thorough, there is a lack of theoretical explanation and further analysis on 'why certain algorithms perform better at larger scales'.
W2: Lack of in-depth discussion on certain algorithms. Methods such as Consensus and EMR require storing masks or reconstructing them during inference. Although FlexMerge can mitigate this, its additional overhead is not discussed in depth in the paper.
W3:Experiment: There is a lack of comparison with similar data-free methods on the same task.
W4: Lack of theoretical analysis. The algorithm uses a greedy strategy to merge blocks, and no explanation of optimality is given.
W5: Limited external validity. While the experimental suite is thorough within classification and homogeneous Transformer backbones, the evidence may not fully generalize to other settings. It would be helpful to include evaluations on detection/segmentation/generation tasks and on heterogeneous architectures to more convincingly demonstrate robustness across modalities and model families.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes FlexMerge, a data-free model merging framework enabling flexible merged models across sizes (including non-integers) via greedy block merging. It balances accuracy-storage, unifies merging algorithms. Experiments on vision/NLP (up to 30 tasks) show steep accuracy gains with modest size increases, shifting algorithm rankings, and efficiency.

### Strengths
1.The paper is well organized and well written.

2.The authors present a well-motivated approach .

3.It conducts numerous experiments, validates the experimental results on models of various series and sizes, and covers a wide range of evaluation tasks.

### Weaknesses
1. **The necessity of the proposed architecture is not clearly justified.**
   According to previous findings from *DARE*[1], *Twin-Merge*[2], and *Consensus*[3], only a small portion of task-specific parameters truly contribute to performance, and task information can be effectively compressed through pruning techniques (e.g., SVD compression or masking).
   Given that the labels of the assumed dataset are known, the authors should compare *FlexMerge* with other methods under the same level of task information compression to demonstrate the superiority of their architecture.
   For example, the authors could compress the task vectors in *TA* using SVD and dynamically select vectors based on task labels for a fair comparison.

2. **In Figure 4(a), the performance of FlexMerge + TA decreases between 5 and 11 tasks.**
   What causes this degradation? The paper should provide an explanation or analysis of this phenomenon.

3. **In the OOD setting, task labels are not available.**
   How does *FlexMerge* decide which branch to activate in this case? The paper should clarify the mechanism used for branch selection under the OOD scenario.

[1]  Language models are super mario: Absorbing abilities from homologous models as a free lunch.

[2] Twin-merging: Dynamic integration of modular expertise in model merging. 

[3] Localizing task information for improved model merging and compression.

### Questions
see weaknesses

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose FlexMerge, a unified framework for data-free, block-level greedy model merging, which enables the combination of multi-task fine-tuned models at arbitrary scales from 1× to M×. The paper systematically demonstrates a favorable trade-off between accuracy and model size across vision, NLP, and multimodal scenarios, showing that scaling from 1× to 2× typically yields substantial accuracy improvements, while the performance approaches the single-task fine-tuning upper bound at scales far below M×, a phenomenon consistently observed across multiple merging algorithms.

### Strengths
1.The paper reframes model merging from a fixed single-target (1×) setup into a controllable continuous spectrum of model scales. This data-free formulation reflects real-world trade-offs between accuracy and resource budgets, turning a theoretical question into a deployable engineering solution.

2.A block-wise greedy strategy decomposes Transformers into Attention, MLP, and other units, enabling fine-grained control over model capacity. The proposed pluggable meta-framework allows the merging primitive  to be instantiated with different data-free algorithms (e.g., TA, TIES, Consensus), demonstrating strong versatility and generality. 

3.Experimental results across ViT-B/32 and ViT-L/14 (covering 8 and 30 task settings), as well as T5/T0-3B under both PEFT and FFT configurations and multimodal scenarios, show consistent trends: performance improves sharply from 1× to 2× capacity and then saturates, confirming the framework’s stability and scalability across merging algorithms.

### Weaknesses
1.Although the proposed block-wise greedy merging strategy demonstrates strong empirical performance, the paper offers no theoretical justification or formal analysis (e.g., approximation guarantees or error bounds). Consequently, it remains unclear under what conditions this greedy selection strategy converges to an effective merging configuration, or what its worst-case behavior might be under large inter-task divergence.

2.All experiments were conducted on models sharing identical architectures. The method’s feasibility and performance under heterogeneous or partially aligned architectures (e.g., differing scales or network structures) remain unexplored, despite such configurations being common in real-world multi-source settings. This assumption limits the method’s generalizability and real-world applicability.

3.Although the paper reports partial results on merging time, inference latency, and reconstruction delay, the overall analysis remains insufficient. There is a lack of explicit complexity characterization for the block-level similarity computation and greedy matching procedure, as well as no quantitative comparison of memory and storage costs across different model scales and merging primitives.

### Questions
1, How about the experiments on merging LLMs ?

2, Definition & Construction of Fractional Models：

The paper introduces fractional model units (e.g., 2.25→), but the mechanism remains unclear.

How are fractional models constructed at the parameter level?

Does “fractional” refer to partial layers, partial parameter blocks, or continuous interpolation over task vectors?

How do you ensure structural consistency and avoid breaking model geometry when only a “fraction” is used?

A more precise definition is needed to understand how fractional capacity is achieved.

3: Merging multiple fine-tuned models is known to cause representational interference. This risk may increase with fractional merging.

4: How do you ensure stability and avoid representation collapse when using fractional model portions?

5: Do fractional models exhibit smooth performance scaling or sudden drops at certain fractional ratios?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a block-level merging enhancement technique: it separates each expert into block groups and greedily merges the most similar pair within the same block using a given merging algorithm. During inference, it requires knowledge of the task ID and the specific block in which the task’s parameters reside, then routes the input to that block. The authors conduct extensive experiments on models across image domains (e.g., CLIP ViT‑B/32, ViT‑L/14) and PEFT/NLP setups.

### Strengths
- S1: This paper proposes a conceptually novel block-wise merging technique, serving as the interpolation between the single merged model and “retain‑everything”. It can balance performance and memory cost.
- S2: FlexMerge can plug into existing algorithms (TA, TIES, EMR, Consensus). The authors propose recomputing pairwise similarities and using a DSU structure to speed up the process.
- S3: The paper tests multiple base models (ViT‑B/32, ViT‑L/14), 8‑task and 30‑task suites, and PEFT scenarios (T0‑3B tasks), and conducts extensive ablation studies.

### Weaknesses
- **W1 (Major):** FlexMerging requires specific task iD. It needs a task ID to route task-specific inputs to the corresponding blocks, which imposes at least two constraints:  
  - It cannot be applied in scenarios where we do not know which task the input originates from. Requiring a known task ID is unfair when comparing against methods that make no such assumption—e.g., task arithmetic or TIES.  
  - It cannot handle out-of-domain evaluation, which is very common in practice. The method assumes that an input from task A must be assigned to the merged block containing the expert parameters for task A. If an unknown task B is presented, the system cannot assign it appropriately.

- **W2 (Major):** The computational complexity for $M$ tasks × $B$ blocks (involving pairwise similarity matrix computation and greedy merging steps) is not fully analyzed. For instance, with very large $M$ (e.g., 100+ tasks) and large $B$ (e.g., DeepSeek with 61 layers or Qwen with 94 layers), the merging cost in terms of memory and time remains unknown.

- **W3 (Minor):** The target size appears more empirical and may count different stored artifacts (e.g., masks for Consensus, per-block scalars for EMR). Choosing a size given a memory or accuracy target is heuristic.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3
