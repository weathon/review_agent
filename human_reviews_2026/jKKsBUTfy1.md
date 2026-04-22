# MIDAS: Mosaic Input-Specific Differentiable Architecture Search

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Differentiable Neural Architecture Search (NAS) provides efficient, gradient-based methods for automatically designing neural networks, yet its adoption remains limited in practice. We present MIDAS, a novel approach that modernizes DARTS by replacing static architecture parameters with dynamic, input-specific parameters computed via self-attention, enriching differentiable NAS with the representational power of self-attention.
To improve robustness, MIDAS (i) localizes architecture selection by computing it separately for each spatial patch of the activation map, and (ii) introduces a parameter-free, topology-aware search space that models node connectivity and simplifies selecting the two incoming edges per node. We evaluate MIDAS on the DARTS, NAS-Bench-201, and RDARTS search spaces. In DARTS, it reaches 97.42% top-1 on CIFAR-10 and 83.38% on CIFAR-100. In NAS-Bench-201, it consistently finds globally optimal architectures. In RDARTS, it sets the state of the art on two of four search spaces on CIFAR-10. We further analyze why MIDAS works, showing that patchwise attention improves discrimination among candidate operations, and the resulting input-specific parameter distributions are class-aware and predominantly unimodal, providing reliable guidance for decoding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes MIDAS, a differentiable architecture search (NAS) method that generates input- and position-specific architecture parameters via an attention-based mechanism and a patchwise feature decomposition. It also introduces a “topology-aware” module that scores edge pairs to enforce two-input constraints. The method is evaluated on standard NAS benchmarks like NAS-Bench-201, DARTS, and RDARTS search spaces, reporting competitive results.

### Strengths
1.The idea of conditioning architecture parameters on input features is interesting and could make NAS more adaptive.
2.The paper provides extensive experiments and ablations across several standard benchmarks.

### Weaknesses
1.I do not see the necessity of partitioning the feature map and learning patch-level architecture parameters, since Eq. (14) simply averages them to obtain a global parameter. This seems mathematically equivalent to directly learning a global weight as in DARTS. The authors should clarify the actual benefit of this design.
2.The use of self-attention is questionable. The defined queries (from raw features) and keys (from operated features) do not form true self-attention, and the semantic meaning of their inner product is unclear. Why should this similarity represent an architecture weight?
3.Compared to DARTS, MIDAS introduces many more architecture parameters and higher computational complexity, but no analysis of FLOPs, memory, or parameter count is provided.
4.The marginal improvement on the DARTS search space is not strong enough to justify the added complexity.

### Questions
see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the self-attention into the differentiable NAS method. I provides a clear motivation for problems of DARTS that are addressed and the weak correlations stemming from the operation choice in DARTS. input-guided mixing can be a potential fix. Experiments are conducted on two datasets that that show improvements over DARTS baselines.

### Strengths
- Interesting idea and novel use of the self-attention technique with a promising new direction. Replacing global architecture parameters with a learned mapping from input features to architecture mixing weights (via attention) is a concrete extension not widely explored in standard DNAS papers.
- Methodological process follows best practices for fairness and reproducibility.

### Weaknesses
- The contributions do not read well. There may be made more explicit and clearer. 
- Not clear how this scales to larger resolution datesets since the self-attention has quadradic scaling.
- I would like to see more varying datasets used in the experiments for searching for networks. CIFAR100/CIFAR10 have the same properties more or less (e.g., content, resolution)
- Other NAS methods make the problem stricter by incorporating additional constraints with regards to FLOPS/Memory/Parameters
- Limited ablations to understand the impact and main mechanisms providing that improve performance

### Questions
- Can you provide more connections to the dynamic routing literature?
- Why are the results between proposed approach and β-DARTS in Table 1 are exactly the same?

### Soundness
3

### Presentation
3

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
This paper presents a variant of DARTS. Using self-attention mechanism to design a architecture selection process. Experiments have been done on common toy search space like DARTS, NASBench201 and RDARTS on image classification task.

### Strengths
Paper is written pretty clear about their technical detail.
They put three space and provide some ablation on their own method.

### Weaknesses
Weaknesses:

As using differentiable archtiecture search is try to minimize the search cost, this field still lacks certain thereotical support, why using a shared weights super net and differentiable target towards certain architecture selection process can work. Any research without clearly addressing direction seems to be meaningful only on practical point of view.

In this regard, this paper presents a result that only on-par or even worse compared to some old baselines, like beta-DARTS in 2022. 

The only motivation on this one is to leverage the full representation power of self-attention mechanism, which is quite straightforward and trivial honestly speaking. 

As such, I have not seen any reason to accept this paper, for lacking interesting motivation, compelling result.

### Questions
As weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
