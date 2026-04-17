# WANT TO TRAIN KANS AT SCALE? NOW UKAN!

- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Kolmogorov–Arnold Networks (KANs) have recently emerged as a powerful alternative to traditional multilayer perceptrons. However, their reliance on predefined, bounded grids restricts their ability to approximate functions on unbounded domains. To address this, we present Unbounded Kolmogorov–Arnold Networks (UKANs), a method that removes the need for bounded grids in traditional Kolmogorov–Arnold Networks (KANs). The key innovation of this method is a coefficient-generator (CG) model that produces, on the fly, only the B-spline coefficients required locally on an unbounded symmetric grid. UKANs couple multilayer perceptrons with KANs by feeding positional encoding of grid groups into the CG model, enabling function approximation on unbounded domains without requiring data normalization. To reduce the computational cost of both UKANs and KANs, we introduce a GPU-accelerated library that lowers B-spline evaluation complexity by a factor proportional to the grid size, enabling large-scale learning by leveraging efficient memory management, in line with recent software advances such as FlashAttention and FlashFFTConv. Performance benchmarking confirms the superior memory and computational efficiency of our accelerated KAN (warpKAN), and UKANs, showing a $3-30\times$ speed-up and up to $1000\times$ memory reduction compared to vanilla KANs. Experiments on regression, classification, and generative tasks demonstrate the effectiveness of UKANs to match or surpass KAN accuracy. Finally, we use both accelerated KAN and UKAN in a molecular property prediction task, establishing the feasibility of large-scale end-to-end training with our optimized implementation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Introduces UKAN (Unbounded Kolmogorov–Arnold Networks): removes the bounded grid constraint in KANs by generating local B-spline coefficients on-the-fly with a coefficient-generator (CG) MLP conditioned by a positional encoding of grid groups.

Presents warpKAN, a GPU-accelerated implementation that exploits compact B-spline support to cut evaluation from O(K·G·d_in·d_out) to O(K·d_in·d_out) (K = degree+1; G = grid size), reporting 3–30× speedups and up to ~1000× memory reduction vs “vanilla” KANs.

Benchmarks across regression, classification (moons, MNIST), a PINN toy ODE, a toy diffusion model, and molecular property prediction; often matches or beats KAN/MLP baselines.

Notes a clean UKAN→KAN mapping for inference over finite intervals (materialize coefficients).

### Strengths
Clear systems contribution: the matrix-form B-spline evaluation and careful GPU design are well-motivated; the roofline analysis is a nice touch. 

Practical unboundedness: the CG strategy elegantly avoids brittle grid updates and out-of-support issues while keeping local B-spline stencils. 

Broad applicability: results span classic toy problems to ADME prediction; shows UKAN is drop-in across tasks (including swapping into GNN blocks). 

Reproducibility extras: explicit complexity table, training details (optimizers/schedules), and appendix with architectural specifics.

### Weaknesses
Ablations on UKAN design are thin. How sensitive are results to: CG MLP depth/width, positional-encoding dimension, group size K, grid step h, and the presence/absence of the extra scaling parameter for coefficients? Provide loss-vs-epoch and final accuracy curves for these. 

Scope of real-world evals. ADME results are promising but limited to four datasets with fixed featurizers; stronger evidence would include graph featurization end-to-end (e.g., message-passing + UKAN) on larger public sets, with statistical tests and variance reports across multiple splits.

Interpretablity of UKANs? especially it would be good to cite the kan 2.0 paper

### Questions
Controlled ablations: (a) CG width/depth × PE dim grid; (b) vary K (spline order) and h; (c) remove the coefficient scale parameter. Report compute/accuracy Pareto. 


Harder benchmarks: CIFAR-10/100 with MLP-Mixer-style backbones swapped to UKAN layers; long-sequence regression (e.g., synthetic operators) versus S4/Hyena. 


Robustness & OOD: evaluate when inputs shift far outside training support—UKAN should shine here; measure error growth vs distance from training domain.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles two significant limitations of Kolmogorov-Arnold Networks (KANs): their reliance on predefined, bounded grids for B-spline activations, restricting applicability to unbounded domains or requiring data normalization, and the computational inefficiency of existing B-spline evaluation methods, hindering large-scale applications. The authors propose two main contributions: 1) Unbounded KANs (UKANs), which replace the fixed B-spline coefficient grid with a coefficient-generator (CG) MLP that dynamically produces necessary coefficients locally on an unbounded grid, enabling function approximation on unbounded domains without normalization; and 2) warpKAN, a GPU-accelerated library using NVIDIA Warp that implements B-spline evaluation with significantly reduced computational complexity (from $\mathcal{O}(Kd_g d_{in} d_{out})$ to $\mathcal{O}(Kd_{in} d_{out})$) and memory footprint by exploiting local support and basis matrix representations. Benchmarks show warpKAN achieves 3-30x speedups and up to 1000x memory reduction compared to a baseline KAN implementation. UKANs are demonstrated to match or exceed KAN accuracy on regression, classification, physics-informed approximation (PINN), and generative (DDPM) tasks. Finally, the scalability enabled by warpKAN is showcased in a large-scale molecular property prediction task using both accelerated KANs and UKANs.

### Strengths
1. The paper directly confronts and provides solutions for two major practical roadblocks for KANs: the bounded grid problem and computational inefficiency, significantly enhancing their usability and scalability.

2. The development of the warpKAN library provides a crucial practical contribution, demonstrating substantial speed and memory improvements through optimized algorithms and GPU acceleration techniques. 

3. The effectiveness of UKANs and the scalability benefits of warpKAN are validated across a diverse set of ML tasks (regression, classification, PINN, DDPM, molecular prediction), demonstrating wide applicability.

### Weaknesses
1. Introducing the CG-MLP adds complexity and parameters to the UKAN model compared to a standard KAN. The paper lacks a detailed comparison of total parameter counts, training times, and inference latency between KAN and UKAN when achieving similar performance levels on the same task.

2. UKAN introduces new hyperparameters related to the CG-MLP (architecture, embedding dimensions, positional encoding details) and the grid spacing $h$. Guidance on selecting these parameters and sensitivity analysis (especially for $h$) is limited.

3. KANs are often valued for interpretability, but the MLP is lack of interpretability. The introduction of the MLP in KAN is not good for interoretability. For example, how to ensure the accuracy of the coefficients selected by MLP?

### Questions
1. Can you elaborate on the potential generalization trade-offs introduced by the CG-MLP in UKANs compared to directly optimizing KAN coefficients?

2. How does the CG-MLP component affect the overall interpretability of the UKAN model compared to standard KANs?
﻿
3. Could you provide guidance or sensitivity analysis regarding the choice of UKAN hyperparameters, particularly the grid spacing $h$ and the CG-MLP design? Could you discuss the computational and memory costs associated with materializing UKAN coefficients for inference, especially in scenarios requiring a large range?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This article introduces two main technical innovations regarding KANs. First, they allow KANs to be applied with unbounded grids by using a neural network to encode an infinite number of spline coefficients (only finitely many of which need to be evaluated for any given input), rather than storing the B-spline coefficients themselves as parameters. Second, they provide improved GPU implementations of the spline evaluation which significantly improve computational efficiency. The authors provide numerous numerical experiments demonstrating the effectiveness of their methods.

In my opinion, the paper represents a solid empirical contribution which provides a much more efficient implementation of KANs.

P.S. The title is very creative!

### Strengths
Training KANs is an important problem. This paper makes substantial progress in developing more efficient and flexible implementations of KANs.

The numerical results look very strong. The runtimes remain nearly constant even for very large grids in Table 2, which demonstrates the parallelism of their method. The UKAN method consistently outperforms vanilla KANs for wide variety of problems, including practical drug discovery problems.

### Weaknesses
The paper doesn't contain any theoretical analysis, although this is to be expected given the empirical nature of the contribution.

I also hope the authors will release their code publicly if the paper is accepted.

### Questions
I don't quite understand equation (8). Is u supposed to be a position in $\mathbb{R}$ or a vector? It seems from the definition following the equation that u has components for each i. Is this correct? Does this mean that u is a vector? Please provide a bit more careful explanation of equation (8).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses two bottlenecks when using Kolmogorov-Arnold Networks (KANs): their computational inefficiency and their reliance on predefined, bounded grids.

The authors present two core contributions:
- A high-performance, GPU-accelerated library (using NVIDIA Warp) that re-implements B-spline evaluation.
- A new architecture that removes the need for a bounded grid. It uses a "coefficient-generator" (CG) model to dynamically generates the required B-spline coefficients on the fly.

### Strengths
The paper's strength lies in its both engineering and architecture contributions. Both the warpKAN library and the UKAN architecture  extends KAN's capabilities.

The performance benchmarks for warpKAN are not just marginal improvements; a 3-30x speedup and 1000x memory reduction represent a step-change in feasibility.

### Weaknesses
UKAN Re-introduces a "Black Box": A primary appeal of KANs over MLPs is their potential for greater interpretability (i.e., "white-box" view of learnable spline coefficients). The UKAN architecture, by using an MLP (the CG) to generate these coefficients, re-introduces a black-box component. The claim that interpretability is retained by "materializing" coefficients post-training feels like a workaround, as the reason for those coefficient values is now hidden inside the CG-MLP.

Overhead of the Coefficient-Generator (CG): The paper introduces the computational cost of the CG-MLP (O_CG in Table 1) but does not deeply analyze the practical trade-offs. For a given problem, it's unclear when the computational overhead of running this new MLP (at every forward pass for every input) becomes more expensive than simply using a larger, fixed grid enabled by warpKAN.

Not really scalable experiments: The experiments remain relatively small and rely on the same dataset used in the original KAN paper—a choice that has been repeatedly criticized by the community. To substantiate the claim of “TRAIN KANs AT SCALE”, it is essential to evaluate performance on large-scale, real-world datasets in domains such as image, audio, or language.

### Questions
See above

### Soundness
3

### Presentation
1

### Contribution
2
