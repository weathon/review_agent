# Beyond Linear Processing: Dendritic Bilinear Integration in Spiking Neural Networks

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
As widely used neuron model in Spiking Neural Networks (SNNs), the Leaky Integrate-and-Fire (LIF) model assumes the linear summation of injected currents. However, recent studies have revealed that a biological neuron can integrate inputs nonlinearly and perform computations such as XOR while an LIF neuron cannot. To bridge this gap, we propose the Dendritic LIF (DLIF) model, which incorporates a bilinear dendritic integration rule derived from neurophysiological experiments. At the single-neuron level, we theoretically demonstrate that a DLIF neuron can capture input correlations, enabling it to perform nonlinear classification tasks. At the network level, we prove that DLIF neurons can preserve and propagate correlation structures from the input layer to the readout layer. These theoretical findings are further confirmed by our numerical experiments. Extensive experiments across diverse architectures—including ResNet, VGG, and Transformer—demonstrate that DLIF achieves state-of-the-art performance on static (CIFAR-10/100, ImageNet) and neuromorphic (DVS-Gesture, DVS-CIFAR10) benchmarks, surpassing LIF and other advanced alternatives while maintaining comparable computational cost. This work provides a biologically plausible and computationally powerful spiking neuron model, paving the way for next-generation brain-inspired computing.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces the DLIF neuron—a variant of the Leaky-Integrate-and-Fire model that implements dendritic bilinear integration. Compared with the standard LIF, DLIF adds a bilinear dendritic term that can be evaluated with minimal overhead, requiring only AND operations. The authors theoretically prove that a single DLIF unit can discriminate between spike trains with identical mean firing rates but different correlation structures, and that a two-layer DLIF-based SNN preserves input correlation structures more effectively than its LIF counterpart. Extensive experiments on multiple datasets and across many existing architectures or training methods show consistent accuracy gains when LIF neurons are replaced by DLIF.

### Strengths
- Theoretical analysis rigorously supports the advantages of DLIF.
- Exhaustive empirical evaluation: DLIF consistently improves performance when plugged into a wide range of models.

### Weaknesses
- DLIF appears difficult to realize at the neuron level alone, because it requires storing the presynaptic spike vectors from the previous layer rather than just the aggregated synaptic current. Implementing this seemingly demands stateful synapses, potentially increasing both training time and memory usage.
- The proposed method was evaluated on multiple datasets by replacing the LIF neurons in prior works with DLIF neurons. Tables 1 and 2 show consistent and stable improvements, a result that is highly encouraging and suggests DLIF could universally outperform LIF and become a new standard neuron model for SNNs. However, the absence of released source code and experiment logs reduces credibility and prevents readers from examining implementation details.

### Questions
The authors claim that DLIF neurons incur “no significant increase in computational cost”; however, in my experience introducing stateful synapses into an SNN usually raises training expenses dramatically. Could the authors supply a direct comparison of training/inference time and memory consumption between DLIF and vanilla LIF neurons?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a DLIF neuron that augments the input current with a bilinear dendritic term ($s^\top K s$), argues theoretically that DLIF captures pairwise input correlations (single-neuron) and preserves/propagates correlation structure (network-level), and reports consistent gains across CIFAR-10/100, ImageNet, DVS-Gesture, DVS-CIFAR10, and Atari RL with small overheads. The idea is timely and simple to integrate into existing SNN stacks. Results are encouraging, but the claims about complexity/energy, the parameterization and sparsity of ($K$), and the scope/limits of the theory need clearer, checkable evidence.

### Strengths
- DLIF replaces linear current summation with an additive bilinear term (s^\top K s); formulation is simple, but performance is good. 
- Static and neuromorphic vision, multiple architectures (VGG/ResNet/Transformer), and training regimes (SLTT, OTTT, TET, STBP-tdBN, etc.) show consistent improvements with small reported overheads.

### Weaknesses
1. The text states ($K$) reflects spatial relationships and is intensity-independent, yet in networks it is learned. Are there structured priors (locality, block-diagonal, cross-channel only) or sharing tied to receptive fields? A comparison of fully learnable ($K$) vs. structured/low-rank/near-neighbor ($K$) would clarify the interpretability–performance–cost trade-off. 

2. Since $K \in \mathbb{R}^{N\times I \times I}$ (although sparse), computational cost, parameter accounting, and training speed need to be better compared.

### Questions
1. How about replace sparse matrix to low-rank matrix. It can better reduce parameter number and computational cost.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes the Dendritic LIF (DLIF) model that extends conventional LIF by incorporating bilinear dendritic integration. Through theoretical analysis and empirical simulations, the authors show that a single DLIF neuron can effectively capture correlations in input spike trains. Experiments across diverse deep SNN architectures and datasets demonstrate that DLIF can achieve SOTA performance, highlighting its scalability.

### Strengths
1.Introducing dendritic computation to large-scale neuromorphic computing is an advanced and compelling research topic.

2.The paper is clearly structured (from single neuron to large-scale SNNs) and is generally well written. The mathematical derivations are rigorous. The figures and tables effectively convey the key ideas and results.

3.Experiments cover a wide range of tasks and SNN architectures. DLIF consistently outperforms LIF on (almost) all the tasks, showcasing the method’s effectiveness and generalizability.

### Weaknesses
1.The novelty of the proposed model is not clearly stated. What are the key differences or innovations of DLIF with the model proposed by (Li et al., 2019)? 

2.Although comparisons are made with point neuron models like PLIF, GLIF, QIF and EIF (Table 3), no comparison is made with dendritic or multi-compartment spiking neurons like (Zheng et al., 2024). The authors claim that DLIF is inspired by dendritic computation, so comparing it with other dendritic neuron models would strengthen the work’s soundness.

3.While inference cost metrics like FLOPS and theoretical energy consumption are provided, there is no information regarding training cost. How much extra training time and memory overhead does the bilinear operation introduce? A comparison with LIF or other spiking neuron models should be made.

Li, Songting, et al. "Dendritic computations captured by an effective point neuron model." Proceedings of the National Academy of Sciences 116.30 (2019): 15244-15252.

Zheng, Hanle, et al. "Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics." Nature Communications 15.1 (2024): 277.

### Questions
See “Weakness” for major questions. Below are some additional minor questions and suggestions:

1.More details should be provided regarding the computation of theoretical energy cost. The explanation in Appendix cannot address my key concern: how is the energy consumption of a bilinear operation calculated?

2.The parameter counts in Table 1 and 2 are weird for me. For a synapse layer with $N$ presynaptic neurons and $M$ postsynaptic neurons, the linear weight is a $M\times N$ matrix, while the bilinear weight is a $M\times N \times N$ tensor, as stated in Line 192 of the manuscript. Even with a 90% sparsity constraint and an all-zero diagonal, there should still be about $0.1\times M \times (N^2-N)$ more learnable parameters, which is much larger than $M \times N$. However, the reported parameter increase is only about 10%. Please clarify in detail how the parameter counts are computed.

3.The authors report the results of DLIF networks trained with OTTT. However, OTTT is an eligibility-trace-based online learning method designed for optimizing $\mathbf{W}$, the linear weights. Its extension to bilinear weights is not trivial. Please explain how to use OTTT to optimize bilinear weights.

4.Typos: “biliear” in Related Work should be bilinear; “as a binary” in Page 3, Line 159 should be “as a binary matrix”.

5.Citation formatting: please use `\citep{}` instead of `(\cite{})`. The displayed text should appear as (xxx et al., 20xx) instead of (xxx et al., (20xx)).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study proposed a new spiking neuron model called DLIF, which can conduct dendritic computations.  Experimental results show that this neural model achieves SOTA performance on five computer vision datasets

### Strengths
1. The proposed neuron model is biologically-inspired and the single neuron theorem is solid.
2. The experiments on XOR verified the theory.
3. Results are validated on several datasets

### Weaknesses
1. The energy metrics are estimated using numbers from a paper from a decade ago. Further justifications should be provided, and the energy benchmarking setting should be clearly presented, e.g., which GPUs are the target, how SNNs are set up and run on such GPUs, and how many GPU it required.
2. The paper compared many SNN algorithms and reproduced their results. It can be beneficial to provide codes (Assuming the ability to replace any neuronal models on any SNN algorithm is one of the main contributions of this paper), instead of just listing the experimental settings in the Supplementary materials for each SNN algorithm. It can make other researchers much easier to follow this study and develop a new spiking neuronal model on top of this study.

### Questions
1. Section 4 "Specifically, we set the sparsity level to 90%, a
choice that is both biologically inspired and empirically validated by an ablation study, as shown in
Section 4.4.". Please explain which dimension the sparsity is applied and how it is controlled before settingthe  sparsity level.

2. Is there any open-source plan for this study?

### Soundness
2

### Presentation
3

### Contribution
2
