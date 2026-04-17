# Fractional-Order Spiking Neural Network

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Spiking Neural Networks (SNNs) draw inspiration from biological neurons to enable brain-like computation, demonstrating effectiveness in processing temporal information with energy efficiency and biological realism.
Most existing SNNs are based on neural dynamics such as the (leaky) integrate-and-fire (IF/LIF) models, which are described by *first-order* ordinary differential equations (ODEs) with Markovian characteristics. This means the potential state at any time depends solely on its immediate past value, potentially limiting network expressiveness.
Empirical studies of real neurons, however, reveal long-range correlations and fractal dendritic structures, suggesting non-Markovian behavior better modeled by *fractional-order* ODEs.
Motivated by this, we propose a *fractional-order* spiking neural network (f-SNN) framework that strictly generalizes integer-order SNNs and captures long-term dependencies in membrane potential and spike trains via fractional dynamics, enabling richer temporal patterns. We also release an open-source toolbox to support the f-SNN framework, applicable to diverse architectures and real-world tasks.
Experimentally, fractional adaptations of established SNNs into the f-SNN framework achieve superior accuracy, comparable energy efficiency, and improved robustness to noise, underscoring the promise of f-SNNs as an effective extension of traditional SNNs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a fractional-order spiking neural network (f-SNN) framework, which replaces traditional first-order membrane potential dynamics in SNNs with fractional-order ordinary differential equations (f-ODEs). The key idea is that fractional derivatives introduce non-Markovian memory effects, allowing neurons to integrate long-term dependencies following a power-law decay rather than exponential decay. The paper provides (1) theoretical analysis of fractional memory; (2) efficient discretization; and (3) extensive experiments on neuromorphic vision and graph datasets, showing consistent improvements in accuracy, energy efficiency, and robustness.

### Strengths
1. The introduction of fractional dynamics into SNNs is novel and biologically inspired. It directly addresses the Markovian limitation of conventional LIF/IF neurons, linking to real neuronal behaviors and fractal dendritic structures.

2. The derivation from Caputo derivatives to the discrete f-LIF formulation is rigorous. Theorem 1 and the comparison of exponential vs. power-law decay are clearly presented and intuitive.
 
3. The f-SNN framework is a strict superset of conventional SNNs (α = 1), making it easily integrable into existing architectures and toolkits.
 
4. Results across multiple domains (neuromorphic datasets and graph benchmarks) show consistent gains. The robustness tests (noise, occlusion, temporal jitter, etc.) are comprehensive and convincing.

### Weaknesses
1. The theory mainly discusses long-term dependence qualitatively via the Mittag–Leffler function; deeper analyses such as stability, gradient dynamics, or expressive power would strengthen the theoretical contribution.

2. Although the paper mentions O(NlogN) and truncated-memory approximations, there is no clear empirical benchmark on training/inference speed or memory consumption compared to standard SNNs.

3. Lack of ablation on fractional order $\alpha$. While $\alpha$ is a key hyperparameter, the paper does not analyze its sensitivity or interpretability across tasks.

4. Some sections (e.g., Sec. 3.1 discretization) could benefit from clearer notation and step-by-step explanation. A schematic showing the memory kernel evolution over time would help readers intuitively grasp the effect of $\alpha$.

### Questions
1. How scalable is f-SNN to large datasets (e.g., DVS-Gesture full size) in wall-clock time and memory usage?

2. Have the authors tried integrating f-LIF with surrogate gradient training, jointly optimizing α and synaptic weights?

3. Can the same principle apply to recurrent  SNN architectures?

4. Why do SpikingJelly and snnTorch produce such different results under the same dataset and network architecture?

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
4

### Summary
This paper proposes a fractional-order spiking neural network (f-SNN) framework that replaces traditional integer-order ODEs with fractional-order ODEs to capture long-term dependencies in biological neurons, outperforming conventional SNNs in accuracy, energy efficiency, and noise robustness across neuromorphic vision and graph tasks while providing an open-source toolbox spikeDE for easy adoption.

### Strengths
1. The proposed fractional-order spiking neural network (f-SNN) framework effectively captures long-term dependencies in membrane potential and spike trains via fractional dynamics, addressing the limitation of traditional integer-order SNNs (based on IF/LIF models) that only rely on immediate past states.

2. The f-SNN demonstrates superior performance in multiple aspects: it achieves higher accuracy than conventional SNNs on neuromorphic vision

### Weaknesses
1. The concept of fractional-order has already been proposed by others; the authors seem to have only put forward a framework and tested it on different datasets, which is more like an engineering problem.

2. The datasets used are limited, and results on widely tested datasets such as CIFAR-10, CIFAR-100, and ImageNet are missing.

### Questions
The energy consumption analysis of SNNs only includes synaptic analysis, while the energy consumption of the neurons themselves should also be compared.

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
2

### Summary
This paper proposes fractional-order spiking neural networks (s-SNN). The idea is to replace the linear ODE that describes the pre-spike potential dynamics of a (L)IF model with a fractional ODE. This allows for better capturing of long-term dependencies in the membrane potential evolution, enhancing the expressivity of the corresponding models. In particular, this paper shows how f-SNNs can be seen as a generalization of the (L)-IF SNNs and how to effectively discretize f-SNNs to get an implementable model using the Adams-Bashforth-Moulton discretization scheme. Finally, the author empirically demonstrates that f-SNNs consistently outperform traditional SNNs in both performance and robustness to input perturbations across multiple tasks and architectures. The authors further developed an open-source toolbox compatible with PyTorch, simplifying the design and implementation of custom SNN neurons.

### Strengths
The paper is well-written, precise, and enjoyable to read. The definition of the continuous-time f-SNN and the derivation of the discrete-time counterpart are well-explained. Also, the experimental setup is clear and well-documented, and the results are consistent with the claims.

### Weaknesses
The paper contains some minor imprecision that can be easily fixed. For instance, in equation (1) line 126, there is an a which I suppose is meant to be a 0. Moreover, in line 172, the authors mentioned "learnable synaptic weights" that are not introduced since in line 170, only the input current is mentioned $I_{in, k}$. The caption of Figure 4, page. 7, could be a little bit extended by explaining how to interpret Figure 4 (at least to me, it is not straightforward). 

I would appreciate seeing a related work section in the main part and a more detailed comparison with the existing models that exploit fractional derivatives. The authors mention them Appendix A2, but it is not clear how it compares with your model. This would be essential for the novelty assessment and the main reason for the overall score.

The theoretical result (Theorem 1 in Section 3.2) seems somewhat modest to be presented as a theorem, as it addresses only the constant input current case and relies on known solutions of the corresponding ODE and fractional differential equation. It might be more appropriate to present it as a remark.

### Questions
I ask the following questions to the authors:

1) In the article "Time to Spike? Understanding the Representational Power of Spiking Neural Networks in Discrete Time" the authors study the expressivity of the LIF SNN model both in terms of approximation properties and in terms of linear regions in the static case. Would you expect the f-SNNs neuron to produce more regions in the input space? Would you expect that f-SNNs neuron regions are in more general positions?

2) Combining the fact that $\alpha$ is selected through a hyperparameter tuning and that training SNNs in general is hard, what can you say about the overall efficiency of the training of an f-SNN model?

### Soundness
3

### Presentation
3

### Contribution
2
