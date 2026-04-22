# A Biologically Plausible Dense Associative Memory with Exponential Capacity

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 6, 6

## Abstract
Krotov and Hopfield (2021) proposed a biologically plausible two-layer associative memory network with memory storage capacity exponential in the number of visible neurons. However, the capacity was only linear in the number of hidden neurons. This limitation arose from the choice of nonlinearity between the visible and hidden units, which enforced winner-take-all dynamics in the hidden layer, thereby restricting each hidden unit to encode only a single memory. We overcome this limitation by introducing a novel associative memory network with a threshold nonlinearity that enables distributed representations. In contrast to winner-take-all dynamics, where each hidden neuron is tied to an entire memory, our network allows hidden neurons to encode basic components shared across many memories. Consequently, complex patterns are represented through combinations of hidden neurons. These representations reduce redundancy and allow many correlated memories to be stored compositionally. Thus, we achieve much higher capacity: exponential in the number of hidden units, provided the number of visible units is sufficiently large relative to the number of hidden units. Exponential capacity arises because all binary states of the hidden units can become stable memory patterns. Moreover, the distributed hidden representation, which has much lower dimensionality than the visible layer, preserves class-discriminative structure, supporting efficient nonlinear decoding. These results establish a new regime for associative memory, enabling high-capacity, robust, and scalable architectures consistent with biological constraints.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a 2-layer associative memory network that extends Krotov and Hopfield (2021). The key innovation is a replacement of the softmax/WTA hidden layer nonlinearity with a Heaviside step function, which effectively enables multiple hidden neurons to represent a memory instead of forming grandmother cells. The paper performed theoretical study of the capacity of the network and found that the network can achieve $2^{N_h}$ capacity with certain assumptions (e.g., $N_v >> N_h$), and it also studied the size of the attractor basin.

### Strengths
- It proposes a simple yet effective fix to a known limit of Krotov and Hopfield 2021; theoretical proof and experiment analysis demonstrates the effectiveness of this proposal
- The theoretical analysis is correct, thorough and clearly presented

### Weaknesses
Conceptually, I feel this paper has the following weaknesses:
- The idea, though effective, is incremental to Krotov and Hopfield 2021. This is exacerbated by the fact that Krotov and Hopfield 2021 already discussed different possibilities of hidden/sensory nonlinearities so this paper does not serve as a generalization.
- Despite the point about, there is still opportunity for this idea to be influential if the authors could extend it along the lines of biological plausibility (e.g., what if the weights are not symmetric?) and ML applications (e.g., how can a transformer benefit from the step function nonlinearity if it can be plugged into attention?). In other words, the paper can go 'wide' as it is incremental from a 'depth' point of view
- The claimed biological plausibility is a direct inheritance of the biological plausible implmentation of Krotov and Hopfield 2021; no novel plausible implementation is introduced. The theoretically optimal $\theta=0.5$ may even be too strict as a biological constraint. Therefore I suggest the authors not to sell this point too much (for example in the title).

In addition, I also feel this paper lacks analytical and experimental depth:
- It lacks a direct comparison to other nonlinearities (e.g., models in Krotov and Hopfield 2021) - I understand theoretically it should have a bigger capacity, but a complete result section needs explicit experimental results to support the theory
- Many experimental setting violates assumptions in theory. For example, in experiments, the $\Theta$ function is a smoothed step-function; in both MNIST and CIFAR experiments $N_v$ is not much bigger than $N_h$. 
- The capacity results do not seem consistent with the theory: for example, 55k unique minima is far less than $2^{N_h}=2^50$. This makes me wonder the importance of the violated theoretical assumptions above.

### Questions
- After training on MNIST/CIFAR10, how many distinct stable attractors exist beyond the training set, and how close is that to the theoretical $2^{N_h}$? Can't you experiment with a smaller $N_h$ e.g., 16, so the theoretical capacity is closer to the training set size?
- How sensitive is stability / capacity to smoothing the Heaviside (which you actually use in training) and to adding noise / heterogeneity in thresholds?
- How crucial is $\tau_h << \tau_v$? Can you show recall robustness curves as a function of this ratio?
- Can you concretely map your hidden code / retrieval dynamics onto a transformer attention head or a feedforward block, and/or show a toy transformer experiment?
- Can you tone down / clarify the “biological plausibility” claim, or alternatively provide evidence that asymmetric weights or mild threshold heterogeneity don’t affect capacity?

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
3

### Summary
This work assess a biologically plausible two-layer associative memory network that overcomes the linear capacity constraint of previous models. The core innovation is replacing the winner-takes-all nonlinearity with a threshold activation function, which enables distributed representations where single hidden neurons encode components shared across many memories. Consequently, the network achieves an exponential storage capacity of $2^{N_h}$, supports compositional storage of complex patterns, and demonstrates robust recall from noise on datasets like MNIST and CIFAR-10.

### Strengths
1. The network is shown to have exponentially higher capacity than the prior two-layer implementations.
2. The model's fixed points possess large basins of attraction, making the memory retrieval process robust to substantial noise in the visible inputs.

### Weaknesses
1. While I certainly agree pairwise interactions between neurons is biological, I do not agree with the implication that setwise interactions are not. I think the claim of this point should be more nuanced, e.g., see appendix A.2 of Burns & Fukai (2023).
2. The theoretical exponential capacity relies on the condition that the number of visible units ($N_v$) is much larger than the number of hidden units ($N_h$), i.e., $N_v \gg N_h$ (see L188-193). This does not seem obviously "biological". It would help if the authors could reference some related neurobiological literature on this point.

### Questions
1. The theoretical capacity holds strongest in the limit $N_v \gg N_h$, but the authors achieved successful recall on CIFAR-10 with a ratio of $N_v/N_h \approx 2$ ($1024$ visible vs. $500$ hidden units) (see L346-355). What is the rigorous theoretical lower bound on the ratio $N_v/N_h$ that is necessary to maintain near-exponential capacity and robust recall, especially when dealing with real-world, highly correlated memories?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Associative memory models often struggle with a trade-off between biological plausibility and storage capacity. Previous biologically plausible two-layer models, such as Krotov and Hopfield (2021), were limited by winner-takes-all dynamics, resulting in a storage capacity that scaled only linearly with the number of hidden neurons.
This paper proposes a modification to this two-layer architecture: replacing the winner-takes-all nonlinearity with a simple threshold Heaviside activation function. The authors claim this allows for distributed, compositional representations in the hidden layer, rather than localized representations that tie memory to neurons.
The primary contribution is the theoretical and empirical demonstration that this network achieves an exponential storage capacity, $2^{N_h}$, in the number of hidden neurons ($N_h$), provided the number of visible neurons ($N_v$) is much larger than the hidden neurons ($N_v \gg N_h$).

### Strengths
The paper's main idea is simple and effective. It directly addresses a key limitation of the Krotov and Hopfield (2021) model—linear capacity—by changing the activation function. It also provides a simpler alternative to other recent work that required combining multiple modules to achieve exponential capacity.

The theoretical analysis is also nice. The paper provides a clear derivation for how the $N_v \gg N_h$ regime leads to a decoupled hidden layer where all $2^{N_h}$ binary states can be stable fixed points. This theory is backed by numerical experiments. The stability of these points is also analyzed. The experiments on MNIST and CIFAR-10 show that the model can store tens of thousands of highly correlated, real-world patterns. The assumptions made are mostly in line with derivations in related literature and so are the experiments.

The paper is written clearly. It identifies the limitations of prior work and effectively explains how the proposed mechanism overcomes them.

### Weaknesses
In the CIFAR-10 experiment, the assumption used to derive the theories, $N_v \gg N_h$, seems to not to be the case as $N_v=1024, N_h=500$, it would be nice to discuss more about the implications of this.

Although not a fault of the paper per se but rather symptomatic of the computational neuroscience literature at large, biological plausibility is a loaded expression and can mean very different things. Also, although the architecture (pairwise synapses, two layers) avoids the implausible interactions of other models , the learning rule is a standard gradient-based optimization of a global cost function, which, despite various research on the biological basis of backprop, remains unclear whether it is truly biologically plausible.

### Questions
1, In your classification results table, the nonlinear classifier on original MNIST images achieves 100% accuracy, but on the recalled visible patterns, it only achieves 91%, can you briefly explain this performance drop?

2, What values did the learned threshold $\theta$ converge to for the MNIST and CIFAR-10 experiments?

3, Does the CIFAR-10 results imply the $N_v \gg N_h$ condition is sufficient but not necessary, or is a different mechanism (perhaps from the learning rule) responsible for the high capacity in this regime?

### Soundness
4

### Presentation
4

### Contribution
3
