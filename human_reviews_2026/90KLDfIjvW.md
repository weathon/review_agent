# Causal pieces: analysing and improving spiking neural networks piece by piece

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
We introduce *causal pieces*, a novel concept for analysing spiking neural networks (SNNs), inspired by *linear pieces* used to study expressivity and trainability in artificial neural networks (ANNs). Causal pieces partition the input and parameter space of an SNN into distinct regions where the same subnetwork causes the output spikes of the SNN. For networks of integrate-and-fire neurons with exponential synapses, we show that within each causal piece, output spike times are locally Lipschitz continuous with respect to the input spike times and network parameters. We also prove that the number of causal pieces is a measure of the approximation capabilities of SNNs. Empirically, we find that parameter initialisations yielding more causal pieces on the training set strongly correlate with SNN training success. Remarkably, even SNNs with only positive weights can exhibit a high number of causal pieces, allowing them to achieve competitive performance on diverse benchmarks such as Yin-Yang, MNIST, and EuroSAT, compared to fully-connected ANNs. These results establish causal pieces as a powerful and principled tool for analysing and improving the computational capabilities of SNNs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Causal pieces that is a method to analyze the degree of expressivity of a given SNN. This method is theoretically backed only for a particular type of neurons (IF neurons with exponential synaptic inputs, which fire once at the most) though. It is found that the number of causal pieces is a good measure of SNN expressiveness and the learning capabilities.

### Strengths
Effort on theoretical understanding of SNNs. This paper theoretically attempts to understand SNNs (time-dependent models with rich dynamics), particularly, their expressiveness and learning capability, in terms of causal pieces. The deduction from theoretical understanding well aligns with the experimental results.

### Weaknesses
Considering simple models only: the proposed theoretical analysis is for IF neurons with exponential synaptic inputs, which fire merely once at the most. For realistic neurons that fire the unlimited number of spikes, it is not very straightforward to define causal subnets and pieces since they likely vary upon time. 

Limited new findings out of theoretical understanding: The deductions from the understanding are not very surprising, like the relation between SNN expressiveness and number of causal pieces, learning capability and number of causal pieces, and the good learning performance for SNNs with positive weights only. The last one is quite clear even without this theoretical understanding since negative fan-in weights often cause dead neurons that degrade the expressiveness and learning capability. Further, the conclusion was drawn from SNNs on only toy datasets.

### Questions
Have the authors attempted to prove the theoretical deductions on datasets of high complexity?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces “causal pieces,” a way to decompose spiking neural networks into locally Lipschitz regions, provides algorithms to count them, and proves that more pieces imply greater expressivity (under stated nLIF assumptions). Empirically, it shows that initial piece count strongly correlates with training success, that depth, especially early layers, inflates piece counts, and that positive weight SNNs can attain many pieces and competitive accuracy on simple benchmarks.

### Strengths
The paper provides up to five contributions.

The figures are beautiful, the theoretical proofs are solid, and the experiments align well with the claims.

### Weaknesses
Writing of Abstract and Introduction. The abstract looks messy. It mainly lists contributions and is almost the same as the second-to-last paragraph in the Introduction. I suggest improving the writing and readability of both sections.

“We believe that causal pieces are a powerful and principled tool for improving SNNs, and may also provide new ways of comparing SNNs and ANNs in the future.” The benefits of the proposed method for comparing SNNs and ANNs are not further described in the main text, which creates a mismatch between the abstract and the main paper.

### Questions
1. The writing of Section 3.4 should be reorganized. It describes the experimental setting and results after the demonstration and the argument derived from it.

2. Figure 3. The correlation study uses only a single dataset, which is insufficient to support the conclusion: “In particular, we demonstrate in simulation that parameter initialisations which yield a high number of causal pieces on the training set strongly correlate with SNN training success.”

3. Section 3.6. The results are credible for these experimental setups (flattened inputs, simple architectures), but “competitive” performance is shown only against specific baselines, not state-of-the-art spiking systems or modern vision-task backbones.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an innovative analytical tool called "Causal Pieces" for spiking neural networks (SNNs). The paper demonstrates that the number of causal pieces is a measure of the expressive power of an SNN, and experiments show that the number of causal pieces at network initialization is highly positively correlated with the final accuracy. Furthermore, the paper finds that even SNNs with only positive weights can exhibit a large number of causal pieces and achieve good performance on benchmark tasks.

### Strengths
- Introduces and formalizes the "causal piece," a novel and natural abstraction for SNNs that partitions the input-parameter space into discrete regions, each governed by an identical causal subnetwork.
- Provides rigorous proof that the number of causal pieces serves as a quantitative measure of an SNN’s expressive capacity, directly linking this number to the network's approximation bounds.
- Empirically demonstrates a strong positive correlation between the number of causal pieces at initialization and final accuracy, offering a principled and highly valuable metric for SNN initialization.

### Weaknesses
1. **Model-Specific Limitations**: All theoretical proofs and primary experiments are based on a simplified neuron model: the nLIF, which assumes no leakage and that each neuron fires at most once (single-spike coding).

2. **Questionable Generalizability**: While the authors suggest in the discussion that the "causal piece" concept could extend to more common and complex models (e.g., leaky LIF, multi-spike coding, and recurrent SNNs), the paper provides no rigorous proof or experimental support for these claims. This generalization currently remains speculative "future work."

3. **Expressivity vs. Generalization**: The paper proves a link between the number of causal pieces and approximation ability (fitting capability), correlating it with training accuracy. However, the authors rightly concede (end of Sec 3.1) that "having many pieces does not translate into the SNN generalising well." A network with an excessive number of pieces might simply be overfitting the training data. The relationship between causal piece count and the more critical metric of generalization remains unclear.

4. **Limited Experimental Validation**: The empirical evaluation is primarily restricted to relatively simple datasets (like MNIST and Yin Yang). It lacks a systematic comparison on larger-scale, temporally complex tasks against strong, established baselines.

5. **Finding is Conceptually Intuitive**: The positive correlation between the number of "causal pieces" (as a measure of partitioning complexity) and the network's expressive power is, at a high level, conceptually intuitive. While the paper provides a valuable, SNN-specific formalization, the core finding itself is not entirely surprising.

### Questions
1. **Computational complexity**: For modern large scale networks, what are the time and memory complexities of computing causal pieces with your method?

2. **Saturation vs exponential growth**: Why does the number of pieces in deep networks appear to saturate rather than grow exponentially?

3. **Training dynamics**: During training or across different samples, how does the total number of causal pieces change? Is there a sample level correlation between prediction accuracy and the assigned causal piece regions?

4. **Generalization on more complex datasets**: Beyond training accuracy, when controlling for model size and dataset size, what is the relationship between the number of pieces and test accuracy on datasets such as CIFAR100 or ImageNet?

### Soundness
3

### Presentation
3

### Contribution
2
