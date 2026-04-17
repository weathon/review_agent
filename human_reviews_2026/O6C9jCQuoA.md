# Neural Darwinism: A Theoretical Framework for Representation Evolution in Convolutional Neural Networks

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
We introduce a Darwinian framework that mathematically formalizes representation evolution in deep networks, viewing each neuron as an adaptive entity competing for survival during training. In this perspective, learning is governed by a unified Darwinian Score that reflects three essential dimensions of neuronal fitness—informational diversity, functional contribution, and temporal adaptability. This score induces a principled constrained optimization objective that balances model compactness with predictive fidelity, supported by new approximation guarantees showing that preserving high-fitness neurons retains the network’s functional capacity. We then operationalize this framework through Neural Darwinism Culling (NDC), which serves as a practical instantiation of the Darwinian Score. NDC dynamically removes neurons with persistently low fitness while allowing high-value neurons to specialize. NDC captures the intrinsic evolutionary dynamics of neural representations: neurons with collapsed activations, negligible causal impact on loss reduction, or stagnant parameter trajectories are pruned, whereas differentiated and adaptable neurons are retained. This yields pruning decisions that are interpretable, layer-aware, and aligned with the competitive pressures naturally emerging across network depth. Experiments across diverse methodological settings demonstrate that NDC, as a direct application of the Darwinian Score, achieves substantially higher sparsity with improved generalization compared to SOTA methods, particularly under extreme compression. Ablations further confirm that the Darwinian Score is the key driver of these gains. Overall, our work provides both a general evolutionary lens for understanding representation dynamics and a practical, theory-grounded path toward efficient and adaptive deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper propose to conduct neural network pruning by using the proposed the Darwinian score, which is designed to measure the information richness, functional relevance, and adaptive dynamics of each neuron.

### Strengths
None.

### Weaknesses
1. The format of citation is wrong, which is hard to read.

2. There are some strange statements, such as ' Because information-theoretic quantities appearing below are sensitive to .....', which is hard to understand.

3. For the evolution-based pruning method, there should be a discussion about 'Towards evolutionary compression. KDD 2018.', which also implements network pruning with evolutionary algorithm.

4. The calculation of the histogram/probability is problematic, how to cover the output range of activations such as ReLU? Which is impossible.

5. For the Darwinian Entropy, the diversity of the probability does not equals to the diversity of the information, especially when there is no alignment between the variables. 

6. The compared methods are old (works from 2020 and 2022), which is hard to verify the effectiveness of the proposed method. Moreover, the reference for Cropit should be included.

### Questions
Please refer to the Weaknesses.

### Soundness
1

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
2

### Summary
This paper introduces Darwinian framework that provides a mathematical foundation for
neural representation dynamics, conceptualizing deep learning as an evolutionary
process governed by selection and adaptation. This notion is formalized by defining a Darwinian Score (DS) that quantifies a neuron’s evolutionary fitness through three measurable components:

Neuron Darwinian Entropy (NDE): information diversity and non-redundancy of activations;

Activation–Gradient Contribution (AGC): functional contribution to loss minimization;

Neuron Adaptivity Score (NAS): the degree of adaptive evolution of neuron parameters over training.

These measures are combined multiplicatively to enforce a “survival-of-the-fittest” criterion within neural architectures. Based on this theory, the authors propose the Neural Darwinism Culling (NDC) algorithm, which prunes neurons dynamically during training according to their Darwinian Score. Theoretical results guarantee bounded approximation error after pruning, and experiments on CIFAR-10 and Tiny-ImageNet demonstrate that NDC achieves superior accuracy and sparsity trade-offs compared to state-of-the-art pruning methods (SNIP, SNAP, CroPit, DPF, etc.).

### Strengths
Concept and biological inspiration - The idea of using Darwinian evolutionary process in deep learning is interesting and novel.
Theory - The algorithm is supported by theorems and proofs.
Unified view of pruning and adaptation- By tying representational diversity, functional importance, and adaptability into a single fitness score, the authors unify multiple pruning philosophies (magnitude-, sensitivity-, and gradient-based) under one framework.
Experimental results - many experiments were conducted to support the claim.

### Weaknesses
Theory - The main theorem assumes a long list of many assumptions that are not so natural or trivial to me. 
Proofs - The proofs are quite straightforward and follow from importance sampling, without too many new observations.
Experiments - The experiments are on small data sets and CNN type of networks, but I could not see a more modern results on transformers and larger datasets. In general, it is not clear how the idea is scalable, especially when we need to maintain all these scores along the training phase. 
Relation to existing evolutionary optimization – The connection to neuroevolution and evolutionary strategies could be elaborated further, particularly distinguishing this work from gradient-free evolutionary algorithms. Also, evolutionary algorithms are not considered a successful or common tool, both in the industry or academy.

### Questions
Please explain in more details the quantitative analysis of the relative impact of NDE, AGC, and NAS components.

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
2

### Summary
In this work, a novel pruning framework, called Neural Darwinism Culling (NDC), is proposed. The key idea is that neurons are assumed to behave like competing agents, with which the authors derive the 'Darwinian score', that combines (a) an entropy/divergence–based activation statistic (NDE), (b) an activation–gradient coupling (AGC), and (c) a weight-trajectory “adaptivity” score (NAS). Theoretical guarantees for the efficacy of the proposed method are also provided. Last, empirical studies are conducted on various standard image processing tasks on standard CNN architectures, where the NDC method is shown to match or outperform near baselines.

### Strengths
* The Darwinism score, combining change in weights, information diversity (NDE, Def 3.2), and functional utility (based on the gradient score, AGC, in Def. 3.3) is useful and captures significant information about relevant neurons for pruning/retaining.
* NDC appears to outperform several near baselines, with models pruned with it retaining higher accuracy across all sparsity levels.
* The mathematical derivations appear to be correct (with a few caveats regarding assumptions, see Questions), and adds guarantees that motivates hte use of this method. 
* The writing of the paper is generally quite clear and easy to follow, though there is an issue with the Definitions in section 3 (see 'Weaknesses').

### Weaknesses
* The definitions proposed in Section 3 are overly long. For instance, in Definition 3.1 (NDE), the equation defining the NDE score is stated 18 lines after the definition begins. The setup should be kept outside the definition, which in turn should be crisp and self-contained.
* The experimental slate should contain ImageNet experiments, which by now are standard in the pruning literature, instead of just TinyImageNet.
* Theorem 3.6 states a worst-case change in prediction (over samples), while an Expectation bound would be more useful.

### Questions
* How does the proposed method perform when the number of epochs is increased/decreased?
* Have the authors considered incorporating the NDC method while training a model from scratch? That is, given a few initial training epochs, NDC would then be applied during training, thereby obtaining a sparse model from scratch. Is there any reason to think this is not possible?

### Soundness
3

### Presentation
2

### Contribution
2
