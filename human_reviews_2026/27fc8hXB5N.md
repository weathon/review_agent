# Geometric Compression in Grokking: The Three-Stage Modular Dynamics of Transformers

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 2, 2

## Abstract
A central puzzle in deep learning is how generalized algorithms emerge from training dynamics, particularly in the phenomenon of grokking. Existing approaches track function complexity (Linear Mapping Number) or representation dimensionality (Local Intrinsic Dimension). We take a different perspective: a unified algorithm should manifest as geometrically consistent transformations across inputs. We introduce the \textbf{Geometric Coherence Score (GCS)}, which measures the directional alignment of local Jacobian transformations across the data manifold. GCS provides a geometric signature of mechanistic unity—consistent transformations indicate a unified computational strategy, while scattered transformations suggest input-specific memorization. Combined with a fixed final geometry protocol that isolates mechanistic evolution from geometric drift, GCS reveals a \textbf{Construct-then-Compress} dynamic invisible to complexity or dimensionality metrics. In single-layer Transformers, this dynamic unfolds in three distinct phases: (1) \textit{Coherence Collapse}, where initial symmetry breaks to memorize data; (2) \textit{Asynchronous Construction and Compression}, a critical silent phase where Attention initiates geometric reorganization, followed by MLP with temporal offset; and (3) \textit{Post-Grokking Refinement}, where the mechanism consolidates into a unified solution. We validate the construct-then-compress principle across activation functions (ReLU, GeLU, SiLU) and modular tasks (addition, subtraction, multiplication, division), establishing GCS as a principled diagnostic tool. Extending to multi-layer networks (2--3 layers), we observe that final layers exhibit iterative construct-compress cycles rather than a single three-phase trajectory, while early layers show path-specific stability. These findings reveal depth-dependent dynamics that warrant further investigation into how hierarchical structure shapes algorithmic formation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work offers a mechanistic perspective on grokking and introduces a principled geometric framework to analyze the dynamic interplay of complexities that drive learning in neural networks.

### Strengths
The proposed method is novel and elegant, offering a new way to study neural networks.

- geometric coherence is a new way for understanding grokking dynamics, going beyond static complexity metrics
- modular method allowing to study separately the attention and the feed-forward layer with the same method
- clear experimental setup

### Weaknesses
- missing scalability discussion for larger networks

- the experiments are currently confined to a single, clean algorithmic task (modular addition). While this is a standard and valid testbed for grokking, the paper would be significantly strengthened by a demonstration on another task (e.g., a different modular operation or a simple symbolic regression task) to show the generality of the dynamic beyond a single function.

### Questions
- Does the three-stage dynamic appear for other grokking tasks? This is critical for claiming a "universal" mechanism.

- Liu et al. 2024 also measured compression during grokking. How do GCS and LMN compare? Do they capture different aspects or the same phenomenon?

- You show correlation between GCS and generalization, but is there evidence that GCS causes or predicts grokking before it happens?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a novel metric for analyzing the way in which a network changes the local structure of the representational space (which they assume is captured by a low-dimensional manifold) using a metric ("Geometric Coherence Score") that measures how strongly the model locally compresses this space. They then use this score to measure changes in GCS in Transformers trained on modular addition (a task on which Transformers show grokking behavior). They connect the grokking behavior to three distinct phases indicated by different behaviors of the GCS.

### Strengths
- The paper is generally well-written.
- Understanding the representational dynamics of Transformers over training is an important challenge and introducing new tools for this purpose can be very valuable.
- The expand-then-compress mechanism the authors identify in the grokking Transformer is interesting.

### Weaknesses
1. Primarily, I think the paper currently insufficiently shows why the phenomena the authors measure requires the introduction of a novel method. It seems to me that the phenomenon of the expansion and compression could also be captured by general measures of dimensionality (such as the participation ratio). While I think the introduction of novel measures is certainly valuable, I think the paper would benefit substantially from demonstrating why this novel measure is useful and provides insight beyond existing measures.

2. Further, I had trouble understanding how your metric is defined. You characterize the local geometry by using the k nearest neighbors of each data point $x_i$. That makes me think that the resulting singular vectors $v_{i,k}$ are separately defined for each data point. But then in constructing $G$, you're again considering a k-NN neighborhood? Is this one different from the first one? Moreover, in equation 3, you're measuring the coherence between those neighborhoods by measuring the correlation between the different pairs of the transformed singular vectors $v_{i,k}'$. Doesn't that make the assumption that the singular vectors $v_{i,k},v_{j,k}$ are aligned? E.g. if two data points' neighborhoods have literally the same mapping $f$ and the same singular vectors, but in a different order, wouldn't $G_{ij}=0$? If that's true, that seems problematic. Am I missing something?

3. Relatedly, I also think it would be important to provide an intuition for what each step is doing and why the equations are defined as they are defined. E.g. it would be great to explain what equation (3) is measuring --- my understanding is that similar mappings should have similar transformed singular vectors $v'$ and so by measuring the average correlation between those vectors, we measure how similarly those mappings are in different parts of the space. Is that correct? One suggestion for how you could provide this intuition would be to give a couple of different examples of different local geometries in the input and different mappings and explain how that affects the overall mapping. E.g. what happens when the local geometries in the input are very different but the mappings are very similar? What happens in the contrary case?

4. Finally, I think it would be helpful to discuss how this metric relates to established methods in the field. E.g. have people previously used entropy to characterize coherence in this kind of context?

In its current form, these concerns prevent me from recommending acceptance. However, I think that the authors address an important problem with a potentially promising approach. I am looking forward to the rebuttal and am certainly open to improving my score.

### Questions
- Could you explain whether my understanding in Weaknesses, paragraph 2, is correct? If so, why is the situation I'm describing not problematic?
- Would other measures of representational geometry (e.g. the participation ratio) also pick up on the transitions in network training you're identifying?
- Why is setting $G_{ij}=0$ for non-neighboring pairs reasonable? Couldn't the tangents still be highly correlated?
- Can you elaborate on the "competing circuits" hypothesis and how it relates to your findings?
- Can't you see the initial stages of Phase II at the end of Figure 4? You can see the attention path GCS decreasing and the MLP Path GCS increasing.
- Figure 5 is really difficult to see right now due to its size. Further, is it possible to provide a more quantitative measure for how the attention scores at these different phass are different? While I can see certain visual differences, it is difficult to get a sense of what they mean in particular comparing "peak state" and "valley state". You're also introducing this terminology for the first time here, I would suggest keeping your terminology consistent with the one you're using earlier.

**Minor suggestions**
- It might be useful to visually indicate the epoch where grokking starts setting in in the flow scores in Fig. 3
- It would be helpful to visually indicate the phases in Fig. 2

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Introduces a new method for measuring the dynamics of grokking called the Geometric Coherence Score (GCS) and applies it to modular addition, a well-studied task exhibiting grokking. Training 1-layer transformers and tracking GCS the authors describe a three-stage construct-then-compress dynamic. GCS also appears to distinguish between overfitting and generalization in this particular setting.

### Strengths
1. Potentially novel perspective on the dynamics of grokking.
2. The use of multiple activation functions.

### Weaknesses
**Some missing citations**:

*Lazy-rich training dynamics* (mentioned on lines 315, 431, 465):

[1] "Grokking as the Transition from Lazy to Rich Training Dynamics" Kumar et al. ICLR 2024

[2] "Feature Learning beyond the Lazy-Rich Dichotomy: Insights from Representational Geometry" Chou et al. ICML 2025

*Connection to double descent* (line 465):

[3] "Unifying Grokking and Double Descent" Davies et al. 2023

[4] "Unified View of Grokking, Double Descent and Emergent Abilities: A Perspective from Circuits Competition" Huang et al. COLM 2024

*GELU*:

[5] "Gaussian Error Linear Units (GELUs)" Hendrycks et al. 2016

*Related complexity measures*:

[6] "Deep Networks Always Grok and Here is Why"  Imtiaz Humayun et al. ICML 2024

*Previous works studying modular addition, features/representations*:

[7] "Grokking modular arithmetic" Gromov. 2023.

[8] "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks" Zhong et al. NeurIPS 2023.

[9] "Feature emergence via margin maximization: case studies in algebraic tasks" Morwani et al. ICLR 2024

[10] "Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks" McCracken et al. NeurIPS 2025


**Narrow experimental scope**: only study one architecture that is a single layer. This is problematic because large changes in what's learned have been observed by different papers studying different experimental settings and/or architectures in modular addition. It's worth noting this paper doesn't cite any of these works, being: [8], found that uniform attention models learn ``pizza circuits'' instead of the clock circuits described by Nanda. [9], found that 1 layer networks learn all n-1/2 frequencies (mod n), importantly explained that the generalizing features emerged due to margin maximization and tracked it throughout training. [10], empirically found that 2, 3, 4 layer networks learn O(log(n)) frequencies and proved it. It remains an open problem [9, 10] why there's such a stark difference in the number of features that emerge between 1-layer vs multilayer networks, and it's conjectured it has to do with training dynamics, which your paper studies.

In light of the above, for this paper to be convincing, it's necessary to see experiments on MLPs of 1-3 layers and transformers of 1-3 layers, and I would hope to see that the geometric coherence score works over all experimental conditions consistently. I'd also like to see quadratic activations studied, since [7] proved an exact solution exists in networks using them, and many papers followed this up contrasting ReLU with quadratic activations.

**Poor plot quality**: the plots are hard to read with tiny fonts and are pixelated. They're also oversized and taking up much more space than necessary, and this space should be used to include other experiments.

Overall, this work seems to be preliminary, and were it reorganized with additional experimental results, it could be an interesting piece of the modular addition story (were it to answer any of the remaining open problems on modular addition, e.g. what changes in the training dynamics between 1 and 2 layer networks. However, if instead its goal is to serve as a work aiming to give insights into grokking, it's far too limited in scope---grokking was originally studied on modular addition but has since been studied on many other datasets including natural data (e.g. MNIST, CIFAR-10, etc). Thus, I would like to see this framework working on these datasets as well (unless this framework resolves the aforementioned open question, or any other ones on modular addition).

### Questions
Q1. In the main paper why are you using d=8 if the appendix states that d=2 is the best? 

Q2. If you were looking at models that weren't a transformer (e.g. MLP without attention), what aspect of those networks would correspond to the attention flow you saw in the transformer? And what would be the corresponding version of Figure 5? Would neuron activation plots capture this?

### Soundness
1

### Presentation
2

### Contribution
2
