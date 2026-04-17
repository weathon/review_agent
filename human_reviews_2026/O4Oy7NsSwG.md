# Topology and geometry of the learning space of ReLU networks: connectivity and singularities

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Understanding the properties of the parameter space in feed-forward ReLU networks is critical for effectively analyzing and guiding training dynamics. After initialization, training under gradient flow decisively restricts the parameter space to an algebraic variety that emerges from the homogeneous nature of the ReLU activation function. In this study, we examine two key challenges associated with feed-forward ReLU networks built on general directed acyclic graph (DAG) architectures: the (dis)connectedness of the parameter space and the existence of singularities within it. We extend previous results by providing a thorough characterization of connectedness, highlighting the roles of bottleneck nodes and balance conditions associated with specific subsets of the network. Our findings clearly demonstrate that singularities are intricately connected to the topology of the underlying DAG and its induced sub-networks. We discuss the reachability of these singularities and establish a principled connection with differentiable pruning. We validate our theory with simple numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents their study on the function spaces parameterized by polynomial neural networks (i.e., those whose activation functions are polynomial). There are two main contributions: identifiability and singularity of functions in the neuromanifold (i.e., functions representable by neural networks). For the former, the authors show that for generic functions in neuromanifold, the set of parameters realizing these functions is at most finitely many or singleton, for Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN) architectures respectively. For the latter, they characterize singularities as functions realized by sparse subnetworks and links this discovery to the sparsity bias of MLPs.

### Strengths
The paper are generally well-written and the results are well-presented. While I do not dive into the proof, their results look sound to me. Two contributions are mathematically interesting and suggest further following work.

### Weaknesses
Several points deserves to be further polished:
1. Since most architectures use ReLU, I find that it is better to connect the current results to the ReLU cases (authors did admit this limitation in section 5).
2. The bound on the degree of the activation in Theorem 4.1 is vacuous in the dimensions of the neural network architecture. Hence, I am not sure if this result reflects what we truly observe in practice.
3. If I understand it correctly, the definition of critically exposed implies that there exists a positive probability that mappings $u$ admit a weight in a critically exposed set as critical points of the training dynamics (provided that we have sufficiently data). However, since we are unable to quantify this probability, they might be negligible and might vanish when dimension increases. I am not sure if we can use this notion to explain the so-called ``bias towards sparse subnetworks'' as in the paper.

### Questions
1. In section 3.2, the link between optimization on the parameter space and on the neuromanifold is rather hand-waving. I wonder if there is a real relation between these two (under suitable conditions).

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
This paper studies the feed-forward ReLU networks defined over directed acyclic graphs, examining the (dis)connectedness of the parameter space and the existence of singularities within it. The conservation laws under gradient flow are identified. Due to the disconnectedness of certain parameter configurations, certain singularities are unreachable, reducing the expressivity of ReLU networks at initialization.

### Strengths
- The paper is relatively well-written and polished. Illustrative figures are provided to accompany the theoretical results and aid understanding.
- The theoretical formulation is clean.
- The result on the disconnectedness of the parameter space is somewhat surprising. The implication of losing expressivity at initialization seems interesting.
- Some numerical experiments are conducted to validate theoretical results.

### Weaknesses
- I am wondering whether the disconnected case occurs in fully-connected ReLU networks or not, since the example network given in Figure 2(d.1) does not look like a fully-connected network. If the disconnection only occurs in networks that are not fully connected, then the statement in line 358 may be inaccurate: "the expressivity can be reduced to the extent that they lose their universal approximation capability"; because ReLU networks that are not fully connected are not universal approximators to begin with. Please feel free to correct me if I have misunderstood your results.
- In line 423, the authors state that: "given a random initialization, the probability of $\mathcal H_G(c)$ having singularities is zero." I trust that this statement itself is correct. However, it doesn't necessarily mean that the gradient flow/descent cannot go near singularities. It's quite common that ReLU networks can have saddle-to-saddle dynamics, in which the gradient flow path passes near a sequence of fixed points [1]. In those cases, even though the dynamics from random initialization never puts the parameters exactly in an invariant set, going near those fixed point is still a very prominent, if not the most prominent, trait of the learning dynamics. If I didn't misunderstand the result, the paragraph "singularities are rare" should probably come with more nuance or caveat -- "probability of having singularities being zero" doesn't mean that learning dynamics doesn't go near singularities.
- The conservation laws arising from symmetries are also studied in [2]. I am wondering how their results relate to yours results in "local conservation laws under gradient flow" in line 160.
- It might be useful to also discuss the limitation of studying gradient flow in place of SGD. Because the quantities that obey conservation laws under gradient flow can actually be time-varying in SGD [3,4].

[1] Boursier et al. "Gradient flow dynamics of shallow relu networks for square loss and orthogonal inputs." NeurIPS 2022.

[2] Ziyin. "Symmetry induces structure and constraint of learning." ICML 2024.

[3] Liu et al. "Noise and fluctuation of finite learning rate stochastic gradient descent." ICML 2021.

[4] Chen et al. "Stochastic collapse: How gradient noise attracts sgd dynamics towards simpler subnetworks." NeurIPS 2023.

### Questions
Is there a particular reason to use the uncommon notation of double angle brackets $《》$? I struggled to understand it from a short inline definition given in line 177. I also didn't know if this notation is essential for reading and understanding the main results.

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
3

### Summary
This paper studies the properties of the parameter space of ReLU networks, notably in order to decide whether this space is connected and/or contains singularities, which are relevant questions to consider when targeting an optimally trained network, or to prune the network without losing performance/expressiveness, respectively.

The authors consider the framework of Directed Acyclic Graphs (DAGs), which is more general than layered architectures,and focus in this paper on properties of homogeneous activation functions, in particular here the ReLU activation function.

They show a complete characterization of the connectedness of the learning space under GF with any given initialization, analysing to this end the role of bottleneck vertices in the network (that is, vertices with only one out-going arc, or only one in-coming arc) and the balance conditions (invariant under GF once the initialization is done) on related sets of vertices.

Moreover, the authors study singularities, namely parts of the learning space where part of the network stops contributing to the computation. They prove a link between the existence of such singularities and the already mentioned balance conditions, and that even when the conditions are gathered, a GF algorithm will not stumble upon a singularity in finite time. 
The authors circumvent this impossibility to favor "self-pruning" by using regularization, and provide numerical experiments showing which regularization helps driving the model towards singularities.

### Strengths
Provides a sound and thorough theoretical analysis of the connectivity of learning space for ReLU-activated DAGs Networks trained under GF after arbitrary initialization.

Theoretical analysis of the conditions of existence of singularities, and of the possibility to reach them, complemented with experimental results on tools to reach these singularities in practice.

### Weaknesses
The results on connectivity might be achievable with simpler tools and less technicality.

The experimental part on connectivity does not bring anything to the discussion. 

The introduction of some notions and symbols is lacking.

### Questions
p3, discussing on re-scaling: Do you assume here and in the rest of the paper that all biases are 0?

p4, top of the page: do you have any other requirement on $\\ell$, other than it being differentiable? For instance $\\ell(x,x)=0$ ?

p5, Definition 1: $\\theta^2$ is the vector obtained from $\\theta$ by squaring each individual element? Or do you here implicitly use some other product?

p5, Proposition 3: the point of view of network flows can be obtained in a simpler way as what is done in Appendix A.3. Indeed, since the source and sinks have unconstrained flows, it would suffice to initialize all edge weights with 1, and then correct the balance for each node $u$ with a simple edition of the weights along a path from an arbitrary source to an arbitrary sink going through $u$. Does the algebraic point of view give, in some way, more insight for this paper?

p6, Theorem 1: I think the proof could take a shortcut (following the idea of the precedent remark, and the intuition-providing  text at the beginning of page 7: first prove that if the conditions are not satisfied, it is unfeasible to satisfy the responsible set of vertices, and if it is, show that fixing first the edge weights incident to $Anc(v)$ (or $Desc(v)$) to satisfy the local constraints, and then construct the rest of the solution greedily without editing any edge incident to $Anc(v)$ (which is then possible by definition of this set, anything outside is on at least one path from source to sink avoiding $Anc(v)$). 
For the trivial case where the deleted $e$ does not correspond to a bottleneck, the result is immediate.
Then, the Proposition 4 directly yields the result.
Is there a reason for taking the long and more technical way?

p6, Figure 2d: the figure is a good illustration of the proved theorem. I don't understand however, what the additional experiments on real data (Appendix A.9.1) bring to the paper, since it needs no further empirical demonstration that the space is disconnected. As I am less acquainted with the experimental side, could you indicate what I am missing here?

p8, Proposition 6: is the converse known to be true/false? 

p9: When and why is self-pruning interesting to have? I understand why one wants to self-prune when the initialization was made such that some singularity exist, but is there an advantage in how expressive the network can be when initialized with a reachable singularity, versus when initialized such that none can be attained by GF?


Typos and Suggestions:

p3, Symmetries of ReLU networks: $\\sigma$ is not introduced, which could be done by adding "the activation function" in front.  Moreover, the formulae for ReLU and Leaky ReLU are both wrong: ReLU: $\\sigma(z)=\\max\\{z, 0\\}$ and Leaky ReLU: $\\sigma(z)=\\max\\{z, \\gamma z\\}$.

p3, Local conservation laws under gradient flow: the variables $d$ and $e$ are clear from context but should be defined nonetheless. 

p6, Definition 2: prefer the use of "...with $V^-_B$, $V^+_B$ the sets..., respectively."

p6, Figure 2c: (ii) this case is misleading, since it is not obvious without the text that the case (iii) can "override" it. It also is technically not true, since there could be a completely independent vertex $v'$ in the network making the space disconnected. Maybe find an alternate formulation meaning roughly "a priori connected".

p7, Corollary 2: unless some intermediate layer has a single neuron! It might not be an interesting case, but the soundness of the corollary requires excluding it.

p7: "Concretely, it means that the balance condition ... will forbid sign switches ..." This is the important intuition behind this section, it would be valuable to highlight it more, and potentially to merge it with the previous paragraph. 



I am open to updating my rating of the paper, depending on the answers provided by the authors.

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
The paper studies invariant sets for gradient flow training of DAG-based ReLU architectures and singularities within those invariant sets.

### Strengths
The paper is very well written and provides some insights on properties of the training dynamics.

### Weaknesses
To me the results seem to be relatively minor and easy extensions of previous results. The authors suggest that formulating these conservation laws with the use of the incidence matrix of the DAG gives significant new insight. But as far as I can see, the main insight is that there a singularities when parts of the graph become disconnected, which does not seem to be surprising.

### Questions
On one hand, singularities could be a concern, because they cannot be escaped once reached. On the other hand you suggest that they may be desirable in the sense that they can be seen as the model performing some automatic pruning during training (and indeed you suggest that one may want to induce singularities intentionally). May question is whether there could not be a worry that inducing singularities prematurely limits the model and prevents it from later converging to more favorable solutions which require all neurons (or at least some of the prematurely pruned ones).

### Soundness
4

### Presentation
4

### Contribution
2
