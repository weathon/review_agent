# A Theoretical Study of Neural Network Expressive Power via Manifold Topology

- Decision: Reject
- Scores: 6, 6, 5, 6, 6

## Abstract
A prevalent assumption regarding real-world data is that it lies on or close to a low-dimensional manifold. When deploying a neural network on data manifolds, the required size, i.e., the number of neurons of the network, heavily depends on the intricacy of the underlying latent manifold. While significant advancements have been made in understanding the geometric attributes of manifolds, it's essential to recognize that topology, too, is a fundamental characteristic of manifolds. In this study, we investigate network expressive power in terms of the latent data manifold. Integrating both topological and geometric facets of the data manifold, we present a size upper bound of ReLU neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper provides results on:
- an upper bound on the network size required to approximate the indicator function of a union of balls / solids
- a lower bound for the training set size necessary to learn the homeomorphism between a complex data manifold and union of balls / solids, approximating it by a simplicial homeomorphism
- disentangling topological and geometric complexities reasoning on two properties rather than a single quantity. 

The main theorem brings these together.

### Strengths
I like the ideas in this paper relating data, the network and the classification task. Although preliminary, the results can be used to derive further bounds that encompass other aspects of deep learning .

### Weaknesses
- Since the setting is not defined until the Main Thm, the statement remains very hard to grasp. For example, which manifold is this theorem talking about? Some manifold constructed from the network weights/architecture or the data manifold? This doesn't become clear until the problem setup. Why don't we define the RELU network already?

- The following two works (uncited currently) present the connection between the intrinsic dimension and the topology of weight trajectories:
    * Andreeva et al. Topological Generalization Bounds for Discrete-Time Stochastic Optimization Algorithms. NeurIPS 2024
    * Birdal et al. Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks. NeurIPS 2021

  This is also why I believe that the claim in Ln. 72 should be softened, e.g. we can say 'data topology' instead of topology. 

- This paper only investigates network size in regards to data topology. However, other crucial elements such as the optimizer, task and the network architecture are disregarded. I don't see how the simplistic view of this paper can fully capture the intricate nature of expressive power.

- Betti numbers are quantities that are somehow hard to work with. Can we use for example silhouettes or other descriptors extracted from the persistent homology? This also better relates the paper to the existing literature. Note that PH is also computed through gradual thickening.

- Can we have a better name for 'topological complexity' as this notion is extremely general and does not explain the particular notion in this paper. 

- Unfortunately, we cannot see any empirical validation of the bounds presented. Could we validate at least the approximation errors, maybe on toy set-ups? Do the estimated quantities correlate well with the actual expressive power?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel bound on the size of ReLU neural networks which, as presented in the paper, are fully-connected feed forward neural networks (NNs) with ReLU activations. Previous work has focused on geometric descriptors in order to offer bounds on the size of NNs, but this work's novel contribution adds topological descriptors in order to calculate upper bounds on the size. In particular, the size is bounded by a function of the square of the sum of the Betti numbers of the manifold, among other variables.

The paper restricts itself to characterization of manifolds that are described as being from the family of thickened 1-manifolds, which are essentially the combinations (i.e. connected sums and disjoint unions) of spheres and tori. Defining an overall network size then is simply the aggregation of the network sizes on these individual ‘pieces’.

The main result is given as a constructive proof and offers a method for creating a ReLU neural network capable of approximating the indicator function of the class labels within a given error of $\epsilon$ with probability $1 - \delta$. It’s important to point out that the class of manifolds which this works for are assumed to be from disjoint sub-manifolds of the classes, e.g., $M = M_1 \sqcup M_0$ where it’s assumed that the objective is binary classification.

### Strengths
The authors offer some originality in their use of topological descriptors to give upper bounds on the size of neural networks, which is an oversight in other methods that seem to ignore the topology of data manifolds altogether.

The paper is well written in that it makes use of various results from previous authors with relevant related work. The relevant citations are present and their inclusion has a clear purpose. The authors also give ample justification for their results in the form of mathematical proofs.

Being able to determine an upper bound on the size of a neural network given a data manifold (for classification problems) is significant, given that the size of a neural network is a major factor in determining its efficacy, trainability and usability. Therefore, in that respect this paper makes a significant contribution in a narrow domain (see Weaknesses).

### Weaknesses
Weaknesses in this particular method spring from two main sources: 1) computational complexity of computing Betti numbers and 2) its applicability and usability.

Betti Numbers: a well known problem with computing Betti numbers is their computational time-complexity, which is dependent on the size of the input and the dimension to which we wish to compute. In the paper the authors define the total topological complexity of the manifold as $\beta(M) = \sum_{i=1}^{d-1}\beta_i(M)$, where $\beta_i(M)$ is the $i$-th Betti number of the manifold. Given that we must compute the Betti numbers out to dimension $(d-1)$ is a serious limitation assuming that our manifold is even a few thousand points big. Therefore, computing the proposed upper bound given that we must first compute $\beta(M)$ is already infeasible for most real world applications.


The paper describes the main process as working for classification problems only. While the number of classification problems is significant, this is a serious limitation for anyone wanting to perform regression or any other objective. Further, given that this process is already computationally infeasible, it seems that practitioners are better off by simply increasing the size of their model, training it and then iterating if results aren’t good enough, instead of taking more time to find out the worst case size that their network would need to be.

### Questions
1) In the paper you define the size of the network as $\sum_l w_l$ where $1\leq l < k+1$. Why do you not take into account the biases and layer $k+1$ when determining the size of the network?

2) From Proposition 3 we can approximate the simplex by only taking $n$ samples from $M$. However, the number of samples is dependent upon the volume of the $d$-dimensional $\epsilon$-balls in the covering. If $d$ is very large, then the volume of the balls is almost zero (depending on the metric used). Therefore, $n$ could also be very large and even bigger than the number of samples present. What should be done in this case? Have you tried to use this method with high-dimensional data?

3) As was mentioned, it is currently intractable to calculate high dimensional Betti numbers for even a few thousand sample points. Have you used or do you recommend using some sort of approximation or heuristic to calculate the Betti numbers?

4) Could you elaborate on any potential theoretical or analytical insight from employing this method, given that it may not be practical in all scenarios?

5) Also, is this method easily extensible to other objectives like regression?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper discusses geometrical and topological aspects of the latent manifold in relation
with the expressibility power of neural network.

The main results state as follows. Theorem 1 gives an upper bound on the network size required
to approximate the indicator function of a certain manifold linked topologically to the data.
Theorem 2 given another bound on complexity by again the network size.

The paper is reasonably written and enough readable, examples are presented with figures to help the intuition, concepts like Betti numbers are communicated in a satisfactory way also to non specialists.

However since the authors mix differential geometry techniques as the condition number linked to the curvature of a manifold with respect to its embedding in R^D with purely topological concepts, as the Betti numbers, which are a topological invariant of a manifold, thus making the reader confused about the purpose of the paper.

Moreover results are presented without giving a clear connection with possible applications in the machine learning domain, not even
with a small practical example that could elucidate, as proof of concept, a future application of their main result.

### Strengths
1) This is a novel topological approach to the expressive power of neural networks.

2) The main result (Theorem 2) is an improvement of existing bound pushing the result in ICLR 2016.

### Weaknesses
1) I have some concerns about practical relevance, since I am unable to see the importance of thickened 1-manifolds in machine learning especially in connection with the latent manifold. Perhaps (see questions below) the authors could explain a simple example, like as proof of concept, to elucidate their finding in some simple application. There is no evidence in this paper that the topological structure of thickened 1-manifolds plays a role in a better practical description of the latent manifold.

2) Though the authors generalize the work  "Depth-width tradeoffs in approximating natural functions
with neural networks" published in ICLR 2016, they are not providing any experimental evidence that what they do
go beyond the results in there from a practical point of view.

3) Technical remark: the condition number is linked with the Riemannian structure (curvature etc), while this paper is focussing mostly on
topology. Once we take homeomorphism, we are changing the condition number of a manifold, then I am not sure about the soundness of the arguments regarding condition number and topology.

### Questions
1) Can you provide with some easy example elucidating your main results? Even proof of concept is ok.

2) In particular (refinement of question 1), can you provide a specific example or case study demonstrating how thickened 1-manifolds relate to real-world machine learning problems? Can you illustrate how the theoretical results apply to a practical ML task?
This would help elucidate the importance of this concept in machine learning.

3) Regarding weakness (2). Can you provide with a specific experiment on benchmark datasets for binary classification that can demonstrate the practical advantages of the authors' approach compared to the ICLR 2016 work? This would provide an empirical validation now missing.

4) Can you explain better how topology (Betti numbers) and differential structure (condition number) can be considered together as means of bounding the complexity of a network for a task? (here binary classification)

5) In particular (refinement of question 4), how do you reconcile the use of condition number (a geometric property) with topological invariants as the Bettin numbers? More specifically, explain how the homeomorphism affects the condition number and how this impacts their theoretical results.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates a NN's expressive power in relation to the data manifolds underlying geometry and topology. They provide theoretical bounds on the size of ReLU classifiers in terms of the manifold's Betti and condition numbers. The condition number measures how much a manifold "curves" in its ambient space.

### Strengths
The paper is very well written. The introduction surveys the field and the current state of the art. The authors provide a nice overview of their problem with many references. The authors provide explicit and intuitive background on all the mathematical objects they intend to use. Their description of their family of thickened 1-manifolds is clear. They clearly state their problem and provide a reasonable answer.

### Weaknesses
The theory seems to be a bit backwards. It appears the authors found something they could prove about a small class of manifolds, and then tried to write a paper about this class. They claim that the thickened 1-manifolds are representative of data manifolds, but provide no empirical nor theoretical proof here.  
The geometric and topological characteristics they use are also somewhat random. The reach (or condition number) is a min over the entire manifold, and as such only captures geometry at a single point. The authors bemoan how difficult it is to incorporate topological information because of its discrete nature, but the data is discrete so they should be able to use TDA to assist them here.

### Questions
Why is TDA not used in this paper?  Is there empirical evidence that real datasets can be modelled as 1-manifolds? There are no experimental results in the paper.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the relationship between neural network size and the topology and geometry of data manifolds, providing a theoretical framework that integrates these manifold properties into network size bounds. The authors present an upper bound on neural network size based on Betti numbers and the reach (inverse of the condition number), representing topological and geometric aspects of the data manifold, respectively. This work aims to address the gap in the literature regarding how topological complexity influences network size, which has been previously underexplored compared to geometric factors. The theoretical results, exemplified through the construction of topological representatives, offer a novel perspective on data manifolds and potential implications for future neural network architectures.

### Strengths
* The theory presented is well-structured and mathematically elegant, making effective use of both topological and geometric complexity measures, notably Betti numbers and the reach. This duality could be beneficial for refining future analyses in manifold learning and deep learning.
* The construction of a neural network classification framework that factors in manifold topology is innovative, especially in how it leverages topological representatives for improved upper bounds.

### Weaknesses
1. Relevance to Neural Representations: While the theoretical results are indeed intriguing, there is limited discussion on their practical relevance for the study of neural representations. Without a thorough analysis of how these bounds translate into tighter or more representative metrics, the applicability of these results to real-world neural networks may be difficult to gauge. Further connections to neural representations in active fields like feature or metric learning would enhance this work’s value to the ICLR community.
2. Upper Bound Tightness: The paper provides an upper bound on network size influenced by manifold topology and condition number, but it does not clarify the tightness of this bound. The condition number could potentially introduce loose dependencies on manifold dimensions, especially since only an upper bound is given without a complementary lower bound or empirical justification. It would be useful if the authors could discuss the conservatism of their bound and any assumptions that might affect its applicability.
3. Lack of Empirical Examples: Although the paper includes theoretical constructs (such as Figure 3), there is a noticeable lack of concrete examples or even toy simulations that showcase the behavior of neural networks with varying topological complexities. Empirical demonstrations—even simple ones—would support the theoretical claims and help readers better understand the practical implications of the upper bound.
4. Connection to Broader Research Themes: Given the potential implications of this work for a range of ML themes (e.g., representational learning, metric learning, compositional modeling, reinforcement learning), the authors could benefit from speculating on or directly addressing possible applications of their results. It would be insightful to hear how they envision the integration of these theoretical insights with active areas like uncertainty quantification, large-scale learning, and non-convex optimization.

### Questions
1. On the Upper Bound Dependency: Could the authors clarify whether the dependence on the manifold dimension d in the Main Theorem is a consequence of the upper bound’s conservativeness? Specifically, might the condition number be introducing an artificial inflation in the bound?
2. Empirical Justification: Are there plans to include practical or toy examples of neural network training on manifolds with varying topological complexities, beyond the “construction” cases in the paper? Such examples could help verify the utility of the theoretical bounds.
3. Potential Applications: Could the authors share any insights or speculations on how their findings might extend to practical applications? For example, how might this framework influence neural network design choices in contexts such as feature or representational learning?

### Soundness
3

### Presentation
3

### Contribution
2
