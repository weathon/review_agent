# Tropical Expressivity of Neural Networks

- Decision: Reject
- Scores: 5, 3, 3, 6

## Abstract
We propose an algebraic geometric framework to study the expressivity of piecewise linear activation neural networks.  A particular quantity of neural networks that has been actively studied is the number of linear regions, which gives a quantification of the information capacity of the architecture.  To study and evaluate information capacity and expressivity, we work in the setting of tropical geometry---a combinatorial and polyhedral variant of algebraic geometry---where there are known connections between tropical rational maps and feedforward neural networks. Our work builds on and expands this connection to capitalize on the rich theory of tropical geometry to characterize and study various architectural aspects of neural networks. Our contributions are threefold: we provide a novel tropical geometric approach to selecting sampling domains among linear regions; an algebraic result allowing for a guided restriction of the sampling domain for network architectures with symmetries; and a new open source OSCAR library to analyze neural networks symbolically using their tropical representations, where we present a new algorithm that computes the exact number of their linear regions. We provide a comprehensive set of proof-of-concept numerical experiments demonstrating the breadth of neural network architectures to which tropical geometric theory can be applied to reveal insights on expressivity characteristics of a network.  Our work provides the foundations for the adaptation of both theory and existing software from computational tropical geometry and symbolic computation to neural networks and deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors study some theoretical and computational aspects of neural networks with piecewise linear activation functions. In Section 2, they are interested in the number of linear regions of such a network. They define the minimal effective radius around an input point as the smallest radius such that a ball of that radius intersects all the linear regions. They bound that radius from above by a Hoffman constant associated with the network. The main advertised application of that is that one can in principle enumerate all linear regions by sampling dense enough inside a ball of their estimated radius.

In Section 3, the authors consider networks that are invariant under a given group action. They provide lower and upper bounds on the number of linear regions by making use of the fundamental domain of the group action. Since the fundamental domain gives a tiling of the input space, the authors further explain that sampling inside the fundamental domain suffices to enumerate all linear regions.

In Section 4, the authors explain their software contribution. They provide a symbolic algorithm that finds all linear regions of a network once one has a representation of the network as a tropical rational function. The authors also suggest to measure the expressivity of a network by the number of monomial required in a tropical rational function representation. They provide an algorithm that reduces unnecessary monomials in a given representation. 

The authors conclude with some experiments and discussion, but their main contributions are the theoretical contributions in Sections 2 and 3, and their software contribution described in Section 4.

### Strengths
Originality: The definition of the Hoffman constant and its usage for sampling linear regions is novel, to the best of my knowledge. Also, I believe that the monomial count expressivity measure has not been studied before.

Significance: I think it will be very helpful for future research that the authors made their software available within the open-source OSCAR library. The proposed ideas of the Hoffman constant and the fundamental region might play a significant role in the future understanding of finding all linear regions of a piecewise linear network.

Quality and clarity: The paper is well-written. The mathematical statements are sound and the experiments make sense.

### Weaknesses
My main concern is whether the paper's contributions are strong enough, which consist mainly of the author's software and their idea of the Hoffman constant to find a bounded region in the input space that intersects all linear regions. Making use of the fundamental domain of a given group action is a rather folklore idea, and regarding the monomial count expressivity measure I have some questions (see below). Moreover, and also very important: The paper oversells its contributions. I strongly suggest to tone down the description of the contributions and focus on describing what exactly is done instead.

1) For instance, the last sentence of the abstract is way too bold. The authors claim to provide "the foundations" but it is rather a small piece of the puzzle.

2)  Another example is the sentence "We study more than simply the number of linear regions, we provide further insight on their geometry". The only new geometric aspect I can see is the definition of the minimal effective radius. But that does not really give us new insights about the geometric properties of the regions, such as their size/volume, their number of facets or vertices, or any other information about the combinatorial type of the polytopes. I suggest to remove the sentence in question completely.

3) Similarly, in line 088, the authors write that the Hoffman constant "effectively gives a geometric characterization". I do not agree with that half-sentence, since the constant does not tell us (almost) anything about the geometric properties of the linear regions; it only tells us how "far away" the regions are at most located. If the authors have something else in mind, I suggest to replace the vague and way too broad statement "geometric characterization" by the exact properties the Hoffman constant tells us.

4) I also think that the phrase "guarantee for the linear regions" in line 088 is too strong. Please be more specific on what that guarantee means. The authors provide a largest region for where one has to sample. But if the sample is not dense enough, one will still miss linear regions (as the authors note in the appendix).

5) It is always hard to claim to have the "first work" as in line 531. Symbolic experiments on other deep learning architectures have for instance been done in "Pure and Spurious Critical Points: a Geometric Study of Linear Networks" by Trager, Kohn, Bruna or in "Geometry of Linear Convolutional Networks" by Kohn, Merkh, Montúfar, Trager.

Finally, here some smaller comments:

- Please add "piecewise" before "linear" in the first line of the abstract.

- I think it is inaccurate to call tropical geometry a "reinterpretation" of algebraic geometry. It is rather a "discrete version" of algebraic geometry.

- In line 139.5, remove one of the 2 words "its the"

- In lines 176 and 177, the row vectors a and a' should have an index i. That is, they should be a_i resp. a'_i.

- In line 191, it would be better to spell out "for all" before "U" to increase clarity.

- In line 205, should m_p and m_q be equal?

- In line 212, insert "in" after "x"

- typo in caption of Figure 3

### Questions
1) In Theorem 3.3, what happens if the group is infinite? Shouldn't your theorem imply that the number of linear regions is infinite in this case? Or will it also be true that the set U_c is empty such that your lower bound becomes trivially equal to 0?

2) A priory, more monomials does not necessarily mean more expressivity, since the monomials could satisfy some internal (algebraic) relations/constraints. For instance, in classical algebra, the polynomials that are large powers of linear terms (i.e., that are of the form (a*x+b)^N for some fixed large N) have many monomials but they can only express a small portion (namely, a 2-dimensional subset) of the function space. Does this behavior happen for the tropical representations of ReLU neural networks? If yes, then a more accurate measure of expressivity would be the dimension of the semi-algebraic set that consists of all tropical rational functions that a fixed network architectures parametrizes, as studied in the article "Functional dimension of feedforward ReLU neural networks" by Grigsby, Lindsey, Meyerhoff, Wu. 

3) I don't understand what is meant with the number of monomials in Figure 9, since both the numerator and denominator have monomials. Do you plot the sum of the number of monomials of both numerator and denominator? Wouldn't in be better to also plot the product to get an idea of how far the upper bound in Example E.2 is?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper investigates counting the linear regions of neural networks through tropical geometry. The authors relate the Hoffman constant of a tropical rational map to the minimal effective radius, which is the smallest radius $r$ such that a ball of radius $r$ intersects all linear regions. They further show that for an invariant network (with respect to a group $G$), one can sample from the fundamental domain of $G$ to obtain an upper bound on the number of linear regions. Lastly, the authors introduce an algorithm using symbolic representation to compute the number of linear regions and propose a new measure of complexity for ReLU neural networks, termed the “monomial count.”

### Strengths
The number of linear regions is a standard measure of complexity for neural networks with piecewise linear activation functions, and the authors provide new insights into computing this count for a given network. Specifically, they relate the minimal effective radius to the Hoffman constant of a tropical Puiseux rational map. Additionally, for invariant neural networks, their observation allows a significant reduction in the number of samples required to estimate the number of linear regions.

### Weaknesses
Overall, the contributions of the paper do not appear substantial enough, leading me to recommend rejection. The theoretical depth is limited, and I see few immediate practical benefits from the results, see also the questions below.
-  While relating the Hoffman constant to the radius of the ball may have theoretical appeal, I do not see its practical benefit, as it is NP-hard to compute the Hoffman constant. The appendix discusses computing the Hoffman constant of a matrix, but there is no discussion about the Hoffman constant for a rational Puiseux function, which is what relates to the radius of the ball. 
- The result concerning sampling from the fundamental domain of the group action for an invariant network appears to be a straightforward observation. 
- The algorithm for determining linear regions using Puiseux polynomials does not seem to offer clear advantages over existing methods. The paper states, “Our tools consider the full input domain of these neural networks and provide an exact geometric characterization of the linear regions. This now makes previously inaccessible avenues available for analyzing the geometry of linear regions of networks (see Appendix I)." However, algorithms for obtaining the entire polyhedral complex of neural networks already exist (e.g., Masden, 2022, https://arxiv.org/abs/2207.07696).

### Questions
- What is the benefit of relating the Hoffman constant to the radius of the ball? Are there any approximations for the Hoffman constant of a rational Puiseux function? Has the Hoffman constant been applied to Puiseux polynomials (and/or rational functions) in prior work?

- Assuming we know the Hoffman constant, are there theoretical guarantees that using it would lead to better sampling to estimate the number of linear regions?

- Is the definition of the Hoffman constant for a Puiseux rational map well-defined (Definition 2.2)? How do you deal with the fact that there can be infinitely many quotients?

- In the introduction, the paper mentions, “Our library opens the door for the extensive theory and existing software on symbolic computation and computational tropical geometry to be used to study neural networks." Could you provide an example of a property of a function computed by a neural network that is now accessible through symbolic computation but was previously inaccessible? A more detailed explanation of these advantages would be helpful.

- Why is the native tropical representation for neural networks useful for measuring complexity? Can you comment on how this representation compares to the best representation as a tropical rational function in terms of monomial complexity?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies the expressivity of neural networks through a tropical lens. Through a suite of algebraic tools, the authors characterize the linear regions of a neural network using the Hoffman constant, discuss sampling improvements for invariant neural networks through the fundamental domain, and provide a symbolic framework for computing linear regions of neural networks.

### Strengths
The paper tackles important problems that can aid our understanding of deep networks. Tropical geometry has been getting increasingly popular in recent years, and has been the tool of choice for geometric analyses using polyhedral geometry. The introduction of the manuscript is crisp and easy to follow, introducing the main subject areas and discussing the contributions clearly. Finally, there is a significant number of contributions.

### Weaknesses
The weaknesses of the work can be summarized in the following axes:

- a severe lack of polish after the introduction,
- confusing results, mainly due to the number of contributions,
- it is unclear what the benefits of the contributions are, either at a theoretical or a empirical level.

**Lack of polish**

All main sections of the paper are introduced very abruptly, and the defining concepts of each of them are delegated to the appendices. This makes the paper extremely difficult to read, and is currently not self-contained without the appendices. Examples of this is the complete omission of a discussion on tropical polynomials, the discussion of Lemma A.19 or Equation 9 without their exposition, and all algorithms being discussed, but never presented, to name a few.

There are many typos and there is a lack of explanations at multiple points in the paper. An example of this lack are Figures 2 and 3, where, beyond the typos, Figure 2 lacks any real caption and what is portrayed in these figures, what the colors mean, or why there is a different number of lines on Fig 2(a) and Fig 2(b) is not clear.

Overall while the three different contributions are connected, the connection is not emphasized and the presentation really hinders the readability and the flow of the work. By trying to include all of them in the same work, it becomes extremely difficult to present each contribution in a clear and rigorous manner, especially at such a page limit. I believe the work could greatly benefit from either splitting the contributions, or investing a significant effort in crafting a coherent thread that connects the contributions, most likely at a venue with a much higher page limit.

**Confusing results**

In terms of the Hoffman constant, it seems that the authors compute the constant, then use it to find the radius of a ball that is needed to intersect all the linear regions. However, if I’m understanding things correctly, this gives no estimate on the number of linear regions. Essentially we have to compute the Hoffman constant (which is NP-hard), and then have to sample many points in that radius to get a probabilistic statement about the number of regions given a sampling size, and even then I don’t see any statements of that effect. Most works I’m aware of on counting the number of linear regions do it without having access to a trained network, and are statements about the architecture, not the specific network, so I’m unsure exactly what the main contribution is here.

Section 4.1 is also confusing and the contribution is again unclear. For example, in line 359 it is stated that we can detect when some intersections may be empty. The way this is stated implies that we have a way of checking if intersections are empty, but not a way to find such intersections. That seems to actually be the case, since on 377 a set of indices I is assumed instead of being computed. A similar comment as above can be made, that this works for specific networks and does not characterize the network class. Also, how are we computing these regions? Are we combinatorially testing all combinations?

**Experimental results**

Simply put, the experimental section is very rushed and the results are not convincing (more in the next section). Starting of with the characterization of width versus depth, what exactly is Figure 4 conveying? There is no comparison between depth versus width, for the same number of neurons. The fact that expressivity increases when we increase the number of nodes is not surprising. A minor comment is also that the range of $k$ doesn’t correctly correspond to Figure 4.

For Figure 5, the trained and untrained networks seem to hardly have any difference in their pruning rate, and the size of the network is so small where no trends could be observed.

### Questions
In terms of questions:

- Early in the introduction the authors mention that expressivity is one of the most important approaches to measure performance. Are the authors aware of any references to support that claim? Usually overly complex networks (i.e., very expressive ones) tend to overfit the data and lead to poor performance, exactly the opposite of what the authors are arguing.
- Given that the purpose of the paper is to aid our understanding of deep networks, there seems to be a limited intuitive explanation of the methods that are used. For example, what is an intuitive explanation of the Hoffman constant? Section 2.1 makes an attempt at explaining, but actually describes the distance $d(u, P(A, B))$ and not the constant, and does so by restating the equation instead of an intuitive explanation.
- About the fundamental domain:
    - It seems that things only work for discrete groups, if so that should be a concrete discussion.
    - There is mention of groups acting on vector spaces, but no discussion about group representations (or at least how these groups act on the vector spaces) is presented.
    - The authors state that in the specific example of 3.2, a factorial improvement is made. However, clearly, $\Delta$ is an infinite set. There is no asymptotic analysis as to how many samples we need from $\Delta$, so how are you claiming an improvement without a sampling statement when sampling from an infinite set?
- Line 395 is very imprecise. What does information mean here? The expression and the function both encode the same function.
- On 423, the authors argue their use of small networks saying that they help highlight intuitive insights that can aid understanding. What are the insights that are gained in doing so? Of the three presented experiments, the first one has no comparison on depth vs width (which is the titular motivation for the section), the second one shows a marginal change which is unclear if it would persist when the layers/nodes increase, and the final one has no takeaways listed.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a novel approach to investigating the expressivity of neural networks through the framework of tropical geometry, a branch of algebraic geometry that employs combinatorial and polyhedral methods, with key contributions including a new framework for selecting bounded sampling domains using the Hoffman constant, a method for enhancing computational efficiency through symmetry in network architecture, and a Julia library integrated into the OSCAR system for symbolic representation and analysis of neural networks using tropical techniques. The paper builds on foundational work that connects tropical rational maps with feedforward neural networks, providing theoretical insights supported by experiments that demonstrate its applicability across various network architectures. Overall, I believe this paper makes a significant contribution to the theoretical foundations of deep learning.

### Strengths
**Clarity:** The paper is well-organized, exhibiting a clear and logical progression from the theoretical background to the main results and their implications. Each section is thoughtfully introduced with specific motivations that enhance the reader's understanding. Notably, the three contributions are effectively aligned with their corresponding sections, reinforcing the connections among the presented ideas.

**Novelty and Significance:** The innovative application of tropical geometry to evaluate the expressivity of neural networks introduces valuable insights. By translating neural networks into tropical representations, the authors utilize tools from an underexplored area of deep learning theory. Their approach, which provides an exact count of linear regions and introduces monomial complexity as a measure of expressivity, has the potential to inspire future research on model architecture and optimization.


**Practical Implementation:** The integration into OSCAR allows for symbolic manipulation of neural networks, which has practical implications for further research and applications.

### Weaknesses
**Limited Empirical Validation:** The experimental demonstrations, while useful as proofs of concept, are mainly limited to small-scale networks.

**Context within Existing Methods:** A more in-depth discussion on how this tropical approach compares to or enhances traditional methods of region counting and expressivity analysis would provide valuable context.

**Notation Clarity:** Some notations in the main paper are not defined in a timely manner. For example, $P(A,b)$ is defined in Line 673 of the appendix but is used in Line 146 of Section 2 without prior explanation.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
