# Constructing Invariant and  Equivariant Operations by Symmetric Tensor Network

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Design of neural networks that incorporate symmetry is crucial for geometric deep learning. Central to this effort is the development of invariant and equivariant operations. This work presents a systematic method for constructing valid invariant and equivariant operations. It can handle inputs and outputs in the form of Cartesian tensors with different ranks, as well as spherical tensors with different types. In addition, our method features a graphical representation utilizing the symmetric tensor network, which simplifies both the proofs and constructions related to invariant and equivariant functions. We also show how to apply this method to design the equivariant interaction message for the geometry graph neural network and general neural network models incorporating symmetry.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper claims to introduce a graphical tensor network framework for constructing SO(3)-invariant and equivariant operations for Cartesian tensors and spherical tensors. In particular, they claim to build generators of invariant functions which they apply in the geometric graph neural network domain, specifically in designing the equivariant interaction message between nodes.

### Strengths
- I commend the authors on contributing a practical framework for SO(3)-equivariance that is based upon results that are found in classical invariant theory.
- The formulation in terms of tensor networks is useful in terms of diagrammatically representing the operations.
- The ability to model both Cartesian tensors and spherical tensors using tensor networks is potentially significant and should result in a wide range of applications.
- Their experiment in Section 5 clearly demonstrates the potential of their approach.

### Weaknesses
- I think the theoretical contribution is very limited. All of the theory used (the results in Section 3) appear in the classical invariant theory literature. In my view only the reformulation of these results in tensor framework notation is original, but I think that originality (in terms of theoretical contribution) is very limited.
- A crucial point for me is the following, which I'd like the authors to address: I think the related work section is 1) in the wrong place - it should be included wholly in the main part of the paper and 2) is missing what I think is key literature that needs to be compared against. For example, in my view, the paper [1] surpasses the theoretical results in this paper: it came up with a complete, explicit basis for modelling all O(n) and SO(n) invariant and equivariant maps between tensors based on the paper [2], which is stronger than the results given in section 3 of the current paper. Indeed, firstly, the current paper only looks at SO(3), whereas [1] looks at all of SO(n). Secondly, the results used in section 3 only tell you what kinds of invariants generate the algebra, and moreover create redundant generating functions, so there is no minimal set of generators necessarily. In contrast, [1] gives a fully constructive description and tells you specifically how to build the invariants exactly. I definitely think a comparison between the results here and those in [1] need to be included explicitly in the paper.
- I think section 4.1 needs to be much clearer, as the link between the tensor network notation introduced in section 2 is not made clear in its application in this section. I also think the notation is somewhat ambiguous and sloppy (e.g in 9), despite the subscript. 
- Given my view that the strength of this paper is solely in its practical application, I think having only one application of their method is not sufficient. I would have liked to have seen more applications where the authors' framework could be applied, especially since their framework works for both Cartesian tensors and spherical ones. 
- I appreciate that the authors have used LLMs to improve the grammar and refine sentences in the paper. However, I think there are still far too many typos and grammatical errors in the paper which makes it more difficult to read. I recommend that the authors go through this paper again carefully, correcting as many of them as they can in order to improve the paper's readability.

[1] Pearce-Crump, E. - Brauer's Group Equivariant Neural Networks (ICML 2023)

[2] Brauer, R. - On Algebras Which Are Connected with the Semisimple Continuous Groups (1937)

### Questions
- I would like the authors to comment on the weaknesses I have listed, especially regarding the paper [1] above.
- I would like them to explain/demonstrate what sorts of problems their practical framework can be applied to, since for me this is their main contribution (which is only demonstrated so far by the one experiment in the paper itself).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a general framework for constructing SO(3)-invariant and equivariant operations using symmetric tensor networks. The authors further illustrate how this applies to geometric GNNs by manually deriving some message-passing operators.

### Strengths
- Studying efficient and reliable equivariant/invariant operators is a grand goal.
- The graphical representation constructed in this paper is very interesting and seems to have some relevance to Penrose graphical notation.

### Weaknesses
> **Confusing Writing Logic.**

If I understand correctly, the core contribution of this article lies in the Tensor Diagrams (TD) notation it provides (since I haven't seen any novel equivariant/invariant constructions). 

Therefore, the appropriate overall layout should be: briefly introduce the definitions of Cartesian and spherical tensors and formalize equivariance, then directly provide a systematic introduction to TD, followed by a discussion in conjunction with geometric graph neural networks. 

However, the article currently begins with a lengthy discussion of lemmas, which is extremely unfriendly to those outside the field and extremely trivial and boring to those within it, making it difficult to quickly grasp the article's contributions. 

Below, I will specifically address the shortcomings of each paragraph.

> **Overclaim of Contribution.** 

In the abstract, the authors claim that their work can "construct valid invariant and equivariant operations", but looking at the entire paper, it seems that this paper is simply reformulating existing work. Of course, this is also a good contribution, but it is necessary to revise the statement of contribution.

In addition, the formalism in this paper makes extensive use of cross products, so it may be very difficult to generalize to O(3) groups. Therefore, I think the authors need to emphasize the scope of application of this paper at the beginning.

> **Lack of Relevant References**

This paper lacks references in several areas. On the one hand, it lacks an introduction to the origins of TD, and on the other hand, it lacks information on geometric graph neural networks. In conjunction with my previous suggestion to the authors to rewrite the paper, I list the discussion that should be supplemented below.

Regarding TD, relevant introductions and discussions such as Penrose graphical notation should be supplemented, and at least the differences between the TD in this paper and Penrose graphical notation should be clearly explained. The following references are recommended:

> - [A] The Tensor Cookbook

Regarding geometric graph neural networks, I suggest a discussion and summary similar to MPNN [B], which should be divided into the following parts:
- From EGNN [C] to HEGNN [D]: Generate scalars through inner products, from 1st-degree irreducible representation to higher-degree irreducible representation.
- From TFN [E] to MACE [F]: Construct and screen through tensor product, from two-body action to multi-body action.
- From LieConv [G] to Frame-Averaging [H]: Construct equivariants/invariants through group convolution, from continuous group convolution to discrete group convolution.
- E2former [I]: Relate Cartesian and spherical tensors and achieve acceleration.
- Theoretical articles [J-N]: used for simple theoretical discussions to illustrate the rationality of the construction.

> - [B] Neural Message Passing for Quantum Chemistry
> - [C] E(n) Equivariant Graph Neural Networks
> - [D] Are High-Degree Representations Really Unnecessary in Equivarinat Graph Neural Networks?
> - [E] Rotation- and translation-equivariant neural networks for 3D point clouds
> - [F] MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields
> - [G] Generalizing Convolutional Neural Networks for Equivariance to Lie Groups on Arbitrary Continuous Data
> - [H] Frame Averaging for Invariant and Equivariant Network Design
> - [I] E2Former: An Efficient and Equivariant Transformer with Linear-Scaling Tensor Products
> - [J] Scalars are universal: Equivariant machine learning, structured like classical physics
> - [K] On the Universality of Rotation Equivariant Point Cloud Networks
> - [L] On the Expressive Power of Geometric Graph Neural Networks
> - [M] Universally Invariant Learning in Equivariant GNNs
> - [N] On the Completeness of Invariant Geometric Deep Learning Models

> **Insufficient experiments** 

The experiments in this article are unimportant and seem to have been forced in by the authors to fill the main text. Their inclusion here is a major flaw and should be removed (rather than moved to the appendix). Specifically, because the objective function is complex, we are uncertain which features are needed and which are not. Therefore, when there are very few features, adding features is likely to improve model performance, while when there are many features, adding features will actually reduce performance. This is common knowledge and requires no additional experimental explanation.


---
 
Overall, as a submission intended to systematically reformulate the field, this paper currently falls far short of the standard necessary for acceptance. The vast majority of its content either simply reiterates domain knowledge or lacks integration with existing work. Both its logical presentation and depth of thought are far below the community average, so I recommend rejection.

Considering that ICLR has accepted resubmitted PDFs in previous years and that authors are not required to perform additional experiments, I believe the above revisions are minimal and can be completed during the discussion phase. If the shortcomings I've raised are addressed, I will reconsider my score.

### Questions
See Weakness.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a paradigm for representing equivariant neural networks using tensor network notation. For the specific case of the group $G = SO(3)$, the author discusses the structure of the equivariant neural network when expressed as a tensor network. This provides a methodology for designing networks intended for scientific problems that exhibit corresponding symmetries.

### Strengths
The author introduces notation from tensor networks in quantum many-body theory, representing symmetric polynomials as contractions of symmetric tensors. The modeling of equivariant operators is achieved by representing these symmetric tensors with tensor networks. This demonstrates the author's solid theoretical foundation, and the content is presented with relative clarity.

### Weaknesses
1. Lack of graph-related experiments. The author mentions equivariant graph neural networks in the title of Section 4, and the proposed method is also aimed at graph neural networks. However, in the experiments in Section 5, the author only conducted experiments related to the prediction of constitutive relations in solid mechanics, with no experiments related to graphs. The author also specifically emphasizes applications on geometric graphs in the contributions, so I believe it is necessary for the author to add relevant content.

2. Contribution is somewhat lacking, and the discussion is not thorough. The author's discussion of $G = SO(3)$ is built upon the structure of $SO(3)$-invariant polynomials, a topic that can be found in [1]. The author rewrites the conclusions regarding invariant polynomials into the tensor network structure of Theorem 3.3. However, after this reformulation, the author does not delve into how this new representation connects different equivariant neural networks or what effective ideas it can bring to network design, which is a topic of interest to many readers. The author should mention the graphical representations of existing equivariant neural networks under this paradigm. Furthermore, based on this paradigm, the author could proceed to discuss the design of new network structures and compare results before and after the modifications. This would make the paper much more substantial.

3. Some imprecise statements and missing explanations. For example, in Section 3.2, lines 179-180, higher-order tensors are mentioned. The author states that for higher-order tensors, these generators would not be minimal. However, these generators include the cross-product symbol, and the definition of a cross product between higher-order tensors is not straightforward. The author should revise the relevant statement. For instance, the notation $(1)^{\otimes l}$ is ambiguous; it should represent a type of tensor or tensor space. But in lines 209-212, where $P_{l}(\boldsymbol{x})$ is a tensor network, the statement that $P_{l}(\boldsymbol{x})$ becomes the tensor $(1)^{\otimes l}$ appears imprecise. Perhaps the author means that the output of $P_{l}(\boldsymbol{x})$ is of type $(1)^{\otimes l}$. As another example, in the tensor network diagrams in Section 3.3, the meaning of the numbers on the edges is unclear. They might indicate that an index from each of the two tensors corresponds to the same representation space, thus allowing contraction. This requires clarification from the author.

4. Suggestion for notational change. In Definition 2.2, the summation indices for the direct sum spaces of both the input and output are $i$. One of them could be changed to $j$.

Overall Assessment: In my personal view, this is an interesting paper that could be extended in meaningful ways. For instance, after representing equivariant neural networks as graphs, one could emulate [2] to perform a search for network architectures to find better-performing models. However, the author's exploration of the problem stops at merely reformulating the network structure. This limits the discussion to a superficial level of representation. As it stands, I believe the paper may not meet the standard for acceptance. If the author were to conduct a more in-depth discussion building on this foundation, I believe this could become a paper with high potential and a significant impact on the community.

Reference: 

[1] Villar, Soledad, et al. "Scalars are universal: Equivariant machine learning, structured like classical physics."

[2] Elsken, Thomas, et al. "Neural architecture search: A survey."

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
