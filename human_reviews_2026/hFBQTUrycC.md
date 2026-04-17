# An algebraic approach to approximately equivariant networks

- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Equivariant neural networks incorporate symmetries through group actions, embedding them as an inductive bias to improve performance. Prominent methods learn an equivariant action on the latent space, or design architectures that are equivariant by construction. These approaches often deliver strong empirical results but can involve architecture-specific constraints, large parameter counts, and high computational cost. 
We challenge the paradigm of complex equivariant architectures with a parameter-free approach grounded in representation theory. We prove that for an equivariant encoder over a finite group, the latent space must almost surely contain one copy of the regular representation for each linearly independent data orbit, which we explore with a number of empirical studies. Leveraging this foundational algebraic insight, we impose the regular representation as an inductive bias via an auxiliary loss, adding no learnable parameters. Our extensive evaluation shows that this method matches or outperforms specialized models in several cases, even those for infinite groups.
We further validate our choice of the regular representation through an ablation study, showing it consistently outperforms defining and trivial representation baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper suggests training neural networks using an equivariance loss on an intermediate latent space. Given a priori knowledge of a task symmetry group, a representation of the group can be chosen and used to define the equivariance loss. Furthermore, theoretical and empirical analysis is presented that suggests that for finite groups, using multiple copies of the regular representation is the optimal choice of representation.

In experiments, the suggested approach is shown to work well.

### Strengths
1. The paper contributes empirical evidence that using a pre-chosen group representation in latent space and training a network to be equivariant wrt to it can work just as well as more complicated methods for obtaining (approximate) equivariance.
2. The flow from checking what representation is learned to fixing the representation is natural and well presented.

### Weaknesses
**Missing references:**

Encouraging equivariance by using a fixed representation in latent space is an idea that goes back at least to Cohen & Welling (Transformation Properties of Learned Visual Representations, ICLR 2015), see also Worrall & al (Interpretable transformations
with encoder-decoder networks, ICCV 2017).

Studying the case of a learnable group representation in latent space was done by Bökman & al, albeit only for keypoint description (Steerers: A framework for rotation equivariant keypoint descriptors, CVPR 2024). However, Bökman & al found that the final distribution of eigenvalues in the learned group representation depends on their initialization, which is not tested in the submitted paper.

Notably, the above works also consider infinite groups (by using finite representations), while the submitted paper states that “a direct extension to infinite groups is a non-trivial challenge for future work”.

**Theory:**

Theorem 1 contains an “almost surely”, which requires a probability measure on the space of equivariant functions considered. Such a probability measure is not presented. Even changing to “almost everywhere” or similar would require defining a measure on the function space.

Theorem 1 further only considers injective functions. Neural networks for classification are typically less and less injective the deeper layer one considers (Mahendran & Vedaldi, Understanding Deep Image Representations by Inverting Them, CVPR 2015).

### Questions
1. Does changing the initialization of $\rho_Z$ in Section 4.2 change the conclusions? For instance if $\rho_Z(g)$ is initialized close to the identity?
2. What layer of the network is $Z$? Is it in the middle or towards the end of the network? Do the experimental conclusions change if different layers are considered for $Z$?
3. What is meant by "almost surely" in Theorem 1? I.e. what is the measure considered?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors consider the problem of learning network that respect an equivariance condition. They argue for not confining the architecture to be exactly equivariant, but rather to penalize non-adherence to the equivariance conditions. 

A problem with imposing equivariance by architecturial means in general is the a-priori non-fixed action of the group on the intermediate spaces. The authors argue theoretically as well as empirically that the regular action of the group is a reasonable choice. Using this choice, the authors propose to train an architecture consisting of an encoder E to a latent space Z (on which the group is acting regularly) and then a decoder/classifier/etc. D from Z to the output space, regularizing E to be equivariant with respect to the fixed input action and regular action on Z during training.

### Strengths
The manuscript is easy to read. The main ideas are easy to digest. The main idea, to achieve equivariance through regularization rather than hard constraints, is sound.

### Weaknesses
* Clarity *  While the manuscript as a whole is simple to read, there are many details that remain unclear. 

1. The statement and proof of the main theorem is correct, but I am unsure of its significance. It is only applicable to quite small groups because of the condition $\dim(Z)\geq \abs{G}$ (think about the group of permutation on, say, more than 10 elements). If the action $\alpha_X$ is linear, one can always $\rho_Z=\alpha_X$ and $\calA = X$ and choose $E$ as the identity to obtain an injective, equivariant map, irregardless of whether $X$ contains the regular representation.
 
Also, the interpretation the authors make (the 'key theoretical insight' on page 4) is unclear to me. First, it is unclear what the meaning of a 'linearly independent orbit' is. Does it mean that the spans of the images of the two orbits have a trivial intersection? That they are not contained in each other? It is also unclear how the statement follows from Theorem 1 : While the group acts transitively on each orbit, it does not act transitively on the union of the orbits.

2. In the exploratory experiments, the authors seem to be making a point out of that the irrep counts always exactly corresponds to the number of linearly independent embedded orbits. It is however unclear how this number is measured, also after looking at the code.

3. The authors claim in the introduction that their method leads to lower parameter counts. I do not understand this. In fact, a constrained architecture will in some sense always have a lower number of effective parameters -- the dimension is reduced by the restriction.

See questions on smaller details below.

* Novelty * What the authors ultimately propose is to penalize non-conformance to the equivariance condition both for the network as a whole and the encoder part. This ultimately is very close to simply using augmentations, and is the driving idea behind e.g. residual pathway priors (see also [1] and references therein). If the only novel idea is the use of the regular representation on the latent space, it is limited.

* Experimental validation* The authors test their method in three settings, and they do showcase good performance. However, their method never outperforms their baseline by more than a standard deviation over three runs, which is slightly unconvincing. 


[1] Pertigkiozoglou, S., Chatzipantazis, E., Trivedi, S., & Daniilidis, K. (2024). Improving equivariant model training via constraint relaxation. Advances in Neural Information Processing Systems, 37, 83497-83520.

### Questions
See the questions under weaknesses. Consider also the following more detailed questions below:

1. In the main body of the text, the TMNIST experiments use a latent dimension of 8, but in the appendix and the code supplement, a latent dimension of 6 is claimed. Which is correct?

2. For the MNIST/D3 experiments, the authors say that the digits are augmented by 'arbitrary' rotations. Does this mean that the rotations are random $\mathrm{SO}(2)$-rotations, or something else?  The group is then chosen as $D_3$, which also contains reflections. Can the authors comment why this choice is made (it does not seem to be compatible with the symmetries of the dataset?)

3. In the main text, the rotations building D3 are 120 degrees, but in the appendix, they are specified as rotations of 60 degrees. Which is correct?

4. In Table 4, the wrong number in the Nodule column seems to have been underlined.

5. In the appendix, the inclusion of Figure 7 is confusing,  in that it is never referenced. In which experiment is reorientation applied/not applied?

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
This work proposes a parameter-free approach for promoting equivariance in neural networks by imposing, through an additional loss term, that the intermediate latent representation of the network contains multiples of the regular representations. The authors' main theoretical argument motivating this choice is that, for an injective function equivariant to a finite group, if its codomain is sufficiently large, then it almost surely contains the regular representation. The authors empirically evaluate this claim by first testing whether simple networks trained to be equivariant indeed learn latent representations that contain the regular representation. Additionally, they showcase across various approximate equivariant tasks that promoting this structure in the latent space of an encoder improves performance compared to previous approximate equivariant methods, which typically achieve equivariance through specific network design.

### Strengths
- The simplicity of the proposed method allows it to be clearly presented and easily implemented, which may encourage broader adoption, especially compared to more complex works on approximate equivariant architecture.
- The experimental results provide empirical evidence that unconstrained networks tend to learn latent spaces containing the regular representations, and that explicitly promoting this structure through a loss can improve overall performance. Although, as stated in the weaknesses, these results are limited to relatively small finite groups, the method can still be of interest to the community and may be applicable to a broader range of tasks.

### Weaknesses
- The main weakness of this work is the inconsistencies in the proof of Theorem 1, which serves as the motivating result for the method. In the proof, the authors assume an injective equivariant function $E$ and construct a linear map $\tilde{E}$, represented in matrix form as $M$. They argue that if $\det(M)\not =0$ then the output of the equivariant function contains the regular representation, and that $\det(M)\not=0$ almost surely, since random matrices are almost surely invertible. However, this probabilistic argument is problematic, since $M$ is not a randomly sampled matrix. On the contrary, $M$ is constructed from an optimized/trained function $E$. Thus, the claim that $\det(M)\not=0$ almost surely lacks formal justification and doesn't hold in the current version of the proof. There may exist deeper conditions under which $M$ is generically invertible, but these are not discussed in the current proof.
- The experimental results are limited to small finite groups,  such as $C_4$ and $D_3$. In most cases, the more challenging settings for equivariant and approximately equivariant networks involve larger or continuous groups. There is no clear indication whether the results presented in this paper can scale as the dimensionality of the group increases, which will also result in an increase in the latent representation's dimensionality and potentially its computational cost.

### Questions
- How can we make a probabilistic argument about the determinant of $M$ and, more broadly, about the nature of the representation of the encoder, when $M$ is constructed by the learned network $E$ and not randomly sampled? Is there any other connection that makes $M$ almost surely invertible?
- How does the method scale for larger finite groups where the size of the latent representation also increases? Did the authors observe any tradeoff, or does the regular representation empirically always seem to be the best choice?

### Soundness
2

### Presentation
3

### Contribution
3
