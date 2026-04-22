# Approximate Equivariance via Projection-Based Regularisation

- Avg Score: 6.50
- Decision: Reject
- Scores: 6, 6, 8, 6

## Abstract
Equivariance is a powerful inductive bias in neural networks, improving generalisation and physical consistency. Recently, however, non-equivariant models have regained attention, due to their better runtime performance and imperfect symmetries that might arise in real-world applications. This has motivated the development of approximately equivariant models that strike a middle ground between respecting symmetries and fitting the data distribution. Existing approaches in this field usually apply sample-based regularisers which depend on data augmentation at training time, incurring a high sample complexity, in particular for continuous groups such as $SO(3)$. This work instead approaches approximate equivariance via a projection-based regulariser which leverages the orthogonal decomposition of linear layers into equivariant and non-equivariant components. In contrast to existing methods, this penalises non-equivariance at an operator level across the full group orbit, rather than point-wise. We present a mathematical framework for computing the non-equivariance penalty exactly and efficiently in both the spatial and spectral domain. In our experiments, our method consistently outperforms prior approximate equivariance approaches in both model performance and efficiency, achieving substantial runtime gains over sample-based regularisers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a projection based regularizer to control the equivariance level of relaxed equivariance architectures.
The key idea is that equivariant linear layers form a subspace of the vector space of all linear weights and that the projection to this subspace is precisely given by the Reynolds operator.
Hence, by considering the decomposition of linear layers' weights into their projections on the equivariant subspace and its orthogonal space, the authors regularize the norms of these two components, effectively controlling the equivariant and non-equivariant behavior of the model during training.

### Strengths
The manuscript is well written and well motivated.
The proposed idea is quite elegant and flexible.

### Weaknesses
While the paper has some new ideas, its novelty with respect to existing methods such as Residual Pathway Priors RPP (Finzi et al., 2021) is not completely clear to me.

Indeed, while the authors here explicitly decompose each linear layer into an equivariant and a complementary non-equivariant component, which are then separately regularized, - if I remember correctly - RPP parameterizes its linear layers as a sum of two separate equivariant and a non-equivariant ones, with an L2 regularization on them. While RPP overparameterizes the network (the equivariant subspace is parameterized twice), the resulting method seems theoretically quite similar since the two linear layers can be merged into a single one with two different regularizations on the equivariant and non-equivariant subspaces.
Am I missing something? Could the authors comment more on this aspect?

Moreover, I think the authors missed a relatively relevant work by (Veekfind & Cesa, 2024) which also studies this decomposition in terms of the Reynolds operator for any steerable CNNs by leveraging the Peter-Weyl theorem in a similar way, although with a slightly different goal.

Veekfind and Cesa, A Probabilistic Approach to Learning the Degree of Equivariance in Steerable CNNs, ICML 2024

### Questions
Lemma 3.2: shouldn't this depend somehow on a Lipschitz constant of T? Also what kind of norm is it used here? This statement seems a bit ambiguous

Do I understand correctly that Eq 9 depends on the norm of $W^{(l)}$ twice, one as explicitly stated in the equation and one, implicitly, via the constant $C$? Is it possible to make this dependence explicit? Is it reasonable to approximate this as a constant?

Sec 4.2: why not applying the proposed method to the standard CNN baseline or the Equivariant baselines too, rather than just the relaxed equivariance methods?

### Soundness
4

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
2

### Summary
The paper proposes a new method for imposing approximate equivariance to a (classical-style, not transfomer-derived) neural network. Put very shortly, the idea is to penalize the difference between an equivariant and a general linear map, and put a penalty term on the difference. An equivariant linear map decomposes in general into a block-diagonal map in a basis corresponding to a irreducible linear representation of the group, while a general map could go beyond this. In the framework of the paper, linear maps are parametrized in the same finite basis of blocks. For the standard case of a 1D Fourier basis (for rotations), this corresponds to only permitting band-limited signals and then penalizing non-equivariance within this subspace of band-limited functions. The result is a conceptually simple scheme to model approximate equivariance of a linear layer of a deep network (which, together with a suitable treatment of nonlinearities, yields an overall approximately equivariant DNN).
The paper also shows some examples of applying the method to 2D image processing tasks, applied to smoke simulation (rotational/translational and scaling symmetry) and CT-slice processing (rotational/translational symmetry). The method works well in practice and seems to be less costly than prior approaches.

### Strengths
The paper is well written and mostly easy to read, despite being rather strict in its formal treatment of the subject matter (I should state that I know the underlying theory only from an application/engineering perspective; I am not an expert in representations of Lie groups or harmonic analysis; so all of my analysis should be read with this caveat in mind).
The basic idea sounds rather straightforward and easy to understand (equivariant maps are a subspace of all linear maps, so one can just penalize "non-equivariant" coefficients in the right basis).
Practical results are limited to 2D and smaller pieces of data (see below), but the method seems to work well and even perform a bit better than competing approaches (again, to note: I am not deeply familiar with the literature, so I cannot really judge novelty; my rating is based on the assumption that the described approach has not yet been tried out in this form). It is also good to see examples beyond E(2) symmetries (including scale-symmetries for fluid dynamics).

### Weaknesses
My impression is that performance is one of the highlighted positives of the method, but it seems to me that the method is still costly. In general, networks employing equivariant linear layers are expensive, as they basically have to sample the whole transformation group with their filters. In the case of this paper, some additional costs occur on the side of parameters: The off-diagonal elements of the linear layer introduce additional degrees of freedom that need to be represented before they can be penalized (although, conceptually, the penalty is computed by projection and subtraction, which, on its own, does not incur costs beyond the standard equivariant linear representation, but the degrees of freedom for violations need to be provided). This might not be a critical problem, as this only affects parameters, not requiring additional sampling of the transformation group, but the overall cost per operation do go up (now quadratically instead of linear in the number of basis vectors, which would be the cut-off frequency and thus sampling density in the Fourier case).

The method is also rather simple (which is also an advantage, see above; I am talking only about its negatives here): Penalizing the difference between an equivariant and a general map seems to be a very straightforward solution. The elaborate exposition makes this a bit difficult to see initially, but in the course of the paper, this is explained very well and clearly. Whether this aspect is a pro or con depends a bit on what background literature is available; I do not feel able to judge this properly, so I would just point out the observation at this point.

The method is also only applied to small 2D data sets (we do not see something like 3D object recognition on ModelNet or the similar), which is likely due to its inherent costs. I would not ask for more, but one could discuss experimental limitations.

One small additional detail: Invariance and equivariance is only considered for the linear layers, as we are dealing with linear representation theory. Thus, Lemma 3.6 only applies to a single linear layer, not a stack of layers with nonlinerities included. Thus, savings are only possible on such a single layer, and a network would probably not use invariant layers at every layer to obtain invariance for expressivity but rather make only the final layer invariant and previous layers equivariant. It might be worth discussing this aspect that might affect practical deployment and utility.

### Questions
It seems to me that the method is inherently more expensive than a classical "GCNN", i.e., a equivariant network based on linear representations in its linear layers. Is this true? If so, can you state the costs (maybe depending on application scenarios; I would guess that projection is still possible in cases where not all coefficients are stored explicitly)?

For the fluid dynamics experiment: Does it use the invariances of the underlying fluid-dynamics equations (Navier-Stokes, I guess; this would be particularly nice), or just general similarity transforms?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a projection-based regularization method to encourage approximate equivariance. This method relies crucially on a work by Elesedy and Zaidi, and involves orthogonally projecting linear operators to the equivariant subspace and penalizing the complement (this is done using the averaging operator). This is in contrast with sample-based penalties which rely on data augmentation over group elements. It is then shown that the proposed penalty is theoretically tied (within a factor of two) to the worst-case equivariance defect. Further, it gives efficient spectral Fourier/Peter–Weyl-based recipes to compute the projection as block-diagonal "masking" plus in-block averaging. This is especially simple for finite groups and steerable layers. Experiments in a toy SO(2) setup, on imperfectly symmetric dynamics, and CT MAR (with C4) show competitive or improved accuracy and better training throughput than sample-based equivariance penalties.

### Strengths
The method basically uses an idea from Elesedy and Zaidi, but it is conceptually clear and simple. It recasts approximate equivariance as operator-space projection instead of sample-space penalties, aligning with representation theory.

The theory is generally nice: It shows equivalence (up to constants) between the projection distance and equivariance defect; this provides per-layer bounds for networks.

Using the Fourier picture gives a clear (and broadly applicable), implementable "mask-and-average" procedure, and closed-form projector for common discrete rotation groups and steerable layers.

The method could be efficient as it avoids extra forward passes and sampling variance. The throughput results seem encouraging. 

The method is also quite general. It could works with non-equivariant architectures by only modifying linear operators, creating a tunable equivariance knob.

### Weaknesses
In the beginning of the paper efficiency and complexity are used as motivational devices, but their scope is not fully quantified. The efficiency claims lack explicit computational complexity and per-epoch wall-clock comparisons across all settings.

The experimental tables sometimes lack explicit metric definition/units and consistent uncertainty reporting (e.g. the dynamics table). Makes it somewhat hard to parse. The breadth of baselines in CT MAR is limited: Only one sample-based method compared. no "train-then-project" or RPP in this domain. Could the authors clarify why? Further, the equivariance defect is not consistently reported: Only the toy example visualizes it. 

The major weakness of the paper in my view is that the experiments are basically reasonable but too "toy-ish" and insufficiently ambitious relative to the theory.​ For the practice around equivariant networks to change, the experiments need to be more ambitious at this point.

### Questions
Do the authors have per-epoch wall-clock and memory comparisons for all experiments (not only throughput) vs. sample-based penalties and vs. no penalty? What is the asymptotic cost of your projection per layer for finite and continuous groups?

Which norms did you actually use for $|| PW ||$ and $|| W - PW ||$ during training? I am curious how sensitive are results to this choice? 

What happens when $\lambda_G = 0$ and $\lambda_{\perp} = 0$ Do you observe any degenerate solutions ?

It would be good to have plots for the defect vs performance. Do the authors have this for any of the experiments?
	​
For SO(3) on point clouds/meshes, how do you compute the projector efficiently? Would you restrict projection to steerable layers only, or do you have a practical recipe for unstructured operators?

Could you include constraint-relaxation (train non-equivariant, project at test) and RPP within the same MAR setup?

I am also curious how sensitive are gains to the degree of symmetry breaking/noise? Any failure cases where projection hurts (e.g., highly non-symmetric regimes)?

I would also like to point out a paper that has a somewhat esoteric operator picture too, but nowhere like the generality of the paper. "Symmetry-Based Structured Matrices for Efficient Approximately Equivariant Networks." (AISTATS 2025) Moreover it only works for discrete groups.

### Soundness
3

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
4

### Summary
This work proposes a projection-based regularization method for learning approximate equivariant networks. Instead of enforcing strict equivariance by design or sample-based penalties that depend on augmented data,  the authors propose a regularizer that operates directly in the parameter space. Specifically, each linear layer $T$ is decomposed, using an orthogonal projection into the equivariant weight space, into the equivariant component P(T) and the non-equivariant residual T-P(T). After the decomposition, they propose to add a regularizer $\lVert T-P(T) \rVert $ that penalizes the non-equivariant part of each layer and depends only on the learned parameters without relying on noisy augmented input samples. The authors show how this regularizer can bound up to a factor of 2 the equivariant error and also describe an efficient implementation in the Fourier domain. Empirically, the method is evaluated in both synthetic and real tasks, showcasing consistent improvements in both task performance and runtime efficiency when compared to a sampled-based regularizer.

### Strengths
- The paper provides a clear formulation of the problem that unifies the equivariance and approximate equivariance under a single operator-level framework. By expressing the regularization in the space of model parameters, the method is independent of input augmentations and data sampling strategies.
- The authors, in addition to proposing the equivariant regularizer, also provide an efficient way to implement it in the Fourier space, allowing for easier adaptation to current pipelines that are limited by resources. This can significantly increase the ability of the method to scale in more complex tasks and architectures.
- The authors perform an extensive evaluation of the method in a large range of tasks, showing how their proposed regularizer can provide general performance benefits that are not task-specific.

### Weaknesses
- There is limited discussion regarding the effect of the introduced hyperparameters $\lambda_G$ ,$\lambda_\perp$ . Since there are two different regularizers that interact both with each other and with the task loss, it would be interesting to see how much the performance gains are sensitive to their values.
- While the authors provide a clear motivation for the $\lVert T-P(T)\rVert$ regularizer, they do not provide any significant insight regarding the $\rVert P(T)\rVert$ term.
- There is no discussion and comparison with the work "A Probabilistic Approach to Learning the Degree of Equivariance in Steerable CNNs" (L.Veefkind and G.Cesa, 2025), which also introduces a similar mechanism that acts directly on the weight space to control the level of equivariance. Discussing conceptual similarities and differences with this work would provide a more complete picture of the related field.

### Questions
- How sensitive is the method to the choice of hyperparameters $\lambda_G$ and $\lambda_\perp$? Do these interact linearly, or does one dominate the other?
- What is the connection of this work with the paper "A Probabilistic Approach to Learning the Degree of Equivariance in Steerable CNNs" which also modulates equivariance at the parameter level?

### Soundness
3

### Presentation
3

### Contribution
3
