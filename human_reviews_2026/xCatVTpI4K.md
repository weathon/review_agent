# Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Learning conditional distributions $\pi^\star(\cdot|x)$ is a central problem in machine learning, which is typically approached via supervised methods with paired data $(x,y) \sim \pi^\star$. However, acquiring paired data samples is often challenging, especially in problems such as domain translation. This necessitates the development of *semi-supervised* models that utilize both limited paired data and additional unpaired i.i.d. samples $x \sim \pi^\star_x$ and $y \sim \pi^\star_y$ from the marginal distributions. The usage of such combined data is complex and often relies on heuristic approaches. To tackle this issue, we propose a new learning paradigm that integrates both paired and unpaired data **seamlessly** using data likelihood maximization techniques. We demonstrate that our approach also connects intriguingly with inverse entropic optimal transport (OT). This finding allows us to apply recent advances in computational OT to establish an **end-to-end** learning algorithm to get $\pi^\star(\cdot|x)$. In addition, we derive the universal approximation property, demonstrating that our approach can theoretically recover true conditional distributions with arbitrarily small error. Furthermore, we demonstrate through empirical tests that our method effectively learns conditional distributions using paired and unpaired data simultaneously.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new method, based on optimal transport, for semi-supervised learning using inverse entropic optimal transport. In a nutshell, the authors consider the problem of learning a conditional distribution $\pi(\cdot|x)$ given a paired set $\set{x_i, y_i}$ and two unpaired sets $\set{y_j}$ and $\set{x_k}$. This corresponds to the semi-supervised learning case, where one has a smaller set of paired, supervised set, and two sets of unsupervised unpaired data. The authors then draw a relationship between the problem of learning $\pi(\cdot|x)$ and inverse OT. Inverse OT, on its own, is a sub-problem within OT where, instead of finding the OT plan, one finds the ground-cost. The authors demonstrate that their method beat other baselines in an toy example and a real case scenario of weather prediction.

### Strengths
__S1.__ I think the authors do a good job in linking the inverse EOT problem with the semi-supervised domain translation objective.

__S2.__ The use of energy based models is also insightful, and nicely decouples the learning terms involving paired and unpaired data.

__S3.__ I also think the authors do a nice job in devising a practical algorithm for optimizing equation 13.

### Weaknesses
__Weakness 1 (Incremental Novelty).__ While the paper is well motivated, its main contribution seems incremental over __(Mokrov et al., 2024)__. For instance, looking at Algorithm 1 in the main paper, the only difference with respect Algorithm 1 of __(Mokrov et al., 2024)__ is the loss function. Other aspects of this submission, such as,

1. The usage of the Gibbs-Boltzmann parametrization, and,
2. The energy function $E(\cdot|x)$

are the same as in the aforementioned paper. __As a consequence, this submission does not meet the novelty criterion for publication at ICLR__.

__Weakness 2 (Limited Experiments).__ The main paper only contains 2 experiments: a toy example, and a weather prediction task. In terms of scale, the second problem is very limited, as it contains only 692 samples. __As a consequence of this remark, this submission does not meet the significance bar for publication at ICLR.__

- __Side Note.__ Given the similarity with __(Mokrov et al., 2024)__ I think a comparison with their method is warranted.

__Weakness 3 (Too general title, too restrictive setting).__ The title of this paper claims that inverse EOT solves semi-supervised learning. However, upon reading the paper, the semi-supervised setting the authors are referring to is actually semi-supervised domain translation. I think this is an important distinction that must be made in the title of the paper if the authors don't actually experiment with general semi-supervised learning.

# References

__(Mokrov et al., 2024)__ "Energy-guided entropic neural optimal transport." arXiv preprint arXiv:2304.06094 (2023).

### Questions
From the overall discussion of the authors method, I feel it could be applied to semi-supervised learning in general (e.g., in the classificaiton/regression settings). Is that the case?

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
The aim of the paper is to learn an unknown conditional distribution in a semi-supervised manner,
where both paired and unpaired training samples from the joint and marginal distributions are available.
This is a well-studied problem in the literature,
for which there exist several numerical algorithms.
The goal of the paper is to address this problem in a novel manner
by establishing a connection to the so-called inverse entropic optimal transport.
This is done using specific models and parametrizations
of the unknown conditional distribution, which finally is modeled as Gaussian mixture.
The derived algorithm exploits the connection to optimal transport and employs efficient methods from this field.
The presented approach is compared to other methods in two numerical experiments.

### Strengths
Overall,
  the theoretical part of the paper is well-developed and nicely written.
  The problem is well explained and motivated.
  The related literature and algorithms are comprehensively reviewed,
  embedding the paper and its approach in the broader fields of machine learning and optimal transport.
 The employed model of the unknown conditional distribution
  and the relation to inverse entropic optimal transport,
  which is one of the main contributions,
  is well presented.
  Besides the brief calculations in the main text,
  the detailed rearrangements are worked out in the appendix,
  making the paper also accessible for non-expert readers.
 The presented relation between semi-supervised domain translation
  and inverse entropic optimal transport is
   interesting.

### Weaknesses
- Without Appendix C.3 and D.1,   the experimental illustrations in §5 are extremely hard to follow.
  Since the information in these appendices is essential,
  they should be briefly included in the main text
  to make §5 self-contained. 
- The first example (§5.1) deals with the approximation
  of an synthetic conditional distribution.
  At first glance,   it seems that the goal is to estimate optimal transport plan,
  which in fact is not entirely true.
  The construction of the *ground truth* should be more highlighted,
  especially why the conditional distribution spread out
  and that the plan $\pi^*$ is not an optimal one.
  This would ease the interpretation of the results.
- The second example (§5.2) deals with real-world weather data.
  Although based on real-world data,
  the experiment seems to be highly synthetic.
  More details regarding the dataset,
  like the considered measurement locations (local or world-wide),
  as well as a motivation of the preprocessing step are missing.
  This information however could be helpful to understand
  the aim of the example and the relation to actual applications.
- Some references may be added concerning 
conditional generative models like Hagemann et al. ICLR 2024 or
Ardizzone et al., 2019 ...
and
concerning inverse entropic OT and related metric learning like 
Huizing, Cantini and Peyr\'e, ICLR 2022,
Auffenberg et al., Unsupervised Ground Metric Learning, 2025.
- My main concern is the improvement of the numerical presentation.

### Questions
- How exactly the log-likelihood values in Table 1 can be interpreted?
More precisely,
how are these values are used
to evaluate the quality of the estimated conditional distribution
and why are they interesting?
How does Table 1 looks like if the CFD,
which is easier to interpret,
is used instead?
- To improve the presentation of the first numerical illustration: could
  the ground truth and the employed data be moved to
  a separate figure that
  is presented before the results?
  At present   data and results are mixed
  which reduces readability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed a semi-supervised algorithm using inversed OT. The author showed that the formulation is related to the likelihood maximization of an energy-based model. The universal approximation property is derived and some numerical experiments are conducted.

### Strengths
This paper showed that the entropy-regularized inverse OT problem can be formulated as a likelihood maximization problem of an energy-based model. This is an interesting result. The universal approximation property is derived, showing the soundness of the method.

### Weaknesses
1. One limitation is that this formulation requires that the marginal of the paired data also follows $\pi_x$ and $\pi_y$. For example, if the paired data is artificially selected, i.e., they do not follow $\pi_x$ and $\pi_y$, then the method no longer works: the first term in Eqn (18) is no longer an approximation of the first term in Eqn (13). I suggest making this clearer in the paper.

2. Clearness: There are too many bold, italic, underlined words throughout the paper, even in the abstract. Many unimportant words like "sequence of", "proofs", "seamlessly" and "extended discussion" are underlined. Such emphasis adds no novelty to the paper and makes it rather difficult to read. My suggestion is to reduce the use of them.  I also suggest refraining from using the objective words like "fancy".

3. The author discussed the use of neural networks, which can be more useful in practice. However, this is hidden in the appendix. I suggest moving it to the main paper, and moving the discussion of Gaussian mixture models to the appendix.


4. All experiments are in small scale. Some larger scale experiments like those in Gu et al., 2022 should be considered.

### Questions
I do not have further questions.

### Soundness
3

### Presentation
2

### Contribution
2
