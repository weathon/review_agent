# Discovering Lie Groups with Flow Matching

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Symmetry is fundamental to understanding physical systems, and at the same time, can improve performance and sample efficiency in machine learning. Both pursuits require knowledge of the underlying symmetries in data. To address this, we propose learning symmetries directly from data via flow matching on Lie groups. We formulate symmetry discovery as learning a distribution over a larger hypothesis group, such that the learned distribution matches the symmetries observed in data. Relative to previous works, our method, LieFlow, is more flexible in terms of the types of groups it can discover and requires fewer assumptions.  Experiments on 2D and 3D point clouds demonstrate the successful discovery of discrete groups, including reflections by flow matching over the complex domain. We identify a key challenge where the symmetric arrangement of the target modes causes "last-minute convergence," where samples remain stationary until relatively late in the flow, and introduce a novel interpolation scheme for flow matching for symmetry discovery.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper seeks to identify the subgroup which preserves a distribution of data, given a matrix group acting on the data space. This task is accomplished by means of flow matching on Lie groups. Given a parametric family of group elements, a prior distribution is assumed on the group, and the flow matching task maps the prior distribution to a new distribution, which new distribution is representative of the subgroup of distribution-preserving group elements (under the group action).

### Strengths
1. This appears to be a novel approach for symmetry detection and it may be well-suited for discovering subgroups that preserve data distributions.
2. This work demonstrates significant potential, though the weaknesses must be addressed. But this seems like a promising research direction that I would like to see developed.

### Weaknesses
1. *Limited experimental comparison.* I see no experimental comparison with existing methods of symmetry detection. Such comparison is crucial so that readers can understand whether the novel method performs better than existing methods at similar tasks. As scalability is beginning to be a topic of interest in symmetry detection, it seems that a comparison of running times is also appropriate.

2. *Experimental results do not validate some of the authors' claims.* Around line 175, it is said that $SO(d)$ and $GL(d)$ are certainly feasible. But there is no theoretical or empirical justification for this statement. In fact, the notion that the method struggles when the symmetry group increases in size seems to cast doubt on the ability of the method to generalize well to higher dimensions, where the subgroups will consist of many more elements. Additionally, the authors claim that their method is applicable to continuous symmetries, as one only need look for an output distribution which is "spread." This seems to open up a can of worms relating to approximate symmetries, which could potentially be addressed with a simple 3d experiment--perhaps when the point cloud takes a familiar shape, such as a paraboloid or (even better due to mixed symmetry types) a hyperboloid of two sheets.

3. *The writing is unclear at times.* Though it appears that there are typos throughout, there are potentially more serious writing issues that disable proper ingestion of the methodology. (some of these moments may arise from a misunderstanding on my part.) For example, in the *training* portion of 4.2, it is not clear what is meant by the statement "$x_0$ is now a sample from the prior and lives in the group $G$:" how can this be, given that $x_0$ is defined as $g x_1$, being a transformation away from $x_1$ which was sampled from the data distribution? Next, the authors claim to predict $A$ given $x_t$: and yet, in algorithm 1, $A$ is directly calculated from $g$, and it appears that $v_{\theta}(x_t,t)$ is being predicted. Algorithm 2 may also be missing a line.

4. *The topic of flows and vector fields is not entirely precise.* A one-parameter flow arises from a time-*independent* vector field, not a time-*dependent* vector field. The time-*dependent* case introduces an additional parameter *beyond the flow parameter*, so that the "flow" for a time-*dependent* vector field should actually have two parameters attributed to it. As constructed, the map $\phi_t(x)$, likely gotten by evaluating the flow parameter at $0$, is actually not a flow and is better described as an evolution operator.

5. *Missing some related work.* There are additional works that should be considered, owing in particular to this work's alleged tie between flows and symmetry: "Symmetry Discovery Beyond Affine Transformations" (Shaw et al., 2024), "Learning Infinitesimal Generators of Continuous Symmetries from Data" (Ko et al., 2024), and "A Unified Framework to Enforce, Discover, and Promote Symmetry in Machine Learning" (Otto et al., 2023).

6. *Sub-optimal presentation.* The figures are hard to read (font size). There appear to be typos and other oddities: lines 59-61 may be missing content, line 81 should have "at the identity" instead of "to the identity", typo around line 161, etc.

### Questions
1. I don't quite understand how the distribution-preserving group elements $h$ are distinguished from the others. This may be related to weakness 3, where I felt the writing was unclear around the discussion of the algorithms.

2. There has been some level of discussion in the community about not merely distribution-preserving transformations, but also structure-preserving transformations. That is, group actions that preserve the manifold structure of the data, not necessarily the way in which the data is distributed on the manifold. Would this method be adaptable to the manifold-preserving case?

3. My other questions relate to the weaknesses noted above: (a) time to compute symmetries; (b) experiments involving continuous symmetry; (c) comparison with other methods.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a method for learning subgroups of Lie groups from examples transformed by random elements of the subgroup. It uses flow matching on manifolds, the larger (superset) Lie group being the manifold in this case, to model the problem: The subgroup samples form a distribution on the larger (superset) Lie group, which can be identified by learning a corresponding generative model. When applying the method to shapes, it uses and requires (to my understanding) models with corresponding vertices which are geometrically identical up to transformations from the symmetry group and possibly small amounts of noise.
In experiments, the method is able to recover subgroups such as 4-fold reflection/rotation groups of larger sets of 2D and 3D transformations (SO(), GL(d)). The method begins to struggle with larger groups, such as octahedral discrete symmetry, where the dense coverage seems to lead to convergence problems. A final section gives an intuitively sound explanation in that attraction to local basins develops only slowly and accelerates quickly in the final training steps due to the initially (likely to be) symmetric attraction. This can be fixed by adapting the time scheduling of flow matching.

### Strengths
Discovery of symmetry and invariance in data is a very interesting and potentially fundamental problem in data science and machine learning; as such, I would see a high "prior likelihood" of impact for any progress on this matter (I could also say, I like the problem the paper is addressing).
The idea of trying to model the issue as just a density reconstruction / generative learning problem on a suitable space of parameters - transformations in a Lie group in this case - appears to be interesting and promising (although it seems that there has been some related work).
The concrete implementation using Lie groups and their Lie algebras for spanning the space and discovering regularity is quite elegant, and pulling the problem into the data domain by transforming the geometry accordingly seems to be the main technical idea needed to implement the rather abstract idea. This seems simple, straightforward and not too difficult to implement, all of which are in principle a big advantage.
I like the last part concerning the "last minute convergence" phenomenon (symmetric attraction being broken late in optimization) in particular: It does not only discover and honestly disclose one of the numerical issues facing the approach but also provides an interesting insight on when generative models in the style of flow-matching run into issues, which is probably interesting on its own as a case study of invariance being harmful to training, which might appear unintuitive at first sight. The paper also provides a strategy that can alleviate the issues.
Finally, I would also like to mention that I would think that the paper is very well written and enjoyable to read (it would, nonetheless, be useful to point out the main workings behind the formalism, which focuses mainly on the workings of the Lie-groups; in particular, one could discuss a bit more the geometry aspect, i.e., what kind of training data is expected and how data items are related to each other).

### Weaknesses
My main concern is that the method is only addressing a rather simple scenario: Having only one instance of the symmetric object given, which is replicated in symmetric copies with, as far as I understand, full vertex-to-vertex correspondence leads to a rather simple scenario. One could probably discover discrete symmetries by simple clustering (maybe meanshift in vertex-space) and continuous symmetry by shape matching (meaning here: computing relative transformations from corresponding vertices in the larger group) and applying PCA to the log-Matrices (in their local frame). I do not see where the full "power" of representation learning with deep networks is required or exploited. Or put differently: The method would probably not be able to discover that a randomly rotated sample of ModelNet-10 is frame invariant from looking at the examples, and, in my understanding, will certainly fail if there are not many copies of the same model included in the training data.
If true (I might misunderstand the approach or overinterpret the experiments - this would be a point to address in the discussion phase, in case), this does not mean that there is no value in the proposed ideas and analysis. I found the basic idea interesting, and the findings about the "last-minute" issues, which arise already in a simple scenario, quite insightful and potentially influential for building a more complex method that can handle strong invariance to geometry. I think that the paper makes an interesting point with valuable ideas but the overall contribution seems to be a bit too small for a top venue.

### Questions
Do you need (vertex-to-vertex) correspondences among the pieces of training data? In general, how much potential would there be for (more) invariance towards geometric variability mixed with varying transformations?

Also, in case I misunderstood the overall workings and idea, please correct me.

### Soundness
3

### Presentation
3

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
The authors propose training a flow matching model on Lie groups as a means of detecting symmetries within a dataset. Specifically, they train a flow matching model with data domain as a Lie group $G$, and assume the to-be-detected symmetry group $H$ is a subgroup of $G$, and symmetry is detected by examining if the learned distribution concentrates around a known symmetry group. To train via flow matching, LieFlow relies on computing exponential and logarithmic maps for interpolating training inputs and constructing training targets. The network is trained to predict tangent vectors (elements of the Lie algebra). Sampling relies on integrating the learned vector field with many small exponential maps. Experiments show that LieFlow can detect simple, low-order symmetries, but struggles with higher symmetry groups such as tetrahedral, octahedral, and icosahedral groups. The main challenge is found to be "last-minute mode convergence", which the authors analyze in detail, and partially overcome with a proposed power distribution timestep schedule.

### Strengths
To my knowledge, the idea of training a flow matching model as a means of detecting symmetries is novel.

The presentation of the background and LieFlow's formalism is elegant. The authors do a good job of contextualizing their work in the symmetry-detection literature.

Experiments are applied to novel toy problems, especially in learning a generative model on $GL(2, \mathbb{C})$.

The authors carefully analyze the "last-minute mode convergence" problem and provide insight with informative and visually appealing figures.

The paper proposes a power distribution time schedule to partially overcome the "last-minute mode convergence" problem.

### Weaknesses
I believe the direction is promising, but the paper could use more rounds of improvement.

A noticeable gap is that the paper does not make the connection of "last-minute mode convergence" with the pathologies of Riemannian flow matching (RFM) with geodesics on compact spaces. These pathologies are that the target vector field is discontinuous at the cut locus ([the surface/poles where geodesics meet](http://www2.mat.dtu.dk/info/mathematics/EmMa/cut_locus/clgif.html)), and the support of the probability path vanishes over time. This phenomenon has been reported in multiple works:
- Figures 2 & 3 in [1] demonstrate that RFM has difficulty learning complicated distributions on the 2D torus, echoed in Figure 4 of [2].
- Appendix G.2 of [3] shows that learning vector fields of geodesics on SO(3) is numerically unstable due to the sharp boundaries of the shrinking uniform distribution.
- Naive flow matching on the simplex results in learning a discontinuous vector field, while the support of the probability distribution vanishes over time. [4]

Figures 6&7 of LieFlow find similar conclusions to the above works, but the paper does not discuss these parallels. Had the authors been aware of this connection, they would know that Riemannian diffusion does not suffer from these pathologies. However, Riemannian diffusion is only briefly mentioned in the paper when citing [2], and [1] is not cited by the authors.

[1] Lou, A., Xu, M., Farris, A., & Ermon, S. (2023). Scaling Riemannian diffusion models. Advances in Neural Information Processing Systems, 36, 80291-80305.

[2] Yuchen Zhu, Tianrong Chen, Lingkai Kong, Evangelos A Theodorou, and Molei Tao. Trivialized momentum facilitates diffusion generative modeling on lie groups. arXiv preprint arXiv:2405.16381, 2024.

[3] Holderrieth, P., Havasi, M., Yim, J., Shaul, N., Gat, I., Jaakkola, T., ... & Lipman, Y. (2024). Generator matching: Generative modeling with arbitrary markov processes. arXiv preprint arXiv:2410.20587.

[4] Stark, H., Jing, B., Wang, C., Corso, G., Berger, B., Barzilay, R., & Jaakkola, T. (2024). Dirichlet flow matching with applications to dna sequence design. arXiv preprint arXiv:2402.05841.

Due to these pathologies, the performance of LieFlow is handicapped, limiting experiments to toy datasets and diminishing the impact of the work. On toy datasets, it is straightforward to find relative transformations between all data points and detect symmetries in that way. Because experiments are only performed on toy datasets, LieFlow is unable to demonstrate its potential advantages for detecting unknown symmetries in real-world data.

The actual detection of symmetry needs more explanation. It is easy to check if LieFlow's learned distribution concentrates around an intended symmetry group, but how will symmetries be detected when LieFlow is applied to real-world data, where the target symmetry group is a priori unknown?

The authors also need to explain how the framework of LieFlow will be able to detect soft symmetries in real-world data. The authors mention valency and electronegativity as symmetries in materials chemistry, as well as local symmetries in images, but it is unclear to me how LieFlow will be able to detect these symmetries.

There should be at least one experiment dedicated to an application with potential practical use, such as a real-world dataset of 3D molecules or point-clouds of 3D objects. One potential application is to rediscover torsional symmetries of 3D molecules.

### Questions
1. Given that Riemannian flow matching with geodesic paths on compact spaces suffers from pathologies, whereas Riemannian diffusion does not suffer from these problems, what is the justification for training a flow matching model rather than a diffusion model, especially in the context of recently developed efficient methods for Riemannian diffusion [1,2]?
2. How will soft symmetries, such as local image symmetries, or valency and electronegativity, be detected within the framework of LieFlow?
3. What is the motivation for detecting symmetries by training a generative model, versus detecting symmetries by directly checking whether the data satisfies transformations of certain groups?
4. Why is it necessary to choose a restricted hypothesis group? What stops us from starting with (if I understand correctly) the most general hypothesis group of $GL(n, \mathbb{C})$ for transforming $n$-dimensional data?
5. What is the simplest but practical real-world dataset that LieFlow could be tested on? What practical situations require detection of discrete groups?

nit-picking:
1. There's an extra "Text" element in Figure 1.
2. extra newline at line 60
3. line 413: "that C4 symmetry in clearly identified." - "in" should be "is"

[1] Yuchen Zhu, Tianrong Chen, Lingkai Kong, Evangelos A Theodorou, and Molei Tao. Trivialized momentum facilitates diffusion generative modeling on lie groups. arXiv preprint arXiv:2405.16381, 2024.

[2] Mangoubi, O., He, N., & Vishnoi, N. K. (2025). Efficient Diffusion Models for Symmetric Manifolds. arXiv preprint arXiv:2505.21640.

### Soundness
2

### Presentation
3

### Contribution
2
