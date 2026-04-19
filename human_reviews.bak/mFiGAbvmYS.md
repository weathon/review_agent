# Towards Generalization under Topological Shifts: A Diffusion PDE Perspective

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 5

## Abstract
The capability of generalization is a cornerstone for the success of modern learning systems. For non-Euclidean data that particularly involves topological features, one important aspect neglected by prior studies is how learning-based models generalize under topological shifts. This paper makes steps towards understanding the generalization of graph neural networks operated on varying topologies through the lens of diffusion PDEs. Our analysis first reveals that the upper bound of the generalization error yielded by local diffusion equation models, which are intimately related to message passing over observed structures, would exponentially grow w.r.t. topological shifts. In contrast, extending the diffusion operator to a non-local counterpart that learns latent structures from data can in principle control the generalization error under topological shifts even when the model accommodates observed structures. On top of these results, we propose Advective Diffusion Transformer inspired by advective diffusion equations serving as a physics-inspired continuous model that synthesizes observed and latent structures for graph learning. The model demonstrates superiority in various downstream tasks across information networks, molecular screening and protein interactions.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The work examines the task of generalizing GNNs across different graphs by adopting insights from diffusion PDEs. The authors make theoretical claims about generalization errors under different model settings and use these theoretical insights to design a GNN architecture (ADIT). They then evaluate their method's performance against several baselines under different settings.

### Strengths
1. The paper is clear and easy to understand.
2. Experimental evaluations were conducted carefully with hyperparameter tuning, and details were well documented.

### Weaknesses
1. The theory part contains several flaws. First, the current proof of theorem 3.1 appears incorrect, as U should be unknown. Without conditioning on U, X, A and Y are all dependent, causing the whole proof to fail. Moreover, corollary 3.3 and proposition 3.4 show upper bounds for D_{ood}, while theorem 4.1 considers the lower bound (through explicit constructions). Therefore, these theoretical results are not comparable. Finally, a smaller but still important issue is that the assumption regarding the bijectivity of g is actually very strong. While there may be other potential issues, these are already crucial ones.

2. Even if the theory is correct, it only provides (loose) bounds on the generalization error. It is difficult to determine whether any of these results will hold in practice. The tightness of these bounds could be evaluated using synthetic datasets, if possible.

3. There exists a whole line of work exploring when GNNs perform better than MLPs (answer: not always) [1-4]. These works essentially suggest that concatenating the latent space returned by GNNs and MLPs may lead to better overall performance. As this work also directly considers a hybrid architecture (by combining GNNs and global attentions), one may speculate that the performance gain comes from this combination, rather than the explanation provided in this paper. Additional discussions and experiments should be provided to address this point. The difference between ADIT and GNN + attention should also be highlighted.
4. The experimental results do not convincingly demonstrate ADIT's superiority or support the arguments regarding generalization. For example, I expect that several methods (GCN, GAT, SGC...) would perform well on various synthetic data tasks, and the Diff-nonlocal baseline would achieve optimal performance in DPPIN node regression. Overall, since generalization itself is difficult to define, the authors should consider potential improvements in their real data experiments.


[1] Ma, Yao, et al. "Is Homophily a Necessity for Graph Neural Networks?." International Conference on Learning Representations.

[2] Gomes, Diana, et al. "When Are Graph Neural Networks Better Than Structure-Agnostic Methods?." I Can't Believe It's Not Better Workshop: Understanding Deep Learning Through Empirical Falsification. 2022.

[3] Dong, Mingze, and Yuval Kluger. "Towards understanding and reducing graph structural noise for GNNs." International Conference on Machine Learning. PMLR, 2023.

[4] Luan, Sitao, et al. "Revisiting heterophily for graph neural networks." Advances in neural information processing systems35 (2022): 1362-1375.

### Questions
See above

### Soundness
1

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
3

### Summary
This paper introduces a graph diffusion model motivated by considering the topological shifts between training and testing sets of the data. The structure of the discretized version of their proposed model contains a fully connected global attention and a local diffusion process on the graph. The paper also studied the extrapolation behavior of the diffusion models and quantified the generalization gap, which is sourced from (train/test) distributional differences via graph topology and generation of the node features. The proposed model shows promising results via both synthetic datasets and real-world applications.

### Strengths
1. The paper combines global attention and local diffusion by considering the topological shift problem via GNN training. The notion of topological shift serves as a common concern in both the academic and industrial worlds. 

2. The approximation methods in Section 4.3 for replacing the eigendecomposition of the Laplacian are of interest for the training on different spectral GNNs. 

3. High diversity of the empirical studies for verifying the model's learning capabilities.

### Weaknesses
1. While the authors show that the error bond can be reduced by the proposed model (which is a relatively new aspect), the model architecture, e.g., attention and diffusion, is not novel.

2. The newly proposed model has a closed-form solution (e.g., mentioned in section 4.1), and it is believed that similar results can be achieved by those spectral GNNs since essentially the model is equivalent to the spectral filtering on a fully connected weighted graph plus a sparse graph that originally given, e.g., equation 9 and its closed form solution.  However, lack of comparisons can be found via the empirical studies. 

3. It is found that the theoretical assumption/claims may be hard to be verified via the practical experiments.

### Questions
Few questions at this stage: 

1. In theorem 4.1, the underlying assumption for the theorem to be effective is that $g$ must be bijective, which is a fairly strong condition, and I am not sure how the author verifies this in real practice. 

2. Is there any metric in practice to show that the generalization error in the proposed model is strictly lower than other diffusions? Is learning accuracy always the chosen one? If it is, how could we verify that the accuracy gain is not from the increased complexity of the model? 

3. Similar to the one mentioned above, how do we measure the distribution shift via real practice data? It seems the ways of shifting are mixed, and the advantage of the model via this aspect is only verified via synthetic datasets.

4. Perhaps some additional experiments on heterophilic graphs; the underlying reason for this is the global weighted transformer owns the potential of over-smoothing the node features, and the inclusion of the local diffusion term brings the variation back to the dynamic, and in some cases, both dynamics can shrink the energy out.  Accordingly, the model is expected to prefer a higher $\beta$ value for heterophily and a relatively low $\beta$ for homophily graphs.  

5. In response to the baseline comparison, it is recommended to compare the spectral-based models.

### Soundness
2

### Presentation
3

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
Unfortunately, this is a negative review. However, I think there are some promising ideas and am optimistic that future versions of this paper will be quite good. 

This paper seeks to understand the generalization abilities of GNNs when the testing distribution is significantly different than the training distribution (``open world” vs ``closed world”). As an illustrative example, they consider training a GNN on an arxiv citation network from one year and testing it on the corresponding network from subsequent year (with similar experiments for other real/synthetic data sets. 

The theoretical understanding in this paper builds upon a string of recent works (GRAND, GRAND++, etc) which show that message passing GNNs (if you ignore the ``combine” portion of an ``aggregate-combine” network) are effectively Forward-Euler discretize equations of dispersive PDE on graph domains highlighting the relationship be ``Graph Diffusion” and Graph PDE on Riemannian manifolds. They then propose a data generation model based on Graphons and a latent ``environment.” Next, they prove some theoretical results analyzing the generalization errors of GRAND-like architectures (essentially train an MLP on the solution to the PDE) analyzing the stability to distribution shifts. Their analysis provides upper bounds which are quite loose, and the authors therefore conclude that such architectures are unstable to even small distribution shifts. (I am unconvinced in this inference, as I will discuss below.)

They then propose to address these theoretical limitations by adding in non-local diffusion and an advection term and prove Theorem 4.1 which they claim shows that their modification overcomes the limitations of other models. (I again disagree with this interpretation.) Additionally, they show some promising numerical results.

I think there are some promising ideas here, but there is a lot of inconsistency in the presentation (graphons vs manifolds) and I am unsure if their continuum-to-discrete model of the advection diffusion equation captures the proper spirit. Additionally, I do not think that their theory implies what they claim in implies. Therefore, acceptance is not recommended at this iteration. 

Disclaimer: I did not check the proofs, because I am not recommending acceptance at this point in time, even under the assumption that the proofs are correct. That said, based on the statements, I do believe that the theoretical results are likely correct (from a mathematical standpoint).

### Strengths
The problem is well motivated as is the ``open world” vs ``closed world” framing of the problem.

Going from diffusion to diffusion-advection is an interesting idea. 

Intriguing experimental set up and some good numerical results.

### Weaknesses
The data generation model (Sec 3.1) is based on the implicit assumption that the graph is a finite realization of a graphon. However, the theoretical motivation is based on the assumption that GNNs are effectively PDE on Riemannian manifolds. The paper would be much stronger without this inconsistency, e.g., if the data generation model also assumed the graphs were generated from manifolds. (One could probably replace W with the heat semi-group P=e^{-tL} or something.) 

It's unclear what the implied constants (from the big-O) notation depend on in Theorem 3.1. In particular, if they depend on the Lipschitz constant of the classifier $\phi$ that would significantly weaken the result since this quantity is hard to control.

In, e.g., Proposition 3.2, it is unclear what $\|A\|_2$ means. Is it the Frobenius norm or the $\ell^2$ operator norm? 

The assumption of conditional independence in Prop 3.4 is very strong to the point where it likely renders the result uninteresting. (To be fair, the authors acknowledge this as a potential weakness.) 

Equation 8 does not seem to be the natural generalization of (7) to the graph setting. Instead V should be some sort of ``graph vector field” defined on the edges of the graph (where you would likely need to involve the incidence matrix as well). 

I don’t think the theory implies, from a qualitative point of view, what the authors believe it implies. Yes, the upper bounds in Corollary 3.3 are somewhat weak. However, corresponding lower bounds are needed in order to ensure that this is a real limitation of GRAND-style networks rather than an artifact of the analysis. Indeed, the experiments in Table 3 actually show sublinear deterioration, which seems not so bad.

Additionally, I don’t think that Theorem 4.1 shows that the proposed model addresses these concerns. It’s a ``Universal-Approximation-based” argument which shows that a good model exists. However, I am unconvinced that if a model is trained on distribution 1, it will learn a model that actually works well on distribution 2. Instead, I think it will likely learn the model which is best for distribution 1, which may or may not be the ``good model” guaranteed via universal approximation.

### Questions
Section 3.2 proposes non-local diffusions based on an attention-based graph rewiring? How does this approach compare to utilizing a non-local operation (such as the Fractional Laplacian) on the original graph. (See, e.g., Mesky, 2023, ``A Fractional Graph Laplacian Approach to Oversmoothing”) 

Why is $C$ time-independent in Section 4.1 when it depends on time in section 3.2.2?

Does this proposed method give any insights to the oversmoothing problem?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
1

### Summary
This paper explores how GNNs generalize under topological shifts using diffusion PDEs as a framework. The authors demonstrate that non-local attention, in theory, offers better control over generalization error under such shifts compared to local attention. They support their claims with both theoretical analysis and experimental results. Furthermore, they introduce a Neural-ODE-inspired model, called Advective Diffusion Transformer, which shows superior performance across multiple downstream tasks.

### Strengths
This paper is clearly written, with well-founded assumptions in the theorem. The experiments effectively validate the theoretical claims, providing solid empirical support.

### Weaknesses
This isn't a significant weakness, but the conclusion that extending the diffusion operator to a non-local counterpart improves generalization feels somewhat intuitive. It would be valuable to explore whether the analysis can yield additional, deeper insights beyond this.

### Questions
1. In Fig. 3, the behaviors of diff-linear and diff-multilinear are less consistent on Homophily Shift and Block Shift, but more consistent on Density Shift. Could you please provide your interpretation of this phenomenon and discuss potential implications for the generalization capabilities of these models?

### Soundness
2

### Presentation
2

### Contribution
3
