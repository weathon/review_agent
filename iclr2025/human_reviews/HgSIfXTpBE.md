## Human Reviewer 1

### Summary
The authors propose an extension of decision trees and random forests, which typically assume Euclidean space, to mixed-curvature representations using product manifolds based on constant curvature spaces: Euclidean space, hyperspherical space, and hyperbolic space (focusing on the hyperboloid model).

### Strengths
- The paper generalizes decision trees and random forests to mixed-curvature representations which can lead to more expressive/flexible classifiers.
- They leverage ideas from previous papers in deep learning which use mixed-curvature, to improve random forests and decision trees.
- The proposed model may have applications in biology: which I believe is true and the authors mention in the introduction, but they do not really test it in the paper.

### Weaknesses
- It seems that the method suffers from similar issues to those discussed in: Neural Latent Geometry Search: Product Manifold Inference via Gromov-Hausdorff-Informed Bayesian Optimization. How does one select the product manifold signature? This should be discussed as part of the limitations. The paper mentioned above suggests using Bayesian optimization to select the signature, for instance.
- In general section 4 is poorly written, there are a lot of baselines, but it is unclear how DTs and RFs are used in different tasks. It is not clearly explained. Maybe additional explanations about the different tasks in the appendix would be helpful: that is, describing the problem setup rather than mainly focusing on reporting hyperaparameters.
- Table 2 has too many symbols. It is not clearly explained what is meant by Max Ambient and Max Product. Could you please define them in the main text or the caption.
- In table 2, for Cora and Citeseer if you are doing node classification, the performance is surprisingly low. What would be the results in terms of accuracy? In the case of Cora and Citeseer is more typical to see the results in accuracy than F1, so it would be helpful to know what they are to compare for instance to the performance of a GCN.
- In Section 4.4. the authors mention they use variational autoencoders for classification. Do you mean autoencoders? Variational autoencoders are generally used for generative modeling. Or do you mean you use the KL loss to regularize the latent space but then use it for classification. It is unclear how classification is performed: in the regularized latent space of an autoencoder? The performance seems very low too for MNIST, etc. KL regularization may be hurting classification performance, if you remove it the autoencoder has more freedom to encode the samples in the latent space without having to stay Gaussian-like and they are probably easier to classify. Could you provide a step-by-step description of the experimental procedure used please.
- The conclusion section lacks a discussion on limitations. As previously mentioned: it is not obvious how to choose the signature. Also, why use RF when there are other better performing methods for most of the tasks discussed in the experiment section? One could argue that RFs may be easier to implement for less experienced practitioners, but introducing the mixed-curvature representations makes the method more unaccessible to a general audience. Potential points to discuss: 1) Challenges in selecting appropriate signatures, 2) Trade-offs between RFs with mixed-curvature representations versus other high-performing methods, 3) Potential barriers to adoption for practitioners less familiar with non-Euclidean geometries

### Questions
Suggestions:

- It should be made clear in the text that product manifolds based on constant curvature manifolds cannot generate any arbitrary manifold, as part of limitations.

- Relevant works are missing: 

Historical context on manifold learning and machine learning: Algorithms for manifold learning by Cayton, Representation Learning: A Review and New Perspectives by Bengio

Hyperbolic spaces and machine learning: Before product manifolds, there were many works on using hyperbolic spaces which served as predecessor which are worth mentioning, for instance, Poincare embeddings for learning hierarchical representations, Neural embeddings of graphs in hyperbolic space, Hyperbolic entailment cones for learning hierarchical embeddings

Product manifolds and product manifold derived features: Heterogeneous manifolds for curvature-aware graph
 embedding, Neural Spacetimes for DAG Representation Learning

Computationally tractable manifolds beyong constant curvature manifolds: One of the motivations for using product manifolds of Euclidean, hyperbolic, and hyperspherical spaces, is that they allow machine learning algorithms to model data on more general manifolds than Euclidean space, while still providing a closed-form solution for geodesic distances, exponential maps, etc. However, product manifolds of this kind are also limited: it is not possible to model any arbitrary manifold by just using this construction. Other computationally tractable alternatives also exist: Neural Snowflakes: Universal Latent Graph Inference via Trainable Latent Geometries, which is based on fractals, and Computationally tractable riemannian manifolds for graph embeddings, which is based on matrix manifolds.

- Curvature K is not defined in Section 2.1.2 for hyperspherical space.

- In equation 10 the wrong symbol is used for Cartesian product
- In equation 12 the wrong symbol is used for Concatenation

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a method for training decision trees that takes into account a product manifold structure, formed by combining manifolds with constant curvature. A notable technical contribution is the angular reformulation of tree splits. The effectiveness of the proposed method is demonstrated through experiments on both synthetic and real-world datasets.

### Strengths
The paper provides a thorough background knowledge, making it accessible even to readers unfamiliar with geometry and decision trees. Visualization quality is also good. Additionally, the experimental settings described in the Appendix are detailed, and the paper overall does not have unclear sections.

### Weaknesses
While the paper makes a solid contribution, in my understanding, it falls short in terms of significance and novelty. For instance, Section 3, which serves as the core of the proposed method, raises some concerns. Section 3.1 discusses Euclidean decision trees, which is straightforward. Section 3.2 builds on existing research. Although Section 3.3 may introduce a new perspective, the authors themselves describe it as “quite simple”. The choice of does not seem to involve a particularly non-trivial approach. Furthermore, Section 3.4 primarily combines elements from Sections 3.1 through 3.3. While the paper mentions the advantages of redefining decision tree learning through angular comparisons, it is difficult to claim that these benefits are especially significant.

Additionally, although the experiments are well-executed, the impact of the proposed method could be better highlighted by including a more diverse set of baselines for comparison. For example, it would be beneficial to compare the method against non-tree models or conventional decision trees applied to real-world data, such as traffic data, without considering manifold structures. Even if the technical novelty is limited, demonstrating significant experimental results would enhance the paper’s relevance to the machine learning community. However, the current experimental results do not sufficiently showcase the advantages of the proposed method.

### Questions
- I believe it would be helpful to clearly identify the technical differences between your proposed approach and previous research. As I reviewed your method, I struggled to find any significant technical novelty, as indicated in the Weaknesses section. If I have misunderstood something, please kindly let me know. 

- Is it possible to consider oblique decision trees? Specifically, can the method be extended to split not only within the same type of space but also across different spaces with oblique boundaries?

- Regarding the third contribution listed—“We extend techniques for sampling distributions in non-Euclidean manifolds to describe mixtures of Gaussians in product manifolds”—am I correct in understanding that this pertains to the creation of datasets, as described in Section 4.1 and Appendix A, rather than being directly related to learning decision trees on product manifolds? 

- In the Introduction, the intention to use decision trees and random forests is mentioned in the fourth paragraph. What is the motivation behind developing these tree-based models? If the motivation lies in accuracy, I believe a comparison with other non-tree-based models would be necessary. On the other hand, if the choice is driven by factors like interpretability, then it would be beneficial to provide experimental results or analysis supporting this aspect.

### Soundness
3

### Presentation
4

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper introduces an extension of traditional decision tree (DT) and random forest (RF) algorithms, designed to work within product manifolds. These manifolds are constructed as Cartesian products of hyperbolic, spherical, and Euclidean spaces, making them suitable for efficiently capturing and analyzing datasets that exhibit diverse geometric properties.

The authors propose angular reformulations of DTs tailored to the specific geometry of product manifolds, ensuring that splits are geodesically convex, maximize margin separation, and can be effectively combined across different components of the manifold. This approach not only generalizes existing techniques for Euclidean and hyperbolic spaces but also introduces new algorithms for handling data in hyperspherical spaces. Through a series of benchmark evaluations, including classification, regression, and link prediction tasks, the proposed methods demonstrate strong performance, often surpassing existing approaches and achieving top rankings on various synthetic, graph-based, and real-world datasets.

Overall, this work provides a robust framework for utilizing decision trees and random forests in mixed-curvature spaces, enabling more effective analysis and interpretation of data with intricate geometric structures.

### Strengths
The paper presents a novel extension of traditional decision trees (DTs) and random forests (RFs) to mixed-curvature product manifolds. This is a substantial departure from prior work that primarily focuses on Euclidean spaces or, at best, single constant-curvature manifolds (hyperbolic or spherical). By integrating these different spaces, the method captures complex geometric patterns more effectively than existing techniques. The use of angular reformulations and geodesic convexity for decision boundaries is an original approach that leverages the intrinsic properties of non-Euclidean spaces. 

It presents a comprehensive framework that not only generalizes existing Euclidean and hyperbolic methods but also introduces new algorithms for spherical manifolds. This integration is both theoretically and practically novel, offering a unified approach to decision tree learning across mixed-curvature spaces.

The paper is grounded in mathematical principles and the theoretical derivations are detailed along with intuitive explanations.  The authors provide extensive empirical validation of their method on 57 datasets, including synthetic data, graph embeddings, and mixed-curvature variational autoencoder outputs. The benchmarks highlight the method's robustness and versatility across different tasks, such as classification, regression, and link prediction. The performance metrics and statistical significance tests give credibility to the claims made. It also compares the proposed approach against several baselines, including ambient space models, tangent plane methods, and k-nearest neighbors.

The method’s ability to handle data with hierarchical, cyclical, and mixed geometric structures makes it relevant for a wide range of domains.

### Weaknesses
1. One of the main limitations of the proposed method is the computational complexity involved in extending decision trees (DTs) and random forests (RFs) to mixed-curvature product manifolds. The algorithm requires handling several costly geometric operations, such as exponential maps, parallel transport, and angle computations. The paper could be improved by including a detailed complexity analysis and discussing possible optimizations or approximations that might reduce computational overhead.
In addition, given the need for complex manifold operations, the approach may struggle with datasets having millions of points or high-dimensional spaces. Including discussions about scalability and memory usage would strengthen the work.

2. The paper assumes that the underlying product manifolds have constant curvature components. This may not hold for many real-world datasets where curvature varies continuously or where the data distribution does not fit neatly into hyperbolic, spherical, or Euclidean categories. It could be helpful to discuss how deviations from this assumption impact performance and whether there are ways to relax or adapt this assumption.

3. The reliance on tangent plane approximations to map points from the manifold to Euclidean space could introduce inaccuracies, in particular in regions of high curvature. The paper does not quantify or analyze these potential inaccuracies. 

4. The mathematical derivations, notably those involving angle computations and midpoint selection, assume well-behaved data distributions. The method may not perform optimally in edge cases, such as when points are near the boundaries of hyperbolic space or when spherical data are tightly clustered around a pole. Discussing these limitations and potential solutions could help overall.

5. The paper compares its method primarily to other non-Euclidean tree-based models and baseline classifiers. However, deep learning models that can handle non-Euclidean data, such as graph neural networks (GNNs) with Riemannian embeddings, are not considered. A comparison to these models would provide a more comprehensive assessment of the proposed method’s strengths and weaknesses.

### Questions
Q1. The method’s ability to fit complex geometric patterns could lead to overfitting, for example on datasets with noise or less pronounced geometric structures. What potential regularization techniques could you use?

Q2. How does the method handle data distributions that do not perfectly conform to the constant-curvature assumption? Are there scenarios where mixed or varying curvature significantly degrades model performance?

Q3. Given the reliance on tangent plane approximations, how does the method account for distortions introduced in regions of high curvature? Can you quantify the impact of these distortions?

Q4. Are there edge cases where the proofs of geodesic convexity or margin maximization might not hold? If so, how does the method handle such situations?

Q5. The method involves complex geometric operations like exponential maps and parallel transport. How efficient are these operations in practice, in particular for large datasets? 

Q6. How sensitive is the method to hyperparameter choices, such as the variance scale in Gaussian sampling or the number of 2D projections considered for splitting?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper extends DT and RF algorithms to product manifolds, yielding splits that are geodesically convex, maximum-margin, and composable. Extensive experiments demonstrate the effectiveness of the proposed algorithms.

### Strengths
The proposed algorithms apply to constant-curvature manifolds, effectively capturing the mixed and complex structure of real-world datasets.

### Weaknesses
-	Novelty. The proposed algorithms appear to be a simple and direct extension of previous work [1,2], which integrates the notion of product space proposed by [1] with the HyperDT framework [2]. The method is somewhat incremental and lacks enough novelty.
-	Applicability. The framework's current restriction to constant-curvature manifolds presents a significant limitation. The applicability to more general scenarios, particularly stochastic-curvature manifolds, remains an open question that warrants investigation.
-	Comparison. The empirical validation would benefit from a comprehensive comparative analysis against contemporary approaches in the field. Notably absent are performance comparisons with state-of-the-art methods such as HORORF [3] and HyperDT [2].


[1] Tabaghi P, Pan C, Chien E, et al. Linear classifiers in product space forms.

[2] Chlenski P, Turok E, Moretti A, et al. Fast hyperboloid decision tree algorithms.

[3] Doorenbos L, Márquez-Neila P, Sznitman R, et al. Hyperbolic Random Forests.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 5

### Summary
Summary
This paper proposes to extend decision trees (DT) and random forests (RF) to mixed curvature spaces -- meaning that the data (here assumed to be in ambient space $\mathbb{R}^d$) is embedded on a curved manifold, and the random forest or DT works directly on this space. The authors argue indeed that curved manifolds are better suited to represent certain data types (e.g. hyperbolic spaces for taxonomies or phylogeny).  Product manifolds are even better at capturing the complexity of real-world data, and do not confine the analysis to choosing a single curvature, which could increase the potential of these methods in a number of real-life scenarios.

To this end, the paper is organized as follows:
- the authors start by generalizing DT and RF on manifolds with constant curvature. The authors start with a compact review of different Riemannian spaces (including hyperspherical spaces and hyperbolic spaces, which are the ones that have received the most attention), as well as the product manifolds. They then show that the  approach proposed by Chlenski et al for hyperbolic DT can actually be extended to any constant curvature manifold.
- the authors then show how to extend the approach to handle product manifold (i.e. they provide an algorithm).
- they then benchmark their approach on many different simulated and real-world datasets, showing the performance of the method.





*Disclaimer: I am not an expert in non-Euclidean geometry. I have followed the developments in hyperbolic Neural networks from a reasonable distance --- enough to know what they are and their advantages, but not enough to make me an expert on the topic or on related literature. I am happy to discuss with other reviewers and the authors any misconception that might transpire from this review.*

### Strengths
1. Curved manifolds (and product manifolds) have received already som attention from the ML community. Most of these approaches (at least, to my shallow knowledge of the literature) however, have focused on extending deep neural network methods. The authors contend here that an extension of the DT and RF would add a missing brick to the set of available tools for data analysts.

2. **Contribution of the authors:** From what I understand, the contributions of the authors are as follows:
- *Generalization of DT and RF to constant-curvature manifolds* : Chlenski et al seem to have already considered the problem of extending DT to work in hyperbolic spaces. The authors here generalise this approach to constant curvature manifold (not just negative curvature), which allows them to also work on hyperspheres. It also allows to generalize to Euclidean space:this is perhaps less interesting, but it goes to show that they devised a generalized method. The authors show rigorously in the appendix that their approach is equivalent to DT in basis space. The authors also provide explicit characterizations of the splits in different curved manifolds.
- *Generalization of DT and RF to product manifolds* :The authors borrow the idea of product manifolds from Gu et al, but this idea had not been explored in the existing RF and DT literature

3. **Strong experiments**: the authors seemed to have gone to great lengths to benchmark their approach, showing its performance on a variety of regression, classification, and even, link prediction tasks. They also perform ablation studies and test the effect of the different hyperparameters.

### Weaknesses
Overall, I find this to be a strong and very well-written paper.
A couple of criticisms perhaps:
- *Computational Time is not reported*: the authors do not report the compute time that their proposed method requires compared to, say, kNN or the ambient space methods. My guess would be that their method might be a bit more computationally expensive than these. That would be fine though, I think the method is interesting and, even it were considerably more expensive than the other methods, it would still be a worthy contribution that could spark follow up works. I would however encourage the authors to report the time, so that anyone interested in the idea has a sense of the current computational complexity of the method.
- *Motivation: why RF and DT?*: To come back to the motivation of the paper, why should we consider RF and DT and not just go with neural networks, for which there have been extensions and can perform better than RF and DT? I could see RF and DT being better alternatives in the low sample regime, or easier/ faster to train. I don't think the authors compare with any NN based method, if Im not mistaken?

### Questions
1- Could the authors comment on this statement: "To this end, Chlenski et al. (2024) impose homogeneity and sparsity constraints on the hyperplanes they consider for hyperbolic DTs". If I understand correctly, the sparsity constraint imposes that the hyperplanes are defined by only 2 quantities (the timelike coordinate and the x_d, which makes the number of candidate subspaces tractable). However, could the authors explain why homogeneity is an important attribute of these subspaces? If  $\mathbb{P} \cap \mathbb{H}$ was not closed under shortest paths, what would happen?  Or is it just another attribute that is necessary for the thresholding?

2- Interpretability: in the euclidean case, RF and DT are great because they're non linear and more powerful than linear regression, but they're reasonably interpretable too (DT are very interpretable, RF get away with the feature importance sorting). Is there any notion that this could be the case in the curved manifold setting?

3- Parameters: I am a little unclear about how to use the algorithm though:
- how do we choose the product manifold? In the experiments (e.g. MNIST), there is a combination of S, E  and H. How do we go about determining what is the right combination? Is there some selection/tuning involved?
- In figure 7: "feature subsampling approaches in RFs." Is the number of features referring to the standard procedure in the RF algorithm where only a fraction of the samples are seen, to de-correlate trees?
- in figure 7: I don't understand what the caption a is referring to? Which part of the algorithm?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3