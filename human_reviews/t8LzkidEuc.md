# Failure-Proof Non-Contrastive Self-Supervised Learning

- Decision: Reject
- Scores: 5, 6, 5, 6, 6, 6

## Abstract
We identify sufficient conditions to avoid known failure modes, including representation, dimensional, cluster and intracluster collapses, occurring in non-contrastive self-supervised learning. Based on these findings, we propose a principled design for the projector and loss function. We theoretically demonstrate that this design introduces an inductive bias that promotes learning representations that are both decorrelated and clustered without explicit enforcing these properties and leading to improved generalization. To the best of our knowledge, this is the first solution that achieves robust training with respect to these failure modes while guaranteeing enhanced generalization performance in downstream tasks. We validate our theoretical findings on image datasets including SVHN, CIFAR10, CIFAR100 and ImageNet-100, and show that our solution, dubbed FALCON, outperforms existing feature decorrelation and cluster-based self-supervised learning methods in terms of generalization to clustering and linear classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposed a method named FALCON to resolve known failure mode in non contrastive learning. Theoretically, the obtained representation are both decorrelated and clustered. They demonstrate the effectiveness of the method in several datasets.

### Strengths
1. This work introduces a novel approach to address known failure modes in non-contrastive learning. By identifying and tackling these specific limitations, the proposed method offers a valuable contribution to advancing the stability and accuracy of non-contrastive learning models.
2. The decision to draw W from a Rademacher distribution in the proposed method is a key innovation that enhances the FALCON loss. This modification may contribute to the regularization effect, thereby improving model generalization.

### Weaknesses
1. It is unclear how to select the hyperparameter $\beta$ in the loss function, which controls the contribution of specific components within the loss. Does optimal $\beta$ vary significantly across different datasets?
2. According to Lemma 1, we cannot infer that FALCON objective guarantee to avoid representation and cluster collapses. Lemma 1 shows that FALCON loss has a lower bound. The existence of a lower bound means that the loss cannot decrease indefinitely; however, it does not ensure that the model's learned representations will avoid collapsing to trivial solutions. Can the authors derive an upper bound for FALCON loss?
3. In the experiments, the accuracy of FALCON can outperform several methods but contrastive learning method can get better performances in these dataset. It is hard to convince readers to use FALCON loss. For example, a work: Revisiting a kNN-based Image Classification
System with High-capacity Storage (Kengo Nakata, et al.)

### Questions
1. One know failure mode is dimensional. How can FALCON scale up with high dimensional data? Especially how it can outperforms other methods.

### Soundness
2

### Presentation
2

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
This paper pinpoints conditions to prevent common failures in non-contrastive self-supervised learning, such as representation and dimensional collapses. Using these insights, this paper develops a structured approach for the projector and loss function. The design inherently encourages learning representations that are both decorrelated and clustered, enhancing generalization without explicitly enforcing these traits. This appears to be the first method to ensure robust training against these failures while also improving generalization in subsequent tasks.

### Strengths
1. The writing logic of this paper is very clear, and the introduction part concisely explains the existing problems and the contributions of this paper.
2. This paper is the first to propose sufficient conditions to address the dimension and cluster collapse issues in non-contrastive self-supervised learning, making certain theoretical contributions to this field.
3. The figures are simple and clear, without unnecessary flourishes, which aid in understanding the FALCON method proposed in this paper.
4. The design of the loss function in Eq.2 is rational, and the role of the two terms in Eq.2 is clearly described. Moreover, based on the properties of Rademacher distribution, utilizing Rademacher distribution to avoid collapse is consistent with general intuition. Lemma 1, which concludes the existence and boundedness of global minima, also demonstrates the theoretical effectiveness of FALCON in avoiding collapse.
5. This paper presents a simple method that improves upon the state-of-the-art and is supported by solid theoretical foundations. It is a substantial and valuable contribution. The reviewer look forward to seeing the follow-up research mentioned in the conclusion by the authors in the future.

### Weaknesses
1. The author didn't provide source code in the supplement material.
Please refer to the question part.

### Questions
1. As the author mentioned, this paper proposes sufficient conditions to prevent potential mode collapses. The reviewer is concerned about one issue: if further considering the sufficient and necessary conditions to solve the collapse problem, would it be more helpful for future model design in this field? For example, it might be useful to consider narrowing down the function class where the projector resides and exploring what properties of the function class can prevent mode collapse.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposed a non-contrastive self-supervised learning method that avoids critical failure modes.
1. It introduces a specially designed projector and loss function that decorrelates and clusters embeddings without ssl heuristics.
2. Provides theoretical guarantees for failure-proof training.
3. The method outperforms existing SSL methods on SVHN, CIFAR-10, CIFAR-100 and ImageNet-100 for discriminative tasks.

### Strengths
FALCON shows a theoretically grounded approach to non-contrastive SSL that avoids common collapse issues without standard heuristics. The paper is clear and well-organized, with intuitive figures and thorough validation across multiple datasets. It is the first paper that systematically deals with various failure modes and strengthens generalization at the same time.

### Weaknesses
Scalability Concerns: FALCON is not evaluated on the full ImageNet dataset, which limits comparison with standard ssl benchmarks. Running experiments on ImageNet would provide a stronger validation of FALCON’s scalability and practical performance.

Limited Training Duration: Most experiments are conducted over 100 epochs, which may not be sufficient for ssl methods to fully converge. Extending the training period could offer a more robust assessment of FALCON’s potential.

Similarity to Prior Work: The method shows resemblance to prior approaches like [1], particularly in using a uniform prior to prevent collapse (see Section 3.1 of the paper, “Why Degenerate Solutions are Avoided”). A clearer explanation of how FALCON differs, alongside comparisons with results from these prior methods, would strengthen the novelty claim.

[1] Ji, Xu, Joao F. Henriques, and Andrea Vedaldi. "Invariant information clustering for unsupervised image classification and segmentation." Proceedings of the IEEE/CVF international conference on computer vision. 2019.

### Questions
One common failure mode in discriminative self-supervised learning is the tendency for models to collapse to the color histogram of the input image, as addressed in SimCLR (see fig.6) and many subsequent works through the use of heavy data augmentation [1,2,3]. Have the authors considered such failure modes in their approach, and does FALCON have the potential to reduce reliance on data augmentations to avoid these issues?

[1] Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020.
[2] Caron, Mathilde, et al. "Unsupervised learning of visual features by contrasting cluster assignments." Advances in neural information processing systems 33 (2020): 9912-9924.
[3] He, Kaiming, et al. "Momentum contrast for unsupervised visual representation learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

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
The paper analyzes the failure modes of self-supervised learning—representation, dimensional, cluster, and intracluster collapses—and proposes a principled approach that is robust to these failure modes. The proposed approach, FALCON, maps feature representations to a frozen dictionary that follows a uniform distribution. The self-superivised learning loss is comprised of a invariance loss and a prior matching loss against the uniform distribution. Empirical results demonstrate competitive down-stream performance on SVHN, CIFAR-10, and CIFAR-100 datasets. Empirical results also demonstrate the benefits of increasing the size of the dictionary and training is more stable.

### Strengths
The proposed approach for self-supervised learning is simple and backed by a principled analysis that guarantees the proposed approach is robust to representation, dimensional, cluster, and intracluster collapses. I liked the idea of imposing a large uniform dictionary codes to facilitate representation learning, which brings together cluster-based SSL and feature decorrelation.

The empirical results show that the proposed FALCON with overcomplete dictionaries outperform all the baseline approaches and is more robust to collapses. The paper also carries out ablation to demonstrate the positive impact of increasing dictionary sizes.

### Weaknesses
Overall, I liked this paper from the motivation to the conclusions. It is unclear how to determine the hyperparameter \beta that balances the consistency and prior-matching loss and if the learning is robust to this hyperparameter. Also unclear if this approach would scale well with larger models.

### Questions
1. How to determine the hyperparameter \beta? Is learning robust to different values of \beta?
2. How would this approach scale with larger models? Would the size of dictionary create computational burden if c >> f?

-- 
Post-rebuttal:
I want to thank the authors for their response; I maintain my original rating.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a new non-contrastive loss function as well as projector layer design, motivated by theory to avoid the known problems of representation collapse, dimensional collapse and intra-cluster collapse for these methods. Previous methods have provided more complicated losses (e.g. generative loss) or heuristics (e.g. stop gradient, momentum) to achieve this, but this paper proposes an alternative by modifying the projector layer and the loss function. The theoretical contributions are backed with experiments to show that the proposed method FALCON can outperform prior methods in generalization while avoiding the known problems in a principled manner.

### Strengths
- The problem is very well motivated. It is true that non-contrastive methods have been relying on heuristic solutions to deal with the different types of collapse problems 
- The paper provided theoretical justification for the FALCON loss and projector design
- This is also supported by experiments showing not just the performance, but the distribution of representations to show how the method avoids collapse.

### Weaknesses
Personally, I found the theory section extremely difficult to parse carefully. There is far too many technical expressions before the problem is clearly formulated and insufficient explanation to understand the key ideas of the proof.  

As an example, the presentation of Lemma 1 is inscrutable. It is hard to know 1) what does the extrema condition intuitively mean? 2) On line 227, the authors ask the reader to note that the invariance loss can be decomposed into a "sum of entropy" term and a "KL-divergence" term -> this is not easy to do without the authors showing this decomposition. 

Overall, I feel this paper can benefit greatly from rewriting to present the technical content in a more digestible manner.

### Questions
Covered in weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces FALCON, a clustering based self-supervised approach that tackles various kinds of collapses through architectural or loss related changes. This allows it to avoid classical collapsing behaviour (total/dimensional) but also collapses related to clustering.
The author demonstrate that this non-collapsing behaviour is avoided both theoretically and empirically, proving that these problems are indeed alleviated.
This work thus extends dimensional collapse analysis to clustering based methods while proposing a method avoiding all of these collapsing behaviours, leading to increased performance.

### Strengths
- The in depth analysis of clustering related collapse is very interesting as cluster based methods have gained in popularity recently (e.g. DINOv2). This extends previous work on dimensional collapse focusing primarily on dimensional collapse.
- Having both the theoretical and empirical evidence that FALCON effectively avoids diverse kinds of collapse leads to a convincing approach.
- FALCON demonstrates improved performance over the baseline methods considered (with some caveats, see weaknesses)

### Weaknesses
The experiments appear a bit weak to support the claims of improved performance.

For the experiments, the use of ResNet-8 leads to extremely low performance for all methods considered (At least for CIFAR-10/100). Comparing to a more standard ResNet-18 (see https://github.com/vturrisi/solo-learn?tab=readme-ov-file#cifar-10 for performance for various methods) we see gaps of 20-30 points. It’s thus unclear how these results may transfer to more widely used architectures.The same analysis on ResNet-18 backbones would be more convincing.
For ImageNet-100, it would also be beneficial to add similar baselines as Table 1 (e.g. Barlow Twins, Swav) to understand how FALCON relates to other methods in this larger scale setting.

### Questions
- Lines 169-170, key components are described as a BN then L2-norm. This seems very similar to doing one step of the Sinkhorn-Knopp algorithm. Was this the motivation for the predictor design ? It may be worth discussing this link more to give an intuition as to how this design helps avoid cluster collapse (due to the sinkhorn knopp algortihm helping to spread cluster assignments)

- Regarding the description of collapse types, a type that exists in methods such as DINO is that some prototypes become identical, and thus we get a kind of “prototype collapse” (see appendix C in [1]), where even if we define $n$ prototypes, there are only $n<m$ in practice. This is directly visible as it is identical to dimensional collapse for clustering methods (since each dimension after clustering corresponds to a dimension). This can for example be avoided by decorrelating features (and thus clustering prototypes). It may be good to discuss the link between this type of collapse and dimensional collapse when discussing all kinds of collapse.

- For the comparison to DINO on dictionary size in table 2, what are the “actual” number of prototypes (and thus dimensions) as the dictionary size increases ? Do we see a stronger collapse for DINO which FALCON avoids ? an analysis such as done in appendix C of [1] may help demonstrate clearly that FALCON does a better job at using the clustering than DINO.

Notation:

- In section 3, $f$ is used as the dimension of the representations and $g$ to denote the encoder. As $f$ is usually used for the encoder, another choice for the embedding dimension (e.g. $d$ and using something like $d’$ for the input dimension) may clarify the notation. 

[1]Garrido, Q., Balestriero, R., Najman, L., & Lecun, Y. . Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank. ICML 2023.

### Soundness
3

### Presentation
3

### Contribution
3
