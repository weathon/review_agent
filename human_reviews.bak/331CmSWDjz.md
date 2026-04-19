# Unsupervised Feature Learning with Emergent Data-Driven Prototypicality

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5

## Abstract
Given an image set without any labels, our goal is to train a model that maps each image to a point in a feature space such that, not only proximity indicates visual similarity, but where it is located directly encodes how prototypical the image is according to the dataset.

Our key insight is to perform unsupervised feature learning in hyperbolic instead of Euclidean space, where the distance between points still reflects image similarity, and yet we gain additional capacity for representing prototypicality with the location of the point: The closer it is to the origin, the more prototypical it is.  The latter property is simply emergent from optimizing the usual metric learning objective:  The image similar to many training instances is best placed at the center of corresponding points in Euclidean space, but closer to the origin in hyperbolic space.

We propose an unsupervised feature learning algorithm in **H**yperbolic space with sphere p**ACK**ing. **HACK** first generates uniformly packed particles in the Poincare ball of hyperbolic space and then assigns each image uniquely to each particle. Images after congealing are regarded more typical of the dataset it belongs to.  With our feature mapper simply trained to spread out training instances in hyperbolic space, we observe that images move closer to the origin with congealing, validating our idea of unsupervised prototypicality discovery.  We demonstrate that our data-driven prototypicality provides an easy and superior unsupervised instance selection to reduce sample complexity, increase model generalization with atypical instances and robustness with typical ones.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes HACK for unsupervised learning that can arrange images in hyperbolic space. HACK optimizes image assignments to a fixed set of uniformly distributed particles in the hyperbolic space. It's found that the prototypicality property is emergent from such optimization: images similar to many training instances (more prototypical) are closer to the origin in hyperbolic space. The authors validate the effectiveness of HACK using synthetic data with natural and congealed images. They also test the method on the real MNIST and CIFAR datasets to reveal prototypicality. Lastly, the discovered prototypical and atypical examples are shown to reduce sample complexity and increase model robustness to some extent.

### Strengths
- The proposed unsupervised method HACK does have clear distinctions with existing methods: unlike supervised learning, HACK allows the image to be assigned to any target (particle). Unlike existing unsupervised learning method, HACK learns to match to
a predefined geometrical organization in hyperbolic space (uniformly distributed).
- The core instance assignment problem is cast as a bipartite matching problem and solved with the well-known Hungarian algorithm that has good convergence properties.
- Besides validating the efficacy of HACK in learning prototypicality, the authors also explored its use in sample complexity reduction and model robustness aspects.

### Weaknesses
I think the presentation of this paper needs improvements. One main issue is that the authors keep talking about how HACK works and how it can encode both visual similarity and prototypicality, without enough explanations about the reason why. It's suggested to list the intuitions upfront, so readers won't always question why HACK is designed this way and why it works at all. Specifically,
- Missing intuition everywhere about why images should be assigned to uniformly distributed particles. Only until Section 4.2, it's mentioned that this is to achieve maximum instance discrimination as in (Wu et al., 2018).
- Follow-up questions: is such uniform target the best option? Ablations on other targets will help.
- Missing another intuition: why prototypicality will merge from optimizing for maximum instance discrimination? This is never explained but super important.
- Figs 5,6,8 are supposed to show evidence that HACK indeed captures 1) visual similarity. Unfortunately I don't have the same observations from the very small image examples. Clearer examples will help. Also, image retrieval experiment is an important alternative. 2) prototypical examples (in the center of the Poincare ball) vs. atypical examples around the boundary. Again, such trend is not clear from the given small image examples.

### Questions
Questions around reducing sample complexity:
- Fig.9(a) shows that models trained on atypical examples performs better than on typical examples, especially when the amount of training examples used is small. This is a bit counter-intuitive and different from many other studies, where DNNs are shown to pick up regularities in typical data and then further benefit from or memorize noise/atypical data. Any comments?
- Fig.9(a) shows that with increasing amount of data (either typical or atypical) converges to similar test accuracy. Is that close to the optimal accuracy, or performance will keep improving with more data? Another (maybe more practical) way to prove sample complexity reduction is to compare to the "best" model performance and measure how much less data are used, rather than in the low-data regime where performance is far from ideal.

Questions around robustness:
- Fig.9(a) basically shows "more atypical data, better generalization accuracy", while Fig.9(b) says that using fewer atypical data improves model robustness. The observations are a bit contradicting and it seems hard to strike the balance between accuracy and robustness. Any comments?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a novel approach to map images into a feature space that not only indicates visual similarity but also encodes the prototypicality of the image based on its location in the dataset. Instead of using Euclidean space, the authors utilize hyperbolic space for unsupervised feature learning. In this space, the proximity of a point to the origin signifies its prototypicality. They present an algorithm called HACK, which assigns each image to uniformly packed particles in hyperbolic space, optimizing the dataset's organization. The method grounds the concept of prototypicality in congealing, aligning images to appear more common and similar, which aligns with human visual perception. The paper's contributions include the first unsupervised feature learning method capturing both visual similarity and prototypicality, and the demonstration that identified prototypical and atypical examples can optimize sample complexity and model robustness.

### Strengths
Strength:
1. Paper is well organized.
2. The use of hyperbolic space instead of Euclidean space is well-motivated.

### Weaknesses
Weakness:
1. CIFAR and MNIST are too toy. ImageNet experiment and fair comparison with previous unsupervised learning (especially contrastive learning) are important, but missing in this work.
2. LeNet is also too toy for a fair comparison with the latest results on unsupervised learning. A model of the ResNet level is a must.
3. Some related works on prototype learning are not cited, like “Prototypical Contrastive Learning of Unsupervised Representations”.

### Questions
Questions:
1. In terms of optimization, the proposed method also needs to alternatively optimize the encoder (θ) and the assignment (π), which show no advantage over previous “prototype contrastive learning work” that requires to optimize both sample features and prototype assignments ("centroids")
2. “pack the particles into a two-dimensional hyperbolic space” Is it possible to expand the embedding space to over two dimensions? I believe representing high-dimensional data into a two-dimensional space is too limited for practically useful embeddings.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors propose an unsupervised feature learning algorithm, HACK, that captures visual similarity and prototypicality. Specifically, HACK first generates uniformly packed particles in the Poincare ball of hyperbolic space. Then, it optimizes data assignments to a uniformly distributed particle set by naturally exploring the properties of hyperbolic space, in which prototypical and semantic structures of data emerge finally.

### Strengths
In this work, the authors propose the unsupervised feature learning method from a novel perspective that aims to capture both visual similarity and prototypicality.

### Weaknesses
1.	The motivation of the proposed method is not clear. It lacks clarification of motivation to state that: what are the shortcomings of existing methods that do not consider prototypicality? Why does the unsupervised feature learning method need to consider prototypicality? The motivation mentioned in the first paragraph of Section 1 is too vague.
2.	In the paper, the work has limited motivation, which seems to be a combination of existing technologies with introducing existing concepts.
3.	It is also necessary to analyze the unique points of the proposed method compared to existing related methods, so as to further clarify the motivation and novelty. However, the paper lacks concrete analyses of the difference between the proposed and existing related methods.
4.	The writing of this paper needs to be improved. Some sentences include too many prepositions, which decreases readability.
5.	The layout of the article needs to be improved, for example, there is too much white space on page 8.

### Questions
Please see the Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
