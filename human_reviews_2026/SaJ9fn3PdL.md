# Unifying Low Dimensional Observations in Deep Learning

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Empirical studies have revealed low-dimensional structures in the eigenspectra of weights, Hessians, gradients, and feature vectors of deep networks, consistently observed across datasets and architectures in the overparameterized regime. In this work, we analyze deep unconstrained feature models (UFMs) to provide an analytic explanation of how these structures emerge at the layer-wise level, including the bulk–outlier Hessian spectrum and the alignment of gradient descent with the outlier eigenspace. We show that deep neural collapse underlies these phenomena, deriving explicit expressions for eigenvalues and eigenvectors of many deep learning matrices in terms of class feature means. Furthermore, we demonstrate that the full Hessian inherits its low-dimensional structure from the layer-wise Hessians, and empirically validate our theory in both UFMs and deep networks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper analyzes Hessian spectra in deep learning. The authors formally connect the deep neural collapse phenomenon with the empirically observed eigenvalue bulks of the Hessian in well-trained DNNs. The authors mostly operate in the deep unconstrained feature regime, although some of their experiments are done with real networks. The authors first focus on the deep linear case and analyze per-layer Hessian spectra. They express the Hessian of the UFM-optimal solutions and compute its eigendecomposition, showing that they can be expressed in the form of class-means and the eigenvalues form bulks similar to those observed in the related work. Then they interpret the Fisher Information Matrix of UFM-optima and its decomposition into within-class, cross-class and class component in terms of the DNC structure. They also show that gradients align with the dominant eigenspace. Within the deep linear setting, they also provide generalizations into the full Hessian. In the non-linear deep unconstrained feature model regime, they show that DNC solutions’ Hessians exhibit the same structure and show that global optima among DNC solutions are those that can be expressed linearly as well. They support their theory with experiments done either directly on deep unconstrained features model or on real networks.

### Strengths
-	S1: The paper establishes a formal connection between deep neural collapse and outlier eigenvalues / bulks in the Hessians. This is an important (and rather surprisingly) missing piece of the puzzle. 
-	S2: The paper is very well written and sound.

### Weaknesses
-   W1: At times, the authors slightly overstate their contribution. In particular, the related work already did make the connection between the neural collapse and Hessian spectra. Already [1], as also described by the authors, talk about the within-class, between-class and class contributions to the eigenvalue distributions. The link is later made explicit in [2] by the same author(s). It is also mentioned by some other works, such as [3] (although not explicitly as neural collapse). This goes against some of the sentences of the paper such as the third paragraph of introduction or line 63. While I agree that this paper certainly makes the connection most explicit and shows it formally, it is unfair to say that the entire idea is novel. I also think the title of the paper could be a bit less generic, as there are many types of low-dimensional phenomena in deep learning and this is only a very small subset of them. 
-	W2: The attempts the authors make to generalize their linear UFM findings on per-layer Hessian to either full Hessian or non-linear setting are not very strong. As for the per-layer to full Hessian generalization, I find the regime of Theorem 6 to be very unrealistic and thus the connection with full network is unconvincing (W2A). As for the non-linear case, there are even two issues. The first one is that the analysis is only done for DNC solutions, however we know from [4, 5] that different structures also often emerge both in DUFM as well as in real networks (W2B). In general, the authors should unify (to live up to the title’s standard) the analysis of neural collapse with other low-rank phenomena, such as the bottleneck rank [5]. The second issue is the acknowledged non-differentiability of the loss at global minima. While the authors say that limit of smoothed ReLUs would solve the issue, I disagree as the limit should be independent of the way how the ReLU is smoothed, yet different smoothings would lead to different reasonable derivative definitions. 
-	W3: The authors only analyze the perfect scenario. However, in real networks, the collapse is not only not achieved perfectly (smaller problem which would only require a perturbation analysis (still would be nice, though)), but, more importantly, it is achieved gradually. This means that at a constant fraction of the layers, the within-class variability is expected to only be removed to an extent that depends on that fraction. This relationship might be linear or log-linear or more complex (depending on the architecture), but one should not expect full collapse already at a constant fraction of layers. Authors should, therefore, discuss and potentially explain how the Hessian eigenvalue laws can be explained in this progressive collapse regime. In this regard, I would also recommend the authors to rephrase lines 310-312 as the Theorem 6 certainly doesn’t cover the “worst-case” scenario. 
-	W4: A minor issue – the definition of DNC3 in the paper seems to be rather weak. This condition is almost trivially satisfied in most reasonable convergence assumptions. [4, 6] and other works on deep neural collapse work with stronger definitions.

[1] Papyan, Vardan. "Traces of class/cross-class structure pervade deep learning spectra." Journal of Machine Learning Research 21.252 (2020): 1-64.

[2] Papyan, Vardan, X. Y. Han, and David L. Donoho. "Prevalence of neural collapse during the terminal phase of deep learning training." Proceedings of the National Academy of Sciences 117.40 (2020): 24652-24663.

[3] Granziol, Diego, Stefan Zohren, and Stephen Roberts. "Learning rates as a function of batch size: A random matrix theory approach to neural network training." Journal of Machine Learning Research 23.173 (2022): 1-65.

[4] Súkeník, Peter, Marco Mondelli, and Christoph H. Lampert. "Deep neural collapse is provably optimal for the deep unconstrained features model." Advances in Neural Information Processing Systems 36 (2023): 52991-53024.

[5] Jacot, Arthur. "Bottleneck structure in learned features: Low-dimension vs regularity tradeoff." Advances in Neural Information Processing Systems 36 (2023): 23607-23629.

[6] Rangamani, Akshay, et al. "Feature learning in deep classifiers through intermediate neural collapse." International conference on machine learning. PMLR, 2023.

### Questions
-	Q1: What is the difference between $\mu$ and $\hat \mu$? 
-	Q2: The eigenvectors that the gradient is aligned with identified in Theorem 3 do not seem to correspond to the Theorem 2 (c) class eigenvectors. This is somewhat surprising because in the cross-entropy case where the gradient is aligned with the top K eigenvectors, these should be the class eigenvectors. What is the cause of this seemingly counterintuitive difference? 
-	Q3 / comment: The proof of Theorem 8 could perhaps be simplified. To prove that global optima of the non-linear case must have non-negative activations, a simple single-layer contradiction and the use of Theorem 4 could be used where one would construct higher norm solution for the same cost that could then be compensated with appropriately smaller weight matrix in the subsequent layer. 

**Summary:**
Although I do have some concerns and I don’t consider the contribution of the paper to be groundbreaking (in particular because the paper only formalizes in simplified setting what has already been voiced and empirically measured), I still think the paper fills in an important gap in theory of the optimization landscape and soundly formalizes the ideas from the related work. Therefore, should the authors satisfiably address my concerns, I would be open to recommend it for acceptance. However, I cannot do so yet, without a successful discussion with the authors.

### Soundness
4

### Presentation
4

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
The authors study the spectral dynamics of the Unconstrained Features Model (UFM) for deep linear networks and deep networks with RELU activations, which assume that you directly optimize the representations input to the neural network, along with the intermediate weight matrices. The authors find that the UFM models exhibit the emergence of bulk outlier values in the gradient and Hessian spectra theoretically and empirically. The authors argue that these experiments and theory unify low-dimensional learning under the deep neural collapse framework, as these outlier eigenvalues correspond to different classes.

### Strengths
The goal of the paper is nice. There are lots of observations across deep learning that can be reduced to low dimensionality. The authors ask if we precisely describe this dimensionality using an established framework – deep neural collapse.

### Weaknesses
-	I find it overall very confusing how sections 3 and 4 related to “unifying low dimensional structure”. I guess they claim to show that for UFM the gradients / Hessian are low rank and have eigenvalues corresponding to different classes, but low dimensional structures in deep learning are much broader. Even in the context of the paper it is not clear to me what the reader should take away from each section.
-	It is very unclear that UFM captures the dynamics of real neural networks. The authors claim that they established DNC as “they establish DNC as a unifying mechanism shaping curvature, gradient alignment, and weight structure”, yet it is essentially impossible to establish a connection between their results and the UFM model to generalization on real datasets. The connection to the spectral dynamics of real neural networks on real datasets is not even established empirically in the paper. For example, I find it hard to expect that the Hessian and gradients in practice always have these outlier eigenvalues exactly corresponding to different classes.
-	I am especially concerned that weights being low rank is too general of a property to say anything useful regarding the purpose and structure of large singular directions of the weights. This is true even if you can predict eigenvalues/eigenvectors in the UFM model and their ranks. It is not just the singular values, but also, mainly, the singular vectors that are important.
-	The work claims to unify low-dimensional observations in deep learning without addressing the many settings and analyses that prove NNs learn low-dimensional structure and that this structure improves generalization. In particular, there is a long line of work on neural networks learning staircase functions multi-index models that is not connected to their observations in UFM or deep neural collapse (e.g. [1,2,3,4]).

[1] “The staircase property: How hierarchical structure can guide deep learning “. Abbe et al., 2021.

[2] “Neural Networks can Learn Representations with Gradient Descent”. Damian et al., 2021.

[3] “How two-layer neural networks learn, one (giant) step at a time”. Dandi et al., 2024.

[4] “Repetita Iuvant: Data Repetition Allows SGD to Learn High-Dimensional Multi-Index Functions”. Arnaboldi et al., 2024.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studied the low-dimensional structures of deep unconstrained feature models (UFMs), from weights, gradients and hessian matrices. The papers demonstrated the driving factors and solutions of deep neural collapse (DNC) and that the full Hessian inherits the layer-wise Hessian structures.

### Strengths
This study provided an unified perspective towards the dimensional structures in weights, gradients and hessian matrices, leading to explanations towards the DNC phenomenon.

### Weaknesses
1. The contributions of this paper is limited in significance, and the conclusions on DNC is rather unclear; It could be further emphasized in the concluding remarks. 
2. The optimization perspective is studied less in this paper, for which the low dimensional structure interplay with learning dynamics is also a critical factor to the DNC phenomenon.
3. The scope of the numerical experiments are limited in the main paper.

### Questions
1. As an further demonstration of the DNC phenomenon, can the authors further present the low-dimensional structure formation in learning, and towards solving the DNC problem?
2. The story-telling could be improved on the full Hessian spectrum side, e.g. how may this structure derivation assist future studies, and what research questions may benefit from this conclusion.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper derives results about the structure of the Hessian associated with a variant of deep learning classification networks called UFM. The main difference between UFM and standard deep nets is the fact that the input are themselves assumed to be features computed via such a highly flexible function that one can treat them as optimization parameters.  With this approximation, analytical derivations provide support for many empirical or semi-empirical observations reported in earlier papers. 

One intriguing property of this approach, which I would like to see commented, is the fact that making the inputs subject to optimization allows the network to simply make them directly express what we are trying to predict. So in that sense, the optimal solution for an UFM is trivial. Although we can still compute gradients and Hessians, they only describe what's happening in the layers that follow a collapsed representation. These layers, essentially, do nothing.  

However it would be too fast to say that this observation totally undermines the paper. The empirical fact remains that the observed Hessian in actual deep networks display the same structure as the Hessians of these rather degenerate UFMs. That in itself demands an explanation!

### Strengths
+ analytical derivations on UFMs match empirical observations on DNNs.

### Weaknesses
- needs a more intentional discussion of the degenerate nature of UFMs

### Questions
- please discuss the degenerate solutions of the UFM formulation. 
- please discuss the possible causes for the match between the empirical DNN Hessians and the UFM ones.

### Soundness
2

### Presentation
3

### Contribution
3
