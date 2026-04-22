# Transformers as Unsupervised Learning Algorithms: A study on Gaussian Mixtures

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 8, 6, 4

## Abstract
The transformer architecture has demonstrated remarkable capabilities in modern artificial intelligence, among which the capability of implicitly learning an internal model during inference time is widely believed to play a key role in the understanding of pre-trained large language models. However, most recent works have been focusing on studying supervised learning topics such as in-context learning, leaving the field of unsupervised learning largely unexplored.
    This paper investigates the capabilities of transformers in solving Gaussian Mixture Models (GMMs), a fundamental unsupervised learning problem through the lens of statistical estimation.
    We propose a transformer-based learning framework called Transformer for Gaussian Mixture Models (TGMM) that simultaneously learns to solve multiple GMM tasks using a shared transformer backbone. The learned models are empirically demonstrated to effectively mitigate the limitations of classical methods such as Expectation-Maximization (EM) or spectral algorithms, at the same time exhibit reasonable robustness to distribution shifts.
    Theoretically, we prove that transformers can efficiently approximate both the Expectation-Maximization (EM) algorithm and a core component of spectral methods—namely, cubic tensor power iterations. These results not only improve upon prior work on approximating the EM algorithm,
    but also provide, to our knowledge, the first theoretical guarantee that transformers can approximate high-order tensor operations.
    Our study bridges the gap between practical success and theoretical understanding, positioning transformers as versatile tools for unsupervised learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies whether transformers can simulate unsupervised learning algorithms with a focus on Gaussian mixtures. The paper provides both empirical and theoretical evidence showing that there exists a transformer that approximates the EM algorithm well.
Next, they propose a learning framework named TGMM for transformers to learn Gaussian mixtures.

### Strengths
- The figures are well presented
- The empirical evidence well demonstrated the proposed TGMM framework is effective.

### Weaknesses
- The theoretical result seems to be weaker than [1], where [1] shows the softmax version and this paper mainly focuses on ReLU transformers.
- I think the presentation of this paper can be further improved. To me, this paper is more of a theoretical focus than empirical based on its settings. However, most of the main text are used to describe empirical results. 
- The proposed TGMM framework seems to be trivial. Its setup is very similar to the pretraining setup in (Bai et al.2023), where it samples tasks from a meta distribution $\pi$.
- It is unclear that how the proposed TGMM framework benefits training from a theoretical perspective.



[1] Transformers versus the EM Algorithm in Multi-class Clustering (He et al, 2025)

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The classical Gaussian mixture model problem (GMM) states as follows.
Suppose we have a set of $N$ samples, $X_1,\dots,X_N$, drawn from a mixture of $K$ (isotropic) Gaussians $\sum_{i=1}^K \pi_i \phi(\cdot\;; \mu_i)$ where $\phi$ is the $d$-dimensional Gaussian with covariance being the identity matrix, $\pi_i > 0$, $\sum_{i=1}^K \pi_i = 1$ and $\mu_i\in \mathbb{R}^d$ for all $i = 1,\dots, K$.
Out task is to design an algorithm that takes $X_1,\dots,X_N$ as input and returns $\hat \pi_1, \dots, \hat \pi_K, \hat \mu_1,\dots, \hat\mu_K$ such that there exists a permutation $\sigma$ on $\{1,\dots, K\}$ that $\hat \pi_{\sigma(i)}$ (resp. $\hat \mu_{\sigma(i)}$) is close to $\pi_i$ (resp. $\mu_i$) for all $i = 1,\dots, K$.

This paper studies the approach of using transformers to solve GMM.
More precisely, the authors propose an architecture that first encodes the samples $X_1,\dots,X_N$, parses them into a multi-layer transformer and then decodes the output.
The authors use the $\ell_2$-error to measure the quality of the output, i.e. $\frac{1}{K}\sum_{k=1}^K (\frac{1}{d}\lVert \hat \mu_{\sigma(i)} - \mu_i\rVert^2 + (\hat\pi_{\sigma(i)} - \pi_i)^2)$.
The authors show that the aforementioned architecture can approximate the Expectation-Maximization (EM) algorithm and the spectral algorithms which are the classical algorithms for solving GMM.

### Strengths
- The paper provides an interesting connection between the modern approach and the classical approach in machine learning. 
I believe this connection can open up a new pathway to explaining the fundamental structures of different models.

- The presentation is clear.
Readers of different levels of expertise should be able to grasp the central idea of this paper.

### Weaknesses
- The paper is generally good.

### Questions
- Abstract: It may be helpful to spell out the full name of TGMM for the first time.

- Line 154: ``... i.i.d. ...''

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper considers the use of transformers to solving Gaussian Mixture Models (GMMs) by learning to directly map from the input samples to the associated GMM parameters considering a simple GMM with identity matrix as covariance whereas mean and cluster proportions are learned. The resulting TGMM is trained to solve multiple GMM tasks using the same transformer backbone. The work is related especially to He et al. (2025b) but the current work focuses on estimating the parameters of the GMM as opposed to the cluster labels. It is demonstrated that L conventional EM steps can be approximated using \mathcal{O}(L) transformer layers with bounds relying by a constant on the attention heads with polynomial scaling in dimension, consequently improving upon the scaling laws of the work of He et al. It is further discussed how power iterations in spectral methods can similarly be approximated. Notably, the proposed TGMM can also learn GMMs with different number of clusters as the transformer structure used an encoder that takes as input an embedding representation of K concatenated the inputs and a decoder structure that is specific to the number of components to be produced in the output. The TGMM is demonstrated to perform well when compared to EM and spectral GMM estimation whereas a Mamba2 backbone is further considered and found to have inferior performance to the transformer backbone.

### Strengths
•	The approach is sound and improves upon the related work especially of He et al 2025b.

•	The results are compelling and the procedure appears promising when compared to EM and spectral estimation of GMMs.

•	The design to accommodate different number of components seems logical and useful.

Originality: The approach expands upon current efforts utilizing Transformers in the context of unsupervised learning in GMM especially leveraging the recent works of He et al 2025a,b with substantial improvements as outlined in the summary. Theoretical properties wrt. the Transformer surrogating EM and spectral inference procedures is further investigated. The contribution in all provides an overall sufficient and interesting contribution.

Clarity: The paper is very clearly written and well presented.

Quality: The paper well presents the developed methodology as well as theoretical insights and the experimentation is generally well executed but limited to synthetic data.

Significance: The approach expands upon previous recent work and make significant contributions including better scaling properties and insights to Transformers capabilities mimicking unsupervised learning procedures in the context of EM and Spectral based inference for the GMM. The experimentations are solid but limited to synthetic data whereas the GMM assuming isotropic variance makes the practical use very limited reducing the impact of the work.

### Weaknesses
The considered GMM formalism is quite limited and it would strengthen the contribution to consider more realistic settings such as GMMs with diagonal covariance clusters. It is unclear why this would form a major challenge in the presented framework as it technically just requires expanding the readout function to have parameters for the variances of similar dimensionality as the produced means in the readout function and loss functions based on an additional squared error loss term for the diagonal variances.

The analysis are purely considering synthetic data and it is unclear why the approach is not also tested on real data. This would substantially expand and verify the practical utility of the developed framework. Currently, the limited variance formulation makes it likely impractical for real data analyses.

The presented performance metric is in terms of least squares error to ground truth cluster centers as well as mixing proportions. I appreciate the L2 metric in the main paper but think the log-likelihood metric considered in the supplementary should also be part of the main-paper as the EM-based procedure is based on optimizing the log-likelihood whereas the TGMM is minimizing the LS error of the parameters. This will provide a more comprehensive evaluation in the main paper in terms of recovering the underlying parameters and how well the model accounts for the underlying density. Furthermore, it would be interesting to display the mean and mixture proportion errors separately at least in the supplementary to understand if there are discrepancies in these errors. It would also be natural to quantify the mixture proportion errors using cross-entropy as optimized.

### Questions
The loss function optimized in equation (2) contains an implicit weighting between accuracy of mean estimates by least squares and accuracy of cluster sizes by cross-entropy loss. How should these losses be balanced in general? And why not use the standard log-likelihood as loss?

EM and spectral methods are widely used, but it would be relevant also to compare against a simple SGD procedure based on Adam or similar optimization minimizing the GMM likelihood for comparison. How would such simple procedure compare?

What is the reason for not including a diagonal variance in the modeling which would make the approach more useful and seems to be straight forward to implement?

What is the practical relevance of the procedure – it would strengthen the paper to consider real datasets and the quality of the learned clusters. This would however likely also require expanding the modeling procedure to at least have cluster centers with diagonal variance parameters. What is hampering the procedure to be expanded to such setting and evaluated on real data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes TGMM, a transformer-based learning algorithm that estimates parameters of Gaussian Mixture Models across varying dimensions d and numbers of components K. Empirically, TGMM competes with or exceeds EM and matches spectral methods while handling cases ($K>d$) where spectral fails, and shows robustness to distribution and sample-size shifts. Theoretically, the authors provide interpretations of their algorithm by proving transformers can approximate EM updates and tensor power iterations used in spectral methods.

### Strengths
1. This paper proposes a new algorithm that uses transformer architectures for GMM learning.  
2. The authors empirically validate the effectiveness, robustness and flexibility of TGMM, and provide theoretical justifications by proving approximability results of their method.

### Weaknesses
1. While the authors try to demonstrate the effectiveness of TGMM via experiments, it actually performs worse than spectral methods in Figure 2 and Figure 4\. These figures also fail to decouple statistical errors and algorithmic errors. Spectral methods, as theoretically guaranteed, will have errors converging towards 0 with enough samples. The authors should at least show the same (empirically or theoretically) for TGMM. To demonstrate effectiveness, in my opinion, they should also compare sample complexity of TGMM to baselines and show improvements.  
2. While spectral methods fail in the degenerate case of $K\>d$, there are sum-of-squares methods that overcome this issue (see \[1\]). Similarly, while EM easily gets trapped in local minima, people have proposed pruning based variants to overcome this (e.g., see \[2\]). The authors should conduct more comprehensive baseline comparisons before claiming performance superiority., 

References.

1. Liu, Allen, and Jerry Li. "Clustering mixtures with almost optimal separation in polynomial time." Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing. 2022\.  
2. Dasgupta, Sanjoy, and Leonard Schulman. "A two-round variant of em for gaussian mixtures." arXiv preprint arXiv:1301.3850 (2013).

### Questions
1. What is the data used for training TGMM? How does the model training cost compare in quantity with the computational cost of EM/spectral methods?   
2. How does the model perform in small separation regimes close to the information theoretical lower bound (see \[1\])? 

References:

1. Regev, Oded, and Aravindan Vijayaraghavan. "On learning mixtures of well-separated gaussians." 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS). IEEE, 2017\.

### Soundness
3

### Presentation
2

### Contribution
2
