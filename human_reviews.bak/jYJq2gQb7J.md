# PROVABLY EFFICIENT FEDERATED ACTIVE MULTI-TASK REPRESENTATION LEARNING

- Decision: Reject
- Scores: 5, 6, 6

## Abstract
Multi-task representation learning is an emerging machine learning paradigm that integrates data from multiple sources, harnessing task similarities to enhance overall model performance. The application of multi-task learning to real-world settings is hindered due to data scarcity, along with challenges related to scalability and computational resources. To address these challenges, we develop a fast and sample-efficient approach for multi-task active learning  with linear representation when the amount of data from source tasks and target tasks is limited.  By leveraging the techniques from active learning, we propose an adaptive sampling-based alternating projected gradient descent (GD) and minimization algorithm that iteratively estimates the relevance of each source task to the target task and samples from each source task based on the estimated relevance. We present the convergence guarantees and the sample and time complexities of our algorithm.  We evaluated the effectiveness of our algorithm using numerical experiments and compared it against four benchmark algorithms using synthetic and real MNIST-C datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This work addresses the problem of multi-task representation learning for linear regression models and linear representations. The solution estimates the relevance of each source task to the target task, enabling efficient use of data from relevant source tasks. The paper also provides convergence guarantees for the algorithm, along with an analysis of its sample and time complexities.

### Strengths
This paper provides theoretical guarantees for the proposed representation learning approach. It has two practical improvements over existing methods (lines 170-182):

* It removes the need for knowledge of an optimal solution to the non-convex optimization cost function.

* It does not require the number of data samples to exceed the problem’s dimensionality.

### Weaknesses
Several aspects of the paper could be improved:

1. The scope of the paper is not clearly stated or positioned in the abstract and introduction. The work focuses specifically on linear representation functions and linear regression models; however, both the abstract and introduction discuss general low-dimensional representation and multi-task learning without mentioning this important, narrower focus.

2. The paper claims applicability to image-related learning problems. However, most of these problems, including the MNIST-C dataset used in the paper, are classification tasks typically addressed with non-linear models. This raises questions about how the proposed solution, which is designed for linear regression models, applies in such contexts. Additionally, Assumption 2.2 requires that $x_{m,n}$​ follows an i.i.d. standard Gaussian distribution, which is rarely satisfied in image-related learning tasks. How is this assumption justified for these applications?

3. The technical content of the paper is not fully self-contained, with some notations either undefined or introduced before being defined.

    a. $n_m$​ is defined for $m \in [M]$ (line 89), but the paper later uses $n_{M+1}$, which is not explained.

    b. $\Theta^*$ and $\theta^*_1,\dots,\theta^*_M$ are undefined. This makes the discussion of their SVD and their connection to $W$ and $B$ in lines 116-119 difficult to follow.

    c. The Notations paragraph (lines 129-137) appears after many of the notations have already been used. It would be clearer if this paragraph were placed at the beginning of Section 2.

    d. It is unclear how $X^{(i)}_{M+1}$ is obtained in Algorithm 1.

4. The high probability statement is presented with a probability in the form $O(1 - \delta - d^{-10} - ....)$ (lines 331) without clarifying its meaningfulness. Since the $O$-notation implies potential hidden constants, this probability may actually be small. Additionally, reducing $\delta$ does not necessarily bring this probability closer to $1$ (indicating high probability) due to the presence of other terms, particularly the exponential ones. Therefore, it is unclear if the theoretical result of this paper is an improvement over Chen et al. (2022).

### Questions
Apart from the weaknesses mentioned above, I have the following questions:

1. What new techniques are introduced to handle unknown relevance parameters (lines 181-182) compared to the case with known relevance parameters? The problem setup and the algorithm appear similar to those in Chen et al. (2022), aside from the cost function optimization, which applies to both known and unknown relevance parameters.
   
2. Is the $\epsilon$ in this paper comparable with the $\epsilon$ in Chen et al. (2022)? The discussion in lines 386-390 suggests that the two are comparable. However, in this paper, the guarantee is $ER \le \epsilon$ whereas in Chen et al. (2022) the guarantee is $ER \le \epsilon^2$.

3. In the experiments, the paper should clarify how a regression model can be applied to MNIST-C, which is a classification problem.

4. Chen et al. (2022) assumes knowledge of an optimal solution to the cost function. How is this assumption implemented in the experiments? What justifications are provided for the proposed approach, which does not rely on knowing the optimal solution but instead searches for it using GD, being able to outperform a baseline that has access to the optimal solution?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is built on the premise that in multi-task learning scenarios one typically lacks a sufficient amount of data to train on. This paper addresses the data scarcity problem by developing an adaptive sampling scheme that takes the form of an alternating projected gradient descent and minimization algorithm. The algorithm iteratively estimates the relevance of a source task based on estimated relevance and then samples more heavily from tasks deemed most relevant. As is standard in the literature, the problem formulation assumes there are $M$ source tasks each with $n_m$ iid samples. In particular, the data scarcity is modeled by the fact that $n_m < d$ where $d$ is the input dimension, and $n_{M+1}\ll \{n_1, \dots, n_M\}$.  It is assumed that the underlying linear representation that is shared across all tasks is of low dimension. A convergence and sample complexity analysis shows that both scale as $\mathcal O (\log (1/\epsilon))$, moreover the number of target samples with the rank of the latent feature space $k$. The theory is reinforced through the simulations using the MNIST-C data set.

### Strengths
I believe the paper addresses a relevant problem, and the proposed solution performs well in theory and practice. The algorithm is somewhat intuitive and by that there are no big surprises in any steps. 

- The authors have included a detailed literature review which clearly situates the problem they address
- The paper is specifically concerned with the setting where the relevance parameter $\nu^\star$ is unknown. Estimating $\nu$ introduces errors in the estimate of $\Theta$ and this requires a non-trivial analysis to ensure convergence. 
- Comparing the case where $\nu^\star$ is known and contrasting the algorithmic performance is appreciated (even if this is buried - see next section).

### Weaknesses
My biggest concern is with the flow of the paper and the clarity of exposition. It is quite well written but there are areas that need to be improved.
- The main theoretical results of the paper are presented in Theorems 5.1 and 5.2 presented in section 5. The section 5.1 header indicates that there are two results, one for when $\nu^\star$ is unknown which corresponds to Algorithm 1 and one for  Algorithm 2 where it is known. While reading the paper, it is never even mentioned that there is an algorithm 2.
- Some of the core notation is confusing. Specifically, $\nu^\star$ is defined as a vector of dimension $M$ but also as function of $m$ on the same line. 
- A significant part of the paper depends on the alternating gradient descent minimization algorithm described in the papers by Lin, Neyer, and Collins. This algorithm has been shown to badly suffer in a number of scenarios, including when source noise covariance is not described by a diagonal matrix - see for example "Sample-Efficient Linear Representation Learning from Non-IID Non-Isotropic Data", Zhang et al.

### Questions
- I wonder how important it is to have an accurate singular value decomposition? I'm curious to see how a randomized SVD may perform c.f. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions", Halko et al.
- Is there anything that can be said if the "ground truth" tasks don't exactly share a common representation but instead there is a slight mismatch?

### Soundness
3

### Presentation
2

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
This paper studies efficient active learning in a multitask setting, where the goal is to learn a shared representation and a target linear predictor. Data from source tasks are actively sampled by discovering the relevance of each source task to the target task. The proposed method adopts alternating gradient descent to obtain the shared linear representation.  Theoretical analysis on the convergence and generalization guarantees are provided. The proposed method is tested with numerical experiments on synthetic data and MNIST.

### Strengths
This paper introduce an active multi-task learning theoretical  framework. The proposed method is based on intuitive idea, i.e., essentially eq (4) where it adapts the relevance estimate of each source task to the target task. Although the idea is intuitive, the technical details, both algorithmically and theoretically, are non-trivial. Numerical experiments verifies the effectiveness of the proposed method on simple cases with linear models.

### Weaknesses
* There seems to be an important assumption that is not clearly elaborated. The introduction of the condition number $\\kappa$ at line 119 essentially requires $\\Sigma$ is full-rank. In other words, it is assumed that the the target task $\omega^\star\_{M+1}$ is always a linear combination of  the set of  $\\{\omega\_m\\}\_{m\in M}$ corresponding the $M$ tasks. This is necessary to require $M\\geq k$ where $k$ is the data manifold dimension, i.e., there must be enough number of source tasks. This seems to be a rather strong assumption, and it seems that it can be relaxed, e.g., suppose there is only a few source tasks but they are all close to the target task, so the learned representation should transfer to the target task even if $\omega^\star\_{M+1}$ is not a linear combination of  the set of  $\\{\omega\_m\\}\_{m\in M}$. Can the authors elaborate more on this assumption and how it may be alleviated?  
* The paper has the term "Federated" in the title, suggesting a federated setting, but it seems that it is not really a federated learning paper. e.g., the word "federated" is only mentioned twice in the main paper. I understand that the proposed method can be directly applied in a federated setting, but it seems a misleading to emphasize this in the title. Or more discussion about the application to federated learning, including relate work, should be included. 
* It would be nice if there are discussions on how the proposed method may be applied to non-linear settings. 
* There are many typos and the presentation could be improved. For example, the matrix $W$ is used at line 116 before its actual definition at line 137. The parameter $\beta$ at line 222 seems to be undefined? Is the $\epsilon_3$ at line 331 a typo? Theorem 1 and 2 are quite hard to read and some more discussion to introduce each part before the theorems should be better, e.g., think of a sketched version of the theorem.

### Questions
* see previous section for questions

### Soundness
3

### Presentation
2

### Contribution
2
