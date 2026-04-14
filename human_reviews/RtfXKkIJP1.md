# A Dynamic Low-Rank Fast Gaussian Transform

- Decision: Reject
- Scores: 3, 8, 6, 6

## Abstract
The Fast Gaussian Transform (FGT) enables subquadratic-time multiplication of an $n\times n$ Gaussian kernel matrix $\mathsf{K}_{i,j}= \exp ( -  \|| x_i - x_j \||\_2^2 ) $ with an arbitrary vector $h \in \mathbb{R}^n$, where $x_1,\dots, x_n \in \mathbb{R}^d$ are a set of fixed source points. This kernel plays a central role in machine learning and random feature maps. Nevertheless, in most modern data analysis applications, datasets are dynamically changing (yet often have low rank), and recomputing the FGT from scratch in (kernel-based) algorithms incurs a major computational overhead ($\gtrsim n$ time for a single source update $\in \mathbb{R}^d$). These applications motivate a dynamic FGT algorithm, which maintains a dynamic set of sources under kernel-density estimation (KDE) queries in sublinear time while retaining Mat-Vec multiplication accuracy and speed.  

Assuming the dynamic data-points $x_i$ lie in a (possibly changing) $k$-dimensional subspace ($k\leq d$), our main result is an efficient dynamic FGT algorithm, supporting the following operations in $\log^{O(k)}(n/\varepsilon)$ time: (1) Adding or deleting a source point, and (2) Estimating the kernel-density of a query point with respect to sources with $\varepsilon$ additive accuracy. The core of the algorithm is a dynamic data structure for maintaining the projected ``interaction rank'' between source and target boxes, decoupled into finite truncation of Taylor and Hermite expansions.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper studies the following problem.
Given $n$ points in $\mathbb{R}^d$ lying in a $k$ dimensional subspace, we would like to construct a data structure that supports insertion, deletion and query.
More precisely, we can insert a new point into the data structure or delete an existing point from the data structure.
Also, we can compute the kernel desnity estimation of the points in the data structure at a certain point within $\epsilon$ error efficiently.
The result relies on fast Gauss transform.
The authors constructure a data structure such that the running time of these operations is at most $\log^{O(k)}(1/\epsilon)$ time.

### Strengths
The problem seems well-motivated.

### Weaknesses
The result relies heavily on the previous results. 
I am not sure if there is a new idea introduced.

### Questions
.

### Soundness
2

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
The fast gaussian transform (FGT) can be used to efficiently apply a gaussian kernal matrix to a vector. However traditionally the number of points in the gaussian kernal matrix is already fixed, so even adding one single point can require us to recompute the FGT from scratch, which makes it very hard to deal with datasets that are dynamically changing. In this work, the authors propose a dynamic FGT algorithm that supports adding or deleting a source point, and also estimating the kernel-density of a query point, under the assumption that the added data-points lie in a low dimensional subspace of the original vector space.

### Strengths
I think this paper is well-written. The introduction clearly outlines the history of FMM and FGT, and states their drawback clearly: they are static and it is hard to add/delete points. I also think the main result section is also well-written, and state the contributions clearly.

I also think the technical contributions are solid: the problem of efficiently updating FGT for changing datasets is important, and this work seems to be one of the first to address this problem.

### Weaknesses
I do not see any major weaknesses, but I think it would be better if the authors could include some numerical experiments to showcase their results. In particular, how much faster is the dynamic FGT compared with recomputing from scratch?

### Questions
See above.

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
2

### Summary
The paper introduces a Dynamic Low-Rank Fast Gaussian Transform (FGT), designed to handle efficiently evolving datasets in kernel density estimation (KDE) and matrix-vector multiplication. The Fast Gaussian Transform (FGT) traditionally enables subquadratic-time multiplication of an n×n Gaussian kernel matrix with a vector, but it struggles with dynamically changing data. The proposed dynamic FGT algorithm supports fast updates and KDE queries in sublinear time, with quasi-linear time complexity for full matrix-vector multiplication.

The paper proposes a new data structure that dynamically maintains low-rank subspaces, supporting efficient insertion and deletion of points, along with fast KDE queries. The method builds on classical fast multipole methods (FMM) and uses Taylor and Hermite expansions to manage “interaction ranks” between source and target boxes. By focusing on data in low-dimensional subspaces, the method bypasses the curse of dimensionality, achieving sublinear update and query times under certain conditions. This is particularly useful for large, evolving datasets common in fields like machine learning. Importantly, they answered a question in an affirmative way:

Is it possible to ‘dynamize’ the Fast Gaussian Transform, in sublinear time? Can the exponentialdependence on d (Greengard & Strain, 1991) be mitigated if the data-points $x_i$ lie in a k-dimensional subspace of R^d?

### Strengths
The key contribution is the new methodology proposed by this paper to handle efficiently evolving datasets in kernel density estimation (KDE) and matrix-vector multiplication in a dynamic setting. This question itself is very important in practice and the new method has good time complexity that supports fast updates and KDE queries in sublinear time, with quasi-linear time complexity for full matrix-vector multiplication. Overall, the paper’s strengths lie in its innovative adaptation of FGT to a dynamic setting, its computational efficiency, and its broad relevance to various machine learning and statistical applications. The work addresses a crucial need for scalable kernel-based methods in dynamic, real-time data environments.

### Weaknesses
I think one problem or actually a question that is worth thinking and needs to be clarified is that the method is mainly tailored specifically for Gaussian kernels. Although in Section 3.4, it is generalized to a broader class of "fast-decaying kernels". It is still unclear if it allows other important kernels such as polynomial kernels, Moreover, as I see the kernel is defined using the Euclidean distance, maybe in manifold settings, it is restrictive and not the proper kernel choice.

The other issue is that the paper does not provide detailed guidance on choosing parameters, for example, in Section 3.1, the kernel bandwidth $\delta$ and the length of the smaller boxes $L$, as it is mainly a theoretical paper without real experiments.

While the paper lists theoretical contributions, since the paper proposes a Fast Gaussian Transform (FGT), which is an algorithm on the issue of fast computation, I think it would be good to do some, at least simple simulations on a dataset that is dynamically changing (as claimed by the paper) to see its empirical performance.

### Questions
See weakness section on how to extend the results to other types of kernels? how to handle tuning parameters as mentioned in the weakness section?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a dynamic data structure for maintaining the Fast Gaussian Transform (FGT) under insertions and deletions of source points. The main contribution is an algorithm that supports polylogarithmic-time updates and kernel density estimation queries when the data points lie in a low-dimensional subspace while retaining quasi-linear time for matrix-vector multiplication. The key technical innovation is a hybrid approach combining Taylor and Hermite expansions to "disentangle" source and target points, enabling efficient dynamic updates.

### Strengths
1. This paper has a strong theoretical contribution by solving an open problem of dynamizing the FGT algorithm, which has applications in machine learning and data analysis.
2. This paper includes a novel technical approach combining Taylor and Hermite expansions to enable efficient updates, with careful error analysis and proofs.
3. This paper improves complexity bounds when data lies in low-dimensional subspaces, addressing the curse of dimensionality issue in previous FGT work.

### Weaknesses
1. This paper has limited experimental evaluations, the paper focuses primarily on theoretical analysis without empirical validation of real datasets.
2. The assumptions of data points lie in a low-dimensional subspace, which may not hold in practice for many applications.
3. The paper has limited discussion of numerical stability issues that may arise in practice when computing Taylor/Hermite expansions.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
2

### Contribution
3
