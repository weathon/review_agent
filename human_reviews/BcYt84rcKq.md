# Fourier Sliced-Wasserstein Embedding for Multisets and Measures

- Avg Score: 6.25
- Decision: Accept (Poster)
- Scores: 6, 8, 5, 6

## Abstract
We present the _Fourier Sliced-Wasserstein (FSW) embedding_—a novel method to embed multisets and measures over $\mathbb{R}^d$ into Euclidean space.

Our proposed embedding approximately preserves the sliced Wasserstein distance on distributions, thereby yielding geometrically meaningful representations that better capture the structure of the input. Moreover, it is injective on measures and _bi-Lipschitz_ on multisets—a significant advantage over prevalent methods based on sum- or max-pooling, which are provably not bi-Lipschitz, and, in many cases, not even injective.
The required output dimension for these guarantees is near-optimal: roughly $2 N d$, where $N$ is the maximal input multiset size.

Furthermore, we prove that it is _impossible_ to embed distributions over $\mathbb{R}^d$ into Euclidean space in a bi-Lipschitz manner. Thus, the metric properties of our embedding are, in a sense, the best possible.

Through numerical experiments, we demonstrate that our method yields superior multiset representations that improve performance in practical learning tasks. Specifically, we show that (a) a simple combination of the FSW embedding with an MLP achieves state-of-the-art performance in learning the (non-sliced) Wasserstein distance; and (b) replacing max-pooling with the FSW embedding makes PointNet significantly more robust to parameter reduction, with only minor performance degradation even after a 40-fold reduction.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
**Summary:**  
The paper introduces the "Fourier Sliced Wasserstein (FSW) embedding" for data in \(\mathbb{R}^d\).

**Theoretical Contributions:**  
1. The authors prove that the embedding preserves or approximates the sliced Wasserstein distance.  
2. They also demonstrate that the embedding technique is injective and bi-Lipschitz.

**Numerical Experiments:**  
1. The authors evaluate the approximation error of the proposed Fourier Sliced Wasserstein embedding.  
2. They showcase an application of FSW for approximating the Wasserstein distance using a Multi-Layer Perceptron (MLP).

### Strengths
1. The combination of the Fourier/cosine transform and the sliced Wasserstein distance (see Eq. (6)) is a novel approach.
2. Theoretical properties for this new technique with respect to the uniform distribution, along with its empirical approximation, are proposed (see Theorem 3.2, Corollary 3.3).
3. Injectivity and bi-Lipschitz properties of the embedding have been investigated.

### Weaknesses
1. I recommend adding a section to introduce baseline methods. For example, explaining how Sinkhorn [Cuturi, 2013] can be used to train a neural network as a Wasserstein distance estimator. Currently, the experimental setup (E1, E2, Phi, Leaky-ReLU) appears tailored only to the proposed method in this paper.
2. It would be beneficial to introduce a real-data application of the proposed Sliced Wasserstein distance embedding technique to illustrate its practical utility.
3. I’m unclear on why 'bi-Lipschitz' is considered a crucial property. Could you provide an example to clarify? For instance, in which applications would the lack of a bi-Lipschitz property cause issues, and where having this property could offer distinct advantages?

### Questions
1. Regarding point (2), is "Multisets" simply another term for "discrete distributions"?  
2. Could you clarify what "E2" refers to in lines 473-474?

### Soundness
2

### Presentation
3

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
The paper seeks to establish a mapping from multisets and measures over $ \mathbb{R}^d $ into Euclidean space, ensuring that the sliced Wasserstein distance corresponds to the distance between their mappings in the target space. The authors propose a mapping that is bi-Lipschitz for multisets and injective for measures. Additionally, they demonstrate that a bi-Lipschitz map for measures does not exist.

### Strengths
The paper is well-structured, and its message is clear. The proofs provided are rigorous and exceptionally clear. This particular problem is quite interesting. I really enjoyed reading the paper.

### Weaknesses
I don't see any weaknesses. Therefore, I recommend accepting it.

### Questions
I haven't any question.

### Soundness
4

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
2

### Summary
This paper presents a novel approach to high-dimensional dataset embedding. The authors provided theoretical performance guarantees and numerical study to show the superior performance of the framework.

### Strengths
The theoretical contribution seems to be sound, with explicitly stated technical assumptions and results. Numerical study is solid.

### Weaknesses
1. The authors denovted much space to describe the p-wasserstein and infinity-type Wasserstein distance. Why it is necessary to introduce infinity-type Wasserstein distance?
2. In line 222, the authors mentioned that in the special case of d=1, Wasserstien can be computed significantly fast. So what is the complexity rate?
3. In line 344, what is the definition of STD???
4. The authors should provide proof ideas for the main technical results in the main content.

### Questions
I am new to this field. Could the authors elaborate more on the practical motivation and applications of this approach?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper considers Fourier slicing embedding both for a collection of probability distributions and multisets over $\mathbb{R}^d$ and supported at $n$ points. The embedding consists of a projection sample on a 1-dimensional vector on the sphere then calculates a cosine transform of the projected quantile function. Under a specific probability distribution of the frequency, the authors prove that the expectation of the estimation error between the embedded measures is exactly the sliced Wasserstein distance. A second part of the theoretical results consists of proving the injectivity of the embedding under the assumption that the dimension embedding $m \geq 2n(d+1) +1$. Numerical experiments are conducted on point cloud classification.

### Strengths
- The paper is well-written and easy to follow. Proofs are rigorous. 
- Proposing the sliced embedding Wasserstein (SEW) through a cosine transform of the projected quantile function. Sampling the quantile function via cosine transform is novel.
- Injectivity and bi-Lipschitz properties of FSEW on the collection of multisets.
- Numerical experiments showcase better Wasserstein approximation on simulated datasets and three real datasets than NProductNet, WPCE, NSDeepSets, and Sinkhorn.

### Weaknesses
- Several approaches for the derivative of sliced Wasserstein distance like, distributional sliced Wasserstein (Nguen et al, ICLR'21), max-sliced Wasserstein, etc ... Could you highlight the difference between FSW and the SOTA derivative of sliced Wasserstein?

### Questions
See Weaknes section.

### Soundness
2

### Presentation
3

### Contribution
2
