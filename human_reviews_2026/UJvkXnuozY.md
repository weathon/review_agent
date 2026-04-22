# Tucker-FNO: Tensor Tucker-Fourier Neural Operator and its Universal Approximation Theory

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Fourier neural operator (FNO) has demonstrated substantial potential in learning mappings between function spaces, such as numerical partial differential equations (PDEs). However, FNO may suffer from inefficiencies when applied to large-scale, high-dimensional function spaces due to the computational overhead associated with high-dimensional Fourier and convolution operators. In this work, we introduce the Tucker-FNO, an efficient neural operator that decomposes the high-dimensional FNO into a series of 1-dimensional FNOs through Tucker decomposition, thereby significantly reducing computational complexity while maintaining expressiveness. Especially, by using the theoretical tools of functional decomposition in Sobolev space, we rigorously establish the universal approximation theorem of Tucker-FNO. Experiments on high-dimensional numerical PDEs such as Navier-Stokes, Plasticity, and Burger's equations show that Tucker-FNO achieves substantial improvement in execution time and performance over FNO. Moreover, by virtue of the compact Tucker decomposition, Tucker-FNO generalizes seamlessly to high-dimensional visual signals by learning mappings from the positional encoding space to the signal's implicit neural representations (INRs). Under this operator INR framework, Tucker-FNO gains consistent improvements on continuous signal restoration over traditional INR methods in terms of efficiency and accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work present the Tucker-FNO, an approach modifies the FNO to support the processing of high-dimensional inputs. Using a Tucker style functional decomposition, high-dimensional inputs may be processed into several low-dimensional inputs and processed in parallel by 1-dimensional FNOs. The authors provide a universal approximation result experiments on PDEs, image in-painting, and image denoising.

### Strengths
- Performing tensor factorization on the operator/function space as opposed to the parameter space is an interesting idea.
- The authors provide a functional Tucker decomposition UAT, which is, as far as I am aware, novel and could be useful in other applications.
- A broad set of baselines beyond PDEs is investigated, where Tucker-FNO shows relatively good performance.
- Several ablation studies are also present, which show notable improvements in speed and parameter efficiency in comparison to the FNO and alternative approaches.

### Weaknesses
1. Prior work and novelty. The manuscript does not adequately situate itself amongst existing literature, particularly the Multi-Grid Tensorized FNO [https://arxiv.org/abs/2310.00120]. This work also relies on Tucker-factorization. Although it acts on the weights of the FNO as opposed to the inputs, the works share substantial similarities in their goal to process large inputs and maintain parameter efficiency. Given this similarity, this work must be cited and compared both conceptually and empirically to clarify the novelty.
2. Limited 3D validation. While this work repeatedly claims applicability to 3D (or general high-D) PDEs, there are only experiments on 2D or 2D+time PDEs. The problems investigated are also relatively simple, and model performance has become saturated in the larger literature. Experiments on more difficult 3D or 3D+time baselines are required to support claims of scaling. Baselines include GINO [2309.00583] and GenCFD [2409.18359].
3. Implementation clarity and code. I feel many of the details of how the approach actually works in practice are obfuscated by rigorous theoretical derivations. The code provided in the supplementary material, likewise, is incomplete and does not resolve key implementation questions. 

Minor typos: line 174 "construct an mapping", line 240 "multi-layer perception" (I believe perceptron is meant here), line 247 big O notation of O(log(n^3)) should be simplified to O(log(n)).

### Questions
1. Is it correct that the input does not undergo an explicit Tucker factorization, but rather it is processed in a way that assumes such a factorization exists? And then the inverse of the Tucker factorization is performed at the end, multiplying the factorized inputs by the core tensor? 
1b. If my understanding above is correct, is this process factored in to the complexity calculation? How does this process scale in the case that the Tucker rank is large, even if it is chosen as a hyperparameter? 
2. Is there any theoretical justification for Tucker factorization of the operator/input as opposed to the weights? Or can these be shown to be equivalent?
3. How does the approach perform on a PDE describing a flow in 3 spatial dimensions? Please provide an experiment with baselines.

### Soundness
3

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
1

### Summary
This paper aims to address the inefficiency of fourier neural operator (FNO) in solving large-scale and high-dimensional PDEs. To reduce computational complexity, this paper proposes Tucker-FNO, a neural operator that decomposes the high-dimensional FNO into a series of 1-dimensional FNOs through Tucker decomposition. It also proves the universal approximation theorem of Tucker-FNO. Experiments are done on PDEs and video reconstruction. The proposed method outperforms previous methods on both tasks.

### Strengths
1. The idea is simple yet effective.
2. The theoretical and practical computation decreases significantly.

### Weaknesses
1. Figure 4 looks weird to me. The Carphone video reconstruction gets worse when the Fourier frequency truncation k increases. It is explained in the paper that "excessive parameters hinder convergence efficiency". However, it is not the case in the PDE problem in Table 4. Why does this difference exist? What if we simply wait the model to converge?
2. Related to 1), if the model is not scalable, then decreasing the computation complexity seems not as important as claimed. A figure of scaling would be appreciated.

### Questions
1. Why Tucker rank is directly set to the latent dimension for PDEs? How does the performance change with Tucker rank?
2. Why not compare with F-FNO since it is already mentioned in the Related Work along with others that are compared with?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Tucker-FNO, a neural operator that decomposes high-dimensional Fourier Neural Operators (FNO) into multiple 1-dimensional FNOs using Tucker decomposition. The method targets computational efficiency for partial differential equations (PDEs) and signal restoration tasks while maintaining expressiveness.

Tucker-FNO decomposes a d-dimensional FNO into d separate 1-dimensional FNOs arranged in Tucker format. A pre-lifting module extracts factor inputs from the condition function, processing them through individual 1-dimensional FNOs before aggregating outputs via Tucker decomposition. This reduces computational complexity from O(n³ log n³) to O(3 n log n) for 3D PDEs. The authors establish a universal approximation theorem using functional decomposition tools in Sobolev spaces, proving Tucker-FNO can approximate any continuous operator.

### Strengths
- Significant efficiency gains: Reduces parameters by and execution time compared to standard FNO on 256×256 data while improving performance​.

- The paper is well-written.

- Rigorous theoretical foundation: First tensor-decomposed neural operator with proven universal approximation capability​.

- Dual applications: Successfully handles both PDE approximation and signal restoration tasks​.

- Good empirical performance: Outperforms FNO, D-FNO, and Com-FNO on Navier-Stokes, Plasticity, and Burger's equations​.

- Better high-frequency signal handling: Shows advantages in regions with significant numerical variation​.

### Weaknesses
- Limited novelty in decomposition: Applies existing Tucker decomposition techniques to FNOs without fundamentally new mathematical insights. Tensorized FNOs, which use Tucker decomposition have already been implemented as part of the neuraloperator library and also published in the literature. While this in itself is not a problem, there is no mention of prior related works such as [1].

- Exprimental evaluation is mainly conducted on toy datasets and more complex examples are omitted. Baselines are also limited and lack other tensorized approaches such as [1] or the factorized FNO. This should also include comparisons to other tensor decomposition than just Tucker (e.g., tensor train, CP decomposition)

- Moreover, it is unclear whether this approach can hold in more complex settings which include geometry. It's quite likely that this approach would fall apart in such a setting, given that it breaks symmetry and even just a single test with a simple geometry would be very informative.

- Restrictive assumptions: Universal approximation theorem requires analyticity on compact sets (Theorem 3), limiting applicability to non-smooth or unbounded domains​.

- Additional hyperparameter sensitivity: Performance depends on Tucker rank selection and frequency truncation threshold, requiring careful tuning​. It's unclear how the rank should be set for a given problem.

- Scalability questions: While efficient for moderate dimensions, scaling behavior for very high-dimensional problems (d > 10) remains unexplored

[1] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., & Anandkumar, A. (2023). Multi-grid tensorized fourier neural operator for high-resolution pdes. arXiv preprint arXiv:2310.00120.

### Questions
- I wonder why the authors chose these examples and whether they explored other tensor decompositions. It would also be interested to include such results in the theory section

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
This paper proposes Tucker-FNO, a tensor-decomposed variant of the Fourier Neural Operator (FNO) designed to improve efficiency and scalability in high-dimensional operator learning. The main idea is to employ Tucker decomposition to represent a high-dimensional Fourier operator as a composition of a low-dimensional core tensor and multiple independent 1D FNOs. This formulation substantially reduces computational costs, particularly the cubic scaling of high-dimensional FFTs, while maintaining expressive power. In addition to the algorithmic design, the paper establishes a universal approximation theorem (UAT) for Tucker-FNO using tools from functional analysis in Sobolev spaces, providing a theoretical guarantee for its representational capacity. Empirical results on several PDE benchmarks, including Navier–Stokes, Burgers’, and Plasticity equations, as well as continuous signal learning tasks, show that Tucker-FNO achieves improved efficiency and accuracy over standard FNOs.

### Strengths
The paper’s strength lies in its clear and well-motivated attempt to address one of the key limitations of FNOs—their inefficiency in high-dimensional function spaces. By leveraging Tucker decomposition, the proposed model reduces both memory and computational complexity while preserving the operator’s structural expressiveness. This approach is both elegant and practical, as it connects classical tensor factorization with modern neural operator design.
The inclusion of a rigorous universal approximation theorem significantly strengthens the contribution by grounding the empirical observations in a solid theoretical foundation. This theorem not only supports Tucker-FNO’s representational power but also provides a framework for future theoretical extensions.
Experimentally, Tucker-FNO demonstrates consistent improvement in speed and accuracy compared to baseline FNOs, with notably lower FFT costs in three-dimensional settings. The paper also extends Tucker-FNO to continuous signal learning via implicit neural representations (INRs), showing that the proposed decomposition generalizes well to non-PDE tasks. Overall, the method provides both practical computational benefits and theoretical depth, positioning it as a meaningful advance in scalable operator learning.

### Weaknesses
While the paper makes a valuable contribution, there are several limitations that should be addressed. First, although the method is advertised as being effective for high-dimensional settings, most experiments are conducted on relatively low-dimensional PDEs (2D or 3D). The current results therefore do not convincingly demonstrate scalability to genuinely high-dimensional problems (e.g., 5D+), which is central to the paper’s motivation.
Second, rank selection in Tucker decomposition is a critical factor for both performance and efficiency. The paper treats Tucker ranks as fixed hyperparameters but does not provide a mechanism to learn or adapt them during training. Over- or under-factorization can degrade performance or efficiency, and the absence of an adaptive rank selection strategy limits the robustness of the approach.
Moreover, while the theoretical result is a strong addition, the practical interpretability of the UAT remains limited — it guarantees existence but does not clarify how rank or decomposition depth affect approximation accuracy in practice. Finally, when Tucker ranks are large, the method may still suffer from bottlenecks due to the size of the core tensor, potentially reducing the claimed computational advantage.

### Questions
- Can the Tucker ranks be learned or dynamically adapted during training, rather than being fixed a priori?
- How does Tucker-FNO behave in genuinely high-dimensional PDEs (e.g., 5D or higher)? Are there benchmark results or computational analyses for such settings?
- For large Tucker ranks, does the core tensor introduce new bottlenecks that offset the decomposition’s computational gains?
- In the universal approximation theorem, can the approximation error be explicitly related to Tucker rank or decomposition depth?

### Soundness
3

### Presentation
3

### Contribution
3
