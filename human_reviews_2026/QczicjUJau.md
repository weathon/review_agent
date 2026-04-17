# Efficient Learning of Probabilistic Tensor Networks

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Tensor networks (TNs) enable compact representations of large tensors through shared parameters. Their use in probabilistic modeling is particularly appealing, as probabilistic tensor networks (PTNs) allow for tractable computation of marginals. However, existing approaches for learning parameters of PTNs are either computationally demanding and not fully compatible with automatic differentiation frameworks, or numerically unstable. In this work, we demonstrate a conceptually simple approach for learning PTNs efficiently, that is numerically stable.  We stabilize the computation of the negative log-likelihood computation by iteratively rescaling intermediate computations using logarithmic scale factors. We show our method provides significant improvements in time and space complexity, achieving 10× reduction in latency for generative modeling on the MNIST dataset. Furthermore, our approach enables learning of distributions with 10× more variables than previous approaches when applied to a variety of density estimation benchmarks. Our code is publicly available at https://github.com/ptensnet/ptn.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work studies the problem of gradient-based training of probabilistic tensor networks, e.g., MPS. Directly using gradient-based optimizers would be very unstable for long MPS, due to the long sequence of matrix multiplication. This paper proposes a method, which normalizes every matrix factor, and then adds the scaling factor on the logarithm. Empirically, the authors show that this scaling makes the training stable, enabling gradient-based training of long MPS, which is impossible for SGD.

### Strengths
1. I think the problem of gradient-based optimization of MPS is meaningful and important in this field. As the authors introduced, the unstable training using SGD and the complexity in both training and implementation of DMRG have been a bottleneck of applying such methods.

2. The proposed method is simple to use, does not introduce additional latency, and seems to be much more stable than SGD.

### Weaknesses
1. My main concern is that a similar trick has been applied to existing libraries. The ContractTN (https://github.com/jemisjoky/ContracTN/blob/main/ContracTN_QTNML_2021_Workshop.pdf) library also uses the rescaling in the logarithm for numerical stability. The scaling factor is also based on the norm of the tensor factors. Could the authors compare with it and emphasize the difference?

2. The analysis in Section 3.4 seems nonrigorous. The authors assume the elements in the tensor factors follow standard Gaussian distribution with unit variance. This assumption needs to be justified. When training neural networks, we need to carefully adjust the initialization and learning rates for a stable training, e.g., https://arxiv.org/pdf/2310.17813. I guess it is also true for training MPS.

3. The authors do not mention the details of the baseline SGD. Using the TensorNetwork library (https://github.com/google/TensorNetwork), it can also achieve competitive performance on high-dimensional data like MNIST simply using Adam (https://arxiv.org/pdf/1906.06329). So I think more comparisons with existing libraries would be useful.

4. When training neural networks, normalization techniques like layer and batch normalization are important. So I am curious how would simply applying these techniques compare with the proposed one.

### Questions
Please see above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper consider density estimation by non-negative tensor network representations based on stochastic gradient descent optimization stabilized through logarithmic scaling factors enabling numerically stable and efficient learning of very large densities. The approach is compared to DMRG for Borne Machines (i.e. squared representations to form non-negative tensors) as well as previous work using SGD for optimizing squared representations as well as imposing a non-negative tensor network factorization demonstrating that the existing SGD based approach fail whereas the proposed LSF works when having many cores being computationally efficient when compared to DMRG based inference.

### Strengths
The approach enables simple SGD learning of non-negative tensor network decompositions for density estimation ensuring numerical stability and enabling large scale density estimation which is useful. The approach is further very simple to implement.

Originality: The approach is in my opinion not original and essentially the logarithmic scaling factors is very similar to the widely used logsumexp operation to achieve numerical stability when evaluating the log of densities constituted by non-negative sums (typically exponentiated but here formed by applying the \sigma operator to project from unconstrained to positive elements in the non-negative decomposition) with the log scaling factor similarly ensuring numerical stability.

Quality: The paper is insufficiently related to existing modeling procedures for density estimation by use of tensor decompositions and should include more extensive comparisons.

Clarity: The paper is clearly written and generally easy to follow. 

Significance: The results are compelling but the novelty is very limited of the procedure and straightforward. Furthermore, existing non-negative tensor decomposition procedures relying on the probabilistic circuit formulation does not suffer as far as I understand of the described problems of inability to scale to large dimensions and being hampered by numerical instability. I also believe the existing tensor train density estimation TTDE framework (see below) as part of their code address numerical instabilities. I thus find the papers contribution very limited and not sufficiently positioned in the literature.

### Weaknesses
The main weakness is that the approach is straightforward and the contribution of limited novelty. Although the approach is practical useful it is very simple and straightforward and using what appears to be very close to existing standard procedures to achieve numerical stability of density estimates exploring their log-domain representations and how log-scaling factors through the decomposition can decouple into factors suitable for numerical stable inference.

### Questions
There is a vast literature on tensor network based density estimation that is not accounted for and that can operate on high-dimensional problems to my understanding without numerical issues. The paper needs to carefully position and establish this work also in relation to this literature, that includes:
TTDE: Tensor Train Density Estimation using Riemmannian optimization considering the least squares objective and squared representation for non-negative tensors as presently considered for tensor train density estimation (TTD) i.e. identical to the presented MPS BM formalism in model structure.
https://proceedings.mlr.press/v161/novikov21a/novikov21a.pdf
This procedure to my understanding use similar tricks to stabilize inference by suitable scaling factors. Please check their code and compare how their procedure performs to yours in the high-dimensinal setting.

Tensor Ring Density Estimation (TRDE/TERM): The tensor ring density estimation procedure described n in:
https://arxiv.org/abs/2312.08075
considers log-likelihood as opposed to least squares loss minimization and should extend to high-dimensional problems.

Tensor network based probabilistic circuits: There is a recent review of probabilistic circuit and its relation to tensor network representations for density estimation including code examples etc., see:

Loconte, Lorenzo, et al. "What is the Relationship between Tensor Factorizations and Circuits (and How Can We Exploit it)?." Transactions on Machine Learning Research (02/2025).

Notably such non-negative decompositions using probabilistic circuits also rely on non-negative tensor network representations but by additionally inducing distributional constraints on the carts this ensures that the resulting tensor-network representation naturally form valid densities. In this context it is also unclear why similar constraints on the non-negative carts in equation 5 cannot be imposed? Such constraints also naturally handle the numerical instabilities encountered when fitting models and makes learning probabilistic circuit based tensor network structures even when the densities are very high-dimensional possible without numeric issues. In fact the log-scaling factors used in LSF seem to indirectly also explore this property. Tensor factorization based probabilitic circuits including the tensor train formalism has also been considered in:

N. Ciolli, M. Mørup and M. N. Schmidt, "TNSPC: Learning from Partially Observed Data Using Tensor Network Structured Probabilistic Circuits," 2025 IEEE 35th International Workshop on Machine Learning for Signal Processing (MLSP), doi: 10.1109/MLSP62443.2025.11204307.

with code provided considering also MNIST sized problems in the GitHub repository.

Non-negative tensor factorization procedures for density estimation: Non-negative tensor factorizations have been widely used for density estimation with associated EM-based procedures used for stable inference,  see also the framework presented in

https://arxiv.org/pdf/2405.18220

and references therein. These methods can also exploit that the non-negative decompositions naturally can be expressed by suitably normalized carts which also ensures numerical stability and scaling to high-dimensional problem in my understanding.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a numerically stable and fully automatic-differentiation-compatible method for training probabilistic tensor networks (PTNs) based on Matrix Product States (MPS). The authors identify that prior approaches—DMRG and vanilla SGD—either incur high computational cost or suffer from numerical instability. They introduce a logarithmic-scale-factor (LSF) training scheme that stabilizes gradient computation, enabling efficient end-to-end learning within mainstream deep-learning frameworks. Experiments demonstrate significant speed and memory gains, along with the ability to train on sequences 10× longer than before, while achieving competitive likelihood performance on benchmarks including MNIST and density estimation datasets.

### Strengths
1. Well-written paper which addresses critical limitations of prior PTN training methods, namely numerical instability and incompatibility with automatic differentiation (e.g., DMRG overhead) .

2. Analyzes SGD instability for PTNs and provides bounds showing exponential scaling in instability, motivating the method.

### Weaknesses
1. My major concern is about the application breadth: While MNIST and tabular benchmarks are reasonable, evaluation on larger or more modern datasets (e.g., CIFAR, text) would strengthen claims of broad usability. Could the author provide more experimental results on datasets such as CIFAR and ImageNet?

2. The MNIST experiment reports negative log-likelihood but does not include quality metrics such as FID. Given that the task is described as generative modeling, including FID (or qualitative samples) would strengthen the evaluation and demonstrate visual fidelity, not just likelihood performance.

3. In Table 3, not all MPS+LSF variants achieve large speedups. Can the authors explain when LSF is most critical and when its advantage shrinks?

4. (Minor) In line 457-458, it is said MPS_{\sigma+LSF} achieves 10x speedup than MPS_{BM+DMRG}. But in Table 3, it is MPS_{BM+LSF} has 10x speed up than MPS_{BM+DMRG} right?

5. Can the LSF scheme be applied to other tensor-network architectures (e.g., TT, TR, PEPS, MERA)? Or is it specific to the chain structure of MPS?

### Questions
Please see the weakness above.

### Soundness
2

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
In this paper the authors theoretically analyze the cause of numerical stability when learning parameters of MPS-based TPNs used SGD. Consequently, they develop a numerically stable approach for computing the negative log-likelihood through the use of logarithmic scale factors. They then show that their developed approach cab be used to process sequences that are 10x longer than managed by previous SGD-based approaches, while being 10x faster than alternative, numerically stable approaches relying on DMRG.

### Strengths
- The paper is overall well-written

- The authors clearly motivate the problem they're solving, supporting their motivation for the problem with empirical insights.

- The proposed approach avoids numerical overflow suffered by using SGD.

- The proposed approach leads to a 10x drop in latency compare to DMRG at a negligible decrease in negative log-likelihood

### Weaknesses
- I think the authors would benefit from summarizing their approach in the abstract. As it stand, the only information a reader is able to extract from the abstract is that the authors propose a simple approach for efficiently learning probabilistic tensor networks.

- In my opinion, the authors provide very little by way of intuition and hand-holding in section 3.5, which I understand to be their main contribution. A toy example with a figure would've been very helpful here as well. Moreover, I would've expected the authors to highlight the properties of the probabilistic models that make this possible, since I presume that these local normalizations are only possible due to the form of the MPS model.

- Consequently, I am left feeling a bit skeptical that the contribution in section 3.5 would be novel, as it is my understanding that it is simply a straight-forward application of the logsumexp trick, although I am happy to be corrected. For example, I believe that the EiNets paper makes use of a similar trick.

- I would've like to see more of a comparison between the proposed approach and DMRF in terms of NLL, in addition to MNIST.

### Questions
- What does line 4 in Figure 1 (a) correspond to? Does that denote matrix multiplication?

- The authors mention that DMRG is non-differentiable. I expected that to translate to some form or biased relaxation, and therefore a higher NLL. However, that does not seem to be the case, at least on the MNIST experiments. 

- Please see the "Weaknesses" section for a list of other questions/concerns.

### Soundness
3

### Presentation
3

### Contribution
2
