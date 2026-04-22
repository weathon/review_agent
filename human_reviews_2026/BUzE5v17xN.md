# Deep Coupling Learning for Solving PDEs

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 8, 4, 2

## Abstract
Physics-Informed Neural Networks (PINNs) represent a significant advancement in computational methods for solving partial differential equations (PDEs).
However, the adoption of deeper neural network architectures presents significant challenges, as they struggle to address differential-related complications that arise during the computation of derivatives over the input of PINNs.
These complications extend beyond traditional vanishing and exploding gradients to include vanishing and exploding differentials, with both phenomena becoming more severe as networks grow deeper.
By examining the computation graph of derivatives in deep neural networks, we identify key bottlenecks causing numerical instabilities in deep architectures. 
In response, we introduce a novel approach that utilizes Coupling Layers with carefully regulated spectral norms of Jacobian matrices to stabilize and facilitate deep PINN training, effectively addressing differential-related challenges and improving model stability.
Our proposed architecture successfully mitigates the fundamental constraints of deeper PINNs while maximizing their capabilities through consistent differential propagation.
Comprehensive evaluations show that our approach surpasses conventional shallow PINN methods and alternative deep PINN designs across a range of challenging problems, particularly in cases featuring high-frequency solution components.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper solves a key problem with PINNs. Deep PINNs fail to train because derivatives either vanish or explode during computation. The authors propose couplednet, a new architecture that controls the size of Jacobian matrices at each layer. The design uses coupled blocks to keep the determinant equal to 1, preventing derivatives from vanishing. It also uses layernorm and output normalization to prevent derivatives from exploding. Experiments show that couplednet can train networks with 50 layers successfully. It achieves significantly better accuracy than existing methods on high-frequency and complex PDEs.

### Strengths
1.This paper identifies a distinct problem beyond traditional gradient issues. It shows that derivative explosion is separate from gradient vanishing in PINNs. This insight is very good and well-supported by theoretical analysis and these experiments.
2. This paper demonstrates strong mathematical reasoning throughout. It provides detailed gradient computation analysis in Appendix. This paper provides formal proofs showing that coupledNet maintains spectral norm control while preserving universal approximation capability. The connection between Jacobian determinants and derivative stability is mathematically rigorous.
3. The experiments systematically demonstrate the depth-performance relationship. The paper shows clear evidence that deeper coupledNet consistently improves accuracy, while baseline methods fail or degrade with depth. Results on high-frequency PDEs are particularly strong.

### Weaknesses
1. The authors acknowledge that couplednet is slower than PirateNet but provide no quantitative analysis or optimization strategies. This is a significant practical limitation that needs more discussion.
2. The output layer normalization restricts the method to PDEs with bounded solution ranges. The paper does not clearly specify when this constraint applies.
3. CoupledNet underperforms on simpler benchmarks (Allen-Cahn, Burgers). The paper does not adequately explain why the method excels on complex PDEs but struggles on simpler ones.

### Questions
1. The paper shows strong results on high-frequency PDEs but weaker results on some benchmarks. What problem characteristics make CoupledNet superior? Is there a theoretical criterion to predict when this architecture is beneficial?
2. All experiments use relatively small 2D domains with known analytical solutions. How does couplednet perform on real physical systems without ground truth?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper addresses the problem of vanishing and exploding differentials when training Physics-Informed Neural Networks (PINNs) with deep architectures, both theoretically and empirically, and it presents an architecture called CoupledNet to overcome these issues. The paper theoretically links the problem of stability in automatic differentiation for physics-informed loss functions to the value of the spectral norm of the hidden layers’ Jacobian matrix. By controlling the magnitude of this spectral norm, the proposed CoupledNet architecture ensures stable and robust training of PINNs. Finally, the authors demonstrate that CoupledNet enables deep PINNs to accurately solve high-frequency and challenging PDE problems, offering new insights into the spectral bias of PINNs and advancing the understanding of their training dynamics.

### Strengths
Among the main strengths of the paper, I highlight the following:

- A comprehensive literature review of Physics-Informed Neural Networks (PINNs) and their training challenges.  
- A clear theoretical analysis of PINN training dynamics, demonstrating that when the norm of the Jacobian matrix of the network output with respect to the input is smaller or larger than one, the derivatives respectively vanish or grow exponentially with depth, leading to vanishing or exploding gradients.  
- An original application of normalizing flow architectures to control the lower bound of the spectral norm, resulting in a more stable and effective architecture for training PINNs.  
- A robust theoretical and experimental validation pipeline, supporting all major claims of the paper:  
  1. *Effectively addresses differential-related challenges* → Figure 2  
  2. *Facilitates deep PINN training* → Figure 3  
  3. *Improves model stability* → Proposition 3.1 + spectral norm analysis of Jacobian matrices  
- Extensive PDE benchmark evaluations, demonstrating the generality and robustness of the proposed approach.

### Weaknesses
- The proposed method can be viewed as an application of normalizing flow architectures, which can be highly valuable for the PINN community, although its impact outside this domain might be more limited.  
- The discussion on spectral bias could be strengthened, particularly by including additional experimental results from the literature that support your hypothesis. Examples include the Poisson 1D equation or the wave equation discussed in [1].  
  Furthermore, it has been analytically shown in [1] that PINNs are biased toward learning functions along the dominant eigen-directions of their limiting NTK. Specifically, the leading eigenvectors corresponding to large eigenvalues of the NTK determine the frequencies that the network learns first.  Therefore, it is important to analyse the differences in the NTK eigen-decomposition for a standard feedforward network, PiratNet, and CoupledNet, because it could be that CoupledNet results in a better spectrum than the standard Feedforward network, validating the already established theory in [1].

[1] Wang, Sifan, Hanwen Wang, and Paris Perdikaris. "On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks." Computer Methods in Applied Mechanics and Engineering 384 (2021): 113938.

### Questions
- It would be helpful to include loss convergence plots for the different loss components and compare them with PiratNet for a clearer assessment of performance.  
- Regarding the JaxPi benchmark, the baseline approach uses time-window training by splitting the \([0, 1]\) domain into 10 subdomains and training separate neural networks for each. Do you follow a similar approach? If not, what are the main differences in your training procedure compared to the JaxPi baseline? This distinction should be explicitly highlighted, and ideally, a similar training paradigm should be used to ensure a fair comparison.  
- Please specify which software framework you used (e.g., TensorFlow, PyTorch, JAX) and include this information in the main text, and ideally, can you provide a link to the code repository to support reproducibility.  
- Why did you not include a comparison with CoupledNet + NTK in Table 1?  
- Equation (6) should use a total derivative, $\frac{d}{dx}$, instead of a partial derivative,$\frac{\partial}{\partial x}$.  
- On page 4, it should read *“normalizing flow architectures”* instead of *“normalization flow architectures.”*  
- In Equation (4), $x_1$ should be bold.  
- On page 4, the expression should be  
 $ 
\mathbf{s} = \exp(\text{LayerNorm}(\sigma(\mathbf{W}_2 \sigma(\mathbf{W}_1 x_1 + \mathbf{b}_1) + \mathbf{b}_2)))
$
  and not  
$
  \mathbf{s} = \exp(\text{LayerNorm}(\sigma(\mathbf{W}_2 \sigma(\mathbf{W}_1 x_1' + \mathbf{b}_1) + \mathbf{b}_2))).
$

### Soundness
4

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
4

### Summary
The paper diagnoses a depth-related failure mode in PINNs: vanishing/exploding differentials caused by products of layer-wise Jacobians when computing derivatives required by PDE residuals. To stabilize these derivatives, the authors propose CoupledNet, built from Coupled Blocks inspired by normalizing flows. Each block preserves a controlled Jacobian to enforce a lower bound on the spectral norm (preventing vanishing derivatives) and introduces two mechanisms to contain the upper bound: (i) a probabilistic bound on the LayerNorm’s variance and (ii) a runtime output-layer constraint that scales features by tracked max $s_i$ across blocks. They prove a simple spectral property (∥J∥₂≥1) and a universal approximation result via a degenerate construction. In experiments, the authors show stable first-order derivative norms at initialization across depth and CoupledNet’s error decreases monotonically with depth (up to 50 layers), unlike MLP/ResNet PINNs that either vanish or explode. On high-frequency PDEs (Eq. 7), CoupledNet attains the lowest residual and relative L2 errors vs PirateNet and Jaxpi (Table 1). On a harder PDE (Eq. 8), a 16-layer CoupledNet (rel. L2 ≈ 0.0718) outperforms deeper PirateNet variants.

### Strengths
1.	The paper reframes PINN instability as derivative pathology and targets it architecturally with coupling layers that explicitly control Jacobian determinants/spectral properties, rather than only reweighting losses or injecting frequency features. This offers a fresh axis of control compared to ResNet-style skips or NTK-based training heuristics.

2.	The paper clears analysis linking PINN residual gradients to products of Jacobians (Appendix A), motivating why depth aggravates differentials even if output-gradients are okay. The spectral claims are formalized (Prop. 3.1) and supported by initialization diagnostics (Fig. 2). Ablations across depth (Fig. 3) convincingly show the architecture drives stability.

### Weaknesses
1.	The determinant-based argument relies on square, equal-width hidden layers and a bias-free LayerNorm ensuring $\sum \log s_i=0$ (det $=1$). Real implementations often vary widths; input/output layers are non-square; permutations complicate block-structure.
2.	The Upper-bound control feels heuristic. The LayerNorm variance bound assumes independence and log-normality; the output-layer constraint only applies when solution ranges are bounded and may change the effective scale of the PDE residual. 
3.	For upper-bound analysis, the paper explicitly assumes absence of permutation layers to get block-diagonal Jacobian products; yet the actual CoupledBlocks do use permutations, then argues this is harmless because a permutation matrix has spectral norm $\|P\|_2=1$. This statement is necessary but insufficient: $\|P\|_2=1$ only guarantees permutations don't increase a given norm via submultiplicativity; it doesn't restore the block-diagonal structure used to get the simple product-of-diagonals bound, nor does it address cross-block mixing created by interleaving permutations with scaling.

### Questions
1.	regard of weakness 1, could you clarify and test: (a) unequal widths, (b) non-square transitions, and (c) the effect of permutations on the claimed lower bound? 
2.	How is  ∣det J∣=1 enforced numerically across blocks? Is it purely a consequence of the bias-free LayerNorm in the exp(·) path, or clamp/renormalize s during training? What is the observed drift of $\sum \log s_i$ in practice?
3.	Could the authors provide a controlled ablation on one representative PDE with: (i) determinant control only, (ii) LayerNorm- $\sigma$ only, (iii) output-layer constraint only?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses numerical instabilities in deep Physics-Informed Neural Networks that arise during automatic differentiation by a new network architecture. It regulates the spectral norms of Jacobian matrices through determinant constraints (det(J)=1), enforcing both a lower bound and an upper bound.

### Strengths
- The paper clearly identifies the issue in deep PINN training (differential instabilities) beyond standard vanishing/exploding gradients (compared to ResNet).
- Theories around CoupleNet are established (e.g., Universal Approximation Theorem)

### Weaknesses
- While the paper shows strong results on adviction, HJB, and custom high-frequency PDEs, the performance on the Jaxpi benchmarks in Table 2 is not impressive. CoupledNet falls behind the PirateNet 4/6 times, by a large margin (over 2x). The authors argue that CoupleNet is more advantageous in high-freq problems, but for example Burgers and GS are such problems where CoupledNet still suffers.
- Lack of Quantitative Efficiency Analysis. The authors say, "CoupledNet’s computational efficiency was lower than that of PirateNet." But it is impossible for the reader to assess the practical cost of the proposed architecture.
- As a new architecture, it needs additional experiments on for example training stability and efficiency (how difficult it is to optimize), sensitivity to hyperparameters, parameter efficiency, memory usage and scaling laws. It is not necessary to provide everything but quite many are not there.
- The architectural choices are somewhat heuristic. Although it satisfies some guarantees, it does not show optimality.

### Questions
Many are mentioned in Weaknesses. Plus, for Figure 2, it could be better to show the distribution of the gradient norms (instead of the total gradient norm) and how the distribution shifts as training goes.

### Soundness
3

### Presentation
3

### Contribution
2
