# KANO: Kolmogorov-Arnold Neural Operator

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
We introduce Kolmogorov–Arnold Neural Operator (KANO), a dual‑domain neural operator jointly parameterized by both spectral and spatial bases with intrinsic symbolic interpretability. We theoretically demonstrate that KANO overcomes the pure-spectral bottleneck of Fourier Neural Operator (FNO): KANO remains expressive over a generic position-dependent dynamics for any physical input, whereas FNO stays practical only to spectrally sparse operators and strictly imposes fast-decaying input Fourier tail. We verify our claims empirically on position-dependent differential operators, for which KANO robustly generalizes but FNO fails to. In the quantum Hamiltonian learning benchmark, KANO reconstructs ground‑truth Hamiltonians in closed-form symbolic representations accurate to the fourth decimal place in coefficients and attains $\approx6\times10^{-6}$ state infidelity from projective measurement data, substantially outperforming that of the FNO trained with ideal full wave function data, $\approx1.5\times10^{-2}$,  by orders of magnitude.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces the Kolmogorov–Arnold Neural Operator (KANO), a novel neural operator that learns pseudo-differential symbols in both spatial and spectral domains using Kolmogorov–Arnold Networks (KANs). The authors identify a theoretical limitation in Fourier Neural Operators (FNOs), the "pure-spectral bottleneck", which restricts their expressivity for position-dependent dynamics. They provide a rigorous theoretical analysis showing that FNOs suffer from super-exponential scaling in model size for such operators, while KANO achieves polynomial scaling. Empirical validation includes synthetic PDE operator learning and quantum Hamiltonian learning, where KANO demonstrates strong out-of-distribution generalization, high parameter efficiency, and symbolic recovery of operator forms.

### Strengths
This paper makes a significant and well-structured contribution to SciML. It identifies a fundamental flaw in a popular architecture, proposes a principled solution, and provides convincing evidence of its superiority. The authors present a rigorous theoretical diagnosis of FNOs, exposing the “pure-spectral bottleneck.” They show that for position-dependent dynamics, including quantum mechanics and fluid flow, FNOs suffer from super-exponential scaling. This is framed not as a minor drawback but as a deep architectural flaw.

In response, the KANO is both novel and well aligned with the mathematics of the problem. By jointly parameterizing operators in spatial and spectral domains through a pseudo-differential framework, KANO represents each term in its naturally sparse basis. This dual-domain approach provides the right inductive bias for robust learning.

The empirical results reinforce the theory with striking effect. KANO does not merely improve modestly over FNO but shows dramatic gains: flawless out-of-distribution generalization, 0.03% of the parameters, and symbolic operator recovery accurate to the fourth decimal. This interpretability moves the model from black-box approximation toward genuine scientific discovery.

### Weaknesses
The main weakness is the narrow experimental validation. Results are confined to 1D synthetic operators and quantum systems, leaving scalability to high-dimensional PDEs untested. Absent are standard benchmarks such as 2D Navier–Stokes or 3D elasticity, along with any runtime or stability analysis, making the practical utility unclear.

Comparisons are also limited. Most results benchmark only against vanilla FNO, whose flaws are already established, creating a straw-man dynamic. Stronger baselines like U-FNO, AM-FNO, PDNO, or multi-scale FNO variants are missing, making it hard to judge whether KANO’s advantage stems from its design or from outdated comparisons.

Finally, while symbolic recovery is compelling, it remains demonstrated only in simple, smooth cases. Whether this interpretability extends to non-smooth, higher-dimensional, or more complex operators is left open. Overall, the idea is strong, but the evidence is not yet broad enough to confirm its readiness for widespread use.

### Questions
How does KANO perform on widely used PDE benchmarks such as 2D Navier–Stokes or heterogeneous material modeling?

Could you include comparisons with stronger baselines like U-FNO, PDNO, or adaptive FNO variants?

What are the runtime and memory costs of KANO compared to FNO during training and inference? Given the dual-domain computations, how does training time scale?

How robust is symbolic recovery when operator coefficients are non-smooth or discontinuous?

What optimization challenges or instabilities were encountered when training the KAN sub-networks within the operator framework, and how were they addressed?

Did you perform ablation studies to understand the individual contributions of the dual-domain design versus the KAN sub-networks? For instance, what is the performance of a KANO variant that uses MLPs instead of KANs?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors proposed the Kolmogorov-Arnold Neural Operator KANO that uses the KAN to construct the neural operator, which has very solid advantages over FNO both from theory and experiment.

### Strengths
1. The explanation of why the FNO have problems is very clear. Strongly support the contribution and advantages of the proposed KANO.

### Weaknesses
1. The idea is quite natural, if picky. But it does not affect the contribution.
2. The experiment is quite limited. I think this method should benefit a lot from the flexibility of the KAN. Extra experiments of diverse operators can make this paper attractive to a broader audience.

### Questions
1. Are there other results that demonstrate the advantages of this method?

2. For the demonstration of the spectrum limitation of FNO, the author uses the linear approximation. Will the off-diagonal contribution arise in higher orders? What will that affect in the argument in this paper?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors consider the neural operator learning setup and consider a class of operators which involves spatial multipication and term it as position dependent dynamics to highlight the shortcoming of the existing fourier neural operator framework. Since the multiplication and differentiation have a dual relationship under the fourier transform, the authors argue that the considered class defined by the multiplication map is just a two step up-shift sparse matrix in the spatial basis, however, takes the form of a dense Teoplitz matrix in the spectral basis and the fourier operator cannot efficiently approaimte the off-diagonal elements of this matrix. The authors first show that this position dependent dynamics will induce a super-exponential scaling in the fourier operator. To adress this issue the authors propose the KANO framework which replaces the nodes of standard MLP with simple sum operations and learn the univariate 1D functions. This reparameterization is similar in expressivity to MLPs upto constant depth and width empirically. Also, they propose using a KAN sub-network jointly parameterized by both spatial and spectral basis and enjoys sparse representations in both spatial and spectral domains. They also discuss that the projection error scales efficieitnyl by its width and the latent network error also scales efficiently independent of the projection error and thus, it is more efficient as compared to the FNO for the considered class. Furthermore, they also verify this empirically by considering a dataset of the considered class where they observe that KANO outperforms FNO with much lesser size required.

### Strengths
Investigating the bottlenecks of the FNO by considering this position dependent dynamics class and then drawing the inference that this will incur dense representation in one basis and sparse in another is really interesting. Further utilising the KA network based parameterization and resolving this issue is also interesting and provides new insights for various scenarios of this operator learning framework. Further verification by empirically investigating this under different testing /training families and demonstrating that KANO can outperform FNO with much lesser paramter. Also the section 5.2 further including two-position dependent quantum dynamics benchmark further strenghten their claims.

### Weaknesses
No major weaknesses but the authors could have also compared the FNO and proposed KANO on the original equations considered in the FNO paper to see how does it perform in that setup to understand its overall use case as against this specific class. It is a bit unclear how this new framework will do on the  other tasks that FNO can perform very efficiently.

### Questions
Since the FNO based layers have been also used for a lot of other tasks like weather forecasting [1] or like token mixers for transformers [2], do the authors have any comments on the broader applicability of this framework?

[1] Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators.
[2] Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers. ICLR 2022.

### Soundness
3

### Presentation
3

### Contribution
3
