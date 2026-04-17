# On the Convergence of Two-Layer Kolmogorov-Arnold Networks with First-Layer Training

- Decision: Accept (Poster)
- Scores: 8, 2, 4

## Abstract
Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to traditional neural networks, offering enhanced interpretability based on the Kolmogorov-Arnold representation theorem. While their empirical success is growing, a theoretical understanding of their training dynamics remains nascent. This paper investigates the optimization of a two-layer KAN in the overparameterized regime, focusing on a simplified yet insightful setting where only the first-layer coefficients are trained via gradient descent.

Our main result establishes that, provided the network is sufficiently wide, this training method is guaranteed to converge to a global minimum and achieve zero training error. Furthermore, we derive a novel, fine-grained convergence rate that explicitly connects the optimization speed to the structure of the data labels through the eigenspectrum of the KAN Tangent Kernel (KAN-TK). Our analysis reveals a key advantage of this architecture: guaranteed convergence is achieved with a hidden layer width of $m=\mathcal{O}(n^2)$, a significant polynomial improvement over the $m=\mathcal{O}(n^6)$ requirement for classic two-layer neural networks using ReLU activation functions and analyzed within the same Tangent Kernel framework. We validate our theoretical findings with numerical experiments that corroborate our predictions on convergence speed and the impact of label structure.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a theoretical study of the optimization dynamics of two-layer Kolmogorov–Arnold Networks (KANs) in the overparameterized regime, under the setup where only the first-layer coefficients are trained and the second-layer coefficients remain fixed. The authors derive global convergence guarantees for gradient descent and provide a label-dependent convergence rate that depends on the eigenspectrum of the KAN Tangent Kernel (KAN-TK).

Their main results show:

Global convergence with zero training error for sufficiently wide KANs.

Polynomial width scaling of n^2, is much smaller than MLPs

A fine-grained convergence bound connecting label alignment with KAN-TK eigenvectors to convergence speed.

Empirical experiments verifying width-dependent and label-structure-dependent convergence.

### Strengths
Timely theoretical contribution.
KANs are a rapidly growing architecture class (with numerous 2024–2025 references), yet theoretical understanding of their training dynamics is scarce. This work fills an important gap by analyzing convergence properties in a simplified yet meaningful setup.

Mathematical rigor.
The paper provides clear theorems (Theorem 4.1 and 4.5) with appropriate assumptions and proof sketches. The use of lazy training analysis and the explicit derivation of the KAN Tangent Kernel are technically sound and well-aligned with the NTK literature.

Improved scaling over MLPs.

Label-dependent convergence analysis.
Theorem 4.5 and the supporting experiments (Figures 3a–b) provide nice intuition about how label structure interacts with the kernel spectrum — a useful insight beyond the standard NTK treatment.

Experimental verification.
Although synthetic, the experiments are cleanly executed and directly corroborate the theoretical claims (width scaling and label alignment).

### Weaknesses
Limited novelty beyond applying NTK analysis.
While the paper is well-executed, its main theorems closely parallel classical NTK results (Du et al., 2019; Arora et al., 2019), with the main novelty being the explicit KAN-TK form. The analysis may be perceived as incremental rather than fundamentally new.

Suggestion: Clarify what features of KAN-TK qualitatively differ from NTK (e.g., structure induced by spline or RBF bases) and how they affect learning beyond improved scaling.

No discussion of second-layer training dynamics.
The paper emphasizes “first-layer only” training for tractability, but it remains unclear how this extends to full training, or whether the key conclusions persist when both layers are updated. For KANs the 2 layer setting is more realistic than MLPs though

Missing discussion on practical implications.
Although the theory is strong, the practical takeaway (e.g., how this informs KAN training in real models like FastKAN or WavKAN) is underdeveloped. Readers would appreciate even brief remarks connecting this to design choices (e.g., number of basis functions, initialization scales). It would be good to cite the KAN 2.0 paper on scientific applications, and also the study of initialization schemes for KANs https://arxiv.org/pdf/2509.03417

### Questions
please provide motivations 
1. why KAN is O(n^2) much smaller than MLP O(n^6); this is not clear to me
2. training only first layer is a bit weird, can you train only the second layer instead (more like random feature KAN, which would be interesting)

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyses the convergence behaviour of Kolmogorov-Arnold Networks (KAN) with the NTK framework, proving that they require less overparameterisation compared to MLP.
The proof is based on standard NTK convergence proof based on stability of parameter, which leads to the lazy training, concluding with the constancy of NTK throughout training and its equivalence to kernel gradient descent.
The supporting experiments show that wider KAN shows faster convergence behaviour, and also validates the lazy training phenomenon.

### Strengths
The organisation of the paper is clear, the main claim and strength of the paper, proof sketch, comparisons to the prior research are easy to follow.
Theoretical results are stated without ambiguity, and the detailed requirement of the network hyperparameters are exactly described.
The proof in the supplementary material is also easy to read, each of the derivations describes which computations were performed, and each step is well separated to follow.
Two experiments in the main text support the main claims of the theoretical results, like global convergence and lazy training in Figure 2. 
Figure 3 and Figure 6 in the supplementary material shows interesting behaviours of KAN-TK, which allows one to understand inductive bias of overparameterised KAN.
Finally, the theoretical results obtained is well analysed, highlighting the gains (less overparameterisation w.r.t. sample size, less dependency on condition number) and losses (stronger requirement on learning rate, and slower convergence).

### Weaknesses
Kolmogorov-Arnold network has gained interest due to its interpretability, particularly for its applications in scientific discovery.
Therefore, it is questionable whether the overparameterised KAN remains interpretable, which seems unlikely according to the proof techniques. 
Specifically, lazy training implies the parameters remain close to their initialisation, making them nearly random, and overparameterisation prevents direct interpretation..
Furthermore, one of the core details for interpretability in (Liu et al., 2025) is the use of sparsification and pruning, which incorporates L1 or entropic regularisation.
This makes this paper's setting far from the practical use and the global convergence proof less interesting.

While the paper claims to prove the superior convergence behaviour of KANs compared to MLPs, the NTK-based convergence proof (from Du et al., 2019) is not a state-of-the-art technique for proving the global convergence of neural networks. 
For instance, Table 1 in (Polyaczyk and Cyranka, 2024) presents multiple sharper convergence rates. (Poyaczyk and Cyranka, 2024) proves a stronger bound of $n^{1.25}$, and under a similar data distribution assumption, (Liu et al., 2022) shows better convergence behaviour.

In the assumptions of Section 4, the positive-definite kernel property applies only to MLPs; therefore, the proof needs to be modified for KANs.
To the best of the reviewer's knowledge, this property holds if (1) the activation function is analytic and non-polynomial (Prop F.1 of Du et al., 2019) or (2) the Hermite polynomial expansion of the activation function has infinitely many non-zero coefficients (a modification of Thm 3.2 of Nguyen et al., 2021). 
Conversely, it can be shown that if the activation function is polynomial, the infinite-width NTK has a finite rank, causing the minimal eigenvalue to become zero when the sample size is sufficiently large.
This may rule out some of the basis functions described in the paper, such as Chebyshev polynomials.
Therefore, it will be helpful if the authors discuss what basis functions will satisfy the assumptions.

While the experiments support the claim of theoretical results, a gap exists between the theoretical result and experimental result.
In Figure 2(b), although wider networks show a smaller weight distance from initialisation, it is unclear whether these distances are upper-bounded throughout the training as claimed in Lemma 4.2.
It would be better if the training time were extended to show a plateau in the distance plot.
Furthermore, one of the paper's main claims is a reduced dependency on sample size; hence, it would be beneficial to show how convergence or lazy training is affected as the number of samples increases.

The target functions used in the experiments are nearly trivial, employing either random labels or $y=x$.
Considering that Figures 6 and 7 show more challenging target functions, it would be better to use such functions in these experiments, or, if possible, to use the target functions or tasks from (Liu et al., 2025).

I am willing to increase my score to 4 if the third, fourth, and fifth issues in this section are addressed. 
Furthermore, I am willing to raise my score to 6 or 8 if the authors provide promising theoretical results on interpretable global convergence (addressing the first weakness) or demonstrate a faster convergence rate that substantiates their claim (addressing the second weakness).

(Liu et al., 2025) KAN: Kolmogorov-arnold networks, ICLR 2025. https://openreview.net/forum?id=Ozo7qJ5vZi

(Du et al., 2019) Gradient descent finds global minima of deep neural networks, ICLR 2019. https://openreview.net/forum?id=S1eK3i09YQ

(Polyaczyk and Cyranka, 2024) Improved Overparametrization Bounds for Global Convergence of SGD for Shallow Neural Networks, TMLR. https://openreview.net/forum?id=RjZq6W6FoE

(Liu et al., 2022)  Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. Applied and Computational Harmonic Analysis, 2022. https://www.sciencedirect.com/science/article/pii/S106352032100110X

(Nguyen et al., 2021) Tight Bounds on the Smallest Eigenvalue of the Neural Tangent Kernel for Deep ReLU Networks, ICML 2021. https://proceedings.mlr.press/v139/nguyen21g.html

### Questions
1. Is slower convergence rate by (Gao & Tan, 2025) result of training both layers, or loose analysis? Can the proof strategy by authors be used to improve (Gao & Tan, 2025)’s result?
2. On the other hand, was it impossible to prove the same dependency of overparameterisation on the sample size while training both layers? If so, does it actually require larger width empirically?  
3. Can you give a table of the assumptions that each basis functions mentioned in the last paragraph of Section 2.1 satisfy? 

(Gao & Tan, 2025) On the convergence of (stochastic) gradient descent for kolmogorov–arnold networks. IEEE Transactions on Information Theory, 2025. https://ieeexplore.ieee.org/document/11079726

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper provides a theoretical analysis of two-layer Kolmogorov-Arnold Networks (KANs) in the overparameterized regime, focusing on the simplified case where only the first-layer coefficients are trained via gradient descent while the second layer remains fixed. The authors prove that under mild assumptions and sufficient network width, gradient descent converges to a global minimum with zero training error. They further derive a label-dependent convergence rate, linking optimization dynamics to the eigenspectrum of the proposed KAN Tangent Kernel (KAN-TK). Theoretical results are supported by synthetic experiments validating convergence behavior and illustrating the impact of label alignment on optimization speed.

### Strengths
1. The paper provides the first convergence analysis for two-layer KANs under partial parameter training (first layer only), which meaningfully extends existing theoretical work on neural tangent kernel dynamics to a new and interpretable architecture.
2. Experiments, though simple, effectively confirm theoretical predictions, particularly the “lazy training” regime and label-structure-dependent convergence.

### Weaknesses
1. Limited experimental setting. The experimental validation is confined to synthetic datasets and simple settings. It remains unclear whether the theoretical findings hold in realistic, high-dimensional KAN applications (e.g., vision).
2. Incomplete comparison to full-layer training. Although Table 1 provides asymptotic comparisons, the paper lacks experimental or theoretical discussion on how fixing the second layer affects expressivity or generalization, which could impact interpretability and optimization flexibility.
3. Potential overemphasis on width reduction. While the polynomial improvement from $O(n^6)$ to  $O(n^2)$ is mathematically striking, the constants and dependence on other factors (e.g., $d^2g^6/\lambda_0^2$) may diminish practical advantage, which is not empirically examined.

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
3
