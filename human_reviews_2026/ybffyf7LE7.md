# Iterative Training of Physics-Informed Neural Networks with Fourier-enhanced Features

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Spectral bias, the tendency of neural networks to learn low-frequency features first, is a well-known issue with many training algorithms for physics-informed neural networks (PINNs). To overcome this issue, we propose IFeF-PINN, an algorithm for iterative training of PINNs with Fourier-enhanced features. The key idea is to enrich the latent space using high-frequency components through Random Fourier Features. This creates a two-stage training problem: (i) estimate a basis in the feature space, and (ii) perform regression to determine the coefficients of the enhanced basis functions. For an underlying linear model, it is shown that the latter problem is convex, and we prove that the iterative training scheme converges. Furthermore, we empirically establish that Random Fourier Features enhance the expressive capacity of the network, enabling accurate approximation of high-frequency PDEs. Through extensive numerical evaluation on classical benchmark problems, the superior performance of our method over state-of-the-art algorithms is shown, and the improved approximation across the frequency domain is illustrated.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes IFeF-PINN, an algorithm for iterative training of PINNs with Fourier-enhanced features. The core of IFeF-PINN is to enrich the latent space using high-frequency components through Random Fourier Features (RFF).  This work provides detailed theoretical derivations for several key conclusions. Experiments also show that IFeF-PINN outperforms other SOTA methods.

### Strengths
- The paper is well-written and easy to follow.
- The claims of this paper are substantiated by multiple theoretical derivations.
- The effectiveness of IFeF-PINN is validated on various problems.

### Weaknesses
- The two-stage training process is a key claim of this work, but it lacks sufficient justification and experimental evidence to demonstrate its necessity. The reviewer acknowledges the importance of decoupled representation and downstream tasks (e.g., large-scale pretraining). However, as the significance of this pipeline in the scope of PINNs is not yet established, the authors should provide a more thorough discussion. An ablation study comparing against an end-to-end training approach using RFF basis extension would help clarify this issue.
- The paper lacks comparison with a basic baseline: the method described in Equation 4. The reviewer disagrees with the authors' claim in Line 155 ("when the input is of low dimension, such as for PINNs, this could limit its asymptotic approximation capabilities"). The method in Equation 4 has been successfully validated in NeRF (NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis), which also utilizes low-dimensional inputs (3D location and 2D viewing direction).
- IFeF-PINN involves multiple training phases, potentially requiring more training steps. The authors mention in Line 344 that baseline methods are trained/evaluated with their default settings. The reviewer questions the fairness of comparisons and suggests that ensuring all methods use the same total number of training steps would provide a more equitable evaluation.
- The impact of the number of Fourier-enhanced features (D) on performance is unclear. Is there a correlation between the value of D and downstream task performance?

### Questions
Please refer to the weaknesses. The reviewer would consider raising the score if the first two weaknesses are adequately addressed.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To address the spectral bias of PINNs, IFeF-PINN is proposed. It is an algorithm for iterative training of PINNs with Fourier-enhanced features, enriching the latent space using high-frequency components through Random Fourier Features. This creates a two-stage training problem: (i) estimate a feature basis, and (ii) perform regression to determine the coefficients. For an underlying linear model, it is shown that the latter problem is convex and that the iterative training scheme converges. Numerical evaluation on classical benchmark problems shows the superior performance of IFeF-PINN over SOTA algorithms is shown.

### Strengths
* (Clarity) The paper is easy to follow. Contributions are well clarified and compared with other studies addressing the spectral bias.
* (Technical contribution) The proposed methods effectively combines PINNs and RFFs in a novel way, addressing the limited approximation ability of RFFs in low dimensions.
* (Broader impact & Technical contribution) An application to general neural architectures is straightforward, with some careful treatments described in Remark 2 in Section 2.
* (Theoretical contribution) Uniqueness & existence of the solution of the lower-level problem (Proposition 1), Lipchitz continuity of the solution map (Proposition 2), convergence to a stationary point (Theorem 1), and the universal approximation (Theorem 2 & Corollary 1) are proved, ensuring theoretical soundness of the proposed method.
* (Empirical contribution) The proposed method outperforms baselines.
- (Reproducibility) Error bars are provided.

### Weaknesses
- Although the theoretical convergence is proved, the proposed method requires a bilevel optimization, which may cause training instability, necessitating warm-start training outlined in Section 3 (please correct me if I missed singing); that said, pretraining or warm-start is often required when training PINN's variants.
- Currently, the code is not available, although it will be released upon acceptance (line 353). Could the authors provide the code? Just a snippet is acceptable.


## Review summary

The paper is well-written and easy to follow. The contribution of the proposed method is clear. I am currently inclined to recommend its acceptance.

### Questions
- Can the proposition 1 be extended to nonlinear problems?

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
4

### Summary
The paper addresses the challenge of spectral bias in physics-informed neural networks (PINNs) by introducing an iterative training framework that utilizes random Fourier features (RFFs). The approach is based on two types of bases, nominal and extended, and applies a regression method on the extended basis coefficients to learn the PDE solution. The method is evaluated on canonical PDE problems. The paper claims that it is potentially generically applicable to different neural PDE solvers. However, that is not shown in the current version. Some aspects of the paper are unclear, for instance, the current write-up, particularly the detailed construction of the basis, and the motivation for the chosen network layer for RFF extension. Also, the empirical demonstration on nonlinear and highly oscillatory PDEs is a limitation.

### Strengths
1. The paper addresses a well-known and important problem of spectral bias in PINNs.

2. The proposed approach is conceptually general and can be integrated with various PINN frameworks.

3. Provides theoretical convergence analysis and demonstrates improvements on canonical problems.

### Weaknesses
1. The paper is unclear in how the nominal and extended bases are created or selected, and lacks an explanation for the motivation behind modifying the last layer to improve spectral properties. The presentation of the methodology could be improved.

2. The discussion of the bi-level optimization framework is limited, and different possible formulations and the rationale for the chosen structure are not justified.

3. Computational aspects, such as training cost, convergence behavior, and initialization of the bi-level optimization, are not adequately discussed and compared with prior works.

4. The numerical experiments primarily focus on linear PDEs; the method’s effectiveness for nonlinear oscillatory problems is not demonstrated.

5. A comparison with related approaches, such as [1] and [2], is lacking, and no clear discussion is provided on computational trade-offs or advantages over these methods.

[1] De Ryck, Tim, et al. "An operator preconditioning perspective on training in physics-informed machine learning." The Twelfth International Conference on Learning Representations.

[2] Moseley, Ben, Andrew Markham, and Tarje Nissen-Meyer. "Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations." Advances in Computational Mathematics 49.4 (2023): 62.

### Questions
1. Could the authors clarify in detail how the nominal and extended bases are constructed and why the RFF extension is applied specifically at the last layer?

2. What are the computational costs and convergence characteristics of the proposed method compared to [1] and [2]? How intensive is the iterative training in practice?

3. How is the initialization of both upper- and lower-level optimizations handled, and how sensitive is the performance to this initialization?

4. Can the authors include results on nonlinear PDEs (e.g., the KdV or Kuramoto–Sivashinsky equations) to demonstrate the method's generality beyond linear problems?

5. Would an ablation on hyperparameter choices (especially for the physics-informed Gaussians and the proposed method) help in understanding performance sensitivity?

6. If the approach primarily targets linear problems, could the authors explicitly position it in relation to [1] and [2], and clarify this limitation in the discussion?

Minor comment:

The formula for computing the relative L2 error is not stated in the paper.

### Soundness
2

### Presentation
2

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
The paper introduces an iterative training method (IFeF-PINN) that mitigates the spectral bias of Physics-Informed Neural Networks, favoring low-frequency solutions. It separates training into two alternating stages: learning a latent basis through a standard PINN, then extending it with Random Fourier Features (RFF) to capture high-frequency components, followed by a convex regression step. The authors prove convergence for linear PDEs and show that the RFF extension expands the network’s expressive power. Experiments on benchmark PDEs (Helmholtz, convection, Burgers, etc.) demonstrate that IFeF-PINN consistently outperforms existing PINN variants, achieving accurate high-frequency solutions and reduced spectral bias.

### Strengths
The paper excels in its clear, well-motivated algorithmic innovation and solid theoretical–empirical balance. 

First, it provides a principled bi-level reformulation of PINN training that explicitly decouples feature learning from coefficient regression. This structure not only clarifies optimization dynamics but also leads to provable convexity and convergence for linear PDEs, a strong theoretical contribution. 

Second, the integration of Random Fourier Features into the latent space is clever and effective. It directly targets spectral bias instead of relying on heuristics like adaptive weights or resampling. 

Third, the paper delivers rigorous ablation and benchmark comparisons (across Helmholtz, convection, Burgers equations) showing that IFeF-PINN achieves one to two orders of magnitude lower relative L2 error than strong baselines, especially under high-frequency and multi-scale regimes where PINNs typically collapse. 

Finally, the frequency-domain analysis using FFT provides quantitative evidence that the model indeed learns higher frequencies, reinforcing its claims with interpretable diagnostics.

### Weaknesses
A clear weakness of the paper is its limited treatment of nonlinear PDEs, where the proposed convex lower-level formulation breaks down. While the authors acknowledge that the lower regression problem becomes nonconvex in such cases (and suggest it to be promising direction), they did not offer quick remedies or convergence guarantee beyond gradient descent heuristics. This has limited its applicability as many practical PDEs of interest, such as fluid dynamics, reaction–diffusion systems, nonlinear elasticity, are inherently nonlinear. The experiments also mostly focus on linear or mildly nonlinear cases (e.g., Burgers’ equation), leaving open whether the method’s strong performance extends to more challenging nonlinear systems.

### Questions
How stable is the iterative bi-level optimization when the Random Fourier Feature dimension D becomes large or when the feature distribution is poorly scaled? Does the alternating update between basis learning and convex regression amplify noise or cause overfitting to spurious high-frequency components?

### Soundness
3

### Presentation
3

### Contribution
2
