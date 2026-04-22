# Generative Modeling from Black-Box Corruptions via Self-Consistent Stochastic Interpolants

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 0, 8

## Abstract
Transport-based methods have emerged as a leading paradigm for building generative models from large, clean datasets. However, in many scientific and engineering domains, clean data are often unavailable: instead, we only observe measurements corrupted through a noisy, ill-conditioned channel. A generative model for the original data thus requires solving an inverse problem at the level of distributions. In this work, we introduce a novel approach to this task based on Stochastic Interpolants: we iteratively update a transport map between corrupted and clean data samples using only access to the corrupted dataset as well as black box access to the corruption channel. Under appropriate conditions, this iterative procedure converges towards a self-consistent transport map that effectively inverts the corruption channel, thus enabling a generative model for the clean data. We refer to the resulting method as the self-consistent stochastic interpolant (SCSI). It (i) is computationally efficient compared to variational alternatives, (ii) highly flexible, handling arbitrary nonlinear forward models with only black-box access, and (iii) enjoys theoretical guarantees. We demonstrate superior performance on inverse problems in natural image processing and scientific reconstruction, and establish convergence guarantees of the scheme under appropriate assumptions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work addresses the problem of restoring clean samples from their corrupted measurements under two key constraints: 1) access is limited to the corrupted data, without corresponding clean samples, and 2) the forward map from the clean signal to the measurements is treated as avilable in a black-box manner. The authors approach this problem using Self-consistent Stochastic Interpolants, an extension of standard Stochastic Interpolants (SI) framework. This method adds a self-consistency mechanism, ensuring that the trained model's outputs map back to the true measurements. The method is introduced with rigorous mathematical formalism, solid theoretical results, and a broad analysis of convergence guarantees. The experimental section covers both standard benchmarks and a sophisticated application to quasar spectra recovery.

### Strengths
S1. The considered problem of obtaining clean samples when only the measurements and the black-box forward model are available seems important and challenging, yet the authors' approach based on self-consistency is both novel and effective.

S2. The proposed approach has strong theoretical foundations and is accompanied by a detailed convergence analysis.

S3. While the paper often utilizes more sophisticated mathematical tools, the authors took great care in writing the paper in a clear way with good flow.

### Weaknesses
W1. My primary concern relates to the lack of evaluation of the SDE version of the algorithm. The theoretical analysis focuses mainly on the SDE case, so it is natural to ask whether these results translate to practice. Moreover, as the authors also mention, the toy example (section 5.1) indicates the superiority of the SDE-based approach, and it seems counterintuitive that no other non-toy evaluations appeared in the paper. While I understand that it might be more computationally demanding and slower in terms of convergence, some kind of comparison in a standard experiment (section 5.2) should be provided.

W2. Equally concerning is the lack of standard evaluation metrics used in typical inverse-problems-related papers.

1. For the standard benchmarks (Table 1), only LPIPS is used. Methods for inverse problems are typically evaluated with the perception-distortion tradeoff in mind [1], so I would recommend adding PSNR, SSIM (distortion) and FID (perception).

2. For the comparison with other inverse generative models, only FID is provided. This is not crucial, but some metric like the Inception Score [2] should also be provided for a more balanced comparison.

3. For the quasar spectra recovery, no quantitative results are given. I'm not an expert in this subfield, but I expect the authors to provide proper metrics used in the evaluation scheme in such scenarios.

W3. The scope of the chosen baselines is very small. DPS is a good starting point, but as the authors correctly point out, its assumptions largely differ from their work. It would be beneficial for the paper to also include other methods that better cover the spectrum. Examples include: unsupervised methods that make different assumptions about the forward model [3,4], supervised bridge methods that ignore the forward model [5,6,7,8] or assume access to it [9], unpaired-data-based bridges [10,11] and the approaches explicitly cited by the authors as the closest ones in terms of assumptions and recency (lines 044-053). I don't expect the authors to include all of the above, but the comparison should be more representative.

W4. (Minor) I believe that the black-box assumption is lost when the method is additionally using either the mask in the random masking experiment or the compression magnitude in the JPEG one. The text should be slightly rephrased to emphasize that.

[1] Blau and Michaeli, The Perception-Distortion Tradeoff, CVPR, 2018

[2] Salimans et al., Improved Techniques for Training GANs, NeurIPS, 2016

[3] Wang et al., Zero-shot image restoration using denoising diffusion null-space model, ICLR, 2023

[4] Song et al., Pseudoinverse-guided diffusion models for inverse problems, ICLR, 2023

[5] Liu et al., I2SB: Image-to-image Schrodinger Bridge, ICML, 2023

[6] Zhou et al., Denoising Diffusion Bridge Models, ICLR, 2024

[7] Luo et al., Image restoration with mean-reverting stochastic differential equations, ICML, 2023

[8] Yue et al., Image restoration through generalized ornstein-uhlenbeck bridge, ICML, 2024

[9] Sobieski et al., System-embedded Diffusion Bridge Models, arXiv, 2025

[10] De Bortoli et al., Schrodinger bridge flow for unpaired data translation, NeurIPS 2024

[11] Kim et al., Unpaired image-to-image translation via neural schrodinger bridge, ICLR, 2024

### Questions
I think that the method is very elegant and I am generally sympathetic to the paper. However, I think some important points should be addressed, which I mentioned in the Weaknesses above. I have one additional question related to the theoretical results.

Q1. How do the considerations from lines 308-319 apply when the considered SDE has matrix-valued coefficients? Moreover, how do these results connect with the matrix-valued SDE proposed in [2], which embeds the Gaussian linear forward model into the coefficients? 

[1] Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, ICLR, 2021

[2] Sobieski et al., System-embedded Diffusion Bridge Models, arXiv, 2025

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
4

### Summary
The authors consider a problem of inverse generative modelling in which they have access to samples of the distribution of corrupted samples and access to a black-box corruption map. The authors propose and theoretically justify a novel iterative algorithm based on iterative learning and inference of a stochastic interpolant model until it generates data such that the corruption map produces exactly the distribution of corrupted samples. The authors further evaluate their approach on image setups with different corruption maps and one scientific setup.

### Strengths
- The proposed approach is novel and theoretically justified.
- The proposed approach does not require knowing or using the specific functional form of the corruption map used.
- The authors evaluate their method on a wide range of image setups and one scientific setup, showing that the proposed method solve the problem.

### Weaknesses
- All the setups are more like synthetic, rather than real setups mentioned in the introduction: “... medical imaging, where we observe tomographic projections of internal structures, astronomical observations affected by atmospheric distortion, and other measurement processes that introduce noise and information loss”. The quasar spectra setup also assumes the usage of the synthetically provided process F. While the considered setups clearly show that the method works, the addition of one in the mentioned real-life examples would further strengthen the method. Specifically, they would show that the requirements of the method (like injectivity of F) could be satisfied in real-life problems and that there is a direct practical result.

### Questions
Do you have any understanding of it it possible to extend this framework to discrete domains by considering, e.g., masking or uniform diffusions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper addresses the problem of inferring a probability density function from data that is obtained by a forward mapping F.
The topic is interesting and important. But the paper contains serious mistakes and therefore cannot be published.

### Strengths
The main strength of the paper is in problem definition and casting. The rest has some serious errors

### Weaknesses
The main problem with this paper is that it has a significant error that propagates through the paper.
They define the path 
I = \alpha x_t + \beta F(x) + \gamma z

But this path is wrong. In particular, F(x) is not even in the same space (functional space) of x and in the finite dim case often has a very different dimensions. Take the example you propose, tomography. F(x) is projection data and x is the image. The size of F(x) depends on the number of projection I take. This path does not make any sense!

Hint for next time - try the path I = \alpha x_t + \beta (\grad F)^T F(x) + \gamma z
This additional operation is crucial both in the finite and infinite setting

### Questions
see above

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a method to solve the inverse problems, formulated as a generative modeling problem, but in an unusual setup. While usually, when solving the inverse problem, one knows the *clean* data $x$ and corrupt operator $\mathcal{K}$, the problem setup authors solve when one knows the *corrupted* data $y$ and corrupt operator $\mathcal{K}$. Since this setup doesn't have clean data samples, it doesn't allow for the straight application of already developed generative modeling methodologies and requires the development of new ones. 

The authors propose to utilize the Stochastic Interpolants (SI) framework and an iterative  (Expectation Maximization-like) scheme for learning it in a no-clean data setup. The key idea of the method is to build such an SI that the corrupted data distribution $Y$ mapped by SI into clean generated data would be mapped back by the corruption operator $\mathcal{K}$ to the initial corrupted data distribution $Y$, i.e., self-consistency property, see lines 177-182. The practical algorithm consists of the iterative repetition of: 1) sampling data from the current SI ($\Theta^{(k)}$), i.e., Expectation step 2) learning the next SI ($\Theta^{(k+1)}$) on the data generated in the previous step, i.e., Minimization step.

In addition, the authors provide a theoretical analysis of their iterative procedure under different conditions and provide convergence guarantees of their procedure.

In practice, the authors evaluate their model on a range of tasks, including toy 2D problems, image inverse problems, and quasar spectrum restoration, and provide comparisons with competing approaches within the image inverse problem setting.

### Strengths
- The SI framework allows for a corruption operator $\mathcal{K}$ with black box access, which is less restrictive than previous works  and allows for wider range of corruption operators $\mathcal{K}$.
- The SI is, in general, known to be a good generative modeling framework for the inverse problem solving [1, 2], so the application of SI to such an inverse problem is very well motivated even in $\mathcal{K}$  cases, e.g. linear inverse problems, where one can utilize other diffusion methodologies [3, 4]
- The theoretical analysis is rather comprehensive and convincing. Section 4 results deliver valuable theoretical analysis on the convergence of the method and the possible room for error, which can be helpful in practice and is good as the theoretical result by itself.
- In the image inverse problems experiments method slightly outperforms DPS method that learns *using the clean data* and outperforms other methods that learn only from corrupted data.

### Weaknesses
- Some results in the paper are made under the assumption that the corruption operator is *injective*. That seems as not a mild assumption. Can authors comment on the restrictions of such an assumption on practice and show some examples of injective/non-injective mappings?
- The learning procedure is non-simulation free, since one has to sample from learned SI during the "Expectation"-step. That leads to a computationally heavy algorithm. Can authors provide an ablation study on the parameters that control the computational demand of the procedure, i.e., the number of procedure iterations $K$ and the number of gradient updates  $T_{tr}$ inside of one iteration.
- The authors explain the *trade-off* between the condition number $\mathcal{X}$ and restriction of the SI approximation class $\mathcal{S}_\lambda$. This trade-off is shown in Theorem 1, but as far as I understand, authors do not test this trade-off in practice. So, can authors implement the restrictions on SI approximation class in practice and indeed show 1) faster convergence, 2) bigger error. That trade-off and the way to enforce it could be very useful in practice and seems like a useful hyperparameter that hasn't been explored.
- The method's performance gap with EM posterior in Table 2 is marginal and the comparison of the proposed method with other methods that learn only on corrupted data has been carried out on the generation task. Which is strange, as far as I understand, at least some of the reference methods (Ambient Diffusion, EM Posterior or others) can solve inverse problems by starting from corrupted observations. It would be nice to compare the proposed method to competitors in particularly in the image restoration setup and not the generation setup, i.e., add methods that learn only on corrupted data in Table 1.
- The applications of the proposed method are not described properly. Can authors describe the possible real-world scenarios where one has only corrupted observations and a black-box corrupt operator?


[1] Liu, G. H., Vahdat, A., Huang, D. A., Theodorou, E., Nie, W., & Anandkumar, A. (2023, July). I $^ 2$ SB: Image-to-Image Schrödinger Bridge. In _International Conference on Machine Learning_ (pp. 22042-22062). PMLR.

[2] Albergo, M. S., Goldstein, M., Boffi, N. M., Ranganath, R., & Vanden-Eijnden, E. (2024, July). Stochastic Interpolants with Data-Dependent Couplings. In _International Conference on Machine Learning_ (pp. 921-937). PMLR

[3] Francois Rozet, Gerome Andry, Franc¸ois Lanusse, and Gilles Louppe. Learning diffusion priors from
observations by expectation maximization. Advances in Neural Information Processing Systems,
37:87647–87682, 2024.

[4] Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alex Dimakis, and Adam Klivans. Ambient diffusion: Learning clean distributions from corrupted data. Advances in Neural Information
Processing Systems, 36:288–313, 2023.

### Questions
See Weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3
