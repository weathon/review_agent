# Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 2

## Abstract
Recovering true signals from noisy measurements is a central challenge in inverse problems spanning medical imaging, geophysics, and signal processing. Current solutions nearly always balance prior assumptions regarding the true signal (regularization) with agreement to noisy measured data (data-fidelity). Conventional data-fidelity loss functions, such as mean-squared error (MSE) or negative log-likelihood, seek pointwise agreement with noisy measurements, often leading to overfitting to noise. In this work, we instead evaluate data-fidelity collectively by testing whether the observed measurements are statistically consistent with the noise distributions implied by the current estimate. We adopt this aggregated perspective and introduce $\textit{distributional consistency (DC) loss}$, a data-fidelity objective that replaces pointwise matching with distribution-level calibration. DC loss acts as a direct and practical plug-in replacement for standard data consistency terms: i) it is compatible with modern unsupervised regularizers that operate without paired measurement–ground-truth data, ii) it is optimized in the same way as traditional losses, and iii) it avoids overfitting to measurement noise without early stopping or the use of priors. Its scope naturally fits many practical inverse problems where the measurement-noise distribution is known and where the measured dataset consists of many independent noisy values. We demonstrate efficacy in two key example application areas: i) in image denoising with deep image prior, using DC instead of MSE loss removes the need for early stopping and achieves higher PSNR; ii) in medical image reconstruction from Poisson-noisy data, DC loss reduces artifacts in highly-iterated reconstructions and enhances the efficacy of hand-crafted regularization. These results position DC loss as a statistically grounded, performance-enhancing alternative to conventional fidelity losses for an important class of unsupervised noise-dominated inverse problems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new 'distributional consistency' data fidelity term that can be used for solving inverse problems. The loss cheks whether the prediction residuals follow the noise distribution by applying a variant of the probability integral transform (similar to the Von Mises statistic). Experiments on single image denoising using the deep image prior and on PET reconstruction with Poisson noise show that the proposed loss performs better than the standard negative log-likelihood setting.

### Strengths
- presents an interesting data fidelity term for inverse problems that exploits full knowledge of the noise distribution by checking that residuals are consistent at a distributional level.
- the proposed loss avoids over-fitting on the deep image prior and improves performance on PET (regularized) reconstruction.
- presents an analysis that motivates links with the standard negative log-likelihood loss

### Weaknesses
- the paper lacks a discussion of existing literature on the distributional loss, and other existing alternatives to negative log-likelihood. For example, the loss seems quite similar to the Von Mises statistic, which is well-known in statistics. It would be also good to include discussions on other related work, like robust estimators (eg Huber losses, etc).

- There is no guarantee that the proposed loss should avoid overfitting in general problems, except for the empirical evidence on the two presented 'toy' examples, which involde a single image at a time. 
   - I would expect at least one experiment on learning on a dataset or its usage with modern solvers such as plug-and-play, diffusion methods, etc.
   - I find sentences like "it avoids overfitting to measurement noise even without the use of priors" problematic, since there is no theoretical guarantee that this is the case - it is easy to construct counter-examples with low distributional loss and high error (as already done in the paper in 3.2) - and the results rely on 'implicit regularizations' such as the deep image prior autoencoder architecture or the reduced number of voxels in PET.

### Questions
I would expect the loss to be highly sensitive to mispecified or approximate noise models - did you check the proposed method on real measurement data (not synthetically generated measurements)?

### Soundness
3

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
4

### Summary
This paper proposes the Distributional Consistency (DC) Loss, a new data-fidelity objective for imaging inverse problems that goes beyond traditional pointwise losses such as MSE or negative log-likelihood. Instead of enforcing pixelwise accuracy, DC loss measures whether the residuals are statistically consistent with the assumed noise distribution. The method applies the Probability Integral Transform (PIT) to map residuals into a uniform space and computes the Wasserstein distance (after a logit transform) between the resulting distribution and a standard logistic reference. Experiments on Deep Image Prior (DIP) denoising and Poisson PET reconstruction show that DC loss effectively mitigates overfitting and improves robustness without relying on early stopping or manual regularization tuning.

### Strengths
- The DC loss provides a principled alternative to traditional pixelwise data terms by enforcing consistency at the distributional level. The use of PIT and Wasserstein distance forms a coherent, probabilistically interpretable framework.
- Experiments on DIP and PET reconstruction show that DC loss mitigates overfitting and achieves a better balance between noise suppression and structural preservation, illustrating strong potential for broader applications.

### Weaknesses
- The scope of evaluation is limited. The paper validates the proposed DC loss only on two tasks—DIP-based image denoising and Poisson PET reconstruction. While these are representative, it remains unclear whether the method generalizes to other inverse problems (e.g., deblurring, inpainting, or diffusion-based approaches) or to more complex, learned priors.

- The core concept is not immediately intuitive. The key theoretical basis—the Probability Integral Transform (PIT)—is crucial for understanding why the CDF-based approach produces uniform residuals under a correct model. However, this important explanation is only presented in the supplementary material, leaving readers unfamiliar with the concept confused at first glance. As written, the method appears pointwise rather than distributional, and the main text would benefit from a concise, self-contained explanation or visual illustration of how PIT ensures distributional consistency.

- The discussion of related work is insufficient. Those who are experts in imaging inverse problems might not necessarily experts in the design of data-fidelity terms. Thus, I find it surprising that the paper does not discuss any closely related works or precedents. This lack of context makes it difficult to judge whether the proposed formulation is genuinely new or an adaptation of existing statistical consistency ideas.

- The approach has practical limitations:
  - Computational cost is higher than standard MSE or NLL losses due to the sorting and Wasserstein distance computation required per iteration.
  - The method assumes known and independent noise models for each measurement, which may not hold in real scenarios involving correlated, unknown, or spatially varying noise.
  - The sensitivity to noise model mismatch is not studied, leaving uncertainty about the robustness of the approach in practical imaging systems.

### Questions
Please see the weakness section.

### Soundness
4

### Presentation
3

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
The paper introduces a loss function for inverse problems, based on matching the distribution of residual errors with a known noise distribution.

### Strengths
- Well written paper.

- The overall idea of matching the noise distribution to derive a loss function is very interesting.

- Results are given for several settings.

### Weaknesses
- The claim that the noise model is known or can be estimated in many practical settings seems overly optimistic. In practice, there can be many additional artifacts that might not be well captured by noise terms and/or are difficult to estimate/model. In the case of tomography for example, real-world data might include artifacts caused by not well calibrated or defective detector pixels (ring artifacts), isolated high-intensity values caused by stray X-rays, scatter components that are caused by non-linear interactions with the sample, parts of the data replaced by nonsensical values (caused for example by the X-ray beam going down during an experiment). It is unclear, but important, to assess how robust the proposed method is to such artifacts. In other words, it would be useful to know how well the method works when the noise model used does not match the data (in various degrees of severity).

- The authors now use the proposed loss in combination with Deep Learning based methods. It would be interesting to see its performance for traditional image reconstruction (e.g. reconstructing the pixel values of an image directly), comparing with traditional reconstruction methods.

- There is quite a bit of work in the inverse problem community about appropriate loss functions. Examples include the student's t loss [1] and the Huber loss [2]. Such works should be discussed in the introduction, and ideally compared with.

- Similarly, there is quite a bit of knowledge about the effect of noise on the convergence of iterative reconstruction methods in inverse problems [3]. Based on this, several stopping criteria have been developed [4]. It would be good to discuss this prior work, and to compare directly with such stopping criteria (as you assume known noise distribution).

[1] Kazantsev, D., Bleichrodt, F., van Leeuwen, T., Kaestner, A., Withers, P. J., Batenburg, K. J., & Lee, P. D. (2017). A novel tomographic reconstruction method based on the robust Student's t function for suppressing data outliers. IEEE Transactions on Computational Imaging, 3(4), 682-693.

[2] Mohan, K. A., Venkatakrishnan, S. V., Gibbs, J. W., Gulsoy, E. B., Xiao, X., De Graef, M., ... & Bouman, C. A. (2015). TIMBIR: A method for time-space reconstruction from interlaced views. IEEE Transactions on Computational Imaging, 1(2), 96-111.

[3] Elfving, T., Hansen, P. C., & Nikazad, T. (2014). Semi-convergence properties of Kaczmarz’s method. Inverse problems, 30(5), 055007.

[4] Hansen, P. C., Jørgensen, J. S., & Rasmussen, P. W. (2021, September). Stopping rules for algebraic iterative reconstruction methods in computed tomography. In 2021 21st International Conference on Computational Science and Its Applications (ICCSA) (pp. 60-70). IEEE.

### Questions
- How does the method perform when the noise model used does not match the data?

- How does the method perform in combination with classical image reconstruction?

- How does the proposed method compare with existing custom loss functions used in inverse problems?

- How does the method compare with existing work based on using known noise models, for example as a stopping criterion?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a novel data-fidelity loss function, the Distributional Consistency (DC) loss, designed for inverse problems. The DC loss evaluates whether the entire set of observed measurements is statistically consistent with the noise distributions implied by the current signal estimate. The authors demonstrate that this method avoids overfitting, eliminates the need for early stopping in methods like Deep Image Prior (DIP) , and improves the efficacy of regularization in tasks like PET image reconstruction.

### Strengths
1. The paper provides a comprehensive statistical analysis of the proposed DC loss.

2. The idea of exploring distributional-level calibration as an alternative to data-fidelity loss is meaningful.

### Weaknesses
1. The method's core premise requires that the measurement noise distribution is known or can be well-estimated. The experiments are confined to purely synthetic data using clean Gaussian or Poisson noise models. This ideal scenario is rarely met in practice, where noise is often complex, correlated, or misspecified. The paper provides no experiments on real-world data, making it impossible to assess the method's robustness or practical generalization ability.

2. The paper makes a very broad claim to be a "performance-enhancing alternative... for inverse problems", yet the experimental validation is narrow. It is limited to only two applications: image denoisin and PET reconstruction. Furthermore, the PET experiments are entirely simulated using a Brain Web phantom. This limited testing on "toy problems" and synthetic data is insufficient to support the paper's general claims.

3. Follow my previous comments, another major limitation is the study's exclusive focus on unsupervised regularization methods like Deep Image Prior (DIP), a 2017 work. It does not demonstrate its effect in any state-of-the-art, end-to-end trained frameworks. Without even a proof-of-concept on generative methods or deep unrolling networks (which is way more common to use compared with DIP), the claim that this loss is a convincing alternative for modern inverse problems is unsupported and feels overclaimed.

4. The practical usage of DC loss is questionable due to its computational overhead. The method requires computing a CDF for every measurement and, more importantly, sorting the entire set of $N$ transformed values at each iteration to compute the Wasserstein-1 distance. 

5. The paper's mathematical notation is inconsistent and confusing. For example: the unknown parameter to be estimated is introduced as $\theta$ in the general formulation (Equation 1). In the PET application, this parameter is abruptly changed to $x$ (Equation 8) without any connection. Most critically, Equation 8 introduces a term $b$ (in $[Ax+b]$) which is never defined anywhere in the paper.

### Questions
Same as the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2
