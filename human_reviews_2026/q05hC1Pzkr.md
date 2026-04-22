# Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Multi-Resolution Hash Encoding (MHE), the foundational technique behind Instant Neural Graphics Primitives, provides a powerful parameterization for neural fields. However, its spatial behavior lacks rigorous understanding from a physical systems perspective, leading to reliance on heuristics for hyperparameter selection. This work introduces a novel analytical approach that characterizes MHE by examining its Point Spread Function (PSF), which is analogous to the Green's function of the system. This methodology enables a quantification of the encoding's spatial resolution and fidelity. We derive a closed-form approximation for the collision-free PSF, uncovering inherent grid-induced anisotropy and a logarithmic spatial profile. We establish that the idealized spatial bandwidth, specifically the Full Width at Half Maximum (FWHM), is determined by the average resolution, $N_{\text{avg}}$. This leads to a counterintuitive finding: the effective resolution of the model is governed by the broadened empirical FWHM (and therefore $N_{\text{avg}}$), rather than the finest resolution $N_{\max}$, a broadening effect we demonstrate arises from optimization dynamics. Furthermore, we analyze the impact of finite hash capacity, demonstrating how collisions introduce speckle noise and degrade the Signal-to-Noise Ratio (SNR). Leveraging these theoretical insights, we propose Rotated MHE (R-MHE), an architecture that applies distinct rotations to the input coordinates at each resolution level. R-MHE mitigates anisotropy while maintaining the efficiency and parameter count of the original MHE. This study establishes a methodology based on physical principles that moves beyond heuristics to characterize and optimize MHE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a method to analyze multi-resolution hash encoding used in implicit neural representations. The authors observe that while MHE is widely adopted, its hyperparameters have been tuned empirically with little mathematical analysis of its behavior. Thus, this paper introduce a point spread function based theoreritcal framework to analyze how MHE operates. Under multi-resolution, they prove that the influence region of each point feature scales with 𝑁min; they also study differences that arise during optimization and propose a model to explain them. They further propose a speckle/noise approximation to model hash collisions, and introduce R-MHE to mitigate these issues. On high-resolution image regression and 3D NeRF training, they show that the analysis is consistent with practice and yields useful hyperparameter guidance.

### Strengths
Motivation: While practitioners recognize certain behaviors of hash encodings, providing a mathematical model and using it to select hyperparameters is a meaningful contribution.

Clarity of modeling: The paper steps through the math cleanly, organizing the required formulas at each stage.

### Weaknesses
**3D experiment observations**
- Even when values are similar, the baseline consistently outperforming the theory in 3D suggests some assumptions may not hold in 3D. Could this stem from the alignment assumption in the theory? A toy misalignment study (small perturbations) could demonstrate how trends evolve and help validate the proposed modeling more precisely.

**Theoretical growth factor experiments.**
- Results are shown by varying b with ±0.1 steps, but: NeRF training is stochastic; single runs are not sufficient—use N repeated trials to report robustness. Use wider resolution than 0.1 steps to confirm the monotonic/expected trend across denser b values.

**Practical calibration of the broadening factor.**
- Does the broadening factor depend on training iterations or optimizer choice? If training dynamics materially affect it—even if grids don’t—then practical calibration may be bottleneck for real applying this analysis on real hyperparameter tuning.

### Questions
- The anisotropic behavior seems applicable to other encodings (e.g., multi-plane or multi-resolution plane encodings). Do the authors have a unified formulation or insights that transfer to those settings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
## Summary
* This paper study the Multi Resolution Hash Encodings (MHE) proposed by instant-ngp, which is used to accelerate representation of 3D scene.
* This paper characterize the PFS of MHE in collision free assumption, concluding both theoretically and empirically that the effective resolution of MHE is not controled by max resolution $N_{MAX}$ as described by instant-ngp, but by average resolution $N_{AVG}$.
* Besides, this paper makes two practical contribution to HME when considering collision: a scheme for parameter selection and an improvement named R-MHE.

### Strengths
## Strength
* The characterization of MHE using PFS is novel and interesting. The conclusion that FWHM is direction dependent and its width is determined by $N_{max}$ is a very useful insight that might inspire later works.
* The empirical on 2D Image Regression is limited but convincing.

### Weaknesses
## Weakness
* The proposed approach is said to have two advantages, avoiding anisotropy and avoiding hashing collision. Both of the two advantages improve the empirical result. However, either because it is not possible to study the two factors independently, or the authors have not done it, I find it hard to evaluate the contribution of avoiding anisotropy to practical results. In fact, the majority of this paper is about how MHE introduces anisotropy. And I am definitely more interested in how the anisotropy property harms the performance of MHE beyond toy examples. While the current empirical result is not strong enough to show it.
* The MHE in its original paper (instant-ngp) is well evaluated with different tasks and contains a lot more formal comparsion with other methods. I do agree that this paper has its unique theoretical insights. However, implemented their proposed approach for other tasks and include more comparsion strengthen this paper a lot.

### Questions
* Is it possible to study the effect of anisotropy reduction and collision avoidance? Which one contributes mainly to the gain in PSNR?

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
3

### Summary
This paper presents to provide understandings on multi-resolution hash encoding (MHE) originally presented from Instant-NGP.

In summary, the paper:
- Frames MHE as a physical system via its point spread function (PSF); derives a closed-form showing logarithmic decay and grid-induced anisotropy. 
- Shows effective resolution follows the broadened empirical Full-width half-maximum (FWHM) = 1/N_avg (instead of N_max) due to optimization-induced broadening (beta=3). 
- Analyzes finite-capacity collisions causing speckle noise and SNR loss; proposes Rotated MHE (R-MHE) with independently hashed, rotated grids to improve isotropy and SNR.
The paper backs its claims with experimental results, through 2D and 3D evaluations.

### Strengths
The paper provides interesting extension of multi-resolution hash encoding. It provides:
- Clear theory–practice link: closed-form PSF, anisotropy factor, and FWHM = 1/N_avg give actionable design rules.
- Comprehensive empirical validation: broadened PSF (beta=3) consistently matches measurements; two-point resolution aligns with FWHM, not N_max. 
- Practical payoff with fixed memory: R-MHE markedly improves isotropy and PSNR in 2D (+8.06 dB to 31.30 dB at M=8) and yields steady 3D gains up to +0.65 dB.

### Weaknesses
Despite its strong theoretical foundations, the paper contains following potential shortcomings.
- Broadening factor of beta = 3 is largely empirical; sensitivity to optimizer/architecture/data is not exhaustively characterized in additional experiments.
- PSF-guided hyperparameter (b) underperforms a tuned baseline in 3D, and R-MHE’s 3D gains are modest (+0.65 dB).
- The paper fundamentally relies on MHE, of which multiple extensions such as dictionary fields has been proposed already. However, the paper does not compare with these extensions which limits its interpretability.

### Questions
1. How are the hyperparameters calculated? Where does beta value come from?
2. Why does performance gap between 2D and 3D MHE exist?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes rotated multi-resolution hash encodings, widely used for implicit neural representations in image regression and neural radiance fields. Previous hash encodings are dependent on axes, leading to anisotropy and hindering empirical performance. They fail to capture well frequency patterns like sinusoidal functions because hash encodings are implicitly aligned with the axes of Cartesian coordinates. In contrast, this paper proposes rotated multi-resolution hash encodings that allow us to be independent of Cartesian coordinates. It achieves this by using multiple rotation functions for input position $x$, resulting in an average of randomly sampled rotation functions. This approach mitigates the reliance on representation aligned with Cartesian coordinates and improves prediction performance without degradation in frequency-based patterns. One of the randomly sampled rotation functions may potentially match the directions of the frequency patterns. This paper demonstrates its effectiveness in theoretical and practical cases.

### Strengths
- This paper introduces a novel approach that explicitly uses multiple random rotation transforms. To maintain the total number of parameters, the paper splits the original feature dimensions into the number of rotation and new feature dimensions. Although it employs a smaller number of feature dimensions, the random rotation function for input $x$ enables it to effectively capture signals that do not align with Cartesian coordinates.

- The paper demonstrates its effectiveness through both mathematical and practical cases. Specifically, it shows that the proposed approach consistently improves prediction performance in image regression and neural radiance fields, while maintaining the same total number of parameters across each case.

### Weaknesses
- I believe this paper falls short of meeting the standards of top-tier AI conferences like ICLR. The presentation lacks an effective and efficient way for readers to grasp the main contribution of the paper. While the mathematical contributions are included in the Appendix, and very abstracted equations are elaborated in the manuscript, I disagree that all the content in the manuscript must be abstracted. The Appendix is necessary to fully digest the mathematical details, but it doesn’t justify the need for the entire manuscript to be abstracted. I believe important equations (Eq 2 and 3) and their impact (e.g., FWMH) should be provided in the manuscript, but they aren’t. 

- Empirically, maintaining the total number of parameters is crucial for improving prediction performance. However, this paper artificially designs baselines and their parameterization. For instance, its baselines started with a feature dimension of 8, which is not the official parameterization of feature dimensions in NeRFs, which sets it to 2 (To my best knowledge). Therefore, it lacks the space to adjust and accommodate random rotation functions. In this case, the proposed approach naturally faces challenges in its practical implementation. 

- Following the second concern in Weakness, they should identify new cases where high-dimensional encodings are required to achieve accurate prediction performance. In that sense, they must demonstrate that rotated MHE not only maintains performance but also reduces the total number of parameters.

### Questions
- I’m curious about Figures 1, 2, and 3. The colors in these figures indicate different parameterizations and the value seems to be continuous in all of them. However, the color information is actually discrete because of the parameterization used in this experiment. Adopting a legend for continuous values makes it challenging to understand graphical explanations easily. 

- Figure 1 and 2 are particularly difficult to grasp the main messages from. They require more detailed explanations and concepts to understand.

### Soundness
3

### Presentation
1

### Contribution
2
