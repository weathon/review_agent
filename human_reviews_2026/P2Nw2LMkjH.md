# ETGS: Explicit Thermodynamics Gaussian Splatting for Dynamic Thermal Reconstruction

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
We propose ETGS, a method for reconstructing dynamic thermal scenes by embedding explicit thermodynamic modeling into 3D Gaussian Splatting. Each Gaussian is equipped with physically interpretable thermal parameters, and its thermodynamics evolution is described by a first-order heat-transfer ODE with an analytical closed-form solution. This formulation avoids numerical integration, enables efficient rendering at arbitrary timestamps, and naturally handles irregular sampling and out-of-order observations. We also introduce the Rapid Heat Dynamics (RHD) dataset, which provides millisecond-aligned RGB–IR image pairs covering typical thermal processes such as cooling, warming, heating, and heat transfer. Experiments on RHD show that ETGS captures rapid thermal dynamics more accurately than existing static and dynamic baselines, while maintaining training and rendering efficiency close to that of static 3DGS. Code and dataset are available at https://github.com/jankin-wang/ETGS.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes ETGS, a novel extension of 3D Gaussian Splatting (3DGS) that enables the reconstruction of dynamic 3D temperature fields from RGB–IR image pairs at arbitrary viewpoints and time instants.
Unlike conventional 3DGS methods, where each Gaussian stores color information, ETGS instead assigns to each Gaussian a set of parameters that model temporal variations in temperature.
Using these parameters together with a Fourier basis representation, the authors derive an ordinary differential equation (ODE) describing the temporal evolution of temperature and show that its solution can be expressed in a closed form (Eq. (8)). Consequently, ETGS can directly estimate the temperature at any given time using Eq. (8).
The authors also introduce a new dataset, Rapid Heat Dynamics (RHD), which captures thermal phenomena such as warming, cooling, heating, and heat transfer at millisecond temporal resolution using synchronized RGB and IR cameras.
Experimental results on the RHD dataset demonstrate that ETGS outperforms existing 3DGS-based temperature reconstruction methods in terms of accuracy, training efficiency, and rendering speed.

### Strengths
* The paper proposes a natural approach to incorporating physically governed variables into the 3DGS framework.

* The experiments demonstrate that the proposed method can effectively model temperature dynamics without compromising the computational efficiency of 3DGS.

### Weaknesses
* Although the authors provide an aligned RGB–IR dataset, the proposed model appears to rely solely on IR data and does not utilize RGB information. If RGB input is indeed unnecessary for the model, then constructing an aligned RGB–IR dataset may not be essential. In that case, evaluations such as those shown in Figure 4 could potentially be misleading.

* The only dynamic variable in the current formulation is temperature, while the spatial configuration of objects remains fixed.

### Questions
* In Eq. (2), the term $f_i$ from Eq. (1) is omitted. Would it be possible to extend the model to simultaneously render RGB images by retaining $f_i$? It would be sufficient to mention this as a potential direction for future work.

* Regarding Eq. (11), the definitions of $\mathcal{L}\_1$ and $\mathcal{L}_{D-SSIM}$ should be explicitly stated (for example, in the appendix). At the very least, the references from which these loss terms are derived should be clearly cited in the main text.

### Soundness
3

### Presentation
4

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
This paper proposes ETGS (Explicit Thermodynamics Gaussian Splatting), a method for reconstructing dynamic thermal scenes by integrating physics-based thermal modeling into 3D Gaussian Splatting. The approach extends standard 3D Gaussians with thermal properties: equivalent heat capacity, heat transfer coefficient, and time-evolving temperature. Temperature evolution is modeled via a first-order ODE derived from Newton's cooling law plus a heat source term expanded on a harmonic basis using Fourier features. The authors derive a closed-form analytical solution that enables efficient rendering at arbitrary timestamps without numerical integration. The paper also introduces the Rapid Heat Dynamics (RHD) dataset: 10 scenes with 2,410 pixel-aligned RGB-IR views, millisecond-accurate timestamps, and coverage of canonical thermal processes (cooling, warming, heating, heat transfer). Experiments show substantial improvements over baselines while maintaining training efficiency comparable to static 3DGS.

### Strengths
- The paper tackles an important and difficult problem at the intersection of computer vision and graphics, with several potential downstream applications.
- The idea of integrating explicit (albeit simplified) thermodynamical modeling into 3DGS is creative, and addresses a real gap in the literature.
- The ODE closed-form solution which avoid numerical integration is elegant and principle
- The model is strongly grounded on physics, enhancing its interpretability
- The dataset introduced in this paper can be of a significant value to the literature. Its temporal resolution, variety of thermodynamic process and adequate hardware design make it a solid dataset with potential reuse in the field and beyond. 
- The results are strong, with significant improvements compared to baselines, and with a training cost comparable to static 3DGS, which is much faster than dynamical baselines.
- Evaluation is comprehensive and uses appropriate metrics
- The paper is generally well written and is easy to follow,

### Weaknesses
- The thermodynamic model presented in the paper is very simplified, missing or oversimplifying important factors like conduction or radiation, non-linear dynamics, and no phase changes. This undermines the "faithful thermodynamic model" claim. At best, the model is physically plausible for simplified scenarios. 
- The usage of Fourier Basis for the harmonic expansion requires further justification. While Fourier approximations make sense, it would be benefitial to understand how different alternatives (eg polynomials, splines, or learned basis) would impact the model. The use of K=24 should be ablated.
- Computational cost should be studied further.
- The paper misses important technical details, including how and why the hyperparameters were chosen.
- There is little validation in term of physics plausibility: For example, the learned h for each material (eg fabric or metal), do they match with known material properties? Is energy conservation preserved? Can it extrapolate beyond training time ranges?
- The dataset has limitations in terms of spatial resolution, and only contains 10 scenes, which limits impact. 
- Evaluation is done exclusively on perceptual metrics, missing the temperature accuracy which I argue is as important for this problem. 
- The paper is not reproducible in its current state (although code was promised). Many hyperparameter values were not reported. 
- Failure cases could be analysed more thoroughly.

### Questions
Questions:
- Why is the PNSR improvement across scenes show such a high variance?
- What do the learned A_i,k, B_i,k represent physically?

Other comments:
- Abstract could be written in a less dense way, it lacks conciseness
- Please fix inconsistent notation (eg G_i)

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
The paper introduces ETGS (Explicit Thermodynamics Gaussian Splatting), a method for dynamic thermal scene reconstruction that embeds explicit thermodynamic equations into Gaussian Splatting. Each Gaussian incorporates physically interpretable parameters—heat capacity, heat-transfer coefficient, heat source, and temperature—with a closed-form solution to the heat ODE that eliminates numerical integration and enables efficient, stable rendering under irregular sampling. The authors also propose the RHD dataset, featuring pixel-aligned RGB–IR pairs with millisecond timestamps. Experimental results demonstrate strong performance and clear advantages over prior methods.

### Strengths
* First explicit embedding of thermodynamic laws into Gaussian Splatting.
* Derivation from heat ODE to closed-form analytical solution with physical interpretation.
* State-of-the-art metrics.
* First ms-timestamped RGB-IR benchmark for thermal dynamics.

### Weaknesses
* The framework largely relies on established heat-transfer equations and standard ODE solutions, with novelty mainly in integrating these principles into the Gaussian representation rather than introducing new thermodynamic insights.
* The framework is demonstrated on controlled thermal scenes; performance in outdoor or multi-object settings remains unclear.
* Only first-order linear heat transfer is considered; nonlinear effects (e.g., phase change or radiation coupling) are ignored.
* More analysis on frequency-grid resolution ($K$, $\omega_{min}$, $\omega_{max}$) would strengthen robustness claims.

### Questions
* Could ETGS extend to coupled RGB + thermal rendering where radiance and temperature interact? This coupling seems crucial for realistic multimodal reconstruction and may reveal new advantages of the proposed explicit thermodynamic formulation.
* How sensitive is the closed-form temperature solution to errors in $\tau_i$ and $h_i$? Would it remain stable for heterogeneous materials?

### Soundness
3

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
3

### Summary
The paper proposes ETGS, a method for **dynamic thermal scene reconstruction** that embeds explicit thermodynamic modeling into the Gaussian Splatting (3DGS) framework. Instead of relying on numerically integrated ODEs, ETGS introduces closed-form temperature evolution equations derived from thermodynamic principles (Newton’s cooling and harmonic heat source excitation).
It also presents a new dataset, RHD (Rapid Heat Dynamics), which includes pixel-aligned RGB–IR image pairs with millisecond timestamps covering heating, cooling, and heat-transfer scenarios. ETGS achieves state-of-the-art reconstruction performance while maintaining efficiency comparable to static 3DGS.

### Strengths
1. The explicit thermodynamic modeling is well-motivated and technically sound, enabling elegant combination with 3DGS.
2. The proposed method maintains the merits of 3DGS on training and inference speeds by avoiding costly neural integral inference in prior works.
3. The established RHD dataset is well-designed with pixel-aligned RGB-IR pairs, meaningful for future research.
4. The comparison experiments are comprehensive, including both static and dynamic baselines.

### Weaknesses
1. The Gaussians are assumed to be independent of each other in the proposed method, which ignores the real-world heat transfer across nearest Gaussians. Therefore, a video or image demonstration of continuity from Gaussian to Gaussian (or part to part) is helpful.
2. The established dataset (RHD) is relatively small and lacks diversity. Moving heat sources or other factors that influence thermodynamics can be involved to improve the dataset complexity.
3. What is the performance when combining RGB and thermal supervisions?
4. The writing of this paper should be improved before acceptance.

### Questions
Please address my concerns in the weakness part.

### Soundness
3

### Presentation
2

### Contribution
2
