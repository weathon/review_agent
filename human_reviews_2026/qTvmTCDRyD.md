# DRIFT-Net: A Spectral-Coupled Neural Operator for PDEs Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Learning PDE dynamics with neural solvers can significantly improve wall-clock efficiency and accuracy compared with classical numerical solvers. In recent years, foundation models for PDEs have largely adopted multi-scale windowed self-attention, with the scOT backbone in Poseidon serving as a representative example.
 However, because of their locality, truly globally consistent spectral coupling can only be propagated gradually through deep stacking and window shifting. This weakens global coupling and leads to error accumulation and drift during closed-loop rollouts.
To address this, we propose DRIFT-Net. It employs a dual-branch design comprising a spectral branch and an image branch. The spectral branch is responsible for capturing global, large-scale low-frequency information, whereas the image branch focuses on local details and nonstationary structures. Specifically, we first perform controlled, lightweight mixing within the low-frequency range. Then we fuse the spectral and image paths at each layer via bandwise weighting, which avoids the width inflation and training instability caused by naive concatenation. The fused result is transformed back into the spatial domain and added to the image branch, thereby preserving both global structure and high-frequency details across scales. Compared with strong attention-based baselines, DRIFT-Net achieves lower error and higher throughput with fewer parameters under identical training settings and budget. On Navier--Stokes benchmarks, the relative $L_{1}$ error is reduced by 7%-54%, the parameter count decreases by about 15%, and the throughput remains higher than scOT. Ablation studies and theoretical analyses further demonstrate the stability and effectiveness of this design. The code is available at https://github.com/cruiseresearchgroup/DRIFT-Net.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces DRIFT-NET, a neural PDE solver aimed at solving the problem of error accumulation and "drift" observed in long-term, autoregressive predictions. The authors identify the cause of this drift in the use of local window attention in popular attention-based models, which only allow slow propagation of global information through the layers. To this end, DRIFT-NET proposes a frequency-weighted loss and a dual-branch architecture within a U-Net style backbone: an image branch using standard ConvNeXt blocks to capture local details and a spectral branch to capture global information. These branches are fused using learned low-frequency mixing and a band-wise radially-varying gate to combine low- and high-frequency components. DRIFT-NET is benchmarked against scOT and FNO on several 2D Navier-Stokes tasks, where it can improve performance with a lower parameter count and faster inference.

### Strengths
- The paper targets a well-known and critical failure mode of neural PDE solvers: long-horizon rollout drift.
- While the dual-branch approach with spectral mixing is not new, the *fusion mechanism* using band-wise radial gating seems to be an effective new component.
- The model demonstrates consistent performance improvement over the scOT and FNO baselines. In particular, it reduces accumulation errors in the rollouts with fewer parameters and a higher throughput. Moreover, an ablation study provides insights into the importance and effect of each of the three main components (LFM, RG, and FWL).

### Weaknesses
- While the paper claims to provide a modular and reusable design for PDE foundation models that “can be swapped in for windowed self-attention blocks in multi-scale operator backbones”, the effect of DRIFT-Net is only evaluated for a single backbone and only on variants of 2D Navier-Stokes equations. The spectral/image split is strongly motivated by fluid dynamics (large-scale flow vs. small-scale eddies), but may not be as optimal for other systems (e.g., hyperbolic equations, reaction-diffusion systems). Other PDE foundation models are not considered or mentioned, e.g. https://arxiv.org/pdf/2310.02994, https://arxiv.org/pdf/2403.03542, https://arxiv.org/abs/2304.07993, and https://arxiv.org/abs/2403.12553. To judge the benefit of the DRIFT-Net module, one needs to evaluate its effect on other types of PDEs, with other backbones, and also other dimensions (since, as mentioned in the limitations, the improvements in throughput might not hold anymore when using 3D FFTs). In the current form, there is also not enough evidence to claim that the drift during rollouts is *directly* caused by windowed self-attention.
- It would be important to compare the performance and throughput to adapted versions of scOT/FNO with a comparable parameter count. Moreover, it is important to note that the performance of (multi-scale) windowed self-attention heavily depends on the underlying kernel and hardware optimizations used for the attention module.
- Strictly speaking, the proposed architecture does not seem to be a neural operator (in the definition of https://arxiv.org/abs/2108.08481), since it is not discretization-invariant, i.e. it does not generalize to resolutions other than the training resolutions.
- DRIFT-Net is only compared to two baselines and no details are provided on the FNO baseline. While some variants of FNO are mentioned in the related works (e.g., U-NO and AFNO), they are not compared against and there exist further FNO versions that bear similarity with DRIFT-Net (e.g. https://arxiv.org/pdf/2109.03697 and https://arxiv.org/pdf/2402.16845).
- While the limitations mention that DRIFT-Net will be tested on irregular domains and complex boundary conditions, it is unclear how the current version can be *efficiently* extended to such (practically more relevant) geometries.

### Questions
Please see weaknesses above

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new model for operator learning for PDEs. The model is motivated by solving the issue of local coupling in existing benchmark models that use local self-attention. This leads to weak global coupling, which manifests as poor error in long term autoregressive solutions for hard, chaotic, PDEs like Navier-Stokes.  DRIFT-Net aims to mitigate this by having two paths in the model. The spectral branch operates on globally coupled low-frequency components by explicitly operating on the low frequency components of the FFT (while preserving high-frequency information), and the image branch focuses on locally coupled details. The two branches are mixed with bandwise weighting instead of concatenation for improved training stability. Experiments show that DRIFT-Net improves error over attention baselines on Navier-Stokes benchmarks, while also using slightly fewer parameters and having faster inference speed.

### Strengths
* The proposed model is well-motivated, and the solution to the global coupling problem intuitively makes sense.
* Experimental evidence clearly shows improvement over strong baselines.
* Evaluation includes some ablation studies.

### Weaknesses
* While the error improves over attention baselines, it is somewhat marginal in the hard cases like the long-horizon Kolmogorov dataset.
* There is no/little exploration of potential theoretical justification for the improvement this model brings.
* Scope of practical settings is narrow, the paper only explores 2D Navier-Stokes instances only -- it is unclear how well the improvement of DRIFT-Net would generalize to other settings.
* Some components of the model like the low-frequency cutoff requires hand tuning.
* Scaling is unclear with needing FFTs over large domains.
* It seems like there is insufficient exploration of training settings (or at least, these experimental details are not provided in enough detail).

### Questions
Based on the weaknesses,

1. Are there more mathematically grounded / theoretical justifications for using this model?
2. Have the authors considered exploring more examples of PDEs and settings to further robustify the main claim that DRIFT-Net improves error on tasks that rely on global coupling?
3. Are there potential solutions to needing hand-tuning of hyperparameters like bandwidth?
4. Could the authors elaborate on the specifics of the different learning rates, optimizers, and data amounts, and total training budget that the authors experimented with? Did the best settings for DRIFT-Net also beat the best settings for the baselines?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
DRIFT-Net aims to tackle one of the core challenges in neural PDE solvers and operator learning -- how to capture both global, long-range dynamics and fine-scale local structure efficiently and stably. It builds on the recent wave of “AI for PDEs” and foundation-model efforts (such as FNO, Fourier Neural Operators, and POSEIDON) but introduces a spectral–spatial coupling mechanism that improves physical fidelity and numerical stability.

The technical innovations in the paper are:
1. Dual-branch operator design -- introduces a spectral branch (for global, low-frequency coupling) and an image branch (for local, high-frequency detail), fused additively at each U-Net scale to preserve both global structure and fine features.
2. Controlled low-frequency mixing -- applies learnable transformations only on a subset of low-frequency Fourier modes.
3. Bandwise fusion with radial gating -- fuses the spectral and spatial signals smoothly across frequency bands using a radial weighting function α(k)∈[0,1]; this yields isotropic, energy-non-expansive blending that prevents spectral discontinuities and stabilises training.
4. Frequency-weighted loss function -- augments the base loss with a spectral weighting term that increases sensitivity to high-frequency reconstruction errors, mitigating neural networks’ spectral bias and improving multi-scale fidelity.

### Strengths
The paper is clearly written and the presentation is clear. The technical innovations seem sensible and well-motivated. Since the technical advances are quite simple and lack e.g. rigorous theory (this isn’t a problem), I think that puts the onus on a very thorough experimental evaluation.

I think the paper is quite strong in this regard. It evaluates the methods on at least three different classes of PDE (2D Navier–Stokes, shallow-water, and Kolmogorov flow), covering both laminar and turbulent regimes, and involving different grid sizes and forecast horizons. Error metrics (L1, relative L2, spectral energy error) are supported by Fourier energy spectra and visual reconstructions, showing both numerical and physical fidelity. Ablation studies examine (i) the low-frequency window size, (ii) the radial gating function, and (iii) the frequency-weighted loss — showing their independent and joint benefits. Baselines are reasonable, though not comprehensive. 

I think that is a reasonable effort, but a few additions would strengthen the evidence base for the DRIFT-Net approach.

### Weaknesses
The comparisons which are present are a good start, but I’d like to see more in a paper which focusses on an architectural improvement. The POSEISON paper, for example, has a substantially more thorough experimental comparison.

In terms of training, I’d liked to have seen a few different sizes of model trained for all methods to understand how the various approaches (these could be factors of two smaller)

I would have liked to have seen a more thorough comparison of the computational aspects in terms of memory, inference-time cost, training duration / cost and how these trade with the sizes of the models.

In terms of baselines, there was arguably not a wealth of methods compared to. POSEIDON seems like a reasonable baseline, but "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers” would be another choice which can perform strongly. 

Minor comments

line 295: Koehler et al. (2024) should be a \citep
sometimes Poseidon’s scalable Operator Transformer is referred to as “scOT” sometimes "SCOT"
line 434 "DRIFT-Net has limitation" -> "DRIFT-Net has limitations"
the pseudocode in the appendix appears just before the empty pseduocode section -- it looks odd (this is of course a very minor issue!)

### Questions
Have the authors investigated how sensitive each method is to hyper-parameters like the learning rate schedule? These settings are shared across approaches -- is this really fair? Were they optimised for DRIFT-Net?

POSEIDON uses 15 tasks, why were only a subset considered here? POSEIDON also evaluates all models in terms of scaling curves which
plot the test error for each task vs. the number of task-specific training examples. Have you considered doing this?

### Soundness
3

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
This paper propose a neural surrogate model for time-dependent PDE simulation based on scOT architecture, The proposed model use convolution and frequency processing together. Special treatments are applied to the separated low frequency and high frequency information to achieve better performance. The model is applied on several fluid dynamics datasets and is compared against FNO and scOT baseline.

### Strengths
- The presentation is clear and easy to follow.
- The model design reflects the motivation about improved frequency processing.
- The experimental evaluation shows improved better results against FNO and scOT, as well as improved efficiency against scOT.

### Weaknesses
- While some design choices are validated in ablation studies, some other main design choices may not be covered by the presented ablations. For example, the importance of having three branches (conv, spectral, linear) in the model design.

- Figure 2 and Figure 3 do not seem very meaningful. If a model has lower onestep error, it will naturally have better step-wise rollout error.

- Some other related baselines could be considered. E.g., U-Net with ConvNeXt blocks.

### Questions
- Figure 1: is the convolution applied after the rFFT2?

- How does DriftNet compare to scOT in terms of overall design choices?

### Soundness
3

### Presentation
3

### Contribution
2
