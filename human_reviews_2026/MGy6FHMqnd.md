# ClimateLLM: Efficient Weather Forecasting via Frequency-Aware Large Language Models

- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Recent progress in deep learning has advanced global weather forecasting, with larger and higher-resolution models steadily improving skill. In parallel, spectral methods provide an efficient basis for global dynamics. Yet most spectral approaches treat the complex spectrum as generic features, conflating the distinct physics encoded in amplitude (energy evolution) and phase (spatial propagation). **We propose ClimateLLM, a physics-aligned, frequency-domain forecasting framework powered by SAED-Former.** At its core, **SAED-Former** explicitly separates these two processes via a *dual-state representation*, computes interactions through a *phase-centric propagation kernel*, and injects wave-number–aware priors using *scale-conditional projection*. This physics-aligned design yields compact, robust frequency-domain representations. On standard reanalysis benchmarks, ClimateLLM matches or exceeds state-of-the-art accuracy across short- and medium-range horizons while training on a single GPU within hours. Moreover, the model supports *cross-variable transference*: networks trained on data-rich variables produce robust zero-shot forecasts for data-scarce variables. By elevating spectral structure to first-class status, ClimateLLM improves forecast quality, efficiency, and generalization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents ClimateLLM, a frequency-domain foundation model for global weather forecasting built on a new architecture called SAED-Former (Scale-Aware Entangled Dynamics Transformer). The key insight is that traditional spectral or deep learning models treat complex-valued frequency representations as flat vectors, conflating amplitude (energy evolution) and phase (spatial propagation). ClimateLLM explicitly decouples these two dynamics into dual states and models their interactions through:

A phase-centric propagation kernel that governs interactions solely based on phase (spatial propagation),

A scale-aware evolution module (SAEM) that applies band-conditioned projections to encode wave-number–dependent physics,

A dual-state representation for amplitude and phase evolution.

The model operates autoregressively in the frequency domain, predicting future weather states using FFT-based embeddings and reconstructing the spatial field via inverse FFT.

### Strengths
Novel spectral formulation: The decoupling of amplitude and phase in a dual-state architecture is a clear conceptual and mathematical innovation, providing a more physically interpretable model of atmospheric dynamics.

Physics-aligned inductive biases: The phase-centric propagation kernel and scale-conditional evolution module are elegant, well-motivated, and contribute to both interpretability and efficiency.

Strong empirical performance: The model demonstrates consistent or superior results to leading deep learning weather models (FourCastNet, ClimODE) on ERA5 benchmarks, with massive reductions in compute and memory.

Zero-/few-shot generalization: The cross-variable transfer results are particularly impressive, hinting at reusable representations across meteorological variables.

Thorough evaluation: Ablation, sensitivity, and real-case analyses (Ahvaz heat event) are well-presented, reinforcing claims about physical fidelity and robustness.

Clarity and reproducibility: The paper provides detailed algorithmic pseudocode and well-documented baselines, enhancing reproducibility.

### Weaknesses
Limited scale of validation: Experiments are limited to low-resolution (5.625°) ERA5 grids; generalization to higher resolutions or local/regional models is untested.

Theoretical justification remains heuristic: While the phase–amplitude separation is intuitively justified, the physics–mathematics linkage (especially in phase-centric attention) lacks formal derivation or stability analysis.

Dependence on FFT assumptions: The method relies heavily on global Fourier bases, which might underperform for non-stationary, non-periodic regional phenomena (e.g., tropical convection, localized storms).

Comparison scope: While comparisons to neural operators and weather transformers are included, it lacks benchmarking against emerging 2025–2026 hybrid or foundation models (e.g., EarthGPT, DeepClimaNet).

Terminological overreach: Calling ClimateLLM an “LLM” may be overstated—while GPT-style architecture is used, there is no natural language interface or token-level semantic modeling.

### Questions
How does ClimateLLM handle localized, non-periodic boundary conditions where FFT-based global modes may be inefficient or physically inconsistent?

Can the phase-centric kernel be related to or derived from wave propagation equations (e.g., Helmholtz or Navier–Stokes dispersion relations)?

How does performance scale with resolution—does the efficiency advantage persist for 0.25° or higher-resolution grids?

Could the authors compare to physics-informed hybrid models that incorporate conservation laws directly in spectral space?

Are there stability concerns when phase wrapping is handled via the angular wrap() function for long temporal horizons?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduce an architecture called ClimateLLM that operates in the frequency domain. The key contribution is the Transformer architecture that is able to process two distinct physical processes of energy evolution (amplitude) and spatial propagation (phase) of atmospheric waves. It achieves this through three key mechanisms: a dual-state representation  for amplitude and phase, a phase-centric propagation kernel for computing interactions, and a scale-conditional projection to apply specialized dynamics across different wavenumbers. The authors demonstrate that their model matches or exceeds the performance of state-of-the-art models on the ERA5 benchmark, while being more computationally efficient. Furthermore, the model shows zero-shot and few-shot generalization capabilities.

### Strengths
- The idea of explicitly separating aptitude (energy) and phase (propagation) is crucial for effective spectral forecasting. This paradigm is a relevant contribution from models that treat the spectrum as a generic feature vector.
- The architecture is deeply thoughtfully designed. The dual-state representation, the phase-centric kernel, and the Adaptive Dynamics Arbiter are all non-trivial. The architecture is optimised to preserve the physical assumptions.
- The architecture achieve state of the art even though it is trained on one single GPU
- The zero-shot and few-shot results appealing. The ability to produce robust forecast for unseen variables suggest that the model is able to learn generalisable psychical principles.

### Weaknesses
- The primary weakness is that the work is validated on a very coarse 5.625° ERA5 grid. While operational weather model operate in 0.25° grid. The number of token would increase by order of magnitude at higher resolutions. The Transformer's quadratic complexity in sequence length could become a problem
- The paper employs a standard 2D Fast Fourier Transform, which assumes a periodic, toroidal domain. While this is appropriate for longitude, it is physically incorrect for latitude (the North Pole does not connect to the South Pole). This can introduce boundary artifacts and misrepresent polar dynamics. While these effects may be minimal at low resolution, they could become a significant source of error at higher resolutions. A discussion of this limitation and the potential use of more appropriate basis functions (e.g., Spherical Harmonics) is missing.
- The paper is benchmarked on relatively smooth, large-scale variables.  The model should be benchmarked on more challenging, non-linear, and multi-scale variables like precipitation.

### Questions
Considering the weakness described previously, could the author:
- Elaborate the anticipated scaling properties of ClimateLLM? How does the computational cost scale with increasing grid resolution. Please state those with both 1° or 0.25°. Have you performed any preliminary experiments that suggest the efficiency gains will be preserved?
- The choice of a 2D FFT for a spherical globe is a key simplification. Could you discuss the potential impact of the periodicity assumption at the poles? Have you considered using Spherical Harmonics, and if so, what are the trade-offs you see between their physical fidelity and the computational/architectural complexity they would introduce?
- How do you hypothesise the  model would perform on variables like precipitation? Such variables have less defined wave-like structures and  are often characterized by sharp gradients and stochasticity. Would the phase-centric kernel still be as effective, and would the ADA be able to capture the complex, scale-dependent physics of convection and cloud formation?
- The evaluation extends to 144 hours (6 days). Have you tested the model's stability in a long-term autoregressive rollout (e.g., 30+ days)? Do the forecasts remain physically plausible, or do they eventually diverge, which can be an issue for learned models that do not explicitly enforce physical conservation laws?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper “ClimateLLM: Efficient Weather Forecasting via Frequency-Aware Large Language Models” introduces a frequency-domain forecasting framework for global weather prediction. Built on the SAED-Former (Scale-Aware Entangled Dynamics Transformer), the model explicitly decouples amplitude (energy evolution) and phase (spatial propagation) in spectral space. It employs a phase-centric propagation kernel and scale-conditional projection to encode physically meaningful dynamics. Experiments on the ERA5 reanalysis dataset show that ClimateLLM achieves high accuracy with low computational cost, supports zero-shot cross-variable transfer.

### Strengths
The model introduces an explicit amplitude–phase separation in spectral space, aligning deep learning with physical weather dynamics. The paper includes detailed ablations, sensitivity analysis. Demonstrate the potential of zero/few shot predictions and efficient training.

### Weaknesses
* The use of the term “LLM” may be misleading. Although the backbone adopts a GPT-2 architecture, it is not a true language model. The paper should clarify that “LLM” here refers either to the Transformer architecture itself or to the “latent language of dynamics.”

* The dataset scale is too small and not consistent with common practices in this research area. The experiments are conducted on data with a resolution of 32×64 and only four meteorological variables. In contrast, typical setups use 0.25° spatial resolution, 13 vertical levels, and include variables such as wind, temperature, humidity, and pressure.

* The comparison omits key baseline backbones widely recognized in the field. In particular, Pangu-Weather and GraphCast, two representative models for atmospheric prediction, are not included in the evaluation.

* The model shows limited 6-hour forecasting ability. In Table 2, its 6-hour performance for Z (139.5) is significantly worse than that of ClimODE (112.3).

* The FFN contribution lacks novelty. FFNs are not new in this domain—they have already been used in FourCastNet [1]. However, the ablation study (Line 430) shows that the FFN contributes far more than other architectural components. This suggests that the main performance gain of ClimLLM comes from a modeling technique that has already been introduced, and the paper does not compare with FourCastNet [1], which also employs FFT-based modeling.

---
[1] Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators  [ICLR 2022]

### Questions
ClimODE is also a physics-informed model. What are the key differences between your approach and ClimODE in terms of physical integration or model formulation?

In Figure 2, why does the variable Z show very little sensitivity to data scale in terms of ACC performance? Could you elaborate on the underlying reason for this behavior?

### Soundness
1

### Presentation
2

### Contribution
1
