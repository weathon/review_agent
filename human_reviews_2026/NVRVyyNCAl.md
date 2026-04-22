# Bio-inspired Working Memory for Online Auditory Pattern Drift Detection

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6, 2

## Abstract
Recent advances in Audio Language Models (ALMs) have attracted unprecedented attention. However, transformer-based ALMs face challenges in long-form audio understanding due to inefficient attention allocation. To address this, we introduce a biologically inspired working memory module, BioWM (Bio-inspired Working Memory), which leverages unsupervised online drift detection as an adaptive attention allocation strategy. BioWM detects auditory pattern drifts by monitoring energy fluctuations induced by spatio-temporal shifts, enabling the model to focus on salient changes. The BioWM does not require long-term historical data or offline pretraining; instead, it adapts online with only a few steps of threshold adjustment. Our approach captures novel events while remaining robust to transient perturbations. Furthermore, BioWM exhibits oscillatory frequency-band dynamics that resemble cortical activity during working memory tasks, thereby strengthening its biological plausibility. We present comprehensive experiments demonstrating the effectiveness of BioWM and provide visualizations of its evolving internal states to highlight both performance gains and interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces NAACA (NeuroAuditory Attentive Cognitive Architecture), a biologically-inspired framework designed for unsupervised online auditory pattern drift detection without requiring historical data or training phases. The main component is BioWM (Biologically-inspired Working Memory), which is a novel 2D recurrent field model defined on a G × G lattice. Incoming audio streams are transformed to oscillatory drive signals that drives the BioWM. Drift detection is done by monitoring energy fluctuations within BioWM with an adaptive threshold. Qualitative results shows that BioWM can detect and distinguishe different forms of drift beyond low-level acoustic fluctuations.

### Strengths
- Novel 2D recurrent field model that uses oscillatory dynamics and spatial coupling for online drift detection.
- Does not require historical data or training phases, making the method straightforward to implement.

### Weaknesses
- While the qualitative results are shown and discussed, they lack a comprehensive quantitative comparison. 
- There is a lack of evidence that the proposed NAACA helps in long-form audio understanding.

### Questions
- Noted on the lack of benchmarks, do the authors have results on a proxy dataset to validate their performance claims?
- Any quantitative evidence that the proposed NAACA helps in long-form audio understanding?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces NAACA (NeuroAuditory Attentive Cognitive Architecture), a bio-inspired framework for unsupervised online auditory pattern drift detection in long-form audio streams. The core component is BioWM (Bio-inspired Working Memory), a 2D recurrent neural field model governed by damped wave equations with spatially-varying propagation speed. The system processes audio through a pretrained encoder, modulates event probabilities as frequency-specific oscillatory inputs to BioWM grids, and detects drift via energy fluctuations against adaptive thresholds. The authors provide theoretical analysis showing that binary striped wave-speed distributions optimize drift sensitivity, and demonstrate the approach on urban soundscape recordings, claiming advantages over cosine-similarity based baselines in distinguishing genuine pattern changes from transient variations.

### Strengths
1. Online unsupervised approach, requiring no retraining on large datasets, making reproducibility more feasible.
2. Clear algorithmic description in sections B1 - B2, enhance reproducibility.
3. The integration of wave equation dynamics, and auditory processing is creative and can inspire future interdisciplinary work in neuroscience and machine learning.

### Weaknesses
1. Comparison only to one baseline. Omitted comparison to other plausible baselines like MCD-DD or DriftLens or other statistical methods (e.g. Page-Hinkley)
2. No quantitative benchmark or statistical comparison beyond DDR; lacks confidence intervals. 
3. The correspondence between BioWM oscillations and cortical gamma/beta activity is superficial. Real neural oscillations emerge from spiking dynamics and synaptic plasticity, not discretized wave equations with hand-tuned parameters.
4. No runtime or computational-efficiency analysis to support “lightweight” claims.
5. Hyperparameter selection seems arbitrary (e.g. damping, persistence P=3, cooldown C=3). No ablation or justification has been provided.

### Questions
1. How sensitive is detection performance to the adaptive threshold parameters (α, window size W)?
2. What is the runtime per second of audio? How does it compare to encoder inference time and similarity computation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes NAACA, a bottom-up framework for online, unsupervised auditory pattern drift detection. Its core is BioWM, a 2D wave-based recurrent field with "primary" and "velocity" neurons. Audio is windowed, encoded by a pretrained audio model to event probabilities, modulated into frequency-specific sinusoidal drives over spatial parcels, integrated by BioWM, and an energy-based change score is compared to an adaptive threshold with persistence filtering to flag drifts. The theory argues that binary, striped spatial distributions of wave speed $c(x, y)$ maximize sensitivity. Experiments on USoW urban soundscapes provide qualitative cases (novel onsets, pause robustness, sub-category changes) and report a Drift Detection Rate (DDR) comparison against a cosine-similarity baseline.

### Strengths
- Well-motivated and biologically grounded formulation. The paper identifies a clear gap in existing online drift-detection methods—namely, their inability to distinguish meaningful pattern changes from natural variability without long-term history—and convincingly argues for a bio-inspired approach. By linking BioWM’s oscillatory dynamics to empirical findings on gamma-band activity during auditory working memory, the authors provide a conceptually coherent bridge between neuroscience and computation.

- Novel recurrent-wave design with self-sustained memory. BioWM’s 2D spatial field governed by wave equations, with primary and velocity neurons, supports both frequency selectivity and short-term persistence. This architecture enables the model to retain recent auditory context without explicit history buffers, providing robustness to transient pauses and noise.

- The proposed NAACA pipeline performs drift detection through online modulation and adaptive thresholding, requiring no pretraining or labeled data. This characteristic directly addresses the computational and data-efficiency bottlenecks of prior contrastive or statistical drift-detection methods.

- Experiments on urban soundscapes (USoW) demonstrate detection of three conceptually distinct drifts—novel-event onset, transient-pause robustness, and subcategory-level changes—along with reduced false positives compared to a cosine-similarity baseline, as quantified by a compact Drift Detection Rate (DDR) metric.

### Weaknesses
- Limited experimental rigor and quantitative evaluation.
The evaluation relies almost entirely on qualitative visualizations and a custom Drift Detection Rate (DDR) measure without annotated ground truth. There are no objective detection metrics (e.g., precision, recall, latency, false-alarm rate) or statistical significance tests. As a result, the claimed improvements over the cosine-similarity baseline remain suggestive rather than conclusive.

- Insufficient baseline comparisons.
The paper compares NAACA only to a cosine-similarity drift detector. Other relevant unsupervised or adaptive detection methods—such as change-point models, reconstruction-based detectors, or recurrent attention mechanisms—are not evaluated. This omission makes it unclear whether BioWM offers consistent advantages beyond this minimal baseline.

- Ambiguity in system design and implementation details. The description of how the pretrained audio encoder (e.g., PANN) interfaces with BioWM lacks clarity. The paper alternately refers to event probabilities, feature embeddings, and carrier modulations without specifying the mapping between these representations and the BioWM input field. Reproducibility would benefit from explicit dimensionalities, parameter values, and update rates.

- Heuristic adaptive thresholding without calibration study.
The adaptive-threshold formula and persistence filtering are chosen heuristically, but the paper does not analyze their stability or sensitivity. It remains uncertain how threshold drift or parameter tuning affects detection performance across environments or drift types.

- All experiments are conducted on a single dataset (USoW) with selected case studies. The method’s generality across domains, such as speech, music, etc. Broader validation would strengthen claims of domain-independent applicability.

### Questions
See above.

### Soundness
3

### Presentation
3

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
The authors focus on the problem of auditory attention, where non stationary auditory patterns require models to correctly attend to salient events and filter out background. To do so, they take inspiration from biological working memory to introduce BioWM (Bio-inspired Working Memory). BioWM is a 2D recurrent neural network that functions as a working memory module. Auditory events are mapped to carrier frequencies that are then stored in BioWM. BioWM is the core module of what the authors call the NeuroAuditory Attentive Cognitive Architecture (NAACA). NAACA uses BioWM to track the history and strength of various auditory events. NAACA uses this information to calculate the relative energy fluctuations of auditory patterns which, combined with an adaptive threshold, allows the model to identify when new auditory events or changes occur. The authors show that this model works in identifying salient events in various auditory clips better than a cosine similarity baseline. They also show theoretical proofs for design choices in their model. Finally, they discuss how oscillatory dynamics in their model resemble that of cortex.

### Strengths
- The suggested model is original and unique, using a 2D spatial network and different carrier frequencies to encode sound "memories". 
- The authors explored the theoretical implications of their setup well, showing proofs to motivate various design choices.  
- The model also seems to work well while using biologically plausible components.

### Weaknesses
- The experimental comparisons rely on a few examples of event detection in real world auditory clips. I felt like the task goals were not clearly defined and case-by-case subjective. For instance, in Figure 3A, the goal is for pauses between baby cries to not be detected. Yet in Figure 4, the desired behavior is for each percussion event to be detected separately. Overall, it is hard to evaluate the method and baseline when the desired behaviors are not carefully defined. 
- I also felt that the comparisons to baseline methods were limited. The only comparison is to a naive cosine similarity baseline, which I think could be given a better shot at doing well. The BioWM model, for instance, also has an adaptive threshold for event detection. My understanding is that the cosine similarity method uses a fixed threshold. It would be more comparable if the cosine similarity method is also given the adaptive threshold. I also think it would be useful to show the cosine similarity line in the plots as well (sort of like how the energy metric is whose for the BioWM model). 
- I was unsure about what the biological inspiration angle is adding to the paper. It's unclear whether this method is an improvement over non-biological methods. Given that, I would expect the biological plausibility of the model be leveraged as a way to propose new theories of how auditory regions in the brain may be conducting computation, or explaining existing experimental findings. However, I don't think either of these are convincingly discussed.

### Questions
- Is the carrier frequency for each event learned or assigned by the experimenter? My impression is that it's the latter. If so, can you discuss how frequencies may be realistically assigned in a real world setting?
- I was also uncertain about the memory mechanism of BioWM. $\Omega_i$ is described as the attractor for an event, but there’s no weight updates mentioned. Are the attractor states already incorporated into the weights of the network from the beginning?
- A key result of section 3.4 is that the oscillatory activity is clustered rather than non-uniform. Isn’t this reflective of the experimenter design, where events are assigned carrier frequencies and attractor dynamics would bias one of those to be picked out?

### Soundness
2

### Presentation
3

### Contribution
2
