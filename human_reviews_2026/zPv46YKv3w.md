# OWL : Geometry-Aware Spatial Reasoning for Audio Large Language Models

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Spatial reasoning is fundamental to auditory perception, yet current audio large
language models (ALLMs) largely rely on unstructured binaural cues and single-
step inference. This limits both perceptual accuracy in direction and distance
estimation and the capacity for interpretable reasoning. Recent work such as BAT
demonstrates spatial QA with binaural audio, but its reliance on coarse categorical
labels (left, right, up, down) and the absence of explicit geometric supervision
constrain resolution and robustness. We introduce the $\textbf{Spatial-Acoustic Geometry
Encoder (SAGE}$), a geometry-aware audio encoder that aligns binaural acoustic
features with 3D spatial structure using panoramic depth images and room-impulse
responses at training time, while requiring only audio at inference. Building on this
representation, we present $\textbf{OWL}$, an ALLM that integrates $\textbf{SAGE}$ with a spatially
grounded chain-of-thought to rationalize over direction-of-arrivals (DoA) and
distance estimates. Through curriculum learning from perceptual QA to multi-step
reasoning, $\textbf{OWL}$ supports o’clock-level azimuth and DoA
estimation. To enable large-scale training and evaluation, we construct and release $\textbf{BiDepth}$,
a dataset of over one million QA pairs combining binaural audio with panoramic
depth images and room impulse responses across both in-room and out-of-room scenarios. Across two benchmark datasets, our new $\textbf{BiDepth}$ and the public SpatialSoundQA, $\textbf{OWL}$ reduces mean DoA error by $\textbf{11$^{\circ}$}$ through $\textbf{SAGE}$
and improves spatial reasoning QA accuracy by up to $\textbf{25}$% over BAT. Our dataset and code are available at: https://github.com/BASHLab/OWL

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel framework for spatial audio understanding, marked by three core contributions. First, it presents BiDepth, a new large-scale synthetic dataset for spatial audio tasks. Second, it proposes SAGE, a novel spatial audio encoder designed to align binaural acoustic features with 3D spatial structure. Finally, it introduces OWL, an audio LLM that utilizes the SAGE encoder to extract and process acoustic and spatial features. The integrated OWL model is shown to achieve superior performance on both the proposed BiDepth dataset and the existing SpatialSound-QA benchmark.

### Strengths
- The development of the BiDepth dataset is a significant contribution to the field. A large-scale, well-structured dataset for spatial audio understanding can facilitate further research.
- The SAGE encoder is also a good design.

### Weaknesses
- Lack of Evaluation on Real-World Data. The model's performance is exclusively validated on simulated data. While the results are encouraging, models trained solely on synthetic data often fail to generalize to real-world scenarios due to the domain gap. The paper would be significantly strengthened by testing the model's robustness and transferability on real-world datasets, such as those from the DCASE Challenge on spatial audio tasks.

### Questions
- What is the model's performance on real-world data? For example, on the DCASE Challenge?
- In the results presented in Table 2, the model using Chain-of-Thought (CoT) reasoning shows improved performance on Type I and Type II tasks. This is counterintuitive, as these tasks do not seem to require complex reasoning. Why CoT is beneficial for these tasks?
- What is the performance of the SAGE encoder for general audio perception tasks? Besides, what are the advantages of a unified encoder like SAGE, which models audio and spatial information jointly, compared to a multi-encoder architecture that models them separately? For instance, [1] demonstrates success with separate modeling. Have you experimented with combining a state-of-the-art audio encoder with the SAGE encoder?

[1] Tang, C.,et al. Can Large Language Models Understand Spatial Audio? Proc. Interspeech 2024

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
5

### Summary
The paper introduces SAGE (Spatial-Acoustic Geometry Encoder), a geometry-aware audio encoder, and OWL, an audio large language model (ALLM) that integrates SAGE with a chain-of-thought (CoT) spatial reasoning mechanism for auditory tasks. To support this approach, the authors curate BiDepth, a large-scale simulated dataset of over 1.1 million question-answer (QA) pairs, combining binaural audio, room impulse responses (RIR), panoramic depth images, and spatially grounded QA annotations. SAGE is supervised with paired geometric and acoustic data during training (but requires only audio at inference), and OWL builds on SAGE to enable structured spatial reasoning with interpretable rationales. Experiments across both new (BiDepth) and existing (SpatialSoundQA) datasets show consistent improvements in direction-of-arrival (DoA) estimation, spatial QA, and multi-step reasoning compared to prior work.

### Strengths
1. The introduction of SAGE, trained with auxiliary geometric supervision (via depth/RIR) but requiring only audio at inference, effectively bridges a key gap in current multimodal audio-language models (ALLMs), which often lack geometric grounding.

2. BiDepth represents a substantial, balanced, and diverse resource for training and evaluating geometry-aware audio-language systems.

3. Quantitative results demonstrate that SAGE and OWL outperform strong baselines (such as BAT and Spatial-AST) on both event detection and fine-grained DoA and spatial reasoning accuracy.

### Weaknesses
1. Reliance on synthetic data limits real-world generalizability.
2. The BiDepth dataset’s elevation coverage is heavily skewed toward the horizontal plane (Figure 5); this is realistic for indoor sources but may result in models struggling with uncommon vertical positions.

### Questions
1. Given the elevation bias displayed in Figure 5, how severe is OWL/SAGE’s performance degradation for sources at highly non-horizontal elevations?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- This paper introduces OWL, an ALLM for geometry-aware spatial reasoning. It integrates an audio encoder with a spatially grounded CoT, enabling it to perform multi-step reasoning over direction and distance estimates. A large-scale dataset, coined BiDepth, is constructed to power this framework.
- The Spatial-Acoustic Geometry Encoder is trained to align binaural acoustic features with 3D spatial structure using panoramic depth images and simulated RIRs.
- Evaluated on BiDepth and the public SpatialSoundQA dataset, OWL demonstrates state-of-the-art SELD performance and spatial reasoning QA accuracy compared to major baselines.

### Strengths
- This work provides a comprehensive end-to-end framework for geometry-aware spatial reasoning, addressing all key components from the SAGE encoder and the OWL LLM to the specialized BiDepth training data.
- The paper introduces BiDepth, the first large-scale dataset of its kind, and its construction is sufficiently detailed in the appendix.
- The framework demonstrates significantly superior performance against both open-source and closed-source baselines, supported by extensive experimental validation and ablation studies.

### Weaknesses
- My main concern lies in the complexity and potential over-fitting of the proposed training pipeline. The framework requires a two-stage SAGE encoder pre-training, followed by a three-stage curriculum for the OWL model. While the ablation studies confirm this complex, multi-step process is necessary for the reported performance, it gives a strong impression of being tightly tailored to the custom-built BiDepth dataset. This raises significant questions about whether the approach is generally applicable or a highly specific solution.
- The contribution of using panoramic depth maps feels overstated, as this is a standard modality provided by the underlying SoundSpaces and Matterport simulation platforms. This is further confused by diagrams (Fig. 2) that imply the presence of specific sounding objects in depth maps (e.g., a cat), but the methodology does not discuss augmenting the 3D scenes with such new visual assets, only placing sound sources at coordinates.
- The design choice of dividing azimuth into 12 "o'clock" sectors feels ad-hoc and is not justified against other discretization schemes or as an established practice.

### Questions
Please refer to the weaknesses for detail. Almost all citations in the main draft are ill-formatted, making it quite uncomfortable to read.

### Soundness
2

### Presentation
2

### Contribution
3
