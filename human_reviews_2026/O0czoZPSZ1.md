# Scalable RF Simulation in Generative 4D Worlds

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Radio Frequency (RF) sensing has emerged as a powerful, privacy-preserving alternative to vision-based methods for various perception tasks. However, building high-quality RF datasets in dynamic and diverse environments remains a major challenge. To address this, we introduce WaveVerse, a prompt-based, scalable framework that simulates realistic RF signals from generated indoor scenes with human motions. WaveVerse introduces a language-guided 4D world generator and a physics-based signal simulator that enables the realistic simulation of RF signals in diverse environments. Experiments validate the effectiveness of our method, and we present two case studies showing WaveVerse not only enables data generation for RF imaging for the first time, but also consistently achieves performance gains in both data-limited and data-adequate scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents WAVEVERSE, a scalable framework that generates realistic RF sensing data by combining LLM-driven 4D world generation (text-based scene and motion synthesis) with phase-coherent ray tracing for physically accurate signal simulation, enabling diverse and high-fidelity RF datasets synthesis.

### Strengths
+ The integration of LLM-driven environment and motion generation with a phase-coherent ray-tracing simulator is thoughtfully designed and demonstrates strong engineering effort, providing a scalable pipeline for synthesizing diverse RF datasets.

+ The work is conceptually innovative in its vision of bridging generative AI and physical simulation, laying a solid foundation for future research that combines semantic scene synthesis with RF propagation modeling.

### Weaknesses
- Although the framework is well engineered, its core innovations build on existing foundations such as LLM-based 3D environment generation, SMPL-driven motion synthesis, and conventional ray-tracing techniques. The work would be stronger with clearer methodological advances that are unique to RF simulation, for example through learned physical priors, adaptive ray selection, or a formal analysis of phase coherence that distinguishes it from prior generative or physics-based systems.

- The idea is interesting and potentially impactful, yet the method inherently faces a fundamental limitation: it remains unclear how closely the simulated RF data align with real-world RF measurements. Because the 4D scene generation process creates entirely synthetic environments and motions that cannot be exactly replicated in the physical world, there is no definitive way to quantify the realism gap between simulated and real signals. Although the downstream case studies show encouraging performance gains, these results do not fully establish physical correspondence, leaving uncertainty about how faithfully WAVEVERSE captures real propagation characteristics.


- The path-based motion conditioning, while scalable, lacks explicit temporal control, which may result in unrealistic motion timing or velocity patterns that undermine physical plausibility in dynamic scenes. Moreover, the phase-coherent ray tracing, though conceptually well-motivated, lacks formal error analysis to verify its phase preservation accuracy.

### Questions
Please see the points raised in the Weaknesses section.

### Soundness
3

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
This paper proposes a framework for scalable RF simulation inside generative 4D (space–time) worlds, featuring a language-guided world generator and a state-aware causal transformer for human motion (“WaveVerse”). It is novel to couple generative scene/motion synthesis with RF propagation to accelerate scenario creation at scale.

### Strengths
1. If the world/motion generator is language-guided, it could make scenario coverage and data diversity dramatically easier.
2. Potential to unify CV-style 4D generative assets with RF rendering, bridging two active communities.

### Weaknesses
1. No verified evidence here of RF accuracy vs. ground truth (ray tracing/EM solvers/measurements).
2. Treatment of multipath, diffraction, penetration, materials, etc, unclear.
3. Generators may induce distribution shift; need calibration showing RF outputs remain physically plausible under varied prompts.

### Questions
1. Do you calibrate generative geometry/materials to match measured RF responses? Any domain-gap mitigation (e.g., distribution alignment, correction nets)?
2. Sensitivity to prompt wording, motion model, mesh resolution, material catalog size; which factors dominate RF error?
3. How do you model material EM properties, antenna patterns, phase noise/CFO, timing offsets, and human-body scattering?

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
4

### Summary
This paper addresses the critical challenge of acquiring large-scale, high-quality datasets for RF sensing by introducing WAVEVERSE, a scalable, prompt-based framework for simulating realistic RF signals in dynamic 4D worlds. The system integrates two main innovations: a language-guided 4D world generator and a physics-based, phase-coherent signal simulator. Experiments validate this approach, showing the phase-coherent simulation yields high-fidelity signals for beamforming and respiration monitoring. Case studies on high-resolution RF imaging and human activity recognition demonstrate that WAVEVERSE-generated data not only enables RF imaging simulation for the first time but also consistently improves downstream task performance in both data-limited and data-adequate settings.

### Strengths
Since RF Genesis, there have been few impressive papers in the field of RF simulation for quite some time. Overall, I hold a positive view of this paper for the following reasons:

- Case Study. I believe the most important aspect of evaluating RF simulation is the gain in real experiments. The current case study provides strong evidence of this. Based on this, I give a score of 6, and if the authors can address the weaknesses I have raised, I would be happy to increase it to 8 or higher.

- Introducing of Large Models. For RF simulation, introducing large models is an easily conceivable idea, but how to effectively introduce large models to improve system performance is not easy. The paper's approach to introducing large models is quite innovative and indeed effective.

- Signal Simulator. The newly designed signal generator demonstrates a certain degree of innovation.

### Weaknesses
1. My primary concern focuses on the RF baseline methods: As mentioned earlier, I believe the case study is very important. However, the current case study only demonstrates that the proposed method is effective compared to having no simulation data. But is it effective compared to data generated by other simulation methods? I suggest adding comparative experiments with data generated by other simulation methods.

2. The paper should provide more evidence that the motions generated by the LLM are physically plausible in terms of dynamics; it only proves that the motions are semantically aligned. A motion that "looks like slipping" does not mean it is physically possible. Similarly, the dielectric properties assigned by the LLM based on semantics have not been physically validated. The "realism" foundation of this framework (LLM input) is semantics-based, while its simulator (ray tracing) is physics-based; the paper does not validate the "physical realism" gap between these two domains.

3. There are some potentially overclaimed aspects in the paper:

(1) How to define "data-adequate." The paper claims to improve performance in data-adequate scenarios. Theoretically, when data is sufficient, the improvement from simulation data should be minimal. If the improvement is significant, then the data is not sufficient.

(2) "Data generation for RF imaging for the first time" is also strange. Although data generation for RF imaging is difficult, it is not necessarily the first time."

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents WAVEVERSE, a framework that generate realistic RF signals from pure text prompts. 
The system combines LLM-driven 3D scene and human motion generation with a phase-coherent ray tracing simulator that preserves spatial and temporal phase consistency, ensuring both physical fidelity and environmental diversity. 
Experiments show that generated data (when combined with or replacing limited real-world RF datasets) significantly improves model performance in both data-limited and data-adequate settings, demonstrating its value as a scalable source of high-fidelity synthetic RF data.

### Strengths
1. Novel problem statement and formulation: RF signal in time series is really challenging.
2. Comprehensive dataset usage like Lai et al., 2024 for imaging and Singh et al., 2019 for activity recognition.
3. (Despite the overall work looks more like an itegration of existing techs rather than tacking theoretical challenges) The usage of LLM to align user's language with RF generation context and the implementation makes sense for potetial product use cases. The phase-coherence design aligns with SOTA.
4. Experiment setting makes sense to use generated data to boost current sensing task performance.

### Weaknesses
1. Limited technical depth: The work primarily integrates existing components (LLM-based scene generation, motion modeling and ray tracing) rather than introducing new theoretical insights or core algorithmic innovations (like NeRF^2 (MobiCom'23)[1] and RF Genesis(SenSys'23))[2].

2. Lack of real-data calibration: The generation workflow is not calibrated or partially trained with real RF measurements, which could clearly have had the chance to help bridge the domain gap and validate simulation fidelity.

3. Unclear inference overhead. Although the algorithm looks not very heavy, it's still needed for a multi-modal model/workflow to demonstrate the computation overhead.

4. Lack of real-world case evaluation / case-specific analysis. Despite it may be too harsh to criticize the work for lacking real-world experiments, it is important to dive into concrete real-world examples (materials or layouts for which the workflow performs worse, any external factors like surrounding WiFi/cellular signal may interfere, etc) to identify the true potential of the work.

[1] Zhao, Xiaopeng, et al. "Nerf2: Neural radio-frequency radiance fields." Proceedings of the 29th Annual International Conference on Mobile Computing and Networking. 2023.

[2] Chen, Xingyu, and Xinyu Zhang. "Rf genesis: Zero-shot generalization of mmwave sensing through simulation-based data synthesis and generative diffusion models." Proceedings of the 21st ACM Conference on Embedded Networked Sensor Systems. 2023.

### Questions
1. generalizability: In which circumstances would WAVEVERSE's performance degrade? How you intepret these cases?

### Soundness
3

### Presentation
3

### Contribution
2
