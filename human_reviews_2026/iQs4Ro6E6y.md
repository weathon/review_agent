# OmniFace: Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer

- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Video Face Swapping (VFS) requires seamlessly injecting a source identity into a target video while meticulously preserving the original pose, expression, lighting, background, and dynamic information. Existing methods struggle to maintain identity similarity and attribute preservation while preserving temporal consistency. To address the challenge, we propose a comprehensive framework to seamlessly transfer the superiority of Image Face Swapping (IFS) to the video domain. We first introduce a novel data pipeline SyncID-Pipe that pre-trains an Identity-Anchored Video Synthesizer and combines it with IFS models to construct bidirectional ID quadruplets for explicit supervision. Building upon paired data, we propose the first Diffusion Transformer-based framework OmniFace, employing a core Modality-Aware Conditioning module to discriminatively inject multi-model conditions. Meanwhile, we propose a Synthetic-to-Real Curriculum mechanism and an Identity-Coherence Reinforcement Learning strategy to enhance visual realism and identity consistency under challenging scenarios. To address the issue of limited benchmarks, we introduce IDBench-V, a comprehensive benchmark encompassing diverse scenes. Extensive experiments demonstrate OmniFace outperforms state-of-the-art methods and further exhibits exceptional versatility, which can be seamlessly adapted to various swap-related tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces OmniFace, a Diffusion Transformer-based framework for Video Face Swapping (VFS). It leverages advances from Image Face Swapping (IFS) to address challenges in maintaining identity similarity, attribute preservation, and temporal consistency in videos. The authors propose a data pipeline, SyncID-Pipe, and a new benchmark, IDBench-V, for video face swapping. Experiments show that OmniFace achieves superior performance and versatility compared to state-of-the-art methods.

### Strengths
- This paper is well written and easy to follow.

- IDBench-V is a new and diverse benchmark for video face swapping.

- The authors conduct various experiments to demonstrate the effectiveness of the proposed method, both quantitatively and qualitatively.

### Weaknesses
- While the paper is well executed and well written, I do not consider the core idea to be particularly novel. Both the data pipeline SyncID-Pipe and the OmniFace framework build on existing algorithms, and few new ideas or inspirations are proposed. The authors are encouraged to better highlight their contributions.

- Simply being the first to apply an existing architecture (such as a Diffusion Transformer) to a new domain (video face swapping) is not always a significant or impactful novelty. The authors should provide more evidence as to whether unique challenges of the domain are addressed and whether the resulting system delivers meaningful improvements or insights.

- The results in Figure 13 show that although this method can better change the identity, it does not seem to handle hair occlusion very well.

### Questions
- What are the specific benefits of using a Diffusion Transformer (DiT) for face swapping, especially considering the computational cost, compared to other existing architectures?

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
4

### Summary
This paper presents OmniFace, a comprehensive framework that transfers the strengths of image face swapping (IFS) into the video face swapping (VFS) domain. The approach introduces (1) SyncID-Pipe, a data generation pipeline that creates bidirectional ID quadruplets using an Identity-Anchored Video Synthesizer (IVS); (2) a Diffusion Transformer (DiT)–based architecture with Modality-Aware Conditioning (MC) for identity, structure, and context fusion; and (3) a Synthetic-to-Real Curriculum and Identity-Coherence Reinforcement Learning (IRL) mechanism to enhance realism and temporal consistency.
The authors also propose IDBench-V, a new benchmark for VFS, and show that OmniFace achieves state-of-the-art performance across identity similarity, attribute preservation, and video quality metrics, outperforming DreamID, CanonSwap, and Stand-In.

### Strengths
1. Data and benchmark contribution: SyncID-Pipe provides explicitly supervised video–image pairs, and IDBench-V offers a much-needed standardized evaluation dataset.

2. Strong empirical results: Extensive experiments, ablations, and user studies consistently demonstrate superior performance in both quantitative metrics and perceptual quality.

3. Well-structured paper: Clear methodology, detailed theoretical grounding, and transparent training setup.

### Weaknesses
1. Lighting robustness. It remains unclear how the model performs under complex or rapidly changing illumination conditions. An ablation or robustness analysis in such scenarios would strengthen the paper.

2. Identity change in Figure 1. I noticed that the identity appears to change in the first row of Figure 1. Does the method work reliably only when the source and target identities share similar facial structures or appearances? Could this limitation be related to the pose guidance mechanism?

3. Ethical considerations. The paper does not sufficiently address the ethical implications of high-fidelity face swapping, such as potential misuse and the need for responsible use guidelines. Including a short discussion on these aspects would improve completeness and balance.

### Questions
1. Lighting robustness. It remains unclear how the model performs under complex or rapidly changing illumination conditions. An ablation or robustness analysis in such scenarios would strengthen the paper.

2. Identity change in Figure 1. I noticed that the identity appears to change in the first row of Figure 1. Does the method work reliably only when the source and target identities share similar facial structures or appearances? Could this limitation be related to the pose guidance mechanism?

3. Ethical considerations. The paper does not sufficiently address the ethical implications of high-fidelity face swapping, such as potential misuse and the need for responsible use guidelines. Including a short discussion on these aspects would improve completeness and balance.

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
4

### Summary
To address the challenges of video face swapping task in maintaining identity similarity and attribute preservation while preserving temporal consistency, the paper proposes the first video face swapping framework OmniFace based on DiT. A novel data pipeline SyncID-Pipe is introduced to transfer the superiority of IFS to VFS. IDBench-V, a comprehensive benchmark tailored for the video face swapping task is also introduced.

### Strengths
1. New Data Pipeline: The proposed SyncID-Pipe pipeline transfers IFS’s strong ability of identity preservation to VFS task.
2. Well-structured Writing: Each module is clearly described with informative figures.
3. Comprehensive Experiments: The evaluation includes three major dimensions (identity consistency, attribute preservation, and video quality) with several SOTA models. A user study is also conducted.
4. Thorough Experiments: Ablation study convinces the effect of each proposed component and their necessity.
5. New Benchmark: The proposed IDBench-V benchmark fills a gap and provides a standardization for VFS evaluation.

### Weaknesses
1. The introduction is redundant and overlapped with related works. The logic is confusing since modules of OmniFace are stacked together and lack correspondence with the addressed challenges.
2. The novelty is incremental. DiT has been used in video generation tasks and Stand-In uses Wan2.1 as the video generation base model, which adopts DiT architecture. This undermines the contributions in terms of novelty.
3. The training requires 6,000 GPU hours, which is costive. There should be a comparison about the computation cost and inference speed.
4. The introduction of IDBench-V benchmark is insufficient. The IDBench-V includes only 200 samples, which is limited in size and diversity.

### Questions
1. A comparison about the computation cost and inference speed should be provided.
2. The modules in OmniFace is more like an integration of several strategies. Their correlation and correspondence with the goal should be better illustrated.
3. Ablation study doesn’t include the Modality-Aware Conditioning module.
4. Stand-in project page provides application in video face swapping. The comparison in experiment should adopt this version.
5. Visualization and examples of the challenging scenarios (extreme head poses, severe occlusions, complex and dynamic expressions, and cluttered multi-person scenes) in IDBench-V should be provided.

### Soundness
4

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
4

### Summary
This paper introduces OmniFace, a comprehensive framework designed to bridge the gap between image and video face swapping using a Diffusion Transformer (DiT) architecture. The authors propose a novel SyncID-Pipe data pipeline for constructing explicitly supervised paired data (bidirectional quadruplets) by integrating strengths from state-of-the-art image face swapping models and an Identity-Anchored Video Synthesizer (IVS). OmniFace employs a modality-aware conditioning mechanism to disentangle and inject multimodal information, a synthetic-to-real curriculum learning scheme, and an Identity-Coherence Reinforcement Learning (IRL) approach to improve robustness and consistency. The work further contributes a new benchmark, IDBench-V, to better evaluate video face swapping systems. Experimental results on IDBench-V and ablation studies demonstrate state-of-the-art performance, high versatility, and adaptability to extended human-centric swapping tasks.

### Strengths
1.The SyncID-Pipe pipeline ingeniously leverages strong image face swapping performance and adapts it for video by crafting bidirectional ID quadruplets, ensuring explicit and effective supervision for video models. This pipeline is well illustrated in Figure 2 and is a core element in improving alignment between image and video domains.
2.The Diffusion Transformer–based formulation is appropriately tailored for video, with a carefully crafted Modality-Aware Conditioning (MC) module for disentangling spatiotemporal, structural, and identity information (explained and visualized in Figure 3).
3.The introduction of IDBench-V (as shown in Figure 8) fills an acknowledged gap by systematically evaluating face swapping methods in diverse real-world scenarios.
4.All core equations (e.g., adaptive pose attention, Q-value definition, IRL loss, guidance purification) are specified with notation, and Appendix A.5 provides a proper theoretical justification for the IRL concept, relating it to reward-weighted likelihood maximization.

### Weaknesses
1.Some aspects of the training process are not entirely transparent.
2.The baseline selection (as shown in Table 1) is mostly well done, but some important recent works in video face swapping, especially those involving subject-agnostic or reenactment models (e.g., direct comparison with FSGAN, which is not referenced), are not included. Even though these may not be directly SOTA, including them would clarify how much improvement is due to architectural advances versus data curation. More detailed explanations for the omission of certain non-diffusion approaches would help.
3.The limitations section (Appendix A.8) briefly notes issues with speed and lighting preservation but lacks concrete error analysis on where (or why) the model produces visible artifacts, identity drift, or temporal inconsistency.

### Questions
1.Could the authors provide a more detailed breakdown of common failure cases for OmniFace, including qualitative examples and quantitative error rates for scenarios such as extreme lighting changes, multi-subject videos, or minority demographics?
2.Is overfitting to the synthetic domain a concern?

### Soundness
4

### Presentation
3

### Contribution
3
