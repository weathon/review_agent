# SmartDJ: Declarative Audio Editing with Audio Language Model

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Audio editing plays a crucial role in VR/AR immersion, virtual conferencing, sound design, and  interactive media. 
However, recent generative audio editing models depend on template-like instruction formats and are restricted to mono-channel audio.
Moreover, existing systems require users to specify low-level editing actions, rather than expressing the desired outcome at a higher semantic level.
We introduce SmartDJ, a novel framework for stereo audio editing that enables declarative audio editing, where the users describe the desired outcome while delegating the underlying editing operations to the system.
Given a high-level instruction, SmartDJ decomposes it into a sequence of atomic edit operations, such as adding, removing, or spatially relocating sound events.
These operations are then executed by a diffusion model trained to edit stereo audio. 
To enable this capability, we design a scalable data synthesis pipeline that produces paired examples of declarative instructions, atomic edit operations, and audios before and after each edit operation. 
Experiments demonstrate that SmartDJ achieves superior perceptual quality, spatial realism, and semantic alignment compared to prior audio editing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SmartDJ, a novel framework for stereo audio editing that combines the reasoning capability of audio language models with the generative power of latent diffusion.

### Strengths
1. Novel Task Formulation: The paper introduces declarative editing for stereo audio, an area that remains relatively underexplored and represents a meaningful contribution to the field.

2. Valuable Dataset Contribution: The newly collected SmartDJ dataset provides a solid foundation for training and evaluating complex audio editing tasks, facilitating future research in declarative audio manipulation.

3. Effective Methodology: The proposed ALM-assisted pipeline demonstrates promising performance in handling complex editing scenarios.

### Weaknesses
1. I note that the paper primarily uses mono-channel audio clips for SmartDJ data synthesis. However, it lacks sufficient details on how spatial audio effects—such as binauralization or room acoustics (e.g., reverberation)—are implemented or considered.

2. Due to the cascaded architecture and separate training strategy, the performance of declarative editing heavily relies on the accuracy of the generated atomic instructions. It would be valuable to include quantitative results on the accuracy/feasibility of atomic instruction generation.

3. The connection between the proposed declarative audio editing and precise spatial effects manipulation remains unclear. For instance, in Example 1 (Appendix C.1), the instruction "Make this sound like a quiet afternoon in a garden" is fulfilled by adding a gentle breeze, but the spatial direction of this sound seems arbitrary. 

4. The novelty of the single-step editing part appears limited, as its training protocol and model architecture closely follow those established in prior works. I am somehow confused.

5. The citation style does not conform to ICLR guidelines; mixing \citet{} and \citep{} reduces readability.

6. The only stated distinction between stereo and general audio editing is “direction,” yet only three direction types are considered. More directional categories are expected.

7. The dataset description is insufficient. “We sample 2–5 audio events and use GPT-4o to create 50k training pairs and 1k evaluation pairs of high-level audio editing data to train an audio language model and evaluate the editing pipeline.” This suggests the data are entirely generated and curated by GPT-4o. The evaluation set should be verified by human annotators for accuracy.

8. There is no comparison with existing editing datasets. Given the proposed 50k examples, please provide comparisons for both training and evaluation sets against prior editing datasets.

9. There are actually training-free editing methods, like DDPM inversion or better methods, that can be easily implemented for stereo editing. The authors should compare these models with SmartDJ.

### Questions
I will reconsider my score if my concerns are thoroughly addressed.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a framework for high-level audio editing that allows users to modify stereo audio through semantic instructions rather than low-level waveform manipulation. The system introduces an Audio Language Model (ALM) that generates step-by-step editing commands, guiding an audio diffusion model to perform multi-stage modifications until an optimal output is reached. The paper explores the feasibility of such semantic audio editing and evaluates its ability to handle complex instructions such as spatial or environmental sound changes.

### Strengths
•  The paper addresses a novel and forward-looking task — semantic, high-level audio editing — which goes beyond traditional waveform-level operations.

•  The idea of integrating an Audio Language Model that provides editing steps is creative and can inspire future work combining reasoning and audio generation.

•  The model supports stereo audio editing, which is technically more challenging and rarely explored in prior works.

•  The volume control (±dB) feature is interesting, as most baseline editing systems cannot precisely handle such parametric changes.

•  The overall direction contributes to bridging the gap between natural-language interaction and multi-step sound manipulation, a promising area for interactive audio systems.

### Weaknesses
•  Although the task is conceptually interesting, the training and testing data are artificially concatenated, which raises concerns about the naturalness and generalizability of the results. Real-world editing tasks often involve continuous, unstructured recordings rather than segmented mixtures.

•  The proposed framework mainly focuses on environmental or ambience-level edits (e.g., changing the spatial context or reverb), while many practical audio-editing scenarios involve precise, localized edits (e.g., removing or amplifying a specific sound event), which existing baseline models can already handle effectively.

•  The system architecture resembles a captioning + LLM planning + diffusion editing pipeline. The novelty is somewhat limited since similar modular approaches could be implemented by combining existing captioning and LLM-based reasoning models.

•  The Audio Language Model (ALM) component’s added value is unclear — it might be replaceable by prompting a strong general LLM (e.g., ChatGPT, Llama 3) with audio captions and editing prompts to produce similar step-wise operations.

•  The diffusion-based editor uses a standard DiT backbone without architectural innovation; exploring more modern generative frameworks like flow-matching or bridge-flow models could potentially improve performance and efficiency.

•  Some tables and metrics (e.g., Table 1) are difficult to interpret — it is not clear whether they measure end-to-end editing quality or only the performance of the editor module， especially it seems that there are a lot of new editing instructions generate by the ALM, such as in/decreasing the volume by dB.

### Questions
•  How does the proposed LDM-based editor compare to existing audio separation or editing models in quantitative and perceptual metrics? Only on the editing level, not combining with the ALM guided steps.  

•  What is the dataset used on result table 2, can we compare the results on the same evaluation set used by Audit? 

•  Can the ALM component be replaced by an audio-captioning model + LLM prompting pipeline? If not, what specific advantages does ALM bring?

•  Have the authors considered evaluating intermediate steps of ALM-guided editing (e.g., SmartDJ examples) to show how the step-wise process improves the result? Currently the demo only show the comparision on the final results.

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
5

### Summary
This paper introduces a framework for audio editing that breaks down high-level (declarative) instructions into a series of basic editing steps, using an LDM model trained to carry out these edits. A data synthesis pipeline is employed to create paired datasets for training. Experimental results indicate this approach delivers better perceptual quality and greater alignment with user instructions compared to several baseline methods.

### Strengths
Innovative use of both Audio Language Models (ALM) and Large Language Models (LLM):
1. The ALM acts as a planner by understanding input audio, interpreting broad user instructions, and segmenting them into a sequence of simpler editing operations.
2. The LLM contributes as a designer during data creation, generating a wide range of high-level edit commands and their corresponding atomic steps.

The study includes comprehensive experiments and ablation analyses.

Allow editing of stereo audio direction.

### Weaknesses
Breaking down user instructions into atomic editing actions isn't a novel concept for audio editing; for instance, WavCraft already transforms commands into several subtasks.

Complicated instructions may not always be easily reducible to a fixed set of basic operations.

### Questions
Are there any additional atomic operations to consider beyond the current ones?

When using Audit as a baseline, do authors use the same training data? If so, what explains the significant subjective preference when comparing SmartDJ and Audit in single-step test?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SmartDJ, a novel framework for declarative stereo audio editing. It leverages an Audio Language Model (ALM) to decompose high-level user instructions (e.g., "make this sound like a sunny forest") into a sequence of atomic editing operations (e.g., add/remove sounds, adjust volume/direction). These operations are then executed sequentially by a Latent Diffusion Model (LDM) to edit the audio while preserving spatial realism. 

The authors also propose a scalable data synthesis pipeline using LLMs and signal processing to generate paired training data, addressing the lack of such datasets. Key contributions include: (1) the first declarative stereo audio editor combining ALM reasoning with LDM generation; (2) a controllable data pipeline for editable audio scenes; and (3) empirical results showing superior perceptual quality, semantic alignment, and spatial fidelity over baselines like Audit and WavCraft, validated through metrics (e.g., FD, CLAP, FSAD) and human studies.

### Strengths
The paper demonstrates strong originality by adapting multimodal large language models (MLLMs) and diffusion-based editing techniques from the vision domain to audio, specifically addressing the underexplored area of declarative (high-level) stereo audio editing. 

The integration of an ALM as a "planner" for decomposing instructions into atomic steps, combined with an LDM "editor," is a clever modular design that allows for interpretability and potential human-in-the-loop refinement. The data synthesis pipeline is a standout contribution: using GPT-4o as a "designer" to generate instructions and steps, paired with rule-based audio composition, enables scalable training data creation, which is a practical solution to the data scarcity problem in audio editing. 

Experiments are comprehensive, including quantitative metrics (e.g., lower FD and higher CLAP scores in Table 1), ablations (e.g., multi-round editing quality in Table 3), and human evaluations showing preference for SmartDJ in fidelity and alignment. Overall, the work is well-substantiated and advances reasoning-guided audio manipulation.

### Weaknesses
While the synthetic data pipeline is innovative, the heavy reliance on composed audio from clean datasets (e.g., AudioCaps, FSD50k) may limit generalization to real-world recordings, which often include overlapping events, noise, or reverberation not fully captured in the synthesis. For instance, the atomic operations are restricted to a predefined set (add/remove/extract/volume/direction), which might not handle more complex edits like timbre changes or temporal alignments, potentially reducing flexibility for diverse instructions. 

Computationally, the sequential multi-step editing (e.g., 13.1s inference time in Table 1) is slower than end-to-end baselines, which could hinder real-time applications. 

The baselines are appropriate but could be expanded to include more recent audio editing methods (e.g., adaptations of IP2P or P2P from images to audio, beyond just Audit and WavCraft). Human studies are mentioned but details (e.g., number of participants, inter-rater agreement) are sparse in the main text, making it hard to assess robustness without appendices.

### Questions
1. How well does SmartDJ generalize to real-world audio recordings (e.g., field recordings with ambient noise or overlaps) versus the synthetic mixtures used in training? Have you tested this, and if not, what challenges do you anticipate?

2. The ALM decomposes instructions effectively in your examples, but how does it handle ambiguous, abstract, or conflicting high-level prompts (e.g., "make it eerie yet cheerful")? Could you provide failure cases or metrics for decomposition accuracy?

3. Are there plans to expand the set of atomic operations (e.g., to include reverb, pitch shifting, or event timing adjustments) to support broader editing scenarios?

4. In the human evaluation, how many participants were involved, what was the setup (e.g., A/B preference, rating scales), and what was the statistical significance of the preferences? This could strengthen the subjective claims.

5. Given the modular design, how sensitive is the overall performance to the choice of ALM backbone (e.g., LTU vs. Audio Flamingo)? Did you ablate different ALMs?

### Soundness
3

### Presentation
3

### Contribution
3
