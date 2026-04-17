# The World is Not Mono: Enabling Spatial Understanding in Large Audio-Language Models

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Existing large audio-language models perceive the world as "mono"—a single stream of audio that ignores the critical spatial dimension ("where") required for universal acoustic scene analysis. To bridge this gap, we first introduce a hierarchical framework for Auditory Scene Analysis (ASA). Guided by this framework, we introduce a system that enables models like Qwen2-Audio to understand and reason about the complex acoustic world. Our framework achieves this through three core contributions: First, we build a large-scale, synthesized binaural audio dataset to provide the rich spatial cues. Second, we design a hybrid feature projector, which leverages parallel semantic and spatial encoders to extract decoupled representations. These distinct streams are integrated via a dense fusion mechanism, ensuring the model receives a holistic view of the acoustic scene. Finally, we employ a progressive training curriculum, advancing from supervised fine-tuning (SFT) to reinforcement learning via Group Relative Policy Optimization (GRPO), to explicitly evolve the model's capabilities towards reasoning. On our comprehensive benchmark, the model demonstrates comparatively strong capability for spatial understanding. By enabling this spatial perception, our work provides a clear pathway for leveraging the powerful reasoning abilities of large models towards holistic acoustic scene analysis, advancing from "mono" semantic recognition to spatial intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces "The World is Not Mono" (TWNM), a comprehensive framework for instilling spatial audio understanding in large audio-language models (LALMs). TWNM combines a synthetic large-scale binaural audio dataset, a Mixture-of-Experts (MoE) architecture that decouples semantic and spatial processing, and a multi-stage training curriculum culminating in reinforcement learning with Group Relative Policy Optimization (GRPO). The authors benchmark their system on an auto-generated, multi-task evaluation suite that probes perception, integration, and reasoning skills.

### Strengths
1. The paper tackles an often-overlooked limitation in LALMs—spatially aware auditory reasoning. The framing underscores the need for models capable of holistic scene analysis rather than mono-dimensional semantic understanding.
2. The use of decoupled semantic and spatial encoders combined via a conditional MoE is interesting.
3. Paper is well written and easy to follow.

### Weaknesses
1. This work lacks a proper baseline and only compares results from different training stages of its own model.

2. It would be better if the analysis also included sim-to-real performance.

3. The use of multiple encoders cannot guarantee complete disentanglement of audio information, and to some extent, it may even lead to desynchronization between different information streams. As discussed in the paper, the system may correctly identify sound sources A and B but confuse their directions or distances or others. The performance of spatial-relationship reasoning at 34.02% and attribute binding integration at 37.07% further prove this. Therefore, is such a disentangled multi-encoder approach combined with MoE truly an appropriate method?

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates equipping Large Audio Language Models (LALM) with spatial perception besides semantic understanding. To accomplish this, an MoE framework with four-stage learning is trained sequentially: (1) Learn the stem spatial audio representation with a spatial encoder, decoupled from the semantic representation; (2) Learn the MoE experts and router to process the concatenated semantic and spatial embeddings; (3) (SFT1.0) Align the MoE weights with LLM (finetuned with LoRA), with additional router supervision; and (4) (SFT2.0) Remove router supervision to further align MoE with LLM in end-to-end training. Additionally, GRPO is employed to further optimize the LLM under these spatial tasks. Moreover, this work proposes a new dataset and benchmark for LALM’s spatial audio reasoning by synthesizing binaural audio from single-channel audio data and querying LLM for question-answer pairs.

### Strengths
-	Originality: This paper accurately identifies the research gap in LALM’s spatial understanding capabilities. This is a nontrivial problem as humans have strong spatial cognition in hearing.

-	Quality:  The engineering efforts of this work are solid and well-planned. The four-stage training strategy appears justified with ablations, especially by the contrast of performance between SFT1.0 and SFT2.0.

-	Clarity: The proposed pipeline is illustrated clearly in a sequential manner.

-	Significance: This work is among the first to address the spatial audio understanding problem for LALMs.

### Weaknesses
-	The presentation of problem formulation is general and unclear. The authors define spatial audio understanding by examples in introduction, but do not categorize this understanding from a broad concept into specific tasks and define each. One can only grasp the outline of these tasks until the experiments section where dataset is introduced. Paragraph 2 in introduction is especially confusing because of this over-abstraction. Please see question 1 below for a request of clarification.

-	While it’s understandable that the proposed problem is relatively novel and lacks baseline models, sufficient experiments are still needed to demonstrate the proposed solution’s legitimacy. Instead of the complex training involved, one could prompt a spatial audio model for spatial localization, and another LALM for semantic understanding of the scene. Combining these predictions and prompting them to another LLM would resolve the “spatially deaf” limitation of current LALMs. Further experiments need to be conducted on these alternative baselines for comparison, delineating the necessity of the proposed method.

-	It’s curious why the proposed training pipeline requires this much complexity. From an engineering perspective, employing MoE is sound for its ability to dedicate parameter groups to semi-explicit subtasks. However, theoretically how much more gain can be achieved with MoE than a unified encoder-LLM mapper is under-studied here. It’s hard to justify this sheer amount of engineering tricks without seeing the performance tradeoff. The originality and novelty of this work thereby are heavily affected by the lack of this ablation.

### Questions
-	What are the major tasks in spatial audio understanding? Why is each dependent on binaural cues instead of single-channel semantics?

-	How much performance gain can be attributed to the fusing module? The model could be separately picking up semantic cues from the binaural channels to accomplish certain tasks.

-	Why MoE is needed to address this task? Could a unified encoder-LLM mapper achieve similar/better performance?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a framework called “The World is Not Mono (TWNM)”, which aims to make Large Audio-Language Models (LALMs) spatially aware. Most existing models only process mono audio, this paper tackles that by giving models the ability to reason about spatial cues like direction, distance, and room acoustics. The authors build a synthetic binaural dataset using physically simulated environments with BRIR and HRTF filters to generate spatially accurate audio scenes. They further propose a task-aware Mixture-of-Experts (MoE) architecture that separates semantic and spatial processing, with specialized experts for handling aspects like direction, distance, reverberation, and source count. The training follows a progressive curriculum, beginning with supervised fine-tuning and culminating in reinforcement learning through GRPO to enhance spatial reasoning. The paper presents a new spatial reasoning benchmark comprising 1,000 multiple-choice questions designed to test perception, integration, and reasoning capabilities. Overall, the model demonstrates strong improvements across all task types, achieving an overall accuracy of 61%, with particularly notable gains in complex reasoning tasks after GRPO training.

### Strengths
- The paper addresses a fundamental gap in LALMs — spatial reasoning, which few have tackled.
- The staged curriculum (SFT → SFT 2.0 → GRPO) helps stabilize training.

### Weaknesses
- Only one LALM i.e. Qwen2-Audio is tested. 
- Limited experimental comparison with other spatial LALMs.
- Missing human evaluation.

### Questions
- Can this be generalized to other models like SALMONN, AudioGPT, or Whisper-based LALMs?
- Can spatial LALMs like BAT, etc. be compared?
- Have you tested on real spatial datasets (e.g., STARSS23 or L3DAS23)? If not, how do you expect it to handle real-world acoustics that deviate from simulation?

### Soundness
2

### Presentation
3

### Contribution
3
