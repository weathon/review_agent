# FilMaster: Bridging Cinematic Principles and Generative AI for Automated Film Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Existing AI-based film generation systems can generate high-quality videos, but struggle to design expressive camera language and establish cinematic rhythm. This deficiency leads to templated visuals and unengaging narratives. To address these limitations, we introduce FilMaster, an end-to-end automated film generation system that integrates real-world cinematic principles to generate professional-grade, editable films. Inspired by professional filmmaking, FilMaster is built on two key cinematic principles: (1) camera language design by learning cinematography from extensive real-world film references, and (2) cinematic rhythm by emulating professional post-production workflows. For camera language, our Multi-shot Synergized Camera Language Design module introduces a novel scene-level Retrieval-Augmented Generation (RAG) framework. Unlike shot-level RAG which retrieves references independently and often leads to visual incoherence, our approach treats an entire scene, comprising multiple shots with a shared spatio-temporal context and narrative objective, as a single, unified query. This holistic query retrieves a consistent set of semantically similar shots with cinematic techniques from a large corpus of 440,000 real film clips. These references then guide an LLM to synergistically plan coherent and expressive camera language for all shots within that scene. To achieve cinematic rhythm, our Audience-Aware Cinematic Rhythm Control module emulates professional post-production, featuring a Rough Cut assembly followed by a Fine Cut process that uses simulated audience feedback to optimize the integration of video and sound for cinematic rhythm. Extensive experiments show superior performance in camera language and cinematic rhythm, paving the way for generative AI in professional filmmaking.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes FilMaster, an automated system for film generation that integrates cinematic principles—specifically, camera language design and cinematic rhythm control—into the video generation pipeline.
It introduces:

* A Multi-shot Synergized Camera Language Design module using a scene-level Retrieval-Augmented Generation (RAG) from a 440k film dataset to plan expressive, coherent shots.

* An Audience-Aware Cinematic Rhythm Control module that simulates post-production editing with “audience feedback” and multi-track sound design.

### Strengths
* Expressive camera work: Produces more natural and film-like results; shows improvement in “camera language.”

* Comprehensive multi-stage pipeline: Covers both pre- and post-production, integrating video, audio, and editing.

* Quantitative and qualitative validation: Outperforms baselines on FilmEval and user studies.

### Weaknesses
* Pipeline complexity: The system is overly intricate, requiring multiple LLMs and retrieval steps; real-time practicality is unclear.

* Manual intervention unclear: The amount of human selection or post-curation is not specified—e.g., how many outputs are generated and chosen.

* Failure cases not discussed: The paper lacks analysis of when and why the method fails (e.g., scene inconsistency, unnatural transitions).

* Audio-visual mismatch: Lip sync and timing issues occasionally appear, affecting realism.

* Aesthetic awkwardness: Some close-up (“in-your-face”) camera shots feel forced or unnatural compared to simpler long-shots like in LTX-Video.

### Questions
* Could modules (e.g., video generation, editing) be replaced with other backbones besides Kling? How modular is the system?

* What level of human intervention or curation is required—how many outputs are generated per scene, and how is the best one selected?

* What are the typical failure cases (e.g., incoherent motion, desynced audio, unrealistic rhythm)?

* Have you evaluated the generalization to unseen domains (e.g., documentaries, dialogue-heavy content)?

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
This paper presents a fully automated, end-to-end clip-level video generation agent. By providing multiple reference images and an initial prompt as input, the system invokes different models to iteratively modify both the prompt and the video. During the Coordination stage, the agent also retrieves suitable audio from an audio library for adaptation. The construction of cinematic language, agent-level video editing (such as acceleration and scene switching), and audio synchronization are particularly novel aspects of the approach. The experimental results, as well as the videos included in the supplementary materials, are impressive and refreshing.

### Strengths
1. An automatic end-to-end clip-level video generation agent with impressive performance.
2. The Coordination Stage could edit the order and duration of the generated videos, which is reasonable as inter-clip videos do not have strict temporal order constraint. The audio fusion manner is also intuitive and suitable for agent like methods (retrieval and synchronized).
3. The whole paper is well writen and easy to understand.

### Weaknesses
The performance of this paper is great, my main concern lies on fair comparison with existing methods. This paper utilizes Kling 1.6 as the video generation model, which is much better than existing open-source model that previous method used, such as CogVideoX and LTX-Video. So is the performance improvement simply caused by the basic ability of Kling? It is recommended to have a fair comparison with MovieAgent to see the contribution of this paper.

### Questions
See the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces FilMaster, an end-to-end automated film generation system designed to address key limitations in existing AI-based video generation. FilMaster proposes a Multi-shot Synergized Camera Language Design module which enables scene-level RAG that learns cinematography from movie references. To achieve a professional narrative, FilMaster further introduces Audience-Aware Cinematic Rhythm Control module for post-production workflow. The visualization experiments provide qualitative evidence for the effectiveness of the proposed method.

### Strengths
- The overall method is easy to follow.
- The visualizations in the supplementary materials offer clear qualitative support for the method's effectiveness.

### Weaknesses
**1. Limited Novelty**

I do not typically raise concerns about novelty lightly, but I must state that the technical contribution of this paper is highly limited. At its core, the proposed method is a relatively straightforward application of RAG. Crucial generative capabilities, such as identity preservation and high-fidelity video synthesis, appear to be inherited from the underlying foundational models used, rather than being novel contributions of the FilMaster framework itself. While the academic community certainly welcomes simple yet effective methods, such contributions usually offer a fundamental insight into the problem being studied. I do not believe this paper achieves that. Instead of solving a core technical challenge, the work primarily focuses on orchestrating existing components. For these reasons, the paper reads more like a well-executed technical project than a piece of novel research.

**2. Subjective Evaluation**

The primary evaluation is based on a user study, which, while valuable, is inherently subjective. Although the authors supplement this with an automatic evaluation using Gemini, this approach is also a form of subjective assessment. The paper would be much stronger if it included more objective and popular metrics. For example, text-video similarity scores could be used to measure the faithfulness of script elaboration, and a quantitative metric for identity consistency across shots would provide more robust evidence of the system's capabilities.

I understand a potential reason why authors did not discuss these metrics might be that these objective metrics would primarily test the performance of the base video generation model, not the proposed framework. However, this argument circles back to the core issue of novelty. If the main contribution is limited to the design of using cinematic language, and the measurable technical improvements are attributable to the underlying models, then the novelty of the framework itself is insufficient.


**Conclusion**

The paper is methodologically sound, but its contribution feels incremental. It does not present significant shortcomings, but conversely, it lacks any clear and compelling advantages that would distinguish it from prior work. The contribution feels unsubstantial and does not appear to meet the bar for acceptance.

I‘m willing to reconsider my score if the authors can provide a convincing rebuttal that thoroughly addresses my concerns.

### Questions
Please See Weaknesses

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aims at automated film video generation, i.e., automatically generating film videos given a text and sets of reference images for characters and locations as input. The proposed framework, named "FilMaster", attempts to surpass existing methods in terms of camera language and cinematic rhythm. The core contributions lie in the introduced Multi-shot Synergized Camera Language Design and Audience-aware Cinematic Rhythm Control, where the former is designed for camera language by introducing a shot-level RAG, and the latter module serves as a post-production process to refine the "Rough Cut" to "Fine Cut" for cinematic rhythm. FilMaster is evaluated on the proposed FilmEval benchmark and outperforms recent methods (Anim-Director, MovieAgent, and LTX-Studio) in both camera language and cinematic rhythm, which is further evidenced through a user study. The paper also presents ablation studies regarding the two proposed modules.

### Strengths
1. The paper tackles an important and underexplored area—bridging cinematic principles and AI-based film generation. The topic is both academically relevant and practically impactful.
2. The authors explicitly ground their work in film principles (camera language, cinematic rhythm, audience perception, etc.) and emulate professional filmmaking workflows. This fills an existing academic gap between generative modeling and film studies.
3. The scene-level retrieval and coordinated camera planning improve shot-level incoherence in previous RAG-based methods.

### Weaknesses
1. While the system design is well-engineered, it lacks a deep technical innovation or theoretical insight at the algorithmic level. I would prefer to see more technically substantive modules rather than a purely workflow-oriented system.
2. The system’s multi-stage pipeline (retrieval -> shot planning -> rough cut -> audience feedback -> fine cut -> sound production) introduces fragility. The paper doesn’t investigate how errors propagate or how robust the system is when upstream stages fail.
3. Despite providing reference images, generated scenes may still lack consistency across shots, especially for partially occluded or unseen parts of reference subjects/scenes. The paper does not discuss this, though it is crucial for multi-shot storytelling.
4. Some generated results in the supplementary video exhibit noticeable audio–video misalignment, including unsynchronized sound effects and inaccurate lip movements.
5. The workflow depends on GPT-4o, Gemini, and Kling video generators. This makes reproducibility extremely difficult and limits transparency.

### Questions
It remains unclear whether the system can produce varied cinematic styles or if retrieval biases constrain creativity?

### Soundness
2

### Presentation
3

### Contribution
2
