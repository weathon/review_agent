# PICABench: How Far are We from Physical Realistic  Image Editing?

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Image editing has achieved remarkable progress recently. Modern editing models could already follow complex instructions to manipulate the original content. However, beyond completing the editing instructions, the accompanying physical effects are the key to the generation realism. For example, removing an object should also remove its shadow, reflections, and interactions with nearby objects. Unfortunately, existing models and benchmarks mainly focus on instruction completion but overlook these physical effects. So, at this moment, how far are we from physically realistic image editing? To answer this, we introduce PICABench, which systematically evaluates physical realism across eight sub-dimension(spanning optics, mechanics, and state transitions) for most of the common editing operations(add, remove, attribute change, etc). We further propose the PICAEval, a reliable evaluation protocol that uses VLM-as-a-judge with per-case, region-level human annotations and questions. Beyond benchmarking, we also explore effective solutions by learning physics from videos and construct a training dataset PICA-100K.After evaluating most of the mainstream models, we observe that physical realism remains a challenging problem with large rooms to explore. We hope that our benchmark and proposed solutions can serve as a foundation for future work moving from naive content editing toward physically consistent realism.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses a critical gap in image editing by introducing PICABench, a benchmark designed to evaluate the physical realism of edited images, moving beyond mere semantic fidelity. The authors argue that while modern models can follow complex editing instructions, they largely fail to simulate accompanying physical effects (e.g., shadow/reflection updates, mechanical interactions), which is crucial for realism. To systematically assess this, PICABench categorizes physical consistency into eight fine-grained sub-dimensions spanning Optics, Mechanics, and State Transitions, covering most common editing operations. The paper also proposes PICAEval, a reliable evaluation protocol that leverages a VLM-as-a-judge with per-case, region-specific questions grounded by human annotations, thereby reducing hallucinations and improving sensitivity to nuanced physical violations. Beyond benchmarking, the authors explore a solution by learning physics from videos, constructing the PICA-100K dataset. Their experiments show that fine-tuning on PICA-100K significantly enhances a model's physical consistency. The comprehensive evaluation of 11 models reveals that physical realism remains a significant challenge, establishing this work as a foundational step towards physically plausible image editing.

### Strengths
- The research on complex instruction-based image editing is highly meaningful, and the proposed method demonstrates improvements over current state-of-the-art approaches.
- The paper presents substantial work, including the introduction of a new benchmark, the provision of corresponding training data, and fine-tuning of the FLUX model.
- The writing is clear, and the experimental comparisons and figure illustrations are professional.

### Weaknesses
- The PICA-100K construction pipeline relies on video generation models, meaning the final benchmark performance heavily depends on the quality of WAN. I am concerned about whether such data can ensure sufficient effectiveness, even if it performs well on the proposed benchmark. Given the proliferation of benchmarks for world-knowledge image editing, I question whether this data would generalize well to other benchmarks.
- The authors fine-tuned the FLUX image generation model and achieved promising results. However, I believe that tasks involving world-knowledge image editing should be improved based on VLM+Diffusion model frameworks, such as Bagel, UniWorld, or Qwen-Image-Edit. While the proposed method outperforms these unified generation-understanding models on PICABench, it may be attributed to the PICA-100K dataset being tailored specifically for this benchmark, allowing a pure diffusion model to overfit to it. It may be unreasonable to request extensive additional experiments during the rebuttal, so I encourage the authors to focus on analyzing this point. My final score will depend on the analysis provided.
- There has been a surge of related work on world-knowledge editing in recent years. While citing the latest work is not mandatory, some early relevant studies should be referenced:

[1] Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation \
[2] Anyedit: Mastering unified high-quality image editing for any idea \
[3] Editworld: Simulating world dynamics for instruction-following image editing \
[4] Reasonpix2pix: instruction reasoning dataset for advanced image editing \
[5] Smartedit: Exploring complex instruction-based image editing with multimodal large language models

### Questions
See the "Weaknesses" section. Additionally, as I have reviewed other higher-quality papers on world knowledge editing and find the choice of the fine-tuned model in this paper somewhat inappropriate, my current score is relatively low. I will adjust my rating based on the rebuttal and other reviewers' comments.

### Soundness
3

### Presentation
2

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
This paper introduces PICABench, a benchmark for systematically evaluating the physical realism of instruction-based image editing models. The authors argue that existing models and benchmarks primarily focus on semantic fidelity and instruction completion, neglecting crucial physical effects like correct shadows, reflections, and object interactions, which are key to true realism. To enable reliable evaluation, the paper proposes PICAEval, a region-grounded, VLM-as-a-judge protocol that uses per-case, region-level human annotations and binary Yes/No questions (VQA) aligned with specific physical sub-dimensions. Finally, authors propose the PICA-100K, a large-scale, synthetic training dataset (100k examples) and experiments show that fine-tuning an existing model (FLUX.1 Kontext) on PICA-100K significantly enhances its physical consistency.

### Strengths
1. PICABench covers eight **fine-grained sub-dimensions** and a variety of common editing operations. And PICAEval addresses the known limitations (hallucination, low sensitivity) of standard VLM-as-a-judge setups by incorporating region-level annotations and targeted Q&A.

2. The paper is very clear and **well-structured**. The core dimensions (Optics, Mechanics, State Transition) are immediately intuitive. Key concepts like the eight sub-dimensions are defined with concrete, checkable criteria

3. It introduces a critical **new evaluation standard** that forces the community to move beyond mere content editing toward physically consistent realism.

### Weaknesses
1. The authors note a slight drop in the State Transition Accuracy for their fine-tuned model and speculate this is due to only using the first and last frames of a video to represent meaningful state changes. This highlights a potential weakness in the PICA-100K data generation pipeline for this specific dimension. The use of only two frames might fail to capture the coherence and physical consistency of the transition process itself, as the model only sees the 'before' and 'after' states. While they mention plans to explore fine-grained strategies, the current limitation is a weakness in the proposed training data solution.

2. The metrics used (Chamfer Distance, F-Score) are standard but insufficient to evaluate perceptual quality or shape plausibility. No perceptual or task-level metric is proposed

3. The paper notes that unified multi-modal models consistently underperform compared to dedicated image editing models, speculating that their enhanced world understanding doesn't translate to physical realism due to a lack of "internalized physics principles". This is a crucial observation, but the analysis remains largely speculative. A more in-depth qualitative analysis, perhaps showing how a unified model (like Bagel or OmniGen2) fails to integrate its supposed world knowledge into the generation process (e.g., how its explicit reasoning about a scene's physics is decoupled from its final generative output), would strengthen this claim

4. Table 2 shows that performance improves with the specificity of the prompt, with a notably small gain between "superficial" and "intermediate" prompts compared to "explicit" prompts. While the authors speculate this is due to a lack of internalized physics, it could also indicate that the models are highly sensitive to explicit guidance (i.e., the "explicit" prompts describe the desired physical outcome) rather than being able to infer the physics from the instruction and scene context (which the "intermediate" prompts attempted to provide). This suggests a potential weakness in the model's ability to generalize physical principles from scene context alone

### Questions
Please see the weakness

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
3

### Summary
This paper focuses on the issue of low physical consistency in instruction-based 2D editing results, and proposes PICABench, a benchmark for physical realism preservation capabilities of image editing models. By analyzing common cases of low realism, the paper assesses the physical consistency in terms of optical, mechanical, and state transition aspects of edited image content. Specifically, this paper proposes the VLM-based evaluation metric PICAEval, combining human annotation mechanisms and QA patterns to evaluate the physical realism of 2D editing results. Furthermore, this paper constructs the synthetic image editing dataset PICA-100K based on video generation modalities to fine-tune and improve the physical preservation capabilities of large-scale 2D editing models.
Through the evaluation of existing models, this paper shows that "most existing large-scale models are still unable to achieve physically realistic 2D editing" and analyzes the gap between instruction understanding and physical preservation in existing models. Moreover, by fine-tuning models with the proposed dataset PICA-100K, this paper demonstrates the effectiveness of PICA-100K.

### Strengths
- This paper is a benchmark paper. It analyzes common physical realism issues in existing 2D editing models, and proposes PICABench and PICAEval to evaluate model capabilities from multiple dimensions. The motivation is clear, and it is important for image editing models.

- The paper explores the capabilities of VLM models, the QA evaluation method of LLM, the human annotation process and the advantages of video for dataset construction and method evaluation, which is novel and effective.

- The paper is well-organized and easy to follow.

### Weaknesses
1.  Physical awareness is important for 2D editing mode, but I still concern about requirements that editing with physical awareness directly. 
- It is more like the issue of prompts/instructions. In addition, the definition of hallucination in this paper should be clearer. The 2D editing model needs to balance between precisely edition according to the prompts, and the certain ability to extrapolate. For example, in Fig.1 bottom right (remove the scooter), it is kind of hallucination if the rider stands on the floor, which also violates our prompt. Maybe, the user only simulates a snapshot that the rider flies. 
- The middle or long instructions in Fig.4 (d) are more reasonable. Tab.2 should include more methods in the main paper. 
- Tab.2 constructs different levels of prompt, but the performance is highly impacted by the long-text comprehension. It feels like the problem has changed to 2D editing with long prompts. Therefore, it would be better to compare with methods based on multiple steps instead of one step with long prompts.

2. There is a lack of detailed explanation of metrics Acc and Con. 

3. Line 424 claims that "model performance improves as prompts become more detailed". However, Tab.2 shows that the consistency (Con) decreases in most cases. Moreover, Line 425 claims that "the gain from intermediate prompts is much smaller than that from explicit prompts". However, it is hard to verify on the basis of tables.

### Questions
See weaknesses

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
This manuscript proposes a benchmark to evaluate physical realism in image editing. The authors consider physical effects beyond simple instruction completion. They have collected a physics-aware benchmark, PICABench, to evaluate the performance of existing image editing models. Meanwhile, the authors also utilize a VLM-as-a-judge (PICAEval) to evaluate the edited images. Beyond the benchmark, the authors also propose constructing a physics-aware dataset (PICA-100K) from videos. Experimental results demonstrate that existing models still struggle to achieve physical realism. Furthermore, the proposed dataset is shown to enhance the physical realism of a fine-tuned base model.

### Strengths
This manuscript considers the role of physics in the image editing task, which is an interesting and rarely explored area of research. The proposed benchmark is well-constructed and provides a comprehensive analysis of existing models regarding the physical realism of their edits. In addition, the proposed PICA-100K dataset is a clever approach to learn physics from synthetic videos.

### Weaknesses
1. The primary weakness is that the proposed solution, fine-tuning on PICA-100K, yields very small gains. While this is a positive result and better than the baseline model, the small margin doesn't present PICA-100K as a definitive solution. Moreover, the final fine-tuned model still underperforms other top-tier models (e.g., GPT-Image-1, Seedream 4.0, Qwen-Image-Edit) in the overall benchmark, as shown in Table 1.

2. To better demonstrate the dataset's effectiveness, the authors should try fine-tuning other base models (besides Flux.1 Kontext) and report their performance improvements.

3. Meanwhile, the concept of constructing image editing pairs from synthetic videos is not entirely novel and has been explored in previous works (e.g., Frame2Frame[1], ByteMorph[2]). The reviewer recommends the authors explicitly to describe the differences and contributions of their data generation pipeline compared to these prior works.

[1] Pathways on the Image Manifold: Image Editing via Video Generation

[2] ByteMorph: Benchmarking Instruction-Guided Image Editing with Non-Rigid Motions

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
