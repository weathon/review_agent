# Rethinking Visual Intelligence: Insights From Video Pretraining

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Large language models (LLMs) have demonstrated that large-scale pretraining enables systems to adapt rapidly to new problems with little supervision in the language domain. This success, however, has not translated as effectively to the visual domain, where models, including LLMs, continue to struggle with compositional understanding, sample efficiency, and general-purpose problem-solving. We investigate Video Diffusion Models (VDMs) as a promising direction for bridging this gap. Pretraining on spatiotemporal data endows these models with strong inductive biases for structure and dynamics, which we hypothesize can support broad task adaptability. To test this, we design a controlled evaluation in which both a pretrained LLM and a pretrained VDM are equipped with lightweight adapters and presented with tasks in their natural modalities. Across benchmarks including ARC-AGI, ConceptARC, visual games, route planning, and cellular automata, VDMs demonstrate higher data efficiency than their language counterparts. Taken together, our results indicate that video pretraining offers inductive biases that support progress toward visual foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores Video Diffusion Models (VDMs) as a new direction for pre-training visual intelligence models, aiming to bridge the generalization and data efficiency gap between language models (LLMs) and visual models.

The authors propose a unified framework to rephrase visual tasks as image-to-image as video transition, and systematically compare LLMs and VDM with the same LoRA fine-tuning method across a range of tasks. Results demonstrate that VDMs significantly outperform LLMs in tasks requiring spatial structure understanding, logical or abstract reasoning, and route planning, while also exhibiting higher data efficiency. The paper argues that the inductive bias introduced by spatiotemporal pre-training can contribute to building more general visual foundation models.

### Strengths
This article proposes an interesting research direction, rightly noting that visual pre-training (particularly with video) remains underexplored compared to linguistic pre-training. It offers valuable insights, as its title (insights from video pretraining) suggests, and presents experimental evidence that video-pretrained VDMs can generalize more effectively to downstream tasks (e.g., spatial structure understanding, logical or abstract reasoning, and route planning) than language-pretrained models. That's why I give good **Soundness**.

### Weaknesses
While I find the core topic of this paper highly interesting, the current manuscript still has significant issues in terms of **sufficiency** and **completeness**.

- **Overly Broad Title & Narrow Focus**: The title is general, while the actual research scope is limited primarily to Video Diffusion Models (VDMs). It should be noted that video foundation models are not exclusively built on diffusion-based approaches—autoregressive (AR) models, for example, represent another important paradigm. The experiments presented are not substantial enough to support the broad claim made in the title. A more accurate title should explicitly reflect that the study focuses on video-diffusion model pre-training?

- **Insufficient Solid Content in Section 3**: The content of Section 3 reads as overly conceptual, covering mostly established background knowledge. Dedicating two full pages to such content does not effectively advance the paper's argument (There are many cases where a single sentence is a standalone paragraph, seeming unnecessary to spend so much space?). It would be more valuable to incorporate additional experimental results and analyses from the appendix into the main body. In its current form, this section gives the impression of a lack of refinement, which is the main reason for my low score in **Presentation**.

- **Limited Scope of Comparative Evaluation**: The study only demonstrates VDM's advantages over LLMs in tasks that can be naturally framed as image-to-image transitions. However, it remains unclear how VDMs would be applied to tasks that are not easily represented as image inputs—such as mathematical reasoning, code generation, or knowledge tasks. What would the input form be for such cases? Would VDMs still maintain an advantage? Most video foundation models start with image foundation models. Since the title mentions **rethinking visual intelligence**, why is it limited to discussing video pre-trained models and not including pure image pre-trained models? The experimental part conducted in the article is not sufficient to support the title of this article. In addition, why do not compare with VLM (image-text input, text out), and t2i/t2v model with text input, image ouput?

**In summary, I believe the current manuscript requires considerable improvement and comes across as somewhat rushed. With substantial revision and expansion, I think it may be a good article**.

### Questions
See Weakness.

Another point concern is that how to determine the correctness of the VDM's output image, specifically how it is evaluated against the ground truth. I did not find the corresponding descriptions for this in the main text.

### Soundness
3

### Presentation
1

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
The authors evaluate video diffusion models and LLMs on visual tasks like ARC-AGI, path planning, sudoku etc. using a LoRA based finetuning setup, to demonstrate that video diffusion models are more data efficient than LLMs when it comes to visual reasoning tasks.

### Strengths
1. The authors have curated a set of interesting visual tasks to benchmark the spatial reasoning capacity of VDMs and LLMs from cellular automata to visual games.
2. The ARC-AGI results are quite novel and timely and highlight drawbacks of current LLMs.

### Weaknesses
1. It seems all the visual tasks require spatial reasoning, not spatio-temporal reasoning, which begs the question why not evaluate image diffusion models as well instead of video diffusion models where the uathors practically discard the temporally intermediate frames generated by the model, essentially not using/evaluating the temporal reasoning capacity of these models.

2. The data efficiency plots compare cog-x with qwen without controlling for pre-training FLOPs/data-volume. This is a very important factor that can determine baseline model performance and should be reported.

3. The authors need to show results for more than one VDM and LLM across all these tasks in order to make general claims about model families.

4. The authors also need to demonstrate scaling behaviors of these VDMs, showing improvement in data efficiency/visual reasoning performance, with increase in model params/pretraining FLOPs etc. to support the claim that VDMs can become foundational vision models. 

5. Finally, for the sake of completeness, the authors should also analyse VDMs vs LLMs for sequence reasoning tasks, to give a full picture of the strengths and weaknesses of these large vision/language models. 


Overall the paper tries to do interesting and relevant analysis of VDMs and LLMs but still lacks crucial results and experiments.

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper compares video diffusion models (VDMs) and large language models (LLMs) under a symmetric, frozen-backbone + LoRA protocol: VDMs perform image-to-image prediction by reframing each input–output grid as a short transition video with discrete interpolation and a neutral fixed text embedding, while LLMs do JSON-to-JSON sequence prediction. Evaluation spans ARC-AGI/ConceptARC, visual games, route planning, and cellular automata with sample-efficiency curves; results show that VDMs are often more sample-efficient on spatial/temporal structure, supporting the claim that video pretraining offers a powerful foundation for visual intelligence.

### Strengths
- Novel Hypothesis and Reframing: The paper's core strength is its originality in reframing VDMs as general problem-solvers rather than just generators. The hypothesis that spatiotemporal inductive biases are key to visual intelligence is a significant and insightful contribution.
- Focus on Data Efficiency: The evaluation wisely focuses on skill acquisition efficiency instead of just final SOTA performance. This provides much deeper evidence for the VDM's superior learning properties in low-data regimes.

### Weaknesses
1. Fundamental Asymmetry in Task Representation and Modality: The comparison's fairness is highly questionable due to a core mismatch in task modalities. 
(1)	The LLM must perform a text-to-text translation on JSON-serialized grids , while the VDM performs a direct pixel-to-pixel mapping. These two representations have fundamentally different information densities, processing complexities, and inherent difficulties. (For example, given a 5x5 grid structure, VDM needs to process an image of 256x256 pixels, while LLM needs to process 25 numbers.)
(2)	The LLM faces a "dual burden" of mastering a complex JSON syntax in addition to the task's core logic. Could the LLM's poor data efficiency be a result of this syntactic and representational overhead, rather than a true failure of its inductive bias for logic?
2. Limited Task Scope and Ecological Validity: The paper makes strong claims about visual intelligence and visual foundation models , but the evaluation is confined to a curated set of synthetic, grid-based tasks with explicit human-defined rules (e.g., ARC, Sudoku, Mazes). This success on toy problems, which are highly amenable to grid-based serialization, does not guarantee the VDM's advantage will generalize to the ambiguity, noise, and implicit physical rules of real-world perception. 
3. Low Absolute Performance: Despite relative efficiency gains, the VDM's low absolute accuracy on key abstract tasks (like ARC-AGI) suggests it also has fundamental limitations in abstract generalization. This low ceiling challenges the claim that this bias is a sufficient solution.
4. Un-ablated Text Embedding: The VDM is conditioned on a "neutral fixed text embedding" ($e_{text}$). Without an ablation study, it is unclear if this provides a crucial task hint, acting as an unfair advantage for the VDM over the LLM, which received no such meta-prompt.

Open to increasing my score, provided my concerns are addressed.

### Questions
Please refer to the questions mentioned in 'weaknesses' section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors investigate whether VDMs can serve as a foundation for visual intelligence, like how LLM did for NLP. They use spatiotemporal pretraining to enhance generalization and data efficiency and provide experimental results. They also curated 3 synthetic benchmarks to test those. For controlled experiments, both were adapted with LoRA fine-tuning while keeping backbones frozen. Each model receives tasks in its natural modality.

### Strengths
1. The motivation of this work is good; the questions the authors raised deserve to have a work to study them. 
2. The curated tasks of interest are interesting; they designed synthetic tasks to test them
3. They show some results that pretraining on VDM modality specific tasks would improve its downstream performances.

### Weaknesses
1. The synthetic tasks are too simplified to be indicative to downstream or other tasks performance. If the authors can show some downstream application enhancements, even just 1 example, then it would be more convincing.
2. The authors compare LLM and VDM, which are two different architectures. There may be transformer-based video LLMs available, such as VideoPoet and VAR, among others. I am sure there are also open-sourced alternatives that are more suitable for these comparisons.
3. The ablations of experiments can be more extensive to be convincing, e.g., add 1 or 2 more families of LLMs and VDMs, add different sizes of those LLMs/VDMs, or, since the authors use LoRA, maybe there can be one more config, etc.

### Questions
See weaknesses. Also, maybe the authors can compare different family or Visual foundation models, e.g., VDMs vs transformer-based VLMs. I wonder how those results would fare?

### Soundness
2

### Presentation
2

### Contribution
2
