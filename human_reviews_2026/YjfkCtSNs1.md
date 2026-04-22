# Stitch: Training-Free Position Control in Multimodal Diffusion Transformers

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Text-to-Image (T2I) generation models have advanced rapidly in recent years, but accurately capturing spatial relationships like “above” or “to the right of” poses a persistent challenge. Earlier methods improved spatial relationship following with external position control. However, as architectures evolved to enhance image quality, these techniques became incompatible with modern models. We propose Stitch, a training-free method for incorporating external position control into Multi-Modal Diffusion Transformers (MMDiT) via automatically-generated bounding boxes. Stitch produces images that are both spatially accurate and visually appealing by generating individual objects within designated bounding boxes and seamlessly stitching them together. We find that targeted attention heads capture the information necessary to isolate and cut out individual objects mid-generation, without needing to fully complete the image. We evaluate Stitch on PosEval, our benchmark for position-based T2I generation. Featuring five new tasks that extend the concept of Position beyond the basic GenEval task, PosEval demonstrates that even top models still have significant room for improvement in position-based generation. Tested on Qwen-Image, FLUX, and SD3.5, Stitch consistently enhances base models, even improving FLUX by 218% on GenEval's Position task and by 206% on PosEval. Stitch achieves state-of-the-art results with Qwen-Image on PosEval, improving over previous models by 54%, all accomplished while integrating position control into leading models training-free.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes STITCH, a training-free positional control method for MMDiT T2I models. An external MLLM decomposes the prompt and predicts bounding boxes. The model first constrains attention inside each box (“Region Binding”), then uses one attention head to extract object latents (“Cutout”), stitches them with the background, and finishes generation.

### Strengths
- Works with off-the-shelf MMDiT/flow models and needs no finetuning.
- Clear mechanism (box-guided masking + attention-based Cutout).
- PosEval adds harder position tasks.

### Weaknesses
- Not end-to-end. It depends on an external MLLM for layout planning; the core model does not learn position.
- Novelty is modest. Bounding-box guidance, stitching, and attention-head probing are well-trodden ideas; the paper mainly adapts them to MMDiT.
- Sensitivity and brittleness are underexplored. Results may hinge on box quality, the chosen attention head, S steps, and η thresholds. Automatic head selection at test time is unclear.
- The approach adds test-time cost (generate per object for S steps), and the blend/accuracy trade-off suggests practical tuning overhead.
- Competent engineering with a useful benchmark, but the method is not end-to-end and the main ideas are incremental. PosEval has value, yet it does not offset the limited novelty of the core approach.

### Questions
What are the time/memory costs per image vs. base models and other training-free layout methods

### Soundness
3

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
The authors address the problem of position control in Flow Matching Text-to-Image (T2I) models. They propose a new benchmark, POSEVAL, to evaluate incorrect object placement, extending current benchmarks that only assess a narrow aspect of the problem (e.g., only two items, no negations). The authors also propose a method to mitigate this issue. Their approach uses a Multimodal Large Language Model (MLLM) to decompose the full prompt into sub-prompts and corresponding bounding boxes. It then applies block attention within these regions to ensure correct item placement and refines the bounding boxes using specific attention heads found to be responsible for focusing on foreground objects to make a cuttoff and merge it into the final image. Finally, the authors provide quantitive evidence that their method is effective at improving positioning compared to vanilla T2I models anda few other baselines.

### Strengths
The authors address a genuine and important problem, providing both a new benchmark for evaluation and a new method to solve this problem.

### Weaknesses
The main weakness of the paper is that the authors overlook previous works that have proposed similar methods to solve this problem, such as [1, 2]. I would expect the authors to include these methods in their experiments and discuss how their proposed method differs from them.

[1] Region-aware text-to-image generation via hard binding and soft refinement (Chen et al.)

[2] Training-free regional prompting for diffusion transformers (Chen et al.)

### Questions
1. Why are you using an MLLM for the first step? Is there any use for the multimodality of the output? 
2. Which MLLM (L) was used in your experiments?

Additional Suggestions:
1. You discussed how you selected the attention heads responsible for positioning foreground items by using 80 images with the same prompt template. I find this discovery interesting, but it is not well-evaluated. After choosing the attention heads, I would expect an evaluation on an additional, different set of prompts and seeds to ensure that these heads generalize in their role.

2. Your introduction cites only one work (Ghosh et al., 2023). Please make sure to discuss previous works in this section, explain how your method differs, clarify why previous methods could not be used for this problem, and add the appropriate citations.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes STITCH, a training-free, test-time method to improve spatial relations in modern MMDiT/flow T2I models, and introduces PosEval, a harder position-focused benchmark extending GenEval  (3–4 objects, negative/relative relations, positional attribute binding).

### Strengths
1. Clear, practical contribution: a training-free procedure that plugs into state-of-the-art MMDiT/flow models and measurably improves spatial adherence.
2. Well-motivated attention masking (Region Binding): three simple constraints that isolate intra-box generation and reduce positional leakage across regions.
3. New benchmark adds breadth and difficulty: PosEval targets real failure modes (negatives, relative relations, scaling to 3/4 objects, attribute-position binding) and demonstrates that “Position is solved” is premature.

### Weaknesses
1. Occlusion/overlap and complex layouts: Bounding boxes are non-overlapping on a coarse grid. Many real prompts imply occlusions or partial overlaps; it’s unclear how well Cutout and Region Binding handle tight interactions or interpenetrating objects.
2. Head selection practicality: Selecting a single “good” attention head per base model uses a small calibration process (80 prompts with SAM-based IoU/IoT). While not training, it’s extra pre-deployment effort.

### Questions
1. How sensitive is Stitch to the quality of the LLM-generated bounding boxes and sub-prompts? Do you have failure analyses where LLMs produce impractical layouts or inconsistent object parsing?

2. Is there an automated, model-agnostic way to choose the Cutout head at test time without using SAM-based ground-truth masks? Could you use self-consistency or internal signals to pick heads dynamically?

3. How stable is the chosen head across prompts, resolutions, and seeds? Does a single head suffice for all object categories, or do some categories benefit from different heads?

4. Can Stitch handle overlapping or occluded objects (e.g., partial overlaps of boxes)? If not, what would be required to support this (e.g., layered depth ordering, soft box masks, or iterative compositing)?

### Soundness
3

### Presentation
3

### Contribution
3
