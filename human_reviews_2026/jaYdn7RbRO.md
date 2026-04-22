# PhysToolBench: Benchmarking Physical Tool Understanding for MLLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
The ability to use, understand, and create tools is a hallmark of human intelligence, enabling sophisticated interaction with the physical world. For any general-purpose intelligent agent to achieve true versatility, it must also master these fundamental skills. While modern Multimodal Large Language Models (MLLMs) leverage their extensive common knowledge for high-level planning in embodied AI and in downstream Vision-Language-Action (VLA) models, the extent of their true understanding of physical tools remains unquantified. To bridge this gap, we present PhysToolBench, the first benchmark dedicated to evaluating the comprehension of physical tools by MLLMs.
Our benchmark is structured as a Visual Question Answering (VQA) dataset comprising over 1,000 image-text pairs. It assesses capabilities across three distinct difficulty levels: 1) Tool Recognition: Requiring the recognition of a tool's primary function. 2) Tool Understanding: Testing the ability to grasp the underlying principles of a tool's operation. 3) Tool Creation: Challenging the model to fashion a new tool from surrounding objects when conventional options are unavailable. Our comprehensive evaluation of 32 MLLMs—spanning proprietary, open-source, specialized embodied, and backbones in VLAs—reveals a significant deficiency in the tool understanding. Furthermore, we provide an in-depth analysis and propose preliminary solutions. Code and dataset are publicly available at https://github.com/PhysToolBench/PhysToolBench.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces *PhysToolBench*, a visual question answering (VQA) benchmark designed to assess multimodal LLMs’ understanding of physical tools across realistic, resource-constrained scenes. Each example presents a task description and a 1024×1024 image with numerically labeled objects; models must select the applicable tools or answer None. The benchmark spans four domains (Daily Life, Industrial, Outdoor, Professional) and three difficulty tiers: Easy (tool recognition), Medium with subtracks named M1 (attribute understanding), M2 (tool combination), M3 (availability/viability), and Hard (creative tool creation/substitution). Model evaluation shows a wide gap to human performance and highlights especially poor robustness on availability and long-tail categories. The paper further proposes a vision-centric reasoning (VCR) agent that decomposes reasoning via object detection and localized analysis to improve M3 scores.

### Strengths
1. The paper is well written and easy to follow.
2. The benchmark shifts tool use evaluation from generic affordance Q&A toward task-conditioned tool selection under availability constraints, which better mirrors embodied settings.
3. A broad model suite (32 MLLMs) across proprietary, open-source, embodied-specific, and VLA-backbones enables nuanced comparisons.

### Weaknesses
1. Around 90% of images are generated (GPT-4o-image) rather than captured in real homes/worksites. While curated, this may under-represent clutter, occlusion, wear, and domain shift typical of robotics. It would help to quantify the 10% real photos’ distribution and to report performance gaps between synthetic vs. real subsets. Since text prompts sometimes hinge on brand/model specifics, generated scenes risk subtle stylistic cues that models could exploit. A leakage/stylization analysis would further strengthen the claims.
2. For M2 Tool Combination, success may depend on understanding hidden state not visually disambiguated. To clarify on how such requirements are visually conveyed and how answer correctness is adjudicated would help.
3. The proposed VCR pipeline is only evaluated on M3 with two backbones (GPT-4o/5). It’s promising, but without broader coverage (M1/M2/Hard), latency/runtime overheads, or robustness checks (e.g., detector errors), it’s hard to gauge its generality or practicality in agents.
4. The approach relies on an object detector (DINOX) and crops. But the performance may hinge on detector recall for small/occluded tools. The paper should further quantify detector metrics and report end-to-end sensitivity to detection failure.

### Questions
1. Could you report per-subset results (synthetic vs. photographed), to help readers better assess the benchmark's real-world transfer?
2. How do you treat cases with multiple acceptable substitutes or partially adequate tools?
3. Beyond evaluation, have you explored training with VCR-style intermediate supervision (crop-level rationales, attribute labels, or damage present tags)? A small tool-centric visual-reasoning dataset might turn your agent into a method contribution, not only a benchmark.
4. Typo: The LaTeX quotes are incorrect—use the backtick ` for the left quote, not the curly apostrophe '.

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
4

### Summary
This paper introduces PhysToolBench, the benchmark designed to evaluate how well Multimodal Large Language Models (MLLMs) comprehend physical tools. Structured as a Visual Question Answering (VQA) dataset with over 1,000 image-text pairs , its primary contribution is a three-level evaluation framework assessing Tool Recognition (basic function) , Tool Understanding (attributes, combination, and availability) , and Tool Creation (flexible usage). By testing 32 MLLMs, the authors demonstrate a significant deficiency in current models , finding that even the best score no higher than 63% (compared to human performance over 90%) and revealing critical weaknesses, especially a failure to recognize when tools are damaged or non-functional. The paper also provides an in-depth analysis of these failings and proposes a "vision-centric reasoning" framework as a preliminary solution.

### Strengths
- Definition of a new problem: While many benchmarks test LLMs on digital tools (like API calls) or MLLMs on simple affordance recognition (i.e., "what is this for?") , this paper is the first to create a dedicated benchmark to evaluate the practical, deep understanding of physical tools.
- The experimental quality is a major strength. The authors conducted a massive evaluation of 32 MLLMs.
- The paper is well written and easy to follow.

### Weaknesses
- The benchmark's design, while original, has limitations in what it truly evaluates. Its core weakness is the potential confusion between deep physical reasoning and visual perception combined with "common sense" world knowledge.
  - This is most evident in the "Availability Understanding" (M.3) sub-task. The paper highlights models' failures on this task as a critical finding. However, this task primarily tests the model's ability to perceive visual damage (e.g., "cracked," "damaged," "empty") and associate it with a learned fact ("cracked things are unusable"). This does not necessarily validate that the model understands the underlying physical principles of why the tool fails (e.g., why a plunger requires an intact seal to create a pressure differential). The benchmark's scope is thus limited in its ability to truly test the comprehension of physics.
  - The "Tool Creation" (Lv.3) level, while innovative, is constrained by the VQA format, which implies a single correct answer. In reality, "tool creation" (or more accurately, tool substitution) is an open-ended problem. For a task like "tighten a screw," a coin might be the ground truth, but a nearby metal ruler, a butter knife, or a different flat-edged object could also be valid solutions. The paper does not address how the benchmark handles this evaluation ambiguity or credits multiple correct, creative solutions.
  - The provided images, while high-quality, appear to be heavily staged and simplified. The paper notes that items are numerically labeled and are the "only available things". This is a significant simplification of a true "open-world" scenario. A real environment (e.g., a kitchen, a workshop) is cluttered, and the correct tool may be partially occluded, surrounded by many similar-looking distractors, or require searching (e.g., in a drawer).
- The paper's contribution, while significant, is centered almost entirely on the introduction of a new dataset and the resulting analysis, with less emphasis on a novel methodological advancement. The paper's scope and structure align more with the goals of a benchmark track than the main conferences

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

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
The paper proposed a benchmark for evaluation MLLMs' performance on understanding tool using tasks. This benchmark is designed to metric the MLLM's ability in three hierarchical difficulty levels. The experimental results indicates that this benchmark reveals several bottleneck of current MLLMs in tool understanding, such as model scale, backbone, vision v.s. text, etc. Then the author proposed a new method to help improving the tool understanding ability for MLLMs by more emphasizing the vision information.

### Strengths
1. The paper claimed the first benchmark work for physical tool understanding.
2. The paper is clearly narrated. The tables and figures are clear to read.
3. The experiment is conducted based on a large group of current MLLMs, provides convincing conclusion.

### Weaknesses
1. Verbal polish: e.g. task of "open a TV" should be "turn on a TV" in Figure 6. Also, incorrect usage of quotation mark throughout the paper.
2. While the experiments evaluated the performance of MLLMs in different difficulty levels designed by the authors. However, to demonstrate that the metric dimensions are sufficiently unique and efficient, the author should also report the MSE of each metric dimension, so that the reader can understand if this dimension is able to largely distinguish different MLLMs.
3. What would be the result if MLLMs are inferencing visually similar but different tool materials? For example, there are 2 forks on the table, one is made of medal, and another is made of plastic. The task is to pick some heavy food material, so the ground truth could only be the medal fork. I assume sometimes the MLLMs might make incorrect decision.

### Questions
See weaknesses

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a benchmark to evaluate the ability of VLMs and VLAs to understand physical tools.

### Strengths
The paper conducts a thorough evaluation across a wide range of baselines and designs multiple difficulty levels in its benchmark for more detailed analysis.

### Weaknesses
The paper is primarily focused on benchmarking, making it more suitable for a benchmark-oriented venue rather than ICLR.

### Questions
1. Is GPT achieving the highest score because the images in the benchmark were generated by GPT itself

### Soundness
3

### Presentation
3

### Contribution
2
