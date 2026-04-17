# Zebra-CoT: A Dataset for Interleaved Vision-Language Reasoning

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Humans often rely on visual aids, such as diagrams or sketches, when tackling complex problems. Teaching multimodal models to adopt similar strategies, a process known as Visual Chain of Thought (visual CoT), is much more difficult. The main challenges are: (1) weak performance of off-the-shelf visual CoT, which hinders reinforcement learning, and (2) the lack of high-quality visual CoT training data. We introduce **Zebra-CoT** a diverse large-scale interleaved text-image reasoning dataset with 182,384 reasoning traces across 18 domains with over 50 distinct tasks.
This dataset is specifically designed to train models to natively perform visual CoT. 
We emphasize four categories of tasks where sketching or visual reasoning is especially natural, spanning (a) *scientific questions* such as geometry, physics, and algorithms; 
(b) *2D visual reasoning tasks* like visual search and jigsaw puzzles; 
(c) *3D reasoning tasks* including 3D multi-hop inference, embodied and robot planning; 
and (d) *visual logic problems and strategic games* like chess. 
Fine-tuning Anole‑7B model on Zebra-CoT yields a +12\% improvement in our test‑set accuracy and up to +13\% performance gains on standard VLM benchmarks. 
Similarly, fine-tuning Bagel‑7B produces models capable of generating high-quality interleaved visual reasoning chains, underscoring Zebra-CoT's effectiveness in advancing multimodal reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a visual chain-of-thought (CoT) collection methods that can provide visual CoT trajectories with strong correlation between the text and the images. They use the methods to collect 182k trajectories across 18 domains. They authors also fine-tune two models to support its usefulness of their collected dataset, where the fine-tuned models shows better visual CoT capacity or even get interleaved visual-text reasoning ability from scratch.

### Strengths
1. The dataset focus on an important problem: how to collect visual chain-of-thought (CoT) data. Their collected dataset provide valuable resources for training better visual reasoning models.
2. The dataset is large-scale and diverse. It contains 182k traces across 18 domains.
3. The authors also finetune two models to support the usefulness of the dataset.

### Weaknesses
1. In the data curation process, the images are first prepared and them the text reasoning traces are generated based on the images. It means that the reasoning steps must be pre-decided and cannot easily modified and extended. For these synthetic problems, it also means it heavily depends on the human prior knowledge (especially on some authors manual design?)

### Questions
1. I am not sure whether the analysis in section 4 can support the claim the visual CoT is important. In the experiments, the 1MT and 2MT provides additional finished reasoning steps for the models. It should significantly reduce the difficulty of the reasoning tasks. So the performance gain may come from the provided finished reasoning steps, instead of the visual information itself. I guess it should be compared to a baseline where the test-reasoning steps are also provided as text-only CoT.
2. I believe a static dataset is useful but not enough because the VLM or LLM with both visual understanding and generation ability are rapidly developing. In my opinion, a better way is to collect data in a dynamic way. For example, the model can generate useful visual CoT by itself and then collect them and finetune on them in an iterative way. Do you believe your data collection method can be extended to such dynamic way?
3. How to create the visual CoT is an important problem. I believe the images in your section 3 should also contain some information that how to create them. The figure 5 is not directly related to the specific tasks.
4. A minor suggestion: There are too many related works in section 2 and it is hard to align them to the table 1 because in your table 1, the dataset use their abbreviated name while in section 2, you use the authors' name. I believe it is better to also provide the dataset name in section 2 for better alignment. And I also wonder whether all mentioned datasets are included in Table 1? If not, I suggest to provide a more complete table in the appendix.

### Soundness
4

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
This paper introduces Zebra-CoT, a large-scale, multi-domain interleaved text-image reasoning dataset designed to address the core issue of scarce high-quality visual chain-of-thought training data. The dataset covers four major categories of natural tasks particularly suited for visual reasoning: scientific problems, 2D/3D visual reasoning, and logical strategy tasks. Models fine-tuned on this dataset achieved significant performance improvements.

### Strengths
Scale and Diversity: The dataset is extensive (over 180K reasoning traces) and covers a wide range of domains (18 domains, 50+ distinct tasks), providing rich and comprehensive learning material for visual reasoning.

Relevance and Naturalness: The dataset is specifically constructed for task types where visual aids naturally enhance reasoning, ensuring high relevance and learning efficiency.

Empirically Validated Effectiveness: Experiments on multiple mainstream models demonstrate that the dataset significantly improves visual reasoning accuracy and the ability to generate high-quality visual reasoning chains.

The main contribution is the training dataset for the community.

### Weaknesses
1. After fine-tuning, the model demonstrates improved overall performance. However, to pinpoint which specific subcategories contribute to its gains on benchmarks like MathVision, MathVista, VisuLogic, EMMA, MMVP, Blink, and Vstar, a more detailed analysis is required due to the limited baseline comparisons currently available in these areas.

2. A detailed comparative analysis with prior benchmarks should be provided in the main paper, quantitatively examining aspects such as dataset scale and evaluation dimensions.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces ZEBRA-CoT, a large-scale dataset of 182K interleaved text–image reasoning traces designed to teach models to perform Visual Chain-of-Thought (visual CoT) reasoning. It contains 18 domains across four categories, e.g., scientific reasoning, 2D and 3D visual reasoning, and visual logic games.
Experiments show that frontier multimodal models (including GPT-5, Claude-4 Sonnet, Gemini 2.5 Pro) perform poorly on ZEBRA-CoT , but their performance improves when given the first one or two multimodal reasoning steps as scaffolding.
Overall, ZEBRA-CoT provides the first established dataset enabling models to think with images, representing a significant step toward interpretable and general multimodal reasoning.

### Strengths
- The paper addresses an important research question to enable models to reason visually rather than purely through text. It proposes a novel curated dataset (ZEBRA-CoT) that directly facilitates this capability beyond simple captioning or visual QA.

- The four categories (scientific, 2D, 3D, strategic) demonstrate careful task design and strong diversity, enabling generalization across reasoning types rather than a single narrow task.

-  The scaffolding experiments convincingly show that models benefit from intermediate multimodal reasoning cues, quantifying the value of explicit visual CoT supervision.

- Fine-tuning both Anole-7B and Bagel-7B demonstrates the dataset’s practicality and shows real gains on external benchmarks such as MathVista, VisuLogic, and MMVP.

- Establishing visual CoT as a training paradigm parallels the importance of textual CoT in LLM reasoning, with potential long-term implications for interpretable and grounded AI.

### Weaknesses
- While diversity is emphasized, there is little quantitative or human evaluation of logical coherence, correctness, or visual–text alignment within reasoning traces. Including human consistency checks or inter-annotator agreement would strengthen credibility.

- The paper does not isolate how different types of visual reasoning steps (e.g., sketches vs. crops vs. bounding boxes) contribute to performance. A finer-grained analysis would clarify which modalities drive improvement.

- Many examples rely on synthetic visual reasoning (mazes, jigsaws, diagrams).  Demonstrations on real-world, noisy visual data would better illustrate applicability.

- The limitation of the proposed dataset is not explicitly discussed. Elaborated discussions on the limitations or possible improvements would enhance the clarity of the paper.

### Questions
- How is visual coherence between text and generated sketches quantitatively verified?
- How does model scale affect the emergence of visual CoT?
- What are the major limitations of the proposed dataset? Is it ensured that it does not contain any noise or mislabeled examples?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new vision-language reasoning dataset Zebra. It contains 182k reasoning chains featured with interleaved visual and textual reasoning steps. Experiments on two VLM backbones show substaintial performance gains after fine-tuning on the proposed Zebra dataset.

### Strengths
+ It proposes a high-quality dataset featured with interleaved visual reasoning chains on multiple domains, fill in the gap of previous visual cot datasets
+ The pilot scalffording experiments and the fine-tuning experiment shows that the dataset could improve downstream visual reasoning performance.
+ The paper is well written and easy to follow.

### Weaknesses
+ Some details of the dataset curation process is missing. From Appendix A.2 we know that the dataset contains both real-world data and synthetic data, but the author does not reveal the detailed source of the real-world data or the specific process of how the synthetic data is generatd. This is important because we don't know if the evaluation dataset like MathVista or MathVision is contaminated or not. Testing the high-similarity overlap between the proposed dataset and the evaluation benchmark would make the performance gains more convincing.

+ I would suggest that In section 5 the fine-tuning experiment to include fine-tuning on previous CoT dataset for a more direct comaprision of your data quality with previous ones.

### Questions
See the weaknesses above.

### Soundness
4

### Presentation
4

### Contribution
4
