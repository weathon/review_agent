# Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Document parsing from scanned images into structured formats remains a significant challenge due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Existing supervised fine-tuning methods often struggle to generalize across diverse document types, leading to poor performance, particularly on out-of-distribution data. This issue is further exacerbated by the limited availability of high-quality training data for layout-aware parsing tasks. To address these challenges, we introduce layoutRL, a reinforcement learning framework that optimizes layout understanding through composite rewards integrating normalized edit distance, paragraph count accuracy, and reading order preservation. To support this training, we construct the Infinity-Doc-400K dataset, which we use to train Infinity-Parser, a vision-language model demonstrating robust generalization across various domains. Extensive evaluations on benchmarks including OmniDocBench, olmOCR-Bench, PubTabNet, and FinTabNet show that Infinity-Parser consistently achieves state-of-the-art performance across a broad range of document types, languages, and structural complexities, substantially outperforming both specialized document parsing systems and general-purpose vision-language models. We will release our code, dataset, and model to facilitate reproducible research in document parsing.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces LayoutRL, a novel reinforcement learning framework for end-to-end parsing of scanned documents. To support reinforcement learning training, the authors also construct Infinity-Doc-400K, a large-scale dataset of ~400K scanned document pages paired with ground-truth Markdown parses. A vision-language model is trained is trained with LayoutRL on this data to yield Infinity-Parser, an end-to-end document parser.

### Strengths
1. The authors introduce Infinity-Doc-400K, a very large dataset. combining synthetic and pseudo-labeled real documents.
2. Infinity-Parser achieves state-of-the-art performance on multiple benchmarks.

### Weaknesses
1. The novelty of the paper is limited. GRPO optimization is already a part of any VLM architecture. The authors provide a task-specific loss, which we typically do during the retraining of a VLM on a particular task. It shouldn't be considered as the novelty of the work

2. In Table 2, the text edit and formula edit scores for English are lower than the Chinese ones. This suggests the training data of the Infinity parser is biased to the Chinese language.

3. The papers lack ablation. It only demonstrates how reinforcement learning is beneficial. It would be interesting to show the choice of VLM trained on Infinity-DOC-400K and the combination of loss functions. 

4. As reported in Table 9 in the supplementary material, the method doesn't perform well on financial reports, academic papers, but performs well on notes and newspapers. However, the former ones have more structural representations than the latter ones. Then, how does the layout information help to generate structured machine translations?

5. In the introduction, it mentions the no. of documents is 400,482, then in the method, it says 400,066, and in the supplementary Figure 7, it says infinity doc 55K. I think the author should be consistent with the numbers.

### Questions
1. Why was only a 43K subset of the 400K dataset used for RL? Were the remaining documents reserved for testing/val, or was it a computational limitation? How might performance change if more data were used? Although the dataset comprises ~400,000 pages, the RL fine-tuning reportedly uses only a 43,000-page subset. It is unclear why the other data were not utilized and whether further gains could be achieved with full-scale RL training.

2. The three reward components are simply summed. Did you tune or validate these weights? Would different weights (or learned weights) affect the trade-off between local accuracy vs global structure?

### Soundness
2

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
4

### Summary
This paper introduces LayoutRL, a novel reinforcement learning framework designed to improve the parsing of scanned documents by making models explicitly layout aware. To achieve this, authors developed a multi-aspect reward model that evaluates content accuracy, paragraph segmentation, and reading order, moving beyond simple token-level supervision. Supporting this framework is the new, large-scale Infinity-Doc-400K dataset, which combines diverse synthetic and real-world documents. Infinity-Parser leverages this training to set a new state-of-the-art performance on various benchmarks, outperforming both specialized pipelines and general-purpose VLMs in accurately capturing complex document structures.

### Strengths
The paper introduces LayoutRL, an original and effective use of reinforcement learning with a multi-aspect reward model to explicitly teach models structural document layout, overcoming the generalization limits of standard fine-tuning. 
It establishes a new state-of-the-art through exhaustive experiments across four diverse benchmarks, outperforming both specialized pipelines and general-purpose VLMs, thereby substantiating its claims with robust empirical evidence. 
By creating and releasing the large-scale Infinity-Doc-400K dataset and the high-performance Infinity-Parser model, the work provides great resources that will accelerate future research and establish a new standard for the document AI community.

### Weaknesses
The paper does not investigate the relative importance of its different reward components (edit distances/paragraph counts/reading order) or adequately explain the negative interaction between SFT and RL, which does not provide deeper insights about why this framework works.

### Questions
If I’m correct, this paper only compares existing VLM baselines without fine-tuning them on the 400K Doc data. Including results after fine-tuning on different VLMs could better show the dataset’s significance.

### Soundness
3

### Presentation
3

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
The paper proposes LayoutRL, an end-to-end reinforcement learning framework for layout-aware scanned document parsing. The authors build Infinity-Doc-400K, a large dataset combining synthetic pages and real-world pages. Trained on this data, Infinity-Parser achieves SOTA results across OmniDocBench, olmOCR-Bench, PubTabNet, and FinTabNet, with stronger OOD generalization and training stability than SFT baselines.

### Strengths
1. The paper presents LayoutRL, a complete reinforcement learning framework for document parsing with three well-designed reward functions that explicitly enhance the model’s understanding of document layout. The method is conceptually sound and experimentally comprehensive, covering multiple benchmarks and settings.

2. The authors introduce Infinity-Doc-400K, a large-scale 400K-document dataset built through multi-model collaborative annotation and template-based synthesis. This dataset provides a valuable and scalable foundation for future reinforcement learning research in layout-aware document understanding.

### Weaknesses
Weakness 1 – Limited analysis of experimental results
While the experiments are extensive, the analysis remains largely descriptive, focusing on trends and metrics rather than underlying causes. For instance, the claimed advantages of RL in stability and generalization are not sufficiently supported by detailed reasoning or ablation-based explanation.

Weakness 2 – Missing appendix
The paper repeatedly refers readers to an appendix for experimental settings and benchmark details, but no appendix is provided. This omission reduces the overall professionalism and completeness of the submission.

Weakness 3 – Insufficient experimental details
Key experimental configurations are missing, including training/testing splits, hyperparameters, and baseline evaluation dates. The lack of transparency limits reproducibility and makes it difficult to fully assess the reported results.

### Questions
1. The data construction section states that the authors built a 400K-document dataset, but the ablation studies repeatedly refer to experiments using 43K samples. Could the authors clarify how these two dataset scales are related and why only 43K samples were used for ablation?

2. Is the edit-distance reward computed over the entire document, or at a finer granularity (e.g., paragraph level)? If it is document-level, what is the specific motivation and necessity of the other two rewards (count and order)? If not, please provide a clearer description of the calculation scope.

3. In Table 5, the difference between English and Chinese edit distances varies dramatically between the second and third rows, while the average edit distance across document types remains almost unchanged; the opposite trend appears between the fourth and sixth rows. What factors contribute to these contrasting behaviors?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes LayoutRL, a layout-aware RL framework for end-to-end scanned document parsing. It trains a VLM to output the final structured document and scores it with three automatic rewards — edit distance, paragraph-count consistency, and reading-order preservation — so the model learns structure, not just tokens. A new 400K document dataset (Infinity-Doc-400K) is built to make this RL feasible.

### Strengths
1.Motivation is clear and well grounded (SFT struggles on page-level / OOD structure).

2.Writing is clear and figures are intuitive, so the method is easy to follow.

### Weaknesses
1.The methodological novelty is somewhat concentrated on constructing a large, structurally aligned corpus and on formulating task-specific, verifiable rewards; the RL component itself follows existing group-relative / rule-based RLFT paradigms and does not introduce a fundamentally new optimization mechanism.

2.A key experiment is missing, namely training Qwen2.5-VL-7B on the proposed Infinity-Doc-400K dataset without applying the RL stage, in order to explicitly verify how much of the performance gain should be attributed to reinforcement learning.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2
