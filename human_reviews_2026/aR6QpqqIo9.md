# T2I-ConBench: Text-to-Image Benchmark for Continual Post-training

- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Continual post‑training adapts a single text‑to‑image diffusion model to learn new tasks without incurring the cost of separate models, but naïve post-training causes forgetting of pretrained knowledge and undermines zero‑shot compositionality. We observe that the absence of a standardized evaluation protocol hampers related research for continual post‑training. To address this, we introduce **T2I‑ConBench**, a unified benchmark for continual post-training of text-to-image models. T2I-ConBench focuses on two practical scenarios, *item customization* and *domain enhancement*, and analyzes four dimensions: (1) retention of generality, (2) target-task performance, (3) catastrophic forgetting, and (4) cross-task generalization. It combines automated metrics, human‑preference modeling, and vision‑language QA for comprehensive assessment. We benchmark ten representative methods across three realistic task sequences and find that no approach excels on all fronts. Even joint “oracle” training does not succeed for every task, and cross-task generalization remains unsolved.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces T2I-ConBench, the first unified benchmark for continual post-training of text-to-image diffusion models, addressing the lack of standardized evaluation in this emerging area. The benchmark focuses on two realistic scenarios—item customization and domain enhancement—and evaluates models across four key dimensions: generality retention, target-task performance, catastrophic forgetting, and cross-task generalization. It integrates automated metrics, human-preference modeling, and vision-language QA for comprehensive assessment. Extensive experiments on ten representative methods reveal that no existing approach achieves consistent performance across all dimensions, highlighting the need for more robust continual post-training strategies.

### Strengths
1. This paper focuses on the impact of continual post-training algorithms themselves on text-to-image generation performance, eliminating the influence of model architecture on performance.
2. A comprehensive evaluation pipeline facilitates future research on continual text-to-image generation. 
3. The paper evaluates multiple existing methods and thoroughly analyzes experimental results, demonstrating that current approaches perform poorly under challenging imbalanced streams.

### Weaknesses
1. "Item Customization and Domain Enhancement differ in granularity and learning objectives, demanding distinct strategies for knowledge updating and retention." is a novel and interesting perspective. Previous work pipelines typically separated specific objectives from overall style. Previous work pipelines typically separated specific objectives from overall style, meaning they only added objectives or only added style. However, subsequent experiments and analyses did not delve deeper into this point.
2. The domains covered in Domain Enhancement are relatively limited. Compared to the example given at the beginning of Section 3 (e.g., portrait photography, wildlife images, or natural landscapes)., the proposed benchmark ultimately covers a more limited domain. I didn't quite grasp the specific meaning of the “Nature” domains.
3. The experimental sequence in Table 2 can be optimized. Items and domains are learned sequentially. If the order is disrupted, for example "item-domain-item-domain", would this pose a greater challenge to the algorithm?
4. What is the difference between "V1 dog" and "V2 dog" in line 142?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel benchmarking framework for evaluating text-to-image diffusion (T2I) models after continuous post-training. Because real-world training requires models to continuously learn new styles, which can lead to catastrophic forgetting, the authors propose a new framework for evaluating continuous learning in T2I models. Evaluation dimensions include forgetting, generality, target task performance, and cross-task generalization.

### Strengths
1. A new and comprehensive post-training evaluation framework for continuous learning diffusion models is necessary.

2. The proposed evaluation aspects are diverse, adding generality retention and cross-task generalization to the forgetting aspect, which is the primary focus of continuous learning, thus meeting current requirements for AI.

### Weaknesses
1. The dataset seems to have some limitations in size and diversity; for example, the tasks only include dogs, cats, and sneakers. This may lead to shortcomings in evaluation.

### Questions
See the Weaknesses above.

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
3

### Summary
Motivated by the observation that the absence of a standardized evaluation protocol hampers related research for continual post-training, this paper presents T2I-ConBench, a comprehensive benchmark for continual post-training of T2I diffusion models. T2IConBench focuses on two scenarios (i.e., item customization and domain enhancement) and analyzes four dimensions. (i.e., (a)retention of generality, (b) target task performance, (c) catastrophic forgetting, and (d) cross-task generalization). Experiments are conducted on PixArt-α and Stable Diffusion v1.4 to benchmark 10 representative methods across three task sequences.

### Strengths
1.	This work develops an automated evaluation pipeline to assess preservation of pretrained generality, target-task performance,  forgetting, and cross-task generalization for continual T2I post-training.
2.	Building upon the proposed T2I-ConBench, this paper evaluates ten representative baseline methods on mixed order streams.

### Weaknesses
1.	In Table 2, it’s surprising that the replay method yields a zero performance for Unique-Sim in Order 2 training. Is there any insight regarding this?
2.	The findings summarized in the work do not offer particularly novel or compelling insights, as most of them were straightforward or previously pointed out.

### Questions
1.	What is the intuition of utilizing Class-Sim as a forget measure?
2.	It’s not clear why “NA” appears for forgetting metrics in Table 1.
3.	For the data replay method, how many samples are replayed for each task in different settings?

### Soundness
3

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
3

### Summary
This paper introduces T2I-ConBench, a benchmark for evaluating continual post-training methods for text-to-image diffusion models. The benchmark comprises two task types, curated datasets, and an automated evaluation pipeline that measures performance in 4 areas. The authors evaluate 10 baseline methods across three sequential task settings and find that no single method excels across all dimensions, joint training does not always succeed, and cross-task generalization remains challenging.

### Strengths
- Continual post-training is practically important but lacks standardized evaluation. This benchmark provides much-needed infrastructure for fair comparison and reproducible research.
- The four-dimensional assessment is thorough and well-motivated. This holistic view goes beyond typical continual learning benchmarks.
- Testing whether models can compose concepts from different tasks (e.g., "astronaut riding horse" after learning "astronaut" and "horse riding" separately) is creative and important for understanding true knowledge integration versus memorization.
- The key takeaways (no universal winner, joint training limitations, cross-task generalization challenges) are interesting, well-supported, and challenge common assumptions in the field.
- Investigating how task order affects learning and generalization (Order 1 vs Order 2) provides practical insights for deployment scenarios.

### Weaknesses
- The benchmark includes only 4 items and 2 domains, which is insufficient for evaluating continual learning at scale. Real-world scenarios involve many more tasks. Expanding to at least 8-10 items and 4-5 domains would significantly strengthen the benchmark's validity and challenge methods more realistically. The task types are also limited to only customization and enhancement, other important scenarios like style transfer or concept editing are missing.
- Domain enhancement relies entirely on FLUX generated synthetic data, inheriting its biases and limitations. The 96% rejection rate is concerning and inadequately explained. What specific quality issues caused such massive filtering? The "meticulous manual screening" lacks transparency: no annotation guidelines, inter-rater reliability measures, or screening criteria are provided. This undermines the validity of domain enhancement experiments.
- No human evaluation is conducted to validate automated metrics. How well does Unique-Sim correlate with human judgments of identity preservation? Does HPS truly capture domain-specific improvements? Does the VQA pipeline accurately assess cross-task generalization? Without human validation, we cannot be confident these metrics measure what they claim to measure.
- No error bars, confidence intervals, or multiple runs with different seeds are reported. This makes it impossible to assess whether observed differences between methods are statistically significant or due to random variation.
- Reliance on a single VLM (Qwen2.5-7B) introduces potential systematic biases. The binary yes/no scoring for cross-task generalization seems coarse (partial successes cannot be captured). The LLM-based decomposition of prompts into questions could introduce errors. Authors could consider comparing multiple VLMs or using more fine-grained scoring.
- The paper acknowledges omitting recent continual learning methods but doesn't justify these specific choices well. Why not include more recent parameter-efficient methods like adapters or prompt tuning?
- Only PixArt-$\alpha$ and SD1.4 are tested. Missing evaluation on SDXL, SD3, or other state-of-the-art architectures limits generalizability of findings.
- How were "challenging concepts" identified? The description suggests prompting the base model and manually checking, which this is subjective. What were the specific criteria? What inter-annotator agreement was achieved? An improvement in rigor would be beneficial.


The paper makes a valuable contribution by introducing the first unified benchmark for continual post-training of T2I models with a novel cross-task generalization evaluation, providing useful infrastructure and insights. However, significant limitations prevent a stronger recommendation and require substantial improvements before serving as a definitive standard.

### Questions
Main concerns and questions were raised in the weaknesses section; human evaluation, 96% rejection rate, statistical significance.

- Have authors validated that the LLM correctly decomposes prompts into comprehensive questions? Can they compare results using different VLMs to assess robustness?
- The cross-task tests seem to only involve pairwise combinations. Have you considered more complex compositional scenarios involving 3 or more tasks?
- What are the computational requirements for the benchmark? How long does full evaluation take, and what resources are needed?

### Soundness
2

### Presentation
3

### Contribution
3
