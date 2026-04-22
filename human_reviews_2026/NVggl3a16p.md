# HiPOOD: Hierarchical Prompt-Aware Zero-Shot Out-of-Distribution Detection

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Reliable image recognition systems must not only classify known categories accurately but also detect instances of novel, unseen classes in open-set scenarios. Achieving this in a zero-shot setting—without any training examples—remains a significant challenge. In this paper, we propose a zero-shot out-of-distribution (OOD) detection approach that leverages semantic class hierarchies to enrich each known label with fine-grained subcategory sets, capturing subsumption relationships between classes. To generate these hierarchies, we query a large language model (LLM) with structured prompts, producing semantically coherent candidate subcategories that are subsequently filtered with a lexical ontology to ensure domain alignment. We incorporate the resulting label hierarchy into CLIP’s classification pipeline, a pre-trained vision–language model (VLM). This design enables the model to distinguish fine-grained categories within the known classes and to recognize when an input does not fit any known class—effectively identifying it as an unknown object. Notably, our approach operates in a zero-shot manner, requiring no additional training. Experiments on several standard OOD detection benchmarks show that our method achieves state-of-the-art performance. Furthermore, by organizing predictions within a semantic hierarchy, the model’s outputs become more informative and easier to interpret, including for inputs that it flags as unknown.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes HIPOOD, a method for zero-shot out-of-distribution (OOD) detection that leverages hierarchical structures of in-distribution (ID) labels. Specifically, the approach uses a large language model (LLM) to generate fine-grained subcategories for each ID label, thereby creating a semantic hierarchy. By utilizing both the ID labels and the newly derived fine-grained labels, the model achieves superior detection performance compared to existing methods.

### Strengths
- S1. The problem setting is interesting. While previous studies have used LLMs to generate OOD labels, this paper instead utilizes them to subdivide ID labels, which represents a novel perspective.

- S2. The explanations throughout the paper are clear, and the distinction from existing work is well articulated, making the contribution easy to understand.

- S3. The paper conducts a large number of experiments, and the ablation studies.

### Weaknesses
-  W1.In the results section, all comparison methods use the original ImageNet labels, while the proposed method uses a different set of labels. It would be better to align the labels across methods to ensure a fair comparison.

- W2.Within OOD detection, hard OOD detection has recently become an important focus, yet this paper does not report results for that setting. Since performance on common OOD detection is already quite high in general, it would be important to include and discuss hard OOD detection results as well [1]. It would be better to include the results with NINCO [2] and SSB-Hard [3].

-  W3. For the evaluation of ImageNet variants, the NegLabel method performs better. Therefore, the effectiveness of the proposed method is not fully demonstrated. It would be beneficial to examine whether combining the proposed approach with NegLabel improves performance.

-  W4. The method appears to be sensitive to hyperparameters, particularly to the relationship between depth and M. The optimal value of M seems to vary depending on the depth, and performance changes significantly with different M values. This sensitivity may make the method less practical or difficult to use in real-world scenarios.

[1] Noda+, A Benchmark and Evaluation for Real-World Out-of-Distribution Detection Using Vision-Language Models, ICIP2025

[2] Bitterwolf+, In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation, ICML2023

[3] Vaze+, Open-Set Recognition: a Good Closed-Set Classifier is All You Need?, ICLR2022

### Questions
- I would like to see the performance of the comparison methods reported when they use the same coarse labels as the proposed method.

- It would also be valuable to include results on hard OOD detection.
 
- When the domain differs, the NegLabel method performs better, which suggests that the effectiveness of the proposed approach is not fully demonstrated. It may be beneficial to explore combining the proposed method with NegLabel to further improve performance.

- Regarding the sensitivity to hyperparameters, it would strengthen the paper if there were a clear and convincing justification for why such sensitivity is acceptable.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes HiPOOD, a zero-shot OOD detection method that augments CLIP with LLM-generated hierarchical subclass labels. It constructs per-class fine-grained label sets via prompting and vocabulary filtering, then computes a coarse-to-fine consistency score to flag OOD samples. The method claims SOTA zero-shot OOD performance on standard benchmarks and improved interpretability.

### Strengths
- The method is zero-shot and training-free, requiring no ID images, fine-tuning, or auxiliary classifiers—only pre-trained CLIP, one-time LLM queries per class, and a static WordNet vocabulary—enabling instant deployment on resource-constrained devices.

- By targeting naturally hierarchical label spaces, HiPOOD achieves superior sensitivity to near-OOD anomalies via parent-conditional reweighting, correctly flagging fine-grained unknowns (e.g., wolf under "dog") where coarse confidence misleads MSP/MCM, leveraging VLM calibration and avoiding exhaustive negative label coverage.

### Weaknesses
- Novelty is limited as the core hierarchical framework—parent-conditional softmax, superclass reweighting, coarse-to-fine consistency—is a near-direct extension of CHiLS, with HiPOOD’s contribution reduced to LLM-based subclass regeneration plus percentile filtering, an engineering tweak rather than a conceptual advance; given ImageNet-1K’s existing WordNet hierarchy, regenerating it via GPT-4 may add noise without justification, failing to address any fundamental prior limitation.

- Heavy, unstable reliance on closed-source LLMs (GPT-4) undermines reproducibility and robustness, with no ablation across open-source models, prompt variations, or failure modes (e.g., generic/irrelevant outputs like "pet"), risking entire subclass tree corruption and performance collapse without controlled external dependencies.

- Computational and deployment overhead is drastically underestimated—for ImageNet-1K (N=1000, M=10), offline phase requires ~1,000 costly LLM calls plus potentially millions of CLIP text encodings over a ~80,000-noun WordNet vocabulary, while online inference demands ~11,000 cosine similarities and 1,000 softmaxes per image, orders of magnitude slower than MSP/energy scoring, with zero discussion of latency, cost, or scalability trade-offs.

- Evaluation is severely overstated and misleadingly narrow, claiming applicability to real-world hierarchical domains like medical imaging (SNOMED/UMLS) while testing exclusively on ImageNet-1K—a dataset with a pre-existing, clean WordNet hierarchy that creates implicit data leakage and an artificially favorable setting; the paper presents zero results on non-hierarchical datasets (CIFAR-10/100), domain-specific data (medical/satellite/e-commerce), and omits critical ablations on LLM quality, prompt design, percentile threshold, or vocabulary source.

### Questions
If the ID category itself is fine-grained, how can authors generate subcategories to accommodate the proposed method? Additionally, certain categories may not have subcategories, such as traffic lights.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a zero-shot out-of-distribution detection method named HiPOOD, which enhances the representation of known categories by constructing a semantic class hierarchy and uses the CLIP model to conduct coarse-grained and fine-grained consistency evaluation at the hierarchy, thereby identifying OOD samples.

### Strengths
1. The paper is well written.
2. Experiments are adequate, verifying the advantage of the proposed HiPOOD.

### Weaknesses
1. The authors argue that "There will always be unseen unknowns that are not represented by any pre-collected negative label". However, such a issue also exists when using hierarchy, where unseen unknowns could always appear out of the admitted hierarchy.
2. Is the hierarchy consistency on the semantic relation kept the same as the CLIP-based embedding similarity? And corarse-level classification may not be that accurate for CLIP-based prompt, since the pre-training data would pair the images with class names from various granularity.
3. The idea of introducing hierarchy into OOD detection has been widely used in related methods, leading to the method lack of novelty.
4. Comparison to a Stronger "Hierarchical" Baseline: While compared to many flat-label methods, the paper could be strengthened by a direct comparison to a simpler hierarchical baseline.
5. The performance hinges on the quality of the generated subclasses. The ablation shows that a predefined ontology can outperform the LLM-generated hierarchy. This raises questions about the method's robustness across diverse domains, especially where high-quality lexical resources like WordNet are unavailable.

### Questions
See the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a closed-loop optimization strategy that suppresses harmful frame relevance scores through universal perturbation refinement, effectively reducing their selection during sampling.
Additionally, the authors formulate a zero-shot OOD detection score, HiPOOD, which leverages hierarchical representations by comparing image–text alignment at both coarse and fine levels.

### Strengths
The method is simple yet effective, and the overall design is conceptually reasonable.

The paper is clearly written and provides theoretical justification for the formulation. The description of class-level cleaning and hierarchy construction is also well-explained.

### Weaknesses
It would be interesting to investigate whether introducing learnable prompts could further improve performance compared to fixed prompts.

As the capabilities of foundation models (e.g., CLIP) continue to improve—particularly their domain invariance and zero-shot generalization—it would be valuable to evaluate the method on harder tasks or larger datasets, such as cross-domain ImageNet variants.

The paper would benefit from stronger baselines for comparison, especially including recent SOTA methods such as CATEX [1].


Reference
[1] Category-Extensible Out-of-Distribution Detection via Hierarchical Context Descriptions.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
