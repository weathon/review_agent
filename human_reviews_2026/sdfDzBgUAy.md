# M$^{3}$T2IBench: A Large-Scale Multi-Category, Multi-Instance, Multi-Relation Text-to-Image Benchmark

- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Text-to-image models are known to struggle with generating images that perfectly align with textual prompts. Several previous studies have focused on evaluating image-text alignment in text-to-image generation. However, these evaluations either address overly simple scenarios, especially overlooking the difficulty of prompts with multiple different instances belonging to the same category, or they introduce metrics that do not correlate well with human evaluation. In this study, we introduce M$^{3}$T2IBench, a large-scale, multi-category, multi-instance, multi-relation along with an object-detection-based evaluation metric, *AlignScore*, which aligns closely with human evaluation. Our findings reveal that current text-to-image models perform poorly on this challenging benchmark. Additionally, we propose the Revise-Then-Enforce approach to enhance image-text alignment. This training-free post-editing method demonstrates improvements in image-text alignment across a broad range of diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the poor alignment of Text-to-Image (T2I) models with complex prompts, particularly those with multiple instances of the same category. It introduces **M³T2IBench**, a large-scale (10,000-sample) benchmark for evaluating multi-category, multi-instance, and multi-relation generation. To evaluate this, the work proposes **AlignScore**, an object-detection-based metric that decomposes performance into instance counting ($Bias$) and attribute/relation accuracy ($Acc$), using an exhaustive search to map prompted and detected objects. Empirically, `AlignScore` achieves a high correlation with human judgment (Pearson $r=0.6711$), and results show current SOTA models perform poorly on this new benchmark. Finally, a training-free post-editing method, **Revise-Then-Enforce**, is proposed, demonstrating consistent alignment improvements across diffusion models.

### Strengths
1. The primary contribution, `M³T2IBench`, is well-motivated and addresses a significant, acknowledged gap in T2I evaluation: the difficulty with prompts containing multiple instances of the *same* category (e.g., "two bowls", "three clocks"), a feature missing from benchmarks like GenEval and T2I-CompBench (Table 1).
2. The proposed `AlignScore` metric demonstrates a significantly higher correlation with human judgment (Pearson $r=0.6711$) than established metrics like VQAScore ($0.4022$) and CLIPScore ($0.2296$). Its design, which separates `Bias` (count) from `Acc` (attribute/relation) and uses an optimal matching search, is methodologically sound for the target problem.
3. The `Revise-Then-Enforce` method is simple, training-free, and shown to be broadly effective, improving all tested open-source models (Table 2). The ablation study (Table 2, SD3 results) effectively justifies its design (Eq. 6) over simpler negative-prompt (Eq. 7) or positive-prompt (Eq. 8) variants.

### Weaknesses
1. The `AlignScore` metric is fundamentally dependent on the capabilities of its component tools: an object detector (Mask2Former) and a color classifier (CLIP). The authors acknowledge a failure mode where the detector double-counts an object (Figure 10), which directly impacts the `Bias` score. The robustness of the metric to such detector failures is not systematically evaluated.
2. The `AlignScore` calculation relies on an exhaustive search (Algorithm 2) to find the best matching $f$ between prompted and detected instances. The paper states this is "efficient," but provides no complexity analysis. This search space grows factorially with the number of instances, which may become computationally infeasible for prompts more complex than those in the benchmark (max 5 instances of the same category).
3. The proposed `Revise-Then-Enforce` method is a post-editing technique, requiring a full generation pass *before* applying the correction. This doubles the inference cost (one pass to find errors, one to correct). This practical limitation is not discussed in the main paper.
4. The benchmark's prompts are generated from rigid templates (Section 3.1), which, while ensuring quality and avoiding LLM subjectivity, limits linguistic diversity. This is acknowledged as a limitation but is a notable tradeoff, as models may overfit to this specific templated sentence structure.

### Questions
1. Regarding the exhaustive search in Algorithm 2 for `AlignScore`: What is the computational cost of this step as the number of instances ($N$) and detected objects ($M$) increases? Could this step be replaced with a more efficient optimal transport or bipartite matching algorithm, and what would be the impact on the final score?
2. The `Revise-Then-Enforce` method (Section 4.2) requires constructing paired prompts $c_1$ (what should be) and $c_2$ (what is wrong). In the case study (Figure 12), $c_2$ is "1 bowl. The second bowl is not white." How sensitive is the method's performance to the precise phrasing of $c_2$? For example, would "1 brown bowl" or simply "1 bowl" have worked as well?
3. The analysis in Figure 9 suggests a "Relation Bias," where models are better at "left" and "above" than "right" and "below". Does this bias correlate with the order of object mentions in the prompt templates? Could this be an artifact of the template-based prompt construction rather than an inherent model bias?
4. The detector's failure case in Figure 10 shows one clock detected twice, which would incorrectly increase the `Bias` score. How prevalent is this duplicate-detection failure in the 10,000-sample benchmark, and does it significantly skew the reported `Bias` results for the models in Table 2?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce M3T2IBench, a text-to-image benchmark featuring multiple categories, instances, and relations, along with AlignScore, a metric that correlates better with human judgment. Their evaluation of various T2I models exposes difficulties in generating images that align well with complex prompts. To address this, they present a training-free "Revise-Then-Enforce" post-editing approach that improves alignment between generated images and text prompts.

### Strengths
1. The paper proposes to evaluate both acc and bias, with an exhaustive searching method to determine the final acc. The idea of evaluating two metrics is reasonable.

2. The paper is well-written and easy to follow.

### Weaknesses
1. Lack of novelty of the proposed revise-then-enforce method. First, the method appears to be hardly related to the proposed benchmark and scoring metrics. This makes the paper separate into two unrelated parts. Additionally, a similar idea has already been explored in earlier works, but the paper fails to mention [1].
2. In Line 201, the benchmark refrains from mentioning words like fresh and majestically to ensure accurate evaluation. This idea does not seem reasonable and therefore severely limits the scope of testing for the benchmark. A holistic benchmark should encompass all the frequently mentioned words by users, rather than focusing on words that are easy to judge and discarding the subjective ones. There are far more attributes that often appear in the prompts, such as moods and shapes. 
3. Besides, the object-centered design seems to limit the comprehensiveness of the benchmark. There are aspects more than objects, like complex relations and styles. 
4. The weakness of QG/A is not very clear. Why can this method not correctly deal with multiple instances belonging to the same category in the prompt?

[1] Wang, Zihao, et al. "Concept algebra for (score-based) text-controlled generative models." Advances in Neural Information Processing Systems 36 (2023): 35331-35349.

### Questions
1. Clarify the connection between the proposed method and [1].

2. What is the relation between the proposed benchmark and method?

3. More clarification and examples of the limit of QG/A methods.

Please also refer to the weakness part.

### Soundness
2

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
This paper introduces M3T2IBench, a large-scale benchmark designed to evaluate text-to-image models on complex compositional prompts. It contains 10,000 structured prompts, which is significantly larger and more diverse than prior benchmarks. The prompts include multiple objects, categories, attributes, and spatial relations, allowing for rigorous evaluation of multi-instance and relational reasoning. The benchmark uses automated metrics to separately measure bias (errors in the number of generated instances) and accuracy (correctness of attributes and relations), relying on object detection and color classification systems. Experiments show that state-of-the-art generators—including Stable Diffusion 3, FLUX, and DALL-E 3—perform poorly, with accuracy below 70% and consistent object count errors. These results highlight the gap between current T2I model capabilities and real compositional understanding.

### Strengths
1. Significant Benchmark Scale and Complexity: It presents largest T2I compositional alignment dataset to date (10k prompts), and supports long prompts, many relations, and multiple instances per category

2. Fine-Grained and Structured Metric Design. The paper evaluates object counts, colors, spatial relations via automated detectors, and distinguishes between bias (count errors) and accuracy (attributes/relations)

### Weaknesses
The benchmark depends heavily on automatic detectors for object, color, and relation evaluation. While this enables scalability, it also inherits all failure cases of those detectors, particularly in stylized or abstract generations. The lack of reported human validation raises questions about the accuracy of the scores and whether false detector errors might inflate failure rates.

Another limitation is the synthetic nature of prompt construction. Prompts are generated using probabilistic rules for relations and attributes, which may not reflect real-world usage patterns. As a result, while the benchmark is rigorous, it may evaluate models on scenarios that are technically difficult but less representative of natural user intent.

The paper also does not deeply analyze failure modes or propose guidance for model improvement beyond brief mentions of an included approach. More detailed discussion or qualitative insight into model breakdowns would strengthen the contribution. Finally, the benchmark over-emphasizes structural correctness, which means it may undervalue creative fidelity or semantic coherence compared to strict compositional rule-following.

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces M3T2IBench, a new large-scale, multi-category, multi-instance, multi-relation benchmark. It focuses on prompts that have many objects, repeated categories, colors, and spatial relations. The benchmark comes with an automatic metric called AlignScore which combines two parts (1) Bias for counting category mistakes and (2) Accuracy for checking if attributes and relations are correct after matching each text object to a visual object. The paper shows that AlignScore correlates better with human judgement than alternatives. It also shows that popular models struggle as prompt complexity increases. The authors propose Revise and Enforce that edits the prompt using what went wrong and pushes generation toward the failed parts without retraining helping improving alignment across several diffusion models.

### Strengths
The paper’s original contribution focuses on real world scenarios which other metrics lack. The prompt construction method is simple, general and easily applicable across diffusion models, and needs no retraining. The metrics are defined clearly and human ratings align better with AlignScore compared to alternatives. The paper is easy to read and the information flows logically. The paper’s contribution lies in coming up with a structured large-scale benchmark making it valuable for real world use cases and the gains from R&E method is consistent and model-agnostic.

### Weaknesses
In the paper, the attributes used are mostly focused on color. Adding additional attributes like shape can help make accuracy apply to real world use cases and not just color and position. DALLE-3 comparison has only 100 prompts which is relatively smaller. Revise and Enforce is assumed to depend on correctly identifying the failed parts. It would be great to explain the method used to identify the failed parts and turn them into prompts.

### Questions
1. How much will the scores change if the detector or color classifier is changed? Any plan to ensemble or calibrate per category?
2. If you change the balance between Bias and Accuracy in AlignScore, do model rankings stay stable relative to human judgments
3. Can you describe in more details about the post editing method from detection errors to two edited prompts?

### Soundness
4

### Presentation
4

### Contribution
3
