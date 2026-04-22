# ChainMPQ: Interleaved Text-Image Reasoning Chains for Mitigating Relation Hallucinations

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
While Large Vision-Language Models (LVLMs) achieve strong performance in multimodal tasks, hallucinations continue to affect their reliability. Among the three categories of hallucinations, which include object, attribute, and relation, relation hallucinations account for the largest proportion but have received the least attention. To address this challenge, we propose ChainMPQ (Multi-Perspective Questions guided Interleaved Text-image Reasoning Chain), a training-free method that improves relational inference in LVLMs by utilizing accumulated textual and visual memories. ChainMPQ first extracts subject and object keywords from the question to enhance the corresponding image regions. It then constructs multi-perspective questions that focus on the three core components of a relationship: the subject, the object, and the relation that links them. These questions are sequentially input to the model, with textual and visual memories from earlier steps providing supporting context for subsequent ones, thereby forming an interleaved chain of image and text that guides progressive relational reasoning. Experiments on multiple LVLMs and benchmarks show that ChainMPQ substantially reduces relation hallucinations, while ablation studies further validate the effectiveness of its three core modules.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces ChainMPQ, a novel method designed to mitigate relation hallucinations in large vision-language models (LVLMs). Relation hallucinations, a type of error in multimodal reasoning, occur when the model fails to accurately infer relationships between recognized objects. The proposed method works by first extracting subject and object keywords, enhancing the visual tokens associated with these entities, and then constructing multiple interleaved questions focusing on different components of a relationship. These questions guide the model step-by-step, leveraging both textual and visual memory to progressively refine relational inference. The paper demonstrates the effectiveness of ChainMPQ through experiments on multiple LVLMs and benchmarks, showing significant improvements in reducing relation hallucinations.

### Strengths
1. The decomposition of relational reasoning into multiple, focused questions is a unique approach. By systematically addressing each component of a relationship (subject, object, and relation), ChainMPQ offers a more structured and interpretable reasoning process compared to previous methods.

2. The method does not require additional training or fine-tuning, making it a practical solution for mitigating hallucinations in pre-trained models. This is a key advantage in real-world applications where retraining large models may not be feasible.

3. The experiments show consistent improvements across different LVLMs (LLaVA-1.5 and InstructBLIP) and benchmarks (MMRel and R-Bench), demonstrating that ChainMPQ is effective in various contexts.

4. The method’s success across different architectures further validates its generalizability and robustness.

### Weaknesses
1. There are other training-free methods[1,2] that adjust attention maps to enhance MLLM performance. These works should be cited and discussed in comparison to ChainMPQ to provide a broader context for the proposed method.

2. R-Bench contains both image-level and instance-level setups. Which setup is used for the experiment in Table 1? If the results in Table 1 are based on the image-level setup, can this method also perform well in the instance-level setup?

3. The current baselines only include LLaVA-1.5 and InstructBLIP. Given the rapid advancements in LVLMs, it would be beneficial to include newer models, such as Qwen2.5-VL[3] and InternVL3.5[4], to demonstrate the method’s effectiveness on state-of-the-art models.

[1] Controlmllm: Training-free visual prompt learning for multimodal large language models

[2] MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs

[3] Qwen2.5-VL Technical Report

[4] Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency

### Questions
See weakness.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The goal of this work is to tackle relational hallucination in LVLMs which is under explored in previous literature in comparison to object and attribute based hallucinations. The proposed method is called ChainMPQ, which is essentially a training free pipeline that converts subject and object keywords from the prompt into a series of questions about the subject, object, and their relation, and uses these to guide cross-attention via an interleaved chain-of-thought mechanism. The goal is to sharpen attention on relation-relevant regions and reduce errors without additional training.

The main findings of the paper show that across two different model families, ChainMPQ is able to reduce relational hallucination on two benchmarks that measure relation hallucinations. More ablation studies are provided to show which types of the questions contribute to the overall improvement of the final results. Their main finding here states that multi-perspective questions and the interleaved chain are the biggest contributors to performance gain.

### Strengths
- Interesting application of interleaved chain-of-thought prompting to reduce relational hallucinations. The staged prompts around subject, object, and relation give the model clear subgoals and make the intervention easy to understand.
- Results clearly demonstrate the benefit of this approach on relational hallucination benchmarks. The ablations help isolate where gains come from, e.g. multi-stage enhancement of visual tokens seems to help the most.
- Training free and low cost to deploy, which makes the method practical for settings where fine-tuning is not feasible.

### Weaknesses
- Organization and detailing of content can be improved a lot. For instance, implementation details of question generation from the text prompt are not included in the main text. Similarly, the baselines are not described in detail (trained/training-free, how does their approach differ from ChainMPQ, etc). In the ablations section, the description of ablations was not very clear to me (e.g. "five constructed questions with a single relationship question" -- how is this constructed?, "we remove the multimodal chain and only use the text context from the previous answer." -- is this the entire multi-turn text context or just the last turn answer?). The paper would benefit by including these details/examples in the main/appendix sections.
- I think the multi perspective question generation aspect of this work is interesting, but currently lacks a general framework for the benefit of future practitioners who might want to extend this work to new benchmarks or new domains. Some discussion or a framework around that can help enhance the generality of this work. 
- Contributions seem siloed around relational hallucinations so the novelty seems limited. The work can be improved by showing how this module complements more general hallucination mitigation strategies.

### Questions
- Can the authors elaborate on the questions raised above on distinguishing details about the ablations: (i) replaces five constructed question with a single question, and (ii) removal of multimodal chains that was raised in the Weaknesses section?
- The case study section says "we present two real examples from MMRel, namely an action case and a spatial case, which together account for over 90% of the MMRel dataset" -- are there any experiments to check the diversity of cases in the two datasets beyond action and spatial? My worry is that this question construction approach is more geared towards the templates most suitable for action spatial domains.
- Can the authors highlight how the exact differences between ChainMPQ and the baselines compared (Constraint-Aware Prompting and Detect-then-Calibrate) and the rationale behind choosing these over other related works?

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
The paper proposes ChainMPQ, a training-free method designed to reduce relation-hallucinations in LVLMs. To do so, they first extract subject and object keywords from a given question,  then generate multi-perspective follow-up questions focusing on subject, object, and the relation.  Experiments across multiple existing LVLMs reportedly show a reduction in relation-hallucination rates, with ablation studies proving the importance of different modules.

### Strengths
1. The proposed training-free hallucination mitigation method is useful and could be used to mitigate the hallucination issues.
2. The motivation of this paper is clear, along with detailed method descriptions and experiments.

### Weaknesses
1. My main concern on this paper is the tested baseline models, which are a little it outdated. It will be better if the authors could add more advanced LVLMs in the experiment, which will largely increase the reliability of the results.

2. The evaluation seems to focus mainly on reduction of relation-hallucination rates, but less so on downstream multi-modal task performance (e.g., does improved relation reasoning lead to better general reasoning or application outcomes).

3. Missing related work on relation hallucination of LVLMs: Wu J, Chung T T, Chen K, et al. Unified triplet-level hallucination evaluation for large vision-language models[J]. arXiv preprint arXiv:2410.23114, 2024.

### Questions
Please see my reviews above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Hallucinations have hindered LVLMs’ reliability, which are categorized into three aspects. The paper focuses on relation hallucinations in LVLMs in which models correctly recognize subjects and objects but misidentify relationship between them. The proposed ChainMPQ (Multi-Perspective Questions guided Interleaved Chain of Image and Text) is a training-free, multi-step, interleaved text–image reasoning framework. It extracts subject/object keywords from the input question to enhance corresponding relation-relevant image regions via cross-attention. Furthermore, it constructs a set of multi-perspective aware text prompt, where sub-questions are raised centered on subject, object, and relation through entity localization and masking-based queries. Then, it builds an interleaved text-image reasoning chain where both textual and visual information are taken into considerations, where text encoder introduces question tokens while context is accumulated using answers from precious steps. From the third sub-question onward, it extracts attention over keyword tokens from late decoder layers, selects top‑k visual tokens via an entropy‑based rule, and applies a confidence‑weighted visual bias to steer subsequent steps; these biases are aggregated across rounds, and the original relation question is finally answered using the enhanced visual tokens and the accumulated textual/visual memories. Ablations indicate that multi-perspective question construction and the interleaved visual memory are the main contributors to the method's effectiveness. As a training‑free, model-agnostic plug-in, the approach can contribute to strengthen relational grounding in LVLMs, and by highlighting a simple decomposition-plus-memory transfer design that others can build on. It may further motivate work on causality-based attribution mechanism and more robust visual representations to reduce relation hallucinations in real applications.

### Strengths
	Innovative Methodology: The core concept of ChainMPQ—decomposing a relationship into subject, object, and relation components and constructing a multi-perspective, interleaved reasoning chain—is creative and well-grounded. The integration of both textual memory (previous answers) and visual memory (attenuation maps) is a key differentiator from text-only prompting methods.
	Strong and Extensive Evaluation: The experimental design is robust. Evaluating on two state-of-the-art LVLMs (LLaVA-1.5 and InstructBLIP) and two dedicated relation-focused benchmarks (MMRel and R-Bench) provides convincing evidence of the method's effectiveness and generalizability.
	Comprehensive Ablation Studies: The ablation study is a major strength. It clearly demonstrates the individual contribution of each of the three core modules (Attention Enhancement, Multi-Perspective Questions, Interleaved Chain), showing that they work synergistically. The sensitivity analysis of hyperparameters k_max and λ adds rigor.
	Clear and Illustrative Case Studies: The qualitative examples are excellent. They effectively translate the complex methodology into an intuitive narrative, showing precisely how the step-by-step process corrects the model's reasoning and attention, which greatly enhances the paper's persuasiveness.

### Weaknesses
	Computational Overhead and Practicality: The most significant weakness is the substantial increase in inference time (∼4x). While the authors rightly note that the method is "training-free" and can process thousands of samples in a few hours, this overhead limits its practicality for real-time applications. A discussion on potential optimizations or a trade-off analysis between accuracy and latency would be beneficial.
	Baseline Comparability and Reproducibility: The paper states that baseline results (e.g., for Prompting Wu et al. and Calibrate Zheng et al.) were obtained by "re-implementing their experiments." It is crucial to clarify if these re-implementations were performed with the authors' best effort or if they were validated against the original papers' results. Slight differences in implementation (e.g., prompt wording, calibration parameters) could affect the fairness of the comparison. Providing the exact prompts used for all methods would improve reproducibility.
	Limited Analysis of Failure Cases: The paper shows where ChainMPQ succeeds but provides little insight into where it fails. A brief analysis of typical failure modes (e.g., errors in the initial keyword extraction, cases where the attention bias propagates errors) would provide a more balanced view and offer valuable directions for future work.
	Clarity on "Visual Memory" Propagation: While the concept of propagating attention maps is clear, the description of how these maps from different steps are combined and applied (Eq. 6) could be more precise. Specifically, it would be helpful to clarify if the masks M_i from all previous steps (3, 4...) are applied with equal weight α_j or if there's a decay mechanism.

### Questions
This paper presents ChainMPQ, a novel, training-free framework designed to mitigate relation hallucinations in Large Vision-Language Models (LVLMs). The method is built on a compelling core idea: decomposing relational reasoning into a sequence of interleaved text and image steps, inspired by human-like reasoning and extending the Interleaved Modal Chain-of-Thought (ICoT) concept. The work is timely, addressing a significant and under-explored problem (relation hallucinations) with a method that demonstrates clear and consistent improvements across multiple models and benchmarks. However, some revisions are needed to enhance the readability and clarity for the reader.
	The use of cross-attention with subject/object keywords to enhance visual tokens is a solid foundation. It would be interesting to know if the performance gain from this module is more pronounced for images with cluttered backgrounds where object localization is inherently harder.
	The masking strategy for generating questions is excellent. However, the dependency on the spaCy NLP toolkit for keyword extraction is a potential point of failure, especially for ambiguous or complex subjects/objects. A quick error analysis of the extraction step would strengthen this section.
	The adaptive top-k selection based on attention entropy is a nice touch. The description of the chain's operation is generally clear, but as noted in the weaknesses, the exact mechanics of combining multiple previous visual biases (Eq. 6) could be elaborated. The assumption that confidence in the textual answer correlates with the reliability of the visual attention map is reasonable but could be discussed briefly.
	While the text mentions 4x inference time overhead, the table contains no efficiency metrics (inference time, computational cost, memory usage). This omission hides the practical trade-off between accuracy gains and computational cost.
	The "Vanilla" baseline represents standard model performance, but there's no comparison to other relevant baselines like standard Chain-of-Thought prompting The methods selected for comparison may not represent the current state-of-the-art in relation hallucination mitigation
	No t‑tests, Wilcoxon tests, or other statistical comparisons are reported; these would confirm that observed gains are not random.
This is a strong paper with a novel contribution, solid experimental validation, and clear presentation. The weaknesses, primarily related to computational overhead and baseline reproducibility, are manageable and do not undermine the core contributions. The paper is likely to be of significant interest to the ICLR community.

### Soundness
3

### Presentation
3

### Contribution
3
