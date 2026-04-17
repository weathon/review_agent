# Bridging Explainability and Embeddings: BEE Aware of Spuriousness

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Current methods for detecting spurious correlations rely on data splits or error patterns, leaving many harmful shortcuts invisible when counterexamples are absent. We introduce BEE (Bridging Explainability and Embeddings), a framework that shifts the focus from model predictions to the weight space and embedding geometry underlying decisions. By analyzing how fine-tuning perturbs pretrained representations, BEE uncovers spurious correlations that remain hidden from conventional evaluation pipelines. We use linear probing as a transparent diagnostic lens, revealing spurious features that not only persist after full fine-tuning but also transfer across diverse state-of-the-art models. Our experiments cover numerous datasets and domains: vision (Waterbirds, CelebA, ImageNet-1k), language (CivilComments, MIMIC-CXR medical notes), and multiple embedding families (CLIP, CLIP-DataComp.XL, mGTE, BLIP2, SigLIP2). 
BEE consistently exposes spurious correlations: from concepts that slash the ImageNet accuracy by up to 95\%, to clinical shortcuts in MIMIC-CXR notes that induce dangerous false negatives. Together, these results position BEE as a general and principled tool for diagnosing spurious correlations in weight space, enabling principled dataset auditing and more trustworthy foundation models. The source code is publicly available at https://github.com/bit-ml/bee.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes BEE, a weight-space diagnostic to identify class-specifc spurious correlations learned during fine-tuning. Specifically, it can be divivided into two steps: (1) Initialize linear probe with zero-shot class embeddings from a foundation model and finetune the probe; (2) Rank class-neutral concepts by their similartiy in terms of class weights and thresholding to select SCs. The method is evauate across different modalities including vision (e.g., ImageNet) and language (e.g., CivilComments) with embeddings from different foundation models. It shows strong generalization ability across different tasks with spurious correlation settings.

### Strengths
1. The author proposes a novel method with clear motivation. Rather than error- or data-driven SC doscovery, them operate the measurement in weight space. This makes spurious concept can be dicsovered from the geometry of embedding space, which is simple and aligned with human intuition.

2. The experiments are evaluated on broad scope with multi-modal validation, demonstrating strong generalization. The pipeline is also simple and easy-to-understand, which can improve its practical usage.

### Weaknesses
1. The concept extraction, LLM filtering and dynamic thresholding the spurious concepts are heuristic and heavily relied on the other foundation model which can potentially also suffer from spurious correlation.

2. The gemoetric view is novel, but the paper lacks theoretical analysis about when weight drift towards concept embeddings necessarily indicates spuriousness versus legitimate subclass structure.

3. The defnition of class-neutral is ambiguous. The filtering relies on LLM heuristics. It's possible to assign the attributes relevant to class context (but not spurious) incorrectly as spurious concept. It will be better to discuss it with precison/recall of the class-neutral label.

### Questions
1. Could you provide the details of the computation overhead for the spurious correlation detection?

2. Is there any broader impact of detecting such spurious correlation apart from classification? Will it affect downstream tasks?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces BEE (Bridging Explainability and Embeddings), a novel framework for detecting spurious correlations in fine-tuned foundation models without requiring counterexamples or group annotations. 

The method uses a transparent linear probing setup, identifies class-neutral concepts in the embedding space, and ranks them based on their alignment with classifier weights.

Experiments cover a wide range of modalities (vision, language, healthcare) and embedding models (CLIP, BLIP2, mGTE, etc.).

### Strengths
- a conceptually elegant and model-agnostic approach to uncover spurious correlations from classifier weights, which is both simple and broadly applicable.

- tightly aligned with foundation models: BEE leverages their shared embedding spaces to analyze both classifier weights and textual concepts. This makes BEE natively compatible with large pre-trained models, including CLIP, BLIP2, mGTE, and others. 

- strong empirical results on diverse tasks and datasets.

- the methods provides explicit concept-level explanations for model decisions, thus good interpretability.

### Weaknesses
- BEE operates entirely within the embedding space of large foundation models such as CLIP or mGTE. If the embedding model itself has already encoded biased or spurious associations, BEE may merely surface these existing biases, rather than revealing new or independent shortcuts learned during downstream training. This raises the question of whether BEE is diagnosing the fine-tuning process or simply interpreting the biases already present in the frozen embeddings.

- BEE relies on linear probing to identify correlations between classifier weights and concept embeddings. As such, it is inherently limited to capturing linear or near-linear relationships. What will happen if BEE is applied on more complex spurious patterns? 

- BEE does not offer any mechanism for diagnosing or correcting the biases that may exist in the foundation model’s embedding space. This is a critical limitation, as the root causes of many spurious correlations may lie within the embedding model itself.

### Questions
NA

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
3

### Summary
This manuscript introduces BEE (Bridging Explainability and Embeddings), a framework designed to diagnose spurious correlations learned during linear probing of frozen model. It uses linear probing to reveal spurious features that persist after full fine-tuning and transfer across diverse SOTA models. BEE is evaluated on vision (Waterbirds, CelebA, ImageNet-1k), language (CivilComments) and medical notes (MIMIC-CXR); and across embedding families (CLIP, CLIP-DataComp.XL, mGTE, BLIP2, SigLIP2). BEE surfaces concepts that can slash ImageNet accuracy by up to 95% and exposes clinical shortcuts in MIMIC-CXR dataset that cause false negatives.

### Strengths
1. Core idea is clear: both the linear head and concepts are in the same embedding space. Direct comparison of learned weights and concept embeddings constitutes a direct probe into the decision mechanisms of the classifier.
2. Efficient as it eliminates the need for expensive backbone retraining or the construction of elaborate counterexample data splits.
3. Empirical validation across multiple modalities / datasets and different foundational embedding families (e.g., CLIP and BLIP-2).
4. ImageNet-1k generative experiment is good evidence: prompting generator with BEE surfaced concepts leads to large drops in accuracy and even to class swaps.

### Weaknesses
1. Missing implementation details for reproducibility about the construction of the concept pool (prompts used with Llama-3.1-8B-Instruct, the exact filtering and de-duplication rules applied with WordNet and the final vocabulary size for each dataset).
2. BEE has a variable number of concept prompts per class. No details on whether the baselines like B2T / SpLiCE were given an equivalent prompt budget - difficult to ascertain if the observed gains are because of the higher quality of the BEE selected concepts or because of a larger query allowance.
3. Meaning of reversed diagonal reference, specification of the smoothing window and the exact roles & sensitivity of the r and p are not clearly defined. Sensitivity or ablation study is needed to confirm the stability of the concept selection process.
4. Missing empirical results / experiments scoring for negatively correlated concepts introduced in the paper.
5. Suggest adding multi-seed results with CIs for all tables.

### Questions
1. Could you please share epochs, batch sizes, learning rate, temperatures (for CLIP and non CLIP encoders), and hardware / compute used for each experiment?
2. Do you actually use negative SCs anywhere? If yes, where and how are they incorporated? If not, please explain why and provide an example showing whether negatives are useful.
3. Could you please provide the full procedure for building C_all?

### Soundness
3

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
4

### Summary
The paper introduces BEE (Bridging Explainability and Embeddings), a framework for identifying spurious correlations by analyzing model weight space and embedding geometry rather than relying solely on data splits or prediction errors. BEE examines how fine-tuning perturbs pretrained representations and uses linear probing as a transparent diagnostic tool to reveal hidden shortcuts. It demonstrates that many spurious features persist even after fine-tuning and can transfer across diverse state-of-the-art models and domains, including vision and language tasks. Experiments show that BEE effectively uncovers severe and transferable spurious correlations, establishing it as a principled framework for model auditing and trustworthiness.

### Strengths
1. The authors showcase a wide range of use cases, including zero-shot classification, discovering spurious correlations within datasets, and applications to both text-based and image-based datasets.

### Weaknesses
1. The proposed method closely resembles the concepts of Label-free CBM [1] and Post-hoc CBM [2], both of which also utilize CLIP-based image /text encoders. In particular, Post-hoc CBM (see Table 10 in its Appendix) demonstrates a similar approach to identify biased concepts residing in the dataset. While the authors could emphasize the spurious concept discovery component as their main contribution, the process of using LLMs and captioning models to enumerate and filter potential concepts appears similar to LLaVA-CBM [3]. The paper should more clearly articulate its unique contribution and novelty relative to these existing works.

2. What would happen if the encoder component were fine-tuned rather than kept frozen?

3. The Method section requires more detail. Some crucial explanations are deferred to Section 3, making it difficult to follow the approach before reaching the experiments. Including a concrete example using a well-known dataset could substantially improve clarity.

4. Section 4.5 (Experiment Setup) is difficult to understand. The authors should provide a clearer description, particularly regarding the setup and implementation details of GroupDRO.

5. Since the performance of the image captioning model appears to play a crucial role, it would be helpful to show example outputs and explain how LLMs are used (e.g., prompts, output format, etc.). Have the authors experimented with different captioning or language models, and if so, how do the results vary?

6. **Writing:** Some sentences are overly long and contain excessive commas, which makes them difficult to read. 

[1] Label-Free Concept Bottleneck Models, ICLR 2023

[2] Post-hoc Concept Bottleneck Models, ICLR 2023

[3] Constructing Concept-based Models to Mitigate Spurious Correlations with Minimal Human Effort, ECCV 2024.

### Questions
1. Section 4.5 (Experiment Setup) is unclear and requires further clarification. What is the exact setup for GroupDRO? Did you use the same CLIP-based encoder and train only the classifier? Were all worst-group samples removed from the original training set. And if so, what is the rationale behind this choice? In that case, labeling the baseline as “GroupDRO” may not be appropriate, as the setup differs from the original formulation. The authors should provide a detailed justification for this experimental design.

2. In Table 5, the improvement on the CelebA dataset appears noticeably smaller compared to other datasets. Could the authors explain the reason behind this limited gain?

### Soundness
2

### Presentation
2

### Contribution
2
