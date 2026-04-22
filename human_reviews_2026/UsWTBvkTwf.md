# Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
The widespread adoption of text-to-image (T2I) diffusion models has raised concerns about their potential to generate copyrighted, inappropriate, or sensitive imagery learned from massive training corpora. As a practical solution, *machine unlearning* aims to selectively erase unwanted concepts from a pre-trained model without retraining from scratch. While most existing methods are effective for single-concept unlearning, they often struggle in real-world scenarios that require removing multiple concepts, since extending them to this setting is both non-trivial and problematic, causing significant challenges in unlearning effectiveness, generation quality, and sensitivity to hyperparameters and datasets. In this paper, we take a unique perspective on multi-concept unlearning by leveraging model sparsity and propose the **F**orget **I**t **A**ll (FIA) framework. FIA first introduces *Contrastive Concept Saliency* to quantify each weight connection’s contribution to a target concept. It then identifies *Concept-Sensitive Neurons* by combining temporal and spatial information, ensuring that only neurons consistently responsive to the target concept are selected. Finally, FIA constructs masks from the identified neurons and fuses them into a unified multi-concept mask, where *Concept-Agnostic Neurons* that broadly support general content generation are preserved while concept-specific neurons are pruned to remove the targets. FIA is training-free and requires only minimal hyperparameter tuning for new tasks, thereby promoting a plug-and-play paradigm. Extensive experiments across three distinct unlearning tasks demonstrate that FIA achieves more reliable multi-concept unlearning, improving forgetting effectiveness while maintaining semantic fidelity and image quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Forget-It-All, a training-free framework for multi-concept machine unlearning in text-to-image diffusion models. FIA computes Contrastive Concept Saliency to quantify each weight's contribution to target concepts, identifies Concept-Sensitive Neurons through temporal and spatial aggregation, and constructs masks that preserve Concept-Agnostic Neurons (which respond broadly across concepts) while pruning concept-specific neurons. The method is evaluated on three unlearning tasks (multi-object, explicit content, artistic style) using Stable Diffusion, achieving SOTA forgetting performance, good generation quality, at under 0.3% overall sparsity.

### Strengths
- To my knowledge, the paper presents the first work to explicitly connect model sparsity with multi-concept unlearning through unstructured neuron masking. The introduction of concept-agnostic neurons is a particularly insightful contribution.
- FIA operates training-free, unlike most competing methods. This makes the method efficient and less prone to overfitting or hyperparameter sensitivity.
- The paper includes experiments across distinct unlearning tasks demonstrating consistent state-of-the-art performance. The forgetting accuracy results are particularly impressive (e.g., Table 1, 1.9% on Imagenette).
- The three-component approach (Contrastive Concept Saliency, Concept-Sensitive Neurons, and mask fusion with concept-agnostic neurons) is systematic and well-motivated.
- The paper provides detailed ablations on key design choices including concept-agnostic ratio, pruning target, and pruning granularity (Section C), validating the importance of each component.
- The paper shows the method can scale to unlearning up to 50 concepts (Appendix E), which is important for practical applications.

### Weaknesses
- The paper lacks theoretical connections to related continual learning frameworks, particularly Elastic Weight Consolidation (EWC) [1] and related Fisher Information-based approaches.
- The Contrastive Concept Saliency formulation (Eq. 1) combines weight magnitude, activation norm, and cosine similarity, but the paper doesn't provide theoretical justification for this specific combination or discuss its relationship to established importance metrics in the literature.
- While the paper reports CLIP scores on MS COCO-30K for most experiments, FID scores are not consistently reported across all tasks. Table 1 (multi-object unlearning of 10 concepts) and Table 2 only show CLIP scores, while Table 3 (explicit content) and Table 4 (artistic style) include both FID and CLIP. This inconsistency makes it difficult to fully assess generation quality degradation across different unlearning scenarios.
- The experiments are restricted to Stable Diffusion v1.4 and v1.5, which are relatively older architectures. The paper does not evaluate on more recent models such as Stable Diffusion XL [2], Stable Diffusion 3 [3], or orther transformer-based architectures like FLUX.1 [4] or Sana [5]. This limits the generalizability claims of the approach.
- Section 3.1 introduces concept prompts and base prompts but doesn't specify how many prompts are used for each concept, whether prompts are manually designed or automatically generated, and how sensitive the method is to prompt variations. While Appendix B.1 provides templates, the number of instances and the variance in results across different prompt sets are not reported.
- While Table 17 reports execution time and memory for FIA, similar metrics are not provided for baseline methods, making it difficult to assess the true computational advantage. The paper claims efficiency but doesn't quantify the training time and resources required by fine-tuning-based baselines for fair comparison.
- The paper doesn't adequately discuss when the method fails or performs suboptimally. For instance, in Table 1, some classes like "golf ball" have higher forgetting accuracy (4.8%) compared to others. Understanding why certain concepts are harder to forget would strengthen the paper.

In conclusion, the paper presents a novel and effective approach with strong empirical results and comprehensive experiments. However, due to the lack of theoretical grounding (particularly connections to EWC and continual learning literature) and limited architectural coverage, among the rest of the issues raised, I am inclined to lower my score.


[1] Kirkpatrick, James, et al. "Overcoming catastrophic forgetting in neural networks." Proceedings of the national academy of sciences 114.13 (2017): 3521-3526.
[2] Podell, Dustin, et al. "Sdxl: Improving latent diffusion models for high-resolution image synthesis." arXiv preprint arXiv:2307.01952 (2023).  
[3] Esser, Patrick, et al. "Scaling rectified flow transformers for high-resolution image synthesis." ICML 2024.  
[4] Black Forest Labs, "FLUX", 2024.  
[5] XIE, Enze, et al. "Sana: Efficient high-resolution image synthesis with linear diffusion transformers." ICLR 2025.

### Questions
- What happens if you need to unlearn additional concepts after already applying FIA? Can the method be applied iteratively, or does this require recomputing masks from scratch?
- Under what conditions does FIA fail to adequately unlearn concepts? Are there specific types of concepts or concept combinations that are particularly challenging for your approach?

### Soundness
2

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
2

### Summary
This paper proposes FIA (Forget-It-All): a training-free multi-concept machine anti-learning framework for concept erasure in text-to-image diffusion models. FIA first proposes Contrastive Concept Saliency (CCS) to quantify the contribution of each weight connection to the target concept. Secondly, it combines temporal and spatial information to identify and select consistently responsive concept-sensitive neurons. Finally, it constructs masks from the identified neurons and fuses them into a unified multi-concept mask. While the framework is simple, efficient, and empirically effective, several important issues arise: Lack of Theoretical Grounding, Prompt Sensitivity, Writing Errors and Concept Interference Risk.

### Strengths
Training-free and lightweight: FIA achieves effective multi-concept unlearning without any retraining or fine-tuning, making it computationally efficient and easy to deploy in large-scale text-to-image models.

Unified saliency formulation: The proposed Contrastive Concept Saliency (CCS) provides a principled way to quantify each connection’s contribution to a concept, integrating structural, activation, and similarity information into a single interpretable metric.

Fine-grained neuron control: By combining temporal aggregation and spatial sparsity, FIA identifies concept-sensitive neurons precisely, enabling selective pruning that minimizes collateral forgetting and preserves overall generative capacity.

Scalable multi-concept handling: The multi-concept mask fusion strategy effectively handles multiple simultaneous unlearning targets, avoiding the catastrophic interference and re-learning issues seen in sequential approaches.

### Weaknesses
1) Does CCS have any theoretical basis? 
2) CCS relies on the construction of "concept prompts/base prompts." Is its selection stable if the base scenario/combination grammar varies significantly or is ambiguous? 
3) On page 6, there is a typo: "We present comprehensive ablation results in Section ??." 
4) If the target concepts are highly correlated, can this method avoid cross-contamination?

### Questions
See above

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
The paper introduces Forget-It-All (FIA), a training-free framework for multi-concept unlearning in text-to-image diffusion models. The framework identifies Concept-Sensitive Neurons via Contrastive Concept Saliency (CCS), which measures each neuron’s contribution to target concepts across both temporal and spatial contexts. These neurons are selectively pruned, while Concept-Agnostic Neurons, which contribute to general image generation, are preserved. Experiments on multi-object, explicit content, and artistic style unlearning tasks show that FIA outperforms prior state-of-the-art methods, achieving strong forgetting efficacy with minimal degradation of generative quality.

### Strengths
1. Comprehensive Evaluation: The experiments cover multiple domains: objects, NSFW content, and artistic styles
2. Strong Results: The proposed method maintains both high CLIP and FID scores for concepts to preserve, and strong forgetting scores for target concepts.

### Weaknesses
1. Overstated and Incorrect Claims: The two claimed benefits of this method are (1) that their method requires fewer sensitive hyperparameters compared to existing methods. This is claimed on lines 89 and 137. However, in this paper's appendix, they show three hyperparameters that must be tuned $r_1$ ,$ r_2$, and $\alpha$. In comparison, ConceptPrune[1], which is the most similar work, requires just the sparsity level $k$. If the authors claim their method requires fewer hyperparameters and that their hyperparameters are easier to tune (less sensitive), there should be empirical evidence. The second claimed benefit is that (2) "To the best of our knowledge, this is the first work to explore multi-concept unlearning through unstructured neuron masking" (line 108). This is blatantly incorrect, as ConceptPrune has an explicit section (5.3) on multi-concept removal. I am certain the authors knew of this as on line 137, they explicitly mention "ConceptPrune removes neurons tied to an undesired concept; however, handling multiple concepts is difficult due to complex neuron interactions". It would've been more transparent for the authors to claim their method shows SOTA multi-concept removal via pruning rather than claiming pioneering work. 
2. Method Novelty: As this paper focuses on proposing a new method, I believe it is important to evaluate the novelty of their proposed method.  In particular, I would like the authors to clearly mention the differences in their method with the similar work ConceptPrune. For instance, how is the "Contrastive Concept Saliency" procedure different from ConceptPrune's "Importance Score"? Looking at equation 1, it appears the only difference is adding a cosine similarity score between the input and output, with the motivation being a vague mention of "effectiveness of signal transmission". Additionally, the proposed method isolates "Concept-Sensitive" neurons by subtracting the mean of $U_{l,t,i,j}$ and the standard deviation for the concept and base prompt. The idea of isolation is not unique, as ConceptPrune similarity sets an inequality between the importance score of their target and reference prompt. Without an ablation, it is unclear whether these small changes are actually significant. Alternatively, there is a lack of theoretical or empirical justification for why their modification is better motivated than ConceptPrune's method. Without either ablation or theoretical/empirical justification for additions (including time-integrated sensitivity), it is unclear whether the additions are significant or added just for the sake of novelty. 
3. Evaluation Metric: The metric used for artistic style erasure is the CLIP-score of the target style. Additionally, "Forget-Success Rate" measures whether the CLIP-score decreased compared to the original model. For general quality preservation, they evaluate over MS COCO-30k via FID and CLIP. I strongly suggest that authors show experiments under a standardized benchmark such as UnlearnCanvas[2] for the following reasons. CLIP-based metrics consider redundant factors such as the object content, not just the style. This metric may not isolate style removal compared to a dedicated classifier. Secondly, using MS COCO-30k as the preservation set is limiting. MS COCO-30k is a general dataset; however, we don't only care about whether the model is able to generate general concepts but also preserve concepts close to the erased concepts (e.g other styles). UnlearnCanvas explicitly measures this with its in-domain and cross-domain retention accuracy metric. I think evaluating on a standard benchmark will provide stronger evidence for the FIA's improvement over ConceptPrune.


[1] Chavhan, Ruchika, Da Li, and Timothy Hospedales. "Conceptprune: Concept editing in diffusion models via skilled neuron pruning." arXiv preprint arXiv:2405.19237 (2024).

[2] Zhang, Yihua, et al. "Unlearncanvas: Stylized image dataset for enhanced machine unlearning evaluation in diffusion models." arXiv preprint arXiv:2402.11846 (2024).

### Questions
As mentioned in the weaknesses section, I'm curious about specific ablations and theoretical/empirical justifications for changes in the method compared to Concept Prune.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Forget-It-All (FIA), a training-free framework for multi-concept machine unlearning in text-to-image diffusion models. The method addresses a limitation of existing approaches that struggle to simultaneously remove multiple unwanted concepts without degrading image quality or requiring extensive hyperparameter tuning. FIA works by computing a Contrastive Concept Saliency metric to identify which neurons respond specifically to target concepts, then uses temporal and spatial sparsity criteria to select Concept-Sensitive Neurons for pruning. Critically, the method preserves Concept-Agnostic Neurons—those that respond broadly across many concepts and support general image generation—while only removing neurons truly specific to unwanted concepts. The authors demonstrate state-of-the-art performance across three unlearning tasks (multi-object, explicit content, and artistic style removal) on benchmarks including Imagenette and I2P, achieving superior forgetting effectiveness (e.g., 1.9% average accuracy on Imagenette vs. 7.34% for the next best method) while maintaining competitive generation quality with less than 0.3% model sparsity.

### Strengths
- No fine-tuning required, making it computationally efficient and reducing overfitting risks compared to existing methods that require extensive training
- First work to connect unstructured neuron masking with multi-concept unlearning, introducing the concept of "concept-agnostic neurons" as a principled way to preserve generation quality
-  Achieves state-of-the-art performance across all three unlearning tasks (1.9% forgetting accuracy vs. 7.34% next best on Imagenette)
- Comprehensive evaluation: Tests on diverse tasks (objects, explicit content, artistic styles) and includes robustness evaluation against adversarial attacks (Ring-A-Bell, MMA, UnlearnDiffAtk)

### Weaknesses
- The paper does not clearly state how many images/samples are generated per concept to compute the Contrastive Concept Saliency scores and resulting masks, which is a significant reproducibility issue.

### Questions
1. How many images/samples are generated per concept to compute the saliency scores?

- Specifically, what are the values of the sample sizes for computing μc, μb, and σb in Equation 2?
- Are these sample sizes the same across all three tasks (objects, explicit content, artistic styles)?
- Please provide this information in a clear table format for reproducibility

2. Sensitivity analysis on sample size:

- How sensitive is your method to the number of images used for mask computation?
- What is the minimum number of samples needed for stable mask identification?
- Did you perform any ablation studies varying the sample size (e.g., 10, 50, 100, 500 images)?
- How does performance degrade with fewer samples?

### Soundness
3

### Presentation
2

### Contribution
3
