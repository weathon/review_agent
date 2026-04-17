# IC-Custom: Diverse Image Customization via In-Context Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Image customization, a crucial technique for industrial media production, aims to generate content that is consistent with reference images. However, current approaches conventionally separate image customization into position-aware and position-free customization paradigms and lack a universal framework for diverse customization, limiting their applications across various scenarios. To overcome these limitations, we propose IC-Custom, a unified framework that seamlessly integrates position-aware and position-free image customization through in-context learning. IC-Custom concatenates reference images with target images to a polyptych, leveraging DiT's multi-modal attention mechanism for fine-grained token-level interactions. We propose the In-context Multi-Modal Attention (ICMA) mechanism, which employs learnable task-oriented register tokens and boundary-aware positional embeddings to enable the model to effectively handle diverse tasks and distinguish between inputs in polyptych configurations. To address the data gap, we curated a 12K identity-consistent dataset with 8K real-world and 4K high-quality synthetic samples, avoiding the overly glossy, oversaturated look typical of synthetic data. IC-Custom supports various industrial applications, including try-on, image insertion, and creative IP customization. Extensive evaluations on our proposed ProductBench and the publicly available DreamBench demonstrate that IC-Custom significantly outperforms community workflows, closed-source models, and state-of-the-art open-source approaches. IC-Custom achieves about 73\% higher human preference across identity consistency, harmony, and text alignment metrics, while training only 0.4\% of the original model parameters.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents IC-Custom, a unified framework for both position-aware and position-free customized generation. The approach formalizes the task under a universal equation, $p(\hat{X}|C_{I},C_{I'},M,C_{T})$, and utilizes an In-Context Diptych Format where the reference identity image and the fill-in image are concatenated and jointly encoded into token sequences using the DiT architecture. The key technical contribution is the In-Context Multi-Modal Attention (ICMA) module, which incorporates learnable task-oriented register tokens to explicitly signal the customization type and Boundary-aware Positional Embeddings to help delineate the input regions in the diptych configuration.

### Strengths
1. The paper proposes a unified framework that seamlessly integrates position-aware and position-free image customization, addressing the key limitation of non-unified, task-specific approaches found in prior work.
2. The diptych framework and its associated training strategy are well-designed and appear reasonable for achieving the model's unified capability.
3. The motivation for using both task-oriented register tokens and boundary-aware positional embeddings within the architecture is clear.

### Weaknesses
1. The paper lacks sufficient qualitative ablation studies to clearly validate the effect of the Task-oriented Register tokens (TR) and Boundary-aware Positional Embeddings (PE). While quantitative ablation results are provided in Table 3(b), the observed metric values are not significantly compelling compared to the full model design. More visualizations for the ablation studies should be provided to demonstrate the necessity and efficacy of these modules.
2. The Redux Encoder is integrated into the framework; it remains unclear whether the resulting performance in identity consistency is highly reliant on the embeddings provided by this Redux Encoder.

### Questions
See Weaknesses

### Soundness
3

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
3

### Summary
The paper introduces IC-Custom, a unified framework for integrating position-aware (mask-guided insertion) and position-free (text-guided generation) image customization tasks using in-context learning on Flux1. architectures. It proposes an In-Context Multi-Modal Attention (ICMA) module with learnable task-oriented register tokens and boundary-aware positional embeddings to handle diverse scenarios. The authors curate a 12K identity-consistent dataset and evaluate on a new ProductBench for position-aware tasks and DreamBench for position-free, demonstrating superior performance over baselines OmniCtrl, DreamO, Insert Anything, and GPT-4o in metrics such as DINO-I, CLIP-I/T, and human preferences.

### Strengths
1. Novel unification of two customization paradigms using in-context learning on DiT, with innovative ICMA incorporating task-specific register tokens and positional embeddings to handle ambiguity.

2. Robust dataset curation (12K samples, real+synthetic) and comprehensive evaluations, including ablations that validate key components.

3. Intuitive diptych formulation and clear training strategy enable seamless handling of diverse tasks without separate models.

4. Advances efficient, identity-consistent generation for practical applications, outperforming SOTA with minimal parameters trained.

### Weaknesses
1. No quantitative analysis of inference time, failure cases beyond visuals. The paper does not address how long the model takes to generate results, nor does it provide a comparison to baseline models in terms of speed. 

2. Unclear Handling of Multi-Reference Customization. The paper briefly mentions multi-reference customization as future work, but does not provide sufficient detail on how this will be incorporated into the current framework. 

3. Limited Exploration of Dataset Diversity. The paper mentions that the authors curated a dataset with 12K identity-consistent samples, combining real-world and synthetic data. However, there is limited discussion about the diversity of this dataset, particularly with respect to edge cases.

### Questions
1. Could the author share more specific details on the computational resources used, such as the type and number of GPUs, as well as the total GPU hours required for the 20K iterations?
2. In the ablation studies, why do the results show only a minor drop when real data is excluded? How was the synthetic data filtered to ensure it matches the quality of real-world data?
3. The metrics from the user study are significantly higher than those of the baselines; could the author clarify why IC-Custom outperforms them to such a large extent?
4. How does IC-Custom perform when faced with more challenging or extreme customization scenarios, such as very complex images, unclear user input, or ambiguous mask regions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes IC-Custom, a unified framework for both position-aware (e.g., image insertion, try-on) and position-free (e.g., text-guided generation) image customization. It reformulates both tasks through a diptych in-context formulation, processed by a new In-Context Multi-Modal Attention (ICMA) module with task-register tokens and boundary-aware positional embeddings. A related dataset (CustomData) and a new benchmark are also introduced. IC-Custom achieves strong SOTA results while fine-tuning only 0.4 % of the base model.

### Strengths
+ Elegant unification: A clear and practical integration of position-aware and position-free customization within one architecture, avoiding the need for separate specialized models.

+ Well-designed ICMA mechanism: Task tokens and boundary embeddings effectively address task ambiguity and spatial confusion, with strong ablations supporting their contribution.

+ Comprehensive evaluation: Extensive comparisons show consistent superiority over GPT-4o and recent open-source baselines, supported by high-quality qualitative examples.

### Weaknesses
+ The “In-context learning” framing is somewhat overstated—the model is trained end-to-end rather than showing adaptive few-shot behavior.

+ There is no explicit modeling of 3-D or geometric consistency; limited robustness tests for large viewpoint changes.

### Questions
1. How do task tokens learn to specialize—via supervision or implicitly from data mixing?

2. Can IC-Custom handle multi-reference or style-blending scenarios?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose IC-Custom, a unified framework integrating position-aware (mask-guided editing) and position-free (text-guided identity generation) image customization. It adopts an in-context diptych format by concatenating reference and fill-in images into a unified input and introduces ICMA module, which incorporates task-oriented learnable register tokens and boundary-aware embeddings to resolve task ambiguity and boundary confusion. They also curate CustomData (12K samples: 8K real-world e-commerce diptychs, 4K filtered synthetic ones) to avoid unrealistic synthetic data flaws, with strict filtering (DINOv2 similarity ≥0.2) and auto-annotation (Qwen-VL2.5 for captions, Grounded SAM for masks). Experiments results show the effectiveness of the proposed method.

### Strengths
1. The paper addresses the long-standing gap of isolated position-aware and position-free image customization by proposing a unified framework. 
2. The method balances performance and efficiency: built on pre-trained FLUX.1-Fill, it only trains 0.4% of parameters via LoRA fine-tuning, while outperforming baselines on ProductBench and DreamBench.
3. Overall, the writing is clear.

### Weaknesses
1. Limited Scenario Generalization: IC-Custom only validates single-object customization. Its performance in multi-object scenarios (e.g., inserting a backpack and a laptop simultaneously) remains untested.
2. The paper does not mention any performance in the absence of text input. Position-aware customization relies on text to define identity and scene constraints. How the model behaves when it lacks clear text guidance is unclear.

### Questions
How the model behaves when it lacks clear text guidance?

### Soundness
3

### Presentation
3

### Contribution
3
