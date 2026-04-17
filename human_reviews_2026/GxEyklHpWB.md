# SCOPE: Selective Cross-modal Orchestration of Visual Perception Experts

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Vision-language models (VLMs) benefit from multiple vision encoders, but naively stacking them yields diminishing returns while multiplying inference costs. We propose SCOPE, a Mixture-of-Encoders (MoEnc) framework that dynamically selects one specialized encoder per image-text pair via instance-level routing, unlike token-level routing in traditional MoE. SCOPE maintains a shared encoder and a pool of routed encoders. A lightweight router uses cross-attention between text prompts and shared visual features to select the optimal encoder from the routed encoders. To train this router, we introduce dual entropy regularization with auxiliary losses to balance dataset-level load distribution with instance-level routing confidence. Remarkably, SCOPE with one shared plus one routed encoder outperforms models using all four extra encoders simultaneously, while reducing compute by 24-49\%. This demonstrates that intelligent encoder selection beats brute-force aggregation, challenging the prevailing paradigm in multi-encoder VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This authors presents SCOPE, an Mixture-of-Encoders-based framework designed to enhance VLMs by dynamically selecting the most suitable off-the-shelf visual encoder for a given image-text pair. SCOPE utilizes a lightweight router, guided by cross-attention between the text prompt and shared visual features, to achieve instance-level routing. The authors demonstrate that selective orchestration significantly reduces computational cost (up to 49%) while outperforming models that naively aggregate multiple encoders. This work challenges the prevailing paradigm of encoder aggregation and offers a highly efficient and valuable alternative.

### Strengths
1.  **Efficient Architecture:** The core idea of using instance-level routing to select from a pool of off-the-shelf visual encoders tackles the critical problem of high inference costs in multi-encoder VLMs. The approach effectively decouples the benefits of specialized encoders from their cumulative computational burden.

2.  **Principled Router Training:** The introduction of the Dual Entropy Regularization and Auxiliary Losses provides a well-motivated mechanism to train the non-differentiable routing process. This structure addresses the inherent conflict between requiring load balancing(for robust training) and demanding high-confidence selection (for performance).

3.  **Strong Empirical Efficiency Claim:** The paper provides compelling evidence that intelligent selection beats brute-force aggregation. The result that SCOPE, using only one shared plus one routed encoder, outperforms models that simultaneously use all four extra encoders in OCR-related and chart understanding scenes.

### Weaknesses
**1.  Necessity of Strict Load Balancing:** 
The strict imposition of load balancing losses ($\mathcal{L}_{be}, \mathcal{L}_{ba}$) requires further discussion and justification. Unlike traditional MoE where experts are randomly initialized FFN blocks prone to "dying" without balancing, SCOPE utilizes powerful, off-the-shelf vision encoders whose utility is naturally non-uniform across tasks. Forcing the router to use demonstrably weaker experts to maintain balance might introduce noise and counteract the primary goal of performance maximization, especially if one encoder is overwhelmingly strong. An investigation into loss annealing (gradually reducing the balancing weight) or an adaptive expert selection/pruning mechanism (similar to the concept in [1]) would allow the model to learn the intrinsic value of each expert and achieve more self-adaptive selection.

**2.  Scope of Experimental Evaluation:** 
The empirical evaluation is heavily focused on OCR-related and chart understanding benchmarks (DocVQA, InfoVQA, TextVQA, ChartQA). The paper lacks evaluation on general-purpose visual reasoning and perception benchmarks (e.g., MME, MMStar, MMBench). This limitation makes it difficult to assess the framework's versatility and its ability to generalize across a broader spectrum of visual understanding tasks, which is my major confusion point.

**3.  Baselines and Router Input:**
* The performance comparison against the original Qwen-2.5-VL is missing. The authors use the Qwen-2.5-VL vision encoder and Qwen-2.5-7B LLM, but the reported baseline performance (e.g., ChartQA 68.3) appears significantly lower than the established performance of the original Qwen-2.5-VL model (87.3). This discrepancy makes it challenging to accurately gauge the actual gain brought by SCOPE over a strong, optimized monolithic VLM baseline.
* Additionally, the choice to use the Qwen-2.5-VL vision encoder's output as the router's visual input is a key design choice. The authors should investigate and report on the effect of using alternative visual features to drive the routing decision.

**4.  Expert Pool Exploration:** 
The paper only explores a constrained set of visual experts. There is no discussion or experimentation with incorporating outputs from other prominent or specialized vision encoders. For instance, the authors could explore experts providing direct dense visual outputs (like depth estimation used in [1]) or integrating other powerful foundational models (like SAM or EVA-02 as mentioned in [2]).

**5.  Challenges in Real-World Scenarios:** 
The paper does not discuss or provide solutions for the challenges of **multi-image input** or **multi-turn conversation** scenarios. The current instance-level routing design seems inherently limited to single-turn, single-image inputs. Clarification on how SCOPE would be adapted or whether it is intended to handle more complex, sequential VLM interactions is necessary.

### Questions
Please see the above comments.

### Soundness
4

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
5

### Summary
This paper proposes Scope, a VLM framework coupled with a mixture of vision encoders. Scope relies on a learnable router that, conditioned on image and text embeddings from Qwen models, dynamically selects a vision encoder per instance as additional image features for vision-language reasoning. To avoid mode collapse during model training, the proposed method incorporates both batch-level and instance-level entropy and auxiliary losses to encourage the model to explore all options of the included vision experts while maintaining high confidence in the selected expert. Scope was extensively ablated with varying auxiliary loss weights and the number of activated experts on standard VQA-related benchmarks.

### Strengths
The paper is well-written, and I appreciate the use of matched colours in Figure 2, making it easy to navigate and understand the design pipeline. The proposed method offers new insights into designing VLM systems in conjunction with encoder designs.

### Weaknesses
The overall paper has a really good flow on introducing the architecture design and training losses. However, I found the experiments are very insufficient and reading like a novel with a rushed ending. This might be due to some many other design choices in the model implementation, which I will explain below. 

- Scope has poorly chosen vision experts. For other notable related VLM works in this domain like Prismer and Eagle; both models chose vision experts spanning across multiple perception domains and tasks, like 3D understanding: depth/surface normal experts, OCR detection models, object segmentations, etc, to cover a wide range of diversity of expert domains. However, the chosen experts here are all general vision models aimed for general-purpose vision backbones (with large-scale self-supervised training and language-image training). In particular, I feel like the inclusion of the ViT + ConvNext variants of the same DINOv3 model seems to be very redundant. I hope the author could explain why Scope chose the vision experts in such a way, and why not consider domain-specific experts like depth estimation, object detection experts heavily explored in the prior work?
- Related to the previous point, it makes the model nearly incomparable to other recent state-of-the-art VLMs, and it is hard to make sense of the notion that more experts lead to worse results. It's also difficult to interpret the chosen experts, e.g. whether the OCR expert really helps on OCR-related tasks; and whether depth experts really help on visual spatial reasoning type of tasks. 
- I found it is very difficult to understand Table 2: I thought baseline-1 should be equivalent to Scope-CA as they both use 1 expert? Or is it possible to report: 
    - The same single expert across all questions, to confirm the router design and selection is really helpful. 
    - Additional experts (top-k) from the router, to make sure the router ranking really makes sense. 
- Some other prior designs on the encoder ensembling techniques are missing: e.g. 1. the expert resampler (perceiver-like) design to merge all experts’ embeddings into a fixed length of tokens proposed from Prismer. 2. Channel-concat, sequence-concat design space proposed by Eagle, all seem to be very important. The first one will significantly reduce the training complexity, without having the proposed auxiliary losses, as we always consider all experts (and with the bounded inference compute). 2. We will know the performance upper bound by having no compression on the expert knowledge.

### Questions
See the weaknesses.

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
3

### Summary
This paper introduces SCOPE, a novel approach for improving the efficiency and performance of VLMs through dynamic selection of vision encoders. By maintaining a pool of specialized vision encoders and a shared encoder, SCOPE uses a lightweight routing mechanism to dynamically select the most appropriate encoder for each image-text pair. This results in improved computational efficiency and performance, achieving significant reductions in inference costs (24-49%) compared to models using all encoders simultaneously. The proposed routing mechanism is trained with dual entropy regularization and auxiliary losses, aiming to balance load distribution and routing confidence.

### Strengths
1.	The dynamic selection of encoders based on input image-text pairs is an innovative approach that helps optimize the use of vision encoders, enhancing both computational efficiency and model performance. This allows the system to select the most relevant expert encoder without incurring the cost of always using all encoders.


2.	The paper demonstrates that SCOPE outperforms static multi-encoder models on several benchmark tasks, such as VQA and document understanding, even when using fewer encoders. This is a solid demonstration of the proposed model's effectiveness.

### Weaknesses
1.	While the dynamic encoder selection reduces computational overhead, SCOPE still requires all encoders to be loaded into memory, especially in batch inference settings. In these cases, if all encoders must be preloaded to ensure fast dynamic selection, the memory consumption could become prohibitive. This significantly diminishes the model’s utility in memory-constrained environments, such as edge devices or real-time applications.

2.	Although SCOPE outperforms models that use all four encoders, there are existing methods (eg ToVE: Efficient Vision-Language Learning via Knowledge Transfer from Vision Experts)that achieve similar or better results by integrating the strengths of different visual experts during training. These approaches do not require dynamic selection at inference time, which reduces the overhead of loading multiple encoders.

3.	The top-1 routing strategy limits the ability of the model to leverage the diversity of visual features that multiple encoders might provide. In scenarios where a more complex fusion of features from multiple encoders is beneficial, the current approach might not capture all relevant information. Top-k routing or alternative mechanisms could enhance the model's ability to integrate diverse visual information, especially in more complex tasks.

### Questions
As the weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
