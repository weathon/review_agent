# LaTtE-Flow: Layerwise Timestep-Expert Flow-based Transformer

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 6, 2

## Abstract
Recent advances in multimodal foundation models capable of both image understanding and generation have opened exciting avenues for building unified systems that seamlessly handle diverse vision-language tasks. Despite the progress, existing unified models typically require extensive pretraining and struggle to achieve the same level of performance compared to models dedicated to each task. Additionally, many of these models suffer from slow image generation speeds, limiting their practical deployment in real-time or resource-constrained settings. In this work, we propose Layerwise Timestep-Expert Flow-based Transformer (LaTtE-Flow), a novel and efficient architecture that unifies image understanding and generation within a single multimodal model. LaTtE-Flow builds upon powerful pretrained Vision-Language Models (VLMs) to inherit strong multimodal understanding capabilities, and extends them with a novel Layerwise Timestep Experts flow-based architecture for efficient image generation. LaTtE-Flow distributes the flow-matching process across specialized groups of Transformer layers, each responsible for a distinct subset of timesteps. This design significantly improves sampling efficiency by activating only a small subset of layers at each sampling timestep. To further enhance performance, we propose a Timestep-Conditioned Residual Attention mechanism for efficient information reuse across layers. Experiments demonstrate that LaTtE-Flow achieves strong performance on multimodal understanding tasks, while achieving competitive image generation quality with up to 48× faster inference speed compared to existing unified multimodal models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes LaTtE-Flow, a novel architecture for extending image understanding models to perform image generation. Its core innovations are the Layerwise Timestep Experts (LTE), which partition transformer layers into groups specialized for different flow-matching timesteps to reduce inference computation, and Timestep-Conditioned Residual Attention (TCRA), a mechanism for reusing attention maps across layers. The method significantly reduces inference computational cost by using different activation layers on different time steps, while maintaining and even improving on both image generation and image understanding.

The proposed method is novel and seems to do lower computational cost at inference without sacrificing the performance. I suggest a weak accept.

### Strengths
1: The idea of LaTtE-flow is simple but effective. Based on its extension nature, it may be applied on any VLMs and trains them to also perform image generation with low computational cost. 

2: LaTtE-flow yields better image generation and image understanding compared with advanced models of similar scale at much fewer activation parameters and average inference time.  

3: The paper delivers a thorough ablation study.

### Weaknesses
1: The authors do not include the performance of the base model, Qwen2-VL-2B in Table 2, making it hard to compare how much gain LaTtE-Flow gives to the image understanding task. 

2: The description of Table 2 does not match the content. It says "we report the number of activated parameters", however it doesn't. It also does not perform the computation cost(or time cost) for different models like in Table 1. 

3: As an extension method, the authors only experiment LaTtE-Flow on one base model, Qwen2-VL-2B and does not generalise the method to other VLMs.

### Questions
1: Could you perform a more complete Table 2, with baseline performance(base model Qwen2-VL-2B) and computational cost( FLops, or other metrics)?
2: Is LaTtE-Flow able to perform on other base models besides Qwen2-VL-2B? How does it perform?

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
This paper presents LaTtE-Flow, a unified MLLM which incorporates flow-matching-based image generation with VLM. First, they propose Layerwise Timestep Expert, distributing transformer layers into different timestep-specific experts, to improve the inference efficiency of unified MLLM. Second, they propose a gate attention approach to reuse the attention map in previous layers. Experiment results show that LaTtE-Flow outperform previous unified MLLM while being more efficient.

### Strengths
1. **The idea is interesting.** The idea of decouple multiple flow matching steps into multiple transformer blocks is interesting and results in good performance.

2. **Efficient.** LaTtE-Flow is very efficient, 6 times faster than Janus Pro. The author provides real running time to verify their claim.

3. **Effective.** LaTtE-Flow achieves low latency while keeping strong image understanding and generation performance.

### Weaknesses
1. **Compare and discuss with concurrent works.** Although using different data and model size, I suggest the author compare and discuss with newer Unified MLLM, including LMFusion, Blip3o and Bagel. 

2. **Unification of generation and understanding.** LaTtE-Flow use different visual encoders and different sets of parameters for image understanding and generation. If the model first generates an image and then performs VQA based on the generated image, it requires two forward passes. I hope the authors can discuss this scenario.

3. **Lack of scale-up experiments.** This paper reports experiments only on 2B models. This is understandable, possibly due to computational constraints. It would be better if the authors could also run an 8B-scale model to demonstrate scalability.

### Questions
See weakness

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
4

### Summary
This paper proposes LaTtE-Flow, a unified and efficient framework for multimodal large language models that jointly handle visual understanding and image generation. The key idea is to enhance generation efficiency without sacrificing understanding capability. To achieve this, LaTtE-Flow introduces a Layerwise Timestep-Expert (LTE) design, where different Transformer layer groups specialize in specific timesteps of the flow-based generation process—thus reducing redundant computation during sampling. Additionally, a Timestep-Conditioned Residual Attention mechanism enables effective information reuse across layers and timesteps, improving coherence and stability in generation. With these designs, LaTtE-Flow substantially accelerates the flow-based generation process while maintaining high-quality visual outputs and strong understanding performance, outperforming prior unified models in both accuracy and efficiency.

### Strengths
1. The paper presents a clear and well-motivated problem statement, effectively highlighting the efficiency–quality trade-off in unified multimodal generation and offering a logically coherent solution through a flow-based Transformer design.

2. The proposed Layerwise Timestep-Expert mechanism is both elegant and practical, significantly improving inference efficiency by activating only relevant Transformer layers at each timestep.

3. The integration of Timestep-Conditioned Residual Attention is innovative, allowing effective feature reuse across layers and timesteps, which enhances both generation quality and training stability.

### Weaknesses
1. The paper does not provide a direct comparison between the proposed LaTtE-Flow and the original VLM backbone on multimodal understanding tasks, leaving unclear how much the unified training or flow-based adaptation affects understanding performance.

2. The work lacks quantitative results on standard text-to-image generation benchmarks, which limits the evaluation of LaTtE-Flow’s true generative capability and generalization to open-ended visual synthesis.

3. Although the architecture introduces several novel components, the ablation studies are relatively insufficient — many key design choices, such as the number of timestep experts or the specific contribution of residual attention, are not systematically analyzed.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The core idea of this paper is to distribute the timestep modelling of flow matching models to different transformer layers, improving inference speed by activating only a small subset of layers at each sampling timestep. Moreover, a Timestep-Conditioned Residual
Attention mechanism is proposed to incorporate attention results across timesteps groups. The proposed method is instantiated with a VLM (i.e., Qwen2.5VL-2B), in the context of unified multimodal models. Following LMFusion, the LLM part is frozen to preserve the understanding ability of the original VLM. For image generation, the generation layers are trained on the ImageNet dataset.

### Strengths
1. The paper is well written and easy to follow. The sampling efficiency of visual generation is a significant research question.

2. The solution of distributing timestep modelling across transformer layers is intuitive. And the proposed time-conditioned residual attention effectively incorporates cross-layer information, boosting convergence and overall performance.

3. Comprehensive studies on the design choices, such as expert groups and the effects of residual attention, are conducted in the experiments.

### Weaknesses
1. *Unclear Motivation:* As stated in the abstract, the paper studies unified multimodal models that struggle to achieve the same level
of performance compared to specialist models. However, the paper only addresses the problem of sampling efficiency, which seems to have digressed from the core issue of unified models.

2. *Experiments Are Incomprehensive:* Although the paper is for unified multimodal models that include both text and image, the image generation of the model is only trained and evaluated on the class-conditial generation dataset---ImageNet.

### Questions
The idea of Layerwise Timestep-Expert seems a universal solution to all diffusion/flow-matching models, would it be applicalble to a wider range of DiTs for visual generation?

### Soundness
2

### Presentation
2

### Contribution
2
