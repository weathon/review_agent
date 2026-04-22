# TAMP: Task-aware Multimodal Pre-Interaction for fine-grained Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Current Multimodal Large Language Models (MLLMs) primarily rely on image-level visual-linguistic alignment, limiting their capability in fine-grained visual perception tasks. Existing solutions either serialize coordinates as text inputs, which lose spatial semantics, or introduce specialized expert modules that increase inference latency and exhibit task bias. To address these limitations, we propose TAMP, a Task-aware Multimodal Pre-Interaction for Fine-Grained Multi-modal LLMs, that automatically recognizes key task-relevant information from instructions and extracts corresponding region features through an  unified and detector-free paradigm. A task-aware region connector with a dual-branch is designed that dynamically handles both referring and grounding tasks. By introducing a instruction template with region placeholders, we seamlessly integrate fine-grained region features into the LLM's reasoning process. Extensive experiments demonstrate that our approach achieves state-of-the-art performance on both referring and grounding benchmarks while maintaining strong general VQA capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents a Task-aware Multimodal Pre-Interaction Framework (TAMP) aimed at enhancing fine-grained visual perception in multimodal large language models (MLLMs).  The proposed task-aware region connector with a dual-branch architecture allows for dynamic extraction of task-relevant region features. Experiments are conducted on several referring and grounding benchmarks.

### Strengths
The writing is good. The paper is easy to follow.

### Weaknesses
[1] In Fig. 1, the authors mention that the position response of existing methods is inherently adept at processing discrete symbols, but
lacks capabilities for modeling continuous spatial coordinates. However, the proposed methods still rely on the discrete symbols to encode coordinates with text. Moreover, compared with ROIAlign-based method, the differences of the proposed method are the cross-attention on the visual features. The proposed method cannot address the challenges of existing methods.

[2] A task-aware instruction template cannot be a contribution.

[3] It seems that the proposed task-aware region connector cannot be used for conventional VQA, limiting the application of the proposed methods.

[4] The method still relies on the 224px visual encoder. However, the more recent MLLM can deal with high resolution, e.g., Qwen-VL 2.5, LLava-ov. The compared methods are old.

### Questions
Please refer to Weaknesses.

### Soundness
2

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
4

### Summary
This paper proposes a task-aware multimodal pre-interaction for fine-grained MLLMs, which extract key task-relevant information from instructions and according region features. By employing a instruction template with region placeholders, fine-grained region information is integrated into the reasoning process. Extensive experiments demonstrate the effectiveness of the proposed method on referring and grounding benchmarks.

### Strengths
1. The paper is well-written and easy to understand. The figure 1 and 2 are clear to understand the motivation and the whole picture of the proposed method.

2. The proposed method is simple yet effective to improve the fine-grained capability of MLLMs. The proposed task-aware region connector is somewhat novel to integrate fine-grained region feature.

3. Extensive experiments demonstrate the effectiveness of the proposed method on referring and grounding benchmarks.

### Weaknesses
1. The base model for experiments is out of date. CLIP ViT-L-224px and LLaMA-2-7B suffers very limited performance for evaluating the effectiveness of the proposed method. I suggest the author to conduct experiments with siglip2-384 and qwen2.5-7b to truly validate the effectiveness.

2. In addition to Lora training, I suggest the authors to conduct the full training to prove the effectiveness in commonly used training settings.

### Questions
How much data was used for pre-training and sft? Is there some grounding VQA data used for training?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the fine-grained visual perception capabilities of multimodal large language models (MLLMs) and proposes a dual-branch, task-aware region connector to enhance performance. The module automatically identifies task-relevant information from user instructions, extracts corresponding regional features, and integrates them into the instruction following a task-specific template. Experimental results demonstrate improved performance on downstream tasks.

### Strengths
The proposed method is intuitive and easy to implement.

### Weaknesses
**Limited applicability:**  The method is only designed and evaluated on grounding and referring expression tasks. However, modern MLLMs are expected to handle a wide range of complex tasks, such as reasoning, generation, or editing. The proposed approach lacks generalizability and may not be easily adapted to these scenarios.


**Lack of complexity analysis:** The paper does not analyze the computational or memory overhead introduced by the additional module and training. A thorough comparison of training cost and inference efficiency with other methods is necessary to fairly evaluate the trade-off between performance and complexity.


**Insufficient experimental evaluation:** The experiments are mainly conducted on VQA and RefCOCO datasets. Performance on other widely used benchmarks such as MMMU, POPE, or MMBench is not reported. A more comprehensive evaluation across diverse datasets is needed to validate the general effectiveness of the method.


**Limited novelty:** The proposed method resembles an attention mechanism over image regions based on textual input. However, the design is relatively straightforward and requires additional training and inference overhead, which diminishes its novelty and practical appeal.

### Questions
Can the proposed method be compared with saliency-based approaches? For instance, instead of using the proposed connector, could one directly use attention maps between text and image regions to select relevant features, thereby avoiding extra modules and training?

### Soundness
2

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
4

### Summary
This paper studies fine-grained visual recognition based on Multimodal Large Language Models (MLLMs). The basic idea is to generate a task-aware region token as input to LLM. Experiments show boosted performance.

### Strengths
1. This paper is well-motivated and fig.1 clearly shows the differences of illustrated frameworks. 
2. The proposed method is reasonable, i.e., to acquire more regional cues which might be important for fine-grained visual recognition.
3. Experiments are conducted on multiple datasets, and shows better performance than many existing works.

### Weaknesses
1. One of concerns is the computational overhead, which is not discussed in depth in the paper. 
2. The strong performance is achieved based on strong baselines. Compared with the baseline, the performance enhancement seems marginal in some cases. 
3. Another important concern is the limited generalization capability. This paper is limited to distinguishing between referring and grounding. This degrades the generalization capability of MLLM, although boosts its performance in specific tasks. It would be important to see if this framework could be extended to other fine-grained tasks.

### Questions
1. Need to provide more indepth discussion and comparison on efficiency and computational overhead.
2. It is important to show illustrations of the effectiveness of learned task-aware region token, and compare the learned tokens of referring and grounding.
3. Need to clarify the limited performance enhancement and generalization capability.

### Soundness
3

### Presentation
3

### Contribution
3
