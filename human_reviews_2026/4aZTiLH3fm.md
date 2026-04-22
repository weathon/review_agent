# TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 2, 4, 8

## Abstract
Modeling semantic and structural information from tabular data remains a core challenge for effective table understanding. Existing Table-as-Text approaches flatten tables for large language models (LLMs), but lose crucial structural cues, while Table-as-Image methods preserve structure yet struggle with precise semantics. Recent Table-as-Multimodality strategies attempt to combine textual and visual views, but they (1) statically process both modalities for every query-table pair within large multimodal LLMs (MLLMs), inevitably introducing redundancy and even conflicts, and (2) depend on costly fine-tuning of MLLMs.  In light of this, we propose TableDART, a training-efficient framework that integrates multimodal views by reusing pretrained single-modality models. TableDART introduces a lightweight 2.59M-parameter MLP gating network that dynamically selects the optimal path (Text-only, Image-only, or Fusion) for each table–query pair, reducing redundancy and avoiding conflicts that arise when textual and visual views of the same table provide inconsistent cues. By routing to the most appropriate view, our framework improves both accuracy and efficiency. In addition, we propose a novel agent to mediate cross-modal knowledge integration by analyzing outputs from text- and image-based models, either selecting the best result or synthesizing a new answer through reasoning. This design avoids the prohibitive costs of full MLLM fine-tuning. Extensive experiments on seven benchmarks show that TableDART establishes new state-of-the-art performance among open-source models, surpassing the strongest baseline by an average of 4.02%. The code is available at: https://github.com/xiaobo-xing/TableDART.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed TableDart, a method designed for understanding semantic and structual information in table data. It adopted an MLP gating network that dynamically selected the optimal path, avoiding prohibitive costs of full MLLM fine-tuning. It also designed an LLM-based Agent to deal with outputs of other two single-modality experts.

### Strengths
- This method only have few parameters to train, avoiding cost of fully finetuning
- The designed dusion agent consider different conditions of other two outputs of single-modality experts, including similar and conflict answers
- This paper do many experiments to validate its performance and efficiency

### Weaknesses
There are some core issues that have significant influence on my rating:
- How do the MLP gating network choose the path after giving the logits? Choose the path with the highest score or paths whose scores exceed certain thresholds. If it is the later condition, what if the gating network choose more than one path?
- In **Inference efficiency of section 4.3**, what is the model used in Non-Adaptive baseline? Why the mean inference time can be reduced? From my perspectives, if single-modality expert is chosen, the inference time remain the same. And if the fusion path is chosen, the other two path will be executed, where the inference time will double.
- What's the standard when you choose the single-modality MLLM and the fusion agent as these three models have a significant influence on the performance of your methods.
- Part results in Table 1 are the same with paper "Multimodal Tabular Reasoning with Privileged Structured Information" but no reference is mentioned.

Other discussions:
- Why do you choose three paths, including text-only, image-only and fusion. Why not add another path with MLLM modality? I take the thought that this path will provide more insight of the information than the single-modality experts do.
- I wonder the number of times each path is chosen as this maybe can help understand the inference efficiency.

### Questions
Please refer to section Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes TableDART, a framework that adaptively selects the optimal processing path for each query to make the most of modality-specific features for table understanding.

### Strengths
Proposes a novel dynamic routing framework that adaptively selects the optimal modality (text, image, or fusion) for each table–query pair.

Achieves good performance gains across multiple benchmarks with minimal additional parameters. The parameter-efficient training makes it practical and resource-friendly.

### Weaknesses
1. Your results seem insufficient to demonstrate that you achieve Inference Efficiency (the main text only contains a one-sentence description of the results). From the appendix, it can be seen that most tasks on the datasets can be completed with a single modality; however, a fusion strategy is still predominantly used.

2. The two single-modality based methods you used already perform far beyond most models, making many methods seem completely meaningless for comparison. Aren't there any other relatively stronger baselines in this field?

3. Three paths are a bit too few for routing. More models should be adopted to provide more options.

4. How does computational efficiency change as $lambda$ varies? This should be presented clearly. Furthermore, I believe it would be normal for the performance to be optimal when $ \lambda$ is 0, but this does not seem to be the case.

5. From the results, Fusion appears to be the primary source of your performance improvement, and this deserves a detailed explanation.

6. Why not use Qwen2.5-VL or even Qwen3-VL as your baseline? 

7. The experimental results in Table 1 on Ovis2 seem partially the same as "Multimodal Tabular Reasoning with Privileged Structured Information" (https://arxiv.org/pdf/2506.04088), except for the result on the TABMWP dataset. However, no reference or explanation can be found in the paper. The authors also did not compare with this paper. Could the authors explain the reason?

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TableDART—a training-efficient framework that reuses pretrained single-modality models via a lightweight 2.59M-parameter MLP gating network, dynamically selecting the optimal path (Text-only, Image-only, or Fusion) for each table–query pair to mitigate redundancy and conflicts.

### Strengths
TableDART addresses limitations of existing table understanding methods by introducing a training-efficient framework with a lightweight MLP gating network, dynamically selecting optimal (text, image, or fusion) paths per table-query pair to reduce redundancy/conflicts. It further incorporates a cross-modal agent for knowledge integration without costly MLLM fine-tuning, achieving SOTA on seven benchmarks.

### Weaknesses
1. The paper is difficult to comprehend, as many concepts are employed directly without any corresponding explanation of their meanings—
   - For instance, what does the term "fine-grained" on line 46 specifically refer to?
   - As noted in the abstract: "statically process both modalities for every query-table pair within large multimodal LLMs (MLLMs), inevitably introducing redundancy and even conflicts," what exactly does the term "conflicts" refer to herein?
   - In the Introduction, it is mentioned that table data exhibits the property of heterogeneity—could the authors provide an explanation for this?
   - What modality exactly is Table T on line 145—textual or visual?
2. The literature review is inadequate, and the compared methods lack several representative approaches, such as the domain-specific VLM TabPedia、SynTab-LLaVA, and the general VLM Qwen2.5VL.
3. How are the three token sequences (e_q, e_t, e_v), which vary in length and differ in token dimension, concatenated?
4. As the authors state, "We employ TableGPT2-7B and Ovis2-8B as TableDART’s single-modality experts," it is queried whether ablation experiments have been conducted using other alternative single-modality experts to verify the robustness and generalizability of the proposed framework.

### Questions
see the weakness

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes TableDART, a dynamic adaptive multi-modal routing framework for table understanding tasks. The key innovation lies in replacing traditional monolithic multimodal LLM fine-tuning with a lightweight 2.59M-parameter gating network that dynamically selects among three inference paths, Text-only, Image-only, or Fusion, depending on each table–query pair’s characteristics. The fusion stage employs an LLM agent that arbitrates between unimodal outputs or synthesizes a new answer when both fail. TableDART reuses frozen pretrained unimodal models, drastically reducing training cost. Experiments on seven benchmarks show TableDART surpasses strong baselines.

### Strengths
- The dynamic adaptive routing approach is impressive. It represents a conceptual shift from static multimodal fusion to instance-specific modality selection.

- The paper demonstrates strong empirical rigor.

- TableDART addresses a key bottleneck in multimodal table understanding, redundant modality processing and high fine-tuning cost

### Weaknesses
- While the paper highlights the LLM-based Fusion agent's "Arbitrator" and "Rescuer" roles, its decision logic is black-box. There is no systematic evaluation of when or why it succeeds beyond anecdotal cases.

- The use of Gemini 2.0 Flash for fusion raises reproducibility and accessibility concerns. A variant using a fully open LLM would improve transparency.

- The gating network's supervision relies on pre-computed correctness vectors, but the procedure for labeled paths is not critically examined.

- It would be instructive to show a variant where fusion uses a simple weighted ensemble rather than an LLM agent, to disentangle the benefits of adaptive routing at the fusion stage.

### Questions
- How sensitive is the routing policy to the composition of the training mixture? Would domain-specific fine-tuning alter routing distributions?

- Does the gating network rely primarily on query semantics or table structure cues when choosing between text and image paths?

### Soundness
3

### Presentation
3

### Contribution
3
