# Zhyper: Factorized Hypernetworks for Conditioned LLM Fine-Tuning

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Large Language Model (LLM) conditioning refers to instructing an LLM to generate content in accordance with the norms and values of a specific culture, beliefs of a particular political orientation, or any desired text-specified semantic conditioning. Unfortunately, prompt engineering does not ensure that LLMs behave in accordance with a desired conditioning due to the inductive bias of the pre-training and alignment datasets. Prior works have focused on fine-tuning LLMs by directly conditioning the LoRA weights; however, such methods introduce a large number of parameters. As a remedy, we propose Zhyper, a parameter-efficient factori$\textbf{z}$ed $\textbf{hyper}$network framework that generates context-aware LoRA adapters from textual descriptions. Experiments on multiple benchmarks show that Zhyper achieves competitive performance with up to $\textbf{26x}$ fewer parameters than the state-of-the-art baselines. Furthermore, we extend Zhyper to cultural alignment, demonstrating improved generalization to out-of-domain settings and a better capturing of fine-grained contextual values.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Zhyper, a parameter-efficient, factorized hypernetwork framework designed for context-aware fine-tuning of Large Language Models (LLMs). Zhyper employs a lightweight hypernetwork that generates LoRA adapters from textual inputs—such as task instructions or cultural descriptions—enabling effective adaptation to both specific tasks and cultural contexts. Compared to existing methods like T2L and HyperLoRA, Zhyper achieves this with significantly fewer parameters. Comprehensive evaluations across multiple benchmarks show that Zhyper matches or exceeds the performance of prior approaches while reducing parameter overhead by up to 26 times. It also exhibits strong generalization to unseen domains and supports fine-grained cultural alignment.

### Strengths
1. Unlike earlier hypernetwork-driven LoRA approaches such as T2L and HyperLoRA, Zhyper delivers on-par or superior results while drastically cutting down the number of parameters required per context, marking a significant leap in parameter efficiency.

2. By introducing a novel factorized design that separates diagonal and square modulation pathways, Zhyper advances beyond conventional hypernetworked LoRA strategies, enabling more efficient use of low-rank intermediate representations for conditional adaptation—this leads to lower memory consumption and computational load with minimal impact on model accuracy.

3. Evaluations across diverse standard benchmarks demonstrate that Zhyper consistently performs competitively with or even surpasses state-of-the-art baseline methods, highlighting its robustness and broad applicability.

### Weaknesses
1. While Zhyper’s core innovation, which replaces dense hypernetwork outputs with a factorized diagonal/square modulation mechanism, does lead to notable gains in parameter efficiency over predecessors like T2L and HyperLoRA, the underlying conceptual framework remains closely aligned with existing hypernetwork-based LoRA methods. The approach appears more as a refined optimization within the current paradigm rather than a transformative rethinking of how contextual signals are integrated into LLM fine-tuning.

2. The paper overlooks several key developments in efficient LLM adaptation that are directly relevant to its methodology [1,2,3,4]. Incorporating comparisons and contextualizing Zhyper against these approaches would strengthen the paper’s contribution claims and better delineate its novelty.

[1] HyperTuning: Toward Adapting Large Language Models without Back-propagation

[2] LoSiA: Efficient High-Rank Fine-Tuning via Subnet Localization and Optimization

[3] S²FT: Efficient, Scalable and Generalizable LLM Fine-tuning by Structured Sparsity

[4] LoQT: Low-Rank Adapters for Quantized Pretraining

3. The experimental evaluation is entirely based on Mistral-7B-Instruct, with no testing on larger-scale models, multilingual benchmarks, or alternative architectures outside the standard decoder-only transformer family.

### Questions
1.  Can the authors provide empirical evidence of Zhyper’s parameter efficiency and accuracy in LLMs larger than Mistral-7B, and non-transformer models?

2.  Beyond asymptotics, can the authors discuss or compute tighter data-dependent or task-specific generalization bounds for Zhyper-diag vs. T2L? Does reduced Rademacher complexity translate into observable gains as $n$ grows?

### Soundness
3

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
This paper proposed a new method for parameter efficient finetuning. 
The key idea is that the Lora weight for making a model meet a new requirement can be generated from the textual description using a hypernetwork.
The method generate the parameters for the Lora adapter by using an embedding that can tell the contextual information, the layer number and the attention module for the lora.
Experiments show that the proposed method can generate useful weights for LLMs to transfer to unseen tasks or align with an unseen culture during the hypernetworks training.

### Strengths
1. The task is an important one, and the paper proposed method seems to be effective.
2. The proposed method can save a magnititude of number of parameters compared to similar methods.

### Weaknesses
1. The experiments should try to cover a wide range of base LLM to show the proposed method is not just useful for one base LLM.
2. Would reasoning LLM benefit from this method?
3. The method only modifies a very small part of the base LLM, thus I think the models abilities are still bounded by the base LLM. It would be great if the paper can discuss on that and maybe show some negative results that the proposed method and other method cannot help base LLM learn tasks beyond the base LLM.

### Questions
See weaknesses

### Soundness
3

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
Conditioning Large Language Models (LLMs) to align with specific textual descriptions, such as tasks or cultural values, often introduces a large number of parameters when using existing hypernetwork-based LoRA methods. This paper proposes Zhyper, a parameter-efficient factorized hypernetwork framework that generates a compact, context-aware modulation signal z from these textual descriptions. Instead of generating the full adapter weights, Zhyper injects this small z signal between shared, trainable LoRA matrices A and B to compute the final weight update. Experiments show Zhyper achieves competitive performance on task-conditioning benchmarks with up to 26x fewer parameters than baselines, while also demonstrating improved generalization in the novel use case of cultural alignment.

### Strengths
* This paper proposes ZHYPER, which enables efficient conditional generation while significantly reducing the number of parameters compared to previous methods.
* Experimental results show that ZHYPER achieves fine-tuning performance comparable to larger models while using only one-tenth of the parameters.
* The authors provide a theoretical analysis demonstrating the superior generalization ability of ZHYPER.
* The paper is well-written.

### Weaknesses
* Overall, this work presents an improvement over T2L. Instead of generating the entire LoRA, it only needs to generate a low-rank embedding. However, the contribution is still largely incremental.
* The experiments are conducted on too few models. The authors only evaluate on Mistral-v0.2, lacking experiments on a wider range of models such as Qwen3, Llama3, and Gemma3. I believe the authors should test across different model families and scales.
* There is no ablation study on the embedding model. Since all experiments use the same embedding model, the authors should perform ablation experiments to examine its influence. Additionally, what would happen if the model’s own embedding outputs were used as inputs?
* The paper lacks comparison with Task Vector, which serves a similar purpose by injecting domain-specific information.

### Questions
* Is it necessary to apply LoRA to every layer? Would applying it only to certain layers lead to better performance?

* Could the authors provide experiments with more embedding models and large language models (LLMs)?

* Could the authors compare their method with the Task Vector approach and clarify the differences between the two?

* Why does the method generate fewer parameters? Since both approaches use LoRA, the total number of additional model parameters is not reduced — only the number of generated parameters is smaller.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper is built on top of Text-to-LoRA (T2L). T2L trains a hypernetwork that generates a task-specific LoRA matrix given a textual task description. This paper proposes two architecture modifications that reduce the number of the parameters of the hypernetwork used to generate task-specific LoRA. The proposed model performs competitively with SOTA methods with reduced number of parameters.

### Strengths
- Thorough empirical comparison with existing baselines on various tasks
- The architecture is simple and effective at reducing the number of parameters

### Weaknesses
- This paper misses important prior works, i.e., LoRA-XS [2], VeRA [3], which propose exactly the two parameterizations used in this work (square and diag). The existence of these two prior works directly dilutes the contributions of the paper, making it hard to justify acceptance as the paper's core idea is simply chaning the output space of T2L from LoRA to LoRA-XS or VeRA without much insights gained.
- This paper focuses specifically on reducing the number of parameters of the hypernetwork. However, the cost (time and memory) of running a forward pass over the hypernetwork is a fraction of generating a response done by the target LLM. Therefore, I respecfully doubt that the propose achitecture would impact any meaningful metrics related to LLM inference (e.g., VRAM, latency, etc.) 
- Section 2.1 and 2.2 are largely based on T2L [1] but is not explicitly mentioned in the text, which makes it hard to identify the novelty of this paper. I believe that by explicitly mentioning how T2L formalizes the problem and what its architecture looks like in Section 2.2 would be beneficial for reader's understanding and scientific credit assignment.
- Various notation mistakes, causing significant confusion (see Typos and Notation Mistakes)
- Table 5 is highly subjective without any supporting empirical evidence

#### Typos and Notation Mistakes
- $r$ is reused with different meanings in line 107 and 108
- At line 108, the hypernetwork maps a layer-specific description vector to the entire parameters of the target network
- $H_\phi$ is defined three times with different meanings (line 108, 130, and 142)
- $z$ is defined with two different meanings (line 108 and 132)
- $\theta$ is defined with two different meanings (line 109 and 146)
- $j$ is not defined in Eq. 4 making it illegible
- line 175: 'HyprLoRA' should be 'HyperLoRA'
- 'Zhyper-full' is never defined but used at line 186
- line 460-461; opening parentheses should have a preceding white space

Overall, I believe that there are several major concerns in the current version of the submission and, therefore, I recommend reject. 

[1] Charakorn, Rujikorn, et al. "Text-to-LoRA: Instant Transformer Adaption." ICML 2025

[2] Bałazy, Klaudia, et al. "Lora-xs: Low-rank adaptation with extremely small number of parameters." arXiv preprint 2024

[3] Kopiczko, Dawid J., Tijmen Blankevoort, and Yuki M. Asano. "Vera: Vector-based random matrix adaptation." ICML 2024

### Questions
- What does "conditioning the LoRA weights" (line 17) mean?
- What does "large contextual spaces" (line 50) mean?
- How come the hypernetwork can "induce descriptive information" (line 94)?

### Soundness
2

### Presentation
1

### Contribution
1
