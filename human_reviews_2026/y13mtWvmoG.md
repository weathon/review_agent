# Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Large Multimodal Models (LMMs) often rely on in-context learning (ICL) to perform new visual question answering (VQA) tasks with minimal supervision. However, ICL performance, especially in smaller LMMs, does not always improve monotonically when increasing the number of examples. We hypothesize that this happens because the LMM is overwhelmed by extraneous information in the image embeddings that is irrelevant to the downstream task. To address this, we propose a meta-learning approach that induces few-shot capabilities in LMMs through a fixed set of soft prompts distilled from task-relevant visual features, which are adapted at test time using a small number of examples. We facilitate this distillation through an attention-mapper module that can be easily integrated with any LMM architecture and is jointly learned with soft prompts. Evaluation on the VL-ICL Bench shows that our method successfully achieves task adaptation in low-data regimes with just a few gradient steps, outperforming  ICL by 21.2%.  Comparisons with parameter-efficient finetuning methods demonstrate that meta-learning further enhances this adaptation by 7.7% for various VQA tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a meta-learning–based approach for few-shot adaptation in Large Multimodal Models (LMMs). They observe that training-free in-context learning often yields inconsistent performance in smaller LMMs and fails to scale reliably with more shots. To address this, they introduce learnable soft prompts distilled from image features using a first-order meta-learning framework. Additionally, they replace the standard shallow visual–language projector with a multi-layer self-attention “attention-mapper” module, enabling richer interactions between visual embeddings and prompt tokens before projection into the language space.

### Strengths
The paper is well written and clearly explained. It proposes a parameter-efficient approach to fine-tuning Large Multimodal Models (LMMs) using simple yet effective ideas such as learnable soft prompts and meta-learning. The work is timely and highlights an important limitation of standard in-context learning, namely its inconsistent scaling with increasing shots. The proposed method shows strong empirical performance, achieving substantial improvements over standard in-context learning and other baselines on few-shot VQA tasks with LMMs of up to 7B parameters.

### Weaknesses
1. Limited technical novelty: The paper borrows heavily from existing approaches. The use of soft or learnable prompts is already well established in few-shot learning and LLM adaptation [1, 2, 3]. Moreover, the meta-learning framework and the attention-mapper module are directly borrowed from Antoniou et al. (2019) and Najdenkoska et al. (2023), respectively.

- [1] Hou et al., MetaPrompting: Learning to Learn Better Prompts, COLING 2022. 

- [2] Wang et al., Towards Unified Prompt Tuning for Few-shot Text Classification, EMNLP Findings 2022.

- [3] Khattak et al., Self-Regulating Prompts: Foundational Model Adaptation Without Forgetting, ICCV 2023.

2. Latency and efficiency: Although the method demonstrates strong few-shot performance, one of the main advantages of training-free in-context learning is its low latency. The paper does not provide any analysis or discussion of the computational cost or response-time impact of meta-learning-based adaptation.

3. Insufficient explanation of motivation: The authors claim that additional soft prompts mitigate the issue of overwhelming image embeddings in smaller LMMs, but this is not analyzed beyond aggregate performance results. A visualization of attention patterns could have helped verify whether the prompts actually improve focus. Moreover, it seems counterintuitive that adding more embeddings (via soft learnable prompts) and information would resolve a capacity limitation.

4. Limited scalability evaluation: While the method is tested on multiple LMMs, all models are limited to 7B parameters. It remains unclear how well the approach scales to larger backbones (e.g., 13B or 34B) or whether the benefits persist at that scale.

5. Potential over-claiming in contributions: The contributions section slightly overstates novelty. MAPD largely reuses the first-order meta-learning framework of Antoniou et al. (2019), applied to prompt adapters. Similarly, the authors did not clearly distinguish their flexible adapter design from that of Najdenkoska et al. (2023). Although the authors claim to have proposed the mapper (inspired by Najdenkoska et al.), it appears to be essentially the same module.

### Questions
I am looking forward to the authors addressing the weaknesses. In particular, I would appreciate them answering the following questions:

1. How does the proposed attention-mapper differ from the adapter module introduced by Najdenkoska et al. (2023)?
2. How does the meta-learning–based adaptation affect latency or response time of the LMMs compared to standard training-free in-context learning?
3. Does the proposed approach scale to larger LMMs (e.g., 13B or 34B), and do the performance gains persist at that scale?
4. How does adding more soft embeddings (via learnable prompts) help reduce the issue of overwhelming image information in smaller LMMs?

### Soundness
3

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
4

### Summary
The paper presents MAPD, a meta-learning framework that enables few-shot adaptation in large multimodal models (LMMs) for visual question answering (VQA) by distilling task-relevant visual features into a fixed set of soft prompts via an attention-mapper integrated into the projection layer. Unlike in-context learning (ICL), which can degrade in performance with more examples in smaller LMMs due to irrelevant image token noise, MAPD leverages first-order MAML to train the attention-mapper and prompts on meta-tasks constructed from support/query splits. At test time, it adapts to new VQA tasks with a few gradient steps on the support set. Evaluated on the VL-ICL Bench, MAPD consistently outperforms ICL and parameter-efficient fine-tuning methods (e.g., LoRA, TPT), with performance improving monotonically as the number of shots increases.

### Strengths
1. Novel and effective integration of meta-learning with multimodal prompt distillation: MAPD is the first approach to apply MAML-style bi-level optimization to distill visual features into soft prompts for LMMs.  Its lightweight attention-mapper (~24M trainable parameters) enables efficient task adaptation, achieving state-of-the-art few-shot VQA performance across diverse VL-ICL tasks, while scaling reliably with support set size.
2. Rigorous experimental design with strong baselines and reproducibility: The authors provide comprehensive ablations (Multi-TaskPD, In-ContextPD, NoMeta-taskPD, Model-AvgPD) and leverage public code, datasets, and a standardized benchmark.  The observed monotonic improvement with increasing shots, coupled with substantial gains over ICL and PEFT methods, offers compelling evidence of MAPD’s practical superiority in low-data regimes.

### Weaknesses
1. The paper attributes the non-monotonic improvement in ICL performance of small-parameter LMMs to "irrelevant information interference in image embedding," but it does not rule out other contributing factors, such as the inherent limitations of small models or deviations in instruction understanding.
2. While much prior work has focused on optimizing the projection layer of LMMs, the advantages of this paper relative to existing methods appear limited. his study focuses on scenarios involving small models, single images and limited samples. Does this narrow scope limit its applicability? Additionally, the paper claims that the approach can be "easily incorporated into the projection layer of any LMM architecture," but it remains unclear whether it is effective for other architectures, such as variants of models like Qwen3-VL.

### Questions
Please refer to weakness

### Soundness
3

### Presentation
3

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
This paper proposes a test-time fine-tuning method to enhance the accuracy of MLLMs in visual question answering (VQA) compared to in-context learning. The proposed method introduces an attention-based mapping network on top of visual tokens that utilizes soft prompt tokens to distill visual information into soft prompts that are then provided to the LLM. The mapping network is pretrained, then fine-tuned on meta-tasks using meta-learning techniques, and finally is fine-tuned for each task during test-time for few-shot VQA. The paper also constructs several alternative methods for fine-tuning the proposed mapping network for comparison with the meta-learning technique. The results show that the proposed method can outperform in-context learning in few-shot VQA.

### Strengths
The proposed method is novel, and shows promising improvements on few-shot VQA. The experiments also provide several interesting insights about the role of attention mapper, soft prompts, and various fine-tuning strategies.

### Weaknesses
1. The description of how meta tasks are constructed lacks clarity. The paper must provide a clear table describing the number and composition of meta tasks used for fine-tuning and test-time fine-tuning. It is therefore hard to understand how the test accuracies are computed and can be compared.

2. Looking at the Appendix, it seems that different baselines are fine-tuned with different composition of meta-tasks (Appendix A.1.3): e.g., MAPD has 10-10 (support-query), whereas Multi-Task has 5-5, and In-Context has 10-1. No explanation is provided for these choices, and how they might affect the reported results.

3. The choice of bounding the number of test-time fine-tuning iterations (K<=30) seems arbitrary and makes the results of comparisons unreliable. For example, without any other information, it is possible that at K=40, which may take only a few more minutes, another method outperforms MAPD or the ranking of the methods changes completely. Comparing how long it takes each method to reach a certain accuracy level would be a more clear comparison.

4. The paper seems to miss the simple baseline of test-time fine-tuning just the connector (mapper MLP) of other open-source SOTA MLLMs (LLaVA-OV and Qwen-VL). This is critical to show whether the proposed adaptor matters in practice.

5. The results in Table 9 of Appendix show that ICL is still the SOTA on 3 out of the 4 tasks in VL-ICL when used with other MLLMs (LLaVA-OV and Qwen-VL). These results show that the best current method for few-shot VQA is still performing ICL with SOTA MLLMs. This limits the practical relevance of the proposed method.

6. All the reported results lack confidence intervals, so it is unclear how statistically significant the differences are. I suggest reporting binomial confidence intervals for accuracy to clarify this.

7. The paper seem to not report the time and computation over-head of its method.

### Questions
1. Can you provide more details about the number of meta-tasks, their composition, and why they differ between different methods?

2. Can you provide a baseline of fine-tuning the mapping network of LLaVA-OV, Qwen-VL?

3. Can you provide confidence intervals for the results?

4. Can you report time overhead of your method compared to ICL?

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
5

### Summary
This paper proposes a meta learning based approach for adding few shot capabilities to LLMs. The paper gives a convincing argument why this is needed and also provide a comprehensive set of baseline methods to show how the new method compares to those. Claim is that through meta learning the method is not hampered by the problem of others methods that they cannot leverage the longer context with more images. There is an extensive set of experiments both in the paper and in the appendix. State-of-the-art results are obtained.

### Strengths
- The paper is very clear and comprehensive. Experiments are well defined and serve a clear purpose.
- New approach to the problem which gives clear improvement in the results. 
- Good ablation study to analyze specific aspects of the methods.

### Weaknesses
- The paper indicates that existing methods cannot leverage the benefits of having longer contexts. This is clearly demonstrated through experimentation. But some more theoretical and intuition basis for this could be given. 
- The method can be applied to any existing method but only shown for a limited number of models. It is also not made explicit what is required for a model to be able to apply your method. You are using prompts and not every model is using exactly the same type of prompts.

### Questions
Please see the two elements indicating for weaknesses and reflect on those.

### Soundness
3

### Presentation
3

### Contribution
3
