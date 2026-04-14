# TorchTitan: One-stop PyTorch native solution for production ready LLM pretraining

- Decision: Accept (Poster)
- Scores: 10, 6, 6, 6, 6, 5

## Abstract
The development of large language models (LLMs) has been instrumental in advancing state-of-the-art natural language processing applications. Training LLMs with billions of parameters and trillions of tokens requires sophisticated distributed systems that enable composing and comparing several state-of-the-art techniques in order to efficiently scale across thousands of accelerators. However, existing solutions are complex, scattered across multiple libraries/repositories, lack interoperability, and are cumbersome to maintain. Thus, curating and empirically comparing training recipes requires non-trivial engineering effort.

This paper introduces **TORCHTITAN**$^1$, a PyTorch-native distributed training system that unifies and advances state-of-the-art techniques, streamlining integration and reducing engineering overhead. TORCHTITAN enables seamless application of 4D parallelism in a modular and composable manner, while featuring elastic scaling to adapt to changing computational requirements. The system provides comprehensive logging, efficient checkpointing, and debugging tools, ensuring production-ready training. Moreover, TORCHTITAN incorporates innovative hardware-software co-designed solutions, leveraging cutting-edge features like Float8 training and SymmetricMemory to maximize hardware utilization.

As a flexible experimental test bed, TORCHTITAN facilitates the curation and comparison of custom recipes for diverse training contexts. By leveraging TORCHTITAN, we developed optimized training recipes for the Llama 3.1 family and provide actionable guidance on selecting and combining distributed training techniques to maximize training efficiency, based on our hands-on experiences.

We thoroughly assess TORCHTITAN on the Llama 3.1 family of LLMs, spanning 8 billion to 405 billion parameters, and showcase its exceptional performance, modular composability, and elastic scalability. By stacking training optimizations, we demonstrate accelerations ranging from 65.08% on Llama 3.1 8B at 128 GPU scale (1D), 12.59% on Llama 3.1 70B at 256 GPU scale (2D), to 30% on Llama 3.1 405B at 512 GPU scale (3D) on NVIDIA H100 GPUs over optimized baselines. We also demonstrate the effectiveness of 4D parallelism in enabling long context training.

$^1$ GitHub: [https://github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper introduces LEGOSCALE, an open-source, PyTorch-native distributed training system that unifies and advances state-of-the-art techniques, streamlining integration and reducing engineering overhead. LEGOSCALE enables 3D parallelism in a modular and composable featuring elastic scaling to adapt to changing computational requirements. By providing an accessible and extensible platform, LEGOSCALE democratizes large language model pre-training, empowering a wider range of researchers and developers to tap into the potential of LLMs and accelerate innovation in the field.

### Strengths
Thank you for submitting this paper to ICLR! I loved reading this paper.

1. Important problem. LLM training is challenging and resource-consuming.

2. Very graceful, interesting and timely solution. Pytorch-native is what the community needs, which is different with existing systems like Megatron.

### Weaknesses
1. Too many modules are introduced in limited pages, which is not friendly to general readers without machine learning system background.

2. Limited model support. I would like to see more application on MoE model, Multi-modal LLM, etc.

3. 3D-parallelism is not comprehensive / state-of-the-art at the time of 2024. Consider adding more support on context parallel (also mentioned in paper), expert parallel, etc.

### Questions
Although this work is more of an engineer's contribution than a research novelty contribution, I still think it is important and should be accepted by ICLR. There are some questions I would like to see more discussion of.

1. For Pytorch native training framework, there is another framework implemented by Bytedance: https://github.com/volcengine/veScale. Could you compare the differences between them? It seems they support more parallisms.

2. I would like to see some results of end-to-end throughput comparison with the latest Megatron, Deepspeed.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This submission comprises a technical report / white paper about the development of LegoScale, a pytorch-based solution for scalable LLM training in distributed settings. The proposed framework extends the capabilities of vanilla pytorch for LLM training in many aspects, ranging from improved scalability through a principled combination of numerous optimised parallelism techniques and memory management, to improved logging and debugging mechanisms.

### Strengths
- The proposed framework offers a streamlined approach for LLM training across distributed systems with different scales of compute. This scalability is achieved through a principled and well-engineered framework that allows tunable exploitation of different parallelism dimensions and other optimisations. This abstracts tons of complexities from deep learning practitioners. 
- Evidently, the majority of design choices in the proposed framework have relied on insights and experience from both the LLM training and the GPU deployment communities.
- The reported experiments demonstrate that the proposed framework can achieve notable speed-ups at training time.

### Weaknesses
- It is difficult to identify much research novelty in the proposed framework, since its contribution can mostly be found in the design and tons of non-trivial engineering efforts behind LegoScale. 
- It is unclear whether this contribution will last or may wind down due to enhancements of the original pytorch framework towards the same direction, or lack of maintenance on the proposed framework which may lead to lack of support from latest pytorch versions or for latest GPU models.

### Questions
This review is intentionally short and high-level, due to the technical excellence, but also lack of research novelty, of the proposed system. 
In my opinion, white papers of successful training frameworks can still be published in ML venues; but will be willing to adjust my score considering the outcome of the upcoming discussion with other reviewers'. 

A piece of information that can facilitate this process is any insights that the authors have obtained and can share with us about the usage of their framework from the community (without violating the double-blind process). Has the framework been open-sourced? How many people are actively contributing to it? How many people are currently using it? Are the reported results easily replicated out-of-the-box on different clusters? Is long-term support and maintenance of the framework part  of the plans of the authors?

Post rebuttal edit:
 I am increasing my score from 5 to 6. Although I maintain the position that the novelty in the manuscript is limited, I believe that the proposed framework can have notable impact to the community, and as such it is worth being presented to an ML venue in my opinion.

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
4

### Summary
This paper describes a feature-rich PyTorch native hardware-software system for efficiently training of LLMs.

### Strengths
The paper describes a useful tool that addresses an important problem.

The tool substantially improves training time.

The system appears to have sufficient features for production use.

The paper has some tutorial value in that it briefly explains a wide range of features appropriate for pre-training systems.

I think this paper's greatest value is in advertizing the pre-training system to researchers and developers who may want to use it.

### Weaknesses
There are few new ideas. The focus is on  making a powerful tool by building on existing ideas. However, this weakness is partially offset by the fact that the work may enable others to develop new ideas by reducing the time required for pre-training.

Interest will be bimodal: those who need to pre-train LLMs will be very interested, especially if they have access to the pre-training system, and those who do not will probably find little of interest in the paper.

There is not a strong distinction between engineering efforts to add features to the pre-training system and research efforts to develop new system design ideas. The paper explains the design of the tool but does little to justify design decisions. Improvements to writing, alone, might address these problem.

With the exception of the sensitivity analysis in the tables on page 8, the paper does not contain scientific findings.

Figures 1 and 2 are essentially impossible to read without magnification, which circumvents conference page limits.

### Questions
Is there a plan to make the software components of the pre-training system available to other researchers?

What do you view as the most novel new ideas within the design of your pre-training system?

### Soundness
4

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
5

### Summary
The paper presents LegoScale, a framework that enables easy comparison and composition of different LLM training functionalities and optimizations. LegoScale can significantly improve the productivity of model scientists and practitioners by reducing the engineering burden of model development. LegoScale incorporates Pytorch-native implementation of popular techniques like mixed precision, 3D parallelism, ZeRO, activation checkpointing, model checkpointing, logging, etc. The evaluation studies usage of LegoScale for training Llama-3 model family and demonstrates model and hardware scaling scenarios by composing 3D parallelism, mixed precision, and compiler optimizations.

### Strengths
- LegoScale tackles an important problem of making it easier for model practitioners and researchers to compare and compose SOTA training techniques. Advanced LLM training techniques are typically scattered across different frameworks making it difficult if not impractical for practitioners and researchers to leverage these techniques. LegoScale could be a very valuable tool for the community. 
- By incorporating pytorch-native implementations of training optimizations, LegoScale lowers the bar for integration into existing pytorch workflows, likely fostering adoption. 
- The evaluation results highlight intuitive compositions of relevant optimizations for model and hardware scaling, which helps the user appreciate the developer benefits of LegoScale.

### Weaknesses
- Considering the growing importance of context scaling, to millions of tokens, LegoScale seems to be missing an important scenario. Although, this is discussed as ongoing work, it is unclear how context scaling fits into LegoScale. This is because the current tensor abstraction focuses on model partitioning, whereas SOTA context scaling optimizations such as Ulyssess involve input partitioning. 
- While LegoScale allows composition of various optimizations, it is unclear how users will be guided to make optimal choices in terms of performance metrics such as throughput, latency, or hardware utilization. 
- While the evaluation does a good presenting the scaling performance results, it is missing results of the production-ready-training features (2.3). For example, it would be useful to show convergence plots demonstrating that effectiveness of parallelism changes using DCP.
- The performance evaluation lacks a common baseline metric across Tables 1-4, which makes it difficult to understand LegoScale efficiency across model and hardware scaling dimensions. While the authors rightly identified how different precisions hampers MFU usage, I think a good compromise is to report bfloat16 and float8 MFUs separately.

### Questions
1. Can you clarify whether the throughput results in Tables 1-4 are per GPU or aggerate.
2. How are the optimization configurations (e.g., PP vs FSDP, AC vs SAC) used in Tables 1-4 obtained? Was there some exploration of the configuration space to discover optimal settings?
3. Can you clarify that LegoScale applies to other model families besides Llama? In particular, line 279 refers to Llama model when discussing the torch.compile integration (2.2.2). 
4. What ZeRO stages are the FSDP configurations in Tables 1-4?
5. HSDP is not mentioned in Figure 1 or 2, although mentioned as an important FSDP option. Is HSDP not available in LegoScale?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces `LegoScale`, an open-source, PyTorch-native distributed training system designed to unify and enhance current engineering efforts in LLM pre-training. `LegoScale` improves training efficiency through a modular, composable design that supports various parallelism techniques, including Fully Sharded Data Parallel (FSDP), Tensor Parallel (TP), and Pipeline Parallel (PP), as well as memory optimization methods such as FP8 precision and activation checkpointing. It also integrates with `torch.compile` for further optimizations. Experiments on the Llama 3.1 family of LLMs demonstrate that `LegoScale` accelerates training by 65.08% on the Llama 3.1 8B model at a 128-GPU scale (1D), 12.59% on the Llama 3.1 70B model at a 256-GPU scale (2D), and 30% on the Llama 3.1 405B model at a 512-GPU scale (3D), all on NVIDIA H100 GPUs, compared to optimized baselines.

### Strengths
This paper is well-organized and clearly presents the design of  `LegoScale`  across parallelism strategies, memory optimization techniques, and integration with  `torch.compile`. While prior research has addressed various aspects of LLM training, this paper delivers a comprehensive, feature-complete open-source solution that focuses on key challenges, including:

1.  Composability of parallelism techniques,
2.  Extensibility for incorporating new optimizations,
3.  Efficient hardware utilization, and
4.  Production-ready features.

The figures, particularly Figures 1 and 2, effectively illustrate how the system supports parallelism and customization features.

Notably, the authors provide clear abstractions for N-D parallelism in training and analyze strategies for maximizing efficiency when combining parallelism methods. The experiments demonstrate promising performance improvements compared to strong baselines. Beyond these contributions, the engineering effort invested in this project is substantial and valuable for future work in this area.

### Weaknesses
The proposed training system builds extensively on prior work, especially in parallelism design and memory optimization. While the tensor abstraction is novel and more expressive, the primary contribution is from an engineering perspective. Additionally, possibly due to page limitations, much of the detail is relegated to the appendix. There are a few areas where the paper could strengthen its credibility.

First, the paper could benefit from more discussion on how the modular design facilitates the integration of future optimizations and enables further parallelism without compromising performance.

For the PyTorch integration, although  `torch.compile`  is powerful, the paper lacks specifics on how it sustains high GPU efficiency in a highly parallel training environment—for example, insights on GPU usage.

An ablation study on the effect of individual features, such as transitioning from FSDP1 to FSDP2, and on performance changes as GPU scale increases within 1D, 2D, and 3D parallelism settings, would also help readers better understand the system's effectiveness.

**Minor issue:**  In line 315, "torch" is misspelled as "torchao."

### Questions
-   Could you elaborate on the improvements from FSDP1 to FSDP2, and how these enhancements impact different parallelism settings?
-   How does each parallelism strategy scale with the number of GPUs?
-   What does GPU utilization look like during training? Is GPU efficiency maximized?
-   Is this method extensible to other platforms, such as TPU clusters?
-   Could you provide more details on the system's fault tolerance and crash recovery design?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The proposed LEGOSCALE framework aims to enhance the training efficiency of the Llama 3.1 model in distributed training by integrating several established techniques, including n-D parallelism support, efficient checkpoint, compiler enhancement, debugging tools, and Float8 support. 
Authors evaluate the scaling performance of LEGOSCALE using the Llama 3.1 family of models (8B, 70B, and 405B) on scales from 8 to 512 H100 GPUs.  
The paper claims 65.08%, 12.59%, and 30% improvement on training speed on the 8B model with data parallel, 70B model with data- and tensor-parallel, and 405B model with 3D parallel, repsectively.

### Strengths
1. Using the latest Llama 3.1 family of models is promising
2. The scale with up to 512 H100 GPUs, though about two orders of magnitude smaller than the original Llama 3.1 paper, is sufficient.

### Weaknesses
1. The primary concern is that none of the proposed techniques is new or invented by the authors, significantly limiting LEGOSCALE's originality. Most of the performance improvement in Table 1-4 is attributed to the incorporation of existing solutions, not the new design of LEGOSCALE.
2. LEGOSCALE employs a composable multi-dimensional parallelism approach by combining Fully Sharded Data Parallel (FSDP), Tensor Parallel (TP), and Pipeline Parallel (PP) methods, which have been extensively validated in systems like DeepSpeed (https://github.com/microsoft/DeepSpeed), Megatron-LM, and GPipe for efficient model scaling.
3. Additionally, LEGOSCALE leverages hardware-software co-designed solutions such as Float8 mixed precision and optimized activation checkpointing to improve hardware utilization and memory efficiency. Most of the recent LLM training was done with BF16 and FP32. FP8 was rarely used due to practical concerns in optimization stability. To justify the innovation of FP8, the authors need to provide end-to-end training results and downstream task performance.
4. Furthermore, LEGOSCALE incorporates PyTorch’s Distributed Checkpointing (DCP) to enable efficient failure recovery and state persistence, building upon checkpointing strategies already prevalent in systems like ByteCheckpoint and Gemini. 
5. LEGOSCALE optimizes memory and computation efficiency through `torch.compile` regional compilation, a technique similarly found in frameworks like JAX and ONNX, which use compiler optimizations to enhance training throughput and resource utilization.

### Questions
Please address the points in weakness.

### Soundness
2

### Presentation
2

### Contribution
2
