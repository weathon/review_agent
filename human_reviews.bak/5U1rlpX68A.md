# SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning

- Decision: Accept (Oral)
- Scores: 8, 8, 6, 8

## Abstract
Continual Learning (CL) with foundation models has recently emerged as a promising paradigm to exploit abundant knowledge acquired during pre-training for tackling sequential tasks. However, existing prompt-based and Low-Rank Adaptation-based (LoRA-based) methods often require expanding a prompt/LoRA pool or retaining samples of previous tasks, which poses significant scalability challenges as the number of tasks grows. 
To address these limitations, we propose Scalable Decoupled LoRA (SD-LoRA) for class incremental learning, which continually separates the learning of the magnitude and direction of LoRA components without rehearsal. Our empirical and theoretical analysis reveals that SD-LoRA tends to follow a low-loss trajectory and converges to an overlapping low-loss region for all learned tasks, resulting in an excellent stability-plasticity trade-off. Building upon these insights, we introduce two variants of SD-LoRA with further improved parameter efficiency. All parameters of SD-LoRAs can be end-to-end optimized for CL objectives. Meanwhile, they support efficient inference by allowing direct evaluation with the finally trained model, obviating the need for component selection. Extensive experiments across multiple CL benchmarks and foundation models consistently validate the effectiveness of SD-LoRA. The code is available at https://github.com/WuYichen-97/SD-Lora-CL.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This manuscript focused on continual learning with the pre-trained model (CL-PTM). By indicating that the existing prompt-based methods rely on an unreliable prompt selection mechanism which can lead to the scalability issue, this manuscript proposed Scalable Low-Rank Adaptation (S-LoRA). Specifically, S-LoRA incrementally decouples the learning of the direction and magnitude of Low-Rank Adaptation (LoRA) parameters. The theoretical and empirical analysis indicated that the proposed method tends to follow a low-loss trajectory that converges to an overlapped low-loss region. Experiments on standard benchmarks were conducted to support the proposed method.

### Strengths
1. The idea of decoupling the learning of LoRA’s direction and magnitude is novel to me. This method addresses issues with scalability and efficiency in class-incremental learning, providing a valuable contribution to the field.
2. The motivation of this manuscript was driven by extensive empirical support. Some interesting findings were observed through the experiments, which were not investigated before.
3. Theoretical analyses were conducted to support the empirical findings. The explanation of the shared low-loss region and how S-LoRA converges to it effectively supports the results presented.

### Weaknesses
1. Some core ideas were not explained clearly. The description of how the decoupling of magnitude and direction need to be better defined.
2. While S-LoRA shows strong performance, some newer methods outside the scope of the prompt-based methods need to be compared for further validation.

### Questions
1. Regarding the magnitude and the direction of the LoRA parameters. In Section 4.1, the decomposition of magnitude and direction of LoRA's parameter updates $\Delta \mathbf{W}$ was only casually mentioned. However, I wonder how the magnitude and direction of such a matrix were defined in the authors' derivation. I believe they should be clearly defined to help the readers to understand the core idea of your method.
2. In Findings 3, the concept of "low-loss path", "linear low-loss path", and "low-loss region" need to be explained with more intuitive expressions.
3. In the first equation after Algorithm 1, I wonder if there is a typo for the definition of $r$. Should the first term in RHS be $\mathbf{A}_j \mathbf{B}_j$ rather than $\mathbf{A}_i \mathbf{B}_i$? If not, please provide further explanations about this point.
4. Still with the same equation. Since the RHS consists of a linear combination of "directions" mentioned by the authors, I wonder how "the residual" of a direction approximation should be defined. In my understanding, $r$ should also be a direction analog to the operations on vectors. However, the authors also had some expressions like "the residual $r$ is less than the threshold $\tau$". I didn't get how to control a residual direction by a scalar threshold.
5. Before Theorem 1 in Section 5, the authors mentioned that $\|\cdot\|$ refers to the operator norm. Do all $\|\cdot\|$ notations within the manuscript stand for the operator norm?
6. I noticed that the proposed S-LoRA method was mainly compared to the prompt-based baselines except for InfoLoRA. However, S-LoRA is not an improvement of the prompt-based methods. It would be better if the authors provide comparisons with a wide range of recent studies like EASE [1], or at least some recent prompt-based methods like [2-5].

I will accordingly change my rating if my concerns are addressed.

References:
[1] Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning. CVPR 2024.
[2] RCS-prompt: learning prompt to rearrange class space for prompt-based continual learning. ECCV 2024.
[3] Prompt Gradient Projection For Continual Learning. ICLR 2024.
[4] PromptFusion: Decoupling Stability and Plasticity for Continual Learning. ECCV 2024.
[5] Consistent Prompting for Rehearsal-Free Continual Learning. CVPR 2024.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a new method called S-LoRA for addressing the Class Incremental Learning (CIL) problem. Currently, prompt-based methods are the most effective method that does not rely on memory-buffer methods in CIL. However, as highlighted by the authors, these approaches have some limitations, particularly regarding prompt selection: they could be more efficient since they require double passes through the model to identify similar prompts and depend heavily on choosing the correct prompt. The proposed method (S-LoRA) aims to alleviate these issues by enabling each task to learn a set of LoRA weights over a pre-trained model, along with a separate set of coefficients that determine the importance of each LoRA-weight. The authors provide theoretical and empirical evidence demonstrating that their proposed method achieves strong results across multiple benchmarks.

### Strengths
- The paper is very well written. It presents the constraints and motivations the authors want to work with and clearly explains a simple but effective method. The experiments shown support the majority of what is presented in the paper.
- The S-LoRA method represents an advancement over current methods that use pre-trained models. It mitigates the issues introduced by prompt-based approaches and reduces the inference time. The idea of splitting the LoRA trainable weights into direction and magnitude is interesting, as it helps each task have the plasticity to add new information while encouraging the reuse of weights learned in past tasks.
    - The authors mentioned that an ideal CL method needs to be rehearsal-free, which I may partially agree with as this depends strongly on the context and scenario of the problem. However, I do agree that it needs to be more efficient at the time of inference.
- The experiments performed in Section 4 help to understand the proposed method's behaviour. It is also interesting to find limitations in the model and propose alternatives that mitigate these problems.
    - Moreover, they are well accompanied by a suitable experimental ablation.

### Weaknesses
- Some sections of the paper mention that the proposed method improves up to 31.05% across various CL benchmarks. It could be helpful to change this as it can present confusion, especially as this number does not necessarily relate to the current best method.
- Algorithm 1 needs to be more precise. In lines 1 and 2, you assume you are working in a Task_j, but then iterate over the T task in lines 6 to 8. This iteration may confuse readers and make them think that a task identifier conditions parameters alpha, A_j and B_j.

### Questions
- I agree with the first finding shown in section 4.2. However, this can change drastically when the input data distribution changes abruptly.
    - I understand that this may be outside the scope of the paper, but did you measure the relative distance with other benchmarks? It would be helpful to compare elements outside the ViT pre-training distribution.
- Can you explain the intuition behind the lack of forgetting in the proposed method?
    - For example, Figure 3.A shows how the alpha values change as different tasks are trained. I imagine this may strongly affect the representation generated for the first task examples, which are only expected to be affected by the weights of A_1 and B_1.
    - Do you have forgetting results?
- Some implementation details are not clearly explained:
    - How are alphas initialised? In most routing methods, the initialisation of the alphas plays a critical role; I can imagine that this can also happen here.
    - Does any regularisation apply? Or do you train them freely?
- Can you explain how the values in Table 5 were calculated?
    - Specifically, I have doubts about the comparison of L2P's trainable parameters versus ES-LoRA's. In L2P, only the prompts pool is trained, which should be smaller than LoRA's A and B.
    - On the other hand, in ES-LoRA, the A and B values are accumulated, which is not accounted for in the table.
    - Can you add the S-LoRA values for comparison?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Existing methods depend on the precision of the selection mechanism. This paper propose a class incremental learning (CLS) method called S-LoRA to decouple the learning of the direction and magnitude of LoRA parameters. It can incrementally learn task-specific LoRA and importance parameters while preserving the optimization directions of previously tasks. The experimental results show their superiority on multiple benchmarks.

### Strengths
1. The ablation study is interesting and validate the effectiveness of the learned importance factors $a$.
2. The paper is well organization and the proposed approach is easy followed.

### Weaknesses
1. My main concern is that the idea of this paper is just like a MOE-LoRA, which has been studied in [1-2] in continual learning. The learned importance parameters are just like the gates of MOE.  We expect the author to discuss more differences between the proposed "reweighting LoRA" and "MOE LoRA".
2. The experiments are insufficient. There are some other methods in CLS that should be compared, like RanPAC[3], and ADAM[4]. To the best of our knowledge, the RanPAC currently performs best among CIL methods, where the details can be found in this survey[8] about PTM-based CL.  Additionally, some SOTA LoRA methods for continual learning (e.g., O-LoRA[6] and N-LoRA[7]) should be also compared. These two works are very related to LoRA for continual learning, even though they are performed on language process tasks.
3. The two efficient versions of S-LoRA seem unrelated to the core idea of this paper, and the effect is not significant. The author should discuss more details about the ES-LoRA1 and ES-LoRA2. 

If the author can answer my issues well, I am willing to improve my score. 


Reference

[1] Liu J, Wu J, Liu J, et al. Learning Attentional Mixture of LoRAs for Language Model Continual Learning[J]. arXiv preprint arXiv:2409.19611, 2024.

[2] Dou S, Zhou E, Liu Y, et al. LoRAMoE: Alleviating World Knowledge Forgetting in Large Language Models via MoE-Style Plugin[C]//Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024: 1932-1945.

[3] McDonnell, M. D.; Gong, D.; Parvaneh, A.; Abbasnejad, E.; and van den Hengel, A. 2024. Ranpac: Random projections and pre-trained models for continual learning. Advances in Neural Information Processing Systems, 36.

[4] Zhou, D.-W.; Ye, H.-J.; Zhan, D.-C.; and Liu, Z. 2023b. Revisiting class-incremental learning with pre-trained models: Generalizability and adaptivity are all you need. arXiv preprint arXiv:2303.07338.

[6] Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang. 2023a. Orthogonal subspace learning for language model continual learning. arXiv preprint arXiv:2310.14152.

[7] Yang S, Ning K P, Liu Y Y, et al. Is Parameter Collision Hindering Continual Learning in LLMs?[J]. arXiv preprint arXiv:2410.10179, 2024.

[8] Continual Learning with Pre-Trained Models: A Survey.

### Questions
See the Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors propose to incrementally learn how to merge different LoRA models by keeping old LoRA modules fixed and only update their weights and a new LoRA module in the memory-free CL scenario. The authors compare their method S-LoRA as well as variants of it on multiple benchmarks and different benchmarks against multiple baselines.

### Strengths
The authors present a simply, but principled idea. It makes a lot of sense to learn to fuse models in the context of CL.

Besides the choices mentioned in "Weaknesses", the empirically evaluation is convincing.

### Weaknesses
The presented work uses ideas from model merging/fusing, e.g., "Editing Models with Task Arithmetic" by Ilharco et al., but doesn't discuss these works in the related work. Adoption of these ideas to CL, such as "Task Arithmetic with LoRA for Continual Learning", are not discussed or not compared to.

The authors compare apples with oranges. Most baselines use inferior PEFT methods (prompt tuning) while the authors use LoRA which naturally will give them an edge. It would be great if the authors could show that the improvement over the baseline is not coming from the PEFT method. See a discussion on this topic in "Choice of PEFT Technique in Continual Learning: Prompt Tuning is Not All You Need".

Table 1 mixes desired properties of CL methods with general properties of methods. End-to-end optimization is no desired property, high predictive performance is.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
