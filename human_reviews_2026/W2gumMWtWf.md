# CoSA: Compressed Sensing-Based Adaptation of Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Parameter-Efficient Fine-Tuning (PEFT) has emerged as a practical paradigm for adapting large language models (LLMs) without updating all parameters. Most existing approaches, such as LoRA and PiSSA, rely on low-rank decompositions of weight updates. However, the low-rank assumption may restrict expressivity, particularly in task-specific adaptation scenarios where singular values are distributed relatively uniformly. To address this limitation, we propose CoSA (Compressed Sensing-Based Adaptation), a new PEFT method extended from compressed sensing theory. Instead of constraining weight updates to a low-rank subspace, CoSA expresses them through fixed random projection matrices and a compact learnable core. We provide a formal theoretical analysis of CoSA as a synthesis process, proving that weight updates can be compactly encoded into a low-dimensional space and mapped back through random projections. Extensive experimental results suggest that CoSA provides a principled perspective for efficient and expressive multi-scale model adaptation. Specifically, we evaluate CoSA on 10 diverse tasks including natural language understanding and generation, employing 5 models of different scales from RoBERTa, Llama, and Qwen families. Across these settings, CoSA consistently matches or outperforms state-of-the-art PEFT baselines while requiring over 68.4% fewer trainable parameters than LoRA and PiSSA.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CoSA (Compressed Sensing Adapter), a compressed sensing-based adapter architecture for parameter-efficient fine-tuning. Unlike low-rank approaches, CoSA expresses weight updates in a compressed form using a learnable core matrix \( Y \) and fixed random projection matrices \( L \) and \( R \). The method is theoretically motivated by the Restricted Isometry Property (RIP), which is used to justify both training stability and the preservation of expressivity.

### Strengths
- The idea of applying compressed sensing to parameter-efficient fine-tuning is interesting.
- The paper is clearly written and easy to follow.
- There is an effort to ground the method in compressed sensing theory, including the use of the Restricted Isometry Property (RIP).

### Weaknesses
- The proposed method lacks novelty. Several works have already explored tri-matrix adapter structures. In particular, TLoRA [1] (in arxiv) presents a structurally identical approach, using frozen random matrices $A$ and $C$, and a learnable small matrix $B$.  Additionally, PMSS [2], which trains frozen A, B, and learnable cores in the same way, but with different initialization methods, was proposed in COLING'25. The authors did not provide a sufficient comparison of these tri-matrix adapters.

- Although the paper claims that RIP leads to stable training, it is unclear whether such constraints are always beneficial. Since RIP restricts the amount of change after projection, it could potentially limit the expressivity of the model during fine-tuning, especially in low-data or few-shot scenarios where greater flexibility might be required. For instance, LoRA-GA [2] aims to better approximate full fine-tuning by mimicking full gradients, while CoSA instead constrains the update space, which may actually hinder learning. While the paper emphasizes the benefits of RIP, it lacks concrete theoretical, empirical, or quantitative evidence to support this claim.

- The set of baseline comparisons is too narrow. The paper does not compare CoSA against recently proposed methods such as NoLA [3] and VeRA [4], which also use frozen/random bases or $AB$-structured adapters. In addition, there is no experimental comparison with TLoRA, which appears to be the most closely related work. Such comparisons are essential to fairly assess CoSA’s effectiveness.

- Unlike standard LoRA, CoSA introduces \( ab \) parameters. Therefore, to ensure fair comparison with LoRA, the number of trainable parameters should be explicitly reported in each experiment. For example, in the NLG task with LLaMA, the paper mentions using (a, b) = (1024, 256), while the LoRA rank is set to 128. Since LLaMA-3B has a hidden dimension of 2048, this results in:
    - LoRA: 2048 × 128 × 2 parameters
    - CoSA: 1024 × 256 parameters
    
Therefore, CoSA uses about half the parameters compared to LoRA. However, other methods such as VeRA [4] or Vb-LoRA [5] introduce even fewer parameters, which makes CoSA to be less contributed. Therefore, the paper should clearly report the exact parameter counts and include comparisons against a wider range of parameter-efficient baselines.



---

>[1]Islam, Tanvir. "TLoRA: Tri-Matrix Low-Rank Adaptation of Large Language Models." arXiv preprint arXiv:2504.18735 (2025).
>
>[2] PMSS: Pretrained Matrices Skeleton Selection for LLM Fine-tuning, COLING, 2025
>
>[3]Wang, Shaowen, Linxi Yu, and Jian Li. "Lora-ga: Low-rank adaptation with gradient approximation." Advances in Neural Information Processing Systems 37 (2024): 54905-54931.
>
>[4] Koohpayegani, Soroush Abbasi, et al. "NOLA: Compressing LoRA using Linear Combination of Random Basis." The Twelfth International Conference on Learning Representations. 2024
>
>[5] Kopiczko, Dawid Jan, Tijmen Blankevoort, and Yuki M. Asano. "VeRA: Vector-based Random Matrix Adaptation." The Twelfth International Conference on Learning Representations. 2024
>
>[6] Li, Yang, Shaobo Han, and Shihao Ji. "Vb-lora: Extreme parameter efficient fine-tuning with vector banks." Advances in Neural Information Processing Systems 37 (2024): 16724-16751.

### Questions
- In Table 1, the paper claims that the storage requirement is $\mathcal{O}(1)$, but the learned core matrix $Y \in R^{a \times b}$ still needs to be stored. To me, this suggests that the storage complexity should be $\mathcal{O}(ab)$, not $\mathcal{O}(1)$. If I’m misunderstanding something, clarification would be appreciated. Moreover, in the NLG task, it seems that $a, b$ can be as large as 1024, which is not negligible in practice.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Compressed Sensing–based Adaptation (CoSA), a PEFT method for LLMs.

Inspired by compressed sensing thoery, the general idea of this work is to parameterize every weight update as are fixed random projection matrices and only the compact core is trained. 

CoSA is compared against LoRA, AdaLoRA, and PiSSA on GLUE and on math/code generation, on top of Llama-3.2-1B, Llama-3.1-8B, Qwen2-7B, showing competitive or better accuracy with substantially fewer parameters.

### Strengths
+ This paper is overall well-written and clearly-presented, making the readers easy to follow.

+ The proposed method shows a clear parameter and memory benefits over LoRA, AdaLoRA, PiSSA.

+ The ablation study is extensive.

### Weaknesses
- The technique soundness is open to doubt, at least in its current form. For example, the framing is not tied to an actual sparsity prior or to constraints. Besides, there is no theory level proof to justify the stability guarantees.

- The core idea to fix random $L$, $R$ and learn a compact core is not sufficiently distinguished from VeRA and/or other related random-projection PEFT methods, making the contribution to the community difficult to justify.

- This paper does not provide a theory-level proof on the emperical risk bound of either sparisty or the regularized training.

- The compared state-of-the-art PEFT methods are significantly missing. Some more recent and much stronger PEFT methods are mssing for comparison, for example:

[1] DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024.

[2] VeRA: Vector-based Random Matrix Adaptation. ICLR 2024.

[3] Foura: Fourier low-rank adaptation. NeurIPS 2024.

[4] SSH: Sparse Spectrum Adaptation via Discrete Hartley Transformation. NAACL 2024.

- In the $(a, b)$ ablation, is the rank $r$ rigorously matched? Please clarify. 

- If comparing with these more recent PEFT methods, the performance of the proposed method is rather limited and even inferior. 

- The experimental validation, to be honest, is rather limted. The authors only validate on two benchmarks, where GLUE is already out-of-date. It should be benchmarked on more recent yet more challenging instruction-tuning or multi-task mixture benchmarks, like [1-4] do. 

- Still regarding the performance, this paper lacks a convincing discussion on the performance variance caused by the random seeds.

- The training time and latency is not either reported or compared. 

- Some typos and presentation issues still remain.

### Questions
Please refer to the weakness section, and address them point-by-point.

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
This paper introduces a new PEFT method for LLMs. The authors argue that the low-rank assumption in LoRA limits expressivity. Inspired by compressed sensing, they propose CoSA, which treats the target weight update matrix as sparse. CoSA employs frozen projection matrices as the sensing matrices and fine-tunes a lower-dimensional measurement matrix. The key contribution is framing the compression of the target weight update matrix through the lens of compressed sensing. The authors also prove that the frozen projection matrices L and R satisfy RIP with high probability. Experiments show that CoSA matches or outperforms strong PEFT baselines while using over 68% fewer trainable parameters, with consistent improvements across NLU and reasoning/code benchmarks.

### Strengths
1. Viewing the PEFT problem through the lens of compressed sensing is an interesting and novel perspective.
2. The writing and presentation is clear.
3. The experiments are comprehensive and include tasks of different domains.

### Weaknesses
1. The proposed approach substantially overlaps with existing methods such as Tied-LoRA and VeRA [1,2], yet the paper makes no mention of them. Both Tied-LoRA and VeRA also employ frozen random matrices as down- and up-projection matrices, making it unclear how CoSA differs conceptually or empirically from these prior works.
2. The claim of O(1) complexity for CoSA in Table 1 appears inaccurate. Given the formulation, the complexity should be O(ab).
3. The method assumes that the target weight update matrix is sparse, which may not hold in practice. The authors should provide justifications for this sparsity assumption.
4. Theorem 1 offers only a superficial guarantee that the Kronecker product of two sensing matrices satisfies RIP with high probability. This result does not provide deeper insights into why the proposed approach should work better than existing PEFT methods.
5. The baseline comparisons are limited. Stronger and more recent baselines such as DoRA are not included. It would also strengthen the paper to evaluate CoSA on instruction-tuning tasks to demonstrate its generality.

References
1. Tied-LoRA: Enhancing parameter efficiency of LoRA with Weight Tying
2. VeRA: Vector-based Random Matrix Adaptation

### Questions
1. Why would a sparsity assumption work better for the $\Delta W$ than low rank assumption?
2. Is there any computational overhead from the additional matrix multiplication of CoSA compared to LoRA?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present a new PEFT method inspired by compressed sensing. They parameterize the update as a sequence of three matrices 
deltaW = L Y R, where L and R are independent random matrics and Y is learnable core matrix. Using random projections reduces the number of training parameters and their RIP property ensures that training is not destabilized. The resutls show improvements over LoRA based methods showing that there are cases where Low-rank is not a correct hypothesis over updates.

### Strengths
1. The observation that Low rank is not always a good hypothesis is valuable (although it appears in some recent works)
2. The paper is well written and generally a good read with discussion around compressed sensing,etc

### Weaknesses
1. Lack of baselines ( and hence related work)  (my main concern is this)

The experiments are okay (benchmark wise) but are lacking baseline wise. For instance, very similar and more recent PEFT baselines are excluded. SketchTune, for instance is also based on sketching matrices (a special case of projection matrices which also have RIP property). Also, some other baselines such as S2FT etc are missing. It is important to compare against these methods to ensure that we are indeed making progress in PEFT domain. 

Zhang, Tianyi, Junda Su, Aditya Desai, Oscar Wu, Zhaozhuo Xu, and Anshumali Shrivastava. "Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation." arXiv preprint arXiv:2410.06364 (2024).

Yang, Xinyu, Jixuan Leng, Geyang Guo, Jiawei Zhao, Ryumei Nakada, Linjun Zhang, Huaxiu Yao, and Beidi Chen. "S $^{2} $ FT: Efficient, scalable and generalizable LLM fine-tuning by structured sparsity." Advances in Neural Information Processing Systems 37 (2024): 59912-59947.

2. The current formulation also is low-rank (rank = min(a,b)). Am i missing something?
       a. why do you expect CoSA to handle cases when deltaW is not low rank
       a. related but different, can authors elaborate how does CoSA provide extra expressive power.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
