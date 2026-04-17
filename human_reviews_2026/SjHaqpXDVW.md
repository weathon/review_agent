# PlanMoGPT: Flow-Enhanced Progressive Planning for Text to Motion Synthesis

- Decision: Reject
- Scores: 8, 6, 2, 2

## Abstract
Recent advances in large language models (LLMs) have enabled breakthroughs in many multimodal generation tasks, but a significant performance gap still exists in text-to-motion generation, where LLM-based methods lag far behind non-LLM methods. We identify the granularity of motion tokenization as a critical bottleneck: fine-grained tokenization induces local dependency issues, where LLMs overemphasize short-term coherence at the expense of global semantic alignment, while coarse-grained tokenization sacrifices motion details. To resolve this issue, we propose PlanMoGPT, an LLM-based framework integrating progressive planning and flow-enhanced fine-grained motion tokenization. First, our progressive planning mechanism leverages LLMs' autoregressive capabilities to hierarchically generate motion tokens by starting from sparse global plans and iteratively refining them into full sequences. Second, our flow-enhanced tokenizer doubles the downsampling resolution and expands the codebook size by eight times, minimizing detail loss during discretization, while a flow-enhanced decoder recovers motion nuances. Extensive experiments on text-to-motion benchmarks demonstrate that it achieves state-of-the-art performance, improving FID scores by 63.8% (from 0.380 to 0.141) on long-sequence generation while enhancing motion diversity by 49.9% compared to existing methods. The proposed framework successfully resolves the diversity-quality trade-off that plagues current non-LLM approaches, establishing new standards for text-to-motion generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents PlanMoGPT, an LLM-based framework that addresses the trade-off between global coherence and motion detail in text-to-motion generation. It introduces a progressive planning mechanism that uses the LLM’s autoregressive abilities to generate motion tokens hierarchically—starting from sparse global plans and refining to full sequences—and a flow-enhanced fine-grained tokenizer that doubles temporal resolution and expands the codebook eightfold to reduce discretization loss. A flow-enhanced decoder further restores motion nuances. Experiments on standard benchmarks show state-of-the-art performance, with a 63.8% FID improvement on long-sequence generation (0.380 → 0.141) and a 49.9% boost in motion diversity, effectively resolving the diversity–quality trade-off that limits current non-LLM methods.

### Strengths
1. The paper points the granularity bottleneck in motion tokenization and tackles it with a coherent “plan-then-detail” pipeline—progressive planning for global-to-local consistency and a flow-enhanced fine-grained tokenizer to retain details, plus a flow-matching decoder to restore nuances.
2. Strong long-sequence performance: On newly built long-motion benchmarks, PlanMoGPT delivers large gains (FID 0.380→0.141, +49.9% diversity), effectively breaking the diversity–quality trade-off that hampers non-LLM approaches and showing excellent long-range semantic alignment.
3. Comprehensive and careful experimentation: The authors evaluate across standard datasets (HumanML3D, KIT-ML) and introduce two extended long-motion datasets (HumanML3D++, KIT-ML++) constructed via motion concatenation with GPT-4 text merging and human QC. They report extensive baselines (diffusion and token-based), ablations (codebook, flow vs residual VQ-VAE, plan intervals, text encoders), diversity–quality analysis, inference cost, and user studies.
4. Generality and robustness: The flow-enhanced tokenizer improves other frameworks (e.g., MoMask), indicating that the proposed tokenization/decoding scheme is transferable beyond their own LLM planner.

### Weaknesses
The paper primarily reports results with a single decoder-only LLM (TinyLLaMA-1.1B), but it lacks a systematic study across multiple LLM sizes and families (e.g. Qwen, Gemma).

### Questions
1. What is the length and the corresponding motion length of the motion interval?
2. What's the speed of your model? how many frames can you generate in one second on average?
3. Why do you choose TINY-LLaMA 1.1B as your base model?
4. Does your model support reasoning-driven motion generation?

### Soundness
3

### Presentation
4

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
This paper aims to solve the issue of the granularity of motion tokenization to improve the performance of text-to-motion. To relieve this issue, this paper introduces PlanMoGPT, an LLM-based framework integrating progressive planning and flow-enhanced fine-grained motion tokenization. Extensive experiments on HumanML3D, HumanML3D++, and KIT-ML++ demonstrate the effectiveness ofthe  proposed methods.

### Strengths
This paper focuses on the issue of the granularity of motion tokenization and introduces flow-matching into motion tokenization to propose flow-enhanced fine-grained motion tokenization. This paper also introduces progressive generation for an LLM-based motion generation model. Comprehensive ablation experiments demonstrate the effectiveness of the proposed method.

### Weaknesses
There are two experimental results in this paper that cannot support the contribution of the paper：
1. PlanMoGPT achieves suboptimal results on the KIT-ML dataset.
2. Introducing time interval 8 does not improve the text-to-motion performance, and time interval 6 leads to higher FID.

### Questions
1. Reducing the time sampling rate and introducing a multi-granularity time interval will cause the sequence to become longer. Do authors consider the issue of reduced generation efficiency due to longer sequences?
2. Since the author mentioned that PlanMoGPT's poor performance on KIT-ML is due to the small size of the dataset, have the authors tried training on a larger dataset, such as SnapMoGen[1] or Motion-X[2]?
[1] Guo C, Hwang I, Wang J, et al. SnapMoGen: Human Motion Generation from Expressive Texts[J]. arXiv preprint arXiv:2507.09122, 2025.
[2] Lin J, Zeng A, Lu S, et al. Motion-x: A large-scale 3d expressive whole-body human motion dataset[J]. Advances in Neural Information Processing Systems, 2023, 36: 25268-25280.

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
4

### Summary
The paper proposes PlanMoGPT, a LLM–based framework for text-to-motion generation. It identifies the local dependency problem in fine-grained motion tokenization as a key limitation of existing approaches and addresses it through a progressive planning strategy, where motion is generated from coarse global plans to fine-grained details, and a flow-enhanced motion tokenizer that improves motion representation and reconstruction. Experiments on their proposed datasets suggesting that PlanMoGPT achieves superior motion quality and diversity compared to prior methods.

### Strengths
The paper proposes PlanMoGPT, which demonstrates notable performance improvements on the authors’ customized benchmarks

### Weaknesses
1. Lacks novelty:
    - the paper appears to be an incremental improvement, and the scientific contribution is not clearly articulated. Much of the work seems engineering-oriented (e.g., “doubles the downsampling resolution and expands the codebook size by eight times” as stated in the abstract).
2. Writing and presentation issues.
    1. The overall narrative lacks clarity. The introduction discusses problems of LLMs, but the method actually targets issues inherent to Transformers in general, not specifically LLMs.
    2. Additionally, the first paragraph attributes the issue to LLMs, while the second paragraph shifts focus to tokenization as the core challenge, implying the problem lies in the tokenizer rather than the LLM. This weakens the logical coherence of the argument.
    3. Missing results in Table 2. Table 2 includes KIT++ results but omits KIT, although the implementation details (lines #306–312) suggest that experiments on KIT-ML were conducted.
    4. Ambiguity in Table 3(b). Table 3 states that “base” refers to not using the residual Transformer or flow-enhanced method. However, the table includes rows labeled with both “Flow” and “Base,” which creates confusion about whether the flow-enhanced method was used.
    5. In lines #74–89, “first” and “firstly” are used, but there is no corresponding “secondly”
3. Experiments:
    1. Limited comparison on proposed datasets. In Table 2, results on HumanML3D++ and KIT-ML++ are compared with only two other models. It lacks comprehensive comparison and thus weakens the empirical support. It would be helpful to include results of BAMM and other LLM-based approaches.
    2. Limited performance gains on commonly used dataset HumanML-3D. The method shows only marginal improvement on HumanML-3D.
4. Insufficient explanation of flow-enhanced method: The paper does not clearly explain how the proposed flow-enhanced method addresses the issue of overemphasizing short-term performance, which is highlighted as a key motivation in the abstract.

### Questions
1. What are the results on the KIT-ML dataset?
2. Could you provide further clarification on the issues described in Weakness 2(d)?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
PlanMoGPT is an LLM-based framework for text-to-motion generation that tackles fine-grained tokenization bottlenecks by integrating progressive planning and flow-enhanced tokenization. It leverages LLMs' autoregressive capabilities to refine sparse global plans into full sequences and expands the tokenizer's codebook while minimizing discretization loss, achieving state-of-the-art results on benchmarks with a 63.8% FID improvement and 49.9% diversity boost.

### Strengths
- The writing and structure of the paper are clear and easy to follow.
- The authors conducted comprehensive experiments on multiple public datasets, demonstrating improvements in numerical metrics for the proposed method.

### Weaknesses
- The paper lacks video samples. For a 3D motion generation model, providing diverse generated video samples is crucial, as it intuitively showcases the model's generation capabilities and quality. Without video samples, it is difficult for me to assess the model's actual performance, and as a reviewer, I cannot accept a 3D motion generation paper without any video samples.
- The baseline methods compared are outdated. The authors should include comparisons with the latest state-of-the-art approaches, such as works [1-5], in 3D human motion generation. The current comparisons fail to convincingly show the proposed method's superiority.
- Additionally, this paper is an LLM-based 3D motion generation model, yet numerous LLM-based related works are not cited or compared, such as [6-8]. The absence of comparisons with these relevant works makes the paper's contributions unclear and hinders the evaluation of its novelty and effectiveness.
- Similarly, while the paper focuses on long-sequence motion generation, it lacks comparisons with many related works in long-sequence motion generation, such as [9-10].


[1]: Guo C, Hwang I, Wang J, et al. SnapMoGen: Human Motion Generation from Expressive Texts[J]. arXiv preprint arXiv:2507.09122, 2025.

[2]: Meng Z, Xie Y, Peng X, et al. Rethinking diffusion for text-driven human motion generation[J]. arXiv preprint arXiv:2411.16575, 2024.

[3]: Zhang J, Fan H, Yang Y. Energymogen: Compositional human motion generation with energy-based diffusion model in latent space[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 17592-17602.

[4]: Yuan W, He Y, Shen W, et al. Mogents: Motion generation based on spatial-temporal joint modeling[J]. Advances in Neural Information Processing Systems, 2024, 37: 130739-130763.

[5]: Zhang Z, Kong B, Liu Q, et al. Towards robust and controllable text-to-motion via masked autoregressive diffusion[C]//Proceedings of the 33rd ACM International Conference on Multimedia. 2025: 9326-9335.

[6]: Wang Y, Huang D, Zhang Y, et al. Motiongpt-2: A general-purpose motion-language model for motion generation and understanding[J]. arXiv preprint arXiv:2410.21747, 2024.

[7]: Xu H, Xu G, Zheng Z, et al. VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models[J]. arXiv preprint arXiv:2508.12081, 2025.

[8]: Wu B, Xie J, Shen K, et al. MG-MotionLLM: A unified framework for motion comprehension and generation across multiple granularities[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 27849-27858.

[9]: Zhang Z, Liu A, Reid I, et al. Motion mamba: Efficient and long sequence motion generation[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 265-282.

[10]: Lee T, Baradel F, Lucas T, et al. T2lm: Long-term 3d human motion generation from multiple sentences[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 1867-1876.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
