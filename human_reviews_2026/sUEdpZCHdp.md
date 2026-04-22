# Composition of Memory Experts for Diffusion World Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 4, 4

## Abstract
World models aim to predict plausible futures consistent with past observations, a
capability central to planning and decision-making in reinforcement learning. Yet,
existing architectures face a fundamental memory trade-off: transformers preserve
local detail but are bottlenecked by quadratic attention, while recurrent and state-
space models scale more efficiently but compress history at the cost of fidelity. To
overcome this trade-off, we suggest decoupling future–past consistency from any
single architecture and instead leveraging a set of specialized experts. We introduce
a diffusion-based framework that integrates heterogeneous memory models through
a contrastive product-of-experts formulation. Our approach instantiates three
complementary roles: a short-term memory expert that captures fine local dynamics,
a long-term memory expert that stores episodic history in external diffusion weights
via lightweight test-time finetuning, and a spatial long-term memory expert that
enforces geometric and spatial coherence. This compositional design avoids mode
collapse and scales to long contexts without incurring a quadratic cost. Across
simulated and real-world benchmarks, our method improves temporal consistency,
recall of past observations, and navigation performance, establishing a novel
paradigm for building and operating memory-augmented diffusion world models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a memory-compositional diffusion world model that integrates multiple specialized “memory experts”—short-term memory (STM), long-term LoRA-based memory (LTM), and spatial long-term memory (SLTM)—into a unified model via a **Product of Contrastive Experts (PoCE)** mechanism. The PoCE combines multiple memory outputs while reducing redundant features through contrastive weighting, encouraging each expert to specialize. Experiments on navigation and reconstruction datasets show clear improvements in long-horizon consistency, memory recall, and generalization to novel sequences.

### Strengths
* The conceptual idea of composing multiple memory experts through a contrastive product is innovative and well-articulated.

* The use of test-time LoRA as long-term memory is practical, allowing scalable specialization without retraining the core model.

* Experiments convincingly demonstrate that the composition improves both reconstruction quality and navigation performance.

* The paper is clearly written, with mathematical formulation and visual illustrations that make the mechanism understandable.

* The approach connects well to emerging work on compositional generative models and memory-based reasoning.

### Weaknesses
* The assumptions behind the contrastive composition (such as the independence of expert outputs) are not empirically validated.

* Computational overhead and runtime comparisons to single-memory baselines are missing.

* While results are strong, the evaluation scope is somewhat limited to a few datasets.

### Questions
1. How sensitive is performance to the number of experts or their relative weights?

2. Could you provide compute and runtime comparisons to conventional diffusion world models?

3. Are there stability issues when experts disagree strongly during sampling?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors propose a novel diffusion world model that employs a composition of multiple types of memory experts to overcome the *memory-compute trade-off* inherent in existing architectures (e.g., Transformers, State Space Models (SSMs)). This model generates the next frame by probabilistically fusing the predicted probability distributions from each expert into a single, unified distribution.

The paper introduces three types of experts: a **Short-Term Memory (STM)** expert that predicts based on recent context, a **Long-Term Memory (LTM)** expert that stores long-term context in its weights via test-time fine-tuning, and a **Spatial Long-Term Memory (SLTM)** expert that utilizes spatial knowledge for prediction . In their experiments, the authors validate the positive impact of each memory component on model performance. They also demonstrate that performance **monotonically increases** as more context is provided to the LTM.

Furthermore, the authors propose a novel formulation, the **Product of Contrastive Experts (PoCE)**, to ensure that each expert focuses more intently on the given context . This method selectively amplifies predictions that have a higher relative probability when conditioned on the context, compared to an unconditional baseline . Through experiments, they show that this approach mitigates the **consistency collapse** phenomenon—often observed in the conventional Product of Experts (PoE) method—and achieves superior performance .

Finally, the authors experimentally validate that their full model, combining the STM, LTM, and SLTM experts with the PoCE unification formulation, outperforms existing memory-augmented world models. In the appendix, they provide a detailed description of their experimental setup and demonstrate the effectiveness of PoCE through a toy example . They also validate the superior performance of their world model through a discrete recall evaluation (Memory-Cards environment) and a deterministic environment analysis (DMLab) . Notably, in the discrete recall evaluation, their model improves recall accuracy by approximately **66%** compared to a standard diffusion baseline (79.2% vs. 13%).

### Strengths
- **Addresses a significant research question and proposes a viable alternative:** The authors tackle the critical _memory-compute tradeoff_  in world models, a problem where models are either limited to using short-term memory or incur excessive computational overhead when employing long-term memory . To address this, they introduce the concept of a Product of Experts (PoE), proposing a model where specialized memory experts operate compositionally, and they experimentally validate its effectiveness through significant performance improvements.
    
- **Novel formulation for a contrastive Product of Experts:** The authors do not simply adopt the standard PoE framework. They identify a potential issue of consistency collapse when applying it to memory experts and propose a novel contrastive formulation to mitigate this problem . The superiority of this method is convincingly demonstrated through both an intuitive toy visualization and an empirical ablation study .
    
- **Thorough experimental validation and detailed methodology:** The authors demonstrate the effectiveness of their proposed compositional memory experts through a comprehensive set of ablation studies. For the Long-Term Memory (LTM) expert, in particular, they show a monotonic performance improvement as the context length increases . Furthermore, they strengthen their claims by including additional insightful experiments in the appendix, such as the PoCE toy visualization and a discrete recall evaluation, which experimentally prove the model's efficacy . The paper is also well-written, with the methodology and experimental settings described in sufficient detail in both the main paper and the appendix to ensure reproducibility.

### Weaknesses
- **Insufficient discussion on whether the proposed method truly resolves the memory-compute tradeoff:** The authors frame their work as a solution to the memory-compute tradeoff. However, their proposed method, Composition of Memory Experts (CoME), requires forward passes for each expert and its contrastive counterpart, leading to a computational complexity increase of approximately $2\times n$ (where $n$ is the number of experts), as discussed in Appendix F. This raises doubts as to whether CoME fundamentally solves the issue, as requiring more memory (i.e., more experts) still leads to a direct increase in computation. While the authors mention in Appendix F that this overhead can be reduced by using smaller expert models, this point is critical to the paper's central research question and warrants a more in-depth discussion with concrete experimental results in the main paper, not just the appendix .
    
- **Lack of discussion on limitations:** The paper lacks a thorough discussion of the limitations of the CoME framework. For instance, as the number of experts with diverse characteristics increases, the computational overhead will grow, and the process of fusing their predicted distributions could become more challenging (e.g., tuning the hyperparameters $\alpha_i$ ​may become prohibitively complex). Moreover, while the framework is effective if the chosen memory experts are well-suited for the task, inappropriate experts could have a detrimental impact, potentially leading to performance worse than the base model. A discussion of these potential failure points and scalability challenges should be included.
    
- **Insufficient experimental analysis for Short-Term Memory and Spatial Long-Term Memory:** While the paper demonstrates a monotonic performance improvement for Long-Term Memory (LTM) with increased context, similar in-depth analyses for Short-Term Memory (STM) and Spatial Long-Term Memory (SLTM) are missing. The ablation studies confirm that using these modules is beneficial, but they do not provide clear evidence that the performance gains are achieved for the intended reasons—i.e., that STM excels by focusing on recent observations and SLTM by leveraging spatial knowledge. This makes it difficult to verify whether these components are functioning as hypothesized.

- **The Statement about the Use of Large Language Model:** The authors doesn’t state explicitly how they used LLM in their submission, which is guided in https://iclr.cc/Conferences/2026/AuthorGuide.
    
- **(Minor) Comments on Presentation:**
    - In line 173, explaining the "subsets of frames" with a concrete example would help prevent reader confusion about how these subsets are determined.
    - In line 221, there appears to be a small formatting artifact at the end of the line.
    - In Section 3.3, the description of SLTM is ambiguous about its use of fine-tuning. This should be clarified in the main text, as it is mentioned in Appendix E.1.
    - The discussion on training cost (lines 359-364) could be more detailed with the training time for each model.
    - In Section 4.3, the dataset used for the ablation studies should be explicitly mentioned.
    - In Figure 3, the meaning of each row and what the reader should observe (e.g., which frames are forward vs. reverse) is not clearly explained in the caption.
    - In line 975, it should be explicitly stated that the "$2\times$" increase is in comparison to the base model.
    - In Figures 5-10, the red box annotations appear misaligned or incorrectly drawn in several places (e.g., all frames are boxed in "Ours" in Figure 5; boxes are unclear in Figure 10).

### Questions
- Could you please share a more detailed experimental analysis for the Short-Term Memory (STM) and Spatial Long-Term Memory (SLTM) experts? For instance, it would be insightful to see an experiment where, in the absence of SLTM, the model's performance on a dedicated spatial reasoning task (e.g., as explored in [1]) is shown to decrease. This would provide stronger evidence that the SLTM is indeed contributing the specific spatial capabilities as intended.

[1] Cho, Junmo, Jaesik Yoon, and Sungjin Ahn. "Spatially-aware transformer for embodied agents." _arXiv preprint arXiv:2402.15160_ (2024).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
To leverage the benefits from different memory models, the proposed approach (CoME) utilises a diffusion framework with a contrastive mechanism to compose information from memories. Experiments across different settings showed that the strategy improves performance.

### Strengths
- The mechanism to compose heterogeneous experts is well-presented in Section 3.2, and different memory models (long-term, short-term, and Spatial Long-term Memory) are detailed clearly in Section 3.3.
- Experiments were conducted on a range of datasets, including Memory Maze, RealEstate10K, DeepMind Lab, Memory Cards Dataset, and RECON.  The comparisons of different memory combinations are also highlighted in Table 1. The setting in section 4.2 shows that the composed mechanism can handle a stream of observations.

### Weaknesses
Since the approach proposed to use all memory models, it will be more expensive than using one memory. However, the authors also discussed and provided compute analysis in Appendix F.

### Questions
The author could highlight more in the baseline, which one is the baseline that contains the simpler way of combining knowledge from memory models.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
- The authors propose approaches for augmenting diffusion world models with external memories (a short, long, and short-long term memory) that enable diffusion world models to have better long-term rollouts. The authors propose a product of contrastive experts approach that improves performance, and demonstrate broadly that their proposed approach has good properties empirically and loosely theoretically.

### Strengths
- The results are generally very strong, with the proposed CoME generally outperforming all existing models in LPIPS, SSIM, and PSNR.
- The writing and motivation are good, as I agree that incorporating short/long/short+long term memory into existing models would be very impactful.
- The ablations are generally very strong in terms of supporting the author's claims.

### Weaknesses
- It's not clear that a fair comparison was done to me, which is my biggest concern. The authors don't include rich details regarding the parameters/FLOPs of models in Table 1, Table 2, and table 3. This makes it hard to determine whether or not the memories are working as intended and really providing strong performance for their size, or are just giving the model more capacity. 
    - I see this as the largest issue. Is it possible to reveal this information as well as do a comparison between a baseline model with more FLOPs/parameters to match the size of the baseline with the memory experts?
    - Further, is it possible to do comparisons at multiple different model sizes, to see if the approach scales well?
- Visual perceptive metrics are the primarily method for evaluating the model. While this is common, pixel level reconstruction has been shown to be less effective than modeling in a learned latent space [1], meaning the world models may produce higher fidelity rollouts but not actually be better world models. What do the authors think of this?

[1] https://arxiv.org/abs/2506.09985

### Questions
- Overall, I'm open to revising my score, but that primarily relies on addressing my concerns/questions given in the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CoME, which effectively combines long-term and short-term memory in Diffusion-based World Models to address memory limitations. Based on the existing limitations and concerns, I recommend **weak reject**. However, I am willing to raise my score if the authors can address these limitations and concerns.

### Strengths
* Well-motivated problem: The fundamental trade-off between context length and computational cost in world models is clearly articulated. The human cognition analogy (STM/LTM) provides good intuition.
* Effective and elegant solution: Using test-time finetuning to overcome context length limitations in the DiT architecture is an effective and concise approach.
* Thorough ablations: Table 4 convincingly demonstrates the necessity of the contrastive formulation; Table 5 analyzes LTM capacity trade-offs.

### Weaknesses
* Insufficient baselines: Missing comparisons with key related works (WorldMem [1], StateSpaceDiffuser [2]).
* Low computational efficiency.
* CoME uses 4 models vs single models for baselines, is this a fair comparison? 
* No failure case analysis.

[1] WORLDMEM: Long-term Consistent World Simulation with Memory

[2] StateSpaceDiffuser: Bringing Long Context to Diffusion World Models

### Questions
* How does CoME world model condition on action input?
* Can the authors provide additional experiments on Minecraft, since many existing works experiment on this environment (Dreamer4 [1], Oasis [2], MineWorld [3])? Or compare with other existing World Model methods on standard datasets (StateSpaceDiffuser [4], RLVR-World [5], GameNGen [6])?
* What is the difference between SLTM and LTM besides patch size and conditioning? Do they both use the same test-time finetuning method? Section E.1 states "similar to LTM" but provides no details.
* If the test set does not include relative and absolute position information (i.e., only contains images and action sequences) would this method still be effective?
* Can the authors provide detailed computational efficiency comparisons with baselines, including end-to-end latency and throughput?
* What is the total parameter count of CoME, can author provides parameter count comparison with existing baselines.
* Why does SLTM use doubled patch size?

[1] Training Agents Inside of Scalable World Models

[2] Oasis: A Universe in a Transformer

[3] MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft

[4] StateSpaceDiffuser: Bringing Long Context to Diffusion World Models

[5] RLVR-World: Training World Models with Reinforcement Learning

[6] Diffusion Models Are Real-Time Game Engines

### Soundness
3

### Presentation
2

### Contribution
3
