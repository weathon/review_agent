# DIVA: Discrete Diffusion Vision-Language-Action Models for Parallelized Action Generation

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Vision-Language-Action (VLA) models have shown promising results in robot control, yet prevailing auto-regressive frameworks suffer from inherent limitations, such as error accumulation and temporal rigidity in action generation. To address this, we introduce a DIscrete diffusion Vision-language-Action model (DIVA), a discrete diffusion-based VLA framework that reformulates action generation as an iterative denoising process over discrete latent representations. The innovation of DIVA lies in the unified discrete diffusion architecture that systematically integrates three core designs: first, a learnable discrete action tokenization process bridges continuous action with the structural multimodal space. Second, A latent-driven policy learning strategy is proposed to align the representative space of the vision-language backbone and the policy head through a joint optimization. Third, a selective group unmasking strategy is introduced during the discrete diffusion decoding to preserve spatiotemporal coherence. Extensive evaluation demonstrates that DIVA achieves state-of-the-art performance in both simulated and real-world environments, validating its advantages in generating coherent, precise, and generalizable robot behaviors. Our work establishes a robust and scalable paradigm for future embodied decision-making systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DIVA, which use discrete diffusion to generate actions. It contains three designs, including a discrete action tokenization, a policy learning, and a selective group unmasking strategy.

### Strengths
- The writing is easy to follow.

- The method contains a lot of careful designs.

- The performance on LIBERO & real world robot surpass the baselines.

### Weaknesses
- The main idea of the method is using the discrete diffusion instead of continuous diffusion for VLA models. The novelty is limited.

- The policy contains a lot of designs, including the unmasking trick, regularization trick, etc. These designs are separate from the main idea and can also be applied to other method. Together with the limited performance gain (+3% success rate), makes the source of gain unclear.

- See questions below.

### Questions
(1)	In L071-L073, the authors claim that: continuous diffusion models “often operate as decoupled decoders that are misaligned with the pretrained representations of the underlying vision-language backbone, leading to inefficiencies and training instability.” Is there any evidence to support this claim? How to evaluate the efficiency and training stability with of the continuous models vs other models? 

(2)	In L222, the action tokenizer is quantized via nearest-neighbor lookup in the VLM-derived codebook. Just for verification, will the VLM use the updated codebook also for the language part or just for the action part? Is the VQ size the same for action and language? If so, I think the VQ size for language might be too big for action. If not, why using a shared codebook as initialization? Ablation is needed to prove this design. 

(3)	For the policy learning part, what is the difference between reconstructed action $\hat{S}_A$ and the decoded action $O_A$? Both of them seem to be the output from the hidden states. The final loss seems to contain both losses ($MSE(\hat{S}_A, S_A)$ and $MSE(S_A, O_A)$). 

(4)	For the OpenVLA-OFT baseline, from the official paper, the LIBERO score is 97.6 (Sp) / 98.4 (Obj) / 97.9 (Goal) / 94.5 (Long) / 97.1 (Avg).  While in the paper, the OpenVLA-OFT has an averaged success rate of 94.5. How did that score calculate? Compared with the official score, the performance gain (+0.3% success rate) seems to limited.

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
4

### Summary
The paper introduces DIVA, a discrete diffusion-based vision-language-action (VLA) framework that addresses the misalignment between pretrained vision-language representations and downstream action learning in prior VLA methods. DIVA introduces two strategies, LDPL and SGU, and demonstrates their effectiveness through ablation studies. In both real-world and simulation experiments, DIVA outperforms the strong VLA baseline π₀, demonstrating the advantages of the discrete diffusion model.

### Strengths
- DIVA introduce discrete diffusion large language models into the VLA domain, leading to significant performance improvements.
- The ablation studies are comprehensive, and the proposed tricks provide noticeable performance improvements.
- Clear presentation of methodology

### Weaknesses
- **Relatively weak experiment**  
 The paper evaluates its method only in the LIBERO simulation environment, while the real-world experiments involve relatively simple tasks and scenarios. Moreover, the absence of demonstration videos raises concerns about the model’s real-world performance.

- **Limited Empirical Evidence Supporting the Conclusions**  
 Many VLA methods already achieve high performance in the LIBERO simulation environment due to its relatively simple tasks and low generalization requirements, resulting in only marginal improvements for DIVA. Moreover, in the ablation results, removing either the LDPL or SGU tricks yields performance comparable to the baseline method OpenVLA-OFT, making it difficult to substantiate the claim that DIVA effectively addresses the issue of misalignment with the pretrained representations of the underlying vision-language backbone.

- **Figure error**  
 According to the description in Equation (5), $L_{\text{recon}}$ should be computed using $H_A$; however, in Fig.2 the loss is shown as being computed at $H_{A}'$.  This appears to be a minor inconsistency/error.
​

### Questions
1. In real-world experiments, for tasks where DIVA achieves a higher success rate than π₀,  what are the corresponding failure cases of π₀ and successful cases of DIVA?  Please provide detailed video demonstrations for comparison.
2. How do the real-world experiments demonstrate the model’s ability to handle long-horizon tasks?
3. The experiments on LIBERO seem to show that, without the two proposed tricks,  the discrete diffusion-based VLA framework performs comparably to the baseline methods  (OpenVLA, π₀, etc.).  How can the authors substantiate the claim that this framework effectively alleviates the disconnect between the pretrained backbone and the policy head? It is recommended to include comparisons on more challenging benchmarks (such as Simpler[1], Robotwin[2], or Calvin[3]) to better demonstrate the model’s effectiveness and generalization ability

[1] Li, Xuanlin, et al. "Evaluating real-world robot manipulation policies in simulation." arXiv preprint arXiv:2405.05941 (2024).

[2] Mu, Yao, et al. "Robotwin: Dual-arm robot benchmark with generative digital twins (early version)." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[3] Mees, Oier, et al. "Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks." IEEE Robotics and Automation Letters 7.3 (2022): 7327-7334.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces DIVA, a Discrete Diffusion Vision-Language-Action (VLA) model designed to address limitations of existing autoregressive and continuous diffusion VLA approaches.
DIVA reformulates action generation as a discrete denoising diffusion process operating over quantized latent action tokens.

### Strengths
1. The paper’s discrete diffusion formulation for action generation is novel.
2. Selective Group Unmasking is a good design and is verified in experiments.

### Weaknesses
1. The real-world experiments are relatively simple and do not adequately demonstrate the policy’s performance in real-world settings.

2. The authors claim that autoregressive-style VLAs suffer from error accumulation and temporal rigidity in action generation, but they do not provide specific experiments to directly validate these claims.

### Questions
1. The authors claim that autoregressive-style VLAs suffer from error accumulation and temporal rigidity in action generation. Are there any specific experiments that directly verify these claims?

2. Why do the authors use an additional policy head to decode the actions instead of directly decoding the tokens generated by the diffusion VLM into raw actions by VQVAE encoder? If an additional policy head is used, how can the generalization ability inherited from the VLM preserved?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces **DIVA** (Discrete Diffusion Vision-Language-Action), a framework that reformulates **action generation** as an **iterative denoising process** over **discrete latent representations**. The proposed pipeline integrates three key components: **learnable discrete action tokenization**, **latent-driven policy learning**, and **selective group unmasking**. Extensive experiments on **LIBERO** and **real-robot** benchmarks demonstrate that **DIVA** achieves superior performance compared to existing **imitation learning** and **Vision-Language-Action (VLA)** models.

### Strengths
1. The paper tackles a **highly important and timely research question**, particularly as diffusion models are becoming increasingly influential in the fields of **imitation learning**, **reinforcement learning**, and **Vision-Language-Action (VLA)** modeling.

2. The paper presents **extensive experiments** in both **simulation** and **real-robot** settings, providing strong empirical validation of the proposed method’s effectiveness.

3. The paper is **well-written**, **clearly structured**, and **easy to follow**, making the technical content accessible and well-motivated.

### Weaknesses
1. The authors should report the **number of random seeds** and the corresponding **standard deviations** for the main results presented in **Table 1**.

2. In **Table 2** of the ablation study section, **Model 1 (OpenVLA)** achieves a performance of **95.2%**, whereas in the main results of **Table 1**, **OpenVLA** reports an average performance of only **76.5%**. The authors should clarify the reason for this **data discrepancy**.

3. What is the **inference time** of **DIVA** for generating a sequence of actions? Does the use of **discrete diffusion models** increase inference latency, potentially limiting the method’s **practical applicability in real-world settings**?

### Questions
1. **(Related to Weakness 2)** What explains the **performance discrepancy** of **OpenVLA** between **Table 1** and **Table 2**?

2. **(Related to Weakness 3)** What is the **inference time** of **DIVA**? Does the use of **discrete diffusion models** introduce additional **inference latency**?

### Soundness
3

### Presentation
3

### Contribution
3
