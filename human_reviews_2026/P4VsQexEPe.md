# Decoupled MeanFlow: Turning Flow Models into Flow Maps for Accelerated Sampling

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Denoising generative models, such as diffusion and flow-based models, produce high-quality samples but require many denoising steps due to discretization error. Flow maps, which estimate the average velocity between timesteps, mitigate this error and enable faster sampling. However, their training typically demands architectural changes that limit compatibility with pretrained flow models. We introduce \emph{Decoupled MeanFlow}, a simple decoding strategy that converts flow models into flow map models without architectural modifications. Our method conditions the final blocks of diffusion transformers on the subsequent timestep, allowing pretrained flow models to be directly repurposed as flow maps. Combined with enhanced training techniques, this design enables high-quality generation in as few as 1–4 steps. Notably, we find that training flow models and subsequently converting them is more efficient and effective than training flow maps from scratch. On ImageNet 256$\times$256 and 512$\times$512, our models attain 1-step FID of 2.16 and 2.12, respectively, surpassing prior art by a large margin. Furthermore, we achieve FID of 1.51 and 1.68 when increasing the steps to 4, which nearly matches the performance of flow models while delivering over 100$\times$ faster inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Decoupled MeanFlow (DMF), a way to convert a pretrained flow (or diffusion-as-flow) model into a flow-map model without changing the backbone: treat the early blocks as an encoder conditioned on the current time $t$ and the late blocks as a decoder conditioned on the next time $r$. This decoupling lets the model learn the average velocity between $(t,r)$ while reusing the original timestep embedding and velocity head. With a flow-matching warm-up then flow-map fine-tuning (plus model-guidance and a robust Cauchy loss), DMF targets ultra-few-step sampling (1–4 steps).

### Strengths
1. This paper proposes simple, compatible architecture. The key idea—drop $r$ from the encoder and $t$ from the decoder, i.e., $u_\theta(x_t,t,r)=g_\theta(f_\theta(x_t,t),r)$—is elegant and backbone-agnostic, enabling plug-and-play conversion of existing flow models.


2. Strong empirical case that “your flow model is secretly a flow map.” Without any finetuning, the converted model can already outperform the base SiT in FID at equal step counts; with decoder-only finetuning it gets further gains. The depth ablation is informative.


3. State-of-the-art 1–4 step results under fair comparisons. Tables show consistent wins over MeanFlow baselines at comparable training budgets, and competitive/best few-step FIDs on ImageNet 256/512.

### Weaknesses
See Questions

### Questions
1. Table 1 suggests both model guidance (MG) and REPA are crucial for DMF’s final performance. DMF typically splits at later layers (e.g., layer 18), whereas REPA often targets earlier layers (e.g., layer 8 in SiT-L/2). Is there a principled relationship between the REPA alignment layer and the DMF encoder–decoder split?
2. The paper mentions visual artifacts in generated outputs in limitations. Could you include representative ‘bad-case’ examples to illustrate these failures?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors explore architectural improvements for training flow map models and propose a series of techniques to enhance the performance of MeanFlow under comparable compute budgets. The resulting Decoupled MeanFlow (DMF) framework introduces a decoupled encoder–decoder design that enables pretrained flow matching models to be seamlessly converted into flow map models without architectural modification. By conditioning the decoder on the next timestep while keeping the encoder focused on the current one, DMF improves both efficiency and transferability of pretrained representations. This approach, combined with flow-matching warm-up, adaptive Cauchy loss, and tailored timestep sampling, achieves state-of-the-art few-step generative performance on ImageNet 256×256, reaching a 1-step FID of 2.16 and matching standard flow model performance at 4 steps.

### Strengths
1. **Training-free flow map transformation**

The proposed decoupled architecture allows pretrained flow models to be directly repurposed as flow maps without additional fine-tuning, which is both conceptually elegant and practically impactful. This demonstrates a viable paradigm for transformer-based flow map models that leverages existing large-scale flow model checkpoints, reducing training cost and broadening applicability.

2. **Broader applicability of the fine-tuning paradigm**

The proposed fine-tuning strategy has strong practical implications. Because DMF can be directly initialized from large-scale pretrained flow checkpoints (e.g., FLUX-dev, SD3, etc.), it enables transforming existing models into few-step generators with a modest additional compute budget. This makes the method immediately applicable in real-world or large-foundation-model scenarios where retraining from scratch is infeasible.

3. **Comprehensive empirical validation**

The paper includes detailed ablations, consistently validating the design claims, particularly the benefit of decoupling timestep conditioning and the effectiveness of flow-matching warm-up for stability and efficiency.

### Weaknesses
1. **Stability of the JVP term**

The proposed method does not directly address the well-known stability issue of the JVP term. This instability has been repeatedly identified as the primary bottleneck in scaling consistency-based methods to large-scale applications such as text-to-image or text-to-video generation (Lu & Song, 2024; Chen et al., 2025; Zheng et al., 2025). Therefore, while the techniques presented in the paper for improving MeanFlow training remain valuable, the overall scope of the paper remains somewhat limited. It would nonetheless strengthen the paper to include an empirical comparison of training stability between DMF and vanilla MeanFlow (e.g., visualizations of training losses, gradient norms, or JVP magnitudes).

2. **Insufficient ablations**

The use of the adaptive weighted Cauchy loss (Eq. 6) for MeanFlow optimization is intriguing but under-justified. The paper should better explain why this choice improves robustness beyond prior alternatives (e.g., the adaptive normalization proposed in Geng et al., 2024 and the tangent normalization proposed in Lu & Song, 2024) with supporting evidence. Also, the paper should cite Dao et al., 2025, as they also proposed applying a near identical Cauchy loss to consistency training.

3. **Limited architectural exploration**

While the encoder-decoder design for training-free flow map transformation is intriguing, the current experiments primarily rely on the vanilla DiT. It remains unclear whether the encoder–decoder decoupling generalizes to other variants of DiT (i.e. LightningDiT (Yao et al., 2025) or DDT (Wang et al., 2025)), and the paper would be greatly strengthened from some discussions of this.

Lu, Cheng, and Yang Song. "Simplifying, stabilizing and scaling continuous-time consistency models." arXiv preprint arXiv:2410.11081 (2024).

Chen, Junsong, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. "Sana-sprint: One-step diffusion with continuous-time consistency distillation." arXiv preprint arXiv:2503.09641 (2025).

Zheng, Kaiwen, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. "Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency." arXiv preprint arXiv:2510.08431 (2025).

Geng, Zhengyang, Ashwini Pokle, William Luo, Justin Lin, and J. Zico Kolter. "Consistency models made easy." arXiv preprint arXiv:2406.14548 (2024).

Wang, Shuai, Zhi Tian, Weilin Huang, and Limin Wang. "Ddt: Decoupled diffusion transformer." arXiv preprint arXiv:2504.05741 (2025).

Yao, Jingfeng, Bin Yang, and Xinggang Wang. "Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models." In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15703-15712. 2025.

Dao, Quan, Khanh Doan, Di Liu, Trung Le, and Dimitris Metaxas. "Improved training technique for latent consistency models." arXiv preprint arXiv:2502.01441 (2025).

### Questions
1. **Representation quality**

Does the proposed decoupled architecture improve the representation quality of the transformer encoder itself compared to vanilla MeanFlow (e.g., linear probing accuracy) under similar training budget?

2. **Diminishing gains with REPA**

The paper notes that REPA fine-tuning yields diminishing returns (Tab. 1). Could the authors provide quantitative comparisons (apply REPA only for MF fine-tuning or apply REPA at both stages) or intuition on why representation alignment offers less benefit under the DMF design or consistency training in general?

3. **Loss formulation**

When doing MF fine-tuning, the authors use sum of MF & FM losses instead of batch splitting proposed in MeanFlow. However, the adaptive weighting function itself breaks the "correctness" of the FM objective (e.g., the optimal solution is no longer $\mathbb{E}[v | x_t = x]$). Have the authors evaluated DMF performance with the vanilla flow-matching loss without such weighting for ablation, and how does it compare in stability and convergence?

4. **Timestep sampling**

Have the authors experimented with alternative timestep proposal paradigms (e.g., $p(t, r) = p(t)p(r | t)$)?

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
3

### Summary
The paper proposes Decoupled MeanFlow (DFM), a method to accelerate diffusion or flow models by converting them into flow maps which can do large denoising jumps. DMF decouples the model's architecture into an encoder and a decoder, where only the encoder receives information about the start of the jump and only the decoder receives information about the end of the jump. DMF achieves the state-of-the-art performance on ImageNet, outperforming shortcut models, MeanFlow models and GANs.

### Strengths
- The proposed DMF model consistently outperforms the MeanFlow baseline across all evaluated datasets and variants. Notably, it achieves high-quality 1-step ImageNet generation which highlights its efficiency and strong generative capacity.
- The model can be trained from scratch, yet it also seamlessly integrates with existing pretrained models without requiring any architectural modifications while yielding improved results. 
- The analysis of the encoder–decoder decomposition is interesting. It is particularly noteworthy that a pretrained model, when utilized under the proposed scheme and without any training, can outperform the original model in few-step generation regimes.

### Weaknesses
- While the separation of encoder and decoder components is appealing, it also seems natural to consider joint conditioning mechanisms that integrate information from both the current and target timesteps in some blocks, potentially via lightweight modifications such as joint AdaLN conditioning or LoRA adapters. Have the authors explored such hybrid alternatives?
- Given that MeanFlow already incorporates both timestep conditionings, one might expect the model to implicitly learn to attenuate or emphasize the relevant conditioning at different stages (e.g., down-weighting target timestep signals early and source signals later). Why does the explicit disentanglement in DMF lead to consistently better results? A more detailed discussion or theoretical analysis on this would strengthen the conceptual clarity of the paper.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper mainly proposes an alternative time-conditioning design for MeanFlow models, named decoupled MeanFlow (DMF). Instead of modifying the architecture of a flow model to accommodate two timesteps,  DMF treats the DiT as an encoder-decoder architecture, and feed $t$ to the encoder and $r$ to the decoder, which leads to notable gains in generation quality. Combining this design with other technics such as flow matching warm-up, adaptive weighted Cauchy loss, and representation alignment, DMF achieves the state-of-the-art 1-step FID of 2.16 on ImageNet 256x256.

### Strengths
- The central contribution of this work is the simple time conditioning modification. This leads to notable performance gains as shown in the ablation studies in Table 1 and 2, validating the effectiveness of the decoupled design.
- Overall, a 1-step FID of 2.16 on ImageNet 256x256 is an impressive state-of-the-art result.

### Weaknesses
- This work is motivated by the encoder-decoder design. Yet the experiments does not provide direct evidence to prove that encoder-decoder is the key to high quality. For example, there are many other ways to condition the network without modifying its architecture, like interleaving $t$ and $r$ for the DiT blocks. Discussing alternative design choices in the ablation could strengthen the argument.
- The authors claim that an existing SiT can be converted into a flow map without finetuning, yet the results in Fig. 3 only show a very minor gap between DMF and SiT. Sometimes it's even worse than the baseline (Fig. 8 left). The slight improvement in general is unsurprising since the flow map tends to be closer to the average of start- and end-point velocity, thus it's reasonable that mixing start and end time in the model should be better than the plain SiT baseline.
- FM training with REPA is not really something new. It's not surprising that initializing from REPA models generally improves MF and DMF, which is not unique to DMF. Thus its relevance to the main contribution is weak.
- Writing issues: 
  - Flow models trained on random noise-data pairs predicts the expectation of $\alpha_t^\prime x_0 + \sigma_t^\prime \epsilon$, i.e., $v(x, t) = \mathbb{E}[\alpha_t^\prime x_0 + \sigma_t^\prime \epsilon]$. The definition of $v(x, t)$ in L146 is not accurate.
  - Metric consistency: L422 says "DMF-XL/2 achieves 1-step FID=3.10", yet table 2 shows FID=2.83.

### Questions
In L208, it is stated that feeding the next timestep to both the encoder and decoder is redundant, which motivates DMF. Why would redundancy cause issues for the model? Neural networks are generally redundant by design, so I do not really understand the intuition of why feeding the timestep to selective blocks would be better than feeding it to all the blocks, apart from the concerns of architecture modifications.

### Soundness
2

### Presentation
3

### Contribution
2
