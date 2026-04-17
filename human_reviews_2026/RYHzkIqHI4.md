# Image Tokenizer Needs Post-Training

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Recent image generative models typically capture the image distribution in a pre-constructed latent space, relying on a frozen image tokenizer. However, there exists a significant discrepancy between the reconstruction and generation distribution, where current tokenizers only prioritize the reconstruction task that happens before generative training without considering the generation errors during sampling. In this paper, we comprehensively analyze the reason for this discrepancy in a discrete latent space, and, from which, we propose a novel tokenizer training scheme including both main-training and post-training, focusing on improving latent space construction and decoding respectively. During the main training, a latent perturbation strategy is proposed to simulate sampling noises, i.e., the unexpected tokens generated in generative inference. Specifically, we propose a plug-and-play tokenizer training scheme, which significantly enhances the robustness of tokenizer, thus boosting the generation quality and convergence speed, and a novel tokenizer evaluation metric, i.e., pFID, which successfully correlates the tokenizer performance to generation quality. During post-training, we further optimize the tokenizer decoder regarding a well-trained generative model to mitigate the distribution difference between generated and reconstructed tokens. With a $\sim$400M generator, a discrete tokenizer trained with our proposed main training achieves a notable 1.60 gFID and further obtains 1.36 gFID with the additional post-training. Further experiments are conducted to broadly validate the effectiveness of our post-training strategy on off-the-shelf discrete and continuous tokenizers, coupled with autoregressive and diffusion-based generators.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates the causes of performance discrepancies in discrete latent tokens and introduces a new tokenizer training framework composed of main-training and post-training stages. The main-training phase incorporates a latent perturbation strategy to mimic sampling noise and improve the robustness and efficiency of tokenizers. A training method and a new evaluation metric, namely pFID, are also proposed to better align tokenizer performance with generation quality. In the post-training phase, the tokenizer decoder is further optimized with respect to a trained generative model to reduce the gap between generated and reconstructed tokens.

### Strengths
[1]. The core idea is simple yet effective, and the authors provide solid experimental evidence to demonstrate its validity.
The comparisons with existing methods are fairly comprehensive, which helps position the contribution clearly within the current literature.

[2]. The paper presents extensive experiments and analyses that enhance the reader’s understanding of the proposed approach.
These results and discussions are valuable for future research in this area and may inspire subsequent work on improving image tokenizers and post-training strategies.

### Weaknesses
[1]. The design of the loss function appears overly complex, and there is no verification of parameter sensitivity for each hyperparameter. I suggest providing a table summarizing the optimal values of these parameters and explaining how those optimal values were determined.

[2]. It would be helpful to include experiments on the text-to-image task using JourneyDB. Without T2I results, it is difficult to assess the generalization ability of your method. Considering the complexity of your loss function, it may also be challenging for future researchers to effectively apply your approach in practice.

1. Sun K, Pan J, Ge Y, et al. Journeydb: A benchmark for generative image understanding[J]. Advances in neural information processing systems, 2023, 36: 49659-49678.

[3]. There is a typo in Table 4 (c): **Perservation ratio** should be corrected to **Preservation ratio.**

[4]. There are several issues in both the references and the OpenReview abstract, which suggest that this work may have been completed under significant time constraints.
4.1. The abstract on OpenReview contains the character sequence \ie, which appears to be an unprocessed LaTeX command.
4.2. The reference section includes numerous duplicated entries and formatting inconsistencies, indicating that the manuscript was not thoroughly proofread before submission. This raises concerns about the overall accuracy and attention to detail in the paper’s presentation.

*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
 Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An
 image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint
 arXiv:2010.11929, 2020.*

*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
 Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszko
reit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at
 scale, 2021. URL https://arxiv.org/abs/2010.11929.*

*Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with
 vq-vae-2. Advances in neural information processing systems, 32, 2019a.*

*Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with
 vq-vae-2, 2019b. URL https://arxiv.org/abs/1906.00446.*

*J Ning, C Li, Z Zhang, Z Geng, Q Dai, K He, and H Hu. All in tokens: Unifying output space of
 visual tasks via soft token. arxiv 2023. arXiv preprint arXiv:2301.02229.*

*X Zhu, W Su, L Lu, B Li, X Wang, and J Dai. Deformable detr: Deformable transformers for
 end-to-end object detection. arxiv 2020. arXiv preprint arXiv:2010.04159, 2010.*

4.3. Some of the cited works appear to have low relevance to your study and could be removed.

[5] Figure 16 still contains many noticeable artifacts. For example, the text on the clock in the first row, first column is distorted; the bird’s tail in the first row, third column appears blurry; the human figure in the second row, first column fails to reconstruct properly; and the dog in the second row, second column is unclear. These examples demonstrate that, while your method shows certain improvements compared to the pre–post-training baseline, the overall visual quality remains subpar. This suggests that the effectiveness of your improvements is somewhat limited and raises questions about the actual contribution of the proposed pFID metric.

### Questions
[1]. Why not use adversarial noise? It would be interesting to explore how the model behaves when adversarial noise is applied. Would the proposed tokenizer or post-training strategy still maintain robustness under such perturbations?

[2]. How many GPU hours were required for training, and what kind of GPUs were used for inference? Please provide detailed computational requirements, including total GPU hours and the type or number of GPUs used. This information is important for assessing the reproducibility and scalability of your method.

[3]. What happens when using codes that are weakly correlated with the latent code? An analysis or ablation using latent codes that are less correlated with the original ones would help clarify the stability and robustness of the learned representation.

[4]. How does your method perform on datasets other than ImageNet? Evaluating on additional datasets would strengthen the claim of generalization and demonstrate that the proposed post-training method is not limited to ImageNet-specific settings.

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
4

### Summary
The paper studies the gap between reconstruction and generation in image tokenizers and introduces a two‑stage scheme called RobusTok. In the main training stage, a latent perturbation strategy is applied in discrete latent space to simulate sampling noise, yielding a tokenizer that is more robust to generation errors. A new metric, pFID, is proposed to better correlate tokenizer quality with downstream generation performance. In the post‑training stage, the tokenizer decoder is further optimized with respect to a well‑trained generator to mitigate the remaining distribution gap between reconstructed and generated tokens. With empirical experiments, it indicates effectiveness across both autoregressive and diffusion‑based generators and for off‑the‑shelf discrete and continuous tokenizers.

### Strengths
1. Clear analysis on the reconstruction–generation gap.
- In this paper, it analyzes the cause of the discrepancy in discrete latent space and introduces pFID, a metric designed to align tokenizer evaluation with downstream generative quality. The empirical study provides detailed evidence supporting the conjecture that robustness in latent space improves generation.

2. Method aligns the conjecture with practice.
- The proposed two-stage design, consisting of latent perturbation for robust latent construction and post-training for decoder alignment, closely follows the stated hypothesis and is validated through comprehensive experiments on both autoregressive and diffusion generators.

### Weaknesses
1. Loss design complexity and attribution.
- The loss formulation in Appendix A.2 makes tokenizer training appear somewhat complex. While the central conjecture is that simulating sampling error via latent perturbation reduces the reconstruction–generation gap, the perturbation is combined with several other loss terms, making the specific contribution of the perturbation harder to isolate. 
- It remains unclear whether the observed gains persist without semantic regularization from an external vision encoder (e.g., DINO). A clear ablation that removes semantic regularization and isolates the effect of each loss component would strengthen the claims.


2. Unclear criteria for when post-training is beneficial.
- The paper motivates post-training as a means to address the residual gap between latents produced by a trained generator and those seen during tokenizer training. However, it remains ambiguous under what circumstances post-training should be applied. Since pFID is introduced as a diagnostic measure correlated with generation quality, clarifying whether a specific pFID threshold is used to trigger or terminate post-training would improve the claim.


3. Limited scope of experiments.
- The experiments are conducted primarily on the ImageNet 256×256 benchmark. It would strengthen the empirical validation to include additional experiments on ImageNet 512×512 or MS-COCO, demonstrating the generalization of the proposed method to higher resolutions and more diverse datasets.


4. Additional baselines for completeness.
- In Table 2 (class-conditional ImageNet 256×256), it would be beneficial to include references to FlexTok [1] and ResGen [2] within the related work discussion. Including these recent methods would strengthen the comparative context and highlight where the proposed approach stands relative to concurrent advances in tokenizer design.


References
- [1] FlexTok: Resampling Images into 1D Token Sequences of Flexible Length
- [2] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens

### Questions
1. Figure 7 (t‑SNE) controls.
- In the t-SNE visualization comparing latent spaces with and without latent perturbation, were all other losses and regularizers, including semantic regularization from any external encoder, kept identical across conditions? Providing an explicit description of the controls or including an ablation without semantic regularization would make the attribution to latent perturbation more convincing.

2. Post‑training scheduling and stopping criteria.
- What metric and threshold determine when to start and when to stop post-training in practice? Is pFID used as a signal to initiate post-training, and is gFID or pFID employed as the stopping criterion? If pFID is intended as a practical diagnostic, it would be useful to know whether it consistently converges toward a target threshold during post-training across different models and datasets.

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
5

### Summary
The authors argue that conventional tokenizers, optimized solely for reconstruction fidelity, lack the necessary robustness to handle the noisy and out-of-distribution latent codes produced during the generative inference stage. To solve this, they propose a two-stage training scheme called "RobusTok". The first stage introduces a latent perturbation strategy to build a robust latent space by simulating sampling noises. The second stage, post-training, further refines the tokenizer's decoder to align it with the specific distribution of latents generated by a pre-trained generative model. The paper presents a new evaluation metric, pFID, to correlate tokenizer performance with generation quality. Through extensive experiments on autoregressive models, the authors demonstrate that their method significantly improves generation quality.

### Strengths
- The paper provides a novel analysis of the discrepancy between reconstruction and generation performance in token-based generative models. It argues that tokenizer robustness, not just reconstruction fidelity, is a critical factor for high-quality image synthesis.
- The proposed two-stage training scheme presents a practical and effective solution.

### Weaknesses
- The proposed method of injecting noise into the latent space to achieve robustness bears a strong resemblance to VAE. To properly contextualize this contribution, the authors should provide a direct comparison with VAVAE [1], another relevant work that also leverages VAE and DINO features.

- The paper's ablation studies should be expanded to include experiments conducted *without* the DINO distillation loss. First, to disentangle the performance gains attributable to the proposed latent perturbation from those provided by DINO; and second, to validate whether the noise injection strategy remains effective in the absence of DINO semantic guidance. Furthermore, the motivation for incorporating DINO distillation is not explicitly justified and appears to be treated as a default choice, whereas its role and necessity should be clearly articulated.

- As shown in Tables 5 and 7, the proposed latent perturbation strategy, while boosting generative performance, consistently degrades reconstruction quality. This introduces a critical trade-off between generation and reconstruction that the authors should discuss in detail. A thorough analysis of this trade-off is needed to understand its implications and to clarify whether this degradation is a fundamental compromise or a side effect that could be mitigated.

[1] Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models

### Questions
See weakness

### Soundness
3

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
4

### Summary
The authors address the mismatch between clean encoder latents and generated tokens produced by a pretrained generator.
To mitigate this discrepancy, they first train the tokenizer with random latent perturbations, improving decoder robustness to noise.
In a second stage, they fine-tune the decoder on the generator’s actual reconstruction errors to further align it with the generator’s distribution.

### Strengths
- Achieves strong FID despite using a relatively small model.
- Clearly demonstrates the weakness of training tokenizers independently from generators.
- Introduces a simple yet effective strategy to simulate generator errors through post-training.

### Weaknesses
- Requires an additional ~50 epochs of training — could this process be done jointly with generator training?
- The generator’s parameter count is small, but encoder–decoder size and efficiency are not discussed.
- While rFID improves, comparisons on PSNR or SSIM would strengthen the evaluation.

### Questions
- Is each tokenizer specific to a given generator, or can it generalize across generators?
- FID is strong, but how does it perform on the validation set? Could the model be overfitting? What about visualization on more challenging classes where other generator usually fails?
- Could the authors elaborate on the role and impact of the DINO contrastive loss?
- pFID correlates more closely with gFID than rFID. why does this relationship hold?

### Soundness
3

### Presentation
3

### Contribution
3
