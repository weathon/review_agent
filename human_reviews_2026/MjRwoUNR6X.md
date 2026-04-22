# OSPA: Enhancing Identity-Preserving Image Generation via Online Self-Preference Alignment

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Identity-preserving text-to-image generation has recently received increasing attention, yet it remains a challenging task. Existing approaches typically fine-tune diffusion models, but they often fail to preserve identity information reliably. Reinforcement learning with human feedback (RLHF) can improve identity consistency, but it requires expensive reward models and carefully curated annotations, limiting its practicality. We present Online Self-Preference Alignment (OSPA), a plug-and-play framework that achieves identity-preserving generation without relying on external reward models or high-quality datasets. OSPA exploits self-preference signals through three components: (1) a self-preference sample generation module that perturbs a frozen policy model to produce paired samples with explicit preferences; (2) a self-reward preference optimization mechanism that updates the policy using group preference optimization; and (3) an online curriculum learning strategy that continuously refines the sample generator with feedback from the evolving policy model. Comprehensive experiments on four state-of-the-art identity-preserving text-to-image models demonstrate that OSPA substantially improves identity fidelity while maintaining visual quality, offering a general and effective alignment strategy for generative models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents Online Self-Preference Alignment (OSPA), a novel framework designed to improve identity-preserving text-to-image generation. The core motivation is to bypass the reliance on expensive human-annotated datasets or separately trained external reward models typically used in RLHF. OSPA achieves this by constructing a self-contained alignment loop: it generates its own preference pairs by perturbing identity embeddings, utilizes a frozen, pre-trained identity encoder to calculate "self-reward" scores based on cosine similarity, and employs an online curriculum learning strategy to iteratively refine the generative policy. Extensive experiments on four ID-preserving models (IP-Adapter, IP-AdapterPlus, InstantID, and InfiniteYou) demonstrate that OSPA consistently enhances identity fidelity while maintaining visual quality.

### Strengths
The most significant strength of this work lies in its pragmatic approach to a major bottleneck in personalized generation: the high cost of alignment. By ingeniously formulating a completely automated, self-referential optimization loop, OSPA effectively removes the need for external supervision during the alignment phase. This "plug-and-play" nature, verified across multiple diverse and strong baselines, highlights excellent generality and high potential for practical application in scalable model customization.

The quality of execution and clarity of presentation are also commendable. The paper is well-structured, and the core methodology is communicated effectively, particularly through well-designed figures like Figure 3. The empirical validation is robust, showing consistent quantitative improvements across various metrics (Face Sim, CLIP-I, FLIP-I) and baselines, suggesting that the proposed Group Preference Optimization (GPO) on self-generated data is a stable and effective training objective.

### Weaknesses
A primary conceptual weakness is the potentially misleading framing of "Self-Preference" and the unverified reliance on the specific pre-trained identity encoder. The "reward" signal is not truly intrinsic to the generative model but is distilled from a frozen, external discriminative model (the face encoder). The entire framework's upper bound is thus locked by this specific encoder's ability to act as a perfect proxy for human identity perception. The paper lacks a critical analysis of how sensitive the entire system is to the choice of this encoder. If the encoder has biases (e.g., focusing too much on low-level textures rather than high-level facial structure), OSPA will amplify these biases.

Furthermore, the experimental section lacks a crucial, simpler baseline to justify the complexity of the proposed preference optimization framework. Since the method relies entirely on the frozen ID encoder for scoring, a straightforward "Rejection Sampling Fine-tuning" approach—generating N samples, selecting the best one based on the same ID encoder score, and performing standard supervised fine-tuning—should be compared. If OSPA does not significantly outperform this much simpler strategy, the necessity of the complex paired-sample generation and GPO loss becomes questionable.

Finally, the method appears highly sensitive to the noise intensity hyperparameter α, as indicated in Figure 6, where performance degrades sharply outside a narrow window. The manuscript does not sufficiently detail the strategy for selecting α across the widely different baselines (e.g., IP-Adapter vs. InstantID). If fine-grained, per-model grid search is required for this parameter, it undermines the claimed "plug-and-play" ease of use and suggests potential fragility in new applications.

### Questions
Could you provide an ablation study or at least a discussion on replacing the currently used identity encoder with a different one (e.g., a different face recognition architecture, or a weaker generic encoder)? This is critical to verify whether the success of OSPA is due to the general framework or specifically tied to the high quality of the chosen "judge" encoder.

How does OSPA compare quantitatively to a simpler "Generate-Filter-Finetune" baseline using the same ID encoder as the filter? Demonstrating a clear margin over this baseline is necessary to robustly justify the added complexity of your online paired-preference learning framework.

Please clarify the exact strategy used for selecting the critical noise hyperparameter α for each of the four baselines. Was a single value universally effective, or was individual tuning necessary? If tuning was needed, how sensitive are the results to small variations in α for the different architectures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focuses on addressing limitations in identity-preserving text-to-image generation (e.g., supervised fine-tuning lacking feedback, RLHF relying on costly external resources) and proposes OSPA (Online Self-Preference Alignment), a plug-and-play framework. OSPA uses three core modules: self-preference sample generation, self-reward optimization, and online curriculum learning. Experiments have shown the effectiveness of the method. .

### Strengths
1. OSPA eliminates the need for expensive external reward models and high-quality curated preference datasets, which are required by existing methods like RLHF-based approaches. Instead, it leverages self-generated preference signals and intrinsic self-reward mechanisms, reducing costs and practical constraints .
2. As a flexible framework, OSPA can be seamlessly applied to multiple SOTA identity-preserving text-to-image models without modifying their core architectures. This broad compatibility makes it highly applicable to existing systems.
3. Extensive experiments show OSPA enhances identity preservation while maintaining or even improving visual quality, addressing the trade-off faced by many existing methods .
4. This paper is well-written and easy to follow.

### Weaknesses
1. OSPA operates on top of pre-trained identity-preserving text-to-image models, and its effectiveness is directly influenced by the quality of these underlying baselines. As explicitly stated in the paper, stronger baselines lead to larger performance gains, while weaker baselines limit the extent of improvement. This means OSPA cannot independently address the inherent flaws of poor-quality baseline models, which restricts its applicability.
2. The self-preference sample generation module relies on Gaussian noise perturbation to create preferred/unpreferred sample pairs. However, experiments show that increasing noise intensity significantly reduces facial identity similarity. This indicates that OSPA’s noise perturbation strategy requires careful parameter tuning. The lack of an adaptive noise adjustment mechanism makes it less robust.
3. For evaluation, this work relies on just 30 reference images from FFHQ and 40 prompts. The small scale and narrow scope of the datasets may limit the generalization of OSPA’s performance to real-world scenarios with more varied identity types and text prompts.
4. The experiments only validate OSPA under standard conditions. There is no testing on extreme scenarios, such as low-quality reference images (blurred, occluded, or low-light), complex text prompts, or cross-domain identity preservation (e.g., generating real-world photos from stylized images).

### Questions
1. OSPA operates on top of pre-trained identity-preserving text-to-image models, and its effectiveness is directly influenced by the quality of these underlying baselines. As explicitly stated in the paper, stronger baselines lead to larger performance gains, while weaker baselines limit the extent of improvement. This means OSPA cannot independently address the inherent flaws of poor-quality baseline models, which restricts its applicability.
2. The self-preference sample generation module relies on Gaussian noise perturbation to create preferred/unpreferred sample pairs. However, experiments show that increasing noise intensity significantly reduces facial identity similarity. This indicates that OSPA’s noise perturbation strategy requires careful parameter tuning. The lack of an adaptive noise adjustment mechanism makes it less robust.
3. For evaluation, this work relies on just 30 reference images from FFHQ and 40 prompts. The small scale and narrow scope of the datasets may limit the generalization of OSPA’s performance to real-world scenarios with more varied identity types and text prompts.
4. The experiments only validate OSPA under standard conditions. There is no testing on extreme scenarios, such as low-quality reference images (blurred, occluded, or low-light), complex text prompts, or cross-domain identity preservation (e.g., generating real-world photos from stylized images).

### Soundness
2

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
4

### Summary
The paper introduces a novel method, Online Self-Preference Alignment (OSPA), designed to overcome the requirement for human input or external reward models in current methods for identity-preserving text-to-image generation. By utilizing only the policy model for alignment, OSPA demonstrates commendable performance.

### Strengths
1. The three core components of the proposed OSPA method are well-motivated and specifically designed to address distinct problems, although some of them appear incremental.

2. The visualization and experimental results presented are compelling.

### Weaknesses
#### Disadvantage/Limitations

##### Minor Limitations

###### a) Typographical and Formula Errors

1. In Algorithm 1, line 8 (or line 227 of the manuscript), the term $\mathrm{sim}(\bm{v}_{\mathrm{ref}}, \mathcal{E}_{I}(\bm{x}_{\mathrm{gen}}^{u_{i}}))$ appears to be a typo. I suspect it should instead be $\mathrm{sim}(\bm{v}_{\mathrm{ref}}, \mathcal{E}_{I}(\bm{x}_{\mathrm{gen}}^{p_{i}}))$ to align with the intended logic of comparing the generated preferred image with the reference vector.

2. If Equation (6) is correct as written, the term involving $\epsilon_{\mathrm{ref}}$ cannot function as a regularizer for the parameter $\epsilon_{\theta}$ as they appear disconnected. I suggest the authors verify if the second term should be $\|\epsilon_{\theta}(\mathbf{x}_{t}^{i}, t) - \epsilon_{\mathrm{ref}}(\mathbf{x}_{t}^{i}, t)\|^{2}_{2}$. Furthermore, the bracket placement seems incorrect; the scaling factor $A_{i}$ should likely only multiply the first term of the loss function.

###### b) Writting

1. The paper references $\mathrm{DDIM}_{\mathrm{sample}}$ in Equations (3) and (4). While DDIM is a common method, the authors must provide the detailed algorithm and corresponding settings in the Appendix for reproducibility (e.g., the time schedule, the exact number of sampling steps, and whether stochasticity is used). Similarly, Equation (5) references the similarity function, $\mathrm{sim}$. The authors must state how this similarity is calculated (e.g., cosine similarity or Euclidean distance).

##### Major Limitations

1. The precise configuration of the "noise identity" (specifically the scale factor $A_i$) is unclear, leading to questions about the critical initial **self-preference sample generation**. This initial step is vital, as subsequent Reinforcement Learning (RL) and online fine-tuning rely entirely on this synthetic dataset. I think the authors should provide an **ablation study** to explicitly validate the effect and necessity of the scale factor $\alpha$.

2. The experiments are exclusively conducted on facial datasets. Given that related work (e.g., IP-Adapter [1]) has validated identity-preserving techniques on diverse non-face datasets, the proposed OSPA method should also be tested on broader image categories to demonstrate its generalizability. Furthermore, I think OPSA should be compared against methods that use external reward models (like ID-Aligner [2]) or human-annotated data. Even a slight performance degradation is acceptable, as a direct comparison is necessary to validate the claim that the cost reduction (no external models/human data) justifies the trade-off.

[1] Ye, H., Zhang, J., Liu, S., Han, X., & Yang, W. (2023). IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721.
[2] Chen, W., Zhang, J., Wu, J., Wu, H., Xiao, X., & Lin, L. (2024). ID-Aligner: Enhancing identity-preserving text-to-image generation with reward feedback learning. arXiv preprint arXiv:2404.15449.

### Questions
1. Regarding the use of DDIM, since the sampler is typically deterministic, I am confused by Figure 3(a), which shows a diverse set of images in the preference group generated from a single image and text input. Can the authors clarify where the randomness is introduced during the sampling process (apart from the initial noise injection for the preference/unpreference images)? Additionally, while the paper specifies the timesteps used for gradient updates, the total sampling steps for DDIM is missing (e.g. 50 steps or 100 steps). Also, can the authors offer the memory cost for a single gradient descent step and the overall time cost for training the model.

2. The authors validate Gaussian noise over "salty noise," but there are many other viable perturbation strategies for the embedding space. If possible, could the authors explore and conduct experiments using alternative perturbation methods, such as randomly replacing parts of the embedding vector with Gaussian noise or randomly swapping values within the embedding?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Online Self-Preference Alignment (OSPA), a plug-and-play framework for enhancing identity-preserving text-to-image generation. Instead of relying on external reward models or high-quality curated datasets (limitations of existing supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) approaches), OSPA achieves identity preservation via self-generated preference signals.

### Strengths
1. The paper repurposes generative models’ stochasticity to generate self-preference pairs via Gaussian noise on embeddings, eliminating the need for external annotations or models.
2. As a plug-and-play framework, the method adapts to 4 SOTA baselines (e.g., IP-Adapter, InstantID) by only updating adapters/projectors, ensuring easy deployment.
3. Multiple metrics (face similarity, CLIP-I, FLIP-I) and ablation studies (noise, online/offline updates) confirm the reliability of OSPA’s performance.

### Weaknesses
1. The paper compares OSPA to SFT-based baselines but not to RLHF/DPO methods (e.g., ID-Aligner, Diffusion-DPO) on the same dataset. Without this, readers cannot fully assess whether OSPA’s gains are due to its design or simply the choice of baselines. 
2. While Fig. 6 shows noise intensity impacts face similarity, the paper does not explain how to choose α (noise coefficient) for different models. For example, IP-Adapter uses α=0.025, InstantID uses α=0.04—what guides this choice? 
3. The GPO loss (Eq. 6) references "shifted timestep sampling (H) from SD3" but provides no details on how H is configured (e.g., timestep range, sampling frequency). Without this, researchers cannot replicate the loss function accurately.

### Questions
1. What are OSPA’s main failure cases? For example, does it struggle with: (a) reference images with occlusions (e.g., glasses, masks)? (b) prompts with extreme style changes (e.g., "person as a cartoon")?
2. In online curriculum learning, continuously updating the policy with the optimized target model may cause distribution drift. Have the authors theoretically proven the convergence conditions of the OSPA framework? In experiments, how is it determined that the model has reached a stable state? Is there a risk of decreased identity fidelity in the later training stages due to over-exploration?

### Soundness
2

### Presentation
2

### Contribution
2
