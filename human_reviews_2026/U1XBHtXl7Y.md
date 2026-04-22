# Video Unlearning via Low-Rank Refusal Vector

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Video generative models achieve high-quality synthesis from natural-language prompts by leveraging large-scale web data. However, this training paradigm inherently exposes them to unsafe biases and harmful concepts, introducing the risk of generating undesirable or illicit content. To mitigate unsafe generations, existing machine unlearning approaches either rely on filtering, and can therefore be bypassed, or they update model weights, but with costly fine-tuning or training-free closed-form edits. We propose the first training-free weight update framework for concept removal in video diffusion models.
From five paired safe/unsafe prompts, our method estimates a refusal vector and integrates it into the model weights as a closed-form update. A contrastive low-rank factorization further disentangles the target concept from unrelated semantics, it ensures a selective concept suppression and it does not harm generation quality. Our approach reduces unsafe generations on the Open-Sora and ZeroScopeT2V models across the T2VSafetyBench and SafeSora benchmarks, with average reductions of 36.3% and 58.2% respectively, while preserving prompt alignment and video quality. This establishes an efficient and scalable solution for safe video generation without retraining nor any inference overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a training-free weight update framework for removing unsafe concepts from video diffusion models. The method leverages low-rank refusal vectors derived from pairs of safe/unsafe prompts, refined via contrastive PCA  to disentangle target concepts from unrelated semantics. By integrating these vectors into model weights through closed-form updates, the approach achieves permanent suppression of unsafe content without retraining or inference overhead. Experiments on OPEN-SORA and ZEROSCOPET2V across T2VSafetyBench and SafeSora benchmarks show average reductions in unsafe generations, while preserving video quality and prompt alignment.

### Strengths
1. The work introduces a training-free weight update framework for video unlearning, addressing a critical gap in scalable safety for video generative models. Unlike filtering methods or fine-tuning approaches, it enables irreversible, weight-level suppression with no computational overhead at inference.

2. The method demonstrates strong empirical performance across diverse benchmarks and models, with significant reductions in unsafe content while maintaining visual quality and semantic alignment. The use of only five prompt pairs and closed-form updates enhances its practicality for real-world deployment.

3. The integration of contrastive low-rank factorization to isolate unsafe concepts from neutral semantics is a thoughtful extension of prior concept-editing work to video domains, leveraging both text and image conditioning for improved precision.

### Weaknesses
1. Overreliance on linear representation assumptions: The core assumption—that unsafe concepts correspond to linear directions in latent space—is justified via references to LLM studies (e.g., Nanda et al., 2023), but video diffusion models have distinct architectures. No empirical validation such as visualization or concept disentanglement tests is provided to confirm this assumption holds for video models.

2. Ambiguity in "Neutral Concepts" for cPCA: The paper does not clearly define how "neutral concepts" are selected for cPCA. If neutral concepts share semantic overlap with unsafe concepts (e.g., "knife" vs. "violence"), subtracting their covariance (Ce) could inadvertently weaken the unsafe concept direction, reducing suppression efficacy. This risk is not discussed or evaluated.

3. While the method is tested on two models and benchmarks, it is unclear how it scales to multi-concept erasure (e.g., simultaneous removal of nudity and gore) or rare/unseen unsafe concepts. The mass erasure experiment (Section C.3) shows degraded performance for combined concepts, but the cause (e.g., overlapping directions) is not explored.

### Questions
Given that video diffusion models differ structurally from LLMs, have you validated that unsafe concepts in these models indeed form linear, separable directions in latent space? 

How are "neutral concepts" operationalized? If a neutral concept is highly correlated with an unsafe concept, does subtracting Ce reduce the magnitude of the unsafe direction in Cr - αCe? Please provide experimental analysis.

### Soundness
3

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
3

### Summary
This paper proposes the first training-free method for permanent concept removal in text-to-video diffusion models.
The method introduces a Low-Rank Refusal Vector (LRRV), computed from a small set of paired safe/unsafe prompts.
By applying contrastive PCA to isolate the target concept, the model performs a closed-form weight update that suppresses unwanted content (e.g., nudity, violence, trademarks) directly in the model weights. This contrasts with existing works that require expensive fine-tuning or unreliable inference-time filtering.

### Strengths
1. Efficiency: While training-free methods are well established in T2I unlearning, there was a timely need for such methods in T2V. The authors propose a simple and efficient algorithm to accomplish this task.
2. Strong Analysis: Rather than just proposing a new method, the authors provide strong ablations and hyperparameter sensitivity studies. This included the choice of rank, layers, prompt pairs, etc. These ablations provide a transparent view into the practicality of their proposed method. 
3. Strong Performance: Empirical results support the proposed method having superior unlearning performance while maintaining strong FVD and NM-Notox scores compared to existing T2V erasure methods.

### Weaknesses
1. Uniqueness to T2V: On line 119, the authors note that existing training-free unlearning methods developed for T2I generation are difficult to adapt to T2V models because of differences in text encoders and frame-independent architectures. However, after reading the method section, it remains unclear how the proposed approach is specifically tailored to the video setting. The described procedure of computing activation differences and applying PCA appears applicable to T2I models as well. This raises questions about how challenging it truly is to extend existing T2I unlearning methods to T2V. It would strengthen the paper if the authors either compared their method directly with established T2I training-free approaches such as UCE [1] and ConceptPrune [2], or provided a more detailed justification for why such methods cannot be effectively adapted to video models.
2. Robustness: Training-free unlearning methods for T2I models have been shown to be highly vulnerable to adversarial prompts [3]. It would be valuable to investigate whether this vulnerability also appears in the T2V setting, as doing so would provide a more comprehensive understanding of the robustness and real-world reliability of the proposed unlearning approach.
3. GPT Evaluation Metric: The authors use GPT-4o to assess whether each video frame contains an unsafe concept, following the evaluation protocol from prior work. However, I remain skeptical about the robustness and accuracy of this approach. Compared with specialized classifiers such as NudeNet [4], which is widely adopted for nudity detection in T2I unlearning. It would strengthen the evaluation if the authors validated GPT-4o’s judgments against established classifiers.


[1] Gandikota, Rohit, et al. "Unified concept editing in diffusion models." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2024.

[2] Chavhan, Ruchika, Da Li, and Timothy Hospedales. "Conceptprune: Concept editing in diffusion models via skilled neuron pruning." arXiv preprint arXiv:2405.19237 (2024).

[3] Zhang, Yimeng, et al. "To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[4] Bedapudi Praneeth, et al. bedapudi6788/NudeNet: Place for Checkpoint Files. v0, Zenodo, 19 Dec. 2019, https://doi.org/10.5281/zenodo.3584720.

### Questions
1. The authors state layers 17-18 are found to be empirically the best for erasing target concepts while preserving general generation quality. Are there any theoretical insights or empirical studies that could explain why this is the case?
2. Recent advances in T2I unlearning have focused on removing multiple concepts either simultaneously or sequentially [1,2]. It would be helpful to understand how well the proposed method scales in such settings. Can the approach effectively handle multiple concepts without significant performance degradation, or do interference effects arise when multiple refusal vectors are combined?


[1] Lu, Shilin, et al. "Mace: Mass concept erasure in diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[2]Cywiński, Bartosz, and Kamil Deja. "SAeUron: Interpretable concept unlearning in diffusion models with sparse autoencoders." arXiv preprint arXiv:2501.18052 (2025).

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a closed-form weight update for video diffusion models to suppress unsafe concepts by projecting out directions identified from a small set of safe/unsafe prompt pairs. 
It argues this yields no inference-time overhead, low compute to apply, and better utility–safety balance than other baselines evaluated on OPEN-SORA and ZeroScopeT2V.

### Strengths
1. The method is simple and fast to apply

2. Clear safety framing with concrete categories (copyright/tm, public figures, etc.) and ablations on rank/regularization.

3. quantitative results (FVD, MM-Notox) suggest limited quality drop on chosen backbones

### Weaknesses
1. The paper evaluates on OPEN-SORA and ZeroScopeT2V only and compares mainly to SAFREE (filtering) and NullSCE (fine-tuning). That omits several strong, contemporary T2V backbone, like Wan series and Hunyuan series. 

2. Safety measurement is narrow; key semantic-fidelity metrics are missing. I think more prompt-faithfulness semantic metrics (e.g., CLIP-text/video alignment, TIFA) to ensure you aren’t quietly degrading non-safety semantics that are not captured by FVD/MM-Notox.

3. Heavy reliance on an automated LLM judge; limited human validation. Do you have a human agreement study on this?

minor: ZEROSCOPET2V --> ZeroScopeT2V?

### Questions
please see the weaknesses section

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
4

### Summary
This paper introduces Refusal Vectors, a training-free, closed-form framework for suppressing unsafe or undesirable concepts in video diffusion models. The method operates by deriving a low-rank direction in weight space, the refusal vector, that encodes the difference between safe and unsafe prompt pairs. This vector is then embedded directly into the model weights as a closed-form update, enabling concept-level unlearning without retraining, original data access, or extra inference cost.

The key components include:

- A contrastive principal component analysis (cPCA) refinement that isolates unsafe semantics from safe content

- A low-rank update formulation that selectively suppresses unsafe concepts while maintaining generation quality.

- Empirical validation across multiple unsafe categories (e.g., pornography, violence, copyrighted content) on established benchmarks such as T2VSafetyBench.

- Results show that Refusal Vectors achieve substantial reductions in unsafe content while preserving prompt alignment and visual fidelity

### Strengths
- The work addresses an important and timely problem—ensuring the safety of large generative models without costly retraining or compromising utility.
- Unlike prior approaches that rely on retraining, reinforcement learning, or prompt engineering, the proposed Refusal Vector method directly identifies and suppresses unsafe directions in weight space using only a small number of safe/unsafe prompt pairs.
- The study’s evaluation on multiple unsafe categories and benchmarks provides solid evidence for both effectiveness and generality of the method.

### Weaknesses
- The paper has limtied novelty
- The concept of refusal vectors [1], low rank updates to parameters/representations [2] has been introduced in previous works before but not for the video domain. So the its novelty is probably only the video domain
- The paper primarily reports a censorship rate metric and some qualitative visualizations. However, there is no systematic evaluation of how much the method preserves prompt alignment or perceptual quality. Without these measures, it is hard to verify that suppression does not lead to over-filtering or semantic drift.
- The paper’s analysis focuses on isolated unsafe concepts but does not discuss how the method performs on prompts that combine safe and unsafe elements
- The paper has incorrect citations: SAFREE (Schramowski et al., 2023) and  SAFREE (Yoon et al.,2025) 

[1] Arditi, Andy, et al. "Refusal in language models is mediated by a single direction." Advances in Neural Information Processing Systems 37 (2024): 136037-136083.

[2] Meng, Kevin, et al. "Locating and editing factual associations in gpt." Advances in neural information processing systems 35 (2022): 17359-17372.

### Questions
- How were the five safe–unsafe prompt pairs chosen? Were they manually designed or sampled from a benchmark?  Would using more diverse or adversarially generated prompt pairs alter the resulting refusal vector? It would help to understand whether the vector’s effectiveness stems from
- Could you provide quantitative metrics (e.g., CLIP text–video similarity, FVD) to show how much semantic fidelity is lost?
- Have you compared your approach to concept erasure or model editing methods such as ROME [2]?
- How does the method behave when unsafe concepts co-occur with safe or semantically entangled ones?
- Can multiple refusal vectors be composed without interference?

### Soundness
3

### Presentation
2

### Contribution
3
