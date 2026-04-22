# Adapting in the Dark: Towards Stable and Efficient Black-Box Test-Time Adaptation

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Test-Time Adaptation (TTA) for black-box models accessible only via APIs presents a significant yet largely unexplored challenge. 
Existing truly black-box methods are scarce; post-hoc output refinement shows minimal benefit, while naively introducing Zeroth-Order Optimization (ZOO) for prompt tuning at test time suffers from prohibitive query costs and catastrophic instability. To address these challenges, we introduce **BETA** (Black-box Efficient Test-time Adaptation), a novel framework that enables stable and efficient adaptation for both standard Vision Models and large Vision-Language Models.  BETA uniquely employs a lightweight, local white-box steering model to create a tractable gradient pathway for optimization, circumventing the need for expensive ZOO methods. This is achieved through a prediction harmonization technique that creates a shared objective, stabilized by consistency regularization and a prompt learning-oriented filtering strategy. Requiring only *a single API call per test sample*, BETA achieves a +7.1\% gain on a ViT-B/16 model and a +3.4\% gain on powerful CLIP models; remarkably, its performance *surpasses* that of certain white-box and gray-box TTA methods (e.g., TENT and TPT). This practical effectiveness is further validated on a real-world commercial API, where BETA achieves a +5.2\% gain for just \$0.4—a 250x cost advantage over ZOO—establishing it as a robust and efficient solution for adapting models in the dark at test time.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a black-box test-time adaptation method motivated by black-box visual prompting literature. The proposed method leverages a white-box surrogate steering model to produce a harmonized prediction with a black-box target model, and leverages the gradient of this steering model to update the visual prompt and its layer-normalization parameter. The authors propose two additional mechanisms -- input consistency (prompted image v.s. clean image) regularization and confidence-based data filtering -- to stabilize the training. They demonstrate the proposed method, BETA, on ImageNet distribution shifts benchmarks with VIT and CLIP-VIT.

### Strengths
* `problem definition`
  * The authors are targeting a very important, promising, yet under-explored problem -- _black-box test-time adaptation_
  * Many frontier models do not reveal their parameters, and the authors' problem setup, assuming almost completely black-box (only the logit-accessibility), is very close to the real-world foundation model adaptation setup. I believe the problem they bring and want to address is one of the contributions of this work.
* `effectiveness of the solution`
  * The proposed method shows a somewhat impressive performance gain across diverse experiments, and sometimes beats the white-box method.
  * Although not sure about this due to largely ambiguous technical details, the proposed method seems like a query-efficient way to achieve such improvements.

### Weaknesses
* `weak rationale and logical gap`
  * The authors do not successfully justify the use of _prediction harmonization with asymmetric optimization_ and even _prediction harmonization_ at all. The results of Figure 2 are trivial because the $g_{\text{Ideal}}$ is derived by the interpolation of $g_{\text{BETA}}$ and $g_{\text{Black}}$, and here: (1) as the green line-- Obj. Relevance--is computed between $g_{\text{Ideal}}$ and $g_{\text{Black}}$, it does not support anything about $g_{\text{BETA}}$; (2) The gold gradient is $g_{\text{Black}}$ rather than $g_{\text{Ideal}}$, so the closeness between $g_{\text{BETA}}$ and $g_{\text{Ideal}}$ does not explain why BETA will achieve a desired gradient.
  * There is a logical gap between the authors' suggestion of _consistency regularization_ and its justification. Figure 3 does not tell us the necessity of the KL regularization between the clean input and the prompted input. It just tells us how unstable the optimizations are across different runs. Why do the authors claim that this is due to the prompt collapse? And why does the consistency regularization address this? 
* `missing formulation details and technical details`
  * What do they mean by $\mathcal{H(p_{x};f_{x})}$ in L214-218? they only provide definition of $\mathcal{H(p_{x})}$ in L193.
  * How many test samples (before and after filtering) were leveraged to learn your model?
  * How many epochs or iterations did you run for the optimization? This is very important for assessing the efficiency of the black-box methods.
* `naive baseline implementation`
  * The considered implementations of black-box baselines, ZOO (CMA, RGF, SPSA), are too naive. It looks like just an adoption of ZOO for visual prompt learning (Bahng et al. 2022). They do not consider advanced implementations of black-box visual prompt learning that are already available (Oh et al. 2023; Park et al., 2024).
  * This makes it hard to assess the necessity of BETA's exclusive components -- the surrogate steering model and two stabilizing mechanisms.
* `improper comparison in the number of API calls`
  * The authors' statement "the existing ZOO method requires 16 API calls per sample, while 1 API call per sample for our method". However, I believe this is an improper statement given the difference in settings of black-box visual prompting and this work. 
  * The ZOO-based visual prompt learning method, e.g., BlackVIP (Oh et al. 2023) and ZIP (Park et al. 2024), was proposed in a few-shot classification setup where the number of samples required per task is $K\times C$ for the K-shot and C-class, where they set K as 16. Therefore, if C is 10, e.g., CIFAR-10, then the required number of samples for the prompt learning is 160 in total. Then, the learned prompt (or reparameterized prompt like BlackVIP) is used for all the upcoming test samples. => It does not mean it requires 16 API calls per sample.
  * Meanwhile, the proposed method BETA also learns a visual prompt from all the test samples (maybe tens of thousands or more for ImageNet variants). Therefore, BETA's prompt learning requires far more samples to learn a prompt than the existing ZOO method, and **the total number of API calls should be compared in terms of the total number of forward passes of the black-box model rather than per-sample API requirement.**

---

> Reference
- Bahng et al. 2022, "Exploring Visual Prompts for Adapting Large-Scale Models"
- Oh et al. 2023, "BlackVIP: Black-Box Visual Prompting for Robust Transfer Learning"
- Park et al., 2024, "ZIP: An Efficient Zeroth-order Prompt Tuning for Black-box Vision-Language Models"

### Questions
* Please clarify the technical and formulation details mentioned above.

---

### If there is any misunderstanding from me, please feel free to point it out. I would be happy to discuss.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a Test-Time Adaptation method (TTA) that is applicable to black-box setting. Through a steering model, and optimizing a visual prompt, the proposed BETA operates and adapts the prediction of black-box models and improves their performance under distribution shifts.
Experiments are carried out on standard ImageNet-C/S/R benchmarks showing consistent performance gain.

### Strengths
The main strengths of this work are: 

1) This paper studies a unique setting: adaptation under black box setting. This is very useful for many applications in the era of strong models deployed as APIs

2) The proposed BETA is simple, cost efficient, and effective

3) Experiments are thorough and results are convincing

### Weaknesses
Generally, there are no major weaknesses in my opinion for this paper. The following points are suggestions to make the argument stronger:

1) The writing of this paper can be vastly improved. Here are a list of suggestions to make the narrative more convincing and easier to follow:

1a) There is no clear mention why should we study TTA under the black-box setting as generally adaptation is a responsibility of the model provider, thus having full white-box access. One possible argument that can be added in the introduction is developing stronger inference of the already strong API at the user side.

1b) Table 1 is not referenced in the text. 

1c) Figure 1 can be aesthetically improved significantly. 

1d) It is not completely clear in Figure 2 why considering $\alpha \in [0.4, 0.6]$ is optimal. The text accompanying that figure describing the experiment should be revised and improved for better clarity.

2) While the experiments considered several baselines, there are two relevant competitors that went missed. Both DDA [A] and DiffPure [B] operate in the black box setting. Particularly, DDA denoises the received input with added perturbation (through diffusion process), in a slightly similar way to BETA. I would highly recommend including experimental comparison against them, especially [A]

3) There are some experimental ablations that are interesting to show and could complement the results in the paper:

3a) Does the learnt visual prompt have any semantic information? Does it denies the received corrupted image?

3b) The cost argument comparing BETA with ZO methods is convincing. One possible standardized way to compare these two schemes is through the computational constrained evaluation [C].

3d) While SAR is included in the comparison tables, ETA/EATA [D] are not. Any reason why? 

3f) All experiments in the main paper are conducted under the episodic evaluation setting. I would also recommend including evaluations under the continual / label imbalanced settings [E].

[A] Back to the source: Diffusion-driven test-time adaptation, CVPR 2023

[B] Diffusion Models for Adversarial Purification, ICML 2022

[C] Evaluation of Test-Time Adaptation Under Computational Time Constraints, ICML 2024

[D] Efficient test-time model adaptation with- out forgetting, ICML 2022

[E] Robust test-time adaptation in dynamic scenarios, CVPR 2023

I am happy to raise my score if the aforementioned points are addressed.

### Questions
Please refer to the weaknesses section

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The article addresses the problem of test-time adaptation in the black-box setting, i.e., where the developer has no access to the model parameters but only to the input and output, treating the model as an API. The article formalizes this setting and proposes a model for it, BETA. BETA instantiates a local steering model that is trained to minimize the entropy for the prediction of steering and the original model combined on samples with high confidence predictions from the latter. A visual additive prompt is also tuned using consistency on the original model output with clean inputs. Finally, the normalization layers of the steering model are also updated using a steering loss, accounting for reliability and non-redundancy of samples. Results show advantages w.r.t. alternative approaches on various benchmarks.

### Strengths
1. The article gives a clear formalization of the problem, with 3.1 describing the various TTA settings and their assumptions. The black-box one indeed is very challenging, and of clear importance despite the little attention given to it.

2. The analyses of Fig. 4 showing the API actual cost on a commercial platform are a very interesting experiment, clearly demonstrating the (monetary) advantages of adopting the proposed pipeline. Given that black-box models are treated as APIs, this table makes a strong point in favour of adopting the proposed approach.

### Weaknesses
1. The idea behind the approach (i.e., improving test-time performance using an auxiliary model) is interesting, yet it might be arguable whether we are adapting the target model or mostly distilling its knowledge into a smaller one. In particular, while Tab. 7 shows the limitations of performing TTA on the steering model, it would be interesting to show the results of simple distillation of the output from the target to the steering model, as distillation strategies have already been shown to be effective for boosting the performance of smaller models on TTA [a].

2. This might be subjective, but I found Sec. 3 quite hard to follow. The method is complex, involving several design choices (e.g., perturbing the input, learning a prompt, updating normalization layers), and the flow between the different components requires some back and forth. For instance, the model introduces the aggregated prediction score with Eq. (1), but the first loss that is introduced is the consistency loss on the predictions of the steering model. Eq. (1) and its definition are used only in Eq. (3) later, whilst being the first thing that is defined. What is updated for each objective is not always clear (i.e., the consistency updates only the input or also the normalization layers?). Changing the structure by starting with the parts where the target and steering model communicate and then moving to the various regularization objectives (a flow followed also in the introduction) might ease the understanding of the approach. 

3.  The set of baselines is limited. In particular, there exist TTA methods based on augmentation strategies (e.g., [b,c]), not requiring any training or access to the parameters of the model, but just input modifications. Moreover, there are approaches tailored for black-box TTA on VLMs (i.e., Meng et al. 2025/[d]), and it is unclear why they are categorized as gray boxes (lines 150-151), as, to my knowledge, it does not assume access to internal features/tokens. Note that the VLM approaches even outperform the proposed strategy on VLM benchmarks: e.g., in Tab. 4, the proposed BETA performs 76.0 on ImageNet-S against the 77.2/80.8 of [c] and 76.6 of [d]. Enlarging the set of baselines for the black-box part (even by adapting approaches, when feasible) would strengthen the experimental results and thus the main claims in terms of effectiveness.

4. While the approach is claimed as "efficient", this refers to the number of API calls (e.g., Tab. 8) rather than the computational cost. Compared to alternatives (e.g., augmentation strategies above, Meng et al. 2025, Boudiaf et al. 2022), the approach may require more time due to the training objectives and updates. A deeper analysis (e.g., memory, speed) would be helpful to understand the tradeoff between API and computational costs of various approaches.

5. Following on the previous point, the article points to the stability of the approach, mostly referred to as the robustness ot noisy/signals that can lead to degenerate solutions (e.g., lines 70-76). However, there is no analysis on, e.g., robustness of the approach w.r.t. order of test data (e.g., i.i.d. vs non i.i.d., as in [e,f]), batch size, and/or other changes in the testing conditions. Adding these analyses would strengthen the claim of the efficacy of the approach. 

6. The way the steering is initialized might have a huge impact on the performance. Specifically, while Tab. 6  demonstrates that assumptions on the architectures can be easily overcome, there is no analysis on whether the pretraining dataset between the source and target should match. This could be interesting to understand which type of assumptions w.r.t. the target model are needed to ensure the effectiveness of the framework.

7. While the motivation of the work is clear, given that VLMs are (arguably) more prone to be released as black-boxes, plus they have been the focus of various recent TTA works, it would have been interesting to extend the analysis to more settings of those for test-time adaptation of VLMs (e.g., fine-grained datasets, full set of ImageNet variants (e.g., (Manli et al., 2022) [c,d]) instead of limiting it to a single architecture and two scenarios as of Tab. 4.

**References:**

[a] Zhao, Shuai, et al. "Test-time adaptation with clip reward for zero-shot generalization in vision-language models." ICLR 2023. \
[b] Shanmugam, Divya, et al. "Better aggregation in test-time augmentation." ICCV 2021.\
[c] Farina, Matteo, et al. "Frustratingly easy test-time adaptation of vision-language models." NeurIPS 2024.\
[d] Meng, Fan'an, et al. "Black-Box Test-Time Prompt Tuning for Vision-Language Models." AAAI 2025.\
[e] Gong, Taesik, et al. "Note: Robust continual test-time adaptation against temporal correlation." NeurIPS 2022.\
[f] Niu, Shuaicheng, et al. "Towards stable test-time adaptation in dynamic wild world." ICLR 2023.

### Questions
1. What are the advantages of the approach vs distillation based strategies?
2. Which of existing approaches for TTA or test-time augmentation could be included as competitors? How would the model perform against them?
3. Is the model efficient in terms of computational cost?
4. Is the model robust to batch ordering and size?
5. How is the steering model initialized?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a novel framework, BETA, designed to address the challenge of adapting pre-trained models in the black-box setting, where only API access is available. The algorithm itself is quite innovative, particularly in how it combines prediction harmonization, steering model regularization, and visual prompt optimization to achieve efficient adaptation. The design of the framework is well thought out, and it successfully balances the trade-offs between optimizing the visual prompt and ensuring model stability during test-time adaptation.

### Strengths
(1) Novel Algorithm Design: The combination of prediction harmonization, steering model fine-tuning, and entropy-based filtering introduces an innovative way to improve black-box adaptation. This approach is technically sound and presents a significant improvement over existing methods like ZOO, LAME, and other black-box TTA strategies.

(2) Stabilization Mechanisms: The use of consistency regularization and non-redundant sample filtering addresses the common issue of instability in adaptation, ensuring more reliable performance.

### Weaknesses
However, I have serious concerns regarding the paper's claim that it solves the black-box adaptation problem. The authors state:
"In this work, we addressed the critical challenge of adapting powerful models in the strict black-box setting where only API access is available."

While the approach is indeed innovative, there is a major inconsistency between this claim and the method proposed. The key issue is that the authors rely on a steering model that is pre-trained on source data, which contradicts the idea of black-box adaptation. In the strict black-box setting, the very essence is to prevent access to model parameters to avoid data leakage, model distillation, and related security risks.

However, the introduction of a steering model—which is an accessible, pre-trained model—is essentially a workaround that compromises the security and privacy principles of black-box models. The steering model, though lightweight, still allows access to certain information that the black-box framework intends to protect. This introduces a critical flaw, as real-world black-box models (such as those deployed in secure environments) are designed specifically to avoid such exposure. Thus, the approach outlined in this paper is unlikely to be directly applicable in many practical scenarios where security and privacy are paramount.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
