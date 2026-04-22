# AdPO: Enhancing the Adversarial Robustness of Large Vision-Language Models with Preference Optimization

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Large Vision-Language Models (LVLMs), such as GPT-4o and LLaVA, have recently witnessed remarkable advancements and are increasingly being deployed in real-world applications. 
However, inheriting the sensitivity of visual neural networks, LVLMs remain vulnerable to adversarial attacks, which can result in erroneous or malicious outputs. 
While existing efforts utilize adversarial fine-tuning to enhance robustness, they often suffer from significant performance degradation on clean inputs. 
In this paper, we propose AdPO, a novel adversarial defense strategy for LVLMs based on preference optimization. 
For the first time, we reframe adversarial training as a preference optimization problem, aiming to enhance the model’s preference for generating normal outputs on clean inputs while rejecting the potential misleading outputs for adversarial examples.
Notably, AdPO achieves this by solely modifying the image encoder, e.g., CLIP ViT, resulting in superior clean and adversarial performance in a variety of downstream tasks.
Due to the computational cost of training large language models, we show that training on smaller LVLMs and transferring to larger ones achieves state-of-the-art performance with efficiency comparable to previous methods.
Our comprehensive experiments confirm the effectiveness of the proposed AdPO which highlights the potential of preference-based learning in adversarially robust multimodal systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper reframes robustness optimization for vision–language models as a preference optimization problem. The authors propose Preferred Image Optimization (PIO) to encourage normal outputs on clean inputs, and Adversarial Image Optimization (AIO) to reject potentially misleading outputs on adversarial examples. Extensive experiments demonstrate the effectiveness of the approach.

Regarding novelty, I am not an expert in preference optimization and am unfamiliar with the detailed trajectory of this line of work, so my assessment of novelty is tentative and I will defer to other reviews.

### Strengths
1. The method is simple and effective, with empirical validation across a broad set of experiments.

2. Framing VLM robustness as a preference optimization problem is an interesting perspective.

### Weaknesses
1. Motivation is insufficient. On line 90 the authors state: “we explore fine-tuning the image encoder of a smaller LVLM and subsequently transferring it to a larger LVLM model.” While reasonable, I do not view this as a contribution by itself. Cost reduction by using a smaller LVLM is expected; the paper does not explain why this transfer should work or provide guarantees.

2. Typos. For example, in the caption of Figure 3, TeCoA is misspelled as CoTeA.

3. Figure readability. The axes in the three plots of Figure 4 are not clearly visible.

### Questions
1. What happens if the encoder is changed? Please provide an ablation on the image encoder choice and discuss sensitivity to encoder architecture/initialization.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes AdPO, a novel adversarial defense strategy for Large Vision-Language Models (LVLMs) based on preference optimization. The key insight is to reframe adversarial training as a preference learning problem, encouraging the model to favor correct outputs for clean inputs while rejecting misleading ones for adversarially perturbed inputs. The method combines Preferred Image Optimization (PIO) and Adversarial Image Optimization (AIO), the former maintaining clean accuracy and the latter enhancing robustness through dynamic fine-tuning.

Experiments across multiple LVLM benchmarks demonstrate that AdPO achieves state-of-the-art robustness with minimal degradation on clean performance. The paper also shows efficient transferability by training only the image encoder of small models and transferring it to larger LVLMs.

### Strengths
1. The study of adversarial training for LVLMs is timely and relevant. While many recent works have focused on CLIP-style vision-language models, the increasing real-world deployment of LVLMs (e.g., Gemini) makes it necessary to explore adversarial training specifically tailored for such models.
2. Figure 2 effectively helps readers quickly grasp the core idea of the AdPO method.
3. I particularly appreciate AdPO’s adversarial modeling approach based on preference optimization. It effectively optimizes two complementary objectives: increasing the probability of generating correct outputs for clean inputs while reducing the likelihood of producing malicious outputs for adversarial inputs. Moreover, it explicitly optimizes the probability of generating correct outputs under adversarial perturbations. This innovative training paradigm enables the model to achieve both excellent clean accuracy and strong adversarial robustness.
4. Extensive experiments across multiple benchmarks and models demonstrate the effectiveness and generality of the proposed method.

### Weaknesses
1. Typo error — the citation for CroPA in Line 467 is missing parentheses.
2. The experiments primarily evaluate results under low attack intensities (2/255, 4/255). Evaluating higher attack strengths (8/255, 16/255) would make the assessment more comprehensive.
3. PIO is described as a relative preference, yet $y_l$ is not necessarily incorrect in every generation, which could potentially affect the results.
4. The subheadings in Figure 4 appear to be a bit too small.

### Questions
Q1: If I understand correctly, in the main experiments neither the baseline methods nor AdPO have seen the subsequent language models during training. I would like to get confirmation from the authors that the comparison is entirely fair.

For other questions, please see Weaknesses

Overall, I believe this paper is a solid, well-executed work with clear motivation and convincing results.

### Soundness
4

### Presentation
3

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
This paper introduces AdPO, a novel adversarial training framework built upon preference optimization (DPO). Specifically, the authors propose Preferred Image Optimization (PIO) and Adversarial Image Optimization (AIO) to simultaneously enhance clean-image performance and adversarial robustness. The formulation of adversarial robustness through preference optimization is insightful, and the experiments demonstrate strong performance across multiple datasets. Furthermore, the method exhibits excellent transferability to larger LVLMs, achieving state-of-the-art results.

### Strengths
1. The formulation of adversarial robustness through preference optimization is insightful.
2. The performance of the proposed method demonstrate the effectiveness.
3. The paper is easy to follow and golocally presented.

### Weaknesses
1. Compared to previous methods such as TeCoA, AdPO requires substantially more computation. In particular, it needs to generate full responses for both the clean and adversarial images, which makes the overall training cost extremely high. It would be desirable if similar performance could be achieved with offline response generation.
2. In Equation (7) (line 299), it seems that y_l should be y_w instead.

### Questions
In Equation 5, what is y_w and y_l? Is y_w the response from clean image and y_l is the response from adversarial image?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes AdPO, a preference-optimization–based adversarial defense for large vision-language models (LVLMs). By reframing adversarial training as a preference learning problem, AdPO enhances robustness while preserving clean performance. The method introduces Preferred Image Optimization (PIO) and Adversarial Image Optimization (AIO), applied only to the image encoder for efficiency and transferability. Experiments on multiple LVLM benchmarks show that AdPO achieves state-of-the-art robustness without significant degradation on clean data.

### Strengths
1. The paper is well written and clearly structured, making the methodology and experimental results easy to follow.

2. The motivation is sound and well justified, addressing the important problem of improving adversarial robustness in large vision-language models.

3. The work presents a creative combination of existing ideas, integrating preference optimization with adversarial training for LVLMs.

4. The proposed defense method demonstrates strong and consistent performance, significantly improving adversarial robustness while maintaining clean accuracy.

5. The experimental evaluation is comprehensive and convincing, covering multiple open-source LVLM architectures, diverse benchmarks, and an analysis of various DPO variants to validate the generality and robustness of the approach.

### Weaknesses
1. The paper does not analyze potential failure cases of AdPO. It would be helpful to evaluate its robustness under adaptive attacks (where the attacker knows the defense) and transfer-based attacks generated from unseen models, to better understand the method’s practical limitations.

2. While the empirical performance of AdPO is impressive, the paper lacks a theoretical discussion connecting its objective to adversarial risk minimization. The approach largely builds upon the known DPO framework, and providing a clearer theoretical justification would strengthen the work’s originality and rigor.

3. (Minor) There are several typos in the submission. For example, in the caption of Figure 3, it should be TeCoA; in line 471, it should be “Flickr30K”; and there are similar minor errors around line 376. Careful proofreading is recommended before the camera-ready version.

### Questions
1. How would AdPO perform under transfer-based (black-box) attacks generated from unseen models?

2. Could the authors provide examples or qualitative analyses of failure cases, showing when AdPO fails and what patterns those failures share? This would help readers understand the limitations and potential future improvements.

3. Can the authors clarify whether AdPO’s optimization objective can be interpreted as a form of adversarial risk minimization? If so, what assumptions or approximations are required for this equivalence?

4. Is there a theoretical intuition or derivation showing how preference optimization contributes to robustness beyond empirical observations?

### Soundness
2

### Presentation
3

### Contribution
3
