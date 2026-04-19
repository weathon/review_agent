# Gradient-based Jailbreak Images for Multimodal Fusion Models

- Decision: Reject
- Scores: 3, 3, 3, 8, 8

## Abstract
Augmenting language models with image inputs may enable more effective jailbreak attacks through continuous optimization, unlike text inputs that require discrete optimization. However, new *multimodal fusion models* tokenize all input modalities using non-differentiable functions, which hinders straightforward attacks. In this work, we introduce the notion of a *tokenizer shortcut* that approximates tokenization with a continuous function and enables continuous optimization. We use tokenizer shortcuts to create the first end-to-end gradient image attacks against multimodal fusion models. We evaluate our attacks on Chameleon models and obtain jailbreak images that elicit harmful information for 72.5% of prompts. Jailbreak images outperform text jailbreaks optimized with the same objective and require 3x lower compute budget to optimize 50x more input tokens. Finally, we find that representation engineering defenses, like Circuit Breakers, trained only on text attacks can effectively transfer to adversarial image inputs.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a gradient-based jailbreak method for multimodal fusion models. The authors introduce tokenizer shortcuts to solve the problem of continuous optimization not being carried out in the multimodal fusion model due to the discretization of input modalities. The experimental evaluation is carried out on the Chameleon multimodal fusion model. The results show that their method can trigger the generation of harmful information.

### Strengths
1. The approach is straightforward, relying on a fully connected network structure to approximate image tokenization.
2. The research addresses an important problem by targeting vulnerabilities in multimodal fusion models.

### Weaknesses
1. The effectiveness of the proposed approach is not well-validated. Without the tokenizer shortcut, the method's performance declines significantly, suggesting it may lack robustness in different settings.

2. From Table 2, the attack success rate drops when adding the refusal prefix part. The enhanced loss function, which aims to reduce the probability of generic refusal tokens, does not demonstrate a clear benefit in the experiments.

3. The approach's effectiveness is further limited when defenses are in place, raising concerns about its resilience against common protective measures.

4. Practical applicability is limited as the approach relies on assumptions that may not align with realistic conditions.

4.1 In direct attack scenarios, the method presumes the target model has been modified to include the shortcut, but it is unlikely defenders would incorporate this modification.

4.2 The approach also lacks sufficient transferability, reducing its usability across different models or settings.

5. The compared baselines are limited, just focusing primarily on text-based attacks GCG. A broader selection of attack methods would improve the robustness of the evaluation.

6. The use of $\Delta$PPL to measure adversarial prompt effectiveness lacks sufficient validation as a reliable metric.

### Questions
1. How can the robustness of the proposed method be improved to maintain effectiveness across diverse settings, especially in the absence of the tokenizer shortcut? Can the authors evaluate performance under different tokenization schemes?

2. Why do we need the enhanced loss function? 

3. Can the authors evaluate their method on other defenses, such as those mentioned by Jain et al. [1]?

5. What are the ablation results of changing the number of layers in the fully connected network or replacing it with other simple architectures?

6. Can the authors include additional baseline methods to more comprehensively assess the robustness and effectiveness of the proposed method, such as the FGSM, PGD, or any other reliable attack methods?

7. How can $\Delta$PPL be further validated as a reliable metric? For example, evaluating $\Delta$PPL's effectiveness in multimodal fusion models with an F1 score would provide a clearer, more reliable assessment.

[1] Jain, N., Schwarzschild, A., Wen, Y., Somepalli, G., Kirchenbauer, J., Chiang, P. Y., ... & Goldstein, T. (2023). Baseline defenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposed jailbreak attacks on multimodal fusion models by introducing a *differentiable* tokenizer shortcut. This allows for continuous optimization of adversarial images intended to bypass model safeguards. It evaluates the effectiveness of such attacks on Chameleon models, achieving a higher attack success rate than text-only jailbreaks. The results suggest that representation engineering defenses for text attacks could also adapt to adversarial image inputs.

### Strengths
- **Well-structured:** The paper is well-written and describes the proposed method clearly.
- **Introduced Differentiable Tokenizer:** This paper proposes using a two-layer neural network to make image tokenization in a multimodal fusion model feasible, enabling continuous optimization and revealing its threats to jailbreak.

### Weaknesses
- The proposed method of modifying the model architecture (replacing the original tokenizer) to elicit the jailbreak does not make much sense; also, the perturbed (attacked) images lack transferability. Given that a text-based attack is already feasible to pose such threats, I tend to buy the proposed method that applies the traditional method of generating adversarial perturbations to a multimodal fusion model. This method, however, is neither novel nor practically applicable to my understanding.

- Using adversarial images to elicit model jailbreak is also not novel; the paper lacks some discussion and comparison with existing works on VLLM [1].

[1] Visual Adversarial Examples Jailbreak Aligned Large Language Models (AAAI 2024)

### Questions
- Could you provide more insights into why the experimental results demonstrated a higher Attack Success Rate (ASR) using the embedding shortcut compared to the 1-hot shortcut?
- While it is understandable that jailbreak images optimized for Chameleon-7B might not transfer effectively to larger models, have you explored or observed whether jailbreak images optimized on larger models could be effectively transferred to smaller ones, such as from a Chameleon-30B to a Chameleon-7B model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a method generating jail-break images that cause early-fusion VL models (especially those with discrete image tokens) to generate harmful content when jail-break images are appended with harmful prompts.
Unlike adapter-based VL models, which do not use image token discretization, the discrete tokenization of images in early-fusion VL models makes direct optimization through gradients challenging and limits the applicability of existing methods.
To address this, the paper proposes a tokenizer shortcut that bypasses the discrete tokenization process by replacing quantization with a shallow MLP module, enabling the computation of gradients. 
The experiments demonstrate the effectiveness of the proposed method for generating jail-break images under certain settings—specifically, a white-box attack.

### Strengths
1. This paper addresses an important reasearch topic of jail-breaking in VL-LLM models, considering the significant growing use of VL models in real world applications. Research in this direction seems essential.
2. This paper is well presented, making the paper easy to follow and understand.

### Weaknesses
1. There is a lack comparison or discussion with other condidates to make quantizqation differentiable. If the proposed method achieves very strong performance in generating jail-breaking iamges, current approach would be acceptable. However, it seems that the proposed method can generate jail-break images in very limited settings: with shortcut or non-transfer setting.
2. As far as i understand, the white-box attack scenario is important because, although it may be impractical and unrealistic, it serves as a useful benchmark for black-box attacks. However, for the "with shortcut" results, it effectively becomes equivalent to altering the model itself, which makes discussions of attack performance somewhat meaningless. Nonetheless, the proposed method is primarily evaluated using the shortcut when demonstrating its strong performance, (Table 1, 2, 3, 4).
3. Optimizing within the input (image and text) space is important, as it is a prerequisite for black-box settings or model transfer. However, as shown in Table 5, the proposed method fails to produce transferable samples and underperforms compared to the baseline.
4. (Minor) The paper seems to contain overclaims or insufficient explanations. For example:
	- The title of Table 3 is "Image jailbreaks outperform text attacks," but the proposed method performs worse than the text-only attack, GCG, in the Circuit Breaker setting. Additionally, comparing GCG with the proposed method "with shortcut" seems unfair, as "with shortcut" is equivalent to changing the model.
	- In discussions and future works, the paper states, "(412) Our work is the first attempt to jailbreak multimodal architectures using end-to-end gradient attacks" and "(423) this problem also persists in multimodal models,". I guess the "fusion-based model" shall be more appropriate.

### Questions
1. Please address the weaknesses.
2. (Suggestion) Moving the related work section front or refering that more related work is in the later part shall improve the understanding of the paper.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies white box attacks against multimodal fusion models. This setting is considered interesting because these models convert all inputs - both text and images - into a shared tokenized space. This approach could make the models vulnerable to more efficient attacks through image optimization, since images offer a continuous space to optimize (unlike text which is discrete). In order to optimize potential attack inputs, they develop the tokenization shortcut method, mapping image embeddings to a continuous model input space before quantization. They find that for whitebox optimization attacks, images are more effective than text, however they do not beat other competitive baselines like representation engineering.

### Strengths
- The choice of studying robustness of multimodal fusion models is timely.
- The selection of research questions is fitting for a first study in a fast-paced field. The hypothesis that it may be easier to attack models with this architecture is interesting, and is very useful to study early in the uptake of architectures.
- The paragraph writing style is easy to read, and the work can serve as an interesting log of experiments for other practitioners.

### Weaknesses
- The choice of the two shortcut is not clearly explained in section 3. It would be useful to spell it out.
- It would be useful to have more qualitative analysis or at least examples of jailbreaking images vs images that fail.

### Questions
- Your experiments seem interesting, and it seems like you may have opinions on future work. While you have already provided motivation for experiment design, it would be useful to add more detail to your results so that it is easier to judge what puzzles are worth investigating. For instance, it would be great to spell out the details of transferability experiments. The observation itself is cool, but the current presentation of the work makes it so that readers will have to reimplement your work to get started on forming hypotheses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
* The paper develops a white box gradient-based image jailbreak method for multimodal fusion models. Prior work on gradient-based image jailbreaks has focused on VLMs due to the lack of open source fusion models, but this has recently changed with the release of Chameleon.  
* The core challenge of doing this with fusion models is that gradients do not flow through to the input image due to a non-differentiable step in tokenization.   
* The authors solve this problem by introducing a novel “tokenizer shortcut” technique, where they train a small MLP to approximate the image tokenizer in a differentiable way. The tokenizer is then replaced by this approximation during adversarial image optimization, allowing gradient-based optimization to succeed.  
* Two versions of the tokenizer shortcut are developed, one mapping directly to embedding space and one producing a one-hot vocabulary encoding.  
* A comprehensive set of experiments are conducted. Key findings:  
  * Both shortcut methods produce images with high Attack Success Rate, but only the 1-hot shortcut images transfer to versions of the model that do not use the shortcut.  
  * Circuit breakers substantially reduce ASR  
  * Jailbreak images transfer easily across prompts but do not transfer across models  
* The authors also conduct a series of ablations, including on response prefix, softmax temperature, and number of train prompts.

### Strengths
* This is a novel method that solves the core challenge of creating gradient-based image jailbreaks for multimodal fusion models.  
* Understanding the vulnerabilities in multimodal models is important for developing more robust systems, and gradient-based jailbreaking of fusion-based models has been under-explored.  
* The authors use good baselines for their experiments (GCG and refusal direction attacks), and convincingly demonstrate the success of their method  
* The experiments are thorough and informative, testing attack transfer as well as defence using circuit breakers. Several ablations are also performed.

### Weaknesses
* The dataset used is quite small, with only 80 prompts in the test set for direct attacks and 20 in the test set for transfer attacks. The results would be more convincing if done on a larger dataset. In addition, only a single dataset is tested.  
* The paper does not include any examples of jailbroken model responses - these are helpful for qualitative understanding of the attack.
* With the exception of table 1, the results given are all for models using the tokenizer shortcut. It would be helpful to also include  the results when using the 1-hot jailbreak images on models without the shortcut in Tables 2 and 4.

### Questions
* Do the authors have an explanation for why the embedding space shortcut attacks do not transfer to non-shortcut models while the 1-hot attacks do?

### Soundness
3

### Presentation
4

### Contribution
4
