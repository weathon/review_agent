# TLDR: Token-Level Detective Reward Model for Large Vision Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Although reward models have been successful in improving multimodal large language models, the reward models themselves remain brutal and contain minimal information. Notably, existing reward models only mimic human annotations by assigning only one feedback to any text, no matter how long the text is. In the realm of multimodal language models, where models are required to process both images and texts, a naive reward model may learn implicit biases toward texts and become less grounded in images. In this paper, we propose a **T**oken-**L**evel **D**etective **R**eward Model (**TLDR**) to provide fine-grained annotations to each text token. We first introduce a perturbation-based method to generate synthetic hard negatives and their token-level labels to train TLDR models. Then we show the rich usefulness of TLDR models both in assisting off-the-shelf models to self-correct their generations, and in serving as a hallucination evaluation tool. We show that TLDR automatically trains a token-level likelihood optimization, and can improve the base model's performance significantly. Finally, we show that TLDR models can significantly speed up human annotation by 3 times to acquire a broader range of high-quality vision language data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the limitations of current reward models in multimodal large language models (MLLMs), which often provide only binary feedback regardless of text length. The authors propose a Token-Level Detective Reward Model (TLDR) to offer fine-grained, token-level feedback instead of coarse binary annotations. The TLDR model uses a perturbation-based method to generate hard negatives and their token-level labels for training. The paper demonstrates the benefits of TLDR models in helping models self-correct and serving as a hallucination evaluation tool. Additionally, TLDR models can accelerate human annotation by three times, improving the acquisition of high-quality vision-language data.

### Strengths
1 The TLDR model provides detailed token-level feedback, addressing the issue of overly simplistic binary annotations, which is a significant improvement in the evaluation process for MLLMs.

2 The perturbation-based approach for generating synthetic hard negatives adds diversity and robustness to the model training, improving the model’s ability to handle difficult scenarios.

3The model can be applied in multiple contexts, both for self-correcting MLLM outputs and as a hallucination evaluation tool, making it a versatile contribution.

### Weaknesses
1. Although the paper introduces the concept of token-level feedback, this approach feels like a natural extension of existing reward modeling techniques, rather than a fundamentally new innovation. The idea of fine-grained feedback has already been explored in other contexts, and the paper does not sufficiently differentiate its contribution from prior work.
2. This paper didn't compare the proposed model with existing hallucination detection and mitigation methods, such as Woodpecker or other state-of-the-art techniques. Additionally, Furthermore, the paper introduces self-correction as a novel advantage of TLDR, but other metrics and models used in hallucination detection could also be adapted to perform self-correction.
3. In Section 5.3, the authors evaluate the self-correction capabilities of the TLDR model using only one dataset, WinoGround. Relying on a single dataset raises concerns about the generalizability and reliability of the results.
4. The proposed hallucination evaluation didn’t show the superiority over the existing hallucination evaluation methods.
5.  Lack of comprehensive survey of hallucination on Large Vision-Language Models.
[1] Object hallucination in image captioning
[2] Evaluating Object Hallucination in Large Vision-Language Models
[3] FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models
[4] Analyzing and mitigating object hallucination in large vision-language models
[5] FGAIF: Aligning Large Vision-Language Models with Fine-grained AI Feedback
[6] Negative Object Presence Evaluation (NOPE) to Measure Object Hallucination in Vision-Language Models
6. Lack of evluation for the synthetic data. Meanwhile, the robustness of trained models highly rely on the quality of the synthetic data.
7. The models used to validate the SELF-CORRECTION WITH TLDR MODELS are too few, which limits the robustness of the results.

### Questions
See weakness

### Soundness
2

### Presentation
3

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
This paper presents a token-level reward model for large vision-language models, where the reward model would produce reward labels for each token during the generation. 

To obtain a training dataset for the reward model, this paper proposes to perturb image caption/VQA answer with an LLM (i.e., LLaMA-70B), with specific templates defined to generate more hallucination/bottleneck-oriented perturbations (e.g., counting objects, color identification etc.)

After training with the dataset, a base LVLM becomes a Token-Level Reward Model for various applications:

(i) hallucination evaluation: where the reward labels could be used to develop token-, sentenc- and response-level accuracy for the generated responses. 

(ii) self-correction with the token-level reward for reducing hallucination;

(iii) enhanced performance as a by-product of the reward model training;

(iv) speed-up the caption annotation

### Strengths
- Overall, I think the idea of developing a token-level reward model for LVLMs is interesting and the authors propose a feasible dataset synthesis method to achieve this.
- The paper is well-written and well-organized. 
- The applications of TLDR seem to be promising for future LVLM developments.

### Weaknesses
My major concerns lie in the experimental settings and evaluation setups, which makes the results less convincing to me.

- Backbone choices: I found the models used in this paper are somewhat arbitrary: the reward model is trained upon PaliGemma; human evaluation is conducted with captions generated by MiniCPM; GPT-4V is used to perform self-correction experiments. Are there any specific reasons to choose different models in these experiments? 

- Insufficient experiments regarding model selection:
   - The reward model is only trained with a PaliGemma-3B model, which as demonstrated in Table 3, performs the worst in the hallucination. Wouldn't a stronger backbone lead to stronger Token-level reward models? An ablation study comparing TLDR models trained on different backbones, including stronger ones like phi and llama-3.2, is recommended.
   - The correlation between MMMU scores is misleading. As pointed by Cambrian-1 and MMMU-Pro, MMMU score does not faithfully reflect the visual understanding and reasoning performance but more about the LLM capability. This renders the correlation analysis less convincing and I recommend the authors incorporate visual-oriented benchmarks to justify this claim.
   - Why LLaVA-series and Qwen-VL are not included in the evaluation? These are commonly adopted LVLMs.  Including these models would provide a more comprehensive comparison across different LVLM architectures and training approaches.


- In Table 5, why are only two tasks (Counting and spatial relation) of BLINK adopted as in-domain tasks? There are also in-domain  tasks such as Visual correspondence & Object localization. Adding the comprehensive results on BLINK could better illustrate the full picture.

### Questions
- Could you explain their rationale for choosing different models for each experiment?



Minor:
There are empty lines after Eq. 6 and Eq. 7.

This paper ignores many relevant papers regarding LVLMs alignments and reward modeling. It would be more beneficial to compare the reward models or DPO training performance of these studies:

- LLaVA-RLHF: Aligning Large Multimodal Models with Factually Augmented RLHF
- RLAIF-V: Aligning MLLMs through Open-Source AI Feedback for Super GPT-4V Trustworthiness
- Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback 
- VLFeedback: A Large-Scale AI Feedback Dataset for Large Vision-Language Models Alignment 
-  Strengthening Multimodal Large Language Model with Bootstrapped Preference Optimization

### Soundness
3

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
This paper presents a token-level reward model for VLMs. The challenge it aims to address is that the commonly used binary reward is often biased towards linger captions. The proposed TLDR model can provide per-token feedback and solve issues such as hallucinations. The method achieves this using synthetic data generation by perturbing correct captions.

### Strengths
- The paper is well motivated
- The method section is clear and easy to follow
- Some very interesting findings, such as that fine-tuning a VLM for token-level rewards improves the model itself.

### Weaknesses
- I would not really call TLDR a reward model, as the paper has not shown that it can actually be used as a reward model (in the RLHF sense). In its current form, this is a per-token correctness model.
- Table 9 shows that in the response-level evaluation, TLDR is only marginally better that a naive response-revel reward model. While this is not possible for the response-level model, it would be interesting to see if TLDR outputs the correct prediction for the right reason, i.e. manually evaluate per-token accuracy, recall and precision, rather than just the global metrics.
- Is Equation (8) backed by anything? That seems very arbitrary.

### Questions
-

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a model that addresses the limitations of existing reward models, which provide only binary feedback for entire text sequences. Instead, it assigns rewards at each token, enabling more precise and interpretable feedback, improving self-correction, hallucination detection, and human annotation efficiency in vision-language tasks.

### Strengths
1. The idea is interesting, as it shifts from traditional binary feedback to a more detailed token-level approach.
2. The model provides token-level rewards, offering more precise and interpretable feedback compared to traditional binary reward models.
3. The proposed HALLUCINATION RATE (%) is a novel evaluation metric.
4. The token-level errors can guide multimodal large language models in self-correction, enhancing their performance.
5. The model can serve as a data correction tool, effectively speeding up the process by three times.

### Weaknesses
The model is trained using a perturbation-based data generation process based on simple factual statements, which introduces some limitations:
1. Its performance may be limited when interpreting information-rich images like posters, where elements for the same noun are not unique and are arranged in a complicated layout.
2. The model may also face challenges in understanding text-rich images, such as documents where relationships between concepts are described in text and require logical reasoning.

### Questions
1. In Section 5.4, despite the model’s low performance on tasks listed in Table 5, does the model with ($\tau$ = 1) still achieve the best performance on token-level hallucination detection?
2. Do you have any insights into why the finetuned LoRA weights initially enhance the model’s performance on other tasks but then cause a decline as ($\tau$) increases?

### Soundness
3

### Presentation
3

### Contribution
3
