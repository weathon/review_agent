# Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Vision–language models (VLMs) extend large language models (LLMs) with vision encoders, enabling text generation conditioned on both images and text. However, this multimodal integration expands the attack surface by exposing the model to image-based jailbreaks crafted to induce harmful responses. Existing gradient-based jailbreak methods transfer poorly, as adversarial patterns overfit to a single white-box surrogate and fail to generalise to black-box models. In this work, we propose **U**niversa**l** and **tra**nsferable jail**break** (**UltraBreak**), a framework that constrains adversarial patterns through transformations and regularisation in the vision space, while relaxing textual targets through semantic-based objectives. By defining its loss in the textual embedding space of the target LLM, UltraBreak discovers universal adversarial patterns that generalise across diverse jailbreak objectives. This combination of vision-level regularisation and semantically guided textual supervision mitigates surrogate overfitting and enables strong transferability across both models and attack targets. Extensive experiments show that UltraBreak consistently outperforms prior jailbreak methods. Further analysis reveals why earlier approaches fail to transfer, highlighting that smoothing the loss landscape via semantic objectives is crucial for enabling universal and transferable jailbreaks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces UltraBreak, an optimization-based attack framework for universal, transferable jailbreaks for VLMs. Unlike previous attacks that overfit models, UltraBreak achieves cross-model transferability and cross-target universality with one adversarial image. Experiments show UltraBreak outperforms prior methods in transferability and universality.

### Strengths
+ The experiments are comprehensive, including open-source and closed-source victim models. Quantitative results show UltraBreak has higher ASR than baselines on AdvBench and SafeBench. Ablation results assess each component's contribution.
+ The paper is clear, well-structured, with logical flow from motivation to analysis. Mathematical formulations are precise.

### Weaknesses
1.	The selected baselines are relatively outdated. For instance, typography-based methods, apart from FigStep, should also consider more recent benchmarks such as MM-SafetyBench [1]. Furthermore, when it comes to optimization methods, several advanced approaches like AdvDiffVLM [2], SSA-CWA [3], AnyAttack [4], and M-Attack  [5] have not been taken into account. Incorporating these would provide a more comprehensive evaluation and enhance the robustness of the study.
2.	The paper reveals that UltraBreak is the first to achieve effective universality and transferability against VLMs using a single surrogate model. Of the five open-source models tested, three are from the Qwen family (Qwen-VL-Chat, Qwen2-VL, Qwen2.5-VL). Though the authors say they come from distinct training pipelines, their shared architecture might lead to common vulnerabilities, possibly inflating transferability measurements.
3.	The paper states that when evaluating commercial models like GPT-4.1-nano and Gemini-2.5-flash-lite, the most harmful targets were excluded. This indicates that test conditions varied and were less challenging than those for open-source models. The 32.26% average ASR may not accurately reflect the attack's effectiveness against advanced safety measures.
4.	The paper highlights high ASR, like 71.83% on SafeBench, but lacks analysis of the ~28% failures. Understanding why some attacks fail or succeed, such as the comparison between VAJM and UltraBreak on GLM-4.1V, is essential. The brief mention of dataset toxicity as a reason is speculative. A detailed failure analysis is vital for assessing threats and guiding defenses.
5.	The paper states the benefit of a smoother loss landscape. However, excessive smoothness can lead to a shallow minimum, which might not be optimal for attack purposes. As the temperature parameter τ increases, the landscape becomes smoother, but the model's focus may shift to irrelevant outputs. Thus, a moderate level of smoothness is necessary, and the idea that the smoother the better is flawed, challenging the method's theoretical basis.

[1] MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models
[2] Efficient generation of targeted and transferable adversarial examples for vision-language models via diffusion models  
[3] How robust is Google’s Bard to adversarial image attacks?  
[4] AnyAttack: Towards large-scale self-supervised generation of targeted adversarial examples for vision-language models  
[5] M-Attack: A Simple Baseline Achieving Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4

### Questions
1. Given that recent methods like M-Attack report success rates over 90%, how does UltraBreak compare to these more advanced and effective baselines?
2. Considering that much of your open-source evaluation involves the Qwen model family, could the high transferability partly be inflated by similarities in architecture?
3. Since the evaluation on commercial models excluded the most harmful targets, how can we fairly assess the threat?
4. How do you reconcile the main claim that a smoother loss landscape is better with your own finding that excessive smoothness hampers the attack's success?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a new VLM jailbreaking algorithm that tries to develop universal and transferable adversarial attacks on VLMs.

### Strengths
- Strengths
    - I think the paper is targetting and important problem—the development of universal and transferable jailbreaks on image models. Previous work showed this was challenging.
    - The method, as explained in the introduction, broadly makes sense to me and is intuitive. I like the shift from a log-likelihood to semantic based loss.
    - The paper is clear and well written.
    - The ablation studies show the components make sense and help performance.
    - I liked the extra analysis in Fig 2, and it made sense.

### Weaknesses
- Weaknesses
    - I think the exposition could be tightened up in places (see questions below).
    - Ideally I'd love to see a big stronger jailbreak evaluation using something like StrongReject.
    - Ideally I'd love to see some additional target models, like Claude Sonnet 4.5, GPT-5.
    - "UltraBreak consistently outperforms all gradient-based baselines across target models and both test sets. One exception is ..." Please tone down the writing e.g., with "tends to outperform"
    - It would be nice to add a baseline like Best-of-N jailbreaking to see how well you can do without needing something transferable or universal. You could look at a "universal and transferable ASR gap"

### Questions
- Questions
    - Minor: "Extensive experiments on benchmark datasets demonstrate that UltraBreak surpasses prior gradient-based methods by over 50% on black-box models and unseen targets, establishing strong universality across targets and transferability across models." I don't know what the 50% number is or refers to—can you make this more precise?
    - As I understand, it seems like the approach used is a __token-level__ semantic loss. I can see how this would improve the loss function (many tokens might have similar meanings), but I think a better thing would be a sentence level embedding loss. I think you're trying to approximate this using the cosine similarity to future embeddings and weighted loss, but is that right?
    - Does the loss function work autoregressively and require model sampling, or do you simply plug in the target completion?

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
3

### Summary
This paper proposes specific optimizations for two important properties in VLM attacks—universality and transferability. The authors introduce an adversary attention semantic loss and a total variation loss. The combination of these two modules effectively mitigates the shortcomings observed in previous works, namely the issues of surrogate overfitting and weak transferability.

Overall, the paper presents effective results, and the experimental findings are well-explained. The ablation studies are also thorough and provide clear insights into the contribution of each component.

However, since my reading in the area of VLM safety is limited, I am not fully confident in assessing how novel this work is compared to prior research.

### Strengths
1. The paper is logically well-organized, and the motivation is clear.

2. The proposed method achieves strong performance, though the degree of novelty is uncertain.

### Weaknesses
1. Figure 1 is hard to follow. I suggest adding a more explicit flow in the caption, or introducing a small algorithm box to walk through the pipeline step by step.

2. There are minor typos—for example, line 199 uses w*t,j; I believe this should be $w_{t,j}$, right?

3. It’s unclear how the method deals with the potentially spiky loss landscape induced by the total variation loss.

### Questions
1. Is the plain average an appropriate aggregate? A simple mean can be distorted by one relatively poor result among several strong ones, which could unfairly lower the overall ranking. Have you considered alternatives such as rank-based aggregation before averaging?

2. What is the contribution of Targeted Prompt Guidance to overall performance? Please provide analysis/ablations to quantify its importance and interactions with the two loss modules.

3. Since the method adds two loss terms, what is the computational overhead compared to baseline?

### Soundness
3

### Presentation
3

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
The paper introduces UltraBreak, a method to train universal and transferable image jailbreaks against VLMs (universal meaning they work against different prompts, and transferable meaning they work against different models not used during attack training).

The method uses gradient optimization to train the attack image. They introduce a number of components to the attack training algorithm to promote transferability:
1. Semantic adversarial target. Instead of calculating the loss as cross entropy to a specific harmful completion, they use a cosine similarity loss in a custom output embedding space.
2. Input space constraints (random transformation of attack image in batch, a projection, and a total variation regularization loss).


While prior works have failed to find image jailbreaks that transfer, UltraBreak has impressive results. Table 1 shows that the attacks transfer very well between models. Table 2 contains useful ablations.

### Strengths
## Originality

This is the first paper, that I know of, to present a method that can produce image jailbreaks that transfer between models. The idea of constraining the input space is not novel, but the semantic loss (and implementation) is new to me.

## Quality

The quality of experiments is good. The leave one out ablations also give insight into which components of the algorithm are important. I was initially skeptical about the need for attention in the semantic loss, but the results in Figure 3 were convincing.

## Clarity 

Overall the paper is well written and easy to follow.

## Significance

There have been many works that have shown VLMs are vulnerable to image jailbreaks. With this being said, almost all assume a white-box threat model. Finding attacks that transfer to other models is an important step that shows image jailbreaks to VLMs are a real concern that require specific mitigations when deploying VLMs, even in a black box manner. In this sense, the findings of the paper have reasonable significance.

### Weaknesses
I think there should be more focus on transfer to frontier models. The bottom section of table 1 has some good results in this area. The paper would be improved by adding results with more current frontier models. 

I think some of the language in the introduction is too strong. For example you state "We present UltraBreak, the first jailbreak framework to achieve effective cross-target universality and cross-model transferability against VLMs." This could be interpreted as meaning no prior work has achieved cross-target transfer, but this is false, for example [1] and [2] achieve this.

In addition it is worth noting [1] is related but not cited, in particular they also use input space constraints and seem to find similar high level features (Figure 6 bottom).

Nit: Line 160 typo "Figure 1 summaries our approach" 

[1] Bailey, Luke, et al. "Image hijacks: Adversarial images can control generative models at runtime." _arXiv preprint arXiv:2309.00236_ (2023).

[2] Qi, Xiangyu, et al. "Visual adversarial examples jailbreak aligned large language models." _Proceedings of the AAAI conference on artificial intelligence_. Vol. 38. No. 19. 2024.

### Questions
1. Can you provide any results on more frontier models, e.g. GPT-5 and Claude 4?
2. Although the ablation is convincing, can you provide more intuition as to why the attention mechanism is needed in the semantic loss?
3. What embedding matrix do you use in equation (3)?

### Soundness
3

### Presentation
3

### Contribution
3
