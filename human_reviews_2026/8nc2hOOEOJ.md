# Jailbreak Connectivity: Towards Diverse, Transferable, and Universal MLLM Jailbreak

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
While multimodal large language models (MLLMs) have shown immense potential, their susceptibility to security threats, particularly through the visual modality, poses serious concerns for real-world deployment. Existing jailbreak studies, which successfully induce harmful responses, suffer from three key limitations: a lack of diversity, poor transferability across different models, and ineffectiveness against multiple targets simultaneously. To address these challenges, we introduce the Jailbreak Connectivity (JC) framework. JC framework includes three novel components. First, it generates a diverse range of jailbreak attacks by constructing a continuous path in the image space that connects two jailbreak images. Second, it improves transferability by integrating two types of surrogate classifiers, Safety Classifiers and Jailbreak Success Predictors, to guide the optimization process. Third, JC enables universal jailbreak attacks by modifying the attack objective to elicit any harmful content rather than being tied to a specific harmful question, thereby inducing the target MLLM to answer a broad range of harmful queries. Our experiments on the SafetyBench dataset show that JC achieves an average attack success rate (ASR) of \emph{79.62\%}, representing a substantial \emph{36.24\% increase} over the best-performing state-of-the-art method. In addition, JC obtains the lowest perplexity in 12 out of 13 scenarios, indicating that the generated harmful responses are more fluent and natural. This work offers a promising approach for generating diverse, transferable, and universal jailbreak attacks, highlighting critical security vulnerabilities in current MLLMs. \textcolor{red}{\emph{Warning: This paper contains data, prompts, and model outputs that are offensive in nature.}}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a way to generate universal, transferable adversarial images for jailbreaking.  To be detailed, it formulates the generation of diverse jailbreak images as an optimization problem of a quadratic Bezier curve, and utilizes surrogate classifiers for transferability, resulting in a curated optimization target. Experiments demonstrate the performance of such a method.

### Strengths
- Clear writing. The paper has a clear structure of storytelling, making it easy to follow the three aspects.
- The problem is well-defined and valuable.
- The design of the optimization target is interesting.

### Weaknesses
- Lack of analysis. The design of the loss function requires further explanation of the intuition behind it.
- No ablation studies. To make the method convincing, at least there should be ablations on each loss part to demonstrate the effectiveness. Besides, more baselines should be added for comparisons.
- The data display is sub-optimal. For example, in Table 1, maybe the most attention is paid only to the last row of the average numbers, and Table 2 is not expressive, either.
- Some tiny typos. For example, the dataset should be MM-SafetyBench, and maybe the Llava-2-13B refers to llava-1.5 or llava-Next.

### Questions
- I am not fully persuaded by the introduction of the quadratic Bezier curve. For me, the most intuitive way should be a linear combination. Why do we introduce the square of $u$ instead of simply using $ux_1+(1-u)x_2$? Are there some related comparisons?
- What are the settings of baseline methods? Do you pick some specific images as the adversarial visual input to test the transferability across scenarios on the whole dataset, or optimize adversarial images for every sample? 
- Could you provide more explanations on the training of the jailbreak prediction classifier? If the labels for safety classifier training are determined by the model responses, and the labels for jailbreak success predictor are also derived from actual attack outcomes (the model responses), what is the difference between these two classifiers?
- Could you provide a few examples of successful jailbreaks on Gemini or GPT-4o? It will help if a few detailed jailbroken responses are included in the appendix.
-  What is the experiment setup in black-box attacks? I noticed that the target is set to GPT-4o, and in Table 10, the target is set to be Gemini. If the model has a black-box access, what is the difference between different targets? 

Anyway, please correct me freely if I get anything misunderstanding in the weakness as well as the questions part.

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
4

### Summary
The paper introduces the Jailbreak Connectivity (JC) framework, which consists of three key components. First, it constructs a continuous path in the image space to generate a diverse range of jailbreak attacks connecting different jailbreak images. Second, it enhances transferability by using Safety Classifiers and Jailbreak Success Predictors as surrogate models to guide optimization. Third, it enables universal jailbreak attacks by redefining the attack objective to provoke harmful outputs of any kind, allowing the target multimodal LLM to produce unsafe responses across a wide variety of harmful queries. Experimental results show the effectiveness of the proposed method.

### Strengths
1. The paper provides good technical details and the writing is generally easy to follow.
2. The proposed method is very effective compared with previous methods.

### Weaknesses
I can’t say this paper is of low quality. I just feel the novelty is somewhat limited. The authors try to solve several limitations of previous multimodal jailbreaking methods through three separate aspects. These aspects are not strongly related, so it makes the paper look like something with several pieces combined but not fully connected. And the proposed solution for each aspect is somewhat naive. I suggest the authors add a discussion why you consider these limitations together, and whether they have synergies. This is very important. Otherwise I personally think it is not a well written paper.

### Questions
1. For the jailbreak success predictor and safety classifier. Since you train on the distribution of seen data. What if we need to deal with out of distribution images and wish to use them to jailbreak?
2. What is the efficiency of JC? Could you do a comparison with baselines? Since you are dealing with several problems at the same time, do you think your performance gain comes from more computation rather than approach design?

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
This paper introduces the "Jailbreak Connectivity" (JC) framework, a novel method for generating jailbreak attacks against Multimodal Large Language Models (MLLMs) that specifically aims to overcome three key limitations of existing work: lack of diversity, poor transferability, and ineffectiveness against multiple targets (i.e., lack of universality). The core idea is to find a continuous path, modeled as a quadratic Bezier curve, between two distinct jailbreak images. By optimizing this path, the authors can (1) sample a diverse set of effective jailbreak images, (2) improve transferability by incorporating lightweight surrogate classifiers (a Safety Classifier and a Jailbreak Success Predictor) into the optimization loss, and (3) create universal attacks by modifying the objective to target a general distribution of harmful outputs rather than a specific question-answer pair. The authors conduct extensive experiments on both open-source (LLaVA, MiniGPT-4) and closed-source (GPT-4o, Gemini) models, demonstrating state-of-the-art results. Notably, their method achieves an average Attack Success Rate (ASR) of 79.62% on SafetyBench, a 36.24% improvement over the best baseline, while also producing more fluent (lower perplexity) and more toxic responses.

### Strengths
The paper is well-motivated, clearly articulating the limitations of existing jailbreak methods (lack of diversity, poor transferability, and limited universality). The application of mode connectivity from loss landscape analysis to jailbreak attack generation is novel and interesting. Additionally, the use of lightweight surrogate classifiers to model target MLLM behavior is a bold and creative approach that, if properly justified, could have practical implications for scalable transfer attacks.

### Weaknesses
This paper has two major concerns that significantly impact its contributions:

* Insufficient Evaluation of Diversity Claims

The paper claims to address the diversity limitation of jailbreak images by introducing mode connectivity. However, the evaluation does not adequately measure or validate the actual diversity of generated images, which undermines one of the paper's core motivations.
From my understanding, the connectivity-based method primarily enables exploration of different perturbation choices within the same perturbation budget (ε = 32/255). I question whether this constitutes meaningful "diversity", as all generated images are bounded perturbations of the same original image. The paper needs to:

1. Provide quantitative metrics for image diversity (e.g., perceptual distance, feature-space divergence)
2. Clarify what "diverse" means in this context beyond sampling different points on a Bezier curve

As presented, the diversity contribution appears overtstated.

* Lack of Theoretical Foundation for Surrogate Classifier Design

The use of lightweight classifiers (CLIP-ViT-Base-Patch32) to predict target MLLM jailbreak success lacks justification. While Section 4.2.2 demonstrates empirical effectiveness, the paper provides no explanation or insights into why this approach works, which raises serious soundness concerns. I hold this blief because there is an apparent paradox: if a lightweight model can accurately predict whether a target MLLM will be jailbroken (and can even provide gradients precise enough to guide attack optimization), this would naturally constitute a very strong defense mechanism. The trained classifiers appear to model the target MLLM's vulnerabilities so precisely that they go far beyond simple "prediction." Why doesn't the target MLLM simply use this same classifier for defense?

In addtion, why is the Safety Classifier needed? Since the goal is to jailbreak MLLMs, it seems the Jailbreak Success Predictor alone would suffice. An ablation study comparing performance with only the success predictor versus both classifiers would clarify their individual contributions.

* Concern about Evaluation Metric Acurrency

The Detoxify classifier (2020) used for toxicity evaluation is relatively outdated, particularly when applied to datasets proposed after its release (e.g., AdvBench and SafetyBench). I recommend incorporating an LLM-as-a-judge as an additional (and likely more robust) evaluation metric to validate the toxicity assessments. This would strengthen the evaluation by: (1) providing a more contemporary measure aligned with current safety standards, and (2) offering cross-validation against Detoxify to ensure the toxicity findings are not artifact-specific to one classifier.

### Questions
For rebuttal, please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies jailbreaking MLLMs and addresses three key challenges: lack of diversity, poor transferability across models, and multiple jailbreak targets. The paper introduces the Jailbreak Connectivity (JC) framework, which includes three components: 1) a continuous noise path connecting two jailbreak images, 2) surrogate classifiers to guide optimization for better model transferability, and 3) an attack objective to elicit any harmful content.

### Strengths
The motivation to improve the diversity and transferability of jailbreak methods on MLLMs is clear and practical for MLLM safety.

The proposed framework demonstrates higher attack success rates and better transferability in the experiments.

### Weaknesses
The first major weakness of this submission is the writing quality. The Introduction section is not ready to publish, as it includes lengthy content on unnecessary points yet omits necessary details of certain important points, and does not provide a good-enough overview and introduction to this work. The first paragraph should be compressed with less content introducing the structure of MLLMs. The second paragraph should elaborate a bit more on the three categories of existing jailbreak methods on MLLMs to better align readers with the authors' taxonomy. The third paragraph, when introducing the limitations, should better cite related literature to help validate the listed weaknesses. Besides, there is no highlight summary of the JC framework regarding its attack performance, models and datasets used in this study, transfer performance, etc.

Moreover, the section on the related work of attacks on LLMs can be more tailored to this paper's topic, e.g., how does the transfer ability of attacks on LLMs work? How to ensure the diversity of attacks on LLMs? What are the similarities and differences of the idea of JC compared to methods on LLMs? The current version is a list of common attacks on LLMs without proper links and discussion to this paper's focus. For the attacks on MLLMs, when discussing the disadvantages, please cite related methods and papers that validate these weaknesses.

The second major concern is about the experimental setup. The default target model is MiniGPT-4. It is recommended to conduct the main experiments on more recent MLLMs such as Qwen-3-VL or Kimi-VL to prove that the proposed framework also applies to more recent and more powerful models with better alignments. Besides, the evaluator could be more solid if it incorporates other standard judge models such as Llama-Guard series and StrongREJECT rubrics. Also, to showcase the robustness of the framework, the average and std of the performance should be reported.

The third major weakness is in the analysis and case study part. The designed framework has 3 different modules, which leads to a bunch of interesting ablation studies worth investigating. For instance, a thorough collection and specification of the choice of hyperparameters is needed to justify the related design choices. Moreover, concrete output from each model is desired to prove the effectiveness of the model.

Other minor points: Use "an MLLM" instead of "a MLLM" on Line 138.

### Questions
- What is the advantage of the continuous path method, compared to random optimization with different steps and initializations?
- How many training samples and resources are needed for the surrogate classifiers? More details are preferred to justify the training cost of these surrogate classifiers.
- How does the optimized image help to alleviate the separate guardrails of GPT-4o and Gemini? The explicit harmful questions are normally directly detected and lead to refusal for these commercial models — how does the JC help to bypass such defense?

### Soundness
2

### Presentation
2

### Contribution
2
