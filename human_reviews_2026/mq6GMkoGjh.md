# EchoBench: Benchmarking Sycophancy in Medical Large Vision-Language Models

- Avg Score: 6.00
- Decision: Reject
- Scores: 10, 4, 4

## Abstract
Recent benchmarks for medical Large Vision-Language Models (LVLMs) primarily focus on task-specific performance metrics, such as accuracy in visual question answering. However, focusing exclusively on leaderboard accuracy risks neglecting critical issues related to model reliability and safety in practical diagnostic scenarios. One significant yet underexplored issue is sycophancy — the propensity of models to uncritically align with user-provided information, thereby creating an echo chamber that amplifies rather than mitigates user biases. While previous studies have investigated sycophantic behavior in text-only large language models (LLMs), its manifestation in LVLMs, particularly within high-stakes medical contexts, remains largely unexplored. To address this gap, we introduce EchoBench, which is, to the best of our knowledge, the first benchmark specifically designed to systematically evaluate sycophantic tendencies in medical LVLMs. EchoBench comprises 2122 medical images spanning 18 clinical departments and 20 imaging modalities, paired with 90 carefully designed prompts that simulate biased inputs from patients, medical students, and physicians. In addition to assessing overall sycophancy rates, we conducted fine-grained analyses across bias types, clinical departments, perceptual granularity, and imaging modalities. We evaluated a range of advanced LVLMs, including medical-specific, open-source, and proprietary models. Our results reveal substantial sycophantic tendencies across all evaluated models. The best-performing proprietary model, Claude 3.7 Sonnet, still exhibits a non-trivial sycophancy rate of 45.98%. Even the most recently released GPT-4.1 demonstrates a higher sycophancy rate of 59.15%. Notably, most medical-specific models exhibit extremely high sycophancy rates (above 95%) while achieving only moderate accuracy. Our findings indicate that sycophancy is a widespread and persistent issue in current medical LVLMs, uncovering several key factors that shape model susceptibility to sycophantic behaviors. Detailed analyses of experimental results reveal that building high-quality medical training datasets that span diverse dimensions and enhancing domain knowledge are essential for mitigating these sycophantic tendencies in medical LVLMs.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This article systematically reveals the critical safety issue of "sycophancy" in large models within medical scenarios: even top commercial models still exhibit excessive compliance with user biases in clinical Q&A, further demonstrating that high scores do not equate to truly "understanding medicine." The authors construct EchoBench, covering 24 LVLMs, with fine-grained analysis across bias types, medical departments, modalities, and evidence granularity, while preliminarily exploring the causes of sycophancy and desensitization approaches. This benchmark focuses on "reliability and safety," holding practical significance for suppressing hallucinations and guiding future large medical model/agent training.

### Strengths
1. Reveals a critical flaw in large models during the diagnostic process, discovering that even the most powerful commercial models exhibit excessive sycophancy toward users in the medical field, demonstrating that high benchmark scores do not necessarily indicate true understanding of medical knowledge.

2. Provides EchoBench, covering 24 LVLMs with fine-grained analysis across different biases/departments/granularities/modalities. This is a rare benchmark specifically targeting the "sycophancy" phenomenon in the medical domain of large models, holding significant importance for eliminating model hallucinations.

3. Investigates the causes behind sycophancy phenomena, providing insights for future medical domain large model training. Future medical large models or agents should not focus solely on accuracy but rather need to make logical and solid judgments that are not influenced by any external factors.

### Weaknesses
The research is comprehensive, only some minor weaknesses:

1. Usually setting the temperature to 0 for all models may not be reasonable, as some commercial models perform optimally at temperatures other than 0. This temperature=0 setting might trigger models to be more inclined toward sycophantic responses to users. Please provide a brief explanation or supplementary materials.

2. Please provide the breakdown of sycophancy rates distinguishing between "originally correct but misled" versus "originally incorrect but coincidentally aligned" proportions.

3. It would be advantageous to show some case studies analyzing the patterns of how different models are misled.

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a benchmark for evaluating sycophancy in vision-language models in the context of medical visual question-answering (VQA) tasks. They curate a large collection multiple-choice medical visual questions based on 2k medical images, spanning a large number of clinical departments and modalities, and design 90 different prompts (in collaboration with physicians) that simulate different kinds of biases that reflect real-world users (patients, physicians, medical students). They evaluate a large collection of VLMs spanning proprietary models, general-domain open-source models, and medically specialized open-source models on their benchmark and find that the majority of models show high sycophancy rates (i.e., model generates prediction that matches with the answer choice encouraged by the biased prompt), indicating their susceptibility to various forms of biases and compromising their accuracy.

### Strengths
- The paper touches upon the important problem of VLMs' susceptibility to highly biased and opinionated input prompts in the context of medical VQA, which well-reflects various real-world use cases of these models.
- The benchmark provides extensive coverage over various imaging modalities and clinical departments and provides an evaluation of a very large number of VLMs, potentially contributing to improved generalizability of findings.
- The paper is generally well-written and easy to follow.

### Weaknesses
- At a higher level, this paper seems to consider a domain-specific instance of when (i) the user-provided context (in this case, specifically designed to output the wrong answer) potentially mismatches the parametric knowledge of VLMs and (ii) VLMs are sensitive to the prompting details. While reasonably well-motivated and interesting from an applications standpoint (perhaps this paper would be better-suited for healthcare-focused venues?), the findings that VLMs are sensitive to prompting details and that they can be easily distracted by potentially misleading context is not entirely surprising. On the flip side, I think there are also many cases where we would actually want the model to be sensitive to the user-provided context, as the models are not always going to be accurate.
- The finding that medically specialized models often perform worse than general-domain models (even the base models, like in LLaVA-Med V1.5 vs. LLaVA-V1.5-7B) on medical VQA tasks is not entirely surprising, as some prior works have already suggested this (e.g., [1]).
- Some preliminary strategies for mitigating sycophancy (based on prompt engineering) are explored but are not extensive.
- Limitations of the study are not discussed.

References:
[1] Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress? (Jeong et al., 2024)

### Questions
- In the second finding ("Medical-specific models perform worst in aggregate"), it is stated that "the poor performance of these medical-specific models primarily stems from their failure to follow task instructions, leading to irrelevant and incorrect responses." Meanwhile, in the fourth finding ("Sycophancy rates fluctuate across different departments"), it is stated that the negative correlation between sycophancy and accuracy across different departments "likely arise[s] from the models' varying levels of domain knowledge" and that the lack of domain knowledge in particular areas may lead to higher sycophancy. The two statements seem a bit contradictory, as medically fine-tuned models should supposedly encode more, if not similar, levels of domain knowledge. Can you elaborate on this discrepancy? 
- Related to the point above, I wonder if the inability of follow task instructions may be an artifact of not using the "right" prompt to elicit the desired responses for the medically specialized models, given that it is mentioned as the reason for their failure.
- Regarding mitigation strategies, why not consider a combination of adjustments to the system prompt (to avoid relying on the adversarial cues) and the self-correction ability? These evaluations do not seem to be considered.

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
Unlike other medical benchmark, this paper focus on the sycophancy of VLMs in medical field. Authors introduce EchoBench, which is the first benchmark specifically designed to systematically evaluate sycophantic tendencies in medical LVLMs. EchoBench comprises 2,122 medical images spanning 18 clinical departments and 20 imaging modalities, paired with 90 carefully designed prompts. Experimental results indicate that sycophancy is a widespread and persistent issue in current medical LVLMs.

### Strengths
The sycophancy of VLM in medical is indeed an important topic which has been ignored. Authors construct a benchmark to evaluate this problem. The benchmark comprises 2,122 real-world medical images spanning 18 clinical departments and 20 imaging modalities, paired with 90 carefully constructed prompts that simulate 9 distinct bias types from the perspectives of patients, medical students, and physicians. Based on the experimental results, authors provide several suggestions for futher development in future medical vlms.

### Weaknesses
1. Some current VLMs are not evaluated, such as gemini 2.5 pro and Qwen 2.5 VL.
2. Please discussion the limatation of this benchmark in the main body.

### Questions
1. Authors claim "The poor performance of these medical-specific models primarily stems from their failure to follow task instructions, lead-
ing to irrelevant and incorrect responses.", please provide the visualization evidence.
2. Will authors release the data and evaluate code in the future?
3. Without any training, does we have some approaches to prevent sycophancy?
4. Authors should provide visualization results to further support their conclusions.

### Soundness
3

### Presentation
3

### Contribution
2
