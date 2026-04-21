# Have the VLMs Lost Confidence? A Study of Sycophancy in VLMs

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
In the study of LLMs, sycophancy represents a prevalent hallucination that poses significant challenges to these models. Specifically, LLMs often fail to adhere to original correct responses, instead blindly agreeing with users' opinions, even when those opinions are incorrect or malicious. However, research on sycophancy in visual language models (VLMs) has been scarce. In this work, we extend the exploration of sycophancy from LLMs to VLMs, introducing the MM-SY benchmark to evaluate this phenomenon. We present evaluation results from multiple representative models, addressing the gap in sycophancy research for VLMs. To mitigate sycophancy, we propose a synthetic dataset for training and employ methods based on prompts, supervised fine-tuning, and DPO. Our experiments demonstrate that these methods effectively alleviate sycophancy in VLMs. Additionally, we probe VLMs to assess the semantic impact of sycophancy and analyze the attention distribution of visual tokens. Our findings indicate that the ability to prevent sycophancy is predominantly observed in higher layers of the model. The lack of attention to image knowledge in these higher layers may contribute to sycophancy, and enhancing image attention at high layers proves beneficial in mitigating this issue.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the study of sycophancy, a prevalent hallucination issue in Vision-Language Models (VLM). Firstly, a benchmark named MM-SY is introduced to evaluate the severity of sycophancy in VLMs. Subsequently, three methods—prompt guidance, Supervised Fine-Tuning (SFT), and Direct Policy Optimization (DPO)—are explored to mitigate sycophancy. Finally, the author analyzes the impact of attention weights on the sycophancy problem through experiments and proposes a simple, training-free method to alleviate this issue.

### Strengths
1. This paper is well written, clearly articulating the progressively detailed research approach to the sycophancy issue in VLMs.

2. Experiments are comprehensive, thoroughly testing multiple VLM models, various tasks, and different user preferences, and analyzing the relationship between sycophancy and various dimensions.

3. By studying the attention weights at different layers, this work reveals the model's performance in mitigating the sycophancy problem.

### Weaknesses
1. This paper mentions the contradiction between sycophancy and stubbornness issues, so for the VLM model, the real problem that needs to be addressed is to reduce sycophancy while maintaining the acceptance of correct opinions. However, methods such as prompt guidance, DPO, and amplify attention seem to reduce sycophancy but at the same time increase stubbornness to an equal extent. This does not truly solve the problem. It is merely shifting the imbalance from one side of the seesaw to the other. Only the SFT method shows a lower increase in stubbornness compared to the alleviation of flattery, but the paper does not provide a thorough analysis of this point.

2. This paper identifies the impact of amplifying high layer attention on the sycophancy problem but does not propose effective solutions based on this finding to truly address both sycophancy and stubbornness issues.

### Questions
As in Weaknesses, why does SFT perform better than other methods? Does high layer attention help in truly addressing the issues of flattery and stubbornness?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper "Have the Vision-Language Models Lost Confidence? A Study of Sycophancy in VLMs" introduces the concept of sycophancy in vision-language models (VLMs), where models blindly agree with user inputs despite contradictory visual evidence. The authors present the MM-SY benchmark, the first evaluation benchmark for sycophancy in VLMs across ten visual understanding tasks. They find that VLMs exhibit significant sycophantic behavior, influenced by factors like task type, user tone, and model size. To address this, the paper explores three mitigation methods: prompt-based, supervised fine-tuning, and direct preference optimization, showing progressive improvements in reducing sycophancy. However, these methods also make VLMs more resistant to corrections. The authors propose a training-free approach by amplifying high-layer vision attention, which effectively mitigates sycophancy without compromising the model's receptiveness to corrections.

### Strengths
1. The paper offers a thorough analysis of the factors influencing sycophancy in VLMs, providing valuable insights into model behavior across different conditions.

2. The exploration of three distinct mitigation methods, each with varying degrees of success, contributes to the understanding of how to manage sycophantic behavior in VLMs.

3. The proposal of a simple, training-free method to reduce sycophancy by amplifying high-layer vision attention is innovative and has practical implications for model development.

### Weaknesses
1. The mitigation methods were only validated on a single VLM (LLaVA-1.5-7B), which limits the generalizability of the findings. It's unclear how these methods would perform across different VLM architectures.

2. The paper mentions that due to time and computational resource constraints, the analysis was limited. This suggests that the findings may not be exhaustive and could benefit from further exploration with additional resources.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the sycophancy problem in VLMs, which is also a common hallucination issue in LLMs. The authors first design an evaluation benchmark along with 10 visual question answering (VQA) tasks to assess the sycophancy problem in popular VLMs. They then propose three methods from the perspective of prompt engineering, supervised fine-tuning, and direct preference optimization to mitigate this issue.

### Strengths
+ This is the first paper to investigate the hallucination problem in multi-modality language models. To address this issue, the authors construct a new evaluation benchmark that includes 10 different visual question answering (VQA) tasks.

+ Based on the designed benchmark, the authors investigate this problem on various popular VLMs and provide comprehensive experimental results.

+ Besides revealing the sycophancy phenomenon, the authors also provide three different kinds of solution to alleviate this hallucination problem.

### Weaknesses
+ It seems that the definition of sycophancy rate is missing. Could the authors present it in Section 2? This is important for the readers to understand Table 1 and Figure 2.

+ In addition to revealing the sycophancy phenomenon, it would be beneficial to analyze why the current model tends to exhibit sycophancy. For example, is this comes from the training data or the network architecture?"

### Questions
+ From Table 1, it seems the sycophancy rate is not correlated to the designed types of tones. Could the authors provide the analysis for this?
+ In addition to the sycophancy rate, it would be beneficial to also present the baseline accuracy. It is interesting to see if the model's sycophancy rate related to its performance?

### Soundness
4

### Presentation
3

### Contribution
3
