# Decoupled Alignment for Robust Plug-and-Play Adaptation

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 3

## Abstract
We introduce a low-resource safety enhancement method for aligning large language models (LLMs) without the need for supervised fine-tuning (SFT) or reinforcement learning from human feedback (RLHF).  
Our main idea is to exploit knowledge distillation to extract the alignment information from existing well-aligned LLMs and integrate it into unaligned LLMs in a plug-and-play fashion. Methodology, we employ delta debugging to identify the critical components of knowledge necessary for effective distillation. On the harmful question dataset, our method significantly enhances the average defense success rate by approximately 14.41\%, reaching as high as 51.39\%, in 17 unaligned pre-trained LLMs, without compromising performance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes DAPA, an LLM safety enhancement method without SFT/RLHF. This is performed by first identifying the "alignment knowledge" stored in the MLP module of an aligned model, and transfer the knowledge to unaligned models. Empirically, the defense success rate was improved, and the utility was not sacrificed.

### Strengths
The tackled problem, LLM safety and jailbreak, is timely. The proposed delta debugging algorithm for finding alignment knowledge within the MLP module is novel, and the ablation study as well as utility test is comprehensive.

### Weaknesses
The main weaknesses I consider are as follows.
- DAPA requires an aligned LLM. Although claimed as "a low-resource safety enhancement method for aligning LLMs", the preliminary of DAPA algorithm for aligning an unaligned LLM is **the very existence** of an aligned LLM. This makes the stated contribution vague since it cannot serve as a substitution to SFT/RLHF, and thus cannot reduce the resource used for alignment or be an "economic solution to LLM safety enhancement".
- The safety test is weak. Throughout the paper, the defense success rate is measured using AdvBench **directly**. However, given the rapid development of LLM safety (or specifically, jailbreak), there are a suite of jailbreak adversaries proposed and been collected by some benchmark projects (e.g., [1] [2]). Thus, it is insufficient to claim safety improvement without testing the model performance against **any** of these representative adversaries.
- The claim of "the alignment knowledge is stored in the Feed-Forward Network (FFN) layers" lacks supportive evidence. This claim is deduced in the paper via Figure 3, where only **a single harmful prompt** is utilized to compare the influence of the MLP module and the attention module across layers. More quantitative results are needed to support this result.

[1] HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal
[2] JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models

### Questions
My questions to authors are as follows.
- Among the unaligned models, there are several base candidates that have not been tuned to follow instructions. Have you launched additional instruction following on non-harmful datasets in addition to the relative memory substitution described in section 4.1? If not, the result with base models will be hard to compare against instruction-tuned ones.
- When extending Figure 3 to incorporate multiple harmful prompts, is it still true that the MLP's influence is the strongest?
- Can DAPA maintain its benefit when encountering more advanced jailbreak adversaries?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a novel low-resource safety enhancement method for LLM by firstly locating and then editing the specific knowledge, without the need of SFT or RLHF. Experiments on several datasets prove the effectiveness.

### Strengths
1. Clear writing and easy to follow.

2. Comprehensive experimental results demonstrate the effectiveness of the proposed method.

3. In my opinion, it is interesting and somewhat reasonable to enhance the safety capabilities of LLMs by identifying and editing unethical knowledge, with the help of another safer LLM.

### Weaknesses
1. In my opinion, the identification and editing of unethical knowledge within LLMs need to be conducted with great caution. The internal behavior of LLMs may be difficult to measure and analyze accurately, and different knowledge measurement tools may yield varying conclusions about the areas where knowledge is stored. Maybe this could affect the foundational research of this paper, although the conclusions drawn in terms of methodological design and experimentation are somewhat reasonable.  Have you considered using multiple knowledge measurement tools to cross-validate the findings?


2. Modifying knowledge in LLMs raises another issue: it may impact the LLM's ability to respond to general questions. The experimental section of the paper also supports this problem that editing unsafe knowledge can affect its normal response capabilities. Although the decreasing in general capabilities is not significant in experiments, in my opinion, the relationship between the two may not be a trade-off: enhancing safety capabilities does not necessarily require a decrease in general capabilities.

3. The safety knowledge alignment used in this paper depends on a safer teacher LLM from the same family. So, which model will teach these teacher LLMs to be safer? Furthermore, I believe that teaching LLMs to recognize harmful prompts and learn when to refuse to respond may be a more effective way to enhance their safety capabilities.

4. In my view, the ideas and motivations of this paper have some similarities with unlearning, such as in [1-3], both focusing on how to erase specific knowledge. It is necessary to include some discussion on the connection and difference between knowledge unlearning and editing.

[1] Safe Unlearning: A Surprisingly Effective and Generalizable Solution to Defend Against Jailbreak Attacks

[2] Towards Safer Large Language Models through Machine Unlearning

[3] Large Language Model Unlearning

### Questions
1. In the absence of a safer teacher model, how can effective model alignment be achieved? Are there alternative sources of knowledge or alignment strategies?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In this paper, the authors are trying to address a challenging task which is to enhance the safety alignment of large language models without SFT and RLHF. The proposed method(called DAPA), unlike previous methods, only requires limited extra resources by employing a knowledge distillation approach. Two detection/alignment modules, i.e. MLP alignment and Gate alignment, are applied to store the alignment knowledge and check the model's output. The proposed method achieves competitive results with several large base models on the tested datasets.

### Strengths
In general, the paper is well-written and easy to follow.

Unlike previous methods which require costly SFT or RLHF, the proposed method can be an additional module as a "plug-and-play approach" to various large language models with extra low-cost resources required. 

The proposed method is tested on models from different families, which shows the flexibility of DAPA that can potentially adapt to other models.

The use of knowledge distillation for alignment is very interesting.

In this paper, the authors present extensive testing with well-documented evaluation metrics across various models and achieve very competitive results. The ablation studies are also well set.

### Weaknesses
The paper has tested the proposed DAPA mostly on small/mid-sized models (up to 13B models), there is uncertainty around its performance and resource efficiency when scaled to the large LLM models.

The proposed method introduces layer-specific edits without necessarily clarifying the deeper alignment process which may lead to alignment outcomes that are difficult to interpret or predict. This may make it hard to understand how the proposed module changes specific model behaviors.

As the ethical guidelines and alignment standards often evolve, I wonder if the continuous updates or recalibration of the alignment modules will become difficult, especially with large models.

### Questions
Please check the aforementioned issues in "Weaknesses" section. Some additional questions:

I wonder if the proposed method can be easily extended to large multimodal models.

Does the proposed method have potential interpretability?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes an interesting idea of isolating the weights of LLMs that impact the safety alignment the most. It then applies these weights to unaligned models to improve their safety alignment. Various experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper presents an interesting idea of identifying the important weights from the teacher models for safety alignment. By simply replacing student's weights with these weights from the teacher, we can achieve significant DSR improvements with minimum effort.
2. The paper introduces a simple algorithm based on previous works.
3. The experiments conducted are extensive including comparison with lots of models from different model family.

### Weaknesses
1. The paper **doesn't follow the ICLR formatting guideline** by significantly changing the margins. It affects the readability of this paper.
2. The presentation of the paper is poor, e.g. some table and figure titles are too lengthy such as table 2 and 3 and figure 6. Lots of the content should be put into the main text instead of title. I don't know if this is caused by changing the margin or the authors intend to put them in the title.
3. Unnecessary highlight of the figure and table titles. This paper uses lot of bold and highlight which are not necessary.
4. While the authors tried to get more space by changing the margins, they split the tables which should be merged into one into multiple tables such as table 6 and 7, which occupies more spaces. 
5. The contribution is not enough. The main contribution of the paper is a naive search algorithm based on a calibration set. The results are likely highly dataset dependent. There is an experiment in the ablation study to test it on a different dataset, but it's not enough to justify the generability of the proposed method.
6. The evaluation dataset is only 128 prompts. It's very likely the results are overfit to the dataset or calibration set.
7. There is no comparison with the teacher model. E.g in table 1 and 2, after replacing the layers, have we achieved similar or same performances with the teacher model? If not, what's the gap there?
8. Although I understand the motivation of the authors, I don't the palpability of this method from the data provided by the authors. E.g. how is the proposed method compared to just simply fine-tuning using PEFT. Given the number of samples are small, it's possible that fine-tuning with PEFT is more efficient and effective. It's hard to tell from the provided data.
9. The paper claims to that the gate projection has the most significant impact, followed by up projection. But judging from figure 4, it looks like the down projection has a bigger impact than up projection, is this a typo?

Overall, I think this paper presents some interesting and promising findings, but it requires significant more data points to support the generability of the proposed idea.

### Questions
Please see my comments in the weakness.

### Soundness
2

### Presentation
1

### Contribution
2
