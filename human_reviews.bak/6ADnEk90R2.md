# CoMMIT: Coordinated Instruction Tuning for Multimodal Large Language Models

- Decision: Reject
- Scores: 3, 5, 6, 5, 5

## Abstract
Instruction tuning in multimodal large language models (MLLMs) generally involves smooth integration of a backbone LLM and a feature encoder that has non-text input modalities. The major challenge is how to efficiently find the synergy through cooperative learning, so that LLMs can adapt their reasoning abilities in downstream tasks while feature encoders can adjust to provide more relevant modality-specific information. In this paper, we analyze the MLLM instruction tuning from both theoretical and empirical perspectives, where we find unbalanced learning between the two modules, i.e., the feature encoder and the LLM, can cause problems of oscillation learning and insufficient training with diminishing learning gradients. Inspired by our findings, we propose a Multimodal Balance Coefficient that enables quantitative measurement of the learning balance. Based on this, we further design a dynamic learning scheduler that better coordinates the learning between the LLM and feature encoder, alleviating the oscillation and insufficient training. In addition, we introduce an auxiliary regularization on the gradient to promote updating with larger step sizes, which potentially enables a more accurate estimation of the learning balance coefficient and further improves the training sufficiency. Our techniques are agnostic to the architecture of LLM and feature encoder, so can be generically integrated with various MLLM. Experiment results on multiple downstream tasks and modalities in vision and audio, demonstrate the proposed method’s better efficiency and effectiveness in MLLM instruction tuning.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This work analyzes instruction tuning in multimodal large language models (MLLMs) from both theoretical and empirical perspectives, and finds unbalanced learning between the feature encoder and the LLM can cause problems of oscillation learning and insufficient training with diminishing learning gradients. To alleviate this, they propose a multimodal balance coefficient to measure the learning balance, and introduce an auxiliary regularisation on the gradient. Experiments on four multimodal LLMs show the proposed method outperforms the baselines.

### Strengths
1. This paper analyzes instruction tuning in multimodal LLMs and finds unbalanced learning between the feature encoder and the LLM can cause problems of oscillation learning and insufficient training with diminishing learning gradients.
2. They propose a multimodal balance coefficient as well as a dynamic learning scheduler to alleviate oscillation learning and insufficient training.
3. Empirical results on multiple downstream tasks in vision and audio modalities show the proposed method CoMMIT outperforms the baselines.

### Weaknesses
1. The contribution looks limited. The proposed method seems to be hard to follow, since it is customized for multimodal instruction tuning.
2. The experiments are not solid enough to confirm the effectiveness of CoMMIT. This paper might consider more recent MLLMs with different architectures.
3. The presentation can be improved. For example, baselines such as Constant LR, Feature CD and Language CD should be briefly explained, to avoid any confusion.

### Questions
- Although this paper is about multimodal instruction tuning, I am curious about whether the findings and proposed method can be generalised to other post-training schemas such supervised fine-tuning (SFT) and direct preference optimization (DPO). If so, do authors have any experimental results under SFT and DPO settings?
- Why don't select some different LLM backbones such as Cambrian-1, MiniCPM-V-2.6 and Qwen2-VL?

### Soundness
2

### Presentation
2

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
The paper addresses unbalanced learning between the LLM and feature encoder in multimodal instruction tuning, leading to issues like oscillation and insufficient training. It proposes a Multimodal Balance Coefficient and a dynamic learning scheduler to coordinate learning, alongside an auxiliary regularization to improve training efficiency. The proposed techniques are architecture-agnostic and show improved performance across multiple tasks and models.

### Strengths
**Combination of Theory and Empirical Evidence**: The proposed theoretical framework is combined with empirical observations, revealing potential issues of learning imbalance and providing deep insights.

**Dynamic Coordination of Learning**: CoMMIT dynamically adjusts the learning rates of the feature encoder and LLM to effectively balance multimodal learning progress, avoiding oscillations and insufficient training.

**Broad Applicability**: The proposed method can be applied to different optimizers and various LLMs, demonstrating strong general applicability.

### Weaknesses
**Limited Generalizability**: It is unclear whether the observed phenomenon is universal, as the authors only used BLIP-2 model and TextVQ dataset in their empirical studies, raising concerns about generalizability.

**Lack of Novel Model Architecture**: The paper primarily proposes a parameter tuning method. A new model architecture would have been more impactful, rather than just dynamically adjusting learning rates.

### Questions
- Using more models and more data in the empirical analysis would make the findings more convincing.

- The authors used three VQA datasets for testing in Table 2. To my knowledge, multimodal large models have many downstream tasks, more evaluation datasets should be included.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- The paper introduces CoMMIT, a novel method for multimodal instruction tuning that dynamically coordinates the learning rates of the multimodal components and employs an auxiliary loss for gradient regularization. 
- It establishes a theoretical framework to identify and analyze learning imbalances in multimodal large language model (MLLM) instruction tuning and provides a convergence rate analysis based on this framework.
- Experiments on multiple downstream tasks and modalities demonstrate that CoMMIT improves both convergence rate and effectiveness in MLLM instruction tuning.

### Strengths
- The paper identifies the phenomenon of unbalanced learning between the feature encoder and the LLM in the MLLM instruction tuning, which can cause diminishing learning gradients and often lead to sub-optimal results.
    - It introduces a quantitative measure for evaluating learning balance and proposes a coordinated learning rate scheduler with auxiliary loss regularization, effectively coordinating the learning of multimodal components.

### Weaknesses
- MLLMs typically comprise a feature encoder, an LLM, and a multimodal projector (e.g., the q-former in BLIP2), the paper does not discuss the role of the multimodal projector in the proposed method. It is unclear if the projector is considered part of the feature encoder, and if so, the rationale behind this choice is not explained.
- Although the paper demonstrates faster convergence of the proposed method, it lacks empirical comparisons in terms of training time efficiency.
- The evaluation of CoMMIT focuses solely on fine-tuning performance. It is unclear if the proposed method could also lead to better performance in the zero-shot setting. 
- It would be better if the paper could also include a comparison of normalized learning gradients (as in Figure 3) for the proposed CoMMIT.

### Questions
- In Section 5.2, why does using a large learning rate for the feature encoder result in gradient diminishing in the feature encoder S , as shown in Figure 3a?
- Have the authors experimented the proposed method with different optimizers? Can the advantage brought by the proposed method be generalized to different optimizers.
- Typos in L271

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper addresses the issue of imbalanced learning between the feature encoder and the LLM in multimodal instruction tuning, which leads to insufficient training and oscillation problems. To alleviate the issue, the authors propose a new training strategy with a dynamic learning scheduler and gradient regularization to balance and enhance learning. Empirical results demonstrate improved convergence and performance across various multimodal tasks with various MLLM.

### Strengths
•	The problem discussed in the paper, i.e., the imbalanced training in MLLMs, is interesting and meaningful for multimodal learning and broader research communities.

•	The paper proposes CoMMIT, a coordinated learning rate scheduler that effectively balances the training of the feature encoder and LLM.

•	Through theoretical analysis, the paper demonstrates that CoMMIT leads to faster convergence and can be generalized across various optimizers.

•	Empirical results across various downstream multi-modal tasks prove that CoMMIT is both effective and adaptable to different MLLM architectures.

### Weaknesses
•	The conclusions of this paper may not be generalizable due to its limited experiment setup. It seems the authors only investigated the setting of finetuning with LoRAs. But LoRAs finetuning can be very different full finetuning. So the generalizability of the approach and findings under this setup is questionable.

•	The paper does not have a clear definition of “learning insufficiency”. In Hypothesis 4.2, the authors mention “imbalanced learning can cause insufficient learning problem”, but do not establish clear criteria or metrics that differentiate sufficient from insufficient learning. Providing a more rigorous definition (e.g., quantifiable definition or threshold) for “learning insufficiency” could strengthen the theoretical and empirical claims.

•	The empirical experiments do not directly demonstrate that CoMMIT resolves the oscillation and insufficient learning issues. While the learning curves and instruction tuning results on MLLMs show overall improvements, they lack in-depth analysis that proves the specific problems are addressed.

•	The experiment setup is not clearly illustrated and lacks many important details. For example, in Section 8, the authors did not clearly state what instruction tuning datasets they are using, what’s the size of the dataset. They also didn’t provide the setup for InternVL2 and LLaVA-1.5.

### Questions
•	How does the variation of Multimodal Balance Coefficient κ during training correlate with model performance and training stability? It would be helpful if you could add detailed quantitive analysis or case studies to show κ’s impact.

•	Although the paper discusses gradient regularization to prevent diminishing gradients, can you provide a more intuitive and in-depth analysis on how the regularization can affect gradients behaviors? For example, more detailed gradient visualization such as gradient norms would be helpful to demonstrate the effectiveness.

•	There are some typos in the draft. For instance, in lines 24-25, “which potentially prevents enables a more …” seems to be a grammar error.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper focus on the balance of learning between vision encoder and llm in the context of visual instruction tuning.  The imbalanced learning is caused by two problems: (1) insufficient learning and (2) oscillation of gradient. To address these two problem, this paper proposes CoMMIT consisting a coordinated learning rate scheduler and regularization in gradient descent. 

The paper (1) defines the Multimodal balance coefficient (k) as the ration between two KL divergence and proved that k accounts for the upper bound of the gradient for llm and vision encoder.
(2) proposes regularization to avoid gradient diminishing problem.

The results show that proposed CoMMIT and CoMMIT-CLR can accelerate the convergence of the losses on two modalities (image and audio) and help the models to achieve lower losses.

### Strengths
1. this paper points out an important and interesting problem in multimodal training, i.e., the training balance between the vision encoder and the LLM.

2. This paper provide both empirical results and theoretical analysis to support their proposed Multimodal balance coefficient.

### Weaknesses
1. writing needs to be significantly improved. many typos and grammar errors, making the paper hard to follow:
L24-25 prevents enables
L200  observe the dyanmics of κt is different
L203 bewteen
L212 - 213
L 283 show case,  problems that signifies

2.  observations in 5.1 and 5.2 need further explanations. (see Questions 4)

3. What are the benefits of the proposed regularization in terms of empirical results? such as convergence speed of the losses or model's performance? If not, it's hard to justify the usefulness of this method.

4. Missing analysis and discussion:
(1) How often do you need to compute k_t in order to get an accurate estimation? can you discuss the optimal updating interval of k_t?
(2) what is the latency caused by computing k_t?

### Questions
1. in equation (6), what do you mean by logits?

2. in appendix A.1, line 716, what is T_t?

3. line 721 "prediction distribution" of what?

4. why increasing the lr of encoder causes the K_t going to 1 in Figure 2 (c)? shouldn't it go to zero?

5. what is unsupervised instruction tuning on L 311. maybe provide a reference?

6. what is \tilde{X_{\theta}^x_t} in equation (9)?

7. Line 322-323, what is the relationship between N and K?

8. equation (25), what does F(x_k)^2_2 mean?

### Soundness
2

### Presentation
1

### Contribution
3
