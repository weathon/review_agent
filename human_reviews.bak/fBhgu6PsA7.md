# MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning

- Decision: Reject
- Scores: 5, 6, 6, 6, 6, 6

## Abstract
Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. Typically, an LLM is first pre-trained on large corpora and subsequently fine-tuned on task-specific datasets. However, during fine-tuning, LLMs may forget some knowledge acquired in the pre-training stage, leading to a decline in general capabilities. To address this challenge, we propose a new fine-tuning algorithm termed Momentum-Filtered Optimizer (MoFO). 
As an extension of greedy block coordinate descent (BCD) methods, MoFO iteratively selects and updates the model parameters with the largest momentum magnitudes.
MoFO achieves similar fine-tuning performance to the default fine-tuning algorithm while effectively mitigating knowledge forgetting.
Furthermore, MoFO does not require access to pre-training data, making it highly suitable for scenarios where the pre-training data is unavailable, such as fine-tuning checkpoint-only open-source LLMs. 
We validate MoFO through rigorous convergence analysis and extensive experiments, demonstrating its superiority over existing methods in mitigating forgetting.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The motivation of this paper is to address the challenge of forgetting in large language models (LLMs) during fine-tuning. The main contribution is the proposed Momentum-Filtered Optimizer (MoFO), a fine-tuning algorithm based on block coordinate descent that selectively updates high-momentum parameters to prevent forgetting.

### Strengths
(a) The paper presents a convergence analysis of the proposed optimizer.

(b) It includes experiments on a diverse set of LLM models and a range of advanced NLP fine-tuning tasks.

### Weaknesses
(a) Unclear Motivation: In the Introduction (Lines 54-55), the statement “However, regularization-based methods alter the original (fine-tuning) loss function, which can potentially affect the model’s performance on fine-tuning tasks” lacks citations or experimental support, weakening the rationale for excluding regularization-based, replay-free fine-tuning methods as baselines. Providing examples or references of such methods that negatively impact fine-tuning performance would help clarify this claim. Additionally, explaining how modifications to the “original” loss function might specifically affect performance would strengthen the motivation for a regularization-free approach. The approach of keeping fine-tuned model parameters close to pre-trained models to prevent forgetting is already well-established in continual learning [1], limiting the novelty of this finding.

(b) Insufficient Baselines: The exclusion of regularization-based fine-tuning methods as baselines lacks sufficient justification. Adding comparisons with well-known regularization-based approaches, such as proximal gradient and EWC-based methods [1], would provide a more thorough evaluation. Furthermore, the current baselines mainly focus on parameter-efficient fine-tuning techniques, omitting full fine-tuning methods relevant for a comprehensive comparison in the LLM context. Including full fine-tuning baselines, as shown in recent work on LLMs [2], would enhance the fairness and depth of the comparative analysis.

(c) Lack of Clear Novelty and Contribution: The specific contributions and novelty of the proposed method in addressing forgetting in fine-tuning LLMs are difficult to discern, as the forgetting issue is not unique to LLMs and has been explored in other settings. Clarifying how this approach differs from or builds upon existing forgetting mitigation techniques, especially in the LLM context, would make its contributions more apparent.

(d) Minor Issues: The use of non-integer epochs is unclear (Line 103); specifying iterations would be more appropriate. Additionally, please clarify the batch size and learning rate used in each experiment, as these details are essential for reproducibility.

[1] Kirkpatrick, J., et al. “Overcoming catastrophic forgetting in neural networks.” Proceedings of the National Academy of Sciences 114.13 (2017): 3521-3526.

[2] Lv, K., et al. “Full parameter fine-tuning for large language models with limited resources.” arXiv preprint arXiv:2306.09782 (2023).

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
3

### Summary
This paper focuses on the forgetting problem of LLM fine-tuning. It proposes a new selective-updating mechanism to mitigate forgetting in the original data domain.

### Strengths
1. The paper works on an intriguing and important problem, which focuses on mitigating the forgetting problem in LLM fine-tuning.
2. The paper demonstrates its performance on multiple datasets and tasks.

### Weaknesses
1. There is no related work section in the main paper, which makes it difficult for readers to quickly follow up on the relevant topic. I suggest authors add this part in their revised version (not in Appendix).
2. The presentation of some sections is not very clear, such as how the authors calculate the distance of different model states in Section 2.2 and how they make the parameter partition in Section 2.3. 
3. I am a little bit confused about some parts of the experiment setting, please refer to **Questions**.

### Questions
1. In Section 2.1 and Figure 2, how do you exactly obtain the distance between the original pre-trained state and fine-tuning model's state? Also, could you explain your experiment setting here, such as did you fine-tune all parameters here? 
2. Based on your findings where **models that remain closer to their pre-trained state tend to preserve more pre-training knowledge**. A naive solution based on your findings is to adopt the PEFT fine-tuning methods, which only train a small part of model parameters, such as only tuning the parameters of the attention layer of an LLM. Why can we not directly use them to mitigate the forgetting problem? 
3. In Section 2.2, can you detail your parameter partition algorithm, especially under the scope of LLMs? 
4. In the experiment setting, is it fair to directly compare LoRA with the fully fine-tuned MoFO? It looks like a more fair comparison is to set them under the same trainable parameters set, like comparing LoRA with MoFO(PEFT version), and comparing Default FT with MoFO(FT version).

### Soundness
3

### Presentation
2

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
The authors focus on mitigating forgetting of pre-training knowledge in LLM fine-tuning. They propose update models with only a subset of parameters based on momentum during fine-tuning. Experiments on public datasets are conducted to demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper follows a clear, well-thought-out structure that facilitates easy reading and comprehension;

2. The experimental results appear encouraging.

### Weaknesses
1. In Section 2.1, the authors show a strong correlation between forgetting and the distance from the pre-trained model with experiments. However, no analysis or proof is provided to explain the reason.

2. Considering that MoFO only uses a subset of parameters during fine-tuning, it may require more time or iterations compared to other methods to reach a minimum loss. It could be helpful to provide a time cost analysis section.

3. Clarification on the selection process for $\alpha$ would enhance the understanding of this parameter's role.

4. Table 5 demonstrates that MoFO outperforms Gradient-filtered BCD and MV-filtered BCD. A deeper analysis of the factors contributing to MoFO's superior performance is needed to provide further insight into these results.

5. In Fig. 4, the differing numbers of points for various methods and the significance of lighter colors could be further clarified for better interpretability.

### Questions
N/A

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
4

### Summary
This paper proposes a new method, named Momentum-Filtered Optimizer (MoFO) for finetuning LLMs while mitigating the forgetting of the knowledge captured by the pre-trained model. The proposed method leverages block coordinate descent (BCD), where the blocks of parameters to update are chosen based on the top-\alpha largest momentum values. The convergence result of MoFO is discussed. Empirical results are also provided to justify the advantages of MoFO.

### Strengths
1. The paper provides a simple but elegant approach to mitigating forgetting in LLM fine-tuning.

2. Empirical results are convincing.

3. The paper is well-written with illustrative figures.

### Weaknesses
1. Using the default partitioning of model parameters as implemented in PyTorch is quite an ad-hoc choice. Is there any better or more principled way to partition the model parameters?

2. The convergence result is good, but it does not show the advantages of MoFO over the traditional Adam method, especially in mitigating forgetting.

### Questions
1. In Table 1, why LoRA perform well on HumanEval but worse on other tasks? Also, in both Table 1 and Figure 4, the performance of MoFO is not significantly better than HFT.

2. The study of the MoFO approach applied on other optimization methods beyond Adam is recommended.

3. I would suggest the authors include  an efficiency analysis for MoFO.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a novel fine-tuning algorithm called the Momentum-Filtered Optimizer (MoFO), designed to address the issue of knowledge forgetting in large language models (LLMs) during fine-tuning. As a replay and regularization free strategy, MoFO iteratively selects and updates model parameters based on the largest momentum magnitudes, extending greedy block coordinate descent (BCD) methods. The algorithm achieves comparable fine-tuning performance to standard methods while effectively mitigating knowledge forgetting, without requiring access to pre-training data. The authors validate MoFO through rigorous convergence analysis and extensive experiments, demonstrating its superiority in reducing forgetting, making it particularly suitable for scenarios involving checkpoint-only open-source LLMs.

### Strengths
1. The paper deals with an important problem in LLM application and gives a practical solution with fair theoretical support.
2. The illustation and analysis of MoFO strategy is clear and in detail.
3. Authors supported their method with experiments covering various tasks and settings.

### Weaknesses
1. The discussion of this paper did not cover other algorithms that are non replay and non regularization, and therefore doesn't give enough support for the motivation and novelty of their work. 
(including but not limited to: https://arxiv.org/pdf/2309.06256, https://arxiv.org/pdf/2404.10306, https://arxiv.org/abs/2302.03241 etc.)
A brief literature review section might be added specifically comparing MoFO to these other approaches, highlighting key differences and potential advantages.
2. The number and quality of baselines compared in experiments are too limited; the experimental validation of orthogonal methods is somehow unconvincing without proper comparisons.

### Questions
1. Add discussions and reference for more works (see list in Weakness 1) on LLM CF mitigation, especially justify your advantage compared to them
2. Add baselines (see list in Weakness 1) and comparison studies (i.e. is the combination of MoFO and replay better than both MoFO only and replay only?) correspondingly in experimental studies. You might also need to better verify (possibly by adding references) why the experimental success of the combination of MoFO with your choice of HFT, GEM and replay would be representative enough to suggest the effectiveness of combining MoFO with othorgonal strategies.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel fine-tuning algorithm named Momentum-Filtered Optimizer (MoFO). The primary contribution of the paper is addressing the issue of catastrophic forgetting in LLMs during fine-tuning. MoFO selectively updates model parameters with the largest momentum magnitudes to maintain proximity to the pre-trained state, thereby mitigating knowledge forgetting. The authors claim that MoFO achieves similar performance to the default fine-tuning algorithm while effectively preserving pre-training knowledge, without requiring access to pre-training data. The paper is rigorous in its convergence analysis and experiments, demonstrating a good trade-off in maintaining general capabilities and downstream task performance.

### Strengths
1. This paper provides a comprehensive and reasonable proof of the convergence.
2. The effectiveness of the method is verified with LLMs of different sizes.
3. The convergence direction of MoFO and baselines is visualized using loss landscape, proving that MoFO converges to a closer point.

### Weaknesses
In Section 4, the loss landscapes for HFT and LoRA were not reported.

### Questions
1. The fraction of MoFO was varied in the experiments. Appendix E.1 only discusses its impact on the Llama-2-7B model. How does this hyperparameter affect models of different sizes?

2. MoFO updates only a small subset of parameters at each iteration, and the learning rate decreases over time. Will this lead to reduced fine-tuning efficiency?

### Soundness
3

### Presentation
3

### Contribution
3
