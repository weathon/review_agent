# Unleashing the Power of Task-Specific Directions in Parameter Efficient Fine-tuning

- Avg Score: 5.75
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 5

## Abstract
Large language models demonstrate impressive performance on downstream tasks, yet requiring extensive resource consumption when fully fine-tuning all parameters. To mitigate this, Parameter Efficient Fine-Tuning (PEFT) strategies, such as LoRA, have been developed. 
In this paper, we delve into the concept of task-specific directions (TSDs)—critical for transitioning large models from pretrained states to task-specific enhancements in PEFT. We propose a framework to clearly define these directions and explore their properties, and practical utilization challenges. We then introduce a novel approach, LoRA-Dash, which aims to maximize the impact of TSDs during the fine-tuning process, thereby enhancing model performance on targeted tasks. Extensive experiments have conclusively demonstrated the effectiveness of LoRA-Dash, and in-depth analyses further reveal the underlying mechanisms of LoRA-Dash.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To enhance the prevalent parameter efficient fine-tuning (PEFT) strategies (e.g., LoRA) for large language models (LLMs), this paper delves into the concept of task-specific directions (TSDs) which are crucial for the adaptation of pre-trained LLMs to downstream tasks. Based on the proposed mathematical description of TSDs, a simple and effective scheme is designed to approximately identify TSDs from the parameter difference matrix obtained through LoRA. The model parameters derived from LoRA are then aligned with the estimated TSDs to enhance LoRA's performance. The results of extensive experiments confirm the validity of the mathematical description and the effectiveness of the proposed PEFT framework across diverse learning settings.

### Strengths
1. This paper is well organized and easy to follow.
2. In contrast to the conceptual discussion of task-specific directions (TSDs) in existing parameter-efficient fine-tuning methods, this paper offers a deeper understanding by providing a mathematical description of TSDs, which is crucial for enhancing the performance of these parameter-efficient fine-tuning methods.  
3. Te proposed TSDs identification and utilization method are simple and effective.
4. Experimental results across diverse settings validate the significance of task-specific directions in fine-tuning large language models (LLMs), and demonstrate the effectiveness of the proposed LoRA-Dash compared to the prevalent LoRA algorithm.

### Weaknesses
1. As a formal definition, the description of task-specific directions (i.e., “directions whose coordinate values exhibit significantly higher change rates $\delta$ through alteration” in Definition 4) is somewhat vague. The term “significantly higher change rate” would benefit from a more precise definition, such as being specified with reference to a particular threshold.
2. More discussions about how to select the value of hyper-parameter $s$ should be included, as the choice of $s$ directly determines the exact task-specific directions.
3. The statement in Proposition 1 is supported solely by observations from experiments and is not accompanied by rigorous proof.
4. The evaluation results of Fully FT method regarding LLaMA2-7B and LLaMA3-7B models are missed in Table 1.
5. To thoroughly evaluate and compare the effects of task-specific directions and other core directions, it is recommended to include the vanilla LoRA as a baseline method in Figure 5a.
6. The primary contribution of this paper is the proposal of a simple and effective improvement scheme for the LoRA algorithm and its variants. Although this contribution is commendable for acceptance, its impact on the broader research community is still limited, which hinders me from assigning a higher rating.

### Questions
Please refer to weaknesses 1-5 listed in the section above.

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
This paper addresses the resource-intensive issue of fully fine-tuning large language models. It focuses on task-specific directions (TSDs) in parameter-efficient fine-tuning (PEFT). LoRA is examined, and its lack of a clear TSD definition is noted. A framework is then introduced to define TSDs based on the relationship between pre-trained weights and optimal weights for specific tasks. The properties of TSDs are explored, and challenges in using them are identified. Despite the unknowns in practical fine-tuning, it's found that LoRA's ∆W can capture TSD information. The novel LoRA-Dash method is proposed, comprising pre-launch and dash phases. Experiments show that LoRA-Dash outperforms LoRA, is robust to parameter budgets, and excels in various tasks. It also enhances other PEFT methods and provides valuable insights for optimizing model fine-tuning.

### Strengths
- A task-aware approach for identifying critical gradient directions is proposed.

- An effective task-specific fine-tuning scheme Lora-Dash has been introduced.

- Theoretical analysis and extensive experiments have been conducted, providing comprehensive validation of the effectiveness of the proposed algorithm.

### Weaknesses
- More justification is expected. For example, does fine-tuning a general-purpose model for a specific task risk compromising its generalization capability? It is essential to explore strategies that enable a model, even after fine-tuning, to retain the flexibility to handle a range of general tasks. Striking a balance between task-specific adaptation and broad applicability remains a key challenge in fine-tuning.

- The experiments can use more perspectives. For example, the observations in Figure 1 are based solely on commonsense reasoning tasks in the LLama model. Further comparisons with other relevant methods (such as Lora-ga，dora) and other models (such as Qwen, Mistral) should be included for a more comprehensive analysis. 

- In the visualization part, as shown in Figure 4 and Figure 8, a comparison with the results from full fine-tuning should also be provided.

### Questions
Mostly the above comments. Also some minors:

- Is Lora-Dash applicable to multimodal models, such as LLaVA?

- Why does the performance of LoRA-Dash decline when the rank increases beyond a certain point?

- How can we address the issue of excessive memory requirements for SVD in the application of this method?

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
This paper critiques LoRA's prior TSD exploration and provides a precise TSD definition to better understand their role in fine-tuning large language models. The authors introduce LoRA-Dash, which fully exploits TSDs potential, and demonstrate its significant advantages over conventional methods through extensive experimentation. The findings validate LoRA-Dash's effectiveness and highlight the importance of TSDs in parameter-efficient fine-tuning, aiming to inspire further advancements in natural language processing and beyond.

### Strengths
1. The authors observe that TSD can be predicted from delta W in LoRA-based fine-tuning, offering a new perspective on task-specific directions.
2. This paper provides a precise definition of task-specific directions and explores their application in LoRA-based parameter-efficient fine-tuning.
3. The authors propose the LoRA Dash algorithm, which proactively utilizes these influences to enhance the model fine-tuning process.

### Weaknesses
See detailed questions.

### Questions
Thanks for submitting to ICLR'25, I really enjoy reading your paper. I think your paper focuses on a hot and important topic. However, I have several further questions:

1. The paper lacks evaluation of the algorithm's overhead on end-to-end fine-tuning time. Can LoRA-Dash accelerate the fine-tuning phase as suggested by its name?
2. The benefits of the LoRA-Dash algorithm are unclear. Although it outperforms the vanilla LoRA approach when r is small, its improvement over the best fine-tuned LoRA model is minimal. Please clarify the real benefits of the LoRA-Dash algorithm.
3. Can you provide insights on applying TSD observations to other parameter-efficient fine-tuning methods, such as adapter/P-tuning? Are these methods suitable for this approach?

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
This paper proposes a parameter-efficient fine-tuning (PEFT) approach, LoRA-Dash, that explicitly exploits task-specific directions to improve parameter-efficiency. The proposed method consists of two phases: in the pre-launch phase, LoRA fine-tuning is performed for a certain number of steps to identify task-specific directions, which are estimated as the core directions of the pre-trained matrix most amplified by the LoRA update matrix. In the dash phase, changes in these directions are explicitly parameterized alongside the LoRA matrices to further enhance task alignment. The authors empirically demonstrate that LoRA-Dash outperforms standard LoRA and recent PEFT methods across a range of tasks and base models and show the robustness of the estimated task-specific directions.

### Strengths
A substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance. For example, originality may arise from a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results. You can incorporate Markdown and Latex into your review. See https://openreview.net/faq.
1. This paper introduces a novel framework for identifying task-specific directions in pre-trained models by projecting various task-related matrices onto the subspace of the original weight matrix, providing a consistent basis for analyzing task-specific adaptations.
2. The proposed method effectively leverages task-specific directions, achieving significant improvements over LoRA with minimal computational overhead, demonstrating its practical advantage as a PEFT method.
3. Empirical evaluations show that the task-specific directions derived from this approach align with those amplified by full fine-tuning, indicating that the method successfully identifies important directions for task adaptation.

### Weaknesses
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.
1. [Reliance on Preliminary Experiments for Hyperparameter Selection]
The proposed method relies on preliminary experiments to set key hyperparameters, such as the length of pre-launch phase and the number of dash directions. Although the chosen hyperparameters work well on the evaluated benchmarks, there is no guarantee that these settings generalize to other tasks and models. Moreover, the authors do not provide a systematic or principled approach for selecting these hyperparameters in varying scenarios.
2. [Lack of Objective Evaluation for Diffusion Models]
The evaluation of diffusion models relies solely on subjective assessments, which introduces potential biases and limits reproducibility. Incorporating quantitative metrics for diffusion model performance would strengthen the evaluation and provide a more comprehensive view of the model’s effectiveness.

### Questions
1. In Figure 5, the performance on OBQA with TSD in the left-most subplot does not match with Length of Pre-launch Phase set to 100 in the middle subplot. Why is this the case since both settings should correspond to the setting used in the main experiment?
2. For subject-driven generation tasks, how do the task-specific directions correspond to the objects or features in the image?

### Soundness
3

### Presentation
3

### Contribution
2
