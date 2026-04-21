# TP-Eval: Tap Multimodal LLMs' Potential in Evaluation by Customizing Prompts

- Avg Score: 4.67
- Decision: Reject
- Scores: 5, 3, 6

## Abstract
Recently, multimodal large language models (MLLMs) have received much attention for their impressive capabilities. The evaluation of MLLMs is becoming critical to analyzing attributes of MLLMs and providing valuable insights. However, current benchmarks overlook the problem of prompt sensitivity - minor prompt variations may lead to significant performance fluctuations. Thus, inappropriate prompts may obscure the models' capabilities, underestimating the models' performance.  Moreover, different models have different preferences for different prompts, and thus, using the same prompt for all models will cause evaluation bias. This paper analyzes this deficiency in existing benchmarks and further introduces a new evaluation framework named TP-Eval, which introduces a prompt customization method to reduce evaluation biases and tap models' potential. TP-Eval will rewrite the original prompts to different customized prompts for different models. In particular, we propose some well-designed modules for prompt customization tailored to the scenario of MLLM evaluation. Extensive experiments demonstrate the effectiveness of our approach to uncovering models' capabilities, and TP-Eval should benefit the community in developing more comprehensive and convincing MLLM evaluation benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper addresses the issue of prompt sensitivity in the evaluation of multimodal large language models (MLLMs). It argues that current benchmarks overlook the impact of prompt variations on model performance, which can lead to significant fluctuations and underestimation of capabilities. The paper introduces a new evaluation framework named TP-Eval, which customizes prompts for different models to reduce evaluation biases and tap into the models' potential more effectively.

### Strengths
1. The paper analyzes and highlights the problems with existing MLLM benchmarks that lead to underestimation and evaluation bias due to prompt sensitivity.
2. It introduces TP-Eval, an evaluation framework that customizes optimal prompts for different models through automatic prompt optimization, tailored to MLLM benchmarks.

### Weaknesses
1. I think the authors should deeply analyze and summarize prompt improvement based on TP-Eval and provide more insights for building reasonable benchmarks in the future.

### Questions
I have a question about the motivation of the paper. Why optimize the prompt for the MLLM benchmark? I think this paper proposes a good method for automation prompt engineering. I also understand the problem of prompt sensitivity. A slight change in the prompt will cause a drastic change in the evaluation results. 

However prompt sensitivity should be a problem with the MLLM itself. The more robust the model, the more stable the evaluation results. In addition, the same evaluation set can ensure the fairness of the evaluation. Therefore, the optimization direction should be on MLLM, not the evaluation benchmark.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper tries to addresse the challenge of evaluating MLLMs accurately. The authors highlight that current benchmarks often overlook the sensitivity of MLLMs to prompt variations, which can lead to significant performance fluctuations and underestimation of model capabilities. They propose a new evaluation framework called TP-Eval, which aims to reduce evaluation biases and unlock the potential of MLLMs by customizing prompts for each model.
TP-Eval introduces a framework includes several modules for prompt customization, taking into account the unique scenario of MLLM evaluation. The authors conduct extensive experiments to demonstrate the effectiveness of TP-Eval in uncovering the true capabilities of MLLMs. They show that by rewriting original prompts to customized prompts for different models, TP-Eval can significantly improve the performance of MLLMs across various tasks.
The authors argue that the true capabilities of MLLMs should be assessed under optimal prompts, which are derived from slight modifications of the original prompts while maintaining the semantic integrity of the task instructions. TP-Eval ensures fairness across models and manages labor costs by automating the prompt customization process.

### Strengths
The paper proposes a novel evaluation framework, TP-Eval, which addresses the issue of prompt sensitivity in MLLMs. This is an innovative approach to improving the accuracy and reliability of MLLM evaluations.

### Weaknesses
This approach may raise concerns about over-optimization for the test set. MLLMs are capable of understanding a wider range of prompts and following instructions to provide responses, which is an issue that needs to be addressed during the alignment phase using methods such as SFT or RL. Existing evaluations primarily assess the capabilities of MLLMs during the IFT or pre-training stages.
The experiments are not sufficient, failing to consider the consistency at different stages of MLLMs, such as pretrain/IFT/alignment models, nor do they take into account the uncertainties and instabilities brought by personalized prompts to model iterations.

### Questions
1. In user usage scenarios, optimized prompts are unlikely to be used, instead, ‘poor’ or ‘average’ prompts are more probable. Therefore, does the underestimation of MLLMs mentioned in the article actually exist? Is there a suspicion of over-optimization for the test set?
2. TP-Eval represents the performance of each model under their respective best prompts. How correlated is this with the effect of randomly selecting an "average" prompt? It would be appreciated if corresponding conclusions could be supplemented.
3. Pretrained or IFT MLLMs can mostly only follow simple instructions. Can TP-Eval provide corresponding experimental results on such models?
4. How to use this benchmark for model iteration? For example, if change to a more powerful vision encoder or LLM backbone, will it be reflected in the results?

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
This paper introduces an automated prompt optimization framework, TP-Eval, which addresses the prompt sensitivity problem in multimodal large language models.

### Strengths
Strength
1.Innovative Approach (Prompt Customization): This paper introduces an automated prompt optimization framework, TP-Eval, which addresses the prompt sensitivity problem in multimodal large language models. By customizing prompts with a small sample set, this method effectively reduces prompt bias, making an innovative contribution to the evaluation of multimodal models.
2.Introduction of an Introspection Mechanism: The introspection mechanism aids in prompt refinement during optimization, not only improving the prompts but also avoiding simple semantic drift. This mechanism enhances model accuracy across diverse tasks, demonstrating a practical and effective design for improved performance on complex tasks.
3.Exploration of Zero-Shot Optimization for Data-Limited Scenarios: The zero-shot optimization method successfully addresses data scarcity or privacy-restricted scenarios through In-Context Learning (ICL) examples. Experimental results show that, even without samples, the model achieves optimized performance through zero-shot methods, representing a practical extension of TP-Eval.
4.Comprehensive Experimental Design: The paper’s experiments cover various task types and include ablation studies and error analysis, providing a thorough view of TP-Eval’s strengths and limitations with well-supported data. These experiments offer reviewers a solid basis for understanding the method’s effectiveness.
5.Revealing Previously Inflated Metrics with Introspection Mechanism: The introspection mechanism reveals that certain high metrics previously observed in some tasks were inflated due to the model merely “memorizing” specific prompt patterns rather than genuinely understanding task requirements. For instance, in anomaly detection tasks, explicit prompt instructions identifying option A as normal and B as abnormal led the model to rely on option order, resulting in semantic overfitting. The introspection mechanism breaks this rigid pattern, requiring the model to “understand” the task more deeply, thereby exposing misleading high scores previously masked. (Section 5.3, Ablation Study)

### Weaknesses
1.Overfitting Risk Not Fully Addressed: Although the paper introduces introspection and reordering mechanisms, some tasks in the experiments still experience optimization failures or overfitting (as seen in error analysis, Section 5.2.3). This issue may be more pronounced in multimodal tasks with smaller datasets. The robustness and generalizability of TP-Eval require further validation.
2.Strong Model Dependency: Experimental results (Figure 5) show that different models respond quite differently to the same optimized prompts, indicating that TP-Eval requires individual tuning and optimization for each model, lacking general applicability. This raises costs for multi-model or large-scale deployments, limiting the method's practicality.

### Questions
I don't have questions.

### Soundness
3

### Presentation
3

### Contribution
2
