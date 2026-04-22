# Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 0

## Abstract
While recent advances in machine learning have equipped Weather Foundation Models (WFMs) with substantial generalization capabilities across diverse downstream tasks, the escalating computational requirements associated with their expanding scale increasingly hinder practical deployment. Current Parameter-Efficient Fine-Tuning (PEFT) methods, designed for vision or language tasks, fail to address the unique challenges of weather downstream tasks, such as variable heterogeneity, resolution diversity, and spatiotemporal coverage variations, leading to suboptimal performance when applied to WFMs. To bridge this gap, we introduce WeatherPEFT, a novel PEFT framework for WFMs incorporating two synergistic innovations. First, during the forward pass, Task-Adaptive Dynamic Prompting (TADP) dynamically injects the embedding weights within the encoder to the input tokens of the pre-trained backbone via internal and external pattern extraction, enabling context-aware feature recalibration for specific downstream tasks. Furthermore, during backpropagation, Stochastic Fisher-Guided Adaptive Selection (SFAS) not only leverages Fisher information to identify and update the most task-critical parameters, thereby preserving invariant pre-trained knowledge, but also introduces randomness to stabilize the selection. We demonstrate the effectiveness and efficiency of WeatherPEFT on three downstream tasks, where existing PEFT methods show significant gaps versus Full-Tuning, and WeatherPEFT achieves performance parity with Full-Tuning using fewer trainable parameters. The code of this work is available at https://github.com/ShileiCao/WeatherPEFT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Weather foundation models (WFMs) are important.

PEFT failed to address the challenges of weather downstream tasks, such as variable heterogeneity, resolution diversity, and spatiotemporal coverage variations, leading to suboptimal performance when applied to WFMs.

This paper introduces WeatherPEFT, which is tailed PEFT for WFMs.

### Strengths
This paper has done comprehensive study to make the WeatherPEFT works for the weather foundation models. Compared to the standard PEFT like LoRA, WeatherPEFT surpasses them and approaches full-tuning performance even with a minimal parameter budget e.g. ~0.3%.

### Weaknesses
One concern with this work is that the proposed method lacks fair evaluation. From the proposed method, it does not seem to be unique to weather foundation models at all. It seems that the proposed method would work to some extent on the other problems too. So it seems that the evaluation only on the weather foundation model is not a fair comparison.

### Questions
1. Can you apply the proposed PEFT method on any dataset outside of the weather domain?

2. How sensitive is the proposed HW-adaptive in weatherPEFT to the different resolution parameter?

### Soundness
3

### Presentation
3

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
The authors propose WeatherPEFT, a parameter-efficient fine-tuning method for Weather Foundation Models. The novelty in this work is to combine: (i) Task-Adaptive Dynamic Prompting (TADP), which extracts spatial, variable-level, and meteorological patterns from the encoder’s embedding weights and injects them as soft prompts into the backbone during the forward pass, and (iii) Stochastic Fisher-Guided Adaptive Selection (SFAS), which uses Fisher information (plus controlled randomness) to update only the most task-critical parameters, and preserve core pre-trained knowledge. The authors show that, altogether, these components allow WeatherPEFT to achieve performance comparable to (and sometimes better than) full fine-tuning across tasks such as downscaling, ensemble forecast post-processing, and precipitation prediction, while using significantly fewer trainable parameters than existing PEFT methods.

Admittedly, I am not an expert in Weather Foundation Models (WFM), but I believe that, aside from being able to properly judge how this work relates to previous WFM works, this lack of expertise does not affect my ability to judge the main contribution of the paper, i.e. the PEFT part.

### Strengths
1. I congratulate the authors on the clarity of this paper. It is extremely well structured and most of the components are very well explained.

2. Climate change exerts pressure through environmental and societal impacts, including more extreme weather events like floods, droughts, and heatwaves, leading to sea-level rise, and causing water scarcity, food insecurity, and habitat destruction. The relevance of this work does not need any justification, and such techniques will be of great aid in the years to come.

3. WeatherPEFT shows consistently strong results across downstream tasks.

4. The authors do a great work in systematically comparing their results against strong PEFT methods and baselines, providing further support of the usefulness of their approach.

### Weaknesses
1. WeatherPEFT is a more complex technique that involves the sequential implementation of three adapters, on top of the rest of the TADP process and the SFAS. Thus, the results shown come at a computational cost.

2. Although I am not asking the authors to perform more experiments, evaluating their method on further downstream tasks could be of interest for future work.

### Questions
1. Have the authors tried to evaluate the increase of performance relative to the added complexity of their method?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
They aimed to tackle the task-adaptive fine-tuning of the foundation models for weather forecasting. To deal with the domain specific challenges in weather forecasting (eg, task-specific characteristics), they proposed two PEFT ideas, specifically: i) task-adaptive dynamic prompting and ii) stochastic fisher-guided adaptive selection. The proposed PEFT method was tested on three downstream tasks/scenarios: downscaling, ensemble, and regional forecasting.

### Strengths
- A novel adapter module called task-adaptive dynamic prompting looks interesting. 
- Interesting applications of PEFT of foundation models for the weather forecasting domain such as regional forecasting and ensemble forecasting.

### Weaknesses
- Arguments in Line 074-082 on why existing PEFT methods fail on weather task adaptation problems, is not very convincing since they also seem to hold for language and vision tasks (eg, task-wise varying input characteristics, different parameters play varying roles in different tasks). Are there any more convincing arguments on why the weather domain is specific?

- Can the proposed TADP and SFAS be applied to general domains? like vision and language downstream task PEFT? Why is it specifically useful for weather data? Some PoC tests on general language and vision domains can be useful. Eg, the HW-, V-, D-adapters, and the external pattern integration in the proposed TADP appear to be applicable to these domains as well almost straightforwardly without modification. 

- MoE (eg, mixture of lora experts) is also a very powerful and topical tool for FM and PEFT. As it's not compared in the Curious to see how it would perform on the benchmarks they used as a baseline.  

- SFAS seems to be the technique relatively well known in the PEFT literature. So it may not be appropriate to claim it as one of the main contributions or novelty of the paper.

- The PEFT baselines they tested are a bit old, eg, beyond lora there are many more, eg, 
sparse lora, vera, or svd-based low rank selection.

- Also, as sparse PEFT competing baselines, as a comparison to the proposed SFAS, the authors may need to consider various pruning methods, and the second-order sparse selection method like [R2].

[R2] Zihao Fu, Haoran Yang, Anthony Man-Cho So, Wai Lam, Lidong Bing, and Nigel Collier. On the Effectiveness of Parameter-Efficient Fine-Tuning. AAAI, 2023.

- Overall, since the sparse adaptive parameter selection based on Fisher info is relatively well known, the main novelty of the paper seems to be merely adding the TADP adapter module. But it is not very clear if the impact of TADP is so significant for publication in this area. Is it better than some combinations of existing PEFT and sparse adaptive parameter selection techniques that the authors did not explore but are often performed in practice? More empirical exploration may be needed together with better justification of the TADP module.

### Questions
See questions in the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
WeatherPEFT introduces a novel Parameter-Efficient Fine-Tuning framework for Weather Foundation Models. It combines Task-Adaptive Dynamic Prompting for context-aware feature recalibration and Stochastic Fisher-Guided Adaptive Selection for stable parameter optimization, achieving efficient fine-tuning and performance parity with full-tuning using fewer trainable parameters.

### Strengths
- The paper is well-written as origanizing the methods and experiments clear.

- This paper compares many methods, demonstrating that the experiments are extensive and comprehensive.

- The appendix is clear and solid.

### Weaknesses
- The description about related works is actually incorrect from Line 078-082. First, there are a lot of task-specific PEFT methods introduced to dynamically select task-specific features to do the adaptation (SCT IJCV 2023) or implicitly selecting features (AdaptFormer, Adapter). Second, most of PEFT methods trained different sets of parameters for different downstream tasks, hence the "apply the same set of trainable parameters across different downstream tasks" is wrong. Then, it still unknow what is the key difference of conducting weather like downstream tasks with the normal vision tranfer tasks in VTAB. 

- Based on the above factual error, it indicates (1) the motivation in this paper is incorrect; (2) the literature review of this submission is not adequate and fully understood.

- The SFAS module is similar to the paper "Raise a Child in a Large Language Model: Towards Effective and Generalizable Fine-tuning". The novelty is limited. Please be careful give the comparison.

- The selected PEFT baselines are old, ConvPass, FacT, NOAH, SCT, RepAdapter are not included.

- It is confusing why w/o Internal and w/o External shows minor improvements as compared with WeatherPEFT in Table 8.

### Questions
None

### Soundness
1

### Presentation
2

### Contribution
1
