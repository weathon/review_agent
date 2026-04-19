# Automated Filtering of Human Feedback Data for Aligning Text-to-Image Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 5

## Abstract
Fine-tuning text-to-image diffusion models with human feedback is an effective method for aligning model behavior with human intentions. However, this alignment process often suffers from slow convergence due to the large size and noise present in human feedback datasets. In this work, we propose FiFA, a novel automated data filtering algorithm designed to enhance the fine-tuning of diffusion models using human feedback datasets with direct preference optimization (DPO). Specifically, our approach selects data by solving an optimization problem to maximize three components: preference margin, text quality, and text diversity. The concept of preference margin is used to identify samples that are highly informative in addressing the noisy nature of feedback dataset, which is calculated using a proxy reward model. Additionally, we incorporate text quality, assessed by large language models to prevent harmful contents, and consider text diversity through a k-nearest neighbor entropy estimator to improve generalization. Finally, we integrate all these components into an optimization process, with approximating the solution by assigning importance score to each data pair and selecting the most important ones. As a result, our method efficiently filters data automatically, without the need for manual intervention, and can be applied to any large-scale dataset. Experimental results show that FiFA significantly enhances training stability and achieves better performance, being preferred by humans 17% more, while using less than 0.5% of the full data and thus 1% of the GPU hours compared to utilizing full human feedback datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes FiFA (Filtering for Feedback Alignment), a novel automated data filtering approach for efficiently fine-tuning text-to-image diffusion models using human feedback data. 

The main contributions are:
* An automated data selection method that maximizes: preference margin, text quality and text diversity.
* Formulation of data selection as an optimization problem to find a subset that maximizes these components.
* Empirical evidence showing FiFA's effectiveness across various models and datasets.

### Strengths
The paper presents a novel approach to data filtering for fine-tuning text-to-image diffusion models. While data pruning and coreset selection are not new concepts in the domain of text-to-image diffusion models (first documented by Meta’s EMU paper), this work focuses on the automation of coreset selection. The combination of preference margin, text quality, and text diversity in a single optimization framework is an effective and reasonable solution in this problem space.

The paper demonstrates effective results across different models (SD1.5 and SDXL) and datasets (Pick-a-Pic v2 and HPSv2), providing robust evidence for their claims. The inclusion of both automatic metrics and human eval provide a complete picture in terms of metrics. There is also some  theoretical analysis provided in the author’s paper.

its most impressive for the authors to achieve high quality alignment with just 0.5% of the data and 1% of the GPU hours. FiFA also demonstrated reduction in harmful content generation, which is critical for these automatic coreset selection method.

### Weaknesses
I think the biggest issue with this work is that it did not experiment with strong diffusion models like SD3-2B or FLUX models or the Playground models. Those models are much better to start with. It would be very helpful to know if the proposed model can further improve strong models.

### Questions
The authors highlighted that the method can achieve good results with just 0.5% of the data. Do you have results showing how well FiFA filtering works on say 0.1%, 1%, 5%, 10% of the dataset? It could help us understand how tunable FiFA is.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors aim to improve aligning text-to-image diffusion models from the perspective of filtering human feedback data. Specifically, they select the data pairs by maximizing three components: preference margin, text quality, and text diversity. For each component, they design an optimization objective. Finally, several experiments have been conducted to verify the contribution of each component

### Strengths
1.	The motivation of filtering the human feedback data is reasonable. It is well-known that the training of diffusion is very cost. High quality data would contribution both the effectiveness and efficiency of the model.

2.	The paper writing is great.

### Weaknesses
1.	The technical contribution is relatively small. In my opinion, the proposed approaches for filtering data are travel and nature. In addition, as for preference margin, I believe that it is better to maximize preference margin in a limited range, while a very large margin would provide difficulty for optimization. 

2.	Only the pick-a-pic dataset is used in the experiment, which is highly related to Pick Score. Some other datasets should be involved to verify the generalization. For example, the authors can use the HPS score to compute the preference margin, even on the same pick-a-pic dataset.

3.	It is also significant to show the pair-wise human evaluation in Figure 4.

### Questions
How about the effectiveness of the proposed approaches on other datasets with different preference models?

I also want to the win rate of the proposed approaches compared with only the base model or DPO.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents a novel approach to fine-tuning text-to-image diffusion models using human feedback data filtered through an automated algorithm. The proposed methodology optimizes the fine-tuning process by selecting a subset of the available human feedback based on a preference margin criterion, enhancing the reward value while considering both prompt quality and diversity to maintain robustness and mitigate potential harmful content.

### Strengths
Please see questions

### Weaknesses
Please see questions

### Questions
1. The proposed filtering algorithm systematically narrows down the human feedback dataset to a subset that is optimal for model fine-tuning. As a general approach, further discussion on the generalizability of this filtering approach could enrich the analysis, such as how it may integrate with other alignment frameworks like RLHF and DPO-based methods. In addition, expanding the range of comparative methods would strengthen the evaluation.

2. To clarify the novelty of this approach, the specific roles of preference margin and the quality/diversity metrics for text prompts could be further justified. Detailing the design motivations behind these components and their interdependencies would clarify their contributions to the model’s overall performance.

3. DPO requires extensive high-quality preference data, which can be costly and difficult to obtain. The accuracy of preference data is essential, as low-quality feedback may lead to biased or suboptimal model behavior. There appears to be some ambiguity in the statement regarding dataset preparation: "To ensure safety, we manually filter out some harmful text prompts from these test prompts, resulting in 446 unique prompts." It seems that an additional manual filtering step was applied before evaluating the proposed algorithm’s ability to handle harmful prompts. Clarifying this step’s rationale and how it affects the filtering method’s efficacy would add clarity.

4. DPO-based approaches can sometimes narrow the scope of outputs, potentially limiting diversity. To validate the claimed advantage of this filtering method in maintaining diverse outputs, more empirical evidence should be presented. Additionally, a comparison with online DPO and recent DPO variants would help contextualize the findings.

5. More qualitative evidence on how the proposed approach reduces training costs would be valuable, particularly with concrete examples or case studies showing the efficiency gains obtained through this filtering algorithm.

6. Human evaluation is conducted in this work, however, it appears that there is no evidence of human ethics approval, despite it potentially being a low-risk case.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces FiFA, an automated data filtering algorithm designed to optimize fine-tuning of text-to-image diffusion models, aligning model behavior more effectively with human intent. While human feedback datasets are valuable for model alignment, their large size and high noise levels often hinder convergence. FiFA enhances fine-tuning by automatically filtering data based on an optimization problem that maximizes three key components: preference margin, text quality, and text diversity. Experimental results show that FiFA enhances training speed and achieves better performance.

### Strengths
1. The paper propose the FiFA algorithm, which leverages three core metrics—preference margin, text quality, and text diversity—to optimize data filtering automatically. This approach effectively addresses noise in human feedback datasets and improves the fine-tuning of diffusion models, particularly for large-scale datasets.
2. The paper is of high quality, with well-designed and comprehensive experiments, including several ablation and comparative studies that strongly support the effectiveness of FiFA in enhancing training efficiency and image quality.
3. The structure of the paper is clear and well-organized. Key concepts, such as preference margin, text quality, and text diversity, are clearly defined, making the methodology accessible.

### Weaknesses
1. In the paper, some equations lack corresponding equation numbers.
2. In the introduction, the phrasing around "difficulty of convergence" is inconsistent with the discussion of the iterative training required for diffusion models. It is recommended that the authors clarify the logical flow.
3. In Equation 2, there is an extra left parenthesis "(".
4. When using pre-trained models (e.g., CLIP, BLIP) to calculate the preference margin, how is the validity of the results ensured? Given that pre-trained models are trained on noisy and ambiguous datasets, they may also yield incorrect results.
5. From the ablation study results, the effects of Text quality and Text diversity are not very significant. The authors state, "when combined with a high margin, they outperform the model trained solely on margin, highlighting the importance of both components," but where are the results? Is the improvement due solely to the higher margin?

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
