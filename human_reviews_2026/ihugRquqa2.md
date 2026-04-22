# Grounding AI Explanations in Experience: A Reflective Cognitive Architecture for Clinical Decision Support

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Effective disease prediction in modern healthcare demands the twin goals of high accuracy and transparent, clinically meaningful explanations. Existing machine learning and large language model (LLM) based approaches often struggle to balance these goals. Many models yield accurate but unclear statistical outputs, while others generate fluent but statistically unsupported narratives, often undermining both the validity of the explanation and the predictive accuracy itself. This shortcoming comes from a shallow interaction with the data, preventing the development of a deep, detailed understanding similar to a human expert's. We argue that high accuracy and high-quality explanations are not separate objectives but are mutually reinforcing outcomes of a model that develops a deep, direct understanding of the data. To achieve this, we propose the Reflective Cognitive Architecture (RCA), a novel framework that coordinates multiple LLMs to learn from direct experience. RCA features an iterative rule refinement mechanism that improves its logic from prediction errors and a distribution-aware rules check mechanism that bases its reasoning in the dataset's global statistics. By using predictive accuracy as a signal to drive deeper comprehension, RCA builds a strong internal model of the data. We evaluated RCA on one private and two public datasets against 22 baselines. The results demonstrate that RCA not only achieves state-of-the-art accuracy and robustness with a relative improvement of up to 40\% over the baseline but, more importantly, leverages this deep understanding to excel in generating explanations that are clear, logical, evidence-based, and balanced, highlighting its potential for creating genuinely trustworthy clinical decision support systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a Reflective Cognitive Architecture (RCA) that uses multiple LLM roles to iteratively refine a rule base from training data errors and a distribution-aware check, aiming to couple accurate predictions with natural-language explanations. During training, RCA cycles a prediction LLM with reflection and checking LLMs to update rules using the global data distribution; the final rule base is then used for inference. Experiments include three tabular clinical datasets. The method is compared to 22 baselines, including traditional ML (e.g., CatBoost), zero-shot LLMs, reasoning LLMs, and LLM agents using tools/code. Evaluation reports Accuracy/MCC/F1 and doctor-scored explanation metrics for Cognitive Load, Logical Argumentation, Evidence-based Medicine, and Cognitive Biasing.

### Strengths
The paper introduces an interesting multi-LLM reflective architecture that combines prediction, reflection, and checking modules to link accuracy and explanation generation. This setup is conceptually original, with a reasonably thorough experimental section and an attempt to evaluate both performance and explanation quality quantitatively and qualitatively.

### Weaknesses
While the paper presents a creative multi-LLM reflective architecture and makes a commendable attempt to address explainability in clinical AI, its conceptual novelty, evaluation fairness, and empirical grounding are insufficient. The main ideas are overstated relative to prior work, the experimental comparisons are unbalanced, and the evaluation metrics do not convincingly support the key claims. The contribution remains interesting but underdeveloped both theoretically and methodologically.

### Questions
1. Could you clarify whether the RCA system is fine-tuned on training data while baselines are evaluated zero-shot? If so, do you have results for a version of RCA under the same zero-shot condition to isolate the effect of reflective learning rather than data exposure?

2. The contribution needs to be redefined. Re-thinking a problem, preparing dataset and evaluating proposed method are not contributions  

3. The four explanation criteria (Cognitive Load, Logical Argumentation, Evidence-based Medicine, Cognitive Biasing) appear to be author-defined rather than standardised measures. Could you clarify how these were operationalised, validated, or adapted from prior work to ensure they reliably assess explanation quality rather than serve as conceptual framing?

4. The distribution-aware rule check grounds logic in training statistics. How do you ensure that this process mitigates rather than amplifies demographic or sampling biases present in the data?

5.  Dataset accessibility: Since the CRT dataset is private and small, results replicability might be an issue. Please clarify

### Soundness
2

### Presentation
2

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
The paper proposes Reflective Cognitive Architecture (RCA), a framework that leverages multiple instantiations of a large language model to iteratively generate improved predictions along with corresponding explanations. The authors evaluate their approach against several strong baselines on three real-world datasets. They further test the framework under challenging conditions with noisy or suboptimal data, demonstrating consistent and significant improvements over all baselines across these scenarios.

### Strengths
I find this an interesting approach that leverages additional data to enhance LLM performance on specialised, data-grounded reasoning tasks. I particularly appreciate the idea of using misclassifications as a mechanism to dynamically update the set of rules followed by the model. The paper is very well written and exceptionally easy to follow. Moreover, the experimental setup is extensive and thorough, evaluating multiple metrics against common baselines and demonstrating that the method remains robust even under moderately noisy conditions. Each component of the proposed framework is shown to contribute meaningfully to the overall performance.

### Weaknesses
- Since the rule extraction process relies heavily on the available data, the model may become more susceptible to exploiting dataset-specific biases. It would be interesting to analyse how the proposed framework behaves under such conditions. One potential mitigation could be to complement the data-driven rule extraction with external knowledge sources, for example, a curated corpus of research papers, to guide the reasoning process using human-grounded knowledge.
- The comparison with standard LLMs (and, to some extent, with tool-augmented models) may not be entirely fair, as those models are not "trained" or exposed to the same data used in the proposed framework. This difference in data exposure could partially explain the observed performance gap. It would be useful to clarify this distinction and discuss its implications for the fairness of the comparison.

### Questions
- How would the proposed method perform on datasets with strongly biased features? Have you considered evaluating the robustness of your framework under such conditions?
- Given that the baseline LLMs were not exposed to the same training data used in your approach, how do you address the issue of fairness in the comparison? Do you think this difference in data access could have influenced the observed performance gap?

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
3

### Summary
This paper proposes a novel Reflective Cognitive Architecture (RCA) framework to address the important issue in AI-assisted diagnosis, where doctors require both high predictive accuracy and truth-worthy explanations. RCA has two main modules: an iterative rule refinement mechanism and a distribution-aware rules check mechanism. Such modules are implemented with LLMs, where a reflection LLM learns from prediction errors to update a "long-term memory" rule base, and a checking LLM validates these rules against the dataset's global statistics to ensure they are evidence-based and robust. Finally, a prediction LLM generates the predictions and explanations based on the optimized rule base as well as the global data distribution.

For the experiment, the authors evaluate RCA on three medical datasets, including a private one and two public ones. The authors proposed four metrics to evaluate the quality of explorations, and invited three doctors to score each explanation on a scale of
1 to 10 for each criterion. The results compared with 22 baselines claim that RCA achieved the best prediction accuracy and the explanation quality.

### Strengths
1. The paper focused on the important interprebabilty question in AI healthcare area. The proposed model is noval, consisting of learning between three LLMs. The authors rethink predictive accuracy not just as an end goal, but as a crucial reward signal that drives the model to build a more robust and fundamental "experience" with the underlying patterns. 

2. The paper propose to quantify the explanations with four criterions, i.e., low CL, sound LA, EBM, reduced CB, which can be widely-used in AI Interpretable research.

3. The paper conducts comprehensive experiments on three datasets and 22 baselines, toghther with ablation study and qualitative analysis. The code and prompt is provided for reproducibility.

### Weaknesses
1. The paper mentioned two parts in the related work, the traditional ML methods and LLM methods. However, neural-symbolic is also a very related research line. Some works in neural-symbolic AI also try to solve the gap between prediction accuracy and meaningful explanations. Such work can be added to the related work as well as the baselines. Meanwhile, there are some medical LLM like Med-Gemini. More baselines on such domain specific LLMs are recommended.

2. The rules are not showed in the paper. As the key component of this paper are highly-related to rules refinement, some motivating examples of rules are important to understand the model effectiveness. Meanwhile, how to check the justifiability of such rules can be explained. For example, we can invite domain experts to check, or compare with public guidelines.

3. The paper also does not show the detailed example of global data distribution. 

4. All the reported results in the table don't have variance. For instance, three doctors are invited to score the explanations, so the variance of these three doctors can be reported.

5. The results on Table 3-5 shows the score of explanations from three doctors. The rubric of such scores are in A.7. However, the rubric only has wide ranges like 5-7/7-10 points, however, in the Table 3-5, we can find that most scores of each baseline models and RCA can fall into the same range. Thus, it is not convincing that he small difference of explanations can indicate the quality of interpretability.

### Questions
1. The figure 1 can be polished. For instance, some key components like the meaning of $R_k$ and $R_f$ can be added in the caption or in the figure. 

2. The meaning of 'conc' is not defined in the paper.

3. The case study in 4.5 can include more details. For example, what are the rules during the self-reflection process? What is the score of explanations for these four metrics.

4. The location of Reproducibility Statement is recommended at the end of the main text (before references).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces a novel framework called Reflective Cognitive Architecture (RCA) aimed at improving both the accuracy and explanation quality of AI models in clinical decision support. By grounding AI explanations in direct data interaction, RCA iteratively refines rules to build deep understanding, ensuring that predictions are both accurate and supported by evidence. The authors demonstrate superior performance across multiple datasets compared to traditional ML, LLM-based methods, and other advanced approaches.

### Strengths
Some of the key contributions of this paper are as follows:

- The paper proposes RCA as a framework that coordinates multiple LLMs to learn directly from data through iterative rule refinement and distribution-aware checks. The refinement mechanism proposed in the paper is interesting and can be argued to be an extensible area of research.
 - The authors also provide strong empirical evaluation (more on this below) on three datasets (CRT, Diabetes, Heart Disease) demonstrate state-of-the-art accuracy and robustness, with significant improvements in prediction performance.
- Finally, given the extremely hard to quantify aspect of explanations, the authors made a commendable effort to provide a formal evaluation using a comprehensive set of metrics including accuracy, Matthews Correlation Coefficient (MCC), F1-score for predictions, and detailed criteria for explanation quality based on cognitive science and medical practice.

### Weaknesses
Given the above strengths, and the promise of the paper, there are several avenues where this can be improved upon 

- While the framework shows promising results on provided public datasets, the data sample size is quite limited, especially considering the large scale patient data that are available today. This naturally brings the question of real-world scalability of this approach to millions of patient records on a combination of diseases
- The paper also fundamentally suffers from a clear methodological description, especially around how different components (e.g., rule refinement and distribution check) interact. While the authors do provide detailed prompts used for the process in appendix, the overall description still lacks clarity, even considering the page limit
- Finally, while the authors have made a great effort at quantifying explanations, more formal comparison, especially as these are usable to human experts are needed. For example [1] is a great reference on how such evaluations have been performed in practice

[1] Slack, D., Krishna, S., Lakkaraju, H. et al. Explaining machine learning models with interactive natural language conversations using TalkToModel. Nat Mach Intell 5, 873–883 (2023). https://doi.org/10.1038/s42256-023-00692-8

### Questions
Some of key questions for the authors are as follows:

- Have the authors conducted edge case analysis? It would be interesting to examine how RCA handles extreme or rare cases in medical datasets, especially given the noisy nature of real-world data. 
- Furthermore, it would be benefical to understand how RCA generates rules across several patient sub-populations
- Finally have you considered the issues around scalability across datasets? It would be interesting to explore potential challenges and scalability when applied to larger or more diverse datasets beyond the provided examples.

### Soundness
3

### Presentation
2

### Contribution
2
