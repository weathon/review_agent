# Differentially Private Conditional Text Generation with RL-Boosted Control

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Generating high-quality synthetic text under differential privacy (DP) is critical for training and evaluating language models without compromising user privacy. Prior work on synthesizing DP *datasets* often fail to preserve key statistical attributes, suffer utility loss from the noise required by DP, and lack fine-grained control over generation. To address these challenges, we make two contributions. First, we introduce a hierarchical framework that decomposes DP synthetic text generation into two subtasks: *feature learning* and *conditional text generation*. This design explicitly incorporates learned features into the generation process and simplifies the end-to-end synthesis task. Through systematic ablations, we identify the most effective configuration: a rich tabular schema as feature, a DP tabular synthesizer, and a DP fine-tuned conditional generator, which we term ACTG (**A**ttribute-**C**onditioned **T**ext **G**eneration). Second, we propose Anchored RL (ARL), a post-training method that improves the instruction-following ability of ACTG for conditional generation. ARL combines RL to boost control with an SFT anchor on best-of-$N$ data to prevent reward hacking. Together, these components form our end-to-end algorithm **ACTG-ARL**, which advances both the quality of DP synthetic text (+20\% MAUVE over prior work) and the control of the conditional generator under strong privacy guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduced a hierarchical framework (with ACTG as the optimal configuration) and a novel Anchored RL recipe that, together, form our end-to-end algorithm ACTG-ARL. 
Experimental results show improvements over strong baselines (DP-FT, CTCL, Aug-PE) across two datasets and three privacy budgets. The paper is well-motivated, with ablation studies that validate each design choice. It contributes to advancing DP text generation by emphasizing fine-grained controllability—a valuable new dimension alongside utility and privacy.

### Strengths
The paper presents a novel and well-structured framework (ACTG) for differentially private (DP) text generation, which decomposes the synthesis process into feature learning and conditional text generation. This modular approach leads to improved privacy-utility trade-offs and provides interpretability benefits.

The introduction of Anchored Reinforcement Learning (ARL) further enhances instruction-following capabilities under privacy constraints, successfully addressing the problem of reward hacking that often plagues RL-based alignment.

### Weaknesses
1.	This paper aims to design a differentially private text generator, however, the argument for differential privacy is not sufficient:

①	Authors say Stage 0 doesn’t consume any privacy budget. Why treat Stage 0 with a trusted component and other stages not. I didn’t find the discussions in Appendix C.1

②	How to employ the framework and how to prove the framework is differentially private are not clear. For the DP feature generator, DP-FT is a text generator, how to generate feature here? For DP conditional generator, how to perform DP-FT, and the method( “prompting a powerful LLM…”) didn’t protect privacy.

③	The usage of ARL is not clear. Authors did not the illustration of privacy for this step.

2.	The baseline coverage could be broader. The paper mainly compares against CTCL, DP-FT, and Aug-PE; including recent diffusion-based or graphical-model-based DP synthesizers (e.g., Ochs & Habernal 2025; DeSalvo et al. 2024) would further solidify the empirical claims. 

3.	Some technical details—such as PPO background, reward signal stability, and computational overhead of best-of-N sampling—could be expanded for clarity. Additionally, all experiments are performed using a single model size (gemma-3-1b-pt), leaving scaling behavior unexplored.

4.	while ARL is effective, its reliance on an LLM oracle for annotation and evaluation may limit reproducibility or accessibility for smaller labs.

5.	The conclusion in Figure 4(a) does not illustrate the issue because the difference in ε=1/∞ is significant. (The experiment for eps=∞ does not illustrate the issues, as in this case, this is no privacy guarentee.)

### Questions
1.	How sensitive is the performance of ARL to the hyperparameter γ and the number of candidates N in best-of-N sampling?

2.	Could the authors clarify the computational cost of the ARL fine-tuning stage relative to ACTG and ACTG-RL?

3.	Is it possible to integrate diffusion-based DP generators into the Stage-1 feature generation step to improve diversity?

4.	How would the proposed method scale if larger or smaller base models were used?

5.	In the ablation study of 5.3.3.2, why is it said that ground-truth features of D^x_priv is not available? This is just a comparative experiment.

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
4

### Summary
The paper presents a new hierarchical framework for generating high-quality synthetic text under differential privacy (DP). Prior approaches to DP synthetic text generation often struggle with preserving key statistical attributes, suffer from significant utility loss due to injected noise, and lack fine-grained control during generation.To address these issues, the authors decompose the DP synthetic text generation task into two subtasks: feature learning and conditional text generation. Their framework uses a rich tabular schema as a feature representation, a DP tabular synthesizer to ensure privacy during feature learning, and a DP fine-tuned conditional generator for text synthesis. Another key contribution is Anchored Reinforcement Learning (Anchored RL), a post-training method that enhances the instruction-following capability of the conditional generator, ACTG, under DP constraints. Empirically, the proposed method improves both text quality and control compared to prior work. It also offers strong privacy guarantees, allowing the resulting DP synthetic datasets to be reused without additional privacy costs.

### Strengths
- The paper is clearly written and easy to follow, with a logical flow that makes the main ideas and contributions understandable.

- The arguments are well structured, and the overall organisation effectively supports the proposed framework and experimental results.

- The inclusion of code significantly enhances the validity and reproducibility of the work. The codebase is well structured, sufficiently documented, and provides a strong foundation for others to build upon.

### Weaknesses
- The reported performance improvements, while consistent, are modest compared to baselines such as vanilla DP-FT and CTCL. The authors could strengthen their claims by including statistical measures of variability (e.g., variance bars or standard deviations) in the plots to show robustness across multiple runs. 

- It would be valuable to include an additional baseline that applies Anchored RL directly to vanilla DP-FT. This would help isolate and clarify the contribution of Anchored RL to the overall performance gains.

- All experiments were conducted using a single model, which the authors acknowledge as a limitation. 


The paper presents a solid contribution with clear writing, strong methodological grounding, and commendable reproducibility. However, I remain uncertain about how much of the reported improvements can be attributed to noise, given the absence of variance bars or discussion of experimental variability. Therefore, I am assigning a score of 6. I would be open to reconsidering my score based on the rebuttal, particularly if the authors can clarify the robustness of their results across multiple runs and provide additional statistical evidence supporting the observed gains.

### Questions
- I could not locate the script used to produce Figure 6, Appendix C (schema identification) in the provided codebase. Including this script would make the work fully reproducible and ensure that others can replicate the S3 approach end-to-end.

- Could the authors clarify whether the number or quality of extracted features influences the effectiveness of the S3 approach? An ablation study examining how varying the number of features impacts performance would provide deeper insights into the method’s sensitivity and limitations.

### Soundness
4

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
This paper presents a new approach to generating DP synthetic text using a hierarchical framework. The model is designed to address the challenges of privacy, data utility, and controlled generation. The framework decomposes the generation task into two stages: feature learning and conditional text generation. The authors introduce ACTG (Attribute-Conditioned Text Generation), a method that optimizes both DP synthetic text quality and fine-grained control over the generation. They also propose a post-training method, Anchored RL (ARL), which improves the instruction-following ability of ACTG by addressing control degradation under DP.

### Strengths
1.The hierarchical framework (ACTG) combines a DP tabular synthesizer with a DP fine-tuned conditional generator. This separation of tasks allows for improved optimization.

2.The introduction of ARL to address instruction-following is a significant contribution. It demonstrates the balance between privacy and control in DP settings, achieving better performance than previous methods.

3.ACTG-ARL outperforms prior methods in multiple metrics, including MAUVE and attribute distribution matching. This provides solid evidence of its practical utility.

### Weaknesses
1.The framework depends heavily on accurate feature extraction, but does not provide a detailed description of the feature extractor, especially regarding how attributes are selected or normalized before DP processing. 

2.The difference between this work and CTCL isn’t made very clear. Although the paper introduces some refinements, it largely follows CTCL’s existing structure of “public pretraining → private fine-tuning → synthetic generation.”  The overall idea and workflow feel quite similar, and the updates seem more like incremental technical improvements than a fundamentally new approach.

3.The evaluation is limited to one model configuration, leaving open the question of how well the approach generalizes across different model sizes and capacities.

### Questions
1.The evaluation is conducted on a single model configuration, but the chosen model and experimental setup raise questions. In the Aug-PE stage, the paper replaces the original GPT-3.5 (used in prior work) with Qwen, which undermines one of the core advantages of the approach,its compatibility with black-box LLMs. This substitution also makes it unclear whether the reported improvements are due to the framework itself or to differences in the underlying model. Furthermore, Aug-PE should theoretically outperform DP-FT given its reinforcement of privacy-preserving generalization, yet this expected advantage is not reflected in the experimental results.
2.How can the trade-off between control and text fidelity be better balanced during the joint optimization of SFT and ARL?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present ACTG-ARL, a novel framework for differentially private (DP) conditional text generation that aims to improve both the quality of synthetic text and fine-grained control over generation while maintaining strong privacy guarantees. The authors propose a hierarchical framework that decomposes DP synthetic text generation into feature learning and conditional text generation. They also introduce Anchored RL (ARL), a post-training method that uses reinforcement learning (RL) with a supervised fine-tuning (SFT) anchor to boost instruction-following ability and mitigate reward hacking.

### Strengths
S1. The authors clearly highlight the weaknesses in existing DP based text generation techniques. The motivation for the paper is strong.

S2. ACTG is modular and hierarchical potentially allowing better privacy utility tradeoffs.

S3. I like the Anchored RL approach for better instruction following in DP-trained conditional generations. The empirical results look strong and the evaluations are well grounded.

### Weaknesses
W1. The complexity and computational cost of the overall framework might be challenging to implement. More discussion on this should be added in the paper. 

W2. The quality of the generated features and rewards is directly tied to the capabilities of these oracle LLMs. The paper acknowledges this with the discussion on extraction error but doesn't fully explore potential limitations if a less capable or open-source LLM is used as the oracle.

W3. While the structured tabular schema (S3) performs well, the process of designing such a schema (LLM-assisted) is described as "dataset-specific." This raises questions about how much manual effort or expert knowledge is required to create effective schemas for new domains, and whether the LLM assistance is truly robust across highly diverse data.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
