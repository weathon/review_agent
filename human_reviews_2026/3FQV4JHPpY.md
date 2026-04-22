# Expanding Reasoning Potential in Foundation Model by Learning Diverse Chains of Thought Patterns

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8

## Abstract
Recent progress in large reasoning models for challenging mathematical reasoning has been driven by reinforcement learning (RL). 
Incorporating long chain-of-thought (CoT) data during mid-training has also been shown to substantially improve reasoning depth. 
However, current approaches often utilize CoT data indiscriminately, leaving open the critical question of which data types most effectively enhance model reasoning capabilities.
In this paper, we define the foundation model's reasoning potential for the first time as the inverse of the number of independent attempts required to correctly answer the question, which is strongly correlated with the final model performance.
We then propose utilizing diverse data enriched with high-value reasoning patterns to expand the reasoning potential. 
Specifically, we abstract atomic reasoning patterns from CoT sequences, characterized by commonality and inductive capabilities, and use them to construct a core reference set enriched with valuable reasoning patterns. 
Furthermore, we propose a dual-granularity algorithm involving chains of reasoning patterns and token entropy, efficiently selecting high-value CoT data (CoTP) from the data pool that aligns with the core set, thereby training models to master reasoning effectively. 
Only 10B-token CoTP data enables the 85A6B Mixture-of-Experts (MoE) model to improve by **9.58\%** on the challenging AIME 2024 and 2025, and to raise the upper bound of downstream RL performance by **7.81\%**.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CoTP, a framework for expanding the reasoning potential of foundation models by selecting high value chain of thought (CoT) data. The authors define “reasoning potential” as the inverse of expected attempts to solve a problem, extract atomic reasoning patterns to build a core set, and use a dual granularity similarity measure (pattern chains + token entropy) with a Hungarian assignment algorithm to select training data. Experiments on an 85A6B MoE model show significant gains on challenging math benchmarks and improved RL upper bounds.

### Strengths
1. Provides a formal definition of reasoning potential, linking it to expected attempts and offering theoretical grounding.

2. Proposes a novel data selection method combining reasoning pattern chains and token entropy, operationalized with weighted DTW and Hungarian matching.

3. Demonstrates strong empirical improvements on difficult mathematical reasoning tasks while maintaining general performance.

### Weaknesses
1. The construction of the core set heavily depends on strong model annotations and manual filtering, which may introduce bias, limit scalability, and lead to redundancy. 

2. The similarity computation (weighted DTW combined with the Hungarian algorithm) could be computationally expensive when applied to very large data pools, raising concerns about efficiency.

3. The reliance on Chinese‑expressed reasoning patterns for distance computation lacks empirical justification, and the paper does not clearly explain how consistency and quality of the large set of abstracted patterns are maintained at scale.

### Questions
1. Could the authors clarify how they mitigate potential bias and scalability issues in the core set construction process, and provide an analysis of the computational cost of their similarity matching pipeline at scale?

2. The paper relies on reasoning patterns expressed in Chinese to compute distances between pattern chains. However, it is unclear whether there is empirical evidence demonstrating that this language‑specific representation is optimal or robust. Could the authors provide experiments or ablations to justify this choice?

3. Moreover, when abstracting a large number of reasoning patterns from massive datasets, how do the authors ensure that the extracted patterns remain consistent, controlled, and aligned with the intended definition of “high‑value” reasoning steps? What mechanisms are in place to prevent noise or drift in the pattern inventory?

4. The proposed CoTP framework shows consistent and significant improvements on mathematical reasoning benchmarks, yet it does not yield comparable gains on general‑domain tasks, sometimes even underperforming other baselines. Could the authors clarify why the method appears highly effective in math reasoning but does not extend to broader general benchmarks? Are there inherent limitations in the design of CoTP that make it domain‑specific?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper discusses which kind of chain of thought data expand the reasoning potential of a foundation model when used during mid-training and how they should be selected?
The authors annotate CoT chains with an atomic reasoning pattern and token wise entropy and weight patterns using TF-IDF to score them with a weighted DTW and use Hungarian algo to match each core instance with the closest candidate data point. With this, they use a pre-trained 85A6B base model and do a mid training with 10B tokens on selected COTP data. They achieved better performance on mathematical datasets. They further perform ablations to justify the importance of entropy based token selection.

### Strengths
- The paper uses the reasoning potential of a language model to motivate a data selection process. This is very interesting and justifies the need for such a method to be used during mid-training. 
- The paper is clearly presented and each step in the process is well specified with neat plots and algorithms wherever needed. I would say the paper is easy to read and well structured. 
- The results looks promising with gains on mathematical datasets which justifies the need for selecting high quality CoT chains.

### Weaknesses
- I think the major drawback in the paper is the experimentation limited to a single model on a single domain. Moreover, the “atomic patterns” are produced by another external model like Deepseek and the prompt in Chinese (as provided in the Appendix) seems quite restricted to work on a specific kind of model with a specific kind of dataset. I am not sure how generalizable this approach can be based on the prompt used, model used and the dataset. Also, it might work because of the similarity of the teacher model used (Deepseek) alongside the student model 85A6B and might not work with different combinations. Also, the RL used is limited to one RL algo of GSPO and model limited to MoEs. 
- Also, some more ablations are needed to justify all the steps in the process used. For example, a sweep of lambda can be used to show the usefulness of entropy, Hungarian also seems a bit extra and what if we replace it with simple kNN, comparing COTP with a random subset of longCoT given its performance, and so on. This seems that the complexity of each step and all the choices made needs to be studied in more detail. 
- Finally, the use of geometric trails is missing from the constraint or the selection process for COTP it seems. Maybe I am missing something but the distance algo is not bounded by the geometric trials in any way and I am wondering if there is any use for geometric trials in itself? Is it more theoretical ?

### Questions
1. I want the authors to shed more light on the sensitivity of lambda? What happens when we change it?
2. Also, I am not sure if Hungarian algo is truly needed here? What if we use kNN and why it will not work?
3. Some more ablations in terms of using a random subset of longCoT vs COTP is needed to understand the true usefulness of COTP. 
4. Any comment on the generalization of the approach beyond Maths, beyond this specific MoE models (maybe to a dense model), beyond mathematical datasets, using a different model to annotate instead of Deepseek, non chinese prompts and so on. 
5. I am also wondering if the contamination test was only done for MathArena or for all the mathematical datasets?
6. Can the approach be extended to other RL recipes beyond GSPO? 
7. I am also wondering if mid-training is the best place for such a recipe? I have seen some papers like SIKeD [1]  where they choose datasets during the fine-tuning stage (so in this case, post-training probably) and that improves the performance of the model. I am wondering if you want to comment on that ? 

[1] SIKeD: https://aclanthology.org/2025.findings-acl.513/

### Soundness
2

### Presentation
3

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
This paper introduces CoTP, a data selection framework that works by expanding the dataset with the COT patterns to increase what the authors define as the model's reasoning potential. The model reasoning potential is defined as the inverse of the number of independent attempts required to correctly answer a question. Using CoTP data the authors are able to improve an MoE model by 9.58% on AIME dataset.

### Strengths
- **Novelty**: The paper presents an original and clearly articulated approach to expanding reasoning potential in foundation models. Both the methodological design and the underlying problem formulation demonstrate genuine novelty.

- **Elegant Solution**: The combination of core-set construction with a dual-granularity selection strategy represents an elegant and well-motivated solution that advances beyond conventional Chain-of-Thought augmentation methods.

- **Theoretical Rigor**: The framework is supported by a strong theoretical foundation and is further reinforced through detailed empirical analysis, illustrating both conceptual robustness and practical effectiveness.
 
- **Comprehensive Empirical Validation**: The work provides extensive experimental results and ablation studies that empirically validate the proposed theoretical framework, showing consistent improvements across mathematical reasoning benchmarks.

- **Reproducibility**: Code was added as supplementary materials, which enhances the transparency and reproducibility of the work.

### Weaknesses
- **Limited Evaluation Scope**: The experiments focus exclusively on mathematical reasoning tasks, leaving uncertainty about whether the proposed framework generalizes to broader domains such as logical, commonsense, or multimodal reasoning.
- **Dependence on DeepSeek-R1**: The framework relies heavily on DeepSeek-R1 for generating Chain-of-Thought (CoT) traces, which is computationally expensive. It would be valuable to examine whether similar improvements can be achieved using smaller or alternative base models.
- **Limited Cross-Model Evaluation**: The study evaluates the method on only a single MoE (mixture-of-experts) model configuration, providing little evidence of robustness across different architectures or model families. Broader testing across diverse model types would strengthen the general applicability claims.

### Questions
The paper mentions a manual screening process for selecting high-value reasoning traces (line 165). Could the authors clarify what this process entails? Specifically, how were examples chosen, and what criteria or safeguards were used to minimize potential bias or subjectivity in the selection?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a mid-training approach using CoT reasoning patterns. Through a process of selecting long-CoT data that is similar to the core set using similarity measures based on pattern & token entropy. The approach has dramatic empirical gains, advancing SOTA significantly in mathematical reasoning.

### Strengths
This paper presents an innovative and detailed approach for a well studied problem.
* **Empirical Results**: Significant gains in the performance of models when using the proposed method of mid training. 
* **Innovative approach**: The proposed algorithm is clever, elegant and interesting to read. It is nice to see such a well motivated method work well empirically.
* **Ablations**: The authors provide a very nice set of ablations that help to complete the understanding of the design space of the given approach, especially Table 3.

### Weaknesses
Mostly I believe this to be a strong paper, which has interesting and high quality results and methodologies.

There are a lot of nice ablations in the paper, but the results that I think may be missing are: connecting the core set similarity to downstream performance? We see only Figure 6 for LongCotPool and CoTP but not the Table 4 ablations? This will help to prove the claim?

### Questions
1. As I understand from line 227, DeepSeek-R1 is the model used for reasoning trace generation. It seems using a variety of datasets is important. Do you think diversity of generative model would be helpful or harmful?

### Soundness
3

### Presentation
3

### Contribution
3
