# Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspective

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 3

## Abstract
Synthetic data has become a pivotal resource in post-training tasks for large language models (LLMs) due to the scarcity of high-quality, specific data. While various methods have been developed to generate synthetic data, there remains a discernible gap between the practical effects of synthetic data and our theoretical comprehension. To address this challenge, we commence by presenting a detailed modeling of the prevalent synthetic data generation process. Building upon this modeling, we demonstrate that the generalization capability of the post-trained model is critically determined by the information gain derived from the generative model, as analyzed from a novel reverse-bottleneck perspective. Moreover, we introduce the concept of Generalization Gain via Mutual Information (GGMI) and elucidate the relationship between generalization gain and information gain. This analysis serves as a theoretical foundation for synthetic data generation and further highlights its connection with the generalization capability of post-trained models, offering an understanding about the design of synthetic data generation techniques and the optimization of the post-training process. We open-source our code at https://github.com/ZyGan1999/Towards-a-Theoretical-Understanding-of-Synthetic-Data-in-LLM-Post-Training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the use of synthetic data in post-training tasks for large language models. It presents a theoretical framework to analyze the effects of synthetic data on model generalization, focusing on the reverse-bottleneck effect. The authors introduce key metrics like Generalization Gain via Mutual Information (GGMI) and propose a detailed modeling of the synthetic data generation process. They also provide upper bounds on generalization error based on information theory.

### Strengths
1. The paper provides a detailed theoretical analysis of the introduction of synthetic data, clearly explaining the information gain that synthetic data can bring.
2. The authors present thorough and well-supported mathematical proofs to substantiate their claims, offering a solid foundation for their arguments.
3. The reverse-bottleneck theory mathematically captures the essence of synthetic data's impact, offering valuable insights and guidance for the development of synthetic data methodologies.

### Weaknesses
1. The paper primarily focuses on theoretical analysis with minimal experimental validation. It is recommended to include empirical experiments to support the proposed theory.
2. The use of GMM models and non-real numerical data raises questions about whether the findings can effectively reflect real-world LLM outcomes.
3. The theory mainly emphasizes the performance benefits of introducing synthetic data, but lacks analysis in other areas. For example, it could explore potential drawbacks of over-reliance on synthetic data in LLMs, as highlighted in studies like https://www.nature.com/articles/s41586-024-07566-y.

### Questions
1. How would synthetic data that has not been processed by an LLM impact the proposed theory? For instance, data generated through specific rules, such as template-based concatenation or converting game-playing sequences into text.
2. Does this theoretical analysis still hold for non-text, multimodal data? Would the same principles apply, or are there limitations in extending the theory to such data types?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work provides a theoretical framework to analyse the nature and impact of synthetic data on the post-training process of LLMs. This work also proposes a "reverse-bottleneck" framework to visualize information flow in the synthetic generation process. On the basis of these theoretical principles, this work proceeds to mathematically derive useful upper bounds for model generalization when trained with synthetic data, in order to better quantify performance expectations.

### Strengths
- This work has a sound, well-researched theoretical basis
- The provided mathematical proofs are comprehensive and well-structured.
- The experimental setup is well-described and the theory is supported by some experimental results.

### Weaknesses
- This work could benefit from additional explanation of how diversity of generated data influences generalization capabilities, as well as a clearer explanation of the intuition behind how ∆I (the information gain derived from the model M) is associated with diversity (lines 483-485) and not just more unique but similar data points, possibly with examples.
-  The dimension of data on which these experiments are performed is 2; however, real-world data is often more complex and higher-dimensional. It would be illustrative to conduct the experiments documented in this paper on data of higher dimensions to reinforce that the theory holds for datasets of varying dimensionality.
- Additionally, given that this analysis is aimed at studying the behaviour of LLMs, the theoretical proofs presented in this work would benefit from conducting the same experiments either with LLMs or other non-GMM models to corroborate the theoretical inferences.

### Questions
- How does task difficulty affect generalization capability and the derived upper bounds? Are there any factors in the documented theoretical expressions that model or are associated with task complexity?
- Lemma 4.8: η' and W' are not defined prior to their usage in the expression; for clarity, it would be useful to define them as other variables have been.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper does a good job of establishing a theoretical rationale behind using synthetic data for LLM post-training processes like SFT. The modeling of synthetic data, focusing on distributional aspects is elaborate and properly justified as well. It establishes an intuitive information-theoretic based upper bound on the generalization error for an under-aligned LLM fine-tuned on synthetic data, and also delineates how this upper bound compares to the one when the LLM is tuned on just the real world samples (anchor data) - showcasing how the former proves to be a more stringent upper bound, leading to better generalization capabilities. Definitely helps bridge a gaping gap in LLM research!

### Strengths
1. The theoretical modeling of synthetic data and how it interacts with the output distribution of the LLM is intuitive and complete. Great job there!
2. The math behind generalization error bounds is solid.
3. The connection to the classical information bottleneck theory is novel and very intuitive, clearly justifies why synthetic data helps better performance on downstream task generalization.

### Weaknesses
1. The paper presents good theoretical foundations of "why" it is important to use synthetic data in post-training alignment and "how" it aids better generalization capabilities of the model. But it fails to address some really important "hows" that it promises in the Introduction section :
   - How does this study help in developing tailored synthetic data generation - that more effectively address specific gaps in training data, thereby enhancing the overall performance and generalization capabilities of large language models
   - How does this translate to better post-training practices involving synthetic datasets, from a practical perspective.
2. The assumption that the transformation function φT is reversible (Section 3.2) is unclear for real LLM prompting scenarios, as task-to-prompt relationships are often many-to-many.A better delineation of this point would be helpful.
3. ⁠The relationship between "synthetic factors" and actual LLM generation processes is not well defined.
4. The connection between the theoretical "information gain" concept and practical improvements in LLM performance is not clearly established. 
5. I think its very crucial to address how the "diversity" and "quality" in synthetic data samples are a big part of why task generalization actually works when models are post trained on synthetic data. These should be adequately quantified and incorporated in the upper bound equations as well, for a more complete picture.

### Questions
Here are suggested experiments that could help validate the paper's hypotheses and theoretical frameworks:
1. LLM-based Validation Experiments :
     - ⁠Compare different sizes of anchor data and their corresponding synthetic data generation
     - ⁠Measure the relationship between model size and synthetic data quality
     - ⁠Test various prompting strategies and their impact on synthetic data diversity
     - Track information gain across different LLM architectures 
2. Practical Application Questions:
    - How can practitioners use your theoretical bounds to improve their synthetic data generation process?
    - What specific guidance would you give for prompt engineering based on your theoretical findings?
3. Validation Questions:
    - How would your bounds change with different LLM architectures or sizes?
    -  ⁠What metrics would you recommend for measuring the quality of synthetic data in practice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces the framework of understanding the LLM synthetic data generation process, by incorporating the reverse-bottleneck perspective. Using a reverse-bottleneck perspective, the authors model the generation of synthetic data and propose an information-theoretic metric called Generalization Gain via Mutual Information (GGMI) to assess the generalization performance of post-trained models. It also reports the information gain and generalization bound from previous literature.

### Strengths
1. The motivation is easy to follow and the figure is well represented.

2. The equation derived from previous literature, such as Lemma 3.1 and 4.4 is clear. 

3. The paper tries to inject the LLM synthetic generation perspective using previously derived framework, showing a potential direction to understanding the synthetic data.

### Weaknesses
1. The paper lacks the experimental support, which cannot convince the readers of their assumption and derivations.

2. The paper is not organized in a good shape for guiding the readers for understanding their contributions. Instead, it enumerates the equations of generalization errors and gains, but lacking sufficient experimental results to support this.

3. Some derivations may be challenging for readers unfamiliar with advanced information theory, which may requires providing backgrounds about this. Since the authors did not conduct real-world experiments, it is unsure whether this can generalize to real-world settings. The framework assumes ideal conditions for synthetic data and LLMs, which may not generalize well to complex real-world scenarios with noisy data or biases.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2
