# Law of Vision Representation in MLLMs

- Decision: Reject
- Scores: 5, 6, 6, 5

## Abstract
We present the "Law of Vision Representation" in multimodal large language models (MLLMs). It reveals a strong correlation between the combination of cross-modal alignment, correspondence in vision representation, and MLLM performance. We quantify the two factors using the cross-modal Alignment and Correspondence score (AC score). Through extensive experiments involving thirteen different vision representation settings and evaluations across eight benchmarks, we find that the AC score is linearly correlated to model performance. By leveraging this relationship, we are able to identify and train the optimal vision representation only, which does not require finetuning the language model every time, resulting in a 99.7% reduction in computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose the AC score, which combines cross-modal alignment and correspondence in vision representation. Then, they find a linear correlation between the AC score and MLLMs' performance. Extensive experiments across different vision representations validate the linear relationship between the AC score and MLLMs' performance, with an R-squared value of 95.72%.

### Strengths
* The motivation is sound. Building the relationship between vision representation and MLLMs' performance is very interesting and helpful.
* By identifying an optimal vision representation without extensive fine-tuning, the paper contributes to reducing computational costs and makes MLLM training more efficient.

### Weaknesses
* Cross-modal alignment score is the maximum cosine similarity between each pair of embedding vectors from the CLIP embedding and the target embedding. However, there may be bias in CLIP characterization. At the same time, CLIP is also one of the vision representations involved in the test, which results in the CLIP representation having an artificially high score on the cross-modal alignment score. 
* The authors only tested the proposed method on LLaVA v1.5. However, I am concerned that the vision representation law will change after changing the MLLM structure or the base LLMs. Therefore, I strongly recommend the authors to verify the generalization of the proposed method on more MLLM structures or the base LLMs.
* To fairly compare different vision representations, it is usually necessary to control the computation cost (e.g., flops) and model parameters of different vision representations within a similar level, but the proposed method does not take this into account. I suggest providing the computational cost and parameter counts for the different vision representations used and discussing how these factors might impact the AC scores.

### Questions
* How would using a stronger contrastive learning model, like BLIP, or a weaker model affect the results when calculating cross-modal alignment score?
* Does calculating the correspondence score require training an MLP to predict key points? If so, different vision representations may require different training hyper-parameters. Consequently, what approach should be taken to determine these optimal hyper-parameters? If the representation dimensions of the vision representation are not the same, then how can they be fairly compared?
* Why does the AC score not exhibit strong performance on the TextVQA and VizWiz benchmarks? Are there any significant differences between these two benchmarks and other benchmarks?
* How is the number of fine-tuning in Tab. 1 determined? Why does the number of fine-tuning vary across different benchmarks?

### Soundness
3

### Presentation
2

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
This paper introduce a AC score that can be used to predict the performance of LVLM with specific vision encoders.

### Strengths
1. AC policy consistently predicts the optimal vision representation with minimal resources.
2. Provide a insight of the choice and design of vision encoder for LVLM.

### Weaknesses
1. The calculation of the A score by using the image feature of CLIP might not fully fulfill the motivation for quantifying the cross-modal alignment, as it assumes that the image-text features are perfectly aligned in CLIP.

### Questions
1. Please clarify the meaning and origin of the dimension 6 in  $X ∈ R^{k×6}$ in Policy Fitting. Is it related to the coefficients of the AC score?
2. Provide a pseudo code for the AC POLICY algorithm to enhance understanding.
3. Explain the choice of Recall@3 as the main metric and supplement with other metrics (like Recall@1, Recall@2, etc.).
4. Present the specific format (contains the weights and bias) of Policy Fitting and analyze the differences in fitting results across benchmarks. 
5. SigLIP is supposed to have better cross-modal alignment as an improved version of CLIP, but it shows a lower score in your setting. Please explain.
6. Does a higher AC score always mean better performance?
7. Besides the A and C scores, are there other factors that need to be considered?

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
In this paper, the authors analyze the relationship between the cross alignment and correspondence (AC) score and MLLM performance. The AC score is a second-degree polynomial transformation of the alignment and correspondence score. Through a few training runs, the AC Policy can be determined and subsequently used to predict the optimal configurations of MLLMs. Thus, the AC Policy can reduce computational costs when deciding the optimal vision representations for MLLMs.

### Strengths
1. The experiments are well-conducted and quite comprehensive.
2. The analyses of A, C, and AC scores, as well as their correlation with model performances are quite clear and valuable, highlighting the effectiveness of the proposed AC score.

### Weaknesses
1. How should we choose the number of $k'$ for Policy fitting? In Table 1, each benchmark has its own number of finetuning runs required to successfully predict the optimal vision representation. While 3 is the most common number across the benchmarks, there are still some outliers. It is difficult to determine the exact number of $k'$ for a benchmark without having the groundtruth. This suggests that the AC policy may not be very practical.
2. All the experiments are conducted under the condition of freezing the Vision Encoders. The conclusions may change when the Vision Encoders are unfrozen, which limits the generalizability of the AC Policy and these conclusions.
3. It is confusing that “if the vision representation ensures accurate correspondence, more visual details are considered even if not attended by the text token”. This statement implies that a high correspondence vision representation preserves visual details, which is ungrounded.

### Questions
1. In section 3.2, the explanation of key points and their prediction process is missing.
2. In Table 1, could you provide the top-1/2 predictions for a more comprehensive analysis?
3. Why does the 13 setting only require 12 finetuning runs to predict the optimal vision representations in a random setting?

### Soundness
3

### Presentation
2

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
The paper presents the Law of Vision Representation in multimodal large language models (MLLMs), aiming to establish a strong relationship between vision representation quality, cross-modal alignment, and model performance. The authors introduce the AC score to quantify cross-modal Alignment and Correspondence, demonstrating a linear correlation between AC scores and MLLM performance across multiple benchmarks. The AC policy is also proposed as a method to efficiently select optimal vision representations, potentially reducing computational costs and energy consumption significantly compared to traditional methods.

### Strengths
1. The paper identifies the limitations in current methods for choosing vision representations, highlighting the need for a deeper understanding rather than an empirical, trial-based approach.
2. The proposal of the Law of Vision Representation is an innovative concept that attempts to quantify the relationship between cross-modal alignment, correspondence in vision representation, and MLLM performance through the AC score. 
3. The paper highlights an approach that achieves computational efficiency by reducing the need for finetuning with each vision representation change. This leads to a substantial reduction in computational cost.

### Weaknesses
While this work raises an important question in MLLM, there are certain weaknesses that prevent me from assigning a higher score.
1. The experiments in this paper are conducted on a specific MLLM setup, with a particular LLM and projector configuration, which raises doubts about whether the conclusions can be applied if the LLM or projector settings are changed. Given that the paper aims to establish a "law of vision representation," the current experiments seem too limited in scope to substantiate such a universal law.  Additional experiments should be carried out using various LLM sizes and projector designs to showcase wider applicability.
2. The choice of linear regression for modeling the relationship between the AC score and MLLM performance lacks sufficient explanation. It is unclear why a linear regression model is specifically used, and there is no discussion on whether other types of regression (e.g., polynomial, logarithmic) could potentially better capture this relationship. A justification for the linear assumption or a comparison with other regression models would strengthen the analysis. A comparison of various regression models (such as linear, polynomial, and logarithmic) should be performed, and their respective performance metrics should be reported to support the selection of linear regression.
3. Although the authors provide a theoretical explanation of the impact of the AC score on performance, the paper would be strengthened by a more rigorous proof or formal derivation of the mathematical relationship. Additionally, a more in-depth discussion on the design and application of AC scores, encompassing both explainable and theoretical aspects, should be included.

### Questions
1. Typo in Table 1 'Number of Fintuning' -> 'Number of Finetuning'.
2. The experimental setting for the chosen LLM and projector is unclear. Could you give more information about the experimental setting?

### Soundness
2

### Presentation
2

### Contribution
2
