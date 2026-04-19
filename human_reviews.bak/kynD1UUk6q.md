# Collapsed Language Models Promote Fairness

- Decision: Accept (Poster)
- Scores: 8, 5, 8, 6

## Abstract
To mitigate societal biases implicitly encoded in recent successful pretrained language models, a diverse array of approaches have been proposed to encourage model fairness, focusing on prompting, data augmentation, regularized fine-tuning, and more. Despite the development, it is nontrivial to reach a principled understanding of fairness and an effective algorithm that can consistently debias language models. In this work, by rigorous evaluations of Neural Collapse -- a learning phenomenon happen in last-layer representations and classifiers in deep networks -- on fairness-related words, we find that debiased language models exhibit collapsed alignment between token representations and word embeddings. More importantly, this observation inspires us to design a principled fine-tuning method that can effectively improve fairness in a wide range of debiasing methods, while still preserving the performance of language models on standard natural language understanding tasks. We attach our code at https://github.com/Xujxyang/Fairness-NC-main

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the relationship between the neural collapse phenomena and fairness in language models. The paper demonstrates that debiased language models naturally exhibit more "collapsed" representations for fairness-sensitive words, and leverage this insight to develop a neural collapse-based regularization method. The proposed approach enhances fairness across different debiasing methods while maintaining model performance on general language tasks.

### Strengths
1.the paper draws the connection between neural collapse theory and fairness in language models in a principal way. It provides novel analytical tools for understanding debiasing mechanisms and offers new insights into how representation geometry relates to model fairness. 

2.  Comprehensive empirical experiments across multiple models and datasets, demonstrating the practical utility of the proposed approach.

3. The proposed regularization method seems to be simple yet effective. It can be easily integrated with existing debiasing approaches without significant computational overhead, making it highly practical for real-world applications. 

3. The empirical evaluation is thorough, covering both intrinsic and extrinsic fairness metrics. The results clearly demonstrate improvements across various benchmarks while preserving model performance on downstream tasks. 

5. The authors also provide helpful visualizations and analyses that illuminate how their method affects model representations.

### Weaknesses
1. The primary weakness of the paper lies in its theoretical foundation. While the empirical results are promising, the authors don't provide sufficient theoretical justification for why NC3 and calibrated metrics work better than others. The paper would benefit from a formal analysis or proof connecting neural collapse to fairness improvements. 

2. The scope of the evaluation is somewhat limited, focusing primarily on **binary** gender bias. While this allows for deep analysis in one area, it leaves questions about the method's generalizability to other (possibly non-binary) demographic groups.

### Questions
1. Can this approach generalize to non-binary protected attributes? If so, how?

2. How does the method work for decoder-only models, like gpt and llama?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a method to enhance model fairness based on the phenomenon of Neural Collapse (NC). The authors first analyze NC in the context of fairness-related words and find that debiased language models exhibit more aligned token representations with word embeddings. Inspired by this observation, they introduce a regularization method based on NC to reduce bias in gender-sensitive words during fine-tuning. The experiments demonstrate that this method can improve fairness across various debiasing algorithms while maintaining the language model's performance on standard natural language understanding tasks.

### Strengths
The paper is logically structured and easy to follow, with a clear progression from problem statement to method proposal and experimental validation.

### Weaknesses
1. In Section 4, the details of the datasets and metrics can be more concisely presented, with some details moved to the appendix. Conversely, the discussion of results is somewhat inadequate. It would be beneficial to highlight the advantages of the proposed method in comparison to the baselines.
2. Additional analysis should be included in section 4.2. From the results alone, it is observed that many outcomes with the addition of (U)NC3 have significantly decreased.
3. It is unclear whether the proposed method is also effective for LLMs.
4. It is uncertain whether the method remains effective for aspects of fairness beyond gender

### Questions
Clarifications Needed: 
1. The σ symbol on line 161. 
2. On line 235, the authors hypothesize that "noises in language data and the complexity of different fine-tuning methods make measurements of NC1/2/4 unstable and inconsistent." Could the authors elaborate on why they consider these factors to be influential? 
3. Could the authors justify the rationale behind replacing class means with classifier weights in NC1/2?
4. Besides gender-related words, have the authors explored other words that might cause the model to collapse more? This experiment would complement Section 3.3.
5. I am curious about the effectiveness of the proposed method in fairness scenarios beyond gender. Firstly, what if it is challenging to obtain relevant word lists? Secondly, will the model’s general performance being affected by other word lists? I think the authors could counduct experiments by adding random words/high frequency words to the word list.
6. Can the authors provide further insights into the results of the visualization in Section 5? Why do they consider this to be a better feature distribution?

### Soundness
2

### Presentation
2

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
### Summary

- The paper studies the relationship between neural collapse and fairness and tries to answer two research questions
	- "Do debiased LMs exhibit greater neural collapse"
	- "Can this inductive bias be leveraged to improve fairness"
- Background on neural collapse:
	- This is a phenomenon that happens in the last layer representations of classifiers.
	- Since the last layer of LM predicts the next word, it can be viewed as a classifier
	- The phenomenon of neural collapse causes the representations of certain gender sensitive words to collapse closer to their class means in debiased LMs
- The paper presents 4 ways to quantify neural collapse borrowing it from prior work. These metrics measure separability between classes.
	- NC1: intra-class variance over inter class variance
	- NC2: Measures separability from a geometric perspective
	- NC3: Uses dot products
	- NC4: Uses a linear projection

In section 3.2.1 the paper provides a worry about diverging values for various NC metrics but convincingly provides a resolution. This is achieved by calibrating the NC metrics by replacing class means with classifier weights in the expressions.
- The experimental results justify the claim of the presence of neural collapse for gender words.
- The second part of the paper deals with proposing a bias mitigation strategy by explicitly enforcing Neural collapse. To this end, they use the NC3 formulation as an auxiliary objective during fine-tuning.
- Both intrinsic and extrinsic metrics show improvements with this modified objective.

Please consider citing the following additional work in the field of LM fairness and evaluation of bias in LMs
- https://arxiv.org/pdf/2208.01448#page=14.61
- https://aclanthology.org/2022.findings-acl.55.pdf
- https://aclanthology.org/2022.acl-long.401.pdf
 
nits
- 3.2.2 CALIBRATIONS -> CALIBRATION
- Section 4.1.3
	- TPR-1 (Type 1: pro-stereotypical minus anti-stereotypical) and TPR-2 (Type 2: pro-stereotypical minus anti-stereotypical) -> TPR-1 (Type 1: pro-stereotypical minus anti-stereotypical) and TPR-2 (Type 2: pro-stereotypical minus anti-stereotypical) reads a bit odd. I think Type 1 are ambiguous sentences and Type 2 are non-ambiguous sentences. Explicitly call this out with a simple example.

### Strengths
### Strengths

- The paper studies a missing link between neural collapse and fairness and proposes a bias mitigation strategy based on neural collapse objective that is agnostic to pre-training or fine-tuning methods. This also avoids manual data balancing or filtering.
- Both intrinsic and extrinsic fairness metrics see improvements with this approach.

### Weaknesses
### Weakness

- Table 9 does not have the best results bold which makes it reading a little harder. 
- While the paper is well-written, simple to follow and proposes a simple technique which works, all the experiments conducted are on BERT based MLM based models. I understand this is intentional to ensure fair comparison with prior work but this also limits understanding the degradation on non MLM tasks. For eg. in Table 9, the results reported are on the GLUE benchmark which are all multiple choice questions. I am curious if this technique can really prove to be a generic technique even for generation tasks.

### Questions
No questions, see weakness section above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work explores fairness in language models from the perspective of Neural Collapse (NC), a phenomenon observed during the final stages of training neural networks on classification tasks (thus can also be applied to language modeling), where the variability and alignment of the penultimate layer and the final classifier reach a structured state. This work further applies NC to debiased LMs and demonstrates that  $\mathcal{NC}3$, which measures the alignment between token representations (contextual class means) and word embeddings (classifiers), consistently improved with debiasing. The author argues that other NC perspectives are less evident in debiased LMs due to common training configurations that violate NC’s typical conditions—such as vocabulary sizes exceeding embedding dimensions and imbalanced token distributions. To address this, they propose calibrated versions of  $\mathcal{NC}1$ and  $\mathcal{NC}2$ by replacing class means with classifier weights. Debiased LM shows significant collapse on these two metrics, and fairness-sensitive words demonstrate more collapse.

Based on the empirical analysis, the authors suggest using  $\mathcal{NC}3$ as a regularization term during fine-tuning to enhance fairness in LMs. Experiments on four fairness datasets demonstrate consistent gains on fairness metrics over three prior debiasing approaches and standard masked language modeling and also doesn't hurt the model's NLU ability.

### Strengths
1. Analyzing the debiased language models from the perspective of neural collapse is novel. This offers valuable insights into the structure of the model's embedding space across different debiasing approaches.
2. The proposed method is simple but effective and can be easily integrated with other approaches. It's interesting to see that by only adding a NC-based regularization term in the loss function, all the compared debiasing methods can be enhanced.
3. The experiments are comprehensive and well-motivated, with the proposed method grounded in observations of $\mathcal{NC}$ perspectives on both biased and debiased LMs, effectively establishing a strong rationale for the proposed method, making it clear and convincing.

### Weaknesses
1. **Clarify of Notations**: Some notations are not clearly defined, which could impact readability.
    - The class embedding variances mentioned on line 161 and line 240 could benefit from a more explicit definition, similar to how it is presented in Eq. 3 of the Linguistic Collapse[1] paper. 
    - In Eq. 1, to align with the token representation defined at line 137, the notation should ideally be ${h}(E(x_{1:t}))$ rather than ${h}(x_{1:t})$ for clarity and consistency. 
2. **Evaluation Metrics**: I have concerns regarding the evaluation metrics used in section 3.3 and section 4.2.1. Please see question 2 and 3 in the following section.
3. **Definition of Unfairness**: Some definition of unfairness were not explicitly defined, which could make the paper less accessible to readers unfamiliar with the fairness literature. For example, in the *BEC-Pro* subsection of section 4.1.2, the average association score is used to measure model unfairness. While this is defined in [2] as $\log \frac{P_T}{P_\text{prior}}$, it's not explicitly explained here, which may cause confusion.

[1] Wu, Robert, and Vardan Papyan. "Linguistic Collapse: Neural Collapse in (Large) Language Models." arXiv preprint arXiv:2405.17767 (2024).    
[2] Bartl, Marion, Malvina Nissim, and Albert Gatt. "Unmasking contextual stereotypes: Measuring and mitigating BERT's gender bias." arXiv preprint arXiv:2010.14534 (2020).

### Questions
1. In section 3.2.2, the authors present a calibrated version of $\mathcal{NC}1$ and  $\mathcal{NC}2$. While I can understand that the main reason to replace class means with classifier weights is because the authors find that $\mathcal{NC}3$ (solely derived from classifier weights) is significant on the debiased models, could this modification compromise the original physical meaning of these metrics Specifically, 
    - $\mathcal{NC}1$, as proposed in [1], measures the variability via an inverse signal-to-noise ratio. However, in the calibrated $\mathcal{N}\mathcal{C}_{1}^{(w)}$, the numerator is the variance of the _class means_, while the denominator measures the distance of the _classifier weight_, which seems to lack physical coherence.
    - Why is $w_c$ not normalized in $\mathcal{(G)NC_2^{(w)}}$ while $\mu_c$ is normalized in $\mathcal{(G)NC_2}$ to measure uniformity?
    - I hope the authors can elaborate more on the reason they choose these calibrated metrics.
2. In section 3.3, the authors argue that _"debiased BERT models exhibit more different token representations and word embeddings only on gender-related vocabulary"_ based on __numerical gaps__ in $\mathcal{NC}$ metrics. However, the baseline level of the base BERT model differ across gender-related words, full vocabulary, and a randomly selected subset. By looking at the __ratios__ of the gap w.r.t. base BERT model's performance under $\mathcal{NC_1^{(w)}}$, I cannot tell there are significant differences among Table 1-3. Similarly, with $\mathcal{(U)NC_3}$, for the gap of ASE and BEC, it's still hard to tell the difference among these three tables.
3. In section 4.1.2, could the authors clarify why the difference in association scores between female and male groups is measured? If the association score is defined as $\log \frac{P_T}{P_\text{prior}}$, I feel it's more reasonable to measure the absolute value of the association score since if the absolute value is close to 0, it should suggest that <person word> predictions are less likely influenced by <profession>, which indicates a less biased LM.
4. All experiments in this paper are conducted on masked language models (e.g, BERT and RoBERTa). Could the proposed method also generalize to casual language models, as explored in the Linguistic Collapse[1] paper?

[1] Wu, Robert, and Vardan Papyan. "Linguistic Collapse: Neural Collapse in (Large) Language Models." arXiv preprint arXiv:2405.17767 (2024).

### Soundness
2

### Presentation
3

### Contribution
3
