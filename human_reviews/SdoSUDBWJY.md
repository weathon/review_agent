# Attacking for Inspection and Instruction: Debiasing Self-explaining Text Classification

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 6, 5, 3

## Abstract
eXplainable Artificial Intelligence (XAI) techniques are indispensable for increasing the transparency of deep learning models. Such transparency facilitates a deeper human comprehension of the model's fairness, security, robustness, among other attributes, leading to heightened trust in the model's decisions. An important line of research in the field of NLP involves self-explanation using a cooperative game, where a generator selects a semantically consistent subset of the input as the explanation, and a subsequent predictor makes predictions based on the selected subset. In this paper, we first uncover a potential caveat: such a cooperative game could unintentionally introduce a sampling bias between the explanation and the target prediction label. Specifically, the generator might inadvertently create an incorrect correlation between the selected explanation and the label, even when they are semantically unrelated in the original dataset. Subsequently, we elucidate the origins of this bias using both theoretical analysis and empirical evidence. Our findings suggest a direction for mitigating this bias, and we introduce an adversarial game as a practical solution. Experiments on two widely used real-world benchmarks show  the effectiveness of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the sampling bias problem that arises from the Rationalizing Neural Prediction (RNP) framework for text classifications. The authors demonstrate how sampling bias could be introduced by the explanation generator and how it leads to a bad impact on the label predictor. This paper then proposes to introduce an attacker to inspect the bias and instruct the predictor to prevent the predictor from adopting the bias.

### Strengths
Originality: The sampling bias problem of the RNP framework for the text classification setting is first proposed in this paper. Introducing an attacker to alleviate this bias is also novel and interesting.  
Quality: Most of the statements are made with sufficient mathematical derivations. In addition, the experiment results provide a practical validation of the statements.  
Clarity: The paper has been well written and organized.  
Significance: This paper may have a limited impact on the NLP community. The main reason is that this paper focuses on improving the RNP framework for developing self-explainable text classification systems. Although developing such systems is a hot topic in NLP, the RNP framework is just one of the solutions. Also, the RNP framework cannot be well generalized to large pre-trained models such as BERT/GPT (according to the authors), which are more practical and common in recent academic research and industry products.

### Weaknesses
The potential of this work is significantly suffered from the fact that the RNP framework cannot be practically aligned with large pre-trained models. In addition, if the authors manually label some data from broader topics and more diverse targets and conduct experiments on them, there would be evidence that the selecting bias is common and inherent exists in the RNP framework, and the proposed method could well alleviate it.

### Questions
1.	By providing more high-quality rationales, the predictions should be more accurate. However, I found that sometimes, the baseline methods could better identify rationales with A2I, while the accuracy of predicting labels becomes worse. For example, Beer-Appearance-last grouped row, the F1 score of rationales improves from 72.3 to 74.6, but accuracy drops from 90.9 to 89.7. Similar trends could also be observed from Aroma aspect, for the sixth grouped row, where F1 score improves from 68.4 to 71.2, but accuracy drops from 90.5 to 89.7. Also, in Hotel-Cleanliness, F1 improves from 38.7 to 39.4, while accuracy drops from 96.0 to 95.5. Could the authors provide some insights into this phenomenon?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors propose a method to address the problem of rationalizing neural prediction. The goal is to use a generator to identify confounding tokens within a language classification task.

Specifically, the authors focus on a binary classification task that assigns a label to a sequence of word tokens. They acknowledge that among these features, only some are causal variables, while others are spurious. The solution is to train a generator that produces a mask to exclude spurious variables. In their work, the authors introduce an adversarial module that learns to select tokens such that they cause a trained predictor to reverse its label, thereby rendering the tokens invariant to the label.

The authors have compared their method with existing studies in the same domain and demonstrated its effectiveness.

### Strengths
- Overall this paper is well written, and the author delivered their method pretty clearly. 

- The motivation of providing explainable instruction for reasonable about natural language is good, especially for the current era of large language models.

### Weaknesses
One of the concerns regarding this work is its contribution; the proposed method seems to be merely an add-on to a specific type of problem. The conclusions drawn from binary classification may not easily generalize to a multi-class setting. In binary classification, selecting the reversed label represents a clear worst-case scenario. However, it is not clear how this approach could extend to multi-class cases. In the appendix, the authors provide formulations for multi-class scenarios in equations (17) and (18). I encourage the authors to deliberate on the specific method for "choosing the $Y' \neq Y$—should this $Y$
  be sampled from a uniform distribution, for instance? Moreover, the authors' method assumes the availability of a balanced dataset. How would the algorithm be modified in the presence of imbalanced labels?

### Questions
Reasoning from language could be complicated, there could be more structured knowledge in one sentence beyond confounding information. Sometimes the same words may imply opposite meanings under different contexts. I am wondering how the authors are going to address these more complicated problems in NLP.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses the task of eXplainable Artificial Intelligence (XAI) where the goal is to increase the transparency of deep learning models to enhance trust in their decisions regarding fairness, security, and robustness. It explores a self-explaining framework called Rationalizing Neural Predictions (RNP) used in NLP models, which employs a cooperative game involving a generator and a predictor. 

It identifies a potential sampling bias issue in the RNP framework, where the generator might select semantically unrelated trivial patterns as explanations, leading to implausible explanations. The paper proposes an adversarial game-based approach to inspect and identify this bias, and introduces a method to instruct the game to debias the predictor by penalizing it when it learns from the bias. 

Experimental results demonstrate the existence of sampling bias and the effectiveness of the inspection and instruction methods, which are model-agnostic and improve the performance of self-explaining models.

### Strengths
- Originality: the paper proposes an interesting strategy to identify the bias problem within self-explanation, and introduces an efficient combination of an adversarial approach, and the development of an instruction objective for mitigating bias. 

- Quality: the paper runs a decent set of experiments to evaluate their method against existing ones.

- Clarity: the presentation of the methodology, the pipeline and the experimental results are well-structured and easy to follow.

- Significance: contributions made to self-explaining rationalization and interpretable machine learning have wild impact in the literature, and the fact the authors achieved good results with their method is significant

### Weaknesses
- No code to verify the results

-  While the paper discusses the theoretical aspects of sampling bias and introduces a solution for self-explanation, it falls short in discussing the real-world implications of this bias in AI applications. Providing concrete examples or case studies of how sampling bias can impact decision-making systems would enhance the paper's practical relevance.

- The datasets BeerAdvocate and HotelReview seem small and basic. I am curious to see how this method performs on larger scale datasets like CIFAR10

- No reporting of the mean and standard deviation of multiple experiments to see if the results are significant

### Questions
Please address the weaknesses above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents an interesting approach to addressing the issue of sampling bias in self-explaining text classification models. The authors propose a method called Attack to Inspection and Instruction (A2I), which uses an adversarial game to inspect and correct the predictor's behavior in a Rationalizing Neural Predictions (RNP) framework. The paper is well-motivated, and the problem of sampling bias in self-explaining models is a relevant and important one in the field of explainable AI.

### Strengths
1. The paper addresses a significant problem in the field of explainable AI, which is the potential for sampling bias to lead to incorrect correlations between selected explanations and labels.
2. The authors provide a thorough theoretical motivation for their approach, explaining how the adversarial game can detect and mitigate sampling bias.
3. The experiments conducted on two real-world benchmarks demonstrate the effectiveness of the proposed method, with significant improvements in rationale quality over the baseline RNP model and other advanced methods.

### Weaknesses
1. The paper primarily focuses on binary classification tasks, and it is not clear how well the proposed method would generalize to multi-class classification or other types of machine learning tasks.
2. While the authors mention that the proposed method is model-agnostic, the experiments are limited to the RNP framework and its variants. It would be beneficial to see the method applied to other types of models to assess its generalizability.
3. The paper could benefit from a more detailed discussion on the limitations of the proposed method, including potential scenarios where the adversarial game might fail to detect certain types of biases or where the instruction phase might not effectively debias the predictor.
4. The use of GRUs and GloVe embeddings, while understandable for comparison purposes, may not reflect the current state-of-the-art in NLP, where transformer-based models like BERT are prevalent. It would be interesting to see how the proposed method performs with such models.

### Questions
None

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors study the problem of self-explainable models through the lens of the Rationalizing Neural Predictions (RNP) framework, where a generator which selects a subset of the input sequence is trained jointly with the predictor to produce an extractive rationale. Here, they look to explain and tackle one particular problem with RNP -- that it may degenerate into the generator selecting a special semantically unmeaningful token, which the predictor learns to associate with a particular label. The authors first theoretically study the problem through the perspective of sampling bias. Then, they propose a method based on training an attacker which tries to produce a justification for the opposite label, and regularizing the generator with this justification. They show that their method outperforms the baselines on typical RNP datasets.

### Strengths
- The authors beat the baselines on typical RNP datasets.
- The proposed method is intuitive at a high level.

### Weaknesses
1. The authors do not sufficiently show that this degeneration is an issue empirically in my opinion. To start, the authors should show a few real examples where vanilla RNP gives a nonsense justification while the predictor still outputs the correct label; and show that RNP + A2I fixes these cases. In addition, the authors could consider plotting a histogram of the length of the rationale (for RNP and RNP+A2I), and showing that samples with short justifications correspond to degenerate cases (e.g. the punctuation example). Overall, the sparsity of the A2I augmented models (in Table 1) do not seem significantly different from the sparsity of the base models, and so I am not convinced that A2I solves the issue presented.

2. The proposed method makes sense, but there are several much simpler solutions that the authors should try and compare with. First, it seems to me that the root cause of the problem is that the generator is overpowered -- it's able to internally detect the label, and then feed special tokens correlated with the label to the predictor which do not have semantic meaning. As such, some simple solutions would be to reduce the capacity of the generator; add regularization to the generator; or to train the generator and predictor in an alternating fashion, with more steps for the predictor. The final suggestion is similar to how GANs are trained. In addition, the problem examined is very similar to mode collapse in GANs, and some of the solutions there (e.g. a diversity regularizer [1]) could work as well.


3. There are a few edge cases which I am not convinced that A2I will be able to fix. Primarily, these deal with the circumstance where, in the toy example, the $t_+$ do not appear in the negative samples, and vice versa -- so $t_+$ is a token that appears almost exclusively in positive examples, and $t_-$ is a token that appears almost exclusively in negative examples. Such spurious correlations have been found in natural language tasks [3-4]. In these cases, it seems like the attacker would not be able to choose the corresponding token, and would thus still output random noise. 

4. Another concern deals with the singular sentiment assumption. This seems like a strong assumption that is very dataset and task specific, and the authors already discuss its failure modes in the appendices. The presence of negation seems to be another case where the assumption would be violated. As such, I am not convinced in the generalizability of the method to other datasets and tasks.  Regardless, the authors should formulate this assumption mathematically in the text.


5. Overall, the clarity of the paper could be improved. Some of the formulation sections are hard to parse. For example, the authors formulate the problem as one of sampling bias, which makes sense intuitively. However, the mathematical formulation and causal graphs for this section don't follow the prior work in sampling bias [2].


6. The utility of the method is limited in the era of large pre-trained LLMs, which would achieve very high _zero-shot_ accuracy on all of the sentiment tasks evaluated, likely even higher than the GloVe + GRU networks studied in the paper. Such LLMs also have the capability of explaining its own reasoning (as the authors have referenced). To improve the significance of the method, the authors should consider applying their method to finetune a large-scale LLM (though the authors mention that even finetuning BERT is challenging for RNP). They could also consider applying it to images and graphs, as described in the introduction.

7. The authors do not show any confidence intervals for their results, so it is unclear whether performance gains are statistically significant. They also only evaluate on two datasets, though these seem to be standard datasets in the RNP community.


[1] Diversity-Sensitive Conditional Generative Adversarial Networks. ICLR 2019.

[2] Controlling Selection Bias in Causal Inference. AISTATS 2012.

[3] An empirical study on robustness to spurious correlations using pre-trained language models. TACL 2020.

[4] On Feature Learning in the Presence of Spurious Correlations. NeurIPS 2022.

### Questions
Please address the weaknesses above.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
