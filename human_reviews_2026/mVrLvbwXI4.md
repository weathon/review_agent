# Group Fairness Meets the Black Box: Enabling Fair Algorithms on Closed LLMs via Post-Processing

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Instruction fine-tuned large language models (LLMs) enable a simple zero-shot or few-shot prompting paradigm, also known as in-context learning, for building prediction models. This convenience, combined with continued advances in LLM capability, has the potential to drive their adoption across a broad range of domains, including high-stakes applications where group fairness—preventing disparate impacts across demographic groups—is essential. The majority of existing approaches to enforcing group fairness on LLM-based classifiers rely on traditional fair algorithms applied via model fine-tuning or head-tuning on final-layer embeddings, but they are no longer applicable to closed-weight LLMs under the in-context learning setting, which include some of the most capable commercial models today, such as GPT-4, Gemini, and Claude. In this paper, we propose a framework for deriving fair classifiers from closed-weight LLMs via prompting: the LLM is treated as a feature extractor, and features are elicited from its probabilistic predictions (e.g., token log probabilities) using prompts strategically designed for the specified fairness criterion to obtain sufficient statistics for fair classification; a fair algorithm is then applied to these features to train a lightweight fair classifier in a post-hoc manner. Experiments on five datasets, including three tabular ones, demonstrate strong accuracy-fairness tradeoffs for the classifiers derived by our framework from both open-weight and closed-weight LLMs; in particular, our framework is data-efficient and outperforms fair classifiers trained on LLM embeddings (i.e., head-tuning) or from scratch on raw tabular features.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an interesting solution to facilitate fair outcomes from black-box LLMs for classification tasks where you don’t have access to the weights or activations to enable traditional techniques such as fine-tuning or training a classifier on embedding spaces.  The authors tackle this via prompting, treating the log prob outputs as features for probabilistic predictions.  Prompts are designed to compute priors over the output classes and likelihoods of the protected attribute conditioned on the output class, so that the joint probability of the class and protected attribute can be used as a feature for a fair classification algorithm.  The framework is tested against several datasets and popular LLMs both closed-weights and open-weights so that they can compare against algorithms that have access to internal embeddings.  They show that the algorithm does well against a no-mitigation baseline, outperforms algorithms based on embeddings and can beat algorithms trained directly on tabular data under low-data regimes.

### Strengths
The algorithm itself is clearly presented, particularly in figure 1 which does an excellent job of outlining the technique.

For the low data regime, this work presents an interesting solution to mitigating bias in LLM-based classifiers where you don’t have access to the weights or embeddings of the model.  This is a valuable contribution to the community.  

The technique the authors use to extract the prior and conditional likelihoods does a good job of eliciting the inherent bias in the model which they directly correct with a classifier trained on the joint probability computed from these.  

The experimental results are convincing, though ablations are limited.  I’d like to see more discussion on the effect of prompt variations as well as the addition of few-shot and chain-of-thought prompting as mentioned on line 476.

### Weaknesses
The abstract and introduction aren’t as clear as they could be as to what the algorithm is doing.  In lines 024-025 where you mention probabilistic predictions, you could mention that you using prompting to illicit the prior distribution over classes and conditional likelihoods to model the inherent bias in the model, which then form the features for the lightweight fair classifier.

Also, in the abstract you claim that your algorithm outperforms training from scratch on raw tabular features, but this is only true in the low-data regime.  You should change the abstract to reflect this.

The results presented in figure 3 are difficult to interpret.  There is no x-axis (it is just each experimental permutation), there is a mysterious clover leaf in some graphs (presumably to differentiate datasets - consider putting a box around each dataset instead), and there is just too much information in one figure.  I’d like to see these results spread out over a number of figures, each one connecting to a paragraph in section 6.1.

Some of the experimental setup that led to figure 3 is not clear (see question 3 below).

### Questions
1. How sensitive is the algorithm to the prompt phrasing?
2. I’m not sure subtracting the baseline from the AUTC is completely beneficial.  Without knowledge of the baseline, it is hard to tell whether the AUTC is good or not.  It is fine for comparing methods, but what if they are bad?  One case might appear fairer but inherently be lower performing due to a low baseline.  Consider updating the results to make the baseline clear or improving the text to clarify why it is OK to remove the baseline.  
3. How much data did you train your classifiers on for Figure 3?  Was it the low-data regime you mention on line 452?  If so, is that a fair comparison?
4. In the conclusion you mention that you only focus on zero-shot prompting.  You should extend the work to show how you would use few-shot and chain of thought prompting for this work and what the results would be.

### Soundness
3

### Presentation
2

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
This paper addresses the limitation of closed-weight large language models (such as GPT-4, Gemini, Claude, etc.) that cannot perform parameter fine-tuning, and proposes a post-processing-based framework for achieving group fairness classification under the condition of only accessing the model's prediction outputs (such as the log probabilities of tokens).
The authors view LLMs as feature extractors, obtaining the LLM's probability predictions for task labels and fairness-related variables through specific prompt design, and constructing low-dimensional features based on this. Then, they combine existing fairness classification algorithms (such as Reductions, MinDiff, LinearPost, etc.) with the existing LLMs (including GPT-4o) for training lightweight fairness classifiers. The paper conducted experiments on several datasets (Adult & ACSIncome, COMPAS, BiasBios, CivilComments) and four LLMs (including GPT-4o), and the results show that the proposed method achieves better accuracy-fairness trade-off performance in low-data scenarios.

### Strengths
1. The problem background holds significant practical value: The study of fairness for closed-source LLMs is a current issue of considerable practical significance, especially under the realistic constraints where model weights are inaccessible.
2. The framework has universality: The method can be compatible with various fairness algorithms and different fairness definitions, and can be applied to various types of tasks (text and tabular data).
3. The framework has strong universality: The method can be compatible with various fairness algorithms (pre-, in-, post-processing) and different fairness definitions (SP, TPR, EO, FPR, etc.), and can be applied to various types of tasks (text and tabular data).
4. The AUTC indicator was proposed to evaluate the trade-off between fairness and accuracy, which to a certain extent enhanced the quantitative comparability of the results.

### Weaknesses
1. The innovation is limited. The method essentially combines the existing post-processing fairness algorithm with the LLM output. The innovation is limited, especially considering the existing work [1, 2]. 
2. Insufficient analysis depth. The paper mainly focuses on verifying the "feasibility of the post-processing framework", but fails to conduct in-depth theoretical analysis on the mechanism for improving fairness. Besides, there is a lack of statistical significance testing for the improvement extent of fairness indicators.
3. Limited scope of application. The framework is entirely dependent on classification prediction outputs and is not applicable to generative tasks (such as text generation, summarization, fair question answering, etc.). Therefore, the assertion that "it is applicable to closed-source LLMs" should be carefully qualified in terms of scope.
4. The number of models covered in the experiment is relatively small (only 4 LLMs), and most of them are open-source models, which cannot fully verify the applicability of "closed-source LLMs".

[1] Di Gennaro F, Laugel T, Grari V, et al. Post-processing fairness with minimal changes[J]. arXiv preprint arXiv:2408.15096, 2024.
[2] Nguyen D, Gupta S, Rana S, et al. Fairness improvement for black-box classifiers with Gaussian process[J]. Information Sciences, 2021, 576: 542-556.

### Questions
See the above-mentioned weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles group fairness for classifiers built via closed-weight LLMs. The core idea is to treat the LLM as a black-box feature extractor and elicit sufficient statistics for fair post-hoc classification purely from its probabilistic predictions. The predictions of LLMs are calibrated (via logistic regression), featurized as a low-dimensional representation of the joint distribution over (A, Y), and then passed to standard fair algorithms (Reductions, MinDiff, or LinearPost) to train a lightweight fair classifier. The experiments show that the induced classifiers yield competitive or superior accuracy–fairness tradeoffs, especially in low-data regimes.

### Strengths
1. The paper is clearly written, accessible, and easy to follow.
2. It addresses the important and timely issue of achieving group fairness for closed-source LLMs.
3. The experimental evaluation is comprehensive and convincing, spanning five diverse datasets (including multi-class and overlapping-group cases), four LLMs (open and closed), and three fairness algorithms.

### Weaknesses
1. This framework extracts P(A | Y, X) from the LLM, the extracted features inherit the model's own biases towards sensitive attributes. This means that the "sufficient statistics" are not objective measurements, but rather potentially biased surrogate indicators. Therefore, downstream calibration/post-processing may only mitigate measurement discrepancies based on these surrogate indicators without truly correcting for bias.
2. The probability extraction schemes typically require K+1 API calls per sample (one for Y, K for A | Y = k), which becomes extremely expensive and slow for common multi-class classification problems involving hundreds of classes. The limitations of closed APIs (rate limiting) exacerbate this scalability problem, and large-scale training or evaluation becomes impractical without engineering modifications or the adoption of approximation methods.
3. Experimental results show that as the dataset size increases, the performance is surpassed by embedding-based training, indicating an inherent information bottleneck in low-dimensional feature extraction. This suggests that this method is best viewed as a low-resource solution rather than a universally superior alternative, limiting its practical impact when more labeled data is available.

### Questions
1. Can you leverage additional information from LLMs to enrich the low-dimensional probabilistic features and improve performance in high-data regimes—for example, by incorporating chain-of-thought?
2. What is the underlying cause of model performance plateauing as training scale increases?

### Soundness
2

### Presentation
3

### Contribution
2
