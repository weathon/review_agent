# Binary Hypothesis Testing for Softmax Models and Leverage Score Models

- Avg Score: 3.00
- Decision: Reject
- Scores: 3, 3, 3, 3

## Abstract
Softmax distributions are widely used in machine learning, including Large Language Models (LLMs) where the attention unit uses softmax distributions. We abstract the attention unit as the softmax model, where given a vector input, the model produces an output drawn from the softmax distribution (which depends on the vector input). We consider the fundamental problem of binary hypothesis testing in the setting of softmax models. That is, given an unknown softmax model, which is known to be one of the two given softmax models, how many queries are needed to determine which one is the truth? We show that the sample complexity is asymptotically $O(\epsilon^{-2})$ where $\epsilon$ is a certain distance between the parameters of the models.

Furthermore, we draw analogy between the softmax model and the leverage score model, an important tool for algorithm design in linear algebra and graph theory. The leverage score model, on a high level, is a model which, given vector input, produces an output drawn from a distribution dependent on the input. We obtain similar results for the binary hypothesis testing problem for leverage score models.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies binary hypothesis testing, which identifies an unknown distribution given two possible candidates, for the softmax and the leverage score models. 

*Setup and tools*: The analysis focuses on the softmax model commonly used in deep learning and on the leverage score models used in graph theory and linear algebra. The authors conducted their proofs with information-theoretic tools and norm manipulation.

*Contributions*: The authors show that the sample complexity (in terms of queries) of the binary hypothesis testing is $\mathcal{O}(\varepsilon^2)$, where $\varepsilon$ is a distance-based measure between parameters, for both the softmax model and the leverage score models.

### Strengths
- Studying the softmax as a sampling operation is interesting
- The focus on the binary hypothesis testing is original and could have a potential impact on machine learning and deep learning 
- The analysis and the proofs for both models are detailed

### Weaknesses
- The authors state the main problem with respect to LLMs (l053), but they do not explain the theoretical, methodological, or practical implications of their results for LLMs or neural networks. 
- Considering simple models for a proper theoretical study is valid, but in my opinion, studying the softmax operator as a standalone is an oversimplification, especially if the motivation is the use of the softmax in LLMs and neural networks. 
- In the attention mechanism, the softmax matrices depend on the input sequence of tokens ($\mathrm{softmax}(XW_Q^TW_KX/d_{\kappa}$). Hence, I don't think that the current analysis with a fixed matrix $\mathbf{A}$ (see Definitions 2.6/2.7) holds. Could the authors elaborate on this? In particular, what plays the role of $A$ and $x$ in LLMs and transformers?
- Similarly, the authors justify the scaling of the input (Definition 2.8) by the use of batch-norm. However, in most transformers, including LLMs, LayerNorm and its variations are preferred instead of batch normalization. In addition, the product $Ax$ of Definition $2.6$ is rather a $XW_Q^TW_KX$ with learnable $W_Q, W_K$. Again, what plays the role of $A$ and of $x$, and why would it be justified to consider $x$ bounded in such a case? Could the authors elaborate on that matter?
- The writing could be improved: several sentences are unclear to me and there are many repetitions in some paragraphs (e.g., "delve").
- The structure could be improved: currently, it does not highlight the strengths of the paper and makes it look more like a concatenation of results.

### Questions
*Questions related to the weaknesses*
1) What are the practical, methodological, or theoretical implications of the results for LLMs, transformers, and neural networks? 
2) Could the authors think of experiments (even on synthetic data) to illustrate or showcase the importance of their theoretical results? (I feel the need to note that I have nothing against fully theoretical papers, but in its current form, the relevance of the theoretical contributions of this paper is limited and it lacks discussion to fully understand their potential implications in machine learning/deep learning).  
3) The authors mention an analogy between softmax and leverage score models in the abstract. What is the analogy or connection between them, except that both are parameterized by a matrix and take vectors as inputs?
4) In the attention mechanism, the softmax matrices depend on the input sequence of tokens ($\mathrm{softmax}(XW_Q^TW_KX/d_{\kappa}$). Hence, I am not sure the current analysis with a fixed matrix $\mathbf{A}$ (see Definitions 2.6/2.7) holds. Could the authors elaborate on this? In particular, what plays the role of $A$ and $x$ in LLMs and transformers?
5) Similarly, the authors justify the scaling of the input (Definition 2.8) by the use of batch-norm. However, in most transformers, including LLMs, LayerNorm and its variations are preferred instead of batch normalization. In addition, the product $Ax$ of Definition $2.6$ is rather a $XW_Q^TW_KX$ with learnable $W_Q, W_K$. Again, what plays the role of $A$ and of $x$, and why would it be justified to consider $x$ bounded in such a case? Could the authors elaborate on that matter?

*I list below potential typos and sentences I did not understand .*

- l034: missing reference replaced by a "?"
- l043: *"potent capabilities"* --> What does that mean? Do the authors mean "powerful capabilities"?
-l045: *"prevailing prevalence"* --> Is it a typo to have the adjective and noun combined? Do the authors simply mean "prevalence"?
-l047-l048: *"However, their is [...] of the whole"* --> I do not understand the sentence, it seems unfinished.
-l050-l051: *"as well as sparsity [...] above"* --> I do not understand this part of the sentence at all.
- l053: *"can we distinguish [...] sampling"* --> I do not understand the question nor its connection to the paragraph above it (l047-l051). Could the authors elaborate on that?
- l059-l081: 5 repetitions of the word "delve", use of complicated vocabulary (distinguishing ability, intricacies, inquiry, etc.) that hinders the meaning and understanding of the paragraph.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors derive upper bounds and lower bounds on the number of queries needed to distinguish between two different softmax or leverage score model distributions. This is also known as the binary hypothesis testing problem. The proofs require manipulating inequalities between the Hellinger distance and the total variation distance. Additional energy constraints on the input query is also needed for proving the upper bounds of sample complexity.

### Strengths
I’m not an expert in this field, but the lower and upper bounds seem like a novel results for softmax and leverage score models for the binary hypothesis testing problem.

### Weaknesses
- This paper is not properly motivated, at least in how it’s written. The introduction sections touch on LLMs and how attention is an important component of LLMs. In the rest of the paper, the authors go on to prove bounds on number of samples needed for binary hypothesis testing for a generic softmax and leverage score model without circling back to how it relates to LLM. If the research question is understanding the sample complexity of softmax and leverage score model under the binary hypothesis testing problem, which I think it’s a worthwhile research question, the authors shouldn’t motivate it from LLMs without ever relating it to LLMs.
- Similar to the first point, I’m not sure if there are any interesting applications for the provided sample complexity bounds. It’s unclear to me if these bounds are tight or not or do they ever prescribe any meaning quantities in practice. Tentatively, I’m suggesting one experiment the authors can do in an application domain like LLM:
    - Suppose there are two LLMs. Given different input sequences, the next token distribution comes from a softmax over the logit vectors. Then one interesting question could be how many input sequences do we need to tell which LLMs we’re using based on the next token softmax distribution. This could be interesting in the context of knowledge distillation or some sort of toxic speech detection problems. Then the authors can plot the predicted number of samples using the bounds and also the empirical number of samples needed to actually distinguish between the two distributions.

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper analyzes the sample complexity of the binary hypothesis testing problem for softmax distributions and leverage scores, with the motivation of improving theoretical understanding of LLMs. The theoretical contribution of the paper is providing upper and lower bounds on the sample complexity, both for softmax operators and leverage scores, which are obtained by estimating the Hellinger distances between the distributions and then applying them in conjunction with prior results in hypothesis testing literature.

### Strengths
- The theoretical results seem sound. 
- The notations and exposition of the results is clear to follow.

### Weaknesses
- The glaring weakness is that the motivation of the paper is largely unclear and disconnected from the work done in the paper. The motivation of the works is to study LLMs through a better understanding of the softmax attention. On the other hand, the analysis, which is about the sample complexity of distinguishing two softmax distributions in a hypothesis testing problem, seems largely tangential and the results are never connected back to the original motivation or explain how it serves to improve LLM understanding, either in theory or via simulations.
- The exposition on leverage scores seems to have been shoehorned and seems to have no connection to the motivation either, except some vague comments about their usefulnes. I understand that it might be of interest to show the similarities, but the authors need to explain why how this additional perspective connects to the question of interest. 
- Presentation Issues: The paper has some writing issues outside of the theoretical analysis, which makes the paper hard to follow. 
    - The motivation needs clarity, and the chose approach needs to be justified.
        - Line 53: _Can we distinguish different ability parts of large language models by limited parameters sampling?_ I am not sure what the authors mean by "ability parts" or "limited parameters sampling".
        - Line 60-63: I find it hard to follow as to how hypothesis testing and distinguishing one softmax distribution from another, can help improve the theoretical understanding of LLMs or determine which parameters are important for inference. 
   - The related work section is vague; it is not clear which references are relevant to the motivation or techniques used in this work. Similarly, in conclusions and future work, the exposition is largely about solving the hypothesis testing problem and does not substantiate on potential practical applications.
    - Lines 47-48, 56-58 need sentence restructuring.
- Overall, I believe that this work seems to be more hypothesis testing centric, and thus might be a better fit for a different venue both in terms of applicability of the results and audience interest.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies the sample complexity of binary hypothesis testing in the context of softmax and leverage score models. First, the paper identifies the requirement of energy constraints for both problems as otherwise the testing problems are straightforward. Then, the lower and upper sample complexities are identified for the two problems.

### Strengths
The hypothesis testing lens to study properties of softmax models is an interesting idea.

### Weaknesses
1. The write-up needs significant improvements. I find it hard to follow the introduction and the related work section on theoretical LLMs, which include many unrelated works. I am also skeptical of how the authors motivate the problem, focusing purely on LLMs.

2. It seems like the main result depends on the result from Polyanskiy & Wu, and the significance of results on top of these main results is not clear. I believe the key result is Theorem 3.1, which is invoked by controlling Hellinger distance under some perturbations. The upper bound in Theorem 3.1. is directly from Polyanskiy & Wu. So, I believe the main result is the lower bound in Theorem 3.1. Studying the analysis, the proof is straightforward (the contribution is mild) and contains an implicit assumption on the number of queries, $m$, not explained.

3. I believe the example in L261-268 is wrong for the leverage score model. Could you explain why $\delta > 0$ will put all mass on $2 \in [n]$ for $\mathrm{Leverage}_{B}(s)$? In addition, I understand that the problem is straightforward without the energy constraints, but this needs to be justified by motivating the problem in this setting.

### Questions
1. Why do you say "asymptotically" in the abstract?
2. Can you explain the case of $\delta \geq 0.1$ for lower bound in Theorem 3.1.?
3. Why do you assume $m \leq 0.01 \delta^{-2}$? I don't see this assumption in the main body of the paper. Why would this assumption is meaningful in the context of binary hypothesis testing?

### Soundness
1

### Presentation
1

### Contribution
2
