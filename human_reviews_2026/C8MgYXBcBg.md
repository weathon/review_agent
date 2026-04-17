# Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Continual Pre-Training (CPT) is essential for enabling Language Models (LMs) to integrate new factual knowledge without erasing old. 
While classical CPT techniques like data replay have become the standard paradigm, the mechanisms underlying how LMs acquire and retain facts over time, termed as continual Factual Knowledge Acquisition (cFKA), remain unclear. 
In this work, we present a theoretical framework that characterizes the training dynamics of cFKA using a single-layer Transformer with linear attention, offering a unified explanation for the behavior of popular CPT methods. 
Our analysis reveals that regularization-based methods merely adjust the convergence rate of parameters without altering the inherent forgetting tendency, whereas data replay methods shift convergence dynamics and stabilize pretrained knowledge. 
Building on these insights, we propose a novel generative data replay approach, called **S**electing **T**okens via attenti**O**n **C**ontribution (STOC), which identifies influential factual snippets to guide replay generation. 
Extensive experiments on both synthetic and real-world datasets validate our theoretical findings and demonstrate that STOC effectively enhances cFKA by mitigating catastrophic forgetting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper conducts theoretical and empirical analysis of how transformer language models learn and forget facts in a continual knowledge acquisition setting. Theoretically, the paper analyzes a transformer with a single linear attention layer and shows that the parameters converge to values based on the statistical correlations between subject and object tokens. Empirically, the substantiates several predictions with language models trained on real and synthetic data, focusing on how different continual pre-training methods succeed or fail to mitigate forgetting. The paper also proposes a generative replay method motivated by the theoretical analysis, which is shown to mitigate forgetting without reducing continual learning.

### Strengths
- Continual knowledge acquisition is an important problem for language models, and this paper provides theoretical and empirical insights that could lead to a better understanding of how to mitigate forgetting in this setting.

- The theoretical model is very simple (only a single linear attention layer), but it leads to some interesting predictions about model behavior, which are supported by experiments. For example, I think it could be a useful contribution to demonstrate how data augmentation might improve generalization (by leading the model to assign higher attention scores to subject tokens rather than template tokens).

- The paper proposes a new generative replay method, which appears to reduce forgetting without reducing continual learning. This could be useful for future work, and it also suggests that the theoretical analysis can lead to some practical innovations.

- The empirical results are supported with two language models and both real and synthetic data.


- The writing and structure of the paper are generally clear (although see my comments below). The model setup is clear, and I found it helpful that the paper highlighted the main findings in each section in gray boxes.

### Weaknesses
- Beginning in section 2.2, I found the exposition to be difficult to follow, partly because the notation is not always clearly defined. This made it difficult to evaluate the different theorems and intuitively understand their implications. I have included more detailed questions in the Questions section. One thing I found consistently confusing is the subscript $s$, which seems to be used inconsistently throught the paper. For example, in theorem 1, $s$ appears on the left hand side of the equation to index a subject token, but it's also used as the index of summation on the right-hand side, ranging from 1 to $t$ (the time index). Assuming that the $s$ used as the index of summation is different from the $s$ outside the summation, it is difficult to make sense of what these terms mean, which is critical for understanding the main contribution.

- The theoretical model is very simple--it uses linear attention and does not include a feed-forward layer. I think these simplifying assumption are probably reasonable in this case, they are consistent with prior theoretical work, and the predictions still seem to lead to meaningful results. However, I think the authors could expand their discussion of the implications of these limits, and how future work might extend this approach to more realistic architectures. In particular, feed-forward are thought to play an important role in factual knowledge acquisition (see e.g. Geva et al., 2023). The authors discuss some model extensions in the appendix, but the discussion does not mention feed-forward layers or linear attention.

- Some methodological details are not so clear from the text. In particular, it is not clear from the main text how the PT dataset and CPT dataset differ, and it not always clear whether the model was being evaluated on facts from the PT dataset or the CPT dataset. I included more detailed comments in the Questions section.


**Summary:** I think the paper presents a potentially useful contribution to understanding and mitigating forgetting in continual fact learning. However, the clarity issues make it difficult to understand the theoretical contributions or assess the soundness. I would consider increasing my score if these issues could be addressed, and if the limitations section could be extended to some of the other key simplifications (especially feed-forward layers).


_References_


[1] Geva et al., 2023. Dissecting recall of factual associations in auto-regressive language models.

### Questions
- In theorem 1, what is the role of $t$ in $\mathbf{u}_s(t)$? Why is $\mathbf{u}$ sensitive to time?

  
- In theorem 1, how is $\boldsymbol{\xi}_s$ dependent on $s$? In the definition in theorem 1, the $s$ on the right hand side appears only in a sum over $s$, so it's unclear how $\boldsymbol{\xi}_s$ is dependent on $s$.

- In the summation in equation 1, why should $s$--which indexes a subject token--range from 1 to $t$, which indexes the time step?

- In equation 5, is $t$ defined with respect to the beginning of pre-training or CPT? Similarly, is $\mathbf{H}_s$ defined in terms of the subject-object distribution from pre-training or CPT?

- In line 320, can you explain why the relationship between parameter count and knowledge amount means that each component of $\mathbf{y}_s$ will not have substantial importance?

- In section 3.2, could you give a more intuitive explanation for why the the amplitude oscillation term will become larger in this setting, and why this mitigates forgetting?



- In sec 2.2, could you clarify the PT/CPT setup? My assumption is that the CPT data consists of the same subjects and attributes from the PT data, but with different objects. Is this right?


- In Table 2, what do the column headings mean? Does each column (original vs. continual) refer to an evaluation set or a model? In other words, does Pythia-Original mean the (CPT) Pythia model evaluated on the PT facts?


- Regarding the finding in section 2.3: Is it possible to say whether the generalization improvement of data augmentation is driven more by: (1) $z_s$: the model pays less attention to template tokens $s$, or (2) $y_s$: template token $s$ has less interference with the output probabilities?

_Suggestions_

- The introduction states that "data replay amplifies the oscillation amplitude when convergence is reached", but without first introducing what this oscillation is. Could possibly give a brief summary of the relative finding in the previous paragraph.

- I suggest explicitly stating that the model uses tied input and output embeddings, which is another assumption.

- "synthesis" in line 65 should be "synthetic"

- In the middle equation in line 133, should $\log \bar{\mathbf{x}^s}$  be $\log \bar{\mathbf{x}_s}$ ?

- In line 128, should it be $\mathbf{U} \in \mathbb{R}^{D \times D}$?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper analyzes the training dynamics of factual knowledge acquisition and retention during continual pretraining. The analysis of linear 1-layer Transformer with several assumptions reveals that data augmentation can be advantageous compared to regularization-based methods to mitigate factual knowledge retention. Based on the theoretical insights, this paper proposes a generative data augmentation method for replay, STOC, which shows comparable or better factual knowledge retention in CPT compared to previous methods.

### Strengths
S1: This work tackles an important problem of building theoretical explanation of factual knowledge acquisition dynamics in continual pretraining.

### Weaknesses
W1: **Unclear technical novelty & contribution** - Although the authors acknowledge they are building upon the 1-layer linear transformer constructions with strong assumptions used in previous works, what distincts this work from them remains unclear. I’d be happy to hear from authors about the novel points of each section and finding compared to previous insights. For example:
- Theorem 1 reflects the classical SGD dynamics on quadratic program, which is applied to the specific problem of CPT on linearized 1-layer Transformer. 
- The theoretical and experimental result that attention patterns become more selective to subject tokens upon data augmentation can be inferred from the thorough analysis in [1], as they have shown the attention score for common tokens shrink at early phase of training, and data augmentation deliberately induces non-subject contexts become common.

W2: **Gap between the theory and experimental design**
- In the current experimental protocol for data augmentation experiments, where the number of the biographies per individual is controlled, does not control the effect from the dataset size. For example, to rule out this, an experiment comparing two datasets, (1) 10k individuals and 1 biography for each individual (2) 2k individuals and 5 biographies for each individual will be required.
- In L358, the authors claim that the increase oscillation term can explain the advantage of replay methods in knowledge retention, but the connection between oscillation term and knowledge retention is unclear to me, and this is not directly confirmed by theoretical or experimental analysis.


W3: **On methodological design**- While the proposed method, STOC, can be effective in learning orthogonal knowledge during CPT,
- There is no further theoretical analysis that show this method can be strictly better compared to naive baselines or guarantees improved knowledge retention
- I think this design can be problematic when we want to overwrite previous knowledge in practical CPT scenarios, by the nature of the design. Specifically, if we want to continually update the model with new knowledge that conflicts with the previous one, STOC would generate replay data that contains previous knowledge that should be overwritten. I would like to hear from the authors on this potential issue.



[1] https://arxiv.org/abs/2305.16380

### Questions
Please see my questions in the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a theoretical framework that characterizes the training dynamics of cFKA using a single-layer Transformer with linear attention.
The analysis shows that regularization based methods only adjust the speed of convergence and not the final forgetting, but data replay methods can shift the convergence dynamics.
Based on the insight, STOC identifies influential factual snippets to improve upon regular replay methods.

### Strengths
- Analyzing forgetting through the lens of learning dynamics is an important direction; I'm glad to see papers making a push in this direction
- The separation of learning dynamics into Y and Z seem to be a clean choice for analysis
- The experiments on the synthetic Biograph shows significant benefit for STOC

### Weaknesses
- Style-wise, I find the paper is hard to follow:
	- The notation is a bit overwhelming and cluttered.
	- Some terms are introduced without proper explanation (e.g., oscillation amplitude) and can only get a better sense much later on, and wasn't clear to me whether large oscillation is desired or not.
- On a high-level, the studied setting seems to deviate from what is claimed. Specifically, the described setting in 2.1 is not really continued pretraining while in the paper the authors describe "[...] offering a unified explanation for the behavior of popular CPT methods". cFAK as described seems to be a very light post-training stage in terms of scale and characteristics.
- The effectiveness of STOC seems questionable. In Table 4, the perfomance of original is only 2-3 points above (often within 1 point) the baselines and given the number of datapoints in these datasets, it doesn't seem significant.
- Many smaller details seem off (mentioned in the question section).

### Questions
- The regularization coefficient in Table 2 is in the range of 1e6 to 1e8. If I'm not missing anything, this is unnaturally large. Can you explain why this is a justified choice?
- The dimension is off? Y \in R^{DxD} but D is supposed to be the vocab size?
- Eq 144: what's equation 6? There's no equation 6 in the main paper. 
- What is Theorem 1 intended for? It's only showing a first-order tyler expansion of the error term, which is a dirivation step not a theorem. What is the assertion of the theorem?
- In Theorem 1, what is the matrix U and why is it relevant?

### Soundness
2

### Presentation
1

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
They study why continual pretraining makes language models forget old facts. They build a theoretical framework using a one-layer transformer to analyze how models acquire and retain factual knowledge. They show regularization only slows forgetting, while data replay stabilizes previous knowledge. Based on this, the authors propose STOC, a generative replay method selecting important tokens by attention. Their experiments confirm the theory and show STOC reduces catastrophic forgetting.

### Strengths
- They have clear motivation and propose a unified theoretical framework that formalizes continual factual knowledge acquisition (cFKA) as an analyzable training dynamic process.
- They clearly distinguishes how regularization and replay differ in convergence rate versus convergence point.
- They also validated on both synthetic and real datasets

### Weaknesses
- The theoretical analysis relies on a single-layer linear-attention Transformer, which may oversimplify the mechanisms observed in real multi-layer nonlinear architectures. While this abstraction is useful for interpretability, it remains unclear whether the same convergence dynamics generalize to the larger LMs.
- I think the real-data experiments are relatively limited in scope. Including larger-scale continual pre-training results or more diverse domains would strengthen the empirical support for the proposed claims.
-  Also, the current framework focuses mainly on templated and independent factual triplets. It would be valuable to discuss how the theory might extend to more complex knowledge structures, which are common in real-world factual updates.

### Questions
Do you think adapt replay ratio dynamically to task difficulty further mitigate forgetting? 
Are results in Tables 3-4 statistically tested? How many random seeds? Are STOC's improvements within error margins?

### Soundness
2

### Presentation
3

### Contribution
2
