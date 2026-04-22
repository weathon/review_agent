# Test time training enhances in-context learning of nonlinear functions

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Test-time training (TTT) enhances model performance by explicitly updating designated parameters prior to each prediction to adapt to the test data.  While TTT has demonstrated considerable empirical success, its theoretical underpinnings remain limited, particularly for nonlinear models.  In this paper, we investigate the combination of TTT with in-context learning (ICL), where the model is given a few examples from the target distribution at inference time.  We analyze this framework in the setting of single-index models $y=\sigma_*({\langle \beta, \mathbf{x} \rangle})$, where the feature vector $\beta$ is drawn from a hidden low-dimensional subspace.  For single-layer transformers trained with gradient-based algorithms and adopting TTT, we establish an upper bound on the prediction risk.  Our theory reveals that TTT enables the single-layer transformers to adapt to both the feature vector $\beta$ and the link function $\sigma_*$, which vary across tasks.  This creates a sharp contrast with ICL alone, which is theoretically difficult to adapt to shifts in the link function.  Moreover, we provide the convergence rate with respect to the data length, showing the predictive error can be driven arbitrarily close to the noise level as the context size and the network width grow.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors consider a theoretical analysis of single-layer transformers trained to do ICL of single-index models with test-time training.

### Strengths
The authors consider the analysis of an interesting problem: ICL under test-time training. 
- The results found seem reasonable and I believe most of the authors' claims about their model are well-supported. 
- To my knowledge, the work is original and considers new topics
- While I have some reservations (below), the results in the paper are interesting and I believe that this is an important line for continuing research. In particular, the ideas the authors raise about distribution shift of the ICL mapping function are relevant to LLM phenomena.

### Weaknesses
I have a few reservations about the paper:
- The presentation is difficult to follow, and builds on a technical body of work. It would be useful if the authors could provide more intuition for their results 
- It is not clear to me how to make an adequate comparison between theory and experiment from the information presented, and what comparison I can make leads me to suspect that theory & experiment do not agree. 
    - No theoretical curves for the predicted test error are presented in Figure 1 in order to make comparison possible. 
    - On line 430, the authors write "Contrary to our initial expectation, the pretrained model’s ICL ability is powerful enough to adapt to the varying link function, while TTT does not lead to significant improvement of the prediction." If I am understanding this correctly, the authors are saying that their theoretical results in fact do not agree with their experiments, and that they would not have expected from their theory that ICL alone could even learn the task. 
    - On line 437, the authors write that "TTT can adapt to distribution shifts, while ICL fails to do so." From my understanding of Figure 1, this is not true. While TTT does outperform ICL on distribution shifts in the authors' experiments, ICL shows at least some ability to adapt to distribution shifts. 
    - Taken together, these observations leave me with reservations about how well the authors' theoretical analysis reflects reality. If the authors could take steps towards unifying theory & experiment (even in the setting considered where the transformer learns single-index models), I would be happy to update my review.

### Questions
- What aspects of the analysis are leading to the prediction that ICL alone cannot learn a distribution of the link functions $\sigma$'s? How might the analysis be updated?
- It is unclear to me what aspects of the main theorem hold implications for real models trained on real data. Could the authors please clarify this?
- Could the authors please clarify what is meant by the statement that ICL fails to adapt to distribution shifts on line 437?

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
2

### Summary
This work provides a theoretical understanding for why transformers using in-context learning (ICL) can further improve their performance approximating non-linear functions under distribution shift using test time training (TTT). TTT has proven to be an effective approach empirically, but a strong understanding of several properties of this approach, such as its sample complexity, were unknown. The paper compares with previous work, and demonstrates that their new results provide convergence rates in more relevant variables, such as the dimensionality of the task-related subspace and number of training examples. Furthermore, the paper’s guarantees also allow for the non-linearity in the family of single-index models that they analyze to change at test time, providing a more interesting and relevant setting for TTT. Finally, the paper provides results for synthetic experiments demonstrating that TTT specifically allows for adaptation to a changing link-function, something that ICL cannot handle on its own.

### Strengths
- The techniques being analyzed (ICL and TTT) are very relevant right now to empiricists and theorists alike.
- The proof sketch provided makes a great job at not only explaining the reasoning in the proof but also demonstrating how pretraining with ICL and TTT are both necessary for this setting.
- Extensive comparisons with previous literature are provided, setting the stage clearly.
- Synthetic experiments towards the end of the paper provide satisfying evidence that TTT specifically helps with adapting to changes in the non-linear aspect of the single index model.
- The setting provided (link-function changing between test and train) seems more relevant to TTT than when the link-function is static, making the theoretical results more impactful.

### Weaknesses
- [W1] Some things in Algorithm 1 are not clear. See [Q1], [Q2].
- [W2] It is not always clear what the contributions of this work are, specifically in comparison to Nishikawa et al. (2025). See [Q3], [Q4].
- [W3] In real applications of TTT, examples are still provided in-context after TTT (Akyurek et al. (2025)). This seems to demonstrate that using information in context is still important for the performance of transformers using TTT. However, the final predictive function $f_{TF}$ does not use in-context examples, somewhat separating this work from real applications.
- [W4] (Minor) Some typos (line 127, “polylogamistic” -> polylogarithmic?).

### Questions
- [Q1] What is $N_{new}$? Based on line 142/143, I was under the impression that the only sizes of TTT data we had access to were $N_1, N_2, N_3, N_4$.
- [Q2] Where are the projected data in algorithm lines 7 and 16 used? It’s made clear that this operation is meant to put the data in the subspace spanned by the $\beta$s seen during pretraining, but they aren’t used in stage 2 or stage 3 of the algorithm.
- [Q3] In 3.1.1 and 3.1.4, it is stated that Nishikawa et al. proved the important properties of the pre-trained attention matrix, as well as how to recover the link-function with a ReLU MLP, respectively. Is the novelty in this approach in the weak and strong recovery steps? How does this new approach differ from how Nishikawa et al. recover $\beta$, if they do so?
- [Q4] On lines 333-339, Nishikawa et al.’s sample complexity bound is said to be tighter, but that the new bound still has two key benefits. Are these two benefits not also shared with the bound provided by Nishikawa et al.?

### Soundness
4

### Presentation
3

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
The paper discusses the efficiency of Test Time Training (TTT) in contrast with In-Context Learning (ICL) as a means of adapting to the prompt/context of an LLM.

The paper establishes bounds on the expected risk in terms of the number of samples provided in the context.

The improvement of TTT over ICL is demonstrated in a synthetic setting.

### Strengths
The difference between ICL and TTT shown in the experiment, and that it requires a sufficiently hard problem (in this case, a distribution shift) is interesting.

### Weaknesses
There are several strong modeling assumptions in the paper.
1. The “Regression for ICL” model
2. The single index model.
3. Pretraining consists of only one step of GD.
4. The context samples are used in a specific way (first to learn the direction of $\beta$ weakly, and then to learn $\beta$ properly).
5. The “input” weights of the MLP are frozen (and random), only the output weight is trained.


Arguably, points 1 and 2 are now common enough in the literature to be considered benign. There are several works that consider the type of GD that is considered here (point 3 and 5), an early example I remember is on feature learning in 2 layer ReLUs, I consider this to be a significant deviation from practice but refrain from ultimate judgement. Point 4 is novel to this setting afaik. I am not sure if the context is actually split in this way to learn some ground truth in stages. Can this be elaborated upon?

### Questions
Are $N_{\text{new}}$ and $N_2$ the same? Is $w_i$ the $i$th sample in the $\mathbf{X}_{N_2}$ set?

What is $g$?

What is the point of lines 15-16 in the algorithm? It doesnt seem like those x are used again. Also arent all the covariates projected onto $\Gamma$? Might be easier to just say that in line 6-7 if so.

For random feature based learning I would expect the size of the hidden layer to be quite large. Here we have m = poly(d). What does $d$ need to be explicitly? I couldnt tell from Appendix G (may be this is made in explicit in some of the related work).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an algorithm for test-time scaling and validates its effectiveness through theoretical analysis and experiments. The authors derive an upper bound that depends on the feature subspace dimension, the test-time context length, and the network width, suggesting that Transformers can adapt to low-dimensional subspaces. The paper also includes numerical experiments on synthetic data to support the theoretical findings.

### Strengths
The underlying problem the authors investigate is important, given the growing power and influence of LLMs. Understanding the mechanism behind test-time training, which has shown strong empirical performance, is both timely and valuable. Also, the paper makes a clear effort to exposit the structure of the proof, helping readers follow the theoretical reasoning and connect the analytical results to the underlying intuition.

### Weaknesses
The paper has issues in terms of writing quality and overall presentation, placing it below the acceptance threshold.

Firstly, the authors fail to clearly compare their theoretical results with prior work, making it hard to understand what the exact novelty and contribution of this paper are. While it is true that the paper provides theoretical results, one key questions remain:

1. The order of the bounds in the theoretical results appears standard and not particularly impressive. For a theoretical paper, the authors should explicitly highlight how their bounds improve upon existing ones, what technical challenges were involved, and how these were addressed. It seems that the authors assume the reader have read the paper Nishikawa et al. (2025), and therefore a lot of comparison are missing. Even after reading 327-352, I am still not sure what are the difference of the setups, assumptions, and technical improvements.

Moreover, there are many small issues in terms of writing, which really impacts the readability

1. line 074 $N_{pt}$ and $T_{pt}$ are not defined, and the author put this in the statement of a theorem.

2. section 2.3 is introduced without any transition. It is about a student model, but the author never mentioned why we need a student model for TTT. At least a smooth transition for the motivation is required. For instance, if I replace "student model" with "one-layer transformer", then the paper would still read fine. 

3. it is mentioned earlier that the motivation is for developing the theory is that when $r < d$ the sample complexity is much better. But in numerical experiment, the authors set $r = d = 16$.

In general, when reading the paper, I am quite confused about the setup and motivation.

### Questions
See weakness above

### Soundness
2

### Presentation
1

### Contribution
2
