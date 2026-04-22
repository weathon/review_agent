# Predicting LLM Reasoning Performance with Small Proxy Model

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Given the prohibitive cost of pre-training large language models, it is essential to leverage smaller proxy models to optimize recipes before scaling up. However, this approach becomes challenging for reasoning capabilities, which exhibit \textit{emergent} behavior that only appears reliably at larger model sizes, often exceeding 7B parameters. To address this, we introduce \tsc{rBridge}, showing that small proxies ($\leq$1B) can effectively predict large-model reasoning by aligning more closely with \textbf{(1)} the pre-training objective and \textbf{(2)} the target task. \tsc{rBridge} achieves this by weighting negative log-likelihood with task alignment, using reasoning traces from frontier models as gold labels. In our experiments, \tsc{rBridge} \textbf{(i)} reduces dataset ranking costs by over 100$\times$ relative to the best baseline, \textbf{(ii)} achieves the strongest correlation across six reasoning benchmarks at 1B to 32B scale, and \textbf{(iii)} transfers predictive relationships across pre-training recipes at 1B to 7B scale. These findings indicate that \tsc{rBridge} offers a practical path for exploring reasoning-oriented pre-training at lower cost.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper offers an improved method for predicting large language model performance on reasoning tasks using results from smaller proxy models. RBridge is an elegant solution. First, reasoning traces from a frontier model are extracted for some benchmark dataset. Second, the negative log-likelihood of each token in the reasoning trace is computed using a small proxy language model. Finally, this token-wise NLL is weighted by MinMax normed average probability of each character in that token, as defined by the frontier model. This produces a weighted negative log-likelihood of the reasoning trace. The authors show that this value ranks and predicts performance of larger models. 

In the first experiment, they train small proxy models and large target-size models on 25 different datasets. They compute the agreement (Decision Accuracy) between the rankings of which dataset leads to better performance (on specific benchmarks) for the proxy models and for the target models. They show that they outperform other methods for ranking datasets and incur a significant compute saving.

In the second experiment, they show that the RBridge value (weighted NLL) for proxy models at different stages of training is strongly correlated with and predictive of the performance of much larger models.

In the third experiment, they show that fitting a function to predict accuracy given the RBridge value (weighted NLL) based on one dataset, transfers seamlessly to predicting performance for a model trained on an entirely different dataset.

Overall, this work offers an innovative and powerful new metric for optimising pre-training datasets through the use of small proxy models. This will be very useful to the community for improving LLM pre-training efficiently, because they can iterate on different training regimes using a small model, with the knowledge that they can predict the performance of much larger models.

### Strengths
The paper has many strengths:
* The motivation and methodology are well thought through and sound.
* The empirical work is diligent and complete. I commend the authors on the number of training and testing runs they must have performed to complete this project.
* The results are very impressive. The efficiency gain from using RBridge compared to other methods is considerable and will greatly accelerate the development of optimal pre-training datasets for large-scale reasoning models.

### Weaknesses
There are only a few minor weaknesses with the paper at the moment:
1. The authors miss some important literature on predictable AI and the use of assessor models for predicting model performance on unseen tasks. I include relevant references at the bottom.
2. The writing is quite rough in several areas. I outline the key spelling/syntax problems below. I would recommend going over the whole paper to check for sense and grammaticality.
3. I would include the limitations and future directions section in the main text, given the extra page.
4. The authors are right to point out that obtaining the reasoning traces from frontier models is an extra cost and may be imperfect. I would definitely like to see the authors make their datasets public upon publication.
5. This is not really a weakness, but the authors could emphasise more that their results really mean that LLM development can *greener*. Right now, iterating over different data regimes uses a lot of energy, which their method can help to reduce if the community makes use of RBridge.


## Missing References

Kipnis, A., Voudouris, K., Buschoff, L. M. S., & Schulz, E. (2024). metabench--A Sparse Benchmark of Reasoning and Knowledge in Large Language Models. arXiv preprint arXiv:2407.12844.

Pacchiardi, L., Voudouris, K., Slater, B., Martínez-Plumed, F., Hernández-Orallo, J., Zhou, L., & Schellaert, W. (2025). PredictaBoard: Benchmarking LLM score predictability. arXiv preprint arXiv:2502.14445.

Prudêncio, R. B., Lorena, A. C., Silva-Filho, T., Drapal, P., & Valeriano, M. G. (2024, June). Assessor models for explaining instance hardness in classification problems. In 2024 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.

Schellaert, W., Martínez-Plumed, F., & Hernández-Orallo, J. (2025). Analysing the predictability of language model performance. ACM Transactions on Intelligent Systems and Technology, 16(2), 1-26.


## Spelling/Syntax

Approximation line numbers followed by the corrected sentence.

Line 36: "suggests that there may be a limit..."
Line 88: "To bridge the evaluation scheme between small proxy models to large target-scale models..."
Line 94: "We improve task alignment at ...."
Line 95 and the enumerated list: This needs to be fully revised for sense. It should start with "We empirically validate our method in the following five ways:" and then each number should be a full sentence.
Line 269: "which demonstrates that including an SFT stage..."
Line 286: "can help target metric Acc./p@1 to be used as signals"
Line 286-7: The sentence on TED oes not make sense to me.
Line 289: "Last is ScB which visualizes..."
Line 299: "respectively"

### Questions
* Are letter-level probabilities within each token conditional on the previous characters/tokens?
* The performance of $R^\phi$ alone is surprisingly high (e.g., in Table 2). Why bother with the extra hassle of character-wise tokenization etc.? What argument can the authors make about the meaningfulness of the difference in performance between $R^\phi$ and RBridge? To some, these differences might look negligible.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes to reliably estimate the reasoning capabilities of large LLMs using small proxy models. 
The paper shows that, as the reasoning capability is an emergent property of large LLMs, the vanilla loss obtained from small models can be noisy.
There are two key limitations identified: (i) evaluation objective misalignment (e.g., NLL loss v.s. pass@K, and off-policy Y distribution ) (ii) task alignment and at the token-level (e.g., important tokens like branching/searching/backtracking should be taken into consideration)

The proposed rBridge tackles the limitations via (i) using a reasoning trace generated by frontier models as a proxy of Y and (ii) the loss is weighted according to the normalized log-probs of the frontier model, where the potential tokenization mismatch issue is tackled by using the character-level average.

Experimental results show the proposed rBridge successfully predicts the model performance using much smaller proxy models (< 100M) for a 1.2B target model. The correlation stays strong when using the 1B model to predict 13B/32B models and shows promising transferability across pre-training datasets.

### Strengths
- The investigated problem is of great practical value, reducing the computational cost for training large-scale LLMs.
- The rBridge is generally effective and easy to implement, with details well covered.

### Weaknesses
- While the papers examine the choices of frontier models in the appendix (i.e., no meaningful difference), I am still concerned about whether the quality of the reasoning trajectories would influence the estimation. As frontier models remain a black box, it is challenging to understand the underlying training distribution. Would the prediction performance degrade significantly if the reasoning benchmarks are OD even for those frontier models?
- The weighted NLL is not ablated. How much does it contribute to the final prediction performance?

### Questions
See Weaknesses

### Soundness
3

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
This paper introduces RBRIDGE, a method for predicting large language model reasoning performance using small proxy models (≤1B parameters) by aligning evaluation with both the pre-training objective and the target task. RBRIDGE leverages reasoning traces from frontier models as gold labels and introduces a weighted negative log-likelihood metric that emphasizes task-critical tokens, achieving strong correlations across six reasoning benchmarks and reducing dataset ranking costs by over 100×. It further demonstrates zero-shot transfer of predictive relationships across pre-training datasets, enabling efficient performance estimation at large scale.

### Strengths
1. The discussed problem is both intellectually engaging and practically important for advancing efficient LLM development.

2. The proposed RBRIDGE metric shows strong performance, achieving high correlation with large-model reasoning while greatly reducing computational costs.

### Weaknesses
1. The paper shows small models struggle with noisy tasks but doesn't clearly explain how to identify which datasets are noisy—more analysis on dataset characteristics would help.

2. The frontier model choice matters: models like R1 and GPT-4o differ in style and training data, which could affect reasoning trace quality and RBRIDGE’s consistency.

3. RBRIDGE is tested on math and QA tasks, but performance may differ on code (e.g., LiveCodeBench) or subjective tasks (e.g., Arena-Hard), where reasoning patterns vary.

4. Figure 5(b) illustrates RBRIDGE’s strong fit, but similar plots on other benchmarks would strengthen the empirical support.

5. Minor typos need fixing, e.g., “is becomes” in Figure 1’s caption.

6. The paper exceeds 9 pages, which violates the rule.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a new method to use a small proxy model to predict the reasoning performance of LLM. It uncovers that existing methods don’t work well because they fail to align with the pre-training objective and the target task. To address the limitations, the proposed method uses frontier-model generated gold reasoning traces for better alignment. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
+ using frontier-model generated gold reasoning traces for better alignment
+ significantly improved performance in terms of compute cost, proxy model size, and zero-shot transferring from one pre-trained dataset to another

### Weaknesses
- needs a frontier-model to generate gold reasoning traces, making the proposed method less useful
- it is unclear how the small proxy models less than <100M work so well

### Questions
How was the compute cost measured? Is it measured in a real system or just calculated by the model size? What are the system (hardware and software) settings if a real system was implemented? Using the proposed method, can a larger proxy model work better and how much better than a smaller proxy model?

### Soundness
3

### Presentation
2

### Contribution
2
