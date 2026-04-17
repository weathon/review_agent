# Interpretable Transformer Regression for Functional and Longitudinal Covariates

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Predicting scalar outcomes from functional data is challenging when measurements are sparse, irregular, and noisy, as in many scientific and clinical longitudinal studies. We propose IDAT, a dual-attention Transformer that operates directly on masked sampling schedules and avoids ad-hoc imputation. IDAT couples (i) time-point attention, which captures local and long-range temporal dynamics together with the response relationship nonparametrically, with (ii) inter-sample attention, which adaptively shares information across subjects with similar trajectories to stabilize estimation under sparsity. These pathways complement one another: time-point attention captures subject-specific dynamics, whereas inter-sample attention leverages population structure to ``borrow information'' from other subjects, echoing principles from random-effects model in longitudinal analysis. Under a random-effects framework that accounts for irregular sampling and measurement noise, we prove prediction-error bounds and show that IDAT consistently approaches the oracle solution. Across both simulations and real-world applications, IDAT achieves the best overall performance among 19 baselines. Only in the extremely dense case ($>80\%$ observations) TabPFN (a recent method published in Nature) achieve a slight advantage, while IDAT still significantly outperforms all other baselines in this scenario. The learned attention weights are interpretable, revealing predictive time domains and potential clusters. In conclusion, IDAT, an end-to-end sparsity-aware Transformer architecture, offers improvements both in predictive accuracy and interpretability for scalar-on-function prediction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes the Interpretable Dual-Attention Transformer (IDAT), which is a model for predicting scalar outcomes from sparse, irregular, and noisy longitudinal data. Specifically, IDAT integrates time-point attention to capture temporal structure and inter-sample attention to share information across similar subjects. The methods is trained end-to-end without imputation.

### Strengths
- **Sound approach:** IDAT unifies temporal (time-point) and cross-sample (inter-sample) attention mechanisms. This design allows the model to capture both individual trajectory dynamics and shared structures across subjects, and it seems to work well in practice.

- **Mesh size insight:** Personally, I found th mesh size tradeoff provides helpful insights how to balance computational efficiency with prediction bias.

### Weaknesses
- The paper is very **straight forward**. The paper basically uses a grid discretization for functional data input.

- **Dual attention** = known parts, new combo: Time-point self-attention and cross-sample (nearest-neighbor-like) pooling are both established ideas. IDAT simply combines them in one encoder and trains them  end-to-end with masks. 

- **Important parts of the paper (e.g., ALL results on simulated data in Sec. 4.1, all Lemmas) are relegated to the appendix.** I understand that not all results can always make it to the main body of the paper. However, here it seems the authors have moved too much important parts to the supplements.

- **Performance is very regime-specific:** The strongest gains are reported when only < 50% of time points are observed, which is fair since this is the regime the method is designed for. However, as sparsity decreases, it seems like the advantage will narrow outside of the very sparse regime

- **Standard consistency and Lipschitz arguments:** The main results are architectural Lipschitz/approximation lemmas and two theorems: encoder consistency and a training-MSE generalization bound under standard assumptions (boundedness, stable SGD, $p/n \to 0$ ). 
These are mathematically sound but completely expected for modern attention+MLP networks. My issue here is that these insights do not provide information /  do not translate into design guidance (e.g., how to pick layers, heads, grid size beyond asymptotics)

- **Variance reduction claim:** Inter-sample attention reduces embedding noise by $O(B^{−1/2})$ when subjects are similar. However, this is a mini-batch averaging effect. The result does not quantify when the bias from pooling outweighs variance gains (of note, the paper concedes this trade-off empirically).

- **Presentation:** the paper would benefit a lot from improving presentation and writing. As mentioned earlier, many components that appear important are moved to the supplements for no obvious reason.

### Questions
- How does IDAT fundamentally differ from existing attention architectures that combine temporal and relational modeling (such as TabNet, Set Transformers, or hierarchical temporal attention models)? What isthe novelty beyond combining two known mechanisms?

- Why is inter-sample attention considered novel?

- The paper provides consistency and Lipschitz-type generalization proofs, but these are standard for attention networks. Specifically, what new theoretical insight do these results provide about dual-attention?

- The variance reduction lemma assumes similarity among samples. Can the authors provide any conditions or diagnostics for when inter-sample attention introduces bias instead, for example in in heterogeneous populations?

- Regarding the variance reduction claims, how does the train-test mismatch around the Y-token affect consistency in practice?

- The interpretability claim rests on “weights acting as regression coefficients.” Is it possible to somehow interpret these weights quantitatively or only qualitatively as saliency maps?

- Sinusoidal positional encodings do not extrapolate well. How does IDAT handle unseen time grids? Would it be posssible to incorporate relative or learned encodings?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new method, the Interpretable Dual-Attention Transformer (IDAT), for scalar-on-function regression where the functional covariate is observed sparsely and irregularly over time. IDAT discretizes the time axis into a grid and uses explicit missing-value masks, avoiding imputation and enabling end-to-end training. It introduces two attention mechanisms: (1) Time-Point Attention, which encodes both local and long-range temporal structure within a subject's trajectory, and (2) Inter-Sample Attention, which borrows information across similar subjects in a batch. The learned attention weights are interpretable and reveal predictive time windows and cohort clusters. The paper also provides theoretical analysis of prediction error bounds and consistency. Experiments on both simulated and real-world datasets are conducted to evaluate the proposed method.

### Strengths
1. The motivation for addressing scalar-on-function regression with sparse, irregular, and noisy longitudinal data is clearly described.
2. A new solution that combines time-point and inter-sample attention in a Transformer framework is proposed, directly targeting the challenges of missingness and heterogeneity.
3. Theoretical analysis for the error bound and consistency of the training and testing phases MSE is provided.
4. Experiments comparing the proposed method with more than 19 baselines on both simulated and real-world datasets are conducted to evaluate its effectiveness.

### Weaknesses
1. The experiments lack an ablation study to illustrate the importance of each module in the model, such as time-point attention and inter-sample attention, respectively.
2. The hyperparameters used in the experiments should be described clearly. In addition, a hyperparameter sensitivity analysis should be provided for important hyperparameters, such as batch size and grid size.
3. The method for simulating different levels of sparsity should be described clearly. It is recommended to also simulate different missing patterns, if not already done.

### Questions
Please refer to the Weaknesses.

### Soundness
2

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
2

### Summary
The paper proposes IDAT (Interpretable Dual-Attention Transformer), a method for a scalar-on-function regression over time. IDAT uses a dual attention module (time-point attention and inter-sample attention), and, thus, it can handle sparsely and irregularly measured longitudinal data. Furthermore, the learned attention weights are interpretable. The authors also provided the theoretical guarantees for the consistency of  IDAT. Finally, the paper compares the new method against multiple other baselines and demonstrates its effectiveness for the scalar-on-function regression.

### Strengths
The paper provides a comprehensive theory on the consistency of the proposed method. Also, the experimental evaluation is very extensive (it includes 19 baselines and multiple datasets).

### Weaknesses
I have outlined a couple of main concerns for the paper:
- **Lack of assumptions for missingness**. I lacked the assumptions on the missigness (or measurement intensity), namely, whether the observation times are informative of the outcome. If yes, shouldn’t the missingness mask be included as an input itself? 
- **Limited contribution**. I my opinion, the main method of the paper is a simple combination of multiple standard Transformer-based approaches. Yes, technically, IDAT might be new in this specific setting of scalar-on-function regression. Yet, overall, I cannot pinpoint a single non-trivial or unique approach that was used in this setting. 

I am curious to hear the authors’ response, and I am open to further discussion.  

Other minor concerns include the following:
- I found the abstract hard to read, as it is overloaded with very specific, tiny details and misses the broader picture.
- Same for the theoretical part, the implications for the theoretical statements were missing.

### Questions
- What is the main motivation to use the inter-sample attention (especially given that the data is i.i.d.)? I understand that they might help to reveal cluster patterns in data, but don’t they increase the variance of the scalar-on-function regression?
- Line 23. “The learned attention weights are interpretable...” Isn’t it true for all the Transformers?
- I wonder whether the stream of literature on marked point processes is relevant or could be adapted to this work (e.g., [1])?

References:
- [1] Panos, Aristeidis. "Decomposable transformer point processes." Advances in Neural Information Processing Systems 37 (2024): 88932-88955.

### Soundness
2

### Presentation
2

### Contribution
2
