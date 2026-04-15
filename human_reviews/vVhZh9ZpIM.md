# The Pitfalls of Memorization: When Memorization Hurts Generalization

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 5

## Abstract
Neural networks often learn simple explanations that fit the majority of the data while memorizing exceptions that deviate from these explanations. This behavior leads to poor generalization when the learned explanations rely on spurious correlations. In this work, we formalize $\textit{the interplay between memorization and generalization}$, showing that spurious correlations would particularly lead to poor generalization when are combined with memorization. Memorization can reduce training loss to zero, leaving no incentive to learn robust, generalizable patterns. To address this, we propose $\textit{memorization-aware training}$ (MAT), which uses held-out predictions as a signal of memorization to shift a model's logits. MAT encourages learning robust patterns invariant across distributions, improving generalization under distribution shifts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the interplay between learning with spurious features and memorizing the noises in the training data. It shows that the combination of the two can be harmful for model generalization because the model lacks further incentive to learn the actual underlying generalizable solution. Based on this, the paper proposed a training paradigm called memorization-aware training (MAT) by utilizing a model trained with heldout data to shift the current model's logits.

### Strengths
1. This paper presents a study of how memorization impact generalization, with a specific focus on the presence of spurious features and distribution shift between training and testing. 
2. The intuition is presented with both a synthetic data example and a theoretical construction.
3. A new method is proposed based on the intuition that perform competitively compared to the previous baselines.

### Weaknesses
1. Spurious features and distribution shift is at the core of the most of the paper. I believe it is better to reflect this in the paper title, which currently is framed quite broadly.
2. The presentation of the paper could be improved. For example

    1. The synthetic example and discussions in Section 5, while interesting, feels disconnected and unrelated to the rest of the paper. Does it mean the memorization at the presence of spurious feature are always good / bad / ugly type of memorization? Or is the proposed training method going to help mitigate any type of memorization discussed here? Can those types of memorization be easily identified in real world scenarios? 

    2. Section 4.1 ends abruptly with a reference to Table 1 and a list of the baseline methods. Having some discussions of the results would be good since the proposed MAT algorithm is an important contribution of the paper. Right now looking at Table 1 it seems the proposed method is not outperforming previous methods when the group annotations for the validation data is available.

3. The logits shift is estimated with Eq (4) when annotation is not available. It would be great to have an ablation study of the accuracy of such estimation in one of the experiments.

4. The explanation of learning spurious feature + memorizing noise is interesting. However, it is still unclear from the results of the paper if this is what is happening in more general setting (i.e. without explicit spurious features and subpopulation shift). So it would be great to have some results on other standard ML benchmark as well.

5. The analysis in Section 4.2 show that the proposed method reduces self influence in the minority subpopulations. It would be great to include the same analysis on some baseline methods compared in Section 4.1 as well.

### Questions
1. In the experiments, does the validation set used to shift the model logits contain spurious feature?
2. L160: the paper says "the model *first* learns $x_a$ ... Once the model achieve nearly perfect accuracy on the majority examples, it *starts* to learn the minority examples". Could you explain where do we observe such order of learning from Figure 1?


==============
Post rebuttal: Thanks the authors for the response, especially the additional results and analysis. I've raised my rating.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper examines the relationship between spurious correlations and memorization. A model trained with Empirical Risk Minimization (ERM), which has learned spurious correlations and has memorized irrelevant patterns, can have poor generalization. Spurious correlations result from patterns within the input data that are closely associated with the output, but not necessarily predictive (e.g., grass in a scene with cows). Memorization occurs when a model memorizes specific patterns rather than learning robust features that generalize to new examples.

To address this, the authors propose a novel approach, Memorization-Aware Training (MAT), a method that modifies the logits of the cross-entropy loss to discourage the model from relying on spurious features ("the indirect path"). By minimizing the log probability of the "indirect path", where the output y depends on a spurious feature a, MAT encourages the model to learn more generalizable patterns. 

The accuracy of MAT is compared to various baseline methods under different sub-population label settings. Additionally, the paper shows that MAT effectively shifts the self-influence distribution, reducing the reliance on spurious correlations.

The authors additionally provide a thorough theoretical analysis and a detailed description MAT algorithm.

### Strengths
1. I really appreciated the first paragraph of the introduction and how it related to your work. This was very creative and smart, and helps the reader in understanding.
2. I like some the experiments in the paper and I have questions about others. First, for Figure 1, setting gamma to 5 creates a type of "worst-case" scenario with respect to spurious features. MAT is able to overcome this learning a generalizable function in the presence of noise. The experiments in Figure 2 show an improved self-influence score distribution with the use of MAT, which is a convincing way to present the effectiveness of your method.

### Weaknesses
1. One thing that the paper is missing is arguments as to why MAT is a better approach than other invariant methods? Or, alternatively, why should one use MAT over these other methods? What are the advantages and disadvantages, aside from sub-population labelling? The average accuracies among these methods seem to be similar. That is, there is no consistent 'best' approach. This doesn't invalidate the results, but I would like to know why and when I should choose MAT over another approach.
2. I think that this paper would benefit from a more realistic scenario where noise modifies the input features. The concatenation of noise to the input is a simplification, which serves a clear purpose; however, it is not representative of real-world noise, in which the input features are directly modified.
3. The rightmost image in Figure 3, with noisy labels, confused me in the sense that I don't understand how it demonstrates the benefit of using MAT.

### Questions
1. (Continuation of weakness #2) Why did you chose to concatenate the noise to the input, rather than add it to the input? How do you think your results would differ if you added the noise to the input, simulating noisy real-world data? Your results, which rely on this simplification, may not hold true to real-world tasks.
2. (Confusion) Looking at the results depicted in Figure 2, I noted that when training with ERM, the Waterbird on Water has a right-tailed self-influence score (e.g., in the right most figure, WB on Water has a large proportion of samples with a self-influence score of 0.3, yet LB on Land has a very tiny proportion of samples with a self-influence score of 0.3). Do you have any ideas on why this is the case? Is this due to a characteristic of the dataset? Of the sub-population images?
3. You can also address any weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the effect of the combination of spurious correlation and memorization and points out that the latter exacerbates the poor generalization caused by the former. The claim has basis on an experimental and theoretical analysis with a linear model. Furthermore, a method to mitigate the problem is proposed and its effectiveness is checked in experimental setups of subpopulation shift.

### Strengths
- A theoretical analysis gives us an example model where the training with noiseless data leads to a test performance asymptotic to perfect accuracy and the training with noise data leads a test performance asymptotic to perfect fitting to spurious correlation(Theorem 2.2).

- A new training loss for mitigating the harmful effect of spurious correlation without requiring annotations on spurious features is proposed based on the theoretical analysis mentioned above (Section 3.1).

- The effect of the proposed method is checked experimentally with the subpopulation shift problem. Although superiority against existing methods for this problem is not observed clearly, the improvement from naive empirical risk minimization is obvious(Table 1). Analysis with influence function is also conducted, and it is observed that memorization is suppressed by the proposed method(Figure 2).

### Weaknesses
- The theoretical analysis is limited to the case of a linear model.

- It is not clear to me how the logit shift proposed in 3.1 makes the training more "memorization-aware". In my understanding, the shift down-weights the gradient descent updates coming from the training samples with spurious correlation, and it should work, but it is not clear how the shift values change depending on the extent of memorization. It is preferable if a comment about this is added in 3.1. 

- (L691, L694) Duplication of a reference.

- The code is not available with the first submission.

### Questions
- Please consider providing clarification regarding the weaknesses.
- Is it possible to extend MAT for more general spurious correlation? In general situations, it can happen that multiple spurious features may correlate with each other, and the indirect path mentioned in Section 3 may change to the one involving intermediate correrations.

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
5

### Summary
This paper studies memorization in the presence of spurious features and input noise in the training data. To improve generalization the paper proposes memorization aware training (MAT). MAT computes held-out predictions, obtained using XRM (prior work), and then uses the predictions to shift model logits. The shifted logits force the model to learn generalizable features, empirically improving worst-group accuracy across common distribution shift benchmarks with spurious correlations like CelebA and Waterbirds.

### Strengths
- The analysis in Section 5 is interesting, particularly the distinction between the noiseless and small noise setting, where the latter actually generalizes better. Essentially, the results indicate that a small amount of independent noise in the input is essential to fit the label noise. It would have been really nice to see an expanded analysis on this claim, with some real world experiments and formal analysis.

### Weaknesses
- My main concern with the paper is lack of novelty in the main technical claim, i.e., memorization (afforded by overparameterization)+spurious correlations = poor generalization. There are multiple works like Shah et al. 2020, and Sagawa et al. 2020 that make the same claim with supporting evidence. 
- The connection between the objective (shifting the logits) and the final goal of preventing memorization is not very clear. In my understanding, shifting the logits should have the same affect as adding more weight in the loss on minority groups. 
- The theoretical analysis in Section 2.1 is very similar to Sagawa et al. 2020 (An investigation of why overparameterization exacerbates spurious correlations). In the prior work, the analysis in the same toy setup shows that memorization is exacerbated by spurious correlations, and thus the results in this work are almost directly implied by the results in Sagawa et al. 2020.
- The empirical results are not very strong, since XRM+GroupDRO is comparable to MAT.

### Questions
- In the "ugly" memorization setting in Section 5, it is unclear to me why in the noiseless setting, the solution learnt is different from the setting with $\sigma=10^{-4}$. The class of over-parameterized networks contain both the function learnt in the $\sigma=10^{-4}$ setting and $\sigma=0$ setting. So, why is the learning biased to the bad solution?

### Soundness
1

### Presentation
2

### Contribution
1
