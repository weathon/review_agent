# Learning from Label Proportions via Proportional Value Classification

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Learning from Label Proportions~(LLP) aims to use bags of instances associated with the proportions of each label within the bag to learn an instance-level classifier. Proportion matching is a widely used strategy that aligns the average model outputs of all instances in a bag with the label proportions in order to induce the classifier. However, simply fitting the label proportions does not encourage discriminative instance-level predictions and may cause over-smoothing problems, resulting in poor classification performance. In this paper, we propose a novel LLP approach that can mitigate the over-smoothing problems with theoretical guarantees. Rather than fitting the label proportions directly, we treat them as targets for an auxiliary proportional value classification task to induce the target classifier. Our approach only requires the incorporation of an aggregation function after the classification layer. We also introduce an efficient computational approach with a divide-and-conquer strategy. Extensive experiments on various benchmark datasets and under different bag-generation strategies demonstrate that our approach achieves superior performance compared with state-of-the-art LLP methods. The code is publicly available at https://github.com/TianhaoMa5/ICLR2026_LLP-PVC.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel approach for LLP, a weakly supervised learning paradigm where the training data consists of bags of instances annotated only with the overall proportion of each label within the bag, and the goal is to learn an instance-level classifier. The authors identify a limitation in prevailing proportion matching strategies: over-smoothing problem, where the model outputs for all instances in a bag converge indistinguishably towards the proportion value, resulting in poor instance-level classification performance. To address this, the paper introduces PVC, a method that reframes the LLP problem. Instead of fitting the proportions directly, the label proportions are treated as targets for an auxiliary multi-class classification task. This auxiliary task involves predicting the specific proportional value from the bag of instances.

### Strengths
- This paper establishes a theoretical bridge between the posterior probabilities of the bag-level proportional values and the instance-level labels, enabling the induction of the target instance-level classifier. 
The computational algorithm based on a divide-and-conquer strategy is efficient, achieving a time complexity of $O(m \log m)$, which is significantly more efficient than a dynamic programming baseline with $O(m^2)$ complexity.
- The authors prove that under specific conditions, particularly when labels are deterministic, the outputs of the optimal instance-level classifier will converge to 0 or 1. The authors claim this property reveals that the proposed PVC mitigates the over-smoothing problem. I have checked the overall logical soundness of the proofs but have not delved into a detailed verification of the specific derivations.

### Weaknesses
1. The core claim of this paper is that PVC mitigates the over-smoothing problem. The authors present Theorems 1 and 2 as the theoretical foundation for this claim. While Theorem 1 establishes a one-to-one correspondence between the bag-level and instance-level classifier outputs, and Theorem 2 states that the optimal instance-level classifier outputs will be 0 or 1 under deterministic labels, the logical connection from these theorems to the concrete mitigation of over-smoothing remains inadequately articulated. It is unclear how these theoretical results, in a step-by-step manner, directly lead to the conclusion that PVC prevents the over-smooth. 

2. The output layers of neural network classifiers typically employ sigmoid or softmax functions to obtain label probabilities. Directly increasing the temperature coefficient in these functions can also make the model more inclined to output extreme values of 0 or 1. This straightforward strategy appears to offer an alternative approach to addressing the over-smoothing problem. So, compared to the temperature-based method, the merits of PVC is not clear.

3. The experimental validation is insufficient to robustly support the authors' central claim. Although the method demonstrates competitive accuracy, the claim of effectively alleviating over-smoothing is not thoroughly verified. The evidence for alleviation of over-smooth is primarily limited to the four figures presented in the introduction. Besides, relying solely on accuracy as the primary evaluation metric is also insufficient for a classification problem.

### Questions
1. Could you please provide a more detailed explanation of the physical insights behind Theorems 1 and 2, clarifying how they theoretically demonstrate that PVC mitigates the over-smoothing problem?

2. Could you please elaborate on the comparative advantages of PVC over direct temperature scaling, and include an experimental analysis of whether their performance difference is significant?

3. Could you please supply additional experimental evidence demonstrating how PVC alleviates over-smoothing, and present results on metrics such as precision and recall?

### Soundness
3

### Presentation
3

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
The paper introduces a method to solve the learning from the label proportions (LLP) problem, where we'd like to obtain an instance-level classifier given aggregated bag-level data. The proposed method claims to mitigate the oversmoothing problem in the proportion matching strategy. This work introduces a new algorithm for LLP accompanied by theoretical analysis and empirical validations.

### Strengths
1. The method reduces LLP to a bag-level classification problem, while prior work on proportion matching typically maps it to a bag-level classification problem.

2. Theorem 2 points out that the optimal solution of the proposed bag-level objective leads to an optimal instance-level classifier. Theorem 3 shows that the optimizer of the empirical objective converges to the optimizer of the objective as the number of bags grows. Theorem 2 and 3 essentially prove that the proposed algorithm indeed solves the LLP problem under the posed assumptions.

3. The theoretical framework is mapped to an implementable algorithm by an efficient computational strategy to identify some coefficients. Empirical evidence suggests that the proposed method outperforms several baselines on the benchmark datasets.

### Weaknesses
1. This work assumes all bags are i.i.d.. I don't think this is a realistic approximation of real-world problems. For example, in the election polling application in the introduction, I think the label proportions of two voting districts in the same city can be highly correlated. 

2. It is not clear to me how over-smoothing is mitigated by Theorem 1. Is there a 1-1 relationship between group-level and instance-level outputs, held for other proportion matching methods?

3. The bound (17) seems counterintuitive to me due to the presence of m in the numerator. If you have a fixed number of bags but with an increasing number of instances in each bag, the bound does not get tighter. This is basically saying you're not getting better convergence with more data allocated into a fixed number of bags. Does this reveal certain caveats of the proposed method? 

4. Only synthetic datasets are used in the experiments. While many LLP papers used to run experiments on synthetic datasets due to the lack of benchmarks, I think the situation is different with the new LLP benchmarks published in [1].

5. The bag size used in experiments is too small to demonstrate the scalability of the proposed algorithm. The largest bag size in [1] is 512. Papers like [2] even push the largest bag size to 2048. In the tables of [2], the accuracy of some methods rapidly drops after the bag size goes over a certain threshold; I'm wondering if this could also happen to the proposed algorithm.

6. It looks like all instances in the same bag must be pushed to the GPU simultaneously to compute the loss and backpropagate. This seems to be a huge computational disadvantage, especially if the bag sizes are large.

## References: 

[1]Brahmbhatt, A., Pokala, M., Saket, R., & Raghuveer, A. (2024). LLP-Bench: A large scale tabular benchmark for learning from label proportions. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24) (pp. 4374–4381). Association for Computing Machinery. https://doi.org/10.1145/3627673.3680032

[2] Zhang, J., Wang, Y., & Scott, C. (2022). Learning from label proportions by learning with label noise. In A. H. Oh, A. Agarwal, D. Belgrave, & K. Cho (Eds.), Advances in Neural Information Processing Systems. https://openreview.net/forum?id=cqyBfRwOTm1

### Questions
1. How important is it for different bags to have the same size m? I don't think Theorems 2 and 3 depend on a fixed m across all bags (Theorem 3 will probably have a different form due to different bag sizes). Can the authors confirm my observations?

2. While Theorem 2 is acceptable, I'm wondering if an excess risk bound like the styles in [1] can be developed. An excess risk bound could illustrate the sensitivity between $f_k$ and $g_k$ in Theorem 2. To be precise, if we obtained a $g_k$ that is not equal but very close to $g_k^\*$, would $f_k$ also stay close to $f_k^\*$?

3. How would the accuracy look if you fix the number of bags and keep increasing the bag sizes? If (17) is tight, you should not get better convergence.

4. See other questions in weaknesses.

I'm open to increasing the score if some of the concerns and questions can be addressed.


## References:

[1] Steinwart, I. How to Compare Different Loss Functions and Their Risks. Constr Approx 26, 225–287 (2007). https://doi.org/10.1007/s00365-006-0662-3

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
4

### Summary
This paper proposes a novel approach for Learning from Label Proportions (LLP) that may mitigate the over-smoothing problem common in proportion matching strategies with theoretical guarantees. The core contribution is to treat label proportions as labels for an auxiliary proportional value classification (PVC) task, which is then used to induce the instance-level classifier. The authors also introduce a "divide-and-conquer" computational method using FFT to solve the problem with an efficient time complexity. Experiments on text and image datasets show that the method achieves better performance against other LLP methods in settings with random bags with fixed size.

### Strengths
- **Novelty**: The core idea to solve a different, auxiliary problem, PVC, is creative and new for this field.
- **Efficiency**: The paper introduces a clever and practical solution using the FFT to make the computationally complex aggregation step efficient.
- **Theoretical quality**: The paper provides strong theoretical backing for the PVC approach. It includes formal guarantees showing how the method mitigates over-smoothing and connects the bag-level task back to the instance-level classifiers.
- **Clarity**: The paper is well-written and clear

### Weaknesses
1. **Mismatch between motivation and experimental setup**: The authors motivate the LLP problem with real-world applications like election polling (Section 1), which often involve complex, non-random bag generation (see [1] for a real-world application of LLP in election). However, the experimental setup (Section 4.1) relies exclusively on bags generated randomly with a fixed size. This setup only addresses one variant of LLP (termed "Naive" in the taxonomy proposed by [2]) and is unlikely to represent the more complex scenarios used as motivation. Even though using real-world data such as [1] is unfeasible, other works have explored more realistic bag creation methods, such as or clustering [3] to create dependence structures that are more realistic. The paper's claims would be significantly strengthened by testing against these more challenging and representative settings.

2. **Limited bag size exploration**: Even within the chosen random setting, it is known that LLP performance often degrades as bag size increases. The bag sizes tested in the paper are relatively small (a maximum of 128 for text and 32 for images, per Section 4.1). The claims about the method's robustness would be better supported by an analysis of its performance with much larger bag sizes, as seen in other recent work (e.g., up to 2048 in [5]).

3. **Missing baseline comparisons**: The evaluation is missing comparisons against several important and recent baselines, such as [4], [5], [6], and [7]. These methods are highly relevant to the task, and their omission makes it difficult to assess the proposed method's performance in the context of the current state-of-the-art. Notably, [5] is a recent ICLR 2024 publication, making its absence in the comparison tables particularly significant.

4. **Potentially unfair hyperparameter selection**: The paper states that "all methods shared the same hyperparameter settings" (Section 4.1) to ensure fairness. While using the same network architecture is reasonable, fixing hyperparameters like the learning rate (LR) for all methods (detailed in Tables 5 and 6) can unfairly disadvantage baselines, as different optimization methods often have different optimal LRs. A more robust comparison would involve minimal tuning (e.g., testing a small set of LR values) for each baseline and reporting its best performance. This would provide a stronger guarantee that the proposed method's superiority is not an artifact of a single, fixed hyperparameter choice that favors it over others.

**Minor details**: 
- $m$ is used in line 98 before its definition (line 142).
- Figure 1 labels are a bit confusing. I would suggest to have entropy and accuracy as two different colors, and have different methods with different markers.


[1]: Assessing Candidate Preference through Web Browsing History: https://dl.acm.org/doi/pdf/10.1145/3219819.3219884  
[2]: Dependence and Model Selection in LLP: The Problem of Variants: https://dl.acm.org/doi/pdf/10.1145/3580305.3599307  
[3]: (Almost) no label no cry: https://proceedings.neurips.cc/paper_files/paper/2014/file/d3313de3f431fd64513431c4326d237c-Paper.pdf  
[4]: Learning from Label Proportions with Consistency Regularization: https://proceedings.mlr.press/v129/tsai20a/tsai20a.pdf  
[5]: Learning from Label Proportions: Bootstrapping Supervised Learners via Belief Propagation: https://openreview.net/pdf?id=KQe9tHd0k8  
[6]: MixBag: Bag-Level Data Augmentation for Learning from Label Proportions: https://openaccess.thecvf.com/content/ICCV2023/papers/Asanomi_MixBag_Bag-Level_Data_Augmentation_for_Learning_from_Label_Proportions_ICCV_2023_paper.pdf  
[7]: Learning from label proportions by learning with label noise: https://proceedings.neurips.cc/paper_files/paper/2022/file/ac56fb3fab015124b541f6299016a21c-Paper-Conference.pdf

### Questions
1. Could you clarify the intended scope of LLP-PVC? The motivation (Section 1) suggests broad use, but the experiments (Section 4.1) use only random fixed-size bags. Do you expect this performance to generalize to non-random settings (e.g., clustered bags [3])?  
2. Why were experiments limited to small bag sizes (max 128)? Was there another bottleneck? Can you provide insight into the expected performance as bag size grows significantly?  From the literature, we expect the performance to decrease with the bag size increase.
3. Could you provide a brief conceptual comparison to recent baselines [4, 5, 6, 7], especially [5] (ICLR 2024)? What are the key methodological differences and trade-offs versus [5]?
4. Can you justify using a single fixed LR for all methods? This choice may unfairly disadvantage baselines. Is there evidence this choice doesn't create a confound, or that this LR is a reasonable default for all?

### Soundness
2

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
4

### Summary
This paper proposed a novel LLP approach that can mitigate the longstanding over-smoothing problem with only the incorporation of an aggregation function after the classification layer. An efficient computational approach with a divide-and-conquer strategy was also proposed.

### Strengths
1.	The proposed method is simple to implement, only involves with an auxiliary proportional value classification task.
2.	A divide-and-conquer method is adopted to reduce the computational complexity by using the fast Fourier transform, which largely reduces the computational cost.
3.	The theoretical work guarantees that the model outputs can mitigate over-smoothing problems under certain assumptions. It also analyses the convergence rate of the proposed risk estimator by providing an estimation error bound.

### Weaknesses
1.	The datasets and bag sizes for the image classification in the empirical study seems to be small. The authors should consider larger datasets with larger bag sizes, for example, CIFAR-100 and bag sizes of 32, 64, and 128.
2.	The actual running time reported in Table 3 is not consistent to the time complexity results.
3.	The reported results seem to be inferior to that is reported in former work. For example, CIFAR-10 in Table 2.
4.	Most of the compared methods are not state-of-the-arts.
5.	Conclusion is too weak and there is no limitation indicated.

### Questions
1.	What is the reason of the results with larger bag sizes are higher than that of smaller bag sizes for the proposed method, for example, results on K-MNIST in Table 2.
2.	Is there any hyperparamters sensitivity and ablation study should be investigated?

### Soundness
3

### Presentation
2

### Contribution
2
