# OrderDP: A Theoretically Guaranteed Lossless Dynamic Data Pruning Framework

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Data pruning (DP), as an oft-stated strategy to alleviate heavy training burdens, reduces the volume of training samples according to a well-defined pruning method while striving for near-lossless performance. However, existing approaches, which commonly select highly informative samples, can lead to biased gradient estimation compared to full-dataset training. Furthermore, the analysis of this bias and its impact on final performance remains ambiguous. To address these challenges, we propose OrderDP, a plug-and-play framework that aims to obtain stable, unbiased, and near-lossless training acceleration with theoretical guarantees. Specifically, OrderDP first randomly selects a subset and then chooses the top-$q$ samples, where unbiasedness is established with respect to a surrogate loss. This ensures that OrderDP conducts unbiased training in terms of the surrogate objective. We further establish convergence and generalization analyses, elucidating how OrderDP affects optimal performance and enables well-controlled acceleration while ensuring guaranteed final performance. Empirically, we evaluate OrderDP against comprehensive baselines on CIFAR-10, CIFAR-100, and ImageNet-1K, demonstrating competitive accuracy, stable convergence, and exact control---all with a simpler design and faster runtime, while reducing training cost by over 40\%. Delivering both strong performance and computational efficiency, our method serves as a robust and easily adaptable tool for data-efficient learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes OrderDP, a method for dynamic data pruning. OrderDP first randomly selects a subset, and then chooses top-q samples based on the loss function value. Paper shows convergence and generalization bounds for OrderDP. OrderDP outperforms other static and dynamic baselines in terms of accuracy for CIFAR10/100, and ImageNet.

### Strengths
- The paper is well written.
- OrderDP provides both bias and convergence analysis, as well as a generalization error bound.
- OrderDP outperforms both static and dynamic pruning methods across all settings.
- Additionally, paper also shows the method works with different optimizers.

### Weaknesses
Clarity: 
- I believe it could benefit the paper if the authors were more clear when writing things like “near-lossless”, for example, do they consider near-lossless when the relative accuracy drop is within 1% or something different?
- I also believe it could benefit the paper to clarify sooner that OrderDP can work with various optimizers.

For further questions and weaknesses see Questions.

### Questions
1. Does OrderDP end up seeing the entire dataset (since there is a uniform sampling at random s samples)? If yes, could you clarify how long it takes to see the entire dataset depending on the pruning ratio? 

2. [1] have shown that depending on the size of the dataset one should keep either hard or easy examples. Does this finding hold also for OrderDP? Is there a pruning ratio at which one should sample min-q scores?

3. Could you clarify how you implement dynamic random? For example, [2] shows that a simple modification of the random baseline can further improve its results. Furthermore, how does your convergence and generalization bound compare to [2]?

4. The paper does a good job in showing the practicality of OrderDP by showing end-to-end wall clock time, and the convergence analysis. However, I believe it could further strengthen the paper to plot its time-to-accuracy, that is, accuracy on the y-axis and wall-clock time on the x-axis.

5. How does OrderDP perform in situations with noisy labels, could you elaborate whether sorting by top-q could be misleading or not in that scenario? 

I look forward to the responses from the author(s) and would reconsider my score based on the clarifications and revisions provided. Thanks.

[1] Sorscher, Ben, et al. "Beyond neural scaling laws: beating power law scaling via data pruning." Advances in Neural Information Processing Systems 35 (2022): 19523-19536.

[2] Okanovic, Patrik, et al. "Repeated Random Sampling for Minimizing the Time-to-Accuracy of Learning." The Twelfth International Conference on Learning Representations.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents OrderDP, a dynamic data pruning framework that reduces training costs by selectively using informative samples during neural network training. Unlike previous approaches that address gradient bias through importance reweighting (which increases variance and gradient norms), OrderDP takes a different strategy: rather than seeking unbiased estimates of the original loss, it defines a surrogate loss function for which the method provides provably unbiased gradient estimates. Experiments on CIFAR and ImageNet datasets show OrderDP achieves competitive accuracy specially in the challenging high compression regime. The authors also provide theoretical guaranties for convergence and generalisation.

### Strengths
Overall I found this to be an interesting work that addresses the gradient bias induced by data pruning from a different and, to my knowledge, novel angle. The empirical results indicate strong performance of this methodology compared to an extensive list of competing methods, especially in the challenging high compression regime. The theoretical analysis brings interesting guarantees and is an appreciated plus to the methodology.

### Weaknesses
- I find the title somewhat overstated. From my understanding, the theoretical results do not establish that OrderDP is a guaranteed lossless framework. I would interpret such a statement as $\mathcal{L}(\theta^\star_{unpruned})-\mathcal{L}(\theta^\star_{pruned})$ converging to 0 with $n$ for all pruning ratios $r$, which can be proved for unbiased pruning methods using reweighting, for example (see [1]). Such a result would be interesting to include in this framework as well, if it holds.

- OrderDP is presented as a plug-and-play framework, which could be understood as being applicable to any base data scoring method, but the paper only deals with one specific score based on the loss function.

[1] Ayed, Fadhel, and Soufiane Hayou. Data pruning and neural scaling laws: fundamental limitations of score-based algorithms.

### Questions
- The algorithm is described for the general (random) mini-batch SGD setting used in the theoretical analysis. However, it is not clear how this translates to the empirical training setup, where batches are predefined, and the model cycles through them rather than resampling. Is the “exploration” and “exploitation” sampling and selection performed within each batch ?
- A related question concerns the definition of the compression ratio $r = 1 - q/n$, which depends only on $q$ and makes sense for the mini-batch SGD setting. In the empirical setup, however, this seems to differ. For instance, lines 407–408 and Section 5.3 suggest that $r$ depends on both $q$ and $s$.
* On lines 316–317, $s$ and $q$ are defined as fractions (0.5 and 0.6) rather than integers as in the rest of the paper. What exactly are these fractions referring to, and why is $q$ larger than $s$ ?

Minor:
- Lines 240–241: Could the authors provide a proof that the (\gamma_j) sum to 1? This would be preferable to empirical validation alone.
- Regarding the statement that “the analysis of the bias and its impact on the final performance remains ambiguous,” it may be useful to consider [1], which provides such an analysis and first proposes a reweighting-based exploration–exploitation correction (to my knowledge).

### Soundness
3

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
This paper presents OrderDP, a novel framework for dynamic data pruning aimed at addressing the biased gradients and training instability inherent in existing methods. The core idea is a two-stage selection process: uniformly sampling a candidate pool, then selecting the top-q highest-loss examples for the gradient update. The key innovation is theoretical: rather than fixing the bias for the original objective, the authors introduce a surrogate loss function and prove that their method provides an unbiased gradient estimate for this new objective. This allows them to establish formal convergence and generalization guarantees. Empirically, OrderDP demonstrates superior performance and stability on CIFAR and ImageNet benchmarks, achieving significant training speed-ups with minimal to no accuracy loss.

### Strengths
1.	The paper introduces a novel two-stage sampling method that combines uniform random sampling for exploration with a top-q, high-loss selection for exploitation. This straightforward design effectively balances the need for data diversity with the selection of informative samples and is simple to integrate into existing training pipelines.

2.	Theoretical Analysis The work is supported by a relatively thorough theoretical analysis, which is a key differentiator from many heuristic-based approaches. By introducing a surrogate loss function, the authors reframe the biased pruning problem into an unbiased optimization of a new objective. This allows them to establish formal convergence rates and generalization bounds.

### Weaknesses
Major concern: 
1.	The authors motivate their work by highlighting the critical issue of biased gradient estimation in existing dynamic data pruning methods, where the bias is defined with respect to the full-dataset empirical risk, L(θ). However, the proposed OrderDP method does not directly yield an unbiased estimator for the gradient of L(θ). Instead, it cleverly reframes the problem by introducing a surrogate loss, L_q(θ), for which the computed gradient is indeed an unbiased estimator.This constitutes a subtle but significant mismatch between the initial problem formulation and the solution. The central premise of the paper shifts from 'fixing the bias for the original objective' to 'optimizing a different, tractable objective'. The validity of this entire approach hinges on the crucial assumption that minimizing L_q(θ) is a sufficiently good proxy for minimizing L(θ).While the authors attempt to bridge this gap through the generalization analysis in Theorem 4, which bounds the bias between the two objectives, the paper would be strengthened by a more direct discussion of this methodological pivot.

2.	The authors make a strong claim of achieving 'loss-less' performance, which is central to the paper's positioning. The empirical results, however, suggest that this claim holds under specific, moderate pruning ratios (e.g., 30% on CIFAR-10) but not universally across more aggressive pruning settings. This discrepancy between the strong, general claim and the nuanced, conditional results could be misleading. The authors should consider refining their claim to be more precise, perhaps by using terminology like 'near-loss-less' or by explicitly stating the conditions under which true 'loss-less' performance is achieved. This would align the paper's claims more accurately with its empirical evidence.

### Questions
1. Can the authors clarify how minimizing the surrogate loss $L_q(\theta)$ reliably approximates minimizing the original objective $L(\theta)$? Are there empirical results or theoretical analyses demonstrating when this approximation may fail?

2. Would the authors consider explicitly discussing the methodological shift from bias correction for $L(\theta)$ to optimization under $L_q(\theta)$, and how this affects the paper’s claimed contributions?

3. Regarding the ``loss-less'' claim, could the authors specify the exact pruning conditions under which this property holds, and whether ``near-loss-less'' might be a more accurate description?

### Soundness
4

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
The paper proposes a dynamic data pruning framework, termed OrderedDP, to speed up the model training process by reducing the number of samples needed for each epoch. The author proposes to use the loss to measure the importance of the data sample together with a ranking method, where a larger loss indicates a higher demand for retention. The experiments were conducted on CIFAR-10/100 and ImageNet-1k using ResNet-50 as the model backbone. The authors have done some analysis on the generalization of the proposed method.

### Strengths
1. The motivation of the paper is well justified. Also, the authors try to give some theoretical analysis of the proposed method to justify its generalization capability.

2. The paper is properly written and it is not hard to follow the paper's story.

### Weaknesses
1. The selected backbone is not strong enough. The reproduced accuracy of the ResNet-50 backbone is lower than the model trained with a stronger training recipe and data augmentation strategy. Actually, it is important to justify that the proposed method is compatible with state-of-the-art training recipes. It is not clear if the "unimportant samples" are indeed not that important with a stronger data augmentation strategy and learning schedule.

2. The experiments are only verified on classification tasks. It is unclear whether the proposed method can be applied to other tasks, such as segmentation, detection, or generative tasks.

3. The method is only verified on CNN-based networks (ResNet-18/50), which typically show faster convergence speed over transformer-based models. It is also not clear how the proposed method performs on transformer-based model architectures.

### Questions
Please refer to the weakness session.

### Soundness
3

### Presentation
3

### Contribution
2
