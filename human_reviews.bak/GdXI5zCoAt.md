# RaSA: Rank-Sharing Low-Rank Adaptation

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Low-rank adaptation (LoRA) has been prominently employed for parameter-efficient fine-tuning of large language models (LLMs). However, the limited expressive capacity of LoRA, stemming from the low-rank constraint, has been recognized as a bottleneck, particularly in rigorous tasks like code generation and mathematical reasoning. To address this limitation, we introduce Rank-Sharing Low-Rank Adaptation (RaSA), an innovative extension that enhances the expressive capacity of LoRA by leveraging partial rank sharing across layers. By forming a shared rank pool and applying layer-specific weighting, RaSA effectively increases the number of ranks without augmenting parameter overhead. Our theoretically grounded and empirically validated approach demonstrates that RaSA not only maintains the core advantages of LoRA but also significantly boosts performance in challenging code and math tasks. Code, data and scripts are available at: https://github.com/zwhe99/RaSA.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a method to improve the expressivity of LoRA, while keeping the same number of parameters. This is by sharing a portion of each learnable adaptor across all layers.

### Strengths
* The authors tackle an important and meaningful topic that would be of interest to the community. PEFT is gaining more and more attention as the size of language models continues to grow.
* The description of the proposed method is clear and concise. Writing is easy to understand and follow.
* The method is simple, effective, and easy to implement.

### Weaknesses
* It seems that unlike vanilla LoRA, that acts independently on each layer, in RaSA there is an underlying assumption that all shared layers have the same dimension. If so, it is worth mentioning it explicitly in the paper.
* There are LoRA extensions that allow allocating a different rank for each layer, according to its importance (for example PRILoRA and others). It seems that in RaSA the rank should be similar across layers. If so, it is worth mentioning this fact and citing the relevant papers that do allow different rank allocation.
* There are no comparisons in Table 1 to other LoRA extensions (AdaLoRA, PRILoRA), which may be of importance to the reader. Furthermore, it is not clear why the VeRA method is compared against, when its number of parameters is x13 less.

### Questions
* In Figure 1b, the name of matrices A,B are in the square, and the dimension is outside. However, in matrix D the name is outside and there are no shapes. Maybe you want to consider putting the name D_i inside, and adding shape, for readability.  
* In line 208, it says: ‘actual weight updates from model fine-tuning’. Do you mean ‘full fine-tuning’ (FFT)?
* In Table 1, why do you show both LAST and BEST? Why does BEST not suffice?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Low-rank adaptation (LoRA) is limited in expression ability due to low rank constraints. This paper propose Rank-Sharing Low-Rank Adaption (RaSA) to enhance the expressive capacity of LoRA by leveraging partial rank sharing across layers. Specifically, RaSA shares $k$ ranks across all layers and assign $r-k$ for layer-specific parameters. After that, RaSA greatly increase the effective rank of the parameter update by $(L-1)\times k$ without introducing additional parameters. The experimental results on multiple datasets with different model architectures validate the effectiveness of the proposed approach.

### Strengths
1. The problem studied in this paper is valuable. 
2. The proposed RaSA approach is interesting and effective.
3. The paper is well organization. 
4. The theoretical and empirical analysis are insightful. 
5. The experiment is sufficient, including the study of Reconstruction error, the study of $k$, the performance comparison of parameters and time, the study of forgetable.

### Weaknesses
A great paper with almost no weaknesses. A small suggestion, the distinction between the different methods in Figures 4, 5, and 7 is only based on different color (red and blue), as printing on paper cannot distinguish them.

### Questions
I am curious why the performance is best when $k$ is around $r/8$ (line 249-250), and if possible, is there any theoretical explanation?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper highlights that the low-rank constraint limits the expressive capacity of LoRA, and that LoRA’s parameters are not fully utilized. To address this, the authors propose RaSA, an approach enabling partial rank sharing across layers with minimal additional parameters. Theoretical and empirical analyses demonstrate that RaSA achieves a minimum reconstruction error bounded by LoRA, and can be easily optimized. Experiments show that RaSA achieves great performance in complex math and code tasks.

### Strengths
- RaSA can efficiently increase the rank of the original LoRA, and compared to directly expanding the rank, RaSA achieves this with almost no increase in parameter count.
- The authors validate the better expressive capacity of RaSA through detailed reconstruction error analysis, empirical analysis shows that RaSA can achieve much lower reconstruction error and reveal the relationship between rank and the hyperparameter k.
- Experiments on code and math tasks show that RaSA achieves much better performance with only a 10% increase in time, highlighting RaSA's promising advantages.

### Weaknesses
- As written in Lines 30-31, LoRA still lags behind full fine-tuning, particularly on complex tasks, however, the authors only compare RaSA with existing peft methods, not including full fine-tuning.

### Questions
- In line 103, the authors claim that introducing a trainable diagonal matrix enables layer-specific weighting, actually, I am confused about this, maybe it could be explained specifically.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an enhancement to existing fine-tuning methods for large language models (LLMs). It builds on Low-rank adaptation (LoRA), commonly used for efficient fine-tuning, but limited by its low-rank constraint in handling complex tasks like code generation and mathematical reasoning. To address this, the authors propose Rank-Sharing Low-Rank Adaptation (RaSA), a novel method that increases expressive capacity by sharing ranks across layers, allowing for more flexibility without adding parameter overhead. RaSA thus extends LoRA’s efficiency, achieving substantial performance gains in rigorous tasks. Code and data resources accompany the work.

### Strengths
1. Technical Soundness: RaSA is a technically robust extension of LoRA, overcoming its expressive limitations through partial rank sharing. Theoretical and empirical results confirm its lower reconstruction error and improved task performance.

2. Paper Presentation: The paper is clearly organized and easy to follow, with well-explained motivations, methods, and contributions, making RaSA's innovations accessible and engaging.

3. Theoretical Analysis: Theoretical bounds and empirical evidence validate RaSA’s improved capacity over LoRA, providing a solid foundation for its enhanced performance in high-rank tasks.

### Weaknesses
1. Lack of Baseline Comparisons:
The paper lacks a comprehensive comparison with other LoRA variations, such as EM-LoRA, AdaLoRA, and Orthonormal LoRA. Without these baselines, it’s difficult to fully assess the advantages of RaSA, as the performance improvements over the original LoRA may not necessarily hold when compared with these alternative methods.

2. Efficiency Concerns:
The proposed method is considerably less efficient than the original LoRA, with a noticeable increase in computational time (at least 1.4 hours). Despite this added overhead, the performance gains are minimal, with PASS@1 improvements generally below 2%. This limited payoff questions the practical viability of RaSA, especially for real-world applications where computational efficiency is crucial.

### Questions
1. Could the efficiency of the proposed method be further optimized to reduce computational time?

2. The performance improvement appears to be notably greater on Mistral-0.3-7B compared to Llama-3.1-8B. Do you have any insights into the reasons behind this difference?

### Soundness
2

### Presentation
2

### Contribution
3
