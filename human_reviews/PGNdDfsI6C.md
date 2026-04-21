# LoRA vs Full Fine-tuning: An Illusion of Equivalence

- Avg Score: 4.75
- Decision: Reject
- Scores: 3, 5, 6, 5

## Abstract
Fine-tuning is a crucial paradigm for adapting pre-trained large language models to downstream tasks. Recently, methods like Low-Rank Adaptation (LoRA) have been shown to match the performance of fully fine-tuned models on various tasks with an extreme reduction in the number of trainable parameters. Even in settings where both methods learn similarly accurate models, \emph{are their learned solutions really equivalent?} 
To answer this, we study how different fine-tuning methods change pre-trained models by analyzing the model's weight matrices through the lens of their spectral properties. We find that full fine-tuning and LoRA yield weight matrices whose singular value decompositions exhibit very different structure; moreover, the fine-tuned models themselves show distinct generalization behaviors when tested outside the adaptation task's distribution. We first show that the weight matrices trained with LoRA have new, high-ranking singular vectors, which we call \emph{intruder dimensions}. Intruder dimensions do not appear during full fine-tuning. Second, we find that LoRA models with intruder dimensions, despite achieving similar performance to full fine-tuning on the target task, become worse models of the pre-training distribution and adapt less robustly to multiple tasks sequentially. Higher-rank, rank-stabilized LoRA models closely mirror full fine-tuning, even when performing on par with lower-rank LoRA models on the same tasks. These results suggest that models updated with LoRA and full fine-tuning inherently access different parts of the solution space, even when they perform equally on the fine-tuned distribution.
We conclude by examining why intruder dimensions appear in LoRA fine-tuned models, why they are undesirable, and how their effects can be minimized.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper empirically examines behavioral differences between LoRA and full fine-tuning, revealing that LoRA introduces *intruder dimensions* — singular vectors with low cosine similarity to the top $k$ singular vectors in the original weight matrix. Centered on this observation, the paper lists approximately ten empirical differences between LoRA and full fine-tuning.

### Strengths
The paper presents an interesting observation on behavioral differences between LoRA and full fine-tuning, followed by a series of experiments exploring various aspects of these observations.

### Weaknesses
This paper should be rejected because:

1. While the observation is interesting, the paper does not establish why the intruder dimension is significant. Counterexamples can be easily constructed to demonstrate that the intruder dimension may not materially impact the model. For instance, consider a 3-D space where the original weight matrix has two singular vectors, (1, 0, 0) and (0, 1, 0). Now consider a LoRA fine-tuned weight matrix with singular vectors (1, 1, 0) and (1, -1, 0). Both vectors are intruder dimensions, yet they span the same subspace as the original vectors and may not affect the fine-tuning process positively or negatively.

2. The paper points out some weaknesses of LoRA (e.g., full fine-tuning achieves a higher effective rank than LoRA), implying that the intruder dimension may be the root cause. However, the paper does not establish the connection between intruder dimension and effective rank, either theoretically or empirically.

3. The paper does not explore initialization, a key difference between LoRA (in which either $A$ or $B$ is randomly initialized) and full fine-tuning (which requires no initialization). Initialization could directly influence the intruder dimension, raising doubts about whether the empirical results stem from favorable initializations that support the paper's claims.

4. The experiments are narrowly focused, examining only cases where LoRA and full fine-tuning have the same accuracy, only sequence tasks, and only a limited range of hyperparameters, suggesting that the findings may be limited to a small niche rather than universally applicable.

Additionally, the paper has several confusing parts:

- The algorithm in Figure 4 mandates that num_intruders $\leq k$, yet Figure 5 shows num_intruders at O(100) while $k = 10$ .
- The results in Figure 8 do not support the claim; in some cases, full fine-tuning has lower accuracy than LoRA, and in others, full fine-tuning's accuracy is even lower than LoRA with a lower rank.
- The results in Figure 9 suggest there is always a LoRA rank where the pseudo-loss is lower than that of full fine-tuning. Does this mean that LoRA could be superior to full fine-tuning if the LoRA rank $r$ is correctly tuned?
- In some experiments, the paper uses $\alpha$ values that are not multiples of $r$, deviating from common LoRA practices. While these setups yielded interesting results, the paper may benefit from experimenting with different $\alpha / r$ ratios to provide insights into LoRA’s behavior in common configurations.

### Questions
1. Why should readers care about *intruder dimensions*?
2. What is the theoretical or empirical connection between intruder dimensions and effective rank?
3. How might initialization affect intruder dimensions?
4. Do we observe similar behaviors in more general cases, such as when LoRA and full fine-tuning differ in accuracy, for non-sequence tasks, or with different hyperparameters?
5. How are there O(100) intruder dimensions if Figure 4’s algorithm limits the number to $k=10$?
6. Does Figure 9 suggest that, with correctly tuned $r$, LoRA could achieve lower pseudo-loss than full fine-tuning?
7. How would behavior change if $\alpha / r$ is kept constant at a value other than 2?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper finds that full fine-tuning and LoRA yield weight matrices with distinctly different structures in their singular value decompositions, even though their downstream task performance is similar. The authors introduce *Intruder Dimensions* to analyze these differences, offering valuable insights from extensive experiments on LoRA training with varying ranks and $\alpha$ values. Additionally, they explore how *Intruder Dimensions* impact the model’s expressive power and generalization.

### Strengths
- The writing is clear, and the proposed method is well-motivated and sound.

- The authors introduce *Intruder Dimensions* to analyze the differences between LoRA and full fine-tuning, conducting extensive experiments to investigate this phenomenon. They present several findings regarding the generalization of LoRA fine-tuned models and offer suggestions, such as scaling $\alpha$ with rank and freezing A, to reduce *Intruder Dimensions*.

### Weaknesses
The main weakness of this paper is that, as an analysis-focused study, the exploration of *Intruder Dimensions* is somewhat **straightforward**. However, the analysis lacks depth; for example, in Section 5, the proposed hypotheses are not thoroughly explored theoretically. While the authors present several findings, they do not provide additional constructive improvement measures, limiting the paper’s practical contribution to the community.

### Questions
- When comparing LoRA to full fine-tuning, I suggest conducting additional experiments to analyze the impact of hyper-parameters. Specifically, for LoRA, variations in learning rate and total training steps can influence the final parameters. 
  - For instance, in Figure 7, adding a plot showing accuracy over training steps would be helpful.

- I recommend including some analysis of *Intruder Dimensions* for more LoRA variants, such as VeRA [r1], LoRA+ [r2], and FourierFT [r3], which would enhance the significance of the work.

  

  [r1] Kopiczko, Dawid Jan, Tijmen Blankevoort, and Yuki M. Asano. "VeRA: Vector-based Random Matrix Adaptation." ICLR 2024.

  [r2] Hayou, Soufiane, Nikhil Ghosh, and Bin Yu. "LoRA+: Efficient Low Rank Adaptation of Large Models." ICML 2024.

  [r3] Gao, Ziqi, et al. "Parameter-Efficient Fine-Tuning with Discrete Fourier Transform." ICML 2024.

### Soundness
3

### Presentation
3

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
This paper compares LoRA and full fine-tuning in adapting pre-trained large language models to downstream tasks. The study reveals that LoRA introduces "intruder dimensions" in the weight matrices, which do not appear in fully fine-tuned models. The paper provides empirical evidence for that (1) LoRA models with intruder dimensions tend to forget more of the pre-training distribution and adapt less robustly to multiple tasks sequentially compared to full fine-tuning; (2) Higher-rank LoRA models, which more closely resemble full fine-tuning, perform better in terms of generalization and robustness; (3) Intruder dimensions negatively impact model performance outside the fine-tuning task distribution.

### Strengths
1. The paper provides a novel insight into the differences between LoRA and full fine-tuning by analyzing the spectral properties of the weight matrices. The concept of "intruder dimensions" offers a new perspective on understanding these methods.
2. The paper not only identifies the presence of intruder dimensions but also analyzes their impact on model behavior, providing evidence for their detrimental effects.

### Weaknesses
1.While the paper provides empirical evidence for the existence and impact of intruder dimensions, the theoretical explanation for why they appear in LoRA models is still lacking.
2.The experiments are mainly conducted on sequence classification tasks and RoBERTa. The experiments on generative tasks seem insufficient and only preliminary attempts have been made, leading to doubts about whether the conclusions of this paper can be generalized to autoregressive models.
3.The authors focus on the original LoRA setting, while ignoring the other LoRA fine-tuning technology.

### Questions
1. Line 173: If the differences in training tasks lead to completely opposite conclusions with (Biderman et al., 2024), then the theoretical analysis of LoRA in this paper may not be comprehensive. Therefore, I am curious whether this phenomenon is caused by the degree of fitting or other parameters.
2. How do LoRA and full fine-tuning perform on more complex tasks like code generation or long-form text generation and autoregressive LLMs? Do the findings on sequence classification tasks generalize to these settings?
3. I am curious whether the phenomenon studied in this paper can be generalized to some recent LoRA fine-tuning techniques, such as QLoRA, Pissa, LoRA+, etc. If these commonly used training techniques do not encounter "intruder dimensions", the significance of this work will be greatly reduced.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
In this paper, to analyze the difference between LoRA fine-tuning and full-weight fine-tuning, the authors propose a new concept, intruder dimension, that measures the similarity of the column basis of singular-value decomposed matrices between the pre-trained weights and fine-tuned weights. Based on the concept of intruder dimension, the authors empirically analyze the difference by visualizing different models' performance and their number of intruder dimensions, evaluated on multiple datasets. The authors made several observations about LoRA's generalization ability to out-of-distribution datasets and drew several conclusions based on the observations. Finally, the authors compare the performance of continual learning abilities of different ranks of LoRA.

### Strengths
1. This paper provides a new concept for analyzing LoRA's different behavior. 
2. This paper has abundant empirical evidence showing the difference between the LoRA and full-weight fine-tune, based on the proposed concept.
3. The paper is organized in a straightforward way and can be easily comprehended.

### Weaknesses
- **The proposed concept "intruder dimension" is limited in its scope and cannot be readily extended to other scenarios.** 
  + According to Definition 1, the intruder dimension depends on the setting of the threshold $\epsilon$, which varies with different number of dimension (footnote 1, page 4). 
  + The concept of intruder dimension considers only the rotational effect of a linear mapping, and it holds only the weight increment $\Delta W$ is small compared to the pre-trained weights. To see this, consider a pre-trained square matrix $W\_0=U\_0 \Sigma\_0 V\_0^T$ (it can also be extended to non-square matrix), and do the SVD to the weight increment as following: $\Delta W = U \Sigma V^T = U\_0 Q \Sigma Q^T V\_0^T$, where $Q$ is a orthonormal (rotation) matrix. Then the fine-tuned matrix can be represented by: $W\_0 + \Delta W = U\_0 (\Sigma\_0 + Q \Sigma Q^T) V\_0^T$. The transformation in between the orthogonal basis $(\Sigma\_0 + Q \Sigma Q^T)$ is the one we actually want to analyze. Doing a further SVD to this will give us $(\Sigma\_0 + Q \Sigma Q^T) = P \Sigma^\prime P^T$. The similarity matrix on which the **intruder dimension** then can be calculated by the matrix product $(U\_0P)^T U\_0=P^T$. It can be seen as the effective rotational transformation learned by the $\Delta W$ and no consideration is paid to the scaling effect, i.e., $\Sigma^\prime$. 
  + Following the second bullet, I don't see the necessity of proposing a whole new concept for analyzing this rotational transformation $P$, as we have well-established inner-product for two different orthonormal matrices, e.g., Frobenius inner-product $<P, Q>=tr(P^TQ)$. And in this case, $tr(P)$. Could you please explain why do we need to define such a new concept? 
- I'm confused about the conclusion of this paper: **is intruder dimension the reason for worse OOD generalization or the consequence of LoRA?** It would be good to confirm the answer to this question. Note it brings a huge difference: 
  + If it's the cause then ideally we can actively mitigate this issue by limiting the intruder dimension, i.e., restricting the rotation transformation of LoRA and we should expect a gain in performance (see SVDiff [1] and AdaLoRA [2] for reference). 
  + If it's just an observation, which I think would be more possible (but excuse my bias), then it also makes sense and there is nothing surprising about it: as LoRA has limited size of the parameters, altering the scaling of the transformation might not be enough to fit the downstream data, and hence more rotational transformation are included, which is consistent with another observation in the paper: "The total number of intruder dimensions increases proportionally to the size of the fine-tuning dataset."
- **Chapter 4 is a bit off the topic**: I don't see the connection between the intruder dimension and continual learning in this case study. And it seems quite established that lower rank of LoRA tends to forget more. At the same time, **Chapter 5 is not in-depth enough**: the conjectures given are straightforward and does not inspire new understanding of LoRA and intruder dimension. I would like to see more experiments that specifically target on the assumptions made in Chapter 5.

**References**
- [1] Han, Ligong, et al. "Svdiff: Compact parameter space for diffusion fine-tuning." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
- [2] Zhang, Qingru, et al. "AdaLoRA: Adaptive budget allocation for parameter-efficient fine-tuning." arXiv preprint arXiv:2303.10512 (2023).

### Questions
My main questions are included in "Weaknesses".

### Soundness
2

### Presentation
3

### Contribution
2
