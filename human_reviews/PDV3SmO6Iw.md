# SORSA: Singular Values and Orthonormal Regularized Singular Vectors Adaptation of Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5

## Abstract
In this paper, we propose Singular Values and Orthonormal Regularized Singular Vectors Adaptation, or SORSA, a novel PEFT method. Each SORSA adapter consists of two main parts: trainable principal singular weights $W_p = U_p \text{diag}(S_p) V^\top_p$, and frozen residual weights $W_r = U_r \text{diag}(S_r) V^\top_r$. These parts are initialized by performing singular value decomposition (SVD) on pre-trained weights. Moreover, we implement and analyze an orthonormal regularizer, which we prove could decrease the condition number of $W_p$ and make the optimization more efficient. SORSA adapters could be merged during inference, thus eliminating any inference latency. We also introduce a method to analyze the variation of the parameters by performing SVD and discuss and analyze SORSA's superiority in minimizing the alteration in the SVD aspect. After all, SORSA shows a faster convergence than LoRA and PiSSA in our experiments. On the GSM-8K benchmark, Llama 2 7B adapted using SORSA achieved 56.03\% accuracy, surpassing LoRA (42.30\%), AdaLoRA (47.30\%), Full FT (49.05\%), and PiSSA (53.07\%). On the MATH benchmark, SORSA achieved 10.36\% accuracy, outperforming LoRA (5.50\%), AdaLoRA (6.48\%), Full FT (7.22\%), and PiSSA (7.44\%). We conclude that SORSA offers a new perspective on parameter-efficient fine-tuning, demonstrating remarkable performance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors propose Singular Values and Orthonormal Regularized Singular Vectors Adaptation (SORSA) for parameter-efficient fine-tuning (PEFT). They use an analysis of singular values and vectors to show the limitation of existing works and propose to solve it via orthonormal regularization. Theoretical analysis shows that the condition number of the low-rank incremental matrix is improved via regularization, resulting in enhanced stability. The superior performance of SORSA is validated on three NLG tasks compared to other PEFT methods.

### Strengths
1. While orthonormal regularization on incremental low-rank matrices has been used in existing works, the analysis of the condition number is novel to me, and the improved stability makes sense.
2. The approach is simple and effective, as it directly applies orthonormal regularization.

### Weaknesses
1. The method is not well-motivated. The authors begin with an analysis of singular values and vectors, which shows a different updating pattern of SORSA compared to other methods. However, it is unclear how this is connected to the limitation of the generalization ability of LoRA and FT. I can only observe a limitation of the learning capacity of LoRA and FT. Additionally, it is not clear why the orthonormal regularization leads to a different updating pattern, as shown in Figure 2, and why this pattern can give an improvement. I suggest a further theoretical justification for this point.
2. There seems to be a misuse of terminology: the authors use FT to denote "partial fine-tuning" (for example, page 2, line 86). However, in the literature, FT often denotes "full fine-tuning" (for example, in DoRA [1]). If the analysis in Sec. 3.2 is truly inspired by DoRA, the authors may want to compare the updating patterns of "full fine-tuning, LoRA, and SORSA" instead of "partial fine-tuning, LoRA, and SORSA." If I misunderstood, could you provide a definition of partial fine-tuning early in the main paper?
3. The experimental comparison is not extensive enough in terms of benchmarks. Only NLG tasks are used to evaluate the method, and it's not clear whether SORSA works in other NLP tasks. More experiments on other tasks, such as the common GLUE benchmark [2] in natural language understanding (NLU), are expected.
4. It seems SORSA (w/o reg) is essentially the same as PiSSA as described in Related Work, so I encourage the authors to use consistent terminology throughout the paper. If they are different, please give a clear explanation of the differences. Also, in this case, quantitative ablation studies are missing since the only comparison between SORSA and SORSA (w/o reg) is in Figure 2. The performance of SORSA (w/o reg) should also be provided in Table 1.



[1] Shih-yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-
Ting Cheng, and Min-Hung Chen. DoRA: Weight-Decomposed Low-Rank Adaptation.
In Forty-first International Conference on Machine Learning, June 2024.

[2] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman.
Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv
preprint arXiv:1804.07461, 2018.

**Minor comments**

1. Figure quality can be improved. For example, the font size should be larger for better readability.

### Questions
1. Is there any insight into the "Grad Norm" figures in Figure 3?

### Soundness
2

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
4

### Summary
The SORSA method proposed in this paper is a variant of LoRA—a technique for parameter-efficient fine-tuning of pre-trained large language models (LLMs). SORSA combines concepts from two previously proposed LoRA variants: AdaLoRA (Zhang et al., 2023) and PiSSA (Meng et al., 2024). In the original LoRA method, each weight matrix $W$ in an LLM is adapted as $W+AB$, where $AB$ represents the product of two low-rank matrices, $A$ and $B$, which are the trainable parameters. AdaLoRA adapts the weight matrices as $W+USV^T$, where $U$ and $V$ are randomly initialized low-rank matrices, and $S$ is a zero-initialized diagonal matrix. During training, $U$ and $V$ are regularized to form orthonormal bases in their columns, so that $USV^T$ resembles the singular value decomposition (SVD) of a matrix. Similar regularization has been shown to speed up training in many previous works.

PiSSA, on the other hand, directly decomposes each pre-trained weight matrix using SVD as $W=USV^T$. During fine-tuning, only the parameters associated with the largest singular values and the corresponding singular vectors (selected columns of $U$ and $V$) are updated, while the remaining parameters remain fixed. PiSSA does not use any orthogonality-enforcing regularization during fine-tuning. SORSA, the method proposed in this paper, essentially applies PiSSA but regularizes the fine-tuned vectors (selected columns of $U$ and $V$) to stay nearly orthonormal, as in AdaLoRA.

### Strengths
The paper reports superior performance for LLMs fine-tuned with SORSA compared to the PiSSA method. Additionally, the proposed approach is relatively easy to implement. The originality of this work lies in combining the ideas behind AdaLoRA and PiSSA, as described in the "Summary" section.

### Weaknesses
Firstly, the paper suffers from poor writing quality, with numerous grammatical errors and awkward English expressions (too many to list exhaustively in this review). The text is also poorly structured, with some mathematical symbols left undefined. The figures use such small font sizes that they are almost unreadable, even in the electronic version where one can zoom in. More detailed feedback is provided below. The paper appears to be written by an inexperienced author, so I would recommend seeking assistance from someone with more experience and strong English skills to review and refine the text.

Besides the standard full fine-tuning and LoRA, the paper compares the proposed SORSA method only with PiSSA. However, since SORSA builds on AdaLoRA’s concepts, I believe AdaLoRA (and possibly other related methods like OLoRA or DoRA) should also be included in the comparisons. Furthermore, the experimental results are limited compared to other referenced works (e.g., the PiSSA paper).

The author’s limited experience is further evident in presenting some trivial findings as if they were major contributions. For example, Section 4 refers to Appendix A, where readers are promised an optimized and highly efficient version of SORSA. However, Appendix A merely states the obvious: that multiplying by a diagonal matrix is more efficient than naively multiplying by a full matrix with zeroed-out off-diagonal elements. The appendix also includes an analysis of the speedup achieved by this "optimized" implementation, but it yields nonsensical results that do not align with the $O(N^2)$ versus $O(N^3)$ complexity of the optimized and naive approaches. Similarly, the statement that the regularizer is convex and Lipschitz continuous is rather obvious, although the corresponding derivations are reasonably included in an appendix.

### Questions
The paper should clarify its relationship to AdaLoRA and PiSSA in the "Related Works" section (or even the Introduction), which it currently does not.

There appear to be inconsistencies in notation with symbols like $V_{[:r]}$, $U_{[:r,:r]}$ and $V_{[:r,:]}$ in Section 2. Additionally, "range" notation is introduced only in Section 4.

Section 3.1 explains SVD, but this explanation is not well-written and is confusing. Given that SVD is a standard technique, it is unnecessary to explain it to the paper's target audience. Simply introducing the notation should suffice.

What is $\Sigma$ in Section 3.1? It is not defined. Is it the diagonal matrix of singular values? The paper already uses $S$ for the vector of singular values.

It may be clearer to use $\mathrm{diag}(S)$ instead of $\mathrm{diag}(W)$ at the end of page 3 and the symbol $W$ is already dedicated to weight matrices.

Section 3.2, titled "Analysis Method," does not actually describe any analysis method. Instead, it introduces metrics that express the deviation of the singular values and vectors of the updated weight matrices from those of the original pre-trained matrices. However, metrics alone do not constitute an analysis method. Moreover, these ad-hoc metrics and the conclusions drawn from their behavior in Figure 2 are speculative, with no analysis showing a direct correlation between these metrics and fine-tuning performance.

"$\Delta\Sigma_t$ represents singular value variants between $W_0$ and $W_t$"
should perhaps read
"$\Delta\Sigma_t$ represents the distance/difference between singular values of $W_0$ and $W_t$"

Section 3.3 compares the behavior of metrics from Section 3.2 for various fine-tuning methods, including SORSA. However, since SORSA has not yet been introduced, readers may not understand why certain behavior is expected. Therefore, this analysis should appear later in the paper, possibly in the appendix, as it is speculative and not essential for understanding the main advantages of the proposed technique.

"significant adjustments in significant vectors" should read "significant adjustments in singular vectors"

"parallel updating pattern across weights in different layers, which emphasizes a restriction of these methods"
I would say that it shows that there is not any restriction on updating parameters in all layers.

"indicating that the updates in the SORSA are less constrained"
Since the changes in singular values and vectors are smaller for SORSA, it suggests that the updates are actually more constrained, as SORSA uses orthonormality regularization as an additional constraint.

"matrix that preserves the largest significant values and vectors, containing the matrix’s most significant data"
Again, "significant" should likely be "singular". What does it mean "the matrix’s most significant data"? This is very poor description of what the largest singular values and the corresponding singular vectors represent.

"which consist $W_p$" should perhaps read "which constitute $W_p$" ... and similarly for $W_r$.

Equation (10) presents an implementation detail that seems unnecessary for inclusion in the main paper.

What is $k$ in Equation (11)? Without defining it, Equation (11) implies that $\gamma$ is smaller than any arbitrary number.

"... SORSA, we present a novel analysis of its condition number"
In (7), SORSA is presented as a function, so it does not have a condition number. The matrix $W_r+W_p$ does.

What are $\sigma_i^{unreg}$ and $\sigma_i^{reg}$ in Equation (12)? Are they the singular values of the fine-tuned models? The derivation of Equation (12) in the appendix is difficult to follow. Where does Equation (25) originate? What do $W_p^{unreg}$  and $W_p^{reg}$ represent in this equation—the parameters of the regularized and unregularized fine-tuned model? Equation (25) implies that the Frobenius norm of the differences between in trained matrices $W_p^{unreg}$  and $W_p^{reg}$ equals the scaled Frobenius norm of the gradient of the regularizer with respect to a $W_p$ matrix. Which $W_p$ matrix? What point is the gradient evaluated at? Equation (25) is completely unclear.

The symbols  $\delta_{1,t}$ and $\delta_{2,t}$ are not introduced in Theorem 5. Likewise, symbols $\kappa$, $\sigma_{max}$ and $\sigma_{min}$ are not introduced, though their meanings may be guessed.



**===End of the original review ====**

Here, I provide response to the Authors comments on the review as I was not able to respond by the official means:


The paper has been significantly improved in the current updated version. It is much easier to follow thanks to its improved structure and presentation. Additionally, results for AdaLoRA have been added, making the results more convincing. Accordingly, I am increasing my scores for the paper.

However, several issues remain. Some of these were already pointed out in the reviews but have not been addressed in the updated paper:

Multiple reviewers requested clarification on the relationship between SORSA, AdaLoRA, and PiSSA. Specifically, we asked for acknowledgment that SORSA is primarily a combination of ideas from these two methods. This relationship (especially with AdaLoRA) is still not clearly stated in the paper.

One reviewer pointed out that "SORSA (w/o reg) is essentially the same as PiSSA." I agree with this assessment, and it should be explicitly stated in the paper. While I understand that PiSSA merges singular values with the matrices of singular vectors, this correspond to only different parametrization of the same model and should not significantly affect optimization (as you confirm in your response). You mentioned that you avoided explicitly stating this to prevent confusion, but I believe that not clearly acknowledging the equivalence of "SORSA (w/o reg)" and "PiSSA" is actually more misleading.

I still find the "Singular Values and Vector Analysis" section speculative and of limited scientific value. As I noted in my previous review, "the ad-hoc metrics and the conclusions drawn from their behavior in Figure 2 are speculative, with no analysis showing a direct correlation between these metrics and fine-tuning performance." Why must we examine differences between weight matrices in terms of singular values and singular vectors separately? What do the patterns in Figure 2 teach us? Low-rank approximation using SVD minimizes the Frobenius norm of the difference between the original matrix and the approximating matrix, so why not simply measure similarity using the Frobenius norm? Wouldn't this metric be equally expressive? For example, the last graph in Figure 2 shows that in some layers, only singular values change while singular vectors remain constant. Does this observation have any meaningful implications? There is no analysis provided to support its relevance.

You state:

"Figure 2 shows that almost all FT, LoRA, and SORSA layers without a regularizer exhibit synchronized and linear-like updating in singular values and vectors. This shows that all layers are 'locked' with each other, which I interpret as 'restriction.' Although SORSA uses one additional regularizer during training, Figure 2 actually demonstrates its more 'free' updating (evident in how different layers can update more independently)."

What does it mean that "layers are 'locked' with each other"? Are you suggesting that updates are correlated across layers? There is no measurement of such correlations in the paper. Nor do you demonstrate that parameters are not updated independently in FT and LoRA. Figure 2 merely shows that all parameters across all layers are updated in FT and LoRA, while only some layers are effectively updated in SORSA. Perhaps stronger (L2 or L1) regularization toward the original weight matrix would achieve the same effect in LoRA.

I strongly suggest removing the section on the "optimized version of SORSA," even from the appendix. The optimization you describe is the only reasonable way to implement multiplication with a diagonal matrix, something anyone with basic algebra knowledge and programming skills would do by default. Admitting that you considered converting the diagonal matrix to a dense one and multiplying with it is embarrassing. Moreover, what you call element-wise multiplication in your equation is not truly element-wise, as the vector must first be broadcast. Finally, as I noted in my previous review, Figure 4 does not illustrate the quadratic vs. cubic complexity as claimed in the text.

The practical purpose of Equation (11), defining $\gamma$ remains unclear. I had hoped your derivation in the appendix would clarify this, but it did not. You state:

"$k$ will be a constant. This essentially implies $\gamma$ should be inversely proportional to $n_d$"

Here, $n_d$ is a constant hyperparameter (maximum learning rate), and you define $\gamma =k / n_d$, where $k$ is another hyperparameter. Why should $k$ be tuned instead of directly adjusting $\gamma$?

It is still unclear what the singular values in Equation (12) represent. Are they the corresponding values (regularized and unregularized) from the respective updates? They do not depend on any iteration index $t$ in the formula.

Minor Issues:

The figures, especially Figure 2, use fonts that are too small to read, although the authors promise to improve them.
Use diag(S) instead of diag(W), as W is already used to denote a weight matrix, not a vector.
 
The subsection titled "Analysis Method" should be renamed to something like "Measuring Similarity Between Singular Values and Vectors," as the current title does not accurately reflect its content.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces SORSA that is a parameter-efficient fine-tuning (PEFT) method designed to adapt large language models (LLMs) for downstream tasks. The experimental results demonstrate  that it can converge faster than PiSSA and LoRA and achieves the higher accuracy on benchmarks like GSM-8K and MATH.

### Strengths
1. The proposed method outperforms the other PEFT techniques such as LoRA and PiSSA in terms of accuracy on various benchmarks, showcasing its effectiveness. The results look very promising on a variety of experiments.
2. The authors also analyze the variation patterns of singular values and vectors during parameter updates and compare SORSA with other PEFT methods such as LoRA and partial fine-tuning.

### Weaknesses
1. The noverty of this paper is limited. the initialization method is from Pissa [1], and updates in the form of singular value decomposition and the orthonormality regularizer are from AdaLoRA [2].
2. Some symbols in Theorem 3 and Theorem 5 are used without the previous definition, which can be confusing. It is better restate these theorems to make them more straightforward.
3.  Proof of Theorem 2 is not right. Line 1035-1036. "This L is finite because the Frobenius norms
of U and V are bounded (they represent orthonormal matrices in the ideal case)". This statement not ritght. You need to add the condition that Frobenius norms of U and V are bounded. $\mathcal{L}_{reg}$ is Lipshitz only when Frobenius norms of U and V are bounded.
4. Theorem 3 is questionable. Related workes on convergence of optimizers for transformers are not clear. They only give some preliminary results under some strict assumptions and settings. But Therorem 3 needs nothing. The proof of Theorem 3 is based on Eq (21), howerver, it has no basis. 
5. Due to my limited time, I haven't checked all the lemmas and theorems, but they do have some problems. I strongly recommend that all authors carefully examine the theoretical analysis to make sure it is correct.
6. In table 1，the result of RWKV6 7B on GSM-8K for LoRa is only 8.04%. This is very strange (too
low). Make sure the experiment setting is correct. Since SORSA is similar to Pissa and AdaLoRA. It is better to campare with the results of AdaLoRA as well.
[1] Fanxu Meng, Zhaohui Wang, and Muhan Zhang. PiSSA: Principal Singular Values and Singular
Vectors Adaptation of Large Language Models.
[2] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao.  Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. ICLR 2023.

### Questions
The same questions as given above in the section of Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
