# DNT: a Deeply Normalized Transformer that can be trained by Momentum SGD

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Transformers have become the de facto backbone of modern deep learning, yet their training typically demands an advanced optimizer with adaptive learning rate like AdamW, rather than a momentum SGDW (mSGDW). 
Previous works show that it is mainly due to a heavy-tailed distribution of the gradients. 
In this paper, we introduce a Deeply Normalized Transformer (DNT), that is meticulously engineered to overcome 
the heavy-tailed gradients issue, enabling seamless training with vanilla mSGDW while yielding comparable performance to the Transformers trained via AdamW. 
Specifically, in DNT, we strategically integrate normalization techniques at proper positions in the Transformers to effectively modulate the Jacobian matrices of each layer, balance the influence of weights, activations, and their interactions, and thus enable the distributions of gradients concentrated. 
We provide both theoretical justifications of the normalization technique used in our DNT and extensive empirical evaluation on two popular Transformer architectures (\ie, ViT and GPT), validating that: a) DNT can be effectively trained with a vanilla mSGDW; and b) DNT outperforms its counterparts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript targets the demand for advanced optimizers with adaptive learning rates in the training of Transformers. A deep normalization transformer is proposed, in which the heavy-tail gradient problem is overcome by strategically integrating different normalization techniques into the appropriate positions of the Transformer.

### Strengths
The idea is timely, clearly presented, theoretically analyzed, and empirically evaluated on two popular Transformer architectures.

### Weaknesses
However, key clarifications and stronger validation are necessary. The detailed comments are as follows:

1. Based on the normalization techniques mentioned in Figure 4, is the contribution of this manuscript merely an engineering improvement regarding the strategic integration of different normalization techniques? What is the logic of cooperation between them? And how to determine the appropriate positions of each technology in the Transformer?

2. In Tables 2-4 of Appendix C, is the selection of parameters specified by the authors or the relatively optimal settings obtained through search? In the experiment, key hyperparameters and design choices were lacking in ablation.

3. There are inconsistencies or errors in the reference writing, such as:
"Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, et al. Symbolic discovery of optimization algorithms. Advances in neural information processing systems, 36, 2024."
should be:
"Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, et al. Symbolic discovery of optimization algorithms. Advances in neural information processing systems, 36:49205-49233, 2023."
It is suggested that authors should carefully check, revise and improve.

4. In line 426, "See Appendix H for the training parameters." In fact, "Appendix H" does not exist in the manuscript.

5. In the manuscript, "the norm of * is very large", "the norm of * is large" and "* become too large" are mentioned times. How to define or quantitatively describe "very large", "large", or "too large"?  
	
6. What is the definition of σ(∙) in lines 303-304 σ(W_1 ) and σ(W_2 )? In Equation (4), "Y = Self-Attention(X′), where x′ = PreNorm(x) ". However, what kind of variable is X′? Or is there any connection between X' and x'? Furthermore, in line 256, "Y = Self-Attention(X) and Y' = Self-Attention(X')". Is it accurate that it is different from the "Y = Self-Attention(X')" in equation (4)?
	
7. In order to facilitate readers' understanding, the authors should introduce and explain the alphabetic symbols and various operation symbols that appear in the manuscript. For example: The definition of "⨂" in the equation on lines 241-242 and in Equation (5) needs to be declared. What is "C" in Equation (5)? The "⨀" in line 254 needs to declare its definition. Furthermore, x_0, which appears in lines 372-373, seems not to have been used. So, where does it play any role? What parameter is "C_dn" in the formula on lines 730-731? What is "C" in Equation (13)?

8. There is a cross-reference error in "according to Equation ??" in line 754. Please check and confirm whether other citations are standardized.
	
9. In lines 759-758, there are ∂L/(∂vec(Y)) and  ∂L/(∂vec(W_q)),   ∂L/(∂vec(W_k)),   ∂L/(∂vec(W_v)), where L and L should be the same?
	
10. The captions in Figures 7-10 are not consistent with those of the other figures mentioned earlier in the manuscript.
	
11. What are the dimensions W_1 and W_2 that appear in Equation (6) respectively? In line 274, does the W_1 x>0 in "W_2 diag(1(W_1 x>0)) W_1" represent the relationship between the vector W_1 x and 0? How exactly is it defined? Furthermore, what kind of calculation is 1(W_1 x>0)? The authors should supplement the corresponding definitions or calculation methods.
	
12. The "if" in line 754 should be "If".

### Questions
see above section for detailed information

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
3

### Summary
This paper addresses a key limitation in training Transformers: the poor performance of momentum SGD (mSGD) compared to adaptive optimizers like AdamW. The root cause is identified as the heavy-tailed distribution of gradients in Transformers, which causes uneven updates across parameters.

To resolve this, the authors propose Deeply Normalized Transformers (DNT), which strategically apply normalization at specific positions in the architecture to modulate Jacobians, balance weight and activation contributions, and reduce the heavy-tail behavior of gradients. Theoretical justifications are provided for why these normalization choices improve training with mSGD. Empirically, DNT achieves performance comparable to AdamW on ImageNet (Vision Transformers) and OpenWebText (GPT) while requiring less memory and computation, thanks to the ability to use mSGDW.

### Strengths
1. The paper addresses a well-known problem in training Transformers, namely why momentum SGD (mSGD) tends to fail compared to adaptive optimizers like AdamW. It provides a detailed theoretical analysis showing how the placement of normalization layers affects the conditioning of Jacobian matrices and the variance of gradients, explaining why these adjustments are crucial for stable training. 

2. The approach is practical, leveraging existing normalization techniques in new positions without introducing additional components. Empirically, the proposed Deeply Normalized Transformer (DNT) matches AdamW performance on ImageNet and OpenWebText, with gradients that are more concentrated and stable under mSGDW. 

3. Using mSGDW also reduces memory and computational requirements compared to AdamW, which is a significant practical advantage. 

4. Finally, the paper offers analytical insights linking the structure of normalization to optimizer behavior, providing a deeper understanding of the interplay between architecture and training dynamics.

### Weaknesses
1. The paper has several limitations regarding the scope and scale of its experiments. It evaluates the proposed Deeply Normalized Transformer (DNT) only on two benchmarks, ImageNet and OpenWebText, and does not include large-scale or multimodal tasks. This narrow evaluation makes it difficult to assess how well the method generalizes to other domains or to the training of state-of-the-art large models.

2. Another limitation is the increased complexity introduced by multiple normalization placements. While these placements are key to stabilizing mSGD, they also add implementation overhead and require careful hyperparameter tuning.

3. The paper also lacks comparisons with other recent approaches designed to improve stability in Transformers, such as nGPT, Stable Transformer, or LipsFormer. Similarly, there is no evaluation against newer optimizers like Muon, which could provide important context for the relative benefits of DNT and mSGDW (Muon is closer to SGD than Adam).

4. Finally, the evaluation on GPT2-small is somewhat limited in scale and may not reflect the challenges of training modern large language models. The optimizer shows instabilities in some experiments, and the loss gap between mSGDW and AdamW remains non-negligible, which could be a critical concern for training larger or more complex architectures.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Deeply Normalized Transformer (DNT), a Transformer architecture designed to be effectively trained with momentum SGD (mSGDW) instead of adaptive optimizers like AdamW. The authors identify that Transformers exhibit heavy-tailed gradient distributions, which make SGD-based optimizers unstable. DNT addresses this by inserting or repositioning normalization layers (InputNorm, PreNorm, MidNorm, and QKNorm) to modulate the Jacobian of each block, ensuring more concentrated gradient distributions and reducing training instability.

### Strengths
1. Theoretical justification on how each normalization position affects the Jacobian and stabilizes gradient magnitudes.
2. Empirical results showing that DNT trained with mSGDW performs comparably to standard Transformers trained with AdamW, both on ImageNet (ViT) and OpenWebText (GPT2).

### Weaknesses
1. Theoretical assumptions are idealized: The high-dimensional isotropy and orthogonality assumptions may not hold exactly for real Transformer activations.
2. Similar ideas appear in nGPT, StableTransformer, and Lipsformer (which are cited), but the novelty claim is modest—it’s mostly a systematic integration and justification rather than a new normalization method. 
3. The comparison is primarily between mSGD and AdamW. It would be more compelling to see how it performs against other optimizers like Sophia or Lion.
4. While comparing mSGDW and AdamW, the paper does not clarify if both optimizers were optimally tuned (learning rates, weight decay, warmup). The authors state "we did not tune the learning rate too much". However, the hyperparameter tables (e.g., Table 2, 4) show that the settings for mSGDW and AdamW are vastly different. For instance, L-DNT-Small uses LR=1.0 for mSGDW versus LR=6e-4 for AdamW , and V-DNT-Large uses LR=0.5 for mSGDW versus LR=1e-3 for AdamW.
5. In addition, There's also no evidence that DNT maintains benefits under fine-tuning, transfer, or longer training schedules.

### Questions
1. Could you include ablations isolating the impact of each normalization (InputNorm, PreNorm, MidNorm, QKNorm) individually on gradient statistics and performance?
2. How much GPU memory and wall-clock time are saved when training with mSGDW compared to AdamW?
3. How does DNT differ conceptually from “Transformers without normalization” (Zhu et al., 2025)? Could these approaches be unified?

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
4

### Summary
This paper proposes the Deeply Normalized Transformer (DNT), a Transformer variant designed to address the heavy-tailed gradient problem that hinders the performance of vanilla momentum SGD (mSGDW) in training Transformers. The authors provide theoretical analysis connecting the heavy-tail issue to the Jacobian matrices of different Transformer components and argue that strategically introducing normalization at specific positions (InputNorm, PreNorm, MidNorm, and QKNorm) can stabilize gradients and mitigate this issue. DNT is evaluated on both Vision Transformers (ViT) and GPT architectures, showing that it can be trained effectively with mSGDW to reach performance comparable to AdamW. The paper includes gradient distribution visualizations and empirical results on ImageNet and OpenWebText benchmarks.

### Strengths
- The paper tackles a practical and important problem—reducing dependency on adaptive optimizers like Adam—by improving Transformer architectures to work with simpler optimizers. 

- The theoretical analysis provides a clear connection between normalization placement and Jacobian conditioning, offering intuition for the design of DNT. They also did a comprehensive analysis of different normalization techniques.

- Experimental results across both vision and language models demonstrate that DNT narrows the performance gap between mSGDW and AdamW, suggesting potential for simpler and more efficient training pipelines.

### Weaknesses
- The experiments, while promising, are limited in scale (e.g., GPT2-Small/Large, ViT-Large) and lack validation on larger models or diverse datasets to confirm robustness.
- The empirical novelty is modest, as the approach primarily reorganizes existing normalization techniques rather than introducing new mechanisms.
- The paper does not include detailed ablation studies to isolate the contribution of each normalization type, which would strengthen the empirical validation.

### Questions
- How does DNT scale when applied to very large LLMs (e.g., tens of billions of parameters)? Are there architectural or stability issues?
- Could the authors clarify whether the performance parity with AdamW holds when hyperparameters (e.g., learning rates, momentum) are more extensively tuned for mSGDW?
- How does DNT interact with modern optimizers such as Muon—does normalization reduce or amplify their benefits?
- What are the computational overheads introduced by the additional normalization layers in large-scale training?
- It appears that mSGDW with DNT generally performs worse than AdamW during the early stages of training—could the authors provide an explanation for this behavior?

Typo:
- line 158: any a forward layer

### Soundness
3

### Presentation
3

### Contribution
3
