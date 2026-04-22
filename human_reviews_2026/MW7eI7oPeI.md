# Interactive Implicit In-context Learning

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6

## Abstract
In-context learning (ICL) empowers large language models (LLMs) to generalize from few-shot demonstrations but faces challenges with quadratic computational complexity and unstable performance as demonstration counts increase. Implicit ICL (I$^2$CL) addresses these limitations by encoding demonstrations into a unified task vector injected during inference, achieving significant efficiency gains and order invariance. However, existing approaches prioritize inference efficiency, sacrificing interactive capabilities crucial to standard ICL—inter-demonstration and demonstration-query interactions. Substantial research underscores that these interactions are fundamental for strong reasoning performance. To bridge this critical gap, we propose Interactive Implicit In-Context Learning (I$^3$CL), a simple yet effective framework that restores such essential capabilities of standard ICL. I$^3$CL integrates only two lightweight interactive modules, preserving I$^2$CL's core efficiency with minimal computational overhead. Experimental evaluation across nine diverse tasks using four LLMs show that I$^3$CL  delivers a significant average performance gain of 7.26\% over prior I$^2$CL baselines. This substantial improvement strongly indicates that restoring interactive capabilities is essential for advancing the effectiveness of implicit ICL methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes $I^3CL$ based on the previous I2CL framework, with some module modification focusing on the anisotropy of demonstration representations (i.e., context vector). Experiment results show that $I^3CL$ is a new SotA among the hidden state steering methods on zero-shot inputs.

### Strengths
1. The two issues of the prior work (I2CL), i.e., no demonstration interaction and the naive averaging problem, pointed out by this paper, are reasonable. Therefore, the paper proposes $I^3CL$ to intuitively address these limitations.
    
2. The experiments show that the proposed $I^3CL$ method achieves SotA performance among implicit ICL methods. Moreover, the ablation analysis confirms the effectiveness of its components.

### Weaknesses
1. Many claims or assumptions remain unverified, and some heuristic designs lack sufficient motivation.
    
    1. Line 084: “exchange, contrast, and complement information to form a more expressive collective representation” is not directly supported by experimental evidence. I expect a direct measurement of such expressive ability.
        
    2. Line 195: “we compute a representation by averaging two specific last-token representations: the last token of the context (prompt) part and the last token of the label part” is a heuristic design. I can not get the motivation for extracting the tokens from these two positions and averaging them, nor convincing experimental results to justify the necessity of doing so. Specifically, one can expect that the last token of the label already encodes the preceding demonstration information, so extracting only the label token’s hidden states and MLP outputs might suffice. You might conduct more ablation experiments to verify whether extracting both the label token and the demonstration token is truly necessary.
        
    3. Line 089: “enabling the model to selectively emphasize the most informative demonstrations” is also not directly supported by experimental evidence. At least, I would like to see a case study of the selected demonstrations by the softmax processing. Also, another concern is raised on this point, see below.
        
2. The demonstrations are weighted using softmax and cosine similarity.
    
    1. This essentially reproduces a demonstration selection process, which can benefit not only from the similarity but also from the diversity of demonstrations [1]. Since evaluating diversity does not significantly affect computational efficiency, I suggest that the authors incorporate diversity into their demonstration-to-query interaction process. Furthermore, the authors could even consider parameterizing these softmax processes, since gradient-based optimization is already employed for efficient calibration, adding a few more parameters would be acceptable.
        
    2. It is conceivable that, in classification tasks, a demonstration similar to the query is likely to have the same label as the query. Directly copying the hidden states of this consistent label into the zero-shot inference naturally biases the output toward that label, which may cause your method to degenerate into a KNN retriever operating on embedded representations. An intuitive way to address this concern is to conduct experiments on non-classification tasks, as described below.
        
3. The experiments are insufficient. The proposed method is clearly not limited to classification tasks, yet this paper only evaluated it on classification datasets. Moreover, although four models are listed in the paper, this paper conducted a complete evaluation on only one of them. This undermines the credibility of their approach.
    
4. This work is incremental. Although I believe the improvements made in this paper are reasonable, it still largely follows the framework of Li et al. Personally, I do not object to incremental work, but I am unsure whether it fits the standards of ICLR. I therefore raise this point for the AC’s consideration, though it does not affect my main score.
    
5. Some writing issues, see Question.
    

[1] Which Examples to Annotate for In-Context Learning? Towards Effective and Efficient Selection. [https://arxiv.org/abs/2310.20046v1](https://arxiv.org/abs/2310.20046v1)

### Questions
1. Table 5 (Fig. 5 in your notation): As far as I understand, compared with $I^3CL$, each step in I2CL uses a simpler module variant (e.g., average pooling instead of self-attention). Why, then, is its time cost greater than that of $I^3CL$?
    
2. Line 218: I did not fully understand what “global collective regularities” specifically refers to. I can imagine that you might be referring to something similar to a “task representation,” but I would like to know what these “regularities” actually are in numerical terms. Therefore, you may need to provide a more detailed analysis comparing the task vector extracted by I2CL with that of $I^3CL$, highlighting their semantic differences.
    
3. Line 414: What does the “highest weight” refer to? Is the weight the norm of the hidden state?
    
4. Line 456: What does “noise” refer to here?

5. Line 617: Typo.

### Soundness
1

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
4

### Summary
The paper introduces Interactive Implicit In-Context Learning (I3CL), a new framework designed to improve upon existing implicit in-context learning (I2CL) methods. The authors identify two weaknesses of I2CL: 1) I2CL processes each demonstration in isolation, ignoring the valuable relationships between demonstrations; 2) It uses simple average pooling to create the task vector, treating all demonstrations as equally informative for any given query. I3CL solves these problems by introducing two lightweight attention mechanisms for 1) inter-demonstration interaction and 2) demonstration-query interaction, to improve the analytical capabilities of I2CL. 

I3CL weighs the in-context vectors and injects them into the LLM's latent space using learnable weights to guide the final prediction. Experiments are conducted on classification tasks and various LLMs, showing that I3CL significantly outperforms I2CL and even surpasses the performance of fine-tuned LoRA models, while maintaining comparable inference efficiency to I2CL.

### Strengths
- The problem background, motivation, and formulation are clear and easy to understand.
- The paper identifies an interesting problem, connecting the literature on importance attribution of in-context demonstrations and in-context vectors to form an elegant solution.
- The method is grounded on well-defined mathematical formulations, showing its rigor and interpretability.
- There is some empirical backing of the superiority of the proposed method across datasets and LLMs.

### Weaknesses
- The datasets are somewhat limited in scope. All experiments are conducted on text-classification tasks, which is too simplistic. It would be great to see experiments on QA tasks using expert trajectories as demonstrations, similar to [1]. Or, at least some non-classification tasks, for example used in [4]
- While it is interesting to connect importance weighting of in-context demonstrations with I2CL, the literature review lacks references to previous works in importance weighting of in-context demonstrations, for example [2,3].
- I hope the authors provide a more detailed explanation of how to interpret Eq. 6 as the "importance weight" of in-context vectors.
-  As far as I understand the paper, Eq. 8 and Eq. 9 follow from Eq. 4 and Eq. 5 of the I2CL paper. I encourage the authors to properly reference the paper when introducing these two equations.



[1] Monea et al., LLMs Are In-Context Bandit Reinforcement Learners, COLM 2025.

[2] Zhou et al., DETAIL: Task Demonstration Attribution for Interpretable In-context Learning, NeurIPS 2024.

[3] Yang et al., Representative Demonstration Selection for In-Context Learning with Two-Stage Determinantal Point Process, EMNLP 2023.

[4] Todd et al., Function Vectors in Large Language Models, ICLR 2024.

### Questions
- The calibration process seems to require some task demonstrations for training. Does the calibration process take place during inference or is it conducted during a "boostrapping" stage?
- The implementation of I3CL seems to be more complicated than I2CL. I wonder if the authors can provide some explanation for why the inference speed is "near-parity". It would be helpful if the authors could offer some speed comparison between I3CL and I2CL.
- For tasks that require the model to generate multiple tokens, do the in-context vectors need to be injected at every autoregressive decoding step?


I am open to raising the score should the authors sufficiently address my concerns.

### Soundness
2

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
4

### Summary
This paper introduces Interactive Implicit In-Context Learning (I3CL), a framework that enhances Implicit In-Context Learning (I2CL) by incorporating two lightweight attention mechanisms. The first is inter-demonstration interaction during encoding, which allows allows demonstrations to exchange and contrast information. The second is demonstration-to-query interaction during inference, which dynamically weights demonstrations according to their relevance to the query. Experiments in this paper show that I3CL consistently outperforms existing baselines.

### Strengths
1. Clear motivation and well-articulated weaknesses of I2CL: This paper points out that independent encoding and uniform averaging used in I2CL obscure relational and relevance information between demonstrations, providing an intuitive and well-founded motivation for I3CL.

2. Simple and intuitively reasonable design: The addition of inter- and cross-attention modules is conceptually simple, lightweight, and easy to integrate with existing I2CL pipelines.

3. Comprehensive and strong empirical results: Experiments in this paper show that I3CL consistently outperforms existing baselines.

### Weaknesses
1. Although this paper provides clear figures that effectively illustrate the high-level idea of their method, the technical details still require more precise explanations. For example, in lines 211–214, the authors mention adopting a self-attention mechanism, but the exact formulation of this mechanism is not provided — are there any trainable parameters involved? In addition, the loss function defined in Section 3.4 and the procedure for training the learnable coefficients could also be described in greater detail. 

2. In lines 183–186, each demonstration requires the additional task instruction I. Is this design necessary? Could it be redundant or repetitive, and would it cause significant additional computational or memory overhead when the number of demonstrations becomes large?

### Questions
1. What's the exact formulation of the self-attention mechanism in lines 211–214? Are there any trainable parameters involved? If there are  trainable parameters, how are they trained?

2. In lines 192-205, the authors mentioned that they used two specific last-token representations while prior works only used one. Why choose these two specific tokens? Could the authors provide some explanation? Is the improved performance of authors' method compared to previous approaches partly due to the additional two token representations?

### Soundness
3

### Presentation
3

### Contribution
3
