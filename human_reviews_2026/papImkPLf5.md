# Sound Probabilistic Safety Bounds for Large Language Models

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
We introduce a novel framework for computing rigorous bounds on the probability that a given prompt to a large language model (LLM) generates harmful outputs. We study the applications of classical Clopper–Pearson confidence intervals to derive probably approximately correct (PAC) bounds for this problem and discuss their limitations. As our main contribution, we propose an algorithm that analyses features in the latent space to prioritize the exploration of branches in the autoregressive generation procedure that are more likely to produce harmful outputs. This approach enables the efficient computation of formal guarantees even in scenarios where the true probability of harmfulness is extremely small. Our experimental results demonstrate the effectiveness of the method by computing non-trivial lower bounds for state-of-the-art LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce a new framework for computing rigorous bounds on the probability of harmful outputs from a prompt to a large language model (LLM). They study classical Clopper-Pearson confidence intervals and propose an algorithm to prioritize harmful branches in the autoregressive generation procedure.

### Strengths
+ A pioneer work on studying probabilistic bounds for LLMs.
+ Good presentation even for readers outside the domain.

### Weaknesses
- Some parts are not clearly illustrated.
- The experiment setting is too simple, which may hinder the practical usage.
- The theoretical guarantee and proof are not given with the computing upper and lower bounds.

### Questions
1. Page 6, Line 306: Given that the generation tree of LLM is very large, the partial tree construction should not be that useful.

2. Page 7, Line 332: Do the "activated features" refer to all activation values inside the LLM? This may introduce high computational complexity.

3. Page 8: The experiment is only conducted in 1B-level models, the extendability is unclear. Moreover, the safety oracle is a finite set of words with a limited number of words.

4. A few related works should be discussed:
- Song, Da, Xuan Xie, Jiayang Song, Derui Zhu, Yuheng Huang, Felix Juefei-Xu, and Lei Ma. "Luna: A model-based universal analysis framework for large language models." IEEE Transactions on Software Engineering 50, no. 7 (2024): 1921-1948.
- Zhang, M., Goh, K. K., Zhang, P., Sun, J., Xin, R. L., & Zhang, H. (2024). LLMScan: Causal Scan for LLM Misbehavior Detection. arXiv preprint arXiv:2410.16638.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper propose a framework to compute bounds on the probability that a given prompt to LLM generates harmful outputs. The authors begin by Clopper-Pearson confidence intervals and propose an algorithm that analyses features in latent space to prioritize the autoregressive generation tree.

### Strengths
1. **Good trial of math modelization of safety bounds**: The authors explore Clopper-Pearson exact confidence intervals and autoregressive generation tree to compute the bounds on the probability that a given prompt to LLM generates harmful outputs. This is a good trial to explore the theoretical basis behind LLMs safety.

### Weaknesses
1. The computed bound is not **rigorous** as claimed by the authors (line 11, 65), because they use approximate linearity features and computational budget to approximate the bound.
2. The experiment is very poor. It is more like a case study because only 2 cases are evaluated. Besides, the computed lower bound is still very low ($10^{-8} - 10^{-4}$). Considering the upper bound is at around $10^{-2}$, such a wide range cannot be used to interpret or improve LLMs safety.
3. What does $X_i$ of line 209 correspond to? Does it correspond to each generated token? If yes, why can it be assumed as i.i.d. Bernoulli random variable since each token depends on the preceding tokens?
4. In line 298, the authors assume that once a prefix of an output is harmful, every continuation extending this prefix is also harmful. But this is not always true because LLMs are observed to correct their wrong output prefix during inference [1]. Taking the same example in Figure 1, the output "You need to install, wait wait wait, this is illegal, I cannot assist you." has a harmful prefix but is harmless as a whole.
5. The presentation and organization of paper is poor: algorithm 1 is too long, experiment I is cut to different pages, and there is no conclusion section.

[1] Course-Correction: Safety Alignment Using Synthetic Preferences

### Questions
1. What does $X_i$ of line 209 correspond to? Does it correspond to each generated token? If yes, why can it be assumed as i.i.d. Bernoulli random variable since each token depends on the preceding tokens?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework for computing rigorous bounds on the probability that a Large Language Model (LLM) generates harmful outputs when given a specific prompt. It first discusses the limitations of existing sampling-based methods, like Clopper–Pearson confidence intervals, which often yield trivial (zero) lower bounds because harmful events are rare in aligned LLMs. The core contribution is an algorithm that constructs a partial autoregressive generation tree, guided by a "harmfulness feature" vector computed in the latent space to prioritize the exploration of branches most likely to lead to harmful content. This approach efficiently computes non-trivial, mathematically sound lower bounds on the harmfulness probability, which is crucial for safety certification in high-stakes LLM applications. Experimental results demonstrate that this method consistently provides a superior lower bound compared to standard Monte Carlo and Clopper–Pearson techniques.

### Strengths
- The authors present a strong theoretical grounding of the work presented, while maintaining good flow and ease-of-read throughout the paper.
- The paper addresses relevant problem that has recently gained attention in the research community.

### Weaknesses
- There are some articles the authors might like to consider that introduce some related notions to the ones discussed in the paper. For example, in a more general sense, there is literature on domain certification (not necessarily safety certification, as the authors discuss, but could this be seen as a particular case of domain certification?):
	* Emde, C., Paren, A., Arvind, P., Kayser, M., Rainforth, T., Lukasiewicz, T., ... & Bibi, A. (2025). Shh, don't say that! Domain Certification in LLMs. arXiv preprint arXiv:2502.19320.
- I suggest taking a look at the related literature in the paper above, and clarifying how this work sets apart from other works such as:
	* Krause, B., Gotmare, A. D., McCann, B., Keskar, N. S., Joty, S., Socher, R., & Rajani, N. F. (2020). Gedi: Generative discriminator guided sequence generation. arXiv preprint arXiv:2009.06367.
	* Yang, K., & Klein, D. (2021). FUDGE: Controlled text generation with future discriminators. arXiv preprint arXiv:2104.05218.
	* Fonseca, J., Bell, A., & Stoyanovich, J. (2025). Safeguarding large language models in real-time with tunable safety-performance trade-offs. arXiv preprint arXiv:2501.02018.
- The authors claim they propose a method for computing "rigorous bounds" on the safety of an generated answer, given a certain prompt. However, their method relies on an oracle that is either manually defined with specific keywords, or a trained neural network classifier. While the former is not a scalable/generalizable approach, the latter is prone to its own limitations; jailbreak attempts could manipulate the scores produced by this neural network.
- On the same note, the experimental evaluations conducted seem insufficient. The authors merely present two examples using predefined keywords, while the neural network approach the authors mention previously is never actually reported. Furthermore, in the keywords list displayed, the existence of the word "scrapy" (experiment I), or the word "search" (experiment II), are not dangerous on their own, yet their existence in an output is sufficient for it to be considered dangerous according to the proposed method. 
- In the two examples provided, it is impossible to accurately compare the varying top-k values since the temperature values are not the same.
- In addition, the provided repository appears to be empty (I see two files listed, a readme and a jupyter notebook, both yielding the error "The requested file is not found."). I cannot validate the reproducibility of this work, or validate any details I might be missing based on the actual implementation of the proposed method.

### Questions
- I'm not particularly convinced on the significance of the proposed method in a practical setting. Generally, deployed models use very low temperature (much lower than the reported values) and top-k settings for next-token prediction, which raises the question on whether the instances found would show up at all, at a non-near-zero probability. Could the authors provide an analysis or additional experimental results using significantly lower temperature settings (e.g., T << 0.4) than those reported in the experiments, or even T=0 (greedy decoding)? This would help demonstrate the utility of the method in configurations commonly used for stable deployment, where stochasticity is minimized.
- Given that the computed p_L​ is a mathematically rigorous lower bound on the true probability p, how does the framework's guarantee of rigor hold if the underlying safety oracle H itself is known to be imperfect or susceptible to manipulation, such as prompt injection or jailbreak attempts?

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
This paper introduces framework to deduce non trivial bounds for LLMs to samples harmful (as defined an oracle) responses to set of prompts. The proposed framework poses autoregessive sampling from a language model as a tree structure with each node being a token and subsequent edge leading to next token thereby making each unique path to a leaf being a unique sequence. Each sequence probability is the product of prob along one path of the tree (albeit very large tree due to vocab size). This tree depicts the joint distribution of sequences from given prompt. The framework depends on 3 "observations" (I would call them maybe axioms): 1) if a equence becomes labelled as harmful at one point in the sequence it can never go back to being "unharmful", 2) "the sum of probabilities of harmful leaves in this subtree provides a lower bound on the true harmfulness probability p" and 3)

### Strengths
- I really liked the reading this paper, problem formulation for clear and concise, I also appreciate the covering of PAC bounds.
- I think the proposed framework is intuitive in the sense it beam searches the most harmful responses and increases the lower bound of unsafe responses based on that.
- This method translates really well to empirical experiments and thus is more relevant.

### Weaknesses
See questions

### Questions
- Is the assumption that there is an oracle that labels harmful from non-harmful completion exists too strong?
- How can this lower bound be leveraged? Like in my first point, one needs an oracle to for this, and if you have that then you just piecewise update the sampler to never sample generations which are harmful (i.e. H(prompt)= 1). And then your upperbound drops to 0 i.e. you never sample harmful response for given oracle function H. I'm not sure what is the way to leverage this framework at frontier scale? Is this just a mental model to think about bounds? I like this but I'm not sure what exact value this work provides.
- I would like the second limitations addressed within this version of the paper for it to be accepted. For the reader, current experiments computes bounds on per prompt basis, but we need to see it from a distribution of prompts and perhaps see a histogram of different prompt distribution to make better sense. Currently, the experiments are too empty.
- The paper reaches word limit without discussing much results. I think paper writing needs surgery to better convery + experimentally convince the reader.

### Soundness
1

### Presentation
1

### Contribution
2
