# The Information Bottleneck of Chain-of-Thought and How Latent CoT Overcomes It

- Decision: Reject
- Scores: 4, 8, 6, 2, 8

## Abstract
Chain-of-thought (CoT) has become the de facto paradigm for large language models (LLMs) to solve complex reasoning tasks. However, due to the sequential nature of token generation, the inference time can be formidable if the CoT is exceedingly long. This paper identifies a fundamental \emph{information bottleneck} that can cause the CoT to be long: although each forward pass can activate a vast amount of neurons, in the end, the information the model writes down is limited to a single token, making it inevitable to produce many more CoT steps than necessary. We first theoretically establish this bottleneck by showing that for some natural problems, such as pointer chasing and computing parity, either 1-layer transformers or constant-layer finite-precision transformers require a rather long CoT to solve. We then demonstrate that for these same problems, allowing the Transformer to write high-dimensional embeddings to the CoT (i.e., using latent CoT) significantly reduces the CoT length, establishing a provably theoretical benefit for using latent CoT. We further validate our theory with controlled experiments: training a small transformer to simulate Conway’s Game of Life with latent CoT, we vary the per-step write bandwidth to the latent CoT and observe a sharp success threshold proportional to the board size.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper gives a theoretical explanation of why chain-of-thought traces in large language models are often very long: each reasoning step can output only a single token despite internally performing much richer computation. The authors prove theoretically that these problems require long CoTs to be solved, but much shorter ones when using latent CoT, which allows writing high-dimensional vectors. Additionally, it provides experiments demonstrating that latent CoT indeed removes this information bottleneck.

### Strengths
1. The authors identify an inherent drawback of the CoT mechanism--its limited information bandwidth--and provide a theoretical explanation for this limitation. This novel theoretical framing gives new insights into this widely observed empirical phenomenon.

2. The authors derive mathematically rigorous bounds on the number of CoT steps required to solve two classic problems--pointer chasing and parity--providing a clear comparison between single-token CoT and latent CoT.

3. The authors introduce a very interesting connection between the dot-by-dot CoT (Pfau et al., 2024), the single-token CoT and latent CoT, framing them as points along a continuum defined by the amount of information transmitted per reasoning step.

4. The paper presents experimental results that validate the theoretical claims.

### Weaknesses
1. In the current form, the main weakness of this paper is its presentation. I think the writing could be greatly improved and the paper could be organized better to improve its clarity. For instance, a lot of space is dedicated to section 1.1, which introduces in a lot of detail the contributions of the paper, before introducing many of the formal preliminaries. I think the contributions could be summarized more succinctly in the introduction in a more informal way. This can save space that can be used for adding some informal proof sketches for the main theorems. 

2. The observation that single-token CoT is inefficient compared to latent CoT is not new, therefore it seems like the novelty stems from its mathematical framing. However, the paper lacks a precise definition of the information bottleneck, which I think is essential for a theoretical paper of this kind. 

3. Another limitation is the relatively limited experimental results. While the synthetic experiments illustrate the information bottleneck, it would help to include experiments on standard reasoning benchmarks, similar to those used in Hao et al. (2024). Such experiments would help demonstrate whether the theoretical insights hold in more realistic settings.

### Questions
1. I don't fully understand the paragraph from lines 106-109. Could you elaborate further?

2. How do you formally define the information bottleneck?

3. Hanh and Rofin (2024) prove that the sensitivity of parity using chain-of-thought is constant. How does their finding connect to your theoretical results?

4. Do you think it would be possible to derive similar bounds for problems that can be solved by the dot-by-dot CoT? It would be very interesting to compare not only single-token CoT and latent CoT, but also include dot-by-dot CoT in the analysis.

5. How do you explain the mixed results of latent CoT methods such as COCONUT (Hao et al., 2024) on standard reasoning benchmarks? They observed that performance varies across benchmarks-sometimes exceeding, sometimes matching, and sometimes falling below single-token CoT.



Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason E Weston, Yuandong Tian. 2024. Training Large Language Model to Reason in a Continuous Latent Space.

Michael Hahn and Mark Rofin. 2024. Why are Sensitive Functions Hard for Transformers?. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14973–15008, Association for Computational Linguistics.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors argue that CoT suffers from an information bottleneck, as each reasoning token requires that the model compress its progress into a single token, which for a vocabulary of size $|\mathcal V|$ carries only $O(\log |\mathcal V|)$ bits of information. In contrast, latent CoT with an embedding dimension of $d$ (and constant precision) carries $\Theta(d)$ bits of information per CoT step. Using a communication complexity argument, the authors show how this leads to a separation in the number of CoT rounds needed to solve parity and pointer chasing (for constant-depth transformers). Using a bounded-dimension MLP layer to simulate varying information bottlenecks, the authors show in experiments simulating Conway's game of life that as the problem scales, larger information bottleneck dimensions are required to roll out 10 game of life steps correctly.

### Strengths
1. Proving lower bounds on the number of reasoning tokens required to solve a problem is a great contribution. A timely and important problem.
2. The proof techniques, especially applying Fourier analysis, are a nice contribution.
3. The paper is clearly written and presented.

### Weaknesses
1. The connection between the theory and the experiments is very loose. Conway's game of life comes out of nowhere, and the experiments do not actually use the vocabulary size as an information bottleneck, just a small intermediate dimension.


### Suggestions
- Figure 1 is quite hard to parse visually as there is no visual pattern indicating which lines have larger or smaller bottlenecks. Using a consistent palette would help (e.g., darker or redder = larger $d$), perhaps in combination with a non-color feature (thicker line = larger). The colors are also inconsistent between $n$s, which compounds the issue
- Some important related work is missing:
   - This paper is very related to the "token complexity hypothesis": How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach, https://arxiv.org/abs/2503.01141. Also related, the "Sample optimal length" from ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length for Efficient Reasoning, https://arxiv.org/abs/2504.21370. This paper can be seen as providing theoretical backing for these empirical results
   - For the idea of information bottlenecks in attention, which is mentioned a couple of times: Lost in Transmission: When and Why LLMs Fail to Reason Globally, https://arxiv.org/abs/2505.08140


### Overall assessment
The theoretical analysis is strong and novel enough to carry the paper, despite the weak connection to the experiments. I think the paper would be substantially better with experiments on the analyzed problems rather than the game of life, however.

### Questions
1. Why not parity and pointer-chasing experiments to align with the theory?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper analyzes the computational abilities of chain of thought (CoT). In particular, it analyzes how the discrete and one-token-at-a-time nature of CoT limits its efficiency and requires a large number of model evaluations to solve certain problems. Methodologically, they approach this through the lens of information bottleneck, showing that each CoT step only adds $\log |\mathcal{V}|$ information, where $\mathcal{V}$ is the vocabulary. In contrast, if the model does not have to decode at every time step (such as in latent CoT or regression-based transformers), the amount of information passed can be much larger. With this, the authors showcase two problems (pointer chasing and parity), where transformers require a large number of CoT steps (model evaluations) when using decoding, but latent CoT substantially reduces the number of model evaluations required. 
They evaluate their theory by training transformers on the Game of life and controlling how much information the CoT tokens can convey. They find a close correspondence with the theory, as communicating more information starkly improves performance with a limited number of model evaluations.

### Strengths
- I believe the paper tackles a relevant problem that is interesting both theoretically as well as practically.
	- For example, in the theoretical section, it considers languages known in theoretical literature to be difficult for transformers
- I also like that the paper applies standard expressivity results on the relationship between sensitivity and circuit complexity to modern LMs.
- The setup is clear, and the experiments make sense.
	- For example, studying the three different regimes (zero, log, full) make sense
	- The experiments are also well-documented
	- The main results are clearly presented, and I think they are a valuable contribution
- I like that the paper uses a known theoretical transformer framework
- Proofs seem sound and ground the contribution

### Weaknesses
- I believe `Amiri et al., Lower Bounds for Chain-of-Thought Reasoning in Hard-Attention Transformers, 2025` studies a very similar problem of the lower bounds on the number of CoT steps and solves a seemingly more general problem in some ways. However, it is not discussed anywhere.
- I am not familiar with communication complexity, so I found the discussion and presentation of the results on the lower bounds very hard to follow, as many technical terms were used without being introduced, for example:
	- What is communication complexity at all?
	- What does it mean for a protocol to succeed with some probability?
	- What is randomized communication complexity?
	- What is advantage?
- The concepts about sensitive functions seem to be based on Ryan O'Donnell’s Analysis of Boolean Functions; I believe it should be cited.
	- I do think sensitive functions could be introduced later in the text though, as they don’t fit naturally where they are introduced/they are not used immediately
- The extra page (current submission is only 8 pages long) could be used for introducing communication complexity terms and for more discussion and a conclusion.

### Questions
- How is the latent-CoT supervision loss computed when `information_bottleneck` is logarithmic or zero? The one-hot-encoded information would not be of the right size.
- How do you ensure that the tokenizer agrees with the steps of the (latent) CoT?
- Why did you encode game of life in natural language? Wouldn’t it be more targeted to encode it with some specialized syntax?
- Could you comment on the difference between your results and those by Amiri et al.? This would help clarify the paper’s contributions.
- Have you thought about frameworks such as looped/universal transformers and diffusion models? The parallel generation inherent to those models could make problems solvable with fewer model evaluations.
	- In particular, what is the connection between your framework and that of Saunshi et al., Reasoning with latent thoughts and other “latent reasoning” frameworks that perform updates to the residual stream in parallel?
- Are there any practical takeaways for improving CoT? Larger vocabularies? Including latent CoT?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper leverages communication complexity to formally analyze the limitations of token-level CoT. The authors establish that token-level CoT faces an information bottleneck on pointer-chasing and parity tasks, proving lower bounds on the required CoT length. In contrast, this paper demonstrates that a latent CoT approach can overcome this bottleneck, requiring a shorter CoT length and thus possessing greater expressiveness. Experimental results on Conway's Game of Life are presented to empirically support the theoretical claims.

### Strengths
The paper introduces a novel and valuable perspective by applying communication complexity to formally analyze the theoretical properties of latent CoT vs. standard token-level CoT. This is an interesting and promising theoretical direction for understanding the capabilities and limitations of different reasoning approaches.

### Weaknesses
1. This paper appears to be an incomplete draft. It is missing several standard and essential sections, such as Related Work and Conclusion sections. The main text's length (8 pages) also suggests it may not be finalized. For a submission, the paper would need to be presented as a complete work.

2. This paper would be more accessible and self-contained with the inclusion of more preliminaries introduction, particularly the necessary background from communication complexity. Additionally, the problem setups could be clarified: a more detailed description of the tasks (pointer chasing, parity), especially the specific input/output representations for the Transformer is needed for the results to be fully understood.

3. The theoretical claims require more rigorous support. For instance, Theorem 4.1 is presented with only a proof sketch in the main text, and a complete, formal proof appears to be missing from the appendix. All theorems should be supported by complete and verifiable proofs.

4. The experimental validation is currently limited to a single task (Conway's Game of Life). It would be highly beneficial to include experiments on a more diverse set of reasoning tasks. Moreover, there is a notable disconnect between the tasks analyzed theoretically (pointer chasing, parity) and the task used for experiments. Aligning the theoretical and empirical investigations more closely, or clearly justifying the choice of the experimental task in the context of the theory, would substantially strengthen the paper's core argument.

### Questions
1. In Theorem 3.1, if one sets $d = \Omega(n^{0.6})$, the bound $n / d^2$ becomes $O(n^{-0.2})$, which approaches zero as $n \to \infty$. Could the authors clarify if this implies that no CoT steps are needed in this regime?

2. The proof of Theorem 3.1 appears to rely on Definition 3.6, which involves $n / 2d$ functions. Does this reliance imply that Theorem 3.1 holds only under the assumption that $d \leq n/2$? Please clarify the conditions and domain for this theorem.

3. The paper considers finite precision Transformers. Could the authors discuss whether these theoretical results are expected to generalize to models using $O(\log n)$-bit precision?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper theoretically compares transformer with word space CoT against latent space CoT. By looking at specific example tasks, the paper shows that a 1-layer transformer with latent CoT is much faster (at least with a factor of model dimension) than standard CoT. This is done by looking at a specific task (pointer chasing) and providing lower bound for number of tokens in a standard CoT and an upper bound for number of tokens with a latent CoT. Similarly, for a multi layer transformer, as long as the computation precision is finite, the authors show a similar task where there is a gap between CoT and latent CoT (at least with a factor of model dimension). Finally, the paper looks at a special version of solving the simulation of Game of Life. In particular, the settings allows the authors to control the information flow between CoT steps. They show that the task can be solved only if bottleneck is large enough to showcase the importance of this bottleneck.

### Strengths
The paper considers several interesting settings. Focusing on the single layer transformers is interesting as there is no room for a transformer to pass additional information such as intermediate vectors through attention. This is extended through a different method and by cleverly utilizing the finite nature of the available computation precision to multi-layer transformers. Also, proposing a practical problem where the effect of this gap can be clearly observed is also interesting. 


The paper is written mostly clearly and the authors do a good job of keeping things mostly simplified and postponing the details to the appendix.

### Weaknesses
I believe some of the claims (especially some made in the introduction section) are inaccurate. For example, when considering multi-layer trnasformers, it is no longer true that the information that can be passed is bounded by log V. These models generate latent representation in intermediate layers which future tokens can attend to. I strongly suggest to make these claims more accurate.

In Section 4, it would be useful to add a discussion around the effect of number of transformer layers and discuss how it is not possible to asymptotically match the increase of context length using a fixed depth. Similarly, it would be nice to more clearly highlight the effect of fixed precision. Generally, I feel this section could use a bit more discussion to make things more intuitive and clear.

### Questions
1. Is it possible to show what is proved for parity task also in practice?

### Soundness
3

### Presentation
2

### Contribution
3
