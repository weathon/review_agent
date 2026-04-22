# Improving Diffusion Language Model Reasoning through Joint Search in Generation Order and Token Space

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
The order-agnostic generation of Diffusion Language Models (DLMs) presents a promising alternative to autoregressive models for complex reasoning. We model reasoning as traversals of a problem-specific graph of logical dependencies, and view DLM decoding as sampling trajectories from a joint space over generation orders and token values. We show that standard decoding heuristics such as low-confidence remasking collapse this reasoning space. To address this, we introduce Order-Token Search, an algorithm that jointly searches over token content and generation order. Its core is a likelihood estimation function that scores block-level denoising actions, enabling stable path pruning. This allows for efficient exploration of diverse reasoning trajectories. Extensive experiments on mathematical reasoning and planning benchmarks show that our method consistently outperforms baselines, matching or surpassing the gains of fully post-trained d1-LLaDA with diffu-GRPO on Countdown, GSM8K, and MATH500 (e.g. achieving a 13.7% absolute gain on Countdown). Our work establishes structured search as a key missing component for advancing reasoning in DLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper describes a new method for remasking in the denoising process of Diffusion Language Models that searches jointly over generation orders and token choices to maintain high accuracy while promoting diverse reasoning. The paper empirically validates their proposed method in comparison with standard remasking strategies over various mathematical reasoning and problem solving benchmark datasets.

### Strengths
- The paper is very well written
- The problem setting is clearly stated and motivated
- Experimental results are clearly presented

### Weaknesses
- For a paper with only experimental results, the extent of the experiments seem lacking to prove significance. It seems that only the Countdown task exhibits strong performance gains and little explanation is given for why that seems to be.
- The computational complexity of the Order-Token Search is not clearly stated, is there a strong incentive to use this new algorithm despite marginal gains in performance?
- What is being meant by “reasoning” in this paper? What is it about the new masking strategy that unlocks “reasoning” capability in the MDMs, which seems to be a central claim of the study.
- The case study is informative to the intuition behind the proposed algorithm, but one example is not convincing to the soundness of an algorithm. Are there other wider trends or more concrete theoretical explanations that hint at better performance on the benchmark datasets?

### Questions
See Weakness section.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a search strategy for generation with discrete diffusion models to enable both high quality and exploration compared to things like low-confidence remasking which sacrifice exploration for quality. At a high-level it is a kind of beam search over both token orderings and token selections. Specifically they adopt the block-wise autoregressive decoding pattern commonly used in past work. They generate multiple possible blocks and only keep the candidates with the highest scores, similar to beam search. For their proposed scoring function, they unmask all tokens and compute the likelihood of the (re-masked) block conditioned on all other tokens. They report results across different decoding methods for standard math benchmarks and some synthetic puzzle benchmarks.

### Strengths
One of the most interesting aspects of this class of generative models is their improved decoding flexibility compared to autoregressive models. Exploring decoding strategies over both positions and tokens that are only possible for this class of models is an interesting research direction.

The motivation from the limitation of existing approaches (i.e. random is diverse but low-quality, and high-confidence is high-quality but not diverse) is clear and intuitive.

A naive beam search over every decoding choice would be extremely expensive. Performing it block-wise is a clever way to balance the benefits of search while reducing the computational overhead.

Their method outperforms low-confidence sampling (see question 2) and random sampling with majority baseline. Their ablation is Table 2 demonstrates the benefits of their joint search.

### Weaknesses
Although their approach is motivated by balancing exploration and quality, this benefit is not validated for their method. They present Pass@k curves to argue for the limitation of low-confidence remasking strategies, but never present such a curve for their approach to validate that it solves the problem with low-confidence remasking. For autoregressive models, beam search often has diversity issues, so a similar thing could be happening here.

Although the asymptotics of their search algorithm is discussed, the computational cost (at least in terms of NFE) of various decoding settings should be reported alongside all results. There are many ways to improve performance by expending more compute. It is important to quantify this rigorously to ensure that the method does not impose unreasonable tradeoffs. In general for test-time scaling approaches like these, results should be reported across a range of inference-matched settings to get a true picture of the tradeoffs.

Some additional decoding baselines should be included. Based on their own figure 2, autoregressive decoding appears to be a very strong baseline and should be included. Majority voting with both random and autoregressive baselines in compute matched settings should be included. 

In the appendix, the authors mention that they add gumbel noise to the logits to improve exploration. This is not discussed in the main paper. This feels like an important implementation detail that should not be relegated to the appendix. Is this critical? How does Order-Token Search perform without it? Can you similarly apply this perturbation to low-confidence re-masking to improve diversity. A majority voting baseline with low-confidence remasking and gumbel noise should also be reported, given the use of gumbel noise in the proposed method.

The analysis in A.2. is not convincing. Figure 2 shows that restricting the model to the autoregressive ordering is extremely effective, dramatically outperforming the other decoding strategies in most settings. Analyzing whether correct solutions under random sampling arose from landing on the autoregressive ordering is a very indirect way to study it. The odds of actually achieving the autoregressive ordering (or even getting very close to it) when doing random decoding is extremely small as demonstrated by the overwhelming amount of large chaotic values. The finding in Figure 2 pretty strongly suggests that the autoregressive ordering is particularly effective, at least for many tasks and settings.

### Questions
1. Equation two doesn’t seem to batch the text and figure. My understanding is that you are measuring the likelihood of the block given the surrounding context. Is that correct? As written, it is the probability of the clean sequence conditioned on the block.
2. What is the performance of the proposed method without Gumbel noise?
3. What are the total NFEs for all reported evaluation settings?

### Soundness
1

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
This paper proposes Order-Token Search, a novel decoding algorithm for Diffusion Language Models that jointly explores generation orders and token choices to improve reasoning performance. The work addresses a trade-off in DLM decoding strategies provides improvements on reasoning-intensive tasks. The work is generally interesting and highlights the importance of masking order in generation, and introduces a new test-time scaling method for DLM.

### Strengths
1. Novel problem identification: The paper highlights current limitation in diffusion language model decoding, which lacks the generation diversity offered by AR. The systematic analysis of the pass@1 vs pass@k trade-off in existing decoding strategies (Section 3) is compelling, showing that low-confidence remasking improves single-trial accuracy but limits exploration diversity.

2. Improved Empirical Results: The method achieves consistent improvements across multiple benchmarks, with particularly notable gains on Countdown that rival post-training methods. The comparison showing Order-Token Search outperforming computationally expensive baselines like Order Search and Token Search demonstrates the value of the dedicated likelihood estimation.

### Weaknesses
1. The decoding method computation complexity is not shown empirically. How does the method perform compared to baselines under the same FLOPs, e.g. draw a test-time scaling curve? 

2. Only one backbone model is studied. The paper would hold a stronger claim with more than one pretrained backbones studied.
 
3. Scoring Function Stability: How sensitive is the block-level likelihood estimation to the choice of block size? 

Minor Issues

- The notation in Equation 2 could be clearer - the function b(xs, xt, x0) is introduced but not precisely defined.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Order-Token Search, a structured search algorithm that jointly explores both generation order and token space. The method employs a likelihood-based scoring function to evaluate block-level denoising actions and prune unstable paths efficiently. Experiments on mathematical reasoning and planning tasks demonstrate the gains, suggesting that structured search can meaningfully enhance reasoning in DLMs without additional training.

### Strengths
1. Introduces a beam-search-like decoding algorithm (Order-Token Search) tailored for diffusion LMs, demonstrating improvement on challenging reasoning tasks.
2. Provides a clear analysis of the trade-off between accuracy and exploration in standard decoding methods (e.g., low-confidence remasking, AR-order, random sampling).

### Weaknesses
1. The paper fails to discuss sampling diversity controls (e.g., temperature, top-k/top-p sampling used in Dream) that could also improve pass@k performance when using low-confidence remasking methods, as studied in prior work (e.g., DiffuCoder [1], showing higher temperature will lead to more diverse token / order exploration and having high pass@k results).

2. In Figure 2, the “AR-order decoding” baseline lacks a beam search counterpart, leading to an unfair comparison—especially since the “token search” baseline is guided by other selected positions, as in line 412, "Token search is guided by a sequence of greedily-decided positions (selected via low-confidence remasking)". This undermines the validity of the baseline.

3. There is an inconsistency in Countdown accuracy between Table 3 (16–21%) and the main results (25.4–34.4%), which should be clarified.

4. The paper does not mention the extra inference overhead introduced by the joint search algorithm, which is important for evaluating practicality.

[1] https://arxiv.org/abs/2506.20639

### Questions
1. In Figure 1, the legend contains a typo: “within a forwar”
2. How is Order-Token Search integrated with full diffusion generation (rather than block diffusion)? Is it applied only to per-block, and does it scale with the full diffusion length?

### Soundness
2

### Presentation
3

### Contribution
2
