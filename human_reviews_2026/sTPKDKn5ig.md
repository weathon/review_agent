# Continuous Chain of Thought Enables Parallel Exploration and Reasoning

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4, 6

## Abstract
Modern language models generate chain-of-thought traces by autoregressively sampling tokens from a finite vocabulary. While this discrete sampling has achieved remarkable success, conducting chain-of-thought with continuously-valued tokens (CoT2) offers a richer and more expressive alternative. Our work provides new theoretical guarantees and algorithms for CoT2, motivated by logical reasoning tasks that inherently require search capabilities. Theoretically, we establish how CoT2 facilitates the model to track multiple discrete traces in parallel; and quantify the level of achievable parallelism and its benefits for inference efficiency. We also provide a CoT2-based one-layer transformer construction that solves the combinatorial ``subset sum problem'' given a sufficient embedding dimension. These insights arise from a novel and effective supervision strategy where we match the language model outputs to the empirical token distributions of a set of target traces. Complementing this, we introduce sampling strategies that unlock policy optimization methods for CoT2. Our primary strategy samples and composes $K$ discrete tokens at each decoding step to control the level of parallelism. 
Experiments confirm that (i) the optimal level of parallelism is governed by the embedding dimension, (ii) our continuous supervision strategy can outperform alternative methods, and (iii) policy optimization with CoT2 indeed improves the performance of the model beyond its initial discrete or continuous supervision.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Continuous Chain of Thought (CoT2), a novel latent chain-of-thought method in which the model outputs a convex combination of vocabulary embeddings. This allows the model to explore multiple reasoning paths in parallel within a single forward pass. The authors also propose a method for supervised training, called Continuous Supervised Fine-Tuning (CSFT), and a GRPO based reinforcement learning objective for policy refinement. Empirically, they show that CoT2 outperforms prior approaches such as COCONUT, discrete CoT, and no-CoT on the MNNS, ProsQA, and ProntoQA benchmarks, while converging faster. The experiments also show that the embedding dimension determines the model’s capacity for parallel reasoning, and that reinforcement learning further boosts accuracy and confidence.

### Strengths
1. The paper is well-motivated and introduces a clear novelty over previous methods. CoT2 itself, as well as the supervised training and reinforcement learning methods are original.

2. The paper provides a theoretical analysis of CoT2, including a proof showing that a single-layer transformer can solve the MNNS problem, and bounds demonstrating improved sample efficiency compared to discrete CoT.

3. The theoretical claims are backed by empirical results showing that CoT2 outperforms prior approaches such as COCONUT, discrete CoT, and no-CoT on the MNNS, ProsQA, and ProntoQA benchmarks, while converging faster. Additionally, the experiments show that the prosed RL-based method further boosts the accuracy.

### Weaknesses
1. A key limitation of CSFT is the reliance of some ground-truth distribution over all intermediate reasoning steps. This is only feasible for the toy tasks like the ones considered in this paper. This dependence seems to hinder the method's applicability to more open-ended reasoning tasks, where such an oracle not available. 

2. The need for a new task like the MNNS is not discussed. It seems like the existing datasets are perfectly suited for the methods introduced in the paper, therefore in the current form it seems like its main purpose is to be used in the proof from Section 4, showing that it can be solved by a transformer. 

3. Computing each dense vector involves performing a matrix-vector multiplication, where the matrix contains the embeddings of all tokens in the vocabulary. It seems like for large vocabularies, this could be a bottleneck that would reduce the practical gains from parallel explorations of reasoning trajectories.

### Questions
See the comments from the weaknesses section. Additionally, have you tested CoT2 on other benchmarks such as GSM8K? What would training look like in this setting? It would be quite interesting to see how it compares to COCONUT, since the authors observed a performance decrease compared to discrete CoT.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose an approach for continuous chain of thought, where at each step, the model stores a weighted average of all token embeddings weighted by their logit weights. They argue that this allows models to explore several possible reasoning traces in parallel. To perform this technique in practice, the authors propose a supervision approach based on selecting a number of high-performing trajectories in search problems. They show that this approach performs better than standard CoT in the minimum non-negative sum (MNNS) task. They also show theoretically how models with CoT2 are capable of solving this problem. The authors conclude by adapting GRPO for CoT2, which improves accuracy on MNNS.

### Strengths
1. The approach is natural and intuitive, and backed up by experiments and theory
2. The presentation is fairly clear overall
3. The topic is very relevant for the conference and of high impact

### Weaknesses
1. Some points could be described more clearly (see questions)

### Overall evaluation
This seems like a strong paper with a good blend of theory and experiments, proposing a continuous CoT approach that is reasonable and seems to perform well. I'm not familiar enough with the continuous CoT literature to know how novel this method is (it seems like using logits to weight all token embeddings is a very natural thing to do; but perhaps the challenge is figuring out the right way to supervise this method in training).

### Questions
### Suggestions/comments
- "AIME or IOI problems" acronyms should be defined, or avoided (e.g., "challenging math problems")
- "simplex-weighted compositions": isn't "weighted average" a simpler way of saying the same thing?
- The theoretical perspective on embedding capacity is nice


### Questions
1. Figure 1, $B=8$, why in $t=3$ is there only 1 embedding $e_1$? Why isn't it the mean of all 8 traces? (similar question for $B=2$, $t=3$)
2. For teacher forcing, how is a ground truth prefix $z^*_{<t}$ converted into the continuous tokens $z_{t'}^*$? Where do the weights $\alpha^*_{t'}$ come from? Do you feed the prefix at each step into the current transformer and use its output logits to weight the next token?
3. What model is generating the supervision distribution $\alpha_t^*$? Ah, I see in 3.1 that there are problem-specific approaches to generating $\alpha_t^*$. It would be helpful to specify at the beginning of Section 3 where these $\alpha$ will come from.
4. What is the complexity of MNNS?
5. In proposition 1, what embedding dimension is required for this to work?

### Soundness
3

### Presentation
3

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
The paper presents a technique of chain-of-thought with continuous valued tokens (termed CoT2) for transformer models, and aims to study the effectiveness of this approach over standard CoT and COCONUT baselines. In particular, the authors use the setup with minimum non-negative sum,  ProsQA, and ProntoQA tasks to show how a shallow transformer can utilize CoT2 for parallel search over possible trajectories and avoid error propagation with intermediate step mistakes.

The main idea of CoT2 is to use the softmax outputs (probs) during auto-regressive token generation and consider the probability-weighted sum of embeddings as the continuous-valued token for the subsequent generations. Thus, it captures information from the whole vocabulary instead of committing to a single token. Overall, this work deals with small-scale (2-layer) transformers to study the role of this approach for parallel exploration of trajectories.

### Strengths
The work is well presented with small-scale experiments, toy setups, and theoretical results to convey the CoT2 idea (especially the continuous token construction and fine-tuning/RL) and to build intuition about its effectiveness. The authors also discuss potential RL-based extensions for sampling and policy optimization with continuous-valued tokens, which can also inspire future efforts.

### Weaknesses
The extension of this approach to practical scenarios is not clear/discussed. Since CSFT requires ground truth $\alpha^*$ for each step, it is unclear how one can compute them for real-world data. There is also no evidence of the benefit of scaling horizontally with more CoT2 tokens in the paper, whereas standard CoT + majority@K (and increasing K) can indeed surpass CoT2.

### Questions
1. In Figure 2(a), the majority@K performance of standard CoT with sufficient K surpasses the CoT2 results, and I also noticed that CoT2 does not show any scaling behaviour in the results. Is it because the continuous tokens were always limited to "m" steps?

2. How to decide the "m" value for tasks beyond the toy setup? If a standard CoT approach can solve the task with say A CoT tokens, then can CoT2 solve it with B < A continuous valued tokens? A discussion on how to think about these situations can be helpful.

3. Can a model trained on MNNS with 4 input digits and CoT2 transfer the latent reasoning capability to tasks with 5 inputs? Changing the range of values seems to affect the performance, so how well can CoT2 be transferred to out-of-distribution inputs?

### Soundness
3

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
3

### Summary
This paper proposes an interesting framework termed _Continuous Chain of Thought_ (CoT2). Inspired by a previous work titled chain of continuous thought (COCONUT), this work trains transformer models to reason in a continuous token space rather than via discrete linguistic steps. Each reasoning token is represented as a convex combination of vocabulary embeddings, allowing multiple reasoning trajectories to coexist in superposition. The model is trained from scratch using a Continuous Supervised Fine-Tuning (CSFT) objective that aligns predicted token distributions with soft targets derived from the top-BBB teacher trajectories, followed by a Generalized Reinforcement Policy Optimization (GRPO) phase to reinforce correct reasoning paths. Experiments on synthetic reasoning benchmarks such as MNNS and ProsQA show that this approach achieves higher task accuracy compared with standard discrete CoT baselines. In addition, the authors revealed a link between embedding dimensionality and the number of representable reasoning branches via theoretical analysis.

### Strengths
- The framework offers a theoretically principled formulation of parallel reasoning within neural sequence models, on a scale that is testable with single GPUs.
- The CSFT objective is mathematically clean, converting multi-trajectory supervision into a single convex target, and the accompanying proofs clarify the representational capacity of continuous token mixtures. 
- The paper establishes a precise connection between continuous probabilistic reasoning and information-theoretic efficiency, supported by consistent empirical trends in accuracy and entropy reduction.

### Weaknesses
- The study’s experimental scope and scale are limited. All models are toy-sized and trained from scratch with pruned vocabularies on synthetic tasks and symbolized formulations that departure significantly from natural languages, making it unclear whether the proposed principles extend to large, pretrained language models operating over natural text. 
- For reasons explained above, I am not convinced that the comparisons against conventional CoT and COCONUT are fair. The uncontrolled and unfair nature of the benchmarking obscures the contribution and significance of the results.
- Moreover, the self-designation “CoT2” implies direct lineage from the original chain of thoughts. I find this misleading since the proposed method differs fundamentally from both discrete CoT prompting and latent-space reasoning approaches like COCONUT.

### Questions
- Can the proposed continuous-token formalism scale to full-vocabulary pretrained LMs, or is it confined to small synthetic domains? No experiments needed if the authors are pressed for time. Simple speculation suffices.
- How sensitive are results to the trajectory budget B and to the number of reasoning steps? Can one set these hyper-parameters automatically?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CoT2, a new method for continuous chain of thought, which constructs continuous tokens as a linear superposition of tokens. Such models can be trained via SFT on a combination of top-K teacher traces. It is shown theoretically that a one-layer transformer construction can solve a version of subset-sum using CoT2 by tracking multiple computations in parallel. Moreover, it is shown that decoding with $N$ chains of CoT2 by averaging top $K$ sampled tokens, is upper bounded by the expected error of $NK$ ordinary CoT chains, showing the statistical benefits of parallelism. Finally, a GRPO objective is proposed for continuous CoT.

### Strengths
* The proposed CoT2 method is novel and simple to implement on top of existing models.
* Since each continuous token is a superposition of discrete tokens, the CoT should be more interpretable compared to methods which simply append the hidden representations.
* Theoretical analysis supports the intuitive benefits of tracking multiple traces. In particular, the increased information content is reflected in improved sample complexity.
* It is demonstrated that GRPO with MTS can adapt models to continuous rollouts and improve performance on synthetic tasks.

### Weaknesses
* While each continuous token should carry $K$ times more information than a discrete token with MTS as demonstrated by the sample complexity results, we also need to sample $K$ times more tokens when decoding. What is the tradeoff in terms of actual runtime (during SFT, decoding, RL, etc)?
* While a transformer construction solving MNNS with CoT2 is provided in Proposition 2, there is no lower bound for ordinary CoT, os it is unclear whether this constitutes an actual improvement over discrete CoT.
* The GRPO policy ratio proposed in Eq.(4) is not an unbiased estimate via policy gradient theorem.
* Since GRPO with MTS can be used to adapt existing discrete models to output continuous CoT, I would expect to see performance on larger models than GPT-2 and more challenging reasoning benchmarks. Can the authors add any larger-scale experiments?

### Questions
* As mentioned by the authors, the CSFT training scheme can be seen as token level distillation. Can the authors give a comparison with related methods in distillation?
* See also the questions in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
