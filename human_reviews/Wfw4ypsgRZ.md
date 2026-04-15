# Decoding Game: On Minimax Optimality of Heuristic Text Generation Strategies

- Decision: Accept (Poster)
- Scores: 8, 5, 8, 5

## Abstract
Decoding strategies play a pivotal role in text generation for modern language models, yet a puzzling gap divides theory and practice. Surprisingly, strategies that should intuitively be optimal, such as Maximum a Posteriori (MAP), often perform poorly in practice. Meanwhile, popular heuristic approaches like Top-$k$ and Nucleus sampling, which employ truncation and normalization of the conditional next-token probabilities, have achieved great empirical success but lack theoretical justifications. In this paper, we propose Decoding Game, a comprehensive theoretical framework which reimagines text generation as a two-player zero-sum game between Strategist, who seeks to produce text credible in the true distribution, and Nature, who distorts the true distribution adversarially. After discussing the decomposibility of multi-step generation, we derive the optimal strategy in closed form for one-step Decoding Game. It is shown that the adversarial Nature imposes an implicit regularization on likelihood maximization, and truncation-normalization methods are first-order approximations to the optimal strategy under this regularization. Additionally, by generalizing the objective and parameters of Decoding Game, near-optimal strategies encompass diverse methods such as greedy search, temperature scaling, and hybrids thereof. Numerical experiments are conducted to complement our theoretical analysis.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper derives stochastic decoding methods as the optimal strategy in an adversarial two-player game. Due to tractability issues, a one-step version of this game and a first order approximation for the resulting strategy is used. An explicit expression for the resulting regularization term added to the maximum likelihood objective is derived.

### Strengths
* It is valuable to have a proper theoretical derivation of heuristic decoding methods that have long been around in practice, and the paper provides this.
* the paper is well-written and neat
* the connection to robust optimization is interesting

### Weaknesses
* The empirical comparison is not state-of-the-art. For example, there are newer variants of greedy designed to overcome issues with repetitiveness, see contrastive search (see: https://huggingface.co/blog/introducing-csearch).
* The introduction sounds a bit like stochastic sampling strategies would always be preferable over deterministic optimization strategies like greedy search or beam search in practice. While there definitely is evidence that they perform better in some settings as given by the citations in the paper, there is also counter-evidence in other settings, e.g. [1], in particular in closed-end tasks. (Btw, the experiments in the paper are also limited to open-end text generation.) I understand that the authors preliminarily want to motivate stochastic decoding strategies in this paper and I do not want to diminish the motivation for them, but I still think it is important to at least acknowledge that the situation is more nuanced. For a recent empirical comparison, also see [2].

[1] "On Decoding Strategies for Neural Text Generators" [Wiher et al, 2022]

[2] "A Thorough Examination of Decoding Methods in the Era of LLMs" [Shi et al., 2024]

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper explores the gap between theoretical and practical decoding strategies in text generation for language models.The authors introduce a new framework, Decoding Game, which models text generation as a two-player game. They derive the optimal one-step strategy, showing adversarial regularization on likelihood maximization, with popular methods as first-order approximations. The framework also generalizes to include strategies like greedy search and temperature scaling. Numerical experiments support their theoretical findings.

### Strengths
- The analysis of the decoding process of LLMs is thoroughly considered. 
- The proposed decoding is backed up theoretically
- The experimental results show somewhat better performance than baselines.

### Weaknesses
- What is the motivation for choosing the TV-sup norm to compute the difference between the true distribution and the approximated one by LLMs? Is there any theoretical justification behind this selection? If not, at least an ablation study showing the effectiveness of this norm is necessary.
- It is still ambiguous why the authors define the distance between the two mappings by the way around Line 248.
- The result in the Equation (1) is non-trivial. Detailed transformation of it is necessary. At first glance, I questioned the equivalence of the two sides.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a theoretical framework that captures various decoding strategies used in practice, as well as provides a generalized form of such a technique and some experiments to support it.

### Strengths
The theoretical framework is well justified and rigorously supported by proofs
Statements are accurate and notation is consistent. Assumptions are clearly stated and natural.
The problem that is addressed is practically significant.
The framework generalizes previous decoding schemes opening the door to new, theoretically-grounded, heuristics.

### Weaknesses
In Prop 3.1, the authors could mention why we need the $\epsilon < max \hat{p}$ - that if we assign non-zero measure to $x_{<t}$, we’d get a cost of $-\infty$.
Theorem 4.3 doesn’t say anything about how the $p$ that yields the optimal solution looks like, nor what role $\hat{w}$ plays (in the min problem; the max one is clear). Does that lack structure or could it be added to the Theorem statement?
While the game approach to it is intuitive to some, it might be worth emphasizing the perspective of optimizing the worst case scenario: there are a couple of places where you phrase the framework, but don’t necessarily argue why nature “should” be adversarial. Optimizing the worst case scenario makes it more clear to people that are not used to thinking about games (since after all you just choose the approach that has the best guarantees under the TV assumption).
It’d be helpful to present the more general case secondary to what you get when you use a greedy local approach (which is done in section 3.4 and then used throughout). Once you decide to do local optimization rather than global, you can jump straight to Equation (4) and while Section 3.3 is well-written and objectively accurate, it is the hardest part to read of the whole paper and people may lose the important take-aways because of getting confused there.

Minor:
Remark at line 339 seems to be more relevant right before Corollary 4.5.
Line 54-55 repeats “From a statistical perspective”
It is worth making explicit that X_0 is a multidimensional object (as opposed to X_{t>0})
Line 243: $N (\hat{P})$ to be $\hat{P}$.. - an extra $\hat{P}$.

### Questions
“Depending on the desired effect, the truncation threshold or S may be determined from various quantities like the truncated vocabulary size, truncated probability weight, entropy, and so on. ” - this is confusing since the truncated vocabulary size, probability weight etc are themselves a function of S. What did you mean here?
Regarding the experiments, have you tried any of the methods that drop non-tail elements? (such as truncation sampling by Finlayson et al. (2024)). It would be interesting to see how a different-pattern sampling compares.
Is there any reason in the remark around line 376 for $C_{I-1}$ to not dominate the entropy? It’s not obvious how to interpret that.
Can your method be adapted to deal with other sorts of measure distances? The most natural one would be bounded cross entropy since this is what training usually optimizes

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper primarily focuses on text generation for language models.
Firstly, the author presents a novel minimax optimization problem for text generation tasks, where the min-player aims to minimize the worst-case negative cross-entropy.
Subsequently, it is demonstrated that the proposed minimax optimization problem can be interpreted as a likelihood maximization problem with implicit regularization.
Finally, the algorithm for solving the minimax optimization problem is introduced and its performance is empirically evaluated through experiments using GPT-2 models.

### Strengths
* The problem is appropriately motivated. Considering decoding strategies from a theoretical standpoint is essential for the research community.
* The algorithm proposed to solve the suggested minimax optimization problem appears to be easily implementable in practice.

### Weaknesses
I'm wondering to what extent this study provides new insights for existing decoding strategies that have achieved empirical success.
The author states "To resolve this dichotomy, this paper aims to propose a comprehensive theoretical framework of text generation" in the introduction section, meaning that one of the main goals of this paper is to theoretically explain the success of these existing strategies.
However, I'm not convinced that the proposed framework can replicate the widely recognized Nucleus sampling and Top-k sampling strategies simply by selecting an objective function $f$.
While it's demonstrated that temperature sampling can be recovered by appropriately choosing the function $f$, I'm interested in knowing if other strategies can also be replicated.

Moreover, since I'm not an expert in this research field, I'm not sure whether experiments using only GPT-2 models are sufficient for validating the proposed framework.
At the very least, it should be discussed whether the scale of the experiments conducted in this study is not insufficient compared to the current standards.

### Questions
My main concerns and questions are outlined in Weaknesses.
There is a typo in the introduction section:
* (line 55-56) "From a statistical perspective" is repeated.

### Soundness
2

### Presentation
2

### Contribution
2
