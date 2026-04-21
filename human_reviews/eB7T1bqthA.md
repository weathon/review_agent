# Pairwise Elimination with Instance-Dependent Guarantees for Bandits with Cost Subsidy

- Avg Score: 5.75
- Decision: Accept (Poster)
- Scores: 5, 6, 6, 6

## Abstract
Multi-armed bandits (MAB) are commonly used in sequential online decision-making when the reward of each decision is an unknown random variable. In practice however, the typical goal of maximizing total reward may be less important than minimizing the total cost of the decisions taken, subject to a reward constraint. For example, we may seek to make decisions that have at least the reward of a reference ``default'' decision, with as low a cost as possible. This problem was recently introduced in the Multi-Armed Bandits with Cost Subsidy (MAB-CS) framework. MAB-CS is broadly applicable to problem domains where a primary metric (cost) is constrained by a secondary metric (reward), and the rewards are unknown. In our work, we address variants of MAB-CS including ones with reward constrained by the reward of a known reference arm or by the subsidized best reward. We introduce the Pairwise-Elimination (PE) algorithm for the known reference arm variant and generalize PE to PE-CS for the subsidized best reward variant. Our instance-dependent analysis of PE and PE-CS reveals that both algorithms have an order-wise logarithmic upper bound on Cost and Quality Regret, making our policies the first with such a guarantee. Moreover, by comparing our upper and lower bound results we establish that PE is order-optimal for all known reference arm problem instances. Finally, experiments are conducted using the MovieLens 25M and Goodreads datasets for both PE and PE-CS revealing the effectiveness of PE and the superior balance between performance and reliability offered by PE-CS compared to baselines from the literature.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes the pairwise elimination (PE) algorithm for bandits with a cost subsidy model. First, it develops a simple PE algorithm that addresses two simplified versions of the MAB-CS model with a known reference arm. Then, it extends the PE algorithm with a BAI subroutine to address the MAB-CS model without knowing the reference arm. Theoretical analysis verifies that the algorithms have instance-dependent logarithmic regret/cost upper bounds. Experiments also verify the PE-CS’s performance.

### Strengths
1. New theoretical results. This paper proposes the first algorithm with instance-dependent logarithmic regret/cost guarantees for the MAB-CS model.

### Weaknesses
1. The algorithm design and analysis of the paper are not novel, and the theoretical results from the proposed algorithm are not interesting (i.e., expected). The algorithmic technique is very similar to prior literature, and the analytical approaches are also similar to other related works. The reviewer would suggest the author look into some challenging parts of the topic, like the lower bound for the MAB-CS model and proposing near-optimal algorithms (e.g., how to revise the UCB-CS algorithm so as it also works for the MAB-CS model).
2. The writing of this paper is not easy to follow. For example, when introducing the three types of model settings (Lines 93—101), which are supposed to be mathematically rigorous (e.g., for clear math notations and definitions), the paper uses an example to explain these three settings vaguely. Another example is the inconsistency of the notations. The notation $\mu_{\text{CS}}$ was used much earlier before it was formally defined. The two gaps $\Delta_C$ and $\Delta_Q^+$ should both have a plus on the superscript, but one is defined explicitly, and the other implicitly, which is inconsistent and confusing.

### Questions
### Minor Comments

- For the `sample_and_update` subroutine, it would be better to give the details, possibly in the appendix, if there is no space in the main paper.
- In Line 321, the `omega` should be $\omega$.
- Why, in the experiment, the regret and cost are always reported in a summation form. Would it be interesting to plot one as the x-axis and the other as the y-axis and check whether there is a trade-off?

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
2

### Summary
This work studies the Multi-Armed Bandits with Cost Subsidy (MAB-CS) problem, where a primary metric (cost) is constrained by a secondary metric (reward), and there is an inability to explicitly determine the trade-off between these metrics. The authors introduce the Pairwise-Elimination (PE) algorithm for a simplified variant of the cost-subsidy problem with a known reference arm, and generalize PE to PE-CS to solve the MAB-CS problem in the setting where the reference arm is the unidentified optimal arm. The authors provide instance-dependent bounds for PE and PE-CS for both Cost Regret and Quality Regret. They also conduct experiments to support the theoretical claims.

### Strengths
1. The problem is well-motivated. The authors provide interesting examples of applications of the MAB-CS framework.
2. The paper is well-written.
3. This work extends the MAB-CS framework to include two new settings, and develops two novel algorithms PE and PE-CS . The authors also provide instance-dependent bounds for the proposed algorithms.
4. The authors conduct experiments on real-world data to support the theoretical claims.

### Weaknesses
I feel some statements are somehow overclaimed. In Lines 115-126, the authors claim that the regret bounds of their proposed algorithms are $O(\log T)$, while for ETC-CS it is $O(T^{2/3})$. However, their bounds are instance-dependent and are not $O(\log T)$ in the worst case, while the  $O(T^{2/3})$ is the worst-case bound. Therefore, such a comparison seems to be over-claimed.

### Questions
1. See the weakness.
2. Is it possible to provide some lower bounds for this problem?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper considers multi-armed bandits. Instead of the traditional problem of maximizing total reward, they consider the problem of minimizing cost subject to a reward constraint. This formulation has been recently proposed in SSKA '21. In this paper, the authors extend the formulation to two settings:

1. Known threshold where there is a fixed reward threshold that should be met in expectation.

2. A reference threshold to an (initially) unknown arm that should be met in expectation.

For both problems, they consider A) the cost regret which is the zero-clipped total expected cost of the algorithm relative to the lowest-cost arm that meets the reward constraint and B) the quality regret which is the zero-clipped difference in expected reward of the algorithm relative to the reward threshold.

They propose a pairwise elimination algorithm that compares arms in a pairwise manner while recording the history of arms to improve efficiency. For the unknown reference threshold setting, they add an exploration phase where they learn they try to learn the reward of the unknown arm. In both settings, they show logarithmic bounds on the cost and quality regret in terms of the number of steps $T$ and the maximum instance-wise quality and regret differences.

They try these algorithms on a few datasets and find that it performs better than other algorithms (which, to be fair, were not designed for this setting).

### Strengths
* They consider new settings which seem reasonable to study.

* They propose new algorithms.

* They show bounds on the performance of their algorithms.

* They evaluate their algorithms in practice.

### Weaknesses
* Without reading the appendix, it's unclear to me what tools they use in the analysis of their algorithms.

* The statement of the bounds is difficult to parse (classic ML with too many terms which are hard to interpret).

* The algorithms are similarly difficult to understand. For example, I don't see how the history is recorded to "intelligently re-use samples for downstream comparisons".

* The algorithms are only tested on the movielens dataset and a toy dataset under specific hyperparameter settings. I would prefer at least three datasets and plots of performance for each hyperparameter ($\ell$, $\mu_\ell$, and $\mu_{CS}$) to validate that the performance is consistent in different settings.

### Questions
* What tools do you use in the analysis of your algorithms? What is the general strategy?

* How do you re-use samples for downstream comparisons in your algorithms?

* Can you make the case for the zero-clipped comparison again? I wasn't persuaded by the (repetitive) paragraph about stellar performance for different ad products in section 1.1. In addition, does your analysis require the zero-clipped comparison?

With additional experiments on other datasets and settings, and satisfactory answers to the above questions, I would be happy to increase my score to at a 6.

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
This work studied the bandit problem with cost subsidy. It proposed the PE-CS algorithm and derived an upper bound on its cost regret and quality regret. The paper provided a detailed description on the design of the algorithm. It also evaluated its performance with numerical simulations.



=========

After rebuttal: score increased to 6.

### Strengths
1. The paper is overall easy to follow.
1. The paper reviewed the various related works.
1. Algorithms are evaluated with both toy data set and real-life MovieLens data set.

### Weaknesses
1. As the CS model is built on Sinha et al. (2021), I would suggest the author(s) to compare the theoretical results with Sinha et al. (2021) after stating the theorems.
1. There are various related works with slightly different models, are the algorithms and their performance comparable? Can those algorithms work under this setting and what are their performance?
1. Considering the plots in Section 4, the target of the algorithm seems to be minimizing the sum of the cost regret and the reward regret. Is that true? I think the target of an algorithm in this problem should be clarified. Besides, is there a trade off between minimizing the cost regret and the reward regret? Why are we interested in the sum of the two regrets?
1. As there are already a number of similar models, what is the motivation to consider this model?

Minor comment:
1. Line 168: Should there be '' We build up our approach to optimize sth. for the regret objectives ..."?

### Questions
Please see the *Weaknesses* section.

### Soundness
3

### Presentation
3

### Contribution
3
