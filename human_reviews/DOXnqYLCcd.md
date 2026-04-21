# Dynamic Assortment Selection and Pricing with Censored Preference Feedback

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
In this study, we investigate the problem of dynamic multi-product selection and pricing by introducing a novel framework based on a *censored multinomial logit* (C-MNL) choice model. In this model, sellers present a set of products with prices, and buyers filter out products priced above their valuation, purchasing at most one product from the remaining options based on their preferences. The goal is to maximize seller revenue by dynamically adjusting product offerings and prices, while learning both product valuations and buyer preferences through purchase feedback. To achieve this, we propose a Lower Confidence Bound (LCB) pricing strategy. By combining this pricing strategy with either an Upper Confidence Bound (UCB) or Thompson Sampling (TS) product selection approach, our algorithms achieve regret bounds of $\tilde{O}(d^{\frac{3}{2}}\sqrt{T/\kappa})$ and $\tilde{O}(d^{2}\sqrt{T/\kappa})$, respectively. Finally, we demonstrate the performance of our methods through simulations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper considers both multi-product selection and pricing under censorship. The authors use a new way to find sublinear regret that uses LCB to price items and UCB/TS to select assortment. They achieve optimal regret bounds with respect to $T$ and use some simulations to verify though lacking comparable benchmarks.

### Strengths
The way of applying LCB to pricing (actually exploration) is inspiring. The extension from MNL to C-MNL is practical in the real world. The usage of TS is quite innovative.

### Weaknesses
1. My main concern is the statement that $1/\kappa=O(K^2)$. Is it true? If $|S|=K$, since exp>0, we know that $P(i_0)\le1/K$. In the meantime, choosing $i$ with the minimum exp term, it seems that $P(i)\le 1/K$ as well. Therefore, $1/\kappa\gtrsim \Omega(K^2)$. Please let me know if I missed something. Otherwise, the results will become much weaker.

2. Since different items have different contexts, it's more reasonable if the buyer can buy more than one item at the same time. It'll be an interesting extension to consider this scenario.

3. I'm wondering if it's possible to get the logarithmic problem-dependent regret as literature in both dynamic pricing and MAB. Maybe you can do some experiments on this. For example, use regression to find the actual order of $T$ (regress log(Regret) on log(T)).

### Questions
For the TS, you use the maximum of m samples. Then, with high probability, $x^T\tilde\theta^{(m)}$ is larger than the mean. It's somewhat another kind of UCB rather than a "true" TS. Do you have any other intuition or motivation to use TS? Can it reduce computational complexity compared with UCB as it still contains a hard-to-compute argmax?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the problem of dynamic multi-product selection and pricing. In such a problem, the seller must not only set prices but also determine which products to offer. The authors introduce a LCB-based pricing strategy to set prices, which can promote exploration because buyers would be more likely to purchase, avoiding the censorship issue. As for product assortment selection, they employ two strategies, one based on UCB and the other based on Thompson Sampling. Each strategy corresponds to an algorithm with a guaranteed regret upper bound. Extensive experiments on synthetic datasets validate the performance of the algorithms and demonstrate their superiority over existing approaches.

### Strengths
1. The problem studied in this paper introduces product assortment selection in a multi-product sale scenario that has not been considered in previous work.

2. The theoretical results are very solid and accompanied by experimental validation. I really appreciate the proof sketch in the main text, which shows the authors' very clear proof ideas.

### Weaknesses
1. According to my understanding, proposing a framework implies proposing a class of solutions. The authors' real contribution is to find a problem - a problem that fits the actual application scenario - and formalize it, so it is inappropriate to describe it as "proposing a novel framework".

2. I know it is difficult to find and prove a regret lower bound in such a complicated problem that involves both price setting and product selection, but a complete paper should at least have a discussion about the regret lower bound. Perhaps it is to refer to the magnitude of the upper and lower bounds of other similar problems, or perhaps to propose a conjecture of a tight regret lower bound, so that readers can have a clearer understanding of the difficulty of this problem and the contribution of the authors.

3. No experiment on any real-world datsaet.

### Questions
This paper proposes two product selection strategies, which makes the theoretical results of this paper look more fruitful. Give the regret upper bound of the TS strategy is worse than that of the UCB strategy, is there any more benefit to studying this strategy besides being able to announce that "it is the first work to apply Thompson Sampling"? For example, is the TS strategy more convenient in code implementation? Or is there an advantage in computational complexity? ...

Please compare the two strategies in detail. The rating may be reduced if the answer is insufficient.

### Soundness
4

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
4

### Summary
This paper studies the problem of online pricing and assortment optimization, where a seller engages in repeated interactions with a buyer over $ T $ discrete time steps. In this setting, the seller offers a selection of products from a set of size $ N $. At each time step $ t $, the seller chooses an assortment $ S_t \subseteq [N] $ of products to present to the buyer, along with a specific price $ p_{i,t} $ for each item $ i \in S_t $.

The buyer is characterized by latent parameters $ \theta_v $ and $ \theta_{\alpha} \in \mathbb{R}^d $, which influence their purchasing decisions but remain unknown the seller. Each product $ i \in S_t $ is associated with known feature vectors $ x_{i,t} $ and $ w_{i,t} \in \mathbb{R}^d $, representing various characteristics of the items. The buyer's valuation of a product is given by $v_{i,t} =  x_{i,t}^T \theta_v $, while their price sensitivity is described by the parameter $ \alpha_{i,t} = w_{i,t}^T \theta_{\alpha} $. Consequently, the buyer's utility for a product $ i $ is defined as $ v_{i,t} - \alpha_{i,t} p_{i,t} $. The buyer makes a purchasing decision based on a "censored multinomial logit choice function," which acts as a sigmoid function applied to all items $ i \in S_t $ for which the price $ p_{i,t} $ does not exceed the buyer’s value $ v_{i,t} $.

The goal is to minimize regret compared to the optimal assortment and pricing decisions in hindsight. A key challenge is that the seller lacks information on which products are "censored" by the buyer's multinomial logit choice function (i.e., $v_{i,t} < p_{i,t}$). This uncertainty makes it challenging for the seller to glean information from the buyer's purchasing behavior. To address this, the authors employ a combination of algorithms: a Lower Confidence Bound (LCB) approach for setting prices and an Upper Confidence Bound (UCB) method for selecting assortments. By using the LCB algorithm, prices are initially set low, reducing the likelihood that items will be censored in the early stages. This enables the algorithm to gather more accurate data on the buyer's latent parameters, $ \theta_v $ and $ \theta_{\alpha} $. This algorithm obtains a regret bound of $ O(d^{2/3}T^{1/2}) $.

### Strengths
-	The paper offers a natural extension of the multinomial logit (MNL) model, building on previous research that has explored online pricing and assortment optimization for the MNL model without incorporating censorship. Censorship seems natural and adds interesting additional, combinatorial challenges.
-	Additionally, the proposed approach, which combines the LCB and UCB algorithms, feels intuitive and well-aligned with the problem.
-	Finally, the paper well written.

### Weaknesses
-	Including the Thompson Sampling (TS) section should be more thoroughly motivated, especially since the algorithm’s regret is weaker than that of the LCB/UCB approach. From the paper, it’s unclear what advantages TS offers over the LCB/UCB algorithm. If none, this section might not be necessary and could be removed to streamline the paper.
-	Additionally, I would appreciate a more detailed explanation of why the latent price sensitivity parameter $ \alpha_{i,t} $ is essential to the model. It would be helpful to have some real-world examples that illustrate the importance of including $ \alpha_{i,t} $. Moreover, I find it counterintuitive that $ \alpha_{i,t} $ appears in the exponent of the MNL choice function $ \exp(v_{i,t} - \alpha_{i,t} p_{i,t}) $, yet it does not appear in the indicator $ \mathbb{1}(p_{i,t} \leq v_{i,t}) $. Intuitively, it might make more sense if the indicator were formulated as $ \mathbb{1}(\alpha_{i,t} p_{i,t} \leq v_{i,t}) $, as this would more consistently capture the influence of $ \alpha_{i,t} $ on the decision threshold.

### Questions
Please address the confusions I highlighted in the Weaknesses section.

### Soundness
4

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
The authors study the dynamic multi-product selection problem. They introduce a new censored multinomial logit (C-MNL) choice model, capturing buyer behavior by filtering products based on price thresholds. They propose two algorithms that leverage a Lower Confidence Bound (LCB) pricing strategy, combined with either an Upper Confidence Bound (UCB) or Thompson Sampling (TS) approach for selecting product assortments. Both algorithms achieve provable regret upper bounds, and their performance is further validated through simulations.

### Strengths
1) The censored multinomial logit (C-MNL) model is a novel approach that effectively captures buyer behavior by incorporating a price threshold for product selection, reflecting realistic purchasing patterns.
2) The authors introduce an innovative Lower Confidence Bound (LCB)-based valuation strategy for pricing, which can be flexibly integrated with various product selection methods to support this new model.
3) The authors provide comprehensive theoretical analyses, deriving regret upper bounds for both proposed algorithms. Moreover, these algorithms are computationally more efficient than previous methods

### Weaknesses
1) The algorithms involve several input parameters, such as $\lambda$, $\eta$, and $C$. However, the paper lacks guidance on how to select appropriate values for these parameters and does not discuss how they impact the regret bounds.
2) There is no lower bound analysis on the dependency on $d$, making it uncertain whether the proposed algorithms are optimal.
3) In the TS-based algorithm, the prior distribution and prior knowledge used (e.g., Gaussian distributions) are not clearly discussed, leaving the assumptions about the prior unclear.
4) The experimental setup is relatively limited, with tests conducted only for $K=4$, $d=2$, and approximately 20 arms (products), which does not reflect the scale and complexity of practical applications.

### Questions
1) Line 174: "Then we define an oracle policy." Could you clarify if approximate regret is considered in cases where the oracle is based on an approximation algorithm?
2) Line 347: "Additionally, our regret bound does not contain $1/ \kappa$ in the leading term". Could you provide some intuition or high-level explanation as to why the term $1/ \kappa$ is absent in the leading term of our regret bound? This could help readers better understand the underlying reasons for this distinction.
3) It could be interesting to explore whether the LCB-based valuation strategy for pricing can be effectively integrated with product selection methods other than UCB and TS.
4) Section 6: Including examples of real-world applications would enhance the discussion on potential extensions.

### Soundness
3

### Presentation
3

### Contribution
2
