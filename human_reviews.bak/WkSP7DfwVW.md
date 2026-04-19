# Extending Myerson's Optimal Auctions to Correlated Bidders via Neural Network Interpolation

- Decision: Reject
- Scores: 5, 6, 8, 3

## Abstract
We aim to design revenue-maximizing single-item auctions that are deterministic, strategy-proof and ex post individually rational.  Myerson's seminal work on optimal auction design solved this problem for independent bidders. Myerson introduced the novel concept of virtual valuation and showed that revenue maximization is equivalent to virtual valuation maximization. Coincidentally, by greedily allocating the item to the bidder with the highest (ironed) virtual valuation, the resulting allocation is guaranteed to be monotone -- a necessary and sufficient condition for strategy-proofness.

For correlated bidders, Myerson's greedy allocation no longer guarantees monotonicity/strategy-proofness. We propose a simple yet empirically effective approach for designing near-optimal auctions for correlated bidders.  We train a neural network to interpolate the greedy allocation, while enforcing that the interpolation must be verifiably monotone.

Empirically, our method consistently achieves near-optimal revenue across a wide range of distributions, including adversarially generated cases. Compared to existing baselines, our approach shows substantial improvement, often reducing the gap to the (unattainable) greedy upper bound by an order of magnitude.

Furthermore, we demonstrate the generality of our approach by extending it to multi-unit auctions with unit demand, where we achieve similarly strong performance. Additionally, our verification techniques can be integrated into the RegretNet framework to design fully strategy-proof auctions.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In single-item sealed-bid auctions, Myerson characterized the revenue-maximizing mechanism under the assumption that the value distributions of all bidders are independent. However, this result does not hold when bidders' values are correlated. In this paper, the authors propose learning a parameterized auction mechanism from sampled data to maximize revenue while maintaining strategy-proofness.

### Strengths
1. The discussion of preliminary and related work is thorough.
2. The setting of correlated bidders is more general than that of classical independent bidders.
3. The authors conduct experiments across a wide range of valuation distributions.

### Weaknesses
1. The presentation could be further improved. The authors allocate excessive space to the Introduction and preliminaries (Sections 2 and 3), leaving less room for the methodology (Section 4), which I believe is the core of the paper.
2. The methodology is unclear to me. It appears that the authors use a neural network to compute the randomized allocation results. However, how is the neural network trained? Do you set the parameters using Mixed Integer Programming (MIP)?
3. If the authors trained the neural network using MIP, how is the monotonicity condition satisfied? If there are any violations, these should be noted and included in the experimental results, similar to what was done in previous literature RegretNet.

### Questions
See "Weaknesses".

### Soundness
2

### Presentation
1

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
The paper is another in the recent thread of work on deep learning for revenue-maximizing auction design (part of what is sometimes called “differentiable economics”).

Here, the focus is on *single-parameter* auctions. Here, Myerson (1981) was an essential contribution, which gave closed-form solutions for the optimal strategyproof auction assuming independent bidders.

Myerson’s simple idea is to use information about the distribution of bidder types to convert each bidder’s valuation into a “virtual valuation”, and then allocate to maximize that virtual valuation. Myerson also showed that any strategyproof mechanism must have a monotone allocation rule. Under certain reasonable conditions, maximizing virtual valuation is monotone (and if these conditions are not met, virtual valuation can be “ironed”).

Myerson’s full result holds only when bidders are independent. When the bidders are correlated, maximizing virtual valuation (ironed or not) may not be monotone. So finding a revenue-maximizing auction is much harder.

The idea of this paper is to use neural networks to search for high-revenue allocation rules and then, by embedding the neural networks in integer programs, to certify that those allocation rules are monotone (therefore strategyproof). Since these certification programs can also generated counterexamples for monotonicity, those can be used in the training loop to gradually remove monotonicity violations.

The authors show that their approach works well on a number of correlated valuation distributions, and can even be extended to single-parameter multi-unit settings.

### Strengths
- Interesting approach for a reasonable mechanism design problem and some nice generalizations
- Approach works empirically well
- Pleasantly clear discussion of extensions of Myerson to correlated valuations, and the challenges that result
- Certification programs are of general interest (i.e. one can apply them to RegretNet too to certify strategyproofness)

### Weaknesses
- Single-parameter correlated bidder auction design is in some sense a relatively niche problem
- Neural networks used are necessarily tiny to be able to be certified. This has its positives but is a limitation if more representational capacity is required for a given problem.
- The authors make a case for their grid distributions, but it would be nice to see some more common correlated distributions from the literature, in particular those with continuous support. The neural network approaches should be able to handle this.

### Questions
- I’m a little bit confused by the discussion of monotonicity. I think I grasp it, but I didn’t find appendix A.9 all that helpful. It might be useful to clarify why the typical Rochet characterization doesn’t do all the work here, as there are obvious ways of making cyclically monotone allocation rules (e.g. take the gradient of an input-convex NN).
- See weaknesses: why not at least take a shot on continuous-support distributions?

### Soundness
4

### Presentation
4

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
The paper tackles the optimal auction design in a two-bidder case with correlated valuation. The key challenge of naively adopting Myerson’s approach to this problem is that the resulting allocation rule is not guaranteed monotone due to the correlation between the value distributions. This problem was shown to be hard in general more than 10 years ago.

The paper proposes an empirical approach to this problem, where a neural network is trained to approximate the greedy allocation while subject to monotonicity regularization. This is done with the following components:
* Use a small and simple NN (MLP with ReLU) to encode the allocation function, and train it to maximize empirical revenue.
* A MIP (MIP-verify) is designed to verify the monotonicity of the allocation rule encoded by such an NN. MIP-verify in fact also (in some sense) quantifies the non-monotonicity of the NN, so it is also used as a monotonicity regularization during the training.
* Another MIP (MIP-fix) is designed to fix a non-monotone NN and generate a guaranteed monotone allocation rule, which is also the final output.

Empirical results suggest that the mechanisms obtained through this approach significantly outperform those obtained via traditional methods. In many test distributions, the gaps with respect to the unachievable greedy allocation benchmark get reduced by roughly 10x. The paper also shows that this approach can be generalized to multi-bidder cases but with unit demand settings.

### Strengths
This paper significantly improves the results on a fundamental auction design setting where no good enough theoretical results are known. The approach used nicely combined NN with MIP and guarantees provable strategy-proofness, which is usually difficult with empirical driven methods.

### Weaknesses
I don’t see any major weakness of this paper. One concrete suggestion is, whether it is possible to take one specific example distribution and present the corresponding greedy allocation and the allocation obtained from this method. It may help give more concrete intuitions for theorists and may inspire theoretical results as well.

### Questions
1. Why are there some rows in Table 1 where MP is larger than Greedy?
2. What are the major blockers for generalizing this method to general multi-bidder cases?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this paper, they authors propose a neural network based approach to automatically design revenue-optimal single item auctions with correlated valuation. The proposed approach includes (1) a post-processing to interpolate the greedy allocation to make the allocation monotone and (2) a revenue fix step to increase the revenue. The authors run many empirical analysis to verify the efficiency of their approach and show that it can be extended to multi-unit auction settings.

### Strengths
This paper analyzed a very interesting problem and lies in the intersection between machine learning and mechanism design. To understand the revenue-optimal single item auction with correlated valuation is important in practice, given it fits a lot of practical scenario.

The experiment results are sound and promising.

### Weaknesses
I indeed have some concerns in terms of the contribution of this paper.

1. I am not quite convinced about the contribution of the paper. The authors mentioned 4 main contributions in the introduction, however, it seems the techniques are a combination of several previous paper, (1) MIP using MLP with RELU activation is from [Curry et al. 2020]. The counter-example guided training is from [Guo 2024]. Can you clarify what is the main novelty of this paper? Is it just the usage of marginal profit in neural network?

2. The paper is not well-written. I think the authors put a lot of effort to highlight the contribution in words, but missed many details of the exact methodology used in the paper. For example, the "method" section (section 4) is fairly short and very condense. It seems requires a lot of prior knowledge to follow the techniques. When I read it, it seems everything seems already known in the previous literature. The only new is the new representation of NN using marginal profit and this leads to my first point.

3. I understand the theoretical limitation of the MyersonNet, which is basically relies on the independent assumption and each $f_i(b_i)$ is to learn the ironed virtual value function. However, I believe it is still valid to compare it empirical to show it doesn't work in practice.

4. Another concern is that the post-processing seems requires the knowledge of the joint distribution $\phi$, is it true? If we only have sample access to the distribution, can your approach still be applied?

Some minor comments:
Better to define the offer more explicitly, e.g., posted price or fixed price depends on the others' bids. Same comments also apply to other terminology.

### Questions
Besides my questions in the "Weaknesses", I have some questions as follows:

1. The second MIP is not always applied, right? If it passed the monotonicity check in the first MIP, you don't need the second MIP?

2. If I understand correctly, the problem of MyersonNet is due to the independent assumption, is it possible to generalize it as follows: There is a literature called lattice network, which can output a general monotonic network for multi-dimensional input. Therefore, we can have n lattice network for each bidder, where each lattice takes all bids as an input and the $i$th lattice network will make the output increasing with $b_i$ but decreasing with $b_{-i}$. In the final layer we can have a softmax layer of n lattice network to output an allocation probability. Therefore, the network can capture the correlated information across bids and still montone with each of dimension.  Lattice network is already implemented in tensorflow (https://www.tensorflow.org/lattice/overview?hl=en) so it should be easy to try. What do you think?

### Soundness
3

### Presentation
2

### Contribution
2
