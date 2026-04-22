# A Unified Framework for Quantized and Continuous Strong Lottery Tickets

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
The Strong Lottery Ticket Hypothesis (SLTH) asserts that sufficiently overparameterized, randomly initialized neural networks contain sparse subnetworks that, even without any training, can match the performance of a small trained network on a given dataset. A key mathematical tool in the theoretical study of SLTH has been the Random Subset Sum Problem (RSSP). The SLTH has recently been extended to the quantized setting, where the network weights are sampled from a discrete set rather than from a continuous interval. These new results are however far from those in arbitrary-precision setting in several ways. In this work, we provide an analysis of the RSSP in the discrete setting, and use it to derive tight SLTH guarantees in the quantized case. Our analysis obtain tight bounds on the failure probability of finding a strong lottery ticket in the quantized regime, providing an exponential improvement over previous results. Most importantly, it unifies the literature by showing that both approximate representations in the continuous setting and exact representations in quantized settings naturally emerge as limiting cases of our results. This perspective not only sharpens existing bounds but also provides a cohesive framework that simultaneously handles approximation and rounding errors.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents a unified theoretical framework for the Strong Lottery Ticket Hypothesis (SLTH) that bridges the gap between continuous and quantized settings. The authors extend the analysis of the Random Subset Sum Problem (RSSP) to the discrete regime, obtaining exponentially decaying failure probabilities in the quantized case—an exponential improvement over prior polynomial bounds.

### Strengths
I am not an expert in this specific research area, so I cannot confidently assess the novelty of the theoretical contribution, but the results appear technically interesting.

### Weaknesses
I would propose for further clarification on questions below:

### Questions
-As I am not an expert in this field, I am curious why the existence of such an optimal sub-network is particularly interesting. Could you elaborate on the practical implications of this theoretical result?
I am also wondering how one could actually find or reach this optimal sub-network in practice. Since it seems like a combinatorial problem. 
Is there any theoretical insight into whether standard pruning or quantization heuristics could approximate the theoretically optimal sub-network?

-I am also a bit confused about the notation $n$ in the paper. Could you clarify what it represents in this context? In Theorem 3 it is stated that mask matrices should be of the order of $n$? I would propose the authors for clarification. 

-About the exponential decay of the failure probability in Theorem 2:  Could the authors comment on whether this rate depends sensitively on the distributional assumptions of the random weights?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the theoretical bounds of strong lottery ticket hypothesis under the condition that the randomly initialized neural network's weights are quantized. To start, the paper provided an extension to the random subset sum problem by letting the target value and the randomly generated value taken from a discrete set. In this problem, the paper discretizes the proof from prior work and derives a sample complexity that is logarithmic with respect to the target accuracy. Moreover, the paper applied this bound to the strong lottery ticket hypothesis, and leveraged the standard proof technique to show the overparameterization required for a good approximation of a target network with discrete weights.

### Strengths
1. The paper considers the problem of pruning and quantization, which is an important topic by it self.
2. The paper extended the proof of random subset sum problem to a discrete setting.
3. The paper applied the result of the discrete random subset sum to the strong lottery ticket hypothesis and derived corresponding overparameterization requirement to achieve a given approximation error.

### Weaknesses
1. The setting considered by the paper is a little bit confusing. In particular, the paper considers approximating a target network with discrete value, which would intuitively be a simpler task than approximating a continuous value (e.g. approximating any value in $[-M, M]$ is intuitively harder than approximating values in $\{-M, -M+1, \dots, M-1, M\}$). However, the paper claims that approximating the discrete value is more general. Moreover, the paper considers initializing the student network (to be pruned) from discrete distribution. Ideally a more common approach is to initialize from continuous distribution, do the pruning to approximate the target network, and then do quantization. The paper is not clear about why they did not choose the latter approach.
2. The setting of the discrete random subset sum problem is simple. In particular, the paper only considers approximating $Z$ from $\{-M, -M+1, \dots, M-1, M\}$ using subset sum of $X_i$'s sampled from the same candidate set. It is not clear how the derived result is better than simply consider sampling $n$ copies of $X_i$'s, and computing the probability that at least one $X_i$ is $Z$ (which would have probability $1 - (\frac{2M}{2M+1})^n$). This simple bound gives an exact approximation, and logarithmic dependency on the probability. Compared with this bound, the bound provided in the paper seems to be worth since the success probability does not decay exponentially with respect to the number of samples.
. 3The paper lacks necessary experiments to validate the theoretical results.

### Questions
None

### Soundness
2

### Presentation
2

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
Summary: 

This paper presents a unified theoretical framework for the Strong Lottery Ticket Hypothesis (SLTH) that encompasses both continuous (arbitrary-precision) and quantized (finite-precision) neural networks. The SLTH posits that large, randomly initialized networks contain sparse subnetworks ("tickets") that can achieve high performance without any training. A key theoretical tool for studying SLTH is the Random Subset Sum Problem (RSSP). The authors identify a gap in existing literature: theoretical guarantees for quantized SLTH provided only inverse-polynomial decay for failure probability and only applied to exact representations, which could be slightly loosen as approximate cases for tighter bound of failure probability.

The core of this work is a new, sharp analysis of the RSSP in the discrete setting666. This analysis (Theorem 2) establishes an exponentially small failure probability for finding a $\Delta$-approximation, which is a significant improvement. By applying this new discrete RSSP result, the authors derive tight SLTH guarantees for quantized networks (Theorem 3). Their framework successfully unifies the field by simultaneously handling approximation errors ($\epsilon$) and rounding/quantization errors ($\delta$). They show that previous results for continuous-approximate tickets and quantized-exact tickets emerge as natural limiting cases of their single, unified model.


The primary contributions of this paper are: 

1. A Unified Framework: It develops the first theoretical framework that unifies the study of SLTH for both continuous (approximate) and quantized (exact and approximate) networks.
 
2. Exponential Probability Bounds: The paper's analysis of the discrete RSSP (Theorem 2) yields an exponentially decaying failure probability bound, as shown in the equation:$P(\tau \le t) \ge 1 - 2 \exp\left[-\frac{1}{\kappa t}\left(t - C \log \frac{M+1}{\Delta+1}\right)^2\right]$ This is an exponential improvement over previous inverse-polynomial bounds for the quantized SLTH. 

3. Tight Quantized SLTH Guarantees: By applying their new RSSP result, the authors prove (Theorem 3) that a sufficiently overparameterized, randomly initialized network with quantized weights (from $S_\delta$) can $\epsilon$-approximate any target network with exponentially high probability.

### Strengths
* Theoretical Significance: The unification of the continuous and quantized SLTH literature is a major theoretical achievement.
* Improved Bounds: The exponential improvement of the failure probability bound (from inverse-polynomial to exponential) is a very strong result. It tightens the theoretical guarantees significantly, bringing the quantized case in line with the continuous one.
* Novel Analysis: The core of the paper, the new analysis for the discrete RSSP, is a valuable combinatorial contribution in its own right and serves as the engine for the paper's other results.
* Generality: The framework's ability to simultaneously manage approximation error ($\epsilon$) and quantization error ($\delta$) makes it powerful and more reflective of practical scenarios where precision is finite.

### Weaknesses
* No Empirical Validation: The paper appears to be purely theoretical. It does not present any empirical experiments to validate the new, tighter bounds or to demonstrate how the required overparameterization $\mathcal{O}(d \log(1/\delta))$ behaves in practice.
* Assumptions: The analysis relies on weights being sampled uniformly from the discrete set $S_\delta$. The authors state this can be relaxed using a "standard rejection sampling argument", but this critical step is not fully elaborated upon in the main text.
* Proof Deferral: As is common, the full proof of the core technical result (Theorem 2) is deferred to the appendix. The main paper relies on a proof sketch, since I don't have time to read supplementary, this step makes me unsure of the soundness of this major proof, the sketch looks reasonable though.
* Existential (what) vs. Efficiently Search (how): The paper proves the existence of quantized strong lottery tickets but does not (and does not claim to) provide new, efficient algorithms for finding them. The problem of efficiently finding the optimal subnetwork remains a challenge.

### Questions
1. Your theoretical bounds are an exponential improvement. How do these new bounds compare to empirical observations? Can you design an experiment to show that the failure probability of finding a quantized ticket does indeed scale exponentially with overparameterization, as your theory predicts?

2. In the proof of Lemma 1, you use a rejection sampling argument to handle the non-uniform distribution of $Z_i = b_i a_i^+$. Could you elaborate on how this step affects the constants in your final overparameterization bound $n \ge C \log \frac{1/\delta + 1}{\Delta+1}$?

3. How precisely does your framework recover the existing continuous SLTH bounds (e.g., from Pensia et al.) in the limit as the quantization step $\delta \to 0$?

4. The analysis is for fully-connected networks with ReLU activations. What are the primary obstacles to extending this unified (quantized + approximate) framework to more complex architectures like Transformers or ConvNets?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper extends the theoretical foundation of the Random Subset Sum Problem (RSSP) to the quantized (discrete) domain, establishing a theoretical guarantee for the Strong Lottery Ticket Hypothesis (SLTH) under quantization. Specifically, the authors present a unified framework that provides theoretical guarantees for SLTH, encompassing both the approximate representation in the continuous setting and the exact representation in the discrete setting.

### Strengths
+ Table 1 plays a crucial role in highlighting the authors’ contributions relative to prior works. Specifically, it demonstrates that the proposed approach unifies both the approximate solution in the continuous setting and the exact solution in the discrete (quantized) setting with the highlight of previous study limitations. Moreover, Table 1 illustrates that the proposed method exhibits an inverse exponential decay in failure probability within the quantized setting.
+ This paper provides the stronger theoretical bound with the exponential decay in failure probability for the SLTH in the quantized setting.

### Weaknesses
- It would be valuable to investigate how the SLTH failure probability decays across different architectures such as ViTs, VLMs, and LLMs. Visualizing this behavior in real-world networks—with respect to factors like layer depth and width—would provide useful insights into how the likelihood of SLTH failure changes across different architectures. 
- The authors have theoretically demonstrated that SLTH failure decays exponentially in the quantized setting. I strongly encourage them to conduct empirical analyses across different architectures (ViT, LLM, and VLM) to validate this behavior in practice. Such experiments would help confirm whether SLTH can indeed be identified in quantized VLMs without significant accuracy degradation and quantify the extent of model compression achieved in terms of parameter reduction.
- Although the paper establishes a theoretical bound for SLTH in the quantized setting, it lacks contributions toward identifying improved sparse models that offer practical benefits.

### Questions
Please refer to Weaknesses section

### Soundness
2

### Presentation
2

### Contribution
2
