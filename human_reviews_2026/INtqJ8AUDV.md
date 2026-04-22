# Inference-time Alignment with Rewards in Anisotropic Besov Spaces: Superiority of Neural Networks over Linear Estimators

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Inference-time alignment, the approach of adapting pre-trained models to rewards through reinforcement learning, has proven highly effective in enhancing the performance of language models. Despite its practical success, theoretical analysis remains underdeveloped, and in particular, only a limited number of studies address the practical setting where neural networks are employed as reward models. In this paper, we investigate the advantages of neural networks in inference-time alignment. Assuming that the true reward function lies in anisotropic Besov spaces, we derive upper bounds on the regret with respect to the number of oracle queries when using a neural network as a reward estimator. We further investigate the limitations of linear reward estimators, and show that neural networks are superior owing to their ability to adapt to the smoothness of functions. Finally, we demonstrate that, with an algorithm that iteratively and actively learns the reward model from the responses of the trained model, smaller regret can be achieved, as neural networks adapt to local structures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper makes valuable theoretical contributions to reward modeling and inference alignment, with innovative insights into neural networks’ advantages and a novel multi-step learning mechanism. However, critical gaps in experimental validation, practical applicability (strong assumptions, computational cost), and accessibility significantly limit its impact.

### Strengths
The paper pioneers the theoretical analysis of neural networks’ advantages in alignment during inference, and formally proves that neural networks are more suitable for reward modeling compared to linear models (e.g., linear regression, kernel methods). This provides solid theoretical support for subsequent research, such as the design of more effective inference optimization strategies.
A Multi-step Alignment mechanism is proposed, extending traditional single-step reward learning to multi-step reward learning, which enriches the methodological framework of reward modeling in related fields.

### Weaknesses
The paper only presents theoretical derivations without providing experimental validation to support the proposed theories. No empirical results, numerical simulations, or real-world case studies are included to verify the correctness, effectiveness, or practical relevance of the theoretical claims.
The regret reduction in the multi-step training of the final proposed reward model relies on strong conditional assumptions. These assumptions include the reward function belonging to the Besov space and independent and identically distributed (i.i.d.) samples in each round, which limit the method’s generality and practical applicability.
The multi-step update mechanism involves sampling, scoring, and training the reward model in each round. The actual computational cost is likely to grow rapidly with the number of rounds and samples, yet the paper does not discuss the potential computational overhead of the algorithm or provide complexity analysis or optimization strategies.
Neural networks are prone to overfitting under small-sample scenarios. The paper fails to address this critical issue or propose mitigation measures (e.g., regularization, data augmentation, or few-shot learning techniques) to ensure the model’s generalization performance.
The paper has a high reading threshold and does not achieve accessibility. The content lacks intuitive explanations, illustrative examples, or simplified descriptions of core concepts, which hinders the dissemination and practical application of the proposed theories, especially for researchers outside the specialized subfield.

### Questions
Design numerical experiments or real-world case studies to verify the theoretical claims, including the superiority of neural networks over linear models in reward modeling and the effectiveness of the Multi-step Alignment mechanism.

Either relax restrictive assumptions (e.g., extend the framework to non-Besov reward functions or non-i.i.d. samples) and provide corresponding theoretical adjustments, or clearly justify why these assumptions are necessary and discuss the method’s performance under relaxed conditions via sensitivity analysis.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper considers the problem of inference time alignment where the reward is assumed to belong to an anisotropic Besov space. The reward is modeled using a neural network, which offers better modeling capabilties than linear functions and hence can learn the reward functions better. Under this setting, the authors propose a multistage inference-time alignment algorithm based on the InferenceTimePessimissm algorithm that was recently proposed. The authors derive regret bounds for their algorithm and show that the neural network based modelling offers improved regret performance over more traditional linear function estimators. Furthermore, since the authors also show that their mutistep algorithm (as opposed to single step) further improves the regret bound.

### Strengths
The authors clearly establish how multi-step routine helps improve their algorithm over a single step. This seems to be an interesting result.

Moreover, the role of $\beta$ is interesting. It clearly shows how local smoothness near the maximum affects convergence.

### Weaknesses
(This is more of a question than a statement but I would like to mention it separately.)

I am still a bit confused about the motivation and novelty of the paper. So as mentioned, the paper focuses on answering the question "What advantages do neural networks offer for inference time alignment and how can we unlock their full potential?"

To answer this, the authors show that when the reward function is in an anisotropic Besov space, NNs are better than linear estimators at modeling it and hence perform better. This has nothing to do with the inference time alignment --- this is a question about reward modeling. Moreover, the superiority of NNs over linear estimators was already established for learning functions in the paper  (Suzuki & Nitanda, 2021). The authors perform a $L_2 - L_{\infty}$ bound transformation in this work but that is relatively quite minor. So the benefit of NNs is already established. 

In terms of application to inference-time alignment, the performance clearly depends on reward estimation error. The authors build their algorithm on InferenceTimePessimissm, which clearly outlines the role of estimation error. So, one can easily plug and play the improved result. With these results, it almost seems that the main answer they are looking for is a plug and play from these results.

Of course there is a multi-stage algorithm that is novel, but it is not central to the question. And I agree that part is definitely novel and does not follow plug and play.

Am I missing something very basic in the premise and the main results or this is pretty much it?

### Questions
I have several questions/comments about the paper.

- I think the paper needs to work quite a bit on notation. There are several overloaded variables like $p$ (distribution and Besov space parameter), $r$ (radius and Besov parameter), $\tau$ (error bound, pointer in algorithm), $\mathcal{N}$ (covering number and Gaussian). This makes the paper much more difficult to read. Even in the appendix, the notations defined in the paper are not used as is, e.g., for a ball (the metric is often skipped). I think this requires a detailed revision of the paper.

- I am curious about the assumption that the reward function belongs to an anistropic Besov space. These spaces are extremely general function spaces that encode minimum regularity requirements in a sense. They are beneficial to characterize the class where NNs are better than linear function (because one is searching for general classes). Real world functions tend to offer more regularity than this general class and while the generality subsumes such functions, it fails to leverage their additional structure. In fact, the benefit over linear functions is only realized in a regime with $p \in [1, 2)$. Given that the advantage is only to be seen in specific subset of functions, that appear more often in mathematical constructs than they do in real life, I am curious about scenarios in real-world applications, where such a generality *actually* helps.

- In addition to the above question, most of the results omit leading constants and hold for large constants and sample sizes. Are such sample sizes relevant in inference time alignment, where typically we don't have very large datasets?

- Moreover, in typical application scenarios both $\Omega_X$ and $\Omega_Y$ are finite. How does one interpret these results for typical application scenarios like these since the results heavily rely on the continuity of the domain?

- In the definition of anisotropic Besov space, shouldn't $r$ be $\min \lfloor s_i \rfloor$? Typically, in fractionally smooth spaces, the derivative that is required to integrable is the greatest integer smaller than the smoothness parameter of the space.

- In Assumption 6, what do you mean by $s > d/p$. $s$ is a vector so that inequality does not make sense. Did you mean $\tilde{s}$?

- In Assumption 6, A1 is it not too strong to assume that the result holds for all $\epsilon > 0$. Firstly, $\epsilon \in (0,1)$ is a necessary requirement for the inclusion to hold. Secondly, the way the assumption is stated, it seems to be global property. Usually it is more reasonable to assume that it holds locally in a neighborhood.

- In Assumption 6, A2, what is $r$?

- In Lemma 19, you state there is an A3 of Assumption 6? I couldn't find any such assumption.

- In the mollification step, what do you do when $y$ goes beyond the domain $\Omega_Y$? Since $\Omega_Y$ is bounded but Gaussian is not, so it is important to handle such cases. Moreover, does that affect the analysis?

- What is the choice of $\sigma^{(\tau)}$? I see a value of $\tau^{\beta/d}$ in the appendix, but I could not find it in the main paper? What is the motivation behind this choice (except for the math working out)?

### Soundness
3

### Presentation
2

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
The paper studies inference-time alignment when the reward is learned and the true reward lies in anisotropic Besov spaces. It proves (i) a regret upper bound for inference-time alignment when the reward is estimated by a sparse ReLU network (Theorem 4), (ii) a lower bound showing linear estimators are rate-suboptimal in this setting (Theorem 5), and (iii) an improvement from a multi-step procedure with Gaussian perturbation that maintains coverage and yields a sharper regret rate (Theorem 7).

### Strengths
Clear theoretical story linking feature learning to regret. The paper formalizes why neural networks help in inference-time alignment: better L2 estimation over anisotropic Besov classes translates (via volume/coverage conditions) into better regret than linear estimators. Theorem 5 crisply separates the classes.

Concrete algorithmic instantiation. The work grounds the analysis in the chi square-regularized Inference Time Pessimism sampler (Algorithm 1) and uses a standard coverage–estimation trade-off inequality as the backbone (Eq. (2)), which is appropriate for inference-time alignment.

### Weaknesses
Strength of structural assumptions. The main gains hinge on Assumption 3 (non-negligible super-level set volume) and, in the multi-step analysis, Assumptions (A1)–(A2) in Assumption 6 that constrain the geometry/covering of $S_{\epsilon}$. These are plausible but may exclude adversarial or heavily spiked rewards in practice.

### Questions
Are the assumptions A1-A2  in Assumption 6 central to the works or can they be relaxed?

### Soundness
3

### Presentation
3

### Contribution
3
