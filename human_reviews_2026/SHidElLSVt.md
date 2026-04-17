# Transformers Learn Latent Mixture Models In-Context via Mirror Descent

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8, 6

## Abstract
Sequence modelling requires determining which past tokens are causally relevant from the context and their importance: a process inherent to the attention layers in transformers, yet whose underlying learned mechanisms remain poorly understood. In this work, we formalize the task of estimating token importance as an in-context learning problem by introducing a framework based on Mixture of Transition Distributions, where a latent variable determines the influence of past tokens on the next. The distribution over this latent variable is parameterized by unobserved mixture weights that transformers must learn in-context. We demonstrate that transformers can implement Mirror Descent to learn these weights from the context. Specifically, we give an explicit construction of a three-layer transformer that exactly implements one step of Mirror Descent and prove that the resulting estimator is a first-order approximation of the Bayes-optimal predictor. Corroborating our construction and its learnability via gradient descent, we empirically show that transformers trained from scratch learn solutions consistent with our theory: their predictive distributions, attention patterns, and learned transition matrix closely match the construction, while deeper models achieve performance comparable to multi-step Mirror Descent.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors set up a synthetic task where to perform well at each step the transformer must estimate mixing weights for a set of transition distributions (the mixture of transition distributions task). They show that posterior mean inference for this task is intractable and then discuss one step of mirror descent as an attractive estimator of the mixing weights. They then show theoretically and empirically that disentangled transformers implement mirror descent to learn mixture weights for this process.

### Strengths
1. I really like the matrix diagrams in section 4, it made following that section a lot easier. I felt that the paper was well written and structured. 
2. To my knowledge no one has previously proposed that ICL involves transformers performing mirror descent. This gives a new perspective on ICL
3. I had not known of the MTD model before reading this, it appears to be a fairly well motivated synthetic task in this setting.

### Weaknesses
The lack of any experiments involving non-disentangled transformers weakens this papers claims. It's not clear to me that a general non-disentangled transformer would implement this mirror descent step. If you consider the Nichani et al. (2024) paper cited in this work, they also perform a theoretical and empirical analysis with disentangled transformers, but they include experiments involving standard non-disentangled transformers also. Something similar in this case would greatly improve the strength of this papers evidence for it's claims and make the title more justified.

### Questions
- Is there any reason why 3 and 6 (or 5, there appears to be an inconsistency between the main text and the appendix) layers were chosen? Does it also hold for other layer numbers?

- Are there any plans to extend this to a more realistic data generating process where tokens can depend on multiple previous tokens at different time lags?

- Would the 5/6 layer disentangled transformer be amenable to the same sort of analysis as the three layer model, showing that it implements two step MD?

### Soundness
2

### Presentation
4

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
This paper introduces a framework for in-context learning with latent variable modelling, based on a mixture of transition distributions. 
The authors introduce a synthetic task that reframes the estimation of past tokens to learning latent mixture weights. The paper also proves that the learned algorithm is mirror descent. The paper also empirically shows that the learned algorithm for a **disentangled** transformer is mirror descent. Finally, the paper shows that the estimator is an approximation of the Bayesian optimal solution.

### Strengths
The paper has the following strengths:
1. Clear framework presentation with latent variable modelling.
2. Good theory for the disentangled transformer
3. Good link to Bayse optimal solution

### Weaknesses
1. While the theory is solid and I consider it a good contribution, the main problem is that the results are focused only on the disentangled transformer.
 - The paper claims that the results were proved for the general transformer (for example, in lines 480-481). 
-  The authors should make it clear that they refer to a disentangled transformer, which is not the same as the generally used one for practical tasks.
2. The paper should try to extend results to the general case transformer, or state why they do not work. 
3. The experimental section should be improved.
- Could more experiments be added for real tasks like language modelling?
- Could the models in experiments be extended to the general transformer?
4. Small typos:
- 076, "patterns math our"? 
- 205-206: "by a Transformer" Do you mean a disentangled transformer?

### Questions
Could more experiments be added for real tasks like language modelling?
Could the models in experiments be extended to the general transformer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the Mixture of Transition Distribution (MTD) task, where a latent variable must be learned in-context. The authors show that an ad hoc disentangled transformer is able to perform one step of Mirror Descent (MD), a mechanism derived from initializing on a specific prior distribution. This MD step serves as a first-order approximation of the Bayesian posterior mean. For longer sequences, they empirically demonstrate that Multi-step Mirror Descent approximates the Bayesian posterior mean more effectively. They also show that training disentangled transformers match the results obtained by their construction.

### Strengths
The primary strength of this paper lies in its conceptual originality and technical rigor in tackling the challenge of dynamic latent variable inference. The paper's most original step is presenting the Mixture of Transition Distributions (MTD) model as an in-context learning task requiring the dynamic estimation of latent mixture weights. This approach successfully moves the field past simpler models (like linear regression or elementary Markov Chains). By identifying and proving that a disentangled transformer implements one step of the MD algorithm, the paper establishes a new insight: a trained disentangled transformer (or the attention part of transformers) in this context can effectively approximate the Bayesian posterior mean and the natural algorithm to achieve it. This provides an algorithmic explanation for how transformers update their beliefs about dynamic sequence properties, representing a good contribution to understanding transformer capacity.

### Weaknesses
I believe there might be a limitation in the scope of the generalization claim. To truly support the title's broad assertion that transformers learn via Mirror Descent, the paper should include an experiment demonstrating that this mechanism holds up in a standard transformer architecture (including MLPs). Furthermore, since the formal proof only covers a single MD step, it would be highly valuable to provide an explicit discussion or intuition regarding the theoretical path for two or more MD steps and whether an extra iteration naturally leads to a quadratic approximation of the Bayesian posterior mean.

### Questions
1. On line 75 you claim the training of the transformer is using Gradient Descent, then you actually use ADAM and do not give a reason, I do not think they will both converge to the same critical point, could you comment on this?.
2. Could you give a discussion where having a known transition matrix \pi is realistic? I might not get why you were using rows from a Dirichlet distribution.
3. How might non-linear activation functions (like ReLU or GELU) interplay with the smooth, exponentiated gradient algorithm required for the MD update?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies transformers trained on the MTD (Mixture of Transition Distributions) task. They show that a three layer disentangled transformer can naturally represent one step of mirror descent on the maximum likelihood objective. They also show that trained transformers have the same attention maps as their theoretical construction and obtain similar performance to mirror descent with optimal step sizes, supporting the hypothesis that transformers solve this task via mirror descent.

### Strengths
The task is a natural generalization of previous in-context learning tasks studied in the literature and removes the dependence on a fixed causal graph. The paper is also very well written, especially section 4 (the explicit construction). In addition, while the experiments are relatively limited, they do support the paper's thesis that transformers solve this ICL task via mirror descent.

### Weaknesses
While the paper provides a construction for a three layer disentangled transformer that can express one step of mirror descent, it doesn't study optimization so it's still unclear how gradient descent finds this solution. To support the claim that transformers learn this construction, the paper relies on experiments on disentangled transformers which show that both the attention maps and the overall performance match their explicit construction. However, there are a few weaknesses of the experiments:

- They don't verify agreement in weight space between the trained transformer and their construction, they only verify the attention maps/KL divergence. Since the attention maps are input dependent, it's difficult to use a single attention map to verify equivalence of two architectures.
- The agreement in attention maps isn't very convincing, for example while the first attention map matches the diagonal structure of the explicit construction, the actual weights appear to be quite far apart. Additionally, the middle attention map has a strong banded structure which is not captured by the explicit construction.
- I believe that the paper only empirically validates their claim on the disentangled transformer. However, if they only verify final performance and attention maps, it would be good to include a "standard" transformer baseline with projections and MLPs, since you can still easily verify attention maps and performance on these. The main advantage of training a disentangled transformer is to look for weight-space agreement.
- The only evidence that transformers perform multi-step mirror descent are the performance plots in Figure 5. It would be good to verify attention maps or ideally the weights themselves to check whether this is actually true or whether it's implementing a different estimator with similar performance (although this seems unlikely).

Minor points:
- In the first part of the abstract/paper, the claim is phrased as "transformers implement Mirror Descent," however the real claim in these sections is that transformers *can* implement mirror descent, and then the empirical part of the paper tries to argue that they do.
- line 151+152: While predicting $\lambda$ is probably a necessary step for most good estimators, the actual task is just to predict $y_t$ given $y_{1:t-1}$, not to predict $\lambda$?
- line 227: overhang in the main paper

### Questions
- What do the weights of the trained disentangled transformer look like?
- How would the performance of the disentangled transformer compare to a standard transformer on this task? Is it clear that the MLPs won't help?
- For the multi-step transformer, were the step sizes for each step tuned independently or were they tied together? In the trained transformer can you read off the step sizes? Are they the same or does it use a schedule?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The main problem the paper tackles is: How do Transformers perform in-context learning when the task requires inferring latent mixture weights on the fly instead of machine learning algorithms? To study this problem, the authors introduce a synthetic ICL task based on the Mixture of Transition Distributions (MTD) model. Methodologically, they explicitly construct a 3-layer disentangled Transformer that mirrors the one-step mirror descent algorithm. Empirically, the show that the trained transformer can approximate the theoretical construction.

### Strengths
* Extends the “transformers-as-optimizers” view from GD on fully observed tasks to mirror descent (exponentiated gradient) on latent mixture weights, a novel setting (latent variables over the simplex) rather than another linear-regression variant.
* The paper is well-written and easy to understand. The proof idea illustration in the main text is very helpful.
* The theoretical construction is backed up by the empirical results.

### Weaknesses
1. The theoretical analysis relies on a simplified, MLP-free Transformer architecture. Hence, it remains uncertain whether the same mirror-descent behavior holds in standard Transformers with MLPs (But I understand this is always the limitation of all constructive proof for almost all ICL theory papers)

2. Proposition 3 and the constructed model require access to the transition matrix $\pi*$, which limits applicability when base dynamics are not shared/known. This $\pi*$ is later encoded in the output matrix $\tilde{W}_O$. Doesn't this mean that the transformer has no chance to learn the transition matrix of the MTD model? If this is the case, how can the theoretical result justify that the transformer is "learning the latent mixture model in context" if it only learn the mixture weight, not the transition matrix? I'm not an expert in MTD, so I might misunderstand the concept, and will appreciate further explanation on this

### Questions
1. Why did the author choose the disentangled transformer architecture to analyze? What will happen if we add back the MLP layer?

2. I think it would be helpful if the authors could elaborate on the choice of the MTD model as a tool to analyze the latent structure of the data. The authors already provide a formal introduction in Section 3, but it might strengthen the paper if that formal definition were explicitly linked back to the intuitive example in Figure 1, which clearly motivates why we want to analyze ICL beyond the machine-learning algorithms studied in previous works.

### Soundness
3

### Presentation
3

### Contribution
2
