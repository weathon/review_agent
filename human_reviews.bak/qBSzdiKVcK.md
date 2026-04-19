# Reality Only Happens Once: Single-path Generalization Bounds for Transformers

- Decision: Reject
- Scores: 3, 5, 3, 8, 6, 5

## Abstract
One of the inherent challenges in deploying transformers on time series is that \emph{reality only happens once}; namely, one typically only has access to a single trajectory of the data-generating process comprised of non-i.i.d.\ observations.  We derive non-asymptotic statistical guarantees in this setting through bounds on the \textit{generalization} of a transformer network at a future-time $t$, given that it has been trained using $N\le t$ observations from a single perturbed trajectory of a {bounded and exponentially ergodic} Markov process.  We obtain a generalization bound which effectively converges at the rate of $\mathcal{O}(1/\sqrt{N})$.  Our bound depends explicitly on the activation function ($\operatorname{Swish}$, $\operatorname{GeLU}$, or $\tanh$ are considered), the number of self-attention heads, depth, width, and norm-bounds defining the transformer architecture.  Our bound consists of three components: (I) The first quantifies the gap between the stationary distribution of the data-generating Markov process and its distribution at time $t$, this term converges exponentially to $0$.  (II) The next term encodes the complexity of the transformer model and, given enough time, eventually converges to $0$ at the rate $\mathcal{O}(\log(N)^r/\sqrt{N})$ for any $r>0$. (III) The third term guarantees that the bound holds with probability at least $1-\delta$, and converges at a rate of $\mathcal{O}(\sqrt{\log(1/\delta)}/\sqrt{N})$.  
Example of (non i.i.d.) data-generating processes which we can treat are the projection of several SDEs onto a compact convex set $C$, and bounded Markov processes satisfying a log-Sobolev inequality.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper provides generalization bounds for learning from ergodic/mixing/contracting stochastic processes and specializes these to transformer-type archictures via control of certain smoothness terms.

### Strengths
* I think the paper is mostly clearly written and well-structured. 

* The calculation of the derivatives of the transformer block is cool.

* The topic and problem formulation is certainly timely. Unfortunately, I do not think the results really adress the question in a way that was not more or less known a priori.

### Weaknesses
I did not find the results to be surprising and my overall assessment is that the paper unfortunately lacks the novelty to be published at ICLR. 

* The (primary) contribution of the paper is way overstated and I'd argue that the "first to" claims are more or less unwarranted:
  * First of all, the fact that an IID result can be ported to the mixing/contracting setting is standard via a blocking argument (the modern version of this argument being due to Yu, but dating back to Bernstein)
  * The $t$-step ahead risk for a contracting markov chain is morally equivalent to the time-averaged risk in most situations. There are already plenty of bounds for this situation in the literature that are easy to port over to the setting of the authors. Examples: [A,B,C,D]. In fact, I think it is concerning that standard results in learning theory about dependent processes are missing from the related work section (e.g., the work of Mohri and collaborators is completely absent from the ref section).
  * In other words: there is a certain degree of reinventing the wheel here, and the authors have unfortunately missed a line of related work that more or less implies results of this form. 
  * Indeed, I would argue that there are sharper estimates available in the literature (while limited to the square loss): reference [B] does not require any type of contraction whatsoever, and reference [D] only has very mild dependency on such contractivity.


* This also brings me to my second perhaps more conceptual point. To my understanding the authors seek to make the claim that their results is explanatory of LLMs. However, I don't believe natural language is well-explained by mixing stochastic processes: the words in a book certainly don't converge in any meaningful sense to a stationary distribution.  


* Arguably more minor, but the strict realizability assumption y=f(x) is rather more stringent than what has been assumed in past (uncited) work.


In sum, I think the results are mostly (1) standard uniform convergence arguments, (2) coupled with the stochastic regularity of assumption 2/3 and and (3) application of path-norm type bounds specialized to transformer type architectures.  In particular, (1/2) is entirely standard in my opinion and (3) (which is not stated as the primary contribution by the authors) does not really seem to be that surprising either in light of similar such bounds for deep nets.

[A] Mohri, Mehryar, and Afshin Rostamizadeh. "Stability Bounds for Stationary φ-mixing and β-mixing Processes." Journal of Machine Learning Research 11.2 (2010).

[B ] Simchowitz, Max, et al. "Learning without mixing: Towards a sharp analysis of linear system identification." Conference On Learning Theory. PMLR, 2018.

[C] Foster, Dylan, Tuhin Sarkar, and Alexander Rakhlin. "Learning nonlinear dynamical systems from a single trajectory." Learning for Dynamics and Control. PMLR, 2020.

[D] Ziemann, Ingvar, and Stephen Tu. "Learning with little mixing." Advances in Neural Information Processing Systems 35 (2022): 4626-4637.

### Questions
Is the reason your bound (ignoring the deviation term in delta) converges exponentially to 0 solely the fact y=f(x), i.e., strict realizability?

Given that the references [A-D] in my opinion provide similar type of gen/excess risk bounds. How would you say your work differentiates itself from these?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper addresses a challenge in applying transformer models to time-series data: the difficulty in achieving generalization guarantees when only a single trajectory of non-i.i.d. observations is available. The authors provide non-asymptotic generalization bounds for transformers trained on a single perturbed trajectory of a bounded and exponentially ergodic Markov process. The primary contribution is a theoretical result that gives a future-time generalization bound of $O(1/\sqrt{N})$, where $N$ is the number of training observations. This bound is derived considering the properties of the Markov process and the architecture of the transformer model, including activation functions, depth, and number of attention heads.

Specifically, the paper decomposes the generalization bound into three components:

Non-Stationarity Term: The difference between the stationary distribution of the Markov process and its time-dependent distribution, converging to 0 exponentially.

Model Complexity Term: A term that captures the complexity of the transformer network in terms of the number of self-attention heads, depth, width, and the activation function, eventually converging to 0 at the rate of $O(\log(N)^r/\sqrt{N})$ .

Probabilistic Validity Term: A term that guarantees the bound holds with probability at least , and converges at a rate of $O(\sqrt{log(1/\delta)/N})$.

The authors also obtain explicit estimates on the constants in these generalization bounds, which relied on explicitly bounding all the higher-order derivatives of transformers, considering their number of attention heads, activation functions, depth, width, and weight constraints.

### Strengths
1、The paper is well-written, with each section presented in a logical order. The introduction is particularly effective in clearly outlining the main contributions of the paper and providing a thorough discussion of related work.

2、This paper provides the first set of statistical guarantees for transformers compatible with many time-series tasks. More broadly, it presents the first results that estimate the future-generalization of a learning model given that its performance has only been measured using present/past historical training data.

3、The decomposition of the generalization bound into intuitive components helps in understanding the influence of different aspects of the model and data, making the theoretical results more interpretable and accessible to practitioners. The authors' explicit focus on the impact of transformer architecture—such as the number of attention heads, activation functions, depth, and width—on generalization performance is highly informative.

4、The paper provides appropriate examples of non-i.i.d. data-generating processes satisfying their assumptions, which helps readers understand the applicability of the assumptions.

### Weaknesses
1、In Theorem 1, the convergence rate of the Model Complexity Term $rate_s(N)$ for $d>2s$  is only $O(1/N^{s/d})$ , which implies that the generalization error bound may suffer significantly from the "curse of dimensionality"  in high-dimensional settings or when the transformer's architecture lacks sufficient smoothness (i.e., the partial derivatives of the transformer are not sufficiently large).

2、Since the authors point out the connection between their assumptions on non-i.i.d. data-generating processes and denoising diffusion models, it would be beneficial to provide empirical validation or experiments to demonstrate how well the derived bounds apply to practical datasets. Including empirical studies would strengthen the practical relevance of the work.

3、The paper could provide a more formal definition and explanation of the parameter "s" to help readers understand its significance in the context of the generalization bound.

4、The assumptions on the Markov process, such as boundedness and exponential ergodicity, might limit the applicability of the results to more complex real-world scenarios. For example, several dynamical systems and financial markets have long-term memory and are non-Markovian, which could reduce the generalizability of the findings

### Questions
The theoretical results in this paper rely on a key result that the transformer architecture admits continuous partial derivatives of all orders. This result seems quite strict, as in practical applications it is often sufficient for the model to be smooth only up to first or second-order derivatives to ensure stable gradient calculations and feasible training. Could the authors provide examples of transformer architectures or scenarios where the model does not admit continuous partial derivatives of all orders?

What are the implications for the generalization bounds in cases where transformers do not admit continuous partial derivatives of all orders?

In summary, my main concern lies in the result that the transformer possesses continuous partial derivatives of all orders, as well as the possibility that the generalization error bound might suffer from the curse of dimensionality. If the authors could clarify these points, I would be inclined to rate this paper higher, as it is really well-written and addresses a critical gap in applying transformers to time series analysis.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper derives generalization bounds for a Transformer model trained on a single time-series trajectory to generalize at a future points of the time series. It departs from the common i.i.d. assumption and assumes a Markovian time series with all points in its trajectory bounded under some norm and an exponential ergodicity property (this is akin to a fast-mixing property for a Markov chain). To bound the growth rate of the generalization error with respect to the parameters of the Transformer model, the authors define a $C^s$ norm which approximately captures the maximum value of any of the first s derivatives of the loss function with respect to the input. They then impose the condition that the $C^s$ norm grow at the rate of $O(s^r)$ for some $r>0$. The target function takes in the input $X_t$ at each time step and outputs a value $Y_t = f^*(X_t)$.
Under this model the work derives a generalization bound which depends on 3 terms
- (i) a term measuring how fast the chain converges to stationarity/ergodicity.
- (ii) a model complexity term which depends on the partial derivatives of the Transformer model.
- (iii) a probabilistic validity term to get the result to hold with high probability (as is common in such bounds)

A second result the paper provides is an explicit characterization of the model complexity term in the generalization bound above in terms of the $C^s$ norm bounds of different components of the Transformer model.

### Strengths
The paper aims to go beyond the standard modeling assumption of i.i.d. data which rarely ever truly holds in practice. Once you leave this assumption analysis of even simple processes can become hard. Moreover, the paper assumes a single path throughout training instead of assuming restarts.

The paper's notion of Markov data generating processes are general enough to capture multiple interesting settings.

### Weaknesses
A key essence of a transformer model is that it is a sequence model. It models the interactions between different tokens or $X_t$ for different values of t. It is very confusing and strange that the authors then proceed to assume $M=1$. At this point, the results then are simply just for a fully connected neural net and nothing specific about the Transformer really comes into play?

If the process mixes fast (in roughly $log(\kappa)$ number of steps) why cant we simply pay a multiplicative factor of $log(\kappa)$ to obtain essentially i.i.d. samples and then use the i.i.d. generalization bounds?

The main body of the paper does a poor job of explaining what the phase transitions in the model complexity term are. Why do they arise in the first place? i.e. why does the convergence rate accelerate abruptly as $s$ increases? Some intuition would help.

Writing is overly heavy with notation at parts. A key definition (the C^s norm) is defined in the Appendix. Makes it hard to build intuition while reading the beginning of the paper. It is unclear what is the significance of contributions of this paper. There is very limited discussion around the seemingly implicit assumption of $M=1$ which severely limits what a transformer model is capable of.

### Questions
- Q1. If Y_n only depends on X_n for one time instance what is the need of a sequence modeling framework like the transformer?
- Q2. What is Schwartz class?
- Q3. Definition 2 uses the C^s norm. Why is this presented before the actual definition of the C^s norm?
- Q4. What are the phase transitions in the model complexity term? Why do they arise?
- Q5. If $M=1$, there is no point in using an attention layer as all information we need to predict $Y_t$ is present in $X_t$ and looking into the past does not offer any added information. Where do the bounds on model complexity change to reflect this?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper titled "Reality Only Happens Once: Single-Path Generalization Bounds for Transformers" addresses a gap in the theoretical understanding of transformer models, particularly in non-i.i.d. settings commonly encountered in time-series data. Traditional statistical guarantees for transformers often rely on independent and identically distributed (i.i.d.) assumptions, which are seldom met in real-world applications such as natural language processing, finance, and reinforcement learning. This work extends the theoretical framework to single-path trajectories generated by bounded and exponentially ergodic Markov processes, providing non-asymptotic generalization bounds that converge at a rate of $\tetxrm{O}\left ( \frac{1}{\sqrt{N}}\right )$. The bounds are explicitly dependent on the transformer's architecture, including the activation function, number of self-attention heads, depth, width, and norm constraints.

### Strengths
Addressing a Significant Gap: The paper tackles the under-explored area of providing statistical guarantees for transformers trained on non-i.i.d. data, which is highly relevant given the prevalence of sequential data in practical applications.

Explicit Generalization Bounds: By deriving explicit bounds that depend on various architectural parameters of transformers, the paper offers valuable insights into how different components of the model influence its generalization capabilities. 

The paper situates itself well within the existing literature, contrasting its contributions with prior work on i.i.d. settings, in-context learning for linear transformers, and universal approximation theorems. This contextualization highlights the novelty and importance of the presented results.

### Weaknesses
Although the bounds are explicit, the dependence on multiple architectural parameters may raise concerns about scalability. It would be useful to discuss how these bounds behave as transformers scale to very large sizes, which is common in modern applications.

I would like to see more discussion about the limitations (or the lack of them) of the case $M=1$ in terms of how transformers are applied specifically to LLM data etc.

### Questions
The paper is done for the case $M=1$. For the case $M>1$, why can you apply the results of the paper for the augmented Markov chain 
$Y_n = ( X_{n-M+1}, \cdots , X_n   )$. 


Here is a list of typos I found. However for the one in line 2462 I would appreciate a clarification.

line 1124 the R^{(M)} should be R^M
line 1371 the expectation should taken over \nu measure
line 1409 On the statement of proposition there many typos, also in the proof of the proposition
line 2462 (and can easily be seen to be Markovian since Brownian motion has the strong Markov property)
line 2508 typos held, admitted -> holds admits.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper gives generalization bounds for the transformer architecture. The transformer is assumed to be trained on $N$ i.i.d. sample strings coming from a certain type of Markov process, and generalization bounds converging at the rate of $O(1/N^{0.5})$ are shown. 

The paper provides detailed calculations for the constant factor for the convergence rate bound, and studies how the bound is affected by changing the internal dimensions of the transformer model, as well as the activation function.  

The work uses tools from stochastic calculus, together with novel bounds on the partial derivatives of transformers.

### Strengths
- The paper gives bounds for an entire collection of hyper-parameter values and compares the resulting bounds.
- The presentation quality is good, and the introduction is well-written.
- The paper gives bounds for the higher-order partial derivatives of a transformer, which are a natural quantities to study in their own right and can potentially useful for future research.

### Weaknesses
- The human language, which is the primary use case of transformers, is known not to satisfy the Markov property. This somewhat undermines the relevance of this work to real-world generalization of transformers.
- The work does not compare the generalization bounds given here with experimental date on the generalization performance of transformers. My understanding is that when it comes to the current literature on generalization bounds, theoretical bounds often diverge from experimental conclusions. This often limits the usefulness of theoretical bounds when it comes to guiding the practice of machine learning.

### Questions
Are there some data-generating models that are more general than Markov processes, and for which one could hope to obtain generalization bounds for transformers?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 6

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies the generalization error of transformers in the case of non-i.i.d. data under the realizability assumption.
In particular, it is assumed that the data is generated by a discrete-time Markov process, and the generalization error is measured as the difference between the empirical sequential risk up to the first $N$ generated examples and the risk with respect to the distribution at a future time $t \\in [N, \\infty]$ (where the risk $t = \\infty$ is with respect to the stationary distribution of the data-generating Markov process).
The study of such a notion of generalization error (here called future-generalization error) aims at removing the standard i.i.d. assumption, which is unnatural for modeling the environments which transformers are usually employed in.
The authors provide non-trivial generalization bounds for the hypothesis classes consisting of transformer architectures, thus giving insights relative to the inherent properties of these models.

### Strengths
This work is well-motivated and tackles an important problem in the context of transformers.
Transformers are currently one of the most used models in multiple domains, and understanding their generalization properties is definitely a relevant research direction.
The paper has a good structure and the results are clearly presented.
The specification of the generalization bounds to classes of transformer architectures is thus interesting and the authors provide a good and thorough discussion on the implications of their results.
Consequently, the results contained in this work allow the reader to gain further insights into the interplay between structural properties of transformers and their generalization capabilities.

### Weaknesses
Although the paper supports the results with reasonable motivations and a detailed presentation of related work, its main weakness is that the main results are not novel.
More precisely, the study of generalization errors in the case of non-i.i.d. data has been thoroughly studied in some notable work (e.g., see [1-3]) up to more than 15 years ago, with some ideas even dating back to [4].
Not only that, but those results appear to potentially subsume the analysis of the future-generalization error (the central part of this submission), considering both the conditional and the unconditional "future risks" with respect to the historical data $(X_i)\_{i \\in [N]}$, as well as the more general non-stationary processes such as $\\beta$-mixing ones.
These already available generalization bounds do not even require the specialization to the transformer model, introducing a general dependence on the Rademacher complexity of the hypothesis class; hence, it is unclear whether it would have simply sufficed to study the Rademacher complexity of the class of transformers in order to derive analogous results as the ones provided in the current submission.

This line of work has been overlooked as a whole, which is a significant and nonnegligible shortcoming of this submission.
This additionally renders claims such as "this is the first result which estimates the future-generalization [...] of a learning model given that its performance has only been measured using present/past historical training data" (lines 77-81) false; this claim is even presented as the primary contribution of the paper, which is misleading.
The authors should have provided a more detailed discussion on how their results relate to the existing literature on generalization bounds for non-i.i.d. processes.

References:

[1] Mohri & Rostamizadeh. "Stability Bounds for Non-i.i.d. Processes". NeurIPS 2007.

[2] Mohri & Rostamizadeh. "Rademacher Complexity Bounds for Non-I.I.D. Processes". NeurIPS 2008.

[3] Kuznetsov & Mohri. "Generalization bounds for non-stationary mixing processes". Machine Learning, vol. 106, 2017.

[4] Yu. "Rates of convergence for empirical processes of stationary mixing sequences". The Annals of Probability, vol. 22, 1994.

### Questions
- Could you comment further on how your results relate to the existing literature on generalization bounds for non-i.i.d. processes (especially the references mentioned above)? In particular, is there any contribution in this submission that goes beyond (i.e., is not implied by) any of the existing results?
- Could you provide more details about the need of the Markovianity assumption in the data-generating process relative to the proofs of your results? What are the main obstacles in lifting this assumption?
- At line 56, you define the $t$-future risk $\\mathcal{R}\_t(\\mathcal{T})$ as an expectation over $X\_t$. Is this conditioned or not with respect to the data points $X\_1, \\dots, X\_N$ which $\\mathcal{T}$ can depend on?
- Please, address any other issues outlined above.

Minor comments and typos:
- Parenthesization of citations is sometimes incorrect or odd (e.g., line 32).
- In the abstract, use "$1-\\delta$" instead of "1-$\\delta$".
- At the math displays of lines 56 and 61, do you mean to have $\\mathcal{T}$ as the argument of the risks instead of $f$?
- Line 71, "with probability of at least" should be "with probability at least".
- Line 95, "instance-dependant" should be "instance-dependent".
- Line 105, "multilayer perceptions" should be "multilayer perceptrons".
- Line 120, "by Markov processor at least" should be "by a Markov process or at least".
- Line 129, "this means that that" should be "this means that".
- Lines 139-141, the sentence is quite unclear as the phrasing is convoluted. Please, revise.
- Line 150, "are all most equal" should be "are almost equal".
- Lines 319-320, "convergence rate acceleration are expressed using" should be "convergence rate acceleration using".
- Math display at line 330, is $I\_{N \\in [\\tau\_s, \\tau\_{s+1})}$ supposed to be the indicator function? If so, I'm not sure this has been explicitly defined before.

### Soundness
2

### Presentation
3

### Contribution
2
