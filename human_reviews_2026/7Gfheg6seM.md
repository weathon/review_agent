# Minimax Rates for Learning Pairwise Interactions in Attention-Style Models

- Decision: Accept (Poster)
- Scores: 10, 4, 2, 8

## Abstract
We study the convergence rate of learning pairwise interactions in single-layer attention-style models, where tokens interact through a weight matrix and a nonlinear activation function. We prove that the minimax rate is $M^{-\frac{2\beta}{2\beta+1}}$, where $M$ is the sample size and $\beta$ is the H\"older smoothness of the activation function. Importantly, this rate is independent of the embedding dimension $d$, the number of tokens $N$, and the rank $r$ of the weight matrix, provided that $rd \le (M/\log M)^{\frac{1}{2\beta+1}}$. These results highlight a fundamental statistical efficiency of attention-style models, even when the weight matrix and activation are not separately identifiable, and provide a theoretical understanding of attention mechanisms and guidance on training.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper studies the statistical problem of learning pairwise interactions in sequences with attention layers. The learning task is a sequence-to-sequence task, which consists in an attention matrix (i.e., a quadratic form in the input tokens) followed by element-wise activation and averaging. The goal is to learn the activation function (among a class of $\beta$-smooth functions) and the attention matrix (which can incorporate a rank constraint). Authors show that the empirical risk minimizer reaches minimax optimality for this problem, and that the minimax rate depends only on the smoothness of the activation, and not on the dimension of the data and on the sequence length.

### Strengths
The paper tackles important questions related to characterizing the statistical properties of sequence-to-sequence tasks that are representative of settings where Transformer performs well (which typically differ from usual time series models in statistics). In particular the paper studies pairwise interactions between input tokens, which are known to be a crucial feature of attention models, under reasonable assumptions on the data distribution. Obtaining the minimax rate and showing that it is reached by the ERM is a good contribution. The paper is clear to follow, and the proofs seem correct.

### Weaknesses
I did not see any major weakness. While I give suggestions below to improve the impact of the paper, I do not expect the authors to answer on these points during the rebuttal.

Although the paper is well-written, I encourage the authors to revisit their paper to give more interpretation of what is going on. For instance, while the authors state several times that the attention avoids the curse of dimensionality in their model, they do not explain _why_: because the nonparametric estimation problem is a unidimensional problem, while in high-dimension you "only" have to estimate a matrix. Similarly, the authors begin in the introduction by stating that the problem is difficult because we only observe an average interaction over the tokens. I imagine this is solved by the exchangeability assumption, which also explains why the sequence length does not appear in the rate, but this could be explained a bit more clearly. I also think it would be very insightful to state the equation on line 859 as an intermediate result in the main text, before deriving the asymptotic rate by assuming $rd$ is small enough and solving for $K_M$.

Minor remarks:
- line 183: while I appreciate going from independence to exchangeability, I suggest rephrasing this paragraph, because exchangeability is not much more realistic than independence when thinking about NLP.
- line 238: $R_g[X]$ is not defined yet at this point.
- line 243: please explain somewhere in the main text how $K_M$ is chosen. Besides, $K_M$ is a natural number so there is a floor function missing in equation (A.20).
- line 252: state that this estimator is the empirical risk minimizer.
- line 292: I disagree with the formulation of the last sentence of the remark. It suggests that taking a rougher activation function helps to mitigate the curse of dimensionality, which does not seem to be a correct interpretation. Indeed, in the proof, before replacing $rd$ using the assumption of the theorem in eq. (A.19), one sees that increasing either $d$ or $\beta$ can only degrade the bound. I would rather say that the contribution of the dimension $d$ to the error comes from the estimation of $A_*$, and that this term does not dominate as long as $rd \leq (M/\log M)^{1/2\beta + 1}$, and that for a fixed $d$ this condition becomes looser as $\beta$ decreases because the other error terms increase.

### Questions
- line 125: most theoretical works rather assume that the particles are on the sphere rather than on the cube, due to the layer normalization operation in Transformers. Do you think it would be possible to change to the sphere?
- line 150: the formulation is misleading as it may lead the reader to believe that softmax is included in the setting under study, whereas only element-wise activations are included. How difficult would it be to extend the results to a function $\phi_\star: \mathbb{R}^n \to \mathbb{R}$?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies a model of $N$ particles with pairwise interactions which resembles attention networks. In particular, one can consider the problem under study as being given $M$ examples of $N$ tokens each in dimension $d$ and the measures of their interaction strength, defined as an element-wise non-linearity applied to a linear dot-product attention of rank $r$ (plus some noise). Given this data, you wish to recover both the attention matrix and the non-linearity. The authors prove that the error on learning the interaction models decays as a certain dimension-free power of the number of examples $M$ as long as the product $rd$ is smaller than a power of $M/log(M)$. This result is verified in one case on synthetic data.

### Strengths
The paper is clear and well written, making the proof easy to follow. Within the modeled under consideration the results are intriguing and sufficiently novel. In particular the authors have very permissive assumptions on the distribution of the examples and the additive noise. I believe the contribution to the theory landscape can be significant, as not many theory papers tackle the problem of dependent token sequences.

### Weaknesses
Although the mathematical analysis is solid, the link to real attention mechanisms is tenuous. The model studied is not an actual dot-product attention layer: it lacks the row-wise softmax normalization that defines attention, and instead applies an element-wise nonlinearity to pairwise scores. As a result, the interacting-particle formulation captures only a simplified kernel-like approximation of attention. I would have liked the authors to better justify why this simplified model is a meaningful generative proxy for attention data, and why its minimax rates are relevant to understanding transformers. A broader empirical study comparing to actual softmax attention or multi-head layers could have clarified the connection and strengthened the practical implications.

Minor typo: in line 154 you refer to $g_*$ in 2.1, which is not there, but a few lines later.

### Questions
1. Would it be possible to extend your model to include row-wise non-linearities? I believe it would be incredibly interesting, as it would be a data generating model that naturally mimics softmax attention.
2. Could you provide some practical examples of data that you believe can be effectively described by your particle model?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies learning interactions in attention-style models where the parameters are unknown. Specifically, they consider the function $Y_i=\frac{1}{N-1}\sum_{j\ne i} \phi(X_i^\top AX_j) +\text{noise}$ from $M$ i.i.d. samples $(X,Y)$ where input $X=(X_1,\cdots,X_N)$ consists of $N$ possibly dependent $d$-dimensional tokens and output $Y=(Y_1,\cdots,Y_N)$ measures average interactions. When $A$ is a rank $r$ matrix and $\phi$ is a $\beta$-Hölder function, the empirical risk minimizer is shown to have error $M^{-\frac{2\beta}{2\beta+1}}$ when $M\gtrsim (rd)^{2\beta+1}$.

### Strengths
The paper derives near-optimal minimax rates for estimation of interaction models such as a self-attention layer. This goes beyond existing frameworks for analyzing optimality of regression models as each sample consists of a possibly dependent sequence of tokens.

### Weaknesses
My main concern is novelty and motivation. I do not believe the results are interesting enough to merit publication.

* The problem is not clearly motivated. When do we need to estimate unknown parameters of attention models?

* The main result (i.e. the dimension-free rate) seems straightforward. Since $\phi$ is a 1-dimensional $\beta$-Hölder function (not $d$-dimensional), it is intuitive that $d$ does not appear in the exponent, and so the dependency on $d$ only arises through learning $A$ which is $rd$-dimensional due to low rank. So there is no "curse of dimensionality" to avoid in the first place, this does not really point to a "fundamental statistical efficiency" of attention models as the paper claims. Also for the comment in the abstract on weight matrix and activation not being separately identifiable, this is not really surprising as identifiability of parameters does not affect minimax rates in general.

* The paper claims the main challenge is going beyond the independency assumption, however the sample objects $X^m$ for $m=1,\cdots,M$ are assumed to be i.i.d. and only the tokens $X_1^m,\cdots,X_N^m$ can be dependent (with an exchangability assumption which immediately implies coercivity). Since the rate is given only for $M$ and not convergent in $N$, this also does not seem surprising. Other aspects of the proof seem more or less standard in minimax analysis (covering number bound, controlling tails by chaining, lower bound via Fano's inequality).

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper concerns the study of asymptotic sample complexity for learning attention like models.
The main claim is that attention like models are free from the curse of dimensionality.
More precisely, while classical results for function regression leads to $M^{-2\beta/(2\beta+d)}$ scalings,
with beta the Holder exponent characterizing the smoothness of the function and d the embedding dimension,
they get a scaling corresponding to $d=1$.

### Strengths
The paper is making a strong statement, which is interesting in many respects and raises a lot of questions. It seems technically
solid (with quite general assumptions for the noise in particular), being grounded  on proven techniques (which I am not specialist of)
for obtaining minimax rates for non-parametric regressions,  which are adapted here to a context where the function is  observed through an average over particles, hence the function estimation is a deconvolution type inverse problem more involved than the usual single index model inference. The result is nicely backed by numerical  experiments. The paper is clearly written, despite the rather lengthy and technically complex  line of arguments.

### Weaknesses
Beside the technical proof, there is maybe a lack of interpretation of this result, which at least at first glance might look very surprising.
Not much intuition is given, and one wonder then if this quite paradoxical result stems from the assumptions which are taken.

### Questions
In order to better understand this result I have the following questions:

- do the scaling in $M^{-2\beta/(2\beta+1)}$ corresponds to what would be obtained if the attention matrix A would be known in advance,
because then we are back to the parallel regression of N functions of a real variable (x=X_i^tAX_j, leaving aside the extra difficulty of the convolution over the particles j)? If this interpretation is correct,  does this mean that the identification of A (i.e. the r directions among d)  do play a role only in a transient regime? is this the reason for the hypothesis that one need a number of samples $M > (rd)^{2\beta+1}$ (in my understanding this is the amount of samples it takes to approximately align with A), beyond which the sample complexity becomes 1d?
- Is this also related to the reason why the rank of the matrix is not involved in the asymptotic?
- I did not understand the role played by the interacting particle system interpretation for obtaining this result, it seems not to play any role in the proof.
- Concerning the regression assumption: usually the g function is actually not observed, it represents a  kind of virtual supervised function regression setting taking the functional form of an attention layer, so why should we expect this regression setting to be relevant to the full attention layer, i.e. when combining the attention weights with the values?
- if we compare to ordinary NN layers, should we obtain similar behaviour? I presume for a single layer function like $f(x) = W_1\phi(W_2^T x)$ the same rate should hold. What about more complex architectures? a rapid summary of existing results obtained so far for NN functions could be useful.

Overall my recomendation would be to save a little bit a room in the plain text to discuss the interpretation of the results.

### Soundness
4

### Presentation
3

### Contribution
4
