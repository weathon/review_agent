# Provable Convergence of Single-Timescale Neural Actor-Critic in Continuous Spaces

- Decision: Reject
- Scores: 3, 5, 5, 6

## Abstract
Actor-critic (AC) algorithms have been the powerhouse behind many successful yet challenging applications. However, the theoretical understanding of finite-time convergence in AC's most practical form remains elusive. Existing research often oversimplifies the algorithm and only considers simple finite state and action spaces. We analyze the more practical single-timescale AC on continuous state and action spaces and use deep neural network approximations for both critic and actor. 
Our analysis reveals that the iterates of the more practical framework we consider converge towards the stationary point at rate $\widetilde{\mathcal{O}}(T^{-1/2})+\widetilde{\mathcal{O}}(m^{-1/2})$, where $T$ is the total number of iterations and $m$ is the width of the deep neural network.  To our knowledge, this is the first finite-time analysis of single-timescale AC in continuous state and action spaces, which further narrows the gap between theory and practice.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper provides a theoretical analysis of an actor critic-type algorithm in continuous spaces, and contributions are theoretical in nature.

### Strengths
- The paper is written well. 
- The mathematical analysis is easy to read.

### Weaknesses
There are some weaknesses in the proposed approaches, as listed below: 

1. Continuous spaces are important and challenging, but the analysis seems to make significant assumptions that are not realistic in practice. Assumption 4.7 seems unrealistic, and the authors have not discussed it in detail. I think this assumption is standard in tabular settings, where one can easily find an upper bound, but for continuous, it could be tricky.  It would be nice to give some realistic examples where this assumption would hold. 

2. Given the existing analysis in the literature, because the authors are operating in parametrized settings, extending the analysis to continuous settings seems incremental and novelty is limited. it would be nice to further add the specific challenges that exist in extending the analysis from tabular to continuous settings and what efforts were made to deal with such challenges. The proof is almost similar to the following paper, with only some additional generalizations of the assumptions.

Chen X, Zhao L. Finite-time analysis of single-timescale actor-critic. Advances in Neural Information Processing Systems. 2024 Feb 13;36.

3. Also assumption 4.6 also requires discussions; the papers authors have cited are mostly in tabular settings. Is it valid to assume the same assumption for continuous spaces as well? 

4. How assumption 4.5 is related to exploration is not clear. Is there any scenarios where this can hold? 

5. Discussing all these assumptions is important because the paper's motivation is to study the analysis in the most practical settings; if the assumptions are not realistic, that would defeat the purpose.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors analyze single time scale actor critic algorithms for continuous state and action space control problems. The results show that the convergence to a stable point at a rate that depends on the number of iterates and the width of the controlling network.

### Strengths
Strong results for an important problem. 

Considering Markovian sampling makes the result even stronger. 

The assumptions are explained very clearly. This is commendable as there are many assumptions ....

### Weaknesses
It is not clear how $c$ is chosen for THM 4.9 to hold. This seems to be something the algorithm would have to know a-priori. But how can that be? This seems like a pretty complex constant AFAIU. 

I don't understand why one would not take $m \to \infty$: in that case the second term is 0, but wouldn't neural network convergence issues emerge? I mean, this is something that should be reflected in the bound somehow. Or am I missing something?

I would appreciate to see a proof outline. Especially, pointing out to the novel elements of the analysis is crucial. Since the paper is really an exposition to the appendix where the proofs are, it is rather hard to follow where the real novelty is. 

Finally, if there is any empirical evidence for the (low) dependence on $m$ empirically for large $m$, this would go a long way to convincing me in the utility of the paper. 

I'd be happy to raise my score, but I would like to see detailed discussion of the above.

### Questions
Please see weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors consider the convergence of single time scale actor critic algorithm in continuous state and action spaces with infinite horizon average reward as the performance metric. The state and action spaces considered are assumed to be compact and the value functions and the policies are parametrized through deep neural networks. The average reward, the relative value function and the policy are all updated on a similar time scale. Finite time convergence guarantees to a stationary point are provided under the overparametrized neural network regime. They resort to a NTK style analysis where by the virtue of overparametrization, the weights do not change much in magnitude by the end of the training period.

### Strengths
1. Most prior literature in RL (especially average reward RL) considers finite state and action spaces. Hence the setting of continuous state and action spaces is relatively understudied given its significance. 
2. Neural networks (NN) have consistently demonstrated strong capabilities as function approximators, with prior results showing that they can approximate continuous functions to any desired accuracy. However, their theoretical foundations remain relatively underexplored. This paper addresses a crucial problem, particularly given the extensive practical applications of neural networks.
3. They characterize tight bounds in terms of dependence on the number of iterations and number of parameters of the NN.

### Weaknesses
1. Since this paper is motivated by the promise of neural networks in policy optimization in the realm of RL, simulations demonstrating the same would increase the strength of the paper. As of now, the paper analyzes the standard actor critic algorithm and provides finite time bounds, but relating these to whats observed in practice would vastly help with understanding (especially in terms of determining the various step sizes required for updating different quantities of interest).
2. The paper provides convergence to a stationary point which can be a local minima. However, some of the recent works have proposed global convergence guarantees for average reward problems. See ref below. Although these are for finite state and action spaces, the optimization landscape might be the same independent of the finiteness of the problem parameters.
3. The assumptions require the nonlinear activations to be differentiable and smooth, this precludes the use of ReLU. However the authors suggest other nonlinear activations which indeed satisfy these assumptions. Once again, a simulation to demonstrate their practicality would help.
4. Regarding Assumption 4.3: Does this assumption require the original value functions to also be smooth with respect to the underlying parameter $\theta$. Since the parameters are initialized from a normal distribution, it is not clear as to how assumption 4.3 can be satisfied in practice (the uniform boundedness of the optimal parameters corresponding to the value function estimate). In case of linear function approximations, this assumption is trivial since typically $\|\phi\| \leq 1$, where $\phi$ is the feature vector. Its not clear whether this assumption can be satisfied in a straight forward manner when using neural networks. 
5. Regarding Assumption 4.4: Since this assumption holds for every state, when dealing with uncountably infinite states, it is unclear as to how $\lambda_1$ can be uniformly bounded from below across all states.




[1] Improved Sample Complexity for Global Convergence of Actor-Critic Algorithms
N Kumar, P Agrawal, G Ramponi, KY Levy, S Mannor 2024
[2] On the global convergence of policy gradient in average reward markov decision processes
N Kumar, Y Murthy, I Shufaro, KY Levy, R Srikant, S Mannor 2024

### Questions
1. Equation 1, what does the measure $d(a\times s')$ represent?
2. Theorem 4.9 $\alpha = \frac{c}{\sqrt{T}}$, what are the typical values of $c$ in this learning rate?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proves a novel result for average reward MDP's. Prior convergence results for actor critic for average reward were only able to prove convergence for a finite state action spaces. This paper is the first result of its kind and thus deserves acceptance in my opinion. I do however have some concerns.

### Strengths
The paper is well laid out and easily readable. The appendix has been arranged in a way that makes it easy to follow. The biggest strength of the paper is the novel contributions that it has.

### Weaknesses
1. Even though the results proved in the paper are novel, the proof seemed quite derivative of the paper Chen 2024.


References: 


Xuyang Chen and Lin Zhao. Finite-time analysis of single-timescale actor-critic. Advances in Neural Information Processing Systems, 36, 2024.

### Questions
1. I have a concern about the assumption 4.5. Specifically the term ${\mu_{\theta}}{\int_{\mathcal{S}{\times}{\mathcal{A}}}}(\pi_{\theta}-1)\mathcal{P}(s^{'}|s,a)d(a{\times}s^{'})$. Since the term $(\pi_{\theta}-1)$ for all state action pairs, is this term not always negative? This would make the assumption 4.5 not true. It is likely I am misunderstanding the notations, so some clarity here would be helpful.
2.  On page 24. In the second left hand side for $Z_{T}$. I am having trouble figuring out where the $\mathcal{O}\left(\frac{1}{\sqrt{m}}\right)$ has come from.  It would be helpful if some information would be given in this regard.

### Soundness
4

### Presentation
3

### Contribution
4
