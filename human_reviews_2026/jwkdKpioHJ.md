# Tuning the burn-in phase in training recurrent neural networks improves their performance

- Decision: Accept (Poster)
- Scores: 4, 4, 8

## Abstract
Training recurrent neural networks (RNNs) with standard backpropagation through time (BPTT) can be challenging, especially in the presence of long input sequences. A practical alternative to reduce computational and memory overhead is to perform BPTT repeatedly over shorter segments of the training data set, corresponding to truncated BPTT. In this paper, we examine the training of RNNs when using such a truncated learning approach for time series tasks. Specifically, we establish theoretical bounds on the accuracy and performance loss when optimizing over subsequences instead of the full data sequence. This reveals that the burn-in phase of the RNN is an important tuning knob in its training, with significant impact on the performance guarantees. We validate our theoretical results through experiments on standard benchmarks from the fields of system identification and time series forecasting. In all experiments, we observe a strong influence of the burn-in phase on the training process, and proper tuning can lead to a reduction of the prediction error on the training and test data of more than 60% in some cases.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper assesses the effects of training recurrent neural networks using truncated back-propagation through time (TBPTT). In TBPTT the full training sequence is split into a number of overlapping sub-sequences to which learning is applied. Each sub-sequence includes an initial part (the burn-in phase) that is excluded from the loss. This paper seeks to access the loss in performance produced by approximating BPTT with TBPTT, and the effects of the burn-in period on performance.

### Strengths
The paper combines both theoretical and empirical research.

It well structured and generally clearly written.

### Weaknesses
The theoretical and practical consequences of this work are unclear.

The most detailed empirical results are provided for a synthetic task that is so simple it is unlikely to have any practical relevance.

### Questions
For all the empirical experiments, why not provide results for BPTT to provide a baseline to compare to?

For the three real-world data-sets why not perform experiments for a range of different combinations of m and N?

Can you quantify how much TBPTT reduces computational costs? e.g. What is the training time for each method that you compare?

Given that different hyper-parameters are required to obtain good performance with TBPTT on different tasks, how could hyper-parameter tuning be performed efficiently? Is the time taken to perform hyper-parameter tuning smaller than the time it would take to perform BPTT?

Hyper-parameter tuning is a particular concern as the optimal hyper-parameters found for the training data do not seem to produce the best results on the testing data. It would therefore be particularly informative to compare the test-data performance of BPTT with that of TBPTT using the hyper-parameters optimised on the training data.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work studies the Truncated Back-propagation Through Time (TBPTT) algorithm for training recurrent neural networks (RNNs) on time series task. They analyze the theoretical bounds on accuracy and performance loss when optimizing over subsequences compared to the full data sequence. Their bounds reveal that the burn-in period m serves as an important hyper-parameter in guaranteeing better performance. RNN hidden states are 0 initialized and up to time m, the network does not generate any output due to negatively affected internal transients. This paper provides theoretical bounds (on regret/performance loss) showing how the burn-in period m influences performance, and shows empirical evidence across synthetic and real-world time-series/system-identification datasets that tuning the burn-in phase improves results.

### Strengths
- This work analyzes TBPTT with zero initialized hidden states from the lens of regret w.r.t. training benchmark and shows that the burn-in period (m) affects the performance guarantees, both training regret and performance regret.
- Experimental evaluations on the synthetic data survey as a good validation of the proposed bounds. It shows that larger values of the burn-in phase (m) allow for longer transient phases, and enables RNN to exploit the exploration to reach closer to the benchmark output.

### Weaknesses
- Although the theoretical bounds are valid in the synthetic data setup, the theoretical analysis starts to dwindle down as we reach towards harder real-world setup
- It is unclear how the assumption 1 holds true in realistic settings with more complex choice of RNNs (such as LSTMs, GRUs, etc.)
- Although theoretically the burn-in period helps in simplifying and analyzing the regret, it is unclear if the burn-in period is really the bottleneck in more sophisticated TBPTT (such as non-zero hidden state initialization, hidden state propagation between sequences, normalization techniques, etc.)

### Questions
- Other than the contractive dynamics, are there well known dynamics that would satisfy the assumption 1?
- While the impact of the burn-in period is relatively simple to observe in synthetic data setup, but it comes harder as we evaluate on harder and more challenging real-world time series forecasting problem. Does this imply that the analysis breaks down under this setup?
- In the case of real-world time series and sysid problems, the experimental setup uses LSTMs as the RNN choice, is there any clue or intuition as to whether the assumption 1 is valid in this setup ? 
- Since typically in TBPTT, the hidden state is carried over from the previous segment, how far off are the empirical numbers compared to this setup? 
- Have you evaluated the effect of N and S in the Tab.1 ? 
- Although, this paper analyzes the regret w.r.t. TBPTT performance with the optimal hidden state initialization and the correct dynamics, how would the analysis change if we were to compare V* with a V_b which measures the performance of the BPTT algorithm (i.e., to predict the future, one can look at the entire past)? 
- Would it be much harder to transfer the theoretical analysis to other state space models?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper propose a method called burn in phase in training RNN using truncated backpropagation through time (TBPTT) that can improve the performance of existing TBPTT method. The back propagation through time training method for RNN is suffered from the recurrent  unrolling. The TBPTT tries to improve the training speed as a cost of accuracy. It divides long sequences into shorter subsequence and increase the number of mini batch which can be parallelized. However, this method will lower the accuracy as each subsequence has a different initial hidden state compare the original sequence. The author propose a method that can reduce this error. The rationale is that for RNN with decaying memory, it will gradually forget the initial condition, thus by using a loss that discard the first m time indices. The performance can be improved. The author provides theoretical proofs for their proposed method. As well as experimental verifications.

### Strengths
The proposed method is easy yet effective. The author provides theoretical proofs for their proposed method. 

Theorem 1.1 An error bound is developed between the model with different initial hidden states, shows that the difference converges if the model has decaying memory (lambda  small)
Theorem 1.2 A bound for training regret is developed. This is the difference between ideal case approximation error where an ideal initial state is known and the approximation of using subsequence with burn in.


Theorem 2.1 Shows 1 on full sequence 
Theorem 2.2 Shows 2 on full sequence

### Weaknesses
Some of the concept and terms does not have enough motivation, and the intuition and explanation is not very clear for the Theorems, which can cause non expert hard to follow the results.

There is no recall of notations in Theorem part. Need frequently refer back to find notations.

Certain part of the experiments can be improved. 

Details please see questions.

### Questions
1. Line 307-308 says that "Comparing the corresponding solutions therefore allows us to investigate and quantify the effects of using zero initialization in TBPTT compared to using the optimal initializations."

Why is the optimal initialization as defined in equation (11) is minimized over {h_t}.  $h_0$ is the initial condition, which is a free variable that can be optimized. But how does $h_1$ to $h_T$ be optimized? does that depend on the inputs and $\theta$. I do not understand the state in line 305-306 which says that the additional constraint forces the underlying hidden state sequences to be a subsequence of the same hidden state sequence.


It appears to me that the most optimal setting is each subsequence $i \in \{1,...S\}$ has its own initialization,so the minimization is taken over $S$ number of initial hidden state for the $S$ different subsequences. 


 
2. It appears to me that Assumption 1 is saying RNN has exponential memory decay. This means that the dependence on initial condition decays exponentially which is characterized by $\lambda$. Is this equivalent to a more simple assumption which assume the largest eigenvalue of $W_{hh}$ is smaller than 1. And does the $\lambda$ has any relation with the largest eigenvalue of $W_{hh}$?


3. The notation in Theorem 1 means that $\theta^\*$ and $\theta^b$ are any parameters from the parameter space. But the superscript $*$ and $b$ are used before to indicate model using zero initial and optimal initial. The notation here need to be clarified.

4. Is "batch-averaged output differences" a standard term? If not, please explain the physical neaning of this quantity.

5. "As this constant does not depend on N , they must become smaller if N is increased" What is "they" refers here, this argument need to be more 
rigorous 

6. The RHS of Theorem 1.2 has a term $\frac{\lambda^m}{N-m}$, this function first decays and then increases on interval $(0,N)$. And the best possible $m$ depends on $\lambda$ and can be explicitly solved. Why line 356 to line 358 only discuss this optimal $m$ vaguely? Is it true that the this optimal $m$ results from the upper bound is not the actually optimal choice of $m$?

7. What is the trade-off discussed in line 380 to line 384. Currently what I can infer from the discussion here is that we want larger $m$, which requires larger $o_{min}$, which results in larger $S$.  This means that larger $m$ results in larger $S$. 
    Firstly, my question is do we always want a larger $m$. 
    Secondly, is the argument here rigorous. Actually when $m$ is large, the upper bound exploded, however, this does not necessarily means that LHS also explode. To be rigorous, I think a lower bound is needed. If not, at least there should be some numerical evidence.


8. Regarding the synthetic experiment:

    a. For the synthetic data experiment. How does it verifies your theories. Is the exponential decay rate follows that $\lambda$ of the unknown system. 
    b. Is it even possible to know the $\lambda$ of the unknown system.
    c. You talked about best choice of $m$ depends on $\lambda$, but from this experiment, can only see larger $m$ the better. What if when $m$ is very large? Does it make the model behave bad as predicated by the upper bound, if not, this means that the bound is too loose.
    d. Is there any experiment to show the trade-off mention in line 380 to 384.

    e. Why do you conduct experiments on a unknown system, which cannot fully verify your results. Is it more reasonable to conduct experiments on a known system which you can control the $\lambda$, to fully support your theoretical results.

### Soundness
3

### Presentation
2

### Contribution
3
