## Human Reviewer 1

### Summary
This paper proposes a method for improving the performance of distributed optimization mitigating the communication bottleneck. The key idea is to use Markovian compressors in the vanilla Quantized Stochastic Gradient Descent (QSGD) algorithm. In addition, the momentum acceleration method is also used. The proof of convergence is provided and the usefulness of the proposed method is confirmed by experiments.

### Strengths
The algorithm is simple and looks easy to implement. Convergence of the algorithm is guaranteed by proofs under several assumptions. Practical usefulness is shown experimentally.

### Weaknesses
The key idea of "random sparsification" has already been proposed in the vanilla QSGD in an earlier paper. The contribution of this paper is to introduce Markovian randomness to it. The significance of the contribution is smaller than the prior work although the guarantee of its convergence may be non-trivial to prove.

### Questions
There are various hyperparameters. Although some details on their tuning are provided in Appendix H, please describe the guiding principles in more detail.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper proposed two sparsification compressors for distributed optimization problems, where the communications between different devices are first compressed before transmission. Almost all previous compressors do not consider what have been transmitted during the previous iterations. While the paper proposed compressors that consider previous iterations in a Markovian fashion. The paper further proves convergence bounds for the proposed compressors. Some experimental results and numerical examples are presented to show that the proposed compressors result in performance improvements compared with previous approaches.

### Strengths
The paper present convergence bounds for the proposed compressors and distributed optimization approaches. There bounds can be valuable for analyzing similar optimization approaches.

### Weaknesses
* The presentation of the proof part need to be improved. In its current form, the proof is difficult to follow. It is suggested that for each inequality, the author may provide a clue that which fact(s) has been used to get the inequality. It may also be helpful that the author provide more proof sketches. 
* It seems that the convergence bounds hold for any unbiased compressors or asymptotic unbiased compressors. Then, how the author justify that BanLast and KAWASAKI are near-optimal among all Markovian type compressors.
* For the experiment part, in my opinion, more experimental results are needed for justifying that the proposed compressors truly outperform the previous random compressors. It is recommend that more experiments with different types of neural networks and more numbers of devices. The distributed optimization approaches are interesting, typically when people want to train large models on massive datasets. So it would be interesting to see that the proposed compressors outperform other approaches on larger training jobs.

### Questions
* In line 850, the notation is quite confusing for me. The left-hand side is a matrix and the right hand side sounds like a norm for me. Are there any typos here?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper consider a generic distributed optimisation problem, and proposes a solution to the information bottleneck due to the necessary communication between computing nodes working in parallel. The solution is based on two stochastic compression operators that go beyond simple sparsifiers. Theoretical guarantees are provided on convergences rates in terms of distance to the minimum of the objective. The effect of momentum is also analysed. Numerical tests are then presented on distributed logistic regression and neural networks training.

### Strengths
Let me start by stating that I am not *at all* an expert in optimisation, even less in distributed setting. I will therefore not judge the novelty of the paper as I have no clue. 
Given this, I have found the paper interesting. The proposed idea is very natural and quite simple. The theoretical guarantees are strong and provide an clear view on convergence properties (I think).
The numerics are convincing.

### Weaknesses
I was disappointed by the quality of presentation. E.g., the text is sometimes quite cryptic and there are many linking words missing. Some definitions are missing too, and I had to guess a number of times what the authors meant. I recommend making an effort in presentation.
In spite I have found the problem in interesting in itself, it is not clear for me that ICLR is the right place to publish this paper. But the AC and authors probably know more about this than me, this is not a strong view.

### Questions
+ strange referencing in the text: I don't know if this is the standard ICLR referencing, but a lot of text space is lost due to the refs everywhere, that cut also the reading flow.
+ def 1: L-smooth -> L_i-smooth
+ Randm never defined
+ F_tau not defined in Theorem 2. Is this just a constant? That depends on what? 
+ what is "variance reduction" discussed in a whole paragraph?
+ what is the « information sent » axis in fig 1?
+ many linking words missing

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
2

---

## Human Reviewer 4

### Summary
This paper proposed a possible improvement of the random sparsification method in the context of distributed optimization, which is in turn a compression technique aimed at reducing the communication cost. Briefly speaking, random sparsification compresses the data by selecting a subset of coordinates uniformly at random and keeping only these selected coordinates of data. The key idea of the paper is that, instead of using i.i.d. selection of subset, it may help to select the subset with a Markov chain. In this way, more diversity is encouraged by setting a smaller probability to select those coordinates which was just selected in the previous few iterations.

### Strengths
I consider Markovian compressors to be a novel idea, but it is possible that I have missed some related literature. 

The proposed algorithm is easy to implement and fairly understandable, which might broaden its interest to general readers. 

The math looks correct to me, though I have not checked every detail.

### Weaknesses
My main concern is that the theoretical analysis does not seem to touch the real advantage of using a Markovian compressor. It appears that the paper resorted to stationarity analysis, but the stationary distribution of the Markov chain simply falls back to the standard random sparsification. It is thus unclear what is gained by a Markovian compressor. Although a better analysis could be challenging, as argued in the paper, the significance of the proposed algorithm would be greatly strengthened if more theoretical insights were provided.

### Questions
Please refer to the Weaknesses part.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3