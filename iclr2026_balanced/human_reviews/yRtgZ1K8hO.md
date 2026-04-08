## Human Reviewer 1

### Summary
This paper proposes a new algorithm for computing the matrix polar decomposition for the Muon optimizer used in training large neural networks. Unlike classical methods designed for high numerical accuracy, Polar Express is optimized for GPU efficiency and relatively low-precision computation (bfloat16). The authors derive the algorithm by solving a minimax optimization problem that guarantees optimal convergence in the worst case and demonstrate substantial empirical improvements over existing methods.

### Strengths
1. This paper introduces a theoretically grounded algorithm.
2. The method is practical, using GPU-efficient GEMMs and is numerically stable.
3. Experiments on GPT-2 with the Muon optimizer show consistent improvements in validation loss over existing methods.

### Weaknesses
1. It would be interesting to see how well the method works on models other than language models. 
2. Lack of detailed runtime or throughput benchmarks. The experiments show the method works better than baselines under the same number of iterations, but it is unclear whether it is faster.

### Questions
see weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper proposes a better method for computing polar factor decomposition, practical for GPU implementation. Optimality under certain conditions is proven, and improved empirical performance for implementation into the Muon algorithm for synthetic problems and GPT model training.

### Strengths
An important and timely problem is considered with strong potential for practical impact.

A novel method is proposed and analyzed mathematically.

Reasonable empirical experiments illustrate and support the claims.

### Weaknesses
The set of practical ML problems can be diversified a bit, e.g., by considering a vision domain.

### Questions
Does the approximation quality depend on the dimension of the problem?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper seeks to improve the Muon optimizer which is being used in several recent results as an alternative to Adam/AdamW for training small-sized language models. In Muon, one subroutine is calculating the polar decomposition. This is the main task handled in this work. The standard solution involves running Newton-Schulz which works well but convergence can be slow. More recent heuristics to speed it up can work but may not converge to the correct solution. Polar Express proposes to fix by adapting the polynomial at each iteration (a minimax problem is solved). This gives good initial speed of the heuristics and guarantees on convergence. On the practical side, GPT-2 models are trained to a better validation loss. The paper is well written. The technical insights are very nice and the empirical results are convincing.

### Strengths
1.  The paper replaces recent heuristic strategies which can be fast but plateau out with a provably optimal algorithm. The initial progress is rapid and has similar convergence guarantees like classical methods. I found this result very interesting and new. 

2. I was familiar with the Nakatsukasa/Freund paper, but the way in which the minimax paradigm is adapted here is surprising in a good way, giving Thm 3.1 and 3.3. I feel that the idea is quite novel and can be useful beyond the Muon use case. 

3. The experimental results back up the main claims. Fig. 3 is sufficient validation that the idea in this paper deserves consideration. 

4. What is also nice is that this is not just a algorithmic results paper. The authors discuss numerical instability by adjusting the polynomials, showing the adjustments needed for practical gains on modern GPUs. Same with precomputing coefficients etc. Excellent balance of technical findings with practicality.

### Weaknesses
1. Relatively minor. It will be good to show how much slower is Zolo-PD using newer QR implementations. 
2. Also minor. Experiments are only on GPT-2 with Muon. If Muon is effective, why not expand the scope of experiments to other architectures and even finetuning?

### Questions
1. You may consider slightly expanding the scope of experiments. The paper is still a valuable contribution, but even if the idea is not equally effective in other cases, it will be good to point out where the gains are limited.

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
10

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes a new method for orthogonalizing a matrix, with applications in the Muon optimizer. The method, PolarExpress, greedily computes polynomial approximations of the sign function in a residual manner using the Remez algorithm, and then applies these approximations during training to approximately orthogonalize the momentum buffer in Muon. The authors prove results about the convergence of PolarExpress and its optimality with respect to the L_\infty norm to the sign function, and empirically show that PolarExpress improves upon existing orthogonalization methods used in current Muon implementations.

### Strengths
I will preface this review by saying that I am not an expert in numerical methods. However, the presented algorithm seems novel and interesting, and the there is theory showing that PolarExpress is guaranteed to converge at least as fast as the canonical "NewtonShultz5"  method. This paper also has some other nice components, including
- Theoretical justification for using a greedy residual polynomial construction
- Empirical results on toy matrices showing that the method converges faster than existing baselines (Figure 3)
- Results showing the effect of varying the lower bound guess $\ell$ of the smallest singular value in the gradient matrix
- Empirical GPT training experiments showing that PolarExpress achieves better initial convergence over prior Muon implementations.

### Weaknesses
- There seems to be a missing "key" baseline, which is running Muon with the actual `polar' function and computing the SVD online. Although this is obviously impractical, it gives a sense of how far PolarExpress and these other methods are from "true" Muon. Can you run an experiment showing how much benefit there is from getting the optimal solution in real-world GPT training?
- Likewise, does minimizing the L_\infty norm of the sign function maximize the directional similarity to the actual "true" Muon update? Can you generate a plot of the cosine distance of PolarExpress's update (and other baselines) to the SVD update to get a sense of how close these approximations are?
- The empirical experiments only go out to 1B tokens. 125M parameters @ 1B tokens is way below Chinchilla optimal, and my own experiments in the past have suggested that Muon's initial performance gap diminishes in the "overtrained regime." If you have time, can you run an experiment past Chinchilla (125M parameters @ 10B tokens should be easily doable in a day on the 4 H100s used in the paper) to see how much of a gap there is?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3