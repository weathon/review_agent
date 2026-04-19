# Training Overparametrized Neural Networks in Sublinear Time

- Decision: Reject
- Scores: 3, 5, 5, 6

## Abstract
The success of deep learning comes at a tremendous computational and energy cost, and the scalability of training massively overparametrized neural networks is becoming a real  barrier to the progress of artificial intelligence (AI). Despite the popularity and low cost-per-iteration of traditional backpropagation via gradient decent, stochastic gradient descent (SGD) has prohibitive convergence rate in non-convex settings, both in theory and practice.  

To mitigate this cost, recent works have  proposed to employ alternative (Newton-type) training methods with much faster convergence rate, albeit with higher cost-per-iteration. 
For a typical neural network with $m=\mathrm{poly}(n)$ parameters and input batch of  $n$  datapoints in $\mathbb{R}^d$, the previous work of \cite{bpsw21} requires $\sim mnd + n^3$ time per iteration. In this paper, we present a novel training method that requires only $m^{1-\alpha} n d + n^3$ amortized time in the same overparametrized regime, where $\alpha \in (0.01,1)$ is some fixed constant. This method relies on a new and alternative view of neural networks, as a set of binary search trees, where each iteration corresponds to modifying a small subset of the nodes in the tree. We believe this view would have further applications in the design and analysis of deep neural networks (DNNs). We conclude a discussion of lower bound for the dynamic sensitive weight searching data structure we make use of, showing that under {\sf SETH} or {\sf OVC} from computational complexity, one cannot substantially improve our algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a second-order optimization algorithm based on the Gauss-Newton method to efficiently train an overparameterized 2-layer shifted ReLU network with a fixed second layer, using squared loss in the NTK regime. The challenge with second-order methods is their high cost-per-iteration (CPI), despite requiring fewer iterations for convergence compared to first-order methods. The authors aim to reduce CPI while maintaining fast convergence. Under the overparameterized setting, where the width $m$ is polynomial in the number of training points $n$ ($m = \mathrm{poly}(n)$) with input dimension $d$, the proposed method reduces CPI to $O(m^{1-\alpha}nd + n^3)$ for some constant $\alpha \in (0.01, 1)$, improving upon existing second-order methods with a CPI of $O(mnd + n^3)$. The method also achieves $O(\log(1/\epsilon))$ iterations to reach a training error below $\epsilon$, matching the convergence rate of existing second-order methods.

The key idea behind the algorithm is using a tree-based data structure to efficiently identify activated neurons, leveraging the sparsity of shifted ReLU activations. Combined with sketching techniques for solving the online regression problem from the Gauss-Newton method, this leads to a sublinear CPI.

### Strengths
The proposed algorithm achieves a CPI that is sublinear in the network width, which is an improvement over existing second-order methods in overparameterized settings. It is great that the paper provides formal proof of both the sublinear CPI and the ultra-fast convergence rate typical of second-order methods.

### Weaknesses
This paper has several critical weaknesses, including issues with the novelty of the tree-based data structure, poor writing, and a lack of practical implications.

1. **The proposed dynamic tree data structure lacks novelty and proper attribution to prior work.**
-  The dynamic tree data structure is not novel and lacks proper citation of prior work, such as [Alman et al., 2023]. The correlation tree proposed in [Alman et al., 2023] (Algorithms 1 and 2 therein) is nearly identical to the threshold search data structure used in this paper (Algorithms 2, 3, and 4). Notations and text are similar, but the authors do not properly cite this relevant work while claiming novelty.

2. **The writing is unpolished and the presentation of key concepts is imprecise.**
- The paper does not sufficiently describe the model setup or key notations in the main text, making it hard for readers to follow. For example, fundamental elements like the loss function and variables such as $J_t$, $f_t$, and $y$ (which appear on Line 247) are only defined in the appendix. Such definitions should be in the main text or referenced before their use to improve clarity. 
- In Theorem 5.1, important parameters such as $\lambda$ and $R$ appear without definition in the main text. While $\lambda$ is defined in the appendix (Remark A.5), it should be introduced earlier. The definition of $R$ is unclear and absent.
- The assumptions regarding network width $m = \Omega(\max(\lambda^{-4}n^4, \lambda^{-2}n^2d\log(16n/\rho)))$ should be better explained. Since $\lambda$ (minimal eigenvector of kernel matrix) is instance-dependent, its relationship to the training data and parameters (e.g., network width and number of training points) should be discussed. Additionally, $\lambda$ is not always guaranteed to be strictly positive.

3. **The assumptions made by the paper are impractical.**
- The paper relies on using "shifted" ReLU activation $\phi(x) = \max (x, b)$ with shifted parameter $b\ge 0$. The shifted parameter should be appropriately chosen (typically, $b=\sqrt{0.48 \log m}$) to ensure sparsity in the activations, and this plays a key role in the analysis. This raises concerns about the generalizability of the results to practical setups, including standard ReLU networks.
- The assumption that the second layer of the network is fixed during training is unrealistic, as in practical settings both layers are typically trained jointly. Prior works, such as [Du et al., 2019], have analyzed overparameterized networks where both layers are trained together, and it would be important to consider whether this paper's analysis can be extended to such cases.

4. **The lack of numerical experiments weakens the practical implications.**
- While I understand that experiments are not strictly necessary for a theoretical paper, empirical results comparing the total training time and CPI of the proposed method with other second-order methods and SGD would strengthen the paper’s practical relevance.

5. Minor comments:
- Typo in Line 87:  Our main reuilt $\to$ Our main result
- Typo in Line 101: Mismatched parenthesis
- Typo in Line 133: Mismatched parenthesis
- Typo in Line 247: $g_t:=\arg\min_g\lVert J_tJ_t^\top g_t - (f_t-y)\rVert$ $\to$ $g_t:=\arg\min_g\lVert J_tJ_t^\top g - (f_t-y)\rVert$

---

**References**

[Alman et al., 2023] Bypass Exponential Time Preprocessing: Fast Neural Network Training via Weight-Data Correlation Preprocessing, NeurIPS 2023.

[Du et al., 2019] Gradient Descent Provably Optimizes Over-parameterized Neural Networks, ICLR 2019

### Questions
1. In Line 281 and Line 285, the online regression problem is referencing Eq. (4), $\min_x \lVert Q^\top Q x - y \rVert_2$. Is this correct? I believe the correct reference should be $g_t:=\arg\min_g\lVert J_tJ_t^\top g - (f_t-y)\rVert$.
2. In Definition 5.5, what does $R_0 \approx n/\lambda$ precisely mean?

### Soundness
2

### Presentation
1

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
This paper provides a novel algorithm for training overparametrized neural networks. It is proven that for 2-layers ReLU nets, this algorithm can be run in sublinear (in the network's width) per-iteration cost, and it preserves the same convergence rate as previous works.

### Strengths
- The problem of reducing the computational cost of running big machine learning models is an important and relevant line of research.
- The paper contains a rigorous analysis of the algorithm presented.
- The connection between neural nets training and binary search trees might be relevant for future work.

### Weaknesses
- The analysis seems to be limited to shifted-ReLU activation, relying heavily on the sparsity of the Jacobian. To the best of my understanding, the analysis does not extend to other activations. 
- I am unsure about the relevance of this result: can an approximation of the NTK matrix together with kernel ridge regression achieve sublinear computation time? The authors should comment on this.

### Questions
- Can the authors comment on how an algorithm that approximates the NTK matrix and trains only the second layer would compare to their algorithm?
- Can the authors comment on the implicit bias of their algorithm, compared to previous state-of-the art algorithms?
- Does the analysis extend to other activations or deeper networks?

Minor things and typos:
- Line 129: missing `which/that' after `data structure'.
- Line 133: missing closing bracket `)'.
- The informal Theorem 1.4 is hard to understand, if SETH and OVC are not defined.
- Theorem 5.1: it would be good to define lambda.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper focuses on an alternative method for training shallow neural relu networks. The paper provides an analysis for the computational complexity of the proposed algorithm which employs efficient techniques to reduce the complexity per iteration.

### Strengths
Please see the "Questions" section.

### Weaknesses
Please see the "Questions" section.

### Questions
My review is as follows:

- It's a bit confusing the Algorithm 1 has two lines where a sketch matrix S is constructed.
- The wording in Algorithm 1 "Maintain the weight implicitly" could perhaps be made clearer?
- I have realized that Algorithm 5 has the full details of Algorithm 1. Still the point stands that the wording of Algorithm 1 is ambiguous.
- Why the references on Locality Sensitive Hashing?
- In my opinion, some simple numerical results that verify the lower cost-per-iteration claim would improve the paper.
- It is not clear why it's "almost" impossible to improve the algorithm, just by considering the way Theorem 1.4 and Theorem 7.3 are written. "This result shows that it is almost impossible to truly improve our algorithm"
- I think the organization of the paper could be improved. For instance, sketching seems important to the approach, but it's mentioned for the first time in page 6.

Minor:
- phi is not defined in line 86 when it first shows up
- line 114, typo payed --> paid?
- typo line 129, "that" supports
- line 152, typo: "next a few"

There are a few other typos.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed treating the training of a NN as a binary tree search in sublinear time by taking advantage of the sparsity in the Jacobian. The paper's analysis focuses on two-layer ReLU networks.

### Strengths
The paper makes a theoretical good case for using the implicit sparsity of the Jacobian for overparameterized NNs. The method proposes only updating active neurons, therefor reducing computation/gradient calcs.

The paper proposed a binary search tree structure with sublinear time complexity per iteration enables fast updates and queries on non-zero Jacobian entries.

Paper provides theoretical basis, although I did not check everything in the appendix. 

While the method requires the Gauss-Newton method for its sparse Jacobian updates, the paper proposed using conjugate gradient methods to do the computations faster. They also rely on matrix sketching which makes is fair.

### Weaknesses
This paper proposes a very new way (at least to me) to train a NN. It seems to have some nice theoretical benefits but since it's so new there might be a lot of unforeseen issues with the method. Overall, I like the paper but have some concerns on the future empirical side of it. 

One that comes to mind is the complexity (not theoretical complexity -- the authors take care of that in the paper). Adam for example has learning rate learning rate and (beta1,beta2). The proposed method would require conjugate gradient which can be expensive and comes with hyper-params of its own, it has sketching which again has more hyper-params like sketch size, ​sketch tolerance and sketch ​failure probability, also things like tree update frequency need to be taken into account. 

Furthermore, while it would be great to take advantage of sparsity, it is not exactly clear to me how sparse the Jacobian would be, and current hardware isn't really setup to 1 calculate sparse gradients, 2 update weights in a much cheaper way due to the sparsity. That is, harvesting the benefit of sparsity is more difficult than one might think in practice.   

Finally, there are no experiments, it would be really nice to see even some toy experiment to see how this method works in practice.

### Questions
Can the authors speak to the comments in the weaknesses section above?

If at least a toy experiment can be added I think it would really help the case of the paper, at least in my eyes.

### Soundness
2

### Presentation
3

### Contribution
3
