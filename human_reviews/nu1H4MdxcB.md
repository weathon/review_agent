# Sparse spatio temporal reconstruction with Closable Kernel Space

- Avg Score: 2.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 1, 3

## Abstract
Quantifying spatio-temporal (ST) measures of dynamical systems is a crucial problem with wide ranging applications in climate modeling, epidemiology, physical processes to name a few. 
We are interested in the same but motivated by a rather practical scenario where sparse information is collected non-uniformly. 
To reconstruct the underlying dynamical system under such constraints, we propose a novel algorithm for learning the Koopman operator via a Reproducing Kernel Hilbert Space (RKHS) based on the Laplacian Kernel Extended Dynamic Mode Decomposition (Lap-KeDMD).
We further show that our kernel space resolves a fundamental issue that is required for a faithful reconstruction of the Koopman operator of the underlying ST data by proving its closability. 
We demonstrate our method on standard benchmark cases --
Burger's Equation, fluid flow across cylinder and Duffing Oscillator. 
We then reconstruct the Koopman operator for a real ST Seattle traffic flow data that is collected non-uniformly. Necessary comparisons are made between the current state of the art kernel methods corresponding to Gaussian Radial Basis Function (GRBF) Kernel. 
Such empirical comparisons leads us to conclude that Lap-KeDMD remarkably outperforms as compared to that of aforementioned counter-part thereby, making the Laplacian Kernel a robust choice for such ST quantification.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper investigates a kernelized (extended) dynamic mode decomposition (eDMD) for reconstruction of spatiotemporal data under the Koopman operator framework. The authors present _Laplace kernel eDMD_, which uses the Laplace kernel in the kernelized formulation of eDMD. They show that compared to the popular Gaussian RBF kernel, the Koopman operator acting on the RKHS of the hyperbolic sine kernel with linear observables is closable, whereas on the RKHS of the RBF kernel, it is not. This suggests the ability / inability of Laplacian / Gaussian kernel to reconstruct spatiotemporal modes from limited data.

### Strengths
The investigation of closability of Koopman operators over RKHSs and its implications on the corresponding eDMD algorithm using the respective reproducing kernel is an interesting perspective (although I am not sure if I have fully understood it). The main results are furthermore supported by various experiments comparing the kernelized eDMD using the Laplace vs Gaussian kernel, showing that the former leads to better reconstruction results.

### Weaknesses
Overall, I find the writing to be difficult to understand. The logical flow of the paper is hard to follow and I struggle to see how the main theoretical result of the paper tells us  about the proposed algorithm outside from an educated guess (see also questions below). I wish the authors made it clearer how the closabilty of Koopman operators over an RKHS tells us about the ability of the algorithm to reconstruct spatiotemporal data. At present, I find this to be confusing since the kernel in the proposed algorithm is given by the Laplace kernel, but the RKHS used in the main result is _not_ the one corresponding to the Laplace kernel but instead the _hyperbolic sine kernel_. A further explanation of how these two kernels are related would be useful for the understanding of the main result, in addition to why the closability result allows spatiotemporal reconstruction.

In terms of the experiments, insufficient details are provided, with the only explanations available in the captions of the various figures. This makes it difficult for us to understand the purpose and outcomes of the various experiments presented. Overall, more care should be taken in the presentation of the results, highlighting (1) the purpose of each experiment, (2) basic experimental configurations, and (3) what the results tell us about Laplace vs Gaussian kernel. In addition, the labels in the figures are too small to read without zooming the screen. This should be made much bigger.

### Questions
There are some points regarding the main results of the paper where I am unsure and hope the authors could please clarify. I believe these are critical and impact the validity of the theoretical claims made in the paper.
- In the Lap-KeDMD algorithm, the Laplace kernel is used to generate the gram and interaction matrices. However, the main theoretical result (Theorem 3.2) is stated in terms of a different hyperbolic sine kernel (10), which is the reproducing kernel of a Laplace-weighted $L^2$-space $H_{\sigma, 1, \mathbb{C}^n}$. I don't understand the relation between the Laplace kernel and this Laplace-weighted $L^2$-space. In particular, I hope the authors can clarify further why Theorem 3.2, which use the hyperbolic sine kernel, says anything about Algorithm 2, which use the Laplace kernel.
- I am unsure why Theorem 3.2 establishes closability over the RKHS $H_{\sigma, 1, \mathbb{C}^n}$ since they only show the property $  \lim_{n\rightarrow\infty} x_n = 0, Tx_n = y_n \Rightarrow \lim_{n\rightarrow\infty} y_n = 0$ over _a single sequence_ $x_n := \|z_n\|_2K^\sigma(z_n, -z_n)$. Whereas Lemma B.1 states that this should be shown for all such sequences $x_n$ to establish closability.
- I am not convinced with the proof of Theorem 3.3, since the authors only show that _an upper bound_ to $\|z_N\|_2 \exp(-4\|z\|_2^2/\sigma)$ is unbounded. This says nothing about $\|z_N\|_2 \exp(-4\|z\|_2^2/\sigma)$ being unbounded.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
This is a nice attempt by AI to submit a fake technical paper to one of the top ML/AI conferences. It successfully fooled me into reading the first few pages with quite some interest until I found it makes no sense coming to the techniques. Then I downloaded the supplement, only to find a 118.8MB python notebook (no bother to open) and an appendix full of nonsense.

### Strengths
It contains an interesting topic. Submitted is a paper that appears technically sound.

### Weaknesses
* Most part does not make sense. Even the term "closable" is not explained. The whole paper is full of grammatical errors. It surely does not explain how the "fundamental issue of ST reconstruction on irregular grids" is solved by Koopman operator.

* Waste of time for reviewers.

### Questions
Too many to ask.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors propose to use the Laplace kernel for approximating Koopman operators on RKHSs. They insist that with the Laplace kernel, the Koopman operators are closable.

### Strengths
Analyzing Koopman operators theoretically is an important and interesting topic.

### Weaknesses
I mainly have following concerns:  
**Readability**  
This paper is hard to follow since the notation is not consistent and there are undefined notions. For example,
- In Definition 2.2, what is $\mathbb{X}$?
- In Definition 2.3, what is $n$? the authors say $n$ is the time, but $n$ is also the dimension of the state space.
- In line 144, what is the definition of $\xi_k$? I couldn't figure out why $x$ is assumed to be represented using the Koopman eigenvectors and $F(x)$ is in such a form.
- In line 149, what is $\mathcal{F}_{N_k}$?
- In line 160, what is the relationship between $\phi$ and $\tilde\{\phi\}$?
- In Proposition 2.2, what is (-)p? Does this mean psudo inverse?

**Validity of the proposed method**  
The authors propose to use the Laplace kernel for KDMD (Algorithm 2). However, in Section 3.1, they define a Hilbert space equipped with the inner product (9). I could not figure out the relationship between the RKHS associated with the Laplace kernel and the Hilbert space defined in Section 3.1. 

**Correctness of the result**  
In Theorem 3.3, $\mathcal{K}\_{\phi}$ is assumed to be bounded on the whole Hilbert space, but the authors conclude $\mathcal{K}_{\phi}$ is not closable. I could not figure out why a bounded operator defined on the Hilbert space is not closable. In addition, in my understanding, Koopman operators on an RKHS is closed. Thus, I'm concerned with the correctness of the result. Indeed, in the proof of Theorem 3.3, I could not figure out the relationship between constructing a sequence converging to $0$ and the Koopman operator.

### Questions
Could you provide the detailed proof of Theorem 3.3?

### Soundness
1

### Presentation
2

### Contribution
2
