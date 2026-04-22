# Accelerating Eigenvalue Dataset Generation via Chebyshev Subspace Filter

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
Eigenvalue problems are among the most important topics in many scientific disciplines. With the recent surge and development of machine learning, neural eigenvalue methods have attracted significant attention as a forward pass of inference requires only a tiny fraction of the computation time compared to traditional solvers. 
However, a key limitation is the requirement for large amounts of labeled data in training, including operators and their eigenvalues.
To tackle this limitation, we propose a novel method, named **S**orting **C**hebyshev **S**ubspace **F**ilter (**SCSF**), which significantly accelerates eigenvalue data generation by leveraging similarities between operators---a factor overlooked by existing methods. 
Specifically, SCSF employs truncated fast Fourier transform (FFT) sorting to group operators with similar eigenvalue distributions and constructs a Chebyshev subspace filter that leverages eigenpairs from previously solved problems to assist in solving subsequent ones, reducing redundant computations.
To the best of our knowledge, SCSF is the first method to accelerate eigenvalue data generation. 
Experimental results show that SCSF achieves up to a $3.5\times$ speedup compared to various numerical solvers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors present SCSF, a sorting Chebyshev subspace filter, which reduces the time to solve eigenvalue problems numerically. The authors present an approach which uses a truncated FFT in to identify low order modes of operators, and sort them by similarity. The problems are solved sequentially, where the solution of the next problem is sped up by incorporating a bias of the eigenvectors and eigenvalues of the previous solution. The authors present a set of experiments on several families of PDEs as well as with a sufficient comparison with alternative solvers.

### Strengths
- The paper was very well written and easy to follow. Although I do not actively do research in this field, I felt I was able to understand the problems, methods, and contributions quite well. 
- The presented approach has strong and consistent gains in time to solution. 
- The authors present many experiments which demonstrate the effectiveness of their approach in many scenarios. As I read, there were 3 occurrences where I read a claim and then made a note to ask for an ablation, only to find the authors had already performed such an experiment and included results either in the main text, or in the appendix.

### Weaknesses
- The scope of the paper is relatively limited. It mainly benefits Hermitian linear operators. 
- There is an assumption of dataset similarity built in to this approach. I wonder what failure modes might arise in situations where this does not hold true. Perhaps it would be an interesting study to gradually mix two different operator distributions and observe how the approach compares to a baseline, in the average and worst-case situations. Some theoretical understanding could also be beneficial on this point as an alternative to an empirical study.
- The work shares many similarities to other approaches (Chase, ChFSI), which hinder the absolute novelty of their approach. This also makes statements such as "the first method to accelerate the operator eigenvalue data generation" a bit too broad. However, the authors have made sure to provide experiments which demonstrate their empirical advantages over these approaches. Perhaps a short statement which makes the differences in SCSF explicit would be beneficial. 
- I always recommend authors to add a short statement on the limitations of the approach. 

- (very minor typo) line 122 on page 3, should state "exhibit a high similarity" instead of "exhibit a highly similarity".

### Questions
- What modifications are necessary to apply this approach to operators which it is not intended for? Or how might it perform on more general operators?
- Could the first highlighted contribution bullet in the introduction be modified to be more specific? 
- How sensitive is the approach to the choice of sorting metric (the truncated FFT distance)? Would other similarity metrics—e.g., based on operator spectra or principal angles—change performance?
- How does the method behave when the dataset contains operators from distinct distributions (i.e., heterogeneous or discontinuous parameter spaces)? The appendix touches on this, but more quantitative analysis might help clarify failure modes.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Training neural operators for eigenvalue problems requires a lot of data and the main bottleneck for generating data for
eigenvalue problems is doing that linear operation. In this paper, the authors propose SCSF which identifies and solves sequentially problems that have close spectral distributions.

### Strengths
* The paper is well-written and well motivated. It is indeed interesting to see work trying to improve eigenvalue
  computations as they are indeed a core component in several important applications.
* The objective of the paper seems unique and novel.
* The method explanation is thoroughly developed with great attention to detail.
* The baseline comparisons are quite strong.

### Weaknesses
* No code is shared through a anonymized link? Actually the sorting and Chebyshev Filtering does not seem trivial to implement. How are the authors planning to promote adoption and reproducibility?

### Questions
* Are there other types of filtering that could be used? Is there any particular reason as to why Chebyshev polynomials
  are a good choice overall? Maybe something related to the expected decay rate of the linear operators in certain
  problems? I could've missed this discussion in the paper.
* I understand that the goal of the paper is to get rid of a critical bottleneck in generating large-scale eigenvalue
  datasets but there is no training of neural operators. One fear is that indeed SCSF might reduce runtime in the
  data preparation but that the eigenvalue estimates are worse hence worsening the performance of neural operators
  trained on this data. Could you please comment on this concern?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper suggests the use of Chebyshev polynomials to as a filter to accelerate the creation of eigenvalue data sets for training neural nets. The suggested SCSF algorithm uses FFT to transform operators into a representation that allows efficient sorting of operators in terms of their distance. The runs subspace iteration using a Chebyshev filter to accelerate the convergence os the eigen-problem, by initializing from the previously solved-for closest operator. 

Results are presented for several different eigen problems of interest. SCSF is compared with old and fairly basic eigenvalue solvers. The comparison include one baseline that uses Chebyshev subspace iteration as a filter.

### Strengths
Results include a set different experiments in terms of capturing both accuracy, running time, and efficacy of the sorting process.
The stated problem is of interest to the community of numerical analysts.

### Weaknesses
1.	The paper doesn’t cover or compare with the two works of Wang et al. 2024,  and Dong et al 2024. For example, Wang et al. 2024 is using the same similarity search, but uses a Krylov iteration instead Chebyshev filtering. Why is that not compared to in the text and in experiments? What are the advantages of using the Chebyshev polynomials of Wang et al’s Krylov iteration.  I would expect experimental benchmarking to go beyond standard packages of eigensolvers. 

2.	There should be a dedicated related work section to compare and contrast this work and prior relevant art.

3.	I fill there is disconnect between the process described in f and g in fig 2, and what is explained in section 3.2

4.	The paper is missing motivation as to why is it useful to use Chebyshev polynomials for this problem.

5.	There is no explanation as how A is derived from P, is it A=P^T*P? is is A= (P+P^T)/2?

6.	It’s not clear if FFT is run on P or on A? in Fig 2 it is applied on A, in algorithm 2 it is applied on P.

7.	The authors claim in their statement of contribution in the introduction that ‘To the best of our knowledge SCSF is the first method to accelerate the operator eigenvalue problem”, however Wang et al 2024 and Dong et al 2024 have shown acceleration before for a similar type of problems. 

8.	Same problem with the statement about the sorting algorithm novelty – it was already state in section E.2.2 of Wang et al.

### Questions
please see the above

### Soundness
3

### Presentation
1

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
The paper tackles the challenge of accelerating eigenvalue dataset generation for parametric operator learning frameworks (e.g., DeepONet). These frameworks typically require labeled data consisting of operator-eigenpair pairs, which are expensive to generate using traditional solvers. The authors propose a method, **Sorting Chebyshev Subspace Filter (SCSF)**, to reduce this computational cost by exploiting structural similarities among related operators.

The core idea is that by transforming a collection of eigenvalue problems into a correlated sequence, one can recycle subspace information and reduce redundant computation.
Their approach combines two main components:  
- **Parameter sorting via a truncated FFT**, which orders problems such that consecutive operators are spectrally similar.  
- **Chebyshev filtered subspace iteration**, which reuses eigenpair information from previously solved problems to accelerate convergence for subsequent ones.

They evaluate the proposed method with several problem settings to demonstrate the efficacy.

### Strengths
- The paper addresses a practical bottleneck in the data generation pipeline for neural operator training, which is both relevant and timely.  
- The paper proposes a systematic integration of classical numerical linear algebra tools (Chebyshev filtering) with a new idea (FFT-based ordering) in a modern ML context.  
- The experiments demonstrate consistent empirical speedups across diverse PDE operators.
- The paper is generally well written and structured.

### Weaknesses
- In terms of novelty, the proposed FFT-based parameter sorting seems to be the only new component on top of the Chebyshev filter.
- The paper does not analyze any theoretical properties of the proposed sorting algorithm. 
- While the writing is generally clear, the introduction of the Chebyshev filter in Section 2.2 looks unmotivated.

### Questions
1. How sensitive is the proposed parameter sorting to the different parameterization of the PDEs?
2. How sensitive is the sorting algorithm to the choice of truncation level \(p_0\)? Does it generalize across different PDE families? (I found a discussion in Appendix, but I think it's better to mention this explicitly in the main text. Is $p_0=20$ universally acceptable constant over all possible PDE families? Should it scale for a harder problem?)
3. Can the authors have any theoretical insights or potentially feasible direction for an analysis of the FFT-based ordering? 
4. In Table 5, why are the numbers in the column Greedy and column Ours exactly the same?

#### **Suggestions**
- Consider adding a paragraph in Section 2, before Section 2.1, to overview what will come in the section. This would resolve the logical gap in the current version. 
- Table captions are instructed to come above tables, not below.

---

Overall, I believe that the paper addresses a timely and practical problem of community's interest. While the proposed idea is not entirely new, the combination of FFT-based sorting and the classical Chebyshev filter seems to provide a nice, practical method in the modern ML context. This can be a useful technique for neural-net-based, scalable numerical analysis.

### Soundness
3

### Presentation
2

### Contribution
3
