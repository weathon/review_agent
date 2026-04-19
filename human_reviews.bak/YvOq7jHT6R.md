# Fight Fire with Fire: Multi-biased Interactions in Hard-Thresholding

- Decision: Reject
- Scores: 3, 3, 3, 6

## Abstract
$\ell_0$ constrained optimization is widely used in machine learning, especially for high-dimensional problems, as it effectively promotes sparse learning. A prominent technique for solving these problems is Hard-Thresholding gradient descent. However, the inherent expansibility of Hard-Thresholding operators can lead to convergence issues, necessitating strategies to accelerate the algorithm. In this article, we believe the random Hard-Thresholding algorithm can be interpreted as an equivalent biased gradient algorithm. By introducing appropriate biases, we can mitigate some of the issues of Hard-Thresholding and enhance convergence. We categorize the biases into memory-biased and recursively-biased, examining their distinct applications within Hard-Thresholding algorithms. Next, we explore the Zeroth-Order versions of these algorithms, which introduce additional biases from Zeroth-Order gradients. Our findings indicate that recursively bias effectively counteracts some of the issues caused by Hard-Thresholding, resulting in improved performance for First-Order algorithms. Conversely, due to the accumulation of errors from Zeroth-Order gradients during recursive bias, the performance of Zeroth-Order algorithms is inferior to that influenced by historical gradients. To address these insights, we propose the SARAHT and BVR-SZHT algorithms for First-Order and Zeroth-Order Hard-Thresholding, respectively, both of which demonstrate faster convergence speeds compared to previous methods. We validate our hypotheses through black-box adversarial experiments and ridge regression evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper revisit the classic Hard-Thresholding algorithm from the viewpoint of biased gradients. The biases are categorized into "memory-biased" and "recursively-biased" ones. The performance of these two categories are respectively examined with First-order and Zeroth-order Hard-Thresholding algorithms. Experiments including black-box adversarial attacks and ridge regressions are conducted to validate the empirical advantages of the proposed approaches, and analyze the parameter sensitivity. However, the writing and presentation of this submission are very poor, making the paper extremely difficult to interpret and evaluate. Consequently, I recommend a complete rewrite of the paper.

### Strengths
The paper is based on solid theoretical analysis.

### Weaknesses
The major concern for this paper is its writing and presentation, which serves as the primary hurdle to understand the paper. Below are my point-by-point comments. 

1. The paper is not self-contained -- a lot of notations come out of the blue without any definition. As a consequence, the readers of this paper have to search for related works to understand them. Unfortunately, there is even no reference for some notations. For example, symbols $\varphi$, $\nu$, and $\mathbb{N}_0$ appear in line 140, while there is not any definition/explanation/reference regarding their meaning. 

2. The notations are inconsistently used throughout the paper. For instance, in Assumption 3, the smoothness constant is denoted as $L_s$, but the function is referred to as $\rho_s^+$-strongly smooth. Additionally, $v_s$ and $L_s$ are respectively used for RSC and RSS constants in Assumptions 2 and 3, while in Section 4 the function $f_i$ is assumed $v_s$-RSS and $L_s$-RSC (rather than $v_s$-RSC and $L_s$-RSS). Other examples include Lemma 1, where $\gamma_k$ appears in the equation, but line 203 uses $\gamma$ without a subscript. 

3. Many sentences lack clear logic. Remark 1 claims that "the MSE of $g(x)$ does not completely determine the convergence", but the next sentence still suggests "using the MSE of $\nabla_{HT}$ as a substitute". Furthermore, Remark 4 states that "the
Hard-Threshold can counteract some bias, thus accelerating convergence", which directly conflicts with Remark 1. Similar logical inconsistencies appear in e.g., Section 3.3. Additionally, Sections 3 and 4 contain many lemmas and theorems that are either poorly explained or entirely unexplained; cf. Theorem 3. 

4. The experiments are also vague, and there is no reference provided in Section 5. The introduction mentions "large-scale machine learning". In contrast, the numerical tests are limited to merely ridge regression in a space of dimensionality $5$. 

5. There are also numerous grammatical error, typos, and citations of improper formats. For instance, the phrase "recursively bias effectively counteracts some of the issues" appears multiple times, where "recursively" should be corrected to "recursive."

### Questions
I would suggest completely rewriting the paper.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper studies biases in gradient descent and how they might cancel issues caused by hard thresholding. The result of this study are two new algorithms for hard-thresholding stochastic gradient descent.

### Strengths
I think this paper studied a topic that is not very explored. It might give new inside and new ideas.

The success of lasso and l1 regularization cast a shadow on l0 regularization, and I don't remember seeing a lot about it in the literature. Studying it might give new insight and propel new ideas.

### Weaknesses
I believe the papers could improve in their presentation, clarity, and empirical validation.

**Numerical experiments**:  I feel the authors don't provide a very comprehensive set of experiments mentioned in the main text 
   - It is only one experiment with ridge regression in the main text (and it uses simulated data only). There seem to be additional experiments in the appendix, but they are not even mentioned in the main text
  - I think the algorithm should be directly compared with LASSO, since it is a very popular to promote sparsity. It could be interesting to directly compare convergence and runtime to for instance SAGA implemented on Lasso.
  
**Presentation and style**: I would recommend significant improvements in presentation and style
  - I feel the results are presented with very little discussion. I feel the main part of the paper is just a collection of results. Could you maybe provide some interpretation for theorems 1, 2, and 3.
   - There are a lot of words that are Capitalized without a clear reason. For instance, Hard-Threshold, Zeroth-Order,  First-Order.
   - I would recommend improving the use of use of natbib. Avoid writing the name and then using citep, i.e. line 060, "William (de Vazelhes et al , 2022)", or in line 063 "Yuan (Yuan et al, 2024)". Maybe just use citet instead
  - The use of the notation $\nabla^{BSAGA}_t$ feels a bit unusual, and a bit confusing with the notation used for gradient
  - The figures have a hard-to-read title, captions, and numbering.
  - The excessive use of abbreviations makes the text hard to read.

### Questions
About the numerical setup
- It is a bit unclear to me why to use a  l_0 regularization together with ridge regression. One argument in the introduction is that l0 regularization avoids hyperparameters, but in the example you re-introduce hyperparameters. How the algorithm performs in the estimation with this constraint?

Other question:
- What are the asymptotic bounds on the computational complexity of iteration of the algorithm with hard thresholding?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper studies the performance of memory-biased gradient and recursively-biased gradient oracle in first-order and zeroth-order stochastic algorithms for L0-norm constrained optimization problems. Moreover, this paper also proposes several hard-thresholding algorithms of first-order algorithms (such as SARAH, BSVRG and BSAGA) and zeroth-order algorithms such as BVR-SZHT. Some experimental results are reported.

### Strengths
The paper is complete in format.

### Weaknesses
The one main contribution of this paper is to propose several first-order and zeroth-order hard-thresholding algorithms such as SARAH-HT, BSVRG-HT, and BSAGA-HT. There are much first-order and zeroth-order stochastic hard-thresholding algorithms. Therefore, the novelty of this paper is very limited.

### Questions
1.	What’s the advantage of the proposed first-order and zeroth-order stochastic hard-thresholding algorithms against related algorithms in terms of convergence rates and complexities?
2.	The experimental results are not convincing. The authors should compare the proposed algorithms with more recently proposed algorithms.
3.	Both the English language and equations in this paper need to be improved. For example,  Line 225: ‘In Hard-Thresholding algorirthm, The commonly used Zeroth-Order estimation is’.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates how different types of biases interact in Hard-Thresholding (HT) optimization algorithms for L0-constrained problems. The authors reinterpret HT algorithms as biased gradient methods and explore two main types of biases: memory-biased and recursively-biased. They analyze how these biases interact both in First-Order (FO) and Zeroth-Order (ZO) optimization settings. Based on their theoretical analysis, they propose two new algorithms: SARAHT for FO optimization and BVR-SZHT for ZO optimization. The key insight is that recursive bias can help counteract HT-induced bias in FO settings, while memory bias works better with ZO estimation bias. They validate their findings through ridge regression experiments and black-box adversarial attacks.

### Strengths
1. The theoretical framework is novel with clean reinterpretation of HT algorithms as biased gradient methods and thorough analysis of bias interactions through bounded MSE conditions.
2. The technical contributions are sufficient. For example, the authors provide rigorous analysis on bias cancellation effects and detailed convergence analysis for both FO and ZO settings. The authors also provide clear characterization of how different biases interact under HT operators.
3. There are multiple metrics for evaluation such as IFO, IZO and NHT.

### Weaknesses
1. The experiments are somewhat weak. 

- The main paper only presents ridge regression experiments, while important black-box adversarial experiments are deferred to the appendix. 
- I recommend moving key adversarial attack results to the main paper, particularly those demonstrating the practical benefits of bias cancellation in zeroth-order optimization
- No evaluation on real-world large-scale datasets. Specifically, I recommend testing on: a) Sparse feature selection problems using MNIST/CIFAR-10 for computer vision, b) Gene expression datasets like Colon Cancer or Leukemia for bioinformatics applications, c) Text classification with sparse word embeddings using Reuters or 20 Newsgroups datasets. These datasets would demonstrate the practical utility of the proposed methods across diverse domains.

2. As for the theretical analysis, the discussion of when these assumptions might fail is limited and there is no analysis of what happens when conditions are violated.

- The Restricted Strong Convexity (RSC) and Restricted Strong Smoothness (RSS) assumptions are quite strong and their limitations should be discussed. Specifically, these assumptions may fail in:
a) Deep neural network optimization where loss landscapes are highly non-convex.
b) Problems with heavy-tailed noise where smoothness is violated.
c) High-dimensional settings where restricted eigenvalue conditions break down.

- The paper should analyze algorithm behavior when these conditions are violated and propose potential modifications or relaxations of the assumptions.

### Questions
1. How do the proposed algorithms perform on non-smooth optimization problems or problems where RSC/RSS conditions are violated?
2. What is the computational overhead of tracking and managing different types of biases compared to simpler approaches?
3. How would the bias interaction analysis extend to other sparsification operators beyond hard thresholding (e.g., soft thresholding)?
4. Can the framework be extended to handle constrained optimization problems while maintaining the bias cancellation benefits?

### Soundness
3

### Presentation
3

### Contribution
3
