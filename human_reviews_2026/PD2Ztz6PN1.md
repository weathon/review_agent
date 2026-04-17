# Function regression using the forward forward training and inferring paradigm

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Function regression/approximation is a fundamental application of machine learning. Neural networks (NNs) can be easily trained for function regression using a sufficient number of neurons and epochs. The forward-forward learning algorithm is a novel approach for training neural networks without backpropagation, and is well suited for implementation in neuromorphic computing and physical analogs for neural networks. To the best of the authors' knowledge, the Forward Forward paradigm of training and inferencing NNs is currently only restricted to classification tasks. This paper introduces a new methodology for approximating functions (function regression) using the Forward-Forward algorithm. The paper further evaluates the developed methodology on univariate and multivariate functions and benchmarks the framework on open source regression data, while comparing its performance to other regression techniques.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper extends the concept of forward-forward networks, originally used for classification tasks, to regression tasks by carefully assigning positive and negative labels to data points. The paper tests forward-forward for regression methods on eight problems. The results look significantly good for these problems. There are no comparative evaluations with any other methods to show any significant uptake of these FF-regression for solving real-world regression problems or regression benchmarks.

### Strengths
The paper presents a clever trick for converting an FF classification model into an FF regression model.  It has been tested over 8 regression functions and shows the usefulness of the method.

### Weaknesses
The paper lacks comprehensiveness. For example, in the abstract and conclusion, it is mentioned that FF-regression is extended to Kolmogorov-Arnold Networks and Deep Physical Networks. These have not been discussed in the main paper at all. Only two Figures have been described in the Appendix without good detail. 

Methods lack any comparison with other standard methods or analysis to confirm the real-world usability of the FF-regression to achieve the energy efficiency goal.  

Trivial thing, but I am wondering whether the Appendix in the same PDF, which counts to 14 pages, violates the page limit.

### Questions
Are these functions evolved for metrics like R2
What is the performance of FF on the simple regression problem on the UCL repository? Whether authors tested this algorithm on those problems.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper extends the Forward-Forward (FF) algorithm for classification to function regression. It treats regression as binary classification: points within a tolerance of the true value are "in-tol" (label 1), others are "out-tol" (label 0). The network uses cosine similarity as the metric to distinguish between correct and incorrect labels.

### Strengths
1. This is the first to apply the forward-forward (FF) learning algorithm to regression. It effectively extends a classification-only method to continuous function approximation. This fills a clear gap in the FF literature.
2. The pseudocode is clear, and the algorithms are easy to understand. Algorithms 1 and 2 provide step-by-step, reproducible training and inference procedures. The accompanying figures further enhance clarity.
3. The method successfully works on regressing low-dimensional functions. It produces reasonable approximations with meaningful uncertainty estimates on 1D, 2D, and 3D benchmarks. Results are well-visualized and supported by MSE metrics.

### Weaknesses
1. The method is tested only on simple, low-frequency functions, not on complex cases (high-dimensional, non-smooth, or multi-frequency). Such limitations in computational complexity are critical. Focusing on KANs may not be appropriate, because the uniform grids of KAN raise these core issues. I suggest that the author read some recent MLP variants that have solved these problems using the multi-scale mesh (a standard data structure in the finite element method).
2. The approach is low-efficient, with training and inference times orders of magnitude slower than backpropagation (Table 2). Despite avoiding backpropagation, the method still relies on gradient descent with no acceleration in convergence (Algorithm 1). Scaling-law analysis would likely reveal poor sample and computation efficiency.
3. The evaluation metrics are inconsistent: $R^2$ score should be used for noise-free regression, and MSE for noisy data. No comparison is made with standard mathematical or ML regressors (e.g., splines, tree-based models). The paper compares only with physical Neural Networks, which is misleading. Neural networks primarily focus on classification tasks (e.g., image generation, language modeling) and are poor at highly accurate regression tasks.

### Questions
Suggestions for Improvement

1. Broader Experiments: Test on more datasets or higher-dimensional/multi-frequency functions. Compare with traditional mathematical/ML methods.
2. Enhancements to Method: The author can explore using the multi-scale mesh instead of the uniform grid on neural networks, which is a solution to extend the application scope to higher-dimensional/multi-frequency functions and even challenge recognition tasks (such as image classification on ImageNet )

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the forward-forward (FF) framework for training and performing inference with regression neural networks. The proposed FF algorithm adapts the FF approach originally designed for neural classifiers to regression tasks.

Similar to its use in classification, the FF framework for regression relies on positive data, negative data, and a goodness function. The algorithm reframes regression as a classification problem by creating bins of target values based on training data and a user-defined tolerance level. If a bin contains the true value y for a given input  x, the algorithm assigns a label of 1 (positive data); otherwise, it assigns a label of 0 (negative data). This transformation enables binary classification between positive and negative data.

During inference, the FF framework maps a query input x to its predicted value. It generates trial points to define candidate bins, identifies the bins where x qualifies as positive data, and collects the corresponding trial points labeled as 1. The algorithm then computes the mean of these trial points to produce the final prediction.

### Strengths
The paper presents its contribution clearly and organizes its content effectively. The introduction and theoretical background offer relevant context  that positions the work and highlights its contribution. The presentation of the forward-forward approach for neural classifiers helps with understanding the new regression framework as a reader. The suitability of the forward-forward algorithm for the implementation of neuromorphic computing and physical analogs for neural networks underscores the significance of this work.

### Weaknesses
The quality of the experimental result in the paper is low and this undermines the soundness and utility of this paper.  

1)  Experimental results for the hyperparameters: This paper does not present experimental results that illustrates the effect of the hyperparameters (tol, $y_{min}$, $y_{max}$)) on the performance of this method. Subsection 3.1.1 summarizes the effects but fails to provide quantitative evidence to support this.

2). Limited simulations: The simulations in this paper is restricted to datapoints from a closes form expressions. It is unclear if the performance on closed form expressions with translate to typical ML-based regression problems, where you only have access to input-output pairs and not a closed-form expression. 

3). No information about the computational cost: For the inference, the new methods uses trial points to define candidate bins. The inference maps a query input x to the bins that result from these trial points. It is important to understand the computational cost of this approach. In the classification case, the number of runs per inference is K where K is the number of classes or categories. For regression, it seems that the number of runs depends on the number of trial point. But there is no information about the impact on the computational cost.

### Questions
Please refer to the points under weaknesses.

I also noticed a typographical error on line 183: " ... to obtain a the mean and the standard deviation..."

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new methodology for approximating functions, that is, function regression, using the Forward-Forward algorithm.

### Strengths
1. The studied topic focusing on function regression with FFA is interesting.

2. The algorithms are concrete, enabling the reproducibility and contributing to understanding of the proposed method.

### Weaknesses
1. The experiments are almost conducted in a toy environment. For example, the target functions approximated here are of few structures, including only elementary functions and their linear combinations. Such regression tasks are extremely simple for neural networks.

2. This paper does not discuss the computational complexity of the method, let alone how to implement it on dedicated hardware. In the main text and experiments, the authors only demonstrate the regression results. The data examples provided offer very limited support for the overall rationale of the method. The authors do not report the runtime or convergence.

### Questions
1. The function regression in Figure 4(a) does not perform well. I am concerned about the relationship between the training data and the objective function here.

2. What are the challenges in extending this method to multiple dimensions? Is it necessary to design a goodness function or positive-negative data on a case-by-case basis?

### Soundness
4

### Presentation
2

### Contribution
3
