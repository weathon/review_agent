# Revisiting Non-separable Binary Classification and its Applications in Anomaly Detection

- Decision: Reject
- Scores: 3, 5, 6

## Abstract
The inability to linearly classify $\texttt{XOR}$ has motivated much of deep learning.
We revisit this age-old problem and show that $\textit{linear}$ classification of $\texttt{XOR}$ is indeed possible.
Instead of separating data between halfspaces, we propose a slightly different paradigm, $\texttt{equality separation}$, that adapts the SVM objective to distinguish data within or outside the margin.
Our classifier can then be integrated into neural network pipelines with a smooth approximation.
From its properties, we intuit that equality separation is suitable for anomaly detection.
To formalize this notion, we introduce $\textit{closing numbers}$, a quantitative measure on the capacity for classifiers to form closed decision regions for anomaly detection.
Springboarding from this theoretical connection between binary classification and anomaly detection, we test our hypothesis on supervised anomaly detection experiments, showing that equality separation can detect both seen and unseen anomalies.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The standard linear SVM fails in the classification of the XOR problem.  To resolve this problem, the authors proposed a new paradigm, equality separation. Additionally, they integrated the idea of equality separation into the neural network and applied the proposed method to supervised anomaly detection tasks.

### Strengths
* The authors studied the VC dimension of the proposed equality separation. The idea of equality separation can also be applied in the neural network.
* The authors introduced the notion of closing numbers to quantify the difficulty of forming a closed decision boundary.

### Weaknesses
* Many times, I got overwhelmed and distracted by the narration and the layout of the manuscript. For examples,
  * The sentence "Equality separators use the distance from a learnt hyperplane as the classification metric" is confusing: A distance is measured between two objects, but here the authors only mention one object, i.e., the learnt hyperplane.
  * I feel the authors spent too much tutorial-like narration for VC dimension, closing number, and locality, in the main manuscript. In particular, what is the purpose of the over-detailed VC dimension? What useful information can we conclude in this section? Did the authors want to use the VC dimension to give some theoretical bound of generalization errors?
  * In section 2.2, it is difficult to follow the mixed descriptions. The authors may use bullet points to describe case-by-case and use some plots to support the narration, if necessary.
  * In section 3, the popped sentence "The utility of equality separators becomes evident in AD setting" is confusing since there is no particular interpretation of the utility of the equality separator in Anomaly Detection in the previous section after the introduction. As for the "Anomaly detection" in the introduction, it looks more like related works, maybe the authors could consider moving that part into Section 3.

* For the equality separator, what is the necessity of this proposed method? 
  * Even though linear SVM fails to solve the XOR classification while equality separator can, why do not use kernel SVMs?
  * In Figure 3(e), what if the unseen classes fall into the purple region but are far away from the brown points? Will they be classified as brown classes when using $\epsilon$-error separator?  What if the brown class is surrounded by the blue class which consists of several cohorts? In this case, does $\epsilon$-error separator work?
  * When considering the toy example in Figure 3, the authors also use the kernel to improve the shallow equality separator. Does this imply that the proposed equality separator (even though it is simple and linear) in general is not proper without kernel or activation?

* The decision of  $\epsilon$-error separator depends on the value of $\epsilon$, but I cannot see any discussion on the choice or computation for the value of  $\epsilon$.

* Since the paper is titled "in Anomaly Detection", it should contain more well-established anomaly detection benchmarks (http://odds.cs.stonybrook.edu)

* There is no discussion on Deep One-Class Classification [1] or a comparison with it. This related work also targets anomaly detection by forming a circle boundary to the normal classes.

* Is that possible to graphically show the closed decision boundaries on other examples formed by the proposed method? 

[1] Lukas Ruff, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel Müller, Marius Kloft Proceedings of the 35th International Conference on Machine Learning, PMLR 80:4393-4402, 2018.

### Questions
* Multiple minor issues:
  * Line 1, Page 3: do you mean $\mathbb{R}^+\cup\{0\}$?
  * Theorem 2.3.: separators $\mathcal{H}$ "in" Def.
  * Corollary 2.4: do you mean $\mathcal{H}_\epsilon$?

* "modeling a halfspace separator …. with an equality separator requires non-linearity like ReLU":  could the authors explain more about how the ReLU reflects the modeling for a halfspace with an equality separator?

* "equality separators yield a different optimal hyperplane compared to SVMs in binary classification": could the authors articulate the "optimality" here?

* "where equality separation is more conservative in classifying the positive brown class": What do you mean by "more conservative"?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors explore the space of halfspace separator. In this manuscript they explore the equality separator. Instead of dividing the space into two halves, all instances that fall on the hyperplane (or are near to it) are part of one class while the rest belong to the other class. The others then calculate the VC dimension of this equality separator. Furthermore, they also introduce the bump activation function to be used in NNs which is a smoothed version of the equality separator. They propose using this separator for anomaly detection. Finally, they show the efficacy of the proposed method in the experimental section.

### Strengths
1. The proposed equality separator is very interesting. Even though for epsilon-separator is related to SVMs there is still other novel aspects to this. Furthermore the theoretical analysis shown here for VC dimension shows the advantage of the proposed method over regular linear separators.
2. The results for anomaly detection is promising specially on the synthetic data set.
3. The paper is very well written. All required information is provided in a clear manner and explained properly.

### Weaknesses
1. As mentioned above, the anomaly detection results in this paper are promising. However, the gain on the NSL-KDD dataset is not always positive. This limits the application of the proposed method.
2. The authors performed thorough experiments on the NSL-KDD dataset. However, further datasets should also be included in the experimentation to show the efficacy of the proposed method.

### Questions
1. This is related to concern regarding weakness 1. What are the authors intuition regarding the equality separator not always outperforming the other baseline methods for NSL-KDD.
2. I noticed that in Table 3, for DOS, HS-NS result is in bold. Why is that? I though ES-NS performs the best here?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work discusses a novel approach to linearly classify the XOR problem, challenging the conventional wisdom that it cannot be done with linear methods. The authors propose "equality separation" as an alternative to traditional halfspace separation, adapting the SVM objective to distinguish data within or outside the margin. They integrate this classifier into neural network pipelines, highlighting its potential for anomaly detection and demonstrating its effectiveness in supervised anomaly detection experiments, including detecting both seen and unseen anomalies.

### Strengths
- The introduction of an 'equality separator' to address the XOR problem is indeed an intriguing and innovative concept.
- The introductory section is well-structured and easily comprehensible, complemented by the informative Figure 1.
- All the theoretical assertions are substantiated with precise definitions and rigorous proofs.

### Weaknesses
- In order to enhance the accessibility and comprehensibility of the content, it would be advisable to incorporate critical discussions and analyses that are currently relegated to the appendix into the main body of the manuscript.
- The proposed design has exclusively undergone experimentation on toy datasets or relatively straightforward real-world datasets. Consequently, there is uncertainty surrounding the effectiveness of the proposed method when confronted with more intricate, real-world datasets.

### Questions
- Is this network design extensible to more intricate datasets, such as image data?
- Isn't there a gradient vanishing problem with that bump activation design when the layers of the neural network are deep?
- What are the advantages of doubling the VC dimension in contemporary neural network architecture?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
