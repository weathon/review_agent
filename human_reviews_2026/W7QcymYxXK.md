# Bridging Unsupervised and Semi-Supervised Anomaly Detection: A Provable and Practical Framework with Synthetic Anomalies

- Decision: Reject
- Scores: 6, 4, 0

## Abstract
Anomaly detection (AD) is a critical task across domains such as cybersecurity and healthcare. In the unsupervised setting, an effective and theoretically-grounded principle is to train classifiers to distinguish normal data from (synthetic) anomalies. We extend this principle to semi-supervised AD, where training data also include a limited labeled subset of anomalies possibly present in test time. We propose a theoretically-grounded and empirically effective framework for semi-supervised AD that combines known and synthetic anomalies during training. To analyze semi-supervised AD, we introduce the first mathematical formulation of semi-supervised AD, which generalizes unsupervised AD. Here, we show that synthetic anomalies enable (i) better anomaly modeling in low-density regions and (ii) optimal convergence guarantees for neural network classifiers — the first theoretical result for semi-supervised AD. We empirically validate our framework on five diverse benchmarks, observing consistent performance gains. These improvements also extend beyond our theoretical framework to other classification-based AD methods, validating the generalizability of the synthetic anomaly principle in AD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper demonstrates that using synthetic anomalies improves the performance of semi-supervised anomaly detection. 
It first formulates anomaly detection as a binary classification problem, 
then shows why training a model using only normal data and known anomalies is difficult. 
The proposed method theoretically resolves this issue by incorporating synthetic anomalies generated from a uniform distribution in addition to known anomalies. 
Experiments on tabular, image, and text data show that adding synthetic anomalies enhances the performance of semi-supervised anomaly detection.

### Strengths
- By defining a unified anomaly detection framework based on binary classification, this paper can handle both unsupervised and semi-supervised settings. Based on this, this paper also theoretically justify the use of synthetic anomalies.
- The experimental results are very strong. Across a variety of methods and datasets, incorporating synthetic anomalies leads to improved performance.

### Weaknesses
Please see the Questions section.

### Questions
- In the proposed method, noise drawn from a uniform distribution is used as synthetic anomalies. However, since normal data also is a subset of a uniform distribution, would not the synthetic anomalies contain normal data as well? If so, I would expect the detection performance for normal data to drop. Why is the proposed method able to avoid this? (For example, DROCC also uses synthetic anomalies, but it includes mechanisms to avoid overlapping with normal data. That approach feels more natural to me; yet for images and text, adding uniform noise to DROCC actually improves performance.)
- This paper uses autoencoders (AEs) in the experiments, but how about trying DeepSVDD? For tabular data, AEs may be better, but for image data I expect DeepSVDD to yield stronger results. I am interested in how the proposed method would perform within DeepSVDD-based variants such as DROCC, ABC, and DeepSAD.

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
This paper proposes to add synthetic anomalies to diversify the collected anomalies under the semi-supervised setting. Authors connect anomaly detection with binary classification and introduces synthetic anomalies to mitigate two issues in semi-supervised AD: false negative modeling and insufficient regularity of learning. Some theoretical analyses are provided to justify the effectiveness of incorporating synthetic anomalies. Experiments across tabular, image, and text datasets  demonstrate the applicability of the proposed framework.

### Strengths
1.  Some theoretical analyses are conducted on the effectiveness of introducing synthetic anomalies for semi-supervised anomaly detection.

2. The experiments span diverse modalities (tabular, image, text) and multiple AD methods, showing general applicability of the “synthetic anomaly” principle.

### Weaknesses
1. Formulating anomaly task as binary classification is fundamentally inappropriate. Since the type of anomaly is uncountable, anomaly detection is usually formulated as one-class classification to model the distribution of normal data or to learn the pattern of them. Using binary classifier may learn a unreliable decision boundary.
 
2. The theoretical analysis of convergence is narrow to the network using ReLU as activation function. Extending the theoretical guarantees to broader architectures or activation functions would significantly strengthen the generality and impact of the results.

3. The novelty of this paper is weak. While the theoretical framing is elegant, the core idea is adding synthetic anomalies, which is not new. The main contribution lies in extending this idea to a semi-supervised setting, which feels incremental and does not substantially push the frontier of anomaly detection research.

4. Synthetic anomaly generation is overly simplistic. The use of uniformly random noise as synthetic anomalies is questionable, especially for complex or high-dimensional data. This weakens the practical significance of the framework and may not generalize to high-dimensional or structured data. There is no comparison with more informative or adaptive anomaly generation methods.

5. The presented ablation resembles a sensitivity analysis rather than a comprehensive investigation.

6. The writing of this work is terrible and should be significantly improvoed.

### Questions
1. How sensitive is the framework to the way synthetic anomalies are generated? Would a more structured generator improve performance?
2. It is possible to generalize the theoretical guarantees to broader architectures or activation functions?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper deals with semi-supervised anomaly detection. It states that anomaly detection and semi-supervised anomaly detection can be approached as binary classification.

It proposes as algorithmical contribution in section 4.1 to sample from a uniform distribution background to create artificial anomaly samples.
 
They cite several theoretical results on excess risk convergence over a function class for binary classification. They prove certain special cases.

They perform experiments using relu networks.

### Strengths
Negligible in the light of the below weaknesses.

### Weaknesses
The algorithmical proposal of this paper, sampling anomaly background ,  is long known (e.g. the paper by Sipple 2020, but it is known before), is trivial and has no novelty. There has been advantages over that, see eg SMOTE, Chawla et al from 2002 . 

- The claim that Semi-supervised anomaly detection can be treated as classification is not novel. Steinwart 2005 has established this formally for anomaly detection in general (Corollary 3 and theorem 4 in Steinwart 2005). 

Were it not for the old suggestion of sampling negatives and the many mistakes, this paper would feel like a recapitulation of Steinwart 2005, or one tries to confuse the readers over the simplicity of the algorithmical content by citing convergence bounds.

- The paper has a number of severe theoretical mistakes:

1. line 148-149 they state that if a non-negative lower bound converges to zero, then the term which it bounds from below must also converge to zero. Their statement is: $a >= b$, $b>=0$, $b \rightarrow 0$ implies $a \rightarrow 0$ 

for evidence see  "from [4], we see ..."

2. a similar wrong conclusion occurs in lines 240-244

"From Proposition 3.1 and Theorem 3.3, we can see that if the regression function is discontinuous, the approximation error is high (at least 1), which may lead to vacuous excess risk bounds (i.e., excess risk can be high and is not guaranteed to converge). Lacking theoretical guarantees, the Bayes classifier cannot be effectively learned."

If an upper bound diverges, it does not mean that the quantity bounded by it would have to diverge as well. Same kind of logical mistake as in 1., but now with an upper bound.

3.Proposition 4.2 is obviously wrong. They claim continuitity, however if $h_-(X)$ is discontinuous, then $f_P(X)$ can be discontinuous, too.
E.g. choose $s=0.5, \tilde{s}=0.5, h_1 =c$, then 
$f_{P}(x) = \frac{0.5c -0.25 h_-(X) -0.25}{0.5c + 0.25h_-(X) + 0.25}$ 

4. eq (4) is proven in Steinwart (2005) as an upper bound, see Theorem 10 in Steinwart (2005).
Proving the exact same result a lower bound would be very surprising.
They use exactly the same argument as Steinwart 2005 in the proof of theorem 10, but arrive at the opposite direction of inequality.

If this is corrected to the correct  direction of inequality, their extension is straightforward. Steinwart 2005 assumes for the anomaly density to be $\mu$. They assume that it has density $h_2$ with respect to $\mu$ . There is no technical effort in doing this change.

btw, line 1101 makes a lower bound (it should use 3/5 as constant but this is minor) . 


5. Proposition 3.1 is wrong because they do not ensure that $\mu(X_1) >0$ and $\mu(X_-) >0$ . One can choose closed sets such that  $\mu(X_1) =0$ and $\mu(X_-) =0$ Then one can get a zero $\ell_{\infty}$-norm to $f(x)=0$

but even if one would fix that, it would be of no consequence, see point 7 

6. It could be that Theorem 4.5 has an unfavourable rate $O( (log n) ^4  / n ) ^{ (c+\alpha) / (c+d) }$

For $c = \alpha q$ is typically small compared to the input dimensionality $d$ if one wants smooth settings as they state it

for even moderate input dimensionalities d the bound is worse than the typical $O(n^{-1/2})$ results .

7. the "insufficient regularity of learning" problem as they state it is no problem for training a classifier: 

assume $P[Y=1|X]$ makes a jump in direction orthogonal to the decision boundary, but the decision boundary is a standard hyperplane. This is trivially learnable with 1 layer. 

- Ironically, Tsybakovs noise condition, which is repeatedly cited by the submitters of this paper, requires a steepness of $\eta(x) = P(Y=1|X=x )$ around 0.5 for faster convergence rates. They state that this steepness would a problem for learning. This is a direct contradiction to the results from Tsybakov and Steinwart. 

- Overall, the paper has very poor readability.

### Questions
none

### Soundness
1

### Presentation
1

### Contribution
1
