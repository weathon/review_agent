# Rényi Sharpness: A Novel Sharpness that Strongly Correlates with Generalization

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 2

## Abstract
Sharpness (of the loss minima) is widely believed to be a good indicator of generalization of neural networks. Unfortunately, the correlation between existing sharpness measures and the generalization is not that strong as expected, sometimes even contradiction occurs. To address this problem, a key observation in this paper is: what really matters for the generalization is the *average spread* (or unevenness) of the spectrum of loss Hessian $\mathbf{H}$. For this reason, the conventional sharpness measures, such as the trace sharpness $\operatorname{tr}(\mathbf{H})$, which cares about the *average value* of the spectrum,  or the max-eigenvalue sharpness $\lambda_{\max}(\mathbf{H})$), which concerns the  *maximum spread* of the spectrum, are not sufficient to well predict the generalization. To finely characterize the average spread of the Hessian spectrum, we leverage the notion of *Rényi entropy* in information theory, which is capable of capturing the unevenness of a probability vector and thus can be extended to describe the unevenness for a general non-negative vector (which is the case for the Hessian spectrum at the loss minima). In specific, in this paper we propose the *Rényi sharpness*, which is defined as the negative of the Rényi entropy of loss Hessian $\mathbf{H}$. 
Extensive experiments demonstrate that Rényi sharpness exhibit *strong* and *consistent* correlation with generalization in various scenarios. Moreover, on the theoretical side, two generalization bounds with respect to the Rényi sharpness are  established, by exploiting the desirable reparametrization invariance property of Rényi sharpness. Finally, as an initial attempt to take advantage of the  Rényi sharpness for regularization, Rényi Sharpness Aware Minimization (RSAM) algorithm is proposed where a variant of Rényi Sharpness is used as the regularizer. It turns out this RSAM is competitive with the state-of-the-art SAM algorithms, and far better than the conventional SAM algorithm based on the max-eigenvalue sharpness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose a novel sharpness measure to predict generalization performance: Renyi sharpness. They provide generalization bounds based on this measure, and experimental studies on CIFAR and TinyImagenet. Further, they also introduce a variant of the sharpness-aware minimization (SAM) algorithm that is based on an approximation of the Renyi sharpness, with experiments on Cifar and TinyImageNet.

### Strengths
The paper aims to address a fundamental problem in deep learning: the poor correlation between many generalization measures, and the empirically observed generalization performance. Using the Renyi sharpness as generalization measure because it captures uniformity is novel (to the best of my knowledge). In the provided experiments, the Renyi sharpnes shows good correlation (albeit doubts remain, see weaknesses below).

### Weaknesses
I have strong doubts on the conclusions that can be drawn from the experiments provided by the authors. In particular, I have concerns regarding the considered sharpness setup, the baselines that were used, the insightsfulness of the generalization bounds, and the effectiveness of the provided RSAM algorithm. 

1. Sharpness setup:  
The present study on the connection between Renyi sharpness and generalization is much less extensive than other work investigating the relationship between sharpness measures and generalization (e.g. Andriushchenko et al [1]). In [1], the authors also found setups - similar to the ones considered in this work - where sharpness _could_ predict generalization. However, Andriushchenko et al [1] showed that this might only be true for certain subgroups of the training parameters. In particular, when considering their “modern” setup (ViTs, ImageNet-scale, varied pretraining+finetuning schemes, OOD generalization and transfer learning, Language and Vision tasks,  …), the correlation disappeared. To show the effectiveness of the Renyi sharpness, experimental evidence on the scale of [1] would be necessary. 



2. Sharpness baselines and tuning of alpha:  
The provided results are for extensively tuned alpha values, whereas the baselines are apparently not tuned (e.g. the $\rho$ values for SAM and ASAM). Further, many sharpness variants (e.g. the ones from [1]) are omitted from the study, and there are no details on how exactly the baseline measures are computed or what they mean.



3. Generalization bounds:  
The generalization bound in theorem 3.2 is based on upper-bounding a generalization bound from [2] by upper-bounding the log-determinant of the Hessian with the Renyi sharpness. Using the log-determinant of the hessian (like done in [2]) would thus be better motivated by the bound. Same for theorem 3.3, where the log-determinant is also upper-bounded by the Renyi sharpness. There is thus a disconnect between the generalization bounds and the measure used. 


4. RSAM:  
The provided RSAM algorithm seems to be brittle (as admitted by the authors: warmup required, length depends on task) and barely brings improvements beyond error bars over ASAM. Further, from Tables 4 and 5 it seems that $\rho$ has been tuned for RSAM, but not for the other SAM variants. Finally, there exists a plethora of SAM variants, that are all ignored in the comparison, and the study is limited in terms of dataset size (no ImageNet scale) and models (no ViTs). 


5. More comments:

- reparametrization-invariant sharpness measures (w.r.t. layerwise rescaling) have been investigated before (see e.g. [1]), and I assume that the ASAM measure used in the provided study is one variant of the measures in [1]. So reparametrization invariance alone cannot be the reason for the (potential) success of a novel sharpness measure. I recommend discussing this when arguing about reparametrization invariance. 
- the authors claim “in our opinion, what matters the most for characterizing the generalization
is the extent of the spread of the spectrum” when introducing the Renyi sharpness as generalization measure, but it is unclear to me where this opinion stems from. 
- The authors describe how alpha might be chosen, depending on the spectrum, but do not elaborate how this choice might look like in practice, and then tune alpha extensively in their experimental section. Is there an intuition on when to chose low or high alpha? Is it necessary to manually inspect the spectrum and decide on multi-cluster vs uniform? 



Typos:

several times (e.g. Line 147): denote vs donate

Line 215 Ler vs Let


[1]  Maksym Andriushchenko, Francesco Croce, Maximilian Müller, Matthias Hein, and Nicolas Flammarion. A modern look at the relationship between sharpness and generalization. 

[2] Zhiwei Jia and Hao Su, Information-Theoretic Local Minima Characterization and Regularization, ICML 2020

### Questions
I do not have specific questions where an answer would change my opinion on the paper - I think significant changes, addressing the weaknesses outlined above, are necessary to convincingly argue in favour of Renyi sharpness:
- a much extended experimental setup, like in Andriushchenko et al
- more and better tuned baselines 
- a practical way of choosing alpha, without the necessity of tuning it
- convincing arguments why the generalization bounds argue in favour of Renyi sharpness instead of log-det (H)
- convincing evidence that RSAM improves robustly for fair comparison (tuned rho) over baselines

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a novel sharpness measure, Renyi sharpness, to consider the spread of the Hessian eigenvalues.
It shows a better correlation with generalization. They also propose a regularization method, RSAM, which outperforms other SAM variants.

### Strengths
- strong empirical results
    - strong performance of RSAM
    - strong  correlation between Renyi sharpness and generalization
- scaling invariance of Renyi sharpness (see weaknesses)

### Weaknesses
-  **Lack of motivation**. Why do we need to consider the **spread** of Hessian eigenvalues? 
    - The authors said "uniform eigenvalue is the most desirable to ensure good generalization, since if there exists no particularly large eigen direction, small perturbation of data would just incur small loss change"
    - This seems like a wrong statement.
    - If all eigenvalues are large but similar (small spread), e.g., isotropic, then there surely exists large eigen direction, but Renyi sharpness is small as it only consider the spread, not the magnitude.
    - $H_\alpha(cI)=1/(1-\alpha)\log \sum (1/n)^\alpha=\log n$ does not depends on the value of $c>0$.

- The concept in Prop 2.2 is **not "reparameterization" invariance, but "scaling" invariance**. Renyi sharpness may not be (nonlinear) reparameterization invariant.

- It seems like $A$ in the bound in (4) is not constant and depends on $\theta$. 

- There is no definition for capital $N$.

- Not a fair comparison. "We vary $\alpha$ and plot the sharpness that attains the highest correlation coefficient". At least, you should report the best $\alpha$ for each layer. It would be better to draw a scatter plot for $(\alpha, \tau)$-pairs (compared to the other baselines, e.g., trace) for each layer. If you pick the best measure (or $\alpha$) after observing the gap, it is not an useful generalization measure.

- use $\langle w_j', v_j\rangle$ or $w_j^{'\top}v_j$ to improve readability.

- It would be better to write $|g_j|^{2\alpha}$ or $(g_j^2)^\alpha$ instead of $g_j^{2\alpha}$ in (10) to avoid a confusion for the case of $\alpha=0.5$.

- Can you elaborate more on the meaning of the **layerwise** Renyi shaprness? Is it important because Prop 3.1 can be applied to a single layer?

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors study a new sharpness measure motivated by Rényi divergences in statistics. 
The authors argue that uniformity in eigenvalues of Hessian (for a constant sum of eigenvalues) is desirable to promote generalization, as opposed to the original SAM, which aims to only minimize the largest eigenvalue of Hessian, being independent of the uniformity. They show that Rényi entropy, when applied to the normalized eigenvalues of the Hessian, can characterize the spread of the spectrum. Moreover, they prove generalization bounds from Rényi sharpness using two ideas: (1) Rényi sharpness, according to its definition, is a normalized function of Hessian eigenvalues, thus it is invariant to rescaling of parameters in homogeneous neural networks such as ReLU nets. (2) Multiplicative weight perturbations, according to orthogonal transformations. Furthermore, they experimentally show how generalization correlates with Rényi sharpness (Section 5) and propose Rényi-SAM, achieved via a new loss function. The method is validated over various datasets. 
 

 




Here is a detailed summary of their main contributions: Rényi sharpness is introduced, and it is shown that it is invariant to rescaling. Its correlation to generalization is shown. In Proposition 3.1, they show that perturbation via orthogonal matrices allows us to upper bound the population loss (Equation 3). This leads to a PAC-bayesian generalization bound in Theorem 3.2 and Theorem 3.3 that relates the population loss to the empirical risk plus some term involving Rényi sharpness.


They use PyHessian to estimate the Hessian of the neural network, which leads to Section 4.1. This includes a comprehensive discussion of how to choose the parameter alpha involving Rényi sharpness. 

To estimate Rényi sharpness, they propose a method based on writing the quantity in terms of the trace of powers of the Hessian, and they use the Hutchinson method along with other quadrature methods used to approximate integrals/expectations. It is given in Algorithm 1.

### Strengths
- A new notion of sharpness, which is theoretically and practically correlated to generalization, is of potential interest to the community

- The paper is an interesting mix of theory and practice

### Weaknesses
- The algorithm (Rényi sharpness) is poorly explained in the paper

### Questions
This is an interesting paper on SAM, relating Rényi sharpness to generalization. I think the paper is making good contributions, spanning from theory to algorithms and insights about generalization. Here are some comments/questions:


Definition 2.1: What happens if $H$ is non-positive definite? How do you define Rényi sharpness then? Please clarify this in the paper


Reparametrization invariance in Proposition 2.2: Why in Equation 1, alpha has to be different from one? Couldn't we take the limit and conclude that the identity also holds for that case?



Proposition 2.2: At first glance, I thought the invariance holds for the Hessian of the whole network, but looking at the proof, it looks like it holds only for an arbitrary fixed later. I ask the authors to clarify this in the text. The invariance follows from the fact that in the definition of Rényi sharpness, the authors introduced a normalization factor (the trace). 



Please include some explanations about Algorithm 1 in the next version of the paper. Currently, it is really difficult to follow what it means.  



Eq. 9: If the gradients are vectors, then what do you mean by taking squares? You mean $hh^T$?


How do you optimize the objective in Eq. 11? Do you compute the gradients of it? Do you have something like a 'base optimizer' similar to the original SAM? If you compute the gradient of Eq. 11, then you have second-order derivative terms (prohibited). Do you use the same approximations as the original SAM to resolve this? I believe a rigorous explanation of what the Rényi SAM algorithm really is is required for the next version of the paper. Since this is one of the main contributions of this paper, you need to explain this in detail, probably having a clear algorithm dedicated to it. 


There are some other notions of sharpness similar to Rényi in recent works that have never been discussed in the paper: For instance, I found 'Tilted SAM' in [1] and 'Frobenius SAM' in [2] on the web. There might also exist others, but at least these two are pretty similar. At least one expects some discussion and comparison in the paper. Please do search for more because I believe there might be other papers. 


[1] Li, Tian, Tianyi Zhou, and Jeff Bilmes. "Tilted sharpness-aware minimization." ICML 2025

[2] Tahmasebi, Behrooz, Ashkan Soleymani, Dara Bahri, Stefanie Jegelka, and Patrick Jaillet. "A universal class of sharpness-aware minimization algorithms." ICML 2024

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a novel measure using the equation for Renyi entropy. The normalized eigenvalues of the loss Hessian are treated as a distribution, and the Renyi entropy of this distribution is used as a measure. A generalization bound is derived in terms of this entropy, and the experimental results using Renyi entropy regularizer showed small improvements on accuracy on benchmark data.

### Strengths
The presented method is novel. Experiments show improvement in accuracy.

### Weaknesses
The presented method is novel, but it is difficult to recommend the paper for acceptance for several reasons. 

The most problematic aspect is that the entropy is defined for a probability distribution, whereas eigenvalue spectrum is not a probability distribution. It is unclear what probability model the authors assume and whether it is valid. In the derived bound, the authors use a posterior distribution Q, but its definition and justification are not clearly explained.  It is very hard to recognize the eigenvalue spectrum as a distribution because no corresponding random variable is defined. Does the largest eigenvalue represent \the probability of the first random variable? The overall setup and explanation are too vague to assess whether the application of Renyi entropy is conceptually meaningful. For example, I have seen using the information-theoretic measures for the output of the network because the output can be naturally considered as a probability distribution over predicted classes. However, in the case of Hessian eigenvalues, such a probabilistic interpretation is not justifiable.

The explanation in lines 176-184 is unclear What are the functions h(.), g(.) and other perturbation constants A and rho? 

Hessian should be a very large matrix, making its computation and eigenvalue calculation at each optimization step computationally expensive. The authors used the square of gradient to approximate the hessian eigenvalues in the experiment without sufficient explanation. Is the eigenvalue distribution related to the square of gradient components in each dimension? During optimization, the gradient should approach zero, while the hessian does not.

### Questions
Please address the weaknesses to increase score.

### Soundness
2

### Presentation
2

### Contribution
2
