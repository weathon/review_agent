# Inverse Entropic Optimal Transport Solves Semi-supervised Learning via Data Likelihood Maximization

- Decision: Reject
- Scores: 6, 5, 3

## Abstract
Learning conditional distributions $\pi^*(\cdot|x)$ is a central problem in machine learning, which is typically approached via supervised methods with paired data $(x,y) \sim \pi^*$. However, acquiring paired data samples is often challenging, especially in problems such as domain translation. This necessitates the development of *semi-supervised* models that utilize both limited paired data and additional unpaired i.i.d. samples $x \sim \pi^*_x$ and $y \sim \pi^*_y$ from the marginal distributions. The usage of such combined data is complex and often relies on heuristic approaches. To tackle this issue, we propose a new learning paradigm that integrates both paired and unpaired data **seamlessly** through the data likelihood maximization techniques. We demonstrate that our approach also connects intriguingly with inverse entropic optimal transport (OT). This finding allows us to apply recent advances in computational OT to establish a **light** learning algorithm to get $\pi^*(\cdot|x)$. In addition, we derive the universal approximation property demonstrating that our approach can theoretically recover true conditional distributions with arbitrarily small error. Furthermore, we demonstrate through empirical tests that our method effectively learns conditional distributions using paired and unpaired data simultaneously.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a semi-supervised objective for the translation task that can leverage both paired and unpaired samples from the source and target domains.
The paper decomposes the standard MLE objective into three parts corresponding to a joint expectation and two marginal expectations over the unpaired source and target samples.
The key challenge is that parts of the objective require integration.
To avoid this computational challenge, the paper makes several critical assumptions about the joint cost function and the target-specific function that makes the necessary functions known in closed-form.
In particular, a Gaussian mixture model assumption and a simple log-sum-exp form of the cost function.
With these closed-form solutions, the paper then demonstrates the method on a 2D synthetic dataset and a weather prediction task.

### Strengths
- Proposes a promising MLE approach to incorporating both paired and unpaired data for the translation task.

- The proposed algorithm does not seem to have any complex hyperparameters.

- Shows a nice connection between using an energy-based conditional generative model and inverse entropic optimal transport. This opens up using insights and methods from computational OT.

- The paper was easy to read.

### Weaknesses
- Intuition and theoretic understanding of objective function is not clear. This makes it difficult to understand and extend. What is the purpose of the objectives corresponding to the Y and X marginals? Why does this make sense? What if these are misestimated?

- The assumptions are not clearly discussed and their corresponding limitations. What is the limitations of assuming the particular form of the needed parametric functions? It seems restrictive. Also, why is it required to have an energy-based model? For better or worse, score-based/diffusion-based generative models are far more common now. Is using an energy-based model significantly more restrictive?

- The practical motivation of the scenario is lacking. When would you expect this situation to hold in the real-world? Which real-world scenarios naturally have both paired and unpaired data?

- The experiments are relatively weak with only one 2D synthetic experiment and 1 weather experiment, but the weather experiment only includes a simple MLP regression baseline (i.e., none of the other baseline methods).

### Questions
- What is the intuitive interpretation of the new objective? This seems a bit ad-hoc/heuristic to introduce the $f(y)$ term for the sole purpose of using marginal samples from $p(y)$.

- Why use the parametrization in 16 and 17? What is the intuition of this parametrization? Is it a universal type of parametrization (i.e., can it represent all interesting cost functions)? Can you give examples of cost functions that do not have this form?

- How does this compare to the most natural likelihood objective that is $L(\theta) = E_{x,y}[-\log \pi^\theta(y|x)] + E_y[-\log E_x[\pi^\theta(y|x)]]$, which is basically the expected conditional KL plus a regularization term on the conditional that ensures it matches the marginal of p(y), i.e., integrating the conditional over the marginal of x should yield the marginal of y.  This seems to be a more straightforward (albeit hard-to-compute) semi-supervised objective.

- What does it mean that the x and y terms are asymmetric? It would seem that the marginal optimization terms should be symmetric as the unpaired data has no directionality per se.

- What if the paired data is not iid from the joint distribution? For example, if there are certain regions of the space that are more likely to have paired samples? This seems quite natural and reasonable in real-world scenarios. How would your method fair in these scenarios?

- How do you ensure $c^\theta(x,y)$ is a valid cost function?

*Summary of Review*
Overall, I found the paper to be intriguing, particularly in the connection between inverse optimal transport and the proposed objective. However, the derivations and development lacked strong intuition and explanation---rather seeming like mostly algebraic manipulation without a clear reason. The paper could be improved significantly with a much deeper understanding and interpretation of the objective. Furthermore, the assumptions lacked significant discussion while being relatively restrictive. Finally, the empirical results were relatively weak only having one 2D synthetic experiment and 1 weather experiment on a very small dataset with less than 1000 samples. Thus, the scalability and applicability of the method are questionable.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a novel approach to semi-supervised domain translation by introducing a likelihood-based loss function that naturally incorporates both paired and unpaired samples. The key theoretical contribution is establishing a connection between their loss and inverse entropic optimal transport - a connection that hasn't been previously explored in the literature. The authors propose using a Gaussian mixture parameterization that allows closed-form expressions for their loss terms. They empirically validate their method against standard supervised/unsupervised approaches on synthetic 2D data and a weather prediction task.

While the paper presents an interesting theoretical contribution connecting inverse entropic optimal transport with likelihood estimation, the lack of comparisons against competing semi-supervised methods and limited experimental validation makes it difficult to assess the practical value of the proposed approach. Thus, currently the paper does not reach the bar for acceptance.

### Strengths
- Novel likelihood maximization loss for semi-supervised domain translation that elegantly handles both paired and unpaired data within a single objective.
- Novel theoretical connection established between inverse entropic optimal transport and likelihood estimation.
- Empirical validation demonstrating how performance scales with both paired and unpaired sample sizes.

### Weaknesses
- Experimental validation:
	- While the related work section discusses multiple approaches to semi-supervised domain translation, the paper lacks empirical comparisons against any of these existing methods. This makes it difficult to assess the relative advantages of the proposed approach compared to established methods that also leverage both paired and unpaired data.
	- The paper would be substantially strengthened by: 1) clearly articulating the advantages of the proposed method over existing approaches, and 2) providing direct empirical comparisons against state-of-the-art semi-supervised methods. Currently, comparisons are limited to baseline methods that were not specifically designed for semi-supervised domain translation.
	- The experimental validation would benefit from including tests on higher-dimensional problems, such as the image translation tasks explored in [1], to demonstrate the method's scalability beyond simple domains.
- Reliance on Gaussian mixture parameterization:
	- The authors acknowledge that their choice of Gaussian mixture parameterization inherently limits the method's scalability to more complex domains.
	- The discussion of alternative parameterizations seems incomplete, leaving questions about potential extensions to more expressive models.
	- While the authors claim their parameterization is "light-speed," this assertion lacks proper justification or comparative analysis.

[1] Xiaole Tang, Xin Hu, Xiang Gu, and Jian Sun. "Residual-conditioned optimal transport: Towards structure-preserving unpaired and paired image restoration". ICML 2024.

### Questions
- The authors describe their parameterization as "light-speed" - what are the specific computational advantages that justify this claim?
- What makes Gaussian mixture parameterization particularly suitable for the proposed loss compared to alternative parameterizations?
- How does the proposed method compare to existing semi-supervised domain translation techniques in terms of both theoretical properties and empirical performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces a novel semi-supervised learning approach that leverages both paired and unpaired data to learn conditional distributions through a data likelihood maximization framework. The proposed method constructs a new loss function that integrates paired and unpaired data directly into the objective, with a connection to inverse entropic optimal transport (OT). By framing the problem as an inverse entropic OT, the authors develop an algorithm that employs Gaussian mixture parameterization to approximate the conditional distribution in a computationally efficient manner. The empirical evaluation demonstrates that this method can achieve effective learning outcomes even with limited paired data, as long as sufficient unpaired data is available, showing promising results on synthetic and real-world datasets.

### Strengths
Clarity: The paper is reasonably structured and provides some intuition on connecting OT with likelihood maximization.

Empirical Results: The empirical tests show potential for the method to handle limited paired data, a typical challenge in semi-supervised setups.

### Weaknesses
Justification of Loss Function: The paper lacks sufficient theoretical or empirical justification for the specific loss function design. The decomposition into paired and unpaired data components seems plausible, but more discussion on its necessity or optimality would strengthen the work. Additionally, while the paper claims the loss solves inverse entropic OT, the practical implications of this for general semi-supervised tasks are underexplored.

Limited Comparisons: The experimental section lacks comparisons with several prominent semi-supervised learning methods that do not rely on OT or Gaussian mixtures. Without comparisons to these methods, it is challenging to gauge whether the proposed approach offers an advantage over non-OT-based techniques.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
