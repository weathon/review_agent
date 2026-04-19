# On the Crucial Role of Initialization for Matrix Factorization

- Decision: Accept (Poster)
- Scores: 8, 8, 5, 5

## Abstract
This work revisits the classical low-rank matrix factorization problem and unveils the critical role of initialization in shaping convergence rates for such nonconvex and nonsmooth optimization. We introduce Nystrom initialization, which significantly improves the global convergence of Scaled Gradient Descent (ScaledGD) in both symmetric and asymmetric matrix factorization tasks. Specifically, we prove that ScaledGD with Nystrom initialization achieves quadratic convergence in cases where only linear rates were previously known. Furthermore, we extend this initialization to low-rank adapters (LoRA) commonly used for finetuning foundation models. Our approach, NoRA, i.e., LoRA with Nystrom initialization, demonstrates superior performance across various downstream tasks and model scales, from 1B to 7B parameters, in large language and diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers the matrix factorization problem, where one is given a matrix $A$ and the goal is to compute a rank-$r$ factorization that approximates $A$. It studies the setting when $A$ is PSD and the factorization is also PSD, and when $A$ is not symmetric. Three settings are studied: exact, over and under parametrization, meaning the rank of $A$ is equal, smaller or larger than $r$. Their algorithm is preconditioned gradient, and the key of this paper it show that a good initialization significantly improves convergence rate. The initialization is to set $X_0=A \Omega$ for a zero-mean Gaussian matrix $\Omega$. Due to rotational invariance of Gaussian, this initialization has many desired property, such as it aligns with the desired top-$r$ eigenspace. Using this technique, they obtain improved rates for both symmetric matrix factorization and asymmetric variants. They also show a major application for this theory, that is, the initialization for LoRA training. Standard LoRA setting initializes $X_0$ to random Gaussian matrix, and the theory developed in this paper suggests a better alternative is to use $X_0=W_0 \Omega$ where $W_0$ is the pre-trained weights. Experiments are performed on various few-shot learning tasks on benchmarks to show advantages over LoRA and its variants.

### Strengths
This paper presents a simple initialization framework for matrix factorization that significantly improves convergence rates, which is to align the principle directions of $X_0$ with $A$ through a rank-$r$ Gaussian. Algorithmically, this does not incur (asymptotically) extra cost as multiplying $A$ with a small matrix is cheap. It also showcases an interesting phenomenon: in the exact parametrization setting, the proposed algorithm has a linear convergence for iterations proportion to $\kappa^3$, then rapidly switches to quadratic convergence, resembles that of second-order method without explicitly resorting to second-order information. The application to LoRA is nice and natural.

### Weaknesses
I think the paper is overall well-written with no obvious weaknesses. The empirical improvements over LoRA seem rather marginal but I'm not an expert on experiments.

### Questions
What is the role of Gaussian playing in your analysis? To put it in other words, do you think it's possible to replace Gaussian with something that can be applied faster to $A$, or Gaussian is essential for the analysis?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The current work describes a novel initialization method for matrix factorization problems which accelerates convergence of ScaledGD both theoretically and empirically. The authors also consider extension of Nystrom initialization to LoRA model and shows its superiority compared to different baselines experimentally.

### Strengths
The current work studies how initialization affects ScaledGD method in matrix factorization problems, and considers its extension to LoRA model. I personally find the topic valuable. The theoretic bound improvement seems huge for classic matrix factorization problem. For LoRA model, recent work has studied customized optimizers and stepsize scheduler, NoRA proposes a new initialization scheme which has its theoretic grounds. Moreover, compared to PiSSA, NoRA is naturally zero-initialized and is thus a more convenient initialization to LoRA.

### Weaknesses
1. The convergence bound to under-parameterized case involves weak optimality. I'm not very familiar with this notion, the authors show all globally optimal solutions are weakly optimal, is there any intuition of this weak optimality or is it just curated from technical stuff? How larger is the space of weak optimality compared to true optimality?

2. Ideally to apply Nystrom initialization to LoRA, one should learn $\Delta W$, which suggests that one should know how much weight change is in order to initialize finetuning model, and is thus unrealistic.

### Questions
1. In Figure 2b, it kind of suggests that smaller $\xi$ is preferred for quicker convergence? So does it hint that Nystrom initialization benefits from small initializaion as well, which also induces zero saddle point concerns?

2. In NoRA, say we have two pretrained weights $W_0,W_1$ of pretty different magnitudes, does it indicate corresponding NoRA initialization also scales with such magnitudes since $X_\{0,1\}=W_\{0,1\}\Omega$? Probably adding certain weight normalization would make the finetuning more stable?


minor writing issues:
line 340: "modified ScaledGD in (6) have" should be "has"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper studies the convergence of ScaledGD, a special preconditioned GD for matrix factorization (MF) problems, on symmetric/asymmetric MF problems. The authors show that initializing the factorized model as some random sketching of the target matrix significantly achieves faster convergence. Then, the authors propose to initialize the Low-rank adaptation (LoRA) using some random sketching of the pre-trained weights.

### Strengths
Good coverage on matrix factorization problems (symmetric/asymmetric, exact/over/under-parametrization); Writings are clear; Some connection to practical problems.

### Weaknesses
The reviewer has concerns regarding the significance and practical relevance of the theoretical results and the way results are connected to LoRA. Specifically:

1. The matrix factorization problem is not a practical problem to be solved, but rather a problem through which one builds a theoretical understanding of GD and its variants in a nonconvex setting. In this regard, any procedure using the target matrix itself, in the reviewer's opinion, is not allowed: if we know the target matrix, we can directly factorize it (which is why the MF problem is not a practical problem), thus any procedure (other than computing gradient, we are studying GD after all) using some information about the true target matrix is not allowed as it simply moves the factorized model one step closer to the target thus reduce the amount of work GD needs to converge. As such, the crucial role of initialization, as the authors stated, is just taking advantage of knowing the target matrix, which deviates from the original purpose of studying the MF problem: to understand how GD minimizes non-convex objectives even with random initialization. (Disclaimer: This point only pertains to MF problems. For practical problems such as matrix sensing, and matrix completion, the initializations can be special as they are non-trivially improving the convergence of GD)

2. as the reviewer already pointed out in 1., if we know the target matrix, then we can directly factorize it, which is exactly what the one-step modified ScaledGD with Nystrom initialization is doing, so Section 3 is trivial.

3. The connection to LoRA is puzzling in two ways: First, what is proposed is not remotely close to what is theoretically studied, Section 2, 3 studies initialization with the targe matrix sketched. The proposed initialization for LoRA is with the pre-trained weights sketched. Based on this, one should get the conclusion that "pre-trained weights" $\approx$ "targe matrix", which is clearly wrong. (Don't get too serious about this argument, the reviewer is just taking the shortcut to make the point). Moreover, the theoretical claims are special initialization accelerates GD, but none of the empirical experiments are about training time. On top of that, the performance improvement from special initialization seems very little to the reviewer.

### Questions
See "Weaknesses"

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper shows that the low-rank matrix factorization problem can be significantly accelerated using a better initialization method. Specifically, the authors introduce the Nyström initialization, which uses a sketching idea to initialize the low-rank factors to be in the appropriate eigenspaces. Then, using Scaled Gradient Descent (ScaledGD), one can obtain quadratic convergence.

The authors then present Nyström-initialized LoRA (NoRA) and present experimental results that show the practical benefits of using such an initialization compared to using the standard LoRA.

### Strengths
On one hand, I feel that the quadratic convergence with the Nyström initialization is significant. I have not directly worked in the area of matrix completion, so I do not have full confidence in this judgment, but this type of quadratic convergence result seems quite important.

At the same time, however, I'm not quite sure if the quadratic convergence is coming from the scaled GD algorithm or from the Nyström initialization. Since the scaled GD algorithm is basically a Newton update, shouldn't the quadratic convergence happen asymptotically, regardless of the initialization? Is it that the Nyström initialization allows the algorithm to reach the quadratically convergent phase more quickly? Perhaps the authors can clarify this point.

### Weaknesses
On the other hand, I feel that the matrix completion result and the Nyström-initialized LoRA (NoRA) are strongly connected. The theory doesn't apply to the NoRA setting, but I do see that the Nyström initialization is a nice conceptual motivation for a potentially better LoRA initialization.

However, now with the flood of empirical LoRA papers, I think the community has a very high bar for recognizing work that purports to improve LoRA in a practical sense. I am not sure if the experimental validations of this work really clear that bar.

### Questions
.

### Soundness
4

### Presentation
4

### Contribution
4
