# LightSAM: Parameter-Agnostic Sharpness-Aware Minimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 6, 5

## Abstract
Sharpness-Aware Minimization (SAM) optimizer enhances the generalization ability of the machine learning model by exploring the flat minima landscape through weight perturbations. Despite its empirical success, SAM introduces an additional hyper-parameter, the perturbation radius, which causes the sensitivity of SAM to it. Moreover, it has been proved that the perturbation radius and learning rate of SAM are constrained by problem-dependent parameters to guarantee convergence. These limitations indicate the requirement of parameter-tuning in practical applications. In this paper, we propose the algorithm LightSAM which sets the perturbation radius and learning rate of SAM adaptively, thus extending the application scope of SAM. LightSAM employs three popular adaptive optimizers, including AdaGrad-Norm, AdaGrad and Adam, to replace the SGD optimizer for weight perturbation and model updating, reducing sensitivity to parameters. Theoretical results show that under weak assumptions, LightSAM could converge ideally with any choices of perturbation radius and learning rate, thus achieving parameter-agnostic. We conduct preliminary experiments on several deep learning tasks, which together with the theoretical findings validate the the effectiveness of LightSAM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes LightSAM, a new variant of the Sharpness-Aware Minimization (SAM) optimizer that incorporates three adaptive optimization techniques (AdaGrad-Norm, AdaGrad, and Adam). The key innovation is using these adaptive methods for both the perturbation radius and learning rate in SAM, eliminating the need for parameter tuning. The main theoretical contribution is proving $O(log T/\sqrt{T})$ convergence for LightSAM under weaker assumptions than prior work, notably removing the gradient boundedness requirement and achieving parameter-agnostic properties. Empirically, the authors evaluate LightSAM on image classification tasks using MNIST and ImageNet datasets. The experiments demonstrate that LightSAM achieves comparable or better performance than carefully tuned SAM while being insensitive to parameter choices, validating its parameter-agnostic properties.

### Strengths
- Algorithm that makes SAM parameter-agnostic by removing hyperparameter restrictions on convergence
- Provides theoretical analysis with $O(\log T/ \sqrt{T})$ convergence guarantees under weaker assumptions
- Demonstrates consistent improvements over AdaSAM across different datasets

### Weaknesses
- My main issue is the mismatch between theory and empirics: the theoretical results focus on optimization convergence for differentiable loss landscapes, while the empirical results demonstrate generalization performance. This disconnect makes it challenging to assess the practical relevance of the theoretical contribution. 

- Limited discussion of computational overhead, especially the extra overhead that comes from accumulating historical gradients, how long does it take in the experiments? Any quantitative comparison with baseline methods?

- The convergence guarantee doesn't offer new insights into generalization properties. (At least doesn't explain it well in the paper.)

### Questions
- Why the analysis is free to the parameter $\rho$? In Theorem 1, we can see that $A_1$ is related to the term $\eta^{2} -2\rho^{2}\log(\epsilon)$ which would be negative if ρ is too large. Better add some explanation.

- While the perturbation radius $\rho$ does not affect the convergence result, how does it influence the model's generalization ability in this paper? Any empirical or theoretical insights provided?

- In line 228, the authors mentioned "Technical Challenge", but what's the key technique to address this challenge? This needs to be explained.

- In Section 3.3, the authors provide the analysis of LightSAM-II (AdaGrad), but without explanation. How to understand these theorems? What insights do they provide?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents LightSAM, a parameter-agnostic variant of the Sharpness-Aware Minimization (SAM) optimizer, designed to enhance generalization by reducing sensitivity to hyperparameters. 
LightSAM replaces the SGD optimizer in SAM with adaptive optimizers (AdaGrad-Norm, AdaGrad, and Adam) for both weight perturbation and model updating steps, aiming to achieve convergence without the need for parameter tuning. 
The theoretical analysis suggests that LightSAM converges with a rate of $\mathcal{O}\left(\ln T/\sqrt{T}\right)$ under relaxed assumptions, and empirical evaluations on MNIST and ImageNet support its effectiveness.

### Strengths
- LightSAM effectively addresses a main limitation of SAM by making it parameter-agnostic, potentially simplifying hyperparameter tuning in practical applications.
- The empirical results are promising. LightSAM’s performance on MNIST and ImageNet, particularly its insensitivity to hyperparameters, demonstrates its practical potential and competitive accuracy.

### Weaknesses
I have a major concern regarding the soundness of the proof. I think that the authors ignore the correlation of $w_t$ and $\xi_t$. Then, two problems arise. First, due to this correlation, the equality such as Line 836-837 does not hold. Second, when applying $w_t$ on affine variance noise, the authors use a form like $\mathbb{E}\\|\nabla f(w\_t,\xi\_t)\\|^2 \le D_0+D_1 \mathbb{E}\\|\nabla f(w\_t)\\|^2$ which is not the form of Assumption 2 as there is no expectation on RHS of Assumption 2. I think that $\\|\nabla f(w\_t,\xi\_t)\\|$ could not be directly bounded through Assumption 2.

Besides, I think that it's better to use notation like $\mathbb{E}_t$ to denote the conditional expectation. It's confusing for readers to see $\mathbb{E}$ denotes both the conditional expectation and the full expectation

I have another major concern regarding the motivation of the algorithm. The main motivation in this paper comes from incorporating the adaptive perturbation radius to eliminate the tuned parameter. In comparison to the SAM, the major difference comes from using $\\|u\_t\\|$, the cumulative sum of the gradient norm instead of the gradient norm $\\|g_t\\|$ in the perturbation radius. However, I think that both two ways could be regarded as adaptive perturbation radiuses since the radius $\rho\_1 = \rho/ \\|u\_t\\| $ and $\rho\_2 = \rho/\\|g_t\\|$ are all determined by the gradient information. So I am not sure whether it's necessary to incorporate the $\\|u\_t\\|$, or are there any essential differences between the two $\rho$? The author could claim more on this, maybe showing that the original SAM must tune the perturbation radius unless diverging.

There are some other concerns as follows.

- The proof methodology in this paper bears a notable resemblance to that in the referenced AdaGrad convergence paper [Wang et al., 2023]. 
	A more in-depth discussion on how the theoretical contributions in this work differ fundamentally from existing proofs could better clarify the unique theoretical foundation of LightSAM and highlight its distinct contributions.
- The paper could benefit from a comparative discussion on computational complexity between LightSAM and other SAM-related optimizers. As computational overhead is a key consideration for optimizers, especially in large-scale models, providing this comparison would give a more complete view of LightSAM’s practical costs.
- While MNIST and ImageNet are standard datasets, they may not fully demonstrate LightSAM’s potential across various complex, real-world tasks. 
	Including additional experiments on tasks beyond classification, such as natural language processing or reinforcement learning, could provide a more comprehensive evaluation.

### Questions
- There are some typographical errors in the proofs that affect readability and accuracy. For instance, in Equation (22) of the proof for Theorem 4, $\frac{1}{\sqrt{v_{t-1}}}$ is placed outside the summation symbol, which might lead to misunderstandings. 
	Addressing these would enhance the overall precision and presentation quality of the paper.
- The uniform size of parentheses throughout the proofs makes it challenging to follow nested expressions, which affects readability. 
	Adjusting parentheses sizes to reflect the level of nesting could improve clarity and help readers follow the steps more comfortably.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work studies the SAM algorithm with  3 adaptive optimizers and provide theoretical analyses, showing its agnostic properties on the convergence results w.r.t. the perturbation. The perspective is interesting, but lacks sufficient explanations and numerical evidence. I would like to raise my scores, if my concerns are thoroughly addressed both analytically and numerically, preferably.

### Strengths
The convergence results being agnostic to parameter values, i.e., the perturbation radius, are interesting to the field.

### Weaknesses
The benefits of the so-called Light SAM are not sufficiently presented in the experiments. The reasons of parameter-agnostic property in convergences are lacking, so are the extension to other optimizers aside of the presented 3.

### Questions
Questions & some advice:

1. The main contribution of this work to me lies in the parameter-agnostic convergence results of SAM adopting to the adaptive optimizers presented in this work. However, in the experiments, such benefits are not elaborated. Firstly, the shown results only appear with incremental and quite marginal improvement; secondly, why not vary some parameters, e.g., $\rho$, to compare the results on acc and especially on convergence as claimed by the theorems; thirdly, more ablation studies should done for each of the proposed 3 versions.  E.g.,

- can it be understood that the proposed method only gives marginal improvements in table 4 and 5?

- the convergence plots for a range of $\rho$ or $\eta$ values can be helpful to provide a more comprehensive ablation study;

- compare the three proposed versions with other optimizers across different hyperparameter settings, rather than a single setup of hyperparameters in table 5; otherwise, how the hyperparameters are determined in comparisons.

2. Can other optimizers also get such parameter agnostic properties, aside of the presented 3? What’s essential characteristics of those optimizers contributing to such agnostic properties in convergence?  

-  discuss the key properties of the three optimizers that enable parameter-agnostic convergence, and whether these properties might extend to other optimizers.

3. Some minor aspects:  (a) why name it as LightSAM, as there are more efficient SAM-related algorithms [1][2] (not limited to). Please justify the choice of name.  (b) I would suggest specify the footnote 1 in a earlier place, e.g., the abstract, or maybe reconsider the title with more focus on the convergence perspective, because the agnostic property lies in the convergence analysis rather than the hyperparameter tuning, if possible.  

 [1] Revisiting Random Weight Perturbation for Efficiently Improving Generalization, TMLR, 2024. 

[2] Friendly sharpness-aware minimization, CVPR, 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a variant of the Sharpness-Aware Minimization (SAM) optimizer designed to adjust both the learning rate and perturbation radius adaptively for SAM. The proposed method, LightSAM, addresses this by integrating adaptive optimizers, including AdaGrad-Norm, AdaGrad, and Adam, to automatically adjust these parameters without problem-dependent tuning, making it parameter-agnostic. The authors provide theoretical insights demonstrating LightSAM’s convergence rate, $O(\ln T/ \sqrt{T})$.  This paper also provides empirical validation through experiments on MNIST and Imagenet datasets, using LeNet and Vision Transformers. The results show that LightSAM achieves competitive performance to existing SAM-based methods, while reducing the dependency on hyper-parameter tuning.

### Strengths
- The paper develops a theoretical analysis of LightSAM’s convergence properties under adaptive learning rates and minimal assumptions, showing that it achieves a convergence rate of $O(\ln T/ \sqrt{T})$

- The experiments on diverse datasets, including MNIST and Imagenet, demonstrate LightSAM’s robustness and parameter-agnostic nature. 

- The paper includes comprehensive comparisons with SAM, AdaSAM, and other optimizers, showing that LightSAM achieves comparable accuracy while reducing tuning complexity.

### Weaknesses
It is unclear how the hyper-parameter tuning complexity should be evaluated in the setting. While the paper has shown that under nine hyper-parameter settings, the LightSAM's standard deviation is small, it would be better to explain the protocol. Moreover, It would be better to conduct thorough studies on the hyper-parameter tuning (for example, under a larger range of hyper-parameters, the results are similar), and more experimental settings (for other types of tasks).

### Questions
- From the experiments, the proposed algorithm performs on par with the AdaSam baseline, while the AdaSam baseline is claimed to have a better convergence rate. Can the authors extend the discussion with AdaSam and explain the advantages of the proposed method? 

- How should one decide the initial parameter of the LightSAM?

### Soundness
2

### Presentation
3

### Contribution
2
