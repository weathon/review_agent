# Neural Tangent Kernels for Axis-Aligned Tree Ensembles

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
While axis-aligned rules are known to induce an important inductive bias in machine learning models such as typical hard decision tree ensembles, theoretical understanding of the learning behavior is largely unrevealed due to the discrete nature of rules. To address this issue, we impose the axis-aligned constraint on soft trees, which relax the splitting process of decision trees and are trained using a gradient method, and present their Neural Tangent Kernel (NTK) that enables us to analytically describe the training behavior. We study two cases: imposing the axis-aligned constraint throughout the entire training process, or only at the initial state. Moreover, we extend the NTK framework to handle various tree architectures simultaneously, and prove that any axis-aligned non-oblivious tree ensemble can be transformed into an axis-aligned oblivious tree ensemble with the same NTK. 
One can search for suitable tree architecture via Multiple Kernel Learning (MKL), and our numerical experiments show a variety of suitable features depending on the type of constraints, which supports not only the theoretical but also the practical impact of the axis-aligned constraint in tree ensemble learning.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper builds upon the theoretical analysis framework introduced by Kanoh & Sugiyama (2022; 2023) to investigate the learning behavior of soft tree ensembles. It focuses on deriving closed-form solutions for the Neural Tangent Kernel (NTK) induced by infinitely many axis-aligned tree ensembles, considering two scenarios: always axis-aligned (AAA) and axis-aligned at initialization (AAI). Additionally, the paper extends the NTK framework to accommodate multiple tree architectures and demonstrates that any non-oblivious axis-aligned tree ensemble can be transformed into an axis-aligned oblivious tree ensemble while preserving the same NTK. The paper also explores the potential applications of multiple kernel learning (MKL) in identifying suitable tree architectures and features under the axis-aligned constraint. Empirical results are provided to validate the theoretical findings and illustrate the practical significance of the axis-aligned constraint in tree ensemble learning.

### Strengths
1. The paper is original in extending the NTK framework to the axis-aligned soft tree ensembles, which have not been theoretically analyzed before.
    
2. The paper is of high quality in deriving the closed form solution of the NTK for both AAA and AAI cases, and proving the equivalence between axis-aligned non-oblivious and oblivious tree ensembles.
    
3. The paper is clear in presenting its main results and providing intuitive explanations and illustrations for its theoretical findings.

### Weaknesses
1. In Section 3.1, if I interpret it correctly, AAA refers to axis-aligned initialization, with the assumption that the selected split feature at each node remains constant throughout training. This assumption appears rather restrictive, considering that typical axis-aligned trees do not impose such constraints. It would be valuable if the author provided further insights or comments regarding this assumption.
    
2. The paper lacks an exploration of the computational complexity and scalability issues related to employing Multiple Kernel Learning (MKL) for tree architecture search. It is crucial to assess the feasibility and efficiency of utilizing MKL, particularly for large-scale datasets and high-dimensional feature spaces.

### Questions
Please find my main concerns in the weakness part. Additionally, I have another question regarding the analysis framework: does this analysis framework allow non-zero initialized parameters (selected split features) become zero (no split at one node) during the training? If the framework allows for such adaptability, it could potentially accommodate changes in the shapes or depths of trees during training, thus enhancing its overall generality.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a theoretical analysis of the Neural Tangent Kernel for axis aligned decision trees. Experiments are limited to a single dataset.

### Strengths
This is some work towards the goal of better understanding how to optimally train axis aligned decision trees.

### Weaknesses
- The paper is theoretical in nature, but its impact for practical applications seems very limited to me.
- Experimental validation is limited to one dataset. This raises concerns about the practical use of this work.
- The comparison with Random Forest in unfair because RF is limited to depth 3. It is not clear how many attributes were used for splittiong at each node for RF. If the number of attributes is small, RF needs deep trees to find the relevant features.
- There is no comparison with existing modern ensembling techniques such as Gradient Boost.

### Questions
- What is the difference between this paper and Kanoh and Sugiyama. "Investigating Axis-Aligned Differentiable Trees through Neural Tangent Kernels". In ICML 2023 Workshop on Differentiable Almost Everything: Differentiable Relaxations, Algorithms, Operators, and Simulators?
- What are the practical implications of knowing the NTK induced by axis-aligned tree ensembles? Does this imply that we can obtain better/faster training algorithms?

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
[I made a mistake in the form.] I found out that I have accidentally checked the "First Time Reviewer" question, but in fact, I'm not. It seems that I cannot undo it now, so I'm instead writing it here.

This paper proposes a way to analyze the training dynamics of axis-aligned tree ensembles using neural tangent kernels (NTK). The idea is two-fold: (1) using soft trees and assigning proper weights to derive NTK, and (2) using multiple kernel learning (MKL) for finding suitable tree structures. The paper also shows that, from any ensemble of axis-aligned trees, one can find that of axis-aligned oblivious trees with the same limiting NTK, which justifies the use of oblivious trees.

### Strengths
- The paper cleverly derives a way to analyze the training dynamics of axis-aligned trees using soft trees and NTK. The idea here is to deliberately assign weights in NTK so that it represents the behavior of axis-aligned trees, which is interesting.

- The online feature selection in actual tree construction is modeled using MKL, which is also interesting.

- The paper also provides a justification regarding the use of oblivious trees based on the above framework.

### Weaknesses
- I did not find any weaknesses.

### Questions
- The paper argues that one of its contributions is including finite tree ensemble scenarios. However, in Section 4, the proposition is only on infinite trees. Why is this?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors derive a Neural Tangent Kernel for axis-aligned trees, and show several extensions such as non-oblivious trees and multiple different architectures. They also run a numerical experiment on the tic-tac-toe dataset and show the empirical utility of such methods.

### Strengths
The authors build on a few prior papers, Kanoh and Sugiyama 2022, 2023, which establish NTK on tree ensembles.  Though I did not check the proofs carefully, the theory seems sound. Prior work on NTK for tree ensembles seemed limited to all trees having the same architecture, and each tree being oblivious, and the authors extended the theory into those regimes.

### Weaknesses
1. I do not really understand why we would *want* NTK for tree ensembles, or more specifically axis-aligned trees.  Typically, I look to theory because it can provide insight or intuition about why some method is working well.  I did not get any of that here.  What does this theory explain about axis-aligned trees that we did not already know? 

2. There are several papers relating trees/forests to kernels.  The mondrian kernel is probably the most famous, https://dl.acm.org/doi/10.5555/2969033.2969177, though Breiman wrote about it over 20 years ago prior to his random forest paper, https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.24.7078&rep=rep1&type=pdf. And we https://arxiv.org/abs/1812.00029, and others, have some work on it as well, https://ieeexplore.ieee.org/document/7373647.  Those kernels are exact, whereas this kernel is an approximation.  Given that trees/forests directly induce a kernel, I do not understand why we would want to derive an approximate kernel?

3. The primary results seem to depend on all the trees having the same architecture, and also all the features being somehow already selected?  I do not understand where the architecture or selected features come from?

4.  The results also all seem to depend on soft, rather than hard, trees. Is that because the math is easier for soft trees? A few sentences about that in the discussion would be helpful context. I do not know of any tree packages that leverage soft trees, so if people actually use them in practice (I know the papers on Neural Forests, but I do not know whether they are actually used anywhere), that would be helpful context as well.  If nobody uses soft trees, that's ok, it just a limitation, and future work might be about getting similar results on hard trees, unless it is obviously (to you) not tractable. 

5. The empirical results are on a single dataset: tic-tac-toe.  It is a nice illustration.  I wonder, however, why this dataset was chosen specifically? Was it cherry-picked to have good results? Or was it because all the features are binary, and that helps for some reason? Or because depth 3 trees work well (at least the AAA/AAI ones)?  

6.  For me, the fact that AAA works better than RF for certain alpha's is by far the most interesting result.  What features of the distribution is the axis-aligned NTK capturing that the RF fails to acquire? I am guessing the fact that they are depth 3 trees has something to do with it, because the RF can only handle 3 feature splits per path, and more are required to achieve Bayes optimal.  How does the NTK get around this issue, what is happening?  For me, this is by far the most interesting result, and I did not understand or see any text attempting to explain it.

### Questions
My main question is why/when can axis-aligned NTKs outperform axis-aligned forests?  What information can they leverage that is missed by the forests? What is the inductive bias of the NTK relative to the axis-aligned forests.  While the theory, on its own, is fine, I do not find it compelling on its own.  The arbitrarily slow convergence theorem (https://link.springer.com/article/10.1007/BF00534199) implies that any given approach will outperform another with finite data.  So, from that perspective, the point of any paper describing a new approach is to provide insight into when/why it outperforms other approaches.  Curves plotting performance vs sample size, dimensionality, or various simulation parameters can all provide insight into this issue.  If the authors can provide clean compelling explanations about when/why their NTK would/does outperform RF or other kernel forest approaches, I think it would be very interesting.  Without that, however, I am just not that interested.

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
3 good
