# Fast Explanation of RBF-Kernel SVM Models Using Activation Patterns

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 5, 6, 5

## Abstract
Machine learning models have significantly enriched the toolbox in the field of neuroimaging analysis. Among them, Support Vector Machines (SVM) have been one of the most popular models for supervised learning, but their use primarily relies on linear SVM models due to their explainability. Kernel SVM models are capable classifiers but more opaque. Recent advances in eXplainable AI (XAI) have developed several feature importance methods to address the explainability problem. However, these explanations can be affected by noise variables which leads to irrelevant variables being regarded as important variables. This problem also appears in explaining linear models, which the linear pattern can address. In this paper, we propose a fast method to explain RBF kernel SVM globally by adopting the notion of a linear pattern in kernel space. Our method can generate global explanations with low computational cost and is less affected by noise variables. We successfully evaluate our method on simulated and real MEG/EEG datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper extends an existing method for explanation of linear models to RBF kernels. The authors show how to extract approximate explanations from this extension by minimising distances in the embedded space. Experiments on simulated and real data indicate that these explanations are substantially better than some existing state-of-the art generic explanation methods.

### Strengths
Interpretation of non-linear SVMs is a important and impactful problem, particularly in the science where the focus may be more on discovery than performance. This method is straight forward, efficient, and seemingly very effective.

### Weaknesses
There is no exploration of how difficult the problem of extracting a pattern through minimising the MSE is. It's stated that the optimisation is run three times, with the mean being used to summarise them. It would be interesting to explore the stability of the iteration more, as this is the critical part of the algorithm. It is also unclear if the mean is the best summary statistic as there may be multiple equally satisfactory explanations possible.

Only neuroimaging data is considered, both for simulations and the real world dataset. It would be interesting to know how this method scales with dimensionality, and if it still remains useful in p >> n scenarios.

### Questions
- Have high dimensional datasets been investigated?
- It would be great to understand more the stability of the MSE optimisation, and if the mean is appropriate.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Needs more work.

The writing is really not presenting the paper under the most favorable light. I think that should this paper be accepted, which I do not recommend, an extensive rewrite is absolutely needed for the sake of the readers and the conference attendees.
There are too many vague statements and too many unclear sentences that make reading the paper too cumbersome.

The subject is narrow, the theoretical contribution is not exceptional, the experimental results are too terse.
In particular, the experiments are not varied enough to validate the method in general.

### Strengths
1) Some experimental results are encouraging, in particular Figure 2C and the fact that the method proposed by the authors is much faster than the presented competitors.

2) The method is relatively simple and well documented in Algorithm 1.

### Weaknesses
1) The impact is too narrow in my opinion and the technique not innovative enough.

2) The experimental results are not diverse enough to really argue that the method is recommendable for general purpose XAI. The authors might want to have a deeper discussion about other kernels.

3) The writing of the paper is too loose which is detrimental to the reader and the conference's audience. E.g. while XAI is a "hot topic" I would not use that language in an academic paper, specifically a paper with a very narrow focus application-wise.

### Questions
1) Could the authors please expand the experimental section to make it more convincing to the general XAI audience?

2) Could the authors please deepen the discussion on other kernels and why they are not treated in the paper?

Thank you.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper concerns explainability in kernel methods, such as SVMs with RBF kernels.
The important problem of correlated noise typically leads to a confounded importance "heatmap".
Haufe et al. (cited in the paper) solved the linear case, while the non-linear case has been looking for good ideas.  Here the authors propose to use the approximate linearity of latent space in kernel methods to de-noise the explanation using Haufe et al.'s method there.

Simulation and benchmark cases are studied. In the simulation study where feature space ground truth is available, the proposed method outperforms relevant baselines. 
In benchmark data where earlier analyses can be used as a proxy for ground truth, performance looks promising

The mathematical and algorithmic work is basic based on a straightforward application of kernel denoising by pre-imaging.

### Strengths
Simple and productive idea.
Straightforward development of an algorithm with some additional minor heuristics (explained in the algorithm 1 box).
Experimental results (including error estimates for the simulation study) supports the proposal.
Computational cost is much less than baseline methods.

### Weaknesses
Much of the math is so basic that a simple reference would have been enough. The gained additional space could have been used to explore the role and significance of the assumptions and heuristics used to stabilize the solution.

The writing is not always so clear - could have benefitted from a writing assistant.

### Questions
Please address the role and significance of the assumptions (linearity of latent space) and heuristics (ensemble averaging). 

Consider running the manuscript with a writing assistant.

### Soundness
3 good

### Presentation
2 fair

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
This paper proposes a new explainer of the kernel SVM under the framework of interpretable machine learning or XAI, which is an active research area. Haufe et al (2014) proposed an explainer for linear models based on a latent variable approach with applications in neuroimaging analysis. The approach is extended to kernel SVM in this paper. As shown in simulations and real applications, the proposed method gives more cogent explanations but also runs faster when compared with the popular explainers such as SHAP and LIME.

### Strengths
The paper is working on a significant problem. The contribution is significant due to the improvement over the state-of-the-art explainers like LIME and SHAP.

### Weaknesses
1. The "Experiment" section requires more clarity. Simply mentioning that data generation utilizes a MATLAB toolbox isn't comprehensive. There's a need for details of the signal and noise pattern creation processes. It's important to highlight that the true data-generating model isn't linear, which underscores the preference for kernel SVM over its linear counterpart.
2. While the emphasis on neuroimaging data is pertinent, its scope appears restricted. The proposed technique seems to be quite general. Could it be naturally integrated with other kernel methodologies, like kernel logistic regression, for instance? The support vectors in SVM might be instrumental in expediting explanation computation; however, this strategy might be equally effective for kernel logistic regression. Furthermore, adapting this method for conventional tabular data could be valuable. In such a scenario, the resulting explanation vector could serve as a metric for variable importance.

### Questions
In addition to the queries outlined in the "Weakness" section, there is another question. Is it true that the proposed method roughly gives the same result with a naive approach which simply gives the mapped data $\phi(x)$, say by eigendecomposition, and directly applies the linear explainer proposed in Haufe et al (2014)? Even if the current proposal operates without the precise mapping of $\phi(x)$, might we obtain a comparable outcome if $\phi$ is discerned through a more rudimentary approach?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
