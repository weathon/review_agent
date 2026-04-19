# NPEFF: Non-Negative Per-Example Fisher Factorization

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
As deep learning models are deployed in more and more settings, it becomes in-
creasingly important to be able to understand why they produce a given prediction,
but interpretation of these models remains a challenge. In this paper, we introduce
a novel interpretability method called NPEFF that is readily applicable to any
end-to-end differentiable model. It operates on the principle that processing of a
characteristic shared across different examples involves a specific subset of model
parameters. We perform NPEFF by decomposing each example’s Fisher infor-
mation matrix as a non-negative sum of components. These components take the
form of either non-negative vectors or rank-1 positive semi-definite matrices de-
pending on whether we are using diagonal or low-rank Fisher representations, re-
spectively. For the latter form, we introduce a novel and highly scalable algorithm.
We demonstrate that components recovered by NPEFF have interpretable tunings
through experiments on language and vision models. Using unique properties of
NPEFF’s parameter-space representations, we ran extensive experiments to verify
that the connections between directions in parameters space and examples recov-
ered by NPEFF actually reflect the model’s processing. We further demonstrate
NPEFF’s ability to uncover processing strategies actually used by a model by cre-
ating a TRACR-compiled model with known ground truth. We explore a potential
applications of NPEFF in uncovering and correcting flawed heuristics used by a
model. We release our code to faciliate research using NPEFF.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes an interpretability method which allows to obtain representations of concepts. These representations are found by non-negative per-example fisher factorization, which can be done for any end-to-end differentiable model. This method is analogous to non-negative matrix factorization matrices in one of its instantiations and hence decomposes the per-example fisher information matrix into a set of non-negative coefficients and concept vectors, referred to as pseudo-Fishers.
In their experiments, the authors demonstrate the ability of their method to verify what concepts lead to the model's processing. They also show some initial experiments demonstrating how to selectively fix incorrect predictions.

### Strengths
- The authors propose a novel, low-rank decomposition for the per-example fisher.
- The method is applicable to any end-to-end differentiable model.
- The authors did a thorough analysis of the hyperparameters introduced by their method.

### Weaknesses
- While the advantages of the method are well explained, it would be good to add a separate limitations section for transparency. Could you elaborate in your comments on what you see as the biggest limitations.
- I have found the toy model difficult to understand. I think it would be good to provide an intuitive explanation in the main text on the programmatic search.

### Questions
- Could your example about fixing flawed heuristics have applications to the unlearning literature?
- As pointed out by Kunstner et al. 2019, the fisher information matrix is an overloaded object which may or may not refer to the empirical fisher. Since you switch from the true fisher to an approximation which Kunstner et al. 2019, please check section 3.2 of their paper that you followed the correct terminology


Typos / Small errors
- Please review your references, e.g. the lottery ticket hypothesis paper appeared in ICLR 2018.
- The equation equation 5 provides ...


References:
Kunstner, Frederik, Philipp Hennig, and Lukas Balles. "Limitations of the empirical fisher approximation for natural gradient descent." Advances in neural information processing systems 32 (2019).

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a new method for interpreting neural networks called NPEFF. The method applies to classification models and relies on first constructing the Fisher matrix for each example. This matrix can be sparsified by magnitude clipping and also by omitting classes with low probability. Given the Fisher matrices for many examples, the “components” are extracted by approximately factorizing the Fisher matrix in a fashion close to non-negative matrix factorization. The authors also shows how the extracted matrices can be used to relate changes in model parameters to changes in the outputs. Experimentally the authors considers two settings: image classification with Resnets and NLP tasks with a finetuned BERT model. In Figure 1 and 2 the authors visualize some found components, in figure 3 they also show that the model is sensitive in the directions uncovered by NPEFF. Additionally, a comparison with toy data with known ground truth components is done, some ablation experiments are included and a comparison to a previous interpretability strategy is conducted.

### Strengths
Intrepretability is important, especially for LLMs. 

The paper is well written.

### Weaknesses
Interpretability has been studied for a long time, so the novelty is rather low.

The results are only presented for small Bert/Resnet models. It would be better with results for LLMs. 

It is hard to know if the proposed method works well. Interpretability is very subjective, and a few examples (which could be cherry picked) are not very convincing. The authors only compare against a single baseline.

### Questions
Could you compare against more baselines? 

Could you do some kind of human evaluation? E.g. give a sample of 10 people two interpretability models and ask which one they prefer.

Could you give results on LLMs?

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose NPEFF for unsupervisedly discovering the components being used in a learnt model. NPEFF decomposes each example' Fisher information matrix as a non-negative sum of components, so as to discover a set of r components within the network and a coefficient $W_{ij}$ describing the influence of the j-th component on the i-th example's prediction. NPEFF components are examined for two language and one vision dataset, both by viewing the sets of examples strongly associated with each component and by observing the effects of perturbing on components. Lastly, an experiment is performed to discover whether modifying components associated with incorrect predictions will improve predictive performance, resulting in an accuracy gain of 0.5%.

### Strengths
- An unsupervised method for concept discovery that works on general NN would be a useful and significant contribution. While there is some related work as detailed in Section 4, much is focused on vision or does not automatically group features together into sets of discrete concepts (for instance, visualizing CNN convolutions creates human-interpretable patterns as to what the convolutions are picking up, but does not group these patterns together itself)
- The resulting components indeed match to desired, human-recognizable concepts in both the language and vision domains
- Quantitative and qualitative evaluations of the method are both very comprehensive

### Weaknesses
- The experiment on fixing flawed heuristics achieves only a slight improvement of 0.5% accuracy, by improving predictions for 4/48 components associated with incorrect heuristics
- Unclear how the perturbations experiment in Section 3.1.2 demonstrates that NPEFF's discovered parameter space directions are important as stated at the end of the section. Would be nice to see more potential application of NPEFF with strong results
- Minor typo: definition of $a_j(x)$ in section 2.1 should have $y_j$, not $y_i$

### Questions
- In section 2.2's preprocessing section, you state that raw PEFs gave components tuned to outlier examples. Could NPEFF with raw PEFs be useful for some form of OOD detection or prediction confidence measure?
- Could you give more information on how Section 3.1.2 perturbation experiment demonstrates the claims at the end of this section?
- Figure 3: it seems interesting how in QQP and NLI, the PEF norm ratios take a wider range of values than the KL ones, while in ImageNet the opposite is true. Is this meaningful in any way?
- Could you give more details on what constitutes a "concept" and what components resulted in Section 3.2? For instance, did each component tend to focus on one particular concept and components were quite diverse from each other (as likely desirable), or do many components tend to identify with many different concepts so that components are relatively similar to each other?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The current paper focuses on developing an approach to decomposing a fisher information matrix into interpretable components to understand the behavior of a pretrained model. I found the claims and the methodology itself interesting, albeit reminiscent of prior works on importance/saliency estimation of model parameters in network pruning (Molchanov et al., 2017)---the authors already cite some of these papers.

### Strengths
I found the proposed methodology itself interesting, though arguably difficult to implement or work with. The available code can help address this.

### Weaknesses
My biggest apprehension at the moment is that the paper's writeup is extremely unclear at times. While the first half of the papers reads fine and I'm able to follow, the motivation for the experiments, their setup, and the very results themselves are quite unclear to me. For example, the authors claim all "directions" unveiled by their method encodes some "concept" used by Tracr for manually performing the tasks discussed therein. It is entirely unclear though what a "concept" means---is it mere addition, a Tracr primitive, a composition of primitives, etc.? Such lack of clarity generally left me confused throughout the paper, such that I was not certain what the implication of any of the experiments was. This made it hard for me to judge the paper.

Practicality: While the authors conduct experiments on a BERT model, the proposed method requires at least linear in number of parameters memory. I'm uncertain of the scalability of this approach, therefore. Of course several methods suffer from this problem, but it will be worth discussing this in the paper.

**Post rebuttals comment:** I appreciate the authors' comments. Having gone through the updated paper, other reviewers' comments, and the rebuttals to them, I do think the paper has improved. I still found the writing unclear at times (e.g., while I get what "tunings" are, I don't understand why the term is suddenly informally defined and then casually used thereafter, making the paper confusing to me at points). I'm increasing my score to 5 however, given the updates and responses.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
