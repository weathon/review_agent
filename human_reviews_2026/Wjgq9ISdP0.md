# Unlocking Out-of-Distribution Generalization in Transformers via Latent Space Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Systematic, compositional generalization beyond the training distribution remains a core challenge in machine learning---and a critical bottleneck for the emergent reasoning abilities of modern language models. This work investigates out-of-distribution (OOD) generalization in Transformer networks using a GSM8K-style modular arithmetic on computational graphs task as a testbed. We introduce and explore a set of four architectural mechanisms aimed at enhancing OOD algorithmic generalization: (i) input-adaptive recurrence; (ii) algorithmic supervision; (iii) anchored latent representations via a discrete bottleneck; and (iv) an explicit error-correction mechanism. Collectively, these mechanisms yield an architectural approach for native and scalable latent space reasoning in Transformer networks with robust algorithmic generalization capabilities. We complement these empirical results with a detailed mechanistic interpretability analysis that reveals how these mechanisms give rise to robust OOD generalization abilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies Transformer length generalization on a modular arithmetic on computational graph task. The authors show that while some degree of length generalization is possible using standard CoT training, four architectural changes: (i) recurrence; (ii) algorithmic supervision; (iii) anchored latent representations (iv) error-correction enable length generalization far beyond the training length. They also provide a mechanistic interpretability analysis showing how the required (length-generalizing) algorithm can be implemented with two attention layers and an MLP, performing copy, induction head, and modular addition roles.

### Strengths
The paper is clear and well-written, the problem is interesting and important, and the four architectural mechanisms are well motivated and yield an impressive improvement in length generalization on the chosen task.

### Weaknesses
My main concern is that only a single task, namely modular arithmetic on computational graph, is studied in this paper. It is unclear whether the four architectural mechanisms can yield similar improvements on other tasks. I appreciate that this task is somewhat representative of a fairly broad class of compositional problems, but I would be more convinced if improvement in length generalization after applying the four changes could be shown on other tasks as well. The architectural changes seem somewhat tailored to the specifics of the task. In order to broadly useful, the changes would also need to be helpful for a larger class of problems.

The mechanistic interpretability study is similarly limited to the single specific task, and it is not fully clear to what extent it can give insight into other types of problems/tasks. Also the mechanisms identified are well known (which is fine, just not a major source of novelty in the paper).

### Questions
Please see Weaknesses. In particular, can the authors comment on whether the four architectural changes are likely to be helpful for other tasks beyond modular arithmetic on computational graph? Or, what kinds of alternative architectural changes might be required for different types of algorithms tasks?

Also, could the authors please comment on modular arithmetic on computational graph as a representative of compositional problems more generally? Is there a sense in which a large and important class of problems is essentially equivalent to this task? In particular, is there a broader class of problem for which we can be confident that the four architectural changes will yield improvements in length-generalization?

Minor:
Fig 5: var1 appears twice, was is supposed to be var2?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work analyses the OOD length generalization of feed-forward and latent recurrent Transformer architectures on a modular arithmetic task, where a model is trained on computational graphs of size up to $L=32$ and tested on up to $L=128$. After observing the failure of feed-forward Transformers (both in direct and CoT configurations), the authors investigate/propose ingredients that enable length generalization in this task: latent recurrence, step-level recurrence supervision, discretized latent representations, and a latent noise injection to introduce error-correction mechanisms. Combining all the ingredients indeed leads to length generalization. Finally, the inner workings of the trained recurrent latent Transformer are mechanistically analyzed on the attention level.

### Strengths
1.	The paper is very well written and motivated. Algorithmic generalization is an open and important challenge in AI; approaching the problem in a principled way with a controllable dataset that does not impose any secondary effects (as in natural language) is commendable.
2.	The paper indeed goes to incredible lengths to make the performance of individual architectures hyperparameter invariant. Indeed, hyperparameters of every architecture are individually optimized, and the extensive Appendix describes all ablations. The rigor in hyperparameter search is highly appreciated.

### Weaknesses
My main concern is the missing connection to related work. The paper is written like starting from a blank sheet of paper. While this makes the reading very coherent and easy to follow, it makes the distinction to already known insights from the literature difficult. Specifically, I would appreciate the following connections to related works:

1.	Missing connection to arithmetic tasks testing length generalization. There is an active area of research on evaluating the expressiveness of neural networks on algorithmic tasks [Dziri et al. 2023, https://arxiv.org/pdf/2305.18654 ] [Thomm et al. 2024, https://arxiv.org/abs/2402.05785 ], specifically targeting length generalization too. Having a comparison to some of the tasks, would be appreciated. It is crucial to situate the proposed task within existing benchmarks and compare the task's difficulty and characteristics against well-known algorithmic generalization tasks.
2.	Missing connection to results reported in the literature. As already mentioned in the strengths, the paper creates reasonable baselines for their task. Yet, it could still be that some techniques had been missed [Csordás et al. 2022, https://arxiv.org/pdf/2108.12284 ]. One way to have a more credible comparison is to use (or better reproduce) results from the literature. Given that there already exist tasks on algorithmic length generalization (see Weakness 1), one could use these tasks and baseline performance as an anchor point. To strengthen the claim that the current task is uniquely challenging or that the proposed architecture is necessary, the authors should reproduce and report the performance of the standard feed-forward Transformer on a canonical, related task to establish a clear anchor and ground the observed failures.
3.	Missing connection to findings on recurrent neural networks. I find it quite hard to categorize the insights gained in this work. The main issue lies in that many of the techniques have already been proposed and validated in previous works (though in different tasks): recurrence [35] [Abnar et al. 2023, https://arxiv.org/pdf/2310.08866 ], or discretization [Wang et al. 2025, https://arxiv.org/abs/2506.21734 ]. Hence, the main contribution of the paper can be read as finding the necessary ingredients to enable OOD length generalization on a single task. I am convinced that the findings can translate to other algorithmic tasks; it just would have to be demonstrated. The major issue is the task-specific nature of the findings. I strongly encourage the authors to demonstrate the generalizability of the necessary ingredients (recurrence + supervision + discretization) by applying the full successful model to at least one other standard OOD length generalization task from the literature.

### Questions
Besides answering the main weaknesses (W1-W3), I have the following minor questions:

1. Lines 229-230 state that the recurrent iterations (T) are adapted to input complexity. Is this done manually, or is there an automated mechanism?
2. Appendix C.1 describes a task-specific latent state embedding structure. Could this structure also be applied to the feed-forward or CoT approach?

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
This work studies transformer's out-of-distribution generalization on problems longer and more complex than ones trained on. The authors designed a flexible graph-based, arithmetic task domain, and explored how several architectural modifications including recurrence, latent representation supervision and discretization, as well as error correction contribute to an improvement on OOD performance in this task domain better than CoT training. The behavioral gains are complemented by interpretability analysis to reveal how the solution is implemented in the model.

### Strengths
- This work provides a detailed study of transformers generalization capabilities on a target task domain. The transformer variant with additional mechanisms achieve pretty strong generalization to harder tasks than seen during training.
- The addition of the interpretability analysis is a plus to complement the performance evaluations.
- The presentation is quite clear.

### Weaknesses
My main concern is the generality of the results. While the work showed strong OOD performance on this particular task, it's unclear whether the same architectural mechanisms easily transfer to other tasks or improve the overall OOD generalization capabilities in transformers. Specifically:
- Many proposed mechanisms such as per-layer supervision and discretization seem to strongly rely on oracle knowledge of the particular task. This would restrict application to tasks without clearly defined "steps" that progress towards a solution.
- The recurrence is described as adaptive, but it's unclear if there is a learned exit strategy by the model, or if the # of iterations remains a hyperparameter.

I think the paper would be greatly enhanced if the authors could experiment or at least discuss how these methods may extend to additional task domains.

### Questions
- Could you clarify if the recurrence adaptation is learned or if the length of iteration is arbitrarily specified? If not learned, have you considered exploring a version where the exit strategy is learned?
- Do you think the proposed model variant would hinge the learning of general language modeling capabilities?

### Soundness
3

### Presentation
4

### Contribution
2
