# FCLoRA: Low-Rank Adaptation with Fine-Grained Component Injection

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
In recent years, low-rank adaptation (LoRA) has emerged as a significant paradigm, which freezes the pre-trained weights and introduces small, learnable adapters instead of fine-tuning the full set of parameters. In this work, we uncover several key insights regarding to the $\textit{singular}$ components of the network parameters based on Singular Value Decomposition (SVD). Firstly, the dominant singular components with large singular values in pre-trained network parameters can be effectively reused during fine-tuning, whereas the fine-grained components with smaller singular values are more task-specific and require substantial adaptation. Secondly, the growth of singular values in the LoRA adapter leads to the forgetting of pre-trained knowledge $-$ a well-known issue called $\textit{catastrophic forgetting}$. Building upon these observations, we propose $\textbf{FCLoRA}$, which injects learnable fine-grained singular components to the pre-trained model. By employing parameterized SVD and restricting the singular values to an appropriate range, $\textbf{FCLoRA}$ can effectively adapt to new tasks by learning in the fine-grained singular domain and alleviates the catastrophic forgetting problem. We conduct extensive experiments and demonstrate that $\textbf{FCLoRA}$ not only improves performance but also effectively retains pre-trained knowledge.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes FCLoRA, a parameter-efficient fine-tuning framework that enhances LoRA by injecting fine-grained singular components into pretrained models using a parameterized SVD formulation.  The authors analyze how different singular value ranges in pretrained weights correspond to transferable versus task-specific knowledge.

### Strengths
1. The paper provides a principled analysis of the relationship between singular value spectra, task transferability, and forgetting, offering new geometric insight into PEFT.
2. The parameterized SVD with bounded singular values is simple yet effective, easily integrable into existing LoRA pipelines.
3. Directly mitigates catastrophic forgetting — a well-known but underexplored problem in PEFT research.

### Weaknesses
1. I found that the conclusions on the principle subspace are somehow contradict to those in PISSA. Can the author explain the reason?
2. The paper introduces an upper bound on singular values to prevent catastrophic forgetting. However, LoRA’s empirical success partially relies on allowing singular value amplification for better expressivity. How do you justify that a static bound does not suppress task-specific adaptation capacity, especially for highly anisotropic pretrained representations?
3. How does FCLoRA differ in principle from AdaLoRA or PiSSA, which also modulate the singular spectrum during training? Is the key contribution the explicit upper bound, or does the fine-grained injection scheme lead to qualitatively different subspace evolution?
4. Since SVD is required for each adapter, how does the computational overhead scale with model size? Are there efficient approximations to make FCLoRA practical for industrial-scale models?
5. Could the authors provide quantitative evidence that this stabilization directly correlates with reduced forgetting, beyond task accuracy—e.g., via spectral entropy or principal angle evolution?

### Questions
Please see weaknesses

### Soundness
2

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
This paper proposes FCLoRA, a modification of the Low-Rank Adaptation (LoRA) method for parameter-efficient fine-tuning. The motivation stems from two main observations derived from Singular Value Decomposition (SVD) analysis of network parameters: 1) Dominant singular components (large singular values) of pre-trained weights are reusable, while fine-grained components (small singular values) are more task-specific and need adaptation. 2) The growth of singular values in the LoRA adapter ($\Delta W$) during fine-tuning can lead to catastrophic forgetting of pre-trained knowledge, linked theoretically to a decrease in the prior probability term in MAP estimation ($p (\theta|\mathcal{D}_A) $). FCLoRA addresses these by injecting a learnable low-rank update $\Delta W$ formulated using parameterized SVD ($\Delta W = U\Sigma V^\top$). Critically, it restricts the learnable singular values $\sigma_n \in \Sigma$ to be within an "appropriate range" by clipping them below a pre-defined upper bound $\bar{\sigma}$ (e.g., a quantile $\sigma^{(q)}$ of $W_0$'s singular values). This aims to focus learning on the fine-grained domain and prevent excessive singular value growth, thereby improving adaptation while mitigating forgetting. Experiments on NLU, QA, and commonsense reasoning tasks are presented.

### Strengths
1. The paper provides a reasonable motivation based on SVD analysis, distinguishing the roles of dominant vs. fine-grained components and linking singular value growth in adapters to catastrophic forgetting via a theoretical argument (Theorem 3.1). Figure 1 visually supports these claims.

2. The core idea of using parameterized SVD and explicitly constraining the magnitude of learnable singular values ($\Sigma$) with an upper bound $\bar{\sigma}$ is a direct and intuitive way to implement the paper's motivation of focusing on "fine-grained" adaptation and preventing excessive deviation from the pre-trained state.

### Weaknesses
1. Limited Novelty and Distinction from Prior Work: The use of parameterized SVD in LoRA variants is not new (e.g., AdaLoRA). Several recent works also leverage SVD components for LoRA initialization or modification, such as PISSA, MiLoRA, LoRA-XS, and SORSA. FCLoRA's specific contribution lies in clipping the learnable singular values $\Sigma$ to an upper bound $\bar{\sigma}$. The paper fails to sufficiently articulate why this specific mechanism is significantly novel or superior compared to these closely related methods (e.g., MiLoRA which also focuses on smaller components, or AdaLoRA which prunes components). Experimental comparisons against PISSA and MiLoRA are notably absent despite their relevance.


2. Crucial Hyperparameter $\bar{\sigma}$ Underexplored: The choice of the upper bound $\bar{\sigma}$ is critical to FCLoRA's definition of "fine-grained" injection. How should $\bar{\sigma}$ be chosen? The paper suggests using a quantile $\sigma^{(q)}$ of $W_0$'s singular values but provides little guidance or analysis on which quantile works best, how sensitive the results are to this choice, or whether it needs tuning per task/model. The ablation study in Appendix O.2 shows sensitivity but lacks a principled selection method. This ambiguity makes the method difficult to apply reliably.

3. Insufficient Validation of Forgetting Mitigation: The primary claim regarding catastrophic forgetting mitigation relies heavily on Theorem 3.1 (based on Laplace approximation) and experiments measuring performance degradation on the pre-training task after single-task fine-tuning (Section 5.3, Figure 3). While indicative, this does not adequately simulate or evaluate performance in a proper continual learning setting with a sequence of distinct downstream tasks. Standard CL benchmarks and metrics (like backward transfer or average accuracy over a sequence) are needed to robustly validate the claims about mitigating forgetting in practice.

4. Computational Aspects: Parameterized SVD requires learning $U, \Sigma, V$ and potentially enforcing orthogonality via regularization $R(U,V)$ with coefficient $\gamma$. How does the computational cost and optimization stability compare to standard LoRA (learning $A, B$)? The paper claims negligible overhead but provides limited analysis (Appendix P shows increased training/inference time).

### Questions
1. Can the authors clearly differentiate FCLoRA's contribution from related works like PISSA, MiLoRA, AdaLoRA, and SORSA, both conceptually and empirically (e.g., through direct experimental comparisons)?

2. How should the crucial hyperparameter $\bar{\sigma}$ be set in practice? Can the authors provide guidelines or further sensitivity analysis across different models and task types? Is the quantile approach generally applicable?

3. To substantiate the claim of mitigating catastrophic forgetting, could the authors evaluate FCLoRA on standard continual learning benchmarks involving task sequences and report metrics like average accuracy and backward transfer, comparing against relevant CL baselines?

4. Can the authors provide more details on the practical training stability and convergence speed of learning parameterized SVD components ($U, \Sigma, V$ with orthogonality constraints) compared to standard LoRA ($A, B$)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes FCLoRA, a LoRA variant that uses a parameterized SVD of the adapter and focuses the learnable part on fine-grained singular components, while keeping dominant components within a bounded range. The motivation has two parts. First, an empirical SVD analysis suggests that large singular components of pre-trained weights can often be reused across downstream tasks, while smaller components are more task-specific. Second, during fine-tuning, the singular values of a LoRA adapter can grow without control, and the authors link this growth to catastrophic forgetting via a MAP style argument. FCLoRA injects a low rank update through SVD and clips the learnable singular values using a quantile of the spectrum of the pre-trained weight. The goal is to let the model adapt to new tasks without erasing the pre-trained prior. Experiments on NLU, QA and several reasoning tasks show small but mostly consistent gains over common LoRA baselines.

### Strengths
I like the attempt to tell a fully spectral story. The paper explains that dominant components can be reused, that fine grained components need to be adapted, and that uncontrolled growth of singular values weakens the pre-trained prior. This is intuitive and the figures support it. The method itself is conceptually light, it is just LoRA with an SVD front and a clipping rule for the singular values. The experiments cover several models and tasks, which shows that the idea is at least practical.

### Weaknesses
The main problem is limited novelty compared with recent SVD based LoRA work. Learning U, Sigma and V for LoRA, or choosing and bounding the singular values, is already present in several papers. This work does not fully isolate what FCLoRA adds beyond saying that the goal is to mitigate forgetting.

A second problem is that the crucial hyperparameter, the quantile of the pre-trained spectrum that defines the upper bound on Sigma, is under explained. The paper simply says to use a quantile of the singular values, but it does not show how sensitive the results are to this choice, whether different models or tasks need different quantiles, or whether per layer and global quantiles behave differently. At the moment this looks like a hand tuned knob.

A third problem is that the evidence for forgetting is weak. The central story of the paper is catastrophic forgetting, but the experiments mainly check how much performance on the pre-training task is lost after fine-tuning a single task. This is not the same as showing mitigation on a real continual learning sequence with several distinct downstream tasks and with standard CL metrics such as average accuracy and backward transfer. Without this, it is hard to give credit for the claimed mitigation.

Finally, the computational aspect of learning a parameterized SVD, including possible orthogonality regularization, is not quantified. There are no numbers for wall clock time, peak memory, or extra parameters, compared to standard LoRA or to another SVD style baseline.

### Questions
1. Can you run at least one sequential or continual experiment, for example a chain of four or five NLU tasks, and show that FCLoRA forgets slower than standard LoRA and at least one recent SVD LoRA baseline?
2. Can you provide a sensitivity analysis of the clipping quantile, for example per layer versus global, and quantiles such as 0.6, 0.7, 0.8 and 0.9, on two different model and task pairs?
3. Can you report training time, peak memory and number of extra parameters for FCLoRA, for standard LoRA, and for one SVD LoRA baseline, on the same hardware and with the same sequence length?
4. Can you clarify in a short ablation in which way FCLoRA differs from AdaLoRA or MiLoRA, other than the forgetting motivation, for example by replacing your clipping rule with their allocation rule?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FCLoRA, a variant of LoRA fine-tuning that injects "fine-grained" singular components into a frozen pre-trained model via a parametrized SVD. FCLoRA treats the adapter adjustment weights as $U\Sigma V^\top$ and clips each learnable singular value $\sigma_n$ to lie below a fixed threshold $\bar{\sigma}$, which is some q-th quantile of the original extracted singular values. As part of the objective function, the paper applies an orthogonality regularizer to $U$ and $V$. The main claim of the paper is that by focusing learning on smaller (or a range of) singular directions, FCLoRA adapts to new tasks while preserving the pre-trained structure, thus reducing catastrophic forgetting.

### Strengths
- A clear intuition that the dominant singular vectors of pre-trained weights should remain fixed while only finer components adapt. Then, the idea of parametrizing the adapter by SVD with slipping is a plausible way to realize this idea. 
- Forgetting mitigation is demonstrably effective, though there is a drop in the pre-training accuracy. 
- The authors do include an ablation table comparing variants of LoRA w./w.o. SVD parametrization, orthogonality constraint, and clipping. This is a good sign that the authors tried to analyze the components of their method.

### Weaknesses
- Several recent methods already (i) initialize/operate in SVD space, (ii) target minor components (milora), or (iii) decompose weight updates (dora). Here, the primary new element is clipping $\Sigma$ plus an interpretation of it as fine-grained injection. The paper should more rigorously contrast their clipping to: MiLoRA's only minor-component editing, AdaLoRA's rank budgeting, and LoRA^2's multi-scale SVD, with controlled ablations where only clipping differs. 
- The theory is suggestive, but not decisive. The bound is loose and non-constructive wrt. an optimal threshold $\bar{\sigma}$; it motivates clipping but does not characterize when clipping is both necessary and sufficient. Spectral-norm growth study is cited to argue a typical increase in the norm, not the link to adapter-only updates, and optimizer types & batch size in these tasks are not quantified. 
- Ablations are insufficient. Although Table 5 contrasts four variants, there are still mixed changes. The effects of each component (clipping threshold, orthogonalization, and SVD parametrization) are not cleanly disentangled. For example, there is no run with parametrized SVD without clipping, or clipping without orthogonalziation. This makes it harder to know which design choices actually drive the results. 
- Note on the clipping threshold. The choice of $\bar{\sigma}$ is only briefly described as a fixed quantile of the pretrained singular values, and then used without justification. The main text offers no guidance on how it was chosen or how results would vary. Appendix O claims to study it, but those results & explanations need to be in the main paper. In practical usage, one would want to know if performance changes sharply or if fine-tuning on a certain set yields significantly different settings. Without this, a proper description of what is what, the method's robustness is unclear. It appears that, across models and tasks, this hyperparameter setting varies, adding another dimension of complexity. 
- Forgetting is tested only on Bookcorpus with minimal detail on the setup, and just for text. The analysis lacks diversity across pre-training corpora or modalities. 
- Despite "no overhead" claims, the appendix shows slower training due to orthogonal regularization. This contradicts the efficiency claim and should be reported clearly. 
- **One other important note:** While the proposed spectral clipping intuitively restricts representational flexibility, the method consistently outperforms all baselines, including the unconstrained ones and across all datasets. This is counterintuitive. The paper should analyze why a capacity-limiting constraint leads to higher downstream accuracy. Is it because of uneven hyperparameter tuning or something else? For this reason, the empirical claims seem unconvincing.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
