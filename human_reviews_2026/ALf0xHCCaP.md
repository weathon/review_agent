# Task-Level Insights from Eigenvalues across Sequence Models

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Although softmax attention drives state-of-the-art performance for sequence models, its quadratic complexity limits scalability, motivating linear alternatives such as state space models (SSMs). While these alternatives improve efficiency, their fundamental differences in information processing remain poorly understood. In this work, we leverage the recently proposed dynamical systems framework to represent softmax, norm and linear attention as dynamical systems, enabling a structured comparison with SSMs by analyzing their respective eigenvalue spectra. Since eigenvalues capture essential aspects of dynamical system behavior, we conduct an extensive empirical analysis across diverse sequence models and benchmarks. We first show that eigenvalues influence essential aspects of memory and long-range dependency modeling, revealing spectral signatures that align with task requirements. Building on these insights, we then investigate how architectural modifications in sequence models impact both eigenvalue spectra and task performance.  This correspondence further strengthens the position of eigenvalue analysis as a principled metric for interpreting, understanding, and ultimately improving the capabilities of sequence models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies sequence models (SSMs and masked attention) through a **dynamical-systems lens** and argues that **task requirements are reflected in the spectrum of state-transition eigenvalues**. Using the DSF formulation, causal attention is written as a recurrence, so eigenvalue magnitudes capture how much the past state is retained (near 1) or forgotten (near 0). The authors empirically profile eigenvalue distributions across architectures (S4, LRU, Mamba-2; softmax, linear, norm attention), layers, and tasks (ListOps, IMDb, CIFAR-10, MQAR, WikiText).

Main findings are:

* **Spectral signatures align with task needs.** Long-memory tasks concentrate mass near 1; tasks needing selective forgetting show peaks near 0.
* **Architectural choices shift spectra in intuitive ways.** Adding explicit **gates** reduces the need for implicit gating in the dynamics; adding **convolution** offloads local context and shifts mass toward 0; **normalization** choices in norm attention trade retention vs selectivity; making **Mamba-2 pseudo-LTI** (constant discretization) yields S4-like spectra and helps ListOps.
* The spectrum can therefore serve as a **diagnostic/design tool** to steer models toward retention or selectivity.

### Strengths
* **Unifying perspective.** A clear DSF view that lets attention and SSMs be compared apples-to-apples through the same spectral metric.
* **Comprehensive empirics.** Multi-task, multi-model, per-layer analyses with ablations (gating, conv, normalization, depth, pseudo-LTI). This breadth gives credibility to the main trend claims.
* **Actionable insights.** Concrete “design → spectrum → behavior” links (e.g., conv shifts mass to 0; constant discretization moves Mamba-2 toward retention). Practitioners can use this to target spectra for a task.
* **Clarity of interpretation.** The near-0 vs near-1 story is intuitive and connects to long-standing control/SSM theory, helping demystify why models succeed or fail on LRA-style tasks.

### Weaknesses
* **Incremental novelty.** Much of the narrative (poles near the unit circle ↔ long memory; selective forgetting ↔ small eigenvalues) echoes prior SSM/RNN work; S5 and related papers already emphasize spectral placement for retention and task performance. The contribution is more **systematization** than new theory.
* **Limited quantification.** The link between spectra and performance is presented mostly as qualitative alignment. No formal predictive model (e.g., correlation between “mass in [0.9,1]” and accuracy/perplexity) or causal intervention is provided.
* **Coarse metric.** Binning only by **magnitude** discards **phase** information of complex eigenvalues (oscillations) and ignores distributional shape beyond coarse buckets. This may hide important structure.
* **Scale and controls.** It’s unclear how sensitive results are to model size, parameter budget, or training recipe. Using one head for plots can mask head-to-head variance; significance tests across seeds are limited in the main text.
* **Scope.** Benchmarks are mostly small/medium. It remains uncertain how well these spectral diagnostics extrapolate to large-scale language models.

### Questions
1. **Quantification.** Can you report correlations (or simple regressions) between task performance and spectral mass in specific bins (e.g., ([0.9,1]) vs ([0,0.1])) across runs/layers? This would strengthen the predictive value of the metric.
2. **Dynamics over training.** Do spectra evolve monotonically toward the final pattern? Showing trajectories (epochs vs mass in each bin) could clarify causality.
3. **Complex phases.** Did you analyze the angles of complex eigenvalues (oscillatory modes)? Any task where phase structure matters?
4. **Head and seed variance.** The appendix includes more heads; could you summarize variance across heads/seeds with statistical testing?
5. **Capacity control.** When comparing architectures, are parameter counts and FLOPs matched closely? If not, can you add matched-capacity comparisons?
6. **When spectra mislead.** Are there counterexamples where spectra look favorable but performance lags (or vice versa)? Understanding failure modes would bound the tool’s applicability.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
### Summary

1. Extends Eigenvalue Analysis to LPV Models:

   The paper builds on earlier work that analyzed eigenvalues in Linear Time-Invariant (LTI) State Space Models, where eigenvalues capture memory, stability, and selectivity. Eigenvalues near 1 mean good long-term memory, while those near 0 indicate forgetting or gating. 

   The authors extend this idea to Linear Parameter-Varying (LPV) systems using the Dynamical Systems Framework (DSF), which writes models like Attention and Mamba-2 as dynamical systems:
   
   $h_i = \Lambda_i h_{i-1} + B_i u_i, \quad y_i = C_i h_i + D_i u_i$

   Here, the parameters change with input, allowing a similar eigenvalue analysis across model types.

2. Analysis:

   On memory, authors show the intuitive result that models that remember well keep eigenvalues close to 1. On language modeling (WikiText), the authors find that “gating” or eigenvalues near zero help model selectively store tokens.

3. Architectural Modifications Based on the Analysis:

   The paper tries to use these insights to improve models with changes including: Gating, Short Convs, Varying the number of layers etc

### Strengths
1. The paper’s central idea, that is extending eigenvalue analysis from LTI SSMs to LPV systems and attention, is natural and easy to follow.
2. The connection between eigenvalues and memory retention/selectivity is intuitive and aligns with established understanding.
3. The paper conducts experiments across a breadth of models (S4, Mamba-2, Self-Attention, Linear Attention etc) and tasks (LRA, MQAR, WikiText)

### Weaknesses
### Weaknesses

1. **On memory tasks**

   1. The analysis of memory is intuitive and largely reiterates known understanding—that eigenvalues near one preserve information and those near zero lead to forgetting.
   2. The attention results, though potentially interesting, are underexplained. The paper attributes attention’s poor performance on LRA to having both very low and very high eigenvalues, but I suspect this explanation may be correlational rather than causal. Attention is known to perform strongly on other memory-intensive retrieval tasks (e.g., NIAH [1]). It would have been useful if the paper discussed this discrepancy and explored why attention struggles on LRA despite its well-documented ability to retain long-term information in other settings.

2. **On gating/selectivity tasks**

   1. The interpretation that “gating” corresponds to zeroing out eigenvalues in WikiText models like Mamba-2 is reasonable but not novel as it directly follows from the Mamba-2's design of selection mechanism.
   2. The follow-up “add gate” experiment on attention has a conceptual mismatch---The gating mechanism used in the experiments differs from Mamba-2’s within-sequence gating, and is instead applied per-token AFTER sequence mixing, making the analogy weak.
   3. The task choice and conclusions seem inconsistent: gating is shown to help on IMDb, a memory-heavy task where (if i understand correctly) it should theoretically hurt (or at-least not help). Furthermore, ListOps, which is also memory intensive, unexpectedly develops gating, which remains unexplained. 

       I expected to see a gating-improved task, like WikiText, show improvement when gating is added to attention.

3. **On convolution and varying number of layers**

   1. These experiments are not motivated by eigenvalue analysis and are instead motivated from Mamba-2's strong performance which feels disconnected from the main argument.
   2. The claim that short convolutions “take over long-range memory” seems to be incorrect—short convolutions are "short" (of size 4) and cannot provide such capacity. The paper employs this argument to justify why a one layer attention model with convolution could solve MQAR. I believe this justification is incorrect. Prior work [1] has shown that MQAR performance requires "induction-head formation": first sequence mixer mixes keys and values and the second sequence mixer retrieves the correct value. Convolution helps because it performs the first task, not because it replaces long-range retrieval.

4. **On evaluation scope**

   1. For the proposed architectural changes, the experiments are limited to small-scale settings and on synthetic tasks.
   2. It would be more convincing to test these modifications on language modeling and downstream evals at across multiple scales (e.g., 125M, 350M, 750M, 1.3B) to assess their importance.

5. **On scope and completeness**

   1. The paper briefly mentions that eigenvalues evolve during training, suggesting potential task-dependent initialization, but this idea is never explored.
   2. The architectural additions should be tested on language modeling for validation.
   3. Overall, the paper reads as an incremental extension of prior DSF analyses, with several unexplained experimental results.

-------
[1]: Mechanistic evaluation of Transformers and state space models. Aryaman Arora, Neil Rathi, Nikil Roashan Selvam, Róbert Csordás, Dan Jurafsky, Christopher Potts

### Questions
See weakness

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
This paper proposes using a unified dynamical systems framework (DSF) to analyze and compare the behavior of various sequence models, including SSMs (S4, LRU, Mamba-2) and different attention mechanisms (softmax, linear, norm). The core idea is to represent these models as dynamical systems and study their eigenvalue spectra to find "spectral signatures" that correlate with specific task requirements, such as long-term memory or selective forgetting. The authors conduct an empirical study across several benchmarks (LRA, MQAR, WikiText) to support their claims.

### Strengths
- The primary strength of this work is the application of the DSF to provide a common ground for analyzing seemingly disparate architectures like SSMs and attention. The goal of finding a unified principle to understand their internal dynamics is valuable and timely.
-  The paper includes an extensive set of experiments across multiple model classes and a diverse set of tasks.

### Weaknesses
- The authors frame their work as a novel application of eigenvalue analysis of sequential models. However, the paper fails to cite or engage with a crucial body of literature on analyzing non-linear dynamical systems using linear operators. The work misses key papers such as [1]. This is particularly relevant, as it explicitly analyzes the eigenspectrum of the Koopman operator to model the non-linear dynamics of sequence models. This paper's core method—using the DSF to derive a linear-like state transition matrix $\Lambda_i$ and analyzing its spectrum—is conceptually similar. The paper would be significantly stronger if it contextualized its approach within this existing work or compare with it.

- The paper notes that linear attention retains its initialization's spectral shape, “raising the question of the potential importance of a task-dependent initialization.” This question remains unaddressed. The paper poses this as an open question but fails to engage with significant, recent work that directly addresses it. The work [2] (and related works) demonstrates that the initialization of weights (e.g., from a pre-trained model) can significantly improve the performance of softmax attention-based models on the LRA benchmark—the same benchmark used in this paper. This is a critical omission. It strongly suggests that a "good" spectral initialization is vital for success. The paper misses a key opportunity to connect its analysis to this important line of work and investigate, for example, how the initial eigenvalues of a pre-trained model compare to the random initializations used in this study.

- The analysis of architectural modifications in Section 5 only partially demonstrates the framework's explanatory power, leaving the overall contribution and utility of the method in question (for this application).


[1] "An operator theoretic approach for analyzing sequence neural networks", Naiman et al. (AAAI23)

[2] "Never Train from Scratch: Fair Comparison of Long Sequence Models Requires Data-Driven Priors", Amos et al. (ICLR 2024)

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes eigenvalue analysis as a principled metric for understanding the capabilities of sequence models. This work first uses an existing framework to formulate softmax, norm, and linear attention as dynamical systems, and then compare the eigenvalues of these models with SSMs, which are naturally based on dynamical systems. Experiments show that the eigenvalue spectra of these models are task-dependent, capturing the long-range modeling capability of these models under various long-memory tasks. Furthermore, empirical results demonstrate how changing model architectures can change the eigenvalue spectra and in turn the task performance.

### Strengths
1. The paper has interesting experimental results on the shifting roles of sequence mixing operators such as convolution, attention, and SSMs. 
2. The paper evaluates a comprehensive set of models on representative long-range tasks. 
3. The paper is clearly written.

### Weaknesses
1. Novelty and technical difficulty: the paper's contribution is mainly conceptual rather than technical, and in my view the conceptual contribution appears somewhat limited. The dynamical system parameterization of transformers is from an existing work, and eigenvalues are known to be important and a central object of study in SSMs. 
2. For LPV systems, the eigenvalues are different across time steps, what's the time step corresponding to the eigenvalues in the plots?
3. Even though Section 5 has very interesting insights, there are no concrete proposals for new architectures.
4. In Section 5, the analysis of the effects of gating the convolution doesn't seem to apply to ListOps, indicating that the effectiveness of such analysis depends on each task. Can the authors give a characterization of the kind of tasks such analysis would apply to? Where would the analysis break down? Moreover, since ListOps requires all input information, I would expect no gating behavior to occur in SSMs. However, both LRU and Mamba 2 have small eigenvalues. Could the authors explain why that might be the case?

Minor: the plot on page 5 can use different shades to indicate value, instead of different colors.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
