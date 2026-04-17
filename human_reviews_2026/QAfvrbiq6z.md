# CBP: Learning Shared Cognitive Basis Space and Connectivity Patterns for Cross-Task Brain Dynamics Modeling

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Understanding the brain dynamics underlying human cognition requires models that jointly achieve modeling capability and interpretability across tasks. Existing approaches either rely on deep learning models, which learn strong but implicitly encoded representations that are difficult to interpret, or explicit state-space models, which are interpretable but limited to single-task settings. To bridge this gap, we present CBP, an interpretable, non–deep-learning framework for cross-cognitive-task brain modeling. By explicitly leveraging shared information across tasks, CBP accurately recovers latent components while maintaining state-of-the-art performance. Importantly, CBP uncovers a stable set of \underline{\textbf{C}}ognitive \underline{\textbf{B}}ases and connectivity \underline{\textbf{P}}atterns in the human brain. The reliability of these discoveries is supported by extensive quantitative evaluations and a battery of perturbation analyses, demonstrating their robustness. Moreover, leveraging these shared cognitive components allows CBP to generalize effectively to both the HCP and multi-stage learning datasets, where it not only achieves accurate prediction of task states and learning outcomes but also reveals the key cognitive connectivity patterns underlying these behaviors. Together, these results establish CBP as an interpretable framework for large-scale cognitive dynamics, offering mechanistic insight into cross-task cognition.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes CBP, which is a dictionary-learning cross-task modeling framework that learns shared cognitive bases and their connectivity patterns. While existing methods, such as single-task neural dynamics, large-scale multi-task foundation models, and state-space models, cannot simultaneously offer predictive performance and interpretability in multi-task settings, the proposed model improves cross-task modeling accuracy while revealing key cognitive patterns. As a result, CBP surpasses multiple baselines on the 103-task fMRI dataset and generalizes to HCP and multi-stage learning datasets.

### Strengths
1)	Based on MacDowell et al. (2024, 2025), this paper explores the existence of a set of global, temporally invariant cognitive basis spaces and connectivity patterns in large-scale human cognitive dynamics. 

2)	Extensive experiments were performed on 5 datasets, including both synthetic and real-world datasets. Moreover, the paper presents interpretable brain-regional analyses showing that the model captures biologically meaningful and task-relevant activation patterns.

### Weaknesses
1)	There is room to improve the clarity of notations. For example, it would be better to explicitly note the definition of $x_{m,t}^{(s)}$ in line 150, e.g., ‘a latent state at time $t$’, such that readers can easily understand that it is an extension of $X_m^{(s)}$. The explicit role of $c^{p,(s)}_{m,t}$ in Eq. 1 and the definition of $\rho(\cdot)$ in line 199 is not mentioned in page 4, which makes the equations difficult to interpret immediately. 

2)	There are so many hyperparameters to fine-tune. Concretely, the loss function is comprised of 4 main terms, and the last term is a linear summation of five components, which requires at least seven trade-off hyperparameters ($\lambda_x, \lambda_a, \lambda_c, \lambda_{\rho}, \alpha_1, \alpha_2, \alpha_3$) in total. The model performance will potentially be affected by these parameters, and the reproducibility and credibility of the results are hardly guaranteed given the large number of hyperparameters involved in the loss function. It would be helpful if the authors could discuss the sensitivity of the model with respect to these hyperparameters, or provide guidance on how they were chosen (e.g., through grid search, cross-validation, or empirical heuristics). 

3)	In the Introduction, the paper compares the proposed method with existing (1) single-task neural models, (2) large-scale multi-task fMRI foundation models (BrainLM, Brain-jepa), and (3) State-space models and asserts its strength over these methods. However, in the experiment, the (2) foundation models were not adopted as baselines, which weakens the empirical validation of the claimed advantages. It would be helpful to provide additional comparison with these frameworks, and I would like to ask why they were excluded in the current version.

### Questions
1)	Based on Table 1, the CBP and AMAG have comparable performance, although AMAG was originally designed as a single-task neural dynamics model [line 46]. This raises a question of what the technical advantage of the proposed method is over AMAG.

2)	How do the cognitive and connectivity patterns change with different numbers of bases ($K$ and $P$)? Are there any neuroscientific implications regarding the choice of their sizes?

3)	How were the category-wise mean activations (Fig.3-B and D) calculated in detail?

4)	Could the authors provide details on how the baseline methods were tuned (e.g., hyperparameter selection procedure, search range, or validation strategy)?

5)	How efficient is the proposed method in terms of training time and model size (number of trainable parameters) compared with the baselines?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a method of CROSS-TASK BRAIN DYNAMICS MODELING, called CBP, via learning cognitive bases. CBP is a dictionary-learning cross-task modeling framework so that it can leverage shared information across tasking cognitive states. The experiments on two synthetic datasets and three real-world datasets demonstrate the contributions of generalizability and correlations between tasking cognitive states.

### Strengths
1. As the foundation model of brain fMRI is growing, the topic of how to properly use task-state fMRI is important and timely.

2. CBP discovered cognitive bases look diverse in the qualitative results (fig 3), indicating that the proposed method is somehow effective.

3. Results are highly interpretable by the distribution of cognitive bases.

### Weaknesses
1. The terminology "multi-task" is confusing in the paper. This term is commonly used for multi-task learning, which represents a methodology of modeling multiple objectives simultaneously. However, this paper used "multi-task" to refer to the task-state functional MRI under different cognitive states, which misleads readers.

2. The "cross-task" (I follow the term used in this paper, but more precisely, this should be "cross-cognitive states") problem in previous brain foundation models is outstanding [1], but the authors failed to review related works precisely. Inaccurate statements are misleading, e.g., *"Large-scale multi-task fMRI foundation models (e.g., BrainLM Ortega Caro et al. (2024), Brain-jepa Dong et al. (2024)) learn highly transferable shared representations and substantially improve multi-task prediction"*. First, BrainJEPA was pretrained on pure resting-state fMRI, and thus it's not a "cross-task" model. Second, all these foundation models were finetuned on applications with a single objective, having no ability for multi-task prediction.

3. The contribution to real-world applications is not clear in the paper. The experiments of real-world datasets compared with baselines were only on BOLD signal reconstruction. Although the cognitive state classification accuracy for the HCP dataset is present in Fig 4, there is no baseline comparison, and the other two real-world datasets were omitted. 

4. Grammar errors and typos are obvious. E.g., *"CBP Discovers Cognitive Basis connectivity patterns Predictive of Task Variables"*

[1] Ziquan Wei, Tingting Dan, Tianlong Chen, and Guorong Wu "BrainMoE: Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model" In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025

### Questions
None

### Soundness
3

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
The paper introduces CBP (Cognitive Bases and connectivity Patterns), a dictionary-learning–based model designed to jointly capture shared cognitive components and their connectivity patterns across multiple fMRI tasks. Unlike existing foundation or state-space models—which are either hard to interpret or restricted to single tasks—CBP explicitly integrates cross-task information, modeling task-evoked brain dynamics as trajectories of a unified cognitive system. But the experiments are insufficient and the results were not very convincing.

### Strengths
1. The method is simple and easy to understand, and the defined losses are clear.
2. The paper demonstrates extensive experiments on synthetic, OT, HCP, and TLAE datasets. The qualitative analyses (e.g., identification of 12 cognitive bases and 8 connectivity motifs) show clear biological interpretability and align with canonical functional networks, which enhances the neuroscientific relevance of the work.

### Weaknesses
1. The performance is not state-of-the-art. This weakens the overall argument, as the proposed approach is less convincing when it does not outperform existing methods. And there is no statistical verification.
2. The comparison is rather limited; several widely used fMRI-based analysis methods were not considered (such as BrainGNN, BolT,BNT,Graphormer,NAGphormer...and foundation models BrainLM, BrainMass,Brain-JEPA......), which weakens the comprehensiveness of the evaluation.
3. The model introduces many regularization terms (Laplacian, sparsity, smoothness, decorrelation, etc.), but the intuition and relative contributions of each are not fully analyzed. The ablation table exists but lacks sufficient explanation and qualitative interpretation.
4. The paper is quite dense and mixes neuroscience motivation with technical formulation. The exposition could be streamlined to make the central idea—cross-task dictionary learning for cognitive bases—stand out more clearly.
5. While the integration across tasks is interesting, much of the mathematical formulation is reminiscent of existing decomposed LDS or sparse coding frameworks, with relatively incremental novelty at the algorithmic level.

### Questions
1. Are all datasets processed using the same MMP360 atlas for parcellation? Differences in atlas choice (e.g., MMP vs. Schaefer or Yeo) can strongly influence functional patterns and network structure, so clarification on this point would help interpret the cross-dataset results.
2. The paper identifies 12 “cognitive bases” that align with known networks. Can the authors provide quantitative validation (e.g., spatial correlation or network overlap with canonical atlases such as Yeo-17 or Power-264) to substantiate these qualitative findings?
3. How does the training complexity of CBP scale with the number of tasks and voxels? Given the high dimensionality of fMRI data, are there efficiency tricks (e.g., low-rank updates or GPU acceleration) that make the approach practical for larger datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CBP that learns a shared cognitive basis space and a set of connectivity patterns that describe how these bases interact over time. The model is designed to be both interpretable and generalizable across many tasks, and the authors provide experiments on synthetic data, OT fMRI (over 100 tasks), HCP tasks.

### Strengths
1. The work explicitly separates and models different components of brain activity (shared bases, task-specific activation trajectories, and connectivity patterns). This gives the model a high level of interpretability.

2. The experimental design is rich and logically structured. The authors move from controlled synthetic data, to large-scale multi-task fMRI, and then to HCP (task decoding). 

3. The model provides intuitive visualizations. The learned bases are shown as cortical maps that appear aligned with known functional systems.

### Weaknesses
1. The paper assumes that different tasks are different trajectories in the same cognitive system, i.e. that a single shared set of bases and patterns can explain many (or all) tasks. Is this really sufficient? Are the learned bases universal, or are they tied to a specific dataset and experimental protocol? What happens if we change the acquisition condition for the same subject (for example, different scanner runs, different cognitive state, different instructions)? I suggest adding a sensitivity analysis that tests how stable the learned bases are under such perturbations.

2. The paper interprets $A$ as functional subnetworks. Is this mapping predefined in any way, or purely learned in a data-driven way? The paper suggests that using a larger number of bases improves reconstruction and downstream prediction. If so, how should we interpret different choices of the number of bases? Right now $A$ feels partly like a posterior observation, which weakens the claim of intrinsic interpretability.

3. The model explains temporal evolution of brain activity as a combination of multiple linear interaction patterns between bases. Real neural dynamics are often strongly nonlinear and state-dependent. The paper should discuss how well this linear-by-parts approximation can capture nonlinear behavior. For example, under what conditions would this approximation fail? 

4. The transfer experiment mainly goes in one direction, from very rich multi-task dataset OT to a fewer-task dataset HCP. If we had an even more fine-grained cognitive taxonomy than OT (more subtle task distinctions), would the bases learned on OT still work well?
The paper should also report a baseline where the model is trained directly on HCP, and then compare that to the transferred version from OT. How different are the learned bases and patterns in these two cases?

### Questions
How exactly are the learned bases $A$ linked to known functional subnetworks? Is this a qualitative, visual matching (this looks like auditory cortex, so we call it auditory)? Similarly, how are the learned connectivity patterns mapped to cognitive labels like “perception–action loop” or “integration pathway”?

### Soundness
3

### Presentation
2

### Contribution
3
