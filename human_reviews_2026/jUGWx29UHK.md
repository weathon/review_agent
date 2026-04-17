# Understanding Federated Unlearning through the Lens of Memorization

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Federated learning (FL) must support unlearning to meet privacy regulations. However, existing federated unlearning approaches may overlook the overlapping information between the unlearning and retained datasets, leading to ineffective unlearning and unfairness between clients. We revisit this problem through the lens of memorization, showing that only unique memorization information from the unlearning dataset should be removed, while shared patterns should remain. Subsequently, we propose the Grouped Memorization Evaluation, a metric that distinguishes memorized from shared knowledge, and introduce Federated Memorization Eraser (FedMemEraser), a pruning-based method that resets redundant parameters carrying memorization information. The experimental results demonstrate that our method closely matches the retraining baselines and effectively eliminates memorization information compared to other unlearning algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The manuscript revisits the problem of federated unlearning from a memorization perspective, arguing that only the unique memorized information within the forgetting dataset should be removed, while the shared patterns should be retained. The authors propose a metric for distinguishing between memorized and shared knowledge at the instance level: Grouped Memorization Evaluation, and introduce Federated Memorization Eraser, with experiments validating the effectiveness of their method. However, there remain several issues that merit further attention:

### Strengths
1. The manuscript redefines the federated unlearning problem from the perspective of memorization and demonstrates that overlapping or shared information should not be unlearned. 
2. The manuscript proposes Grouped Memorization Evaluation, a novel metric that can measure memorization information at the example level, thereby enabling a fine-grained assessment of unlearning efficacy.

### Weaknesses
1. The effectiveness of the FedMemEraser method relies on the pruning ratio. The manuscript notes that this ratio needs to be determined empirically, which might increase the difficulty and tuning cost of applying the method in different scenarios.
2. The method's core assumption is that redundant parameters relative to the remaining dataset D_r  primarily carry the unique memorization information of the unlearning dataset D_u. The universality and completeness of this link require further theoretical or experimental validation.
3. The proposed "Grouped Memorization Evaluation" metric requires retraining the model multiple times to compute the memorization score for each example. This could introduce significant computational overhead in practice.
4. The results in Fig. 1 are too densely presented, which negatively impacts readability.

### Questions
1. In Section 6.2, during Stage 1, the manuscript uses the average gradient magnitude to evaluate the amount of memorized information in the parameters. However, the authors do not provide sufficient justification and evidence for the superiority and necessity of this step. Moreover, the rationale behind the selection of the threshold hyperparameter for filtering is not thoroughly explained—how is the predetermined percentage of initialized parameters determined? In addition, the method does not appear to account for sensitivity differences across layers.
2. The manuscript discusses the issue of overlapping and non-overlapping information. Since the proposed method aims to remove non-overlapping information while preserving overlapping information, would the presence of retained clients whose data distributions are similar to that of the forgetting client affect the evaluation results?
3. The core assumption that “memorized information is equivalent to non-overlapping information” appears to be overly idealized. In Equation (3), $F_m=F_u-F_g$ is defined as the memorized information, but the manuscript does not explain how to concretely distinguish between “overlapping” and “non-overlapping” components in the feature space. Given that the feature distributions are highly nonlinear, an empirical definition alone cannot guarantee that $F_m$ truly corresponds to the memorized portion.
4. The assumption regarding redundant parameters in FedMemEraser lacks validation in Stage 1. The key premise of the algorithm is that small-gradient parameters correspond to memorized information. This assumption is too absolute and lacks theoretical justification or empirical support.
5. FedMemEraser consists of three phases: positioning, resetting, and fine-tuning. If only positioning and reset phases are performed without fine-tuning, what would happen to the forgetting effect and generalization performance of the model? How much forgetting effect does the reset operation itself contribute, and how much does the subsequent fine-tuning contribute?
6. The manuscript's experiments mainly focus on client level forgetting. How will the FedMemEraser method apply to sample-level or category-level forgetting in federated learning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper re-examines federated unlearning through the lens of memorization, arguing that common metrics cannot verify whether client-specific information is truly removed. It introduces Grouped Memorization Evaluation (GME): estimate per-sample memorization via multiple retrainings, group the to-be-forgotten samples by score, and contrast “unlearning vs. retraining” within groups for fine-grained assessment. The method FedMemEraser identifies potentially memorization-bearing redundant parameters using the remaining clients’ average gradients, resets them, and fine-tunes only on remaining clients to restore generalization. Experiments on CIFAR-10/100 and EMNIST evaluate GME effectiveness, test performance, local fairness, and time/communication cost.

### Strengths
Clear problem reframing and definitions. The decomposition into shared Fg vs. memorized Fm information provides a precise target for Fu and explains why removing overlap harms generalization and fairness. Additionally, the formal “federated memorization unlearning” definition is clean and intuitive.
Evaluation innovation. Grouped Memorization Evaluation (GME) directly probes whether high-memorization examples in Du are “forgotten,” enabling fine-grained, content-aware assessment beyond global accuracy or distance metrics—addressing known shortcomings of prior FU evaluations. 
Simple, FL-compatible method. FedMemEraser requires only server-side aggregation of client gradients already present in FL to locate low-update (redundant) parameters, followed by reinit + fine-tuning. This simplicity is attractive and appears robust to both IID and non-IID data.
Broad evaluation coverage. Beyond Grouped Memorization Evaluation and test accuracy, it also reports local fairness and time/convergence behavior.

### Weaknesses
Cost and practicality of  Grouped Memorization Evaluation (GME). The memorization score requires training J retrained models without Du to estimate probabilities, which is computationally heavy and may be impractical for large-scale FL. The paper should clarify how J is set.
Hyperparameter sensitivity and selection protocol. The method’s main knob ρ is dataset-specific (e.g., different ρ for CIFAR-10 vs. CIFAR-100), and ablations show strong effects. The paper may be should specify a principled, data-independent selection rule and report sensitivity curves. 
Coverage of federated learning in Related Work. The Related Work section barely discusses FL and FU beyond a brief definition in Preliminaries. Please expand the FL-related literature review: position your work against core FL/FU lines (e.g., client heterogeneity, privacy/unlearning, fairness in FL, server-side vs. client-side unlearning), and clarify what is genuinely new here relative to these threads.
Terminology and dataset naming consistency. Please standardize technical terms and dataset names across the paper. For example, consistently use “CIFAR-10/100” (with a hyphen and capitalization) instead of variants like “CIFAR10,” and ensure line 425’s “CIFAR10” is corrected to “CIFAR-10.”

### Questions
GME overhead and configuration. What value of JJJ was used to compute memorization scores? How costly is GME relative to a full retrain, and is it strictly an offline evaluation tool (no influence on hyperparameter selection)? 

Hyperparameter ρ selection. How should practitioners choose ρ without access to Du? Did you fix ρ per dataset or per run?

Precise definitions in Eqs. (9) and (11). In Eq. (9), what exactly is Pr? Is it the same quantity as M in Eq. (11), or is M derived from Pr (e.g., an average over runs/clients/examples)? Please give a rigorous, self-contained definition , and state the intended direction: in Eq. (11), does lower M indicate better unlearning (i.e., less memorization), or the opposite?
Threshold selection in Eq. (10). How are the thresholds in Eq. (10) chosen in practice (fixed a priori, validated on remaining-client data, or tuned per dataset/model)? Please describe the protocol used in experiments and provide a sensitivity analysis or at least the chosen values and their rationale.
Subgroup mapping and high scores in Table 1. Do the subgroups in Table 1—for example, (95%,100%]—correspond exactly to the thresholds defined in Eq. (10)? If so, why do low-memorization bins show extremely high scores (e.g., EMNIST IID achieves 99.97% in Group (0%,80%])

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores federated unlearning from the perspective of memorization and proposes FedMemEraser, a lightweight method combining gradient-threshold-based pruning and fine-tuning to remove memorized information while retaining shared knowledge. The study provides both theoretical insight into the connection between memorization and unlearning effectiveness and empirical results

### Strengths
- Writing is good and easy to follow and understand.

- It provides a clear theoretical formulation linking memorized knowledge to model parameters, which helps explain the trade-off between unlearning effectiveness and performance retention.

### Weaknesses
- Why [1] can not become a baseline of this paper? Please demonstrate the experiment results.

- I have carefully checked, and some relevant studies are not discussed in this paper. For example [1] and so on. Please do a comprehensive investigation.

- The proposed FedMemEraser essentially follows a combination of gradient-threshold-based redundant-parameter pruning and a fine-tuning procedure. Compared with existing unlearning approaches based on weight importance or influence functions, its novelty is a problem.

- This paper conducts experiments with a fixed setup of 10 clients and does not evaluate the method under different numbers of clients, leaving its scalability and stability unverified.

[1] Not:Federated unlearning via weight negation

### Questions
See in weakness

### Soundness
3

### Presentation
3

### Contribution
2
