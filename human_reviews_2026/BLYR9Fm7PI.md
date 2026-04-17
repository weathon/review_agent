# FedQUIT: On-Device Federated Unlearning via a Quasi-Competent Virtual Teacher

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Federated Learning (FL) enables the collaborative training of machine learning models without requiring centralized collection of user data. To comply with the right to be forgotten, FL clients should be able to request the removal of their data contributions from the global model. In this paper, we propose FedQUIT, a novel unlearning algorithm that operates directly on client devices that request to remove its contribution. Our method leverages knowledge distillation to remove the influence of the target client’s data from the global model while preserving its generalization ability. FedQUIT adopts a teacher–student framework, where a modified version of the current global model serves as a virtual teacher and the local model acts as the student. The virtual teacher is constructed by adjusting the global model’s outputs on forget data, penalizing the confidence assigned to the true class while preserving relationships among outputs of non-true classes, to simultaneously induce forgetting and retain useful knowledge. As a result, FedQUIT achieves unlearning without making any additional assumption over the standard FedAvg protocol. Evaluation across diverse datasets, data heterogeneity levels, and model architectures shows that FedQUIT achieves superior unlearning compared to six state-of-the-art methods, while significantly reducing cumulative communication and computational overhead relative to retraining from scratch.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a federated unlearning (FU) algorithm called FedQUIT, which aims to execute 'forget' requests directly on client devices without requiring historical gradients or additional public data on the server side. The core design idea is to modify the logits of the global model on the data to be forgotten i.e. penalizing the logits of the true class while preserving the relationships among non-true classes, thereby constructing a virtual teacher; the local model (student model) then mimics this teacher through one round of on-device knowledge distillation, after which the standard FedAvg training process resumes. In the experimental section, on CIFAR-10, CIFAR-100, and CUB-200, compared with six state-of-the-art methods (FedEraser, PGA, MoDe, FedAU, NoT, FedOSD), the results show higher forgetting effectiveness, along with lower communication and computation overhead. The paper also includes a theoretical analysis, proving that the parameter perturbation induced by FedQUIT is bounded and preserves the convergence properties of FedAvg.

### Strengths
1. No need to modify the FedAvg framework, just add a single round of distillation step to deploy.
2. The boundedness analysis of parameter perturbations and the convergence proof of FedAvg after forgetting provide formal guarantees for the stability of the method, filling the theoretical gap in many empirical federated forgetting works.
3. The experiments cover a wide range of topics, including multiple datasets, multiple architectures (ResNet-18 and MiT-B0), and comparisons with various SOTA methods.

### Weaknesses
1. The paper frames the "quasi-competent virtual teacher" as a key innovation, but it is an incremental adaptation of KD for FU. Prior work (e.g., FedNTD, Lee et al., 2022) already uses KD with modified true-class logits in FL (albeit for heterogeneity, not unlearning), and the "preserve non-true class geometry" idea is standard in KD (Hinton et al., 2015). The only novel tweak, i.e. setting v to min logit, is a hyperparameter choice, not a conceptual breakthrough.
2. All proofs rely on idealized smoothing conditions and lack rigorous definitions for non-independent and identically distributed (Non-IID) cases.
3. This paper only analyzes the sensitivity of v value.

### Questions
1. Table 1 labels FedEraser as "limited in effectiveness", lacking comparative results and not tested in large-scale scenarios. Does this mean that FedQUIT is less effective than FedEraser, and that its advantage may diminish as the number of clients increases?
2. What is the rationale for fixing the distillation temperature τ to 1 during training? Have you tried using higher τ to smooth the probability distribution?
3. Section D.8 excludes MoDe and FedOSD from the sample oblivion experiment due to "unclear how to use holdout data". However, MoDe's memory bootstrapping phase and FedOSD's post-training phase explicitly allow clients to use holdout data, why can't these baselines be adapted? Please either include these baselines in the sample oblivion experiment or provide a detailed technical explanation for the exclusion.
4. How is "on-device" implemented? Are the computing and storage limitations of mobile devices fully considered?
5. Does FedQUIT support forgetting multiple clients simultaneously? If multiple requests arrive in parallel, will the algorithm still converge?

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
5

### Summary
This paper introduces FedQUIT, a novel federated unlearning (FU) framework designed for on-device execution. The method addresses a client's request to remove its data's influence from a global model by employing a single-round, history-free, and proxy-free unlearning procedure based on knowledge distillation. The core idea is a "quasi-competent virtual teacher," a modified version of the global model that, on the forget data, penalizes the true-class logit while preserving the relational geometry of non-true class logits. This simultaneously induces forgetting and retains generalizable knowledge. Extensive experiments demonstrate that FedQUIT achieves a superior unlearning-recovery trade-off compared to six state-of-the-art baselines, while significantly reducing communication and computational overhead relative to retraining from scratch.

### Strengths
1. The method's on-device, single-round, history-free, and proxy-free design makes it exceptionally practical for real-world federated learning, particularly in resource-constrained cross-device settings where client availability is intermittent and storing historical updates is infeasible.

2. The concept of the "quasi-competent virtual teacher" is a significant contribution. This carefully crafted distillation target provides a sophisticated way to balance the removal of specific information with the retention of general knowledge, proving more effective than naive supervision perturbation techniques.

3. The experimental validation is a major strength. The paper compares FedQUIT against six strong and relevant baselines across multiple datasets, data distributions (IID and non-IID), and model architectures, using a comprehensive set of efficacy and efficiency metrics that convincingly support its claims.

4. The paper provides solid theoretical insights that justify the forgetting signal and the stability of the post-unlearning model for continued training. These theoretical claims are further substantiated by strong empirical results and insightful ablation studies that analyze the key design choices.

### Weaknesses
1. While the specific construction of the virtual teacher is novel, the broader concept of "supervision perturbation" has been explored. The paper would benefit from a more detailed discussion in the related work section that explicitly contrasts its sophisticated soft-target manipulation against simpler hard-label modifications (e.g., label randomization) to better highlight its unique contributions.

2. The provided theory justifies the unlearning mechanism but lacks formal guarantees on how closely the resulting model approximates the gold-standard retrained model. The paper should be strengthened by adding an analysis that bounds the divergence between the unlearned and retrained models.

3. The method's reliance solely on the local forget set creates a risk of "catastrophic unlearning," where small or atypical requests could disproportionately damage the global model. A robustness analysis should be conducted using highly sparse forget requests to evaluate this potential failure mode.

4. The paper fails to justify the necessity of its knowledge distillation framework over a more direct alternative, namely simple gradient ascent on the forget data. An essential ablation study comparing against unconstrained gradient ascent should be added to validate the design choice.

5. The core assumption that the teacher's non-true class geometry is beneficial knowledge is questionable, as it may propagate biases from a poorly calibrated global model. The analysis would be more convincing with ablations on more robust teacher designs, such as applying temperature scaling to the non-true logits.

6. The experimental recovery protocol, which stops training once a target test accuracy is met, may unfairly penalize baselines by halting their recovery prematurely across all metrics. A more robust evaluation should compare all methods after a fixed recovery budget or by reporting the cost-versus-efficacy Pareto frontier.

### Questions
1. The concept of supervision perturbation for unlearning has been explored. Could you elaborate on the key advantages of your sophisticated soft-target manipulation compared to simpler hard-label modifications like label randomization, particularly in terms of knowledge retention and recovery efficiency?

2. Your theoretical analysis effectively shows that unlearning occurs and convergence can resume. However, a key claim is that FedQUIT approximates the retrained model. Is it possible to provide any theoretical insights or bounds on the divergence between the model produced by FedQUIT and the ideal, retrained-from-scratch model?

3. How robust is FedQUIT when the unlearning request pertains to a very small or statistically atypical subset of a client's data? Is there a risk that the KD process, focused on this unrepresentative set, could disproportionately damage the model's performance on retained data (i.e., catastrophic unlearning)?

4. To better justify the complexity of the proposed KD framework, could you comment on how FedQUIT compares to a more direct unlearning baseline, such as simple, unconstrained gradient ascent on the forget data's loss function? What are the hypothesized benefits of preserving non-true class geometry over simply maximizing the error on the forget samples?

5. The quality of the "quasi-competent" teacher relies on the global model's outputs. In scenarios where the global model is poorly calibrated or overconfident in its incorrect predictions, could preserving this flawed geometry be detrimental? Have you considered or experimented with more robust teacher designs, such as applying temperature scaling to the non-true logits?

6. The recovery protocol stops when a baseline matches the retrained model's test accuracy. Could this criterion mask deficiencies in other unlearning metrics by halting recovery prematurely for some methods? How might the comparative results change if all methods were evaluated after a fixed recovery budget?

### Soundness
2

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
This paper proposes FedQUIT, a novel on-device federated unlearning (FU) algorithm designed to address the "right to be forgotten" in federated learning (FL) while preserving model generalization. FedQUIT adopts a teacher-student framework: a quasi-competent virtual teacher is constructed by modifying the output of the current global model (penalizing the true-class logit of forget data to induce forgetting, while preserving non-true class output relationships to retain useful knowledge), and the local model (student) mimics the teacher via knowledge distillation (minimizing KL divergence). Evaluations across CIFAR-10, CIFAR-100, and CUB-200 datasets (both IID and non-IID distributions) with ResNet-18 and MiT-B0 models show that FedQUIT outperforms six state-of-the-art FU baselines in unlearning efficacy (closer to the "Retrain" gold standard) and significantly reduces communication (15–60×) and computational (15–60×) overhead compared to retraining from scratch.

### Strengths
+ The idea proposed in this paper is straightforward and easy to understand. FedQUIT performs unlearning directly on the requesting client’s device in a single round, eliminating the need for multi-round client-server coordination (a flaw of MoDe and FedOSD) and reducing latency.
+ Unlike methods like FedEraser (relying on historical updates) and Wu et al. (2022b) (relying on proxy data), FedQUIT only uses the current global model and local forget data, avoiding privacy and storage risks from historical data.
+ The paper provides rigorous theoretical analyses, including proofs that FedQUIT induces a controlled forgetting signal (Lemma 1), bounded parameter perturbation (Lemma 2), and maintains FedAvg convergence guarantees after unlearning (Theorem 1).

### Weaknesses
- It is still unclear what the intuition is behind the whole framework design, and why only a knowledge distillation loss can be used to achieve the goal of both efficiency and preserving performance.

- While the paper mentions extending FedQUIT to sample unlearning, experiments only cover 50% forget data in a single setting (CIFAR-100), lacking validation on varying forget data ratios (e.g., 10%, 30%) or other datasets.

- It seems that only a marginal performance gain has been obtained from the designed framework, i.e, 1~3%. Furthermore, in Table 1, many of the results of FedQUIT were highlighted, but they are not always the best ones, e.g., CIFAR-10, Non-IID, ResNet-18, E=1, Retain Acc. (88.8%). Please check and clarify.
​
- Evaluations use a single GPU (NVIDIA RTX A5000) and do not test FedQUIT on resource-constrained devices (e.g., smartphones or other devices), where its "on-device" advantage is most critical.

### Questions
1. How does FedQUIT perform when multiple clients simultaneously request unlearning? The current design focuses on single-client unlearning, but real FL systems may face concurrent requests—does it maintain efficacy/efficiency under such scenarios?

2. For highly imbalanced forget data (e.g., 1% vs. 90% of a client’s data), does the default $\(v=min_c z_i^g\)$ still work, or is a dynamic v-adjustment mechanism needed?

3. Can FedQUIT be extended to other FL tasks beyond image classification (e.g., NLP or time-series prediction), and what modifications would be required for non-classification model outputs?

4. In cross-silo FL (where clients are organizations with large datasets), does FedQUIT’s 1-round unlearning still outperform baselines, or does dataset size increase its local computational cost significantly?

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
4

### Summary
This paper proposes FedQUIT, an on-device federated unlearning framework that enables clients to remove their data influence without accessing historical updates or auxiliary datasets. The key idea is to use a quasi-competent virtual teacher that penalizes the true-class logit while preserving inter-class relations, achieving efficient forgetting through one-round knowledge distillation. Theoretical analysis proves bounded parameter shifts and preserved convergence under FedAvg.

### Strengths
- The proposed quasi-competent teacher is a novel intermediate-level concept, clearly distinguished from prior “incompetent teacher” or “random-label teacher” paradigms. By penalizing the true-class prediction while preserving the inter-class structure, it effectively balances forgetting and knowledge retention.

- The evaluation metrics are well designed and cover both unlearning efficacy and efficiency, offering a comprehensive view of model performance.

- The idea itself is interesting and provides a fresh perspective on on-device federated unlearning through knowledge distillation.

- The theoretical analysis is mathematically sound and aligns well with the algorithm design

### Weaknesses
- The authors announce that this paper is in on-device federated learning settings in line 81. However, the experiment results under different numbers of clients are not clear. Could authors provide the results on the settings of more clients.

- The paper claims that FedQUIT is more efficient than MoDe and FedOSD. However, those baselines inherently require more local epochs or communication rounds. To ensure fairness, it would be better to compare all methods under the same computation or communication cost budget and observe their relative performance.

- The writing is generally clear, but the technical novelty needs to be emphasized more strongly. Knowledge distillation has already been used in federated unlearning (e.g., Wu et al., 2022a; 2022b), yet the paper only briefly mentions that those methods rely on historical records. The authors should provide a more detailed comparison highlighting the conceptual and technical differences between FedQUIT and these prior KD-based unlearning approaches.

### Questions
- How about the results under different numbers of client settings?

- If the budget is fixed, how about the performance gap?

### Soundness
3

### Presentation
3

### Contribution
3
