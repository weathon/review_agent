# DATS: Distance-Aware Temperature Scaling for Calibrated Class-Incremental Learning

- Avg Score: 3.60
- Decision: Reject
- Scores: 4, 6, 4, 2, 2

## Abstract
Continual Learning (CL) is recently gaining increasing attention for its ability to enable a single model to learn incrementally from a sequence of new classes. In this scenario, it is important to keep consistent predictive performance across all the classes and prevent the so-called Catastrophic Forgetting (CF). However, in safety-critical applications, predictive performance alone is insufficient. Predictive models should also be able to reliably communicate their uncertainty in a calibrated manner - that is, with confidence scores aligned to the true frequencies of target events. Existing approaches in CL address calibration primarily from a data-centric perspective, relying on a single temperature shared across all tasks. Such solutions overlook task-specific differences, leading to large fluctuations in calibration error across tasks. For this reason, we argue that a more principled approach should adapt the temperature according to the distance to the current task. However, the unavailability of the task information at test time/during deployment poses a major challenge to achieve the intended objective. For this, we propose Distance-Aware Temperature Scaling (DATS), which combines prototype-based distance estimation with distance-aware calibration to infer task proximity and assign adaptive temperatures without prior task information. Through extensive empirical evaluation on both standard benchmarks and real-world, imbalanced datasets taken from the biomedical domain, our approach demonstrates to be stable, reliable and consistent in reducing calibration error across tasks compared to state-of-the-art approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Distance-Aware Temperature Scaling (DATS), a post-hoc calibration method for Continual Learning (CL).
Instead of using a single global temperature for all tasks, DATS estimates the prototype-based distance between classes to infer task proximity and assign adaptive temperatures without task labels. By doing so, it aims to achieve more stable and reliable uncertainty calibration across sequential tasks. Experiments on standard benchmarks and biomedical datasets show that DATS effectively reduces calibration errors compared to existing calibration methods.

### Strengths
The paper is well written with clear logic and fluent expression, making it easy to read and understand. The motivation aligns well with the proposed method, and the presentation is visually appealing with informative figures and tables. The experiments are diverse and comprehensive, covering both benchmark and real-world datasets, and the results show consistently strong performance and clear advantages over existing methods.

### Weaknesses
The core assumption that class-prototype distances capture inter-task differences lacks theoretical support. The distance definition is heuristic and may be sensitive to noise. The experiments do not analyze the distance–temperature relationship, leaving the true contribution of distance unclear. The method is only tested on the ER baseline, limiting claims of generality, and no analysis is provided on the effect of memory buffer size.

### Questions
1. The core assumption of this paper is that inter-task distributional differences can be captured through class-prototype distances. However, in practice, class prototypes only reflect differences in the mean of the distributions, while variations at the level of distributional variance cannot be represented. Therefore, the motivation lacks theoretical justification and cannot be considered fully valid.
2. The distance definition in Equation (6) is somewhat heuristic, as using the minimum distance makes it overly sensitive to noisy or outlier classes. It is recommended to include an additional experiment comparing results obtained by replacing the minimum distance with average, Top-K, or weighted similarity measures.
3. The experimental section lacks an analysis of the distance–temperature relationship, since Equation (7) essentially adds a per-class temperature bias. The experiments do not examine how ($ d_c $) actually relates to the learned temperature, and the observed improvements might simply result from the parameter ($ \omega_c$ ). What would happen if ( $d_c$ ) were fixed rather than representing a distance? It is recommended to include such comparisons to address potential doubts.
4. The proposed method appears to be a plug-in approach with claimed general applicability, and it is evaluated on top of Experience Replay (ER). However, ER is a relatively classical CIL baseline with considerable room for improvement. If the proposed method were deployed on more advanced CIL frameworks, would it still demonstrate such a level of effectiveness?
5. Although the paper employs a memory buffer, the experiments do not include any analysis of the buffer size. It is recommended to add such experiments to assess the method’s sensitivity to different memory capacities.

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
3

### Summary
This paper concentrates on the issue of calibration in continual learning and propose a temperature based method, i.e., Distance-Aware Temperature Scaling (DATS), which combines prototype-based distance estimation with distance-aware calibration to infer task proximity and assign adaptive temperatures without prior task information. The main contributions are summarized as:

a. This paper demonstrates that calibration should move from task-agnostic to task-aware and adaptive re-calibration methods and expose the limitations of current task-agnostic data-centric approaches.

b. This paper proposes a distance-aware calibration framework that shifts the focus from data selection and injects task-awareness directly into the re-calibration algorithm.

c. Provide extensive experiments to prove the effectiveness of the proposed method.

### Strengths
a. This paper is well written and easy to follow.

b. It is interesting to consider the issue of calibration in continual learning.

c. This paper is technically solid.

### Weaknesses
a. My major concern is about the theoretical analysis. In Section 2, this paper introduces the definition of perfect calibration. I would like to know how the calibration results obtained by the proposed method compare, in theory, to those of a perfectly calibrated model. Please provide a detailed theoretical analysis.

b. Please compare this paper with [1] in details.

[1] Calibration of continual learning models. In CVPR 2024.

### Questions
N/A

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
1

### Summary
This paper introduces the concept of task-aware calibration into the field of continuous learning and proposes the DATS method. It employs category prototype distance as a proxy for task dissimilarity to achieve adaptive calibration without task labels. The article first reveals the limitations of existing approaches through robust empirical analysis and innovatively introduces two new metrics, ΔLECE and max ΔECE, providing fresh perspectives for evaluating calibration stability. The core of the approach lies in inferring task divergence through prototype distances, alongside designing a lightweight runtime inference mechanism. However, the method lacks validation of buffer prototype stability. Furthermore, the paper fails to adequately explain why greater task distances necessitate higher temperatures, and its fixed-threshold runtime calibration mechanism may falter when model accuracy is low. Its validity is primarily validated within the Experiential Replay (ER) framework, lacking experimental support across other continuous learning paradigms. Overall, this paper makes significant contributions in problem definition, methodological innovation, and evaluation systems. However, the reliability of its conclusions could be further strengthened through deeper validation of core assumptions and broader experimental testing.

### Strengths
1. The paper precisely identifies and substantiates a critical issue. Through experiments (as shown in Figures 3 and 4), it reveals that existing ‘task-agnostic’ approaches cause model confidence to drift between new and old tasks, providing empirical grounds for the necessity of ‘task-aware’ calibration.

2. The authors propose the DATS method, which ingeniously addresses core constraints of task-sensitivity in real-world deployment. It quantifies task differences through ‘category prototype distance’ and designs a lightweight test-time inference mechanism.

3. The paper innovatively introduces two metrics: ΔLECE (impact on the latest task) and max ΔECE (worst-case performance degradation), precisely quantifying calibration stability and reliability.

4. Validation on biomedical datasets exhibiting category imbalance robustly demonstrates the method's resilience and practical value in complex data environments.

### Weaknesses
1. The core of the method lies in the cosine distance of class prototypes at the penultimate layer. Due to catastrophic forgetting, feature representations of older classes may degrade, and the stability of prototypes in the calibration buffer has not been verified.

2. At low accuracy levels, the fixed-threshold calibration mechanism may misclassify the current task as another task, severely compromising calibration effectiveness.

3. The paper assigns higher temperatures to tasks farther from the current task, yet fails to explain why greater distance warrants increased temperature.

4. The paper primarily employs Experiential Replay (ER) as the continuous learning training method, lacking experiments on other types of CL algorithms.

### Questions
1. The author should explain how to ensure the stability and reliability of older class prototypes in the calibration buffer, given that feature representations are susceptible to catastrophic forgetting.

2. The author should give a detailed explanation about the fixed-threshold calibration mechanism prevent misclassifying the current task when its accuracy is low and its prototype is close to a different task's. A sensitivity analysis of the threshold may be needed.

3. What is the justification for assigning a higher temperature to tasks that are farther away? The author is suggested to provide theoretical or empirical explanation for this design choice.

4. The experiments are primarily based on Experience Replay (ER). Can the demonstrated effectiveness of the calibration method be generalized to other families of continual learning algorithms?

### Soundness
3

### Presentation
3

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
This paper introduces Distance-Aware Temperature Scaling (DATS), a method designed to improve model calibration in continual learning settings. The approach combines prototype-based distance estimation with distance-aware calibration to infer task proximity and assign adaptive temperatures that shape predictive distributions accordingly. The authors conduct empirical evaluations across both standard continual learning benchmarks and real-world datasets from the biomedical domain, demonstrating that their approach consistently reduces calibration error across tasks compared to baseline approaches.

### Strengths
- The proposed method is straightforward to implement and demonstrates clear effectiveness in improving calibration performance.
- The empirical evaluation is comprehensive, incorporating both standard continual learning benchmarks and realistic datasets from the biomedical domain, which strengthens the practical relevance of the findings.
- The paper is generally easy to follow.

### Weaknesses
- Tables 1 and 2 lack direct comparisons of classification performance metrics (e.g., accuracy) between the proposed method and baseline approaches. While DATS shows improved calibration, it is essential to verify that classification performance remains competitive with existing methods. Although NLL provides some indication of predictive quality, it does not directly substitute for standard accuracy metrics.

- The test-time calibration procedure described in Section 3.3 appears to require inference in a transductive setting, where optimal temperatures are determined collectively across a batch of test samples. This setup may inherently favor calibration performance. It would be valuable to demonstrate that the method maintains both improved calibration and competitive classification performance under a more challenging inductive setting, where test samples are processed individually rather than in batches.

My current rating is primarily influenced by these two weaknesses, I am willing to revise my score if they can be adequately addressed.

### Questions
- Poor calibration in continual learning often stems from continuously updating the final linear classifier, which can introduce bias toward recent tasks (as illustrated in Figure 3). How does the proposed method compare to simpler baselines that address this issue directly, such as GDumb [1] or other continual learning approaches that retrain the final classifier layer from scratch after each learning episode? Understanding this comparison would help clarify whether the calibration improvements are primarily due to the temperature scaling mechanism or could be achieved through alternative classifier alignment strategies.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the underexplored problem of uncertainty calibration in class-incremental continual learning (CL), where models must maintain both accuracy and reliable confidence estimates as new tasks arrive. Existing methods like RC and T-CIL use a single temperature across all tasks, ignoring task-specific variability caused by catastrophic forgetting, which leads to unstable calibration. The authors propose Distance-Aware Temperature Scaling (DATS), a post-hoc calibration framework that infers task proximity through prototype-based distance estimation and adjusts the temperature adaptively without requiring task labels at test time. By combining distance-aware calibration with prototype matching, DATS introduces task-awareness into temperature optimization. Extensive experiments on standard benchmarks (CIFAR-10/100, TinyImageNet) and imbalanced biomedical datasets (BloodCell, SkinLesions) show that DATS significantly reduces Expected Calibration Error (ECE) and maintains stability across tasks, outperforming task-agnostic baselines while remaining computationally efficient.

### Strengths
* Clear identification of prior limitations: The paper clearly diagnoses the limitations of existing continual learning (CL) calibration methods, which rely on a single, task-agnostic temperature, failing to account for inter-task variability caused by catastrophic forgetting.
* Improved calibration stability: The results demonstrate that DATS consistently reduces Expected Calibration Error (ECE) and avoids severe miscalibration across tasks—especially valuable in medical and safety-critical applications.
* Efficiency and practicality: Despite introducing distance-based components and task-tailored temperature scaling, DATS remains computationally efficient compared to T-CIL and simple to integrate as a post-hoc method.

### Weaknesses
* Misattributed source of miscalibration: The paper attributes inter-task miscalibration mainly to intrinsic differences in task uncertainty, but from the experiment results, the main source of the task uncertainty difference would be a well-known artifact of the class-incremental learning (CIL) setup itself rather than the task difference—specifically, the phenomenon that logits and confidence scores become biased toward the current task. Prior works (e.g., SS-IL, BiC, or bias-correction-based CIL methods) have shown that this stems from representation drift and imbalance in CIL learning, rather than from a true difference in task-level uncertainty. Hence, applying task-dependent temperature scaling might address this symptom. Hence, it would be better to first stabilize representation learning or balance logit distributions before introducing per-task calibration. Furthermore, from the experimental results in this paper, the previous tasks uncertainty gap look much smaller than the gap between previous and current tasks. This paper needs more persuasive experimental setup that can better exhibit the motivation of this paper.
* Assumption of batched test data. A major limitation is that DATS requires access to a set of test samples to compute representative class assignments and estimate the dominant class distances. This assumption does not hold in streaming or online inference scenarios, where test samples arrive one by one.
* Limited integration with CIL algorithms. The proposed method is evaluated as a post-hoc module, independent of the underlying CIL training dynamics. However, calibration behavior in CIL is tightly linked to how tasks are learned and replayed. Without jointly considering the training algorithm (e.g., ER, DER, or SS-IL variants), the effect of DATS may vary, making its generality uncertain.
* Novelty boundary. The novelty lies mainly in introducing a distance-aware temperature modulation rather than a fundamentally new calibration formulation. The method extends existing temperature scaling ideas with an additional distance-dependent weighting, which might be considered incremental.

### Questions
The proposed DATS framework assumes that a representative set of test samples is available to infer task proximity and compute adaptive temperatures. However, in many continual learning scenarios, test samples may arrive sequentially (e.g., in streaming or online inference). How would DATS handle such settings?

### Soundness
3

### Presentation
3

### Contribution
2
