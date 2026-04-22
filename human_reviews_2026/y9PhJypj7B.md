# Using Temperature Sampling to Effectively Train Robot Learning Policies on Imbalanced Datasets

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Increasingly large datasets of robot actions and sensory observations are being collected to train ever-larger neural networks. These datasets are collected based on tasks and while these tasks may be distinct in their descriptions, many involve very similar physical action sequences (e.g., ‘pick up an apple’ versus ‘pick up an orange’). As a result, many datasets of robotic tasks are substantially imbalanced in terms of the physical robotic actions they represent. In this work, we propose a simple sampling strategy for policy training that mitigates this imbalance. Our method requires only a few lines of code to integrate into existing codebases and improves generalization. We evaluate our method in both pre-training small models and fine-tuning large foundational models. Our results show substantial improvements on low-resource tasks compared to prior state-of-the-art methods, without degrading performance on high-resource tasks. This enables more effective use of model capacity for multi-task policies. We also further validate our approach in a real-world setup on a Franka Panda robot arm across a diverse set of tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the issue of data imbalance in multi-task robot learning datasets, where certain action primitives (e.g., 'pick-and-place') are over-represented compared to others. The authors propose a simple data sampling strategy, termed "temperature-based sampling," to mitigate this imbalance during training. The method adjusts the sampling probability of each task based on its dataset size and a temperature parameter, $\tau$. The core of their proposed implementation is a cosine warming schedule for $\tau$ (from 1 to 5), which gradually increases the sampling rate of low-resource tasks as training progresses. The authors validate their method through a toy experiment (sparse parity), simulation experiments on artificially imbalanced versions of RoboCasa and Libero (for both training-from-scratch and fine-tuning foundation models), and a real-world experiment on a Franka Panda arm using a custom-collected dataset. The central claim is that this method substantially improves performance on low-resource tasks without degrading performance on high-resource tasks, thereby making more effective use of model capacity.

### Strengths
- The proposed method is exceptionally simple to implement, requiring only a few lines of code to modify an existing data loader. This low barrier to entry makes it a practical and appealing technique for researchers and practitioners.
- The authors have made a commendable effort to validate their approach across a variety of settings: a controlled toy problem, two different simulation benchmarks (RoboCasa, Libero), two training paradigms (from-scratch and fine-tuning a VLA), and a real-world hardware setup. This demonstrates a degree of diligence in testing the method's applicability.
- The paper includes several useful ablation studies that investigate the method's robustness to model size, the degree of data imbalance, and the choice of temperature schedule. These studies provide valuable insights, for instance, showing that the benefits of the method increase with the severity of the imbalance ratio and that a "warming" schedule is more effective than a "decay" schedule.

### Weaknesses
- The core idea of temperature-based sampling to address data imbalance is not new. The paper itself acknowledges very similar prior work in the domain of multilingual natural language processing (e.g., Wang et al., 2020b; Choi et al., 2023). The contribution is therefore the application of a known technique to a new domain (robotics). While valuable, this is an incremental contribution. Furthermore, the idea of data reweighting is not new to robotics; for example, the RDT-1B model already utilized a form of data weighting in its training protocol to balance datasets. The paper fails to adequately differentiate its contribution from these existing concepts.
- While the inclusion of a real-world experiment is a strength, its execution is weak. The real-world results in Figure 5 only compare the proposed method against "random" sampling. In simulation (Figure 4), the authors compare against "upsample" and "ReMix" baselines, but these are conspicuously absent in the hardware experiments. To make a convincing claim, the method must be shown to outperform other SOTA re-balancing techniques on a physical robot. Without this comparison, the real-world results only show that the method is better than the most naive baseline, which is insufficient.
- The abstract and main body repeatedly claim that the method improves low-resource performance "without degrading performance on high-resource tasks". However, the authors' own results in Figure 4 contradict this. For the high-resource "Pick/Place" task, random sampling achieves a 0.21 success rate, while the proposed temperature sampling method achieves only 0.17. This is a clear degradation in performance. While the overall average improves due to gains elsewhere, the claim of no degradation is factually incorrect based on the provided data and should be revised.
- The simulation results in Figure 7 are highly suspect. For both the 300:50 and 1000:50 imbalance ratios, the success rate for the "Pick/Place" task—the high-resource task—is abysmal across all methods (well below 5%). A policy trained on thousands of demonstrations for a single skill family should achieve much higher performance. This suggests a fundamental issue with the experimental setup, the base policy's capacity, or the evaluation protocol itself. If the model fails to learn the task for which it has the most data, the results on all other low-resource tasks become difficult to interpret and trust.
- The method's effectiveness is highly dependent on the choice of the temperature schedule (start $\tau$, end $\tau$, shape). The authors state they found the cosine warming schedule from $\tau=1$ to $\tau=5$ through a "thorough hyper-parameter search", but provide no details on this search. How does the optimal schedule change for different datasets, model architectures, or imbalance ratios? This introduces a significant tuning burden, undermining the "simplicity" of the method. The authors acknowledge the limitation of a "fixed schedule" but downplay the difficulty this presents for a user trying to apply the method to a new problem.
- The proposed method requires the dataset to be segmented into discrete tasks with known demonstration counts ($|D_i|$). This is a strong assumption that does not hold for large, aggregated, "in-the-wild" datasets like the full Open X-Embodiment (OXE) or DROID, where skills are intermixed and not cleanly labeled. The paper positions itself as a solution for such large-scale datasets but fails to address how its core requirement would be met in those scenarios, limiting its real-world applicability.

### Questions
- The choice of the cosine warming schedule from $\tau=1$ to $\tau=5$ appears critical to the method's success. How sensitive is the final policy performance to these specific hyperparameter choices? For a new dataset with a different imbalance ratio, would a user need to perform another extensive hyperparameter search, and if so, does this not diminish the claimed simplicity of the method?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses action-primitive imbalance in large robot datasets, which harms generalization on low-resource tasks (LRTs). The authors propose a simple "Temperature-based Sampling" strategy, using a "cosine warming" schedule for $\tau$. Experiments on toy tasks, in simulation (RoboCasa, Libero), and on a real Franka robot show the method improves LRT performance.

### Strengths
- The problem of action-primitive imbalance is critical for the community as it moves toward large-scale datasets.
- The method is simple, efficient, and easy to implement. It also appears more stable than complex baselines like ReMix.
- The evaluation is comprehensive, covering a toy task, simulation (training from scratch and fine-tuning), and real-world hardware validation, strongly supporting the claims.
- Solid ablation studies validate the impact of schedules, model sizes, and imbalance ratios.

### Weaknesses
- The paper claims no performance degradation on high-resource tasks (HRTs), but Figure 4 shows a significant drop for the "Pick/Place" HRT (0.21 to 0.12). This trade-off is not discussed.
- The method requires pre-segmented tasks with known counts ($|D_i|$), making it hard to apply directly to unsegmented "in-the-wild" datasets.

### Questions
- Q1.  Can you explain the significant HRT performance drop in Figure 4 (Pick/Place task)? Why did this trade-off only appear in the RoboCasa experiment and not in the Libero or real-world ones? Is this trade-off inherent?
- Q2.  The ReMix baseline performed very poorly (Fig. 4). Can you provide details on its tuning? How can we be sure this isn't just due to sub-optimal hyperparameters?
- Q3.  How can this method be applied to unsegmented "in-the-wild" datasets, given its reliance on known task counts ($\|D_i\|$)? Are there heuristic approaches (e.g., language clustering) that could work, beyond waiting for future skill segmentation research?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a sampling method for training robotic models under imbalance datasets. They propose temperature sampling, which increasingly up weights under represented tasks towards the end of training, and show that this helps the success rate of those under represented tasks.

### Strengths
- The method they propose is simple to understand and easy to implement compared to other works, and is shown to work well in practice in two simulated robotic environments and one real robot set up. 
- The authors provide good ablation studies to show how the method works with different model size, different datasets, and different annealing process.

### Weaknesses
- It is unclear to me how the authors arrived at this particular form of temperature sampling. For example, instead of using |D|^(1/t), why not exp(|D|)^(1/t) for example? What about other forms? It would be nice to see how different forms for temperature sampling impacts policy performance.
- it would be nice to see whether this method works with larger and more realistic datasets. Currently, the authors hand pick a subset of Robocasa & LIBERO for their experiments. It is unclear in the paper how this subset and its composition was chosen, and how the results would differ when the dataset is chosen differently. On the other hand, it is much more natural to use a dataset like DROID, which is collected by human teleop and has natural data imbalance. Would training a VLA (e.g. OpenVLA, Pi0, etc.) over the DROID dataset, with temperature sampling, be better than vanilla random sampling?
- As the authors acknowledged, temperature sampling requires pre-specifying the number of steps to train. However, the optimal checkpoint for a different datamix may be different. For example, when we downweight HRT, we may need to train for longer to get good performance on HRT. The authors only compare on single checkpoint in the paper.
- lack of good baseline comparisons: the authors only compare against on baseline, ReMix, and it seems weird that it always perform worse than random sampling. Is there any reason why? The authors mention that ReMix is sensitive to hyperparameters: did you tune the hyperparameters when reporting its performance? If so, to what extent was the tunning effort? The paper would also benefit from additional baselines, such as [1]
- Temperature based sampling seems to significantly hurt performance on high resource tasks

[1] DataMIL: Selecting Data for Robot Imitation Learning with Datamodels

### Questions
- the authors mentioned that ReMix is sensitive to hyperparameters? Did you tune the hyperparameters when reporting its performance? If so, to what extent was the tunning effort? Why does it almost always perform worse than random sampling?
- Why do you need to use different base models for the Robocasa dataset vs the LIBERO dataset?
- How did you pick the checkpoint (40k)? Do you get similar results (i.e. ranking of different methods) if you pick other checkpoints?

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
The authors explore the problem of choosing sub-dataset sampling frequencies when training imitation learning policies on imbalanced robot datasets. They study in particular this problem for cases where the imbalance arises due to the existence of different splits of the data, each split corresponding to a particular action primitive, with certain splits containing more data than others. This assumption is common in many large robot datasets where pick-and-place primitives are far more common than other types of robot skills, like wiping or throwing. The authors analyze two methods for choosing sampling frequencies based on two prior papers that studied this problem for multilingual LLMs: (1) upsampling low-resource data splits at the beginning of training and annealing it towards the end [1], and (2) increasing the sampling frequency of low-resource datasets towards the end of training [2]. The authors find that method (2) works better on a testbed RoboCasa simulation experiment, and on a real robot experiment. They also when proposing their method run it on a toy sparse parity task, where similar trends as in their main experiments are demonstrated. They also evaluate on the LIBERO simulation and report results of fine-tuning a foundation model.

[1] Li, Tianjian, et al. "Upsample or Upweight? Balanced Training on Heavily Imbalanced Datasets." arXiv preprint arXiv:2410.04579 (2024).

[2] Choi, Dami, et al. "Order matters in the presence of dataset imbalance for multilingual learning." Advances in Neural Information Processing Systems 36 (2023): 66902-66922.

### Strengths
(1) The problem tackled of picking sub-dataset sampling frequencies is an important problem in robotics, especially as more recent work ventures to train large robot policies on more heterogeneous data sources.

(2) The related works section of the paper is thorough

(3) The overall method makes sense, and experiments show better downstream task performance for their proposed sampling method in both real and sim.

(4) The authors compare against relevant baselines, including Re-Mix and the simple baseline of upsampling low-resource datasets and fixing this sampling ratio throughout training.

### Weaknesses
(1) The particular problem statement itself is not thoroughly justified. The problem statement in question is how to pick sampling frequencies when there exist subsets, each with different amounts of data for different action primitives. While action primitive decomposition is indeed one way of splitting data, it is not the only one, and there are many other axes upon which there exist low-resource and high-resource splits, such as language instructions, camera viewpoints, lighting, environment diversity, etc. By focusing just on action primitives, the authors missed the chance to study the robotic dataset sampling problem in its full scope.

(2) The authors mention related work on automatic skill decomposition via learned latent skills, but they do not actually run these methods (or any other method, e.g., based on parsing language instructions) to automatically segment robot datasets into skill-oriented subsets. Instead they assume that such decompositions already exist. However for many real world datasets such skill-oriented decompositions do not exist.

### Questions
(1) While I agree that pick-place tasks are overrepresented in large robot datasets, it would be good to include some evidence of this, e.g., by citing previous work.

(2) Section 5.2 is missing why the authors think the technique of upsampling low-resource datasets later in training works better than other methods.

(3) The authors justify choosing cosine warming after a hyper-parameter search and say it outperforms linear/exponential warmup/decay. The authors should include the actual ablation table or curves in the main text or appendix.

### Soundness
1

### Presentation
2

### Contribution
1
