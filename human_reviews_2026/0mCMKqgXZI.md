# Resolving Interference (RI): Disentangling Models for Improved Model Merging

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Model merging has shown that multitask models can be created by directly combining the parameters of different models that are each specialized on tasks of interest. However, models trained independently on distinct tasks often exhibit interference that degrades the merged model's performance. To solve this problem, we formally define the notion of 'Cross-Task Interference' as the drift in the representation of the merged model to its constituent models. Reducing cross-task interference is the key to improving merging performance. To address this issue, we propose our method 'Resolving Interference (RI)', a light-weight framework which disentangles expert models to be functionally orthogonal to the space of other tasks, thereby reducing cross-task interference. RI does this whilst using only \textit{unlabeled auxiliary} data as input (i.e., no task-data is needed), allowing it to be applied to under data-scarce scenarios. RI consistently improves the performance of existing merging methods by up to 10% and generalization to unseen domains by up to 2.3%. We also find RI to be robust to the source of auxiliary input while being significantly less sensitive to tuning of merging hyperparameters.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces Resolving Interference (RI), a lightweight framework to reduce cross-task interference in model merging, defined as merged models’ representation drift from constituent task-specialized models. RI disentangles expert models into functionally orthogonal subspaces using only unlabeled auxiliary data (no task data needed), enhancing existing merging methods.

### Strengths
1.The paper is well organized and well written.

2.The authors present a well-motivated approach with a simple, easy-to-follow framework.

3.It conducts numerous experiments, validates the experimental results on models of various series and sizes, and covers a wide range of evaluation tasks

### Weaknesses
1. **Although the paper claims to address the scenario where no model-related data are available for improving model fusion, its contribution remains unclear.**
   This is because the proposed method still requires training with additional data, and the source of this auxiliary data is restricted. The comparison settings in the paper do not convincingly isolate the true contribution of the proposed approach, as the method still depends on external data. Moreover, Figure 4 shows that not all auxiliary datasets lead to performance gains, which raises questions about the selection of these datasets (e.g., whether they are similar to the task-specific data). It is recommended that the authors include an analysis of the correlation between the auxiliary and task-specific data, and add experiments in Tables 1 and 2 using *noisy* or *irrelevant* auxiliary data (as in the purple items of Figure 4) to verify that the proposed method can indeed improve model fusion performance **without relying on task-related data**.


2. **The explanation for why weight averaging fails in the in-domain setting is not convincing.**
   The authors attribute this to the small scaling coefficient. However, in the 20-task setting, the coefficient for TA is also small, yet RI still performs effectively. Therefore, this explanation does not sufficiently account for the observed phenomenon. The authors are encouraged to provide a deeper analysis or additional experiments to investigate this behavior, rather than offering a potentially incomplete conclusion.

3. **The computational and memory costs introduced by training the proposed method should be more rigorously discussed.**
   Since the method requires an additional training phase, its practical feasibility needs further evaluation. The paper should include detailed comparisons of GPU memory usage, training time, and computational cost, as well as an analysis of scalability when the number of tasks increases (as the method likely requires more training steps in such cases).

### Questions
1. Would the proposed method fail or degrade in performance when applied to text classification or text generation tasks, where the output representations may be more complex than in the current settings?

2. Although the method does not require task-specific data, the authors could still use such data as auxiliary input to further validate its effectiveness and compare it with other training-based approaches. This would strengthen the paper’s contribution. I am also curious about the method’s performance when in-domain data are used as auxiliary data.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the concept of cross-task interference in model merging: the merged model underperforms its constituent experts.
It proposes a training-based merge method that defines the interference as the KL divergence between the merged and expert models and trains on unlabeled auxiliary data to reduce this cross-task interference. The authors demonstrate that RI improves standard merging across 8/14/20-task benchmarks on ViT-B/32, B/16, and L/14.

### Strengths
- This paper tackles merging interference by matching the expert output with the merged model output and proposes a new Resolving Interference (RI) method.  
- The paper conducts up to 20 datasets experiments and presents improved results on CLIP across various merging methods.  
- It includes a detailed analysis of the use of an auxiliary dataset, which requires no access to labels.

### Weaknesses
- The weaknesses of RI loss:  
  - It only applies to classification tasks and ignores generative tasks and LLMs (which are dominant): all experiments use CLIP-style ViT encoders and vision classification heads.  
  - The algorithm explicitly requires the set of heads $\{h_i\}_{i=1}^N$. This assumption does not hold for many generative settings (e.g., segmentation, detection, diffusion, LLMs).  
  - It cannot handle different task types, which are naturally supported by other merging techniques (e.g., summarization, math, and code tasks can be merged via Task Arithmetic).  
  - It introduces an unnecessary pipeline procedure: it adds a gradient-based adaptation stage for **each expert** (RI), instead of a single gradient-free merge. By default, it uses 2,500 steps (320k samples) on auxiliary data **per expert**, which incurs high training cost, whereas other merging techniques (e.g., Task Arithmetic, TIES) are training-free.  
  - Assuming N tasks, RI requires $O(N)$ forward passes per step per expert; across all experts and S steps, this becomes $O(N^2S)$  head forwards plus $O(NS)$ backbone forwards.  
    - When the number of tasks is very large, the loss incurs high computational complexity.  
    - As model size scales, each forward pass becomes more expensive, making RI even costlier.  
    - In Figure 3, the paper shows an “elbow” around 2,500 steps. With a maximum task number of N = 20 and S = 2,500 RI steps, this requires >1M head forwards and >50k backbone forwards in total.
- The weaknesses of auxiliary data:  
  - The use of arbitrary auxiliary data is justified empirically rather than theoretically. The authors show that even Gaussian noise yields a noticeable performance boost. Why does directly using noise improve model performance? A noisy dataset may drive expert models to produce meaningless outputs, which should not benefit model training.  
  - It implicitly assumes that the auxiliary data comes from the same task type. It remains unclear whether auxiliary data can still be used when merging different task types.
- The weaknesses of the experiments:  
  - Averaging + RI (Ours) consistently underperforms plain Averaging on both ViT-B/16 and ViT-L/14 across 8, 14, and 20 tasks.  
  - The evaluation does not cover most SOTA baselines, such as Surgery, TwinMerging, CatMerging, and LoTMerging.
- Typo: Line 281: “78.7KnOTS”

### Questions
See Weakness

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
The paper proposes resolving interference (RI) to reduce cross-task interference for model merging, with the potential to utilize auxiliary input. The proposed method outperforms reported baselines when merging ViT with 8, 14, and 20 tasks. Overall, the writing and organization of the paper are good.

### Strengths
1) The paper proposes a distillation method to disentangle the output of a task-specific model through the definition of cross-task interference.
2) The proposed method is examined in vision tasks following the model merging recent setup with 8, 14, and 20 tasks, and it showed that it can improve the existing model merging method.
3) The experiment and ablation are well thought out and analyzed to investigate the cross-task interference and the proposed RI.

### Weaknesses
1) In the abstract, the 10% mentioned is confusing. The improvement made by this paper against SOTA is roughly within 2%. This is misleading.
2) Missing analysis of the data-less model merging SOTA "WUDI" merging. By how much can RI improve WUDI?
3) The proposed method is not evaluated on NLP tasks, which are commonly studied in almost all recent model merging methods. Is RI applicable to NLP tasks?
4) The computational requirements should be analyzed, and jointly optimizing the task vectors simultaneously could be very costly.

### Questions
Just curious. Why $\tau_i^*$ need to be optimized separately for each task? Can't we optimize a single $\tau ^ *$ for all tasks?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a model merging framework enhanced by a Representation Independence (RI) Loss. By leveraging auxiliary data and introducing a regularization constraint, the method aims to mitigate task interference during model merging, thereby improving robustness and generalization. The core idea is to enforce task-irrelevance constraints between merged representations, allowing the final model to better retain knowledge from each task. Experiments on multi-task benchmarks demonstrate consistent improvements.

### Strengths
**Clear innovation**: The RI Loss provides a novel regularization mechanism to reduce task interference, offering a fresh perspective for model merging research.

**Simple and practical**: The method can be seamlessly integrated into existing training setups as a plug-and-play module.

**Empirical validation**: Extensive evaluations across multiple benchmarks show promising gains.

Solid motivation: The task-relatedness perspective provides a sound theoretical rationale for reducing interference in merging.

### Weaknesses
**1，Dependency on auxiliary data**: The effectiveness appears tied to the availability and distribution of auxiliary data, yet its limitations are not thoroughly examined.

**2，Missing key baselines**: Recent strong merging methods (e.g., Wudi-Merging) are not included in comparisons.

**3，Limited large-model experiments**: The study focuses on moderate-scale models, leaving the applicability to large-scale vision-language or language models unverified.

**4，Incomplete presentation**: The main figure only explains the RI Loss without clearly visualizing the merging pipeline, which may confuse readers unfamiliar with the topic.

**5，Insufficient training-cost analysis**: More detailed benchmarking against alternatives (e.g., Adamerging, model surgery, naive distillation, multi-task learning) would strengthen the claims.

### Questions
1，How does RI Loss perform when the distribution of auxiliary data significantly differs from that of the target tasks? Are there failure cases or ablation studies? Does the amount of auxiliary data matter?

2，Will the authors include comparisons with recent baselines such as Wudi-Merging?

3，Can the approach scale to large models (e.g., LLMs, CLIP-based multi-task settings)? Any preliminary results or discussion?

4，Could the paper include a clearer pipeline diagram to illustrate the full merging process rather than only the RI Loss component?

5，Could the authors provide quantitative training-time comparisons with Adamerging, model surgery, naive distillation, and multi-task learning?

Minor Issues / Typos

Line 58: Incorrect citation — Task Arithmetic in the Tangent Space should be NeurIPS 2023, not 2024.

### Soundness
2

### Presentation
2

### Contribution
3
