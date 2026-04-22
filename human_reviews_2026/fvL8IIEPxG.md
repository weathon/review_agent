# Forget Forgetting: Continual Learning in a World of Abundant Memory

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 8, 6, 4

## Abstract
Continual learning (CL) has traditionally focused on minimizing exemplar memory, a constraint often misaligned with modern systems where GPU time, not storage, is the primary bottleneck. This paper challenges this paradigm by investigating a more realistic regime: one where memory is abundant enough to mitigate forgetting, but full retraining from scratch remains prohibitively expensive. In this practical "middle ground", we find that the core challenge shifts from stability to plasticity, as models become biased toward prior tasks and struggle to learn new ones. Conversely, improved stability allows simple replay baselines to outperform the state-of-the-art methods at a fraction of the GPU cost. To address this newly surfaced trade-off, we propose Weight Space Consolidation, a lightweight method that combines (1) rank-based parameter resets to restore plasticity with (2) weight averaging to enhance stability. Validated on both class-incremental learning with image classifiers and continual instruction tuning with large language models, our approach outperforms strong baselines while matching the low computational cost of replay, offering a scalable alternative to expensive full-retraining. These findings challenge long-standing CL assumptions and establish a new, cost-efficient baseline for real-world CL systems where exemplar memory is no longer the limiting factor.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Traditionally, continual learning has been studied under the memory-restricted setting (i.e., completely exemplar-free or only limited buffer sizes allowed). Authors argue that this is unrealistic under modern standards: compute, and not memory-costs, are the main drivers for the high costs related to model training. Thus, authors study the setting where larger-than-common buffer sizes are used, and where instead the compute availability is the driving factor for comparison. They find that, as the memory buffer grows and thus allows storing more samples of prior tasks, the challenge in CL shift from forgetting to plasticity. Models are stable wrt old tasks, but not plastic enough to new tasks. To alleviate, they propose using stochastic weight averaging and resetting weights with relatively little gradient activity. They study their method, termed Weight Space Consolidation, under the class-incremental setting on CIFAR100, ImageNet-100, and text-based TRACE benchmark, where competitors are outperformed.

### Strengths
Strengths:
+ novel take on memory buffers in CL
+ simple and effective method to improve plasticity under performance-stable regimes
+ image and text modalities tested
+ easy-to-follow description of algorithm

### Weaknesses
- l 311 to 319: this is a central part of the proposed method, but no experimental results support this thesis
- Paper claims less compute overhead from their method, but no tabular results indicating the overhead versus naive replay are given
- if we can store samples of old tasks, then we could, in turn, learn longer task sequences. Paper should integrate longer task sequences (e.g., TinyImageNet split into 20 or 40 tasks)
- Full re-training is often used as comparison, but missing from the tables
- Its unclear how "from scratch" is interpreted. Is that an entirely fresh model that's trained on increasing unions of memory buffer? Then, no knowledge would be transferred by design
- Hyperparameter tuning on ResNet32 and then using tuned params on ResNet18 seems strange (l 364). Please elaborate on this.
- I do not agree with the definition of post-hoc merging versus in-training merging (l292ff). Post-hoc merging would be, in my understanding, merging after Training on a task, which is thus before a next task -- which is still "during training".

### Questions
I have several questions/ideas for improvement
- You use a rank-based approach --> where are the ranks (i.e., matrix ranks) used here? Do you mean "ranking based approach"?
- Equation 2 and 5 both use $\alpha$ for different jobs
- Equation 2: how is alpha annealed (l 232)
- Equation 5: describe that the [l] are the weights to be reset
- Figure 1: are the memory sizes per class or total? Additionally give percentage
- L 460: Q versus q (in the top-q%)
- Figure 3 and 405ff: how is VRAM usage measured? Do not modern frameworks pre-allocate as much GPU memory as possible (i.e., aiming at 100% GPU memory utilization)

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
Memory-based methods in Continual Learning focus on learning a sequence of tasks with help of a buffer, which store samples from previous tasks and use them when learning the current tasks. Normally, this buffer is limited in the amount of samples they can store due to some constraints of the environment. In this paper, the authors challenge this constraint mainly because storage costs much less than training time. However, as we increase the samples we store, models could face a different challenge, and instead of having forgetting (too much plasticity) this shifts to a high stability and not learning the current task enough. Considering all this, this paper proposes Weight Space Consolidation, which combines a reset strategy to improve plasticity and weight averaging to enhance stability. Experiments on multiple modalities show good performance of the proposed method as they increase the memory budget. Also, multiple ablation experiments help understand why and how the method performs.

### Strengths
- The paper is well motivated and written. The authors raise the challenge that, in current systems, memory is not the most important constraint, as GPU time is more costly. Presenting references and numbers, the paper encourages researchers not to focus on minimising memory size.
    - However, there are scenarios where having a small memory is required, such as when resource constraints are imposed or when the privacy of previously learned data is an issue. These are not mentioned or explained in the paper.
- Along with raising the challenge and presenting a new scenario, the authors analyse the limitations of naively increasing memory size and propose a new approach to increase plasticity and enhance stability in this new scenario.
- The experiments and results clearly help to demonstrate what is discussed in the text. Experiments across multiple benchmarks and modalities provide robustness to the results, and ablation experiments help understand the method's limitations and provide further insights into how it works.

### Weaknesses
- A common approach to increase the plasticity of memory-based methods is to concatenate buffer samples with current-task samples at the batch level. This is different from what is shown in the paper, where the full batch is sampled from the intersection of the current data and the buffer. As shown in the paper, this last approach suffers from plasticity because it is least likely to sample data from the current task; however, by concatenating at the batch level, there is no plasticity problem.
    - How does this sampling method affect the scenario, results and the proposed method? 
    - Concatenating at the batch level can be more costly (in terms of GPU time), but it may mean not keeping a copy of the model in memory, which can increase the batch size.
- Figure 2b is unclear. The orange line makes it impossible to see the blue lines and compare them.

### Questions
- Why not compare against better memory-based methods, like for example:
    - Buzzega, Pietro, et al. "Dark experience for general continual learning: a strong, simple baseline." Advances in neural information processing systems 33 (2020): 15920-15930.
        - It has been shown to scale well with the number of tasks.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors challenge the assumption of traditional CL. Instead, they argue that, instead of the storage, the GPU time is the main bottleneck. They investigated the scenario where memory is sufficient enough to mitigate forgetting, but full retraining from scratch remains the main challenge. As the authors have discovered that models become biased toward prior tasks and struggle to learn new tasks, they propose Weight Space Consolidation, a lightweight method that combines rank-based parameter resets to restore plasticity with weight averaging to enhance stability.

### Strengths
+ The paper investigates a new paradigm/scenario that challenges the previous assumptions. Such new thinking is always valuable.
+ The mathematical formulations are rigorous and I discovered zero mistakes there.
+ The new paradigm is not just an "assumption," they have evidence (Section 3) to empirically demonstrate that the new assumption is valid.

### Weaknesses
- While the mathematical formulation is rigorous, the theoretical foundation on why this new approach would work is lacking.
- The proposed approach reads like A (rank-based parameter resets) + B (weight averaging). A + B isn't always necessarily bad, but both A and B are the results of prior work, so what's the technical contribution here with this approach?

### Questions
See "Weaknesses." What are the technical contributions (s) of this approach, if the A and B components were previously proposed and employed by existing works?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Brief Summary: The paper tackles the task of continual learning. The authors suggest gpu compute cost is the main constraint and find the problem is with plasticity where bottleneck is on struggling to learn new tasks. To address this, authors propose Weight Space Consolidation involving rank-based parameter resets and weight averaging. Experiments are conducted on both image-classificaiton tasks (cifar-100) and llm instruction tuning (trace) showing improvement at significantly reduced costs.

### Strengths
Pros:

1. The overall point about gpu compute constraints being more than storage is good. Exploration of such middle-ground strategies makes sense to me. The core idea of reseting certain parameters for plasticity also is interesting.

2. In the high-sample regime, the proposed method almost always outperforms existing baselines by 2-3 points. 

3. The paper has nice ablation experiments such as comparing replay with reset (table 3), reset strategies (table 5). The plasticity loss experiment (Fig2) in particular suggest the issue with lower plasticity with increasing size of past examples. The additional experiments in the supplementary are appreciated.

### Weaknesses
Cons:

1. My main concern is with the framing of the paper for continual learning and its potential application. The point about GPU cost being dominant with storage is fine, but the main problem is that the data itself might not be available in the first place. So storage was never the real issue, it is access to data. For a practical example, assume we have llama-3.2b instruct model but we don't know what data was used in the base to instruct training. Here, the authors are essentially assuming we have access to the underlying data, which is not the case. A few other points:

(i) While storage cost is not a problem, the cost of downloading setting up s3 buckets high throughput speed for gpu access are all real costs. These need to be highlighted. 

(ii) If the authors are motivating it based on cost, some experiments on exact cost saved for practical reference should be provided.

This is not to say the original point about GPU vs storage cost isn't correct, but that storage is not the only factor, data access itself is a big one. The entire argument essentially dilutes the author's novelty.

2. It seems a naive full-fine-tuning baseline is missing in Table 2 ? Difference between proposed method and full-fine-tuned on the same set would be very helpful. 

3. Authors primarily explore task-based paradigm only, but it seems the method might be more impactful in task-free settings [Ref1]. 

4. For LLM datasets, authors only consider TRACE dataset, so tough to know the generalizability.. Given that authors are exploring image-based and llm-based separately, it might be worth repeating experiments on multi-modal datasets as well such as on VisCOLL [Ref2]. 

5. (Minor) Some qualitative visualizations would be very helpful.

6. (Minor) For Trace experiments, the training hyper-parameters are underspecified. What is the fine-tuning method for llama-3.2-1B? Some experiments using LoRA adaptors would also be interesting. 

---
[Ref1]: Aljundi, Rahaf, Klaas Kelchtermans, and Tinne Tuytelaars. "Task-free continual learning." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11254-11263. 2019.

[Ref2]: Jin, Xisen, Junyi Du, Arka Sadhu, Ram Nevatia, and Xiang Ren. "Visually grounded continual learning of compositional phrases." arXiv preprint arXiv:2005.00785 (2020).


---

Overall Rating: 4/10

The main framing of gpu vs storage cost is not really novel. While the authors get 2-3 points improvement in high-sample regime, it is unclear how interesting that is. Authors miss key experiments such as full-fine-tuning baselines, only considering task-based not task-free, and only one trace dataset for llm experiments.

### Questions
Q1. Can plasticity be measured as a metric on an eval set? Currently, plasticity loss is provided in the graph.

### Soundness
2

### Presentation
3

### Contribution
3
