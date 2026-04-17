# Seeing Like Humans: Task-Driven Token Reduction for Accelerated ViT in Robotic Navigation

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
In robotics, vision is critical for enabling agents to perceive and interact with their environment. Recent advancements in vision models, particularly Vision Transformers (ViTs), have shown remarkable performance in pure vision tasks like object recognition and scene understanding, showing great potential for robotic applications such as object navigation. However, their computational cost grows quadratically with respect to the number of tokens, posing significant challenges for real-time deployment on resource-constrained robotic platforms. To enhance ViT efficiency in robotic tasks, we propose a biologically-inspired token reduction framework that dynamically allocates computation to task-relevant regions in images while neglecting those irrelevant regions for efficiency. Our method introduces two key components: (1) a task-driven spatial attention mechanism that selectively prunes redundant tokens based on the current task, and (2) a temporal feature reusing module that reuses stable visual features across frames to minimize redundant computation. Together, these components enable the visual perception model to focus only on relevant regions, significantly improving inference speed. Experiments show that our method notably reduces inference time in object navigation tasks without significant performance degradation. Additionally, it enables practical ViT deployment on edge devices such as the Jetson Orin (high-performance GPU) and Raspberry Pi 4B (lightweight CPU), achieving 56.5 FPS and 2 FPS, respectively. This represents a 1.5~3× speedup over standard ViTs, making real-time robotic vision more feasible.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a human-like, biological-inspired token reduction method for using ViTs in robotic navigation that accelerates inference. The authors propose to reuse temporal token reusing that reuses previous frame via cross-attention and skips recomputing redundant tokens. The authors also propose task-driven token pruning, which predicts task relevance from the robot’s current state and goal and drops tokens that are irrelevant to the final objective. The experiment results show up to ~1.5× speedup on Orin and nearly 3× speedup on Raspberry Pi with only a small drop in success rate and SPL compared to full baseline methods.

### Strengths
1. From a perspective of pruning to what human sees is quite interesting and novel. The cross-attention between the current frame’s tokens and the previous frame’s final features, plus a learned confidence score that blends attention entropy, cosine similarity, and max attention to decide which tokens can skip recomputation, is quite similar to what human sees through time.
2. The policy-aware pruning is also an intuitive design for this specific robot navigation task. 
3. The paper provides qualitative and quantitative interpretability, where the paper provides visualizations that show that the model keeps tokens around the goal object and navigable path while discarding irrelevant background.
4. The experiment results show acceleration while maintaining an acceptable drop compared to the full baseline method. And the added modules are lightweight.

### Weaknesses
1. The current evaluation seems to be narrowly focused on ObjectNav in Habitat with imitation-learned policies. Would the same pruning technique also work for other embodied tasks, such as manipulation, dynamic obstacle avoidance, and outdoor navigation?
2. The temporal reuse block assumes adjacent frames are similar enough that features can be carried forward. Will the solution still work under tasks involving fast egomotion, motion blur, or abrupt viewpoint shifts that are common on robots in the wild?
3. The baselines (EViT, DynamicViT, ToMe) are standard token pruning methods from vision, as they do not have task-aware pruning for the task paper's focus. Will this create a bias in the comparison in the experiment?

### Questions
Please see the weakness section.

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
This paper aims at addressing the quadratic computational cost of ViTs, which limits the real-time use on edge devices. Specifically, the authors draw inspiration from human vision and introduce two key ideas: (1) task-driven token pruning that selectively processes image regions relevant to the current task/state, while discarding irrelevant visual tokens, and (2) temporal token reuse that reuses stable visual features across frames to avoid redundant computation in sequential video input. The experiments show that the method achieves up to 1.5×–3× speedup with minimal loss in navigation performance compared to standard ViTs.

### Strengths
- The paper directly focuses on embodied AI and practical robotics settings, which sounds useful. The discussion of efficiency constraints and control-loop implications also make sense to me.
- The experiments are conducted on multiple hardware platforms, such as Jetson Orin and Raspberry Pi 4B, which demonstrate meaningful real-world value.
- Based on the provided table, the proposed method achieves significant acceleration (up to 3×) without sacrificing navigation success rates close to baseline models. The results look promising.
- The paper is well written and easy to follow. The figures are well-plotted and informative which can make readers quickly understand the core ideas.

### Weaknesses
- My major concern is that all of the experiments are limited to ObjectNav in Habitat, which show limited diversity on the target task. Such experiments make it questionable that how the proposed method can be generalized to other manipulation or multi-object reasoning tasks.
- Another concern is that the authors only provide FPS and success rate, which are too narrow to evaluate on the target task. The authors should also provide the evaluation results on perception quality, such as visual feature fidelity and localization accuracy.
- The authors only compare with some pruning-based and merging-based methods (e.g., EViT, ToMe, DynamicViT) but lack of the comparisons with some other ViT acceleration approaches, such as some linear attention methods (e.g., Performers [1]) and compression-based methods (e.g., MobileViT [2]).
- The hyperparameters of "temporal token reuse threshold" and "task-driven pruning threshold" directly determine how aggressively tokens are reused or pruned, which could very sensitive to the balance between computational savings and perceptual fidelity. However, the authors naively use 0.5 throughout the paper without further ablation. It would be insightful to see how these hyperparamters affect the model performance.


[1] "Rethinking Attention with Performers", ICLR 2021
[2] "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer", ICLR 2022

### Questions
- Could the authors provide the evaluation on some other tasks with different manipulations or on multi-object reasoning tasks?
- Could the authors provide the scores of visual feature fidelity and localization accuracy in the table?
- Could the authors compare more related acceleration models/approaches?
- Could the authors provide the ablation study of the hyperparameters, such as "temporal token reuse threshold" and "task-driven pruning threshold"?

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
This paper proposes a token reduction framework to optimize effficiency of Vision Transformers (ViTs) for robotic navigation. The authors propose a biologically-inspired token reduction framework with two components: (1) a temporal token reuse module that identifies and reuses stable visual features across consecutive frames to avoid redundant computation, and (2) a task-driven token pruning module that selectively retains tokens relevant to the current navigation task and robot state while discarding irrelevant regions.

The method is evaluated on object navigation tasks in the Habitat simulator using a ViT+RNN architecture with DeiT-Tiny as the visual backbone. The temporal reuse module is inserted after the 3rd ViT layer and uses cross-attention to propagate features from the previous frame, while the task-driven pruning module is placed after the 6th layer and uses a lightweight MLP to predict a "perception focus" based on robot state and task. Unlike prior token reduction methods that use fixed pruning ratios, this approach dynamically adapts token selection based on temporal redundancy and task requirements, allowing a single model to generalize across multiple navigation goals without retraining.

The results demonstrate substantial speedups: 1.5× on Jetson Orin GPU (56.4 FPS vs. 37.4 FPS baseline) and nearly 3× on Raspberry Pi 4B CPU (1.95 FPS vs. 0.69 FPS baseline), while maintaining navigation performance with only minor drops in Success Rate (50.05% vs. 52.10%) and SPL (21.46% vs. 22.03%). The method reduces computational cost to 5.00 GFLOPs compared to the baseline's 8.99 GFLOPs.

### Strengths
1. The paper is well-written and the approach is simple, intuitive, and works reliably
2. Compared to prior toke-pruning methods the proposed approach performs the best by being efficient in using less FLOPs and maintaining high success rates as presented in main results which is quite promising.
3. The analysis and ablations presented in section 4.3 are  well thought through and demonstrate benefits of using the proposed approach over prior methods. It also clearly demonstrates that even after trading off compute through pruning the method maintains high success rates.

### Weaknesses
1. The choice of adding the token reuse module at 3rd layer of ViT and token pruning module on 6th layer of the ViT are not very well motivated and explained in the paper. Can authors elaborate on how this decision was made? Also I believe ablating the choice of layers for each of these modules and adding those results to the paper will be quite valuable.
2. Similarly the choice of threshold values and lambda are not very well supported through relevant ablations. It would be great if authors can add justification for them and add supporting experiments.
3. The paper also doesn’t mention how the output tokens from ViT are converted and passed to the RNN policy. I am curious why authors choose a RNN for policy model instead of using a transformer and I would be interested in seeing same comparison of all methods presented in table 1 with a ViT+Transformer policy. I am concerned that if the ViT outputs are compressed using a spatial compression layer for RNN input that could be causing issues with other methods like ToME and DynamicViT. Operating directly on visual encoder output the transformer policy for navigation could give more insights into this. However this will incur more FLOPs as now we need to maintain K frames in transformer context for reasonable performance as there is no hidden states of past interactions.

### Questions
The contributions in the paper show meaningful performance and efficiency improvements. There are a few decision decisions for which ablations are missing or not justified due which I recommend a borderline reject. I am happy to increase my ratings if authors can address the concerns I mentioned and add supporting experiments

### Soundness
3

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
3

### Summary
This paper proposes a framework to improve the efficiency of Vision Transformers (ViTs) for robotic navigation. The method introduces:

1) Temporal Token Reusing : cross-frame attention to reuse redundant features across consecutive frames.

2) Task-Driven Token Pruning : dynamically removing spatial tokens irrelevant to the robot’s current task or goal.

The authors evaluate the approach on Habitat’s ObjectNav benchmark, showing speedups on Jetson Orin (GPU) and Raspberry Pi 4B (CPU), with small drops in navigation success.

### Strengths
Making visual robotic policies run on-device in real time in embodied settings is an important problem to solve with a general solution that doesn’t assume too much about the problem setting. The approach here shows speed gain, outperforming other pruning baselines that were studied on the task.

### Weaknesses
1) Results are shown only for one task with a specific assumption (the environment will be largely static for this method to work), the empirical evidence would be much stronger if the authors show applicability of method at least on manipulation and Imagenav tasks - both have similar assumptions as the task in the paper (static environment, small localised motions).

2) Even on the single task that is used to study the method, SR/SPL are below the baseline (albeit slightly).

3) I would want to see results on a baseline that is essentially the temporal version of ToMe (trivial modification to reuse tokens across frames if they satisfy the same closeness criteria as that applied in ToMe at the single frame level currently) for fairness

4) I would also compare performance to a smaller model that is flop matched to the proposed method to check if the pruning method proposed outperforms that at least.

5) ToMe is a training free approach without much overhead, so the authors should report how much is the overhead from their approach because they have to train two stages on top of the baseline - since if it is comparable compute to training the baseline itself then that is not very meaningful

6) I see no change in performance in table 3b for temporal ablations when varying the threshold - maybe we need to ablate the range further to get some insights from this ablation?

7) Writing needs to be improved to make the paper more self contained and to make the experimental section easier to parse. The experimental section does not define (sp) and (temp) used in the main results table anywhere. There should also be a description of what the different task subsets (base/long) are actually testing and how they differ.

### Questions
Listed above.

### Soundness
3

### Presentation
2

### Contribution
2
