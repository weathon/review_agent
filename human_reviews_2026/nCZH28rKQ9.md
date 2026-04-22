# SpaceVista: All-Scale Visual Spatial Reasoning from mm to km

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
With the current surge in spatial reasoning explorations, researchers have made significant progress in understanding indoor scenes, but still struggle with diverse applications such as robotics and autonomous driving.  This paper aims to advance all-scale spatial reasoning across diverse scenarios by tackling two key challenges: 1) the heavy reliance on indoor 3D scans and labor-intensive manual annotations for dataset curation; 2) the absence of effective all-scale scene modeling, which often leads to overfitting to individual scenes. In this paper, we introduce a holistic solution that integrates a structured spatial reasoning knowledge system, scale-aware modeling, and a progressive training paradigm, as the first attempt to broaden the all-scale spatial intelligence of MLLMs to the best of our knowledge. Using a task-specific, specialist-driven automated pipeline, we curate over 38K video scenes across 5 spatial scales to create SpaceVista-1M, a dataset comprising approximately 1M spatial QA pairs spanning 19 diverse task types. While specialist models can inject useful domain knowledge, they are not reliable for evaluation. We then build an all-scale benchmark with precise annotations by manually recording, retrieving, and assembling video-based data. However, naive training with SpaceVista-1M often yields suboptimal results due to the potential knowledge conflict. Accordingly, we introduce SpaceVista-7B, a spatial reasoning model that accepts dense inputs beyond semantics and uses scale as an anchor for scale-aware experts and progressive rewards. Finally, extensive evaluations across 5 benchmarks, including our SpaceVista-Bench, demonstrate competitive performance, showcasing strong generalization across all scales and scenarios. Our dataset, model, and benchmark will be released at our [project page](https://mm2km.github.io/).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a large-scale and ambitious effort to achieve unified spatial reasoning across six orders of magnitude. The authors build SpaceVista-1M, a dataset of 1 million QA pairs from 38K videos across 19 task types and 5 spatial scales, an impressive engineering and data-collection effort. They further propose SpaceVista-7B, a multimodal model enhanced with scale-aware LoRA-like experts and progressive reward learning to handle cross-scale knowledge conflicts. Finally, they release SpaceVista-Bench, a carefully curated benchmark for real-world evaluation, showing that SpaceVista-7B achieves strong performance across multiple spatial reasoning benchmarks.

### Strengths
The dataset spans scales from mm to km, addressing a key gap in prior spatial reasoning work.
The LoRA-based scale experts and progressive reward mechanism effectively reduce scale conflicts.
Consistent improvements across diverse benchmarks demonstrate solid performance and generalization.
The work integrates the dataset, model, and benchmark in one framework, a significant contribution to spatial reasoning research.

### Weaknesses
Model novelty mainly lies in adaptation (LoRA experts) rather than in a fundamentally new architecture.

Data quality verification and annotation consistency are underexplained.

The scale of the dataset is impressive, and I respect the authors’ hard work. However, the tasks for evaluating or learning spatial reasoning capabilities remain somewhat superficial. The current tasks seem to focus only on object localization, manipulation planning, distance estimation, camera localization, and area estimation across different scenarios. When discussing spatial reasoning capabilities, more sophisticated scenes, such as maze-solving or optimization problems like the traveling salesman problem or vehicle routing problem, would be expected. I would like to hear from the authors whether such tasks are included or, if not, their reasons for excluding them.

The constructed data appears to consist mostly of multiple-choice or standard grounding and distance estimation tasks, and may lack open-ended spatial reasoning data, which is currently in short supply.

### Questions
The paper is quite long, and I may have missed some details. Below are my questions:

Do the 1M annotations constructed by the authors include questions with open-ended responses?

How is the annotation accuracy of SpaceVista-1M and SpaceVista-Bench quantitatively ensured?

Can the scale router dynamically adapt during inference?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper’s main contribution is two-fold: a SpaceVista-1M dataset and SpaceVista-7B model to address the scaling issue of VLM reasoning. Specifically, SpaceVista-1M incorporates various different scales of data for training. SpaceVista-7B proposes LoRA-based scale-aware experts to avoid knowledge clashes during training. The paper trains SpaceVista-1M on Qwen-VL as well as their own model with various ablations on the vision backbone, number of experts, and the RL algorithms. Results show that their dataset brings improvements on most benchmarks, and their architectures achieve better results than all other specialized open-source models.

### Strengths
Strengths
I find the experiments are quite extensive. Figure 5 is a very interesting visualization and informative proof that the scale-expert is acting the way it should be. Table 3 and Table 5 are great additions of ablations that allow us to better understand SpaceVista-7B. 

The authors trained Qwen-VL 7B on their dataset, which I think is a great addition to understand whether the dataset itself is a valid contribution. I would love to see perhaps another opensource model trained on SpaceVista-7B but the current results seem sufficient to me in this aspect.

### Weaknesses
Questions

Even though the dataset is a great contribution, I have some questions regarding the ability to generalize after training on SpaceVista-1M. 

One thing I noticed is that open-sourced specialized models, such as VG LLM-4B seem to perform better on other data like VSI-bench compared to vanilla SpaceVista-7B trained on SpaceVista-1M without the experts. This could potentially be explained by saying that VSI-bench and SpaceVista-1M still have a domain gap even though the scale is covered within this range. It is also likely that there are some scales that are within 1e-3m to 1e+3m but not part of the dataset. It would be great to understand whether VLM trained on SpaceVista is just better at the multiple scales seen during the dataset or can actually interpolate/extrapolate to new scales. One way of potentially testing this could be Outdoor scenes with the scale slightly zoomed in/out to mimic a scale in between automatic driving and remote sensing and test the results on that. 

It would also be great to test these out of distribution data mentioned above on different numbers of experts. This allows us to understand whether the experts are essentially fitting to a certain scale range or are they making the model more generalizable. Do we have to linearly increase the number of experts when the scale ranges increase?

The authors also missed out on the comparison against VLM-3R (https://arxiv.org/html/2505.20279v2), which is significantly better on the VSI-Bench compared to all the methods the authors listed in the paper (60.9). I wonder whether this model would also generalize to the SpatialVista benchmark.

Another observation is that Qwen2.5-VL-7B without training on SpaceVista-1M actually performs better on MMSI-Bench. I was wondering whether there may actually be some overfitting to SpaceVista and therefore the model starts to perform worse in other benchmarks. 

I have also noticed that the Vanilla model is already performing better than Qwen2.5VL-7B trained on SpaceVista-1M without scale, Dinov3, or experts. I was wondering if you have any intuition on where this improvement comes from?

Some other minor points:
I find Figure 1 and 3 slightly confusing. For Figure 1 it took me a little while to understand the dotted lines are not part of the architecture or model contribution but comments to describe the contribution. Fig 3 (a)’s arrangement is a little messy with the evaluation benchmark in the middle, legend on bottom left, and the construction pipeline going clockwise.

### Questions
To summarize, I think the paper could be improved with the two following questions:

- Understanding whether training with multiple scales allows generalization beyond these scales. Can the model interpolate/extrapolate the scale to ranges beyond the training data and perform better than other specialized/general models?

- Are the experts just fitting to certain ranges that exist in the training dataset range? Does the number of experts required increase linearly with the scale range?

- Comparison against VLM-3R seems to be missing

Overall, even though the paper could be benefitted from answering these issues, I find the current contribution (dataset and model) to be sufficient. I am therefore providing an initial rating of 6.

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
5

### Summary
The paper aims to enable all-scale visual spatial reasoning, spanning from mm-level objects to km-level outdoor and aerial scenes. They introduce ~1M spatial QA pairs curated using a specialist-driven automatic pipeline, along with an evaluation set with manual annotations to ensure reliability. On the modeling side, the paper proposes incorporating DINO/VGGT features for VLMs and scale-aware expert modules to enhance performance. For RL training, the paper introduces a progressive reward design that uses scale as an anchor for progressive rewards.

### Strengths
* The paper is well written and easy to follow.
* Spatial reasoning is a hot topic. Covering a wider range of scales can potentially enable more applications.
* The dataset and benchmark curation effort are valuable and appreciated.

### Weaknesses
* Many components in the paper have already been discussed in prior spatial reasoning works. Important related work such as SpatialVLM is missing in both the comparison and the discussion. The data pipeline (specialist-driven QA generation, depth estimation, etc.) is highly related to SpatialVLM. Also, the use of DINO/VGGT features also overlap with existing works (e.g., Cambrian-1), so the novelty is less clear.
* It is not fully convincing whether the model truly learns spatial understanding across scales, or mainly picks up dataset statistics. In Table 5, the specialized model does not clearly outperform general models. It would help to evaluate on a split of *unusual* or *OOD* scale cases.
* Similarly, the scale-anchored RL training raises the concern that the model may rely on scale priors rather than genuine spatial reasoning. If an object appears at an unusual scale, would the model still behave correctly?
* The impact on general VLM capabilities is unclear, especially after adding DINO/VGGT features. These types of features are often avoided in VLMs because they can degrade general reasoning performance. If the model loses general ability and mainly follows template-style spatial cues, one could argue that running the specialist pipeline alone may suffice.
* The ablation on scale-experts suggests limited effectiveness, making it unclear whether they are necessary.
* The comparison between SFT and RL training variants on SpaceVista (Table 2) does not clearly control for training data type/amount (e.g., with or without CoT, # training samples). A more detailed ablation would make it easier to judge whether the reward design truly improves performance.

### Questions
Please see the weakness section.

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
The paper addresses the limitations of current multimodal models in spatial reasoning, which tend to overfit to indoor environments and lack scalability across diverse real-world settings. It introduces a unified framework for all-scale spatial reasoning—from millimeter- to kilometer-level understanding—covering applications in robotics, manufacturing, autonomous driving, and aerial sensing.

To overcome the dependency on manual 3D annotations and narrow scene coverage, the authors propose SpaceVista-1M, a large-scale dataset of 1 million spatial question–answer pairs collected from 38,000 videos across five spatial scales and 19 task types. The dataset is generated through a specialist-driven automated pipeline that integrates multiple vision models to capture geometry and depth cues.

They further present SpaceVista-7B, a multimodal large language model enhanced with scale-aware experts and a progressive training strategy that uses spatial scale as a guiding signal to mitigate cross-scale conflicts. For reliable evaluation, they build SpaceVista-Bench, a manually verified benchmark with precise real-world annotations.

Experiments demonstrate that SpaceVista-7B generalizes effectively across diverse environments and scales, outperforming prior spatial reasoning models and establishing a new foundation for large-scale, all-scale spatial intelligence.

### Strengths
1. The paper clearly identifies the limitation of current spatial reasoning models that overfit to predominantly indoor scales. Its proposed all-scale formulation is well-motivated.

2. The SpaceVista-1M dataset is comprehensive, spanning five spatial scales and nineteen task types, with dense geometric annotation and QA formats suitable for both supervised fine-tuning and reinforcement learning. The accompanying SpaceVista-Bench provides reasonable ground truth for evaluation.

3. The model integrates three technically sound components—dense self-supervised visual features, LoRA-like scale experts with a routing mechanism to mitigate cross-scale conflicts, and anchor-based rewards for progressive training—all validated through ablation studies.

4. Extensive experiments across five benchmarks (VSI-, STI-, SPAR-, MMSI-, and SpaceVista-Bench) demonstrate consistent improvements, with SpaceVista-7B achieving competitive or superior performance to strong closed-source systems such as GPT-5 and Gemini-2.5 despite its smaller 7B parameter scale.

### Weaknesses
1. The dataset generation pipeline heavily depends on specialized perception models for supervision; errors from these models may propagate. The paper would benefit from quantitative assessments of noise levels and their effects across tasks and scales.

2. While evaluations cover multiple benchmarks, the analysis remains surface-level—some strong proprietary models perform unexpectedly poorly, and fine-grained diagnostics (e.g., per-task or per-scale performance, error taxonomy) are limited to the appendix.

3. Ablation studies focus on architectural variations but do not sufficiently examine the contribution of different data subsets (e.g., scales or task families) to overall cross-scale generalization.

### Questions
1. What is the router’s entropy across layers/scales during inference? Any evidence of specialization collapsing to a single expert?

### Soundness
3

### Presentation
3

### Contribution
3
