# Unified 3D Scene Understanding Through Physical World Modeling

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Understanding 3D scenes requires flexible combinations of visual reasoning tasks, including depth estimation, novel view synthesis, and object manipulation, all of which are essential for perception and interaction. Existing approaches have typically addressed these tasks in isolation, preventing them from sharing a common representation or transferring knowledge across tasks. A conceptually simpler but practically non-trivial alternative is to unify these diverse tasks into a single model, reducing different tasks from separate training objectives to merely different prompts and allowing for joint training across all datasets. In this work, we present a physical world model for unified 3D understanding and interaction 3WM, formulated as a probabilistic graphical model in which nodes represent multimodal scene elements such as RGB, optical flow, and camera pose. Diverse tasks emerge from different inference pathways through the graph: novel view synthesis from RGB and dense flow prompts, object manipulation from RGB and sparse flow prompts, and depth estimation from RGB and camera conditioning, all zero-shot without task-specific training. 3WM outperforms specialized baselines without the need for finetuning by offering precise controllability, strong geometric consistency, and robustness in real-world scenarios, achieving state-of-the-art performance on NVS and 3D object manipulation. Beyond predefined tasks, the model supports composable inference pathways, such as moving objects aside while navigating a 3D environment, enabling complex geometric reasoning. This demonstrates that a unified model can serve as a practical alternative to fragmented task-specific systems, taking a step towards a general-purpose visual world model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes 3WM (3D World Model), a unified framework for 3D scene understanding and interaction based on a probabilistic graphical model (PGM) formulation. Unlike task-specific 3D models, 3WM encodes multimodal scene elements (RGB, optical flow, camera pose) as nodes and performs diverse tasks—such as novel view synthesis, depth estimation, and 3D object manipulation—as different inference pathways through the same graph.

### Strengths
1. The paper introduces a conceptually elegant unification of multiple 3D vision tasks under one probabilistic and autoregressive modeling framework.
2. The local random-access sequence modeling design and pointer-token formulation are interesting and potentially valuable for scalable multimodal 3D reasoning.
3. Both quantitative and qualitative experiments have been conducted in sufficient detail.

### Weaknesses
1. The presentation quality is inconsistent. The Introduction section fails to clearly and explicitly highlight the main contributions of the paper, making it difficult for readers to grasp the motivation and significance of the work.
2. A teaser figure should be included in the Introduction to provide an intuitive overview of the proposed framework and help readers quickly understand the key idea and workflow.
3. The paper lacks comprehensive ablation studies, which are necessary to fully evaluate the generalization ability and effectiveness of the proposed method.

### Questions
I am not very familiar with the research areas of Novel View Synthesis and World Model, so my current evaluation mainly focuses on the presentation quality and writing clarity of the paper. I will update my assessment after considering feedback and technical insights from other reviewers.

### Soundness
3

### Presentation
2

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
This paper aims to propose a method to unify several important tasks of 3D scene understanding. To this end, the authors propose a probabilistic graphical model, where nodes represent multimodal scene elements such as RGB, optical flow, and camera pose. The combinations of these nodes therefore enables different tasks, including novel view synthesis, 3D object manipulation and depth estimation without task-specific training. Experiments show that the unified model outperforms the respective task-specific baselines. The authors also discover that the model supports composable inference pathways, therefore can help other tasks such as navigation and amodal completion.

### Strengths
1. Novelty of the problem. I like the story that the authors want to develop a unified model for different 3D understanding tasks.

2. Novelty of the method. The method to unify the different 3D understanding tasks is also novel and interesting to me.

3. Good evaluation results. Experiment results show the developed unified model generally outperforms the task specific models for novel view synthesis, 3D object manipulation and depth estimation, although it has not been trained for the specific tasks. The results are impressive to me.

### Weaknesses
1. Efficiency. It is good to see the authors have developed such a unified model with good performance. It would be better if the authors can also show the efficiency of their model for different tasks compared with the respective task-specific baselines. If the model are going to be applied to embodied AI systems for navigation, for example, the efficiency does matter.

2. Application scenarios. In section 5 of the paper, the authors discuss about the emergent geometric reasoning abilities of the model, and have provided several qualitative examples. However, it increases the paper's contribution and significance if the authors can provide quantitative results on the tasks, such as navigation, and amodal completion, as they mentioned.

### Questions
Generally I like the paper. I would recommend the authors to address my concerns in the weakness section during the rebuttal.

### Soundness
3

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
The proposed physical world model aims to enable 3D scene understanding enabling improvements in visual reasoning tasks. 
The method proposes a probabilistic graphical model operating on scene information such as RGB patches, optical flow, and depth maps, implemented as a 7B autoregressive transformer performing next-token prediction. The model is trained on Big video dataset and 3D vision benchmarks such as Co3D, ScanNet++, and others. The model enables novel-view synthesis, 3D object manipulation, and depth estimation. The paper presents strong results for each of the tasks on standard test benchmarks demonstrating improvements over existing methods. The authors also contribute a new dataset of 100 image pairs for 3D object manipulation which could be useful for future research.

### Strengths
The paper is well-motivated and proposes a simple yet effective method enabling strong results for Novel-view synthesis, 3D object manipulation, and depth estimation tasks. The paper expresses the ideas with enough detail for understanding each component and its implementation. The evaluation is comprehensive with existing methods on standard benchmarks for the respective tasks. Composable inference pathways for 3D scene understanding as shown in Figure 6 is a strong result and could be useful for downstream robotics tasks. This can be used for complex reasoning and planning in 3D scenes.

### Weaknesses
- The results do not show strong understanding of lighting and appearance understanding based on the results in Figure 3 (the specular highlights on the objects) and 4 (the macbook), as the objects seem to have baked in shading.
- The object manipulations are limited to rigid transformations and do not show compositional understanding of objects such as stacked objects. The demonstrated examples are limited to the new dataset and would require more evidence to support the claim of improved 3D object manipulation capabilities.
- The paper should include a discussion on the limitations of the model to better inform the reader about the scope of the contributions.
Suggestions for improvement of the presentation:
- I recommend the authors to include a short introduction to each section as an overview of the content.
- Introduction can include a concise summary of the contributions of the paper.

Minor:
- L115: an video --> a video

### Questions
- Can the model perform transformation of non-rigid objects such as cloth? If so, how can the transformations be performed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper presents an autoregressive generative model (3WM) for RGB images and optical flow based on an interesting "pointer-content representation" that allows a single autoregressive model to generate image or optimal flow output from sparse or dense image or optical flow input. Effectively, the paper trains a single model to learn to approximate a diversity of types of conditional distributions on image and optical flow tokens, via randomization of the training set. The paper demonstrates the resulting model on several tasks including novel-view synthesis, 3D object manipulation, and depth extraction.

### Strengths
The pointer-content representation that allows for a single model to approximate a number of conditional generation tasks on the joint space of (RGB image, optical flow, RGB image) is significant. The approach of training a model to approximate diverse types of conditional distributions via randomly selecting and re-reorder "pointer-content" pairs is a useful contribution. The qualitative comparisons to baselines shown in the paper are compelling.

### Weaknesses
The use of "graphical model" as the formal framework for describing the work seems somewhat ill-suited, since there are no conditional independences in the model (and not surprisingly, the the paper includes no graphical models visualized). See "Questions" for connections to other formal frameworks for generative modeling that seem more aligned with the "pointer-content" representation.

### Questions
1. The specific way that training examples were generated from the video dataset seems central to the model. It would help to provide more detail on the type of (latent, observed) pairs that were generated synthetically from the video dataset, and the distribution of these different types of examples at training time?
2. Since the autoregressive model represents in principle all possible conditional distributions, it would be interesting to study how self-coherent it is. It would be interesting to compare (i) the approximate amortized inferences obtained by sampling from the trained autoregressive model, with (ii) the idealized conditional distribution that is defined by sampling from the unconditioned joint distribution and performing rejection sampling. One way to do this is via estimating expected symmetrized KL divergence on synthetic data generated from the model itself as in "AIDE: An algorithm for measuring the accuracy of probabilistic inference algorithms" to compare rejection sampling versus the variational approximation learned by the model (this essentially boils down to sampling from the joint model, then evaluating the conditional probability of the latent tokens given the observed tokens under the autoregressive model). Are there certain classes of queries where it is more self-consistent than others? Can you use the joint model to do a little model-based Monte Carlo (importance sampling) to improve upon proposals sampled from the autoregressive amortized approximation of the conditional distribution?
3.  Interesting connection: The "pointer-content" representation doesn't fit well with standard graphical models, but is more closely related to the address-value representation used in the Gen probabilistic programming language (see e.g. section 4.1 of Gen: A General-Purpose Probabilistic Programming System with Programmable Inference) where random choices are assigned labels ("addresses") and generative models are probability distribution on dictionaries that map addresses to values.

### Soundness
3

### Presentation
3

### Contribution
4
