# On the Generalization Capacities of MLLMs for Spatial Intelligence

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 6, 4, 8, 6

## Abstract
Multimodal Large Language Models (MLLMs) that directly process RGB inputs for tasks like 3D localization and navigation have shown remarkable potential. However, we argue that these ``RGB-only'' approaches are fundamentally flawed in their ability to generalize across cameras. By ignoring camera parameters, they entangle an object's physical properties with the camera's perspective, creating an irresolvable ambiguity. We show this leads MLLMs to overfit to the training camera distribution, rather than learning true and generalizable 3D geometric principles. To address this, we propose Camera-Aware MLLM framework for spatial MLLMs. It learns generalizable spatial reasoning by: (i) injecting camera intrinsics via a dense embedding that conditions each visual token; (ii) introducing a camera-aware data augmentation strategy that synthetically varies camera parameters, forcing the model to disentangle camera properties from scene content; and (iii) distilling geometric priors from a 3D vision foundation model. Extensive experiments demonstrate that camera-aware MLLMs substantially outperform their naive counterparts, particularly in cross-camera generalization tests on spatially-grounded tasks, indicating that camera-awareness is not only beneficial but also a prerequisite for robust and generalizable spatial intelligence in MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the limitations of "RGB-only" multimodal large language models in spatial reasoning tasks, particularly their lack of generalization across camera viewpoints. The authors attribute this to the absence of camera intrinsics, which they argue introduces geometric ambiguity and leads to overfitting. To address this, the paper introduces a Camera-Aware MLLM framework that incorporates dense camera embeddings, camera-aware data augmentation, and geometric prior distillation from a 3D vision model.

### Strengths
(1) This paper is well-motivated. It identifies a fundamental limitation in existing MLLMs for spatial tasks, based on both theory and empirical analysis. What’s more, the cross-camera generalization experiments are particularly compelling and clearly demonstrate the value of the proposed method.

(2) The proposed Camera-Aware MLLM framework is evaluated through a series of experiments. The results show consistent improvements over camera-agnostic baselines, suggesting that incorporating camera parameters can enhance the robustness of spatial MLLMs.

### Weaknesses
(1) Limited discussion on real-world applicability. The paper would benefit from a discussion about the real-world scenarios where camera intrinsics may be unknown, unreliable, or hard to estimate depth. 

(2) The ablation study in Table 5 lacks the “Only Prior Distillation” setting, which I consider important for understanding the individual contribution of geometric prior supervision.

(3) It remains unclear whether the improvements stem from the proposed model itself or simply from the availability of additional 3D geometric information. It could be clarified by comparing with methods that directly utilize depth maps or incorporate depth-aware features into the model.

### Questions
See Weaknesses

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
4

### Summary
This paper identifies a fundamental limitation in RGB-only Multimodal Large Language Models (MLLMs) for spatial reasoning tasks: by ignoring camera intrinsic parameters, these models cannot disambiguate geometric properties from camera perspectives, leading to poor cross-camera generalization. The authors propose a Camera-Aware MLLM framework that addresses this issue through three key novelties: (i) dense camera ray embeddings that condition each visual token on its corresponding camera geometry, (ii) camera-aware data augmentation that synthetically varies camera parameters during training, and (iii) geometric prior distillation from a pre-trained monocular depth estimation model (UniDepth v2).

### Strengths
1. The paper demonstrates that camera-agnostic MLLMs fail to generalize across different camera parameters, highlighting a critical but overlooked limitation in current spatial reasoning approaches.
2. The proposed framework combines three complementary components to resolve the identified ambiguity, with experimental results showing improvements in cross-camera generalization on multiple spatially-grounded tasks.

### Weaknesses
1. Table 5 does not include "Prior Distillation only" or "Geom Aug + Prior Dist (w/o Ray Emb)", making it hard to isolate the contribution of camera ray embeddings. Given that UniDepth v2 is pretrained on 10M+ RGB-depth pairs, the geometric prior distillation likely contributes the majority of improvements, but this is never quantified.

2. Outdated and incomplete baselines:
* Table 2/3/4 use Gemini-1.5-Flash/Pro (Feb 2024) instead of Gemini-2.5 or later versions
* Table 4 is missing Qwen2.5-VL-7B and Qwen2.5-VL-72B, which are natural baselines given the authors' use of Qwen2.5-VL-3B
* No comparison with Claude 4.5 Sonnet, GPT-5 or other recent strong MLLMs

3. Table 1 and Figure 6 only test uniform image rescaling (0.8×, 1.2×), which is an extremely limited simulation of camera variation. Real cross-camera generalization should involve different camera models with varying focal lengths (wide-angle vs telephoto) and different aspect ratios and principal point locations.

4. Tables 3-4 show that the proposed camera-aware design provides little to no benefit on standard spatial reasoning benchmarks. The results suggest camera-awareness is only beneficial for a narrow subset of spatially-grounded tasks, not a fundamental requirement for spatial reasoning.

5. Missing computational cost analysis (inference time, memory overhead of running UniDepth v2)

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper argues that for spatial intelligence MLLMs need camera intrinsics as input to the model — prior spatial MLLMs do not use this input and thus are not robust to changes in camera parameters. The paper first quantifies this problem by showing lack of generalization on 3D object detection on ScanNet when models do not use camera information as input. Next, it proposes to inject this information by encoding the cameras rays with the visual features; doing augmentations; and training on wide varieties of datasets. In experiments, it shows that their method with camera information works better than existing methods, and the ablations show that all proposed changes help

### Strengths
- The paper is well-written and easy to follow
- The idea in general makes sense and the results look good
- The experiment section is well designed and answers most research questions raised by the paper

### Weaknesses
No major weaknesses per se, just some comments regarding the writing and presentation of the work

1. About Table 1: This table nicely shows that prior VLMs do not benefit from multiple dataset training and are susceptible to zooming-in/out operations. I was expecting the paper to show at the end that this problem is now resolved with the proposed technique and the paper does show it but it wasn’t straightforward to make this connection. Specifically, Table-1, Figure-6 and Table-5 are connected by this evaluation — yet Table-1 uses top 31 classes while figure-6 and table-5 uses 20 classes. Additionally, an explicit mention of this connection between Table-1 and Figure-6 and 5 will help. 
2. (Minor comment): The section 3.3 describes the scale-depth ambiguity in a lot of words which is an extremely well-known computer vision phenomenon. The authors can consider making this section short and crisp
3. Table-2 is missing highlights on best working method numbers, making it harder to parse quickly. Additionally, some additional descriptions of the datasets and benchmarks will help. For eg. It is unclear what low, medium and high in Table-2 mean.

### Questions
N/A -- no major questions or concerns

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a camera-aware MLLM framework that integrates camera embedding, geometric prior distillation, and camera-aware augmentation into the existing MLLM architecture.  
Extensive experiments are conducted on spatial reasoning benchmarks for MLLM models, and ablation studies for each proposed component are provided.

### Strengths
- The writing is clear, and the logical flow is easy to follow.  
- The benchmark coverage is extensive.  
- Incorporating ray embedding, camera-aware augmentation, and distillation from 3D models is a reasonable and well-motivated approach.

### Weaknesses
- Figure 1 may be misleading. The caption states, “There is no way to know unless I know the camera intrinsics!” However, even with camera intrinsics, it is still not analytically possible to determine the exact 3D location from a single RGB image. It would be helpful to clarify that priors are still needed to estimate the 3D position.  
- The performance reported in Tables 2 and 3 is comparable to that of other spatial MLLMs, which makes the paper’s claims less convincing.

### Questions
- The formatting of Tables 1 and 3 could be improved. It would also be beneficial to include citations for the methods compared in the tables.

### Soundness
2

### Presentation
3

### Contribution
3
