# Compositional 4D Dynamic Scenes Understanding with Physics Priors for Video Question Answering

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
For vision-language models (VLMs), understanding the dynamic properties of objects and their interactions in 3D scenes from videos is crucial for effective reasoning about high-level temporal and action semantics. Although humans are adept at understanding these properties by constructing 3D and temporal (4D) representations of the world, current video understanding models struggle to extract these dynamic semantics, arguably because these models use cross-frame reasoning without underlying knowledge of the 3D/4D scenes.
In this work, we introduce **DynSuperCLEVR**, the first video question answering dataset that focuses on language understanding of the dynamic properties of 3D objects. We concentrate on three physical concepts—*velocity*, *acceleration*, and *collisions*—within 4D scenes. We further generate three types of questions, including factual queries, future predictions, and counterfactual reasoning that involve different aspects of reasoning on these 4D dynamic properties.
To further demonstrate the importance of explicit scene representations in answering these 4D dynamics questions, we propose **NS-4DPhysics**, a **N**eural-**S**ymbolic VideoQA model integrating **Physics** prior for **4D** dynamic properties with explicit scene representation of videos. 
Instead of answering the questions directly from the video text input, our method first estimates the 4D world states with a 3D generative model powered by a physical prior, and then uses neural symbolic reasoning to answer the questions based on the 4D world states.
Our evaluation on all three types of questions in DynSuperCLEVR shows that previous video question answering models and large multimodal models struggle with questions about 4D dynamics, while our NS-4DPhysics significantly outperforms previous state-of-the-art models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work focuses on the understanding of the dynamic properties of 3D objects in videos. It uses a simulator to control physical concepts (velocity, acceleration, collisions, etc.)  to generate videos, and uses pre-defined programs to generate annotations. Based on the proposed dataset, this work introduces a physics prior based VQA model. The experiment shows its effectiveness on the proposed dataset.

### Strengths
1. Overall, the paper is easy-to-follow.
2. The motivation of the dataset is clear.
3. Experiment verifies the effectiveness on the proposed dataset.

### Weaknesses
1. The content of the penultimate paragraph is confusing. Firstly, the motivation of the model is unclear. Why an explicit scene representation should be introduced to answer 4D dynamics questions? Secondly, what exactly is the"explicit scene representation"? Thirdly, what is the relation between the "scene parsing module" and "3D generative model", and how do the "3D generative representation" and "symbolic reasoning module" work together?
2. The significance of the proposed dataset is doubtful. In the reviewer's opinion, the scenarios in the proposed dataset may be too limited and fall far short of covering all real-world situations.
3. The effectiveness of the proposed model is also doubtful. The modules are specifically designed based on the actually unknown priors, i.e., the dataset generation and annotation processes. The authors should conduct experiments on spatiotemporal questions of other datasets, including both synthetic and real datasets.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

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
The paper addresses the dynamic properties of 3D objects in videos in the task of video question answering. It first proposes a new dataset called DynSuperCLEVR that composes multiple transportation objects into a scene and generates videos of these objects moving. The considered properties are speed, acceleration, and collision. Three types of questions are designed to test VLM's ability to understand the 3D dynamics of objects in these videos. A neural symbolic method, NS-4DPhysics, is proposed to address the importance of explicit 4D representation.

### Strengths
- Innovative Dataset: DynSuperCLEVR fills a gap in VideoQA with a focus on 4D dynamics, including velocity, acceleration, and collision, enhancing video-based physics reasoning. The scene is programmed in a way that the ground truth information of object speeds and collision events can be documented and transformed into question-answer pairs. 
- Effective Model Design: NS-4DPhysics uses a physics-informed 4D scene representation and neural-symbolic reasoning, excelling in complex VideoQA tasks over baseline models.
- Comprehensive experiments demonstrate NS-4DPhysics’s superior performance in factual, predictive, and counterfactual reasoning.

### Weaknesses
- The proposed dataset only spans a narrow domain of scenes and objects and may not generalize well to open-domain scenarios. The CLEVR-like setting makes things look nice and clean; for example, the objects have uniform colors, noise-free textures, rigid objects, and much fewer high-frequency details compared to realistic videos. Method comparison on the dataset may not reflect the true ability of the method in real-world videos.
- The reliance on physics priors might reduce flexibility in scenarios where these priors don’t apply as expected.

### Questions
- How did you input the video frames to GPT-4o?

### Soundness
2

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
This paper makes two main contributions: (1) introducing DynSuperCLEVR, a novel video question answering dataset that focuses on understanding 4D dynamics (velocity, acceleration, collisions) of objects in 3D scenes, and (2) proposing NS-4DPhysics, a neural-symbolic model that integrates physics priors with 3D scene understanding for dynamics reasoning. Through extensive experiments, their model significantly outperforms existing approaches, including large multimodal models, demonstrating current limitations in physical reasoning capabilities of video-language models.

### Strengths
- The paper's main objective of addressing multimodal 4D dynamics understanding is well-motivated.
- The authors provide comprehensive evaluation results across three types of reasoning tasks (factual, predictive, and counterfactual), demonstrating the model's capabilities in different scenarios.
- The proposed physics-aware neural-symbolic architecture presents an innovative approach

### Weaknesses
- The dataset only considers rigid objects with linear velocity and acceleration. Real-world scenarios often involve more complex dynamics like non-rigid deformation, rotation-based motion, and fluid dynamics.
- The dataset uses synthetic rendering which may not capture real-world challenges like motion blur, camera shake, varying lighting conditions, and partial occlusions.
- The ablation studies are limited. While the paper shows the importance of physics priors, there could be more detailed analysis of other architectural choices and hyperparameters, like the impact of different CNN backbones or the choice of physics engine parameters.

### Questions
- It would be great if the author analyze the performance of proposed model on more complex dynamic scenarios such as non-rigid object deformation or fluid dynamics
- How do the authors plan to address the challenges in real datasets such as motion blur, camera shake, and varying lighting conditions?
- How do architectural choices  or important hyperparameters such as different CNN backbones or the choice of physics engine parameters impact on the performance of proposed model?

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
2

### Summary
The paper introduces DynSuperCLEVR, a video question answering (VideoQA) dataset that emphasizes understanding dynamic 3D object properties within 4D (3D + time) scenes. 
Additionally, the authors present NS-4DPhysics, a model that combines neural-symbolic reasoning with physics-based priors to analyze these dynamic properties. The model first constructs an explicit 4D scene representation using a 3D generative model, followed by neural-symbolic reasoning to answer questions. 
Experimental results demonstrate that NS-4DPhysics surpasses existing VideoQA models across various question types (factual, predictive, and counterfactual), underscoring its effectiveness in reasoning about object dynamics in complex, synthetic environments.

### Strengths
**1. Novel Dataset:** DynSuperCLEVR is a novel dataset that focuses on 4D dynamics, addressing a critical gap in existing VideoQA datasets which typically overlook explicit physics-based scene understanding.

**2. Innovative Model Design:** The NS-4DPhysics model combines 3D generative modeling with physics-informed priors, represents an innovative approach to handling dynamic 4D scene reasoning.

**3. Comprehensive Benchmarking:** Extensive evaluations against baseline models, including video large language models (Video-LLMs) and other symbolic frameworks, highlight the superior performance of NS-4DPhysics in capturing 4D dynamics.

**4. Future and Counterfactual Simulations:** By leveraging physics-based priors, the model excels at simulating both future and hypothetical states, demonstrating practical value and broad application potential.

### Weaknesses
**1. Synthetic Data Limitations:** While the dataset is suitable for testing dynamic properties, its synthetic nature may limit generalizability to real-world applications. Despite the authors’ efforts to improve aspects like background realism (L201), models trained exclusively on synthetic data often struggle to handle real-world noise and variability.

**2. Computational Complexity:** The NS-4DPhysics model is computationally demanding due to its reliance on 3D generative modeling and physics-based priors, presenting challenges for scalability and use in resource-constrained environments.

**3. Limited Object Diversity:** The dataset is limited to a narrow range of rigid objects, which may not adequately represent the complexity of real-world scenes that often include deformable or articulated objects.

**4. Evaluation of Real-World Applicability:** The paper lacks an analysis of the model’s performance on real-world video data, which is essential for evaluating its practical applicability outside synthetic benchmarks.

### Questions
1. Have the authors considered extending the dataset to include articulated or deformable objects? If so, what challenges or limitations do they anticipate with this extension?

2. Could the authors provide an efficiency analysis of the proposed model, including resource usage and runtime under typical conditions?

3. What modifications to the NS-4DPhysics framework would make it more efficient for real-time performance or deployment in resource-limited environments?

### Soundness
3

### Presentation
3

### Contribution
3
