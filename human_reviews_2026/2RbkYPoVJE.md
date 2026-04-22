# LEO: A Graph Attention-Based Framework for Learned Object Extensions and Adaptive Sensor Fusion for Autonomous Driving Applications

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Accurate shape and trajectory estimation of dynamic objects is a fundamental requirement for reliable perception in Automated Driving (AD). In the classical versions of AD algorithms and stacks, various Bayesian extended object geometric models are used to provide object-related extensions and trajectories. Performance of such approaches are deeply connected with the completeness of a-priori and update-likelihood functions. Recent deep learning approaches improve flexibility by learning shape features directly from raw or fused sensor data, but they often rely on dense annotated datasets and high computational resources, which restricts their applicability in production vehicles. We aim to improve production-level automated driving systems by integrating the computational efficiency and theoretical robustness of geometric methods with the adaptability and generalization capabilities of modern deep learning techniques. We employ a task-specific parallelogram-based ground-truth formulation to represent object extensions, facilitating expressive modeling of complex geometries such as articulated trucks and trailers. Our primary contribution is the development of a novel spatio-temporal Graph Attention Network (GAT)-based model, Learned Extension of Objects (LEO), that demonstrates proficiency in adaptive fusion weight learning, temporal consistency, and multi-scale shape representation from multi-modal production grade sensor tracks. LEO successfully generalizes across various sensor modalities, configurations, object classes, and geographic regions, exhibiting robustness even under challenging conditions and longer range targets. We have presented these observations and evaluations based on the real-world Mercedes-Benz SAE Level-3 (L3) DRIVE PILOT dataset in our article. Furthermore, its computational efficiency makes it a suitable candidate for integration into a real-time production system, although further validation and integration efforts are necessary for deployment in safety-critical systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LEO, a spatio-temporal Graph Attention Network framework for extended object tracking and adaptive sensor fusion in autonomous driving. It introduces a parallelogram-based representation to better capture articulated geometries such as trucks with trailers, and a dual-attention mechanism that jointly models temporal consistency and spatial dependencies across multiple sensor modalities. The method is evaluated on a large-scale real-world Mercedes-Benz DRIVE PILOT dataset, showing improved performance.

### Strengths
1. The paper is clearly written and easy to follow, with a logical structure connecting motivation, design, and evaluation.

2. The proposed parallelogram-based representation and dual-attention fusion mechanism are technically sound and reasonably novel for production-oriented extended object tracking.

3. The framework design is simple and efficient, achieving strong quantitative and qualitative results with real-world data, demonstrating  potential for deployment.

### Weaknesses
1. The paper overlooks a substantial body of work in 3D object detection and tracking that explores similar geometric or graph-based formulations, limiting its positioning within the broader literature.

2. The model does not explicitly consider object elevation or height variation on non-flat surfaces, which could affect performance in urban scenarios.

3. Although the paper claims real-time and production efficiency, it lacks explicit comparisons with other lightweight or real-time baselines to substantiate the efficiency claim.

4. While the parallelogram representation is motivated by articulated geometries, its ability to generalize to curved or multi-joint structures (e.g., articulated buses or deformable trailers) is limited; multiple correlated bounding boxes could better capture such configurations.

### Questions
1. How does the computational efficiency of LEO compare quantitatively with other models such as 3DMOTFormer or TransFusion?

2. Have the authors considered integrating a height or 3D shape component in the parallelogram representation to improve generalization to complex urban geometries?

3. For articulated or multi-joint vehicles, would a hierarchical multi-box or graph-structured decomposition improve modeling accuracy compared to the single-parallelogram abstraction?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces LEO (Learned Extension of Objects), a novel deep learning framework designed for accurate object shape and trajectory estimation in autonomous driving (AD) systems. The central problem it addresses is the gap between classical geometric tracking methods which are computationally efficient but inflexible and modern deep learning approaches, which are more adaptable but often too resource-intensive for production vehicles and rely on dense, annotated raw sensor data.

LEO is proposed as a hybrid solution that integrates the adaptability of deep learning with the constraints of production-level systems, which typically provide object level tracks rather than raw sensor data. The framework uses a spatio-temporal Graph Attention Network (GAT) to perform adaptive fusion of multi-modal sensor tracks from LiDARs, radars and cameras.

A key contribution is the use of a parallelogram based object representation. This allows the model to represent complex geometries, such as articulated trucks with trailers, more accurately than traditional rectangular bounding boxes. The model is trained and evaluated on a large-scale, real-world dataset from the Mercedes-Benz SAE Level-3 DRIVE PILOT system, demonstrating its accuracy, temporal stability, and computational efficiency.

### Strengths
- Production-Focused and Efficient: The system is explicitly designed to work within the constraints of production vehicles.
- Works with Object Tracks: Unlike models that require raw, high-bandwidth sensor data, LEO operates on high-level object tracks.
- Computationally Efficient: The model is fast, with an average inference time of ~13.5 ms (around 30 FPS) on an RTX 2080 Ti, making it suitable for real-time deployment.
- Novelty with parallelogram representation. The model's key innovation is its use of a parallelogram-based representation instead of standard rectangular bounding boxes. It also models articulated vehicles which allows the model to accurately represent complex, non-rectangular shapes like articulated trucks with trailers.

### Weaknesses
- No comparison with other state-of-the-art methods. The paper only discusses limited quantitative and qualitative performance of the proposed method but does not show comparisons with other methods.
- Parallelograms, while more expressive than rectangles, are not universal and they cannot accurately model non-convex shapes (e.g. a pedestrian with an outstretched arm, a forklift, or complex multi-vehicle scenarios). Also, there is not sufficient quantitative evidence in the paper which discusses the benefit of a parallelogram approach.
- While efficient in this application, Graph Attention Networks can become computationally expensive as the graph size (number of objects or sensor tracks) increases.
- The work lacks technical novelty expected for ICLR. The core of the model is a Graph Attention Network (GAT), which is a pre-existing architecture. The parallelogram representation is a clever, domain-specific engineering choice for handling articulated vehicles, which is as an extension of a bounding box rather than a new general-purpose representation.

### Questions
- Could the authors provide a quantitative comparison against other track-level fusion algorithms? 
- A key contribution of the work is the parallelogram-based representation for modeling articulated objects. To isolate the benefit of this contribution, could the authors provide an ablation study? Specifically, what is the performance difference when training a version of LEO that only outputs standard oriented rectangular boxes versus the full parallelogram model?
- Handling of Non-Convex Geometries: The parallelogram representation is more flexible than a rectangle but is still a convex polygon. How does the model perform when faced with distinctly non-convex objects, such as a "jack-knifed" truck, a forklift, or a vehicle with an open trunk? Is the resulting parallelogram considered a "safe" (i.e., over-approximated) representation in these cases, or does the model fail?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, a graph attention based setup is proposed for multiple object tracking. The system takes in inputs (tracked objects) from multiple sensors (camera, lidar, radar), and fuses them using a graph attention transformer. 

I was a little intrigued by the problem and design in that it was not to use learning to track fused features, but as a master fuser of sorts to handle inputs from different modalities. Note that these inputs are themselves part of a tracking pipeline (say, an EKF), and as such they could be fused with another such state machine, but owing to difficulties in disambiguating them (for instance in compound objects), they resort to this learned approach to handle it. Another fascinating point is that they use labels generated from an auto labelling pipeline which itself is, from the looks of it, a performant tracker. So in a sense the learning replaces this pipeline a la distillation. 

The method is geared for engineering use. They have a graph attention transformer handling spatial and temporal nodes, featuring as separate losses.

### Strengths
+ This is a well engineered approach that can be used in real, production cases. 
+ It gets around problematic aspects of handling multisensor inputs through this GAT fusion approach. 
+ Spatial and temporal attention is elegant. 
+ Approach seems to run real time, and seems to get around expensive operations like the Hausdorff (quadratic++) distance calculation
+ Implicit handling of association by GAT.
+ Handling of composite objects (semi-truck, etc). This is a pain point in industrial trackers generally(!).

### Weaknesses
- I like the approach and implementation, but unfortunately have to call out the fact that there are no evaluations on public datasets. Without a public evaluation, it is very hard to judge how well this method performs, other than in a qualitative sense on their own dataset. I admit that this is how we will do it in a real production setup. Also germane is the point that real world datasets with appropriate sensors are much more complex than public datasets like nuscenes. Even so, I think it is useful, if only, to compare methods and to make it easier to replicate for in-house use. 

- The paper does not work on actual features from a detector. I am slightly biased towards approach that work with real features, in so far as to say that they add some richness to the representation. I would like to ask the authors, what benefit does their approach bring to the table as compared with a traditional approach - in terms of numbers. 

- The labelling approach: Generally (again, the biases stem from my practice), we agree that labelling is a painstaking and difficult to scale approach for tracking. But here, they generate 'pseudo labels' from - please correct me if the interpretation is incorrect - a previous in-house solution that uses engineered fusion approaches. So their approach learns these pseudo labels, and not really learning any new aspects of the data like geometry or occlusions. This undermines the novelty of the work in my view. 

- I would have hoped for some analysis of difficult cases like occlusions, disagreements across sensors and so forth.

### Questions
See above. I have questions about the motivation, which seems nuanced. What does the GAT bring to the table? Can you compare results with a KF based solution? 

I would like to see performance on public datasets and across different methods.

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
LEO introduces a graph attention–based model for multi-sensor fusion and object shape estimation in autonomous driving. Using a parallelogram representation and dual attention, it captures both temporal and spatial relations. Tests on Mercedes-Benz DRIVE PILOT data show high accuracy and real-time performance. The framework is efficient, robust, and suitable for real-world deployment.

### Strengths
The paper is well-written and easy to follow. In addition, some advantages of the paper are listed below:
* Adaptive Multi-Sensor Fusion. LEO’s dual-attention mechanism effectively learns to fuse LiDAR, radar, and camera data, adapting to varying sensor reliability and visibility conditions.
* Generalized Object Representation. The parallelogram-based shape model accurately handles both rigid and articulated objects (e.g., trucks with trailers), outperforming traditional rectangular bounding box methods.
* Real-Time, Production-Ready Performance. The framework achieves high accuracy with low latency (~30 FPS) and low computational cost, demonstrating strong potential for real-world automotive deployment.

### Weaknesses
Apart from the strengths listed in the Strength Section, there are some weaknesses:
* Only one dataset is used. The proposed method is only evaluated on one dataset. It would be great if the authors could also evaluate the proposed method on more datasets to prove the generalizability of the algorithm.
* Limited novelty. The proposed architecture mainly extends existing Graph Attention Network and sensor fusion concepts, with modest methodological innovation.
* Insufficient experiments. There is little exploration of how each component (e.g., dual attention, parallelogram representation) contributes to overall performance.
* Comparison with other baselines. The paper does not provide direct quantitative comparisons with state-of-the-art learning-based or geometric fusion methods, making it hard to assess relative performance.

### Questions
Since there are some major weaknesses in the paper (see the Weaknesses Section), I suggest that the authors resubmit the paper to another conference or journal.

### Soundness
3

### Presentation
3

### Contribution
2
