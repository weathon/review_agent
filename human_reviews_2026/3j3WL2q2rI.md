# SpikeGrasp: A Benchmark for 6-DoF Grasp Pose Detection from Stereo Spike Streams

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
We have built Spikegrasp.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper innovatively utilizes spike streams as perceptual signals to predict 7-DoF grasp poses. To achieve this, the paper proposes the SpikeGrasp framework and synthesizes a dataset in Blender for model training. The proposed method demonstrates promising performance on the test data. The main contributions of the paper are as follow:

1.	This paper proposes the SpikeGrasp architecture which process spike streams and outputs grasp pose.

2.	This paper presents the first large-scale synthetic spike stream dataset for 6-DoF grasp pose detection.

### Strengths
1.	This paper introduces an end-to-end 6-DoF pose detection network based on stereo spike streams, which is different from the previous works that are based on RGBD or point clouds.

2.	This paper presents the first synthetic spike stream dataset for 6-DoF grasp pose detection,

### Weaknesses
1.	The dataset presented in this paper constitutes an incremental extension of the GraspNet-1Billion [1] dataset. It retains similarity with GraspNet-1Billion in terms of the included objects, the grasp pose annotations, and the data partitioning strategy. The primary enhancement lies in the incorporation of the spike stream modality. Furthermore, the evaluation benchmark remains consistent with that used for GraspNet-1Billion.
2.	The paper does not clearly explain why spike streams are used as input. According to the results in Table 1, SpikeGrasp only achieves an 38.84 AP on the seen test set, while GSNet which gets single view point clouds already reaches 65.7 AP on the GraspNet-1Billion test set. Furthermore, I am curious why GSNet only achieves 34.52 AP on the synthetic dataset? What are the distinctive features of this synthetic dataset compared to GraspNet-1Billion?
3.	The experiments presented in this paper have certain limitations. First, the proposed method is not evaluated on the widely adopted GraspNet-1Billion benchmark. This would support the results and make them stronger. Furthermore, the study lacks real-world experiments, which are essential for demonstrating the practical applicability and robustness of the approach in physical environments.

[1] Fang, Hao-Shu, et al. "Graspnet-1billion: A large-scale benchmark for general object grasping." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

### Questions
1.	Why does the paper use Stereo Spike Streams as the model input? What advantages does it offer compared to RGB-D or point cloud data for the 6-DoF grasp pose detection task?
2.	Where does the gap between real and simulated Stereo Spike Streams data come from? Can the method proposed in the paper be used in real-world scenarios, and how effective is it?

### Soundness
2

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
3

### Summary
The paper proposes SpikeGrasp, which uses stereo spike streams to obtain  6-DoF grasp poses. This differs from previous methods that depend on explicit geometry reconstruction. The paper also curates a synthetic grasping dataset and defines a custom evaluation protocol. The experiment in the synthetic environments suggests competitive performances against other baselines.

### Strengths
1. The end-to-end “spikes to grasp” pipeline is technically coherent and clearly pipelined.

2. The paper includes reasonable ablations showing the effect of objectness and graspness branches and spike choices.

### Weaknesses
1. The story is not convincing. The introduction starts well, and the motivation is clear, but the subsequent parts, including the methodology, do not really effectively adhere to the points conveyed in the paper. The current writing lacks reasons for designing each module. I agree with the claim that humans do not have an explicit 3D sensing in the brain to manipulate objects.  I also feel interested in a Spike-driven solution. However, I do not feel a very strong connection to why the authors designed the network in this way. 

2. The central narrative is that direct “spikes → grasp” is viable and efficient. However, all validation is synthetic or simulation-only. There should be a real spike-camera experiment for evaluation.

3. Some typos can be found under Supplementary Material. For example, A.2.1 PROBLEM STATEMENT is typed twice.

### Questions
Can you report some real spike-camera experiments with SR@1/5/10 and PR-AUC, plus wall-clock latency?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a bio-inspired 6-DoF grasp detection method called SpikeGrasp, which mimics the biological visuomotor pathway. Similar to retinas, the method processes raw and asynchronous events from stereo spike cameras to directly infer grasp poses and then refines grasp hypotheses via a recurrent spiking neural network without reconstructing a point cloud. The authors built a synthetic benchmark dataset to evaluate SpikeGrasp, and compared it with standard methods. Good results are reported.

### Strengths
1. This paper proposes a novel bio-inspired grasping method that uses stereo spike cameras, which is new in the field of grasping.

2. This paper proposes a new synthetic spike stream dataset for spike-based grasping research.

3. This paper conducts a set of experiments and demonstrates the effectiveness of the proposed method.

### Weaknesses
1. The paper does not clearly analyze the benefits of spike-based grasping, and the principle of SpikeGrasp should be further elaborated.

2. The paper lacks real-world experiments, which is hard to demonstrate the validity of grasping methods.

3. The relation of this work and existing literature should be analyzed in more depth. What is the design rationale of the proposed method should be elaborated more.

### Questions
1. How does SpikeGrasp deal with dynamic environments?

2. The paper states that 6-DoF grasp annotations of objects are sourced from GraspNet-1Billion. Could the authors provide the detailed process of how to source grasp annotations?

3. I am a bit confused why real-world experiments were not conducted. What are the challenges?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The author proposes SpikeGrasp, a deep learning framework for 6-DoF grasp detection using raw spike streams. The model extracts features from the spike streams and then uses the features to generate objectness, graspness, and the final grasp pose. Also, a new synthetic spike stream dataset for 6-DoF grasp detection is proposed.

### Strengths
The use of spike stream offers new insight into grasp detection. The approach eliminates the need for depth information, which may sometimes be inaccurate. 

The author proposes a new synthetic dataset with spike streams, which fills the underserved evaluation gap for spike stream-based methods and is a useful resource for the community.

The model is computationally efficient in theory.

### Weaknesses
The methods used to compare are mostly dated. The author should consider evaluatingmore state-of-the-art methods, e.g., HGGD or EconomicGrasp.

The author should conduct more real-world experiments instead of just using simulated settings. Real robot experiments are very important for validating the effectiveness of the proposed method.

### Questions
Is the depth modal still in the new synthetic dataset? Since many other methods in the quantitative evaluation require depth as a input modal, the author should elaborate this point in the details of the dataset. 
Why is the result of GraspNet on the synthetic dataset indentical to that on the GraspNet-1Billion dataset?
GSNet has a relatively high AP on the GraspNet-1Billion seen dataset. Why is there a huge performance degradation on the seen synthetic dataset, namely 34.52. Is the dataset properly generated?

### Soundness
2

### Presentation
2

### Contribution
2
