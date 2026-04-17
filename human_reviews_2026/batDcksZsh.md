# Boosted Trees on a Diet: Compact Models for Resource-Constrained Devices

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Deploying machine learning models on compute-constrained devices has become a key building block of modern IoT applications. In this work, we present a compression scheme for boosted decision trees, addressing the growing need for lightweight machine learning models. Specifically, we provide techniques for training compact boosted decision tree ensembles that exhibit a reduced memory footprint by rewarding, among other things, the reuse of features and thresholds during training. Our experimental evaluation shows that models achieved the same performance with a compression ratio of 4–16x compared to LightGBM models using an adapted training process and an alternative memory layout. Once deployed, the corresponding IoT devices can operate independently of constant communication or external energy supply, and, thus, autonomously, requiring only minimal computing power and energy. This capability opens the door to a wide range of IoT applications, including remote monitoring, edge analytics, and real-time decision making in isolated or power-limited environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces compressed boosted decision trees for constrained devices. The authors reuse values across thresholds, leaves, etc. and create a compact memory layout. The results indicate that the proposed methods are competitive in accuracy while being significantly better in comression.

### Strengths
1. The paper is clear to read
2. The motivation and the constraints specified as per use cases and practical designs pique the interest
3. The proposed method is well-described and the method appears very intuitive
4. The results and analysis are rigorous and show good performance

### Weaknesses
1. I am not from this exact area but I am a bit surprised if there aer not more related works. Neural network pruning, quantization and deployment on edge devices is common. But, perhaps even for tree-based methods there are more such related work. Atleast, a couple of examples that come to my mind.

• ProtoNN: Compressed and Accurate kNN for Resource-scarce Devices
Chirag Gupta, Arun Sai Suggala, Ankit Goyal, Harsha Vardhan Simhadri, Bhargavi Paranjape, Ashish Kumar,
Saurabh Goyal, Raghavendra Udupa, Manik Varma and Prateek Jain
International Conference on Machine Learning (ICML), 2017

• Resource-efficient Machine Learning in 2 KB RAM for the Internet of Things
Ashish Kumar, Saurabh Goyal and Manik Varma
International Conference on Machine Learning (ICML), 2017


2. Continuation to above -- I think the absence of more baselines esp. in optimized tree implementations is a weakness.

### Questions
1. Can you expand related work to include more methods for optimizing tree-based methods on constrained devices?
2. Can you compare your work empirically to those papers?

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
3

### Summary
This research introduces a novel compression scheme tailored for boosted decision tree ensembles to address the challenge of deploying machine learning models on compute-constrained IoT devices. The approach focuses on training compact models by strategically encouraging the reuse of features and thresholds. Experimental results demonstrate a substantial memory footprint reduction of 4–16x compared to standard LightGBM models without compromising performance. This optimization is crucial for enabling autonomous, low-power IoT applications such as remote monitoring, edge analytics, and real-time decision-making in isolated or energy-limited environments.

### Strengths
Penalizing the use of new features/thresholds encourages reuse across trees. The idea of feature reuse and threshold reuse is interesting and simple yet useful for sustainable ML. Introduction of a new loss function. 

The work is very useful for doing efficient ML on resource-constrained devices.  

Good sensitivity analysis. 

I found the writing to be decent, and the paper is well structured. 

The performance gains with minimal memory consumption compared to SOTA methods.

### Weaknesses
The part on memory layout based on encoding the information in a bit-wise manner is not novel. 

No actual implementation on MCUs, which makes this a purely algorithmic work. A bit more analysis, including power consumption and energy efficiency, is required. 

Domains requiring distinct rules (e.g., heterogeneous datasets) do not allow threshold reuse without performance loss.

The idea is good, but the utility of it is limited. For example, it does not make sense to train ML models on tiny MCU devices with 32 KB of RAM. The authors should motivate the real use case scenarios where their proposed method will be useful in the real world.

### Questions
What are some real-life scenarios where one will need to do training on MCUs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Trees on a Diet (ToaD), a method for compressing GBDT for resource-constrained devices. It employs a regularization strategy that encourages the reuse of features and thresholds during training, coupled with a specialized, bit-wise memory layout. This layout uses global lookup tables and a pointer-less array representation to minimize storage. Evaluated on eight tabular datasets, ToaD compresses models that are 4–16× smaller than baseline LightGBM while maintaining comparable accuracy under tight memory constraints.

### Strengths
1. The work enables complex models like GBDTs to run on severely memory-constrained microcontrollers.
2. The proposed pointer-less array-based memory layout is highly suitable for microcontroller deployment, minimizing memory footprint and avoiding inefficient pointer chasing.
3. The method is orthogonal to many existing compressions like pruning and quantization, and easy to integrate into existing work
4. The experimental evaluation is comprehensive and sound

### Weaknesses
1. Potential inference latency overhead. The decoding process involving bit-level manipulations and lookups in global arrays may inherently more computationally expensive than the direct pointer-based traversal used in standard implementations. The paper would be significantly strengthened by an end-to-end latency evaluation.
2. Linear penalty is not motivated theoretically (e.g., from a Bayesian perspective) or empirically against other potential forms. Would a logarithmic penalty, which might better model diminishing costs, be more effective? 
3. Several figures (e.g., Figure 6)  is so small that makes the text are hard to read. Furthermore, in Figure 5, the label is partially covered.

### Questions
1. The search over 32,076 configurations per dataset is computationally intensive. Do you have any insights or heuristics for navigating the hyperparameter more efficiently in practice?
2. For larger models that must reside in main memory (or SSD) and are evaluated on systems with hierarchical caches, could the non-sequential, lookup-heavy memory access pattern of your method lead to frequent cache misses or I/O bottlenecks compared to the more sequential access of standard array-based tree representations?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this article, the authors propose Trees on a Diet (ToaD), a training‑time compression framework for Gradient-Boosted Decision Tree (GBDT) ensembles, targeting microcontrollers and, more generally, resource-constrained edge/embedded devices.
The core ideas are: *i)* adding linear regularizers that penalize the introduction of new features and new thresholds across the ensemble to encourage reuse during tree growth, and *ii)* deploying a pointer‑less bit‑wise memory layout with global lookup tables for feature thresholds and global leaf values shared across all trees.
The results show that ToaD matches the competitor's performance, reporting 4 to 16 times lower memory usage for the same accuracy in the most relevant memory range (≤128 KB).

### Strengths
The work tackles a concrete real-world problem: DTs on microcontrollers, where RAM and flash budgets are very limited.

The article is well-written, structured, and easy to follow.

The proposed method is described in sufficient detail.
Specifically, the introduction of ensemble-level penalties using features and thresholds is a simple yet effective idea to induce parameter reuse, which complements post-training pruning/quantization.
Furthermore, the design choice to store the threshold bit-width and numeric type per feature provides a flexible precision/size trade-off.

### Weaknesses
The primary motivation for this work is the deployment of GBDT on resource-constrained devices, where memory is a critical constraint, as well as latency (and energy consumption).
The authors provide results of memory savings, but do not present any experimental results on inference speed (and energy consumption).
Without this analysis, the practical utility of ToaD for real-time edge applications remains unproven.

The authors state that the $RF$ is “the ratio between the global number of values and the sum of the nodes and leaves” (line 371).
Thus, if values are reused effectively, the number of global values becomes smaller, while the number of nodes and leaves remains fixed; therefore, a good reuse would produce $RF<1$.
At line 374, the interpretation in the text says the opposite, implying that $RF$ should be (#nodes + #leaves)/(#global values), *i.e.*, the inverse of the statement above.

Memory for baselines is computed under a simplified node model that includes two child pointers, whereas ToaD benefits from a pointer‑less encoding and global sharing.
This risks giving ToaD an advantage in the comparison.

Experiments use a single 80/20 split per dataset with large sweeps and report the best points within memory limits; however, no statistical significance values are presented.

The authors cite several other relevant works on tree compression and optimization (*e.g.*, Koschel et al., 2023, and Buschjäger & Morik, 2023) in their related work section, but do not include them in the experiments.

### Questions
The authors should:
1) Provide inference latency and energy consumption (per prediction) on representative MCUs (*e.g.*, the mentioned ARM Cortex‑M4 @ 48 MHz) for ToaD versus baselines.
2) Clarify the reuse factor formula to align with the intended interpretation.
3) Report memory results under a unified layout, or at least discuss the potential advantages of ToaD against baseline models.
4) Report mean±std for accuracy at each memory budget, and compare against other relevant presented works only mentioned in the state-of-the-art.

### Soundness
2

### Presentation
3

### Contribution
2
