# NPF-$k$CT: A $k$-center clustering solver with neural process filter for continuous POMDP-based object search

- Decision: Reject
- Scores: 3, 6, 8, 6

## Abstract
Efficiently searching for target objects in intricate environments poses a significant challenge for mobile robots, due to perception errors, limited field of view (FOV), and visual occlusion. These factors cause the problem to be partially observed. Therefore, we formulate the object-search task as a high-dimensional Partially Observable Markov Decision Process (POMDP) with hybrid (continuous and discrete) action spaces. We propose a novel sampling-based online POMDP solver named Neural Process Filtered $k$-Center Clustering Tree (NPF-$k$CT). The optimal action is selected using Monte Carlo Tree Search (MCTS) in conjunction with a neural process network to filter out ineffective primitive actions (i.e., basic robot operations), alongside $k$-center clustering hypersphere discretization to efficiently refine high-dimensional continuous sub-action spaces. Adhering to the hierarchical optimistic optimization (HOO) concept, we leverage an upper-confidence bound (UCB) on the action value function within the hypersphere with estimated diameters to guide the MCTS expansion. We extensively tested our approach in Gazebo simulations using Fetch and Stretch robots across diverse target-finding scenarios. Comparative results show higher success rates and faster target detection than baseline methods, with no additional computational cost. We also validated our method on a physical robot in an office environment. Project page: \url{https://sites.google.com/view/npfkct}.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a POMDP solver to find the target occluded objects in complex household environments. The proposed method can select the appropriate action primitives and solve long-horizon goals, such as searching the target objects. The experiments are conducted in both simulation and the real world.

### Strengths
1: The authors work on an important research direction: searching for occluded objects is an interesting and open research problem. 

2: The POMDP formulation is a feasible approach to solving the complex search problem. 

3: The authors compare the proposed approach to several POMDP baselines, which is nice.

### Weaknesses
1: My first concern is that the paper does not discuss any failure cases. Throughout the paper, I find that the proposed approach achieves a 100% success rate in most of the simulation tasks and 10/18 in the real-world tasks. The authors only mention that the real-world failures are due to hardware and communication issues, which are superficial. 

2: My second concern is that the paper misses an important discussion of related works on reasoning about occluded objects in the robot manipulation literature, such as [1][2]. Specifically, [2] proposes a memory model to remember and retrieve the occluded target objects by selecting action primitives to achieve the goal. I understand that both [1] and [2] only present experiments with a fixed-base arm, but it should be feasible to extend their approaches to mobile manipulation by incorporating navigation skills. 

[1]: C. R. Garrett, C. Paxton, T. Lozano-P ́erez, L. P. Kaelbling, and D. Fox, “Online replanning in belief space for partially observable task and motion problems,” in 2020 IEEE International Conference on Robotics
and Automation (ICRA). IEEE, 2020. 

[2]: Y. Huang, J. Yuan, C. Kim, P. Pradhan, B. Chen, L. Fuxin, and T. Hermans. Out of Sight, Still in Mind: Reasoning and Planning about Unobserved Objects with Video Tracking Enabled Memory Models. In IEEE International Conference on Robotics and Automation (ICRA), 2024.

3: The authors do not discuss any limitations of the proposed approach. 

4: The visualizations in the paper can be significantly improved. For example, in Figure 7, it’s hard to understand the difference between the red boxes, and blue, and purple dashed circles without a caption. Furthermore, the approach section is difficult to follow without a clear approach overview figure.

5: This paper makes strong assumptions about known furniture and known object poses. In the real world, would your proposed approach be robust to the noisy segmentations and estimated object poses?

### Questions
1: what’s the difference between your proposed approach and [1][2]? 

2: what are the failure cases and limitations of your proposed approach? 

3: Could you show the performance of your proposed approach with imperfect segmentation and pose estimation in the real world?

### Soundness
2

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
4

### Summary
The paper presents an interesting approach to address the problem of Object Search. It uses the Monte-Carlo Tree Search (MCTS) along with a Neural Process Filter (NPF) and k-center clustering to find the best action based on maps, images and onboard sensors. The NPF helps remove the ineffective actions and the k-center clustering helps to refine the action spaces. This method has been tested in simulation on Gazebo and in real-life, showcasing some promising results.

### Strengths
- The paper is very well-written and structured.
- The paper presents a novel, simple yet elegant solution to the object search problem.
- The real-world demonstration of the method is appreciated.

### Weaknesses
1. The overall methodology might be novel but considering the individual modules, it comes down to a Monte-Carlo Tree Search (MCTS), an action scoring network (NPF) and k-center clustering. This approach seems heavily engineered towards solving the Object Search problem specifically and holds little appeal for the broader learning community.

2. Missing SOTA and comparitive study : The paper has not cited or evaluated against some known methods in the domain of Object search such as [1]. This makes it difficult to understand not just the comparative efficacy of the proposed method but also the complexity of the experimental setup.

3. The paper primarily uses the scenarios created by the authors through Gazebo for evaluation. However it would be better and more helpful to add experiments on some known dataset or benchmarks such as ScanNet [2].

4. The experimental setup seems too simple. In 4 out of the 5 scenarios the paper shows results in Table 1, the baselines and this method achieve well over 90% successful rate of object search within 50 steps. This warrants the need for more thorough evaluation either by increasing the complexity of the simulated setting or testing on some known dataset/benchmark as previously stated.

[1] Francesco Giuliari, Geri Skenderi, Marco Cristani, Alessio Del Bue, and Yiming Wang. 2023. Leveraging Commonsense for Object Localisation in Partial Scenes. IEEE Trans. Pattern Anal. Mach. Intell. 45, 10 (Oct. 2023), 12038–12049.\
[2] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

### Questions
Same as Weakness

### Soundness
3

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
3

### Summary
This paper introduces NPF-kCT, an online POMDP solver designed for efficient object search tasks in complex 3D environments using mobile manipulators. The paper outlines that the major challenges are searching for target objects in cluttered and partially known environments, where perception errors, limited fields of view, and visual occlusions are prevalent. NPF-kCT formulates the object search problem as a high-dimensional POMDP with hybrid action spaces and utilizes a neural process-based network to predict the feasibility of primitive actions and filter out ineffective ones. The method also applies a clustering method to group filtered actions into hyperspheres and employs MCTS in conjunction with a UCB strategy to construct a belief tree and select optimal actions. The paper provides theoretical results for sampling robot actions and empirical results in Gazebo simulations with Fetch and Stretch robot simulators that outperform baseline methods in terms of success rate and efficiency in finding target objects.

### Strengths
The results have a sufficient number of baselines. The work tackles a significant problem as open-world mechanical search is difficult and useful. The introduction and conclusion are clear but section 4 could do a better job with intuition about aspects of the method. The figures are good and informative but Figure 4 should be redone. The main novelty is the POMDP solver which has both empirical and theoretical results thus having good originality.

### Weaknesses
The paper only offers simulation experiments in Gazebo with two mobile robots but would benefit from expanding to other navigation simulators such as one of the THOR variants but ideally real experiments should be conducted to truly corroborate results. It would be interesting to see how this method scales with higher-dimensional belief states and action spaces. Line 246 does not seem to be a full sentence. The clarity for the Motivation about Network Filtering is still not obvious and would benefit from being clearer. The related works would also benefit from being more thorough. It would also be helpful to conduct ablations of different parts of the method.

### Questions
To solve the task of mechanical search, why is better to favor a POMDP solver rather than a VLA (vision-language-action) model?
How much benefit is the action clustering providing?

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
This paper proposes an online POMDP solver, Neural Process Filtered k-Center Clustering Tree (NPF-kCT),  designed to optimize object search tasks for mobile robots in cluttered environments. NPF-kCT integrates Monte Carlo Tree Search (MCTS) with a neural process network and a k-center clustering approach, making action selection more efficient by filtering out sub-optimal actions from high dimensional action spaces.  Experiments in Gazebo and real-world settings show the solver's effectiveness, as it outperforms baseline POMDP methods in success rate, speed, and total reward for the same computation budget.

### Strengths
1. One of the primary drawbacks of tree search based algorithms is increasing sample/time complexity with larger tree widths. In this paper, NPF-kCT effectively combines neural process filtering with k-center clustering and MCTS to filter out sub-optimal actions while expanding the search tree. 

2. Results include both simulated and real-world scenarios of object search in cluttered environments and the proposed method can generalize to different robot platforms. NPF-kCT especially outperforms comparable MCTS-based baseline algorithms in terms of number of actions or steps required to achieve the task.

### Weaknesses
1. NPF-kCT relies on a number of heuristic design choices which might hinder generalization and reproducibility. Some of these are setting the number of clusters $k$, number of simulation episodes in the search tree, and neural process network training which affects the filtered action space.  

2. Certain assumptions, such as a 100% success rate for actions in Gazebo, may not fully reflect real-world constraints, possibly affecting the method’s performance in practical applications.

3. In terms of writing quality, although the authors have provided adequate details for the algorithm, it might help to make the text more succinct to further improve the presentation.

### Questions
1. In Table 1, do the reward values convey anything other than "higher reward is better"? Why not just report the number of steps since that more clearly supports the claim of improved search efficiency over baselines? 

2. Line 246 - why does it start from the middle of a sentence? 

3. Why was the time complexity not reported for any of the algorithms in Table 1? Is it efficient enough to actually use NPF-kCT as an online solver with real robots?

### Soundness
3

### Presentation
2

### Contribution
3
