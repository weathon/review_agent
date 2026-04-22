# Capturing Visual Environment Structure Correlates with Control Performance

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
The choice of visual representation is key to scaling generalist robot policies. However, direct evaluation via policy rollouts is expensive, even in simulation. Existing proxy metrics focus on the representation's capacity to capture narrow aspects of the visual world, like object shape, limiting generalization across environments. In this paper, we take an analytical perspective: we probe pretrained visual encoders by measuring how well they support decoding of environment state—including geometry, object structure, and physical attributes—from images. Leveraging simulation environments with access to ground-truth state, we show that this probing accuracy strongly correlates with downstream policy performance across diverse environments and learning settings, significantly outperforming prior metrics. Our study provides insight into the representational properties that support generalizable manipulation, suggesting that learning to encode full environment state is a promising objective for visual representations for control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a new proxy for manipulation downstream policy success by measuring how well visual encoder pretrained backbones can predict the state of the environment (lighting conditions, robot arm joints and end effector pose, objects' positions, orientations, shapes, materials). Training such regressors is possible in simulation where ground truth state is available. For a plethora of backbone encoders, the proxy is shown to provide significantly better correlation to task performance than other methods both in simulation and on real world deployment. In addition, jointly fine-tuning visual backbones (5 different variants) for policy learning as well as state prediction is shown to  consistently offer clear improvements to policy performance on the MetaWorld benchmark.

### Strengths
The paper is well presented and thorough in situating the work within the current literature.
The manuscript is clear and enjoyable to read with a logical and consistent progression.

The solution presented to the proxy measure learning problem is simple and intuitive which makes it convincing. Its implementation in terms of encoding, multi-object handling, loss design and metric definition are all clear and logical.

The experimental protocol is thorough, at least for the base experiments looking into correlation quality, with many relevant benchmarks and visual encoder backbones evaluated both in simulation and in the real world. 

The results appear to back the claims of the paper very elegantly.

### Weaknesses
**On the evaluation protocol:**
- The authors state at line 302 that the protocol follows that of the SimplerEnv benchmark without any additional details or what part of the evaluation this entails, one suspects it is the averaged success rate measurements but that is not clear
- Furthermore, the authors select the Mean Maximum Rank Violation (MMRV) metric as well as the Pearson correlation between performance and state prediction capacity as the statistical measures of correlation without any introduction, explanation or presentation of the what these metrics represent and why those and not others are utilized in the work. This leaves room for doubt as to whether the metric selection is cherry picked to amplify the paper results or objectively standard. This warrants more clarity as the choice is not explained in the main text, nor is it clarified in the appendix.

**On the results:**
- The success rates vary at most by about 15% percent between the best and worst models, with 3 out of four benchmarks having all models perform at low success rates. Each datapoint is obtained with 100 rollouts. Looking only at the MMRV and r numbers provided (and not explained as mentioned above) and average policy success rates with no standard deviations with different x axis scales, it is difficult to appreciate the statistical significance of the results.

**On the analysis of individual state dimensions:**

I thought this was a very nice ablation to have, yet it is not easy to decipher how it was conducted and what is interesting about the results. After some scrolling back and forth my understanding is that from the full state regressor you extract the specific predicted state entries and compute the MMRV score wrt performance. The idea being that the ones with the lowest would correlate the best with performance should they have been used on their own. In many ways this feels incomplete and perhaps the wrong question to ask:
- The state dimensions are hand designed, and although they seem reasonable one could argue that the possibilities/variants might extend to other attributes. If the authors assume that training a regressor on the full state is the best approach, they should make a case that shows that each attribute or dimension in the state is actually contributing to correlation (at least for some backbone/benchmark combination)
- Otherwise the authors should be looking for the best mix of attributes to consider for correlation and considering the couplings that arise both in the training of regressors as well as in establishing the correlation scores.
- Indeed, with numbers varying quite aggressively across benchmarks for a single state dimension on its own, the only conclusion is that (l 458) "that different environments indeed present different demands for visual representations". But what about doing a leaving one out experiment for example to try to understand not what each brings on its own, but which are not that useful in the mix.
- In summary, it is unclear whether all the attributes in the full state regressor are needed and how this affects the state regression quality should they be reduced/modified which can then couple back into the correlation quality.

**Application demonstration asymmetry :**
- First and foremost, though well written, the paper fails to my taste to clearly explain what are the uses of such proxies and why they are important tools. They might be cheaper to evaluate during architecture design or backbone selection for a policy but the case could be. made clearer as to their exact purpose.
- The most interesting application beyond ranking seems to be joint fine-tuning which the authors show provides substantial performance gains across backbone models. This is a very interesting aspect of the work that is largely neglected compared to the potential interest for practitioners. This part of the work warrants more results and details.

The authors have omitted to include the use of LLMs statement.

### Questions
- Are all the tasks considered on parallel gripper pick and place tasks? 

- This circles back to points about what state attributes to consider, for tasks with end effector camera where the arm body is not visible how would you say things would differ, one can only predict general scene and object attributes in this situation. Would similar correlations hold in your opinion?

- The other proxies seem very generic, are there no intermediate metrics or any works that provide proxies that are closer to capturing the state of the scene? 

- Along these lines, are there ways to combine the pre-existing metrics to see if some combination of them can challenge your approach?

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
Training robot policies, especially in the real world, are expensive.  These policies depend upon visual inputs, which generally need to be encoded before they can be fed as input to a policy network.  There are a wide variety of possible ways to encode pixels to inputs that would be useful for a policy network, and the authors aim to identify which visual encoders are most useful without having to train the expensive robot policy.

The authors posit that learning to predict underlying state representation of a robot scene from a visual representation of that scene is a good predictor how useful the resulting visual encoding is for learning control policies.  This visual encoding comes from a variety of pre-trained visual encoders like masked auto encoders, etc. As ground state information about a scene is hard to attain in the real world, the focus is on measuring this predictive effect in simulation.  

The author carefully formulate the learning problem of predicting the state from the visual simulator, with appropriate normalization and discretization applied, and details such as disambiguating which objects are targeted in the state vector by providing 2D bounding boxes for each object are handled in a systematic manner. 

The process of learning to predict the state from a visual encoding is applied to a variety of different pre-trained visual learnt encoders including self-supervised, manipulation specific, and generative models. This state prediction metric is compared to a variety of alternative proxies like segmentation accuracy, etc.  The ranking via this proxy target is more accurate (Mean Maximum Rank Violation) in ranking how well the trained robotics policy will perform.  The state prediction proxy is  also more computationally efficientcompared to other proxies as learning to predict the state is a relatively low dimensional regression problem.

They then demonstrated that this ranking also transferred across the real to sim gap, and that the best visual representation in simulation also predicted best on robot policy learning performance.  They further explored which aspects of the state prediction best predicted which visualization would be best for a task.

### Strengths
## Originality
The proposed metric is to my knowledge novel and useful.  It can also be a useful proxy when designing new visual encoders, as well as for quality control of the final result (and intermediate checkpoints during training), and potentially as additional auxillary loss during training.

## Quality
The authors cast a wide net, and systematically explore the effects of many different visual representations in many robotics tasks.  They empirically demonstrate that their proxy is both more efficient and predictive than alternatives in the literature, and they demonstrate transfer to real robot tasks.

## Clarity
The paper is clearly written, easy to follow with clear reasoning, and would be reproducible from the provided descriptions.

## Significance
Better visual encoders are likely to help robotics, and approaches like those outlines in this paper may useful in developing that research.  It can be expanded to include other predictors like contact surfaces/other information from the simulation.

### Weaknesses
In general, I would expect that L2 losses on poses would have problems with wrapping causing large error.  I didn't notice in the paper where they tackled this potential problem.  E.g., a scene where the objects are aligned in a specific manner (lying on a table?) could have problems where close poses end up with very difference values per axis of the pose, throwing off the pose for in a set of tasks.  

The representation you can get out of a simulator will be a bit limited - it is unlikely to be useful in learning to fold a jumper as it is unlikely there will be a good "symbolic" target to score the outputs of the model against.  That to some degree limits the information you can get out of this proxy target, but it might still be sufficient for ranking the different representations.

### Questions
Have you considered mixtures of visual representations?  E.g., is it more powerful to just concatenate subsets of the representations?  Can you use your method to tell in advance which representations are likely to complement each other or make up for each others shortcomings?

How strong are the effects of initialization on how good the regression solution is?  Does the learning converge to the same point, and if not can you include error bars?

Pose might be highly ambiguous for many objects, e.g., cube have symmetry, vases have rotational symmetry.  Did you encounter such problems in the simulators?

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
This paper proposes state prediction from visual inputs as a fast, reliable proxy for evaluating visual representations in robotics. Instead of expensive policy rollouts, the authors measure how well a frozen encoder supports decoding of ground-truth environment state (geometry, object structure, and physical attributes). They show that state-prediction accuracy correlates strongly with downstream policy performance across multiple manipulation suites (MetaWorld, RoboCasa, SimplerEnv), is orders of magnitude faster than rollouts, and exhibits non-trivial sim-to-real correlation. The work is a simple, practical idea with useful empirical insights for representation selection and benchmarking.

### Strengths
- Strong empirical validation: Correlation holds across MetaWorld, RoboCasa, SimplerEnv with multiple seeds and error bars.

- Computational efficiency: Good speedups over policy rollouts; actionable for everyday / larger-scale benchmarking.

- Unified evaluation setup: Works across environments with a consistent state target.

- Actionable insights: Per-dimension/attribute analyses give interpretability; sim-to-real correlation is encouraging.

- Clarity: Problem framing and metrics are easy to implement and reproduce.

### Weaknesses
-Task diversity and modality coverage: While the results across manipulation benchmarks are compelling, the current evaluation is limited to manipulation-centric settings. Prior work (e.g., VC-1) has shown a form of multi-modality, where visual representations that excel in certain domains (e.g., R3M on MetaWorld) can perform poorly in others (e.g., navigation tasks in Habitat). This raises the question of whether the proposed proxy generalizes across task families with different perceptual and temporal demands (e.g., navigation, long-horizon multi-stage tasks, language-conditioned control). Including experiments or discussion around such cross-domain generalization—or clarifying the scope of applicability—would strengthen the claims and situate the method relative to known modality gaps in representation learning for robotics.

- Simulator dependence: Ground-truth state access limits direct real-world deployment of the proxy; a concrete real-world surrogate (e.g., 6D pose from VLM/trackers) would elevate practicality.

- Failure cases underexplored: Provide root-cause analysis for outliers (e.g., WidowX MAE)—is it state granularity, visuals, or embodiment mismatch?

- Ablations on proxy design: How sensitive are correlations to the decoder capacity, training budget, or subset of state dimensions?

### Questions
0. Could you discuss/explore more in-detail the first weakness I raised in the previous point?

1. Real-world without GT state: What concrete proxy(s) would you propose—e.g., learned keypoints/pose tracking, depth recon, object state estimators—and have you piloted any?

2.  Can you report correlations when decoding only task-critical subsets of state (e.g., end-effector pose + key object states) rather than the full state? This would help clarify whether the proxy is capturing the information actually needed for control, as opposed to benefiting from full-state regression capacity.

3.  How does correlation change when using smaller decoders or shorter training? Understanding this would help confirm that the proxy reflects encoder quality rather than decoder strength, and that the metric remains reliable under lower-compute settings.

4.  Sim-to-real outlier analysis: What explains the WidowX MAE anomaly? Can you isolate the culprit (visual domain, state scaling, or dynamics mismatch)?

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
The paper proposes a training/evaluation-free proxy for selecting visual backbones for robot manipulation. It includes regressing a unified simulator state from images using the frozen backbone + MLP, then reranking backbones by this state prediction score. They show the proxy correlates strongly with policy success across 3 simulation benchmarks, and provide a real-world validation. The method is computationally cheaper than baselines, and improves policy success when used as an auxiliary objective.

### Strengths
* The method is simple and efficient. The predicted state covers information about object and scene-level variables, finally providing a single score to policy performance.

* The paper evaluates across a breadth of envs, including 3 simulation envs and real-world evaluation on two tasks.

* The paper uses a strong set of baselines, including few-shot, action MSE, Depth, etc.

### Weaknesses
1) The method relies on privileged information from the simulator (state + 2D object boxes) which can not be made available in the real world. It is not clear to me from the current set of experiments if the rankings would correlated if the real world env is substantially different from the simulated environment where the data is collected.

2) As tasks get more complicated, the number of state variables to track will keep on increasing. For example, if the task requires picking up objects of different categories, it would require more variables to track.

3) The authors should compare the real world performance of their approach with other visual backbone selection proxies. Some of these proxies (eg: Action MSE, Depth (can be obtained from the RealSense camera)) do not need privileged information from the simulator and therefore I suggest that the authors also report results when using the data from the real world.

4) I would have liked to see more robotics specific visual encoders like VC-1, MVP [1], LIV [2], etc. They would have also added some diversity in terms of the dataset used in training of the vision encoder.

5) The paper appears to use a baseline that closely matches the method proposed in prior work (SCR: Stable Control Representations [3]), but this connection is not acknowledged or cited; the authors should clarify this and include the appropriate reference.

[1] Xiao, Tete, et al. "Masked visual pre-training for motor control." arXiv preprint arXiv:2203.06173 (2022).

[2] Ma, Yecheng Jason, et al. "Liv: Language-image representations and rewards for robotic control." International Conference on Machine Learning. PMLR, 2023.

[3] Gupta, Gunshi, et al. "Pre-trained text-to-image diffusion models are versatile representation learners for control." Advances in Neural Information Processing Systems 37 (2024): 74182-74210.

### Questions
See point 3, 4 and 5 above.

### Soundness
2

### Presentation
3

### Contribution
2
