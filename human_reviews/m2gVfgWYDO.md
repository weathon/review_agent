# Tracking objects that change in appearance with phase synchrony

- Decision: Accept (Poster)
- Scores: 8, 3, 8

## Abstract
Objects we encounter often change appearance as we interact with them. Changes in illumination (shadows), object pose, or the movement of non-rigid objects can drastically alter available image features. How do biological visual systems track objects as they change? One plausible mechanism involves attentional mechanisms for reasoning about the locations of objects independently of their appearances --- a capability that prominent neuroscience theories have associated with computing through neural synchrony. Here, we describe a novel deep learning circuit that can learn to precisely control attention to features separately from their location in the world through neural synchrony: the complex-valued recurrent neural network (CV-RNN). Next, we compare object tracking in humans, the CV-RNN, and other deep neural networks (DNNs), using FeatureTracker: a large-scale challenge that asks observers to track objects as their locations and appearances change in precisely controlled ways. While humans effortlessly solved FeatureTracker, state-of-the-art DNNs did not. In contrast, our CV-RNN behaved similarly to humans on the challenge, providing a computational proof-of-concept for the role of phase synchronization as a neural substrate for tracking appearance-morphing objects as they move about.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors research how a neural network architecture and training scheme could be set up by drawing inspiration from neuroscience, such that it performs better in object tracking. In particular, in scenarios where the objects may change appearance. The main goal, however, is not simply increasing performance but rather to better understand observations in neuroscience by implementing hypothesized mechanisms in an ANN. 
Concretely, in this task setting, object properties such as shape and color change while an object is moving over a trajectory, and the model has to decide if the selected object reaches the target position. Additionally, the object may be occasionally occluded. For training and benchmarking on this task, the authors construct a new dataset and challenge for object tracking. Human performance is assessed on the test set of this object tracking challenge. S.o.t.a. DNN models for object tracking are benchmarked and their failures analyzed. A new RNN architecture and training scheme with a specific loss function is established that leverages complex-valued attention units to simulate neural synchrony observed in real neurons of the visual pathway. The new architecture is able to reach human performance in several experimental settings and also shows similar failure modes as humans.

### Strengths
1. The paper establishes a new dataset and challenge for object tracking with changing object appearance. The setup to generate this synthetic data allows customization for specific research questions related to object tracking and ablation studies.
2. The authors present a new architecture and establish convincing experiments and results for why the new model (and training technique with specific loss function) brings the model mechanism closer to what is observed in neuroscience in terms of neural synchrony.
3. The paper is clearly structured and introduces the topic as well as explains the main goal in a clear fashion.  I like how the main model and challenge is introduced by first analyzing a more simple setting (shell game) both in humans and an initial model, which then leads to the main dataset and model. Most questions I had during reading were promptly answered in the main text or appendix, and the paper carefully sets expectations in terms of ML and neuroscience perspective, and the required ablations and add-on experiments were done.

### Weaknesses
1. The training and test data is made of synthetic data of single color shapes, it is unclear how well it translates to real videos (the authors leave this point for future work) and objects. An interesting intermediate step toward a more natural setting could have been textured shapes and/or a (ideally non-static) background image.
2. An interesting model to try besides the presented baselines would have been other complex-valued RNNs that are not bio-inspired but compatible with the introduced synchrony loss. This would have added the understanding of the impact of the bio-inspired architecture that InT provides.
3. Although different task variations are benchmarked, the limited scope to just one dataset (that the authors created) make it difficult to compare to other approaches that are specialized on changing appearance, e.g. [2] benchmarks on 4 datasets.

Just a suggestion (space is limited after all): there is a vast amount of literature of complex-valued RNNs (a survey that may also contain relevant RNNs is [1]) presented after the 'old' ones cited in the paper, and some citations could be added.

[1] Lee, ChiYan, Hideyuki Hasegawa, and Shangce Gao. "Complex-valued neural networks: A comprehensive survey." IEEE/CAA Journal of Automatica Sinica 9.8 (2022): 1406-1426.
[2] Cai, Yidong, et al. "Robust object modeling for visual tracking." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

### Questions
1. How do you assess the amount of error information that is provided during training of the baseline models vs the specific loss function (synchrony loss) for the CV-RNN? Could some of the increased test performance also be due to more efficient training with more loss information?
2. As far as I understand the data generation pipeline, all objects are initialized at the start and exist until the end of the sequence (besides occasional occlusions). Do you expect the model would still perform, if e.g. objects move in and out of the frame, or just vanish, or appear mid sequence? What if this only happens during testing?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors implement a particular biologically-inspired hypothesis for visual object tracking into a new-ish neural network (CV-RNN) type an then compare the results to standard network architectures.

### Strengths
I think the question of object tracking is interesting. 

The benchmark is interestingly conceived.

### Weaknesses
The evaluations are very simple and don't really support strong claims about the new architecture being much better than previous architectures.   

The test of "standard" DNNs for the purpose of baselines is perhaps a little shaky. 

The benchmark, while interesting, is limited by being in such a toy condition.    I would be much more convinced if the algorithms here were also tested on recent object tracking benchmarks such as TAP-VID.  (Or if it was convincingly explained why such benchmarks are inapproprioate tests.)

### Questions
There are real-world object tracking benchmarks, such as TAP-VID.  Can the algorithms here work there?  Do they outperform  recent good strategies for tracking in those cases?

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
This paper presents CV-RNN, a deep learning circuit designed to replicate the biological visual system's ability to track objects with changing appearances by leveraging neural synchrony. The CV-RNN encodes object identity using the phase of complex-valued neurons, enabling the network to track objects as their appearance evolves over time. Through the "FeatureTracker" challenge, the authors demonstrate that while humans can effectively track objects despite changes in color and shape, conventional deep learning models struggle with this task. The CV-RNN, however, approaches human-level performance by employing phase synchronization to track objects with changing appearances.

### Strengths
1. The design of CV-RNN is well-motivated by prominent neuroscience theories (e.g., binding-by-synchrony) and serves as a proof-of-concept that neural synchrony aids in object tracking.
2. The combination of the FeatureTracker challenge and human psychophysics experiments provides a valuable framework for studying object tracking in a controlled environment, emphasizing the significant performance gap between current deep learning models and human capabilities.
3. CV-RNN outperforms other baseline models in the FeatureTracker challenge, demonstrating the efficacy of complex-valued representations.

Overall, the paper is well-motivated and clearly written. The main claims are supported by robust evidence.

### Weaknesses
1. Although CV-RNN approaches human performance on the FeatureTracker challenge, the use of synthetic datasets featuring simplified changes in appearance and shape may limit the generalizability of the findings to real-world object tracking, which is more challenging. Using naturalistic videos such as DAVIS would significantly strengthen the paper’s claims.
2. Comparing with self-supervised visual representation learning methods such as VideoMAE and DINO would be useful. The baseline methods in this paper — TimeSformer, MViT, ResNet3D, and MC3 — are supervised video action recognition models trained with specific activity labels, which may limit their generalizability to out-of-distribution datasets like the FeatureTracker challenge. Self-supervised learning methods have been shown to learn visual representations that transfer effectively to downstream vision tasks and could serve as competitive baselines for CV-RNN. If the authors believe self-supervised methods are not suitable baselines, it would be useful to provide a justification.

3. There is a missing comparison to RNN-L in Figure 5. The RNN performs worse than CV-RNN, which could be potentially due to its smaller number of parameters (108,214 vs. 171,580). The supplementary material (Table 5) mentions an RNN-L variant with a model size comparable to CV-RNN. Including its quantitative performance in Figure 5 would help demonstrate that the improvement of CV-RNN is independent of model size. If there was a specific reason for their omission, the authors can provide an explanation to clarify the rationale behind the omission.
4. Legend is missing in Figure A.10. Although efficiency is not the primary focus of this paper, an analysis of inference time and memory requirements for CV-RNN compared to baseline methods would be valuable. While Figure A.10 seems to include such a comparison, the missing legend makes it difficult to interpret.

### Questions
See Weaknesses

### Soundness
4

### Presentation
4

### Contribution
4
