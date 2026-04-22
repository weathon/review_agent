# What Happens Next? Anticipating Future Motion by Generating Point Trajectories

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
We consider the problem of forecasting motion from a single image, i.e., predicting how objects in the world are likely to move, without the ability to observe other parameters such as the object velocities or the forces applied to them. We formulate this task as conditional generation of dense trajectory grids with a model that closely follows the architecture of modern video generators but outputs motion trajectories instead of pixels. This approach captures scene-wide dynamics and uncertainty, yielding more accurate and diverse predictions than prior regressors and generators. Although recent state-of-the-art video generators are often regarded as world models, we show that they struggle with forecasting motion from a single image, even in simple physical scenarios such as falling blocks or mechanical object interactions, despite fine-tuning on such data. We show that this limitation arises from the overhead of generating pixels rather than directly modeling motion.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors focus on efficient approaches to forecasting dynamics in an image. To this end, they propose, instead of modeling pixels directly, to forecast point trajectories. They utilize a beta-VAE combined with Flow Matching to generate future point trajectories conditioned on an image. The authors then comprehensively evaluate the performance of their model, looking at both statistical and physical plausibility. They evaluate on robotic datasets such as Libero as well as synthetic datasets (Kubric). The also include a user study of performance.

### Strengths
1. The paper explores a less studied avenue of visual dynamics: predicting pixel motion versus pixels themselves. This approach makes intuitive sense.

2. Quantitative evaluation is comprehensive and convincing. 

3. The paper is well-written and easy to understand.

### Weaknesses
1. There is a prior work, "An Uncertain Future: Forecasting from Static Images using Variational Autoencoders", Walker et al, ECCV 2016, which is similar and also uses a VAE to forecast pixel trajectories from a single image. This work should be cited.

2. Results of the user study are somewhat weak (preferred over other approaches only 52% of the time).

3. It would be helpful to see a few more qualitative results from the Physics 101 dataset.

### Questions
1. Please cite the paper above. It is similar to approach mentioned in this paper.

2. Please elaborate more on the user study, and would it be possible to include more results from Physics 101?

### Soundness
4

### Presentation
4

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
This paper addresses the problem of predicting object movement from a single static image, where future motion is inherently uncertain. The authors propose a method that directly generates a dense grid of point trajectories (coordinates) conditioned on the input image, rather than generating future video frames (pixels). They utilize a generative architecture (similar to modern video generators, specifically using flow matching) to model the distribution of possible future movements.

### Strengths
1. Generates point trajectories instead of pixels, bypassing the complexity of video generation. This allows the model to efficiently learn physical dynamics and object consistency.
2. Uses a generative model to predict the distribution of trajectories, producing multiple physically plausible futures, which better reflects real-world information gaps.
3. Achieves superior performance in physical reasoning tasks compared to prior trajectory regressors and state-of-the-art video generators pre-trained on massive datasets.

### Weaknesses
1. Predictions are based on a single image, lacking precise initial velocity or force data. Accuracy heavily relies on the model's learned "physical common sense" rather than precise observation. I think this task setting quite strange.
2. Evaluation primarily uses synthetic datasets (e.g., Kubric, Physion). Its generalization capability to complex, "in-the-wild" real-world physics (e.g., fluid, soft bodies) remains questionable.

### Questions
1. The point I understand the least is why, for a scenario where object motion needs to be predicted, it is limited to using only a single static image. Alternatively, is there a way to adapt this model to also support video or multi-frame input? And how would its performance be?
2. It seems to lack some relevant paper citations, such as Flow-grounded spatial-temporal video prediction from still images, papers about future frame synthesis/prediction

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
3

### Summary
This paper addresses the problem of forecasting object motion from a single image—predicting potential object movements without observing velocities or applied forces. It formulates the task as conditional generation of dense point trajectory grids, drawing inspiration from modern video generator architectures but outputting trajectories instead of pixels. The model uses a trajectory VAE for latent space mapping and flow matching for sampling, capturing scene-wide dynamics and uncertainty.

### Strengths
1. It innovatively redefines motion forecasting as dense trajectory grid generation, moving beyond prior work focused on sparse active points (e.g., ATM, Tra-MoE) or pixel-based video generation, enabling better global scene dynamics modeling.
2. The study uses diverse datasets and comprehensive metrics to assess accuracy, diversity, and physical plausibility, with ablations strengthening conclusions

### Weaknesses
- **Lack of Downstream Task Evaluation for Video Generation**: While the paper argues that trajectory generation outperforms video generation for motion forecasting, it does not test how the two approaches perform on downstream tasks that rely on motion, e.g., robot manipulation, video interpolation, or action prediction.   
- **No Analysis of Query Point Sampling Density**: The paper only discusses ablations of latent code dimensions but does not explore the impact of query point sampling density.   
- **Insufficient Real-World Data Results and Visualizations**: Real-world validation is limited to the Physics101 dataset without testing on other real datasets, e.g., egocentric motion datasets.

### Questions
In the absence of a task description (e.g., for the action "Pick up the book on the right and place it under the cabinet shelf"), how does the paper address the diversity of action prediction, and are there any controllable settings integrated into the pipeline?

### Soundness
2

### Presentation
3

### Contribution
2
