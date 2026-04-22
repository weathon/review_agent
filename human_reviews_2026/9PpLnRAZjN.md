# End-to-End One Step Flow Matching via Flow Fitting

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Diffusion and flow-matching models have demonstrated impressive performance in generating diverse, high-fidelity images by learning transformations from noise to data. However, their reliance on multi-step sampling requires repeated neural network evaluations, leading to high computational cost. We propose FlowFit, a family of generative models that enables high-quality sample generation through both single-phase training and single-step inference. FlowFit learns to approximate the continuous flow trajectory between latent noise (x_0) and data (x_1) by fitting a basis of functions parameterized over time (t \in [0, 1]) during training. At inference time, sampling is performed by simply evaluating the flow only at the terminal time (t = 1), avoiding iterative denoising or numerical integration. Empirically, FlowFit outperforms prior diffusion-based single-phase training methods achieving superior sample quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a single-phase method to learn one-step map to approximate the ODE trajectory in flow matching to enable one-step generation. The authors introduce basis functions in time to model the one-step map to reduce the computation of backpropagation in training.

### Strengths
The paper is well organized and easy to follow. The authors introduced fixed basis functions in time to model the one-step map in order to reduce the computation of backpropagation in training. Experiment results also verify the efficiency of the proposed method.

### Weaknesses
The idea is quite naive overall, using a one-step map to approximate the ODE trajectory via matching velocity. Although the author introduces the basis functions, it is basically doing distillation over a pre-trained flow matching model. I understand that this is a single phase method and the training of one-step map can be in parallel with FM, but the computation cost is still doubled. The method is not very appealing to me.

### Questions
Can the authors provide results for multiple step evaluation of the proposed method? I understand the primary goal is to do one-step generation, but I still want to know if the proposed method can work well when increasing computational budget.

### Soundness
2

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
This paper presents FlowFit, a novel framework for flow modeling via basis function fitting that enables high-quality sample generation through both single-phase training and single-step inference, which avoids iterative numerical integration like denoising diffusion models. To do this, the paper directly parameterizes the continuous-time flows using a residual expansion over fixed basis functions. This framework can be used in one step generation and distillation of pretrained models. The paper experiments on CelebAHQ-256 and ImageNet-256 to show that the proposed method outperforms prior diffusion-based single-phase training methods.

### Strengths
•	The paper is clearly organized with direct motives.

•	The proposed method of using basis function to directly distill the learned velocity field is simple and straightforward.

### Weaknesses
•	The content of the paper is inadequate. It would be beneficial to add more discussions on the choice of basis functions and the capacity on approximating certain flows for specific tasks. It remains suspicious to me that the proposed method could scale up to more general complex generation tasks, in which the optimal velocity field and the flow map is highly curved and the degree of the mapping function is large.

•	The paper lacks theoretical analysis of the proposed method. It would be beneficial to theoretically discover how large the basis function family is needed for a given task in mathematical details, a simple synthetic example could suffice. The comparison may utilize the theory in polynomial regression analysis.

•	In my view, the empirical comparison in the experiments is a bit obscure and misleading. It is somewhat unfair to put the proposed method FlowFit in the column of “1-step” with the other method’s “1-step” like standard flow matching and consistency models, since the FlowFit method regards the multi-step trajectory as a single complex function map. The complexity of the “1-step” for FlowFit method is much higher than the “1-step” for a single step of sample update according to the learned velocity field in standard flow matching and consistency models. It would be more convincing to show the effectiveness of the proposed method by presenting the exact numerical comparison of the total time cost.

### Questions
•	Besides the empirical comparison, could the authors provide an understanding on why directly fitting the map could be better than other few step generation methods based on updating samples according to the velocity field? It seems to me that the difficulty in fitting the velocity field is not less than directly updating the samples through the distilled velocity field in previous few step generation methods like diffusion distillation.

•	Could the flow fitting strategy be applied to other classical flow trajectories like diffusion paths in one step generation tasks like diffusion distillation? If yes, will the type of the true flow trajectory affect the choice of the basis function family and the difficulty of flow fitting?  Could the authors provide some understanding on whether and why the flow fitting strategy will outperform other diffusion distillation methods in this case?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This manuscript proposes a new type of 1-step generative models called FlowFit. FlowFit approximates the flow trajectory using basis functions, and trains networks to predict the coefficients of the bases. Training is conducted by simultaneously training a flow matching model to learn the velocity field, and matching the derivative of the approximated trajectory to the learned velocity field. The proposed method shows promising results on the DiT-B architecture. Although the FID is not as strong as the SoTAs (e.g., MeanFlow DiT-B FID is 6.17 on ImageNet), the novelty of the method itself is a solid contribution that could inspire future works.

### Strengths
- The idea of fitting basis functions to approximate the flow trajectory is very original, and the training objective of matching the derivative of the function also differs from previous one/few-step models, such as consistency models. The fact that a network is able to predict all the coefficients of the bases in one step is very intriguing. 
- Apart from the novelty, the method is also relatively simple and elegant, in contrast to SOTA consistency models which often involve adaptive losses, complex scheduling or inefficient JVP.
- The generation quality, as measured by FID, appears strong enough to outperform older generations of consistency models, such as iCT, sCT.
- The presentation is clear and focused, although adding a graphical illustration may further strengthen the clarity.

### Weaknesses
- The major limitation of this work is that the FID is relatively underperforming compared to SOTA consistency models. For example, on ImageNet 256x256, MeanFlow (also using DiT-B) achieves an FID of 6.17, whereas the proposed approach attains an FID of 34.4.
- Experiments are only done on DiT-B. Larger models (e.g., DiT-XL) are not tested.
- The proposed method requires training two models, one for flow matching and one for trajectory matching. This is more like online distillation rather than "end-to-end".
- Minor formatting issue: the reference format is not following the official ICLR template, which should be name + year instead of numbers.

### Questions
The paper shows improving performance with increasing order, but stopped at order 8. In L411, the authors explain that this is due to computational constraints. Why is maximum order limited by computation? I thought most of the computation expenses are from the neural network, and increasing the number of bases would not introduce significant overhead in training and inference (because evaluating these simple functions is usually fast)?

### Soundness
3

### Presentation
3

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
This paper proposes flow fitting that train two networks which are velocity flow matching and flow model that align with the velocity flow matching model. The flow trajectory model is defined in terms of basis function.

### Strengths
1. The model proposes to use separate network called trajectory flow to fit with training velocity model, which is new idea than the unified model like the meanflow and shortcut model.
2. The model performance beats shortcut in some setting. 
3. The code implementation are provided.

### Weaknesses
1. The paper writing seems unprofessional with poor writing in introduction (too many short paragraph). The related works is not well-studied and lacks many recent works like [1, 2, 3].

[1]: Inductive Moment Matching - ICML

[2]: Improved Training Technique for Latent Consistency Models - ICLR

[3]: Consistency Models Made Easy - ICLR

2. The authors do not clarify why not use single deep learning model instead of both basis function and deep network to estimate coefficient.

3. The paper only compares with shortcut model in limited setting (undertraining). How about training model until converge with DiT-XL/2.

4. In ablation study, the experiment details are not provided like what dataset, training iterations and other details. The authors just simply put a table without any explanation to guide the reader.

### Questions
Please see the weakness

### Soundness
1

### Presentation
1

### Contribution
2
