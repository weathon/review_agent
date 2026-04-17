# Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Predicting physical dynamics from raw visual data remains a major challenge in AI. While recent video generation models have achieved impressive visual quality, they still cannot consistently generate physically plausible videos due to a lack of modeling of physical laws. Recent approaches combining 3D Gaussian splatting and physics engines can produce physically plausible videos, but are hindered by high computational costs in both reconstruction and simulation, and often lack robustness in complex real-world scenarios. To address these issues, we introduce **Neural Gaussian Force Field (NGFF)**, an end-to-end neural framework that integrates 3D Gaussian perception with physics-based dynamic modeling to generate interactive, physically realistic 4D videos from multi-view RGB inputs, achieving two orders of magnitude faster than prior Gaussian simulators. To support training, we also present **GSCollision**, a 4D Gaussian dataset featuring diverse materials, multi-object interactions, and complex scenes, totaling over 640k rendered physical videos (∼4 TB). Evaluations on synthetic and real 3D scenarios show NGFF’s strong generalization and robustness in physical reasoning, advancing video prediction towards physics-grounded world models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces NGFF, a framework capable of generating videos with physics-grounded dynamics for scenes containing multi-body interaction. For foreground objects represented by 3D Gaussians, a DeepONet, which consumes the geometric and dynamic attribute of pairwise objects, is employed to predict the received forces and torques in a feed-forward manner. Upon this, local stress fields are predicted to model local deformations. The learned force field is then integrated with an ODE solver to predict object’s motion. In conjunction with the model, they also constructed a dataset containing dynamic Gaussians simulated by MPM for training and benchmark the proposed method. Experimental results show that the proposed feed-forward method can achieve accurate dynamic prediction and demonstrates clear generalization in some aspects.

### Strengths
1.	The paper is well-written and easy to follow. 
2.	This work shows that force fields in multi-object interactions can be efficiently and accurately predicted by NGFF, with strong positional, temporal and compositional generalization.

### Weaknesses
1.	L192 mentions that they leverage 3D generation model to ”address occlusions and invisible parts” from multi-view input, but how to ensure consistency between assets obtained via single (front) view-to-3D generation and the input multi view content in regions invisible in the front view?
2.	In Equation (3), why can the local force field be predicted only from the net force/net torque, contact region and location/velocity of its own each point, which seems under-determined.
3.	The framework does not consider lighting effects induced by motion, which are crucial for realistic video synthesis. This omission weakens its claimed competence as a video generation method.
4.	Fitting the dynamics of only 10 objects already requires a 4T dataset, which appears to suggest a poor data efficiency.

### Questions
1.	Does “10 objects” refer to 10 individual instances or 10 categories? If it means categories, how many objects collected for each category, and why not assess generalization to novel instance? 
2.	Should generative modeling be considered when scaling to more diverse categories?
3.	L965 says that state descriptors are embedded via an MLP, while L208 mentioned that the state $s(t)$ and $\dot{s}(t)$ containing point cloud and its velocity in $M\times3$. Are these the same “state”? If so, how are the point clouds processed in this MLP?
4.	Some notations in Equation (1) lack clear definition. E.g., $I_{t,k}$ and $\hat{G}$ are not defined in the context. Does this loss aim to enforce consistency of parameters and rendered image between simulated and predicted Gaussian?

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
This paper introduces Neural Gaussian Force Field (NGFF) to address the challenge of predicting physical dynamics from raw visual data. It learns an implicit force field to drive the temporal evolution of 3D Gaussian Splatting, thereby modeling 4D dynamics. Experimental results demonstrate that NGFF achieves state-of-the-art performance in the 4D generation of collision motion in both simulation and real-world cases.

### Strengths
1. The idea of using an implicit force field, rather than an explicit physics engine, to govern 3D GS evolution is highly innovative. It effectively tackles the critical bottlenecks of large computational overhead and insufficient robustness inherent in tightly coupling 3D GS with physical simulation.
2. The paper provides extensive quantitative and qualitative results across diverse and challenging evaluations, which demonstrate NGFF's effectiveness in both 4D dynamic prediction and the more challenging real-world simulation.

### Weaknesses
1. Limited Performance: Since the model is primarily trained on data generated by the Material Point Method (MPM) simulation, its learned dynamics are inherently limited by the accuracy, fidelity, and approximations of the underlying MPM solver. This raises a concern that the performance of the trained NGFF model is limited to the quality of the synthetic ground truth.
2. Generalization to Unseen Physics: While NGFF performs excellently on the provided datasets, its ability to generalize to unseen material properties (e.g., how a model trained only on rigid and soft bodies handles sand or fluids) or unseen complex constraints (e.g., complex joints or hinges) remains unclear. This is a crucial factor for real-world applicability.
3. Missing Implementation Details: Modeling and training details are not clear enough, making this paper hard to follow. For example, what attributes of Gaussian kernels need to be supervised, and how is L' calculated in Eq.1? What is $s$ in Eq.6, and how to convert it to the state of Gaussian kernels?

### Questions
My questions are mainly based on the above weaknesses:
1. Do the predictions from NGFF perform better compared to the MPM simulation results? Is it possible to train the model with real captured data?
2. The predicted simulation in the paper is limited to the collision of rigid or soft body objects. Can this method be extended to more complex physical motions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The NGFF framework integrates 3D Gaussian perception with Neural Gaussian Force Fields to generate interactive, physically realistic 4D videos from multi-view inputs. It models dynamics by learning an explicit force field integrated via an ODE solver, achieving two orders of magnitude faster simulation than previous Gaussian methods.

### Strengths
The NGFF framework integrates 3D Gaussian perception and Neural Gaussian Force Fields to generate interactive, physically realistic 4D videos. It models dynamics by learning an explicit force field, achieving two orders of magnitude faster simulation. It also introduces the GSCollision dataset.

### Weaknesses
The NGFF framework currently relies on multi-view inputs for reliable 3D Gaussian reconstruction, needing extension to monocular or partial observations to match human ability. The benchmark covers only 10 representative objects, necessitating scaling to far more diverse materials and articulated structures. Additionally, visual quality is sometimes slightly lower than SOTA video generation models due to 3D reconstruction error.

### Questions
How will you extend NGFF to robustly predict 4D dynamics from monocular or partial RGB inputs?

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
4

### Summary
This work presents the Neural Gaussian Force Field (NGFF) for animating 3D Gaussians. In summary, subjects are reconstructed from multi-view visual observations, and a continuous force field animates these reconstructed objects. The force field (NGFF) is predicted using a neural network. To train this force-prediction network, the authors simulate a large-scale dataset using the Moving Particle Method (MPM). The key contributions of this work are the development of the force-prediction network and the creation of the proposed dataset.

### Strengths
This work introduces a new approach to generating physically plausible 4D Gaussians in a feed-forward manner. The force-field prediction network is innovative.

The motivation and method are clearly presented, making them easy to understand.

The proposed dataset is valuable for studying 4D physics-plausible Gaussians.

### Weaknesses
I have two concerns on this work.

I have two concerns regarding this work:

1. The quality of the proposed dataset:
   (1) Is the MPM simulator truly representative of real-world physics? I worry that a simple MPM may not produce high-quality physics data. If that’s the case, the dataset's value may be diminished.
   (2) The simulations are conducted in a simple box environment, which lacks complexity and reduces diversity.
   (3) Although 3DGS offers high fidelity as a representation, there is still a noticeable gap between the rendered video and real-world captured video, as illustrated in Fig. 3.

2. The soundness of the force field prediction network:
   (1) The predicted target is highly dimensional and ambiguous. I doubt whether the model can effectively learn PDE dynamics from the data.

### Questions
- The author should provide clearer information about the dataset: (1) How many particles are included? (2) How are collisions handled? (3) Which constitutive model is used in the Material Point Method (MPM)? (4) How many simulation steps were performed?

- How can you demonstrate that the predicted results align with PDE knowledge rather than relying solely on mass dynamics? (This is an open question.)

- Have you considered objects with uneven quality distribution?

- If the surface of the environment is more complex, such as featuring uneven slopes, can your model accommodate this?

### Soundness
2

### Presentation
3

### Contribution
2
