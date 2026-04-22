# FALCON-S: Fixed-wing Aerodynamics and Learning Control Suite

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
We introduce FALCON-S, a modular and high-fidelity framework for learning and control of fixed-wing aerial vehicles operating in ground effect. In contrast to existing aerial platforms with simplified dynamics, FALCON-S incorporates full 6DoF simulation alongside detailed modeling of ground-effect aerodynamics, actuator dynamics, and environmental disturbances. It offers a level of physical fidelity and modular component design that enables fine-grained manipulation and systematic analysis of low-altitude flight phenomena, capabilities rarely found in open-source or state-of-the-art simulation platforms. The framework includes both CPU and GPU simulation backends via Python and NVIDIA Warp, supporting high-throughput simulation across up to millions of parallel environments, which makes it suitable for reinforcement learning, sampling-based control algorithms, and large-scale evaluation. FALCON-S features a flexible architecture with interchangeable controllers, supporting optimal control, model-free and model-based RL, as well as a suite of flight control tasks such as altitude regulation and trajectory tracking. We include optional interfaces for validation and comparison through MATLAB/Simulink and XPlane, making it compatible with both engineering workflows and commercial simulators. The framework is released as open-source to facilitate reproducibility and enable controlled benchmarking in realistic flight scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FALCON-S, a new, open-source simulation framework for developing and bench-marking control algorithms for fixed-wing aerial vehicles. The main contribution is a simulator unlike previous work that provides :- 

(1) High fidelity Physics - 6 DoF dynamics model that includes components for Aerodynamics, first or second order actuator dynamics, environmental dynamics and sensor models 
(2) High throughput scalability - Dual backend architecture for CPU/GPU based pipelines with the GPU backend being able to simulate "up to millions of parallel environments" wit ha single step time.
(3) Modularity - The framework supports both reinforcement learning and optimal control methods with a gym api being exposed for RL methods
(4) Cross platform validation interfaces - Validation capabilities for both MATLAB/Simulink and X Plane.

The paper is motivated by the need to bridge sim-2-real gap as existing simulators are either high-fidelity but too slow for RL (JSBSim, FlightGear, and X-Plane) or high-throughput but lack the necessary physical realism (Flightmare, NeuralPlane)

### Strengths
1. Originality - It correctly identifies and addresses an important gap in the existing landscape of aerial vehicle simulators - either they are slow or too oversimplified and lack physics realism. Its the mix of both that is available in FALCON-S that pushes it for the win
2. Quality -  The quality of engineering work presented in this paper is absolutely amazing. The physics model that includes interconnected components for - Aerodynamics, Actuators, Environmental Effects and Sensors- is a significant strength with the dual backend architecture for high throughput simulation
3. Clarity - Absolutely perfect technical clarity with the writing and well structured, making it easy to follow. The figures and tables are also very effective with the appendices providing deep technical clarity
4. Significance - This work is highly significant as an open source community resource as it provides a strong instrument for the robotics, aerospace, and RL communities.

### Weaknesses
1. The paper's primary motivation -bridging the "sim-to-real gap"  - is left entirely unsubstantiated. There is no experiments done on real world hardware.
2. 75% of the experimental results use a classical LQR and just 1 experiment using Dreamer V3. The paper thus provides zero insights about learning and just is flexing the simulator features. And it DreamerV3 is just tested in isolation for some reason. 
3. It is stated - "Our experiments are designed to highlight the flexibility and realism of the FALCON-S framework, rather than to optimize or compare specific learning or control algorithms" - this probably goes against the necessity for a learning based conference.

### Questions
1. Could the authors provide more experimental results for learning based algorithms ? As we are only relying on a non-learning controller for the main results
2. The MPPI controller failures seem to be omitted from main text and are pushed to the appendix (Appendix B2 (Table 9)). What was the reasoning behind LQR passing and MPPI failing in such scenarios ? 
3. Why should the ICLR community accept the fact that this framework helps bridge the gap when there are no real world experiments ? 
4. Why did you use a classical LQR controller for all the key physics tests (like sensor noise, ground effect, and delays)? Since this is a learning conference, shouldn't these experiments have used a learning agent like PPO or DreamerV3 to show how these realistic physics impact the learning process itself ?

### Soundness
1

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces FALCON-S, a modular, high-fidelity simulation framework designed for learning and control of fixed-wing unmanned aerial vehicles (UAVs) operating in ground effect. The simulator integrates full 6DoF rigid-body dynamics, advanced aerodynamic modeling, including semi-empirical ground effect corrections—and realistic actuator and sensor dynamics. It supports GPU-accelerated simulation via NVIDIA Warp, enabling high-throughput training across millions of parallel environments. The framework is highly extensible, with support for both classical (e.g., LQR, MPPI) and modern deep reinforcement learning (e.g., PPO, DreamerV3) controllers. It also includes interfaces for cross-validation with MATLAB/Simulink and X-Plane. The authors demonstrate the platform’s utility through illustrative experiments on multi-task generalization, cross-aircraft evaluation, and robustness to environmental disturbances.

### Strengths
1. **High Fidelity and Realism:**
FALCON-S stands out for its realistic modeling of ground effect, aerodynamic coefficients, actuator dynamics, and sensor imperfections—features often missing or oversimplified in existing simulators like QPlane or NeuralPlane.

2. **Scalable, GPU-Accelerated Simulation:**
The use of NVIDIA Warp enables a single-step simulation time of 0.0022 seconds across 1 million parallel environments - i.e., a 100× speed-up over state-of-the-art. This is a major technical achievement and crucial for efficient RL training.

3. **Modular and Extensible Architecture:**
The framework is well-organized into agent and environment modules, with clean separation of concerns. The ability to toggle physical models (e.g., ground effect, turbulence) and swap controllers (classical vs. learning-based) enables rigorous ablation studies.

4. **Strong Validation and Interoperability:**
The inclusion of X-Plane and MATLAB/Simulink interfaces is a significant asset for cross-platform validation, sim-to-real transfer, and integration with industry-standard workflows.

5. **Open-Source and Reproducible Research:**
The code is released under an open-source license (via anonymous 4open.science), promoting reproducibility, benchmarking, and community adoption.

### Weaknesses
1. **Limited Quantitative Benchmarking:**
While the paper presents several illustrative experiments, it lacks comprehensive, quantitative benchmarks comparing FALCON-S against existing platforms (e.g., NeuralPlane, QPlane) on standard control tasks. A formal benchmarking study would strengthen the claim of superiority.

2. **Lack of Hardware Integration or Real-World Deployment:**
The paper focuses on simulation and learning, but there is no discussion of hardware-in-the-loop (HIL) testing, flight testing, or sim-to-real transfer in practice. This limits the paper’s impact on real-world deployment.

3. **Insufficient Discussion of Computational Overhead and Memory Usage:**
While speed is emphasized, the paper lacks detailed analysis of memory footprint, GPU memory usage, and scalability limits (e.g., how many agents can be simulated before performance degrades). This is critical for practical deployment.

4. **Limited Explanation of Ground Effect Model Parameters:**
The ground effect model is based on empirical corrections (Phillips & Hunsaker, 2013), but the paper does not clearly explain how parameters (e.g., h/b ratio, aspect ratio) are estimated or tuned for different aircraft. This reduces transparency.

### Questions
1. How does FALCON-S compare to existing platforms (e.g., NeuralPlane, QPlane) in terms of simulation accuracy and computational efficiency across a wide range of tasks?
2. Could the authors provide a detailed breakdown of the memory and GPU usage during large-scale parallel simulations (e.g., 1M environments)?
3. How are the ground effect model parameters (e.g., µL, µD) calibrated for different aircraft types in FALCON-S? Is this process automated or manual?
4. Are there plans to include more complex flight dynamics (e.g., post-stall aerodynamics, vortex shedding) in future versions?
5. Has FALCON-S been tested in hardware-in-the-loop (HIL) setups or actual flight tests? If so, please provide details.
6. How does the framework handle non-ideal environmental conditions such as crosswinds, gusts, or sudden atmospheric pressure changes beyond the Dryden turbulence model?
7. Is the framework compatible with reinforcement learning baselines beyond SB3 and DreamerV3 (e.g., SAC, TD3, PPO with attention)?
What is the expected time and computational cost to train a single controller (e.g., DreamerV3) on a standard task using FALCON-S?
8. How does the framework ensure numerical stability and convergence when simulating high-dimensional, nonlinear dynamics over long horizons?
9. Are there plans to support real-time rendering and visualization (e.g., via Unity or Unreal) in addition to X-Plane?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a software package for GPU-accelerated realistic simulation of a fixed-wing aircraft.
The package targets applications in reinforcement learning and sampling-based model-predictive control. The package simulates basic rigid-body dynamics, fixed-wing aerodynamic forces, actuator delays, wind, pressure, ground effect, and sensor delays/noise.
Several reward functions for different tasks are provided.
The package exposes an OpenAI Gym interface and supports both GPU and CPU backends.
It has a bridge to X-Plane to compare the simulation against a realistic industry standard.

The experiments:
- Demonstrate that DreamerV3, a model-based RL algorithm, can learn an altitude-keeping policy.
- Evaluate a LQR controller on different tasks.
- Show that a LQR controller synthesized for one aircraft does not transfer to others.
- Show the impact of different physical and sensor disturbances on the LQR controller.
There is also a MPPI experiment in the appendix, but its results are not discussed in the main body.

### Strengths
- The proposed simulator appears to fill a gap in the landscape of fixed-wing flight simulators: the realistic ones are not amenable to parallel single-stepping for learning, and the other GPU-parallel one is not realistic (if we take the authors' word for it).
- The package seems well-designed, and the GPU acceleration is highly relevant for RL.
- The description of the simulator architectures and features has a good detail level - for a practitioner trying to decide which simulator to use, the paper should have all the information they need.
- Fixed-wing control, especially in scenarios far from straight and level flight, is relatively under-explored by the learning+control community compared to the (arguably less interesting) quadrotors.

### Weaknesses
- Numerous plots are illegible in the text labels and/or plots themselves: Figure 2, bottom row of Figure 4, second row of Figure 5.
- Given the emphasis on supporting a range of control architectures, the paper seems to be missing an obvious experiment: compare the performance of DreamerV3, MPPI, and LQR on the same set of tasks. Some of this information already exists in comparing Table 3 and Table 9, but Table 9 is relegated to the appendix, and it will be easy to make errors looking back and forth.
- Going further, other candidates like model-free RL and optimization-based (deterministic) MPC could make the comparison even more interesting.
- Ground effect modeling is stated as a key improvement relative to other simulators, but only one small experiment in the appendix includes it.
- There is a feature table comparing to other simulators, but there is no quantitative comparison e.g. on framerate, accuracy, etc.
- Lines 302-304 claim that the experiment will analyze ground effect and actuator delay, but the experiment does not actually include those phenomena.
- Table 7 in the appendix is never discussed in the appendix text.

Overall, the software presented in this paper is a solid contribution to the aerial robotics and learning+control research communities. The paper does a good job describing the features and architecture of the software. However, the experiments seem more like basic validation tests; they do not really provide any new information. For a datasets/benchmarks paper in ICLR, I would expect the experiments to do at least one of:

1) shed light on some important phenomenon in the application of learning+control algorithms to fixed-wing aircraft that was not well-known before, or
2) show very convincing evidence that the simulator is an improvement on previous available simulators.

I do not think this paper has met that standard. However, it seems within reach with a bit more effort, and I encourage the authors to keep refining this project.

### Questions
- In Table 1, the authors claim that their simulator is "more realistic" than NeuralPlane in an unspecified way. However, since it is the most closely related work, there should be more detail. In what way exactly is NeuralPlane less realistic?
- The authors claim the package includes "visualization tools", but these are never discussed, and I did not see any screenshots. What are they?
- Line 065: In the intro, it is not clear exactly what "validation capabilities" means.
- Line 090: What do you mean by "3DoF models"? It seems implausible that any flight simulator would not use a 6DoF SE(3) configuration as the foundation.
- Line 161-162: Is there any reference to justify that first- and second-order actuator response models are "realistic", e.g. from the aerospace controls literature?
- Line 203: What does it mean for the agent module to "support" a control model? Most robotics simulators can be used with any control architecture with no special effort.
- Line 272: Typo "luck-up table"

### Soundness
3

### Presentation
3

### Contribution
2
