# Learning the energy relaxation manifold from unrelaxed structures with RelaxNet

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
In an effort to bypass computationally expensive density functional theory (DFT) calculations for energy minimization and structure relaxation, rapid progress in the development of machine learning force fields/interatomic potentials (MLFF/MLIPs) and more robust models that adhere to quantum chemistry/physical paradigms and constraints have been realized. However, most research to date involves static-frame energy predictions only (i.e., given a specific atomic configuration, predict the energy of the current or final instance), neglecting intermediary physical insight-providing contexts. In this work, we developed RelaxNet, the first end-to-end, dynamics-aware, equivariant model that leverages neural ordinary differential equations (ODEs) and message passing neural networks (MPNNs) for predicting the energy relaxation landscape between the initial unrelaxed structure and final relaxed structure. From just the initial structure, which is often the configuration that is fed into DFT simulations, we can accurately recover the energy, forces, and geometric pathways for the trajectory at a competitive prediction accuracy, as evidenced by comprehensive benchmarking with state-of-the-art static models and MLIP-based relaxation methods. Additionally, we provide extensive insights on the use of implicit vs. explicit latent embedding evolution to offer perspectives on optimal strategies for future works that seek to integrate expensive graph-based neural networks and neural ODEs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors tackle the task of predicting the per-structure energy and per-atom forces for a relaxation trajectory, given only the initial positions.

They propose to train a NeuralODE wrapped around a MPNN.
The ODE state consists of positions x(0) and learned “velocity” that is updated via the predicted force (obtained as gradient of the predicted energy).
The authors ablate two design choices to update the latent embeddings h, that are produced by the encoder block.
Implicit latent: h is not an ODE state. It’s recomputed from the current positions x(t) via the encoder at every pseudo-time step and then used to predict energy. 

Explicit latent: h(0) is initialized and included as an ODE state variable, then updated each step by an MLP (along with the position and velocity)
They test their approach on crystal structure trajectories from the JARVIS database.

### Strengths
- Moves beyond single-frame energy prediction to trajectory learning between initial and relaxed states
- On JARVIS, authors report 17.06 meV/atom (final frame) vs. ~27–33 meV/atom for baselines, albeit with caveats

### Weaknesses
- It is unclear to me when one would be interested in predicting a relaxation trajectory compared to a MLIP that can do relaxation and any other dynamics (like MD)
- The approach limits data to sequential frames of fixed length
- The experiments only cover one not-so-common dataset with only 30k datapoints

### Questions
- I think the real benchmark is running relaxations with MLIPs. Can you compare the trajectory and final geometry error between relaxations with MLIPs, RelaxNet, and ground truth trajectories?
- I don’t quite understand how table 3 is computed. As I understand, RelaxNet will arrive at a different final structure than the data. Are the reported MAE_E for the baseline at the final RelaxNet-generated-structure or at the dataset-structure? For the ground truth, do you run DFT on the RelaxNet-generated-structure?
- You report only the MAE of the Energies in table 3, can you also include the Force MAE?
- It would be helpful to highlight (bold) the best numbers in the tables
- Can you measure the difference in inference time cost between running relaxations with an MLIP to using RelaxNet?
- What does this sentence in line 107 mean? "these equivariant models can be extended to periodicity-dependent applications, like molecular modeling and DFT”
- Does the use of a Neural ODE increase the cost of training compared to an MLIP by a lot?

### Soundness
1

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
The paper introduces RelaxNet, a neural ODE-based model that predicts the entire DFT energy relaxation trajectory from just the initial unrelaxed structure.

### Strengths
This is a novel problem with a well-proposed method. Predicting the entire relaxation trajectory is both novel and of high practical value. 

- provides good starting points for DFT simulations, reducing convergence time

- offers a clear comparison of implicit vs. explicit latent evolution across different trajectory lengths

- physics-based model

### Weaknesses
- Expensive training: 133-311 min/epoch for implicit method is prohibitive; even explicit (17-137 min) is slow. How scalable is this model
- No analysis of trajectory smoothness, physical plausibility, or whether intermediate states are actually useful
- limited dataset coverage

### Questions
- Beyond energy and force MAE, how well does RelaxNet reproduce the actual geometric pathway of the relaxation? For example, what is the average RMSD between the predicted and true final structures?

- The model is trained on DFT relaxation trajectories. If you initialized it with a perturbed structure that lies off the training trajectories (but within the distribution of atomic configurations), would it reliably find a path to a reasonable local minimum, or is it primarily learning to interpolate between seen initial and final states?

- How sensitive are the results to the choice of the ODE solver (RK4) and its hyperparameters (tolerances, step size)? Was any exploration done with adaptive solvers?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces RelaxNet, a dynamics-aware and equivariant deep learning model for structure relaxation. To address the limitation of prior works that primarily predict energy for static structures while overlooking intermediate physical insights, RelaxNet leverages neural ordinary differential equations (neural ODEs) combined with message-passing neural networks (MPNNs) to model the energy relaxation landscape between initial and final structural states. Empirical results and extensive ablations are provided to support the rationale of the proposed approach.

### Strengths
* The idea of using neural ODEs to model relaxation trajectories is novel and thoughtfully implemented, offering a promising alternative formulation for this task.
* RelaxNet achieves competitive accuracy compared to state-of-the-art methods on standard benchmarks.

### Weaknesses
* While modeling intermediate states is intuitively appealing, it remains unclear why this should necessarily improve the prediction accuracy of the final relaxed state. As shown in Table 3, the performance gain over strong baselines is marginal. The authors should provide stronger intuition—or theoretical justification—for why this intermediate modeling leads to better final-state predictions.
* Insufficient experimental analysis:
  * There is no clearly defined, chemically meaningful accuracy threshold for evaluating relaxation performance, making it difficult to assess the practical utility of the reported results.
  * A comparison of training cost (e.g., wall-clock time, memory usage) between RelaxNet and baselines is missing. Given the modest performance improvement, such analysis is essential to evaluate the method's efficiency.
  * The requirement that training systems must contain more than $n$ frames (due to the neural ODE formulation) raises concerns about generalizability, since this limitation may restrict the applicability of RelaxNet to systems where long relaxation trajectories are unavailable.

### Questions
* In Equation (2b), the notation $G$ is used without a clear definition. Could the authors clarify its meaning?
* Could the authors provide more detailed explanations of the data processing steps mentioned in line 181?
* The second term in Equation (7), $\hat{E}_{t, s+1} - \hat{E}_{t,s}$, is difficult to interpret. Could the authors explain it more clearly?
* How exactly is the energy MAE computed for RelaxNet for evaluation? 
* For systems with varying numbers of frames, the authors sample $n$ equidistant points. Does this result in different effective time intervals between samples? If so, could this introduce ambiguity in the learned dynamics or physical consistency of the gradient field?

### Soundness
2

### Presentation
1

### Contribution
2
