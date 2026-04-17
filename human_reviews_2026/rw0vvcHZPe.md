# Geometry of Uncertainty: Learning Metric Spaces for Multimodal State Estimation in RL

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Estimating the state of an environment from high-dimensional, multimodal, and noisy observations is a fundamental challenge in reinforcement learning (RL). Traditional approaches rely on probabilistic models to account for the uncertainty, but often require explicit noise assumptions, in turn limiting generalization. In this work, we contribute a novel method to learn a structured latent representation, in which distances between states directly correlate with the minimum number of actions required to transition between them. The proposed metric space formulation provides a geometric interpretation of uncertainty without the need for explicit probabilistic modeling. To achieve this, we introduce a multimodal latent transition model and a sensor fusion mechanism based on inverse distance weighting, allowing for the adaptive integration of multiple sensor modalities without prior knowledge of noise distributions. We empirically validate the approach on a range of multimodal RL tasks, demonstrating improved robustness to sensor noise and superior state estimation compared to baseline methods. Our experiments show enhanced performance of an RL agent via the learned representation, eliminating the need of explicit noise augmentation. The presented results suggest that leveraging transition-aware metric spaces provides a principled and scalable solution for robust state estimation in sequential decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Authors propose a new latent metric (MetricMM) for sensor fusion and state estimation. The key idea is that uncertainty can be inferred from distance thereby ridding the need for explicit uncertainty. Empirically this idea proved robust to different classes of noise in a dozen of RL control domains that other methods are brittle to.

### Strengths
Originality and significance: The idea of utilizing the distance in latent metric space as a measure of sensor consistency thus reliability is novel as far as I am aware of. I like the motivation: it is new and simple (in a positive sense); it bypasses the noise modeling in sensor fusion which is indeed a pain unless one assumes Gaussian.

Quality and clarity: Experimental setup are thorough and results are well organized and presented.

### Weaknesses
I lean towards accept because no detrimental weaknesses stand out to me, though I have a few questions below that will help me become more confident.

### Questions
1. If I understand correctly, the noise is applied per frame by chance. What if the noise is temporally dependent, e.g., in the case of the "failure" noise, it is likely that a camera artifact persist for a few consecutive frames, "sticky artifacts" per se. How does that impact your methods and baselines?
2. How are the seven families of perturbation designed? Puzzle and texture seems too artificial to be application-relevant.
3. Each loss in eq (9) makes sense intuitively, though it would be more informative to show their role through a small ablation study even just in one domain and a few noise levels.

Decision irrelevant suggestions:
1. Table titles are cryptic. I understand you have lots of results to show (which is good signs), but it would be really helpful to expand in the title on the takeaway in natural language, at least those tables in the main body.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a novel method for learning distance metrics between states in reinforcement learning. The approach is to learn mappings from observation modalities to latent states, and measure distance using a well-define metric in the latent state space. The learned state representation can then be used as the basis for reinforcement learning algorithms. The method is tested in a series of benchmarks.

### Strengths
State representation learning is an important problem in reinforcement learning, and in this sense the paper makes a timely contribution. It also looks as if the proposed approach performs well in practice compared to other algorithms.

### Weaknesses
I believe that several concepts are not clearly explained, which makes it difficult to accurately evaluate the contribution. Mainly for this reason my opinion regarding acceptance at ICLR is on the negative side.

Apart from the cited paper by Wang et al., there are other works that explicitly learn a distance estimate between pairs of states. Concretely, the first work also measures distance as the minimum number of actions required to transition from one state to another.

State Representation Learning for Goal-Conditioned Reinforcement Learning
Steccanella & Jonsson, ECML 2022

Park, Kreiman & Levine, ICML 2024
Foundation Policies with Hilbert Representations

### Questions
In the definition of POMDPs, do you assume that the underlying state space S is known to the learner? (even if it is not observable)

The proposed approach is to map *each* observation modality to the same latent space, in the hope that all observation modalities derived from the same state will map to approximately the same latent state. However, the mapping from states to observation modalities is stochastic, so a deterministic map is not likely to map all modalities to the same latent state. Why do you map all observation modalities to the same latent space? An alternative would be to have a different latent space for each observation modality and aggregate the distance measures in another way.

From Equation (3) it appears as if you assume that the function \varphi_T is known, is this indeed the case? I would have expected you to *learn* an approximate transition function in the latent space. If you do not assume that \varphi_T is known, then I do not see how you can make an assumption regarding the transition error of \varphi_T, since this depends on the quality of learning.

What is the motivation for the exact form of Equation (3)?

The description of the experimental setup leaves a lot to be desired. What is the training pipeline? Do you train a distance estimate first, and then a deep RL policy, or is learning simultaneous? Is distance estimation done online or from offline data? What is the input to the deep RL algorithm (SAC), i.e. exactly how are states and state features defined? 

From where did you take the seven corruption families? Can you provide a reference?

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
3

### Summary
The paper addresses the important problem of robust state estimation from multimodal, noisy observations in RL. It proposes METRICMM, a method that learns a shared latent space and uses a simple geometric fusion rule, Inverse Distance Weighting (IDW), to combine sensor estimates. The core contribution is to reframe the problem of sensor uncertainty from a probabilistic one to a geometric one. The experimental results demonstrate a clear improvement in robustness over a wide range of baselines.

### Strengths
The primary strength of this work is its core conceptual shift. Instead of relying on Bayesian filtering, which requires restrictive priors or generative models , METRICMM recasts uncertainty in geometric terms.

The inverse distance weighting (IDW) fusion mechanism is a direct and elegant consequence of the geometric formulation. The dynamics prediction z^t​ acts as a reliable "anchor." Any sensor encoding that is geometrically distant from this anchor is naturally identified as noise and its contribution is automatically suppressed. This is far simpler and, as the evidence suggests, more robust than learned fusion mechanisms like attention ( \alpha-MDF baseline ) or simple concatenation/linear combinations.

METRICMM consistently shows a much "flatter" degradation slope, maintaining high performance even as corruption frequency increases, while all baselines collapse quickly. Empirical results, albeit of simple benchmarks, are quite strong.

### Weaknesses
1) Naive Temporal Distance Loss: The paper's stated goal is to learn a space where distances correlate with the minimum number of actions required to transition between them. However, the actual loss function used is a massive oversimplification. This loss function does not model the minimum number of actions it models all single-step transitions are equidistant. It forces the distance between any two consecutive states to be 1, regardless of the optimality action taken. This is a naive implementation that contradicts the paper's core motivation.

2) I think the above formulation will be optimistic when transitions are stochastic.

3) The paper's core claim is that its learned metric representation is superior. However, the experiments confound this with a hard-coded, non-learned fusion rule (IDW). The high performance could also be due to the simple outlier-rejection property of IDW and not the learned representation. What will happen if the authors take the learned encoders from the strongest baselines (\alpha-MDF or CORAL) and, at test time, replace their learned fusion module with the paper's exact IDW fusion rule?

### Questions
See weaknesses

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
4

### Summary
This paper proposes a novel method for robust multimodal state estimation in POMDPs by learning a structured latent representation. The core idea is to create a metric space where the Euclidean distance between latent states directly correlates with the minimum number of actions required to transition between them.

All sensor modalities are encoded into this shared space, providing a geometric interpretation of uncertainty that avoids the need for explicit probabilistic noise models. The system fuses information by weighting each modality's contribution based on its inverse distance to the predicted state, thereby down-weighting corrupted or unreliable sensor data. Empirical results demonstrate that this approach significantly enhances an RL agent's performance and robustness against unseen sensor corruptions.

### Strengths
- The paper introduces a novel geometrical view for handling state representation and uncertainty in POMDPs. By refraining uncertainty in geometric terms it bypasses the complexities and assumptions of traditional probabilistic models. This idea could inspire a new direction of research for handling partial observability and uncertainty in RL.

- The paper is well written and the methodology is sound. Including implementation details that could make future reproduction of the algorithm and empirical results easy. 

- The experimental results are extensive and provide convincing evidence of the method's effectiveness.

### Weaknesses
- The paper fail in citing previous work from Steccanella et al. (2022), "State Representation Learning for Goal-Conditioned Reinforcement Learning". That work appears to be the first to propose the idea of minimum number of actions distance and motivates very similar objectives for learning an embedding space where distance between states in this embedding space approximates the minimum action distance, by means of leveraging local constraints and the useful upper-bound of the trajectory distance instead of simply trying to maximize the log distance. Please discuss and include this citation in your work. 

- The paper should make more clear to the reader that this approach assume deterministic dynamics. Is not clear to me how this approach will behave in the case of stochastic dynamics where the distance between states becomes an expectation and the minimum action distance will provide just a lower bound on that. And a broader discussion of this limitation is needed. The authors should explicitly state this assumption and dedicate a discussion to this limitation, outlining how the framework will behave in this scenario.

- The paper uses a symmetric metric (Euclidean distance) to approximate the Minimum Action Distance, which is inherently asymmetric in most realistic environments. For instance, in environments with irreversible or asymmetric dynamics. By enforcing a symmetric metric, the model is forced to learn an inaccurate representation that cannot capture such crucial, directional aspects of the environment's dynamics. As noted in prior work (Steccanella et al., 2022), this forces only to learn a symmetric approximation of the true Minimum Action Distance. A more thorough discussion is needed to acknowledge this trade-off between simplicity and representational fidelity.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
