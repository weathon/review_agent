# X2C: Enabling Realistic Human-to-Humanoid Facial Expression Imitation

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
The ability to imitate realistic facial expressions is essential for humanoid robots in affective human–robot communication. Achieving this requires modeling two correspondences: between human and humanoid expressions, and between humanoid expressions and robot control values. During training, we predict control values from humanoid expression images, while at execution time the control values drive the robot to reproduce expressions. Progress in this area has been limited by the lack of datasets containing diverse humanoid expressions with precise control annotations. We introduce **X2C** (Expression to Control), a large-scale dataset of 100,000 $\langle \text{image}, \text{control value} \rangle$ pairs, where each image depicts a humanoid robot expression annotated with 30 ground-truth control values. Building on this resource, we propose **X2CNet**, a framework that transfers expression dynamics from human to humanoid faces and learns the mapping between humanoid expressions and control values. X2CNet enables in-the-wild imitation across diverse human performers and establishes a strong baseline for this task.
Real-world robot experiments validate our framework and demonstrate the potential of X2C to advance realistic human-to-humanoid facial expression imitation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces X2C, a large-scale dataset for realistic humanoid facial expression imitation, consisting of 100,000 image–control value pairs, and X2CNet, a two-stage framework for human-to-humanoid facial expression imitation. In the first stage, LivePortrait is employed to transfer facial motion from human faces to humanoid images; in the second stage, the model predicts 30 continuous control parameters that drive the robot’s actuators. The dataset is synthetically generated through analytical interpolation, ensuring precise annotations, and the proposed method is validated through real-world experiments on the Ameca humanoid robot.

### Strengths
1. High-quality dataset: The paper presents a large-scale (100K) and high-quality humanoid facial expression dataset with precisely annotated actuator control values.

2. Rigorous methodology and real-world validation: The data generation pipeline incorporates asymmetric expressions and other design strategies to ensure annotation accuracy and diversity, and the method is validated on a physical humanoid robot, demonstrating effective sim-to-real transfer and practical applicability.

3. Ethical compliance: The work exhibits solid engineering execution and provides a transparent ethical statement regarding data collection and human participation.

4. Potential impact: X2C has the potential to become a benchmark resource for research on facially expressive robots and affective human–robot interaction (HRI).

### Weaknesses
1. Lack of user evaluation: The study does not include user experiments assessing the perceptual realism or emotional accuracy of the reproduced expressions.

2. Engineering-oriented approach: The proposed two-stage pipeline primarily integrates existing motion transfer and regression methods, with limited algorithmic innovation.

3. Limited baselines: The paper mainly compares against a few simple or outdated methods, lacking stronger or more recent baselines, especially in terms of visual performance.

4. Dataset generalization: The experiments are conducted only on the Ameca humanoid robot, leaving it unclear whether the approach generalizes to other humanoid platforms.

5. Fine-grained expression detail imitation: The fine-grained expression details, e.g. the wrinkles could not be effectively imitated by the proposed algorithm, e.g. in Fig. 6.

### Questions
1. Will the generated control sequences exhibit any abrupt transitions over time?

2. Could the dataset be extended to include audio-conditioned expressions or multimodal cues (e.g., speech, gaze interaction)?

3. Beyond the reported MAE, are there any metrics for perceptual similarity of expressions?

4. Could the proposed method deal with the imitation of fine-grained expression details ?

### Soundness
3

### Presentation
3

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
This paper introduces X2C, a large-scale dataset designed for realistic humanoid facial expression imitation, consisting of 100,000 paired samples of humanoid facial images and 30-dimensional control values. It also presents X2CNet, a two-stage framework for transferring human facial expressions to humanoid robots. The first stage transfers expression dynamics from human faces to humanoid images using motion transfer, while the second stage learns a mapping from these humanoid images to actuator control values. The dataset is collected using a simulated humanoid platform, ensuring precision and safety. The paper validates its contributions through quantitative experiments and real-world robot demonstrations, showing superior accuracy over baseline methods and demonstrating robust human-to-humanoid expression imitation.

### Strengths
**[Dataset Contribution]:** X2C fills an important gap in humanoid expression imitation research by providing the first large-scale, high-quality dataset of humanoid facial expressions annotated with precise control values. The dataset comprises asymmetric and fine-grained expressions, with a detailed analysis of 30 control dimensions and their distributions, as well as validation of the dataset's diversity.

**[Comprehensive experiments]:** The authors conduct both quantitative comparisons and ablation studies (Tables 4–5) showing that X2CNet significantly outperforms random and landmark-based baselines.

**[Real-world demonstrations]:** The paper provides convincing qualitative results (Figure 6, page 9) where a humanoid robot reproduces diverse and nuanced human expressions across performers, indicating successful transfer.

**[Clear presentation]:** The paper is well-written and easy to follow, with detailed methodology and good visual results.

### Weaknesses
**[Limited novelty in model design]:** While the dataset contribution is significant, the X2CNet architecture primarily adapts existing components (LivePortrait + ResNet18 regression), offering limited algorithmic innovation.

**[Insufficient cross-domain generalization analysis]:** The experiments mostly focus on within-dataset performance. It remains unclear how the model handles unseen human identities or lighting variations beyond those shown qualitatively. Is it basically determined by the adopted models?

### Questions
**[Q1]:** How well does X2CNet generalize to unseen human subjects, especially those with non-neutral identities, extreme expressions and inputs with different styles like cartoon?

**[Q2]:** Were any temporal consistency mechanisms considered to smooth control predictions over time?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces X2C, a large-scale dataset containing 100,000 image–control pairs for humanoid facial expression imitation, where each image depicts a humanoid robot face and is annotated with 30 continuous actuator control values. The authors additionally propose X2CNet, a two-stage imitation framework that first transfers human expressions to humanoid images using a motion-transfer model and then predicts control values with a ResNet-based regression network. The dataset is collected in simulation using key-framed animations and interpolation, ensuring precise alignment between visual appearance and control signals. Experiments compare the proposed mapping network against random baselines and a landmark-based model, showing lower mean absolute error in control prediction. Real-world robot demonstrations validate that the model-generated control values can drive a physical robot to imitate diverse human expressions.

### Strengths
Strengths:
- X2C provides 100k image–control pairs with precise continuous annotations. This dataset can be helpful for the broad community.
- The dataset preparation uses key-framed animation, interpolation, SSIM-based frame filtering, and control alignment with careful processing and considerations.
- The authors demonstrate the effectiveness of the method on a physical robot platform.

### Weaknesses
Weakness:
- The model architectures and algorithms do not have enough novel designs. I am not raising this point to question the novelty of the work itself, since I do appreciate the value of the dataset and the real-world demos, but simply not sure if the proposed method will be a good fit for ICLR. This work seems to be much more well-suited for a robotics venue.
- The primary metric is MAE over control values. There is no perceptual, behavioral, or user-study evaluation of expression quality. Including human evaluations will further strengthen the claims.
- Another suggestion is to consider adding more advanced models to explore algorithm improvements.
- Variations of the data, such as lighting changes or out-of-distribution identities, could be interesting to consider in the dataset and evaluations.
- No explicit characterization of when the model fails or limitations of the two-stage approach. Discussing a bit more on those aspects can be very helpful.
- There is a claim on "no sim-to-real gap" in the paper, but this is not carefully validated or quantified. At least some discussions on how the sim2real gap was resolved can be helpful.

### Questions
Please see weakness points.

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
5

### Summary
This paper introduces a dataset and framework to solve the problem of humanoid robots imitating human facial expressions. The authors identify that a key challenge is the lack of data mapping a robot's appearance to its specific motor control commands.

Their solution has two parts:
- X2C Dataset: A new, large-scale dataset of 100,000 (image, control value) pairs. These are not human images; they are images of a virtual humanoid (Ameca) paired with the 30 ground-truth motor control values needed to create that exact expression.
- X2CNet Framework: A two-stage model for imitation. Stage 1 (Motion Transfer) uses a pre-trained model to "retarget" a live human's facial motion onto a static image of the humanoid. Stage 2 (Mapping Network) takes this newly generated humanoid image and, using the X2C dataset, predicts the 30 control values to send to the physical robot.

The method is validated on a physical Ameca robot, successfully imitating diverse human expressions in real-world conditions.

### Strengths
- The paper correctly identifies the critical bottleneck in this field: the lack of a large, high-fidelity dataset that directly links visual expressions to low-level motor commands.
- The X2C dataset is curated using a virtual robot to generate the training data, which is clever, as it bypasses the risk of physical damage and provides perfectly ground-truth control value annotations.

### Weaknesses
- The framework is heavily relying on the pre-trained motion transfer model (LivePortrait). Any artifact from this first stage—such as misinterpreting a subtle expression will be passed directly to the second stage. The mapping network will then correctly predict the control values for the wrong expression.
- The paper presents this as a system for affective human-robot communication, but provides no information on its computational performance (e.g., FPS or latency). It is unclear if this two-stage model can run fast enough for a truly live and natural interaction.
- There are no user study results to quantify the naturalness and imitation capability. While the authors provide strong quantitative metrics (like Mean Absolute Error) to show that their X2CNet framework can accurately predict the 30 ground-truth control values, this only measures engineering correctness. It does not measure the subjective, perceptual quality of the imitation. The paper relies entirely on qualitative evidence for this. Section 4, "REAL-WORLD DEMONSTRATIONS", and its corresponding Figure 6 are presented as the primary validation.

### Questions
- What is the end-to-end latency of the full X2CNet pipeline, from the camera capturing the human face to the robot's motors moving? Can the framework run at a speed (e.g., frames-per-second) that is high enough for a natural, in-the-wild social interaction?
- How do you ensure that the mapping network generalizes well from its clean training data to the (potentially noisy or artifact-filled) images produced by the LivePortrait model?

### Soundness
3

### Presentation
3

### Contribution
3
