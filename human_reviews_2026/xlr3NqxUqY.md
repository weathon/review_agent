# Contact-guided Real2Sim from Monocular Video with Planar Scene Primitives

- Decision: Accept (Poster)
- Scores: 2, 6, 8, 4

## Abstract
We introduce CRISP, a method that recovers simulatable human motion and scene geometry from monocular video. Prior work on joint human--scene reconstruction relies on data-driven priors and joint optimization with no physics in the loop, or recovers noisy geometry with artifacts that cause motion-tracking policies with scene interactions to fail. In contrast, our key insight is to fit simulation-ready convex planar primitives to a depth-based point cloud reconstruction of the scene via a simple clustering pipeline over depth, normals, and flow. To reconstruct scene geometry that might be occluded during interactions, we use human--scene contact modeling (e.g., using human posture to reconstruct the occluded seat of a chair). Finally, we ensure that human and scene reconstructions are physically plausible by using them to drive a humanoid controller via reinforcement learning. Our approach reduces motion-tracking failure rates from 55.2\% to 6.9\% on human-centric video benchmarks (EMDB, PROX), while delivering 43\% faster RL simulation throughput. This demonstrates CRISP's ability to generate physically valid human motion and interaction environments at scale, advancing real-to-sim applications for robotics. Code and interactive demos are available at our project website: https://crisp-real2sim.github.io/CRISP-Real2Sim

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces CRISP, a method for recovering simulatable 3D human motion and scene geometry from monocular video. Its key point is the use of compact planar primitives for scene representation, which are more efficient and robust for physics simulation than traditional dense meshes. CRISP also employs contact-guided modeling to reconstruct occluded interaction surfaces. By validating the reconstruction through Reinforcement Learning (RL), the method achieves a high real-to-sim success rate and reduces motion tracking failure rates, while speeding up RL simulation throughput. This makes the "vid2sim" pipeline practical for embodied AI and robotics.

### Strengths
The paper's strengths lie in its high empirical quality and practical value for the Real-to-Sim pipeline. It replacing noisy dense meshes with convex planar primitives for simulation efficiency. This engineering choice, combined with clever contact-guided scene completion, results in a highly robust system that achieves a  good reduction in motion tracking failures and an acceleration of RL training throughput.

### Weaknesses
The primary weakness of CRISP is its limited conceptual novelty, functioning largely as a highly effective "set of practical engineering approaches" that integrates existing components (HMR, 4D reconstruction, contact prediction) rather than introducing a new framework. Specifically, the use of planar primitives, while practical, relies on a simple, non-learned clustering pipeline that may struggle with complex, non-planar geometries (e.g., curved furniture, organic shapes), limiting its generalizability beyond the "planar-world assumption." The Contact-Guided Scene Completion is a heuristic that depends on pre-trained vision-language models for interaction detection, which introduces a dependency on external, potentially brittle, classifiers. The two core designs—reliance on planar primitives and contact-guided scene completion—may boost computational efficiency but fail to address the key challenges in scene reconstruction, leaving the work unable to grasp the true hurdles of such tasks.
In addition, the experimental comparison in Figure 2 is not fair to the baseline method (VideoMimic). The authors represent the scene using compact planar primitives, which, while simplifying the optimization for motion tracking, lead to significant scene distortion. This representation creates a larger sim-to-real gap for the subsequent humanoid tracking task. For a more equitable comparison and to better prepare the method for real-world deployment, the authors should consider investigating learning a motion prior from these compact scene interactions.

### Questions
no.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a framework for simulation-ready human scene reconstruction from a single video. The authors introduce a planar primitive representation to better model planar regions such as the ground and propose a contact term to improve physical plausibility. The paper is clearly written and provides several qualitative results demonstrating the effectiveness of the approach.

### Strengths
1. The paper is clearly written and easy to follow.

2. The idea of introducing planar primitives to constrain plane reconstruction is intuitive and practical.

3. The proposed contact term helps ensure more physically consistent interactions between humans and the environment.

4. The experiments validate the overall framework, and the inclusion of numerous demos and visualization results makes the paper more understandable and accessible.

### Weaknesses
1. Although I appreciate the effort in building the proposed framework, it somewhat gives the impression of a technical pack, combining many existing components rather than introducing fundamentally new concepts.

2. Table 2: The reported decimal precision is inconsistent across metrics and should be standardized for clarity.

3. Experimental results are not sufficiently comprehensive:

    - The ablation study is limited; for instance, it is unclear how the number of primitives affects reconstruction quality.

    - There is no report of the model’s runtime, computational cost, or efficiency.

    - No failure case analysis is provided.

    - Some of the demos provided by UPP do not seem to align properly.  In several examples, the input video and the output result do not match in duration or content.

4. The paper lacks a discussion of limitations, such as handling complex or irregular objects in the scene (e.g., interactions with yoga balls or sandbags).

5. There are several typos and minor grammatical issues throughout the paper, and need careful proofreading.

### Questions
What will be the result if the video is a human lying on a yoga ball?

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
4

### Summary
The authors introduce CRISP to recover human motion and scene geometry simultaneously from monocular videos. CRISP first fits convex planar primitives to depth-based point clouds, then addresses the occlusions with human-scene contact modeling. Finally, an RL policy is trained to simulate the reconstructed human-scene interaction.

### Strengths
- The overall pipeline is reasonable and practical. The idea of exploiting planar primitives achieves a balance between representativity and efficiency.

- The performance is impressive and superior to its counterparts. 

- Comprehensive analyses are provided, revealing meaningful findings for the real2sim of HSI.

### Weaknesses
- The symbol system is a little complicated. Also, there are duplicated symbols like K in L232 and L243. Annotating them in pipeline figures, such as Figure 2, might help.

- In Table 2, the ground truth jitter should be provided for reference.

- It might be better if some failure cases could be shown.

### Questions
- How was K in L232 decided?

- How long does it take to recover one sample?

### Soundness
3

### Presentation
2

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
In this paper, the authors proposed to use planner primitive to approximate the real world scene "geometry".
The author uses MegaSAM to recover the global scene point cloud and uses planar primitive fitting to approximate the scene geometry.
Combined with RL training, this allows for recovering the human motion and the scene geometry from a monocular video.

### Strengths
1. It provides a nice and simple pipeline for recovering the human motion and the scene geometry from a monocular video.
Generating motions which handles human-scene interactions is a very challenging task
and crucial for downstream applicaitons in animation and robot loco-manipulation.

2. The paper is well-written and the results are convincing.
As can be seen in the paper,
a rich set of experiments and ablation studies are conducted to evaluate the performance of the proposed method.

3. The method appears robust across a diverse set of static scenes where the agent is vaulting, parkouring, and climbing.

4. This paper provides more robust scene reconstruction in sim compared to baselines such as VideoMimic.

### Weaknesses
1. The paper does not present sim2real results as in VideoMimic.
Considering that the two work are very similar in terms of the pipeline,
it would make presenting only real2sim results rather incremental and less convincing.

I would argue the fps in training is not that important as people have been exploring universal sim RL policy,
making retraining less needed.
But I think the proposed method will actually make more sense in real time sim2real setting,
where scene needs to be reconstructed on the fly and perhaps that's where the planer primitive fitting will come in handy.

2. It's hard to tell how scalable and robust the algorithm is given that only a collection of scenes from PROX and EMDB are used.
It will be very helpful if the authors could provide more experiments on more diverse scenes,
and maybe even extend to the in-the-wild videos from youtube for example.

### Questions
1. How fast can the scene reconstruction run and can it be done in real time?

2. Is it possible to do runtime scene reconstruction which only considers a subset of the scene geometry?

3. Is it possible to input the scene information into the policy observation and apply it in a runtime fashion?

### Soundness
2

### Presentation
3

### Contribution
2
