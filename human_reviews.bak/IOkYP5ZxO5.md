# ViTaS: Visual Tactile Soft Fusion Contrastive Learning for Reinforcement Learning

- Decision: Reject
- Scores: 5, 6, 3, 8

## Abstract
Tactile information plays a crucial role in human manipulation tasks and has recently garnered increasing attention in robotic manipulation. However, existing approaches struggle to effectively integrate visual and tactile information, resulting in suboptimal performance. In this paper, we present **ViTaS**, a simple yet effective framework that incorporates both visual and tactile information to guide an agent's behavior. We introduce _Soft Fusion Contrastive Learning_, an advanced version of conventional contrastive learning method, to enhance the fusion of these two modalities,  and adopt a CVAE module to utilize complementary information within visuo-tactile representation. We conduct comprehensive experiments, including $\mathbf{9}$ tasks in simulation environment, across $\mathbf{5}$ different benchmarks, to compare ViTaS with existing baselines. The results demonstrate that ViTaS achieves state-of-the-art performance, with an average improvement of $\mathbf{51}$%. Furthermore, our method significantly enhances sample efficiency while maintaining minimal parameters, underscoring the effectiveness of our approach. The code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposed a framework ViTaS, a contrastive learning based framework to enhance reinforcement learning for vision-tactile manipulation tasks in simulation. By incorporating a soft vision-tactile contrastive learning term with CVAE supervision, ViTaS helps various PPO-based manipulation tasks achieve better performance.

### Strengths
1. Applied cross-modal contrastive learning method to an interesting use case
2. Conducted various experiments with many baselines, showing a significant boost in curves

### Weaknesses
1. The manuscript is not well written. There are dozens of cases that
- Lacks coherence. In Sec 2, 3, and 4 it confused me a lot to see methods or results without knowing your motivation for doing so, or how to interpret the results. The entire paragraph from Line 418 doesn't belong here. 
- Figures and Tables
  + Many referred in context 2 pages before/after itself while you don't have to.
  + Details in Fig 1 are not explained, making this teaser confusing
  + Fig 2 is not clear enough, especially the soft fusion contrastive learning part. You showed lots of details in different colors, while it looks just like time contrastive to me. Also, what is the graph in the CVAE module visualization?
  + Fig 5, where is the drop you mentioned at Line 366? 
  + Fig 6 It's unclear how to evaluate the two settings. i.e. what patterns prove ViTaS is superior? 
  + Why do you use the same abbreviation rule for w/ and w/o in Table 2? Also, you mentioned K=1 and K=50, but the K value for other settings is not mentioned until the ablation for K itself.
- Regarding your contributions, from my PoV (1) and (2) are the same thing. Also, the average improvement of 51% is not shown in your experiment section.
- Missing method details. This is probably due to you mimicking Sec 3.2 of "Self-supervised Co-training for Video Representation Learning" since your Sec 3.1 is mainly adopting this method in your scenario, but I strongly suggest you revise your method sections. Your RL setting has more details to tell than copying theirs, as you showed in your pseudocode in the Appendix. For example, 
  + Your top-K is found in the buffer, which is never mentioned in context as an important implementation detail. 
  + Also, the metric for similarity is non-trivial. You should explain your usage of the encoder in formula (2). 
- Typos
  + Line 151, takes information from two modalities, or simply two modalities'.
  + Line 154, cross-modal
  + Line 215, full stop
  + Line 261, involving
  + Line 286, zero-padded
  + Line 431, results are
  + Line 437, is not explained "below"

2. Your entire framework is based on the observation "We note that when two tactile maps are resembling, the key features derived from the corresponding visual data should likewise bear a strong resemblance and vice versa". This is a strong and dangerous claim without clarifying the scope. For more broad vision-tactile manipulation tasks, I can easily construct counter-examples by replacing every part of the grasped object except the contact surfaces, not to mention the backgrounds. In this case, (1) does this claim work only in your task setting, with restricted object types that satisfy this rule? (2) will applying ViTaS in turn harm the generalizability of the RL agent? Please add more discussion on this part, elaborating either on the nature of your dataset or the limitation of your method. Please correct me if I misinterpreted your claim.

### Questions
1. In your CVAE ablation you showed only the pen rotation test. From Fig 4 it seems the pen is large enough in the visual image. What about other tasks? For example, can your CVAE work in other experiments where the object of interest is small, or its visual states are subtle? Please show some ablations on this. For the same reason, the explanation in Line 381 "This token-based feature extraction is prone to overlooking critical information compared to pixel-level extraction, thus resulting in the capture of fewer essential features" is not convincing. Can you show some qualitative results?
2. The last two questions in the Weaknesses section

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
5

### Summary
This paper presents ViTaS, a framework that combines visual and tactile data to improve robotic manipulation. It employs Soft Fusion Contrastive Learning and a CVAE module to effectively fuse and reconstruct these modalities. Experiments are done on a diverse set of none tasks in five different benchmarks.

### Strengths
- The paper is well-written, organized, and straightforward.
- The experiments are conducted across a diverse set of simulated environments, providing robust validation.

### Weaknesses
- **Lack of Motivation:** The authors incorporate a CVAE as a key component in their method, yet the rationale for this choice remains unclear. Section 3.2 includes a brief explanation, but it lacks depth and fails to adequately justify the use of CVAE. The authors should provide a more explicit motivation, detailing why CVAE combined with Contrastive Learning is preferable to the MAE objective used in [1].

- **Contrastive Objective:**  The theoretical differences between the contrastive loss used in this work and that in [2] requires further clarification. Providing a mathematical comparison between the two approaches could offer valuable insight into the unique contributions of this method, particularly regarding distinctions in formulation and the expected effect on learning outcomes.

- **Reconstruction Objective and Noise:** It is surprising that the authors opted for a pixel-based reconstruction objective. Prior research [2,3,4] has consistently shown that pixel-level reconstruction objectives often underperform in noisy conditions, as they may capture irrelevant elements from the observation. How would this approach fare in real-world scenarios with noise-prone tactile sensory data? I recommend that the authors demonstrate the model's robustness by presenting results on noisy or irrelevant observation components, like Gaussian noise, salt-and-pepper noise (similar to [1]), or missing pixels.

- **Limitations:** The limitations specific to this method are insufficiently addressed. The authors should explicitly discuss these limitations in the paper. For instance, addressing scalability to complex manipulation tasks, representation robustness under noise, and real-world applicability could provide readers with a clearer understanding of the practical constraints and areas for future development.

- **Performance on Rotate Tasks:** ViTaS significantly outperforms all other methods on the rotate tasks. The only provided explanation is that these tasks are “contact-rich and require methods to incorporate visual and tactile information jointly.” However, this characteristic seems relevant to all tasks in general. The authors should offer more insights into why ViTaS performs exceptionally well in the rotate tasks specifically.

- **Clarification Needed:** In Section 4.3, the authors claim, “This token-based feature extraction is prone to overlooking critical information compared to pixel-level extraction, thus resulting in the capture of fewer essential features.” This assertion seems counterintuitive, given that ViTs employ an attention mechanism that typically compensates by aggregating information across tokens, effectively capturing both local and global features. The authors should provide references to support this statement.

- **Experiments:** Although the method has been validated across a range of simulated environments, it has not been empirically tested or combined with tactile sensors in real-world experiments. Real-world tactile data is inherently different and would serve as a valuable addition to the validation. However, this is not a major concern, considering the wide range of simulated experiments.

**References**

[1] Sferrazza, Carmelo, et al. "The power of the senses: Generalizable manipulation from vision and touch through masked multimodal learning." (2023).

[2] Zhang, Amy, et al. "Learning Invariant Representations for Reinforcement Learning without Reconstruction."International Conference on Learning Representations.

[3] Bai, Chenjia, et al. "Dynamic bottleneck for robust self-supervised exploration."Advances in Neural Information Processing Systems 34 (2021): 17007-17020.

[4] JDeng, Fei, Ingook Jang, and Sungjin Ahn. "Dreamerpro: Reconstruction-free model-based reinforcement learning with prototypical representations."International conference on machine learning. PMLR, 2022.

### Questions
Why is only the visual observation reconstructed in the CVAE, and not the tactile data?

The other Questions are in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors introduce ViTaS, a framework for integrating visual and tactile information to improve robotic manipulation. ViTaS employs Soft Fusion Contrastive Learning to better fuse these modalities and a CVAE module to leverage complementary visuo-tactile information. The authors conduct some experiments with the proposed method.

### Strengths
1. The paper is well-structured and easy to follow.
2. The authors conduct extensive experiments in simulated environments.

### Weaknesses
1. Key visual RL baselines, such as DrQ-v2, are missing from the comparisons. The authors should compare their method with DrQ-v2 using both image-only and combined image-tactile observations.
2. More analysis is needed to explain how contrastive learning and the CVAE module contribute to better feature representations.
3. In Figure 5, only the proposed method performs well in the egg, block, and pen rotation environments, while it performs similarly to some baselines in other environments. Are these environments not optimized adequately for the baselines?
4. In Figure 7, the performance variance without CVAE is notably high. Does this indicate that one/multiple of the trials converges to a local optimum? If so, the authors are encouraged to run over more seeds to verify if ViTaS exhibits similar behavior.

### Questions
The authors need to address my concerns in the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The author introduces a framework to enhance robot manipulation by fusing visual and tactile data. They use a soft fusion contrastive learning approach and want to align these modalities in the latent space to improve RL policy.

### Strengths
1. The visualization of t-SNE is good and the overall presentation is easy to read.
2. Extensive experiments and analysis of ablation study, the overall results are convincing.
3. Diverse simulation and environment.

Overall its a good paper to accept.

### Weaknesses
1. No real-world experiments were included to validate the effectiveness of the method.

2. Some minor presentation issue should be fixed.

### Questions
1. [line281-288]. Please specify what is 32x32x3 tactile sensor and what is 3x3x3. Please specify each channel of them. 

2. It would be better for the author to mention what kind of tactile sensors they are using and what kind of sensor signals can be received (eg. normal force, shear force)

3. [line 241,242], there are a lot of papers already showing the dexterous hand's abilities to do in-hand rotation using both vision and tactile or using solely tactile. Thus, the question seems too trivial to answer and cannot be a good point to compare with existing work.

4. To access the generalization of the approach, it would also be good for the author to list the objects used during the testing. Also, It would be better to describe the level of randomization and data distribution for that.

### Soundness
3

### Presentation
3

### Contribution
3
