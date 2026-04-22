# Object-Centric World Models from Few-Shot Annotations for Sample-Efficient Reinforcement Learning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
While deep reinforcement learning (RL) from pixels has achieved remarkable success, its sample inefficiency remains a critical limitation for real-world applications. Model-based RL (MBRL) addresses this by learning a world model to generate simulated experience, but standard approaches that rely on pixel-level reconstruction losses often fail to capture small, task-critical objects in complex, dynamic scenes. We posit that an object-centric (OC) representation can direct model capacity toward semantically meaningful entities, improving dynamics prediction and sample efficiency. In this work, we introduce **OC-STORM**, an object-centric MBRL framework that enhances a learned world model with object representations extracted by a pretrained segmentation network. By conditioning on a minimal number of annotated frames, OC-STORM learns to track decision-relevant object dynamics and inter-object interactions without extensive labeling or access to privileged information. Empirical results demonstrate that OC-STORM significantly outperforms the STORM baseline on the Atari 100k benchmark and achieves state-of-the-art sample efficiency on challenging boss fights in the visually complex game **Hollow Knight**. Our findings underscore the potential of integrating OC priors into MBRL for complex visual domains.

Project page: https://oc-storm.weipuzhang.com

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies object based representations for learning world models in a sample efficient manner. They propose to use SAM2 and Cutie as the frozen models which provide object vectors. These vectors along with the full frame are passed into categorical vae followed by a spatial transformer whose outputs are used to predict the next latents, rewards and termination. They show the effectiveness of their approach on atari100k, hollow knight and some continuous control tasks.

### Strengths
I appreciate the idea of adding object-based inductive biases into world modelling, this allows to maintain object consistency and tracking for all objects especially smaller ones.

I like that the authors opted for wide evaluation suite going beyond atari to hollow knight and continuous control.

### Weaknesses
I think the paper would benifit from comparing against unsupervised object-centric representation learning baseline such as slot attention (https://arxiv.org/abs/2006.15055) and slotformer (https://arxiv.org/abs/2210.05861). The main claim of the paper is that object based vectors help world modelling but it is not clear whether there is something special in vectors provides by SAM2/Cutie or even unsupervised methods can also help. Also world modelling has been a focus in the unsupervised object-centric representation learning community (eg. slotformer, SSWM( https://arxiv.org/abs/2402.03326)) hence it might be useful to copmare against some of these baselines.

It would be great if the authors could clarify and make the setup more clear. For example,
- The main model figure is in the appendix, I think it should be in the main paper
- It is not clear how the FOCUS baseline works - how do you discretize the masks from FOCUS?

The claim for discretization is to limit compounding of errors. I don't fully buy this, shouldn't errors compound in discrete autoregressive prediction too? It would be great to have a baseline which does not perform discretization so only continuous prediction to show the pros and cons


The authors claim that the method is sample efficient which is great, however if given access to more samples, will the method scale as well as non object-based methods?

### Questions
What is the policy to collect rollouts for training the world model?

How do you decide the number of object vectors for each task?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Presents OC-STORM, an object centric model based RL methods:
- The user gives the model a few annotated object masks which are used to get SAM2 / cutie object representations.
- The object vectors and a downsampled image vector are embedded to a categorical latent (via categorical VAE)
- A world model is trained via reconstruction
- A policy is trained inside of the world model for control. 

Performance results:
- The object-centric variants of DreamerV3 and STORM tend to perform better than the non-object centric part. 
- Further, for games where SAM / Cutie is better at detecting objects in, performance is consistently better, which demonstrates the merits of the method when there is a good object extraction
- Further evaluated in a hollow knight 100k setting and find that it out-performs the on-OC variant
- Can also work in continuous tasks (Metaworld)
- Analysis shows that (i) object information is well-learned as scene can be reconstructed from object vectors (players in boxing), and (ii) there is some robustness to randomly zero-ing out detected objects

### Strengths
The idea is novel in using few-shot labelled object masks to get (pre-trained) object representations, which is then used to train a world model for policy control.

The paper presents clearly and is of high quality. It presents positive results against reasonable baselines across a number of environments (atari100k, hollow knight, metaworld). It reads as a comprehensive work that can be informative for future works in object centric RL to build on.

The ablations are comprehensive. The authors do good science to isolate the effect of object encoding (e.g. object detectable vs. badly detected), if the object representation can be reconstructed, and robustness to segmentation model failure.

Finally, I appreciate that authors are also frank in discussing some short-comings of current object extraction methods (such as inability to handle duplicate objects and geometric structures such as walls and floors) for future research to build on.

### Weaknesses
The major weakness of this method is the effort vs. gains trade-off for using this object-centric representation. To use OC-STORM, the user must first generate 6-12 frames of object mask labels for each environment they may wish to run. The gain from doing this, based on the paper, seems to be _mainly_ about better _sample efficiency_. One could argue that instead of going through the effort of labelling, the user can also (i) run the alternative methods longer to get similar performance, or (ii) use a more compute-intensive method to get the same sample efficiency (e.g. Delta-IRIS or DIAMOND in Appendix C seem to show this -- correct me if I mis-understood). 

On the other hand, some of the original work on object-oriented RL were motivated by the hope that modelling object & interactions can allow generalization in fundamentally different ways [1]. More recent works have similarly shown zero-shot generalization [2,3], new ways of doing model learning & efficient exploration [3], and the ability for compositional generalization [4,5]. 

If sample efficiency is indeed the main goal, and being mindful that future work can push this method to be even more performant than the more compute intensive methods, it still feels worthwhile to discuss the particular use case for the current approach as presented currently (for instance, in a low sample, low compute setting?), and how it may differ (or is similar) to other object-oriented approaches.  

-----

[1] Diuk, Carlos, Andre Cohen, and Michael L. Littman. "An object-oriented representation for efficient reinforcement learning." Proceedings of the 25th international conference on Machine learning. 2008.

[2] Sancaktar, Cansu, Sebastian Blaes, and Georg Martius. "Curious exploration via structured world models yields zero-shot object manipulation." Advances in Neural Information Processing Systems 35 (2022): 24170-24183.

[3] GX-Chen, Anthony, Kenneth Marino, and Rob Fergus. "Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction." arXiv preprint arXiv:2408.11816 (2024).

[4] Zhou, Allan, et al. "Policy architectures for compositional generalization in control." arXiv preprint arXiv:2203.05960 (2022).

[5] Haramati, Dan, Tal Daniel, and Aviv Tamar. "Entity-centric reinforcement learning for object manipulation from pixels." arXiv preprint arXiv:2404.01220 (2024).

### Questions
- How do you decide on how many key objects to label in the human annotated frames? Is there any guidance here for what to / not to label?
- Can the segmentation model work without any labelled frames (I was under the impression that SAM can)? If so, how accurate are they at segmenting objects?
- How sensitive is the policy to different human labels? If different users labelled the same game, or labelled a different number of objects, how would this change the segmentation and would the policy still be robust to this?

### Soundness
3

### Presentation
4

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
The paper proposes OC-STORM, an object-centric model-based RL framework that fuses few-shot, pretrained video-segmentation features with pixel inputs to train a spatial-temporal world model, yielding improved sample efficiency on Atari-100k and Hollow Knight.

### Strengths
- The method integrates few-shot object features from SAM2/Cutie into a spatial–temporal world model (Transformer/RNN backbones), with clean modality separation (object tokens + visual token) and categorical VAE discretization; the training and architecture are well specified. 

- On Atari-100k, object-centric variants outperform baselines (e.g., Cutie-OC-STORM reaches HNS mean 134.8% vs. STORM 114.2%, median 43.8% vs. 42.5%); Hollow Knight learning curves show faster convergence on harder bosses. 

- The paper diagnoses why vector features beat mask features, provides module ablations, feature-only reconstructions, and a failure-robustness study by zeroing object features; Meta-World results indicate portability beyond Atari.

### Weaknesses
- Comparative scope. Core comparisons are mainly within-framework ablations (STORM/DreamerV3 variants); external SOTA world-model baselines (e.g., diffusion/tokenization variants) and broader agent baselines are deferred or absent in the main text, and Hollow Knight lacks standardized settings—making cross-paper claims harder to calibrate. 

- Annotation/K configuration burden. The user-set K (objects) and handful of annotated frames (≈6–12) are reasonable but the human-time budget and sensitivity to K are not quantified in the main text; detection incompleteness is acknowledged but only coarsely analyzed. 

- System metrics incomplete. The paper references computational overhead analyses but does not report build/index/latency figures for segmentation + KG-like memory in the main body—useful for scaling to long episodes or high-res inputs.

### Questions
- Labeling cost & K sensitivity. How many minutes of annotation per game are required in practice, and how does performance vary with K (under-/over-specifying the number of tracked objects)? Could you add a curve for returns vs. K and vs. number of annotated frames? 

- External baselines & reporting. Can you include a matched-config comparison against recent token/diffusion world models and standardized Hollow Knight setups (or release your wrapper to make one), plus runtime/latency tables for the segmentation pipeline?

### Soundness
3

### Presentation
2

### Contribution
3
