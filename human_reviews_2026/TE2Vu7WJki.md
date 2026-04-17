# NeuralOS: Towards Simulating Operating Systems via Neural Generative Models

- Decision: Accept (Poster)
- Scores: 8, 6, 2, 6

## Abstract
We introduce NeuralOS, a neural framework that simulates graphical user interfaces (GUIs) of operating systems by directly predicting screen frames in response to user inputs such as mouse movements, clicks, and keyboard events. NeuralOS combines a recurrent neural network (RNN), which tracks computer state, with a diffusion-based neural renderer that generates screen images. The model is trained on a dataset of Ubuntu XFCE recordings, which include both randomly generated interactions and realistic interactions produced by AI agents. Experiments show that NeuralOS successfully renders realistic GUI sequences, accurately captures mouse interactions, and reliably predicts state transitions like application launches. Beyond reproducing existing systems, NeuralOS shows that synthesized training data can teach the model to simulate applications that were never installed, as illustrated by a Doom application, and suggests a path toward learning user interfaces purely from synthetic demonstrations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper suggests that it may be possible to simulate computer programs without having their code through a demonstration of using RNNs + Attention + Diffusion Models with next-pixel-prediction to generate realistic responses to computer input (i.e., keyboard or mouse input).  They show through several experiments that this is highly effective in some toy settings.   They also included evidence that this was not nearly as effective without several novelties added to the process, such as their unique way of representing (x,y) coordinates and their specialized training schedule.

### Strengths
This paper is clearly in line with ICLR's goal, to advance theoretical and empirical knowledge of learning.  It has huge potential implications for both theory and practice, suggesting a) that in a very concrete and systematizable way, neural attention mechanisms can replace symbolic logic, and b) suggesting alternatives to current practice in software development.  While I imagine that these results are limited by the simplicity of the environments that were being simulated, that in itself could become an area of theoretical interest in itself - how do we create complexity classes of software applications which can predict whether or not they can be imitated using such a model?

### Weaknesses
This paper potentially presents significant ethical challenges - if a model can be trained to in effect "become" a software package, then it will become difficult to preserve the current economic approach to selling software.  

Besides that - I understood why they chose to rely on an RNN, but those are famously not as expressive as transformers, and, in general, I would have liked to see more analysis of what kinds of applications \textit{couldn't} be simulated, if any; if literally any software could be simulated, I would have liked to know how they got so much expressivity out of an RNN.  What I would imagine based on their architecture is that the system would be limited in both how complex the software they simulated could be, as well as by how many different software packages a single module could simulate.  I would have liked to see the performance as a function of how many distinct software packages were being imitated by a given module.

### Questions
On line 209 - could you please explain the logic for concatenating the noisy image?

Do you have intuition why your unique spatial encoding did better than just including the x,y coordinates?  I would have guessed that the network could have learned to pay particular attention to these details and to tiny differences in those coordinates without requiring a new map formalism (assuming they were floats passed into the module).

Should we think of the scheduled sampling as a kind of teacher forcing, or could you elaborate on theoretical motivations for using it?

On line 265, where it states “After pretraining, the RNN-generated frames alone tend to be blurry due to averaging multiple plausible outcomes, but crucially provide a strong initialization for subsequent joint training” - could you be a bit more specific about the strengths and weaknesses of every stage of training, perhaps in the form of a small table?

Line 270 - I'm surprised occasionally replacing the input image with the model-generated frame works at all if they are actually different and the output crucially depends on what the image is; it seems that if you give it a different input, it should produce a different output as \textit{correct} behavior.  Could you please clarify?

You said that you were using "random exploration" to mine traces, but your search tree looks nothing like random exploration - did you just mean random in the sense of non-deterministic, or am I misunderstanding the role of the search tree in gathering data?

Line 313 - is the use of Bezier curves to model cursor usage novel?  If not, the literature should be cited.  If so, a simple ablation with using a simpler family of curves would be great (although probably more of a camera-ready than necessary-by-rebuttal) item.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces NeuralOS, a generative framework designed to simulate operating system graphical user interfaces by autoregressively predicting screen frames based on user inputs like mouse movements, clicks, and keyboard events. The architecture combines a hierarchical recurrent neural network to manage and track the internal system state with a diffusion-based neural renderer that generates the final screen image conditioned on the RNN's state. The authors ask an LLM-based computer-use agent (Claude-3.5-Sonnet) to generate interaction traces, and alongside random synthetic traces through a Docker Ubuntu instance to generate the underlying training data with actions.

### Strengths
1. Cleanly split state tracking (hierarchical RNN with attention over the previous frame) from image synthesis, which makes the problem well-posed for long, interactive sequences.

1. This paper is very instructive in providing a recipe for training video based world-models in general. There are several important tricks here that are more broadly applicable than training a neural OS world model. For instance, the model architecture for long memory, multi-stage training, combination of tricks in Section 4, sampling _transitions_ more often, adding random explorations to mitigate spurious correlations etc.

1. Fast 2-step DDIM giving ~18 fps on a single H100, which is useful for interactive demos, not just offline generation.

1. Reproducibility & openness with code, checkpoints, data scripts, and an interactive demo is linked.

### Weaknesses
1. You can't really use the final artifact for much. If you want to train a model to use an operating system, you'd rather just use the Docker container. I'm finding it a bit hard to justify the contribution of this paper beyond that it is a very interesting demo and some of the tricks used to collect the data and make the world-model work.

1. The demonstration paths through the Docker OS to collect training data were generated by a computer-use agent (Claude-3.5-Sonnet). This is obviously very expensive to collect a very large amount of interaction traces.

1. The quality of the model isn't that particularly great. Notably, in the demo, typing into the terminal barely works.

### Questions
1. Were the human evaluations done on the same distribution of interaction traces used to collect the data?

1. Given that a standard Docker container provides a perfect, zero-cost, high-fidelity environment for training an OS agent, what is the practical justification for doing 23,000-GPU-hours to train this model? What tangible use case does this simulation serve today the real environment doesn't solve more effectively and cheaply?

1. The paper admits that "fine-grained keyboard interactions are not reliably supported". How can this be presented as a successful OS simulation if it fails at one of the most basic and critical components of an operating system (the terminal)?

1. How exactly does the "fabricated" Doom demo, which was manually constructed from "spliced in" gameplay, demonstrate a generalizable capability to simulate new or useful applications? Isn't this just a one-off, manually-guided demo rather than an emergent property of the model?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework to simulate operating systems’ (OS) graphical user interfaces (GUIs) through single screen frame prediction in response to user’s inputs, using deep neural networks. The proposed approach spans the entire pipeline - from data collection to the final prediction - and employs two neural architectures: a recurrent neural network (RNN) for state-tracking, and a diffusion-based convolutional neural renderer for image generation. The authors formalized the task of next frame prediction as an autoregressive generative modeling problem. 
The models are trained on synthetic data, utilizing both random and AI-agent human-like generated interactions within Ubuntu XFCE environments. The methodology is clearly stated and well presented. The authors address OS-specific simulation challenges - distinct from those encountered in, for example, game environments. Their proposed solutions include notable design choices, such as the use of an RNN for long-term state tracking, and a Gaussian spatial map for precise cursor modeling. 
Although no formal metrics or benchmarks are mentioned, the authors report that, based on  human-evaluation results, the model achieves visually coherent frame prediction. 
The paper’s main contribution lies in demonstrating that the model can even generalize to previously unseen applications (e.g., clicking an icon for a non-installed app and producing plausible launch behavior). One suggestion regarding this last point, would be to expand on the significance of this finding and its broader implications for the future of generative user interfaces. The work is motivated by the goal of creating learned, interactive simulators of operating systems through synthetic demonstrations, but the research motivation behind this should be stated more clearly.

### Strengths
* The paper is clearly written and well organized; the technical components are formally defined and adequately justified, which makes the work easy to follow.
* The proposed architectures are coherently structured and integrated. 
* The paper represents an original contribution, as generative simulation of Operating Systems remains novel and unexplored, though it draws some parallels with prior work on generative simulation in other fields (i.e.: gaming).
* Code is given to reviewers, allowing the reproducibility of the proposed approach.

### Weaknesses
* The motivation remains vague throughout the paper. The key questions left to be answered are: “How can this contribution advance the field?”, “Why do we need to simulate OSs?”. A more explicit problem statement - together with expected applications (i.e.: human-computer interaction research, AI agents) would strengthen the paper’s significance. 
* No closely related previous work is discussed - hence, no competitors are referenced. The authors only cite examples from game or real-time simulations, omitting directly relevant efforts in neural or generative OS simulations. Providing a stronger contextualization of the work, would help in collocating the study - e.g.: by referencing recent work such as “Simulating a Neural Operating System by Google (at https://developers.googleblog.com/en/simulating-a-neural-operating-system-with-gemini-2-5-flash-lite/) or other generative interfaces models (as in  https://arxiv.org/pdf/2310.04875). This would clarify how the proposed approach advances beyond prompt-driven systems. * This too, would aid the authors in the articulation of why a fully autonomous OS would be desirable. 
* Experimental validation is limited. Although ablation studies are included to assess each components’ importance, no quantitative comparison against state-of-the-art baselines is provided. Qualitative evaluations and human-evaluation results are present and informative, but they would benefit from additional quantitative evidence (i.e.: comparison with SOTA, statistical analyses, benchmark comparisons, graphs). 
* Development is limited to synthetic Ubuntu XFCE environments, hence, the generalization ability to other OS interfaces remains to be verified.
* Regarding the provided demo, the simulation exhibits some technical issues: frame rendering lags; disappearing cursor behavior when the RNN toggle is activated; the “Auto Input” option not working - there is no frame prediction (i.e.: the screen does not change) and after about five seconds of waiting for an autonomous screen update, one gets a “Connection Timeout Warning” and the connection restarts. These issues make it difficult to evaluate the system’s claimed interactivity.
* An additional concern regards the resources required. The high GPU cost required for the proposed OS simulation - which cannot approach performance comparable to real-time OSs - raises questions about utility and scalability. In simpler words: “Why is it necessary to use all those hours of GPU usage in order to simulate an OS that will never reach performance comparable to a real OS (which is also, way cheaper)?”

### Questions
See weaknesses

### Soundness
2

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
3

### Summary
This paper proposes NeuralOS, an end-to-end generative model–based operating system capable of simulating and rendering graphical user interfaces (GUIs) through real-time cursor and keyboard interactions. The study adopts a multi-stage training framework to enhance the quality of GUI simulation. Experiments and demo results suggest that NeuralOS can effectively simulate GUIs via GPU rendering without relying on a physically hosted interface. Overall, this work presents an intriguing step toward developing a world model for GUI simulation.

### Strengths
This work provides a clear and detailed description of the model architecture, input feature encoding, and multi-stage training strategy to ensure rendering quality. The use of LLMs to autonomously collect interaction data, thereby removing human involvement, is particularly interesting. Overall, the paper demonstrates a promising approach to constructing a world model for operating systems, capable of generating real-time screens based on user interactions.

### Weaknesses
#1 The paper should better position its work within the context of existing works. For instance, although video generation is briefly mentioned (line 146), no specific papers are cited. Similarly, despite of the discussion on world models in Section L, the main text does not contextualize NeuralOS in world models. The authors clarify how their approach to modeling long-term trajectories and user actions differs from prior methods used in video generation and world modeling.

#2 The discussion of challenges in lines 214–215 is unclear. In Stage 4, the paper notes that capturing long-term dependencies is difficult due to hardware limitations, yet the input context is later extended to address this. If this extension mitigates the issue, the authors should clarify in what sense this remains a challenge.

#3 The paper mentions that transformers suffer from high inference complexity, yet the proposed model incorporates attention heads which is an essential component of transformers. 

#4 Cursor coordinates are encoded twice, at line 174 and line 199, respectively. The paper should clarify the rationale behind this encoding design or conduct an ablation study to explain how it contributes to the simulation of cursor movement.

#5 In evaluating the performance of NeuralOS, the performance metrics and their formulas should be clarified and referenced. For example, the formula for accuracy is not explicitly provided in the main text. It is unclear whether the accuracy reported in Table 1 (which measures the identification of the real system) is conceptually the same as the cursor position accuracy and overall image simulation accuracy.

### Questions
Please answer the questions in the weakness section.

### Soundness
3

### Presentation
4

### Contribution
3
