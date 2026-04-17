# Can vision language models learn intuitive physics from interaction?

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Pre-trained vision language models do not have good intuitions about the physical world. Recent work has shown that supervised fine-tuning can improve model performance on simple physical tasks. However, fine-tuned models do not appear to learn robust physical rules that can generalize to new contexts. Based on research in cognitive science, we hypothesize that models need to interact with an environment to properly learn its physical dynamics. We train models that learn through interaction with the environment using reinforcement learning, as well as models that learn without interaction using supervised fine-tuning. While both reinforcement learning and supervised fine-tuning appear to improve within-task performance, they fail to produce models with generalizable physical intuitions. Models trained on one task do not reliably generalize to related tasks, even if they share visual statistics and physical principles, and regardless of whether they are trained through interaction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates whether VLMs can learn generalizable intuitive physics through interaction, as opposed to passive SFT. The authors compare two methods—GRPO (an interactive RL method) and SFT (non-interactive)—on tasks involving block tower stability and construction. They test three hypotheses: whether interactive training improves (1) within-task generalization, (2) cross-task generalization, and (3) sample efficiency on new tasks. The results show that while both methods achieve near-ceiling performance on trained tasks, neither leads to robust generalization to related tasks, suggesting that current fine-tuning methods encourage shortcut learning rather than genuine physical reasoning.

### Strengths
1. The paper is one of the first to systematically compare interactive (RL) and non-interactive (SFT) training for intuitive physics in VLMs.
2. The writing is clear, and the experimental setup is well-explained.

### Weaknesses
1. Limited Experimental Scope: The study is confined to a single model family (Qwen2.5-VL) and a narrow set of block-stacking tasks. Broader evaluation across diverse model architectures and more varied physical reasoning tasks would strengthen the conclusions.

2. Limited Innovation: The paper only compares SFT and RL on a small task without further analysis of the underlying reasons or proposing potential solutions—or at least offering improvements specific to the physics subtask. Additionally, it does not explore whether longer training, alternative RL algorithms, or more diverse interaction strategies could enhance generalization.

3. Limited Community Impact: The paper’s point about not overestimating the generalization ability brought by SFT (L459) appears to be a widely recognized observation. Moreover, the study focuses only on the narrow subfield of physical perception and is validated on a small-scale dataset, which limits the significance of its findings.

4. Presentation Issues:
(a) Appendix A.5 Attention Maps: It is unclear what the authors intend to illustrate with these visualizations.
(b) Related Work: The related work section is overly verbose and fails to clearly highlight the paper’s significant contributions compared to prior research.

### Questions
See weaknesses.
I am willing to chat with the authors to improve the paper.

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
The paper asks whether interaction helps VLMs acquire generalizable intuitive physics. Using Two TDW tower datasets and four tasks, the authors fine-tune Qwen2.5 VL with SFT or GRPO. Both reach near ceiling on the task they train on, yet show little transfer to related tasks. Linear probes can decode relevant physical quantities from activations, but this competence does not translate into zero-shot performance. Additional SFT on a new task learns faster than from the base model. Overall the study delivers careful negative results (which I appreciate a lot that the authors shared this.)

### Strengths
- Clear question and hypotheses grounded in cognitive science
- Controlled comparison of SFT and RL with matched PEFT settings and budgets
- Simple tasks with explicit rewards and prompt templates, plus training logs
- Generalization matrix across all train and test task pairs
- Decodability analysis that separates representation competence from output performance
- Useful visualization of reward landscapes and attention maps
- Negative results are reported transparently

### Weaknesses
- Very narrow scope. One model family at one size and one environment
- Interaction is minimal. One step RL with short textual actions, not true multi-step closed loop control
- Fixed camera and block sizes make pixel shortcuts likely, which undermines conclusions about physics learning
- No baselines for multitask SFT, joint training across tasks, or auxiliary representation losses
- Linear probe dataset is small and lacks controls such as image only probes or interventions

### Questions
- What happens with multi-step interaction and longer action horizons?
- Does heavy domain randomization of textures, lighting, camera pose, and block size improve transfer?
- How does joint multitask SFT across all four tasks compare to single task post-training?
- Do larger backbones or different families change the outcome?
- Can you test on a external stability dataset to validate generality?

### Soundness
3

### Presentation
3

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
This paper explores whether supervised fine tuning works better than reinforcement learning fine tuning for vision language models in the context of intuitive physics tasks. They construct a train and test dataset using the ThreeDWorld simulator where structures are constructed with multiple blocks (i.e cubes), and an agent is required to either make a judgement of whether the structure is stable or how much a given block should be moved to make the structure stable. Vision language models are then trained on this task using supervised finetuning or reinforcement learning and tested to compare relative performance. 

The authors find that there is not much difference in the performance of these two classes of models. They perform at ceiling when tested on the same kind of tasks seen in training and generalize equally poorly to new physical tasks.

### Strengths
1. Training models to understand intuitive physics by interacting with the environment  is a well motivated hypothesis, as it is similar to how babies learn. 
2. The proposed metrics to test models seems reasonable, and the authors conduct rigorous evaluations.

### Weaknesses
1. The dataset seems a bit simple and contrived. The fact that supervised learning performs as good as reinforcement learning might be because it’s a really easy task, and not because both methods fundamentally work equally well. 
2. I strongly disagree with the statement made in the conclusion that “these results cast doubt on whether posttraining methods are sufficient for developing models that reason about the world in a human-like manner”. The models not generalizing to new tasks, might just be because the training set is nowhere close to the amount of data that babies see, and not because the training algorithm is limited in some way. So we can’t really conclude anything about which model class is better from this result. 
3. It seems like the rewards are really handcrafted for this particular task. This is certainly not how humans would get rewards in the real world, so I’m curious to know what the authors think about how this method would scale to multiple tasks in different environments. Would rewards need to be defined for each task separately?
4. It’s also not clear whether babies need a particular set of task specifications and goals for being able to learn intuitive physics. Most of intuitive physics might be learnt just by passive object manipulations without any defined goal like stability or placement. So how would we control for this kind of variable in the experiment? It makes me think that there is an inherent limitation in the way the training pipeline is set up here, which again makes me less confident about making any conclusions.

### Questions
My main concerns are about the dataset being too simple. How could this be extended or improved to ensure that the conclusions are reliable?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates whether VLMs better learns physical reasoning through SFT versus RL (GRPO).
They use a custom block tower building benchmark with two tasks:
- binary stability prediction
- block displacement prediction for tower stabilization

They compare two training approaches, using the Qwen2.5-VL 7B model with PEFT:
(1) SFT: Supervised fine-tuning on labeled correct answers
(2) GRPO: RL where the model predicts displacements/stability and receives rewards based on whether predictions would result in stable towers

### Strengths
The research question is well-motivated - comparing RL versus passive supervised learning for physical reasoning is an important question for embodied VLAs and robotics.

The method section is clearly written and self-contained, with sufficient implementation details.

The experimental design is clean and controlled, the writing of the experimental section is ordered nicely,
as each experiment motivates the following ones.

### Weaknesses
While this serves as a useful motivating example, the scope of these experiments (single model, relatively narrow custom task) makes it difficult to draw significant conclusions, and it's unclear whether these findings can be generalized to larger scale training with multiple tasks.
I like the direction of this work but I believe the findings and experiments are too limited for a full publication at ICLR.

The results are hard to parse - a simple table summarizing performance and rewards would complement Figure 2 well.

### Questions
What we can actually take away from this work for larger scale training of spatial / physics understanding for VLMs?
What is the message the authors are trying to convey with this work?

The VLAs in robotics are trained on physically grounded tasks (pick and place, manipulation) and might learn some of this "physics concepts" implicitly through their training.
I wonder if this actually makes a difference compared to the data and task Qwen2.5-VL is trained on.

### Soundness
2

### Presentation
2

### Contribution
2
