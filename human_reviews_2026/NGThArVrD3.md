# Setting up for failure: automatic discovery of the neural mechanisms of cognitive errors

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 10

## Abstract
Discovering the neural mechanisms underpinning cognition is one of the grand challenges of neuroscience. However, previous approaches for building models of recurrent neural network (RNN) dynamics that explain behaviour required iterative refinement of architectures and/or optimisation objectives, resulting in a piecemeal, and mostly heuristic, human-in-the-loop process. Here, we offer an alternative approach that automates the discovery of viable RNN mechanisms by explicitly training RNNs to reproduce behaviour, including the same characteristic errors and suboptimalities, that humans and animals produce in a cognitive task. Achieving this required two main innovations. First, as the amount of behavioural data that can be collected in experiments is often too limited to train RNNs, we use a non-parametric generative model of behavioural responses to produce surrogate data for training RNNs. Second, to capture all relevant statistical aspects of the data, rather than a limited number of hand-picked low-order moments as in previous moment-matching-based approaches, we developed a novel diffusion model-based approach for training RNNs. To showcase the potential of our approach, we chose a visual working memory task as our test-bed, as behaviour in this task is well known to produce response distributions that are patently multimodal (due to so-called swap errors). The resulting network dynamics correctly predicted previously reported qualitative features of neural data recorded in macaques. Importantly, these results were not possible to obtain with more traditional approaches, i.e., when only a limited set of behavioural signatures (rather than the full richness of behavioural response distributions) were fitted, or when RNNs were trained for task optimality (instead of reproducing behaviour). Our approach also yields novel predictions about the mechanism of swap errors, which can be readily tested in experiments. These results suggest that fitting RNNs to rich patterns of behaviour provides a powerful way to automatically discover the neural network dynamics supporting important cognitive functions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes training RNNs to reproduce behavioral distributions including characteristic error modes than optimizing task performance. They do so by first using a generative model to produce synthetic data for training RNNs, then adapting diffusion training procedures for RNNs to produce realistic behavioral distributions. The authors tested their method on the visual working memory task, and found that the DDPM-trained RNNs, instead of the task-optimized RNNs, produces realistic swap errors and neural representations.

### Strengths
1. Training RNNs to produce realistic and complex behavioral distributions is a very important and timely research direction. 
2. This paper is very well-written and easy to follow. 
3. The experiments are well designed and the authors provided analysis at both the behavioral and representational level.

### Weaknesses
1. The BNS model used to generate synthetic data still requires hand-tuning, so “automatic discovery” feels overstated. Could the authors comment on this limitation?
2. In Figure 3A, the correspondence between border colors and RNN models is unclear. Please state this explicitly in the caption (or add a legend).
3. On line 203, (F) has not been introduced, and the network equation is missing. It would help to present Equation 3 (or at least the network update equation within it) earlier.
4. On line 252, I think the intended expression is (x = G(r) = W_x r) (the r appears to be missing).
5. The approach depends on a generative model to create synthetic behavioral data for training the RNNs, which raises several concerns:

   * For more complex tasks, well-founded hypotheses or statistical/generative models of the behavioral process may be unknown or unavailable.
   * The method assumes the generative model captures key characteristics of behavior, yet the paper provides no quantitative evidence for this beyond showing that swap error increases when the probe is closer to the cued item.
   * Lines 146–149 claim that models listed in §1 “can be used to faithfully generate synthetic behavioural data,” but this is not supported with empirical evidence or citations to prior work.

If these concerns are properly addressed, I may consider increasing my score.

### Questions
1. Does well does the method generalize to a more complex versions of the task, for example, with more items?
2. How does the method apply to tasks where an output is required at every time step of the trial? 
3. Fig 3C: for the brown-border model, shouldn’t close probe feature values have more swap error? (It looks like it's the opposite in the figure).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work trained RNNs to exhibit swap errors during a multi-item delayed estimation VWM task. Its success was contrasted with naive training methods applied to task-optimal networks. The proposed method captured a more accurate description of neural mechanisms when trained to adhere to the empirical variability around real swap errors.

### Strengths
1. The research problem itself is interesting and promising, i.e. discovering neural mechanisms of cognitive errors. However, it is doubted whether such simulated errors by RNNs can be really called 'neural mechanisms' of human cognitive errors, since the mechanisms were revealed in RNNs rather than human neural activities.

2. The background section makes it easier for the reader to understand the context.
 
3. The proposed model was evaluated in a working memory task dataset and showed meaningful patterns and signatures of cognitive behaviors.

### Weaknesses
1. Most evaluations are qualitative results, rather than quantitative results. Although the discovered signatures are promising, the lack of quantitative measurements significantly reduces the rigor of this work.

2. Only one task (working memory) is used for evaluation. Although I agree that this is a representative task, the small task domin limits the generalization ability of this proposed model and it is not convincing that it can be effective in other cognitive task domains.

3. Limited baseline or ablation studies to show the unique advantages of the proposed model to discover mechanisms of cognitive errors compared with prior models in existing work. I can only find one naive training method as the baseline.

4. Lack of details in the dataset used for evaluation, including working memory task settings/design, sample size of participants and trials in the task, etc. This makes it hard to evaluate the significance of this work and reliability of the findings.

### Questions
Please check my concerns and questions in Weaknesses section.

### Soundness
2

### Presentation
3

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
The authors train RNNs to fit distributions of behavioral output. This is done by modifying a diffusion based procedure. The algorithm is tested on a synthetic model of behavior. The resulting network’s hidden state statistics has similarities to those observed experimentally.

### Strengths
Behavior is often probabilistic, and not necessarily unimodal. Most models add some noise on top of a deterministic backbone. Using flexible distributions directly is an important contribution.

### Weaknesses
Despite the introduction, there is no fit to actual behavioral data. Is there any guarantee that this can work?

Clarity: There are several places where the writing is not clear. For instance, the paper jumps between index and feature based codes, without explaining the rationale.  

The basic premise in line 41: RNNs are not the only option for testable hypotheses. They are a specific example of a latent variable model. 

Line 110 – biologically plausible is a broad and ill-defined term. Some people only consider spiking networks to be plausible. Others require conductance based models. Others are satisfied with rate models, but insist on Dale’s law.

### Questions
Line 120 – Why dendritic tree?
Lines 208-214 The description of index and feature based is not written clearly. The overuse of the word feature does not help.
Line 253: Clarity. If I understand correctly, the network activity r is split. But the sentence reads as if it is split into x and m. The next sentence reads “this output”, but the word output does not appear in the previous sentence.
Figure 7A: could be useful to explain what the horizontal and vertical axes of s_t are

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
In this submission, the authors propose an alternative method to train models of biological visual processing that don't rely on task-optimization. The proposed method aims to synthetically generate a large amount of behavioral data and introduces a diffusion-based method to train RNNs to fit the generated behavioral data, hence automating the development of brain-aligned dynamical system models without a human-in-the-loop. RNN models of a multi-item delayed estimation problem are trained to display 'swap errors' which are characteristic of human behavior on such visual working memory problems, and a representational geometry analysis is shown to highlight how the trained RNN models capture key indicators of visual working memory behavior.

### Strengths
- Training biologically plausible dynamical systems models using deep neural networks has been saturated for a while with task-optimized neural networks. The proposed approach in this paper to automate the process of generating ecologically valid synthetic behavioral data and a new paradigm to train RNNs on the above generated data with denoising diffusion is clearly a fresh take that is well distinguished from task-optimized network training.
- While the proposed method is shown to model behavior in a visual working memory task, I believe the general framework proposed here has the potential to scale to, and help interpret other dynamic biological processes in perception and decision making.

### Weaknesses
- While task-optimized networks are notoriously difficult to design and train, they provide a direct way to compute representational similarity relative to biological brains on naturalistic stimuli (images, videos, audio, etc). The proposed methods here, however, train RNNs that operate at a different level of abstraction of the input if I understand correctly. This is expected, as the proposed model is one of biological decision making and not necessarily perception. But if one were to apply the proposed method to model perceptual processes, could the authors please comment on how one could bridge this gap in input complexity consumed by their proposed RNN models? Would BNS style synthetic data generation even be feasible for generating coherent, high-dimensional sensory behavioral data?

### Questions
NA. Please refer to my review above.

### Soundness
4

### Presentation
4

### Contribution
4
