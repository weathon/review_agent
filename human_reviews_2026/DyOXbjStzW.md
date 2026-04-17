# Mechanisms of skill transfer from pretraining to target task in recurrent neural networks

- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Pretraining on simpler tasks can often improve learning outcomes on a more difficult target task. Nonetheless, what makes for a good pretraining curriculum and the mechanisms of positive transfer across tasks remain poorly understood. Here we use RNNs trained on fixed length temporal integration to compare curricula with varying degrees of effectiveness. We show that pretraining on simpler versions of the target task is less effective than curricula which take advantage of the target task's compositional structure and train sub-skills needed for solving it. By exploiting the highly structured solution of our target task, we can mechanistically explain improvements in speed and quality of learning in terms of the slow features of the RNN dynamics that the curriculum helps build, and the reuse and adaptation of those slow features during target training. Our results argue that pretraining on tasks that individually hone sub-skills required for the target are particularly beneficial, as they build a scaffolding on which additional dynamical systems structures can be compositionally expanded to achieve the final function. Thus, our results document a novel mechanism for repurposing dynamical systems features in support of cognitive flexibility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper provides a mechanistic perspective on how pretraining shapes the dynamics of RNNs on a fixed-length temporal integration task. The authors argue that pretraining on compositional sub-skills (the counting task) builds a reusable dynamical scaffold (a limit cycle) upon which the target integration behavior can be built upon.

### Strengths
- Understanding mechanisms of transfer across tasks in RNNs is an important and timely question for computational neuroscience. 
- The paper goes beyond studying the speed of learning and analyzes the dynamical features of the task to provide a mechanistic account of why pretraining on counting speeds up subsequent learning on the integration task.

### Weaknesses
1) Limited novelty and scope

Much of the arguments made by the authors has strong overlap with prior work (for example, Driscoll et al. (2023) on dynamical motif reuse in multitask RNNs; Turner & Barak (2023) on attractor reuse when the RNNs are trained sequentially on multiple tasks with overlapping task structure; Hocker et al. (2025) on compositional pretraining speeds up subsequent learning through reuse of dynamical features). The present contribution largely re-articulates these ideas on a single, relatively simple integration task with fixed trial lengths. The authors acknowledges that the task is “too simple to strictly require pretraining,” which undermines the general significance of the claimed mechanisms.

2) Unclear generalizability

The paper acknowledges that the task is “too simple to strictly require pretraining” (Page 3), raising questions about its broader relevance. The focus on low-rank changes and dynamic scaffolds might not translate to scenarios with higher-dimensional or less structured tasks, and this concern is not adequately addressed.

Other comments:
- On line 91, this is a discrete-time RNN.
- Can you also show the x-tick labels (the values of the input/output) in Fig. 1A–B?
- This paper would benefit a lot from clearer writing and better interpretation of the results. For example, in Fig. 3 the final-CL effective rank and the final–initial effective rank are always the same for the C→Int training regime. This is interesting because it implies that the necessary structure for performing the integration task may have been developed during the counting phase and then reconfigured to perform the full task, but the paper does not explicitly make this point.
- On line 215, the authors state that “good solutions in our task often take advantage of representing time within a trial.” However, it remains unclear why the networks develop a limit cycle to track the passage of time. Is this because multiple trials are concatenated with an impulse at the beginning of each trial to indicate a new trial, so that learning a time-tracking limit cycle helps the network reset its hidden state beyond using the impulse cue?

### Questions
See weaknesses

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
This paper investigates how curriculum design affects transfer and representational geometry in recurrent neural networks (RNNs) trained on sequential sensorimotor tasks. The authors frame skill transfer as a dynamical-systems problem and analyze how pretraining on related subtasks alters the effective rank and phase geometry of population activity during subsequent learning.

They implement a family of structured curricula that vary task difficulty and temporal structure and compare them with single-task (“solo”) training on paradigmatic continuous-control tasks such as Integrate-n and Count-n.

The key findings are:

Pretraining on simpler or shorter tasks accelerates convergence on longer or more complex ones.

Transfer efficacy correlates with trajectory alignment and reduced effective rank in neural population activity.

Intermediate “mixed” curricula yield smoother representational transitions than direct or disjoint training.

Figures 7–8 visualize these effects through changes in effective rank and phase alignment of hidden-state trajectories.

### Strengths
1. The paper goes beyond behavioral speedups and analyzes the geometry of hidden trajectories, linking dimensionality reduction to efficient transfer — a clear and interpretable mechanistic contribution.
2. The chosen tasks (Integrate-n, Count-n) and architectures are simple enough to allow transparent interpretation; the analysis of effective rank and phase structure is well justified and reproducible.
3. Figures 7–8 clearly demonstrate how curriculum structure modulates trajectory alignment and representational compression, with consistent trends across curricula and seeds.
4. The work situates itself among both classical and modern studies of curriculum learning and RNN dynamics, citing Soviany et al. (2022), Schuessler et al. (2020), and Sussillo & Barak (2013).
5. Experimental details, data splits, and metrics are described thoroughly; the authors commit to releasing code upon publication.

### Weaknesses
Lack of formal theory.
There is no explicit model explaining why lower effective rank corresponds to faster transfer. The relationship is described qualitatively but not derived or predicted mathematically.

Narrow task family.
Experiments are confined to low-dimensional continuous-control tasks (Integrate/Count). It remains unclear whether the same representational mechanisms hold in higher-dimensional or non-periodic domains.

1. Omission of closely related theoretical work. The paper cites Proca et al. (2025) on learning dynamics but omits Rajan, Kepple, & Engleken ICLR, who analyzed sequential curricula and representational transfer in RNNs — a directly relevant body of work. This omission weakens the contextualization of the present results.

2. Limited quantitative rigor. While figures are compelling, the authors provide no statistical tests or uncertainty estimates for rank and phase measures; relying solely on visual consistency is insufficient for strong claims.

3. Weak link to cognitive and behavioral data. The discussion briefly mentions parallels to human skill-transfer and hierarchical learning, but these analogies are not developed, missing an opportunity to connect the findings to cognitive or neuroscience evidence. Summerfield and Saxe lab paper are useful refs.

### Questions
1. Can you formalize the observed relation between rank reduction and transfer speed—e.g., via a simple linear-subspace or covariance-alignment model?

2. Have you tested whether rank alignment persists if network recurrence weights are trained rather than frozen?

3. Could you provide confidence intervals or statistical tests for rank and phase-alignment metrics?

4. How would curricula behave under non-periodic tasks (e.g., context switching or delayed association)?

5. How does your mechanistic interpretation relate to the Rajan–Kepple–Engleken findings on representational reuse?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents an empirical study on the structure of the dynamics of a simple RNN at different stages during training in the context of pre-training and curriculum learning for an “evidence integration” task. It uses various qualitative (e.g. attractors in the network activity) and quantitative measures (e.g. rank of weight change matrix) based on the hidden state trajectories, network activity, or weight matrix evolution to characterize the geometry of the network-dynamics and efficacy of different pre-training approaches (same task but simpler vs. different counting based task). They convincingly show differences in efficacy between the different pre-training / curriculum-training approaches, and subsequently compare them to similarities / differences in the geometry of the respectively trained network / network dynamics. Interestingly, the slightly different counting task turns out to be the more beneficial pre-training task.

### Strengths
1) The paper tackles interesting and relevant questions regarding the characteristics of good pretraining tasks, which are thoroughly motivated in its introduction.

2) The pretraining / curriculum learning setup seems novel and the chosen qualitative and quantitative metrics suitable for investigating the geometry of the resulting network / network dynamics of different pretraining approaches.

3) The paper clearly demonstrate differences in efficacy as well as geometric structure in the artifacts of the various pretraining approaches, and the geometric explanation of their efficacy seem like a promising research direction (e.g. reuse, reshape, and de novo formation).

### Weaknesses
W1) The abstract is difficult to understand; possibly due to convoluted writing and imprecise use of technical terms.

W2) The paper seems to lack a section on related work, making the contextualization of experimental setup, evaluation methods and results, as well as originality somewhat difficult.

W3) The results are shown for **one simple RNN** (Elman type) architecture and one type of integration problem which is the sum operation. No comparison of different architectures etc. Will you analysis also be applicable to LSTMs? As it is an empirical paper, I am not convinced about generality of the evidences and conclusions.  

W4) The method section is mathematically imprecise (see mu sampling, possibly incorrect e.g. “continous time“ RNN even though equations show discrete time RNN) and seems to lack a lot of detail (e.g. how were the networks trained precisely, exact training setup, which libraries were used for computation, etc.). Also the appendix lacks these details. The paper is unlikely to be reproducible.

W5) Many figures feature cryptic labeling (e.g. figures 2-4), and could benefit from proper scaling and labeling of legends where missing, or additional details on abbreviations in their subtext.

W6) It is somewhat hard to reason about what exactly the experiments (and metrics) show in the end. It often doesn’t seem like an apples to apples comparison between the integration and counting pretraining task, as multiple dimensions are changed at once (e.g. periodicity AND target), and possibly some more informative ablations or controls are missing (e.g. Integrate 6A to Integrate 12A, or Initial to integrate 6A).

W7) The presented explanation for the efficacy of the pretraining in terms of geometry of the network / network-dynamics is not fully convincing / sufficiently substantiated. While differences in efficacy of the pretraining procedure are clear, and that differences in the geometric structure of the network / network dynamics exists due to the different pretraining also seems sufficiently evidenced; however, the argument / evidence as to how or why the pretraining induced geometry in network / network dynamics is better remains somewhat unclear. For example; how could I determine based on the induced geometry in network / network dynamics PRIOR to actually evaluating the final network training whether it will be more or less beneficial than a particular other geometry?

### Questions
Q1) How was the network trained precisely (loss function, optimizer, learning rate, epochs, batch size, etc.), and the details regarding the training data (exact sigma_stim, mu_0, and number of trials during training)

Q2) How common are networks that perform that task well but have no “bird cage“ structured (as mentioned in Lines 198-199 and Fig. 3D.) And what do their dynamical plots look like (analogous to 1C)?

Q3) For the analysis in terms of rank the network weight changes; was W_in, W_rec or W_out or all three analyzed? And if only W_rec why not all three?

Q4) What is the precise threshold mentioned in Figure 2C (L182-L183)

Q5) How were the perturbations sampled / done in Fig 1.C and 4.C?

Q6) What are the results for evaluating metrics in Figure 6B between e.g. Initial A to Count 6A, or Integrate 6A Integrate 12A? (This could serve as additional control.)

Q7a) How exactly does the geometry of the dynamics of a integration pretrained network look like; i.e. Figure 1C or 4B type of plot but for shorter integration T=6?. And is it more similar to Figure 1C than 4B?

Q7b) And if it is more similar, than doesn’t this conflict with the explanation that the low dimensional change to geometry of the dynamics is responsible for the quality of the pretraining? (At least 5B looks way more similar to 5E than 5A).

Q8) Do the results in Figure 3 “C -> Int” “final-initial” mean that the solution (weights) that are found by pretraining with counting are closer to initializaiton (weights) than they would be by using no pretraining or short integration?

Q9) What would the comparison in figure 2 look like, if counting was also pretrained with “short“ sequences? From the graphic it might seem that simply pretraining with any task that has matching cyclicity is the core importance. Not the counting itself.

Q10) How does the integration pretrainig task “hone sub-skills“ more so than using the same task but with shorter sequences? And is this also true if integration pretraining was also half periodicity?

Q11) The question from weakness 7).

Q12) How exactly do the results support “cognitive flexibility“?

### Soundness
2

### Presentation
2

### Contribution
2
