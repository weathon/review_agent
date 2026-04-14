# Conformalized Interactive Imitation Learning: Handling Expert Shift and Intermittent Feedback

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
In interactive imitation learning (IL), uncertainty quantification offers a way for the learner (i.e. robot) to contend with distribution shifts encountered during deployment by actively seeking additional feedback from an expert (i.e. human) online. Prior works use mechanisms like ensemble disagreement or Monte Carlo dropout to quantify when black-box IL policies are uncertain; however, these approaches can lead to overconfident estimates when faced with deployment-time distribution shifts. Instead, we contend that we need uncertainty quantification algorithms that can leverage the expert human feedback received during deployment time to adapt the robot's uncertainty online. To tackle this, we draw upon online conformal prediction, a distribution-free method for constructing prediction intervals online given a stream of ground-truth labels. Human labels, however, are intermittent in the interactive IL setting. Thus, from the conformal prediction side, we introduce a novel uncertainty quantification algorithm called intermittent quantile tracking (IQT) that leverages a probabilistic model of intermittent labels, maintains asymptotic coverage guarantees, and empirically achieves desired coverage levels. From the interactive IL side, we develop ConformalDAgger, a new approach wherein the robot uses prediction intervals calibrated by IQT as a reliable measure of deployment-time uncertainty to actively query for more expert feedback. We compare ConformalDAgger to prior uncertainty-aware DAgger methods in scenarios where the distribution shift is (and isn't) present because of changes in the expert's policy. We find that in simulated and hardware deployments on a 7DOF robotic manipulator, ConformalDAgger detects high uncertainty when the expert shifts and increases the number of interventions compared to baselines, allowing the robot to more quickly learn the new behavior.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an interactive imitation learning method (ConformalDAgger) that has the learner (robot) actively ask for additional supervision when it is uncertain during deployment. As in prior methods, it uses uncertainty quantification to do this, but unlike other methods, it updates the uncertainty quantification online using the expert feedback during deployment. The paper does this by introducing a novel online conformal prediction method called IQT, which is able to deal with intermittent expert labels, unlike prior approaches in online conformal prediction. The proposed method is shown to be effective at dealing with situations where there is Expert Shift, i.e. the expert policy changes between train and test time.

### Strengths
The paper is well-written and mostly easy to follow. The methods presented in the paper are interesting and appear to be novel. There is empirical evidence showing the ability of the proposed method to deal with "Expert Shift" in both simulated and real-world experiments.

### Weaknesses
**Weak and Unclear Motivation.** The proposed method seems to be exclusively useful for the setting of "Expert Shift" where the expert policy changes between train-time and test-time. This does not seem to be a very practical, commonplace issue. Are there other motivating examples for the proposed method as well, and corresponding settings where the proposed method would outperform other baselines? This aligns with Takeaway 3 in Section 5 too, where performance is similar to the baseline in the absence of expert shift. On a related note, a change in the expert policy can be arbitrary - it would be useful to provide some more intuition on what the best behavior should be for an ideal method in sensing and dealing with such a shift.

**Thoroughness and Reproducibility of Experiments.** The paper could benefit from some more thorough simulation experiments. For example, including a simulation experiment with high-dimensional observations (e.g. images) similar to the real-world experiments could be beneficial. Also running such experiments on some standard benchmark tasks in existing simulation frameworks that go beyond goal-reaching would greatly improve the reproducibility of the work and the experiments. Some suggestions include [robomimic](https://robomimic.github.io/), [RLBench](https://sites.google.com/view/rlbench), [ManiSkill](https://www.maniskill.ai/home), or some of the tasks presented in [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/). Also, the paper only compares against EnsembleDAgger but it seems like there are other suitable baselines that could also be appropriate, such as [ThriftyDAgger](https://sites.google.com/view/thrifty-dagger/home).

**Other Issues.**

- Algorithm 1, line 11 is potentially misleading. The expert is only selectively annotating states unlike DAgger where all states observed by the agent would be relabeled. 
- several undefined Sec refs (line 154, line 212, line 252, line 471)

### Questions
- IQT is only applied to the end effector position (bottom of page 9) -- is this a limitation? There might be robot manipulation tasks that require more rotation-heavy actions.
- In the hardware robot experiment, why is the EnsembleDAgger baseline different from the one used in the simulation experiments?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose ConformalDAgger that uses runtime prediction interval to actively query expert’s feedback in the context of imitation learning for robot manipulation. ConformalDAgger introduces Intermittent Quantile Tracking, an uncertainty quantification method that provably minimizes regret about error of following expert’s behavior based on Angelopoulos et al., 2024, but also extends it to situations where the feedback is intermittent. Robot asks for human feedback whenever the prediction interval size exceeds some defined threshold, and the threshold also affects the regret guarantee. ConformalDAgger is validated on both typical time-series prediction settings and also robot imitation learning with human correction, showing improved coverage of expert behavior when the expert behavior distribution shifts, compared to EnsembleDAgger, a Bayesian uncertainty quantification method that can be miscalibrated.

### Strengths
The paper addresses a very important and timely problem in training robot manipulation policies — simple behavior cloning suffers from covariate shift, and rigorously quantifying the policy’s uncertainty and using it to query additional feedback from humans are an intuitive and potentially effective way to address covariate shift. Proper UQ ensures strong coverage while minimizing the amount of feedback needed.

ConformalDAgger is based on the online conformal prediction theory, which provides rigorous guarantees for the policy error, albeit being a regret-based one. The proposed extension to intermittent feedback setting is very novel.

ConformalDAgger shows clear advantage in coverage compared to EnsembleDAgger, which can be miscalibrated (although I have concerns about the setup of the baseline below). I also appreciate the experiments in the appendix varying p_t.

The paper is very well-written. Most experimental details are presented in the main text. All the illustrations are nicely done and convey the central ideas and experimental results cleanly.

### Weaknesses
I have mainly two concerns about the paper and proposed approach. The first one is about the experiment setup comparing ConformalDAgger to EnsembleDAgger. There is no discussion about how some of the hyperparameters, e.g., uncertainty threshold and safety classifier threshold used in EnsembleDAgger, are chosen in practice. I understand that these numbers can be difficult to choose in a rigorous manner, but some discussion will be greatly appreciated. Also, to strengthen the overall results, I suggest trying out a few different sets of thresholds, or even sweeping them, and comparing the two methods across the choice of these parameters. My understanding is that, with rigorous online conformal prediction theory, some of these parameters in ConformalDAgger will be easier to tune, and it would be good to discuss this too (or not if I misunderstand). 

My second concern lies in the overall approach. IQT is a nice theory and provides guarantees, but in practice, if the feedback rate (p_t) is not high, which probably should be the case given human intervention can be tedious, the regret guarantee can be quite loose. The relationship between interval size and robot asking for help also confuses me: if robot doesn’t ask for help, the prediction interval is not updated since there is no new information, and intuitively this leads to less information for the robot deciding whether to ask for help. From my intuition, it makes sense to consider additional contextual information to adjust the prediction set size. I am curious how the authors think of this concern.

### Questions
My question for the rebuttal is about the choice of hyperparameters. Can the authors comment on the choices in Sec. 5 and Sec. 6? If it is not possible to run new experiments, what would the effect of varying the threshold and temperature on the comparison to EnsembleDAgger look like?

I also suggest adding the training details of EnsembleDAgger and e.g., how the action variance is computed, in the appendix.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Interactive imitation learning with queryable experts (e.g. Dagger) are a powerful approach to robot learning from demonstration. Unlike supervised-based offline approaches like behavioral cloning, interactive IL can avoid the distribution shift between the training distribution (i.e. the expert’s state distribution) and the test-time distribution (i.e. the learner’s state distribution). Prior work in uncertainty quantification-based IL has developed methods of actively seeking feedback from the expert (robot-gated), in contrast to the original Dagger-based approach, where the expert dictated the intervention (human-gated). The paper proposes leveraging online conformal prediction as a method for uncertainty quantification during deployment time. The paper extends quantile tracking to the intermittent feedback setting, which is what is encountered in the interactive IL setting the paper considers for robot learning. Using the proposed intermittent quantile tracking (IQT), the paper proposes a new approach to uncertainty quantification-based IL that uses IQT’s prediction intervals and evaluates on simulation and physical robot tasks.

### Strengths
The paper motivates the problem well and presents an improvement to existing uncertainty-aware interactive IL methods (specifically Dagger-based) that incorporates uncertainty estimates in deployment time. The figures are well-done, and the experimental setup is mostly understandable (see points below).

### Weaknesses
Overall, the limitations in the paper’s experimental results are the most significant weaknesses of the paper, which I discuss in detail below.

The paper only compares to one baseline, and they do not provide justification for why other baselines are not considered. Either incorporating additional baselines or providing a discussion of why other Dagger-based or Monte Carlo-based (e.g. Cui et al. 2019) methods were not considered and would strengthen the empirical results.

Continuing with the limitations of the baselines, in the Hardware Experiments section (Sec. 6), the paper implements EnsembleDagger without a safety classifier. Rather than train a safety classifier to detect dissimilarity between the expert’s actions and the robot’s actions, the diffusion policy is sampled 3 times in order to “capture the notion of epistemic uncertainty.” The justification for this decision is unclear. Is replacing the safety constraint with multiple samples from the diffusion policy (which will be stochastic) better than implementing a safety classifier? Or is implementing a safety classifier unfeasible? (Additionally, I was unable to find the relevant discussion of an *implicit* EnsembleDagger-type approach in the referenced Wolleb et al. 2022.)

There are no details of the parameter tuning that was done to select EnsembleDagger’s uncertainty threshold. This is quite an important parameter that can affect its performance. EnsembleDagger is the only baseline compared against, so the lack of discussion of the uncertainty parameter undermines the relevance of the baseline. 

More broadly, the empirical results only consider distribution shift due to expert shift. The paper does not test on environments with environment shift (e.g. changing dynamics). 

On a different but important dimension, Dagger is also known to be a resource-intensive training procedure, requiring a queryable expert. The paper currently lacks a discussion of the computational efficiency of ConformalDagger and how it compares to EnsembleDagger. (e.g. measuring the wall-clock time to establish a comparison, discussing the IQT procedure’s efficiency when integrated into Algorithm 1, etc). 

As a further extension, the original Dagger is known to fail in the setting where the expert policy is unrealizable. Currently, the paper has no discussion on what setting they consider with respect to the expert policy and the learner’s policy class, and whether the ConformalDagger approach would fare better or worse than prior work in the misspecified setting. 

**Minor Comments:**
- Section 3 would benefit from further explanation (e.g. explaining the significance of Proposition 1) would improve the clarity of the section. 
- Appendix section numbers are listed as ???.

### Questions
The phrasing of Line 7 in Algorithm 1 is confusing, as it suggests two of the exact same samplings.

### Soundness
2

### Presentation
3

### Contribution
2
