# On the Adversarial Risk of Test Time Adaptation: An Investigation into Realistic Test-Time Data Poisoning

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 8

## Abstract
Test-time adaptation (TTA) updates the model weights during the inference stage using testing data to enhance generalization. However, this practice exposes TTA to adversarial risks. Existing studies have shown that when TTA is updated with crafted adversarial test samples, also known as test-time poisoned data, the performance on benign samples can deteriorate. Nonetheless, the perceived adversarial risk may be overstated if the poisoned data is generated under overly strong assumptions. In this work, we first review realistic assumptions for test-time data poisoning, including white-box versus grey-box attacks, access to benign data, attack order, and more. We then propose an effective and realistic attack method that better produces poisoned samples without access to benign samples, and derive an effective in-distribution attack objective. We also design two TTA-aware attack objectives. Our benchmarks of existing attack methods reveal that the TTA methods are more robust than previously believed. In addition, we analyze effective defense strategies to help develop adversarially robust TTA methods. The source code is available at https://github.com/Gorilla-Lab-SCUT/RTTDP.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores test-time data poisoning (TTDP) under realistic conditions, proposing a grey-box attack that reflects practical scenarios where adversaries lack full access to benign data and real-time model updates. The study introduces two attack objectives based on entropy manipulation to effectively poison data while remaining within realistic constraints. Through extensive experimentation on state-of-the-art test-time adaptation (TTA) methods, the authors assess the vulnerability of these methods and propose feature consistency regularization as a countermeasure. This work highlights the persistent adversarial risks in TTA setups under real-world scenarios.

### Strengths
- The study is well-written;
- It covers an interesting setting that deserve much more attention from the security aspect.

### Weaknesses
While the study is well-written and formally structured, there are several areas where clarity and methodological rigor could be improved.

- Distinction between RTTDP and TePA in Table 1. In Table 1, the difference between RTTDP and TePA, particularly in terms of online adaptability, is not clear. It would be beneficial to clarify whether TePA is prevented from online applicability, thus highlighting RTTDP’s novelty.

- Inconsistent comparison between poisoning and adversarial attacks. The paper compares the proposed poisoning attack with MaxCE, an adversarial example attack applied at test time rather than during training. This inclusion creates confusion about the setting being considered, especially as MaxCE’s performance is close to that of the proposed attacks, despite being a simpler, suboptimal approach. Additionally, this inconsistency leads to general confusion throughout the paper regarding the exact threat model, setting, and practical cases under consideration. The title’s emphasis on “Realistic” test-time data poisoning also adds to this confusion, as data poisoning traditionally pertains to training rather than test inference. To clarify, the authors should explain that, while poisoning data are collected during test inference, they are subsequently used to update the model through adaptation techniques, which occurs as an offline process. Better descriptions and visualizations of the threat model, attack setting, and relevant practical scenarios would improve clarity and accurately convey the scope and applicability of the proposed approach.

- Not impactful results. MaxCE’s performance, although designed as an adversarial example attack rather than a poisoning attack, remains close to that of the proposed attacks, which raises questions about the actual impact and effectiveness of the proposed methods. The results do not demonstrate a clear or significant advantage of the new attack over MaxCE, suggesting that the proposed methods may lack the robustness or advancement intended. Furthermore, more sophisticated adversarial example attacks (e.g., PGD, C&W, and AutoAttack). I recommend including comparisons with these advanced adversarial attacks to provide a clearer perspective on whether the proposed approach offers meaningful improvements. Currently, the lack of a notable performance gap reduces the impact of the results, leaving it uncertain whether the proposed attacks represent a substantial advancement.

- Unsupported and unclear methodology. The transition from a bilevel optimization formulation to a single-level optimization problem lacks theoretical clarity. Such transitions are complex, and previous research has shown limitations in achieving optimal results through this method. Expanding on this approach and its theoretical basis would improve the rigor of the paper’s claims.

- Confusing experimental setup. The design choices in the "Benchmarking Poisoning Methods" section lack clarity. The authors do not specify how these choices impact prior works or how they differ from established benchmarking methods, such as DIA and TePA. This lack of clarity makes it difficult to understand the setup and its implications for reproducibility.

- Missing details on attack queries. The number of queries used by each attack is a critical factor, yet the experimental comparisons lack this information. Since query efficiency is a key point raised by the authors, detailing query usage across attacks would be essential for a fair comparison.


**Minor Points:**

- In Equation 1, the authors include two Kullback-Leibler Divergence (KLD) terms. To clarify their necessity, the authors should explain why both terms are required rather than just one, even if asymmetry in KLD is a factor. Providing justification in the text would enhance methodological clarity.

- I suggest the authors break Equation 2 into multiple lines to improve readability.

- Expanding the background discussion presented in Section 2 and related work with further background knowledge on adversarial examples, and on white-box, grey-box, and black-box threat models in the context of poisoning. The authors could reference relevant surveys that would help to clarify their threat model and position at the SoA and facilitate non-expert readers understand these distinctions. 

- Figure 1 is overly technical and lacks clarity regarding the threat model and attacker strategy. Simplifying it could enhance its value as a conceptual overview.


**References**
- [AutoAttack] Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks.
- [PGD] Towards deep learning models resistant to adversarial attacks
- [C&W] Towards evaluating the robustness of neural networks

### Questions
How does your approach transition from a bilevel optimization problem to a single-level one, and could you provide additional theoretical support for this methodology?

Can you elaborate on how your benchmarking setup differs from prior works and how these design choices may influence your results?

Given that MaxCE is primarily an adversarial example attack, what was the rationale behind including it in your comparisons, and have you considered additional comparisons with more advanced adversarial example attacks like PGD or AutoAttack?

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
4

### Summary
The paper deals with data poisoning in the test time adaptation setting. The paper exposes some issues with existing poisoning attacks in this setting, and goes on to propose certain alterations/additions to existing poisoning objectives, and evaluates results on several datasets, with several test time adaptation methods.

### Strengths
* The experimentation is thorough, and generally well presented. 
* The authors do a great job of pointing out issues with existing poisoning attacks against TTA training. 
* The authors motivate and propose a poisoning scheme of their own, and show its effectiveness in several settings.

### Weaknesses
* The area of data poisoning in the TTA setting is a bit niche, and to be honest, doesn't seem to present many challenges beyond existing data poisoning settings. So called "availability" attacks have been introduced in [1,2], and several other works. These should at least be cited, and probably compared against as there's a decent amount of overlap. 
* The real "value add" for the authors, in my view, is the addition of the feature clustering regularizer. The other contributions (BLE, notch loss, etc.) seem to be very slight modifications of existing poisoning attacks, or even just existing adversarial attacks. 
* The related work shouldn't be at the end. It would be very helpful for readers to have some more info on TTA at the beginning of the work. But if you're going to keep it at the end, you NEED to cite things earlier, as it's totally unclear what things like TENT, RPL, etc. are when they're introduced. It seems like in section 4.2, only the acronyms are introduced, with no explanation, and no citation to click on and find more information. Note: I didn't deduct any "points" for this weakness, but it really needs to be addressed. 

[1] Huang, Hanxun, et al. "Unlearnable examples: Making personal data unexploitable." arXiv preprint arXiv:2101.04898 (2021).

[2] Fowl, Liam, et al. "Adversarial examples make strong poisons." Advances in Neural Information Processing Systems 34 (2021): 30339-30351.

### Questions
* Does this only work against losses $\mathcal{L}_{tta}$ that are unsupervised? It would be nice to explain this a bit more and give some examples of what this loss function looks like in the main body. 
* Do you ever specify the attack budget? I couldn't find it in the paper. In A.2, you define this quantity, $r$, but I don't ever see details for it. Is this the same thing as $b$ in Table 8? 
* What are the "Source" numbers listed in Tables 2,3,4? Is this just the success of standard adversarial attacks?

### Soundness
3

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
This paper examines the adversarial risks in test-time adaptation (TTA), highlighting that TTA’s exposure to test-time data poisoning may be less severe under realistic attack assumptions. The authors propose two TTA-aware attack objectives and a new in-distribution poisoning method that operates without access to benign data, revealing that TTA methods show greater robustness than expected and identifying defense strategies to enhance TTA’s adversarial resilience.

### Strengths
1. The paper is overall well-written with clear math definitions and extensive experiments and studies an important problem of data poisoning under weaker/realistic assumptions. 
2. The paper studies the popular test-time attack problem from a novel perspective of weaker threat model.

### Weaknesses
1. Line 680, citation format wrong. Missing conference name.
2. Figure 1 is helpful but visually confusing and ovewhelming. Please explain what B_a, B_ab, B_t, B_b in the figure description section. A term (-\lambda L_{reg} and L_{atk}) is confusing to put with the full equation . Also, it is better to present the attack objective in separate part than to fit it in Figure 1.
3. It will be more helpful to have a graph that assign the current popular attack methods into different buckets, where each bucket has different threat model.

### Questions
1. Has the author investigate the effect of the ratio of between the adversarial example and the benign examples? 
2. Did the author experiment and compare with harder attack like in [1] that requires access to benign examples. Or is there quantitative measurement on the tradeoff when relaxing to realistic attacks. 
3. Is it possible to derive any formal guarantee on the attack effectiveness when relaxing the different constraints?



[1]. Chen, J., Wu, X., Guo, Y., Liang, Y., and Jha, S. Towards evaluating the robustness of neural networks learned by transduction. In Int. Conf. Learn. Represent., 2022

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates adversarial risks in test-time adaptation (TTA), which updates model weights during inference to counter distribution shifts. The authors argue that the assumed threat model of prior work is unrealistic and gives the attacker too much access. They propose a more restricted attack model called Realistic Test-Time Data Poisoning (RTTDP), assuming grey-box access to the model and no access to the benign samples of other users. The authors propose a new attack method outperforming previous work in this new setting.

### Strengths
**More Realistic Threat-Model**

The paper critically examines the assumptions made in prior work regarding the threat model of poisoning attacks against TTA systems, particularly regarding the adversary’s capabilities. In particular, the relaxation of (i) white-box access to gray-box access and (ii) restricting access to the targeted samples makes the setting much more realistic and, therefore, the findings more significant and relevant.

**A New Attack Rooted in Theoretical Insights**

The proposed method cleverly extends on previous attacks, making it effective in the new, more restricted attack setting. The method is rooted in theoretical insights that nicely explain the various parts and modifications. It demonstrates that TTA is still vulnerable to poisoning attacks even with more restrictive assumptions.

**Thorough Evaluation**

The experimental evaluation is thorough. It spans three different datasets with increasing complexity (CIFAR10-C, CIFAR1-C, ImageNet-C), multiple different TTA methods, and compares against two different baselines. The evaluation also includes an ablation study, as well as first look at potential defenses, which I appreciate. The defenses show promise but are unable to fully recover the functions of the original models.

### Weaknesses
**Improved Assumptions Still Strong**

While the gray-box assumption is a significant improvement over the white-box assumption in some prior work, it still assumes the adversary's access to the original model (before TTA) and data from the same distribution as the victim benign user. Both of these assumptions are fairly strong and could ideally be relaxed further to, e.g., black-box access.

**Unclear Details**

Several important details of the method remain unclear and should be included in the paper. The most important ones include: why repeated querying of the model is forbidden (L181), how the surrogate model can be trained without querying (Section 3.1), why we can assume to know $\mathcal{B}_{a, t-1}$, and how “Uniform” and “Non-Uniform” attacks are executed. Please refer to “Questions” for more details and additional comments.

**The Presentation Can Be Improved**

The method’s presentation could be improved in multiple ways:

- Figure 1 gives a good overview of the method. It could benefit from some simplifications to reduce visual clutter; see “Suggestions for Improvements” for detailed suggestions.
- Figure 2 is very small and should be enlarged. The four subplots are also unrelated and referred to from different parts of the paper. I suggest splitting it into individual figures and moving those to the corresponding sections.
- In general, the paper’s English is easy to read. However, some parts are hard to understand due to grammatically wrong or overly complex sentences (e.g. L431, L259, …). The paper would benefit significantly from a careful revision, possibly with an English language tool.

**No Code Available**

No code was available for review, and the authors did not specify whether it will be released upon publication.

### Questions
**Questions**

L181: Why is repeated querying prohibited? I understand that the model is constantly updated throughout the querying, but is this really a problem?

Section 3.1: How exactly is the model distilled? L181 says repeated querying is prohibited, but my understanding is that training a surrogate model requires precisely such querying? How can you query for a single $\theta_t$ if it changes by the very fact that it is being queried?

L232/233: How can we assume to know $\mathcal{B}_{a, t-1}$? Other users could have queried the model an unknown number of times between training the surrogate and launching the attack.

Table 2-4: What do “Uniform” and “Non-Uniform” Attack Freq. refer to? I could not find this mentioned anywhere in the paper. The appendix has a figure, but also no sufficient explanation. Since this is a central part of the main results tables, it should be carefully explained in the main part of the paper.

Table 2-4: What does “Source” refer to? this should also be explained.

L259/260: I don’t understand what this sentence means. What is “the optimized”, what is the “optimizing objective”, what is the “problem”?

L431: I do not understand this sentence. Can you please rephrase?

L149/150: “the attacker is assumed to have no access to real-time model weights.” By whom is this assumed?

Table 1: what does “Offline” attack order refer to?

**Suggestions for Improvement**

Figure 1: The overview figure 1 gives over the method is great! I appreciate the difficulty in illustrating the complex system with many aspects. I would like to make some suggestions to simplify the figure, as it took me a long time to work through it, and I believe it would be much more helpful with slightly less detail that distracts from the core parts:

- I suggest entirely removing the two boxes “TTA Model” and “Attack Objectives”. They don’t add much information over the text, and it would allow the viewer to concentrate on the three parties in the system: the adversary, the benign user, and the TTA server.
- Removing the symbols for the attacker and user would further reduce visual clutter, and they are redundant to the titles of the boxes.
- I would stylize the images from the distributions - e.g. squares of one color per distribution - instead of using actual images from the dataset. This has several advantages: (i) it reduces visual clutter, (ii) it makes it immediately apparent which distribution they belong to, (iii) the “poison” symbol becomes more apparent (it took me a while to see the little devils).

There may be more opportunities for improvements, e.g., thicker lines for arrows, but I believe the three points above will already make the figure significantly easier to understand.

Figure 2: Please consider breaking these plots into individual figures.

- They are currently way too small, to the point where labels and points are barely readable when printed. At least doubling their size would be required.
- The several subplots are unrelated and even referenced from different sections of the paper. It would be much more helpful to have the figure close to the text that it refers to.

L298: You refer to Fig 2(c), which contains many acronyms that have not been introduced. E.g., TENT, EATA, SAR, NHE, BLE, …

Tables 5-7: These tables would be easier to read without the grey shading.

### Soundness
3

### Presentation
3

### Contribution
3
