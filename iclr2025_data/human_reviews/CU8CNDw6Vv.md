## Human Reviewer 1

### Summary
The paper investigates the validity and consistency of benchmarks used for evaluating methods that mitigate spurious correlations in machine learning models. Recognizing that current benchmarks often produce conflicting results—with certain methods performing well on one benchmark but poorly on others—the authors aim to understand the root of these disagreements. They propose three key desiderata for a valid spurious correlation benchmark: ERM (Empirical Risk Minimization) Failure, Discriminative Power, and Convergent Validity. To assess a benchmark’s validity, they introduce a model-dependent measure, the Bayes Factor (K), which quantifies task difficulty due to spurious correlation. Through an empirical study across multiple benchmarks, the paper identifies benchmarks that meet the proposed validity criteria and highlights methods that demonstrate robustness across varying benchmarks. Additionally, they offer practical recommendations for practitioners to choose benchmarks and methods tailored to their specific dataset characteristics, advocating for a systematic approach to benchmark selection in real-world applications.

### Strengths
- **Originality**: The paper presents a novel approach to evaluating spurious correlation benchmarks by proposing three validity criteria—ERM Failure, Discriminative Power, and Convergent Validity.
- **Quality**: The study is well-executed, with a thorough empirical analysis to assess the proposed validity criteria. The use of the Bayes Factor as a measure of task difficulty provides a quantifiable metric, helping to identify benchmark inconsistencies.
- **Clarity**: Definitions of key concepts, such as the three validity criteria, are well-explained. The practical recommendations provide actionable insights for researchers and practitioners selecting benchmarks.
- **Significance**: By focusing on the quality of benchmarks themselves, the paper addresses a critical gap in spurious correlation research. The findings could lead to improved benchmark selection practices, which are essential for evaluating and developing robust models across diverse domains.

### Weaknesses
The methods discussed in the paper currently omit some recent state-of-the-art algorithms and techniques in spurious correlation research published before July 1, 2024, which would strengthen both the related work and Section 4. For instance, 
- Wang et al. "On the Effect of Key Factors in Spurious Correlation." AISTATS 2024.
- Yang et al. "Identifying Spurious Biases Early in Training through the Lens of Simplicity Bias." AISTATS 2024.
- Lin et al. "Spurious Feature Diversification Improves Out-of-distribution Generalization." ICLR 2024.
- Deng et al. "Robust Learning with Progressive Data Expansion Against Spurious Correlation." NeurIPS 2023.
    
Including these and potentially other relevant studies would make the paper more up-to-date. Even if not directly compared in Section 4, these works should at least be cited and discussed to reflect the current advancements in the field.

### Questions
1. There is an unresolved comment left in Line 242: “Can we make x axis bigger? Too hard to read even zooming.” This appears to have been unintentionally included in the submitted version and should be removed.
2. In lines 312-313, the paper states that $M_{ERM}$ is trained using ERM, while in lines 866-867, it is mentioned that $M_{ERM}$ is trained using the Adam optimizer. Could the authors confirm which training method was used and clarify any potential discrepancies?
3. Given the emphasis on benchmark selection, do the authors have insights into how the choice of model architecture might impact the validity of a benchmark? Are certain models more or less suitable for assessing spurious correlation benchmarks under the proposed criteria of ERM Failure, Discriminative Power, and Convergent Validity?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper assesses the quality of spurious correlation benchmarks and methods. The paper first develops three criteria desired for spurious correlation benchmarks and checks whether these are satisfied by some commonly used benchmarks. They then check which methods perform well across different benchmarks, and develop a new recommendation for choosing which method to use for a given dataset and model.

### Strengths
The results provide insights both about which benchmarks are good indicators of mitigating spurious correlations and which methods are robust across different benchmarks, which can be useful to a variety of practitioners.

### Weaknesses
1. Calculating K requires two full training runs (one with ERM, one with reweighting). This is extremely resource-intensive, and the empirical results do not seem to show a significant enough improvement to warrant such a cost.

2. The variety of spurious correlation benchmarks is a problem that has been addressed in previous work (Joshi et al., 2023; Yang et al., 2023). A more detailed comparison of the observations in this work versus those in previous work would be appreciated.

3. Some parts of the paper could be reorganized for clarity. A few specific points
- unnecessary comments on line 242
- lack of a dedicated related works section that puts the paper in the context of existing research (see previous comment)
- the discussion section jumps between many topics that are only loosely related to each other and the main paper, making it hard to follow

### Questions
I wonder how some datasets can have negative K (i.e. reweighting decreases performance)? This seems indicative of some confounding factors other than the identified spurious correlation, which may hinder the validity of the experimental results.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper takes a critical look at a set of popular Spurious Correlation Benchmarks (SCBs), and shows that they often disagree with one another. The authors set out three desiderata that they think SCBs should exhibit based on the performance of different methods on the worse group accuracy. Specifically they claim a good SCB should exhibit a failure case of vanilla ERM, have strong Discriminative Power and Convergent Validity. The paper then evaluates how well these desiderata are satisfied by the set of SCB in question. The authors introduce a metric “K” to measure the difficulty of SCBs due to spurious correlations, After establishing a subset of the SCBs that satisfy the three desiderata, common domain generalisation approaches are assessed on this subset. Finally it is recommended practitioner also assess methods on these data sets or on data set similar in term of “K” to their data set of interest. In the Discussion section the authors discuss some weakness of their work and make some general recommendations for which SCB to use.

### Strengths
The papers main finding that Spurious Correlation Benchmarks (SCBs) often disagree with one another is interesting and definitely of interest.

The experiments performed are sound and presented in a clear and manner.

The prose of the paper are of good quality and in general it is easy to read and understand.

### Weaknesses
The biggest weakness of the paper is it is incorrectly titled, and the abstract is misleading. Spurious Correlations can be present outside of data sets with subpopulations shifts, however only subpopulation data sets and approaches have been considered in this work.  While the authors note this in the discussion section, I find this to still be insufficient. In its current state I think the work would be much better titled “REASSESSING THE VALIDITY OF SUBPOPULATION SHIFT BENCHMARKS”. With this title and a little rewriting to narrow the focus to these data sets and mitigation strategies I think the paper would be much better.

“Spurious Correlations” or “shortcuts” are typically defined as decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions, such as real-world scenarios (Geirhos R et al. 2020). This phenomena only requires a distribution shift between test and train environments. The link between group performance and spurious correlations is critically missing from the paper. 

This has the following issues:
1)The assumption of having access to group information limits the usefulness of the desiderata to subpopulation shift benchmarks. 
2) How data set are grouped into subpopulation would likely have a large impact on these desiderata, how robust the desiderata are to the merging or splitting of groups has not been explored. I would suspect that the desiderata would be very sensitive hence more detail here seems necessary.
3) There is no explanation of the different groups for the data sets in question, no detail on how the groups were selected. Or how to select useful groups when they have not be provided
4) Many (possibly all) of the mitigation strategies require group labels. Many Spurious Correlation mitigation strategies that don’t require group labels have not be considered, Feature Sieve, Deep feature reweighing, or the ensemble approach of (Teney et al 2022b) to name just a few.

All in all this paper just focuses on subpopulation shift benchmarks, hence the title and abstract and introduction should reflect that, and the effect of quality of the sub population labels should be explored.

The recipe for practitioners comparing mitigation methods on similar data sets in terms of K (Line 416-420), assumes access to data of the test domain to compute K. This requires the domain shift is known at train time.  This limits the usefulness of the approach as it assumes one has access to “clean” test data but insufficient to train on directly. 

**Typos:**

Line 242: - author comment left in

Line 140: ANother

Line 102: correctionS - should be single

**Refs**

Geirhos R, Jacobsen JH, Michaelis C, Zemel R, Brendel W, Bethge M, Wichmann FA. Shortcut learning in deep neural networks. Nature Machine Intelligence. 2020 Nov;2(11):665-73.

Hermann KL, Mobahi H, Fel T, Mozer MC. On the foundations of shortcut learning. arXiv preprint arXiv:2310.16228. 2023 Oct 24.

Damien Teney, Maxime Peyrard, and Ehsan Abbasnejad. Predicting is not understanding: Recognizing and addressing underspecification in machine learning. In European Conference on Computer Vision, pp. 458–476. Springer, 2022b.

### Questions
**Questions**

What are the groups for the data set you consider?
How would extend your desiderata to setting where you did not have group labels?
How sensitive are your desiderata to the merging of groups or splitting of groups?


**Suggestions**

This paper focuses on subpopulation shift benchmarks, hence the title and abstract and introduction should reflect that, and the effect of quality of the sub population labels should be explored.

In its current state I think this work would be much better titled “REASSESSING THE VALIDITY OF SUBPOPULATION SHIFT BENCHMARKS”. With this title and a little rewriting to narrow the focus to these data sets and mitigation strategies I think the paper would be much better.

The 3.4.1 1. Is commonly referred to as “predictivity” and 2, and 3 are know as the “availability” Hermann et al. 2024 It’s also not clear to me what the different between 2 and 3 are.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper investigates the problem of spurious correlations and the fact that results are inconsistent across benchmarks. It demonstrates that the top-performing methods on one benchmark may perform poorly on another, revealing significant benchmark disagreements. In particular, the authors show that some methods while achieving best on some benchmarks they perform among the bottom 3 on other benchmarks. To address this, the authors propose three desiderata for a valid benchmark: ERM Failure, Discriminative Power, and Convergent Validity. Their analysis shows that many benchmarks and mitigation methods fail to meet these criteria, questioning their effectiveness. The paper also provides guidance for selecting appropriate benchmarks based on specific tasks.

### Strengths
1. The paper's objectives and goals are clearly articulated.

2. The problem addressed is a longstanding challenge in machine learning, as defining spurious correlations and constructing relevant attributes is difficult. This paper delves deeply into the reasoning behind these challenges and explores the properties that datasets should possess to qualify for evaluating spurious correlations.

### Weaknesses
The paper seems to have been rushed for the submission. There are several errors, mistakes, typos, in addition to a comment left out by the authors regarding one of their figures that reads "Can we make x axis bigger? To hard to read even zooming" in line 242.

I will list below a non-exhaustive list:

1. Line 242 "Can we make x axis .. ".

2. Stay consistent "Figure" vs "fig" vs "Fig". It should always be capitalised "F" but at least stay consistent on the abbreviation or not.

3. Similarly to the above, also Appendix X, Figure Y, Table Z, Equation T all need to have the first letter capitalized.

4. Lines 202, 204 are missing extra spaces @ "Citybirdsshould" and "(AvP)has. There are a number of these

5. Figures are poorly presented. Figure 1 for instance, is hard to read (particularly figure 2). Make it bigger or change the presentation. The text is too small.

6. Caption of Figure 1 seems wrong. It reads "best method on Waterbirds (DFR) is the second worst on NICO++". DFR performs 19 (worst) on Waterbirds and second best on NICO++ according to Figure 2b.

7. "of its" > "to its" @ line 92.

8. Figure 4 is very poorly presented, xlabel, ylabels, and legends are all small.

9. Lots of white space in Figure 5. You can make it better. and enlarge the plots.


General weakness:

1. The paper focuses on image classification, which is to some extent an outdated setup and less exciting compared to newer domains.

2. All datasets require attributes, which is less realistic in real-world scenarios. The paper aims to provide a practical guide for practitioners deploying their models, but it is unlikely to encounter a test dataset where the attributes are known a priori.

3. The most interesting experiments are those presented in Table 2, as they provide evidence that filtering benchmarks based on the proposed desiderata helps in capturing and measuring spurious correlations. However, more experiments are needed. The details of this experiment section are poorly presented, and the rationale for selecting particular methods (GroupDRO and ReSample) is not adequately explained. It would also be valuable to investigate the same experiments using a different set of starting datasets to assess the impact of applying the desiderata on final performance.

### Questions
See above. I believe the paper is not yet ready for publication. The presentation is poor and the paper still needs to conduct a few experiments justifying the proposed desiderata in addition to better justify the experiments when attributes are actually required.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
4