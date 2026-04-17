# Randomized trials in EEG classification experiments are mandatory as   drift is pervasive

- Decision: Reject
- Scores: 0, 4, 2, 2

## Abstract
Temporal correlation is demonstrated in three public EEG datasets.
Filtering does not remove the correlation.  The community is cautioned
that a substantial number of recent publications are flawed due to a
confound between temporal correlation and stimulus class, despite
claims that filtering removes the correlation.  The only known way to
avoid this confound is through randomization.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper adresses a temporal correlation in a publicly available dataset, which leads to a confound in classification.  To demonstrate the confound, the authors repurposed the public dataset to argument that without this confound the classification results on this dataset have lower performance.  The authors gave an overview of articles using this dataset, that may have been affected by this confound.

### Strengths
- The paper gives an overview of the articles where the supposedly affected dataset has been used.  
- The paper explains a method which should illustrate that without the temporal correlation, the classification results are

### Weaknesses
- The nature of the confound is not clearly presented.  The authors state this in the first 2 sentences in the first paragraph of section 3, but this is the key to this article. Yet, the nature of the confound is only described in 2 sentences.  This creates little convincing value in their case.  The authors could use other references to experimental designs where the confound did not appear.  
- The authors put evidence of the temporal correlation by repurposing three other datasets towards the design of Spampinato et al, but I question if this is really representative.  The three datasets used in the article do not have a visual attention task, hence the question is if the LSTM network and the EEGChannelNet classifier is well suited for this data. 

- The authors did not clearly explain why the filtering method would decrease the temporal correlation.  

- The paper focusses on the flaw in the article but lacks a good practices statement or lessons learned discussion on how to avoid these confounds in later studies.

- The authors also tried to place an analogy with radio broadcast, but there it is not clear what the link is with EEG signals.  The paper lacks an small and consice overview on how to remove temporal correlation.

### Questions
- Why was the bandpass filtering between 14 and 71 Hz chosen? 

- What is the effect of the post hoc filtereing?  Is the temporal correlation decreased or has it remained the same?

- Did the authors try other means to demonstrate this temporal correlation?.

- In the discussion section, the authors claim that bandpass filtering dos not remove slow spectral change, but what do the authors mean by slow.  Can they give a time estimation?  Do you mean slow spectral change outside of the bandpass limits of the filter?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript highlights a critical methodological limitation pervasive in the EEG classification literature, as typified by Spampinato et al. (2017). The authors contend that inherent temporal drift in EEG recordings, when combined with block-design protocols that are inadequately randomized, produces spurious correlations between stimulus categories and the temporal structure of the experiment. This confounding factor fundamentally undermines the interpretability of results and casts doubt on the validity of conclusions drawn in nearly one hundred related publications.

### Strengths
- This paper demonstrated temporal correlations among three distinct public EEG datasets: Delorme (2020), Hatlestad-Hall et al. (2021), and Babayan et al. (2019). This supports the perspective that temporal correlations are widely present in EEG data.
- The results clearly demonstrate that bandpass filtering (14–71 Hz) does not remove the temporal correlation. Even with filtering, the classification accuracy obtained by both the LSTM and EEGChannelNet classifiers remains far above chance, refuting previous claims (Claim II) that filtering could remove the temporal correlation.
- The paper offers the novel and significant claim that the collection protocol used in studies like Spampinato et al. (2017) is inherently and irreparably confounded.

### Weaknesses
- The authors state explicitly that they do not claim the presence of confounds in the original studies corresponding to the datasets analyzed. Their approach instead repurposes three publicly available EEG datasets, comprising two resting-state and one auditory dataset, in order to reproduce the confounded block-design structure reported by Spampinato et al. (2017).
- The consistent alignment between training and test loss or accuracy observed throughout training is presented as strong evidence for the existence of label leakage.
- The high classification accuracy obtained, nearly perfect without filtering and substantially exceeding chance levels even after filtering, suggests that the classifiers examined, namely LSTM and EEGChannelNet, primarily exploit temporal confounds rather than encoding stimulus-related neural activity.

### Questions
- Q1. Since the data from all participants were combined before being divided into training and test sets, has any analysis been conducted, or is supporting evidence available, that demonstrates the presence of temporal correlation when subject pooling is avoided? Furthermore, is there empirical verification that the confound mechanism generalizes across individual subjects or recording sessions?
- Q2. This study provides clear evidence that applying a 14–71 Hz bandpass filter does not effectively mitigate the issue. In light of the electrical engineering principle concerning filtering and modulation cited in the paper, were alternative high-pass or band-stop filter thresholds tested to determine the precise conditions under which temporal correlation remains and cannot be eliminated after processing?
- Q3. The findings indicate that after filtering, EEGChannelNet achieves substantially higher classification accuracy than LSTM, leading to the interpretation that EEGChannelNet is even more sensitive to the confound. What analyses or theoretical considerations account for the greater susceptibility of EEGChannelNet compared to LSTM to the temporal confound under filtered conditions?

### Soundness
3

### Presentation
2

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
This paper analyzes three public EEG datasets repurposed with simulated block-design visual stimulus trials, showing pervasive temporal correlations that cause confounds in classification. It finds that filtering does not remove these correlations and concludes that only randomized trial designs can avoid such confounds.

### Strengths
The concern identified in about temporal confounds in EEG classification experiments is apt and essential to communicate to the scientific community, because such confounds fundamentally undermine the validity of a large body of published EEG decoding research, potentially misguiding future methods development and applications. 

Addressing this issue is critical for ensuring methodological rigor, truthful interpretation of results, and the ethical advancement of EEG-based neuroscience and brain-computer interface research.

### Weaknesses
The language is generally direct but sometimes overly assertive in tone, bordering on confrontational regarding the ongoing use of flawed datasets in the community.

Detailed methodology for data repurposing and subject simulation could be clearer, possibly aided by schematics. The current method description is difficult to follow, requiring a line-by-line breakdown.

Validation would be stronger with direct experiments or replication on actual randomized visual EEG data rather than entirely simulated repurposed datasets.

### Questions
It would be important for the readers to understand how representative the simulated resting-state and auditory EEG data are for visual stimulus experiments. A discussion or motivation could be helpful to clarify this.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
There seems to be a strong similarity between submissions 4793 and 4802. Both submissions point out a weakness of the experimental design of prior work studying visual stimuli decoding from EEG. While this reviewer agrees that randomized stimuli designs are critical in data collection to reduce the impact of confounding factors like temporal drifts often encountered in biosignals, it seems very likely that both articles originate from the same authors.

### Strengths
N/A

### Weaknesses
N/A

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
1
