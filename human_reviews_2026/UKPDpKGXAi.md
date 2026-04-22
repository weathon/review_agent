# A widely used protocol for EEG classification experiments leads to a confound

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 0, 4, 2

## Abstract
A temporal confound has been previously reported in a (now widely
used) dataset that others have tried to suggest is nonetheless
justified.  Despite attempts to make the community aware of this
confound, a significant number of publications continue to use the
confounded dataset, thereby drawing unsupported conclusions.  We
present a new experiment that conclusively demonstrates that the
identified confound in the dataset cannot be explained away by
recourse to factors such as block design, session duration, number of
subjects, or pooling multiple subjects.  We advise caution when
designing, conducting, and interpreting the results of experiments
that use this problematic protocol.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on temporal drift as a confound or 'embedded clock' in prominent EEG experimental designs. In particular, correlation of stimulus class with experiment timing. Although corrected experimental designs have been suggested, Palazzo et al. 2020 and 2024 claim the original confound is acceptable on several bases. The current paper rebuts this claim and highlights that the confounder remains.

### Strengths
The paper is well motivated and highlights an important problem in the literature (as well as problem in the peer-review system itself). The misapplication and misinterpretation of EEG data may also carry serious ethical concerns, e.g., if deployed in critical healthcare settings. For these reasons, it is very important that all such experimental designs are well-reasoned and understood.

### Weaknesses
I found the presentation and writing to be quite poor. On pages 2 and 3, there are huge numbers of in-text citations. It would be better to put this list in the appendix, for example, and save the limited real estate of the conference paper for more important points. On this point, the paper itself comes in at under 5 pages, with no appendices. This itself is not a showstopper for me: some of the best papers are short. However, if the quantity of content is limited, I expect the content in the main text to be information-dense and of very high quality. I do not find that it is, both in presentation and content.

In particular:
1. The paper primarily replicates and extends prior refutation studies with minor methodological variations, offering limited new theoretical or analytical insight.
2. Only six subjects were tested, which constrains generalisability and statistical power, especially given EEG’s high inter-subject variability.
3. While the study controls the temporal confound, other possible sources of bias (e.g., fatigue, attention drift, order effects beyond class blocks) are acknowledged but not experimentally tested.

### Questions
1. What anonymised information about the human participants did you record?
2. What informed consent and data protection procedures did you put in place?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper contains similar elements as another paper submitted to this conference.

### Strengths
None

### Weaknesses
A similar paper has been submitted to this conference.

### Questions
none

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a replication experiment demonstrating that a widely used block-design EEG classification protocol induces a temporal confound, linking the downstream class with time. It shows that classification accuracy depends heavily on this confound rather than true neural decoding. The paper critiques the continued use of flawed datasets and protocols in the field.

### Strengths
The concern identified in about temporal confounds in EEG classification experiments is apt and essential to communicate to the scientific community, because such confounds fundamentally undermine the validity of a large body of published EEG decoding research, potentially misguiding future methods development and applications.

Addressing this issue is critical for ensuring methodological rigor, truthful interpretation of results, and the ethical advancement of EEG-based neuroscience and brain-computer interface research.

The work is a replication of prior protocol with controlled manipulation of stimulus block order across multiple subjects, providing a thorough critique of defending arguments for block design, short sessions, and pooling subjects.

### Weaknesses
The language is generally direct but sometimes overly assertive in tone, bordering on confrontational regarding the ongoing use of flawed datasets in the community.

### Questions
Why have the authors not included similar analyses directly on the original datasets from Spampinato et al. or other widely used confounded datasets, to further validate the generalizability of their findings?

### Soundness
3

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
