## Human Reviewer 1

### Summary
This paper re-analyzes the ICLR 2025 paper _“Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs”_, arguing that its main claims about min-p sampling’s superiority are unsupported. The authors identify overstatements in the original work’s human evaluations, benchmark tests, and reporting practices. Using this case study, they propose a blueprint for improving rigor and transparency in empirical ML research.

### Strengths
- This paper conducts a detailed and transparent re-examination of a prior work, carefully identifying exaggerations and verifying claims through rigorous statistical and experimental checks.
- The discussion section provides thoughtful and necessary guidance for improving rigor and transparency in scientific research, offering lessons that are broadly applicable across scientific disciplines.

### Weaknesses
- Although the authors claim “From this case study, we derive a blueprint for more rigorous research,” the proposed blueprint only appears briefly in the final discussion section (less than one page). This portion is disproportionately small relative to the claimed contribution and lacks the depth or generalization needed to stand as a substantive methodological advance.

- While the manuscript provides an important and well-executed re-analysis of a previous study, it primarily functions as a reproducibility commentary rather than a novel empirical or theoretical contribution. I recognize its educational and community value, but it does not clearly fit within any of the main ICLR paper categories. The authors might consider submitting it as a position paper or workshop contribution, where its insights on research rigor and transparency would be more appropriately framed.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper presents a detailed case study of min-p, which is a high-visibility paper published in ICLR 2025. The authors have conducted a lot of rigorous experiments and found the results claimed from the min-p no longer hold. The authors thus derive a blueprint for more rigorous empirical ML research. This paper very much reminds me of one of the ML reproducibility challenge papers.

### Strengths
- This paper critically examines the validity of empirical machine learning research through carefully designed statistical tests and rigorous experimental design. It offers valuable insights for the community and serves as a warning about the noise and inconsistency in current ML research reporting.

- The paper is clearly written and presented very nicely.

### Weaknesses
- This paper could have a greater impact if it positioned itself as a position paper outlining guidelines for standard practices in rigorous ML experimentation. At present, its conclusions are drawn from a single case study, and their external validity depends on whether Min-P is truly representative of broader systemic issues.

### Questions
- Regarding 3.1, I get that when controlling for the hyper-parameter tuning budget, there is no distinction between min-p and other sampling methods. Did you run experiments where all the samplers are fully tuned? Does min-p have an advantage in this case?

- The authors are questioning the community adoption. While the original paper may have exaggerated, min-p seems to be well accepted by the community as I can see. Is there further community evidence that does not support min-p’s claim of superiority?

### Soundness
2

### Presentation
3

### Contribution
1

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
This work challenged the experimental results of the min-p method (ICLR 2025 oral), claiming that most of its conclusions are invalid upon closer examination. The paper ends with a list of general lessons for conducting more rigorous empirical ML studies.

### Strengths
- A comprehensive re-examination of a seemingly popular method for token sampling in LLMs

- While most of the key lessons pointed by the authors are not new per se in established guidelines on reproducibility, it is nonetheless a good case study to draw more attention to the current questionable practices in ML research

### Weaknesses
- This work seems to be an odd fit for ICLR, as there is no new result or particularly novel insights. Perhaps the authors should consider submitting this work to the ML reproducibility challenge. 

- Some of the claims (eg., S2.1, S2.4, S4.3), including private communications to the original authors of min-p, are difficult to verify for the reviewers. 

- Since the work is mostly focused on criticizing one particular paper, it feels only just if the original authors were given a chance to respond before publication. However, I do not see how the existing ICLR protocol could accommodate this exchange. 

- I agree with the authors that there is a crisis of rigor in empirical ML research. This is a widespread problem in the entire community (sometimes caused by realistic constraints such as resource). However, I do not think it is a good idea to "go after" one particular paper or group. This work would be much more useful and impactful if it is organized by the current ill practices in empirical ML (like the lessons that the authors listed on Line 28-30), with each lesson illustrated on a **different** high profile paper. We need to distinguish criticizing a paper/group (more subjective and not interesting to the community) from criticizing a practice (more objective and interesting to everyone).

### Questions
Let me be clear: I applaud the authors' efforts and I agree with the listed lessons (though I can't say any of them is really new or unexpected). I just do not think ICLR is the right place for such efforts, and I prefer not to single out any paper or group when the underlying problem is widespread. 

The disclaimer on Line 456 (new evidence might lead to different conclusions) makes me feel even more uneasy, if the original authors of min-p were not given a chance to respond. But I don't know how this could be accommodated.

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
4