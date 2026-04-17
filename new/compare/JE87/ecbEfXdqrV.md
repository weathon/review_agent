# Review

## Summary
The paper proposes a new definition for the counterintuitive phenomenon in anomaly detection using normalizing flows, which has been previously reported in the image domain but not extensively studied in tabular data. The authors conduct a systematic evaluation across 47 tabular datasets, demonstrating that this phenomenon is rare in tabular data compared to image data. They provide theoretical and empirical analyses, focusing on the roles of data dimensionality and feature correlation, to explain why normalizing flows are effective for anomaly detection in tabular domains.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The paper is well-organized, with a logical flow from problem formulation to theoretical analysis and empirical evaluation.
- The authors provide a thorough empirical evaluation across 47 tabular datasets and 10 image datasets, comparing 12 baseline models.
- The authors provide a theoretical analysis explaining why the counterintuitive phenomenon is less frequent in tabular data, linking it to differences in dimensionality and feature correlation.

## Weaknesses
- The definition of the counterintuitive phenomenon is somewhat rigid and may not capture the full spectrum of possible cases. The authors might consider incorporating a more nuanced approach to account for various dataset-specific characteristics and model architectures.
- The paper could benefit from a more detailed discussion on the practical implications of the findings, particularly how they can inform the development of improved anomaly detection methods in tabular domains.
- The paper does not address how the counterintuitive phenomenon might be mitigated or solved in the future work.

## Questions
- Could the authors provide more details on the choice of comparison models used in the empirical evaluation? Specifically, why were these 12 models selected and how do they represent the state-of-the-art in tabular anomaly detection?
- The authors propose a new definition for the counterintuitive phenomenon. Could they elaborate on how this definition compares to previous observations made in the literature? Are there any limitations of the new definition that should be considered?
- The theoretical analysis suggests that dimensionality and feature correlation play a crucial role in the counterintuitive phenomenon. Could the authors discuss potential strategies for mitigating this issue in future work?
- The empirical evaluation focuses on tabular and image datasets. Could the authors discuss the generalizability of their findings to other data modalities, such as text or audio?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4