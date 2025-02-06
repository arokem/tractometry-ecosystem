---
title: 'Computational examples of software for white matter tractometry'
tags:
  - Preprint
  - Jupyter Book
  - Reproducible article
  - Neuroscience

authors:
  - name: John Kruper
    affiliation: "1, 2"

  - name: Ariel Rokem
    affiliation: "1, 2"

affiliations:
 - name: Department of Psychology, University of Washington, Seattle, WA, USA
   index: 1
 - name: eScience Institute, University of Washington, Seattle, WA, USA
   index: 2

date: February 6th, 2024
bibliography: paper.bib
---

# Summary

Tractometry uses diffusion-weighted magnetic resonance imaging (dMRI) to assess
the physical properties of long-range brain connections [@Yeatman2012AFQ].
We present an integrative ecosystem of software that performs all steps
of tractometry: post-processing of dMRI data, delineation of major white matter
pathways, and modeling of the tissue properties within them. This ecosystem
also provides tools that extract insights from these measurements, including
novel implementations of machine learning and statistical analysis methods that
consider the unique structure of tractometry data [@RichieHalford2021SGL,@Muncy2022GAMs], as well as tools for visualization and interpretation of the results [@Yeatman2018AFQBrowser,@Kruper2024-ke]. Taken together, these
open-source software tools provide a comprehensive environment for the analysis
of dMRI data.

# Acknowledgements

This work was funded by National Institutes of Health grants MH121868,
MH121867, and EB027585, as well as by National Science Foundation grant
\#1934292. Software development was also supported by the Chan Zuckerberg
Initiative's Essential Open Source Software for Science program, the Alfred P.
Sloan Foundation and the Gordon \& Betty Moore Foundation.

# References
