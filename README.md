# Response time analysis for sporadic tasks in uniprocessor fixed-priority scheduling with starting and resuming delays

Code & artifacts related to paper *Response time analysis for sporadic tasks in uniprocessor fixed-priority scheduling with starting and resuming delays*, RTNS 2025, by Hadrien Barral, Yasmina Abdeddaïm, Damien Masson and Joël Goossens.

This work has been presented at [RTNS 2025](https://rtns2025.retis.santannapisa.it/).
Slides of the presentation are available in the repository (`slides.pdf`).

# Organization

Main folders:
- `fsimu`: fast C++ simulator. Allows for sporadic and periodic simulations, with many tuning parameters. Contains a taskset generator.
- `simu`: python simulator. Allows computing RiLD and perform periodic simulations. Contains a taskset generator.

Various helpers:
- `draw-schedule`: sub-module to draw chronograms
- `legend`: builds the legend for chronograms, as used in the paper
- `simu/hl`: builds all chronograms used in the paper and presentation
- `plot`: draws the plots used in the paper. Run `plot.py`.

# License

This tool is released under SPDX-License-Identifier: GPL-3.0-only.
