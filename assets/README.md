# Paper figures and abstract

Source: Haijiang Li, Chengyu Lv, Yi Zhang, Rui Qian, Zhibing Zhang, Xiangqing Shen,
Junjie Yang, Yuchen Zhang, Wenyuan Jiang, Hanqing Hu, and Cangqi Zhou,
**VERPO: Verified Evidence Regularized Policy Optimization**, arXiv:2609.06100v2.

- [Paper](https://arxiv.org/abs/2609.06100v2)
- [Source archive](https://arxiv.org/src/2609.06100v2)
- [Paper license: CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

`overview.png` is rendered from `fig/Method_asset.pdf` (Figure 3), with outer
white margins cropped. `grpo-verpo.png` is rendered from `fig/Motivation_RLVR.pdf`
(Figure 1). Content, labels and colors are unchanged. The diagrams were not redrawn
or generated. English Abstract text is reproduced from
`main_sections/section_00_abstract.tex`; the Chinese README provides a translation.
These paper-derived materials retain CC BY-SA 4.0, separately from the code's
Apache-2.0 license.

Rendering commands (Poppler):

```bash
pdftoppm -png -singlefile -scale-to 2600 -x 20 -y 100 -W 2560 -H 980 Method_asset.pdf overview
pdftoppm -png -singlefile -scale-to 2000 Motivation_RLVR.pdf grpo-verpo
```

The source PDFs/archive are kept outside Git. `sources.json` records their SHA256
values, output sizes and source archive member names.
