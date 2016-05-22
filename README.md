# BCFL2016

This repository holds the code for the paper "Pareto weights as wedges in two-country models" by David Backus, Chase Coleman, Axelle Ferriere, and Spencer Lyon.

If there are any questions, issues, or comments about the code feel free to reach out to any of the authors or (better yet) [open an issue](https://github.com/NYUEcon/BCFL2016/issues/new) here on this GitHub repository.

## Notation

Notation in the code should map closely to the notation used in the paper. In the table below we outline the notational conventions used in this code:

| Variable Name |          Mathematical notation           |
|---------------|------------------------------------------|
| `x`           | Any variable at time `t`, or a parameter |
| `xp`          | `x_{t+1}`                                |
| `xm`          | `x_{t-1}`                                |
| `xh`          | `\hat{x}`                                |
| `lx`          | `\log(x)`                                 |

These can be combined in arbitrary ways. For example `lzhp` means `\log \hat{z}_{t+1}}`

## Running the code

To run the code, navigate to the `code` folder in this respository and run `julia main_cv.jl` for the constant volatility model (not featured in the paper) or `julia main_sv.jl` for the stochastic volatility version.

This will compute solutions for all parameterizations of the model highlighted in the text or figures of the paper. Solutions will be stored in a directory `code/solutions`.

## Generating figures

To generate the figures included in the paper, first solve the model by following the instructions above and (from the `code` directory) run the file `julia paper_figures.jl`. This will output all the figures contained in the paper to a directory `code/images`.

You can also see all the figures in the Jupyter notebook `code/PaperPictures.ipynb`. The Julia script above is actually just a download of the code cells from that notebook.

