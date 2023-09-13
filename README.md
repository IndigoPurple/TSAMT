# Cross-Camera Human Motion Transfer by Time Series Analysis
In this repository we provide code of the paper:
> **Cross-Camera Human Motion Transfer by Time Series Analysis**

> Yaping Zhao, Guanghan Li, Edmund Y. Lam

> link: 

<p align="center">
<img src="img/teaser.png">
</p>

# Usage
1. Identify seasonality with fourier series analysis. Check out `fourier_analysis.py`.
2. Build an addictive time series model;
3. find  periodic  points;
4. extract  addictive  factor;
5. transfer  motion  pattern. 
Step 2-5 are implemented with `utils.py`.

# Citation
Cite our paper if you find it interesting!
```
@article{zhao,
  title={Cross-Camera Human Motion Transfer by Time Series Analysis},
  author={Zhao, Yaping and Li, Guanghan and Lam, Edmund Y.},
  journal={to appear}
}
```
