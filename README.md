# Cross-Camera Human Motion Transfer by Time Series Analysis
In this repository we provide code of the paper:
> **Cross-Camera Human Motion Transfer by Time Series Analysis**

> Yaping Zhao, Guanghan Li, Edmund Y. Lam

> supplementary material: https://github.com/IndigoPurple/TSAMT/blob/main/sm/sm.pdf

<p align="center">
<img src="img/teaser.png">
</p>

# Usage
## Preparation

Before using the HMR (End-to-end Recovery of Human Shape and Pose) project, certain preparations need to be made. In this section, we will describe the necessary procedures and models required to utilize the HMR methodology for the reconstruction of 3D human meshes from 2D video frames.

### 1. SMPL Model

The SMPL (Skinned Multi-Person Linear) model is a representation of 3D human bodies. It provides a parameterized model that can be used to represent the shape and pose of a human body. In this project, we employ the SMPL model to accurately represent 3D bodies.

To utilize the SMPL model, please refer to the procedures established in HMR. These procedures will guide you on how to incorporate the SMPL model into your workflow and make use of its capabilities for 3D body representation.

### 2. HMR Methodology

The HMR methodology is the core technique used in this project for the reconstruction of 3D human meshes from 2D video frames. It leverages deep learning models to estimate the shape and pose of human bodies given 2D image inputs.

To reconstruct 3D human meshes using the HMR methodology, follow the procedures outlined in HMR. These procedures will provide you with the necessary guidance to integrate the HMR methodology into your workflow and perform accurate 3D human mesh reconstruction.

By following the above procedures and utilizing the SMPL model and HMR methodology as described, you will be able to effectively reconstruct 3D human meshes from 2D video frames using the HMR project.

For detailed implementation instructions and code examples, please refer to the official GitHub repository of HMR: [GitHub - akanazawa/hmr](https://github.com/akanazawa/hmr).

[//]: # (1. Identify seasonality with fourier series analysis. Check out `fourier_analysis.py`.)

[//]: # (2. Build an addictive time series model;)

[//]: # (3. find  periodic  points;)

[//]: # (4. extract  addictive  factor;)

[//]: # (5. transfer  motion  pattern. )

[//]: # (Step 2-5 are implemented with `utils.py`.)

## Time Series Analysis
To utilize the code provided in this repository for analyzing time series data using Fourier series and implementing an addictive time series model, follow the steps outlined below:

1. **Identify Seasonality using Fourier Series Analysis:**
   - Open the file named `fourier_series.py` in your preferred Python environment.
   - Ensure that you have the necessary dependencies installed, including NumPy and Matplotlib.
   - Specify the input data source by uncommenting either the line that loads data from a CSV file or the line that loads data from a NumPy array.
   - Adjust the value of the variable `pick` to select the desired column from the input data for analysis.
   - Run the script.

2. **Build an Addictive Time Series Model:**
   - Open the `utils.py` file that contains the implementation of the addictive time series model.
   - Review the provided functions, including `mean_smoothing` and `exponential_smoothing`, which are used for data preprocessing.
   - Familiarize yourself with the `periodicDecomp` function, which performs the decomposition of the time series based on periodic points.
   - Understand the purpose and usage of additional functions present in the file.

3. **Find Periodic Points within the Time Series:**
   - In the `periodicDecomp` function of `utils.py`, specify the necessary input parameters:
     - `lr`: The input low-resolution (LR) time series data.
     - `hr`: The input high-resolution (HR) time series data.
     - `lr_points`: An array specifying the periodic points within the LR time series.
     - `hr_points`: An array specifying the periodic points within the HR time series.
   - Adjust any other relevant parameters or configurations within the function as needed.

4. **Extract the Addictive Factor:**
   - In the `periodicDecomp` function of `utils.py`, observe the steps and calculations performed to decompose the HR and LR periods.
   - Understand how the additive factors, represented by the arrays `hr_factor_add_4` and `lr_factor_add_4`, are computed and applied to the LR time series.

5. **Transfer the Motion Pattern to the Target Camera:**
   - Review the code in `utils.py` that handles the transfer of the motion pattern to the target camera.
   - Explore the usage of variables such as `LR_cameras`, `texture_img`, `texture_vt`, and `data_dict` in the relevant functions.
   - Understand any additional steps or transformations required to accomplish the motion pattern transfer.

By following these steps and understanding the code details and parameters provided in `fourier_series.py` and `utils.py`, you can effectively utilize the code for analyzing time series data, building an addictive time series model, identifying periodic points, extracting additive factors, and transferring motion patterns to the target camera.
# Citation
Cite our paper if you find it interesting!
```
@article{zhao,
  title={Cross-Camera Human Motion Transfer by Time Series Analysis},
  author={Zhao, Yaping and Li, Guanghan and Lam, Edmund Y.},
  journal={to appear}
}
```
