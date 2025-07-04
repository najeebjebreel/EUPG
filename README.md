# EUPG: Efficient Unlearning with Privacy Guarantees

This is the official repository containing the code needed to replicate the results from "EUPG: Efficient Unlearning with Privacy Guarantees."

## Paper

[EUPG: Efficient Unlearning with Privacy Guarantees]()

## Usage

The repository includes a Jupyter notebook for each benchmark. These notebooks can be used to reproduce the experiments reported in the paper for their respective benchmarks. Note that for CIFAR10, you need to generate the DP-protected data using the codes in the folder dp_data.

## Datasets

The four datasets used are publicly available( the three tabular datasets are located in the data folder):
- [Adult Income](https://archive.ics.uci.edu/ml/datasets/Adult)
- [Heart Disease](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)
- [Credit Information](https://www.kaggle.com/c/GiveMeSomeCredit)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) will automatically be downloaded from Torchvision datasets.

## Dependencies

- **Python**: 3.8.16
- **TensorFlow**: 2.13.0
- **Torch**: 1.12.0
- **Torchvision**: 0.13.0
- **NumPy**: 1.24.3
- **Pandas**: 1.5.3
- **SciPy**: 1.10.1
- **Scikit-learn**: 1.3.0
- **XGBoost**: 2.0.2

The required Anaconda environment can be installed using the `environment.yml` file.

## Main Results

### Before Unlearning

*The tables below show the performance of EUPG before forgetting.*  
![EUPG performance before unlearning](figures/before.png)

### After Unlearning

*The tables below show the post-unlearning performance of EUPG after forgetting 5% of the training data.*  
![EUPG performance after unlearning](figures/after.png)

## Citation

*Information to be added*

## Funding

Partial support for this work has been received from the Government of Catalonia (ICREA Acad\`emia Prizes to J. Domingo-Ferrer and to D. S\'anchez, and grant 2021SGR-00115), MCIN/AEI/ 10.13039/501100011033 and ``ERDF A way of making Europe'' under grant PID2021-123637NB-I00 ``CURLING'', and  the EU's NextGenerationEU/PRTR via INCIBE (project ``HERMES'' and INCIBE-URV cybersecurity chair).

