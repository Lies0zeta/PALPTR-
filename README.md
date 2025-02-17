# PTR

This repository contains the implementation of the paper:

**Efficient Initial Data Selection and Labeling for Multi-Class Classification Using Topological Analysis**

[Lies Hadjadj](https://orcid.org/0000-0002-7926-656X), [Emilie Devijver](http://ama.liglab.fr/~devijvee/), [Rémi Molinier](https://www-fourier.ujf-grenoble.fr/~molinier/), and [Massih-Reza Amini](http://ama.liglab.fr/~amini/).



## Abstract
Machine learning techniques often require large labeled training sets to attain optimal performance. However, acquiring labeled data can pose challenges in practical scenarios. Pool-based active learning methods aim to select the most relevant data points for training from a pool of unlabeled data. Nonetheless, these methods heavily rely on the initial labeled dataset, often chosen randomly. In our study, we introduce a novel approach specifically tailored for multi-class classification tasks, utilizing Proper Topological Regions (PTR) derived from topological data analysis (TDA) to efficiently identify the initial set of points for labeling. Through experiments on various benchmark datasets, we demonstrate the efficacy of our method and its competitive performance compared to traditional approaches, as measured by average balanced classification accuracy.



## Usage

```python
from ptr import PTRSelector

# Initialize the selector
selector = PTRSelector()

# Get initial points to label
points_to_label = selector.select_points(X)
```

## Datasets

We evaluated our method on several benchmark datasets:

- PROTEIN 
- BANKNOTE
- COIL-20
- ISOLET
- PEN-DIGITS
- NURSERY

## Results

Our method demonstrates superior performance compared to:
- Random selection
- K-means clustering
- Core-set approach
- Uncertainty sampling

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{hadjadj2024efficient,
    title={Efficient Initial Data Selection and Labeling for Multi-Class Classification Using Topological Analysis},
    author={Hadjadj, Lies and Devijver, Emilie and Molinier, R{\'e}mi and Amini, Massih-Reza},
    booktitle={ECAI 2024: 26th European Conference on Artificial Intelligence},
    series={Frontiers in Artificial Intelligence and Applications},
    volume={392},
    pages={2677--2684},
    year={2024},
    doi={10.3233/FAIA240800},
    publisher={IOS Press}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue or contact:
- Lies Hadjadj (lies.hadjadj@univ-grenoble-alpes.fr)
