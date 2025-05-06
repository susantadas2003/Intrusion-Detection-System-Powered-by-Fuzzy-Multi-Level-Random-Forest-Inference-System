
# Intrusion Detection System Powered by Fuzzy Multi-Level Random Forest Inference System

## **Overview**
This project implements an advanced **Intrusion Detection System (IDS)** using a hybrid approach that combines fuzzy logic with a multi-level random forest inference system. The system is designed to accurately identify malicious activities in a network while reducing false positives and maintaining scalability.

---

## **Features**
- **Hybrid Approach**: Combines fuzzy logic and multi-level random forest classifiers for enhanced accuracy.
- **Scalable Design**: Modular and extensible for various intrusion scenarios.
- **Real-time Detection**: Capable of analyzing network traffic in real-time.
- **High Precision**: Reduces false positives using a multi-layered classification approach.

---

## **Algorithm and Workflow**
The system workflow is as follows:

1. **Input Preprocessing**: Collect and preprocess raw network traffic data.
2. **Feature Selection**: Select the most relevant features for intrusion detection using a ranking algorithm.
3. **Fuzzy Logic Layer**: Utilize fuzzy rules to handle ambiguous data and generate preliminary classification.
4. **Multi-Level Random Forest Layer**: Apply multiple random forest classifiers for hierarchical decision-making.
5. **Output Decision**: Classify the input as benign or malicious with high confidence.

A detailed flowchart of the algorithm is available in the documentation.

---

## **Requirements**
To set up the project, ensure the following dependencies are installed:

- Python >= 3.8
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Fuzzy Logic Libraries

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## **Getting Started**
 Clone the repository:
   ```bash
   git clone https://github.com/Subhajit-Nandi/Network-Intrusion-Detection-System-powered-by-Fuzzy-Multilevel-Random-Forest-Inference-System
   cd Network-Intrusion-Detection-System-powered-by-Fuzzy-Multilevel-Random-Forest-Inference-System
   ```

---

## **Results and Observations**
The system achieves high detection accuracy on standard intrusion detection datasets with low false positive rates.

---

## **Future Work**
- Extend the model to detect zero-day attacks.
- Optimize feature selection using deep learning techniques.
- Develop a GUI for real-time monitoring.

---

## **References**
- [Fuzzy Logic Systems](https://www.tutorialspoint.com/artificial_intelligence/artificial_intelligence_fuzzy_logic_systems.htm)
- [Random Forest Classifier](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Network Intrusion Detection](https://www.researchgate.net/publication/369190552_A_Multi-level_Random_Forest_Model-Based_Intrusion_Detection_Using_Fuzzy_Inference_System_for_Internet_of_Things_Networks)

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request for any improvements.

---
