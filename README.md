# Apple Leaf Disease Detection Project

This project use CNN model to detect apple leaves disease using images. It consists of two main parts:
1. **Model Building and Training**: The model is built and trained using a dataset of healthy and diseased apple leaves from Kaggle website.
2. **GUI Application**: A graphical user interface (GUI) to upload an image of an apple leaf and predict whether it is healthy or diseased using the trained model.

---

## Installation
1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

Note:
- `Dataset/` is not included in the repository because it is too large.
- Trained model files such as `trained_model.h5` are also ignored.

## Project Structure
Apple_leaf_disease_project/
├── Dataset/
│   ├── diseased/             # Images of diseased apple leaves
│   └── healthy/              # Images of healthy apple leaves
│
├── GUI/
│   ├── __init__.py           # Init file for GUI module
│   └── AppleLeafGUI.py       # GUI application for prediction
│
├── model/
│   ├── __init__.py           # Init file for model module
│   └── AppleLeafModel.py     # CNN model class: training, evaluation, prediction
│
├── best_model_weights.h5     # Best model weights saved via ModelCheckpoint
├── trained_model.h5          # Final trained model (best weights loaded and saved)
├── main.py                   # Script to run the complete pipeline (model + GUI)
└── README.md                 # Project overview and instructions


### Description of files:
- **Dataset**: Contains the images for training the model. 
  - `diseased/`: Images of diseased apple leaves.
  - `healthy/`: Images of healthy apple leaves.
- **GUI**: Contains the code for the graphical user interface.
  - `AppleLeafGUI.py`: The main file for the GUI that allows users to upload an image and predict the disease status of an apple leaf.
- **model**: Contains the machine learning model's code.
  - `AppleLeafModel.py`: Defines the model architecture and methods for training, evaluating, and predicting.
  - `main.py`: The main script for training and testing the model.
- **trained_model.h5**: The pre-trained model saved after training.


## How to Train the Model
    just go to the main.py class, import AppleLeafModel class, create an object 
    and call the run() function which is
    defined in model class, then click the run button, the model will be trained and evaluated,
    and will be saved in trained_model.h5 file

##how to see the evaluation plot and confusion matrix visualisation
    go to view-Panes-Plots, you will see the plot of training and 
    and validation accuracy as well as confusion matrix
## How to upload an image and see the result
    go to the main.py class, comment out model.run() function, and create a AppleLeafGUI
    class (e.g. gui=AppleLeafGUI), the GUI window will automatically appear
    on your window. Then just upload an image from your computer an it will
    show you the result and probability



