# CEREBRAL VASCULAR

## About this app

This web application aims to equip the  healthcare professional of the imaging department with a viewer that, working with computer vision, is able to extract morphological characteristics of brain blood vessels and work them quantitatively by reducing the visual interpretation. The platform will help the doctor to determine a more accurate prognosis and to be able to predict the degree of susceptibility of certain children to future diseases, such as malformations.

This is achieved by processing a magnetic resonance image of the brain following a series of steps to obtain parameters that allow us to evaluate the symmetry of the two cerebral hemispheres.

## How to run this app locally

(The following instructions are for unix-like shells)

Clone this repository and navigate to the directory containing this `README` in
a terminal.

Make sure you have installed the latest version of pip

```bash
python3 -m pip install --upgrade pip
```

Create and activate a virtual environment (recommended):

```bash
python -m venv myvenv
source myvenv/bin/activate
```

Install the requirements

```bash
pip install -r requirements.txt
```

Run the app. An IP address where you can view the app in your browser will be
displayed in the terminal.

```bash
python app.py
```
