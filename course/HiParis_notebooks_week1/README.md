<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->

<p align="center">
  <a href="https://www.hi-paris.fr/">
    <img src="./src/figures/logo-hi-paris-retina.png" alt="Logo" width="280" height="180">
  </a>
</p>
<h3 align="center">Hackathon 2023 - Training session n°1</h3>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of contents</summary>
  <ol>
    <li><a href="#about-hi!-paris">About Hi! PARIS</a></li>
    <li><a href="#resources-and-documentation">Resources and documentation</a></li>
    <li><a href="#quickstart">Quickstart</a></li>
  </ol>
</details>



<!-- About Hi! Paris -->

## About Hi! PARIS

Hi! PARIS is the brand new Center on Data Analytics and Artificial Intelligence for Science, Business and Society created in the framework of the Institut Polytechnique de Paris (IP Paris) and HEC Paris Allian


<!-- About Hi! Paris -->



## Resources and documentations

### Python library


- The official documentation for [Python](https://docs.python.org/3/)

- [NumPy](https://en.wikipedia.org/wiki/NumPy) is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- [Pandas](https://pandas.pydata.org/) is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.

- [Scikit Learn](https://scikit-learn.org/stable/) is a Python module for machine learning built on top of [SciPy](https://www.scipy.org/)
- [TensorFlow](https://www.tensorflow.org/?hl=fr) library for numerical computation using data flow graphs. 
- [Torch](https://pytorch.org/) provides several tools for fast tensor mathematics, storage interfaces and machine learning models.

### Cheatsheet
- [Python](src/cheat_sheet/mementopython3-english.pdf)
- Data manipulation : [Numpy](src/cheat_sheet/Numpy_Python_Cheat_Sheet.pdf), [Pandas](src/cheat_sheet/Pandas_Cheat_Sheet.pdf)
- Data visualisation : [Matplotlib](src/cheat_sheet/Python_Matplotlib_Cheat_Sheet.pdf), [Seaborn](src/cheat_sheet/Python_Seaborn_Cheat_Sheet.pdf), [Plotly](src/cheat_sheet/python_cheat_sheet.pdf)
- Machine learning : [Sklearn](src/cheart_sheet/Scikit_Learn_Cheat_Sheet_Python.pdf)


## Quickstart

- In a terminal shell, go to the root of the directory
- Install the required python packages:
```
  pip insall -r requirements.txt
```
or, if you're using Anaconda:
```
  conda create --name <env> --file requirements_conda.txt
```
- Activate jupyter notebook nbextensions:
```
  jupyter contrib nbextension install --user
```
- Start jupyter :
```
  jupyter notebook
```
- Once jupyter is launched, click the "Nbextension" tab, then tick the "Exercise2" box.

