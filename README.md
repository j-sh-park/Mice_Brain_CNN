# Image Classifier Using Convolutional Neural Network on Cell Morphology

**Group:** Image7

This was completed as part of our DATA3888 (2023) Capstone project on Biotechnology - Image Analysis.

The aim of our project is to determine the impact of preprocessing techniques on the training dataset that is fed into the deep learning model, on the deep learning model's accuracy. The dataset is obtained from <a href="https://www.10xgenomics.com/resources/datasets/fresh-frozen-mouse-brain-for-xenium-explorer-demo-1-standard">10xGenomics</a>. 

The CNN model used in our project is based on the AlexNet architecture.

## Table of Content

* [Getting Started](#start)
    * [Clone Repository](#clone)
    * [Obtain Models](#models)
    * [RStudio & R](#rstudio)
    * [Install/Check Packages](#packages)
* [Running & Usage](#run)
   * [Usage](#usage)
   * [Merge VS Remove Clusters](#clusters)
   * [Performance Tab](#performance)
   * [Interpretability Tab](#interpret)


## <div id="start">Getting Started</div>

**IMPORTANT:** Please follow these steps sequentially in order to run our Shiny App successfully.

###  <div id="clone">Clone Repository</div>

Clone this repository in order to obtain the Random Forest models, Shiny App script and other resources required.

```shell
git clone https://github.sydney.edu.au/tzar0514/image7-shiny
```

### <div id="models">Obtain Models</div>

1. Obtain our CNN models via this <a href="https://drive.google.com/file/d/1WQvHoY686EXpFOr6gDn77dOmZeEZvRUs/view?usp=share_link">link</a> (**WARNING:** This file is approxmately 2.4GB. It may take a few minutes depending on the strength of your network connection.) 

2. After you have downloaded the zip file, please extract the `cnn_models` folder into the image7-shiny directory (i.e image7-shiny/cnn_models)

### <div id="rstudio">RStudio & R</div>

#### If you already have R & RStudio, skip this step!

Our app runs on R and the shiny app has be to deployed via an R source file. As such, you must have RStudio and R language installed in your local machine. 
Follow the installation guide <a href="https://rstudio-education.github.io/hopr/starting.html">here</a>.

### <div id="packages">Install/Check Packages</div>

Our Shiny App requires the use of the following dependencies:

- shiny
- ggplot2
- plotly
- keras
- EBImage
- SpatialPack
- pracma
- randomForest
- BiocGenerics

Most of these packages can be installed via the standard way of `install.packages("package")`. 

BiocGenerics and EBImage are R packages distributed as part of the Bioconductor project. To install this package, run R and enter:

```shell
install.packages("BiocManager")
BiocManager::install("EBImage")
```

Once installed, simply load the library in using the standard manner with `library(package)`. 

For more information on installation, visit these webpages:
- <a href="https://bioconductor.org/packages/release/bioc/html/BiocGenerics.html">BiocGenerics Installation Guide</a>
-  <a href="https://bioconductor.org/packages/release/bioc/vignettes/EBImage/inst/doc/EBImage-introduction.html#1_Getting_started">EBImage Installation Guide</a>

### <div id="run">Running & Usage</div>

Open the file `image7_shiny` in RStudio and click on 'Run App' on the top right.

#### <div id="usage">Usage/Instructions</div>

The introduction tab should inform you how to use and navigate around the app. To reiterate the instructions:

1. Upload a PNG image of a cell. **IMPORTANT:** It should be named in this specific formart: `cell_<ID>.png` where `ID` is any numeric value.
   - An image will appear on the right. That is your input image. 

2. Select the preprocessing technique of your choice.
   - Another image will appear on the right beneath the text. This is your input image with the chosen preprocessing technique applied.
   
3. If you have chosen a preprocessing technique with boundaries, upload the CSV file containing the x and y vertices of the region of interest in your image.

4. Select between Random Forest and CNN as your model to predict the image with.

#### <div id="clusters">Difference between Merge & Remove Clusters</div>

You can also choose between `Merge` and `Remove` clusters to use models that were trained on the two different datasets.
- Merge Clusters involved merging 28 classes based on their <a href="https://cf.10xgenomics.com/samples/xenium/1.0.2/Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP/Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP_analysis_summary.html">UMAP projection</a> down to 17 clusters
- Removed clusters involved removing clusters that contained too few images

#### <div id="performance">Performance Tab</div>

This tab informs the user about accuracy and robustness of the CNN and Random Forest models, as well as comparisons between the two models.
   
#### <div id="interpret">Interpretability Tab</div>

This tab informs the user about interpretability of the CNN and Random Forest models, as well as comparisons between the two models. More information is provided for the CNN model as it is more of a black-box model as compared to Random Forest.
