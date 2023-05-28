# Image Classifier Using Convolutional Neural Network on Cell Morphology

**Group:** Image7

This was completed as part of our DATA3888 (2023) Capstone project on Biotechnology - Image Analysis.

The aim of our project is to determine the impact of preprocessing techniques on the training dataset that is fed into the deep learning model, on the deep learning model's accuracy. The dataset is obtained from <a href="https://www.10xgenomics.com/resources/datasets/fresh-frozen-mouse-brain-for-xenium-explorer-demo-1-standard">10xGenomics</a>. 

The CNN model used in our project is based on the AlexNet architecture.

The Shiny App Github is located <a href="https://github.sydney.edu.au/tzar0514/image7-shiny">here</a>. To run the Shiny App, folllow the README file in that GitHub.

## Table of Content

* [Reproducing the Report](#start)
    * [Clone Repository](#clone)
    * [Install/Check Packages](#packages)
* [Important Notice](#notice)
* [Other Information/Resources](#others)

## <div id="start">Reproducing the Report</div>

**IMPORTANT:** Please follow these steps sequentially in order to render the report.

Due to the nature of our project, our report relies on cached files. All of which are in this repository or in Google Drive links provided at the end of this README, as well as in the report.

###  <div id="clone">Clone Repository</div>

Clone this repository in order to obtain the RDS files of Random Forest accuracies, models, CNN accuracies and other resources required.

```shell
git clone https://github.sydney.edu.au/anad8554/Image7/
```

### <div id="packages">Install/Check Packages</div>

Our Report requires the use of the following dependencies:

- shiny
- ggplot2
- keras
- EBImage
- SpatialPack
- pracma
- randomForest
- RBioFormats
- tensorflow
- cowplot
- pracma
- tidyverse

Please ensure all these packages are installed before knitting `report_draft.Rmd`.

## <div id="notice">Important Notice</div>

In the event you decide to run the models yourself, you may encounter some errors regarding the file data path. Always ensure that the data paths match.
- A likely example is running `alexnet_multiple.Rmd` or `alexnet_without_boundaries.Rmd`. To run these properly, simply take these models out of the folder `tools` and put it into the same directory as the `report_draft.Rmd`.

## <div id="others">Other Information/Resources</div>

This section will provide links (which are also stated in the report) to additional resources you may need should you run the models yourself.

- Original Data (No preprocessing techniques applied) - <a href="https://drive.google.com/drive/folders/1aA_-F5AWbB9r6-FBjk7jglopN9QfArbq?usp=sharing">Source</a>
   - This link contains:
      - `original_given`: the original dataset given by DATA3888 Teaching Team 
      - `merged_clusters`: the dataset that merged clusters
      - `removed_clusters`: the dataset that removed clusters

- Unread Augmented Data (Preprocessing techniques have already been applied) - <a href="https://drive.google.com/drive/folders/1jp6Fo5gww6vW3T-Rr2S3D-zwoM2b0QIU?usp=sharing">Source</a> 
   - Images that will need to be read in. **WARNING:** Reading images via `get_images()` in `alexnet_multiple.Rmd` may take up to 30 minutes. We provided an alternative to save computational time: **Read Augmented Data**  
   - The conversion script to get this dataset from the original given data by the teaching team can be found <a href="https://github.sydney.edu.au/anad8554/Image7/blob/main/tools/Image_convert.Rmd">here</a>

- Read Augmented Data (Preprocessing techniques have already been applied) - <a href="https://drive.google.com/drive/folders/1S1ASvH6tAZt5VUVI_jazss9LAjhVRmvg?usp=sharing">Source</a> 
   - Images that have been read into a variable and saved into an RDS file for each dataset. Using this will only take a few seconds compared to Unread.
   - The conversion script can be found <a href="https://github.sydney.edu.au/anad8554/Image7/blob/main/readingImages.Rmd">here<a>

- CNN Models (Weights) - <a href="https://drive.google.com/file/d/1LtC88X1hY5QR9BV8P8yoOE3AHAZ5dPMY/view?usp=drive_link">Source</a>
   - Actual model architecure and how to save the full model can be found in `tools/alexnet_multiple.Rmd`
   
