library(shiny)
library(ggplot2)
library(plotly)
library(keras)
library(EBImage)
library(SpatialPack)
library(pracma)
library(randomForest)

# NOTE: Need an 'images' and 'cnn_models' folder in your working directory for user input & cnn models
addResourcePath(prefix = "imgResources", directoryPath = "./images")
addResourcePath(prefix = "cnn_models", directoryPath = "./cnn_models")

addResourcePath(prefix = "rf_models_merged", directoryPath = "./RF Models (Merged)")
addResourcePath(prefix = "resources", directoryPath = "./resources")
addResourcePath(prefix = "rf_models_removed", directoryPath = "./RF Models (Removed)")

#functions
denoise_filter <- function(img){
  img = denoise(as.matrix(img), type="enhanced") 
  return(img)
}

power_filter <- function(img){
  img = img^2.5
  return(img)
}

thresholding_filter <- function(img){
  img = thresh(img, w=9, h=8, offset=0.05)
  img = fillHull(opening(img, makeBrush(5, shape='diamond')))
  return(img)
}

opening_filter <- function(img){
  img = opening(img, makeBrush(5, shape='diamond'))
  return(img)
}

#Use this in the app to convert the users input image
convert_img <- function(img, img_technique){
  if (img_technique == "With Power Law and boundary"){
    return(power_filter(img))
  } else if (img_technique == "With Opening and boundary") {
    return(opening_filter(img))
  } else if (img_technique == "With Denoise and boundary") {
    return(denoise_filter(img))
  } else if (img_technique == "With Thresholding and boundary") {
    return(thresholding_filter(img))
  } 
  
  
  
  
  
  return(img)
  
}
#```


# create model architecture with no weights
create_model <- function(learning_rate = 0.000000000001, input_shape=c(224, 224, 1), cluster_number=17) {
  
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    
    # 1st layer
    layer_conv_2d(filters = 96, kernel_size = c(11,11), strides = c(4,4), activation = 'relu', input_shape = input_shape, padding="same") %>% 
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(3, 3), strides = c(2,2)) %>% 
    
    # 2nd layer
    layer_conv_2d(filters = 256, kernel_size = c(5,5), strides=c(1,1), activation = 'relu', padding = "same") %>% 
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(3, 3), strides = c(2,2)) %>% 
    
    # 3rd layer
    layer_conv_2d(filters = 384, kernel_size = c(3,3), strides=c(1,1), activation = 'relu', padding = "same") %>% 
    layer_batch_normalization() %>%
    
    # 4th layer
    layer_conv_2d(filters = 384, kernel_size = c(3,3), strides=c(1,1), activation = 'relu', padding = "same") %>%
    layer_batch_normalization() %>%
    
    # 5th layer
    layer_conv_2d(filters = 256, kernel_size = c(3,3), strides=c(1,1), activation = 'relu', padding = "same") %>% 
    layer_batch_normalization() %>%
    layer_max_pooling_2d(pool_size = c(3, 3), strides = c(2,2)) %>% 
    
    # 6th layer
    layer_flatten() %>% 
    layer_dense(units = 4096, activation = 'relu') %>% 
    layer_dropout(rate = 0.5) %>% 
    
    # 7th layer
    layer_dense(units = 4096, activation = 'relu') %>%
    layer_dropout(rate = 0.5) %>%
    
    # 8th layer
    layer_dense(units = cluster_number, activation = 'softmax')
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_sgd(learning_rate = learning_rate),
    metrics = "accuracy"
  )
  
  return(model)
  
}


# Define UI for app that draws a histogram ----
ui <- fluidPage(
  # App title ----
  titlePanel(
    ("Image7")
  ),
  
  tabsetPanel( 
    tabPanel (
      "Introduction",
      h2(strong("Image Classification via Deep Learning and Random Forest")),
      h3("Introduction"),
      p("In this app, users can upload an image of a mice brain cell and its accompanying boundaries file and assess the performance and interpretability of a number of pre-trained deep neural network and random forest classifiers."),
      p('Those with a more data science interest will be able to use our learning tool to understand how these data wrangling techniques impact the convolutional neural network and classical machine learning models.'),
      p('For biologists and geneticists, they can use our learning tool to determine how data wrangling techniques affect the accurate classification of mice brain cells.'),
      h3('About the Data'),
      h4('Origins'),
      p('10xgenomics originally obtained a 10µm section from a C57BL/6 mouse from Charles River Laboratories. The tissue was prepared following the demonstrated protocols Xenium In Situ for Fresh Frozen Tissues - Tissue Preparation Guide (CG000579) and Xenium In Situ for Fresh Frozen Tissues – Fixation & Permeabilization (CG000581). A 1,000 image subset was randomly sampled from a larger dataset containing 36,602 images that were clustered base on gene expression using the Xenium In Situ platform. (10xGenomics, 2023)'),
      h4('Clustering Alterations'),
      p('Originally, the dataset came clustered into 28 clusters. In this project, we experimented with two techniques of cluster alterations. In the first technique (Merge Cluster tab), we merged clusters based on their UMAP projections. For the second technique (Remove Cluster tab), we simply removed clusters that contained too few images.'),
      h4('Data Augmentation'),
      p('In addition to removing clusters with too few images, we also augmented images such that there would be an equal number of images in each cluster. To do this, we performed a number of small random changes to the images (rotations and translations) before saving them to the same cluster.'),
      h4('Image Preprocessing Techniques'),
      p('In addition to the two clustering alteration techniques, we also performed four image preprocessing techniques: Power Law, Thresholding, Opening and Denoise. This gives us a total of 8 datasets: 4 image preprocessing techniques * 2 clustering alterations.'),
      h3('Model Selection'),
      h4('AlexNet (Deep Learning)'),
      p('We choose AlexNet because it is easy to train whilst maintaining relatively good top-5 accuracy. It features less computational demand due to the small amount of layers compared to its competitors. We also thought that its simplicity would allow us to see the impact of preprocessing better. Various tests were conducted to fine tune parameters such as learning rate and epoch size.'),
      h4('Random Forest (Classical Machine Learning)'),
      p('We selected Random Forest as it efficiently handles large datasets. We use our random forests on extracted feature data instead of pixel data due to computational time and higher accuracy.'),
      h4('References'),
      p('10xGenomics. (2023). Fresh Frozen Mouse Brain for Xenium Explorer Demo. 10xgenomics.com. https://www.10xgenomics.com/resources/datasets/fresh-frozen-mouse-brain-for-xenium-explorer-demo-1-standard')
    ),
    
    # 2nd tab
    tabPanel(
      "Merge Cluster",
      # Sidebar layout with input and output definitions ----
      sidebarLayout(
        # Sidebar panel for inputs ----
        sidebarPanel(
          fileInput(
            inputId= "filename",
            label="Upload a PNG file",
            multiple=FALSE,
            accept=c("image/png")
          ),
          selectInput(
            inputId = "img_technique", 
            label= "Choose one technique for your model",
            choices = c(
              "No boundaries or techniques", 
              "With boundary", 
              "With Power Law and boundary", 
              "With Thresholding and boundary", 
              "With Opening and boundary", 
              "With Denoise and boundary"
            )
          ),
          fileInput(
            inputId= "boundaries_file",
            label="Upload the CSV file containing the cell boundaries",
            multiple=FALSE,
            accept=c("text/csv")
          ),
          selectInput(
            inputId = "img_model", 
            label = "Choose your model",
            choices = c("Random Forest", "CNN")
          ),
        ),
        # Main panel for displaying outputs ----
        mainPanel(
          fluidRow(
            # Output: Histogram ----
            # column(width = 6, plotlyOutput("plot1")),
            # column(width = 6, plotlyOutput("plot2"))
            
            column(6, offset = 3, 
                   fluidRow(
                     column(2, plotOutput(outputId ='og_image', width='250px', height='250px'))
                     #column(4, "Original Image")
                   )
            )
          ),
          textOutput(outputId = "prediction"),
          textOutput(outputId = "accuracy"),
          br(),
          textOutput(outputId = "chosen_model"),
          textOutput(outputId = "chosen_technique"),
          br(),
          fluidRow(
            # Output: Histogram ----
            # column(width = 6, plotlyOutput("plot1")),
            # column(width = 6, plotlyOutput("plot2"))
            column(6, offset = 3, plotOutput(outputId ='preprocessed_img', width='250px', height='250px'))
          )
        )
      )
    ),
    
    # Third tab
    tabPanel(
      "Remove Cluster",
      # Sidebar layout with input and output definitions ----
      sidebarLayout(
        # Sidebar panel for inputs ----
        sidebarPanel(
          fileInput(
            inputId= "filename_removed",
            label="Upload a PNG file",
            multiple=FALSE,
            accept=c("image/png")
          ),
          selectInput(
            inputId = "img_technique_removed", 
            label= "Choose one technique for your model",
            choices = c(
              "No boundaries or techniques", 
              "With boundary", 
              "With Power Law and boundary", 
              "With Thresholding and boundary", 
              "With Opening and boundary", 
              "With Denoise and boundary"
            )
          ),
          fileInput(
            inputId= "boundaries_file_removed",
            label="Upload the CSV file containing the cell boundaries",
            multiple=FALSE,
            accept=c("text/csv")
          ),
          selectInput(
            inputId = "img_model_removed", 
            label = "Choose your model",
            choices = c("Random Forest", "CNN")
          ),
        ),
        # Main panel for displaying outputs ----
        mainPanel(
          fluidRow(
            # Output: Histogram ----
            # column(width = 6, plotlyOutput("plot1")),
            # column(width = 6, plotlyOutput("plot2"))
            
            column(6, offset = 3, 
                   fluidRow(
                     column(2, plotOutput(outputId ='og_image_removed', width='250px', height='250px'))
                     #column(4, "Original Image")
                   )
            )
          ),
          textOutput(outputId = "prediction_removed"),
          textOutput(outputId = "accuracy_removed"),
          br(),
          textOutput(outputId = "chosen_model_removed"),
          textOutput(outputId = "chosen_technique_removed"),
          br(),
          fluidRow(
            # Output: Histogram ----
            # column(width = 6, plotlyOutput("plot1")),
            # column(width = 6, plotlyOutput("plot2"))
            column(6, offset = 3, plotOutput(outputId ='preprocessed_img_removed', width='250px', height='250px'))
          )
        )
      )
    ),
    #Third tab
    tabPanel(
      "Performance",
      # Sidebar layout with input and output definitions ----
      fixedRow(
        column(12,
               fixedRow(
                 column(6,
                        h3(strong(("CNN (AlexNet)"))),
                        fluidRow(column(12,
                                        h4(strong("Accuracy")),
                                        plotlyOutput(outputId = "cnn_acc_comparison"),
                                        p("Insert text here")
                                        
                        )), 
                        fluidRow(column(12,
                                        h4(strong("Robustness")),
                                        p("Insert text here")
                        )),
                        
                 ),
                 column(6,
                        h3(strong("Random Forest")),
                        fluidRow(column(12,
                                        h4(strong("Accuracy")),
                                        plotlyOutput(outputId = "rf_boxplot"),
                                        
                                        p("Insert text here")
                        )), 
                        fluidRow(column(12,
                                        h4(strong("Robustness")),
                                        p("Insert text here")
                        ))
                 )
               )
        )
      )
    ),
    tabPanel(
      "Interpretability",
      # Sidebar layout with input and output definitions ----
      h3(strong("Random Forest")),
      
      p("In order to understand exactly how the model makes its predictions, we can implement the use of Activation Heatmaps. This will allow us to interpret the AlexNet model."),
      h3(strong("CNN Alexnet")),
      
      
      
      fixedRow(
        column(12,
               fixedRow(
                 column(6,
                        h3(strong(("CNN (AlexNet)"))),
                        fluidRow(column(12,
                                        h4(strong("Interpretability")),
                                        imageOutput(outputId = 'cnn_archi'),
                                        p("In order to understand exactly how the model makes its predictions, we can implement the use of Activation Heatmaps. This will allow us to interpret the AlexNet model.")
                        ))
                        
                 ),
                 column(6,
                        h3(strong("Random Forest")),
                        fluidRow(column(12,
                                        h4(strong("Interpretability")),
                                        p("Random Forests are much easier to understand based on the decision trees they produce. As such we consider the interpretability of the neural network model and investigate the changes that the four pre-processing techniques can cause.")
                        ))
                 )
               )
        )
      )
    )
  )
)

# Define server logic required to draw a histogram ----
server <- function(input, output) {
  
  data <- reactive({
    req(input$filename)
    img <- readImage(input$filename$datapath)
    return(img)
  })
  
  data_removed <- reactive({
    req(input$filename_removed)
    img <- readImage(input$filename_removed$datapath)
    return(img)
  })
  
  add_model_weights_merged <- function(model) {
    if (input$img_technique == "With boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_merged_boundaries_weights.h5')
    } else if (input$img_technique == "With Power Law and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_merged_17_augmented_power_boundaries.h5')
    } else if (input$img_technique == "With Thresholding and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_merged_17_augmented_threshold_boundaries.h5')
    } else if (input$img_technique == "With Opening and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_merged_opening_weights.hdf5')
    } else if (input$img_technique == "With Denoise and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_denoise_merged_weights.h5')
    } else {
      # no boundaries
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_merged_raw_weights.hdf5')
    }
    
    return(loaded_model)
  }
  
  add_model_weights_removed<- function(model) {
    if (input$img_technique_removed == "With boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_removed_boundaries_weights.h5')
    } else if (input$img_technique_removed == "With Power Law and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_removed_17_augmented_power_boundaries.h5')
    } else if (input$img_technique_removed == "With Thresholding and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_removed_17_thresholding_boundaries.h5')
    } else if (input$img_technique_removed == "With Opening and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_opening_removed_weights.h5')
    } else if (input$img_technique_removed == "With Denoise and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_denoise_removed_weights.h5')
    } else {
      # no boundaries
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_removed_raw_weights.hdf5')
    }
    
    return(loaded_model)
  }
  
  apply_boundary <- function(img, cell_boundary) {
    # rescale the boundary according to the pixels
    pixels = dim(img)
    cell_boundary$vertex_x_scaled <- 1+((cell_boundary$vertex_x - min(cell_boundary$vertex_x))/0.2125)
    cell_boundary$vertex_y_scaled <- 1+((cell_boundary$vertex_y - min(cell_boundary$vertex_y))/0.2125)
    
    # identify which pixels are inside or outside of the cell segment using inpolygon
    pixel_locations = expand.grid(seq_len(nrow(img)), seq_len(ncol(img)))
    
    pixels_inside = inpolygon(x = pixel_locations[,1],
                              y = pixel_locations[,2],
                              xp = cell_boundary$vertex_x_scaled,
                              yp = cell_boundary$vertex_y_scaled,
                              boundary = TRUE)
    
    img_inside = img
    img_inside@.Data <- matrix(pixels_inside, nrow = nrow(img), ncol = ncol(img))
    
    return(img_inside)
  }
  
  mask_resize = function(img, img_inside, w = 50, h = 50) {
    
    img_mask = img*img_inside
    
    # then, transform the masked image to the same number of pixels, 50x50
    img_mask_resized = resize(img_mask, w, h)
    
    return(img_mask_resized)
  }
  
  output$chosen_model <- renderText ({
    req(input$filename, input$img_model, input$img_technique)
    paste0("Your chosen model is: ", input$img_model)
  })
  
  output$chosen_technique <- renderText ({
    req(input$filename, input$img_model, input$img_technique)
    paste0("Your chosen pre-processing technique is: ", input$img_technique)
  })
  
  # REMOVED CLUSTER ------------
  output$chosen_model_removed <- renderText ({
    req(input$filename_removed, input$img_model_removed, input$img_technique_removed)
    paste0("Your chosen model is: ", input$img_model_removed)
  })
  
  output$chosen_technique_removed <- renderText ({
    req(input$filename_removed, input$img_model_removed, input$img_technique_removed)
    paste0("Your chosen pre-processing technique is: ", input$img_technique_removed)
  })
  
  output$accuracy <- renderText ({
    req(input$filename, input$img_model, input$img_technique)
    
    accuracy = NULL
    
    if (input$img_model == 'CNN') {
      if (!(is.null(input$boundaries_file)) && input$img_technique != 'No boundaries or techniques') {
        if (input$img_technique == "With boundary") {
          accuracy = "7.40%"
        } else if (input$img_technique == "With Power Law and boundary") {
          accuracy = "6.96%"    
        } else if (input$img_technique == "With Thresholding and boundary") {
          accuracy = "8.20%"
        } else if (input$img_technique == "With Opening and boundary") {
          accuracy = "7.58%"
        } else if (input$img_technique == "With Denoise and boundary") {
          accuracy = "6.83%"
        } 
      } else if (input$img_technique == 'No boundaries or techniques'){
        # no boundaries
        accuracy = "7.58%"
      }
    } else {
      # random forest accuracies
      if (input$img_technique == "With boundary") {
        accuracy = "82.6%"
      } else if (input$img_technique == "With Power Law and boundary") {
        accuracy = "82.1%"    
      } else if (input$img_technique == "With Thresholding and boundary") {
        accuracy = "81.3%"
      } else if (input$img_technique == "With Opening and boundary") {
        accuracy = "83.1%"
      } else if (input$img_technique == "With Denoise and boundary") {
        accuracy = "83.2%"
      } else {
        # no boundaries
        accuracy = "NA"
      }
    }
    if (!(is.null(accuracy))) {
      paste0("Prediction Accuracy: ", accuracy)
    }
    
  })
  
  output$accuracy_removed <- renderText ({
    req(input$filename_removed, input$img_model_removed, input$img_technique_removed)
    
    accuracy = NULL
    
    if (input$img_model_removed == 'CNN') {
      if (!(is.null(input$boundaries_file_removed)) && input$img_technique_removed != 'No boundaries or techniques') {
        if (input$img_technique_removed == "With boundary") {
          accuracy = "5.60%"
        } else if (input$img_technique_removed == "With Power Law and boundary") {
          accuracy = "6.33%"    
        } else if (input$img_technique_removed == "With Thresholding and boundary") {
          accuracy = "6.21%"
        } else if (input$img_technique_removed == "With Opening and boundary") {
          accuracy = "6.21%"
        } else if (input$img_technique_removed == "With Denoise and boundary") {
          accuracy = "6.09%"
        } 
      } else if (input$img_technique_removed == 'No boundaries or techniques'){
        # no boundaries
        accuracy = "4.97%"
      }
    } else {
      # random forest accuracies
      if (input$img_technique_removed == "With boundary") {
        accuracy = "84.1%"
      } else if (input$img_technique_removed == "With Power Law and boundary") {
        accuracy = "84.2%"    
      } else if (input$img_technique_removed == "With Thresholding and boundary") {
        accuracy = "83.3%"
      } else if (input$img_technique_removed == "With Opening and boundary") {
        accuracy = "83.9%"
      } else if (input$img_technique_removed == "With Denoise and boundary") {
        accuracy = "83.1%"
      } else {
        # no boundaries
        accuracy = "NA"
      }
    }
    if (!(is.null(accuracy))) {
      paste0("Prediction Accuracy: ", accuracy)
    }
    
  })
  
  cnn_prediction <- function(img_technique) {
    img = convert_img(data(), img_technique)
    if (img_technique != 'No boundaries or techniques') {
      # apply boundaries
      req(input$boundaries_file)
      cell_boundaries = read.csv(input$boundaries_file$datapath)
      img_inside = apply_boundary(img, cell_boundaries)
      img_resized = mask_resize(img, img_inside, 224, 224)
    } else {
      # no boundaries
      img_resized = resize(img, 224, 224)
    }
    
    x <- array(dim=c(1, 224, 224, 1))
    x[1,,,1] <- img_resized@.Data
    input_shape = dim(x)[2:4]
    model = create_model(input_shape = input_shape)
    loaded_model = add_model_weights_merged(model)
    
    res = loaded_model %>% predict(x)
    predicted_class = apply(res, 1, which.max)
    
    return(predicted_class)
    
  }
  
  cnn_prediction_removed <- function(img_technique) {
    img = convert_img(data_removed(), img_technique)
    if (img_technique != 'No boundaries or techniques') {
      # apply boundaries
      req(input$boundaries_file_removed)
      cell_boundaries = read.csv(input$boundaries_file_removed$datapath)
      img_inside = apply_boundary(img, cell_boundaries)
      img_resized = mask_resize(img, img_inside, 224, 224)
    } else {
      # no boundaries
      img_resized = resize(img, 224, 224)
    }
    
    x <- array(dim=c(1, 224, 224, 1))
    x[1,,,1] <- img_resized@.Data
    input_shape = dim(x)[2:4]
    model = create_model(input_shape = input_shape)
    loaded_model = add_model_weights_removed(model)
    
    res = loaded_model %>% predict(x)
    predicted_class = apply(res, 1, which.max)
    
    return(predicted_class)
    
  }
  
  rf_prediction <- function(img_technique) {
    
    img = convert_img(data(), img_technique)
    if (img_technique == "With boundary") {
      loaded_model = readRDS("RF Models (Merged)/rf_boundaries_features.rds")
    } else if (img_technique == "With Power Law and boundary") {
      loaded_model = readRDS("RF Models (Merged)/rf_power_features.rds")
    } else if (img_technique == "With Thresholding and boundary") {
      loaded_model = readRDS("RF Models (Merged)/rf_thresholding_features.rds")
    } else if (img_technique == "With Opening and boundary") {
      loaded_model = readRDS("RF Models (Merged)/rf_opening_features.rds")
    } else if (img_technique == "With Denoise and boundary") {
      loaded_model = readRDS("RF Models (Merged)/rf_denoise_features.rds")
    } else {
      # no boundaries
      return("NA - Random Forest cannot compute this.")
    }
    # ensure that boundaries csv is uploaded before computing
    if (!(is.null(input$boundaries_file))) {
      cell_boundaries = read.csv(input$boundaries_file$datapath)
      
      # note: ASSUMPTION THAT THE INPUT IMAGE IS NAMED cell_<ID>.png
      cell_id = gsub(".*cell_|.png", "", input$filename$name)
      img_inside = apply_boundary(img, cell_boundaries)
      img_mask = mask_resize(img, img_inside, 224, 224)
      
      xf = computeFeatures(img_inside, img, expandRef = NULL)
      rownames(xf) <- cell_id
      res = predict(loaded_model, xf)
      predicted_class = gsub(".*cluster_|", "", res)
      
      return(predicted_class)
    }
    return(NULL)
  }
  
  rf_prediction_removed <- function(img_technique) {
    
    img = convert_img(data_removed(), img_technique)
    if (img_technique == "With boundary") {
      loaded_model = readRDS("RF Models (Removed)/rf_boundaries_features_removed.rds")
    } else if (img_technique == "With Power Law and boundary") {
      loaded_model = readRDS("RF Models (Removed)/rf_power_features_removed.rds")
    } else if (img_technique == "With Thresholding and boundary") {
      loaded_model = readRDS("RF Models (Removed)/rf_thresholding_features_removed.rds")
    } else if (img_technique == "With Opening and boundary") {
      loaded_model = readRDS("RF Models (Removed)/rf_opening_features_removed.rds")
    } else if (img_technique == "With Denoise and boundary") {
      loaded_model = readRDS("RF Models (Removed)/rf_denoise_features_removed.rds")
    } else {
      # no boundaries
      return("NA - Random Forest cannot compute this.")
    }
    # ensure that boundaries csv is uploaded before computing
    if (!(is.null(input$boundaries_file_removed))) {
      cell_boundaries = read.csv(input$boundaries_file_removed$datapath)
      
      # note: ASSUMPTION THAT THE INPUT IMAGE IS NAMED cell_<ID>.png
      cell_id = gsub(".*cell_|.png", "", input$filename_removed$name)
      img_inside = apply_boundary(img, cell_boundaries)
      img_mask = mask_resize(img, img_inside, 224, 224)
      
      xf = computeFeatures(img_inside, img, expandRef = NULL)
      rownames(xf) <- cell_id
      res = predict(loaded_model, xf)
      predicted_class = gsub(".*cluster_|", "", res)
      
      return(predicted_class)
    }
    return(NULL)
  }
  
  # displays prediction text
  output$prediction <- renderText({
    # waits for technique, model
    req(input$img_technique, input$img_model)
    predicted_class = NULL
    
    if (input$img_model == 'CNN') {
      if (!(is.null(input$boundaries_file)) && input$img_technique != 'No boundaries or techniques') {
        predicted_class = cnn_prediction(input$img_technique)
      } else if ((is.null(input$boundaries_file)) && input$img_technique == 'No boundaries or techniques') {
        predicted_class = cnn_prediction(input$img_technique)
      }
      paste0("Please upload a CSV file containing the cell boundaries.")
      
    } else {
      # random forest
      predicted_class = rf_prediction(input$img_technique)
    }
    
    if (is.null(predicted_class)) {
      paste0("Please choose a model and technique. If you have chosen techniques with boundaries, please include the CSV file containing the boundaries.")
    } else {
      paste0("The predicted cluster is: ", predicted_class)
    }
    
  })
  
  output$prediction_removed <- renderText({
    # waits for technique, model
    req(input$img_technique_removed, input$img_model_removed)
    predicted_class = NULL
    
    if (input$img_model_removed == 'CNN') {
      if (!(is.null(input$boundaries_file_removed)) && input$img_technique_removed != 'No boundaries or techniques') {
        predicted_class = cnn_prediction_removed(input$img_technique_removed)
      } else if ((is.null(input$boundaries_file_removed)) && input$img_technique_removed == 'No boundaries or techniques') {
        predicted_class = cnn_prediction_removed(input$img_technique_removed)
      }
      paste0("Please upload a CSV file containing the cell boundaries.")
      
    } else {
      # random forest
      predicted_class = rf_prediction_removed(input$img_technique_removed)
    }
    
    if (is.null(predicted_class)) {
      paste0("Please choose a model and technique. If you have chosen techniques with boundaries, please include the CSV file containing the boundaries.")
    } else {
      paste0("The predicted cluster is: ", predicted_class)
    }
    
  })
  
  # displays input image
  output$og_image <- renderPlot({
    req(data())
    plot(data(), all=FALSE)
  })
  
  # displays input image for removed cluster
  output$og_image_removed <- renderPlot({
    req(data_removed())
    plot(data_removed(), all=FALSE)
  })
  
  # displays the preprocessed image
  output$preprocessed_img <- renderPlot({
    req(input$filename, input$img_technique, data())
    if (input$img_technique != 'No boundaries or techniques' && !(is.null(input$boundaries_file))) {
      img = convert_img(data(), input$img_technique)
      # apply boundaries
      cell_boundaries = read.csv(input$boundaries_file$datapath)
      img_resized = apply_boundary(img, cell_boundaries)
      img_resized = mask_resize(img, img_resized)
      plot(img_resized, all=FALSE)
    }
  })
  
  # displays the preprocessed image
  output$preprocessed_img_removed <- renderPlot({
    req(input$filename_removed, input$img_technique_removed, data_removed())
    if (input$img_technique_removed != 'No boundaries or techniques' && !(is.null(input$boundaries_file_removed))) {
      img = convert_img(data_removed(), input$img_technique_removed)
      # apply boundaries
      cell_boundaries = read.csv(input$boundaries_file_removed$datapath)
      img_resized = apply_boundary(img, cell_boundaries)
      img_resized = mask_resize(img, img_resized)
      plot(img_resized, all=FALSE)
    }
  })
  
  # PERFORMANCE TAB ------------------------------------------
  
  output$cnn_archi<- renderImage({
    
    list(src = "resources/model_architecture.png", width = "90%")
    
  }, deleteFile = F)
  
  output$cnn_acc_comparison <- renderPlotly({
    cnn_model_technique <-  c("Raw Image", "With Boundary", "Opening", "Power Law", "Denoise", "Thresholding")
    accuracy <- c(0.0757764, 0.074, 0.06956522, 0.08198758, 0.0757764, 0.06832298, 0.04968944, 0.05590062, 0.06335404, 0.0621118, 0.0621118, 0.0608696)
    data_type <- factor(c("Merged", "Merged", "Merged", "Merged", "Merged", "Merged", "Removed", "Removed", "Removed", "Removed", "Removed", "Removed"))
    
    cnn_data <- data.frame(cnn_model_technique, accuracy, data_type)
    p3 <- ggplot(data = cnn_data, aes(x = cnn_model_technique, y = accuracy, fill = data_type)) + 
      geom_bar(stat = 'identity', position = position_dodge()) + 
      labs(x = "Neural Network Model", y = "Accuracy", title = "Accuracies of CNN Deep Learning Models\non Image Preprocessing Techniques (Merged)", fill = "CNN Model")
    
    ggplotly(p3)
  })
  
  output$rf_acc_comparison <- renderPlotly({
    rf_model_technique <-  c("With Boundary", "Opening", "Power Law", "Denoise", "Thresholding")
    accuracy <- c(0.826, 0.831, 0.821, 0.832, 0.813, 0.842, 0.839, 0.842, 0.831, 0.833)
    data_type <- factor(c("Merged", "Merged", "Merged", "Merged", "Merged", "Removed", "Removed", "Removed", "Removed", "Removed"))
    
    df <- data.frame(rf_model_technique, accuracy, data_type)
    p <- ggplot(data = df, aes(x = rf_model_technique, y = accuracy, fill = data_type)) + 
      geom_bar(stat = 'identity', position = position_dodge()) + 
      ggtitle("Comparison of Random Forest Machine Learning Models\nby their Image Preprocessing Technique") + 
      xlab("Random Forest Model") + 
      ylab("Accuracy")
    
    ggplotly(p)
  })
  
  output$rf_boxplot <- renderPlotly({
    cv_acc_rff_bound = c(0.8152941, 0.7917647, 0.8376471,  0.8000000, 0.8000000)
    cv_acc_rff_denoise = c(0.7952941, 0.8470588, 0.8329412,  0.8141176, 0.7858824)
    cv_acc_rff_opening = c(0.8152941, 0.7917647, 0.8376471,  0.8000000, 0.8000000)
    
    cv_acc_rff_power = c(0.7988235, 0.8200000, 0.8317647,  0.8411765, 0.8117647)
    
    cv_acc_rff_thresh = c(0.8070588, 0.8176471, 0.7929412,  0.8082353, 0.8364706)
    barplot_df = data.frame(accuracy = c(cv_acc_rff_bound,
                                         cv_acc_rff_denoise, 
                                         cv_acc_rff_opening,
                                         cv_acc_rff_power,
                                         cv_acc_rff_thresh),
                            model = c(rep("Boundary", length(cv_acc_rff_bound)),
                                      rep("Denoise", length(cv_acc_rff_denoise)),
                                      rep("Opening", length(cv_acc_rff_opening)),
                                      rep("Power Law", length(cv_acc_rff_power)),
                                      rep("Thresholding", length(cv_acc_rff_thresh))))
    
    p2 <- ggplot(data = barplot_df, aes(x = model, y = accuracy, fill = model)) + 
      geom_boxplot() + 
      labs(x = "Random Forest Model", y = "5-fold CV Accuracy", title = "Distribution of 5-fold CV accuracies\nfor Random Forest Models on Image Preprocessing Techniques (Merged)", fill = "RF Model")
    
    ggplotly(p2)
  })
  
}


shinyApp(ui = ui, server = server)