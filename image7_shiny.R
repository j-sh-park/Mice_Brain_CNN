library(shiny)
library(plotly)
library(png)
library(keras)
library(EBImage)
library(SpatialPack)
library(pracma)

# NOTE: Need an 'images' and 'cnn_models' folder in your working directory 
addResourcePath(prefix = "imgResources", directoryPath = "./images")
addResourcePath(prefix = "cnn_models", directoryPath = "./cnn_models")
addResourcePath(prefix = "rf_models_merged", directoryPath = "./RF Models (Merged)")
#addResourcePath(prefix = "rf_models_removed", directoryPath = "./RF Models (Removed)")
#NOTE: rename inputID="filename" to "filename_merge"(line67) and "filename_remove"(line111) to avoid conflics? 

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
  } else if (img_technique == "With everything"){
    #NOTE: we haven't decide what's the best filter(maybe we don't need this?)
    return(img)
  }
  # no boundaries / boundaries
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
    ("Image 7")
  ),
  
  tabsetPanel(
    # First tab
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
              "With Denoise and boundary", 
              "With everything"
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
          textOutput(outputId = "chosen_model"),
          textOutput(outputId = "chosen_technique"),
          fluidRow(
            # Output: Histogram ----
            # column(width = 6, plotlyOutput("plot1")),
            # column(width = 6, plotlyOutput("plot2"))
            column(6, offset = 3, plotOutput(outputId ='preprocessed_img', width='250px', height='250px'))
          )
        )
      )
    ),
    
    # Second tab
    tabPanel(
      "Remove Cluster",
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
              "With Denoise and boundary", 
              "With everything"
            )
          ),
          selectInput(
            inputId = "img_model", 
            label = "Choose your model",
            choices = c("Random Forest", "CNN")
          ),
          actionButton("go", "Run")
        ),
        # Main panel for displaying outputs ----
        mainPanel(
          #textOutput(outputId = "prediction"),
          fluidRow(
            # Output: Histogram ----
            # column(width = 6, plotlyOutput("plot1")),
            # column(width = 6, plotlyOutput("plot2"))
            #column(6, offset = 3, plotOutput(outputId ='raster', width='100px', height='100px'))
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
    } else if (input$img_technique == "With everything") {
      loaded_model = load_model_weights_hdf5(model, 'alexnet_merged_boundaries_weights.h5')
    } else {
      # no boundaries
      loaded_model = load_model_weights_hdf5(model, 'alexnet_merged_boundaries_weights.h5')
    }
    
    return(loaded_model)
  }
  
  add_model_weights_removed<- function(model) {
    if (input$img_technique == "With boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_removed_boundaries_weights.h5')
    } else if (input$img_technique == "With Power Law and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_removed_17_augmented_power_boundaries.h5')
    } else if (input$img_technique == "With Thresholding and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_removed_17_thresholding_boundaries.h5')
    } else if (input$img_technique == "With Opening and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_opening_removed_weights.h5')
    } else if (input$img_technique == "With Denoise and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_denoise_removed_weights.h5')
    } else if (input$img_technique == "With everything") {
      loaded_model = load_model_weights_hdf5(model, 'alexnet_merged_boundaries_weights.h5')
    } else {
      # no boundaries
      loaded_model = load_model_weights_hdf5(model, 'alexnet_merged_boundaries_weights.h5')
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
  
  cnn_prediction <- function(img_technique) {
    img = convert_img(data(), img_technique)
    if (img_technique != 'No boundaries or techniques') {
      # apply boundaries
      req(input$boundaries_file)
      cell_boundaries = read.csv(input$boundaries_file$datapath)
      img_inside = apply_boundary(img, cell_boundaries)
      img_resized = mask_resize(img, img_inside, 224, 224)
    } else {
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
  
  rf_prediction <- function(img_technique) {
    
    img = convert_img(data(), img_technique)
    if (img_technique == "With boundary") {
      loaded_model = readRDS("RF Models (Merged)/rf_boundaries_features.rds")
    } else if (img_technique == "With Power Law and boundary") {
      loaded_model = readRDS("rf_models_merged/rf_power_features.rds")
    } else if (img_technique == "With Thresholding and boundary") {
      loaded_model = readRDS("rf_models_merged/rf_thresholding_features.rds")
    } else if (img_technique == "With Opening and boundary") {
      loaded_model = readRDS("rf_models_merged/rf_opening_features.rds")
    } else if (img_technique == "With Denoise and boundary") {
      loaded_model = readRDS("rf_models_merged/rf_denoise_features.rds")
    } else if (img_technique == "With everything") {
      # NOTE: NOT YET CHOSEN
      loaded_model = readRDS("rf_models_merged/rf_boundaries_features.rds")
    } else {
      # no boundaries
      return("NA - Random Forest cannot compute this.")
    }
    
    if (!(is.null(input$boundaries_file))) {
      cell_boundaries = read.csv(input$boundaries_file$datapath)
      img_inside = apply_boundary(img, cell_boundaries)
      img_mask = img*img_inside
      xf = computeFeatures(img_inside, img_mask, expandRef = NULL)
      rownames(xf) <- colnames(computeFeatures(img_inside, img_mask, expandRef = NULL))
      res = loaded_model %>% predict(t(xf))
      return(res)
    }
    return(NULL)
  }
  
  output$prediction <- renderText({
    # waits for technique, model
    req(input$img_technique, input$img_model)
    predicted_class = NULL
    
    if (input$img_model == 'CNN') {
      if (!(is.null(input$boundaries_file)) && input$img_technique != 'No boundaries or techniques') {
        predicted_class = cnn_prediction(input$img_technique)
      }
      predicted_class = cnn_prediction(input$img_technique)
    } else {
      # random forest
      predicted_class = rf_prediction(input$img_technique)
      #predicted_class = "hello i need help with rf lmao"
    }
    
    if (is.null(predicted_class)) {
      paste0("Please choose a model and technique. If you have chosen techniques with boundaries, please include the CSV file containing the boundaries.")
    } else {
      paste0("The predicted cluster is: ", predicted_class)
    }
    
  })
  
  output$og_image <- renderPlot({
    req(data())
    plot(data(), all=FALSE)
  })
  
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
}


shinyApp(ui = ui, server = server)