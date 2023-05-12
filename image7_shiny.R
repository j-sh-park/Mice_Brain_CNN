library(shiny)
library(plotly)
library(png)
library(keras)
library(EBImage)
library(SpatialPack)

# NOTE: Need an 'images' and 'cnn_models' folder in your working directory 
addResourcePath(prefix = "imgResources", directoryPath = "./images")
addResourcePath(prefix = "cnn_models", directoryPath = "./cnn_models")
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
  if (img_technique == "With boundary"){
    return(img)
  } else if (img_technique == "With Power Law and boundary"){
    return(power_filter(img))
  } else if (img_technique == "With Opening and boundary") {
    return(opening_filter(img))
  } else if (img_technique == "With Denoise and boundary") {
    return(denoise_filter(img))
  } else if (img_technique == "With everything"){
    #NOTE: we haven't decide what's the best filter(maybe we don't need this?)
    return(img)
  }
  
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
          selectInput(
            inputId = "img_model", 
            label = "Choose your model",
            choices = c("Random Forest", "CNN")
          ),
          actionButton("go", "Run")
        ),
        # Main panel for displaying outputs ----
        mainPanel(
          textOutput(outputId = "prediction"),
          fluidRow(
            # Output: Histogram ----
            # column(width = 6, plotlyOutput("plot1")),
            # column(width = 6, plotlyOutput("plot2"))
            column(6, offset = 3, plotOutput(outputId ='raster', width='100px', height='100px'))
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
    #png::readPNG(input$filename$datapath)
    img <- readImage(input$filename$datapath)
    
    return(img)
  })
  
  add_model_weights_merged <- function(model) {
    if (input$img_technique == "With boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_merged_boundaries_weights.h5')
    } else if (input$img_technique == "With Power Law and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'alexnet_merged_boundaries_weights.h5')
    } else if (input$img_technique == "With Thresholding and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'alexnet_merged_boundaries_weights.h5')
    } else if (input$img_technique == "With Opening and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'alexnet_opening_merged_weights.h5')
    } else if (input$img_technique == "With Denoise and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_denoise_merged_weights.hdf5')
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
      loaded_model = load_model_weights_hdf5(model, 'cnn_models/alexnet_merged_boundaries_weights.h5')
    } else if (input$img_technique == "With Power Law and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'alexnet_merged_boundaries_weights.h5')
    } else if (input$img_technique == "With Thresholding and boundary") {
      loaded_model = load_model_weights_hdf5(model, 'alexnet_merged_boundaries_weights.h5')
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
  
  
  output$prediction <- renderText({
    # waits for technique, model then run button
    req(input$img_technique, input$img_model, input$go)
    if (input$img_model == 'CNN') {
      img = convert_img(data(), input$img_technique)
      img_resized = resize(img, 224, 224)
      x <- array(dim=c(1, 224, 224, 1))
      x[1,,,1] <- img_resized@.Data
      input_shape = dim(x)[2:4]
      model = create_model(input_shape = input_shape)
      loaded_model = add_model_weights_merged(model)
      
      res = loaded_model %>% predict(x)
      predicted_class = apply(res, 1, which.max)
      paste0("The predicted cluster is ", predicted_class)
    } else if (input$img_model == 'Random Forest') {
      # insert randomforest code here
    }

  })
  
  output$raster <- renderPlot({
    req(data())
    plot(data(), all=FALSE)
  })
  
  
  rf_no_b_t <- function(filename) {
    #img <- readPNG(filename)
    # apply the loaded preprocessing method to the image
    return(img_processed)
  }
  
  rf_boundaries <- function(filename) {
    # load the preprocessing method from the RDS file
    rf_boundaries_method <- readRDS("rf_boundaries_features.rds")
    # read the image file using the png package
    #img <- readPNG(filename)
    # apply the loaded preprocessing method to the image
    img_processed <- rf_boundaries_method(img)
    return(img_processed)  
  }
  
  rf_power <- function(filename) {
    # load the preprocessing method from the RDS file
    rf_power_method <- readRDS("rf_power_features.rds")
    # read the image file using the png package
    #img <- readPNG(filename)
    # apply the loaded preprocessing method to the image
    img_processed <- rf_power_method(img)
    return(img_processed)  
  }
  
  rf_threshold <- function(filename) {
    # load the preprocessing method from the RDS file
    rf_threshold_method <- readRDS("rf_thresholding_features.rds")
    # read the image file using the png package
    img <- readPNG(filename)
    # apply the loaded preprocessing method to the image
    img_processed <- rf_threshold_method(img)
    return(img_processed) 
  }
  
  rf_opening <- function(filename) {
    # load the preprocessing method from the RDS file
    rf_opening_method <- readRDS("rf_opening_features.rds")
    # read the image file using the png package
    img <- readPNG(filename)
    # apply the loaded preprocessing method to the image
    img_processed <- rf_opening_method(img)
    return(img_processed) 
  }
  
  rf_denoise <- function(filename) {
    # load the preprocessing method from the RDS file
    rf_denoise_method <- readRDS("rf_denoise_features.rds")
    # read the image file using the png package
    img <- readPNG(filename)
    # apply the loaded preprocessing method to the image
    img_processed <- rf_denoise_method(img)
    return(img_processed)  
  }
  
  rf_everything <- function(filename) {
    # load the preprocessing method from the RDS file
    rf_boundaries_features_method <- readRDS("rf_boundaries_features.rds")
    # read the image file using the png package
    img <- readPNG(filename)
    # apply the loaded preprocessing method to the image
    img_processed <- rf_boundaries_features_method(img)
    return(img_processed)  
  }
  
  process_file <- function(file, func) {
    if (input$img_technique == "With boundary") {
      processed_file <- rf_boundaries(file)
    } else if (input$img_technique == "With Power Law and boundary") {
      processed_file <- rf_power(file)
    } else if (input$img_technique == "With Thresholding and boundary") {
      processed_file <- rf_threshold(file)
    } else if (input$img_technique == "With Opening and boundary") {
      processed_file <- rf_opening(file)
    } else if (input$img_technique == "With Denoise and boundary") {
      processed_file <- rf_denoise(file)
    } else if (input$img_technique == "With everything")
      processed_file <- rf_every(file)
      
  }
  
  output$plot <- renderPlot({
    plot(processed_file())
  })
}
      

shinyApp(ui = ui, server = server)
        
