library(shiny)
library(plotly)
library(png)

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
              "With Opening and boundary ", 
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
          fluidRow(
            # Output: Histogram ----
            # column(width = 6, plotlyOutput("plot1")),
            # column(width = 6, plotlyOutput("plot2"))
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
              "With Opening and boundary ", 
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
          fluidRow(
            # Output: Histogram ----
            # column(width = 6, plotlyOutput("plot1")),
            # column(width = 6, plotlyOutput("plot2"))
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
    img <- readPNG(input$filename)
    return(img)
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
        
