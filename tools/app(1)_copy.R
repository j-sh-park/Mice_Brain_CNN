library(shiny)
library(plotly)
library(png)
library(EBImage)
library(SpatialPack)

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
          # Display uploaded file name
          verbatimTextOutput("uploaded_file"),
          
          # Display message indicating successful upload
          textOutput("upload_message"),
          
          fluidRow(
            # Output: Histogram ----
            column(width = 6, plotlyOutput("plot2"))
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
          verbatimTextOutput("uploaded_file"),
          
          # Display message indicating successful upload
          textOutput("upload_message"),
          
          fluidRow(
            # Output: Histogram ----
            column(width = 6, plotlyOutput("plot1"))
          )
        )
      )
    ),
      #Third tab
      tabPanel(
        "Comparision",
        # Sidebar layout with input and output definitions ----
        sidebarLayout(
          fluidRow(
          # Sidebar panel for inputs ----
          ),
          # Main panel for displaying outputs ----
          mainPanel(
            fluidRow(
              # Output: Histogram ----
              column(width = 6, plotlyOutput("plot1")),
              column(width = 6, plotlyOutput("plot2"))
            )
          )
        )
      )
  )
)

# Define server logic required to draw a histogram ----
server <- function(input, output) {
  
  output$uploaded_file <- renderPrint({
    if (!is.null(input$filename)) {
      paste("Uploaded file name: ", input$filename)
    }
  })
  
  output$upload_message <- renderText({
    if (!is.null(input$filename)) {
      "File uploaded successfully."
    }
  })
  
  data <- reactive({
    if(is.null(input$filename)) return(NULL)
    img <- readPNG(input$filename)
    return(img)
  })
  
  
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
}
      

shinyApp(ui = ui, server = server)


