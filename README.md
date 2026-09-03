# 🏥 Medical Report Intelligence Using AI and OCR

## 📌 Overview

Medical Report Intelligence is an AI-powered medical report processing application designed to simplify the process of extracting useful information from medical report images. Medical reports are often provided as scanned documents or images containing pathology parameters, test names, values, reference ranges, and other textual information. Manually reading and entering this information can be time-consuming and error-prone. This project addresses this problem by combining image preprocessing, Optical Character Recognition (OCR), Python-based processing, FastAPI REST APIs, and a Flutter application into an end-to-end system. The application allows a user to capture or provide a medical report image, processes the image to improve text recognition, extracts the textual information using Tesseract OCR, identifies relevant pathology parameters, and converts the extracted information into a structured JSON format that can be used for further AI-assisted analysis. The OCR pipeline achieved approximately 92% text extraction accuracy across more than 1,500 medical report images.

## 🎯 Objective

The main objective of this project is to build a practical system that can transform unstructured medical report images into structured and machine-readable information. Instead of requiring users to manually read and enter values from a report, the application provides an automated workflow in which the report image is captured, processed, analyzed using OCR, and converted into structured information. The project also demonstrates how a computer vision and OCR pipeline can be integrated with a backend API and a cross-platform mobile application to create a complete AI-powered application.

## 🔄 System Workflow

The complete workflow starts from the Flutter application, where the user captures or selects an image of a medical report. The selected image is sent to the backend through a REST API exposed by the FastAPI server. Once the backend receives the image, the Python processing pipeline loads the image and performs image preprocessing using OpenCV. Preprocessing is an important stage because medical reports can contain different layouts, image qualities, lighting conditions, fonts, and background noise. The image is therefore prepared before OCR so that the text-recognition process can work more effectively. After preprocessing, the processed image is passed to the Tesseract OCR engine, which detects and extracts the text present in the medical report. The extracted OCR text is then processed by the Python backend to identify relevant medical and pathology-related information. The system organizes the extracted parameters into a structured representation instead of leaving the information as raw OCR text. These extracted pathology parameters are then converted into structured JSON data, making the information easier for downstream processing and AI-assisted analysis. The FastAPI backend manages this complete processing workflow and returns the structured result through REST APIs to the Flutter application. Finally, the Flutter application receives the processed response and presents the extracted information to the user in a more usable format. In this way, the project connects image capture, computer vision, OCR, backend processing, structured data generation, and a mobile interface into one end-to-end workflow.

## 🧩 Architecture

The project can be understood as a pipeline consisting of several major components:

**User → Flutter Application → FastAPI REST API → Python Processing → OpenCV Preprocessing → Tesseract OCR → Text Processing → Pathology Parameter Extraction → Structured JSON → FastAPI Response → Flutter Application**

The Flutter application acts as the user-facing layer. It provides the interface for capturing or selecting medical report images and displaying the processed information. FastAPI acts as the backend API layer and receives requests from the application. Python handles the core processing logic, while OpenCV is used for image processing and preparation. Tesseract OCR performs the actual text extraction from the processed medical report image. The extracted information is then organized into structured JSON data containing the relevant pathology parameters, which can subsequently be used for downstream AI-assisted processing.

## 🛠️ Technologies Used

* **Python** – Core programming and processing logic
* **FastAPI** – Backend framework for creating REST APIs
* **OpenCV** – Image processing and preprocessing
* **Tesseract OCR** – Optical Character Recognition and text extraction
* **REST APIs** – Communication between the Flutter application and backend
* **Flutter** – Cross-platform mobile application development
* **Dart** – Programming language used with Flutter
* **JSON** – Structured representation of extracted report information

## 🔍 OCR Processing Pipeline

The OCR pipeline is one of the most important parts of the project. When a medical report image reaches the backend, the image is first loaded and prepared for processing. OpenCV is used to perform image preprocessing so that unnecessary visual noise can be reduced and the text can be made more suitable for OCR. The processed image is then passed to Tesseract OCR, which identifies characters and words from the report. The resulting OCR output provides the raw textual representation of the medical report. The backend then processes this extracted text to identify pathology-related parameters and organize them into structured information. Instead of returning only a block of extracted text, the system converts the relevant information into JSON so that it can be consumed programmatically by other parts of the application or used for downstream AI-assisted analysis.

## 📱 Application Workflow

From the user's perspective, the workflow is designed to be simple. The user opens the Flutter application and provides a medical report image through the available scanning or image-selection functionality. The application sends the image to the FastAPI backend using a REST API request. The backend receives the image and starts the OCR processing pipeline. OpenCV prepares the image for recognition, Tesseract extracts the text, and the Python processing layer organizes the extracted information. Relevant pathology parameters are converted into structured JSON data. The backend then sends the processed result back to the Flutter application through the REST API. The application receives the response and presents the extracted information to the user. This removes the need for manually transcribing information from the report and creates a more automated way of handling medical report data.

## 📊 Project Performance

The OCR pipeline was evaluated across more than **1,500 medical report images** and achieved approximately **92% text extraction accuracy**. This evaluation helped demonstrate the effectiveness of the implemented OCR and image-processing workflow across a large collection of medical report images.

## 🚀 Key Features

* Medical report image scanning and processing
* Automated OCR-based text extraction
* Image preprocessing using OpenCV
* Pathology parameter extraction
* Conversion of extracted information into structured JSON
* FastAPI-based REST backend
* Cross-platform Flutter application
* AI-assisted medical report analysis workflow
* Processing of 1,500+ medical report images
* Approximately 92% OCR text extraction accuracy

## 📂 Project Structure

```text
Health-assistent-app/
│
├── frontend/
│   └── Flutter application
│
├── backend/
│   ├── FastAPI application
│   ├── OCR processing
│   ├── Image preprocessing
│   └── JSON generation
│
├── images/
│   └── Sample medical report images
│
├── requirements.txt
├── README.md
└── ...
```

> The exact directory structure may vary depending on the current implementation of the repository.

## ⚙️ How the System Works End-to-End

The system follows an image-to-structured-data workflow. First, a medical report is captured or selected through the Flutter application. The application communicates with the FastAPI backend through REST APIs and sends the report image for processing. The FastAPI server receives the image and forwards it to the Python-based processing pipeline. OpenCV performs the required image preprocessing to prepare the report for OCR. Tesseract OCR then analyzes the processed image and extracts the textual content. The extracted text is passed through the backend processing logic, where relevant pathology parameters are identified and organized. The extracted information is transformed into structured JSON, allowing the result to be handled programmatically rather than as unstructured text. The FastAPI server returns the structured response to the Flutter application, where the information can be displayed to the user. This complete workflow creates a bridge between an image-based medical document and structured machine-readable information that can be used for further analysis.

## 💡 What I Learned

This project provided hands-on experience in developing an end-to-end AI-powered application rather than working with an isolated machine learning model. I gained practical experience with image preprocessing, OCR, Python-based data processing, REST API development, FastAPI backend development, JSON data generation, and Flutter application integration. The project also helped me understand how different components of an AI application communicate with each other, from receiving an image from a user interface to processing that image through an OCR pipeline and returning structured information through an API. Working with more than 1,500 medical report images also provided practical experience in evaluating OCR performance and understanding the challenges involved in extracting text from real-world document images.

## 🔮 Future Improvements

The system can be extended further by improving OCR accuracy for different medical report layouts and image qualities. Additional document preprocessing techniques could be introduced to improve recognition under difficult image conditions. The structured pathology data could also be connected to more advanced AI models for deeper report analysis, while additional validation could be introduced to improve the reliability of extracted values. The application could further support different types of medical documents and provide more flexible processing for reports with different layouts and formats.

## 👨‍💻 Project Highlights

* Built an end-to-end AI-powered medical report processing system
* Achieved **92% text extraction accuracy**
* Tested across **1,500+ medical report images**
* Implemented image preprocessing using **OpenCV**
* Implemented OCR using **Tesseract**
* Developed backend services using **FastAPI**
* Built REST APIs for report processing
* Converted extracted pathology parameters into **structured JSON**
* Developed a cross-platform application using **Flutter**

## 📜 Disclaimer

This project is intended for educational and software-development purposes. The extracted information should not be considered a medical diagnosis or a replacement for professional medical advice. Medical information should always be reviewed and interpreted by qualified healthcare professionals.

