using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

namespace FaceRecognition
{
    class Program
    {
        private static readonly string TrainingImagesFolder = @"D:\CsharpProjects\FaceRecognition\FaceRecognition\TrainingImages"; // Set the path for your training images
        private static readonly string RecognizerModelPath = @"D:\CsharpProjects\FaceRecognition\FaceRecognition\Resources\faceRecognizer.xml"; // Set the path for your recognizer model
        private static readonly EigenFaceRecognizer recognizer = new();

        private static CascadeClassifier? faceCascade;

        static void Main()
        {
            if(!Directory.Exists(TrainingImagesFolder))
                Directory.CreateDirectory(TrainingImagesFolder);

            // Initialize the face cascade classifier
            faceCascade = new CascadeClassifier(@"D:\CsharpProjects\FaceRecognition\FaceRecognition\Resources\haarcascade_frontalface_default.xml");

            // Call the function to capture training images
            //CaptureTrainingImages();

            // Train the recognizer with the captured images
            if (File.Exists(RecognizerModelPath))
                recognizer.Read(RecognizerModelPath);
            else
                TrainRecognizer();

            // Start the face recognition process
            StartFaceRecognition_1();
            //StartFaceRecognition_2();

            //Evaluate Prediction Accuracy
            //EvaluateModel(TrainingImagesFolder);
        }

        private static void CaptureTrainingImages()
        {
            using var capture = new VideoCapture(0);
            Mat frame = new();
            int count = 0;

            Console.WriteLine("Press 's' to save image and 'ESC' to exit.");

            while (true)
            {
                capture.Read(frame);
                if (!frame.IsEmpty)
                {
                    CvInvoke.Imshow("Capture Training Images", frame);
                    var key = CvInvoke.WaitKey(10);

                    if (key == 's') // Save image on pressing 's'
                    {
                        // Resize the frame to 100x100 before saving
                        var resizedFrame = new Image<Bgr, byte>(frame).Resize(100, 100, Inter.Linear);
                        string filePath = Path.Combine(TrainingImagesFolder, $"image_{count}.jpg");
                        try
                        {
                            resizedFrame.Save(filePath);
                            Console.WriteLine($"Image saved: {filePath}");
                            count++;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error saving image: {ex.Message}");
                        }
                    }
                    else if (key == 27) // ESC key to exit
                    {
                        break;
                    }
                }
            }
        }

        private static void TrainRecognizer()
        {

            if (File.Exists(RecognizerModelPath)) 
                File.Delete(RecognizerModelPath);

            var trainingImages = new List<Mat>();
            var labels = new List<int>();
            int label = 0;

            string[] imageFiles = Directory.GetFiles(TrainingImagesFolder, "*.jpg");
            if (imageFiles.Length == 0)
            {
                Console.WriteLine("No training images found. Please capture some images first.");
                return;
            }

            foreach (var filePath in imageFiles)
            {
                // Convert to Mat and resize to 100x100 for consistency
                var img = new Image<Gray, byte>(filePath).Resize(100, 100, Inter.Linear).Mat;
                trainingImages.Add(img);
                labels.Add(label); // Assign a label (increment for each image)
                label++;
            }

            using var trainingImagesArray = new VectorOfMat(trainingImages.ToArray());
            using var labelsArray = new VectorOfInt(labels.ToArray());
            if (trainingImagesArray.Size == 0 || labelsArray.Size == 0)
            {
                Console.WriteLine("No images or labels were found for training.");
                return;
            }

            recognizer.Train(trainingImagesArray, labelsArray);
            recognizer.Write(RecognizerModelPath);
            Console.WriteLine("Training complete and model saved.");
        }

        private static void StartFaceRecognition_1()
        {
            double thresholdDistance = 12000; // Adjust this value based on testing
            using (var capture = new VideoCapture(0))
            {
                Mat frame = new Mat();
                while (true)
                {
                    capture.Read(frame);
                    if (!frame.IsEmpty)
                    {
                        var image = frame.ToImage<Bgr, byte>();
                        var grayImage = image.Convert<Gray, byte>();

                        var faces = faceCascade?.DetectMultiScale(grayImage, 1.1, 10, new Size(30, 30), Size.Empty);
                        foreach (var face in faces!)
                        {
                            image.Draw(face, new Bgr(Color.Green), 2);
                            var faceMat = grayImage.GetSubRect(face).Resize(100, 100, Inter.Linear);

                            var result = recognizer.Predict(faceMat);

                            // Optionally, display the predicted label
                            Console.WriteLine($"Predicted label: {result.Label} with confidence: {result.Distance}");

                            // Calculate confidence percentage based on threshold distance
                            double confidencePercentage = Math.Max(0, 100 - (result.Distance / thresholdDistance) * 100);
                            confidencePercentage = Math.Min(100, confidencePercentage);

                            // Display predicted label and confidence percentage
                            //Console.WriteLine($"Predicted label: {result.Label} with confidence: {confidencePercentage}%");

                            string labelText = $"ID: {result.Label}, Conf: {confidencePercentage:F2}%";
                            CvInvoke.PutText(image, labelText, new Point(face.X, face.Y - 10),
                                             FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 1);
                        }

                        CvInvoke.Imshow("Real-Time Face Detection", image);
                        if (CvInvoke.WaitKey(10) == 27) break;
                    }
                }
            }
        }

        private static void StartFaceRecognition_2()
        {
            double thresholdDistance = 12000; // Adjust this value based on testing
            using (var capture = new VideoCapture(0))
            {
                Mat frame = new();
                while (true)
                {
                    capture.Read(frame);
                    if (!frame.IsEmpty)
                    {
                        var image = frame.ToImage<Bgr, byte>();
                        var grayImage = image.Convert<Gray, byte>();

                        var faces = faceCascade?.DetectMultiScale(grayImage, 1.1, 10, new Size(30, 30), Size.Empty);
                        foreach (var face in faces!)
                        {
                            // Draw rectangle around detected face
                            image.Draw(face, new Bgr(Color.Green), 2);

                            // Extract the face region
                            var faceMat = new Mat(grayImage.Mat, face); // Get the face region

                            // Resize the face region to 100x100
                            var resizedFaceMat = new Mat();
                            CvInvoke.Resize(faceMat, resizedFaceMat, new Size(100, 100), interpolation: Emgu.CV.CvEnum.Inter.Linear);

                            // Predict the identity of the detected face
                            var result = recognizer.Predict(resizedFaceMat);

                            // Optionally, display the predicted label
                            Console.WriteLine($"Predicted label: {result.Label} with confidence: {result.Distance}");

                            // Calculate confidence percentage based on threshold distance
                            double confidencePercentage = Math.Max(0, 100 - (result.Distance / thresholdDistance) * 100);
                            confidencePercentage = Math.Min(100, confidencePercentage);

                            // Display predicted label and confidence percentage
                            string labelText = $"ID: {result.Label}, Conf: {confidencePercentage:F2}%";
                            CvInvoke.PutText(image, labelText, new Point(face.X, face.Y - 10),
                                             FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 1);
                        }

                        CvInvoke.Imshow("Real-Time Face Detection", image);
                        if (CvInvoke.WaitKey(10) == 27) break;  // Press ESC to exit
                    }
                }
            }
        }


        private static void EvaluateModel(string testImagesFolder)
        {
            var testImageFiles = Directory.GetFiles(testImagesFolder, "*.jpg");
            int correctPredictions = 0;
            int totalPredictions = testImageFiles.Length;

            foreach (var filePath in testImageFiles)
            {
                // Load the test image
                var testImage = new Image<Gray, byte>(filePath).Mat;
                var detectedFaces = faceCascade.DetectMultiScale(testImage, 1.1, 10, Size.Empty);

                // Assuming you have a function to extract the actual label from the file name
                int actualLabel = GetLabelFromFileName(filePath);

                foreach (var face in detectedFaces)
                {
                    // Get the face region
                    var faceRegion = new Mat(testImage, face);

                    // Resize the face region to 100x100
                    var resizedFaceMat = new Mat();
                    CvInvoke.Resize(faceRegion, resizedFaceMat, new Size(100, 100), interpolation: Emgu.CV.CvEnum.Inter.Linear);

                    // Predict the identity of the detected face
                    var result = recognizer.Predict(resizedFaceMat);
                    int predictedLabel = result.Label;

                    // Check if prediction is correct
                    if (predictedLabel == actualLabel)
                    {
                        correctPredictions++;
                    }
                }
            }

            // Calculate accuracy
            double accuracy = (double)correctPredictions / totalPredictions * 100;
            Console.WriteLine($"Accuracy: {accuracy}% ({correctPredictions}/{totalPredictions})");
        }
        // Helper method to extract label from the file name
        private static int GetLabelFromFileName(string filePath)
        {
            // Example: If the file name is "image_1.jpg", return 1
            // Adjust this method based on your actual file naming convention
            string fileName = Path.GetFileNameWithoutExtension(filePath);
            return int.Parse(fileName.Split('_')[1]);
        }


    }
}