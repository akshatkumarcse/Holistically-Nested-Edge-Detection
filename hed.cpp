#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/layer.details.hpp>


class CropLayer : public cv::dnn::Layer
{
public:
    CropLayer(const cv::dnn::LayerParams &params) : Layer(params)
    {
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new CropLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &internals) const CV_OVERRIDE
    {
        CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
        std::vector<int> outShape(4);
        outShape[0] = inputs[0][0];  // batch size
        outShape[1] = inputs[0][1];  // number of channels
        outShape[2] = inputs[1][2];
        outShape[3] = inputs[1][3];
        outputs.assign(1, outShape);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {

        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& inp = inputs[0];
        cv::Mat& out = outputs[0];

        int ystart = (inp.size[2] - out.size[2]) / 2;
        int xstart = (inp.size[3] - out.size[3]) / 2;
        int yend = ystart + out.size[2];
        int xend = xstart + out.size[3];

        const int batchSize = inp.size[0];
        const int numChannels = inp.size[1];
        const int height = out.size[2];
        const int width = out.size[3];

        int sz[] = { (int)batchSize, numChannels, height, width };
        out.create(4, sz, CV_32F);
        for(int i=0; i<batchSize; i++)
        {
            for(int j=0; j<numChannels; j++)
            {
                cv::Mat plane(inp.size[2], inp.size[3], CV_32F, inp.ptr<float>(i,j));
                cv::Mat crop = plane(cv::Range(ystart,yend), cv::Range(xstart,xend));
                cv::Mat targ(height, width, CV_32F, out.ptr<float>(i,j));
                crop.copyTo(targ);
            }
        }
    }
};



int main( int argc, char* argv[] )
{
    CV_DNN_REGISTER_LAYER_CLASS(Crop, CropLayer);   //In the line below, enter the Absolute Path to .prototxt and .caffemodel file.
    cv::dnn::Net net = cv::dnn::readNet("Path to .prototxt file", "Path to .caffemodel");
    


    /* For Live video feed.

    cv::VideoCapture cap;
     // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0))
        return 0;

    for(;;)
    {
          cv::Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream

          cv::Size reso(500,500);
          cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, reso, cv::Scalar(104.00698793, 116.66876762, 122.67891434), false, false);

          //cv::resize(out.reshape(1, reso.height), out, frame.size());
        

          net.setInput(blob);
          cv::Mat out = net.forward();
        
          cv::imshow("This is you, smile! :)", frame);
          if( cv::waitKey(10) == 27 ) break; // stop capturing by pressing ESC 
    }
    
    */


    //For Single Images. Works Perfectly
    ///*
    cv::Mat img = cv::imread("../dependencies/dog.jpg");    //Path to the image
    cv::Size reso(500,500);
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, reso, cv::Scalar(104.00698793, 116.66876762, 122.67891434), false, false);
    net.setInput(blob);
    cv::Mat out = net.forward();
    cv::resize(out.reshape(1, reso.height), out, img.size());
    cv::imshow("Image with HED", out);
    cv::waitKey();


    //*/

    return 0;
}

