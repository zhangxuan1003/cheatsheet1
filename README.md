# cheatsheet1
//2D & 3D
template<int Dim>

void Convert( std::string inputFilename, std::string outputFilename,int dimension, float sigma)

{
  using PixelType = unsigned char;
  using ImageType = itk::Image< PixelType, Dim >;
  using ReaderType = itk::ImageFileReader< ImageType >;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(inputFilename);
  reader->Update();

  using FilterType = itk::RecursiveGaussianImageFilter<ImageType ,ImageType >;
  typename FilterType::Pointer filterX = FilterType::New();
  typename FilterType::Pointer filterY = FilterType::New();

  filterX->SetDirection( 0 ); // 0 --> X direction
  filterY->SetDirection( 1 ); // 1 --> Y direction
  filterX->SetOrder( FilterType::ZeroOrder );
  filterY->SetOrder( FilterType::ZeroOrder );
  filterX->SetInput( reader->GetOutput() );
  filterY->SetInput( filterX->GetOutput() );
  filterX->SetSigma( sigma );
  filterY->SetSigma( sigma );

  if(dimension==2){
    using WriterType = itk::ImageFileWriter< ImageType >;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( outputFilename );
    writer->SetInput( filterY->GetOutput() );
    writer->Update();
  }
  else if (dimension==3) {
    typename FilterType ::Pointer filterZ = FilterType::New();
    filterZ->SetDirection(2);
    filterZ->SetOrder(FilterType::ZeroOrder);
    filterZ->SetInputImage(filterY->GetOutput());
    filterZ->SetSigma(sigma);
    using WriterType = itk::ImageFileWriter< ImageType >;
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput(filterZ->GetOutput());
    writer->Update();
  }
  
  int main( int argc, char* argv[] )
{
  if( argc < 5 )
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageDimension sigmaValue inputImageFile outputImageFile" << std::endl;
    return EXIT_FAILURE;
  }
  const unsigned int inputDim = std::stoi( argv[1] );
  const float sigmaValue = std::stod( argv[2] );
  const std::string inputFilename = argv[3];
  const std::string outputFilename = argv[4];

  if (inputDim==2) {
    Convert<2>(inputFilename, outputFilename,inputDim,sigmaValue);
  }
  else if (inputDim==3) {
  Convert<3>(inputFilename, outputFilename,inputDim,sigmaValue);
  }
  return EXIT_SUCCESS;
}

//TRANSLATION IMAGE REGISTRATION


#include "itkImageRegistrationMethodv4.h"
#include "itkTranslationTransform.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "registerCLP.h"

int main( int argc, char *argv[] )

{
  PARSE_ARGS;
  std::cout << "fixedImageFile: " << fixedImageFile << std::endl;
  std::cout << "movingImageFile: " << movingImageFile << std::endl;
  std::cout << "outputImageFile: " << outputImageFile << std::endl;
  std::cout << "differenceImageBeforeFile: " << differenceImageBeforeFile << std::endl;
  std::cout << "differenceImageAfterFile: " << differenceImageAfterFile << std::endl;

  using FixedImageType = itk::Image< float , 3 >;
  using MovingImageType = itk::Image< float, 3 >;
  
  using FixedImageReaderType = itk::ImageFileReader< FixedImageType  >;
  auto fixedImageReader = FixedImageReaderType::New();
  fixedImageReader->SetFileName( fixedImageFile );
  
  using MovingImageReaderType = itk::ImageFileReader< MovingImageType >;
  auto movingImageReader = MovingImageReaderType::New();
  movingImageReader->SetFileName( movingImageFile );

  using TransformType = itk::TranslationTransform< double, 3 >;
  auto initialTransform = TransformType::New();

  using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
  auto optimizer = OptimizerType::New();
  optimizer->SetLearningRate( 4 );
  optimizer->SetMinimumStepLength( 0.01 );
  optimizer->SetRelaxationFactor( 0.5 );
  optimizer->SetNumberOfIterations( 200 );

  using MetricType = itk::MeanSquaresImageToImageMetricv4<FixedImageType,MovingImageType >;
  auto metric = MetricType::New();
  
  using FixedLinearInterpolatorType = itk::LinearInterpolateImageFunction<FixedImageType,double >;
  using MovingLinearInterpolatorType = itk::LinearInterpolateImageFunction<MovingImageType, double >;

  FixedLinearInterpolatorType::Pointer fixedInterpolator =FixedLinearInterpolatorType::New();
  MovingLinearInterpolatorType::Pointer movingInterpolator = MovingLinearInterpolatorType::New();

  metric->SetFixedInterpolator( fixedInterpolator );
  metric->SetMovingInterpolator( movingInterpolator );

  using RegistrationType = itk::ImageRegistrationMethodv4<FixedImageType,MovingImageType >;
  auto registration = RegistrationType::New();
  registration->SetOptimizer(optimizer);
  registration->SetMetric(metric);
  registration->SetFixedImage( fixedImageReader->GetOutput() );
  registration->SetMovingImage( movingImageReader->GetOutput() );





  auto movingInitialTransform = TransformType::New();
  TransformType::ParametersType initialParameters( movingInitialTransform->GetNumberOfParameters() );
  registration->SetMovingInitialTransform( movingInitialTransform );
  initialParameters[0]=0.0;
  initialParameters[1]=0.0;
  initialParameters[2]=0.0;
  movingInitialTransform->SetParameters( initialParameters );
  registration->SetMovingInitialTransform( movingInitialTransform );

  auto identityTransform = TransformType::New();
  identityTransform->SetIdentity();
  registration->SetFixedInitialTransform( identityTransform );

  constexpr unsigned int numberOfLevels = 1;
  registration->SetNumberOfLevels( numberOfLevels );
  RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize( 1 );
  shrinkFactorsPerLevel[0] = 1;
  registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );

  RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize( 1 );
  smoothingSigmasPerLevel[0] = 0;
  registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );

try
  {
    registration->Update();
    std::cout << "Optimizer stop condition: "
    << registration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
  }

  catch( itk::ExceptionObject & err )
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }



  auto transform = registration->GetTransform();
  auto finalParameters = transform->GetParameters();
  auto translationAlongX = finalParameters[0];
  auto translationAlongY = finalParameters[1];
  auto translationAlongZ = finalParameters[2];
  //TODO: Get all parameters
  auto numberOfIterations = optimizer->GetCurrentIteration();
  auto bestValue = optimizer->GetValue();
  std::cout << "Result = " << std::endl;
  std::cout << " Translation X = " << translationAlongX  << std::endl;
  std::cout << " Translation Y = " << translationAlongY  << std::endl;
  std::cout << " Translation Z = " << translationAlongZ  << std::endl;
  //TODO: Print other parameters
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue          << std::endl

  using CompositeTransformType = itk::CompositeTransform<double,3 >;
  auto outputCompositeTransform = CompositeTransformType::New();
  outputCompositeTransform->AddTransform( movingInitialTransform );
  outputCompositeTransform->AddTransform(registration->GetModifiableTransform() );
  using ResampleFilterType = itk::ResampleImageFilter<MovingImageType,FixedImageType >;
  auto resampler = ResampleFilterType::New();
  resampler->SetInput( movingImageReader->GetOutput() );
  resampler->SetTransform( outputCompositeTransform );
  auto fixedImage = fixedImageReader->GetOutput();
  resampler->SetUseReferenceImage( true );
  resampler->SetReferenceImage( fixedImage );
  resampler->SetDefaultPixelValue( 100 );
  using OutputPixelType = unsigned char;
  using OutputImageType = itk::Image< OutputPixelType, 3 >;
  using CastFilterType = itk::CastImageFilter<FixedImageType,OutputImageType >;
  auto caster = CastFilterType::New();
  caster->SetInput( resampler->GetOutput() );

  using WriterType = itk::ImageFileWriter< OutputImageType >;
  auto writer = WriterType::New();
  writer->SetFileName( outputImageFile );
  writer->SetInput( caster->GetOutput()   );
  writer->Update();
  
  using DifferenceFilterType = itk::SubtractImageFilter<FixedImageType,FixedImageType,FixedImageType >;
  auto difference = DifferenceFilterType::New();
  difference->SetInput1( fixedImageReader->GetOutput() );
  difference->SetInput2( resampler->GetOutput() );
  using RescalerType = itk::RescaleIntensityImageFilter<FixedImageType,OutputImageType >;
  auto intensityRescaler = RescalerType::New();
  intensityRescaler->SetInput( difference->GetOutput() );
  intensityRescaler->SetOutputMinimum( itk::NumericTraits< OutputPixelType >::min() );
  intensityRescaler->SetOutputMaximum( itk::NumericTraits< OutputPixelType >::max() );
  resampler->SetDefaultPixelValue( 1 );
  writer->SetInput( intensityRescaler->GetOutput() );
  writer->SetFileName( differenceImageAfterFile );
  writer->Update();
  resampler->SetTransform( identityTransform );
  writer->SetFileName( differenceImageBeforeFile );
  writer->Update();
  
  return EXIT_SUCCESS;
}

//3D Euler Transform
#include "itkImageRegistrationMethodv4.h"
#include "itkTranslationTransform.h"
#include "itkEuler3DTransform.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "HW2IterationLogger.h"
#include "registerCLP.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCenteredTransformInitializer.h"
#include "itkSubtractImageFilter.h"

int main( int argc, char *argv[] )
{
  PARSE_ARGS;
    std::cout << "fixedImageFile: " << fixedImageFile << std::endl;
    std::cout << "movingImageFile: " << movingImageFile << std::endl;
    std::cout << "outputImageFile: " << outputImageFile << std::endl;
    std::cout << "differenceImageAfterFile: " << differenceImageAfterFile << std::endl;
    
    constexpr unsigned int Dimension = 3;
    using PixelType = float;
    using ImageType = itk::Image<PixelType, 3>;
    using ReaderType = itk::ImageFileReader<ImageType>;
    ReaderType::Pointer fixedImageReader = ReaderType::New();
    fixedImageReader->SetFileName(fixedImageFile);
    fixedImageReader->Update();
    ReaderType::Pointer movingImageReader = ReaderType::New();
    movingImageReader->SetFileName(movingImageFile);
    movingImageReader->Update();
    
    using FixedImageType = itk::Image< PixelType, Dimension >;
    using MovingImageType = itk::Image< PixelType, Dimension >;
    using TransformType = itk::Euler3DTransform< double >;
    using OptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
    using MetricType = itk::MeanSquaresImageToImageMetricv4<FixedImageType,MovingImageType >;
    using RegistrationType = itk::ImageRegistrationMethodv4<FixedImageType,MovingImageType,TransformType>;
    
    MetricType::Pointer metric=MetricType::New();
    OptimizerType::Pointer optimizer=OptimizerType::New();
    RegistrationType::Pointer registration=RegistrationType::New();

    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);

    using FixedLinearInterpolatorType = itk::LinearInterpolateImageFunction<FixedImageType,double >;
    using MovingLinearInterpolatorType = itk::LinearInterpolateImageFunction<MovingImageType,double >;

    FixedLinearInterpolatorType::Pointer fixedInterpolator =FixedLinearInterpolatorType::New();
    MovingLinearInterpolatorType::Pointer movingInterpolator =MovingLinearInterpolatorType::New();

    metric->SetFixedInterpolator( fixedInterpolator );
    metric->SetMovingInterpolator( movingInterpolator );



    fixedImageReader->Update();
    movingImageReader->Update();
    FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();

    ImageType::SpacingType fixedSpacing=fixedImage->GetSpacing();
    ImageType::PointType fixedOrigin=fixedImage->GetOrigin();
    ImageType::RegionType fixedRegion=fixedImage->GetLargestPossibleRegion();
    ImageType::SizeType fixedSize=fixedRegion.GetSize();
    using VectorType=itk::Vector<double,3>;

    typename ImageType::IndexType Index1;
    Index1[0]=fixedSize[0];
    Index1[1]=fixedSize[1];
    Index1[2]=fixedSize[2];

    typename ImageType::PointType corner1;
    fixedImage->TransformIndexToPhysicalPoint( Index1, corner1 );
    
    using VectorType=itk::Vector<double,3>;
    VectorType centerFixed;
    centerFixed[0]=(fixedOrigin[0]+corner1[0])/2;
    centerFixed[1]=(fixedOrigin[1]+corner1[1])/2;
    centerFixed[2]=(fixedOrigin[2]+corner1[2])/2;

    MovingImageType::Pointer movingImage = movingImageReader->GetOutput();
    ImageType::SpacingType movingSpacing=movingImage->GetSpacing();
    ImageType::PointType movingOrigin=movingImage->GetOrigin();
    ImageType::RegionType movingRegion=movingImage->GetLargestPossibleRegion();
    ImageType::SizeType movingSize=movingRegion.GetSize();
    
    typename ImageType::IndexType Index2;
    Index2[0]=movingSize[0];
    Index2[1]=movingSize[1];
    Index2[2]=movingSize[2];
    
    typename ImageType::PointType corner2;
    movingImage->TransformIndexToPhysicalPoint( Index2, corner2 );
    VectorType centerMoving;
    centerMoving[0]=(movingOrigin[0]+corner2[0])/2;
    centerMoving[1]=(movingOrigin[1]+corner2[1])/2;
    centerMoving[2]=(movingOrigin[2]+corner2[2])/2;

    std::cout<< centerMoving<< std::endl;
    std::cout<< centerFixed<< std::endl;
    ImageType::SizeType physicalfixed;
    physicalfixed[0]=fixedSize[0]*fixedSpacing[0];
    physicalfixed[1]=fixedSize[1]*fixedSpacing[1];
    physicalfixed[2]=fixedSize[2]*fixedSpacing[2];

    ImageType::SizeType physicalmoving;
    physicalmoving[0]=movingSize[0]*movingSpacing[0];
    physicalmoving[1]=movingSize[1]*movingSpacing[1];
    physicalmoving[2]=movingSize[2]*movingSpacing[2];

    std::cout<<"the size of fixed image" <<physicalfixed<< std::endl;
    std::cout<<"the size of moving image" << physicalmoving<< std::endl;


    TransformType::Pointer initialTransform = TransformType::New();
    initialTransform->SetCenter( centerFixed );
    initialTransform->SetTranslation( centerMoving - centerFixed );
    initialTransform->SetRotation( 0.0,0.0,0.0);
    registration->SetMovingInitialTransform( initialTransform );

    constexpr unsigned int numberOfLevels = 3;
    RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
    shrinkFactorsPerLevel.SetSize( 3 );
    shrinkFactorsPerLevel[0] = 8;
    shrinkFactorsPerLevel[1] = 4;
    shrinkFactorsPerLevel[2] = 1;
    RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
    smoothingSigmasPerLevel.SetSize( 3 );
    smoothingSigmasPerLevel[0] = 4;
    smoothingSigmasPerLevel[1] = 2;
    smoothingSigmasPerLevel[2] = 0;
    
    registration->SetNumberOfLevels ( numberOfLevels );
    registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );
    registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
    registration->SetFixedImage(fixedImageReader->GetOutput());
    registration->SetMovingImage(movingImageReader->GetOutput());



    TransformType::Pointer identityTransform = TransformType::New();
    identityTransform->SetIdentity();
    registration->SetFixedInitialTransform( identityTransform );
    using OptimizerScalesType = OptimizerType::ScalesType;
    
    OptimizerScalesType optimizerScales(initialTransform->GetNumberOfParameters() );
    const double translationScale = 1.0 / 256;
    optimizerScales[0] = 1.0;
    optimizerScales[1] = 1.0;
    optimizerScales[2] = 1.0;
    optimizerScales[3] = translationScale;
    optimizerScales[4] = translationScale;
    optimizerScales[5] = translationScale;
    optimizer->SetScales( optimizerScales );

    optimizer->SetLearningRate( 0.2 );
    optimizer->SetMinimumStepLength( 0.01 );//0.01
    optimizer->SetRelaxationFactor( 0.5 );
    optimizer->SetNumberOfIterations( 500 );

    CommandIterationUpdate<OptimizerType>::Pointer observer = CommandIterationUpdate<OptimizerType>::New();
    optimizer->AddObserver( itk::IterationEvent(), observer );

    try
    {
        registration->Update();
        std::cout << "Optimizer stop condition: "
                  << registration->GetOptimizer()->GetStopConditionDescription()
                  << std::endl;
    }
    catch( itk::ExceptionObject & err )
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    using CompositeTransformType = itk::CompositeTransform<double,Dimension >;
    CompositeTransformType::Pointer outputCompositeTransform =CompositeTransformType::New();
    outputCompositeTransform->AddTransform( initialTransform );
    outputCompositeTransform->AddTransform(registration->GetModifiableTransform() );

    using ResampleFilterType = itk::ResampleImageFilter<MovingImageType,FixedImageType >;
    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetInput( movingImageReader->GetOutput() );
    resampler->SetTransform( outputCompositeTransform );
    resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
    resampler->SetOutputOrigin( fixedImage->GetOrigin() );
    resampler->SetOutputSpacing( fixedImage->GetSpacing() );
    resampler->SetOutputDirection( fixedImage->GetDirection() );
    resampler->SetDefaultPixelValue( 100 );
    
    using OutputPixelType = float;
    using OutputImageType = itk::Image< OutputPixelType, Dimension >;
    using CastFilterType = itk::CastImageFilter<FixedImageType,OutputImageType >;
    using WriterType = itk::ImageFileWriter< OutputImageType >;

    WriterType::Pointer writer = WriterType::New();
    CastFilterType::Pointer caster =  CastFilterType::New();
    caster->SetInput( resampler->GetOutput() );
    writer->SetInput( caster->GetOutput());
    writer->SetFileName(outputImageFile);
    writer->Update();

    using SubtractImageFilterType=itk::SubtractImageFilter<FixedImageType,FixedImageType,ImageType >;
    SubtractImageFilterType::Pointer difference = SubtractImageFilterType::New();
    difference->SetInput1( fixedImageReader->GetOutput() );
    difference->SetInput2( resampler->GetOutput() );
    difference->Update();
    writer->SetInput( difference->GetOutput());
    writer->SetFileName(differenceImageAfterFile);
    writer->Update();

  return EXIT_SUCCESS;

}

//OTSMULTIPULETHRESHOLD

#include "itkOtsuMultipleThresholdsCalculator.h"
using ScalarImageToHistogramGeneratorType =
itk::Statistics::ScalarImageToHistogramGenerator<InputImageType>;
using HistogramType = ScalarImageToHistogramGeneratorType::HistogramType;
using CalculatorType = itk::OtsuMultipleThresholdsCalculator<HistogramType>;

using FilterType = itk::BinaryThresholdImageFilter<InputImageType, OutputImageType >;
ScalarImageToHistogramGeneratorType::Pointer scalarImageToHistogramGenerator= ScalarImageToHistogramGeneratorType::New();
CalculatorType::Pointer calculator = CalculatorType::New();
FilterType::Pointer filter = FilterType::New();

scalarImageToHistogramGenerator->SetNumberOfBins( 128 );
calculator->SetNumberOfThresholds( std::stoi( argv[4] ) );

scalarImageToHistogramGenerator->SetInput( reader->GetOutput() );
calculator->SetInputHistogram(scalarImageToHistogramGenerator->GetOutput() );

const CalculatorType::OutputType &thresholdVector = calculator->GetOutput();
for( auto itNum = thresholdVector.begin();
itNum != thresholdVector.end();
++itNum )
{
std::cout << "OtsuThreshold["
<< (int)(itNum - thresholdVector.begin())
<< "] = "
<< static_cast<itk::NumericTraits<
CalculatorType::MeasurementType>::PrintType>(*itNum)<< std::endl;
}
upperThreshold = itk::NumericTraits<InputPixelType>::max();
filter->SetLowerThreshold( lowerThreshold );
filter->SetUpperThreshold( upperThreshold );
  
filter->SetInput( reader->GetOutput() );
writer->SetInput( filter->GetOutput() );
