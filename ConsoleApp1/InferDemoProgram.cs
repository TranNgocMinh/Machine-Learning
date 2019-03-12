using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace ConsoleApp1
{
    class InferDemoProgram
    {
        static void Main(string[] args)
        {
            //**************Step 1: Define a probabilistic model*************************
            //Assume that we know temperature of three days:
            //Day 1: 13 Celsius, Day 2: 17 Celsius, and Day 3: 16 Celsius
            double[] temp = new double[3] { 13, 17, 16 };
            // Creating a Gaussian distribution and a Gamma distribution
            Variable<double> averageTemp = Variable.GaussianFromMeanAndPrecision(15, 0.01).Named("Average Temperature");
            Variable<double> precision = Variable.GammaFromShapeAndScale(2.0, 0.5).Named("Precision");

            // Train the model
            // using the Range object to handle the array of data efficiently
            Range dataRange = new Range(temp.Length).Named("n");
            VariableArray<double> daysTemp = Variable.Array<double>(dataRange);
            daysTemp[dataRange] = Variable.GaussianFromMeanAndPrecision(averageTemp, precision).ForEach(dataRange);
            daysTemp.ObservedValue = temp;
            //**************Step 2: Creating an inference engine*************************
            InferenceEngine ie = new InferenceEngine();
            ie.ShowFactorGraph = true;
            //**************Step 3: Execution of an inference query*************************
            //Make predictions
            //Add a prediction variable and retrain the model
            Variable<double> tomorrowsTemp = Variable.GaussianFromMeanAndPrecision(averageTemp, precision).Named("Tomorrows Predicted Temperature");
            //get the Gaussian distribution
            Gaussian tomorrowsTempDist = ie.Infer<Gaussian>(tomorrowsTemp);
            // get the mean 
            double tomorrowsMean = tomorrowsTempDist.GetMean();
            //get the variance
            double tomorrowsStdDev = Math.Sqrt(tomorrowsTempDist.GetVariance());
            //Using Expectation Propagation - the default algorithm
            if (!(ie.Algorithm is VariationalMessagePassing))
           {
                // Write out the results.
                Console.WriteLine("Tomorrows predicted temperature: {0:f2} Celsius plus or minus {1:f2}", tomorrowsMean, tomorrowsStdDev);
                // Ask other questions of the model
                double probTempLessThan18Celsius = ie.Infer<Bernoulli>(tomorrowsTemp < 18.0).GetProbTrue();
                Console.WriteLine("Probability that the temperature is less than 18 Celsius: {0:f2}", probTempLessThan18Celsius);
            }
            else
                Console.WriteLine("Not run with Variational Message Passing!");
            Console.ReadKey();
        }
    }
}
