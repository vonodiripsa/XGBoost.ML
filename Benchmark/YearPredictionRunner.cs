/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Globalization;
using System.Linq;
using Microsoft.Extensions.Configuration;

namespace Benchmark
{
    /**
    * Runner for YearPredictionMSD data set
    */
    class YearPredictionRunner
    {
        static void YearPredictionMSDBenchmark(string dataPath, int numTrainRows, int numberTestRows)
        {
            //YearPredictionMSD benchmark
            BinData allData = YearPredictionLoader.LoadYearPredictionDataFromFile(dataPath);
            BinData bData = YearPredictionLoader.GenerateenchmarkData(allData);

            var xgbc = new SimpleLearner();

            long startTime = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;

            xgbc.Train(bData.trainData, bData.label);
            xgbc.Predict(bData.testData);

            long stopTime = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;

            long latency = stopTime - startTime;

            var time = TimeSpan.FromMilliseconds(latency);
            
            Console.Out.WriteLine("Run time: " + time);
        }

        static void Main(string[] args)
        {
            var config = new ConfigurationBuilder()
                .AddJsonFile("appconfig.json")
                .Build();
            string dataPath = config["data_path"];

            int.TryParse(config["train_rows"], out int numTrainRows);
            int.TryParse(config["test_rows"], out var numberTestRows);

            // YearPredictionMSD performance estimate
            YearPredictionMSDBenchmark(dataPath, numTrainRows, numberTestRows);        
        }
    }
}
