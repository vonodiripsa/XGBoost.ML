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
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using CsvHelper;

namespace Benchmark
{
    /**
     * Load Year Prediction data
     */
    class YearPredictionLoader
    {
        public static string YearPredictionFileName = "YearPredictionMSD.txt";
        public static string YearPredictionBinFileName = "YearPredictionMSD.bin";

        public static int NumRows = 515345;

        public static BinData LoadYearPredictionDataFromFile(string path)
        {
            String serFileName = path + YearPredictionBinFileName;

            if (File.Exists(serFileName))
            {
                IFormatter formatter = new BinaryFormatter();
                Stream stream = new FileStream(serFileName, FileMode.Open, FileAccess.Read, FileShare.Read);
                BinData data = (BinData)formatter.Deserialize(stream);
                stream.Close();

                return data;
            }
            else
            {
                List<float[]> X = new List<float[]>();
                List<float> y = new List<float>();

                using (var reader = new StreamReader(path + YearPredictionFileName))
                using (var csv = new CsvReader(reader))
                {
                    csv.Configuration.HasHeaderRecord = false;
                    while (csv.Read())
                    {

                        var r = csv.GetRecord<YearData>();
                        X.Add(r.X);
                        y.Add(r.y);

                    }
                }

                float[][] dTrain = new float[NumRows][];

                for (int i = 0; i < NumRows; i++)
                {
                    dTrain[i] = X[i];
                }

                float[] label = new float[NumRows];

                for (int i = 0; i < NumRows; i++)
                {
                    label[i] = y[i];
                }

                BinData data = new BinData(dTrain, label, null, null);

                IFormatter wFormatter = new BinaryFormatter();
                Stream wStream = new FileStream(serFileName, FileMode.Create, FileAccess.Write, FileShare.None);
                wFormatter.Serialize(wStream, data);
                wStream.Close();

                return data;
            }
        }

        public static BinData GenerateenchmarkData(BinData data)
        {
            int numTrainRows = 10000;
            int numberTestRows = 1000;

            int startRow = numberTestRows + 1;


            float[][] dTrain = new float[numTrainRows][];

            for (int i = startRow; i < startRow + numTrainRows; i++)
            {
                dTrain[i - startRow] = data.trainData[i];
            }

            float[] label = new float[numTrainRows];

            for (int i = startRow; i < startRow + numTrainRows; i++)
            {
                label[i - startRow] = data.label[i];
            }


            float[][] dTest = new float[numberTestRows][];

            for (int i = 0; i < numberTestRows; i++)
            {
                dTest[i] = data.trainData[i];
            }

            return new BinData(dTrain, label, dTest, null);
        }


    }
}
