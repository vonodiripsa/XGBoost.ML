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
using System.Globalization;
using System.Runtime.InteropServices;
using XGBoostBindings;

namespace Benchmark
{
    /**
    * Booster for xgboost, this is a model API that support interactive build of a XGBoost Model
    */
    public class ModelBooster 
    {
        private readonly IntPtr _handle;
        public IntPtr Handle => _handle;

        public ModelBooster(IDictionary<string, object> parameters, Matrix train)
        {
            var handle = new[] { train.Handle};
            var output = Bindings.XGBoosterCreate(handle, unchecked((ulong)handle.Length), out _handle);
            if (output == -1) throw new Exception(Bindings.XGBGetLastError());

            InitParameters(parameters);
        }

        public void UpdateOneIter(Matrix train, int iter)
        {
            var output = Bindings.XGBoosterUpdateOneIter(Handle, iter, train.Handle);
            if (output == -1) throw new Exception(Bindings.XGBGetLastError());
        }

        public float[] Predict(Matrix test)
        {
            var output = Bindings.XGBoosterPredict(
                _handle, test.Handle, 0, 0, out var predictionsLen, out var predictionsPtr);
            if (output == -1) throw new Exception(Bindings.XGBGetLastError());
            return GetPredictionsArray(predictionsPtr, predictionsLen);
        }

        public float[] GetPredictionsArray(IntPtr predictionsPtr, ulong predictionsLen)
        {
            byte floatSize = 4;
            var length = unchecked((int)predictionsLen);
            var predictions = new float[length];
            for (var numberOfFloats = 0; numberOfFloats < length; numberOfFloats++)
            {
                var floatBytes = new byte[floatSize];
                for (var byteIdx = 0; byteIdx < floatSize; byteIdx++)
                {
                    floatBytes[byteIdx] = Marshal.ReadByte(predictionsPtr, floatSize * numberOfFloats + byteIdx);
                }
                predictions[numberOfFloats] = BitConverter.ToSingle(floatBytes, 0);
            }
            return predictions;
        }
        public void InitParameters(IDictionary<string, object> parameters)
        {
            SetParameterValue("max_depth", ((int)parameters["max_depth"]).ToString());
            SetParameterValue("learning_rate", ((float)parameters["learning_rate"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("n_estimators", ((int)parameters["n_estimators"]).ToString());
            SetParameterValue("silent", ((bool)parameters["silent"]).ToString());
            SetParameterValue("objective", (string)parameters["objective"]);
            SetParameterValue("Booster", (string)parameters["Booster"]);
            SetParameterValue("tree_method", (string)parameters["tree_method"]);
            SetParameterValue("nthread", ((int)parameters["nthread"]).ToString());
            SetParameterValue("gamma", ((float)parameters["gamma"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("min_child_weight", ((int)parameters["min_child_weight"]).ToString());
            SetParameterValue("max_delta_step", ((int)parameters["max_delta_step"]).ToString());
            SetParameterValue("subsample", ((float)parameters["subsample"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("colsample_bytree", ((float)parameters["colsample_bytree"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("colsample_bylevel", ((float)parameters["colsample_bylevel"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("reg_alpha", ((float)parameters["reg_alpha"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("reg_lambda", ((float)parameters["reg_lambda"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("scale_pos_weight", ((float)parameters["scale_pos_weight"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("base_score", ((float)parameters["base_score"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("seed", ((int)parameters["seed"]).ToString());
            SetParameterValue("missing", ((float)parameters["missing"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("sample_type", (string)parameters["sample_type"]);
            SetParameterValue("normalize_type ", (string)parameters["normalize_type"]);
            SetParameterValue("rate_drop", ((float)parameters["rate_drop"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("one_drop", ((int)parameters["one_drop"]).ToString());
            SetParameterValue("skip_drop", ((float)parameters["skip_drop"]).ToString(CultureInfo.InvariantCulture));
            SetParameterValue("num_class", "1");
        }

        public void SetParameterValue(string name, string val)
        {
            var output = Bindings.XGBoosterSetParam(Handle, name, val);
            if (output == -1) throw new Exception(Bindings.XGBGetLastError());
        }

    }
}
