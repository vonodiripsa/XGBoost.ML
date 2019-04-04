/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
using System.Linq;
using XGBoostBindings;

namespace Benchmark
{
    /**
    * DMatrix for xgboost.
    *
    */
    public class Matrix : IDisposable
    {
        private bool _disposed;
        private readonly IntPtr _handle;
        public IntPtr Handle => _handle;

        public Matrix(float[][] data)
            : this(data, unchecked((ulong)data.Length), unchecked((ulong)data[0].Length), null)
        {
        }

        public Matrix(float[][] data, float[] labels)
            : this(data, unchecked((ulong)data.Length), unchecked((ulong)data[0].Length), labels)
        {
        }

        public Matrix(float[][] data, ulong nrows, ulong ncols, float[] labels)
        {
            float[] flatData = CreateFlatArray(data);
            int output = Bindings.XGDMatrixCreateFromMat(flatData, nrows, ncols, -1.0F, out _handle);
            if (output == -1)
                throw new Exception(Bindings.XGBGetLastError());

            if (labels != null)
            {
                SetFloatInfo("label", labels);
            }
        }

        private static float[] CreateFlatArray(float[][] data)
        {
            var count = data.Sum(t => t.Length);

            var resultData = new float[count];
            int ind = 0;
            foreach (var d1 in data)
            {
                foreach (var d2 in d1)
                {
                    resultData[ind] = d2;
                    ind += 1;
                }
            }
            return resultData;
        }

        private void SetFloatInfo(string field, float[] floatInfo)
        {
            ulong len = (ulong)floatInfo.Length;
            int output = Bindings.XGDMatrixSetFloatInfo(_handle, field, floatInfo, len);
            if (output == -1)
                throw new Exception(Bindings.XGBGetLastError());
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            int output = Bindings.XGDMatrixFree(_handle);
            if (output == -1)
                throw new Exception(Bindings.XGBGetLastError());

            _disposed = true;
        }
    }
}
