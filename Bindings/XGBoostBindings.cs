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
using System.Runtime.InteropServices;

namespace XGBoostBindings
{
    /**
    * xgboost c api bindings
    */
    public class Bindings
    {
        public const string LibName = "libxgboost";

        /*!
         *  get string message of the last error
         *
         *  all function in this file will return 0 when success
         *  and -1 when an error occurred,
         *  XGBGetLastError can be called to retrieve the error
         *
         *  this function is thread safe and can be called by different thread
         * \return const char* error information
         */
        [DllImport(LibName)]
        public static extern string XGBGetLastError();

        /*!
         *  create matrix content from dense matrix
         * \param data pointer to the data space
         * \param nrow number of rows
         * \param ncol number columns
         * \param missing which value to represent missing value
         * \param out created dmatrix
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(LibName)]
        public static extern int XGDMatrixCreateFromMat(float[] data, ulong nRow, ulong nCol,
                                                    float missing, out IntPtr handle);
        /*!
         * \brief free space in data matrix
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(LibName)]
        public static extern int XGDMatrixFree(IntPtr handle);
        
        /*!
         * \brief set float vector to a content in info
         * \param handle a instance of data matrix
         * \param field field name, can be label, weight
         * \param array pointer to float vector
         * \param len length of array
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(LibName)]
        public static extern int XGDMatrixSetFloatInfo(IntPtr handle, string field,
              float[] array, ulong len);

        /*!
         * \brief get float info vector from matrix
         * \param handle a instance of data matrix
         * \param field field name
         * \param out_len used to set result length
         * \param out_dptr pointer to the result
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(LibName)]
        public static extern int XGDMatrixGetFloatInfo(IntPtr handle, string field,
                                                   out ulong len, out IntPtr result);
        /*!
         * \brief create xgboost learner
         * \param dmats matrices that are set to be cached
         * \param len length of dmats
         * \param out handle to the result booster
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(LibName)]
        public static extern int XGBoosterCreate(IntPtr[] matrices,
                                                 ulong len, out IntPtr handle);

        /*!
         * \brief set parameters
         * \param handle handle
         * \param name  parameter name
         * \param value value of parameter
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(LibName)]
        public static extern int XGBoosterSetParam(IntPtr handle, string name, string val);

        /*!
         * \brief update the model in one round using dtrain
         * \param handle handle
         * \param iter current iteration rounds
         * \param dtrain training data
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(LibName)]
        public static extern int XGBoosterUpdateOneIter(IntPtr bHandle, int iteration,
                                                        IntPtr dHandle);
        /*!
         * \brief make prediction based on dmat
         * \param handle handle
         * \param dmat data matrix
         * \param option_mask bit-mask of options taken in prediction, possible values
         *          0:normal prediction
         *          1:output margin instead of transformed value
         *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
         *          4:output feature contributions to individual predictions
         * \param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
         *    when the parameter is set to 0, we will use all the trees
         * \param out_len used to store length of returning result
         * \param out_result used to set a pointer to array
         * \return 0 when success, -1 when failure happens
         */
        [DllImport(LibName)]
        public static extern int XGBoosterPredict(IntPtr bHandle, IntPtr dHandle,
                                                  int optionMask, int ntreeLimit,
                                                  out ulong pLen, out IntPtr pPtr);
    }
}
