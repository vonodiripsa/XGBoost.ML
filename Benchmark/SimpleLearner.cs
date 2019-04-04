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

using System.Collections.Generic;
using System.Linq;

namespace Benchmark
{
    /**
    * trainer/predictor for xgboost
    */
    public class SimpleLearner
    {
        private IDictionary<string, object> Parameters = new Dictionary<string, object>();
        private ModelBooster Booster;

        public int MaxDepth { get; set; }
        public float LearningRate { get; set; }
        public int NEstimators { get; set; }
        public bool Silent { get; set; }
        public string Objective { get; set; }
        public int NThread { get; set; }
        public float Gamma { get; set; }
        public int MinChildWeight { get; set; }
        public int MaxDeltaStep { get; set; }
        public float Subsample { get; set; }
        public float ColSampleByTree { get; set; }
        public float ColSampleByLevel { get; set; }
        public float RegAlpha { get; set; }
        public float RegLambda { get; set; }
        public float ScalePosWeight { get; set; }
        public float BaseScore { get; set; }
        public int Seed { get; set; }
        public float Missing { get; set; }
        public int NumClass { get; set; }

        public SimpleLearner()
        {
            MaxDepth = 3;
            LearningRate = 0.1F;
            NEstimators = 100;
            Silent = true;
            Objective = "reg:linear";
            NThread = -1;
            Gamma = 0F;
            MinChildWeight = 1;
            MaxDeltaStep = 0;
            Subsample = 1F;
            ColSampleByTree = 1F;
            ColSampleByLevel = 1F;
            RegAlpha = 0F;
            RegLambda = 1F;
            ScalePosWeight = 1F;
            BaseScore = 0.5F;
            Seed = 0;
            Missing = float.NaN;
            NumClass = 1;

            InitiateParameters();
        }

        public SimpleLearner(int maxDepth, float learningRate, int nEstimators,
            bool silent, string objective,
            int nThread, float gamma, int minChildWeight,
            int maxDeltaStep, float subsample, float colSampleByTree,
            float colSampleByLevel, float regAlpha, float regLambda,
            float scalePosWeight, float baseScore, int seed,
            float missing, int numClass)
        {
            MaxDepth = maxDepth;
            LearningRate = learningRate;
            NEstimators = nEstimators;
            Silent = silent;
            Objective = objective;
            NThread = nThread;
            Gamma = gamma;
            MinChildWeight = minChildWeight;
            MaxDeltaStep = maxDeltaStep;
            Subsample = subsample;
            ColSampleByTree = colSampleByTree;
            ColSampleByLevel = colSampleByLevel;
            RegAlpha = regAlpha;
            RegLambda = regLambda;
            ScalePosWeight = scalePosWeight;
            BaseScore = baseScore;
            Seed = seed;
            Missing = missing;
            NumClass = numClass;

            InitiateParameters();
        }

        public void InitiateParameters()
        {
            Parameters["max_depth"] = MaxDepth;
            Parameters["learning_rate"] = LearningRate;
            Parameters["n_estimators"] = NEstimators;
            Parameters["silent"] = Silent;
            Parameters["objective"] = Objective;
            Parameters["Booster"] = "gbtree";
            Parameters["tree_method"] = "auto";
            Parameters["nthread"] = NThread;
            Parameters["gamma"] = Gamma;
            Parameters["min_child_weight"] = MinChildWeight;
            Parameters["max_delta_step"] = MaxDeltaStep;
            Parameters["subsample"] = Subsample;
            Parameters["colsample_bytree"] = ColSampleByTree;
            Parameters["colsample_bylevel"] = ColSampleByLevel;
            Parameters["reg_alpha"] = RegAlpha;
            Parameters["reg_lambda"] = RegLambda;
            Parameters["scale_pos_weight"] = ScalePosWeight;
            Parameters["sample_type"] = "uniform";
            Parameters["normalize_type"] = "tree";
            Parameters["rate_drop"] = 0f;
            Parameters["one_drop"] = 0;
            Parameters["skip_drop"] = 0f;
            Parameters["base_score"] = BaseScore;
            Parameters["seed"] = Seed;
            Parameters["missing"] = Missing;
            Parameters["_Booster"] = null;
            Parameters["num_class"] = NumClass;
        }

        public float[] Predict(float[][] data)
        {
            using (var test = new Matrix(data))
            {
                var retArray = Booster.Predict(test);
                //convert to 0,1
                retArray = retArray.Select(v => v > 0.5f ? 1f : 0f).ToArray();
                return retArray;
            }
        }

        public void Train(float[][] data, float[] labels)
        {
            using (var train = new Matrix(data, labels))
            {
                Booster = Train(Parameters, train, (int)Parameters["n_estimators"]);
            }
        }

        private ModelBooster Train(IDictionary<string, object> args, Matrix train, int nEstimators)
        {
            var boost = new ModelBooster(args, train);
            for (var i = 0; i < nEstimators; i++) { boost.UpdateOneIter(train, i); }
            return boost;
        }
    }
}
