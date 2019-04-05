# XGBoost.ML

This is the first part of Nvidia Rapids and ML.Net integration

XGBoost.ML is the .Net Core package of xgboost. It is an attempt to brings the optimizations
and power of xgboost and Rapids into ML.Net on Linux and Windows.

You can find more about XGBoost on [Documentation](https://github.com/dmlc/xgboost)

To run it you have to download and put YearPredictionMSD.txt file in data folder creted on the same level as project folders. Benchmark bin (Benchmark/bin/Debug/netcoreapp2.2) should have XGBoost libxgboost.so. After you run it first time (slow parsing) a serialized YearPredictionMSD file will be created and the load will take several second.
One of the ways to run it is launching dotnet run in Benchmark folder.
