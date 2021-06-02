#include <iostream>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cout << "Usage: " << argv[0] << " <saved_model_dir>" << std::endl;
        std::exit(1);
    }
    std::string model_path = argv[1];

    tensorflow::SavedModelBundle model;
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;

    auto status = tensorflow::LoadSavedModel(session_options, run_options,
                                             model_path, {"serve"}, &model);
    if (!status.ok()) {
        std::cerr << "Failed: " << status;
        return -1;
    }

    std::cout << "Load model success!" << std::endl;
    return 0;
}
