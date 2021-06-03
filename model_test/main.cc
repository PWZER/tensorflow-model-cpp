#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <sstream>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/platform/status.h"

void print_tensor(const std::string &name, tensorflow::Tensor &tensor,
                  bool keep_shape = false) {
    std::cout << name << ": ";
    if (keep_shape == false || tensor.dims() <= 1) {
        if (tensor.dims() > 0) {
            std::cout << "[";
        }
        auto data = (float *)tensor.data();
        for (int j = 0; j < tensor.NumElements(); ++j) {
            if (j > 0) {
                std::cout << ", ";
            }
            std::cout << data[j];
        }
        if (tensor.dims() > 0) {
            std::cout << "]";
        }
        std::cout << std::endl;
    } else {
        std::cout << "[" << std::endl;
        for (int i = 0; i < tensor.dim_size(0); ++i) {
            auto sub = tensor.SubSlice(i);
            std::cout << "\t[";

            auto data = (float *)sub.data();
            for (int j = 0; j < sub.NumElements(); ++j) {
                if (j > 0) {
                    std::cout << ", ";
                }
                std::cout << data[j];
            }
            std::cout << "]," << std::endl;
        }
        std::cout << "]" << std::endl;
    }
}

tensorflow::Status load(const std::string &saved_model_dir,
                        std::shared_ptr<tensorflow::SavedModelBundle> &model) {
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    return tensorflow::LoadSavedModel(session_options, run_options,
                                      saved_model_dir, {"serve"}, model.get());
}

void run(std::shared_ptr<tensorflow::SavedModelBundle> &model) {
    std::vector<std::pair<std::string, tensorflow::Tensor> > inputs;
    inputs.emplace_back(std::make_pair<std::string, tensorflow::Tensor>(
        "Placeholder:0", tensorflow::Input::Initializer({10, 20}).tensor));

    std::vector<std::string> output_tensor_names = {
        "embedding_lookup/Identity_1"};

    std::vector<tensorflow::Tensor> outputs;
    auto status =
        model->GetSession()->Run(inputs, output_tensor_names, {}, &outputs);
    if (!status.ok()) {
        std::cerr << "Session run failed: " << status;
        return;
    }

    std::cout << "output_num: " << outputs.size() << std::endl;

    for (size_t i = 0; i < outputs.size(); ++i) {
        std::stringstream ss;
        ss << "output[" << i << "]";
        print_tensor(ss.str(), outputs[i], true);
    }

    std::cout << "Session run success!" << std::endl;
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        std::cout << "Usage: " << argv[0] << " <saved_model_dir>" << std::endl;
        std::exit(1);
    }
    std::string model_path = argv[1];

    tensorflow::Status status;

    auto v1 = std::make_shared<tensorflow::SavedModelBundle>();
    TF_CHECK_OK(load(model_path, v1));
    run(v1);

    auto v2 = std::make_shared<tensorflow::SavedModelBundle>();
    TF_CHECK_OK(load(model_path, v2));
    run(v2);
    return 0;
}
