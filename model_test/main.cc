#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

void print_tensor(const std::string &name, tensorflow::Tensor &tensor,
                  bool keep_shape = false) {
    std::cout << name << " shape: " << tensor.shape() << std::endl;
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
                        std::shared_ptr<tensorflow::SavedModelBundle> &bundle) {
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    return tensorflow::LoadSavedModel(
        session_options, run_options, saved_model_dir,
        {tensorflow::kSavedModelTagServe}, bundle.get());
}

void run(std::shared_ptr<tensorflow::SavedModelBundle> &bundle,
         const std::string &signature_name, bool keep_shape = false) {
    auto it = bundle->meta_graph_def.signature_def().find(signature_name);
    if (it == bundle->meta_graph_def.signature_def().end()) {
        std::cerr << "signature_name: \"" << signature_name << "\" not found!"
                  << std::endl;
        return;
    }
    auto signature = it->second;
    auto input_infos = signature.inputs();
    auto output_infos = signature.outputs();

    std::vector<std::pair<std::string, tensorflow::Tensor> > inputs;
    if (input_infos.size() > 0) {
        for (auto &input_info : input_infos) {
            inputs.emplace_back(std::make_pair<std::string, tensorflow::Tensor>(
                input_info.second.name().c_str(),
                tensorflow::Input::Initializer({10, 20}).tensor));
        }
    }

    std::vector<std::string> output_names;
    for (auto &output_info : output_infos) {
        output_names.emplace_back(output_info.second.name());
    }

    std::vector<tensorflow::Tensor> outputs;
    auto status = bundle->GetSession()->Run(inputs, output_names, {}, &outputs);
    if (!status.ok()) {
        std::cerr << "Session run failed: " << status;
        return;
    }

    std::cout << "output_num: " << outputs.size() << std::endl;

    for (size_t i = 0; i < outputs.size(); ++i) {
        std::stringstream ss;
        ss << "output[" << i << "] \"" << output_names[i] << "\"";
        print_tensor(ss.str(), outputs[i], keep_shape);
    }

    std::cout << "Session run success!" << std::endl;
}

tensorflow::Status clone(
    const std::shared_ptr<tensorflow::SavedModelBundle> &bundle,
    std::shared_ptr<tensorflow::SavedModelBundle> &cloned_bundle) {
    cloned_bundle->meta_graph_def = bundle->meta_graph_def;
    cloned_bundle->debug_info = std::unique_ptr<tensorflow::GraphDebugInfo>();
    cloned_bundle->debug_info->CopyFrom(*bundle->debug_info);

    // create session
    tensorflow::SessionOptions session_options;
    tensorflow::RunOptions run_options;
    TF_RETURN_IF_ERROR(tensorflow::LoadMetagraphIntoSession(
        session_options, cloned_bundle->meta_graph_def,
        &cloned_bundle->session));

    // get orginal model embedding tensor values
    auto it = bundle->meta_graph_def.signature_def().find("embedding");
    if (it == bundle->meta_graph_def.signature_def().end()) {
        std::cerr << "signature_name: \"embedding\" not found!" << std::endl;
        return tensorflow::Status::OK();
    }
    std::vector<std::string> output_names;
    for (auto &output_info : it->second.outputs()) {
        std::cout << "output: " << output_info.first
                  << " name: " << output_info.second.name()
                  << " shape: " << output_info.second.tensor_shape()
                  << std::endl;
        output_names.emplace_back(output_info.second.name());
    }
    std::vector<tensorflow::Tensor> output_tensors;
    TF_RETURN_IF_ERROR(
        bundle->GetSession()->Run({}, output_names, {}, &output_tensors));

    std::cout << "output num: " << (output_tensors.size()) << std::endl;
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        std::cout << "output[" << i << "] shape " << output_tensors[i].shape()
                  << std::endl;
    }

    // set cloned model embedding tensor values
    return tensorflow::Status::OK();
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
    run(v1, "predict", true);
    // run(v1, "embedding", false);

    auto v2 = std::make_shared<tensorflow::SavedModelBundle>();
    // TF_CHECK_OK(load(model_path, v2));
    TF_CHECK_OK(clone(v1, v2));
    // run(v2);
    return 0;
}
