#pragma once

#include "model.hpp"
#include "checkpoint.hpp"
#include <filesystem>


class SafetensorsCheckpoint : public Checkpoint {

    private:
        ModelConfig read_config(const std::filesystem::path& file_path) const;
        std::unordered_map<std::string, TensorInfo> read_tensorinfo(const std::filesystem::path& file_path) const;

    public:
        explicit SafetensorsCheckpoint(
            const std::filesystem::path& model_path
        ) : Checkpoint(model_path) {}

        Model load() const override;
};
