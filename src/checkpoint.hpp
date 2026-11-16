#pragma once

#include <filesystem>
#include <string>
#include "model.hpp"


class Checkpoint {
    protected:
        explicit Checkpoint(const std::filesystem::path& model_path)
            : model_path_(model_path) {}

        std::filesystem::path model_path_;

    public:
        virtual ~Checkpoint() = default;
        virtual Model load() const = 0;

};
