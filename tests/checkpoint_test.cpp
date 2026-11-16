#include "safetensor_checkpoint.hpp"

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

static_assert(std::is_same_v<decltype(std::declval<const Checkpoint&>().load()), Model>);

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("Checkpoint test failed");
    }
}

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("locinfer-checkpoint-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);

    std::ofstream(directory / "config.json") << R"({
        "architectures": ["QwenForCausalLM"],
        "vocab_size": 100, "hidden_size": 16, "num_hidden_layers": 2
    })";

    const std::string header = R"({"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"bias":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})";
    {
        std::ofstream file(directory / "model.safetensors", std::ios::binary);
        for (unsigned i = 0; i < 8; ++i) {
            file.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> (8 * i)) & 0xff));
        }
        file << header;
        const char data[8]{};
        file.write(data, sizeof(data));
    }

    Model model = SafetensorsCheckpoint(directory).load();
    require(model.config.architecture == "QwenForCausalLM");
    require(model.config.vocab_size == 100);
    require(model.config.hidden_size == 16);
    require(model.config.num_layers == 2);
    require(model.tensors.size() == 2);
    const auto& weight = model.tensors.at("weight");
    require(weight.dtype == "F32");
    require(weight.shape == std::vector<std::uint64_t>{1});
    require(weight.begin == 8 + header.size());
    require(weight.end == weight.begin + 4);
    require(model.tensors.at("bias").begin == weight.end);
    require(model.tensors.at("bias").end == weight.end + 4);

    bool rejected = false;
    try {
        SafetensorsCheckpoint(directory / "missing").load();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected);

    std::filesystem::remove_all(directory);
}
