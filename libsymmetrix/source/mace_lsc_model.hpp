#pragma once

#include <string>
#include "nlohmann/json.hpp"

// Internal host snapshot: parse once, validate through the CPU reader, then
// prepare a backend-specific layout without reopening the model file.
struct LSCModelData {
    nlohmann::json document;
};
LSCModelData read_validated_lsc_model(const std::string& filename);
