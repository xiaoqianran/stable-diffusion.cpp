#include "stable-diffusion.h"

#include <cassert>
#include <cstring>

int main() {
    for (int i = 0; i < RNG_TYPE_COUNT; ++i) {
        const auto value = static_cast<rng_type_t>(i);
        const char* name = sd_rng_type_name(value);
        assert(name != nullptr && name[0] != '\0');
        assert(str_to_rng_type(name) == value);
    }

    for (int i = 0; i < SAMPLE_METHOD_COUNT; ++i) {
        const auto value = static_cast<sample_method_t>(i);
        const char* name = sd_sample_method_name(value);
        assert(name != nullptr && name[0] != '\0');
        assert(str_to_sample_method(name) == value);
    }

    for (int i = 0; i < SCHEDULER_COUNT; ++i) {
        const auto value = static_cast<scheduler_t>(i);
        const char* name = sd_scheduler_name(value);
        assert(name != nullptr && name[0] != '\0');
        assert(str_to_scheduler(name) == value);
    }

    for (int i = 0; i < PREDICTION_COUNT; ++i) {
        const auto value = static_cast<prediction_t>(i);
        const char* name = sd_prediction_name(value);
        assert(name != nullptr && name[0] != '\0');
        assert(str_to_prediction(name) == value);
    }

    for (int i = 0; i < PREVIEW_COUNT; ++i) {
        const auto value = static_cast<preview_t>(i);
        const char* name = sd_preview_name(value);
        assert(name != nullptr && name[0] != '\0');
        assert(str_to_preview(name) == value);
    }

    for (int i = 0; i < LORA_APPLY_MODE_COUNT; ++i) {
        const auto value = static_cast<lora_apply_mode_t>(i);
        const char* name = sd_lora_apply_mode_name(value);
        assert(name != nullptr && name[0] != '\0');
        assert(str_to_lora_apply_mode(name) == value);
    }

    for (int i = 0; i < SD_HIRES_UPSCALER_COUNT; ++i) {
        const auto value = static_cast<sd_hires_upscaler_t>(i);
        const char* name = sd_hires_upscaler_name(value);
        assert(name != nullptr && name[0] != '\0');
        assert(str_to_sd_hires_upscaler(name) == value);
    }

    assert(sd_version() != nullptr);
    assert(sd_commit() != nullptr);
    assert(str_to_rng_type("__invalid__") == RNG_TYPE_COUNT);
    assert(str_to_sample_method("__invalid__") == SAMPLE_METHOD_COUNT);
    assert(str_to_scheduler("__invalid__") == SCHEDULER_COUNT);
    return 0;
}
